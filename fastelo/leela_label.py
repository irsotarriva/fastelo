"""Label whole games with a Leela network, split so the GPU part can run on another machine.

  local (CPU, 14 cores):
    python -m fastelo.leela_label pack --data data/rapid_v1 --split train \
        --out data/leela_parts/nov --positions-per-part 5000000 --parts 4
  GPU machine (resumable per part):
    python -m fastelo.leela_label run --parts parts --onnx t1.f16.onnx --out labels
  local again:
    labels = load_labels(part_dir, out_dir)   # our-vocab moves + teacher logits per position

A part holds whole games (Leela needs the 7 previous positions, which are then already in the
file), so shipping costs ~90 B/position:
  records.npy   [N,29] uint8   positions in game order (build_dataset.POS_DTYPE bytes)
  ply.npy       [N] uint16     ply of each position within its game (history availability)
  rep.npy       [N] uint8      1 if this position occurred earlier in the same game
  legal_off.npy [N+1] int64    CSR offsets of the legal moves
  legal_lc0.npy [M] int16      legal moves as lc0 policy indices   (shipped)
  legal_ours.npy[M] int16      same moves in our vocabulary        (kept locally, not needed for labelling)
Output per part (zstd-compressed .npy.zst): logits [M] float16 (teacher logits of the legal
moves, CSR as above) and wdl [N,3] float16. `ship` writes .zst copies of the shipped files
(~110 MB per 5M positions); `run` and `load_labels` read .npy or .npy.zst transparently.
"""
import argparse
import glob
import json
import multiprocessing as mp
import os
import time

import numpy as np
import torch

from .build_dataset import FLAG_BLACK
from .features import _OFF, REC, decode_records

SHIP_FILES = ("records.npy", "ply.npy", "rep.npy", "legal_off.npy", "legal_lc0.npy", "legal_ours.npy")  # ours: training


def load_npy(path):
    """np.load that also reads the zstd-compressed `<path>.zst` copies made by `ship`."""
    if os.path.exists(path):
        return np.load(path)
    import io
    import zstandard
    with open(path + ".zst", "rb") as f:
        return np.load(io.BytesIO(zstandard.ZstdDecompressor().decompress(f.read())))


def save_npy_zst(path, arr, level=10):
    import io
    import zstandard
    buf = io.BytesIO()
    np.save(buf, arr)
    with open(path + ".zst", "wb") as f:
        f.write(zstandard.ZstdCompressor(level=level, threads=-1).compress(buf.getvalue()))


def ship(args):
    """Copy the files the GPU machine needs, zstd-compressed (~4.5x smaller), into an upload folder."""
    import shutil
    for part in sorted(glob.glob(os.path.join(args.parts, "part_*"))):
        dst = os.path.join(args.out, os.path.basename(part))
        if os.path.exists(os.path.join(dst, "meta.json")):
            continue
        os.makedirs(dst + ".tmp", exist_ok=True)
        for f in SHIP_FILES:
            save_npy_zst(os.path.join(dst + ".tmp", f), np.load(os.path.join(part, f)))
        shutil.copy(os.path.join(part, "meta.json"), dst + ".tmp")
        os.replace(dst + ".tmp", dst)
        size = sum(os.path.getsize(os.path.join(dst, f)) for f in os.listdir(dst))
        print(f"{os.path.basename(part)}: {size / 1e6:.0f} MB", flush=True)


# --------------------------------------------------------------------- GPU plane encoder
def planes_from_records(raw, ply, rep, idx):
    """lc0 INPUT_CLASSICAL_112_PLANE inputs for positions `idx` of a part, built on the device
    of `raw` from the stored records of the position and the 7 before it in the same game.

    raw [N,29] uint8, ply [N] int64, rep [N] bool (all on one device), idx [B] int64.
    Returns [B,112,8,8] float32. Checked against leela.encode_planes (tests/test_leela_label.py).
    """
    B = len(idx)
    dev = raw.device
    out = torch.zeros(B, 112, 64, device=dev)
    sq = torch.arange(64, device=dev)
    mirror = sq ^ 56
    for t in range(8):
        ok = ply[idx] >= t                                  # history exists
        j = (idx - t).clamp(min=0)
        codes = decode_records(raw[j])["board"]            # [B,64] in the frame of the mover at j
        if t % 2:                                           # opponent's frame: mirror + swap sides
            codes = codes[:, mirror]
            codes = torch.where(codes > 0, (codes + 5) % 12 + 1, codes)
        for k in range(12):                                 # 0-5 ours P..K, 6-11 theirs
            out[:, 13 * t + k] = ((codes == k + 1) & ok.unsqueeze(1)).float()
        out[:, 13 * t + 12] = (rep[j] & ok).float().unsqueeze(1).expand(B, 64)
    cur = raw[idx]
    flags = cur[:, _OFF["flags"]].long()
    for plane, bit in ((104, 1), (105, 0), (106, 3), (107, 2)):  # we O-O-O, we O-O, they O-O-O, they O-O
        out[:, plane] = ((flags >> bit) & 1).float().unsqueeze(1)
    out[:, 108] = ((flags & FLAG_BLACK) != 0).float().unsqueeze(1)
    out[:, 109] = cur[:, _OFF["halfmove"]].float().unsqueeze(1)
    out[:, 111] = 1.0
    return out.view(B, 112, 8, 8)


# -------------------------------------------------------------------------------- pack
def repetition_flags(raw, game_of):
    """1 where the same position (pieces, castling, side, ep) occurred earlier in the game."""
    kb = _OFF["halfmove"]
    key = np.ascontiguousarray(raw[:, :kb]).copy()
    key[:, _OFF["flags"]] &= 0x1F                          # drop the PLAYER_OK bit
    rec = np.zeros(len(raw), dtype=[("g", "<i8"), ("k", f"V{kb}")])
    rec["g"], rec["k"] = game_of, key.view(f"V{kb}").ravel()
    order = np.argsort(rec, kind="stable")
    s = rec[order]
    dup = np.zeros(len(raw), bool)
    dup[1:] = (s["g"][1:] == s["g"][:-1]) & (s["k"][1:] == s["k"][:-1])
    out = np.zeros(len(raw), np.uint8)
    out[order] = dup  # stable sort keeps game order, so every later occurrence is flagged
    return out


def _legal_worker(rec_chunk):
    import chess
    from .board_codec import board_from_codes, decode_boards
    from .leela import lc0_index
    from .vocab import encode_move
    codes = decode_boards(rec_chunk[:, :8].copy().view(np.uint64).ravel(), rec_chunk[:, 8:24])
    lc0, ours, lens = [], [], []
    for i in range(len(rec_chunk)):
        r = rec_chunk[i]
        board = board_from_codes(codes[i], int(r[_OFF["flags"]]), int(r[_OFF["ep"]]), int(r[_OFF["halfmove"]]))
        moves = list(board.legal_moves)
        lens.append(len(moves))
        for m in moves:
            ours.append(encode_move(m, chess.WHITE))  # boards are in the mover's frame
            lc0.append(lc0_index(board, m))
    return np.array(lens, np.int64), np.array(lc0, np.int16), np.array(ours, np.int16)


def pack(args):
    from .shards import load_shards
    shards = load_shards(os.path.join(args.data, args.split))
    rng = np.random.default_rng(args.seed)
    # Whole games, visited in a seeded random order across shards, until the parts are full.
    order = [(s, gi) for s in range(len(shards)) for gi in range(len(shards[s][0]))]
    perm = rng.permutation(len(order))
    os.makedirs(args.out, exist_ok=True)
    cursor = 0
    for part in range(args.parts):
        d = os.path.join(args.out, f"part_{part:04d}")
        if os.path.exists(os.path.join(d, "meta.json")):
            cursor = json.load(open(os.path.join(d, "meta.json")))["cursor_end"]
            continue
        t0 = time.time()
        picked, n = [], 0
        while n < args.positions_per_part and cursor < len(perm):
            s, gi = order[perm[cursor]]
            cursor += 1
            n_plies = int(shards[s][0]["n_plies"][gi])
            picked.append((s, int(shards[s][0]["start"][gi]), n_plies))
            n += n_plies
        picked.sort()  # read shards in order
        recs, plies, game_of = [], [], []
        for k, (s, start, n_plies) in enumerate(picked):
            recs.append(shards[s][1][start:start + n_plies])
            plies.append(np.arange(n_plies, dtype=np.uint16))
            game_of.append(np.full(n_plies, k, np.int64))
        raw = np.ascontiguousarray(np.concatenate(recs)).view(np.uint8).reshape(-1, REC)
        ply, game_of = np.concatenate(plies), np.concatenate(game_of)
        rep = repetition_flags(raw, game_of)
        chunks = np.array_split(raw, max(1, len(raw) // 20000))
        with mp.Pool(args.workers) as pool:
            res = pool.map(_legal_worker, chunks)
        lens = np.concatenate([r[0] for r in res])
        tmp = d + ".tmp"
        os.makedirs(tmp, exist_ok=True)
        np.save(os.path.join(tmp, "records.npy"), raw)
        np.save(os.path.join(tmp, "ply.npy"), ply)
        np.save(os.path.join(tmp, "rep.npy"), rep)
        np.save(os.path.join(tmp, "legal_off.npy"), np.concatenate([[0], np.cumsum(lens)]))
        np.save(os.path.join(tmp, "legal_lc0.npy"), np.concatenate([r[1] for r in res]))
        np.save(os.path.join(tmp, "legal_ours.npy"), np.concatenate([r[2] for r in res]))
        with open(os.path.join(tmp, "meta.json"), "w") as f:
            json.dump(dict(data=os.path.abspath(args.data), split=args.split, seed=args.seed,
                           games=len(picked), positions=len(raw), legal_moves=int(lens.sum()),
                           cursor_end=cursor), f, indent=1)
        os.replace(tmp, d)
        print(f"part {part}: {len(picked):,} games, {len(raw):,} positions, "
              f"{lens.mean():.1f} legal moves/position, {time.time() - t0:.0f}s", flush=True)


def part_names(args, store=None):
    """Part names from the local parts dir, plus (when parts are fetched on demand) the remote list."""
    names = {os.path.basename(p) for p in glob.glob(os.path.join(args.parts, "part_*"))}
    if store and args.parts_prefix:
        for key in store.list(args.parts_prefix.rstrip("/") + "/"):
            rest = key[len(args.parts_prefix.rstrip("/")) + 1:]
            if rest.startswith("part_"):
                names.add(rest.split("/")[0])
    return sorted(names)


# --------------------------------------------------------------------------------- run
def run(args):
    import onnxruntime as ort
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if dev.type == "cuda" else ["CPUExecutionProvider"]
    sess = ort.InferenceSession(args.onnx, providers=providers)
    if dev.type == "cuda" and "CUDAExecutionProvider" not in sess.get_providers():
        raise RuntimeError("GPU present but onnxruntime fell back to CPU (CUDA/cuDNN version mismatch?)")
    half = "float16" in sess.get_inputs()[0].type
    print("onnxruntime providers:", sess.get_providers(), "| input fp16:", half, flush=True)
    os.makedirs(args.out, exist_ok=True)
    store = None
    if args.remote:
        from .cloud import open_store, sync_down
        store = open_store(args.remote)
    names = part_names(args, store)
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        names = names[i::n]  # this process's share (one process per GPU)
    for name in names:
        part = os.path.join(args.parts, name)
        dst = os.path.join(args.out, name)
        t0 = time.time()
        done_in = [p for p in (dst, os.path.join(args.copy_to, name) if args.copy_to else None)
                   if p and os.path.exists(os.path.join(p, "wdl.npy.zst"))]
        if not done_in and store and store.exists(f"{args.labels_prefix}/{name}/wdl.npy.zst"):
            done_in = [f"{args.remote}/{args.labels_prefix}/{name}"]
        if done_in:  # also on a fresh machine, when only the remote copy survives
            print(f"{name}: done ({done_in[0]}), skipping", flush=True)
            continue
        if store and args.parts_prefix:  # fetch this part's inputs on demand
            sync_down(store, f"{args.parts_prefix}/{name}", part)
        raw = torch.from_numpy(load_npy(os.path.join(part, "records.npy"))).to(dev)
        ply = torch.from_numpy(load_npy(os.path.join(part, "ply.npy")).astype(np.int64)).to(dev)
        rep = torch.from_numpy(load_npy(os.path.join(part, "rep.npy")).astype(bool)).to(dev)
        off = load_npy(os.path.join(part, "legal_off.npy"))
        lc0 = torch.from_numpy(load_npy(os.path.join(part, "legal_lc0.npy")).astype(np.int64)).to(dev)
        n = len(raw)
        t_run = time.time()  # rate excludes loading the part
        logits_out = np.empty(off[-1], np.float16)
        wdl_out = np.empty((n, 3), np.float16)
        for s in range(0, n, args.batch):
            idx = torch.arange(s, min(n, s + args.batch), device=dev)
            x = planes_from_records(raw, ply, rep, idx)
            x = (x.half() if half else x).cpu().numpy()
            pol, wdl = sess.run(["/output/policy", "/output/wdl"], {"/input/planes": x})
            pol = torch.from_numpy(pol.astype(np.float32)).to(dev)
            a, b = off[s], off[min(n, s + args.batch)]
            seg = torch.from_numpy(np.repeat(np.arange(len(idx)), np.diff(off[s:min(n, s + args.batch) + 1]))).to(dev)
            logits_out[a:b] = pol[seg, lc0[a:b]].half().cpu().numpy()
            wdl_out[s:s + len(idx)] = wdl.astype(np.float16)
            if (s // args.batch) % 200 == 0:
                done = s + len(idx)
                print(f"{name}: {done:,}/{n:,} ({done / max(time.time() - t_run, 1e-9):,.0f} positions/s)", flush=True)
        tmp = dst + ".tmp"
        os.makedirs(tmp, exist_ok=True)
        save_npy_zst(os.path.join(tmp, "logits.npy"), logits_out, level=3)  # smaller download
        save_npy_zst(os.path.join(tmp, "wdl.npy"), wdl_out, level=3)
        with open(os.path.join(tmp, "meta.json"), "w") as f:
            json.dump(dict(onnx=os.path.basename(args.onnx), positions=n, seconds=round(time.time() - t0, 1)), f)
        os.replace(tmp, dst)
        print(f"{name}: {n:,} positions in {time.time() - t0:.0f}s ({n / (time.time() - t0):,.0f}/s)", flush=True)
        if store:  # a stopped run keeps every finished part
            for f in os.listdir(dst):
                store.put_file(os.path.join(dst, f), f"{args.labels_prefix}/{name}/{f}")
            print(f"{name}: uploaded to {args.remote}/{args.labels_prefix}", flush=True)
        if args.copy_to:  # so a stopped session keeps every finished part
            import shutil
            shutil.copytree(dst, os.path.join(args.copy_to, name), dirs_exist_ok=True)
            print(f"{name}: copied to {args.copy_to}", flush=True)


# ------------------------------------------------------------------------------- local
def load_labels(part_dir, out_dir):
    """Teacher output for one part, joined with the locally kept our-vocab move lists."""
    return dict(records=load_npy(os.path.join(part_dir, "records.npy")),
                offsets=load_npy(os.path.join(part_dir, "legal_off.npy")),
                moves=load_npy(os.path.join(part_dir, "legal_ours.npy")),
                logits=load_npy(os.path.join(out_dir, "logits.npy")),
                wdl=load_npy(os.path.join(out_dir, "wdl.npy")))


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pack")
    p.add_argument("--data", required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--out", required=True)
    p.add_argument("--positions-per-part", type=int, default=5_000_000)
    p.add_argument("--parts", type=int, default=1)
    p.add_argument("--workers", type=int, default=14)
    p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("ship", help="zstd-compressed copies of what the GPU machine needs")
    p.add_argument("--parts", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("run")
    p.add_argument("--parts", required=True, help="directory with part_XXXX folders")
    p.add_argument("--onnx", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--batch", type=int, default=2048)
    p.add_argument("--copy-to", default=None, help="copy each finished part's output here ")
    p.add_argument("--shard", default=None, help="i/N: this process labels parts i, i+N, ... (one per GPU)")
    p.add_argument("--remote", default=None, help="s3://bucket or file://dir: skip parts labelled there, upload new ones")
    p.add_argument("--labels-prefix", default="labels", help="remote prefix for label outputs")
    p.add_argument("--parts-prefix", default=None, help="remote prefix holding the parts (fetched on demand)")
    args = ap.parse_args(argv)
    {"pack": pack, "ship": ship, "run": run}[args.cmd](args)


if __name__ == "__main__":
    main()
