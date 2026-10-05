"""Teacher benchmark: Leela networks vs Stockfish labels on the same real-game positions.

    # local: build a self-contained benchmark file (Leela input planes + legal moves + SF scores)
    python -m fastelo.teacher_bench pack --labels data/sf_labels/val_n1000 --n 50000 \
        --out data/teacher_bench_50k.npz
    # anywhere (CPU or GPU): quality, concentration, throughput, memory
    python -m fastelo.teacher_bench run --bench data/teacher_bench_50k.npz --onnx <net>.onnx \
        --providers CUDAExecutionProvider --batches 64,256,1024,4096

Quality is judged by Stockfish's scores for every legal move (expected-score loss vs the best
move). The same positions also carry the move the human actually played, as a reference.
"""
import argparse
import glob
import json
import os
import subprocess
import time

import numpy as np

from .eval_strength import win_prob

N_BIT_PLANES = 104


# ----------------------------------------------------------------------------------- pack
def pack(args):
    from .features import _OFF
    from .leela import board_with_history, encode_planes, lc0_index
    from .shards import load_shards
    from .vocab import encode_move

    meta = json.load(open(os.path.join(args.labels, "meta.json")))
    shards = load_shards(os.path.join(meta["data"], meta["split"]))
    rows = []  # (source, offsets into sf moves/scores)
    for d in sorted(glob.glob(os.path.join(args.labels, "chunk_*"))):
        src, off = np.load(os.path.join(d, "source.npy")), np.load(os.path.join(d, "offsets.npy"))
        moves, scores = np.load(os.path.join(d, "moves.npy")), np.load(os.path.join(d, "scores.npy"))
        elo, rec = np.load(os.path.join(d, "mover_elo.npy")), np.load(os.path.join(d, "records.npy"))
        for i in range(len(src)):
            mv = int(rec[i, _OFF["move"]]) | int(rec[i, _OFF["move"] + 1]) << 8  # little-endian uint16
            rows.append((src[i], moves[off[i]:off[i + 1]], scores[off[i]:off[i + 1]], elo[i], mv))
            if len(rows) >= args.n:
                break
        if len(rows) >= args.n:
            break

    n = len(rows)
    bits = np.zeros((n, N_BIT_PLANES), np.uint64)
    aux = np.zeros((n, 8), np.float32)
    ours_l, lc0_l, score_l, lens = [], [], [], []
    played = np.zeros(n, np.int64)
    t0 = time.time()
    for i, (src, sf_moves, sf_scores, _, human_move) in enumerate(rows):
        g, p = shards[src[0]]
        board = board_with_history(g, p, int(src[1]))
        planes = encode_planes(board)
        bits[i] = np.packbits(planes[:N_BIT_PLANES].reshape(N_BIT_PLANES, 64).astype(np.uint8),
                              axis=1, bitorder="little").view(np.uint64)[:, 0]
        aux[i] = planes[N_BIT_PLANES:, 0, 0]
        score_of = dict(zip(sf_moves.tolist(), sf_scores.tolist()))
        moves = list(board.legal_moves)
        ours = [encode_move(m, board.turn) for m in moves]
        ours_l += ours
        lc0_l += [lc0_index(board, m) for m in moves]
        score_l += [score_of[o] for o in ours]
        lens.append(len(moves))
        played[i] = ours.index(human_move)
        if (i + 1) % 5000 == 0:
            print(f"packed {i + 1}/{n} ({time.time() - t0:.0f}s)", flush=True)
    offsets = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    np.savez_compressed(args.out, bits=bits, aux=aux, offsets=offsets, ours=np.array(ours_l, np.int16),
                        lc0=np.array(lc0_l, np.int16), sf_cp=np.array(score_l, np.int16),
                        played_local=played, mover_elo=np.array([r[3] for r in rows], np.int16),
                        sf_nodes_per_move=meta["nodes_per_move"])
    print(f"wrote {args.out}: {n} positions, {os.path.getsize(args.out) / 1e6:.1f} MB")


def unpack_planes(bits, aux):
    """[N,104] uint64 + [N,8] -> [N,112,8,8] float32."""
    n = len(bits)
    b = np.unpackbits(bits.view(np.uint8).reshape(n, N_BIT_PLANES, 8), axis=2, bitorder="little")
    planes = np.empty((n, 112, 64), np.float32)
    planes[:, :N_BIT_PLANES] = b
    planes[:, N_BIT_PLANES:] = aux[:, :, None]
    return planes.reshape(n, 112, 8, 8)


# ------------------------------------------------------------------------------------ run
def gpu_memory_mb():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        return float(out.strip().splitlines()[0])
    except Exception:  # noqa: BLE001 - no GPU / no nvidia-smi
        return None


def segment_stats(p, seg, n, offsets):
    """Top-k mass, effective number of moves, tail mass for a per-position distribution."""
    order = np.lexsort((-p, seg))
    ps = p[order]
    rank = np.arange(len(ps)) - offsets[:-1][seg[order]]
    top = {k: np.bincount(seg[order], weights=ps * (rank < k), minlength=n) for k in (1, 3, 5, 8, 16)}
    H = -np.bincount(seg, weights=p * np.log(p + 1e-12), minlength=n)
    return dict(**{f"top{k}_mass": float(v.mean()) for k, v in top.items()},
                top1_mass_p10=float(np.percentile(top[1], 10)), top1_mass_p90=float(np.percentile(top[1], 90)),
                effective_moves=float(np.exp(H).mean()),
                mass_beyond_top16_p99=float(np.percentile(1 - top[16], 99)))


def run(args):
    from .leela import LeelaONNX
    d = np.load(args.bench)
    offsets, lc0, sf_cp = d["offsets"], d["lc0"].astype(np.int64), d["sf_cp"].astype(np.float64)
    n = len(offsets) - 1
    lens = np.diff(offsets)
    seg = np.repeat(np.arange(n), lens)
    wp = win_prob(np.clip(sf_cp, -3000, 3000))
    best = np.full(n, -1.0)
    np.maximum.at(best, seg, wp)
    loss = best[seg] - wp  # expected-score loss of each legal move

    providers = args.providers.split(",")
    mem0 = gpu_memory_mb()
    net = LeelaONNX(args.onnx, providers=providers, threads=args.threads)
    actual = net.session.get_providers()
    planes = unpack_planes(d["bits"], d["aux"])

    # throughput per batch size (on a fixed slice, after warm-up)
    speed = {}
    for bs in [int(b) for b in args.batches.split(",")]:
        x = planes[:bs] if bs <= n else np.repeat(planes, bs // n + 1, axis=0)[:bs]
        net.run(x)
        reps = max(1, args.min_positions // bs)
        t = time.time()
        for _ in range(reps):
            net.run(x)
        dt = time.time() - t
        speed[bs] = dict(positions_per_s=round(reps * bs / dt, 1), gpu_mem_mb=gpu_memory_mb())
        print(f"batch {bs}: {reps * bs / dt:,.0f} positions/s, gpu mem {speed[bs]['gpu_mem_mb']} MB", flush=True)

    # full pass for quality and concentration
    bs = max(int(b) for b in args.batches.split(","))
    logits = np.empty((n, 1858), np.float32)
    wdl = np.empty((n, 3), np.float32)
    for s in range(0, n, bs):
        logits[s:s + bs], wdl[s:s + bs] = net.run(planes[s:s + bs])
    z = logits[seg, lc0]
    zmax = np.full(n, -np.inf)
    np.maximum.at(zmax, seg, z)
    e = np.exp(z - zmax[seg])
    p = e / np.bincount(seg, weights=e, minlength=n)[seg]

    pmax = np.zeros(n)
    np.maximum.at(pmax, seg, p)
    argmax = (p == pmax[seg])
    first = np.flatnonzero(argmax)[np.unique(seg[argmax], return_index=True)[1]]
    sf_best = np.flatnonzero(loss == 0)[np.unique(seg[loss == 0], return_index=True)[1]]
    human = offsets[:-1] + d["played_local"]
    quality = dict(
        leela_argmax_loss=float(loss[first].mean()),
        leela_sampling_loss=float(np.bincount(seg, weights=p * loss, minlength=n).mean()),
        leela_argmax_is_sf_best=float((loss[first] == 0).mean()),
        human_played_loss=float(loss[human].mean()),
        human_played_is_sf_best=float((loss[human] == 0).mean()),
        leela_prob_on_sf_best=float(p[sf_best].mean()),
    )
    result = dict(onnx=os.path.basename(args.onnx), providers=actual, positions=n,
                  gpu_mem_before_mb=mem0, throughput=speed, quality=quality,
                  concentration=segment_stats(p, seg, n, offsets))
    print(json.dumps(result, indent=1))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=1)


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pack")
    p.add_argument("--labels", required=True)
    p.add_argument("--n", type=int, default=50000)
    p.add_argument("--out", required=True)
    p = sub.add_parser("run")
    p.add_argument("--bench", required=True)
    p.add_argument("--onnx", required=True)
    p.add_argument("--providers", default="CPUExecutionProvider")
    p.add_argument("--batches", default="64,256")
    p.add_argument("--min-positions", type=int, default=2048, help="per throughput measurement")
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    (pack if args.cmd == "pack" else run)(args)


if __name__ == "__main__":
    main()
