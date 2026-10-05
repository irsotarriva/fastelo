"""Stockfish labels for positions from real games (no engine games: the positions keep
the full off-policy variety of human play).

    python -m fastelo.sf_label --data data/rapid_v1 --split train \
        --out data/sf_labels/train_n1000 --positions 50000000 --nodes 1000 --workers 12

For every sampled position Stockfish scores *every* legal move (multi-PV = all moves,
`nodes` per move), so any soft target (any temperature, expected-score weighting) can be
built at training time. Positions are sampled uniformly from the dataset, exact duplicates
(mostly opening positions) are dropped, and the order is fixed by the seed so an
interrupted run resumes where it stopped (finished chunks are skipped).

Output per chunk directory (CSR layout):
  records.npy  [N, 29] uint8   the position (build_dataset.POS_DTYPE bytes; mover's perspective)
  mover_elo.npy [N] uint16     rating of the human who faced it (analysis only)
  source.npy   [N, 2] int64    (shard number, position index) in the dataset split, so the game
                               history can be recovered (Leela needs the previous 7 positions)
  offsets.npy  [N+1] int64     moves/scores of position i are [offsets[i], offsets[i+1])
  moves.npy    [M] uint16      vocab index (mover perspective)
  scores.npy   [M] int16       centipawns, mover POV, after the move (+-30000-n for mate in n)
"""
import argparse
import json
import multiprocessing as mp
import os
import time

import chess
import chess.engine
import numpy as np
import torch

from .board_codec import board_from_codes
from .eval_strength import DEFAULT_ENGINE
from .build_dataset import FLAG_BLACK
from .features import _OFF, REC, decode_records
from .pgn_stream import MATE_SCORE
from .shards import load_shards
from .vocab import encode_move

_engine = None


def _init(path, nodes, hash_mb):
    global _engine, _nodes
    _engine = chess.engine.SimpleEngine.popen_uci(path)
    _engine.configure({"Threads": 1, "Hash": hash_mb})
    _nodes = nodes


def pov_cp(score: chess.engine.PovScore) -> int:
    """Mover's (white, since boards are mover-perspective) score in cp, mates -> +-(30000 - n)."""
    s = score.white()
    if s.is_mate():
        n = s.mate()
        return MATE_SCORE - min(abs(n), 999) if n > 0 else -MATE_SCORE + min(abs(n), 999)
    return max(-MATE_SCORE + 1000, min(MATE_SCORE - 1000, s.score()))


def _label(fen):
    """Score for every legal move. With a small node budget Stockfish can stop before it has
    reported every multi-PV line, so any missing moves are searched again, restricted to them."""
    board = chess.Board(fen)
    todo = list(board.legal_moves)
    out = {}
    for _ in range(4):
        if not todo:
            break
        infos = _engine.analyse(board, chess.engine.Limit(nodes=_nodes * len(todo)), multipv=len(todo),
                                root_moves=todo)
        for i in infos:
            if "pv" in i and i["pv"][0] not in out:
                out[i["pv"][0]] = pov_cp(i["score"])
        todo = [m for m in todo if m not in out]
    for m in todo:  # still unscored after retries (never observed): one single-move search each
        out[m] = pov_cp(_engine.analyse(board, chess.engine.Limit(nodes=_nodes), root_moves=[m])["score"])
    return [(encode_move(m, chess.WHITE), s) for m, s in out.items()]


def sample_positions(shards, n, seed, min_ply):
    """Deterministic uniform sample of distinct positions (distinct by board bytes, ignoring
    the move played and the halfmove counter), gathered straight from the memmapped shards
    so the split never has to fit in RAM. Returns (records [n,29] uint8, mover Elo [n],
    sources [n,2] int64 = (shard number, position index in that shard))."""
    rng = np.random.default_rng(seed)
    sizes = np.array([len(p) for _, p in shards], np.int64)
    want = np.floor(sizes / sizes.sum() * n * 1.2).astype(np.int64) + 1
    recs, elos, srcs = [], [], []
    for s, ((g, p), k) in enumerate(zip(shards, want)):
        idx = np.unique(rng.integers(0, len(p), size=k))
        gi = np.searchsorted(g["start"], idx, "right") - 1
        keep = (idx - g["start"][gi]) >= min_ply
        idx, gi = idx[keep], gi[keep]
        r = np.ascontiguousarray(p[idx]).view(np.uint8).reshape(-1, REC)
        black = (r[:, _OFF["flags"]] & FLAG_BLACK) != 0
        recs.append(r)
        elos.append(np.where(black, g["black_elo"][gi], g["white_elo"][gi]).astype(np.uint16))
        srcs.append(np.stack([np.full(len(idx), s), idx], 1))
    recs, elos, srcs = np.concatenate(recs), np.concatenate(elos), np.concatenate(srcs)
    order = rng.permutation(len(recs))
    recs, elos, srcs = recs[order], elos[order], srcs[order]
    kb = _OFF["halfmove"]  # key = occ, pieces, flags, ep (everything before halfmove/move)
    keys = np.ascontiguousarray(recs[:, :kb]).view(np.dtype((np.void, kb))).ravel()
    _, first = np.unique(keys, return_index=True)
    keep = np.sort(first)[:n]
    return recs[keep], elos[keep], srcs[keep]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", required=True)
    ap.add_argument("--positions", type=int, required=True)
    ap.add_argument("--nodes", type=int, default=1000, help="engine nodes per legal move")
    ap.add_argument("--min-ply", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=200_000)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--hash", type=int, default=16)
    ap.add_argument("--engine", default=DEFAULT_ENGINE)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    meta_path = os.path.join(args.out, "meta.json")
    meta = dict(data=os.path.abspath(args.data), split=args.split, nodes_per_move=args.nodes,
                engine=os.path.basename(args.engine), seed=args.seed, min_ply=args.min_ply,
                positions=args.positions, chunk=args.chunk)
    if os.path.exists(meta_path):
        old = json.load(open(meta_path))
        if {k: old[k] for k in meta if k != "data"} != {k: meta[k] for k in meta if k != "data"}:
            raise SystemExit(f"{args.out} was started with different settings: {old}")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=1)

    t0 = time.time()
    rec_all, mover_elo_all, src_all = sample_positions(load_shards(os.path.join(args.data, args.split)),
                                                       args.positions, args.seed, args.min_ply)
    print(f"sampled {len(rec_all):,} distinct positions in {time.time() - t0:.0f}s", flush=True)

    n_chunks = (len(rec_all) + args.chunk - 1) // args.chunk
    with mp.Pool(args.workers, initializer=_init, initargs=(args.engine, args.nodes, args.hash)) as pool:
        for c in range(n_chunks):
            d = os.path.join(args.out, f"chunk_{c:05d}")
            if os.path.exists(os.path.join(d, "scores.npy")):
                continue
            sl = slice(c * args.chunk, (c + 1) * args.chunk)
            rec = rec_all[sl]
            dec = decode_records(torch.from_numpy(rec))
            flags = (dec["castling"] @ torch.tensor([1.0, 2, 4, 8])).int()
            fens = [board_from_codes(dec["board"][i].numpy(), int(flags[i]), int(dec["ep"][i]),
                                     int(dec["halfmove"][i])).fen() for i in range(len(rec))]
            tc = time.time()
            results = pool.map(_label, fens, chunksize=64)
            lens = np.array([len(r) for r in results], np.int64)
            offsets = np.concatenate([[0], np.cumsum(lens)])
            moves = np.array([m for r in results for m, _ in r], np.uint16)
            scores = np.array([s for r in results for _, s in r], np.int16)
            tmp = d + ".tmp"
            os.makedirs(tmp, exist_ok=True)
            np.save(os.path.join(tmp, "records.npy"), rec)
            np.save(os.path.join(tmp, "mover_elo.npy"), mover_elo_all[sl])
            np.save(os.path.join(tmp, "source.npy"), src_all[sl])  # (shard, index) in the dataset split
            np.save(os.path.join(tmp, "offsets.npy"), offsets)
            np.save(os.path.join(tmp, "moves.npy"), moves)
            np.save(os.path.join(tmp, "scores.npy"), scores)
            os.replace(tmp, d)
            dt = time.time() - tc
            print(f"chunk {c + 1}/{n_chunks}: {len(rec):,} positions in {dt:.0f}s "
                  f"({len(rec) / dt:,.0f}/s)", flush=True)


if __name__ == "__main__":
    main()
