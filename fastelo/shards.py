"""Read extracted shards (memory-mapped) and sanity-check them.

    python -m fastelo.shards data/processed/sample100k
"""
import argparse
import glob
import os

import numpy as np

from .board_codec import board_from_codes, decode_boards
from .pgn_stream import EVAL_NONE, TC_CATEGORIES
from .vocab import legal_mask


def load_shards(root):
    """List of (games, positions) memmapped arrays, one pair per shard."""
    out = []
    for d in sorted(glob.glob(os.path.join(root, "shard_*"))):
        out.append((np.load(os.path.join(d, "games.npy"), mmap_mode="r"),
                    np.load(os.path.join(d, "positions.npy"), mmap_mode="r")))
    return out


def verify(shards, n=2000, seed=0):
    """Every stored move must be legal in its decoded board; game rows must tile positions."""
    rng = np.random.default_rng(seed)
    for games, pos in shards:
        assert games["start"][0] == 0
        assert np.all(games["start"][1:] == games["start"][:-1] + games["n_plies"][:-1])
        assert games["start"][-1] + games["n_plies"][-1] == len(pos)
        idx = rng.choice(len(pos), size=min(n, len(pos)), replace=False)
        p = pos[np.sort(idx)]
        codes = decode_boards(p["occ"], p["pieces"])
        for row, c in zip(p, codes):
            board = board_from_codes(c, int(row["flags"]), int(row["ep"]), int(row["halfmove"]))
            assert legal_mask(board)[row["move"]], board.fen()


def summary(shards):
    games = np.concatenate([g for g, _ in shards])
    pos_n = sum(len(p) for _, p in shards)
    evals = np.concatenate([p["eval"] for _, p in shards])
    elos = np.concatenate([games["white_elo"], games["black_elo"]]).astype(int)
    print(f"games {len(games):,}  positions {pos_n:,}  plies/game {pos_n / len(games):.1f}")
    print("split:", {s: int((games["split"] == i).sum()) for i, s in enumerate(("train", "val", "test"))})
    print("tc:", {TC_CATEGORIES[t]: int((games["tc"] == t).sum()) for t in np.unique(games["tc"])})
    print(f"analysed games {games['has_eval'].mean():.1%}, positions with eval {(evals != EVAL_NONE).mean():.1%}")
    print("elo percentiles 1/10/50/90/99:", np.percentile(elos, [1, 10, 50, 90, 99]).astype(int))
    hist, edges = np.histogram(elos, bins=np.arange(400, 3300, 200))
    for h, e in zip(hist, edges):
        print(f"  {e:4d}-{e + 199:4d} {h / len(elos):6.1%} {'#' * int(60 * h / hist.max())}")
    gap = np.abs(games["white_elo"].astype(int) - games["black_elo"].astype(int))
    print("|white-black| elo gap percentiles 50/90/99:", np.percentile(gap, [50, 90, 99]).astype(int))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    args = ap.parse_args()
    shards = load_shards(args.root)
    summary(shards)
    verify(shards)
    print("verify OK")
