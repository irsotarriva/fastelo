"""Stage 2: filter the extracted month into a shuffled, upload-ready training dataset.

    python -m fastelo.build_dataset data/processed/2025-11 [more stage-1 roots...] --out data/datasets/rapid_v1

Human-vs-bot games (stage 1 `--bots`) are kept, but a bot side is never a label: its moves are
not player targets.

Reads the stage-1 shards strictly sequentially (the source disk is an HDD) and
shuffles games with a two-pass bucket shuffle, so memory stays bounded.

Output layout (every shard is self-contained; the first k train shards are a
uniform random subset, which is what the scaling experiments use):
  <out>/manifest.json
  <out>/{train,val,test}/shard_XXXXX/games.npy      GAME_DTYPE, `start` relative to the shard
  <out>/{train,val,test}/shard_XXXXX/positions.npy  POS_DTYPE (29 B): what training reads
  <out>/{train,val,test}/shard_XXXXX/aux.npy        AUX_DTYPE (6 B): clock/eval, analysis only
"""
import argparse
import json
import os
import shutil
import time

import numpy as np

from . import extract
from .pgn_stream import TC_CATEGORIES
from .shards import load_shards
from .vocab import VOCAB_SIZE

SCHEMA_VERSION = 1
POS_DTYPE = np.dtype([
    ("occ", "<u8"), ("pieces", "u1", (16,)),
    ("flags", "u1"),  # bits 0-3 castling (own K, own Q, opp K, opp Q), bit 4 black to move,
                      # bit 5 PLAYER_OK: use as an player training target (not in time trouble)
    ("ep", "u1"), ("halfmove", "u1"), ("move", "<u2"),
])
AUX_DTYPE = np.dtype([("clock", "<u2"), ("spent", "<u2"), ("eval", "<i2")])
GAME_DTYPE = extract.GAME_DTYPE
FLAG_BLACK = 1 << 4
FLAG_PLAYER_OK = 1 << 5


def block_index(starts, lens):
    """Concatenated ranges [s, s+l) for each (s, l), vectorized."""
    starts = np.asarray(starts, np.int64)
    lens = np.asarray(lens, np.int64)
    ends = np.cumsum(lens)
    return np.repeat(starts - (ends - lens), lens) + np.arange(ends[-1] if len(ends) else 0)


def select_games(games, args, tc_ids, established):
    """Rating-certainty, Elo-range and experience filters apply to the human side(s) only;
    a bot's rating is never used as a label."""
    ok = np.isin(games["tc"], tc_ids)
    for bit, side in ((1, "white"), (2, "black")):
        is_bot = (games["bots"] & bit) != 0
        human_ok = np.abs(games[f"{side}_diff"].astype(np.int32)) <= args.max_rating_diff
        human_ok &= (games[f"{side}_elo"] >= args.min_elo) & (games[f"{side}_elo"] <= args.max_elo)
        if established is not None:
            human_ok &= np.isin(player_key(games[f"{side}_id"], games["tc"]), established)
        ok &= is_bot | human_ok
    return ok


def player_key(ids, tc):
    # Ratings are per time-control pool, so experience is counted per (player, tc).
    return ids ^ (tc.astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("src", nargs="+", help="stage-1 extraction root(s), e.g. several months")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tc", default="rapid,classical")
    ap.add_argument("--max-rating-diff", type=int, default=15,
                    help="both players' |rating change| must be <= this (drops uncertain/new ratings)")
    ap.add_argument("--min-player-games", type=int, default=0,
                    help="both players need this many games in the same time control in the source month")
    ap.add_argument("--min-clock", type=int, default=30,
                    help="moves made with less than this many seconds left are not player targets")
    ap.add_argument("--min-elo", type=int, default=0)
    ap.add_argument("--max-elo", type=int, default=4000)
    ap.add_argument("--shard-positions", type=int, default=40_000_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    tc_ids = [TC_CATEGORIES.index(t) for t in args.tc.split(",")]
    src = []
    for root in args.src:
        stats_path = os.path.join(root, "stats.json")
        tail = bool(json.load(open(stats_path))["filter"].get("tail_elo")) if os.path.exists(stats_path) else False
        for g, p in load_shards(root):
            g = extract.conform_games(np.asarray(g))
            if tail:  # human games from a tail-only extraction (its bot games are unbiased)
                g["tail_enriched"] = (g["bots"] == 0).astype(np.uint8)
            src.append((g, p))
        print(f"source {root}: {'tail-only human games' if tail else 'full'}", flush=True)
    tmp = os.path.join(args.out, "_tmp")
    os.makedirs(tmp, exist_ok=True)

    established = None
    if args.min_player_games > 0:
        keys = []
        for g, _ in src:
            g = np.asarray(g)
            m = np.isin(g["tc"], tc_ids)
            keys += [player_key(g["white_id"][m], g["tc"][m]), player_key(g["black_id"][m], g["tc"][m])]
        uniq, counts = np.unique(np.concatenate(keys), return_counts=True)
        established = uniq[counts >= args.min_player_games]
        print(f"players (per tc): {len(uniq):,}, with >= {args.min_player_games} games: {len(established):,}")

    # Pass 0: size each split so the buckets come out at ~shard_positions.
    split_positions = np.zeros(3, np.int64)
    for g, _ in src:
        g = np.asarray(g)
        sel = select_games(g, args, tc_ids, established)
        for s in range(3):
            split_positions[s] += g["n_plies"][sel & (g["split"] == s)].sum()
    n_buckets = [max(1, int(np.ceil(n / args.shard_positions))) for n in split_positions]
    print("selected positions per split:", dict(zip(extract.SPLITS, split_positions.tolist())),
          "buckets:", n_buckets)

    # Pass 1: stream source shards, scatter selected games to random buckets.
    files = {}

    def append(split, b, kind, arr):
        key = (split, b, kind)
        if key not in files:
            files[key] = open(os.path.join(tmp, f"{extract.SPLITS[split]}_{b:05d}.{kind}"), "wb")
        files[key].write(arr.tobytes())

    for i, (g, p) in enumerate(src):
        g, p = np.asarray(g), np.asarray(p)  # one sequential read per shard
        sel = np.flatnonzero(select_games(g, args, tc_ids, established))
        for s in range(3):
            gs = g[sel[g["split"][sel] == s]]
            if not len(gs):
                continue
            buckets = rng.integers(n_buckets[s], size=len(gs))
            for b in np.unique(buckets):
                gb = gs[buckets == b]
                pb = p[block_index(gb["start"], gb["n_plies"])]
                out = np.empty(len(pb), POS_DTYPE)
                for f in ("occ", "pieces", "ep", "halfmove", "move"):
                    out[f] = pb[f]
                # PlayerNet targets: human moves only, not made in time trouble.
                black_to_move = (pb["flags"] & FLAG_BLACK) != 0
                bots = np.repeat(gb["bots"], gb["n_plies"])
                mover_bot = np.where(black_to_move, bots & 2, bots & 1) != 0
                player_ok = (pb["clock"] >= args.min_clock) & ~mover_bot
                out["flags"] = pb["flags"] | np.where(player_ok, FLAG_PLAYER_OK, 0).astype(np.uint8)
                aux = np.empty(len(pb), AUX_DTYPE)
                for f in AUX_DTYPE.names:
                    aux[f] = pb[f]
                append(s, b, "games", gb)
                append(s, b, "pos", out)
                append(s, b, "aux", aux)
        print(f"pass 1: source shard {i + 1}/{len(src)} done ({time.time() - t0:.0f}s)", flush=True)
    for f in files.values():
        f.close()

    # Pass 2: shuffle games inside each bucket and write final shards.
    summary = {}
    elo_hist = np.zeros(40, np.int64)
    for s, name in enumerate(extract.SPLITS):
        n_games = n_pos = n_player = n_bot_games = n_tail_games = 0
        for b in range(n_buckets[s]):
            base = os.path.join(tmp, f"{name}_{b:05d}")
            if not os.path.exists(base + ".games"):
                continue
            g = np.fromfile(base + ".games", GAME_DTYPE)
            p = np.fromfile(base + ".pos", POS_DTYPE)
            a = np.fromfile(base + ".aux", AUX_DTYPE)
            g["start"] = np.concatenate([[0], np.cumsum(g["n_plies"].astype(np.int64))[:-1]])
            g = g[rng.permutation(len(g))]
            idx = block_index(g["start"], g["n_plies"])
            p, a = p[idx], a[idx]
            g["start"] = np.concatenate([[0], np.cumsum(g["n_plies"].astype(np.int64))[:-1]])
            d = os.path.join(args.out, name, f"shard_{b:05d}")
            os.makedirs(d, exist_ok=True)
            np.save(os.path.join(d, "games.npy"), g)
            np.save(os.path.join(d, "positions.npy"), p)
            np.save(os.path.join(d, "aux.npy"), a)
            n_games += len(g)
            n_bot_games += int((g["bots"] != 0).sum())
            n_tail_games += int(g["tail_enriched"].sum())
            n_pos += len(p)
            n_player += int((p["flags"] & FLAG_PLAYER_OK).astype(bool).sum())
            if s == 0:
                elo_hist += np.bincount(np.minimum(np.concatenate([g["white_elo"], g["black_elo"]]) // 100, 39),
                                        minlength=40)
            for kind in ("games", "pos", "aux"):
                os.remove(f"{base}.{kind}")
        summary[name] = dict(games=n_games, human_vs_bot_games=n_bot_games, tail_enriched_games=n_tail_games,
                             positions=n_pos,
                             player_positions=n_player, shards=n_buckets[s])
        print(name, summary[name], flush=True)
    shutil.rmtree(tmp)

    manifest = dict(
        schema_version=SCHEMA_VERSION, vocab_size=VOCAB_SIZE, source=[os.path.abspath(s) for s in args.src],
        filters={k: v for k, v in vars(args).items() if k not in ("src", "out")},
        splits=summary,
        bytes=dict(position=POS_DTYPE.itemsize, aux=AUX_DTYPE.itemsize, game=GAME_DTYPE.itemsize),
        dtypes=dict(positions=POS_DTYPE.descr, aux=AUX_DTYPE.descr, games=GAME_DTYPE.descr),
        train_elo_hist_100=dict(zip((np.arange(40) * 100).tolist(), elo_hist.tolist())),
        seconds=round(time.time() - t0, 1),
    )
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)


if __name__ == "__main__":
    main()
