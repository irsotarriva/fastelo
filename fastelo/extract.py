"""Extract filtered, labeled positions from a Lichess dump into memmappable shards.

    python -m fastelo.extract data/raw/lichess_db_standard_rated_2025-11.pgn.zst \
        --out data/processed/2025-11 --max-games 200000

Output (per shard_XXXXX/ directory):
  positions.npy  structured POS_DTYPE, one row per ply (position before the move)
  games.npy      structured GAME_DTYPE, one row per game; rows [start, start+n_plies)
                 of positions.npy belong to it
plus stats.json with filter counts and rating / time-control histograms.
"""
import argparse
import json
import multiprocessing as mp
import os
import time
import zlib
from collections import Counter
from dataclasses import asdict

import chess
import numpy as np

from .board_codec import encode_board
from .pgn_stream import EVAL_NONE, FilterConfig, check_headers, iter_raw_games, parse_movetext
from .vocab import encode_move

POS_DTYPE = np.dtype([
    ("occ", "<u8"), ("pieces", "u1", (16,)), ("flags", "u1"), ("ep", "u1"),
    ("halfmove", "u1"), ("move", "<u2"),
    ("clock", "<u2"),   # mover's remaining seconds before this move
    ("spent", "<u2"),   # seconds the mover spent on this move (net of increment)
    ("eval", "<i2"),    # Lichess engine eval after this move, white POV centipawns
                        # (+-30000 minus n for mate in n, pgn_stream.EVAL_NONE if not analysed)
])
GAME_DTYPE = np.dtype([
    ("id", "S8"), ("white_elo", "<u2"), ("black_elo", "<u2"), ("base", "<u2"), ("inc", "u1"),
    ("result", "i1"),   # +1 white win, 0 draw, -1 black win
    ("tc", "u1"),       # index into pgn_stream.TC_CATEGORIES
    ("termination", "u1"),  # 0 normal, 1 time forfeit
    ("split", "u1"),    # 0 train, 1 val, 2 test
    ("has_eval", "u1"),
    ("white_diff", "<i2"), ("black_diff", "<i2"),  # rating change; pgn_stream.DIFF_MISSING if absent
    ("white_id", "<u8"), ("black_id", "<u8"),      # hashed usernames
    ("bots", "u1"),     # bit0 white is a BOT account, bit1 black is
    ("tail_enriched", "u1"),  # 1 = from a tail-only extraction (skewed Elo mix; set by build_dataset)
    ("n_plies", "<u2"), ("start", "<i8"),
])
SPLITS = ("train", "val", "test")


def conform_games(g):
    """Upgrade game rows from an older stage-1 schema; fields it lacked are zero
    (e.g. `bots` for extractions made before bot support: all human). Always returns a
    writable in-memory copy (the input is usually a read-only memmap)."""
    if g.dtype == GAME_DTYPE:
        return np.array(g)
    out = np.zeros(len(g), GAME_DTYPE)
    for name in g.dtype.names:
        if name in GAME_DTYPE.names:
            out[name] = g[name]
    return out
TC_INDEX = {c: i for i, c in enumerate(("ultrabullet", "bullet", "blitz", "rapid", "classical", "correspondence"))}


def game_split(game_id: str) -> int:
    h = zlib.crc32(game_id.encode()) % 100
    return 0 if h < 96 else (1 if h < 98 else 2)


def process_game(info, movetext, min_plies):
    """Return (game_row_tuple, pos_rows_list) or (None, reason)."""
    sans, clocks, evals = parse_movetext(movetext)
    if clocks is None:
        return None, "no_clock"
    if len(sans) < min_plies:
        return None, "too_short"
    board = chess.Board()
    rows = []
    prev_clock = [info.base, info.base]  # remaining before move, indexed by ply parity
    try:
        for ply, (san, clk, ev) in enumerate(zip(sans, clocks, evals)):
            move = board.parse_san(san)
            occ, pieces, flags, ep = encode_board(board)
            side = ply & 1
            before = prev_clock[side]
            spent = max(0, before + info.inc - clk) if ply >= 2 else 0
            rows.append((occ, np.frombuffer(pieces, np.uint8), flags, ep,
                         min(board.halfmove_clock, 255), encode_move(move, board.turn),
                         min(before, 65535), min(spent, 65535), ev))
            prev_clock[side] = clk
            board.push(move)
    except ValueError:
        return None, "illegal_san"
    game = (info.game_id, info.white_elo, info.black_elo, min(info.base, 65535), min(info.inc, 255),
            info.result, TC_INDEX[info.tc], 0 if info.termination == "Normal" else 1,
            game_split(info.game_id), int(any(e != EVAL_NONE for e in evals)),
            info.white_diff, info.black_diff, info.white_id, info.black_id,
            int(info.white_bot) | int(info.black_bot) << 1, 0, len(rows), 0)
    return game, rows


def _worker(args):
    chunk, min_plies = args
    games, positions, reasons = [], [], Counter()
    for info, movetext in chunk:
        # One malformed game must never take down a multi-hour run: convert per game and
        # count anything unexpected as a rejection reason instead.
        try:
            game, rows = process_game(info, movetext, min_plies)
            if game is None:
                reasons[rows] += 1
                continue
            g = np.array([game], dtype=GAME_DTYPE)
            p = np.array(rows, dtype=POS_DTYPE)
        except Exception as e:  # noqa: BLE001
            reasons[f"error_{type(e).__name__}"] += 1
            continue
        games.append(g)
        positions.append(p)
    if not games:
        return np.zeros(0, GAME_DTYPE), np.zeros(0, POS_DTYPE), reasons
    return np.concatenate(games), np.concatenate(positions), reasons


class ShardWriter:
    def __init__(self, out_dir, positions_per_shard):
        self.out_dir, self.limit = out_dir, positions_per_shard
        self.games, self.positions, self.n_pos, self.shard_idx = [], [], 0, 0
        os.makedirs(out_dir, exist_ok=True)

    def add(self, g, p):
        self.games.append(g)
        self.positions.append(p)
        self.n_pos += len(p)
        if self.n_pos >= self.limit:
            self.flush()

    def flush(self):
        if not self.games:
            return
        g = np.concatenate(self.games)
        p = np.concatenate(self.positions)
        g["start"] = np.concatenate([[0], np.cumsum(g["n_plies"].astype(np.int64))[:-1]])
        d = os.path.join(self.out_dir, f"shard_{self.shard_idx:05d}")
        os.makedirs(d, exist_ok=True)
        np.save(os.path.join(d, "games.npy"), g)
        np.save(os.path.join(d, "positions.npy"), p)
        self.shard_idx += 1
        self.games, self.positions, self.n_pos = [], [], 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("zst")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-games", type=int, default=0, help="stop after this many games are read (0 = all)")
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 2))
    ap.add_argument("--chunk", type=int, default=500)
    ap.add_argument("--min-plies", type=int, default=10)
    ap.add_argument("--positions-per-shard", type=int, default=20_000_000)
    ap.add_argument("--tc", default="blitz,rapid,classical")
    ap.add_argument("--stats-only", action="store_true", help="header statistics only, no move parsing")
    ap.add_argument("--bots", choices=["exclude", "human_vs_bot", "only_human_vs_bot"], default="exclude")
    ap.add_argument("--tail-elo", type=int, nargs=2, metavar=("LO", "HI"), default=None,
                    help="keep human games only if a player is <= LO or >= HI")
    args = ap.parse_args()

    cfg = FilterConfig(keep_tc=tuple(args.tc.split(",")), bots=args.bots,
                       tail_elo=tuple(args.tail_elo) if args.tail_elo else None)
    reasons = Counter()
    tc_counts = Counter()
    elo_hist_all = Counter()   # (tc, elo bucket) over every rated human game, before tc filter
    elo_hist_kept = Counter()  # elo bucket over players of kept games
    plies_hist = Counter()
    n_read = n_kept = n_pos = 0
    writer = None if args.stats_only else ShardWriter(args.out, args.positions_per_shard)
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()

    def chunks():
        nonlocal n_read
        chunk = []
        for headers, movetext in iter_raw_games(args.zst):
            n_read += 1
            info, reason = check_headers(headers, cfg)
            if info is not None and reason in (None, "time_control", "elo_range"):
                tc_counts[info.tc] += 1
                for e in (info.white_elo, info.black_elo):
                    elo_hist_all[f"{info.tc}:{e // 100 * 100}"] += 1
            if reason:
                reasons[reason] += 1
            elif not args.stats_only:
                chunk.append((info, movetext))
                if len(chunk) >= args.chunk:
                    yield chunk, args.min_plies
                    chunk = []
            if args.max_games and n_read >= args.max_games:
                break
            if n_read % 200_000 == 0:
                print(f"read {n_read:,} games, kept {n_kept:,}, positions {n_pos:,}, "
                      f"{n_read / (time.time() - t0):,.0f} games/s", flush=True)
        if chunk:
            yield chunk, args.min_plies

    if args.stats_only:
        for _ in chunks():
            pass
    else:
        with mp.Pool(args.workers) as pool:
            for g, p, r in pool.imap(_worker, chunks(), chunksize=1):
                reasons.update(r)
                n_kept += len(g)
                n_pos += len(p)
                for e in np.concatenate([g["white_elo"], g["black_elo"]]):
                    elo_hist_kept[int(e) // 100 * 100] += 1
                for n in g["n_plies"]:
                    plies_hist[int(n) // 10 * 10] += 1
                writer.add(g, p)
        writer.flush()

    elapsed = time.time() - t0
    stats = dict(
        source=os.path.basename(args.zst), filter=asdict(cfg), min_plies=args.min_plies,
        games_read=n_read, games_kept=n_kept, positions=n_pos,
        bytes_per_position=POS_DTYPE.itemsize, bytes_per_game=GAME_DTYPE.itemsize,
        dataset_bytes=n_pos * POS_DTYPE.itemsize + n_kept * GAME_DTYPE.itemsize,
        seconds=round(elapsed, 1), games_per_second=round(n_read / max(elapsed, 1e-9)),
        rejected=dict(reasons), tc_counts=dict(tc_counts),
        elo_hist_by_tc=dict(sorted(elo_hist_all.items())),
        elo_hist_kept=dict(sorted(elo_hist_kept.items())), plies_hist=dict(sorted(plies_hist.items())),
    )
    with open(os.path.join(args.out, "stats.json" if not args.stats_only else "header_stats.json"), "w") as f:
        json.dump(stats, f, indent=1)
    print(json.dumps({k: v for k, v in stats.items() if not k.startswith(("elo_hist", "plies"))}, indent=1))


if __name__ == "__main__":
    main()
