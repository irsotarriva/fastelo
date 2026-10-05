"""Convert the synthetic phase-varying games (fastelo.synth generate) to Parquet for Hugging Face.

    python scripts/export_synth_dataset.py --src data/synth_v1 \
        --labels runs/cloud/ladder_3000/labels.json --out data/hf_dataset [--upload --repo <user>/<name>]

One row per game: the moves in UCI, the result, each side's three phase strengths and its rating
label. Released under CC0.
"""
import argparse
import glob
import json
import os

import chess
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from fastelo.vocab import MOVES, mirror_move

UCI_WHITE = np.array(MOVES)
UCI_BLACK = np.array([mirror_move(chess.Move.from_uci(m)).uci() for m in MOVES])
RESULT = {1: "1-0", 0: "1/2-1/2", -1: "0-1"}


def rows(shard):
    g = np.load(os.path.join(shard, "games.npy"))
    p = np.load(os.path.join(shard, "positions.npy"), mmap_mode="r")
    s = np.load(os.path.join(shard, "specs.npy"))
    assert len(g) == len(s), shard
    mv = np.asarray(p["move"])
    for i in range(len(g)):
        a, n = int(g["start"][i]), int(g["n_plies"][i])
        idx = mv[a:a + n]
        uci = np.where(np.arange(n) % 2 == 0, UCI_WHITE[idx], UCI_BLACK[idx])
        yield dict(game_id=g["id"][i].decode(), moves=" ".join(uci), plies=n, result=RESULT[int(g["result"][i])],
                   adjudicated=bool(n >= 200),
                   white_rating=int(g["white_elo"][i]), black_rating=int(g["black_elo"][i]),
                   white_opening_elo=float(s[i, 0]), white_middlegame_elo=float(s[i, 1]),
                   white_endgame_elo=float(s[i, 2]), black_opening_elo=float(s[i, 3]),
                   black_middlegame_elo=float(s[i, 4]), black_endgame_elo=float(s[i, 5]))


def card(labels, counts, repo_model):
    c = labels["linear_coef"]
    return f'''---
license: cc0-1.0
task_categories:
  - other
tags:
  - chess
  - synthetic
  - elo
pretty_name: fastelo synthetic games with phase-dependent strength
size_categories:
  - 100K<n<1M
configs:
  - config_name: default
    data_files:
      - split: train
        path: data/train-*.parquet
      - split: validation
        path: data/validation-*.parquet
---

# Synthetic chess games with known, phase-dependent strength

{counts["train"]:,} training and {counts["validation"]:,} validation games played by a human-like chess policy
([{repo_model}](https://huggingface.co/{repo_model})) in which **each player has three separate
strengths**: one for the opening, one for the middlegame and one for the endgame. Every game
records those six numbers and a rating label for each side.

Why this exists: in human games a player's opening, middlegame and endgame strength go together,
so human databases cannot tell how much each phase is worth, or whether a method that rates
players from their games can be fooled by one of them (for example by memorised opening theory).
Here the phases vary independently, and in half of the games the opponent's strength is
unrelated to the player's, which human matchmaking never provides. In this project the games
showed that a network rating whole games gave the opening about twice the weight that results
justify.

## Columns

| Column | Meaning |
|---|---|
| `game_id` | identifier |
| `moves` | the game in UCI notation, space-separated, from the initial position |
| `plies` | number of half-moves |
| `result` | `1-0`, `0-1` or `1/2-1/2` |
| `adjudicated` | true if the game reached 200 plies and was scored by material (3 points or more = win) |
| `white_rating`, `black_rating` | rating label of each player (formula below), Lichess rapid scale |
| `white_opening_elo`, `white_middlegame_elo`, `white_endgame_elo` | the target strength White played at in each phase |
| `black_opening_elo`, `black_middlegame_elo`, `black_endgame_elo` | the same for Black |

## How the games were made

- **Players.** A middlegame strength uniform in 600-2300; opening and endgame strengths offset from
  it by a normal distribution (sigma 200 Elo); 8% of players have one phase moved by 500-1000 Elo
  in either direction (for example strong theory followed by weak play). All strengths are clipped
  to 500-2500.
- **Phase blending.** The strength used for a move is a smooth blend of the three: the opening
  weight is a sigmoid over the ply (centre 24, width 4), the endgame weight a sigmoid over the
  remaining non-pawn material (centre 20 points, width 4), the rest is middlegame.
- **Moves.** Sampled from the calibrated fastelo policy at that strength. The calibration maps a
  target Elo to settings whose Stockfish-measured move quality matches humans of that Lichess
  rapid rating (600 to about 2400).
- **Opponents.** In half of the games the opponent is an unrelated random player; in the other
  half its middlegame strength is within about 200 Elo of the player's.
- **Game end.** Checkmate, stalemate, insufficient material, threefold repetition, the 50-move
  rule, or adjudication by material at 200 plies. There are no clocks and no resignations.

## Rating labels

A separate ladder of {labels["players"]:,} such players, each playing 40 games against bots of known
strength (opponent re-matched to the player's current rating after every game), gave each one a
result-based rating. A linear fit of that rating on the three phase strengths is the label used
here:

    rating = {c[0]:.1f} + {c[1]:.3f} * opening + {c[2]:.3f} * middlegame + {c[3]:.3f} * endgame

The fit leaves about {labels["model_error"]:.0f} Elo of error beyond the ladder's own measurement noise
(about {labels["ladder_se"]:.0f} Elo per player), so treat the labels as accurate to roughly that.

## Limitations

- These are games between neural-network players, not humans; "Elo" means the calibrated target
  strength, on the Lichess rapid scale.
- The policy tops out at about 2400-2500; targets near the top are played mostly by a stronger,
  less human-like network.
- Labels come from a fitted formula, not from each player's own ladder.

## Licence

CC0 1.0: no rights reserved. The games contain no positions or moves taken from other databases.
The policy that played them was trained on the [Lichess open database](https://database.lichess.org/)
(CC0) with [Leela Chess Zero](https://lczero.org/) as a teacher.
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", default="data/hf_dataset")
    ap.add_argument("--repo", default="irsotarriva/fastelo-synthetic-games")
    ap.add_argument("--model-repo", default="irsotarriva/fastelo")
    ap.add_argument("--rows-per-file", type=int, default=100_000)
    ap.add_argument("--upload", action="store_true")
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out, "data"), exist_ok=True)
    counts = {}
    for split, name in (("train", "train"), ("val", "validation")):
        shards = sorted(glob.glob(os.path.join(args.src, split, "shard_*")))
        buf, files, total = [], [], 0

        def flush():
            path = os.path.join(args.out, "data", f"{name}-{len(files):05d}.parquet")
            pq.write_table(pa.Table.from_pylist(buf), path, compression="zstd")
            files.append(path)
            buf.clear()

        for sh in shards:
            for r in rows(sh):
                buf.append(r)
                total += 1
                if len(buf) >= args.rows_per_file:
                    flush()
        if buf:
            flush()
        n = len(files)  # the Hub's split pattern wants "-of-N" names
        for i, f in enumerate(files):
            os.replace(f, os.path.join(args.out, "data", f"{name}-{i:05d}-of-{n:05d}.parquet"))
        counts[name] = total
        print(f"{name}: {total:,} games in {n} file(s)", flush=True)
    labels = json.load(open(args.labels))
    with open(os.path.join(args.out, "README.md"), "w", encoding="utf-8") as f:
        f.write(card(labels, counts, args.model_repo))
    size = sum(os.path.getsize(p) for p in glob.glob(os.path.join(args.out, "data", "*.parquet"))) / 1e6
    print(f"dataset folder: {args.out} ({size:.0f} MB)")
    if args.upload:
        from huggingface_hub import HfApi
        api = HfApi(token=os.environ.get("HF_API_KEY") or os.environ.get("HF_TOKEN"))
        api.create_repo(args.repo, repo_type="dataset", exist_ok=True)
        api.upload_folder(folder_path=args.out, repo_id=args.repo, repo_type="dataset",
                          commit_message="Synthetic games with phase-dependent strength")
        print(f"uploaded: https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
