# fastelo

A chess bot that plays like a human of a chosen rating, and a rating of *your* play read from the
same network after every move, with statistical and systematic error bars.

- **Play it:** [live demo](https://presonal-page-eight.vercel.app/en/projects/fast-elo-estimation-for-chess)
- **Trained models:** [huggingface.co/irsotarriva/fastelo](https://huggingface.co/irsotarriva/fastelo)
- **Report with all measurements:** [ChessFastElo-Report.ipynb](https://github.com/irsotarriva/MLIdeas/blob/main/ChessFastElo-Report.ipynb)
- **Synthetic games dataset:** [fastelo-synthetic-games](https://huggingface.co/datasets/irsotarriva/fastelo-synthetic-games) (CC0)

## What it does

- **The player** is a 7M-parameter transformer that gives the probability of every move for a
  human of a given rating. It first learned chess by imitating Leela Chess Zero on positions from
  human games, then was trained on 500 million human moves from Lichess rapid and classical games.
  A calibration step (no training) makes it give away the same winning chances per move as humans
  of each rating, from 600 to about 2400.
- **The rating** comes from the same network. For the moves one person played, the summed
  log-probability as a function of the rating input is a measurement of that person's rating. It
  uses only their own moves, so it does not depend on how well the opponent plays. Its bias is
  measured on held-out human games and corrected; what is left is quoted as a systematic error.

## Quick start

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or a CUDA build
pip install -r requirements.txt
```

```python
from huggingface_hub import snapshot_download
from fastelo.play import Player, Rater
import chess, numpy as np

root = snapshot_download("irsotarriva/fastelo")
bot = Player(f"{root}/calibration.json")                        # plays at a target rating
rater = Rater(bot, response=f"{root}/rating_response.json")     # rates a player from their moves

board, boards, moves = chess.Board(), [], []
rng = np.random.default_rng(0)
while not board.is_game_over() and len(moves) < 80:
    move = bot.choose(board, ply=len(moves), self_elo=1500, opp_elo=1500, rng=rng)
    boards.append(board.copy(stack=False)); moves.append(move); board.push(move)

r = rater.rating(boards, moves, side=0)                          # white, after each of its moves
print(r["value"][-1], r["lo"][-1], r["hi"][-1], r["syst"][-1])   # value, 1-sigma bounds, systematic
```

Local demo (board, clock, live rating):

```bash
python -m fastelo.demo.app --calibration <model folder>/calibration.json
```

`fastelo.lite` has the same player and rating in plain NumPy, without PyTorch; the web demo runs
it in a small serverless function (`scripts/build_vercel_function.py` builds that function as one
file).

## Layout

| Module | Role |
|---|---|
| `vocab.py` | 1880-move vocabulary from the mover's point of view (black positions are mirrored) |
| `board_codec.py` | compact board encoding and decoders |
| `pgn_stream.py`, `extract.py`, `shards.py` | stage 1: parse a Lichess monthly dump into shards |
| `build_dataset.py` | stage 2: filter, shuffle, write train/val/test shards |
| `features.py` | `GameStore`: a whole split as one tensor, batch decoding on the device |
| `models.py` | `PlayerNet`, the policy network, and its rating encoders |
| `leela.py`, `leela_label.py` | label positions with a Leela Chess Zero network |
| `distill.py` | train the strong player on the Leela labels |
| `train.py` | train the player on human moves, starting from the strong player |
| `runtime.py`, `cloud.py` | training loop: any number of GPUs, resumable, checkpoints to a bucket or folder |
| `sf_label.py`, `teacher_bench.py`, `eval_strength.py` | Stockfish analysis and move-quality measurement |
| `eval_player.py` | likelihood of human moves against the rating input |
| `calibrate.py` | target rating -> rating input, temperature, share of the strong player |
| `likelihood.py` | rating from the likelihood of a player's own moves; measurement on human games |
| `response.py` | bias correction of the rating, statistical and systematic errors |
| `play.py` | `Player`, `Rater`, games between agents |
| `lite.py` | player and rating in NumPy |
| `demo/` | the interactive demo (session logic, board image, Gradio app) |
| `synth.py` | players with separate opening / middlegame / endgame strength; ladder and game generation |
| `export.py` | weights, calibration and model card for publication |

`results/` holds the measured numbers the report notebook reads (training curves, calibration,
closure tests); `scripts/` rebuilds the report and the web function from them.

## Reproducing the pipeline

Data: a [Lichess database](https://database.lichess.org/) monthly dump (`.pgn.zst`).

```bash
# 1. data
python -m fastelo.extract data/raw/lichess_db_standard_rated_2025-11.pgn.zst --out data/processed/2025-11
python -m fastelo.build_dataset data/processed/2025-11 --out data/rapid --tc rapid,classical

# 2. strong player: label positions with Leela, then distil
python -m fastelo.leela_label pack --data data/rapid --split train --out data/leela_parts
python -m fastelo.leela_label run --parts data/leela_parts --onnx <leela net>.onnx --out data/leela_labels
python -m fastelo.distill train --parts data/leela_parts --labels data/leela_labels --run distill

# 3. player: human moves, conditioned on both ratings
python -m fastelo.train --data data/rapid --run player --init-from runs/distill/ckpt.pt --optimizer muon

# 4. calibration against Stockfish-analysed positions (inference only)
python -m fastelo.eval_strength analyse --data data/rapid --out data/sf_test.npz
python -m fastelo.calibrate cache --player runs/player/ckpt.pt --strong runs/distill/ckpt.pt \
    --analysis data/sf_test.npz --out data/calib_cache.npz
python -m fastelo.calibrate fit --cache data/calib_cache.npz --out runs/player/calibration.json

# 5. bias correction of the rating (inference only)
python -m fastelo.likelihood measure --ckpt runs/player/ckpt.pt --data data/rapid --split val --out lik_val.npz
python -m fastelo.likelihood measure --ckpt runs/player/ckpt.pt --data data/rapid --split test --out lik_test.npz
python -m fastelo.response fit --derive lik_val.npz --closure lik_test.npz --out runs/player/rating_response.json
```

Each module's docstring lists its options. The published models were trained on four months of
rapid and classical games (42.7 million games after filtering) in about 33 hours on one RTX 3090.

## Tests

```bash
python -m pytest -q
```

Tests that need Stockfish or a Leela network are skipped when those are not installed under
`tools/`.

## Licence and attribution

GNU GPL v3.0 (see `LICENSE`).

- Data: [Lichess open database](https://database.lichess.org/) (CC0).
- Teacher: [Leela Chess Zero](https://lczero.org/) (GPL-3.0), network T1-256x10-distilled-swa-2432500.
- Calibration analysis: [Stockfish](https://stockfishchess.org/) (GPL-3.0).
