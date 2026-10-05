"""Package the models and the demo for Hugging Face.

    python -m fastelo.export --player runs/player/ckpt.pt --strong runs/distill/ckpt.pt \
        --calibration runs/player/calibration.json \
        --repo <user>/fastelo --out data/hf [--results results.json] [--upload]

Writes two folders:
  <out>/model  weights only, half precision, safetensors (+ a .json with the architecture), the
               calibration, the model card and the licence -> a Hugging Face *model* repo
  <out>/space  the Gradio demo (this package + app.py + requirements) -> a Hugging Face *Space*
               that downloads the weights from the model repo when it starts
--upload creates/updates both repos (token from HF_API_KEY or HF_TOKEN).
"""
import argparse
import json
import os
import shutil

import torch

LICENSE_FILE = os.path.join(os.path.dirname(__file__), "..", "LICENSE")


def export_weights(ckpt_path, out_base):
    from safetensors.torch import save_file
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = {k: (v.half() if v.is_floating_point() else v).contiguous() for k, v in ck["model"].items()}
    save_file(state, out_base + ".safetensors")
    info = dict(kind="player", model_cfg=ck["model_cfg"],
                step=int(ck["step"]),
                parameters=int(sum(v.numel() for v in ck["model"].values())))
    if "distill" in ck:
        info["distill"] = ck["distill"]
    with open(out_base + ".json", "w") as f:
        json.dump(info, f, indent=1)
    return info


SPACE_APP = '''"""Hugging Face Space entry point: fetch the weights, start the demo."""
import os

from huggingface_hub import snapshot_download

from fastelo.demo.app import build_app
from fastelo.play import CalibratedPolicy, MoveRater

MODEL_REPO = os.environ.get("FASTELO_MODEL_REPO", "{repo}")
# FASTELO_MODEL_DIR: a local copy of the model repo (for running the Space code offline)
root = os.environ.get("FASTELO_MODEL_DIR") or snapshot_download(MODEL_REPO)
policy = CalibratedPolicy(os.path.join(root, "calibration.json"))
demo = build_app(policy, MoveRater(policy, response=os.path.join(root, "rating_response.json")))

if __name__ == "__main__":
    demo.launch()
'''

SPACE_REQUIREMENTS = '''--extra-index-url https://download.pytorch.org/whl/cpu
torch
chess>=1.11
numpy>=2.0
matplotlib>=3.9
zstandard>=0.23
safetensors>=0.4
huggingface_hub>=0.25
pillow
'''


def space_readme(repo, gradio_version):
    return f'''---
title: fastelo - a chess bot that rates you as you play
emoji: "\\u265f"
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: {gradio_version}
app_file: app.py
pinned: false
license: gpl-3.0
models:
  - {repo}
short_description: Play a human-like bot and get a live Elo estimate
---

# fastelo demo

Play rapid games against a bot that plays like a human of your level. After every move the same
network works out at which rating your moves are most likely, which gives an estimate of your
Lichess-style rating with an error bar; each game is one measurement and all your games are
combined into one rating. The bot re-targets your current best estimate before each of its moves.

Models, training details and measured results: [{repo}](https://huggingface.co/{repo}).

Licence: GPL-3.0. Trained on the [Lichess open database](https://database.lichess.org/) (CC0) with
[Leela Chess Zero](https://lczero.org/) (network T1-256x10-distilled) as the teacher.
'''


def rating_section(resp, study):
    """Model-card text for the rating: closure tables from fastelo.response fit, and why the rating
    is read from the player and not from a separate network."""
    if not resp:
        return ""
    k = "40" if "40" in resp["closure"] else sorted(resp["closure"])[0]
    raw, cor, cmb = resp["closure"][k]["raw"], resp["closure"][k]["corrected"], {c["band"]: c for c in resp["combined"][k]}
    rows = "\n".join(
        f"| {a['band']} | {a['bias']:+.0f} | {b['bias']:+.0f} | −{b['mean_minus']:.0f} / +{b['mean_plus']:.0f} | "
        f"{100 * b['coverage']:.0f}% | {resp['systematic'][k][b['band']]:.0f} | {cmb[b['band']]['mae']:.0f} |"
        for a, b in zip(raw, cor) if b["band"] in cmb)
    why = ""
    if study:
        e, m = study["estimator_vs_fixed_bot"], study["move_rating_vs_fixed_bot"]
        line = lambda d, u: " / ".join(f"{d['user'][u][d['bot'].index(x)]}" for x in m["bot"])
        why = f"""
### Why there is no separate rating network

A causal transformer that reads the whole game and predicts both ratings was trained first. On
human games it looked good (about 190 Elo of error after a game), but most of what it knows is
the *level of the game*: human opponents are matched, so the opponent's play gives the rating
away. Against a bot whose strength follows the estimate that is circular, and the estimate
drifts. The rating from the player's own moves does not depend on the opponent:

| Simulated user (20 moves) | Whole-game network, bot at {" / ".join(str(x) for x in m["bot"])} | Own moves, bot at {" / ".join(str(x) for x in m["bot"])} |
|---|---|---|
| engine's best moves | {line(e, "best")} | {line(m, "best")} |
| 2200 | {line(e, "2200")} | {line(m, "2200")} |
| 1500 | {line(e, "1500")} | {line(m, "1500")} |
| 900 | {line(e, "900")} | {line(m, "900")} |

On real players the own-moves rating is also at least as accurate, without using the opponent at
all. The whole-game network is therefore not released.
"""
    return f"""
## Rating a player from their own moves

The player gives p(move | position, rating). For the moves one person played, the summed
log-probability as a function of the rating input is a measurement of that person's rating that
uses nothing but their own decisions. `fastelo.play.Rater` scores every move under 12 rating
inputs (400 to 2600), caps what a single move can contribute (so one slip does not decide the
result) and takes the mean and width of the resulting curve. Because each move is scored on its
own, the curve also shows *where* in a game someone played above or below their level.

`rating_response.json` then removes the remaining bias, measured on held-out human games with
the same number of players in every 200-Elo band: for each number of moves, a monotone map f
with mean f(estimate) = true rating at every true rating. `Rater.rating()` returns

- **value**: the corrected estimate;
- **statistical error** (asymmetric): the estimate plus and minus its width, mapped through f;
- **systematic error**: the bias left after correction on an independent set of games. It is
  common to all games of a player, so playing more games does not reduce it.

Closure test on independent human games after {int(k) // 2} own moves (Elo):

| True rating | Bias, raw | Bias, corrected | Stat. error | Inside 1 sigma | Syst. | Mean error of 5 games combined |
|---|---|---|---|---|---|---|
{rows}

When several games are combined, each is weighted by the typical error of a game of its length,
not by its own error bar: weights that depend on the estimate would bias the average.
{why}"""


def model_card(repo, infos, cal, results, dataset="irsotarriva/fastelo-synthetic-games", response=None, study=None):
    r = results or {}
    a = infos["player"]
    knots = "\n".join(f"| {k['elo']:.0f} | {k['conditioning']:.0f} | {k['temperature']:.2f} | {k['mix']:.2f} |"
                      for k in cal["knots"] if k["conditioning"] is not None)
    return f"""---
license: gpl-3.0
library_name: pytorch
tags:
  - chess
  - human-like
  - elo
  - knowledge-distillation
  - uncertainty
---

# fastelo: a chess player that plays at a chosen rating, and rates you from your own moves

Two small transformers (about 7M parameters each) for human-like chess at a chosen strength:

| File | What it is | Parameters |
|---|---|---|
| `player.safetensors` | **Player** (policy network): the move distribution of a human of a given Elo (input: position, own Elo, opponent Elo) | {a["parameters"] / 1e6:.2f}M |
| `strong_player.safetensors` | **Strong player** (policy network): the player before the human phase, distilled from Leela Chess Zero; mixed in for targets above about 2100 | {infos["strong_player"]["parameters"] / 1e6:.2f}M |
| `calibration.json` | Maps a target Elo to the settings that play at that strength | |
| `rating_response.json` | Bias correction of the rating read from the player, with its systematic error | |

The same player does both jobs: it plays at a target rating, and the likelihood of *your* moves
under it, as a function of the rating, tells how strong you are. Both networks are supervised
policy networks; nothing here is trained with reinforcement learning.

Related: [synthetic games with phase-dependent strength](https://huggingface.co/datasets/{dataset})
(CC0), generated with these models.

## How it was trained

1. **Distillation.** A student transformer (64 square tokens + 3 context tokens, d=256, 8 layers)
   learned the policy and win/draw/loss output of the Leela Chess Zero network
   **T1-256x10-distilled-swa-2432500** on 300M positions taken from real Lichess games
   (never engine self-play, to keep the variety of human positions).
2. **Human phase.** Starting from that student, the player was trained on 500M positions from rated
   **rapid and classical** Lichess games (August to November 2025; 42.7M games after filtering
   out bots, provisional ratings and abandoned games), conditioned on the mover's and the
   opponent's rating.
3. **Calibration (no training).** On 30,000 held-out positions analysed with Stockfish, the
   temperature, the share of the strong player and a small correction of the Elo input were fitted
   so that the policy's expected-score loss per move equals that of humans in each rating band.

## Measured results

**Playing strength.** Top-1 match with the human move on held-out games: {100 * r.get("player_top1", 0.544):.1f}%.
For each rating band, the Elo input under which the human moves are most likely tracks the band
(correlation {r.get("player_band_corr", 0.998):.3f}). Raw strength rises with the Elo input up to about
2400 and then plateaus; with the calibration the policy matches human move quality from 600 to
about 2400, and comes within about one standard error at 2400-2600.

Calibrated settings (target Elo -> Elo input, temperature, share of the strong player):

| Target | Elo input | Temperature | Strong share |
|---|---|---|---|
{knots}
{rating_section(response, study)}
## Limitations

- Strength above about 2400-2500 is not reachable; near the top the policy is mostly the strong
  player played close to its best move, which is less human-like.
- One game is a weak measurement: roughly +-300 Elo after 20 of your moves, about +-200 after 40.
  Several games are needed for a tight rating.
- Ratings should only be quoted between 600 and 2200 (`fastelo.response.REPORT_RANGE`); beyond
  that, report "below 600" or "2200+". The player is calibrated from 600, and bot-generated play
  near the top is rated 100 to 200 Elo too high.
- The rating was measured on rapid games; untimed or blitz play is not calibrated.
- Ratings are on the Lichess rapid scale, not FIDE.

## Use

```python
from huggingface_hub import snapshot_download
from fastelo.play import Player, Rater   # the fastelo Python package
import chess, numpy as np

root = snapshot_download("{repo}")
bot = Player(f"{{root}}/calibration.json")
rater = Rater(bot, response=f"{{root}}/rating_response.json")
board, boards, moves = chess.Board(), [], []
for _ in range(40):
    move = bot.choose(board, ply=len(moves), self_elo=1500, opp_elo=1500, rng=np.random.default_rng(len(moves)))
    boards.append(board.copy(stack=False)); moves.append(move); board.push(move)
r = rater.rating(boards, moves, side=0)                            # white, after each of its moves
print(r["value"][-1], r["lo"][-1], r["hi"][-1], r["syst"][-1])     # value, 1-sigma stat. bounds, systematic
```

`fastelo.lite` has the same player and rating in plain NumPy (no PyTorch), which is what the web
demo runs.

## Licence and attribution

Code and weights are released under the **GNU GPL v3.0** (see `LICENSE`).

- **Data:** [Lichess open database](https://database.lichess.org/), released under CC0.
- **Teacher:** [Leela Chess Zero](https://lczero.org/) (GPL-3.0 engine; network
  T1-256x10-distilled-swa-2432500 from the project's contributed networks). The Leela network
  itself is not redistributed here; the two players were trained on its outputs.
- **Analysis for calibration:** [Stockfish](https://stockfishchess.org/) (GPL-3.0), used as a tool.
"""


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--player", required=True)
    ap.add_argument("--strong", required=True)
    ap.add_argument("--calibration", required=True)
    ap.add_argument("--repo", required=True, help="<user>/<name>, used for both the model repo and the Space")
    ap.add_argument("--out", default="data/hf")
    ap.add_argument("--results", default=None, help="JSON with measured numbers for the model card")
    ap.add_argument("--rating-response", default=None,
                    help="bias correction of the move-likelihood rating (fastelo.likelihood measure + fastelo.response fit)")
    ap.add_argument("--study", default=None, help="results/rating_study.json, for the card's comparison table")
    ap.add_argument("--upload", action="store_true")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--model-only", action="store_true", help="with --upload: skip the Space")
    ap.add_argument("--message", default=None, help="commit message for the model repo")
    args = ap.parse_args(argv)

    model_dir, space_dir = os.path.join(args.out, "model"), os.path.join(args.out, "space")
    for d in (model_dir, space_dir):
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d, exist_ok=True)
    infos = {name: export_weights(path, os.path.join(model_dir, name))
             for name, path in (("player", args.player), ("strong_player", args.strong))}
    cal = json.load(open(args.calibration))
    cal.pop("bands", None)
    cal.update(player="player.safetensors", strong_player="strong_player.safetensors")
    with open(os.path.join(model_dir, "calibration.json"), "w") as f:
        json.dump(cal, f, indent=1)
    results = json.load(open(args.results)) if args.results else None
    rating = json.load(open(args.rating_response)) if args.rating_response else None
    if rating:
        shutil.copyfile(args.rating_response, os.path.join(model_dir, "rating_response.json"))
    study = json.load(open(args.study)) if args.study else None
    with open(os.path.join(model_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(model_card(args.repo, infos, cal, results, response=rating, study=study))
    shutil.copyfile(LICENSE_FILE, os.path.join(model_dir, "LICENSE"))

    import gradio
    pkg = os.path.dirname(os.path.abspath(__file__))
    shutil.copytree(pkg, os.path.join(space_dir, "fastelo"), ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    with open(os.path.join(space_dir, "app.py"), "w", encoding="utf-8") as f:
        f.write(SPACE_APP.replace("{repo}", args.repo))
    with open(os.path.join(space_dir, "requirements.txt"), "w") as f:
        f.write(SPACE_REQUIREMENTS)
    with open(os.path.join(space_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(space_readme(args.repo, gradio.__version__))
    shutil.copyfile(LICENSE_FILE, os.path.join(space_dir, "LICENSE"))
    size = sum(os.path.getsize(os.path.join(model_dir, f)) for f in os.listdir(model_dir)) / 1e6
    print(f"model repo: {model_dir} ({size:.0f} MB) | space: {space_dir}")

    if args.upload:
        from huggingface_hub import HfApi
        api = HfApi(token=os.environ.get("HF_API_KEY") or os.environ.get("HF_TOKEN"))
        api.create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)
        api.upload_folder(folder_path=model_dir, repo_id=args.repo, repo_type="model",
                          delete_patterns=["*.safetensors", "*.json"],  # drop files renamed since the last upload
                          commit_message=args.message or "fastelo models, calibration and model card")
        if args.model_only:
            print(f"uploaded: https://huggingface.co/{args.repo}")
            return
        api.create_repo(args.repo, repo_type="space", space_sdk="gradio", private=args.private, exist_ok=True)
        api.upload_folder(folder_path=space_dir, repo_id=args.repo, repo_type="space",
                          commit_message="fastelo demo")
        print(f"uploaded: https://huggingface.co/{args.repo} and https://huggingface.co/spaces/{args.repo}")


if __name__ == "__main__":
    main()
