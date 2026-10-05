"""Engine-based strength calibration of the player.

    python -m fastelo.eval_strength analyse --data data/dev --out data/sf_dev.npz
    python -m fastelo.eval_strength score --ckpt runs/player_tiny_cpu/ckpt.pt --analysis data/sf_dev.npz

`analyse` (slow, once per dataset): samples held-out positions stratified by the
mover's Elo and has Stockfish evaluate *every* legal move. Each move gets a
loss = drop in expected score (Lichess win-probability model) vs the best move.

`score` (fast, once per checkpoint): the player's expected loss at each
conditioning Elo is sum_m p(m | elo) * loss(m) - the average quality of the moves
it would play when sampling. Compared on the same positions with the loss of
the moves real players of that Elo made. A calibrated player matches the human
curve inside the training range; outside it we see saturation or a cliff.
"""
import argparse
import json
import multiprocessing as mp
import os

import chess
import chess.engine
import numpy as np
import torch
import torch.nn.functional as F

from .board_codec import board_from_codes
from .eval_player import TRUE_BINS, shade_ood
from .features import _OFF, GameStore, decode_records, to_device
from .build_dataset import FLAG_PLAYER_OK
from .models import load_model
from .vocab import VOCAB_SIZE, encode_move
from . import plotstyle
from .plotstyle import plt

DEFAULT_ENGINE = os.path.join(os.path.dirname(__file__), "..", "tools", "stockfish",
                              "stockfish-windows-x86-64-universal.exe")


def win_prob(cp):
    """Lichess expected-score model, cp from the mover's point of view -> [0, 1]."""
    return 1 / (1 + np.exp(-0.00368208 * cp))


def score_to_cp(score: chess.engine.PovScore):
    return score.white().score(mate_score=10000)


_engine = None


def _init_engine(path, nodes):
    global _engine, _nodes
    _engine = chess.engine.SimpleEngine.popen_uci(path)
    _engine.configure({"Threads": 1, "Hash": 64})
    _nodes = nodes


def _analyse(fen):
    board = chess.Board(fen)
    n = board.legal_moves.count()
    infos = _engine.analyse(board, chess.engine.Limit(nodes=_nodes * n), multipv=n)
    return [(encode_move(i["pv"][0], chess.WHITE), score_to_cp(i["score"])) for i in infos if "pv" in i]


def analyse(args):
    store = GameStore(args.data, args.split)
    rng = np.random.default_rng(args.seed)
    ok = ((store.raw[:, _OFF["flags"]] & FLAG_PLAYER_OK) != 0).numpy()
    game_of = np.searchsorted(store.games_np["start"], np.arange(store.n_positions), "right") - 1
    ply = np.arange(store.n_positions) - store.games_np["start"][game_of]
    black = ply % 2 == 1
    elo = np.where(black, store.games_np["black_elo"][game_of], store.games_np["white_elo"][game_of])
    ok &= ply >= args.min_ply  # skip memorized openings
    bins = np.digitize(elo, TRUE_BINS)
    chosen = []
    for b in range(1, len(TRUE_BINS)):
        cand = np.flatnonzero(ok & (bins == b))
        chosen.append(rng.choice(cand, size=min(args.per_bin, len(cand)), replace=False))
    idx = np.sort(np.concatenate(chosen))
    d = decode_records(store.raw[torch.as_tensor(idx)])
    flags = (d["castling"] @ torch.tensor([1.0, 2, 4, 8])).int()
    fens = [board_from_codes(d["board"][i].numpy(), int(flags[i]), int(d["ep"][i]), int(d["halfmove"][i])).fen()
            for i in range(len(idx))]
    print(f"analysing {len(idx)} positions with {args.workers} engines at {args.nodes} nodes/move", flush=True)
    with mp.Pool(args.workers, initializer=_init_engine, initargs=(args.engine, args.nodes)) as pool:
        results = []
        for i, r in enumerate(pool.imap(_analyse, fens, chunksize=4)):
            results.append(r)
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(fens)}", flush=True)
    loss = np.full((len(idx), VOCAB_SIZE), np.nan, np.float32)
    for i, r in enumerate(results):
        wp = {m: win_prob(cp) for m, cp in r}
        best = max(wp.values())
        for m, w in wp.items():
            loss[i, m] = best - w
    played = d["move"].numpy()
    np.savez_compressed(args.out, idx=idx, loss=loss, played=played, elo=elo[idx], split=args.split,
                        data=os.path.abspath(args.data))
    human = loss[np.arange(len(idx)), played]
    print(f"saved {args.out}; human mean loss {np.nanmean(human):.4f}, "
          f"{np.isnan(human).sum()} played moves missing from analysis")


@torch.no_grad()
def score(args):
    a = np.load(args.analysis)
    model, ckpt = load_model(args.ckpt, args.device)
    store = GameStore(str(a["data"]), str(a["split"]))
    idx, loss, played, elo = a["idx"], a["loss"], a["played"], a["elo"]
    legal = ~np.isnan(loss)
    loss0 = np.nan_to_num(loss, nan=0.0)
    lo, hi, step = (int(x) for x in args.grid.split(":"))
    grid = np.arange(lo, hi + 1, step)
    exp_loss = np.zeros((len(grid), len(idx)), np.float32)
    top1_loss = np.zeros((len(grid), len(idx)), np.float32)
    for s in range(0, len(idx), 1024):
        sl = slice(s, s + 1024)
        b = to_device(store.batch(torch.as_tensor(idx[sl])), args.device)
        m = torch.as_tensor(legal[sl], device=args.device)
        L = torch.as_tensor(loss0[sl], device=args.device)
        for i, e in enumerate(grid):
            se = torch.full_like(b["self_elo"], float(e))
            logits, _ = model(dict(b, self_elo=se, opp_elo=se))
            p = F.softmax(logits.float().masked_fill(~m, float("-inf")) / args.temperature, -1)
            exp_loss[i, sl] = (p * L).sum(-1).cpu().numpy()
            top1_loss[i, sl] = L.gather(1, p.argmax(-1, keepdim=True)).squeeze(1).cpu().numpy()

    human = loss[np.arange(len(idx)), played]
    bins = np.digitize(elo, TRUE_BINS)
    rows = []
    for b in range(1, len(TRUE_BINS)):
        sel = (bins == b) & ~np.isnan(human)
        if sel.sum() < 30:
            continue
        center = (TRUE_BINS[b - 1] + TRUE_BINS[b]) / 2
        gi = int(np.abs(grid - center).argmin())
        h = human[sel]
        band_curve = exp_loss[:, sel].mean(1)
        rows.append(dict(band=f"{TRUE_BINS[b - 1]}-{TRUE_BINS[b]}", center=center, n=int(sel.sum()),
                         human_loss=float(h.mean()), human_se=float(h.std() / np.sqrt(len(h))),
                         model_loss=float(exp_loss[gi, sel].mean()),
                         model_greedy_loss=float(top1_loss[gi, sel].mean()),
                         # conditioning Elo at which the player's moves on THIS band's positions are as
                         # good as the moves these players made (with the band's +-1 s.e. range)
                         conditioning=matching_elo(grid, band_curve, h.mean()),
                         conditioning_lo=matching_elo(grid, band_curve, h.mean() + h.std() / np.sqrt(len(h))),
                         conditioning_hi=matching_elo(grid, band_curve, h.mean() - h.std() / np.sqrt(len(h))),
                         band_curve_decreasing=float((np.diff(band_curve) < 0).mean())))
    curve = exp_loss.mean(1)
    out_dir = args.out or os.path.join(os.path.dirname(args.ckpt), "eval_strength")
    os.makedirs(out_dir, exist_ok=True)
    summary = dict(step=ckpt["step"], temperature=args.temperature, positions=len(idx), bands=rows,
                   expected_loss_vs_elo=dict(zip(grid.tolist(), np.round(curve, 5).tolist())),
                   # how much better its moves get from 800 to 2400 conditioning (percentage points of
                   # expected score per move; humans: ~7.0 -> ~2.3). ~0 = Elo input ignored.
                   loss_drop_800_to_2400=float(100 * (np.interp(800, grid, curve) - np.interp(2400, grid, curve))),
                   strictly_decreasing_fraction=float((np.diff(curve) < -1e-4).mean()),
                   # true (human-equivalent) Elo -> conditioning Elo to feed the player; inference-only
                   # calibration, valid only where the player's strength is monotonic in its Elo input
                   lut={str(r["center"]): r["conditioning"] for r in rows if r["conditioning"] is not None})

    plotstyle.setup()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    ax = axes[0]
    c = [r["center"] for r in rows]
    ax.errorbar(c, [100 * r["human_loss"] for r in rows], yerr=[196 * r["human_se"] for r in rows],
                fmt="o-", color=plotstyle.SERIES[1], label="humans of that Elo (played move)", capsize=3)
    ax.plot(c, [100 * r["model_loss"] for r in rows], "o-", color=plotstyle.SERIES[0],
            label="player conditioned on that Elo (sampling)")
    ax.plot(c, [100 * r["model_greedy_loss"] for r in rows], "o--", color=plotstyle.SERIES[0], alpha=0.5,
            label="player, argmax move")
    ax.set_xlabel("Elo"); ax.set_ylabel("expected-score loss per move (%)")
    ax.set_title("Calibration on the same positions")
    ax.legend(fontsize=8); ax.set_ylim(bottom=0)
    ax = axes[1]
    ax.plot(grid, 100 * curve, "o-", color=plotstyle.SERIES[0])
    shade_ood(ax, grid, model.cfg.elo_lo, model.cfg.elo_hi)
    ax.set_xlabel("conditioning Elo"); ax.set_ylabel("expected-score loss per move (%)")
    ax.set_title("All positions: should fall smoothly with Elo"); ax.set_ylim(0, 100 * curve.max() * 1.15)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "strength_calibration.png"), dpi=130)
    plt.close(fig)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))


def _engine_moves(job):
    """Worker: play one engine configuration on every position; returns chosen vocab indices."""
    path, opts, fens, seconds = job
    engine = chess.engine.SimpleEngine.popen_uci(path)
    engine.configure({"Threads": 1, "Hash": 16, **opts})
    out = []
    for fen in fens:
        out.append(encode_move(engine.play(chess.Board(fen), chess.engine.Limit(time=seconds)).move, chess.WHITE))
    engine.quit()
    return out


def matching_elo(grid, curve, target):
    """Smallest conditioning Elo at which `curve` (expected loss vs conditioning, ideally falling)
    comes down to `target`, interpolated; None if it never does on the grid."""
    curve = np.asarray(curve, np.float64)
    if curve[0] <= target:
        return float(grid[0]) if np.isclose(curve[0], target) else None
    for i in range(len(curve) - 1):
        if curve[i] > target >= curve[i + 1]:
            t = (curve[i] - target) / (curve[i] - curve[i + 1])
            return float(grid[i] + t * (grid[i + 1] - grid[i]))
    return None


def equivalent_elo(model_by_band, human_by_band, centers):
    """Elo where the engine's per-band loss equals humans' (linear interpolation of the
    difference across bands). Returns (elo, note)."""
    diff = np.asarray(model_by_band) - np.asarray(human_by_band)  # >0: engine worse than that band
    for i in range(len(diff) - 1):
        if diff[i] > 0 >= diff[i + 1] or diff[i] <= 0 < diff[i + 1]:
            t = diff[i] / (diff[i] - diff[i + 1])
            return float(centers[i] + t * (centers[i + 1] - centers[i])), "crossing"
    return (float(centers[-1]), "better than every band") if (diff <= 0).all() else \
           (float(centers[0]), "worse than every band")


def calibrate_engine(args):
    """Where do Stockfish's strength settings land on the Lichess human scale, judged by
    move quality on the analysed positions?"""
    a = np.load(args.analysis)
    store = GameStore(str(a["data"]), str(a["split"]))
    idx, loss, played, elo = a["idx"], a["loss"], a["played"], a["elo"]
    d = decode_records(store.raw[torch.as_tensor(idx)])
    flags = (d["castling"] @ torch.tensor([1.0, 2, 4, 8])).int()
    fens = [board_from_codes(d["board"][i].numpy(), int(flags[i]), int(d["ep"][i]), int(d["halfmove"][i])).fen()
            for i in range(len(idx))]
    configs = [(f"UCI_Elo {e}", {"UCI_LimitStrength": True, "UCI_Elo": e}) for e in args.uci_elos]
    configs += [(f"Skill {s}", {"UCI_LimitStrength": False, "Skill Level": s}) for s in args.skills]
    jobs = [(args.engine, opts, fens, args.seconds) for _, opts in configs]
    print(f"{len(configs)} engine settings x {len(fens)} positions at {args.seconds}s/move", flush=True)
    with mp.Pool(args.workers) as pool:
        moves = pool.map(_engine_moves, jobs)

    bins = np.digitize(elo, TRUE_BINS)
    bands = [b for b in range(1, len(TRUE_BINS)) if (bins == b).sum() >= 30]
    centers = np.array([(TRUE_BINS[b - 1] + TRUE_BINS[b]) / 2 for b in bands])
    human = loss[np.arange(len(idx)), played]
    human_by_band = [float(np.nanmean(human[bins == b])) for b in bands]
    rows = []
    for (name, _), mv in zip(configs, moves):
        l = loss[np.arange(len(idx)), np.asarray(mv)]
        by_band = [float(np.nanmean(l[bins == b])) for b in bands]
        eq, note = equivalent_elo(by_band, human_by_band, centers)
        rows.append(dict(setting=name, mean_loss=float(np.nanmean(l)), lichess_equivalent=round(eq), note=note,
                         loss_by_band=dict(zip(centers.astype(int).tolist(), np.round(by_band, 4).tolist()))))
        print(f"{name:14s} loss {np.nanmean(l) * 100:5.2f}%  ~ Lichess {round(eq)} ({note})", flush=True)
    out = dict(seconds_per_move=args.seconds, human_loss_by_band=dict(zip(centers.astype(int).tolist(),
               np.round(human_by_band, 4).tolist())), engines=rows)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)

    plotstyle.setup()
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(centers, 100 * np.array(human_by_band), "o-", color=plotstyle.SERIES[1], label="humans (played move)")
    uci = [r for r in rows if r["setting"].startswith("UCI")]
    for r, c in zip(uci, plotstyle.ramp(len(uci))):
        ax.plot(centers, 100 * np.array(list(r["loss_by_band"].values())), "-", color=c, lw=1.5)
        ax.text(centers[-1] + 30, 100 * list(r["loss_by_band"].values())[-1], r["setting"].split()[-1],
                color=plotstyle.TEXT_2, fontsize=8, va="center")
    ax.set_xlabel("Elo band of the positions (human players)")
    ax.set_ylabel("expected-score loss per move (%)")
    ax.set_title("Stockfish UCI_Elo settings (blue, labelled) vs humans on the same positions")
    ax.legend(fontsize=8, loc="upper right"); ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(os.path.splitext(args.out)[0] + ".png", dpi=130)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("engine", help="map Stockfish strength settings to Lichess-equivalent Elo")
    p.add_argument("--analysis", required=True)
    p.add_argument("--out", required=True, help=".json (a .png is written next to it)")
    p.add_argument("--uci-elos", type=int, nargs="+", default=[1320, 1500, 1800, 2100, 2400, 2700, 3000, 3190])
    p.add_argument("--skills", type=int, nargs="+", default=[0, 3, 6, 10])
    p.add_argument("--seconds", type=float, default=0.1)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--engine", default=DEFAULT_ENGINE)
    p = sub.add_parser("analyse")
    p.add_argument("--data", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--out", required=True)
    p.add_argument("--per-bin", type=int, default=150)
    p.add_argument("--min-ply", type=int, default=10)
    p.add_argument("--nodes", type=int, default=20000, help="engine nodes per legal move")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--engine", default=DEFAULT_ENGINE)
    p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("score")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--analysis", required=True)
    p.add_argument("--grid", default="200:3400:100")
    p.add_argument("--temperature", type=float, default=1.0, help="policy temperature (<1 = sharper, stronger)")
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)
    {"analyse": analyse, "score": score, "engine": calibrate_engine}[args.cmd](args)


if __name__ == "__main__":
    main()
