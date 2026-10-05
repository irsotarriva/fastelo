"""Does the player play like the Elo it is conditioned on?

    python -m fastelo.eval_player --ckpt runs/player_tiny_cpu/ckpt.pt --data data/dev

Takes held-out games, and for every position scores the move the human actually
played under a grid of conditioning Elos (including values outside the training
range). Three views of the same likelihood matrix:

1. Likelihood curves: mean log p(human move | conditioning Elo) for each band of
   true player Elo. If the player learned Elo, each band's curve peaks at its own Elo.
2. Player Elo recovery: treat the player as a likelihood model and compute a
   posterior over each player's Elo from their moves in one game. If conditioning
   is meaningful the estimate tracks the true rating, and improves with more moves.
3. Out of distribution: behaviour of the curves and of policy entropy beyond the
   training range - smooth saturation vs. a cliff.
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from .board_codec import board_from_codes
from .build_dataset import FLAG_PLAYER_OK
from .features import _OFF, GameStore, decode_records, to_device
from .models import load_model
from .vocab import VOCAB_SIZE, legal_mask
from . import plotstyle
from .plotstyle import plt

TRUE_BINS = [600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600]
MOVE_COUNTS = [5, 10, 20, 40, 1000]


def select_positions(store, n_games, seed, min_plies=10):
    rng = np.random.default_rng(seed)
    g = store.games_np
    candidates = np.flatnonzero(g["n_plies"] >= min_plies)
    games = np.sort(rng.choice(candidates, size=min(n_games, len(candidates)), replace=False))
    idx = np.concatenate([np.arange(g["start"][i], g["start"][i] + g["n_plies"][i]) for i in games])
    idx_t = torch.as_tensor(idx, device=store.device)
    ok = ((store.raw[idx_t, _OFF["flags"]] & FLAG_PLAYER_OK) != 0).cpu().numpy()
    return idx[ok]


def legal_masks(store, idx):
    rec = store.raw[torch.as_tensor(idx, device=store.device)].cpu()
    d = decode_records(rec)
    flags = d["castling"] @ torch.tensor([1.0, 2, 4, 8])
    out = np.zeros((len(idx), VOCAB_SIZE), dtype=bool)
    for i in range(len(idx)):
        board = board_from_codes(d["board"][i].numpy(), int(flags[i]), int(d["ep"][i]), int(d["halfmove"][i]))
        out[i] = legal_mask(board)
    return torch.from_numpy(out)


@torch.no_grad()
def score_grid(model, store, idx, masks, grid, opp_mode, device, batch=2048, amp=False):
    """Returns logp[A,N] of the played move, top1[A,N], entropy[A,N] under each conditioning Elo,
    plus the same three under the true Elos, and per-position metadata."""
    A, N = len(grid), len(idx)
    logp = np.zeros((A, N), np.float32)
    top1 = np.zeros((A, N), bool)
    ent = np.zeros((A, N), np.float32)
    meta = {}
    true_scores = {"logp": np.zeros(N, np.float32), "top1": np.zeros(N, bool)}
    autocast = torch.autocast(torch.device(device).type, dtype=torch.bfloat16, enabled=amp)
    for s in range(0, N, batch):
        sl = slice(s, s + batch)
        b = to_device(store.batch(torch.as_tensor(idx[sl], device=store.device)), device)
        m = masks[sl].to(device)
        for k, v in (("self_elo", b["self_elo"]), ("opp_elo", b["opp_elo"]), ("ply", b["ply"]),
                     ("game", b["game"])):
            meta.setdefault(k, []).append(v.cpu().numpy())
        meta.setdefault("black", []).append((b["ply"] % 2 == 1).cpu().numpy())

        def run(self_elo, opp_elo):
            bb = dict(b, self_elo=self_elo, opp_elo=opp_elo)
            with autocast:
                logits, _ = model(bb)
            lp = F.log_softmax(logits.float().masked_fill(~m, float("-inf")), -1)
            played = lp.gather(1, bb["move"].unsqueeze(1)).squeeze(1)
            hit = lp.argmax(-1) == bb["move"]
            e = -(lp.exp() * lp.masked_fill(~m, 0)).sum(-1)
            return played.cpu().numpy(), hit.cpu().numpy(), e.cpu().numpy()

        true_scores["logp"][sl], true_scores["top1"][sl], _ = run(b["self_elo"], b["opp_elo"])
        for a_i, a in enumerate(grid):
            se = torch.full_like(b["self_elo"], float(a))
            oe = se if opp_mode == "same" else se + (b["opp_elo"] - b["self_elo"])
            logp[a_i, sl], top1[a_i, sl], ent[a_i, sl] = run(se, oe)
    meta = {k: np.concatenate(v) for k, v in meta.items()}
    return logp, top1, ent, true_scores, meta


def shade_ood(ax, grid, lo, hi):
    """Shade conditioning Elos outside the Elo encoder's range, labelled at the top."""
    for a, b in ((grid[0], lo), (hi, grid[-1])):
        if b > a:
            ax.axvspan(a, b, color=plotstyle.GRID, alpha=0.6, lw=0)
            ax.text((a + b) / 2, 0.98, "outside\nencoder range", transform=ax.get_xaxis_transform(),
                    ha="center", va="top", color=plotstyle.TEXT_2, fontsize=8)


def prior_on_grid(manifest, grid):
    hist = manifest.get("train_elo_hist_100")
    if not hist:
        return np.full(len(grid), 1.0 / len(grid))
    centers = np.array([int(k) + 50 for k in hist])
    p = np.interp(grid, centers, np.array(list(hist.values()), float)) + 1e-6
    return p / p.sum()


def player_estimates(logp, meta, grid, prior):
    """Posterior-mean Elo per (game, color) from its first k moves, for k in MOVE_COUNTS."""
    key = meta["game"] * 2 + meta["black"]
    order = np.lexsort((meta["ply"], key))
    key_s, lp_s = key[order], logp[:, order]
    uniq, first = np.unique(key_s, return_index=True)
    true = meta["self_elo"][order][first]
    rows = []
    log_prior = np.log(prior)[:, None]
    bounds = list(first) + [len(key_s)]
    for j in range(len(uniq)):
        lp = lp_s[:, bounds[j]:bounds[j + 1]]
        cum = np.cumsum(lp, axis=1)
        est = {}
        for k in MOVE_COUNTS:
            kk = min(k, cum.shape[1])
            post = cum[:, kk - 1:kk] + log_prior
            post = np.exp(post - post.max())
            post /= post.sum()
            est[k] = float((post[:, 0] * grid).sum())
        rows.append((true[j], cum.shape[1], est))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--games", type=int, default=300)
    ap.add_argument("--grid", default="200:3400:100", help="lo:hi:step conditioning Elos")
    ap.add_argument("--opp-mode", choices=["same", "gap"], default="same",
                    help="opponent Elo = conditioning Elo, or conditioning Elo + true rating gap")
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    lo, hi, step = (int(x) for x in args.grid.split(":"))
    grid = np.arange(lo, hi + 1, step)
    out = args.out or os.path.join(os.path.dirname(args.ckpt), f"eval_{args.split}")
    os.makedirs(out, exist_ok=True)
    model, ckpt = load_model(args.ckpt, args.device)
    store = GameStore(args.data, args.split)
    idx = select_positions(store, args.games, args.seed)
    print(f"{len(idx):,} positions from {args.games} games; computing legal masks", flush=True)
    masks = legal_masks(store, idx)
    logp, top1, ent, true_scores, meta = score_grid(model, store, idx, masks, grid, args.opp_mode, args.device,
                                                    amp=args.device.startswith("cuda"))
    np.savez_compressed(os.path.join(out, "scores.npz"), grid=grid, logp=logp, top1=top1, ent=ent,
                        true_logp=true_scores["logp"], true_top1=true_scores["top1"], **meta)

    train_lo, train_hi = model.cfg.elo_lo, model.cfg.elo_hi
    true_elo = meta["self_elo"]
    bins = np.digitize(true_elo, TRUE_BINS)
    summary = dict(step=ckpt["step"], positions=len(idx), games=args.games, opp_mode=args.opp_mode,
                   top1_true_elo=float(true_scores["top1"].mean()),
                   mean_logp_true_elo=float(true_scores["logp"].mean()))

    # 1. likelihood curves per true-Elo band
    plotstyle.setup()
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bands = [(i, f"{TRUE_BINS[i - 1]}-{TRUE_BINS[i]}") for i in range(1, len(TRUE_BINS))
             if (bins == i).sum() >= 200]
    colors = plotstyle.ramp(len(bands))
    band_rows = []
    for (i, label), c in zip(bands, colors):
        sel = bins == i
        curve = logp[:, sel].mean(1)
        rel = curve - curve.max()
        a_best = int(grid[curve.argmax()])
        center = (TRUE_BINS[i - 1] + TRUE_BINS[i]) / 2
        band_rows.append(dict(band=label, n=int(sel.sum()), argmax_elo=a_best, center=center,
                              top1_at_true=float(true_scores["top1"][sel].mean()),
                              top1_best=float(top1[:, sel].mean(1).max())))
        ax.plot(grid, rel, color=c, label=label)
        ax.plot([a_best], [0], "o", color=c, markersize=8, markeredgecolor=plotstyle.SURFACE, markeredgewidth=2)
    shade_ood(ax, grid, train_lo, train_hi)
    ax.set_xlabel("conditioning Elo")
    ax.set_ylabel("mean log-likelihood of human moves\n(relative to each band's peak, nats/move)")
    ax.set_title("Each band of players should be best explained at its own Elo (dots = peak)")
    ax.legend(title="true player Elo", loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8, title_fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "likelihood_curves.png"), dpi=130)
    plt.close(fig)
    summary["bands"] = band_rows
    peaks = np.array([r["argmax_elo"] for r in band_rows])
    centers = np.array([r["center"] for r in band_rows])
    summary["band_peak_mae"] = float(np.abs(peaks - centers).mean()) if len(peaks) else None
    summary["band_peak_corr"] = float(np.corrcoef(peaks, centers)[0, 1]) if len(peaks) > 2 else None

    # 2. player Elo recovery
    prior = prior_on_grid(store.manifest, grid)
    rows = player_estimates(logp, meta, grid, prior)
    true_p = np.array([r[0] for r in rows])
    prior_mean = float((prior * grid).sum())
    rec = {"baseline_prior_mean_mae": float(np.abs(true_p - prior_mean).mean())}
    for k in MOVE_COUNTS:
        est = np.array([r[2][k] for r in rows])
        rec[f"mae_{k}_moves"] = float(np.abs(est - true_p).mean())
        rec[f"corr_{k}_moves"] = float(np.corrcoef(est, true_p)[0, 1])
    summary["player_recovery"] = rec

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    est_all = np.array([r[2][MOVE_COUNTS[-1]] for r in rows])
    ax = axes[0]
    ax.plot([400, 3000], [400, 3000], color=plotstyle.TEXT_2, lw=1, ls="--")
    ax.scatter(true_p, est_all, s=10, color=plotstyle.SERIES[0], alpha=0.45, lw=0)
    ax.set_xlim(400, 3000); ax.set_ylim(400, 3000)
    ax.set_xlabel("true Elo"); ax.set_ylabel("estimated Elo (posterior mean)")
    ax.set_title(f"One game per point, all moves (r = {rec['corr_1000_moves']:.2f})")
    ax = axes[1]
    ks = MOVE_COUNTS[:-1]
    ax.plot(ks, [rec[f"mae_{k}_moves"] for k in ks], "o-", color=plotstyle.SERIES[0], label="player likelihood")
    base = rec["baseline_prior_mean_mae"]
    ax.axhline(base, color=plotstyle.TEXT_2, ls="--", lw=1)
    ax.text(ks[-1], base * 0.97, "guess the average rating ", ha="right", va="top", color=plotstyle.TEXT_2, fontsize=9)
    ax.set_xscale("log"); ax.minorticks_off(); ax.set_xticks(ks); ax.set_xticklabels(ks)
    ax.set_xlabel("moves observed from the player"); ax.set_ylabel("mean absolute error (Elo)")
    ax.set_title("Error should shrink with more moves")
    ax.set_ylim(0, base * 1.15)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "player_recovery.png"), dpi=130)
    plt.close(fig)

    # 3. OOD: policy entropy vs conditioning Elo
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(grid, ent.mean(1), "o-", color=plotstyle.SERIES[0])
    shade_ood(ax, grid, train_lo, train_hi)
    ax.set_xlabel("conditioning Elo"); ax.set_ylabel("mean policy entropy (nats)")
    ax.set_title("Stronger conditioning should mean a sharper policy")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "entropy_vs_elo.png"), dpi=130)
    plt.close(fig)
    summary["entropy_vs_elo"] = dict(zip(grid.tolist(), np.round(ent.mean(1), 4).tolist()))

    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "entropy_vs_elo"}, indent=1))


if __name__ == "__main__":
    main()
