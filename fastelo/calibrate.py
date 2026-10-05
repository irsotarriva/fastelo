"""Inference-only strength calibration: target Elo -> (Elo input, temperature, mix).

    python -m fastelo.calibrate cache --player runs/player/ckpt.pt --strong runs/distill/ckpt.pt \
        --analysis data/sf_test.npz --out data/calib_cache.npz
    python -m fastelo.calibrate fit --cache data/calib_cache.npz --out runs/player/calibration.json

The playing policy for a target Elo e mixes the human-like player with a stronger model (the
distilled student, before the human phase) in log-probability space and applies a temperature:

    log p(m) = softmax( (1 - mix) * log p_player(m | elo=c) / T + mix * log p_strong(m) / T_strong )

The strong model gets its own low temperature: Leela-style policies are broad, so the distilled
student is only stronger than the player when it plays close to its top move.

`fit` chooses the knobs so that, on Stockfish-analysed held-out positions of each rating band,
the policy's expected-score loss per move equals that of the human players of the band:
  1. temperature T(e): with c = e and no mixing, the T that matches each band; a continuous
     two-segment line through those (falling with Elo, knot fitted) is the schedule;
  2. mix(e): only for bands that no temperature can bring to the human level (c = e), the share
     of the strong model needed (at T(e) and T_strong), as a ramp from 0 rising with Elo;
  3. Elo input c(e): with T and mix fixed, the input that matches each band exactly - the LUT,
     which should now be a small correction around c = e.

`cache` runs the networks once (player at a grid of Elo inputs, strong model once) and stores the
logits of the legal moves; `fit` then searches the knobs in seconds.
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from .eval_player import TRUE_BINS
from .features import GameStore, to_device
from .models import load_model
from . import plotstyle
from .plotstyle import plt


# ------------------------------------------------------------------------------- cache
@torch.no_grad()
def cache(args):
    a = np.load(args.analysis)
    idx, loss, played, elo = a["idx"], a["loss"], a["played"], a["elo"]
    legal = ~np.isnan(loss)
    K = int(legal.sum(1).max())
    # legal moves of each position, padded to K (vocab index -1 = padding)
    moves = np.full((len(idx), K), -1, np.int64)
    for i in range(len(idx)):
        m = np.flatnonzero(legal[i])
        moves[i, :len(m)] = m
    pad = moves < 0
    mv = np.where(pad, 0, moves)
    move_loss = np.where(pad, np.nan, np.take_along_axis(np.nan_to_num(loss, nan=0.0), mv, 1)).astype(np.float32)
    played_k = np.argmax(moves == played[:, None], 1)
    assert (moves[np.arange(len(idx)), played_k] == played).all(), "played move missing from analysis"

    player, _ = load_model(args.player, args.device)
    strong, sck = load_model(args.strong, args.device)
    strong_elo = float(sck.get("distill", {}).get("elo", 3000.0))
    lo, hi, step = (int(x) for x in args.grid.split(":"))
    grid = np.arange(lo, hi + 1, step)
    store = GameStore(str(a["data"]), str(a["split"]))
    act = np.zeros((len(grid), len(idx), K), np.float16)
    stg = np.zeros((len(idx), K), np.float16)

    def gather(logits, sl):
        lg = logits.float().cpu().numpy()
        out = np.take_along_axis(lg, mv[sl], 1)
        return np.where(pad[sl], -np.inf, out)

    for s in range(0, len(idx), args.batch):
        sl = slice(s, s + args.batch)
        b = to_device(store.batch(torch.as_tensor(idx[sl])), args.device)
        se = torch.full_like(b["self_elo"], strong_elo)
        stg[sl] = gather(strong(dict(b, self_elo=se, opp_elo=se))[0], sl)
        for i, e in enumerate(grid):
            se = torch.full_like(b["self_elo"], float(e))
            act[i, sl] = gather(player(dict(b, self_elo=se, opp_elo=se))[0], sl)
        print(f"  {min(s + args.batch, len(idx))}/{len(idx)}", flush=True)
    np.savez(args.out, grid=grid, player=act, strong=stg, move_loss=move_loss, pad=pad,
             played_k=played_k, elo=elo, strong_elo=strong_elo,
             player_ckpt=str(args.player), strong_ckpt=str(args.strong))
    print(f"saved {args.out}: {len(idx)} positions x {len(grid)} Elo inputs, up to {K} legal moves")


# --------------------------------------------------------------------------------- fit
class Scorer:
    """Expected loss / human-likeness of the mixed policy on a set of positions."""

    def __init__(self, c, sel, t_strong):
        self.t_strong = t_strong
        self.grid = c["grid"].astype(np.float64)
        self.act = torch.from_numpy(c["player"][:, sel].astype(np.float32))       # [G, N, K]
        self.stg = torch.from_numpy(c["strong"][sel].astype(np.float32))         # [N, K]
        self.loss = torch.from_numpy(np.nan_to_num(c["move_loss"][sel], nan=0.0))
        self.played = torch.from_numpy(c["played_k"][sel])
        self.act = torch.log_softmax(self.act, -1)
        self.stg = torch.log_softmax(self.stg, -1)

    def logp(self, elo_in, T, mix):
        x = float(np.clip(elo_in, self.grid[0], self.grid[-1]))
        i = int(min(np.searchsorted(self.grid, x, "right") - 1, len(self.grid) - 2))
        t = (x - self.grid[i]) / (self.grid[i + 1] - self.grid[i])
        a = (1 - t) * self.act[i] + t * self.act[i + 1]          # Elo input between grid points
        z = torch.nan_to_num((1 - mix) * a / T + mix * self.stg / self.t_strong, nan=-1e9, neginf=-1e9)
        return torch.log_softmax(z, -1)

    def expected_loss(self, elo_in, T, mix):
        return float((self.logp(elo_in, T, mix).exp() * self.loss).sum(-1).mean())

    def likeness(self, elo_in, T, mix):
        lp = self.logp(elo_in, T, mix)
        hit = (lp.argmax(-1) == self.played).float().mean()
        return float(lp.gather(1, self.played[:, None]).mean()), float(hit)


def fit_temperature(e, t, segments=2, t_min=0.3, t_max=1.5):
    """Continuous piecewise-linear T(e), non-increasing, fitted by least squares: one line, or two
    joined at the knot (searched every 25 Elo between the 2nd and 2nd-to-last points) that fits best."""
    e, t = np.asarray(e, float), np.asarray(t, float)

    def lsq(knot):
        # basis: 1, (e - knot) below the knot, (e - knot) above; slopes clipped to <= 0
        if knot is None:  # single line, anchored at the first point
            A = np.stack([np.ones_like(e), e - e[0]], 1)
            coef = np.linalg.lstsq(A, t, rcond=None)[0]
            coef = np.array([coef[0], min(coef[1], 0.0)])
            if coef[1] == 0.0:
                coef[0] = t.mean()
            return dict(knot=float(e[0]), value=float(coef[0]), slope_lo=float(coef[1]), slope_hi=float(coef[1])),                 float(((A @ coef - t) ** 2).sum())
        A = np.stack([np.ones_like(e), np.minimum(e - knot, 0), np.maximum(e - knot, 0)], 1)
        coef = np.linalg.lstsq(A, t, rcond=None)[0]
        for _ in range(2):  # refit with any positive slope fixed at 0
            free = [0] + [i for i in (1, 2) if coef[i] < 0]
            coef = np.zeros(3)
            coef[free] = np.linalg.lstsq(A[:, free], t, rcond=None)[0]
        return dict(knot=float(knot), value=float(coef[0]), slope_lo=float(coef[1]), slope_hi=float(coef[2])),             float(((A @ coef - t) ** 2).sum())

    best = lsq(None)
    if segments == 2 and len(e) >= 4:
        for k in np.arange(e[1], e[-2] + 1, 25.0):
            cand = lsq(k)
            if cand[1] < best[1]:
                best = cand
    return dict(best[0], t_min=t_min, t_max=t_max, rss=best[1])


def temperature_at(c, e):
    d = e - c["knot"]
    return float(np.clip(c["value"] + (c["slope_lo"] if d < 0 else c["slope_hi"]) * d, c["t_min"], c["t_max"]))


def solve(f, target, lo, hi, increasing=True, iters=40):
    """x in [lo, hi] with f(x) = target for a monotone f; None if outside the reachable range."""
    flo, fhi = f(lo), f(hi)
    if increasing and not (flo <= target <= fhi) or not increasing and not (fhi <= target <= flo):
        return None
    for _ in range(iters):
        mid = (lo + hi) / 2
        if (f(mid) < target) == increasing:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def fit(args):
    c = np.load(args.cache)
    bins = np.digitize(c["elo"], TRUE_BINS)
    human_all = np.take_along_axis(np.nan_to_num(c["move_loss"], nan=0.0), c["played_k"][:, None], 1)[:, 0]
    bands = []
    for b in range(1, len(TRUE_BINS)):
        sel = bins == b
        if sel.sum() < 100:
            continue
        h = human_all[sel]
        bands.append(dict(band=f"{TRUE_BINS[b - 1]}-{TRUE_BINS[b]}", elo=(TRUE_BINS[b - 1] + TRUE_BINS[b]) / 2,
                          n=int(sel.sum()), human=float(h.mean()), human_se=float(h.std() / np.sqrt(len(h))),
                          scorer=Scorer(c, sel, args.t_strong)))
    t_lo, t_hi = args.t_min, args.t_max

    # 1. temperature with c = e, no mixing: lower T -> stronger (loss increases with T)
    for b in bands:
        b["t_match"] = solve(lambda T: b["scorer"].expected_loss(b["elo"], T, 0.0), b["human"], t_lo, t_hi)
    pts = np.array([(b["elo"], b["t_match"]) for b in bands if b["t_match"] is not None])
    tcurve = fit_temperature(pts[:, 0], pts[:, 1], segments=args.temp_segments, t_min=t_lo, t_max=t_hi)
    temp = lambda e: temperature_at(tcurve, e)

    # 2. mixing only where the player at T(e) cannot reach the band with an Elo-input correction of
    #    at most `max_correction` (loss decreases with the input and with mix)
    g = c["grid"]
    for b in bands:
        s, T, e = b["scorer"], temp(b["elo"]), b["elo"]
        hi_in = float(min(g[-1], e + args.max_correction))
        b["mix_match"] = 0.0 if s.expected_loss(hi_in, T, 0.0) <= b["human"] else \
            solve(lambda m: s.expected_loss(e, T, m), b["human"], 0.0, 1.0, increasing=False)
    knots_e = [b["elo"] for b in bands]
    m_pts = np.maximum.accumulate([1.0 if b["mix_match"] is None else b["mix_match"] for b in bands])
    first = int(np.argmax(m_pts > 0)) if (m_pts > 0).any() else len(bands)
    # ramp: 0 up to the last band that needs no mixing, then through the needed values
    ramp_e = ([knots_e[first - 1]] if first > 0 else []) + knots_e[first:]
    ramp_m = ([0.0] if first > 0 else []) + list(m_pts[first:])

    def mix(e):
        if not ramp_e or e <= ramp_e[0]:
            return 0.0 if ramp_e else 0.0
        if len(ramp_e) >= 2 and e > ramp_e[-1]:  # extrapolate the last segment, capped at 1
            s = (ramp_m[-1] - ramp_m[-2]) / (ramp_e[-1] - ramp_e[-2])
            return float(min(1.0, ramp_m[-1] + s * (e - ramp_e[-1])))
        return float(np.interp(e, ramp_e, ramp_m))

    # 3. residual LUT: Elo input that matches each band with T(e), mix(e) fixed (loss falls with c)
    for b in bands:
        s, T, m = b["scorer"], temp(b["elo"]), mix(b["elo"])
        b["temperature"], b["mix"] = T, m
        # within +-max_correction only: above ~2400 the player's strength stops rising with its input
        lo_in = float(max(g[0], b["elo"] - args.max_correction))
        hi_in = float(min(g[-1], b["elo"] + args.max_correction))
        b["conditioning"] = solve(lambda x: s.expected_loss(x, T, m), b["human"], lo_in, hi_in, increasing=False)
        b["policy_loss"] = s.expected_loss(b["conditioning"] if b["conditioning"] is not None else b["elo"], T, m)
        b["raw_loss"] = s.expected_loss(b["elo"], 1.0, 0.0)
        b["raw_logp"], b["raw_top1"] = s.likeness(b["elo"], 1.0, 0.0)
        ci = b["conditioning"] if b["conditioning"] is not None else b["elo"]
        b["cal_logp"], b["cal_top1"] = s.likeness(ci, T, m)

    out = dict(player=str(c["player_ckpt"]), strong_player=str(c["strong_ckpt"]), strong_elo=float(c["strong_elo"]),
               strong_temperature=args.t_strong,
               temperature_curve=tcurve,
               mix_ramp=dict(elo=[float(x) for x in ramp_e], mix=[float(x) for x in ramp_m]),
               knots=[dict(elo=b["elo"], conditioning=b["conditioning"], temperature=round(b["temperature"], 4),
                           mix=round(b["mix"], 4)) for b in bands],
               bands=[{k: (round(v, 5) if isinstance(v, float) else v) for k, v in b.items() if k != "scorer"}
                      for b in bands])
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)

    print(f"temperature: {tcurve['value']:.3f} at {tcurve['knot']:.0f}, slopes {1000 * tcurve['slope_lo']:+.3f} / "
          f"{1000 * tcurve['slope_hi']:+.3f} per 1000 Elo below / above, clipped to [{t_lo}, {t_hi}]; rss {tcurve['rss']:.5f}")
    print(f"mix ramp: {list(zip(out['mix_ramp']['elo'], [round(m, 3) for m in out['mix_ramp']['mix']]))}")
    print(f"{'band':<10}{'human%':>7}{'raw%':>7} | {'T@c=e':>6} {'mix':>5} | {'T':>5} {'mix':>5} {'input':>6} {'d':>5}"
          f" {'pol%':>6} | logp raw/cal  top1 raw/cal")
    for b in bands:
        ci = b["conditioning"]
        print(f"{b['band']:<10}{100 * b['human']:7.2f}{100 * b['raw_loss']:7.2f} | "
              f"{(b['t_match'] or float('nan')):6.3f} {(b['mix_match'] if b['mix_match'] is not None else float('nan')):5.2f} | "
              f"{b['temperature']:5.3f} {b['mix']:5.2f} {(ci or float('nan')):6.0f} {((ci or np.nan) - b['elo']):5.0f}"
              f" {100 * b['policy_loss']:6.2f} | {b['raw_logp']:.3f}/{b['cal_logp']:.3f}  {b['raw_top1']:.3f}/{b['cal_top1']:.3f}")
    plot(out, args.out.replace(".json", ".png"))


def plot(out, path):
    plotstyle.setup()
    B = out["bands"]
    e = [b["elo"] for b in B]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    ax = axes[0]
    ax.plot(e, [b["temperature"] for b in B], "o-", color=plotstyle.SERIES[0], label="temperature")
    ax.plot(e, [b["mix"] for b in B], "s-", color=plotstyle.SERIES[1], label="mix toward strong model")
    ax.set_xlabel("target Elo"); ax.set_title("Knobs"); ax.legend(fontsize=8)
    ax = axes[1]
    ax.plot(e, [(b["conditioning"] - b["elo"]) if b["conditioning"] is not None else np.nan for b in B], "o-",
            color=plotstyle.SERIES[0])
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_xlabel("target Elo"); ax.set_ylabel("Elo input - target"); ax.set_title("Residual LUT correction")
    ax = axes[2]
    ax.errorbar(e, [100 * b["human"] for b in B], yerr=[196 * b["human_se"] for b in B], fmt="o-",
                color=plotstyle.SERIES[1], capsize=3, label="humans")
    ax.plot(e, [100 * b["raw_loss"] for b in B], "o--", color="0.5", label="player, input = target, T = 1")
    ax.plot(e, [100 * b["policy_loss"] for b in B], "o-", color=plotstyle.SERIES[0], label="calibrated policy")
    ax.set_xlabel("target Elo"); ax.set_ylabel("expected-score loss per move (%)"); ax.set_title("Strength")
    ax.legend(fontsize=8); ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def settings(calibration, elo):
    """(Elo input, temperature, mix) for a target Elo, interpolating the fitted knots (the strong
    model's temperature is calibration["strong_temperature"])."""
    k = [x for x in calibration["knots"] if x["conditioning"] is not None]
    e = [x["elo"] for x in k]
    mr = calibration["mix_ramp"]
    if "temperature_curve" in calibration:
        T = temperature_at(calibration["temperature_curve"], elo)
    else:  # older single-line calibrations
        tl = calibration["temperature_line"]
        T = float(np.clip(tl["intercept"] + tl["slope"] * elo, tl["t_min"], tl["t_max"]))
    if not mr["elo"] or elo <= mr["elo"][0]:
        m = 0.0
    elif elo > mr["elo"][-1] and len(mr["elo"]) >= 2:
        s = (mr["mix"][-1] - mr["mix"][-2]) / (mr["elo"][-1] - mr["elo"][-2])
        m = float(min(1.0, mr["mix"][-1] + s * (elo - mr["elo"][-1])))
    else:
        m = float(np.interp(elo, mr["elo"], mr["mix"]))
    d = np.interp(elo, e, [x["conditioning"] - x["elo"] for x in k])  # flat beyond the end knots
    return float(elo + d), T, m


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("cache")
    p.add_argument("--player", required=True)
    p.add_argument("--strong", required=True, help="stronger model to mix in (the distilled student)")
    p.add_argument("--analysis", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--grid", default="400:3000:100")
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p = sub.add_parser("fit")
    p.add_argument("--cache", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--t-min", type=float, default=0.3)
    p.add_argument("--t-max", type=float, default=1.5)
    p.add_argument("--temp-segments", type=int, choices=[1, 2], default=2,
                   help="temperature schedule: one line, or two joined at a fitted knot")
    p.add_argument("--t-strong", type=float, default=0.15, help="temperature of the strong model in the mix")
    p.add_argument("--max-correction", type=float, default=300.0,
                   help="largest Elo-input correction the LUT may make before mixing in the strong model")
    args = ap.parse_args(argv)
    (cache if args.cmd == "cache" else fit)(args)


if __name__ == "__main__":
    main()
