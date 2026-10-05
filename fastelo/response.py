"""Bias correction of a rating estimate, with statistical and systematic errors.

    python -m fastelo.likelihood measure --ckpt runs/player/ckpt.pt --data <dataset> --split val --out lik_val.npz
    python -m fastelo.likelihood measure --ckpt runs/player/ckpt.pt --data <dataset> --split test --out lik_test.npz
    python -m fastelo.response fit --derive lik_val.npz --closure lik_test.npz --out rating_response.json

An estimate of a noisy quantity is pulled toward the middle of the scale, more so the less of
the game it has seen. A measurement file holds, for the same number of players in every 200-Elo
band of true rating (so the tails are as well measured as the middle), the estimate and its sigma
after k plies. `fit` finds, for each k, a monotone map f_k with <f_k(estimate) | true> = true at
every true rating - the correction is unbiased by construction, which inverting the mean response
<estimate | true> is not where that response bends. The map is piecewise linear in the estimate.

Errors after correction:
  statistical  the estimate +- sigma mapped through f_k -> asymmetric (lo, hi)
  systematic   the bias the correction leaves on independent games (non-closure), per band; it
               is common to all of a player's games, so combining games does not reduce it
`apply()` is what the demo calls.
"""
import argparse
import json

import numpy as np

PLIES = [6, 10, 16, 20, 30, 40, 60, 80, 120, 160]
BANDS = list(range(400, 2800, 200))        # 400-600 ... 2400-2600
GRID = np.arange(400, 2601, 50.0)          # true-rating grid of the response curves
KNOTS = np.arange(200, 2801, 100.0)        # estimate grid of the correction map
REPORT_RANGE = (600.0, 2200.0)             # ratings are quoted inside this; beyond it only "below 600" / "2200+"
VALUE_RANGE = (300.0, 2700.0)              # corrected values stay within 100 Elo of the range the correction was measured on


# --------------------------------------------------------------------------------- fit
def isotonic(y, w):
    """Non-decreasing least-squares fit (pool adjacent violators)."""
    y, w = list(map(float, y)), list(map(float, w))
    blocks = [[v, wt, 1] for v, wt in zip(y, w)]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] > blocks[i + 1][0] + 1e-12:
            a, b = blocks[i], blocks[i + 1]
            blocks[i:i + 2] = [[(a[0] * a[1] + b[0] * b[1]) / (a[1] + b[1]), a[1] + b[1], a[2] + b[2]]]
            i = max(i - 1, 0)
        else:
            i += 1
    return np.concatenate([np.full(n, v) for v, _, n in blocks])


def response_curve(true, est, width=100.0):
    """<est | true> on GRID: Gaussian-kernel local mean, made monotone (a diagnostic: its slope
    says how much of a change in true rating the estimate sees)."""
    ok = ~np.isnan(est)
    t, e = true[ok], est[ok]
    w = np.exp(-0.5 * ((GRID[:, None] - t[None, :]) / width) ** 2)
    n_eff = w.sum(1)
    return isotonic((w * e[None, :]).sum(1) / np.maximum(n_eff, 1e-9), n_eff)


def _hat(x):
    x = np.clip(x, KNOTS[0], KNOTS[-1])
    j = np.clip(np.searchsorted(KNOTS, x, "right") - 1, 0, len(KNOTS) - 2)
    t = (x - KNOTS[j]) / (KNOTS[j + 1] - KNOTS[j])
    P = np.zeros((len(x), len(KNOTS)))
    r = np.arange(len(x))
    P[r, j], P[r, j + 1] = 1 - t, t
    return P


def unbiased_map(true, est, smooth=0.3, bin_width=50):
    """Values on KNOTS of the map f with <f(est) | true> = true in every bin of true rating (least
    squares, second-difference smoothing, made monotone)."""
    ok = ~np.isnan(est)
    t, P = true[ok], _hat(est[ok])
    A, b = [], []
    for lo in np.arange(BANDS[0], BANDS[-1], bin_width):
        sel = (t >= lo) & (t < lo + bin_width)
        if sel.sum() >= 30:
            A.append(P[sel].mean(0))
            b.append(t[sel].mean())
    D = np.diff(np.eye(len(KNOTS)), 2, axis=0) * smooth
    coef = np.linalg.lstsq(np.r_[np.array(A), D], np.r_[np.array(b), np.zeros(len(D))], rcond=None)[0]
    return np.clip(np.maximum.accumulate(coef), *VALUE_RANGE)


def correct(coef, plies, est, sigma, k, n_sigma=(1.0,), sigma_scale=None):
    """Corrected value and asymmetric bounds for estimates after k plies (arrays). Returns
    [value, lo_1, hi_1, lo_2, hi_2, ...], one (lo, hi) pair per entry of n_sigma; the map is
    interpolated linearly between the ply counts it was fitted at. sigma_scale (one factor per
    fitted ply count) widens or narrows the estimate's own sigma before it is mapped."""
    est, sigma = np.atleast_1d(np.asarray(est, float)), np.atleast_1d(np.asarray(sigma, float))
    coef, plies = np.asarray(coef, float), np.asarray(plies, float)
    k = np.clip(np.broadcast_to(np.asarray(k, float), est.shape), plies[0], plies[-1])
    if sigma_scale is not None:
        sigma = sigma * np.interp(k, plies, sigma_scale)
    j = np.clip(np.searchsorted(plies, k, "right") - 1, 0, len(plies) - 2)
    t = (k - plies[j]) / (plies[j + 1] - plies[j])
    xs = [est] + [est + sgn * n * sigma for n in n_sigma for sgn in (-1, 1)]
    out = [np.empty(est.shape) for _ in xs]
    for jj in np.unique(j):
        sel = j == jj
        for o, x in zip(out, xs):
            o[sel] = (1 - t[sel]) * np.interp(x[sel], KNOTS, coef[jj]) + t[sel] * np.interp(x[sel], KNOTS, coef[jj + 1])
    return out


def combine(values, minus, plus, resolution):
    """Several games of one player -> (value, minus, plus). Each game is weighted by the
    resolution (Elo) of a game of its length, not by its own error bar: the error bar depends on
    the estimate, and weights that depend on the value would bias the mean."""
    values, minus, plus = (np.asarray(x, float) for x in (values, minus, plus))
    w = 1.0 / np.broadcast_to(np.asarray(resolution, float), values.shape) ** 2
    w = w / w.sum()
    return float((w * values).sum()), float(np.sqrt((w ** 2 * minus ** 2).sum())), float(np.sqrt((w ** 2 * plus ** 2).sum()))


def resolution(response, plies_seen):
    """Typical error (Elo) of one corrected game of that many plies."""
    return np.interp(plies_seen, response["plies"], np.minimum.accumulate(response["resolution"]))


def band_table(true, value, lo=None, hi=None):
    rows = []
    for b in BANDS[:-1]:
        sel = (true >= b) & (true < b + 200) & ~np.isnan(value)
        if sel.sum() < 20:
            continue
        d = value[sel] - true[sel]
        row = dict(band=f"{b}-{b + 200}", n=int(sel.sum()), bias=round(float(d.mean()), 1),
                   bias_err=round(float(d.std() / np.sqrt(sel.sum())), 1), mae=round(float(np.abs(d).mean()), 1))
        if lo is not None:
            row["coverage"] = round(float(((true[sel] >= lo[sel]) & (true[sel] <= hi[sel])).mean()), 3)
            row["mean_minus"] = round(float((value[sel] - lo[sel]).mean()), 1)
            row["mean_plus"] = round(float((hi[sel] - value[sel]).mean()), 1)
        rows.append(row)
    return rows


def fit(args):
    d, c = np.load(args.derive), np.load(args.closure)
    plies = d["plies"].astype(float)
    use = [j for j in range(len(plies)) if (~np.isnan(d["est"][:, j])).sum() > 500]
    plies_used = plies[use]
    curves = np.stack([response_curve(d["true"], d["est"][:, j]) for j in use])
    coef = np.stack([unbiased_map(d["true"], d["est"][:, j], args.smooth) for j in use])
    # Factor on the estimate's sigma that makes the corrected 1-sigma interval hold 68.3% of the
    # true ratings on the derive set (1 = the sigma was already right).
    scale = []
    for j, jd in enumerate(use):
        ok = ~np.isnan(d["est"][:, jd])
        e, s_, t = d["est"][ok, jd], d["sigma"][ok, jd], d["true"][ok]
        a, b = 0.3, 4.0
        for _ in range(30):
            m = 0.5 * (a + b)
            _, lo, hi = correct(coef, plies_used, e, s_ * m, plies_used[j])
            a, b = (m, b) if ((t >= lo) & (t <= hi)).mean() < 0.683 else (a, m)
        scale.append(round(0.5 * (a + b), 3))
    scale = np.array(scale)
    res = []
    for j, jd in enumerate(use):  # resolution of one corrected game, all bands weighted alike
        ok = ~np.isnan(d["est"][:, jd])
        v = correct(coef, plies_used, d["est"][ok, jd], d["sigma"][ok, jd], plies_used[j])[0]
        res.append(float(np.sqrt(np.mean((v - d["true"][ok]) ** 2))))
    out = dict(step=int(d["step"]), plies=plies_used.tolist(), knots=KNOTS.tolist(), coef=coef.round(1).tolist(),
               resolution=np.round(res, 1).tolist(), sigma_scale=scale.tolist(), closure={}, systematic={}, combined={})
    print("factor on the sigma by plies seen:", dict(zip(plies_used.astype(int).tolist(), scale.tolist())))
    print(f"correction derived from {len(d['true'])} players, closure on {len(c['true'])}")
    print("resolution of one corrected game by plies seen:", dict(zip(plies_used.astype(int).tolist(), np.round(res).tolist())))
    print("slope of the raw response (d estimate / d true) by plies seen, at 1000 / 1500 / 2000:")
    for j, k in enumerate(plies_used):
        s = np.gradient(curves[j], GRID)
        print(f"  {k:>4.0f} plies: {np.interp(1000, GRID, s):.2f} {np.interp(1500, GRID, s):.2f} {np.interp(2000, GRID, s):.2f}")
    rng = np.random.default_rng(0)
    for k in (20, 40, 80):
        if k not in plies_used:
            continue
        jc = list(c["plies"]).index(k)
        est, sig, true = c["est"][:, jc], c["sigma"][:, jc], c["true"]
        ok = ~np.isnan(est)
        est, sig, true = est[ok], sig[ok], true[ok]
        val, lo, hi = correct(coef, plies_used, est, sig, k, sigma_scale=scale)
        raw = band_table(true, est, est - sig, est + sig)
        cor = band_table(true, val, lo, hi)
        out["closure"][str(k)] = dict(raw=raw, corrected=cor)
        # systematic: the residual bias on independent games (and how well it is known), per band
        out["systematic"][str(k)] = {r["band"]: round(float(np.hypot(r["bias"], r["bias_err"])), 1) for r in cor}
        print(f"\nafter {k} plies (closure set)   band: raw bias -> corrected bias (+-stat) | MAE raw -> corr | "
              f"coverage raw -> corr | stat -/+ | syst")
        for a, b in zip(raw, cor):
            print(f"  {a['band']:<10} n={a['n']:>4}  {a['bias']:+7.1f} -> {b['bias']:+7.1f} (+-{b['bias_err']:.0f}) | "
                  f"{a['mae']:6.1f} -> {b['mae']:6.1f} | {a['coverage']:.2f} -> {b['coverage']:.2f} | "
                  f"-{b['mean_minus']:.0f} / +{b['mean_plus']:.0f} | {out['systematic'][str(k)][b['band']]:.0f}")
        # several games of "one player": groups of 5 from the same band, combined as the demo does
        rows = []
        for b in BANDS[:-1]:
            idx = rng.permutation(np.flatnonzero((true >= b) & (true < b + 200)))
            idx = idx[:len(idx) // 5 * 5].reshape(-1, 5)
            if len(idx) < 10:
                continue
            t = true[idx].mean(1)
            w = 1.0 / sig[idx] ** 2
            raw5 = (w * est[idx]).sum(1) / w.sum(1)
            cmb = np.array([combine(val[i], val[i] - lo[i], hi[i] - val[i], 1.0) for i in idx])
            syst = out["systematic"][str(k)].get(f"{b}-{b + 200}", 0.0)
            inside = (t >= cmb[:, 0] - np.hypot(cmb[:, 1], syst)) & (t <= cmb[:, 0] + np.hypot(cmb[:, 2], syst))
            rows.append(dict(band=f"{b}-{b + 200}", groups=len(idx), raw_bias=round(float((raw5 - t).mean()), 1),
                             bias=round(float((cmb[:, 0] - t).mean()), 1),
                             raw_mae=round(float(np.abs(raw5 - t).mean()), 1),
                             mae=round(float(np.abs(cmb[:, 0] - t).mean()), 1), coverage=round(float(inside.mean()), 3)))
        out["combined"][str(k)] = rows
        print(f"  5 games combined, {k} plies each:  band: raw bias -> corrected | MAE raw -> corr | coverage (stat+syst)")
        for r in rows:
            print(f"    {r['band']:<10} {r['raw_bias']:+7.1f} -> {r['bias']:+7.1f} | {r['raw_mae']:6.1f} -> {r['mae']:6.1f} | "
                  f"{r['coverage']:.2f}")
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nsaved {args.out}")


# ------------------------------------------------------------------------------- apply
def systematic(response, value, plies_seen):
    """Systematic error (Elo) at corrected values, from the closure table nearest in plies."""
    ks = sorted(int(k) for k in response["systematic"])
    table = response["systematic"][str(min(ks, key=lambda x: abs(x - float(np.max(plies_seen)))))]
    out = []
    for v in np.atleast_1d(value):
        b = int(np.clip(v // 200 * 200, BANDS[0], BANDS[-2]))
        out.append(table.get(f"{b}-{b + 200}", max(table.values())))
    return np.array(out)


def apply(response, est, sigma, plies_seen):
    """Corrected estimates (arrays): dict(value, lo, hi, lo2, hi2, syst). lo/hi are the 1-sigma and
    lo2/hi2 the 2-sigma statistical bounds."""
    v, lo, hi, lo2, hi2 = correct(response["coef"], response["plies"], est, sigma, plies_seen, n_sigma=(1.0, 2.0),
                                  sigma_scale=response.get("sigma_scale"))
    return dict(value=v, lo=lo, hi=hi, lo2=lo2, hi2=hi2, syst=systematic(response, v, plies_seen))


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("fit")
    p.add_argument("--derive", required=True)
    p.add_argument("--closure", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--smooth", type=float, default=0.3, help="second-difference penalty of the correction map")
    args = ap.parse_args(argv)
    fit(args)


if __name__ == "__main__":
    main()
