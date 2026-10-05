import argparse
import json

import numpy as np

from fastelo import response as R


def simulated(n, seed):
    """An estimator that sees 60% of the distance from 1500 and adds 150 Elo of noise."""
    rng = np.random.default_rng(seed)
    true = rng.uniform(400, 2600, n)
    est = np.stack([1500 + 0.6 * (true - 1500) + rng.normal(0, 150, n)] * len(R.PLIES), 1).astype(np.float32)
    return dict(true=true, est=est, sigma=np.full_like(est, 150.0), plies=np.array(R.PLIES), step=1)


def test_correction_removes_the_bias_and_keeps_coverage(tmp_path):
    np.savez(tmp_path / "a.npz", **simulated(20000, 0))
    np.savez(tmp_path / "b.npz", **simulated(20000, 1))
    out = tmp_path / "response.json"
    R.fit(argparse.Namespace(derive=str(tmp_path / "a.npz"), closure=str(tmp_path / "b.npz"), out=str(out), smooth=0.3))
    resp = json.load(open(out))
    raw, cor = resp["closure"]["40"]["raw"], resp["closure"]["40"]["corrected"]
    assert abs(raw[0]["bias"]) > 300 and abs(raw[-1]["bias"]) > 300
    assert max(abs(r["bias"]) for r in cor) < 40
    assert all(0.55 < r["coverage"] < 0.85 for r in cor)
    assert max(abs(r["bias"]) for r in resp["combined"]["40"]) < 40
    # apply: a raw 1200 after 40 plies came from a weaker player; the stat. error grows by 1 / 0.6
    a = R.apply(resp, [1200.0], [150.0], [40])
    assert 900 < a["value"][0] < 1100 and a["lo"][0] < a["value"][0] < a["hi"][0]
    assert 180 < 0.5 * (a["hi"][0] - a["lo"][0]) < 320 and a["lo2"][0] < a["lo"][0] and a["syst"][0] >= 0


def test_combine_weights_do_not_depend_on_the_error_bars():
    v, minus, plus = R.combine([1000.0, 1400.0], [100.0, 300.0], [200.0, 100.0], 1.0)
    assert v == 1200.0 and np.isclose(minus, np.hypot(50, 150)) and np.isclose(plus, np.hypot(100, 50))
