"""Distillation: CSR softmax helpers, data store, loss and a short training run."""
import os

import numpy as np
import pytest
import torch

from fastelo import distill, leela_label
from fastelo.vocab import VOCAB_SIZE

from test_leela_label import ONNX, part  # noqa: F401  (fixtures)
from test_pipeline import dataset  # noqa: F401


def test_segment_ops_match_dense():
    torch.manual_seed(0)
    lens = torch.tensor([3, 1, 5, 2])
    x = torch.randn(int(lens.sum()))
    seg = torch.repeat_interleave(torch.arange(4), lens)
    got = distill.segment_log_softmax(x, seg, 4)
    start = 0
    for i, l in enumerate(lens.tolist()):
        torch.testing.assert_close(got[start:start + l], torch.log_softmax(x[start:start + l], 0))
        assert distill.segment_argmax(x, seg, 4)[i] == start + int(x[start:start + l].argmax())
        start += l
    flat, s2 = distill.csr_gather_index(torch.tensor([10, 0, 4]), torch.tensor([2, 0, 3]))
    assert flat.tolist() == [10, 11, 4, 5, 6] and s2.tolist() == [0, 0, 2, 2, 2]


@pytest.fixture(scope="module")
def labelled(part, tmp_path_factory):  # noqa: F811
    if not os.path.exists(ONNX):
        pytest.skip("Leela ONNX net not converted")
    out = tmp_path_factory.mktemp("labels")
    leela_label.main(["run", "--parts", str(part), "--onnx", ONNX, "--out", str(out), "--batch", "512"])
    return part, out


def test_store_batch_and_loss(labelled):
    parts, labels = labelled
    store = distill.DistillStore(str(parts), str(labels))
    idx = torch.arange(0, 200)
    b = store.batch(idx)
    assert b["legal_moves"].max() < VOCAB_SIZE and b["legal_seg"].max() == 199
    # teacher WDL was reordered to our [loss, draw, win] and is a distribution
    torch.testing.assert_close(b["teacher_wdl"].sum(-1), torch.ones(200), atol=2e-3, rtol=0)
    from fastelo.models import PlayerNet, PlayerConfig
    model = PlayerNet(PlayerConfig(d_model=32, n_layers=1, n_heads=4))
    loss, m = distill.distill_loss(model, b)
    assert torch.isfinite(loss) and m["kl"] >= -1e-4


def _train_argv(parts, labels, runs, steps, extra=()):
    return ["train", "--parts", str(parts), "--labels", str(labels), "--run", "t", "--runs-dir", str(runs),
            "--model", '{"d_model": 32, "n_layers": 1, "n_heads": 4}', "--steps", str(steps), "--batch", "64",
            "--warmup", "5", "--eval-every", "100000", "--log-every", "20", "--val-positions", "500",
            "--threads", "2", *extra]


def test_nonfinite_batches_are_skipped_then_abort(labelled, tmp_path, monkeypatch):
    import json
    parts, labels = labelled
    real, calls = distill.distill_loss, {"n": 0}

    def flaky(model, b, tau=1.0, value_weight=0.5, every=10):
        loss, m = real(model, b, tau, value_weight)
        calls["n"] += 1
        return (loss * float("nan"), m) if calls["n"] % every == 0 else (loss, m)

    # 1 in 10 batches non-finite (10% > 1%): skipped, dumped, and the run stops at the first log line.
    monkeypatch.setattr(distill, "distill_loss", flaky)
    with pytest.raises(RuntimeError, match="numerically unstable"):
        distill.main(_train_argv(parts, labels, tmp_path, 60))
    assert list((tmp_path / "t").glob("nonfinite_batch_*.pt"))
    rec = [json.loads(l) for l in open(tmp_path / "t" / "log.jsonl")][-1]
    assert rec["nonfinite_batches"] == 2 and np.isfinite(rec["kl"])
    # Tolerated when rare enough: the same run with a 20% budget finishes.
    calls["n"] = 0
    distill.main(_train_argv(parts, labels, tmp_path / "ok", 40, ["--max-bad-frac", "0.2"]))


def test_short_training_reduces_kl(labelled, tmp_path):
    parts, labels = labelled
    distill.main(["train", "--parts", str(parts), "--labels", str(labels), "--run", "t", "--runs-dir", str(tmp_path),
                  "--model", '{"d_model": 64, "n_layers": 2, "n_heads": 4}', "--steps", "150", "--batch", "128",
                  "--lr", "3e-3", "--warmup", "10", "--eval-every", "150", "--log-every", "25",
                  "--val-positions", "500", "--threads", "4"])
    import json
    logs = [json.loads(l) for l in open(tmp_path / "t" / "log.jsonl")]
    train = [r for r in logs if r["split"] == "train"]
    assert train[-1]["kl"] < 0.85 * train[0]["kl"], (train[0]["kl"], train[-1]["kl"])
    assert (tmp_path / "t" / "ckpt.pt").exists()
