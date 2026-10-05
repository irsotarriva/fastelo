"""Runtime for resumable multi-GPU runs: optimizer split, remote resume, 2-process DDP (CPU/gloo)."""
import json
import os
import shutil
import socket

import pytest
import torch

from fastelo import distill
from fastelo.cloud import LocalStore, sync_down, sync_up
from fastelo.models import PlayerNet, PlayerConfig
from fastelo.runtime import make_optimizers

from test_distill import labelled  # noqa: F401  (fixtures)
from test_leela_label import part  # noqa: F401
from test_pipeline import dataset  # noqa: F401


class A:  # minimal args
    lr, wd = 1e-3, 0.05

    def __init__(self, optimizer):
        self.optimizer = optimizer


def test_muon_only_on_block_matrices():
    model = PlayerNet(PlayerConfig(d_model=32, n_layers=2, n_heads=4))
    opts = make_optimizers(model, A("muon"))
    assert [type(o).__name__ for o in opts] == ["AdamW", "Muon"]
    names = {id(p): n for n, p in model.named_parameters()}
    muon = [names[id(p)] for g in opts[1].param_groups for p in g["params"]]
    assert muon and all(".layers." in n and n.endswith("weight") for n in muon)
    assert not any(k in n for n in muon for k in ("elo", "piece", "policy", "value", "norm"))
    adam = {names[id(p)] for g in opts[0].param_groups for p in g["params"]}
    assert len(adam) + len(muon) == len(names) and not adam & set(muon)
    assert [type(o).__name__ for o in make_optimizers(model, A("adamw"))] == ["AdamW"]


def test_store_sync(tmp_path):
    store = LocalStore(str(tmp_path / "remote"))
    (tmp_path / "a" / "x").mkdir(parents=True)
    (tmp_path / "a" / "x" / "f.bin").write_bytes(b"12345")
    assert sync_up(store, str(tmp_path / "a"), "pre") == 1
    assert sync_up(store, str(tmp_path / "a"), "pre") == 0  # idempotent
    assert sync_down(store, "pre", str(tmp_path / "b")) == 1
    assert (tmp_path / "b" / "x" / "f.bin").read_bytes() == b"12345"
    assert sync_down(store, "pre", str(tmp_path / "b")) == 0
    # same size, newer remote copy (a checkpoint saved again): fetched, not skipped
    (tmp_path / "a" / "x" / "f.bin").write_bytes(b"54321")
    store.put_file(str(tmp_path / "a" / "x" / "f.bin"), "pre/x/f.bin")
    os.utime(tmp_path / "b" / "x" / "f.bin", (1, 1))
    assert sync_down(store, "pre", str(tmp_path / "b")) == 1
    assert (tmp_path / "b" / "x" / "f.bin").read_bytes() == b"54321"


def _argv(parts, labels, runs, remote, steps, extra=()):
    return ["train", "--parts", str(parts), "--labels", str(labels), "--run", "r", "--runs-dir", str(runs),
            "--remote", remote, "--model", '{"d_model": 32, "n_layers": 1, "n_heads": 4}', "--steps", str(steps),
            "--batch", "64", "--warmup", "5", "--eval-every", "1000", "--log-every", "10", "--ckpt-every", "20",
            "--val-positions", "300", "--threads", "2", *extra]


def test_resume_from_remote_after_losing_the_box(labelled, tmp_path, capsys):
    parts, labels = labelled
    remote = f"file://{tmp_path / 'remote'}"
    distill.main(_argv(parts, labels, tmp_path / "box1", remote, 20, ["--optimizer", "muon"]))
    assert os.path.exists(tmp_path / "remote" / "runs" / "r" / "ckpt.pt")
    capsys.readouterr()
    # a brand-new box (empty runs dir) continues from the remote checkpoint
    distill.main(_argv(parts, labels, tmp_path / "box2", remote, 40, ["--optimizer", "muon"]))
    assert "resumed from step 20" in capsys.readouterr().out
    ck = torch.load(tmp_path / "remote" / "runs" / "r" / "ckpt.pt", weights_only=False)
    assert ck["step"] == 40 and len(ck["opt"]) == 2
    steps = [json.loads(l)["step"] for l in open(tmp_path / "box2" / "r" / "log.jsonl")]
    assert steps[:2] == [10, 20] and steps[-1] == 40  # log history came back with the checkpoint


def test_human_phase_player_from_distilled(labelled, dataset, tmp_path, capsys):  # noqa: F811
    from fastelo import train
    parts, labels = labelled
    remote = f"file://{tmp_path / 'remote'}"
    distill.main(_argv(parts, labels, tmp_path / "runs", remote, 10))  # run "r" = distilled student
    capsys.readouterr()
    common = ["--data", str(dataset), "--runs-dir", str(tmp_path / "runs"), "--remote", remote, "--steps", "12",
              "--warmup", "2", "--log-every", "6", "--eval-every", "12", "--eval-size", "200", "--threads", "2"]
    train.main(["--run", "human", "--init-from", str(tmp_path / "runs" / "r" / "ckpt.pt"),
                "--batch", "64", "--optimizer", "muon", *common])
    out = capsys.readouterr().out
    assert "initialised from" in out and "top1_by_elo" in out
    distilled = torch.load(tmp_path / "runs" / "r" / "ckpt.pt", weights_only=False)
    human = torch.load(tmp_path / "remote" / "runs" / "human" / "ckpt.pt", weights_only=False)
    assert human["model_cfg"] == distilled["model_cfg"] and human["step"] == 12


def _ddp_worker(rank, world, port, argv, out):
    os.environ.update(RANK=str(rank), WORLD_SIZE=str(world), LOCAL_RANK=str(rank), MASTER_ADDR="127.0.0.1",
                      MASTER_PORT=str(port), USE_LIBUV="0")
    import fastelo.runtime as rt
    real_fit = rt.fit

    def fit_and_dump(model, *a, **k):
        steps = real_fit(model, *a, **k)
        torch.save({n: p.detach().clone() for n, p in model.state_dict().items()}, f"{out}_rank{rank}.pt")
        return steps
    rt.fit = fit_and_dump
    distill.main(argv)


def test_two_process_ddp_keeps_ranks_identical(labelled, tmp_path):
    parts, labels = labelled
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    argv = _argv(parts, labels, tmp_path / "runs", f"file://{tmp_path / 'remote'}", 12)
    out = str(tmp_path / "weights")
    torch.multiprocessing.spawn(_ddp_worker, args=(2, port, argv, out), nprocs=2, join=True)
    w0, w1 = torch.load(f"{out}_rank0.pt"), torch.load(f"{out}_rank1.pt")
    for n in w0:
        torch.testing.assert_close(w0[n], w1[n], msg=n)
    log = [json.loads(l) for l in open(tmp_path / "runs" / "r" / "log.jsonl")]
    train = [r for r in log if r["split"] == "train"]
    assert train[-1]["step"] == 10 and train[-1]["items_per_s"] > 0
    assert log[-1]["split"] == "val" and log[-1]["step"] == 12  # final evaluation on the main rank


def test_plan_data_and_rotation_schedule():
    from fastelo.runtime import Rotating, plan_data

    class D:
        world, device = 2, torch.device("cpu")

    where, groups = plan_data(list("abcde"), [4e9] * 5, D, ram_gb=20)  # 10 GB per rank -> 2 units a group
    assert where == "cpu" and groups == [["a", "b"], ["c", "d"], ["e"]]
    assert plan_data(list("ab"), [50e9] * 2, D, ram_gb=20)[1] == [["a"], ["b"]]  # an oversized unit stands alone
    loads = []
    rot = Rotating(groups, [200, 200, 100], lambda g: loads.append(g) or tuple(g), per_step=10)
    assert [rot.group_at(s) for s in (0, 19, 20, 39, 40, 49, 50, 99)] == [0, 0, 1, 1, 2, 2, 0, 2]
    assert rot.at(0) == ("a", "b") and rot.at(5) == ("a", "b") and rot.at(45) == ("e",)
    assert loads == [["a", "b"], ["e"]]  # a resumed run goes straight to its group


def test_training_rotates_through_data_that_does_not_fit(labelled, dataset, tmp_path, capsys):  # noqa: F811
    parts, labels = labelled
    # ~0.5 MB per part and 0.3 MB of "RAM": one part at a time, ~47 steps each at batch 64
    distill.main(_argv(parts, labels, tmp_path / "runs", f"file://{tmp_path / 'remote'}", 60,
                       ["--data-ram-gb", "0.0003", "--eval-every", "30"]))
    out = capsys.readouterr().out
    assert "2 parts in 2 group(s)" in out and "data group 2/2" in out
    log = [json.loads(line) for line in open(tmp_path / "runs" / "r" / "log.jsonl")]
    assert [r["step"] for r in log if r.get("split") == "val"][-1] == 60

    from fastelo import train
    train.main(["--data", str(dataset), "--run", "a", "--runs-dir", str(tmp_path / "runs"),
                "--model", '{"d_model": 32, "n_layers": 1, "n_heads": 4}', "--steps", "70", "--batch", "64",
                "--warmup", "5", "--eval-every", "1000", "--log-every", "10", "--eval-size", "200",
                "--elo-balance", "0.5", "--data-ram-gb", "0.0002", "--threads", "2"])
    out = capsys.readouterr().out
    assert "data group 2/" in out


def test_global_batch_is_exact_when_micro_batch_does_not_divide_it(labelled, tmp_path, capsys):
    parts, labels = labelled
    distill.main(_argv(parts, labels, tmp_path / "runs", f"file://{tmp_path / 'remote'}", 2, ["--micro-batch", "50"]))
    assert "micro-batch 32 x accum 2 x world 1 = 64" in capsys.readouterr().out
