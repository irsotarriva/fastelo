"""The labeller must reproduce the verified reference encoder exactly."""
import json
import os

import chess
import numpy as np
import pytest
import torch

from fastelo import leela_label
from fastelo.features import _OFF
from fastelo.leela import LeelaONNX, encode_planes, legal_lc0_indices
from fastelo.vocab import decode_move

from test_pipeline import dataset  # noqa: F401  (module-scoped synthetic dataset fixture)

ONNX = os.path.join(os.path.dirname(__file__), "..", "tools", "onnx", "t1-256x10-distilled-swa-2432500.f32.onnx")


@pytest.fixture(scope="module")
def part(dataset, tmp_path_factory):  # noqa: F811
    out = tmp_path_factory.mktemp("parts")
    leela_label.main(["pack", "--data", str(dataset), "--split", "train", "--out", str(out),
                      "--positions-per-part", "3000", "--parts", "2", "--workers", "2"])
    return out


def replay(raw, ply):
    """Yield (i, board-with-history) for every position of a part."""
    board = None
    for i in range(len(raw)):
        if ply[i] == 0:
            board = chess.Board()
        yield i, board.copy()
        mv = int(raw[i, _OFF["move"]]) | int(raw[i, _OFF["move"] + 1]) << 8
        board.push(decode_move(mv, board.turn))


def test_pack_encoder_legal_moves_and_repetitions(part):
    d = part / "part_0000"
    raw, ply, rep = (np.load(d / f) for f in ("records.npy", "ply.npy", "rep.npy"))
    off, lc0, ours = (np.load(d / f) for f in ("legal_off.npy", "legal_lc0.npy", "legal_ours.npy"))
    assert json.load(open(d / "meta.json"))["positions"] == len(raw) >= 3000
    planes = leela_label.planes_from_records(torch.from_numpy(raw), torch.from_numpy(ply.astype(np.int64)),
                                             torch.from_numpy(rep.astype(bool)), torch.arange(len(raw))).numpy()
    for i, board in replay(raw, ply):
        np.testing.assert_array_equal(planes[i], encode_planes(board), err_msg=f"position {i}")
        assert bool(rep[i]) == board.is_repetition(2)
        ref_ours, ref_lc0 = legal_lc0_indices(board)
        assert sorted(zip(ours[off[i]:off[i + 1]], lc0[off[i]:off[i + 1]])) == sorted(zip(ref_ours, ref_lc0))


def test_parts_are_disjoint(part):
    a, b = (np.load(part / p / "records.npy") for p in ("part_0000", "part_0001"))
    ka = {r.tobytes() for r in a}
    assert not any(r.tobytes() in ka for r in b[np.load(part / "part_0001" / "ply.npy") > 12])


@pytest.mark.skipif(not os.path.exists(ONNX), reason="Leela ONNX net not converted")
def test_run_matches_reference(part, tmp_path):
    shipped, out = tmp_path / "shipped", tmp_path / "out"
    leela_label.main(["ship", "--parts", str(part), "--out", str(shipped)])  # the files the GPU machine gets
    assert (shipped / "part_0000" / "legal_ours.npy.zst").exists()  # shipped for training
    mirror = tmp_path / "mirror_out"
    leela_label.main(["run", "--parts", str(shipped), "--onnx", ONNX, "--out", str(out), "--batch", "512",
                      "--copy-to", str(mirror)])
    assert (mirror / "part_0001" / "wdl.npy.zst").exists()
    lab = leela_label.load_labels(part / "part_0000", out / "part_0000")
    raw, ply = lab["records"], np.load(part / "part_0000" / "ply.npy")
    net = LeelaONNX(ONNX, threads=4)
    for i, board in replay(raw, ply):
        if i % 97:
            continue
        pol, wdl = net.run(encode_planes(board)[None])
        _, ref_lc0 = legal_lc0_indices(board)
        got = lab["logits"][lab["offsets"][i]:lab["offsets"][i + 1]].astype(np.float32)
        np.testing.assert_allclose(np.sort(got), np.sort(pol[0, ref_lc0]), atol=2e-2)
        np.testing.assert_allclose(lab["wdl"][i].astype(np.float32), wdl[0], atol=2e-3)
    # resume on a fresh machine: nothing local, but the copy exists -> every part is skipped
    fresh = tmp_path / "fresh_vm_out"
    leela_label.main(["run", "--parts", str(shipped), "--onnx", ONNX, "--out", str(fresh), "--copy-to", str(mirror)])
    assert not fresh.exists() or not any(fresh.iterdir())
