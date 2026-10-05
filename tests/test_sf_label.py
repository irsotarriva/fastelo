"""Stockfish labeller on the synthetic pipeline dataset."""
import os

import chess
import numpy as np
import pytest
import torch

from fastelo import sf_label
from fastelo.board_codec import board_from_codes
from fastelo.eval_strength import DEFAULT_ENGINE
from fastelo.features import decode_records
from fastelo.vocab import legal_mask

from test_pipeline import dataset  # noqa: F401  (module-scoped fixture)

pytestmark = pytest.mark.skipif(not os.path.exists(DEFAULT_ENGINE), reason="stockfish not installed")


def test_labels_cover_every_legal_move_and_resume(dataset, tmp_path, capsys):  # noqa: F811
    out = tmp_path / "labels"
    argv = ["--data", str(dataset), "--split", "train", "--out", str(out), "--positions", "60",
            "--chunk", "25", "--nodes", "50", "--workers", "2"]
    sf_label.main(argv)
    chunks = sorted(p for p in os.listdir(out) if p.startswith("chunk_"))
    assert len(chunks) == 3
    keys = set()
    for c in chunks:
        d = out / c
        rec = np.load(d / "records.npy")
        off, moves, scores = (np.load(d / f) for f in ("offsets.npy", "moves.npy", "scores.npy"))
        assert off[-1] == len(moves) == len(scores) and len(off) == len(rec) + 1
        dec = decode_records(torch.from_numpy(rec))
        flags = (dec["castling"] @ torch.tensor([1.0, 2, 4, 8])).int()
        for i in range(len(rec)):
            board = board_from_codes(dec["board"][i].numpy(), int(flags[i]), int(dec["ep"][i]))
            labeled = moves[off[i]:off[i + 1]]
            assert sorted(labeled.tolist()) == sorted(np.flatnonzero(legal_mask(board)).tolist())
            keys.add(rec[i, :26].tobytes())
    assert len(keys) == 60, "positions must be distinct"
    # Resume: rerunning does no work and leaves the chunks untouched.
    capsys.readouterr()
    sf_label.main(argv)
    assert "chunk" not in capsys.readouterr().out.replace("chunks", "")
    # Different settings into the same directory are refused.
    with pytest.raises(SystemExit):
        sf_label.main(argv[:-4] + ["--nodes", "99", "--workers", "2"])


def test_pov_cp_mate_and_clamp():
    board = chess.Board("6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1")  # Rd8#
    engine = chess.engine.SimpleEngine.popen_uci(DEFAULT_ENGINE)
    try:
        info = engine.analyse(board, chess.engine.Limit(nodes=2000))
    finally:
        engine.quit()
    assert sf_label.pov_cp(info["score"]) == 30000 - 1
