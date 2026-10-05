"""Our Leela input encoder + ONNX network must reproduce lc0's own policy prior."""
import os
import random
import re

import chess
import chess.engine
import numpy as np
import pytest

from fastelo.leela import OURS_TO_LC0, LeelaONNX, legal_lc0_indices, lc0_uci

TOOLS = os.path.join(os.path.dirname(__file__), "..", "tools")
LC0 = os.path.join(TOOLS, "lc0", "lc0.exe")
NET = os.path.join(TOOLS, "nets", "t1-256x10-distilled-swa-2432500.pb.gz")
ONNX = os.path.join(TOOLS, "onnx", "t1-256x10-distilled-swa-2432500.f32.onnx")
pytestmark = pytest.mark.skipif(not all(map(os.path.exists, (LC0, NET, ONNX))), reason="lc0 / net not installed")

_P_RE = re.compile(r"^(\S+)\s+\(\s*\d+\s*\).*\(P:\s*([\d.]+)%\)")


def lc0_priors(engine, board):
    """{uci: prior %} from lc0's VerboseMoveStats after a 1-node search."""
    priors = {}
    with engine.analysis(board, chess.engine.Limit(nodes=1)) as analysis:
        for info in analysis:
            m = _P_RE.match(info.get("string", ""))
            if m and m.group(1) != "node":
                priors[m.group(1)] = float(m.group(2))
    return priors


def positions(n_games=12, seed=0):
    """Positions with full move history from random games; biased towards pawn pushes so
    promotions, en passant and castling rights changes all show up."""
    rng = random.Random(seed)
    for _ in range(n_games):
        board = chess.Board()
        for _ in range(rng.randint(20, 120)):
            moves = list(board.legal_moves)
            if not moves or board.is_game_over():
                break
            pawn = [m for m in moves if board.piece_type_at(m.from_square) == chess.PAWN]
            board.push(rng.choice(pawn if pawn and rng.random() < 0.4 else moves))
            yield board.copy()


def test_repetition_plane_matches_lc0():
    """Knight shuffles repeat positions inside the 8-ply history window."""
    net = LeelaONNX(ONNX, threads=4)
    engine = chess.engine.SimpleEngine.popen_uci([LC0, f"--weights={NET}"])
    engine.configure({"VerboseMoveStats": True, "PolicyTemperature": 1.0})
    board = chess.Board()
    worst, repeated = 0.0, 0
    try:
        for uci in "e2e4 e7e5 g1f3 b8c6 f3g1 c6b8 g1f3 b8c6 f3g1 c6b8 g1f3".split():
            board.push_uci(uci)
            ours, _ = net.move_distribution(board)
            ref = lc0_priors(engine, board)
            worst = max([worst] + [abs(100 * p - ref[lc0_uci(board, m)]) for m, p in ours.items()])
            repeated += board.is_repetition(2)
    finally:
        engine.quit()
    assert repeated >= 4
    assert worst < 0.1, f"max prior difference {worst:.3f} percentage points"


def test_mapping_is_injective_per_position():
    for board in positions(4):
        _, lc0 = legal_lc0_indices(board)
        assert len(set(lc0.tolist())) == len(lc0)
    assert OURS_TO_LC0.min() >= 0


def test_matches_lc0_policy():
    net = LeelaONNX(ONNX, threads=4)
    engine = chess.engine.SimpleEngine.popen_uci([LC0, f"--weights={NET}"])
    engine.configure({"VerboseMoveStats": True, "PolicyTemperature": 1.0})
    worst, n, promo_checked = 0.0, 0, 0
    try:
        for i, board in enumerate(positions()):
            if i % 7:  # a spread of positions keeps the test fast
                continue
            ours, _ = net.move_distribution(board)
            ref = lc0_priors(engine, board)
            assert set(ref) == {lc0_uci(board, m) for m in ours}, board.fen()
            for mv, p in ours.items():
                worst = max(worst, abs(100 * p - ref[lc0_uci(board, mv)]))
            promo_checked += any(m.promotion for m in ours)
            n += 1
    finally:
        engine.quit()
    assert n > 20
    assert promo_checked > 0, "no promotion positions were exercised"
    assert worst < 0.1, f"max prior difference {worst:.3f} percentage points"
