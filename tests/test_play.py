"""Inference path and game harness, with untrained tiny models."""
import os
import random

import chess
import numpy as np
import pytest
import torch

from fastelo.board_codec import board_from_codes
from fastelo.eval_strength import DEFAULT_ENGINE
from fastelo.features import decode_records, records_from_boards
from fastelo.models import PlayerNet, PlayerConfig, config_dict
from fastelo.play import MoveRater, PolicyAgent, PlayerPolicy, StockfishAgent, play_game
from fastelo.vocab import legal_mask


def test_records_from_boards_roundtrip():
    rng = random.Random(1)
    board = chess.Board()
    boards = []
    for _ in range(80):
        moves = list(board.legal_moves)
        if not moves:
            break
        board.push(rng.choice(moves))
        boards.append(board.copy())
    d = decode_records(records_from_boards(boards))
    flags = d["castling"] @ torch.tensor([1.0, 2, 4, 8])
    for i, bd in enumerate(boards):
        rebuilt = board_from_codes(d["board"][i].numpy(), int(flags[i]), int(d["ep"][i]))
        assert np.array_equal(legal_mask(rebuilt), legal_mask(bd))
        assert bool(d["black"][i]) == (bd.turn == chess.BLACK)


@pytest.fixture(scope="module")
def ckpts(tmp_path_factory):
    d = tmp_path_factory.mktemp("ckpt")
    torch.manual_seed(0)
    model = PlayerNet(PlayerConfig(d_model=32, n_layers=1, n_heads=4))
    path = str(d / "player.pt")
    torch.save(dict(model=model.state_dict(), kind="player", model_cfg=config_dict(model), step=0), path)
    return {"player": path}


def test_game_runs_and_both_sides_are_rated(ckpts):
    policy = PlayerPolicy(ckpts["player"])
    game = play_game(PolicyAgent(policy, 1500), PolicyAgent(policy, 1200), MoveRater(policy),
                     np.random.default_rng(0), max_plies=24)
    assert game["plies"] > 0
    assert all(r["rating_lo"] <= r["rating"] <= r["rating_hi"] for r in game["trace"])
    assert all(400 <= r["rating"] <= 2600 for r in game["trace"])


@pytest.mark.skipif(not os.path.exists(DEFAULT_ENGINE), reason="stockfish not installed")
def test_stockfish_agent(ckpts):
    policy = PlayerPolicy(ckpts["player"])
    sf = StockfishAgent(DEFAULT_ENGINE, 1400)
    try:
        game = play_game(PolicyAgent(policy, 1500), sf, None, np.random.default_rng(0), max_plies=10)
    finally:
        sf.engine.quit()
    assert game["plies"] == 10


def test_equivalent_elo():
    from fastelo.eval_strength import equivalent_elo
    centers = [1000, 1500, 2000]
    human = [0.06, 0.04, 0.02]
    assert equivalent_elo([0.05, 0.05, 0.05], human, centers) == (1250.0, "crossing")
    assert equivalent_elo([0.01, 0.01, 0.01], human, centers)[1] == "better than every band"
    assert equivalent_elo([0.09, 0.09, 0.09], human, centers)[1] == "worse than every band"
