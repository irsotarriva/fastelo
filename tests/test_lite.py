"""The NumPy inference path (fastelo.lite) must give the same numbers as the PyTorch models."""
import json

import chess
import numpy as np
import torch
from safetensors.torch import save_file

from fastelo import lite
from fastelo.models import PlayerNet, PlayerConfig, config_dict
from fastelo.play import CalibratedPolicy, MoveRater


def export(model, kind, path):
    save_file({k: v.half() for k, v in model.state_dict().items()}, str(path) + ".safetensors")
    json.dump(dict(kind=kind, model_cfg=config_dict(model), step=1), open(str(path) + ".json", "w"))
    model.load_state_dict({k: v.half().float() for k, v in model.state_dict().items()})  # as load_model sees it


def game(n):
    board, boards, moves = chess.Board(), [], []
    for i in range(n):
        m = sorted(board.legal_moves, key=lambda m: m.uci())[(7 * i) % board.legal_moves.count()]
        boards.append(board.copy(stack=False)); moves.append(m); board.push(m)
    return board, boards, moves


def test_lite_matches_pytorch(tmp_path):
    torch.manual_seed(0)
    for name in ("player", "strong_player"):
        export(PlayerNet(PlayerConfig(d_model=32, n_layers=2, n_heads=4)).eval(), "player", tmp_path / name)
    cal = dict(strong_elo=3000.0, strong_temperature=0.15, player="player.safetensors",
               strong_player="strong_player.safetensors",
               temperature_curve=dict(knot=1650.0, value=0.9, slope_lo=-1e-5, slope_hi=-7e-4, t_min=0.3, t_max=1.5),
               mix_ramp=dict(elo=[2100.0, 2300.0, 2500.0], mix=[0.0, 0.1, 1.0]),
               knots=[dict(elo=700.0, conditioning=690.0), dict(elo=2500.0, conditioning=2600.0)])
    json.dump(cal, open(tmp_path / "calibration.json", "w"))
    engine = lite.Engine(str(tmp_path))
    ref = CalibratedPolicy(str(tmp_path / "calibration.json"))
    board, boards, moves = game(31)
    for target in (500, 1500, 2250, 2600):       # below the knots, no mix, mixed, past the ramp
        for b, ply in ((chess.Board(), 0), (boards[17], 17), (board, 31)):   # white and black to move
            p = engine.distribution(b, ply, target)
            q = ref.distribution(b, ply, target).numpy()
            assert np.abs(p - q).max() < 2e-4, (target, ply)
    # the rating used by the demo: likelihood of the player's own moves, scored incrementally
    rater, state, ref_state = MoveRater(ref), {}, {}
    for n in (1, 2, 9, 31):
        for side in (1,):
            r = engine.rating(boards[:n], moves[:n], side, state)
            q = rater.rating(boards[:n], moves[:n], side, state=ref_state)
            assert len(r["value"]) == n // 2 and np.allclose(r["value"], q["value"], atol=0.5)
            assert np.allclose(r["hi"] - r["lo"], q["hi"] - q["lo"], atol=0.5)
    assert np.allclose(engine.rating(boards, moves, 0, {})["value"], rater.rating(boards, moves, 0)["value"], atol=0.5)


def test_turn_is_stateless_and_matches_the_session(tmp_path):
    """The web demo's request handler: same bot targets and measurements with or without the cache."""
    test_lite_matches_pytorch(tmp_path)          # writes a model folder
    engine = lite.Engine(str(tmp_path))
    states, ucis, out = {}, [], None
    for i in range(6):
        board = chess.Board()
        for u in ucis:
            board.push_uci(u)
        ucis.append(sorted(m.uci() for m in board.legal_moves)[i])
        req = dict(moves=ucis, user_white=True, game_id="g1", previous=[])
        out = lite.turn(engine, req, states, np.random.default_rng(i))
        cold = lite.turn(engine, dict(req, game_id=""), {}, np.random.default_rng(i))   # no cache
        assert out["bot_move"] == cold["bot_move"] and out["curve"] == cold["curve"]
        assert len(out["curve"]["value"]) == i + 1 and out["measurement"]["user_moves"] == i + 1
        assert 600 <= out["bot_elo"] <= 2400 and len(out["moves"]) == len(ucis) + 1
        ucis = out["moves"]
    # a second game as black: the bot opens, aiming at the rating from the first game
    first = out["measurement"]
    out2 = lite.turn(engine, dict(moves=[], user_white=False, game_id="g2", previous=[first]), states,
                     np.random.default_rng(0))
    assert out2["bot_move"] and out2["measurement"] is None and out2["combined"]["games"] == 1
    assert out2["bot_elo"] == float(np.clip(first["mean"], 600, 2400))
    import pytest
    with pytest.raises(ValueError):
        lite.turn(engine, dict(moves=["e2e5"], user_white=True), {}, np.random.default_rng(0))
    assert lite.turn(engine, dict(moves=ucis, user_white=True, bot=False), {}, np.random.default_rng(0))["bot_move"] is None
