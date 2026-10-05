"""Synthetic phase-varying games: batched play, dataset format, label formula."""
import json

import numpy as np
import torch

from fastelo import synth
from fastelo.board_codec import board_from_codes
from fastelo.features import GameStore
from fastelo.models import PlayerNet, PlayerConfig
from fastelo.runtime import make_optimizers  # noqa: F401  (import check)
from fastelo.vocab import legal_mask


def tiny_calibration(tmp_path):
    torch.manual_seed(0)
    paths = []
    for name in ("player", "strong"):
        cfg = PlayerConfig(d_model=32, n_layers=1, n_heads=4)
        m = PlayerNet(cfg)
        p = tmp_path / f"{name}.pt"
        torch.save(dict(kind="player", model_cfg=cfg.__dict__, model=m.state_dict(), step=0), p)
        paths.append(str(p))
    cal = dict(player=paths[0], strong_player=paths[1], strong_elo=3000.0, strong_temperature=0.15,
               temperature_curve=dict(knot=1650.0, value=0.9, slope_lo=0.0, slope_hi=-0.0007, t_min=0.3, t_max=1.5),
               mix_ramp=dict(elo=[2100.0, 2500.0], mix=[0.0, 1.0]),
               knots=[dict(elo=700.0, conditioning=700.0), dict(elo=2500.0, conditioning=2500.0)])
    p = tmp_path / "cal.json"
    p.write_text(json.dumps(cal))
    return str(p)


def test_label_formula_recovers_phase_weights(tmp_path):
    rng = np.random.default_rng(0)
    truth = np.array([50.0, 0.2, 0.6, 0.15, 0.0])
    with open(tmp_path / "players.jsonl", "w") as f:
        for i in range(400):
            spec, kind = synth.sample_player(rng)
            y = float(synth.features([spec])[0] @ truth + rng.normal(0, 30))
            f.write(json.dumps(dict(id=i, spec=list(spec), kind=kind, ladder=y, ladder_se=30.0)) + "\n")
    import argparse
    synth.fit(argparse.Namespace(ladder=str(tmp_path), out=str(tmp_path / "labels.json")))
    lab = json.load(open(tmp_path / "labels.json"))
    assert np.allclose(lab["linear_coef"][1:], truth[1:4], atol=0.03) and lab["model_error"] < 15
    assert abs(float(synth.label(lab, [(1000, 1500, 2000)])[0]) - (50 + 200 + 900 + 300)) < 30


def test_generated_games_are_a_valid_dataset(tmp_path):
    cal = tiny_calibration(tmp_path)
    labels = dict(coef=[68.0, 0.19, 0.61, 0.16, 0.0])
    (tmp_path / "labels.json").write_text(json.dumps(labels))
    out = tmp_path / "ds"
    for part in ("0/2", "1/2"):  # two processes' worth of shards in one dataset
        synth.main(["generate", "--calibration", cal, "--labels", str(tmp_path / "labels.json"), "--games", "12",
                    "--val-games", "4", "--batch", "6", "--max-plies", "30", "--shard-positions", "100",
                    "--out", str(out), "--device", "cpu", "--part", part])
    s = GameStore(str(out), "train", "cpu")
    assert s.n_games == 12 and s.n_positions == int(s.game_len.sum()) and GameStore(str(out), "val").n_games == 4
    assert len(set(s.games_np["id"])) == 12 and (s.game_len <= 30).all()
    b = s.batch(torch.arange(s.n_positions))
    w = torch.tensor([1., 2, 4, 8])
    for k in range(s.n_positions):   # every stored move is legal in its position
        bd = board_from_codes(b["board"][k].numpy(), int(b["castling"][k].float() @ w), int(b["ep"][k]))
        assert legal_mask(bd)[b["move"][k]]
    specs = np.concatenate([np.load(p) for p in sorted(out.glob("train/*/specs.npy"))])
    assert specs.shape == (12, 6)
    # labels follow the formula of each side's three strengths
    lab_w = synth.label(labels, specs[:, :3])
    assert np.abs(np.sort(lab_w.astype(int)) - np.sort(s.white_elo.numpy().astype(int))).max() <= 1


def test_ladder_runs_and_rates_players(tmp_path):
    cal = tiny_calibration(tmp_path)
    synth.main(["ladder", "--calibration", cal, "--players", "6", "--rounds", "2", "--batch", "6", "--max-plies", "20",
                "--out", str(tmp_path / "lad"), "--device", "cpu"])
    P = [json.loads(line) for line in open(tmp_path / "lad" / "players.jsonl")]
    assert len(P) == 6 and all(len(p["path"]) == 2 and len(p["scores"]) == 2 for p in P)
    assert (tmp_path / "lad" / "labels.json").exists()
    # a stopped ladder continues from the rounds on disk instead of starting again
    synth.main(["ladder", "--calibration", cal, "--players", "6", "--rounds", "3", "--batch", "6", "--max-plies", "20",
                "--out", str(tmp_path / "lad"), "--device", "cpu"])
    P2 = [json.loads(line) for line in open(tmp_path / "lad" / "players.jsonl")]
    assert all(p2["scores"][:2] == p["scores"] and len(p2["scores"]) == 3 for p, p2 in zip(P, P2))
    # ... and the remaining rounds can be played by several processes
    synth.main(["ladder", "--calibration", cal, "--players", "6", "--rounds", "5", "--batch", "6", "--max-plies", "20",
                "--out", str(tmp_path / "lad"), "--device", "cpu", "--workers", "2"])
    P3 = [json.loads(line) for line in open(tmp_path / "lad" / "players.jsonl")]
    assert all(p3["scores"][:3] == p2["scores"] and len(p3["scores"]) == 5 for p2, p3 in zip(P2, P3))
