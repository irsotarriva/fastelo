"""End to end on synthetic games: extract -> build -> GameStore -> models."""
import random

import chess
import numpy as np
import pytest
import torch

from fastelo import build_dataset
from fastelo.board_codec import board_from_codes
from fastelo.extract import ShardWriter, _worker
from fastelo.features import GameStore
from fastelo.models import PlayerNet, PlayerConfig
from fastelo.pgn_stream import FilterConfig, check_headers
from fastelo.vocab import legal_mask


def random_game(rng, n_plies):
    board = chess.Board()
    parts, clock = [], [600, 600]
    for ply in range(n_plies):
        moves = list(board.legal_moves)
        if not moves:
            break
        mv = rng.choice(moves)
        san = board.san(mv)
        clock[ply % 2] = max(0, clock[ply % 2] - rng.randint(0, 40))
        c = clock[ply % 2]
        prefix = f"{ply // 2 + 1}. " if ply % 2 == 0 else f"{ply // 2 + 1}... "
        parts.append(f"{prefix}{san} {{ [%clk 0:{c // 60:02d}:{c % 60:02d}] }}")
        board.push(mv)
    return " ".join(parts) + " 1-0"


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    rng = random.Random(0)
    root = tmp_path_factory.mktemp("data")
    chunk = []
    for i in range(300):
        h = {"Site": f"https://lichess.org/g{i:07d}", "WhiteElo": str(rng.randint(800, 2400)),
             "BlackElo": str(rng.randint(800, 2400)), "Result": "1-0", "TimeControl": "600+0",
             "Termination": "Normal", "White": f"p{rng.randint(0, 40)}", "Black": f"p{rng.randint(0, 40)}",
             "WhiteRatingDiff": "+5", "BlackRatingDiff": "-5"}
        if i % 5 == 0:  # every 5th game: white is a bot with a wildly moving rating (must be ignored)
            h.update(WhiteTitle="BOT", WhiteRatingDiff="+90")
        info, reason = check_headers(h, FilterConfig(keep_tc=("rapid",), bots="human_vs_bot"))
        assert reason is None
        chunk.append((info, random_game(rng, rng.randint(12, 90))))
    g, p, reasons = _worker((chunk, 10))
    w = ShardWriter(str(root / "stage1"), 5000)
    w.add(g, p)
    w.flush()
    out = root / "ds"
    build_dataset.main([str(root / "stage1"), "--out", str(out), "--tc", "rapid", "--shard-positions", "4000"])
    return out


def test_store_batch_matches_python_chess(dataset):
    store = GameStore(str(dataset), "train")
    gen = torch.Generator().manual_seed(0)
    idx = store.sample_player_indices(500, gen)
    b = store.batch(idx)
    g = store.games_np
    for k in range(len(idx)):
        board = board_from_codes(b["board"][k].numpy(), int(b["castling"][k] @ torch.tensor([1., 2, 4, 8])),
                                 int(b["ep"][k]))
        assert legal_mask(board)[b["move"][k]]
        gi = int(b["game"][k])
        black = int(b["ply"][k]) % 2 == 1
        assert b["self_elo"][k] == (g["black_elo"][gi] if black else g["white_elo"][gi])
        assert b["result"][k] == (0 if black else 2)  # white won every synthetic game


def test_bot_sides_are_never_labels(dataset):
    store = GameStore(str(dataset), "train")
    g = store.games_np
    assert (g["bots"] == 1).any(), "bot games must survive the rating-diff filter (bot diff is ignored)"
    # No player target is a bot move.
    idx = store.player_indices()
    b = store.batch(idx)
    white_to_move = b["ply"] % 2 == 0
    assert not ((store.bots[b["game"]] & 1).bool() & white_to_move).any()


def test_filter_modes():
    base = {"Site": "https://lichess.org/abcdEFGH", "WhiteElo": "1500", "BlackElo": "1600", "Result": "1-0",
            "TimeControl": "600+0", "Termination": "Normal"}
    bot = {**base, "BlackTitle": "BOT"}
    cfg = lambda **k: FilterConfig(keep_tc=("rapid",), **k)
    assert check_headers(bot, cfg())[1] == "bot"
    assert check_headers(bot, cfg(bots="human_vs_bot"))[1] is None
    assert check_headers(base, cfg(bots="only_human_vs_bot"))[1] == "human_vs_human"
    assert check_headers({**bot, "WhiteTitle": "BOT"}, cfg(bots="human_vs_bot"))[1] == "bot_vs_bot"
    assert check_headers(base, cfg(tail_elo=(800, 2300)))[1] == "not_tail"
    assert check_headers({**base, "BlackElo": "2400"}, cfg(tail_elo=(800, 2300)))[1] is None
    assert check_headers({**base, "WhiteElo": "700"}, cfg(tail_elo=(800, 2300)))[1] is None
    assert check_headers(bot, cfg(bots="human_vs_bot", tail_elo=(800, 2300)))[1] is None  # tails: humans only


def test_tail_enriched_sources(tmp_path):
    """Human games from a tail-only extraction are flagged and kept out of the natural mix."""
    import json
    rng = random.Random(3)

    def stage1(root, n, tail, seed_offset):
        chunk = []
        for i in range(n):
            h = {"Site": f"https://lichess.org/t{seed_offset + i:07d}", "WhiteElo": "2500", "BlackElo": "1500",
                 "Result": "1-0", "TimeControl": "600+0", "Termination": "Normal",
                 "WhiteRatingDiff": "+5", "BlackRatingDiff": "-5"}
            info, _ = check_headers(h, FilterConfig(keep_tc=("rapid",)))
            chunk.append((info, random_game(rng, 20)))
        g, p, _ = _worker((chunk, 10))
        w = ShardWriter(str(root), 10**6)
        w.add(g, p)
        w.flush()
        with open(root / "stats.json", "w") as f:
            json.dump({"filter": {"tail_elo": [800, 2300] if tail else None}}, f)

    stage1(tmp_path / "base", 60, False, 0)
    stage1(tmp_path / "tail", 60, True, 1000)
    out = tmp_path / "ds"
    build_dataset.main([str(tmp_path / "base"), str(tmp_path / "tail"), "--out", str(out), "--tc", "rapid"])
    store = GameStore(str(out), "train")
    ids = store.games_np["id"].astype(str)
    assert (store.games_np["tail_enriched"] == np.char.startswith(ids, "t00010")).all()
    assert store.tail_enriched.any() and not store.tail_enriched[store.natural_games].any()
    assert len(store.natural_games) == int((~store.tail_enriched).sum())


def test_build_accepts_old_stage1_schema(tmp_path):
    """Stage-1 shards written before the `bots` field existed must still build."""
    from fastelo.extract import GAME_DTYPE, conform_games
    old = np.dtype([(n, GAME_DTYPE.fields[n][0]) for n in GAME_DTYPE.names if n != "bots"])
    g = np.zeros(3, old)
    g["n_plies"] = 5
    new = conform_games(g)
    assert new.dtype == GAME_DTYPE and (new["bots"] == 0).all() and (new["n_plies"] == 5).all()


def test_elo_balanced_sampling(dataset):
    store = GameStore(str(dataset), "train")
    w = store.elo_balance_weights(1.0)
    assert float(w.max()) == 1.0
    gen = torch.Generator().manual_seed(0)
    idx = store.sample_player_indices(2000, gen, w)
    assert len(idx) == 2000
    # alpha=1 flattens the Elo distribution, so it must be flatter than natural sampling.
    nat = store.mover_elo(store.sample_player_indices(2000, gen))
    bal = store.mover_elo(idx)
    spread = lambda e: torch.bincount((e // 400).long(), minlength=8).float().std()
    assert spread(bal) <= spread(nat) * 1.2


def test_optimizer_groups():
    from fastelo.train import get_args, make_optimizer
    model = PlayerNet(PlayerConfig(d_model=32, n_layers=1, n_heads=4, elo_smooth=1.0))
    opt = make_optimizer(model, get_args(["--data", "x", "--run", "x"]))
    decay_ids = {id(p) for p in opt.param_groups[0]["params"]}
    assert id(model.elo.knots.weight) not in decay_ids and id(model.encoder.piece.weight) not in decay_ids
    assert id(model.policy.weight) in decay_ids
    assert any(n.endswith("in_proj_weight") and id(p) in decay_ids for n, p in model.named_parameters())
    loss, m = model.loss({k: v for k, v in _dummy_batch().items()})
    assert "elo_smooth" in m


def _dummy_batch(B=4):
    return dict(board=torch.randint(0, 13, (B, 64)), castling=torch.zeros(B, 4), ep=torch.full((B,), 8),
                halfmove=torch.zeros(B, dtype=torch.long), ply=torch.zeros(B, dtype=torch.long),
                self_elo=torch.full((B,), 1500.0), opp_elo=torch.full((B,), 1500.0),
                tc=torch.full((B,), 3), move=torch.zeros(B, dtype=torch.long),
                result=torch.ones(B, dtype=torch.long))


def test_models_overfit_tiny_batch(dataset):
    torch.manual_seed(0)
    store = GameStore(str(dataset), "train")
    b = store.batch(store.sample_player_indices(32, torch.Generator().manual_seed(1)))
    for kind in ("interp", "fourier"):
        player = PlayerNet(PlayerConfig(d_model=64, n_layers=2, n_heads=4, elo_encoding=kind))
        opt = torch.optim.Adam(player.parameters(), 3e-3)
        first = None
        for _ in range(60):
            loss, m = player.loss(b)
            first = first or loss.item()
            opt.zero_grad(); loss.backward(); opt.step()
        assert loss.item() < 0.3 * first, (kind, first, loss.item())
