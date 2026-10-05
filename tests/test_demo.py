"""Demo session logic with stand-in models: game flow, live curve, per-game measurements, combination."""
import chess
import numpy as np
import pytest

from fastelo.demo.board_image import SIZE, render, square_at
from fastelo.demo.session import RatingSession, combine


class FirstMovePolicy:
    def choose(self, board, ply, self_elo, opp_elo, rng):
        return sorted(board.legal_moves, key=lambda m: m.uci())[0]


class FakeRater:
    """The user is rated 1000 + 20 * (own moves - 1); the error shrinks with the moves seen."""

    def rating(self, boards_before, moves, side, state=None):
        n = len(range(side, len(moves), 2))
        mu = 1000.0 + 20 * np.arange(n)
        sd = 400 / np.sqrt(np.arange(n) + 1)
        return dict(value=mu, lo=mu - sd, hi=mu + sd, lo2=mu - 2 * sd, hi2=mu + 2 * sd, syst=np.zeros(n), resolution=sd)


def test_combine_is_inverse_variance():
    mu, sd = combine([(1000, 100), (1400, 100)])
    assert mu == pytest.approx(1200) and sd == pytest.approx(100 / np.sqrt(2))
    mu, sd = combine([(1000, 100), (2000, 1e6)])
    assert mu == pytest.approx(1000, abs=1e-3)
    assert combine([]) is None


def test_game_flow_and_measurements():
    s = RatingSession(FirstMovePolicy(), FakeRater(), seed=0)
    g = s.new_game(user_white=True)
    assert g.bot_elo == 1500 and g.user_turn
    assert not s.user_move("e2e5")                       # illegal
    for _ in range(6):
        assert s.user_move(next(iter(sorted(g.board.legal_moves, key=lambda m: m.uci()))))
    n, mu, sd = s.live_curve()
    assert list(n) == list(range(1, 7)) and mu[0] == 1000 and mu[1] == 1020  # one entry per move of the user
    s.resign()
    assert g.result == "0-1" and len(s.results) == 1
    assert s.results[0]["mean"] == mu[-1] and s.results[0]["user_moves"] == 6
    # next game: the bot adapts to the combined estimate, and the user (black) waits for its move
    g2 = s.new_game(user_white=False)
    assert g2.bot_elo == pytest.approx(s.results[0]["mean"]) and len(g2.moves) == 1 and g2.user_turn
    s.resign()                                            # no move by the user yet: nothing to measure
    assert len(s.results) == 1


def test_board_click_mapping_both_orientations():
    img = render(chess.Board(), flip=True)
    assert img.size == (SIZE, SIZE)
    for flip in (False, True):
        for sq in (chess.A1, chess.E4, chess.H8):
            f, r = chess.square_file(sq), chess.square_rank(sq)
            if flip:
                f, r = 7 - f, 7 - r
            x, y = 20 + f * 64 + 32, 20 + (7 - r) * 64 + 32
            assert square_at(x, y, flip) == sq
            assert square_at(x / 2, y / 2, flip, size=SIZE / 2) == sq   # image shown at half size


def test_bot_retargets_every_move_to_the_live_estimate():
    s = RatingSession(FirstMovePolicy(), FakeRater(), seed=0)
    g = s.new_game(user_white=True)
    for _ in range(4):
        s.user_move(sorted(g.board.legal_moves, key=lambda m: m.uci())[0])
    # with no finished games the target is the live estimate of the user: 1000 + 20 per move
    assert g.bot_elos == pytest.approx([1000.0, 1020.0, 1040.0, 1060.0])
    fixed = s.new_game(user_white=True, bot_elo=1800)
    s.user_move(sorted(fixed.board.legal_moves, key=lambda m: m.uci())[0])
    assert fixed.bot_elos == [1800.0]


def test_clock_runs_on_the_users_turn_and_flags():
    t = [0.0]
    s = RatingSession(FirstMovePolicy(), FakeRater(), seed=0, now=lambda: t[0])
    g = s.new_game(user_white=True, base=60, increment=5)
    first = lambda: sorted(g.board.legal_moves, key=lambda m: m.uci())[0]
    t[0] = 10
    assert s.time_left() == 50
    assert s.user_move(first())                    # 10 s used, +5 increment; the bot answers at once
    assert s.time_left() == 55 and not s.check_flag()
    t[0] = 64.9
    assert s.time_left() == pytest.approx(0.1)
    t[0] = 70
    assert not s.user_move(first())                # too late: lost on time, the move is not played
    assert g.flagged and g.result == "0-1" and s.time_left() == 0 and len(g.moves) == 2
    assert s.results[-1]["flagged"] and s.results[-1]["user_moves"] == 1
    g = s.new_game(user_white=False, base=30)
    t[0] = 200
    assert s.check_flag() and g.result == "1-0"    # the timer tick alone ends the game
