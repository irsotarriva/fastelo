"""Game/rating logic of the demo, independent of any UI.

A RatingSession is one visitor: they play games against the bot, the rater reads every game ply
by ply, and each game - finished or still in progress - is one measurement (mean, sigma) of the
visitor's Elo, taken after their latest move. The measurements are combined as independent
experiments would be; short games carry a large error and a small weight. With a bias-corrected
rating (fastelo.response) each measurement has asymmetric statistical errors and a systematic
error that combining games does not reduce.
"""
import time
from dataclasses import dataclass, field

import chess
import numpy as np


def combine(estimates):
    """Inverse-variance weighted mean of (mean, sigma) measurements -> (mean, sigma)."""
    if not estimates:
        return None
    w = np.array([1.0 / s ** 2 for _, s in estimates])
    mu = float(np.sum(w * np.array([m for m, _ in estimates])) / w.sum())
    return mu, float(w.sum() ** -0.5)


@dataclass
class Game:
    user_white: bool
    bot_elo: float
    board: chess.Board = field(default_factory=chess.Board)
    bot_elos: list = field(default_factory=list)   # strength the bot used for each of its moves
    boards_before: list = field(default_factory=list)
    moves: list = field(default_factory=list)
    result: str = None            # "1-0" / "0-1" / "1/2-1/2" once over
    resigned: bool = False
    flagged: bool = False         # the user ran out of time
    base: float = 600.0           # user's clock: seconds at the start, increment per move
    increment: float = 0.0
    clock: float = 600.0          # seconds left when the user's current turn started
    turn_started: float = None    # session time at which the user's turn started (None = not their turn)
    rating_state: dict = field(default_factory=dict)   # scores of the moves already rated

    @property
    def user_turn(self):
        return self.result is None and self.board.turn == (chess.WHITE if self.user_white else chess.BLACK)

    def push(self, move):
        self.boards_before.append(self.board.copy(stack=False))
        self.moves.append(move)
        self.board.push(move)
        if self.board.is_game_over(claim_draw=True):
            self.result = self.board.result(claim_draw=True)

    def user_plies(self):
        """Ply indices (0-based) of the user's own moves."""
        first = 0 if self.user_white else 1
        return list(range(first, len(self.moves), 2))


class RatingSession:
    MIN_USER_MOVES = 1    # a game counts once the user has moved; its sigma says how much

    def __init__(self, policy, rater, start_elo=1500.0, bot_range=(600.0, 2400.0), seed=None, now=time.monotonic):
        self.policy, self.rater, self.now = policy, rater, now
        self.start_elo, self.bot_range = start_elo, bot_range
        self.rng = np.random.default_rng(seed)
        self.game = None
        self.fixed_bot_elo = None
        self.results = []     # finished games: dict(mean, sigma, result, user_white, bot_elo, moves)

    # ----------------------------------------------------------------- games
    def bot_strength(self, fixed=None):
        """The bot plays at the best current guess of the user's Elo: all finished games combined with
        the live estimate of the game in progress (start_elo before there is any information)."""
        if fixed is not None:
            return float(fixed)
        c = self.combined(include_live=True)
        return float(np.clip(c[0] if c else self.start_elo, *self.bot_range))

    def new_game(self, user_white=True, bot_elo=None, base=600.0, increment=0.0):
        """base/increment: the user's clock in seconds (the rater was trained on rapid games, so the
        demo is timed; the bot answers at once and has no clock)."""
        if self.game is not None and self.game.result is None:
            self.resign()
        self.fixed_bot_elo = bot_elo
        self.game = Game(user_white=user_white, bot_elo=self.bot_strength(bot_elo), base=base, increment=increment,
                         clock=base)
        if not user_white:
            self._bot_move()
        self.game.turn_started = self.now()
        return self.game

    def time_left(self):
        """Seconds on the user's clock right now."""
        g = self.game
        if g is None:
            return None
        if g.result is not None or g.turn_started is None:
            return max(0.0, g.clock)
        return max(0.0, g.clock - (self.now() - g.turn_started))

    def check_flag(self):
        """Ends the game as a loss if the user's time has run out. Returns True when that happens."""
        g = self.game
        if g is None or g.result is not None or not g.user_turn or self.time_left() > 0:
            return False
        g.clock, g.turn_started = 0.0, None
        g.result, g.flagged = ("0-1" if g.user_white else "1-0"), True
        self._record()
        return True

    def user_move(self, move):
        """Play the user's move (chess.Move or UCI/SAN string) and the bot's reply. Returns False if
        the move is illegal or it is not the user's turn."""
        g = self.game
        if g is None or not g.user_turn:
            return False
        if isinstance(move, str):
            try:
                move = chess.Move.from_uci(move)
            except ValueError:
                try:
                    move = g.board.parse_san(move)
                except ValueError:
                    return False
        if move not in g.board.legal_moves:
            return False
        if self.check_flag():
            return False
        g.clock, g.turn_started = self.time_left() + g.increment, None
        g.push(move)
        if g.result is None:
            self._bot_move()
        if g.result is not None:
            self._record()
        else:
            g.turn_started = self.now()
        return True

    def _bot_move(self):
        g = self.game
        g.bot_elo = self.bot_strength(self.fixed_bot_elo)   # re-targeted before every bot move
        g.bot_elos.append(g.bot_elo)
        move = self.policy.choose(g.board, len(g.moves), g.bot_elo, g.bot_elo, self.rng)
        g.push(move)

    def resign(self):
        g = self.game
        if g is None or g.result is not None:
            return
        g.clock, g.turn_started = self.time_left(), None
        g.result, g.resigned = ("0-1" if g.user_white else "1-0"), True
        self._record()

    # ---------------------------------------------------------------- rating
    def rating_curve(self, game=None):
        """The user's rating after each of their moves in the game: dict of arrays with n (move
        number), value, lo/hi (1 sigma statistical), lo2/hi2 (2 sigma), syst, resolution. With a
        bias-corrected rating the bounds are asymmetric."""
        g = game or self.game
        keys = ("value", "lo", "hi", "lo2", "hi2", "syst", "resolution")
        if g is None or not g.moves:
            return dict(n=np.zeros(0, int), **{k: np.zeros(0) for k in keys})
        side = 0 if g.user_white else 1
        r = self.rater.rating(g.boards_before, g.moves, side, state=g.rating_state)
        return dict(n=np.arange(1, len(r["value"]) + 1), **{k: np.asarray(r[k]) for k in keys})

    def live_curve(self, game=None):
        """(user move number, value, symmetrised sigma) after each of the user's moves in the game."""
        r = self.rating_curve(game)
        return r["n"], r["value"], 0.5 * (r["hi"] - r["lo"])

    @staticmethod
    def _measurement(r):
        v = float(r["value"][-1])
        return dict(mean=v, sigma=float(0.5 * (r["hi"][-1] - r["lo"][-1])), minus=float(v - r["lo"][-1]),
                    plus=float(r["hi"][-1] - v), syst=float(r["syst"][-1]), resolution=float(r["resolution"][-1]))

    def _record(self):
        g = self.game
        r = self.rating_curve(g)
        if len(r["n"]) < self.MIN_USER_MOVES:
            return
        self.results.append(dict(self._measurement(r), result=g.result, user_white=g.user_white, bot_elo=g.bot_elo,
                                 user_moves=int(len(r["n"])), resigned=g.resigned, flagged=g.flagged))

    def live_measurement(self):
        """The game in progress as a measurement (same keys as a finished game), or None."""
        g = self.game
        if g is None or g.result is not None:
            return None
        r = self.rating_curve(g)
        return self._measurement(r) if len(r["n"]) else None

    def live_estimate(self):
        """(value, symmetrised sigma) of the user after their latest move in the game in progress, or None."""
        m = self.live_measurement()
        return (m["mean"], m["sigma"]) if m else None

    def combined_full(self, include_live=True):
        """All games as one result: dict(value, minus, plus, syst). Each game is weighted by the
        resolution of a game of its length. The statistical errors average down; the systematic
        error is common to every game and does not."""
        ms = list(self.results)
        live = self.live_measurement() if include_live else None
        if live:
            ms.append(live)
        if not ms:
            return None
        w = np.array([1.0 / max(m["resolution"], 1.0) ** 2 for m in ms])
        w = w / w.sum()
        col = lambda k: np.array([m[k] for m in ms])
        return dict(value=float((w * col("mean")).sum()), minus=float(np.sqrt((w ** 2 * col("minus") ** 2).sum())),
                    plus=float(np.sqrt((w ** 2 * col("plus") ** 2).sum())), syst=float((w * col("syst")).sum()),
                    games=len(ms))

    def combined(self, include_live=True):
        """(value, symmetrised statistical sigma) of all games, or None."""
        c = self.combined_full(include_live)
        return (c["value"], 0.5 * (c["minus"] + c["plus"])) if c else None
