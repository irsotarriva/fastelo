"""The player and the rating in plain NumPy, for places where PyTorch does not fit (a
serverless function). Reads the exported files (fastelo.export): <name>.safetensors + <name>.json,
calibration.json and rating_response.json.

    engine = Engine("data/hf/model")
    move = engine.bot_move(board, ply, target_elo, rng)
    r = engine.rating(boards_before, moves, side, state)     # as play.MoveRater.rating

Depends on numpy and python-chess only. tests/test_lite.py checks it against the PyTorch models.
"""
import json
import os
import struct

import chess
import numpy as np

from .likelihood import GRID as RATING_GRID, curve as likelihood_curve
from .response import REPORT_RANGE, apply as apply_response, resolution, systematic
from .vocab import VOCAB_SIZE, decode_move, encode_move, legal_mask

RAPID = 3                           # pgn_stream.TC_CATEGORIES.index("rapid")
_DTYPES = {"F16": np.float16, "F32": np.float32, "F64": np.float64, "I64": np.int64}


def load_safetensors(path):
    """name -> float32 array (the format is an 8-byte header length, a JSON header, raw data)."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        data = f.read()
    out = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        a, b = info["data_offsets"]
        out[name] = np.frombuffer(data[a:b], _DTYPES[info["dtype"]]).reshape(info["shape"]).astype(np.float32)
    return out


# ------------------------------------------------------------------------ transformer
def _erf(x):  # Abramowitz & Stegun 7.1.26, |error| < 1.5e-7
    s, x = np.sign(x), np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * x)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * np.exp(-x * x)
    return s * y


def gelu(x):
    return 0.5 * x * (1.0 + _erf(x * 0.7071067811865476))


def layer_norm(x, w, b):
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(v + 1e-5) * w + b


def softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


class Layer:
    """One pre-LN nn.TransformerEncoderLayer (GELU)."""

    def __init__(self, w, prefix, n_heads):
        g = lambda k: w[prefix + k]
        d = g("norm1.weight").shape[0]
        W, b = g("self_attn.in_proj_weight"), g("self_attn.in_proj_bias")
        self.wq, self.wk, self.wv = W[:d].T.copy(), W[d:2 * d].T.copy(), W[2 * d:].T.copy()
        self.bq, self.bk, self.bv = b[:d], b[d:2 * d], b[2 * d:]
        self.wo, self.bo = g("self_attn.out_proj.weight").T.copy(), g("self_attn.out_proj.bias")
        self.w1, self.b1 = g("linear1.weight").T.copy(), g("linear1.bias")
        self.w2, self.b2 = g("linear2.weight").T.copy(), g("linear2.bias")
        self.n1, self.n2 = (g("norm1.weight"), g("norm1.bias")), (g("norm2.weight"), g("norm2.bias"))
        self.h, self.scale = n_heads, (d // n_heads) ** -0.5

    def _heads(self, x):  # [..., L, d] -> [..., h, L, d/h]
        return x.reshape(*x.shape[:-1], self.h, -1).swapaxes(-2, -3)

    def attend(self, q, k, v, mask=None):
        """q [..., Lq, d] (already projected), k/v [..., Lk, d] -> [..., Lq, d]."""
        a = (self._heads(q) * self.scale) @ self._heads(k).swapaxes(-1, -2)
        if mask is not None:
            a = a + mask
        o = softmax(a) @ self._heads(v)
        return o.swapaxes(-2, -3).reshape(q.shape) @ self.wo + self.bo

    def ff(self, x):
        return gelu(layer_norm(x, *self.n2) @ self.w1 + self.b1) @ self.w2 + self.b2

    def __call__(self, x):
        m = layer_norm(x, *self.n1)
        x = x + self.attend(m @ self.wq + self.bq, m @ self.wk + self.bk, m @ self.wv + self.bv)
        return x + self.ff(x)

    def step(self, x, cache):
        """Causal incremental pass: x [n, d] new tokens, cache (k, v) of the earlier ones or None."""
        m = layer_norm(x, *self.n1)
        k, v = m @ self.wk + self.bk, m @ self.wv + self.bv
        n, t = len(x), 0 if cache is None else len(cache[0])
        if cache is not None:
            k, v = np.concatenate([cache[0], k]), np.concatenate([cache[1], v])
        mask = np.zeros((n, t + n), np.float32)
        mask[:, t:] = np.triu(np.full((n, n), -np.inf, np.float32), 1)
        x = x + self.attend(m @ self.wq + self.bq, k, v, mask)
        return x + self.ff(x), (k, v)


def board_features(boards):
    """python-chess boards -> piece codes [B, 64] (mover's perspective: 0 empty, 1-6 own, 7-12
    opponent), castling [B, 4], en-passant file [B] (8 = none), halfmove clock [B]. Same encoding
    as board_codec.encode_board + features.decode_records."""
    codes = np.zeros((len(boards), 64), np.int64)
    castling = np.zeros((len(boards), 4), np.float32)
    ep = np.zeros(len(boards), np.int64)
    half = np.zeros(len(boards), np.int64)
    for i, board in enumerate(boards):
        turn = board.turn
        for pt in range(1, 7):
            for color in (chess.WHITE, chess.BLACK):
                m = board.pieces_mask(pt, color)
                if turn == chess.BLACK:
                    m = chess.flip_vertical(m)
                code = pt if color == turn else pt + 6
                while m:
                    lsb = m & -m
                    codes[i, lsb.bit_length() - 1] = code
                    m ^= lsb
        castling[i] = (board.has_kingside_castling_rights(turn), board.has_queenside_castling_rights(turn),
                       board.has_kingside_castling_rights(not turn), board.has_queenside_castling_rights(not turn))
        ep[i] = chess.square_file(board.ep_square) if board.has_legal_en_passant() else 8
        half[i] = min(board.halfmove_clock, 100)
    return codes, castling, ep, half


class BoardEncoder:
    def __init__(self, w, prefix, n_layers, n_heads):
        g = lambda k: w[prefix + k]
        self.piece, self.square = g("piece.weight"), g("square")
        self.cw, self.cb = g("castling.weight").T.copy(), g("castling.bias")
        self.ep, self.half, self.ply = g("ep.weight"), g("halfmove.weight"), g("ply.weight")
        self.layers = [Layer(w, f"{prefix}transformer.layers.{i}.", n_heads) for i in range(n_layers)]
        self.norm = (g("norm.weight"), g("norm.bias"))

    def __call__(self, boards, plies, prefix):
        """prefix [B, P, d] -> the encoded prefix token 0, [B, d]. One board with several prefixes
        (the same position under several rating inputs) is encoded as a batch."""
        codes, castling, ep, half = board_features(boards)
        if len(boards) == 1 and len(prefix) > 1:
            codes, castling, ep, half = (np.repeat(x, len(prefix), 0) for x in (codes, castling, ep, half))
            plies = np.repeat(plies, len(prefix))
        state = castling @ self.cw + self.cb + self.ep[ep] + self.half[half] + self.ply[np.minimum(plies, 255)]
        x = np.concatenate([prefix + state[:, None], self.piece[codes] + self.square], 1)
        for layer in self.layers:
            x = layer(x)
        return layer_norm(x[:, 0], *self.norm)


def _load(root, name):
    cfg = json.load(open(os.path.join(root, name + ".json")))["model_cfg"]
    return load_safetensors(os.path.join(root, name + ".safetensors")), cfg


class Policy:
    """models.PlayerNet: logits over the move vocabulary for one position."""

    def __init__(self, root, name):
        w, cfg = _load(root, name)
        self.encoder = BoardEncoder(w, "encoder.", cfg["n_layers"], cfg["n_heads"])
        self.knots, self.lo, self.hi = w["elo.knots.weight"], cfg["elo_lo"], cfg["elo_hi"]
        self.step = (self.hi - self.lo) / (len(self.knots) - 1)
        self.role, self.tc = w["role"], w["tc.weight"]
        self.pw, self.pb = w["policy.weight"].T.copy(), w["policy.bias"]

    def elo(self, e):
        x = (min(max(float(e), self.lo), self.hi) - self.lo) / self.step
        i = min(int(x), len(self.knots) - 2)
        return (1 - (x - i)) * self.knots[i] + (x - i) * self.knots[i + 1]

    def logits(self, board, ply, self_elo, opp_elo, tc=RAPID):
        prefix = np.stack([self.role[0] + self.tc[tc], self.role[1] + self.elo(self_elo),
                           self.role[2] + self.elo(opp_elo)])[None]
        return (self.encoder([board], np.array([ply]), prefix) @ self.pw + self.pb)[0]

    def move_logp(self, board, ply, move, elos, tc=RAPID):
        """log p(move | position, rating) for each rating input in `elos` (own = opponent input),
        over the legal moves: [len(elos)]."""
        prefix = np.stack([np.stack([self.role[0] + self.tc[tc], self.role[1] + self.elo(e), self.role[2] + self.elo(e)])
                           for e in elos])
        z = (self.encoder([board], np.array([ply]), prefix) @ self.pw + self.pb).astype(np.float64)
        mask = legal_mask(board)
        z = np.where(mask, z, -np.inf)
        z = z - z.max(1, keepdims=True)
        return z[:, encode_move(move, board.turn)] - np.log(np.exp(z).sum(1))


def _log_softmax(z):
    z = z.astype(np.float64)
    z = z - z.max()
    return z - np.log(np.exp(z).sum())


def settings(cal, elo):
    """calibrate.settings: target Elo -> (Elo input, temperature, share of the strong player)."""
    k = [x for x in cal["knots"] if x["conditioning"] is not None]
    c = cal["temperature_curve"]
    d = elo - c["knot"]
    T = float(np.clip(c["value"] + (c["slope_lo"] if d < 0 else c["slope_hi"]) * d, c["t_min"], c["t_max"]))
    mr = cal["mix_ramp"]
    if not mr["elo"] or elo <= mr["elo"][0]:
        m = 0.0
    elif elo > mr["elo"][-1] and len(mr["elo"]) >= 2:
        s = (mr["mix"][-1] - mr["mix"][-2]) / (mr["elo"][-1] - mr["elo"][-2])
        m = float(min(1.0, mr["mix"][-1] + s * (elo - mr["elo"][-1])))
    else:
        m = float(np.interp(elo, mr["elo"], mr["mix"]))
    return float(elo + np.interp(elo, [x["elo"] for x in k], [x["conditioning"] - x["elo"] for x in k])), T, m


class Engine:
    """The calibrated bot and the bias-corrected rating from one exported model folder."""

    def __init__(self, root):
        self.cal = json.load(open(os.path.join(root, "calibration.json")))
        self.player = Policy(root, os.path.splitext(self.cal["player"])[0])
        self.strong = Policy(root, os.path.splitext(self.cal["strong_player"])[0])
        path = os.path.join(root, "rating_response.json")
        self.response = json.load(open(path)) if os.path.exists(path) else None

    def distribution(self, board, ply, target, tc=RAPID):
        """Move probabilities [VOCAB_SIZE] of the bot playing at a target Elo (play.CalibratedPolicy)."""
        c, T, mix = settings(self.cal, float(target))
        mask = legal_mask(board)
        z = _log_softmax(self.player.logits(board, ply, c, c, tc)[mask]) / T
        if mix > 0:
            se = self.cal["strong_elo"]
            z = (1 - mix) * z + mix * _log_softmax(self.strong.logits(board, ply, se, se, tc)[mask]) / self.cal["strong_temperature"]
        p = np.zeros(VOCAB_SIZE)
        p[mask] = np.exp(z - z.max())
        return p / p.sum()

    def bot_move(self, board, ply, target, rng, tc=RAPID):
        return decode_move(int(rng.choice(VOCAB_SIZE, p=self.distribution(board, ply, target, tc))), board.turn)

    def rating(self, boards_before, moves, side, state, tc=RAPID):
        """One player's rating after each of their own moves (side 0 = white, 1 = black), from the
        likelihood of those moves under the player: dict of arrays [n own moves] with value, lo/hi
        (1 sigma statistical), lo2/hi2 (2 sigma), syst, resolution. state: a dict owned by the
        caller, one per game (start with {}); only moves not yet in it are scored."""
        own = list(range(side, len(moves), 2))
        key = [moves[p].uci() for p in own]
        done = len(state.get("moves", []))
        if done > len(key) or state.get("moves", []) != key[:done] or state.get("side", side) != side:
            state.clear()
            done = 0
        for p in own[done:]:
            state.setdefault("logp", []).append(self.player.move_logp(boards_before[p], p, moves[p], RATING_GRID, tc))
        state["moves"], state["side"] = key, side
        mu, sd = likelihood_curve(np.array(state.get("logp", [])).reshape(-1, len(RATING_GRID)))
        if self.response is None or not len(mu):
            return dict(value=mu, lo=mu - sd, hi=mu + sd, lo2=mu - 2 * sd, hi2=mu + 2 * sd, syst=np.zeros_like(mu),
                        resolution=sd)
        seen = 2 * np.arange(1, len(mu) + 1)   # the correction is indexed by plies of the game
        out = apply_response(self.response, mu, sd, seen)
        out["syst"] = np.array([systematic(self.response, v, k)[0] for v, k in zip(out["value"], seen)])
        out["resolution"] = resolution(self.response, seen)
        return out


# ----------------------------------------------------------------- stateless demo turn
BOT_RANGE = (600.0, 2400.0)   # strengths the calibrated bot can play
START_ELO = 1500.0
MEASUREMENT_KEYS = ("mean", "minus", "plus", "syst", "resolution")


def combine_measurements(ms):
    """Games of one player -> dict(value, minus, plus, syst, games), as demo.session.combined_full:
    weights from the resolution of a game of that length; the systematic error does not average down."""
    if not ms:
        return None
    w = np.array([1.0 / max(float(m["resolution"]), 1.0) ** 2 for m in ms])
    w = w / w.sum()
    col = lambda k: np.array([float(m[k]) for m in ms])
    return dict(value=float((w * col("mean")).sum()), minus=float(np.sqrt((w ** 2 * col("minus") ** 2).sum())),
                plus=float(np.sqrt((w ** 2 * col("plus") ** 2).sum())), syst=float((w * col("syst")).sum()), games=len(ms))


def turn(engine, req, states, rng):
    """One request of the web demo. The browser holds the session (moves of the game in progress,
    measurements of the finished games); the server validates, rates the user and answers with the
    bot's move. `states` maps game id -> scores of the moves already rated; it is only a speed-up, a
    miss recomputes.

    req: moves [uci, ...] (all moves so far), user_white, game_id, previous [measurement, ...],
         bot (false = do not move, e.g. the user resigned or lost on time).
    Returns moves (with the bot's reply), bot_move, bot_elo, result, curve (the user's rating after
    each of their moves), measurement (this game), combined (all games)."""
    ucis = list(req.get("moves", []))
    previous = [{k: float(m[k]) for k in MEASUREMENT_KEYS} for m in req.get("previous", [])][-50:]
    user_white = bool(req.get("user_white", True))
    if len(ucis) > 600:
        raise ValueError("game too long")
    board, boards, moves = chess.Board(), [], []
    for u in ucis:
        m = chess.Move.from_uci(str(u))
        if m not in board.legal_moves:
            raise ValueError(f"illegal move {u}")
        boards.append(board.copy(stack=False))
        moves.append(m)
        board.push(m)
    gid = str(req.get("game_id", ""))[:64]
    state = states.setdefault(gid, {}) if gid else {}
    side = 0 if user_white else 1

    def rate():
        return engine.rating(boards, moves, side, state)

    def measurement(r):
        if not len(r["value"]):
            return None
        v = float(r["value"][-1])
        return dict(mean=v, minus=float(v - r["lo"][-1]), plus=float(r["hi"][-1] - v), syst=float(r["syst"][-1]),
                    resolution=float(r["resolution"][-1]), user_moves=int(len(r["value"])))

    r = rate()
    live = measurement(r)
    combined = combine_measurements(previous + ([live] if live else []))
    over = board.is_game_over(claim_draw=True)
    bot_move = bot_elo = None
    if not over and req.get("bot", True) and board.turn != (chess.WHITE if user_white else chess.BLACK):
        bot_elo = float(np.clip(combined["value"] if combined else START_ELO, *BOT_RANGE))
        m = engine.bot_move(board, len(moves), bot_elo, rng)
        boards.append(board.copy(stack=False))
        moves.append(m)
        board.push(m)
        bot_move = m.uci()
        over = board.is_game_over(claim_draw=True)
    rnd = lambda a: [round(float(x), 1) for x in a]
    return dict(moves=[m.uci() for m in moves], bot_move=bot_move, bot_elo=bot_elo,
                result=board.result(claim_draw=True) if over else None,
                curve={k: rnd(r[k]) for k in ("value", "lo", "hi", "lo2", "hi2", "syst")},
                measurement=live, combined=combined, report_range=list(REPORT_RANGE))
