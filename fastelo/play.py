"""Play with the player and rate a player from their own moves.

    # two calibrated bots, both rated from their own moves as the game goes on
    python -m fastelo.play --calibration model/calibration.json --white player:1500 --black player:1100
    # the bot against Stockfish limited to 1400
    python -m fastelo.play --calibration model/calibration.json --white player:1500 --black stockfish:1400

Player (CalibratedPolicy) plays at a target Elo; Rater (MoveRater) reads a rating off the
likelihood of someone's moves under the player (see fastelo.likelihood).
"""
import argparse
import json
import os

import chess
import chess.engine
import numpy as np
import torch
import torch.nn.functional as F

from .features import decode_records, records_from_boards
from .models import load_model
from .pgn_stream import TC_CATEGORIES
from .vocab import VOCAB_SIZE, decode_move, encode_move, legal_mask

RAPID = TC_CATEGORIES.index("rapid")


class PlayerPolicy:
    def __init__(self, ckpt, device="cpu", temperature=1.0):
        self.model, _ = load_model(ckpt, device)
        self.device, self.temperature = device, temperature

    @torch.no_grad()
    def logits(self, board, ply, self_elo, opp_elo, tc=RAPID):
        b = decode_records(records_from_boards([board]))
        b.pop("black")
        b.update(self_elo=torch.tensor([float(self_elo)]), opp_elo=torch.tensor([float(opp_elo)]),
                 tc=torch.tensor([tc]), ply=torch.tensor([min(ply, 255)]))
        b = {k: v.to(self.device) for k, v in b.items()}
        return self.model(b)[0][0].float()

    @torch.no_grad()
    def distribution(self, board, ply, self_elo, opp_elo, tc=RAPID):
        mask = torch.as_tensor(legal_mask(board), device=self.device)
        logits = self.logits(board, ply, self_elo, opp_elo, tc)
        return F.softmax(logits.masked_fill(~mask, float("-inf")) / self.temperature, -1).cpu()

    def choose(self, board, ply, self_elo, opp_elo, rng, tc=RAPID):
        p = self.distribution(board, ply, self_elo, opp_elo, tc).numpy().astype(np.float64)
        return decode_move(int(rng.choice(VOCAB_SIZE, p=p / p.sum())), board.turn)


class CalibratedPolicy:
    """Plays at a *target* Elo using a calibration from `fastelo.calibrate fit`: the target is
    mapped to (player Elo input, temperature, share of the strong model), and the two models'
    log-probabilities are mixed exactly as in the calibration. Same interface as PlayerPolicy, with
    `self_elo` meaning the target; the opponent input is set to the player input, as in the fit."""

    def __init__(self, calibration, device="cpu", player=None, strong=None):
        from .calibrate import settings
        self.cal = json.load(open(calibration)) if isinstance(calibration, str) else calibration
        self.settings = lambda e: settings(self.cal, e)
        base = os.path.dirname(os.path.abspath(calibration)) if isinstance(calibration, str) else "."
        path = lambda p: p if os.path.isabs(p) else os.path.join(base, p)  # relative to the calibration file
        self.player = PlayerPolicy(player or path(self.cal["player"]), device)
        self.strong = PlayerPolicy(strong or path(self.cal["strong_player"]), device)
        self.device = device

    @torch.no_grad()
    def distribution(self, board, ply, self_elo, opp_elo=None, tc=RAPID):
        c, T, mix = self.settings(float(self_elo))
        mask = torch.as_tensor(legal_mask(board), device=self.device)
        la = torch.log_softmax(self.player.logits(board, ply, c, c, tc).masked_fill(~mask, float("-inf")), -1)
        z = la / T
        if mix > 0:
            se = self.cal["strong_elo"]
            ls = torch.log_softmax(self.strong.logits(board, ply, se, se, tc).masked_fill(~mask, float("-inf")), -1)
            z = (1 - mix) * z + mix * ls / self.cal["strong_temperature"]
        return F.softmax(z.masked_fill(~mask, float("-inf")), -1).cpu()

    def choose(self, board, ply, self_elo, opp_elo, rng, tc=RAPID):
        p = self.distribution(board, ply, self_elo, opp_elo, tc).numpy().astype(np.float64)
        return decode_move(int(rng.choice(VOCAB_SIZE, p=p / p.sum())), board.turn)


class MoveRater:
    """Rates a player from the likelihood of their own moves under the player network (see
    fastelo.likelihood). It does not depend on how strong the
    opponent is, so it can be used against a bot whose strength follows the estimate.
    `rating()` returns one entry per own move."""

    def __init__(self, policy, response="auto"):
        """policy: PlayerPolicy, CalibratedPolicy (its human player is used) or a checkpoint path.
        response: bias correction from `fastelo.response fit` on `fastelo.likelihood measure` files."""
        if isinstance(policy, str):
            policy = PlayerPolicy(policy)
        self.policy = getattr(policy, "player", policy)
        if response == "auto":
            response = None
        self.response = json.load(open(response)) if response else None

    @torch.no_grad()
    def move_logp(self, board, ply, move, tc=RAPID):
        from .likelihood import GRID
        b = decode_records(records_from_boards([board] * len(GRID)))
        b.pop("black")
        e = torch.tensor(GRID, dtype=torch.float32)
        b.update(self_elo=e, opp_elo=e, tc=torch.full((len(GRID),), tc), ply=torch.full((len(GRID),), min(ply, 255)))
        b = {k: v.to(self.policy.device) for k, v in b.items()}
        mask = torch.as_tensor(legal_mask(board), device=self.policy.device)
        lp = torch.log_softmax(self.policy.model(b)[0].float().masked_fill(~mask, float("-inf")), -1)
        return lp[:, encode_move(move, board.turn)].double().cpu().numpy()

    def rating(self, boards_before, moves, side, tc=RAPID, state=None):
        from . import likelihood, response as R
        state = {} if state is None else state
        own = list(range(side, len(moves), 2))
        key = [moves[p].uci() for p in own]
        done = len(state.get("moves", []))
        if done > len(key) or state.get("moves", []) != key[:done] or state.get("side", side) != side:
            state.clear()
            done = 0
        for p in own[done:]:
            state.setdefault("logp", []).append(self.move_logp(boards_before[p], p, moves[p], tc))
        state["moves"], state["side"] = key, side
        mu, sd = likelihood.curve(np.array(state.get("logp", [])).reshape(-1, len(likelihood.GRID)))
        if self.response is None or not len(mu):
            return dict(value=mu, lo=mu - sd, hi=mu + sd, lo2=mu - 2 * sd, hi2=mu + 2 * sd, syst=np.zeros_like(mu),
                        resolution=sd)
        seen = 2 * np.arange(1, len(mu) + 1)
        out = R.apply(self.response, mu, sd, seen)
        out["syst"] = np.array([R.systematic(self.response, v, k)[0] for v, k in zip(out["value"], seen)])
        out["resolution"] = R.resolution(self.response, seen)
        return out


# Public names
Player = CalibratedPolicy
Rater = MoveRater


class Agent:
    name = "agent"
    cond_elo = None

    def move(self, board, ply, history, rng):
        raise NotImplementedError


class PolicyAgent(Agent):
    def __init__(self, policy, elo, opp_elo=None):
        self.policy, self.elo, self.opp_elo = policy, elo, opp_elo or elo
        self.name, self.cond_elo = f"player@{elo}", elo

    def move(self, board, ply, history, rng):
        return self.policy.choose(board, ply, self.elo, self.opp_elo, rng)


class StockfishAgent(Agent):
    def __init__(self, path, elo):
        self.engine = chess.engine.SimpleEngine.popen_uci(path)
        self.engine.configure({"UCI_LimitStrength": True, "UCI_Elo": elo, "Threads": 1})
        self.name, self.cond_elo = f"stockfish@{elo}", elo

    def move(self, board, ply, history, rng):
        return self.engine.play(board, chess.engine.Limit(time=0.05)).move


def play_game(white, black, rater=None, rng=None, max_plies=300):
    """One game between two agents. With a rater, both sides are rated from their own moves and the
    trace carries the mover's rating after each move."""
    rng = rng or np.random.default_rng()
    board = chess.Board()
    history = {"boards": [], "moves": []}
    states = ({}, {})
    trace = []
    for ply in range(max_plies):
        if board.is_game_over(claim_draw=True):
            break
        agent = white if board.turn == chess.WHITE else black
        mv = agent.move(board, ply, history, rng)
        history["boards"].append(board.copy(stack=False))
        history["moves"].append(mv)
        row = dict(ply=ply, move=board.san(mv), mover=agent.name, cond_elo=agent.cond_elo)
        board.push(mv)
        if rater is not None:
            r = rater.rating(history["boards"], history["moves"], ply % 2, state=states[ply % 2])
            row.update(rating=round(float(r["value"][-1])), rating_lo=round(float(r["lo"][-1])),
                       rating_hi=round(float(r["hi"][-1])))
        trace.append(row)
    return dict(result=board.result(claim_draw=True), plies=len(trace), trace=trace)


def make_agent(spec, policy, args):
    kind, _, val = spec.partition(":")
    if kind == "player":
        return PolicyAgent(policy, int(val))
    if kind == "stockfish":
        return StockfishAgent(args.engine, int(val))
    raise ValueError(spec)


def main(argv=None):
    from .eval_strength import DEFAULT_ENGINE
    ap = argparse.ArgumentParser()
    ap.add_argument("--player", default=None)
    ap.add_argument("--calibration", default=None,
                    help="calibration.json from fastelo.calibrate: player:<elo> then means a calibrated target Elo")
    ap.add_argument("--response", default=None, help="rating_response.json: bias correction of the rating")
    ap.add_argument("--white", default="player:1500")
    ap.add_argument("--black", default="player:1500")
    ap.add_argument("--games", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--engine", default=DEFAULT_ENGINE)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    policy = (CalibratedPolicy(args.calibration, player=args.player) if args.calibration
              else PlayerPolicy(args.player, temperature=args.temperature))
    rater = MoveRater(policy, response=args.response)
    rng = np.random.default_rng(args.seed)
    games = []
    for g in range(args.games):
        white, black = make_agent(args.white, policy, args), make_agent(args.black, policy, args)
        game = play_game(white, black, rater, rng)
        games.append(game)
        last = {r["mover"]: r for r in game["trace"]}
        print(f"game {g}: {game['result']} in {game['plies']} plies; "
              + "; ".join(f"{name} rated {r['rating']} ({r['rating_lo']}-{r['rating_hi']})" for name, r in last.items()))
        for a in (white, black):
            if isinstance(a, StockfishAgent):
                a.engine.quit()
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(games, f, indent=1)


if __name__ == "__main__":
    main()
