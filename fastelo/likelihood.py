"""Rating a player from the likelihood of their own moves under the player network.

    python -m fastelo.likelihood measure --ckpt runs/player/ckpt.pt --data <dataset> --split val --out lik_val.npz
    python -m fastelo.likelihood measure --ckpt runs/player/ckpt.pt --data <dataset> --split test --out lik_test.npz
    python -m fastelo.response fit --derive lik_val.npz --closure lik_test.npz --out runs/player/rating_response.json

The player network gives p(move | position, rating). For one player's moves in a game, the
log-likelihood as a function of the rating input, L(e) = sum_i log p(move_i | position_i, e), is
a measurement of that player's rating that uses nothing but their own decisions.

Why not a network that reads the whole game and predicts the ratings: in human games the two
players are matched, so the level of the *game* tells such a network most of what it knows. Against
a bot whose strength follows the estimate that is circular: a perfect player was rated 1250 against
a 700 bot and 2250 against a 2300 bot. The likelihood of the player's own moves does not depend on
how well the opponent plays.

The estimate is the posterior mean under a flat prior on [GRID[0], GRID[-1]], with the posterior
standard deviation as sigma. `measure` writes the same file as `fastelo.response measure`, so the
bias correction and the statistical / systematic errors come from `fastelo.response fit` unchanged
("plies seen" = twice the number of the player's own moves).
"""
import argparse

import numpy as np

GRID = np.arange(400, 2601, 200.0)     # rating inputs the moves are scored under (12 forward passes per move)
FINE = np.arange(400, 2601, 25.0)      # the log-likelihood is interpolated onto this for the posterior


def posterior(cum_logp):
    """cum_logp [..., len(GRID)]: summed log-likelihood on GRID -> (mean, sd) of the flat-prior posterior."""
    cum_logp = np.asarray(cum_logp, np.float64)
    flat = cum_logp.reshape(-1, len(GRID))
    fine = np.stack([np.interp(FINE, GRID, row) for row in flat])
    p = np.exp(fine - fine.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    mean = (p * FINE).sum(1)
    sd = np.sqrt((p * (FINE - mean[:, None]) ** 2).sum(1))
    return mean.reshape(cum_logp.shape[:-1]), sd.reshape(cum_logp.shape[:-1])


CLIP = 4.0    # nats; on real games the accuracy is the same as without a cap (tested from 1 to 6)


def robust(logp, clip=CLIP):
    """Limit what one move can say: relative to the rating that explains the move best, no rating
    is penalised by more than `clip` nats. Without it a single blunder (or mouse slip) has almost
    zero probability for a strong player and decides the whole estimate."""
    logp = np.asarray(logp, np.float64)
    if clip is None or not logp.size:
        return logp
    return np.maximum(logp, logp.max(-1, keepdims=True) - clip)


def curve(logp, clip=CLIP):
    """logp [n_moves, len(GRID)] of one player's moves in order -> (mean [n], sd [n]) after each move."""
    if not len(logp):
        return np.zeros(0), np.zeros(0)
    return posterior(np.cumsum(robust(logp, clip), 0))


def summarise(logp, owner, n, clip=CLIP):
    """Per-player estimates after PLIES[c] // 2 own moves: est [n, len(PLIES)], sigma (nan where the
    game is shorter). logp [positions, len(GRID)], owner [positions] sorted player index."""
    from .response import PLIES
    est = np.full((n, len(PLIES)), np.nan, np.float32)
    sig = np.full((n, len(PLIES)), np.nan, np.float32)
    first = np.searchsorted(owner, np.arange(n))
    last = np.searchsorted(owner, np.arange(n), "right")
    for j in range(n):
        cum = np.cumsum(robust(logp[first[j]:last[j]], clip), 0)
        for c, k in enumerate(PLIES):
            if k // 2 <= len(cum):
                est[j, c], sig[j, c] = posterior(cum[k // 2 - 1])
    return est, sig


def measure(args):
    import torch
    import torch.nn.functional as F
    from .eval_player import legal_masks
    from .features import GameStore, to_device
    from .models import load_model
    from .response import BANDS, PLIES
    torch.set_grad_enabled(False)
    model, ckpt = load_model(args.ckpt, args.device)
    store = GameStore(args.data, args.split)
    rng = np.random.default_rng(args.seed)
    g = store.games_np
    ok = (g["bots"] == 0) & (g["n_plies"] >= 12)
    games, side = [], []
    for lo in BANDS[:-1]:
        for s, col in ((0, "white_elo"), (1, "black_elo")):  # the player sampled into the band
            cand = np.flatnonzero(ok & (g[col] >= lo) & (g[col] < lo + 200))
            pick = rng.choice(cand, size=min(args.per_band // 2, len(cand)), replace=False)
            games.append(pick)
            side.append(np.full(len(pick), s))
    games, side = np.concatenate(games), np.concatenate(side)
    max_moves = PLIES[-1] // 2
    # every position where the sampled player is to move, and which player it belongs to
    idx, owner = [], []
    for j, (gi, s) in enumerate(zip(games, side)):
        p = np.arange(g["start"][gi] + s, g["start"][gi] + g["n_plies"][gi], 2)[:max_moves]
        idx.append(p)
        owner.append(np.full(len(p), j))
    idx, owner = np.concatenate(idx), np.concatenate(owner)
    print(f"{len(games)} players, {len(idx)} positions x {len(GRID)} rating inputs", flush=True)
    logp = np.zeros((len(idx), len(GRID)), np.float32)
    for a in range(0, len(idx), args.batch):
        sl = slice(a, a + args.batch)
        b = to_device(store.batch(torch.as_tensor(idx[sl], device=store.device)), args.device)
        mask = legal_masks(store, idx[sl]).to(args.device)
        for k, e in enumerate(GRID):
            elo = torch.full_like(b["self_elo"], float(e))
            logits, _ = model(dict(b, self_elo=elo, opp_elo=elo))
            lp = F.log_softmax(logits.float().masked_fill(~mask, float("-inf")), -1)
            logp[sl, k] = lp.gather(1, b["move"].unsqueeze(1)).squeeze(1).cpu().numpy()
        if (a // args.batch) % 20 == 0:
            print(f"  {a + args.batch:>8} / {len(idx)}", flush=True)
    n = len(games)
    true = np.where(side == 0, g["white_elo"][games], g["black_elo"][games]).astype(np.float32)
    est, sig = summarise(logp, owner, n, args.clip)
    # logp and owner are kept so that the estimate can be redone with other settings (resummarise)
    np.savez(args.out, true=true, est=est, sigma=sig, plies=np.array(PLIES), side=side,
             opp=np.where(side == 0, g["black_elo"][games], g["white_elo"][games]), n_plies=g["n_plies"][games],
             step=ckpt["step"], logp=logp, owner=owner, clip=np.nan if args.clip is None else args.clip)
    print(f"saved {args.out}: {n} players")


def resummarise(args):
    d = dict(np.load(args.inp))
    d["est"], d["sigma"] = summarise(d["logp"], d["owner"], len(d["true"]), args.clip)
    d["clip"] = np.nan if args.clip is None else args.clip
    np.savez(args.out, **d)
    print(f"saved {args.out} (clip {args.clip})")


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("measure")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--per-band", type=int, default=800, help="players per 200-Elo band")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--clip", type=float, default=CLIP)
    p = sub.add_parser("resummarise", help="redo the estimates of a measure file with another clip")
    p.add_argument("--inp", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--clip", type=float, default=None)
    args = ap.parse_args(argv)
    (measure if args.cmd == "measure" else resummarise)(args)


if __name__ == "__main__":
    main()
