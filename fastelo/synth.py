"""Synthetic players with phase-dependent strength, played in large batches (GPU).

    # 1. true strength of random phase-varying players: a ladder against calibrated bots
    python -m fastelo.synth ladder --calibration cal.json --players 3000 --rounds 40 --out runs/ladder
    # 2. label formula: ladder rating as a function of (opening, middlegame, endgame) strength
    python -m fastelo.synth fit --ladder runs/ladder --out runs/ladder/labels.json
    # 3. games between random players, each labelled by the formula
    python -m fastelo.synth generate --calibration cal.json --labels runs/ladder/labels.json \
        --games 300000 --out data/synth_v1

Why: on Lichess a player's opening, middlegame and endgame strength go together, so human games
cannot show how much each phase is worth or whether a rating method leans on one of them. In these
games the three vary independently, and so does the opponent's strength.

All games of a batch advance one ply per step with one network call (the calibrated policy of
fastelo.calibrate: Elo input, temperature and mix with the strong model per position).
`generate` writes the same shard format as build_dataset (positions.npy / games.npy).
"""
import argparse
import json
import os
import time

import chess
import numpy as np
import torch

from .build_dataset import FLAG_PLAYER_OK, GAME_DTYPE, POS_DTYPE
from .calibrate import settings
from .features import decode_records, records_from_boards
from .models import load_model
from .pgn_stream import TC_CATEGORIES
from .vocab import decode_move, legal_mask

RAPID = TC_CATEGORIES.index("rapid")


# ---------------------------------------------------------------- phase-varying players
NPM = {chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
PIECE_VALUE = {chess.PAWN: 1, **NPM}


def phase_weights(board, ply, open_ply=24, open_width=4.0, end_npm=20, end_width=4.0):
    """(w_open, w_mid, w_end), smooth and summing to 1."""
    npm = sum(NPM.get(p.piece_type, 0) for p in board.piece_map().values())
    w_open = 1 / (1 + np.exp((ply - open_ply) / open_width))
    w_end = 1 / (1 + np.exp((npm - end_npm) / end_width))
    return w_open, (1 - w_open) * (1 - w_end), (1 - w_open) * w_end


def elo_map(opp, score, prior_mean=1500.0, prior_sd=600.0):
    """Rating (MAP under a Gaussian prior) and standard error from games against `opp` with scores
    in {0, .5, 1}, under P(score) = 1 / (1 + 10^((R_opp - R) / 400))."""
    opp, score = np.asarray(opp, float), np.asarray(score, float)
    k = np.log(10) / 400
    r = prior_mean
    for _ in range(100):
        p = 1 / (1 + 10 ** ((opp - r) / 400))
        g = np.sum(score - p) * k - (r - prior_mean) / prior_sd ** 2
        h = -np.sum(p * (1 - p)) * k * k - 1 / prior_sd ** 2
        step = float(np.clip(-g / h, -200, 200))
        r += step
        if abs(step) < 1e-3:
            break
    p = 1 / (1 + 10 ** ((opp - r) / 400))
    return r, float((np.sum(p * (1 - p)) * k * k + 1 / prior_sd ** 2) ** -0.5)


def sample_player(rng, lo=600, hi=2300, sigma=200, outlier=0.08, clamp=(500, 2500)):
    """((opening, middlegame, endgame) target Elos, kind)."""
    mid = rng.uniform(lo, hi)
    o, e = mid + rng.normal(0, sigma), mid + rng.normal(0, sigma)
    kind = "normal"
    if rng.random() < outlier:
        which = rng.integers(3)
        shift = rng.uniform(500, 1000) * rng.choice([-1, 1])
        if which == 0:
            o, kind = mid + shift, "opening outlier"
        elif which == 1:
            mid, kind = mid + shift, "middlegame outlier"
        else:
            e, kind = mid + shift, "endgame outlier"
    return tuple(float(np.clip(x, *clamp)) for x in (o, mid, e)), kind




class BatchedPolicy:
    """Calibrated policy for many positions at once."""

    def __init__(self, calibration, device=None, amp=True, player=None, strong=None):
        """player / strong: checkpoint paths overriding the ones recorded in the calibration."""
        self.cal = json.load(open(calibration)) if isinstance(calibration, str) else calibration
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.player, _ = load_model(player or self.cal["player"], self.device)
        self.strong, _ = load_model(strong or self.cal["strong_player"], self.device)
        self.amp = amp and self.device.type == "cuda"
        self._cache = {}
        if self.device.type == "cuda":
            # the CPU side is python-chess on one core; a full thread pool per process only makes
            # several processes on one box fight each other (measured: 12 processes = speed of 1)
            torch.set_num_threads(2)

    def knobs(self, elo):
        key = round(float(elo))
        if key not in self._cache:
            self._cache[key] = settings(self.cal, key)
        return self._cache[key]

    @torch.no_grad()
    def choose(self, boards, plies, elos, gen, recs=None):
        """One move (vocabulary index, mover's perspective) for every board at its target Elo.
        recs: records_from_boards(boards) if the caller already has them."""
        k = np.array([self.knobs(e) for e in elos], np.float32)                 # [B, 3] input, T, mix
        b = decode_records(records_from_boards(boards) if recs is None else recs)
        b.pop("black")
        b["tc"] = torch.full((len(boards),), RAPID, dtype=torch.long)
        b["ply"] = torch.as_tensor(np.minimum(plies, 255), dtype=torch.long)
        b = {key: v.to(self.device) for key, v in b.items()}
        c = torch.from_numpy(k[:, 0]).to(self.device)
        T = torch.from_numpy(k[:, 1]).to(self.device)
        mix = torch.from_numpy(k[:, 2]).to(self.device)
        mask = torch.from_numpy(np.stack([legal_mask(bd) for bd in boards])).to(self.device)
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.amp):
            la = self.player(dict(b, self_elo=c, opp_elo=c))[0].float()
        z = torch.log_softmax(la.masked_fill(~mask, float("-inf")), -1) / T[:, None]
        need = mix > 0
        if need.any():
            sub = {key: v[need] for key, v in b.items()}
            se = torch.full((int(need.sum()),), float(self.cal["strong_elo"]), device=self.device)
            with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.amp):
                ls = self.strong(dict(sub, self_elo=se, opp_elo=se))[0].float()
            ls = torch.log_softmax(ls.masked_fill(~mask[need], float("-inf")), -1) / self.cal["strong_temperature"]
            m = mix[need][:, None]
            z[need] = torch.nan_to_num((1 - m) * z[need], neginf=-1e9) + m * torch.nan_to_num(ls, neginf=-1e9)
        p = torch.softmax(z.masked_fill(~mask, float("-inf")), -1)
        return torch.multinomial(p, 1, generator=gen).squeeze(1).cpu().numpy()


def result_of(board, max_plies, n_plies):
    """+1 white win / 0 draw / -1 black win, or None while the game goes on."""
    if board.is_checkmate():
        return -1 if board.turn == chess.WHITE else 1
    if (board.is_stalemate() or board.is_insufficient_material() or board.halfmove_clock >= 100
            or board.is_repetition(3)):
        return 0
    if n_plies >= max_plies:  # adjudicate by material
        bal = sum((1 if p.color == chess.WHITE else -1) * PIECE_VALUE.get(p.piece_type, 0)
                  for p in board.piece_map().values())
        return 1 if bal >= 3 else -1 if bal <= -3 else 0
    return None


def play_batch(policy, specs, gen, max_plies=200, keep_records=True):
    """Play one game per (white_spec, black_spec) pair, all in parallel.
    Returns per game: result (+1/0/-1) and, if keep_records, its POS_DTYPE records."""
    n = len(specs)
    boards = [chess.Board() for _ in range(n)]
    recs = [[] for _ in range(n)]
    results = [None] * n
    active = list(range(n))
    ply = 0
    while active:
        bs = [boards[i] for i in active]
        side = ply % 2
        elos = []
        for i, bd in zip(active, bs):
            o, m, e = specs[i][side]
            wo, wm, we = phase_weights(bd, ply)
            elos.append(wo * o + wm * m + we * e)
        raw = records_from_boards(bs)
        moves = policy.choose(bs, np.full(len(bs), ply), elos, gen, raw)
        if keep_records:
            r = raw.numpy().view(POS_DTYPE).reshape(-1).copy()
            r["move"] = moves
            r["flags"] |= FLAG_PLAYER_OK
        still = []
        for j, i in enumerate(active):
            if keep_records:
                recs[i].append(r[j])
            boards[i].push(decode_move(int(moves[j]), boards[i].turn))
            res = result_of(boards[i], max_plies, ply + 1)
            if res is None:
                still.append(i)
            else:
                results[i] = res
        active = still
        ply += 1
    return results, ([np.array(x, POS_DTYPE) for x in recs] if keep_records else None)


# ------------------------------------------------------------------------------ ladder
_LW = {}


def _lw_init(calibration, device, player, strong):
    torch.set_num_threads(2)  # the work is python-chess on one core; more threads only spin
    _LW["policy"] = BatchedPolicy(calibration, device, player=player, strong=strong)


def _lw_play(job):
    specs, seed, max_plies = job
    pol = _LW["policy"]
    gen = torch.Generator(device=pol.device).manual_seed(seed)
    return play_batch(pol, specs, gen, max_plies, keep_records=False)[0]


def ladder(args):
    """--workers N: each round's games are split over N processes (each with the policy on the
    GPU); the chess logic is CPU-bound, so this scales with cores until the GPU is busy."""
    import multiprocessing as mp
    pool = None
    if args.workers > 1:
        pool = mp.get_context("spawn").Pool(args.workers, initializer=_lw_init,
                                            initargs=(args.calibration, args.device, args.player, args.strong))
        device = torch.device("cpu")
    else:
        policy = BatchedPolicy(args.calibration, args.device, player=args.player, strong=args.strong)
        device = policy.device
    rng = np.random.default_rng(args.seed)
    gen = torch.Generator(device=device).manual_seed(args.seed)
    players = [sample_player(rng, sigma=args.sigma, outlier=args.outlier) for _ in range(args.players)]
    rating = np.full(args.players, 1500.0)
    opps, scores, paths = [[] for _ in players], [[] for _ in players], [[] for _ in players]
    os.makedirs(args.out, exist_ok=True)
    first, saved = 0, os.path.join(args.out, "players.jsonl")
    if os.path.exists(saved):  # resume after a stop: the same players (same seed), rounds played so far
        try:
            old = [json.loads(line) for line in open(saved)]
        except ValueError:  # a file cut off mid-write by an older version: start again
            old = []
        if len(old) == len(players) and all(np.allclose(o["spec"], p[0]) for o, p in zip(old, players)):
            first = min(len(o["path"]) for o in old)
            for i, o in enumerate(old):
                opps[i], scores[i], paths[i] = o["opps"][:first], o["scores"][:first], o["path"][:first]
                if first:
                    rating[i] = elo_map(opps[i], scores[i])[0]
            rng = np.random.default_rng(args.seed + 1000 * first)
            gen.manual_seed(args.seed + 1000 * first)
            print(f"resuming the ladder after round {first}", flush=True)
    t0 = time.time()
    for rnd in range(first, args.rounds):
        opp = np.clip(rating + rng.normal(0, 50, args.players), 500, 2500)
        white = (np.arange(args.players) + rnd) % 2 == 0          # alternate colours
        pair = lambda i: (players[i][0], (opp[i],) * 3) if white[i] else ((opp[i],) * 3, players[i][0])
        if pool is not None:
            chunks = [c for c in np.array_split(np.arange(args.players), args.workers) if len(c)]
            outs = pool.map(_lw_play, [([pair(i) for i in c], args.seed * 7919 + rnd * 1009 + k, args.max_plies)
                                       for k, c in enumerate(chunks)])
            batches = list(zip(chunks, outs))
        else:
            batches = []
            for s in range(0, args.players, args.batch):
                idx = range(s, min(args.players, s + args.batch))
                batches.append((idx, play_batch(policy, [pair(i) for i in idx], gen, args.max_plies,
                                                keep_records=False)[0]))
        for idx, res in batches:
            for i, r in zip(idx, res):
                sc = (r + 1) / 2 if white[i] else (1 - r) / 2
                opps[i].append(float(opp[i]))
                scores[i].append(sc)
                rating[i] = elo_map(opps[i], scores[i])[0]
                paths[i].append(round(float(rating[i]), 1))
        print(f"round {rnd + 1}/{args.rounds} ({time.time() - t0:.0f}s)", flush=True)
        with open(saved + ".tmp", "w") as f:  # every round, atomically: a stop loses one round at most
            for i, (spec, kind) in enumerate(players):
                r, se = elo_map(opps[i], scores[i])
                f.write(json.dumps(dict(id=i, spec=list(spec), kind=kind, ladder=r, ladder_se=se, path=paths[i],
                                        opps=opps[i], scores=scores[i])) + "\n")
        os.replace(saved + ".tmp", saved)
    if pool is not None:
        pool.close()
    fit(argparse.Namespace(ladder=args.out, out=os.path.join(args.out, "labels.json")))


def features(spec):
    """Regressors of the label formula: the three strengths plus their spread (a weak phase costs
    more than a strong one gains)."""
    o, m, e = np.asarray(spec, float).T
    return np.stack([np.ones_like(m), o, m, e, np.minimum(o, np.minimum(m, e))], -1)


def label(labels, spec):
    return np.asarray(features(spec) @ np.asarray(labels["coef"]))


def fit(args):
    P = [json.loads(line) for line in open(os.path.join(args.ladder, "players.jsonl"))]
    X = np.array([p["spec"] for p in P])
    y = np.array([p["ladder"] for p in P])
    w = 1 / np.array([p["ladder_se"] for p in P]) ** 2
    A = features(X)
    lin = np.linalg.lstsq(A[:, :4] * np.sqrt(w)[:, None], y * np.sqrt(w), rcond=None)[0]
    coef = np.linalg.lstsq(A * np.sqrt(w)[:, None], y * np.sqrt(w), rcond=None)[0]
    res_lin, res = y - A[:, :4] @ lin, y - A @ coef
    noise = float(np.mean([p["ladder_se"] ** 2 for p in P]))
    out = dict(coef=[float(c) for c in coef], terms=["1", "opening", "middlegame", "endgame", "min(phases)"],
               players=len(P), residual_sd=float(res.std()), residual_sd_linear=float(res_lin.std()),
               ladder_se=float(np.sqrt(noise)), model_error=float(np.sqrt(max(res.var() - noise, 0.0))),
               linear_coef=[float(c) for c in lin])
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


# ---------------------------------------------------------------------------- generate
def sample_pair(rng, sigma, outlier, independent):
    """Two players; the opponent's strength is independent of the player's with prob. `independent`
    (so it is no evidence), else within ~200 Elo (as in matched play)."""
    a, _ = sample_player(rng, sigma=sigma, outlier=outlier)
    if rng.random() < independent:
        b, _ = sample_player(rng, sigma=sigma, outlier=outlier)
    else:
        shift = rng.normal(0, 200)
        b0, _ = sample_player(rng, sigma=sigma, outlier=outlier)
        d = (a[1] + shift) - b0[1]
        b = tuple(float(np.clip(x + d, 500, 2500)) for x in b0)
    return a, b


def shard_name(part, k):
    return f"shard_{part:02d}{k:03d}"


def write_shard(root, split, name, games, recs):
    d = os.path.join(root, split, name)
    os.makedirs(d, exist_ok=True)
    np.save(os.path.join(d, "positions.npy"), np.concatenate(recs))
    np.save(os.path.join(d, "games.npy"), games)


def generate(args):
    """--part i/N: this process makes 1/N of the games with its own seed and shard names, so N
    processes can share one GPU (python-chess, not the network, is the bottleneck)."""
    part, parts = (int(x) for x in args.part.split("/"))
    policy = BatchedPolicy(args.calibration, args.device, player=args.player, strong=args.strong)
    labels = json.load(open(args.labels))
    seed = args.seed * 1000 + part
    rng = np.random.default_rng(seed)
    gen = torch.Generator(device=policy.device).manual_seed(seed)
    os.makedirs(args.out, exist_ok=True)
    counts = {"train": args.games // parts, "val": args.val_games // parts}
    meta = dict(schema_version=1, synthetic=True, labels=labels, calibration=os.path.abspath(args.calibration),
                sigma=args.sigma, outlier=args.outlier, independent_opponent=args.independent, splits={})
    t0 = time.time()
    for split, total in counts.items():
        done, shard, g_buf, r_buf, n_pos, specs_log = 0, 0, [], [], 0, []
        sdir = os.path.join(args.out, split)
        existing = sorted(d for d in (os.listdir(sdir) if os.path.isdir(sdir) else [])
                          if d.startswith(f"shard_{part:02d}") and os.path.exists(os.path.join(sdir, d, "specs.npy")))
        if existing:  # resume after a stop: keep this part's finished shards
            shard = len(existing)
            done = sum(len(np.load(os.path.join(sdir, d, "games.npy"), mmap_mode="r")) for d in existing)
            rng = np.random.default_rng(seed + 7919 * shard + (1 if split == "val" else 0))
        while done < total:
            n = min(args.batch, total - done)
            pairs = [sample_pair(rng, args.sigma, args.outlier, args.independent) for _ in range(n)]
            results, recs = play_batch(policy, pairs, gen, args.max_plies)
            G = np.zeros(n, GAME_DTYPE)
            lab_w, lab_b = label(labels, [p[0] for p in pairs]), label(labels, [p[1] for p in pairs])
            for i in range(n):
                G[i]["id"] = f"{part:x}{done + i:07d}".encode()[:8]
                G[i]["white_elo"], G[i]["black_elo"] = int(np.clip(lab_w[i], 0, 4000)), int(np.clip(lab_b[i], 0, 4000))
                G[i]["base"], G[i]["tc"], G[i]["result"] = 600, RAPID, results[i]
                G[i]["split"] = 0 if split == "train" else 1
                G[i]["n_plies"], G[i]["start"] = len(recs[i]), n_pos
                n_pos += len(recs[i])
            g_buf.append(G)
            r_buf.extend(recs)
            specs_log.extend([list(p[0]) + list(p[1]) for p in pairs])
            done += n
            print(f"{split}: {done}/{total} games ({time.time() - t0:.0f}s)", flush=True)
            if n_pos >= args.shard_positions or done >= total:
                write_shard(args.out, split, shard_name(part, shard), np.concatenate(g_buf), r_buf)
                # written last: marks the shard as complete for a resumed run
                np.save(os.path.join(sdir, shard_name(part, shard), "specs.npy"), np.array(specs_log, np.float32))
                shard, g_buf, r_buf, n_pos, specs_log = shard + 1, [], [], 0, []
        meta["splits"][split] = dict(games=done, shards=shard)
    with open(os.path.join(args.out, "manifest.json" if parts == 1 else f"manifest_{part:02d}.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps(meta["splits"]))


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--calibration", required=True)
        p.add_argument("--player", default=None, help="player checkpoint (default: the path in the calibration)")
        p.add_argument("--strong", default=None, help="strong-model checkpoint (default: as in the calibration)")
        p.add_argument("--sigma", type=float, default=200.0, help="spread of opening/endgame around the middlegame")
        p.add_argument("--outlier", type=float, default=0.08, help="share of players with one phase 500-1000 Elo off")
        p.add_argument("--max-plies", type=int, default=200)
        p.add_argument("--batch", type=int, default=2048, help="games played in parallel")
        p.add_argument("--device", default=None)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--out", required=True)

    p = sub.add_parser("ladder")
    common(p)
    p.add_argument("--players", type=int, default=3000)
    p.add_argument("--rounds", type=int, default=40)
    p.add_argument("--workers", type=int, default=1, help="processes playing each round's games in parallel")
    p = sub.add_parser("fit")
    p.add_argument("--ladder", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("generate")
    common(p)
    p.add_argument("--labels", required=True)
    p.add_argument("--games", type=int, default=300000)
    p.add_argument("--val-games", type=int, default=5000)
    p.add_argument("--independent", type=float, default=0.5, help="share of games with an unrelated opponent")
    p.add_argument("--shard-positions", type=int, default=4_000_000)
    p.add_argument("--part", default="0/1", help="i/N: make the i-th of N equal parts (run N processes)")
    args = ap.parse_args(argv)
    {"ladder": ladder, "fit": fit, "generate": generate}[args.cmd](args)


if __name__ == "__main__":
    main()
