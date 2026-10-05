"""Device-resident dataset store and batch builder.

A whole split is kept as one uint8 tensor of raw 29-byte position records
(build_dataset.POS_DTYPE) plus small per-game tensors. Sampling and board
decoding are pure tensor ops, so with the store on the GPU there is no CPU data
pipeline at all (rapid+classical for one month is ~22 GB, which fits in a
48/80 GB card); on CPU the same code runs from RAM.
"""
import glob
import json
import os

import numpy as np
import torch

from .build_dataset import FLAG_PLAYER_OK, FLAG_BLACK, POS_DTYPE

REC = POS_DTYPE.itemsize
_OFF = {name: POS_DTYPE.fields[name][1] for name in POS_DTYPE.names}
ELO_MEAN, ELO_STD = 1600.0, 400.0


class GameStore:
    def __init__(self, root, split, device="cpu", max_shards=None, max_positions=None, rank=0, world=1,
                 dirs=None):
        """Load the first `max_shards` shards (a uniform random subset of games, since
        build_dataset shuffles) of `root/split`; with world > 1 only this rank's round-robin share.
        `dirs` names the shard directories explicitly instead."""
        if dirs is None:
            dirs = self.shard_dirs(root, split, max_shards)[rank::world]
        if not dirs:
            raise FileNotFoundError(f"no shards in {os.path.join(root, split)}")
        parts, games, offset = [], [], 0
        for d in dirs:
            g = np.load(os.path.join(d, "games.npy"))
            p = np.load(os.path.join(d, "positions.npy"), mmap_mode="r")
            if max_positions is not None and offset + len(p) > max_positions:
                keep = int(np.searchsorted(g["start"] + g["n_plies"], max_positions - offset, "right"))
                g = g[:keep]
                p = p[:int(g["start"][-1] + g["n_plies"][-1])] if keep else p[:0]
            parts.append(p)
            g = g.copy()
            g["start"] += offset
            games.append(g)
            offset += len(p)
            if max_positions is not None and offset >= max_positions:
                break
        g = np.concatenate(games)
        self.device = torch.device(device)
        raw = torch.empty((offset, REC), dtype=torch.uint8)  # filled shard by shard: no second copy in RAM
        offset = 0
        for p in parts:
            raw[offset:offset + len(p)] = torch.from_numpy(np.array(p).view(np.uint8).reshape(-1, REC))
            offset += len(p)
        self.raw = raw.to(self.device)
        t = lambda a, dt: torch.as_tensor(np.ascontiguousarray(a).astype(dt), device=self.device)
        self.game_start = t(g["start"], np.int64)
        self.game_len = t(g["n_plies"], np.int64)
        self.white_elo = t(g["white_elo"], np.float32)
        self.black_elo = t(g["black_elo"], np.float32)
        self.tc = t(g["tc"], np.int64)
        self.result = t(g["result"], np.int64)  # +1 white win, 0 draw, -1 black win
        field = lambda name: g[name] if name in g.dtype.names else np.zeros(len(g))
        self.bots = t(field("bots"), np.int64)  # bit0 white is a bot, bit1 black
        self.tail_enriched = t(field("tail_enriched"), np.bool_)
        # Games with the natural rating mix (without the extra games added for the rating tails).
        self.natural_games = torch.nonzero(~self.tail_enriched).squeeze(1)
        self.games_np = g
        self.n_positions = len(self.raw)
        self.n_games = len(g)
        manifest = os.path.join(root, "manifest.json")
        self.manifest = json.load(open(manifest)) if os.path.exists(manifest) else {}

    @staticmethod
    def shard_dirs(root, split, max_shards=None):
        return sorted(glob.glob(os.path.join(root, split, "shard_*")))[:max_shards]

    # ---------------------------------------------------------------- positions
    def mover_elo(self, idx):
        gi = torch.searchsorted(self.game_start, idx, right=True) - 1
        black = (self.raw[idx, _OFF["flags"]] & FLAG_BLACK) != 0
        return torch.where(black, self.black_elo[gi], self.white_elo[gi])

    def elo_balance_weights(self, alpha, floor=1e-3):
        """Acceptance probability per 100-Elo bin so that sampling ~ p(elo)^(1-alpha):
        alpha=0 uniform over positions, alpha=1 flat over Elo (bins below `floor` mass capped)."""
        plies = self.game_len.float() / 2
        hist = torch.zeros(40, device=self.device)
        for elo in (self.white_elo, self.black_elo):
            hist.index_add_(0, (elo // 100).long().clamp(0, 39), plies)
        f = (hist / hist.sum()).clamp(min=floor)
        w = f ** (-alpha)
        return w / w.max()

    def sample_player_indices(self, n, generator, elo_weights=None):
        """Random positions that are valid player targets (rejection sampling; ~99% qualify).
        elo_weights: optional per-100-Elo-bin acceptance probabilities (elo_balance_weights)."""
        out = []
        need = n
        while need > 0:
            draw = int(need * 1.1) + 16 if elo_weights is None else int(need * 4) + 64
            idx = torch.randint(0, self.n_positions, (draw,), generator=generator, device="cpu").to(self.device)
            ok = (self.raw[idx, _OFF["flags"]] & FLAG_PLAYER_OK) != 0
            if elo_weights is not None:
                u = torch.rand(draw, generator=generator).to(self.device)
                ok &= u < elo_weights[(self.mover_elo(idx) // 100).long().clamp(0, 39)]
            idx = idx[ok][:need]
            out.append(idx)
            need -= len(idx)
        return torch.cat(out)

    def player_indices(self):
        """All valid player positions (for evaluation)."""
        return torch.nonzero((self.raw[:, _OFF["flags"]] & FLAG_PLAYER_OK) != 0).squeeze(1)

    def batch(self, idx, elo_override=None):
        """Model inputs for positions `idx` (LongTensor on self.device).
        elo_override: optional (self_elo, opp_elo) float tensors replacing the true ratings."""
        rec = self.raw[idx]
        b = decode_records(rec)
        gi = torch.searchsorted(self.game_start, idx, right=True) - 1
        black = b.pop("black")
        we, be = self.white_elo[gi], self.black_elo[gi]
        b["self_elo"] = torch.where(black, be, we)
        b["opp_elo"] = torch.where(black, we, be)
        if elo_override is not None:
            b["self_elo"], b["opp_elo"] = elo_override
        b["tc"] = self.tc[gi]
        b["ply"] = (idx - self.game_start[gi]).clamp(max=255)
        res = self.result[gi]
        b["result"] = torch.where(black, -res, res) + 1  # mover POV: 0 loss, 1 draw, 2 win
        b["game"] = gi
        return b


def decode_records(rec):
    """rec: [B, 29] uint8 -> dict of model input tensors."""
    B = rec.shape[0]
    # clone() gives fresh aligned storage; a plain slice keeps the 29-byte stride / odd offset
    # (and .contiguous() is a no-op when B == 1), which dtype views reject.
    occ = rec[:, _OFF["occ"]:_OFF["occ"] + 8].reshape(-1).clone().view(torch.int64)
    bits = (occ.unsqueeze(1) >> torch.arange(64, device=rec.device)) & 1
    pieces = rec[:, _OFF["pieces"]:_OFF["pieces"] + 16].long()
    nib = torch.stack([pieces & 0xF, pieces >> 4], dim=2).reshape(B, 32)
    slot = (torch.cumsum(bits, 1) - 1).clamp(0, 31)
    board = torch.gather(nib, 1, slot) * bits
    flags = rec[:, _OFF["flags"]].long()
    move = rec[:, _OFF["move"]:_OFF["move"] + 2].reshape(-1).clone().view(torch.int16).long() & 0xFFFF
    return dict(
        board=board,                                   # [B,64] 0 empty, 1-6 own, 7-12 opponent
        castling=((flags.unsqueeze(1) >> torch.arange(4, device=rec.device)) & 1).float(),
        ep=rec[:, _OFF["ep"]].long(),                  # 0-7 file, 8 none
        halfmove=rec[:, _OFF["halfmove"]].long(),
        move=move,
        black=(flags & FLAG_BLACK) != 0,
    )


def records_from_boards(boards):
    """python-chess boards -> [B, 29] uint8 records (move field left 0)."""
    from .board_codec import encode_board
    recs = np.zeros(len(boards), POS_DTYPE)
    for i, board in enumerate(boards):
        occ, pieces, flags, ep = encode_board(board)
        recs[i]["occ"] = occ
        recs[i]["pieces"] = np.frombuffer(pieces, np.uint8)
        recs[i]["flags"] = flags
        recs[i]["ep"] = ep
        recs[i]["halfmove"] = min(board.halfmove_clock, 255)
    return torch.from_numpy(recs.view(np.uint8).reshape(-1, REC))


def to_device(batch, device):
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
