"""The player: an Elo-conditioned move predictor (policy network)."""
import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .features import ELO_MEAN, ELO_STD
from .vocab import VOCAB_SIZE

N_TC = 6  # pgn_stream.TC_CATEGORIES


class EloEncoder(nn.Module):
    """Maps a rating to a d-dim vector.

    'interp':  learned embeddings at knots every `step` Elo in [lo, hi], linearly
               interpolated; ratings outside the range are clamped, so out-of-range
               requests behave like the nearest edge (no extrapolation, no cliff).
    'fourier': MLP on multi-frequency sin/cos features of the normalized rating;
               smooth and able to extrapolate, but nothing constrains how.
    """

    def __init__(self, d, kind="interp", lo=400, hi=3000, step=100):
        super().__init__()
        self.kind, self.lo, self.hi, self.step = kind, lo, hi, step
        if kind == "interp":
            self.knots = nn.Embedding((hi - lo) // step + 1, d)
            nn.init.normal_(self.knots.weight, std=0.02)
        elif kind == "fourier":
            self.register_buffer("freqs", 2.0 ** torch.arange(6) * math.pi / 4)
            self.mlp = nn.Sequential(nn.Linear(13, d), nn.GELU(), nn.Linear(d, d))
        else:
            raise ValueError(kind)

    def forward(self, elo):
        if self.kind == "interp":
            x = (elo.clamp(self.lo, self.hi) - self.lo) / self.step
            i0 = x.floor().long().clamp(max=self.knots.num_embeddings - 2)
            w = (x - i0).unsqueeze(-1)
            return (1 - w) * self.knots(i0) + w * self.knots(i0 + 1)
        x = ((elo - ELO_MEAN) / ELO_STD).unsqueeze(-1)
        return self.mlp(torch.cat([x, torch.sin(x * self.freqs), torch.cos(x * self.freqs)], -1))


class BoardEncoder(nn.Module):
    """64 square tokens (piece + square embedding) plus `n_prefix` extra tokens
    supplied by the caller, through a pre-LN transformer."""

    def __init__(self, d, n_layers, n_heads, ff_mult=4, dropout=0.0):
        super().__init__()
        self.piece = nn.Embedding(13, d)
        self.square = nn.Parameter(torch.randn(64, d) * 0.02)
        self.castling = nn.Linear(4, d)
        self.ep = nn.Embedding(9, d)
        self.halfmove = nn.Embedding(101, d)
        self.ply = nn.Embedding(256, d)
        layer = nn.TransformerEncoderLayer(d, n_heads, ff_mult * d, dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)

    def forward(self, batch, prefix):
        """prefix: [B, P, d] tokens prepended to the board. Returns [B, P+64, d]."""
        sq = self.piece(batch["board"]) + self.square
        state = (self.castling(batch["castling"]) + self.ep(batch["ep"])
                 + self.halfmove(batch["halfmove"].clamp(max=100)) + self.ply(batch["ply"]))
        prefix = prefix + state.unsqueeze(1)
        return self.norm(self.transformer(torch.cat([prefix, sq], 1)))


@dataclass
class PlayerConfig:
    d_model: int = 256
    n_layers: int = 8
    n_heads: int = 8
    ff_mult: int = 4
    dropout: float = 0.0
    elo_encoding: str = "interp"
    elo_lo: int = 400
    elo_hi: int = 3000
    value_weight: float = 0.1  # auxiliary game-outcome head
    # 'interp' only: penalty on the second difference of adjacent Elo knots. Rare knots
    # (tails of the rating distribution) get almost no gradient and otherwise stay at their
    # random init, which reads as "no Elo information"; this makes them continue the trend
    # of their neighbours instead.
    elo_smooth: float = 0.0


class PlayerNet(nn.Module):
    """p(move | board, own Elo, opponent Elo, time control)."""

    def __init__(self, cfg: PlayerConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.encoder = BoardEncoder(d, cfg.n_layers, cfg.n_heads, cfg.ff_mult, cfg.dropout)
        self.elo = EloEncoder(d, cfg.elo_encoding, cfg.elo_lo, cfg.elo_hi)
        self.role = nn.Parameter(torch.randn(3, d) * 0.02)  # CLS, self-Elo, opp-Elo
        self.tc = nn.Embedding(N_TC, d)
        self.policy = nn.Linear(d, VOCAB_SIZE)
        self.value = nn.Linear(d, 3)

    def forward(self, batch):
        B = batch["board"].shape[0]
        prefix = torch.stack([
            self.role[0].expand(B, -1) + self.tc(batch["tc"]),
            self.role[1] + self.elo(batch["self_elo"]),
            self.role[2] + self.elo(batch["opp_elo"]),
        ], 1)
        h = self.encoder(batch, prefix)[:, 0]
        return self.policy(h), self.value(h)

    def loss(self, batch):
        return self.loss_from_outputs(self(batch), batch)

    def loss_from_outputs(self, outputs, batch):
        """Loss from forward outputs, so a trainer can run the forward through a DDP wrapper."""
        logits, value = outputs
        pl = F.cross_entropy(logits.float(), batch["move"])
        vl = F.cross_entropy(value.float(), batch["result"])
        top1 = (logits.argmax(-1) == batch["move"]).float().mean()
        loss = pl + self.cfg.value_weight * vl
        metrics = dict(policy_loss=pl.detach(), value_loss=vl.detach(), top1=top1.detach())
        if self.cfg.elo_smooth > 0 and self.cfg.elo_encoding == "interp":
            k = self.elo.knots.weight
            smooth = (k[2:] - 2 * k[1:-1] + k[:-2]).pow(2).sum(-1).mean()
            loss = loss + self.cfg.elo_smooth * smooth
            metrics["elo_smooth"] = smooth.detach()
        return loss, metrics


def build(kind, cfg_dict):
    """kind is kept for the checkpoint format; there is one model class."""
    return PlayerNet(PlayerConfig(**cfg_dict))


def config_dict(model):
    return asdict(model.cfg)


def load_model(ckpt_path, device="cpu"):
    """A training checkpoint (.pt) or exported weights (<name>.safetensors + <name>.json, see
    fastelo.export). Returns (model in eval mode, info dict with at least kind / model_cfg / step)."""
    ckpt_path = str(ckpt_path)
    if ckpt_path.endswith(".safetensors"):
        import json
        from safetensors.torch import load_file
        ckpt = json.load(open(ckpt_path[:-len(".safetensors")] + ".json"))
        state = {k: v.float() if v.is_floating_point() else v for k, v in load_file(ckpt_path).items()}
    else:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = ckpt["model"]
    model = build(ckpt["kind"], ckpt["model_cfg"])
    model.load_state_dict(state)
    return model.to(device).eval(), ckpt
