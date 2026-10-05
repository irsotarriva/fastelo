"""Distil a Leela teacher into our player ("learn to play" before the human phase).

    python -m fastelo.distill train --parts <parts dir> --labels <leela_label run output dir> \
        --bench data/student_bench_50k.npz --run distill_t4 --steps 3000 --batch 1024
    python -m fastelo.distill make-bench --sf-labels data/sf_labels/val_n1000 \
        --teacher-bench data/teacher_bench_50k.npz --out data/student_bench_50k.npz

Training data are `leela_label` parts (whole games) plus the teacher output: logits over the
legal moves and WDL. The student (models.PlayerNet) is trained on the legal moves only:
  policy: cross-entropy to softmax(teacher_logits / tau)   (= KL + teacher entropy)
  value:  cross-entropy to the teacher's WDL
The Elo inputs are set to a constant `--elo` (the teacher is not a human of any rating; the
human phase later teaches what the Elo input means).

Strength is measured on the Stockfish-labelled benchmark positions: expected-score loss of the
student's top move and of sampling from it (same metric as teacher_bench / eval_strength).
"""
import argparse
import glob
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from .eval_strength import win_prob
from .features import decode_records
from .leela_label import load_npy
from .models import build
from .pgn_stream import TC_CATEGORIES

RAPID = TC_CATEGORIES.index("rapid")


# ------------------------------------------------------------------ segment (CSR) helpers
def csr_gather_index(starts, lens):
    """Flat indices of the concatenated ranges [s, s+l) and the segment id of each."""
    seg = torch.repeat_interleave(torch.arange(len(lens), device=lens.device), lens)
    first = torch.cumsum(lens, 0) - lens
    flat = starts[seg] + (torch.arange(len(seg), device=lens.device) - first[seg])
    return flat, seg


def segment_log_softmax(x, seg, n):
    # the per-segment max is only for numerical stability: keep it out of the gradient
    m = torch.full((n,), float("-inf"), device=x.device, dtype=x.dtype).scatter_reduce(0, seg, x.detach(), "amax")
    z = torch.zeros(n, device=x.device, dtype=x.dtype).index_add_(0, seg, torch.exp(x - m[seg]))
    return x - (torch.log(z) + m)[seg]


def segment_sum(x, seg, n):
    return torch.zeros(n, device=x.device, dtype=x.dtype).index_add_(0, seg, x)


def segment_argmax(x, seg, n):
    """Flat index of the max element of each segment."""
    m = torch.full((n,), float("-inf"), device=x.device, dtype=x.dtype).scatter_reduce(0, seg, x, "amax")
    hit = (x == m[seg]).nonzero().squeeze(1)
    first = torch.full((n,), len(x), device=x.device, dtype=torch.long).scatter_reduce(0, seg[hit], hit, "amin")
    return first


# -------------------------------------------------------------------------------- data
def labelled_parts(parts_dir, labels_dir):
    """Names of the parts that have teacher output."""
    return [os.path.basename(p) for p in sorted(glob.glob(os.path.join(parts_dir, "part_*")))
            if any(os.path.exists(os.path.join(labels_dir, os.path.basename(p), f)) for f in ("wdl.npy.zst", "wdl.npy"))]


class DistillStore:
    """Labelled parts on one device; `batch(idx)` builds player inputs plus the legal-move CSR."""

    def __init__(self, parts_dir, labels_dir, device="cpu", max_parts=None, elo=3000.0, names=None,
                 train_device=None):
        """device: where the arrays live: "cpu", "cuda[:i]", or "auto" = the training GPU if the arrays
        take < 35% of its memory, else host RAM (batches are then gathered on CPU and moved).
        names: explicit part names (e.g. this rank's share), default all labelled parts."""
        if names is None:
            names = labelled_parts(parts_dir, labels_dir)[:max_parts]
        if not names:
            raise FileNotFoundError(f"no labelled parts: {parts_dir} + {labels_dir}")
        cols = dict(raw=[], ply=[], off=[np.zeros(1, np.int64)], moves=[], tlogits=[], twdl=[])
        base = 0
        for name in names:
            p, o = os.path.join(parts_dir, name), os.path.join(labels_dir, name)
            cols["raw"].append(load_npy(os.path.join(p, "records.npy")))
            cols["ply"].append(load_npy(os.path.join(p, "ply.npy")).astype(np.int16))
            po = load_npy(os.path.join(p, "legal_off.npy"))
            cols["off"].append((po[1:] + base).astype(np.int64))
            base += int(po[-1])
            cols["moves"].append(load_npy(os.path.join(p, "legal_ours.npy")))
            cols["tlogits"].append(load_npy(os.path.join(o, "logits.npy")))
            # lc0 WDL is [win, draw, loss] for the side to move; our value head is [loss, draw, win]
            cols["twdl"].append(np.ascontiguousarray(load_npy(os.path.join(o, "wdl.npy"))[:, ::-1]))
        train_device = torch.device(train_device) if train_device else torch.device("cpu")
        if device == "auto":
            nbytes = sum(a.nbytes for parts in cols.values() for a in parts)
            fits = (train_device.type == "cuda"
                    and nbytes < 0.35 * torch.cuda.get_device_properties(train_device).total_memory)
            device = train_device if fits else "cpu"
        dev = torch.device(device)
        self.device, self.train_device, self.parts, self.elo = dev, train_device, names, float(elo)
        for k in list(cols):  # one column at a time, so the per-part copies are freed as we go
            setattr(self, k, torch.from_numpy(np.concatenate(cols.pop(k))).to(dev))
        self.n = len(self.raw)

    def batch(self, idx):
        """Inputs for positions `idx` (on the store's device), returned on the training device."""
        idx = idx.to(self.device)
        b = self._batch(idx)
        return {k: v.to(self.train_device, non_blocking=True) for k, v in b.items()}

    def _batch(self, idx):
        b = decode_records(self.raw[idx])
        b.pop("black")
        B = len(idx)
        b["self_elo"] = torch.full((B,), self.elo, device=self.device)
        b["opp_elo"] = b["self_elo"]
        b["tc"] = torch.full((B,), RAPID, device=self.device, dtype=torch.long)
        b["ply"] = self.ply[idx].long().clamp(0, 255)
        starts, lens = self.off[idx], self.off[idx + 1] - self.off[idx]
        flat, seg = csr_gather_index(starts, lens)
        b["legal_moves"] = self.moves[flat].long()
        b["legal_seg"] = seg
        b["teacher_logits"] = self.tlogits[flat].float()
        b["teacher_wdl"] = self.twdl[idx].float()
        return b


def distill_loss(model, b, tau=1.0, value_weight=0.5):
    logits, value = model(b)
    B = len(b["ply"])
    seg = b["legal_seg"]
    s = logits.float()[seg, b["legal_moves"]]
    logp_s = segment_log_softmax(s, seg, B)
    logp_t = segment_log_softmax(b["teacher_logits"] / tau, seg, B)
    p_t = logp_t.exp()
    ce = -segment_sum(p_t * logp_s, seg, B)
    ent = -segment_sum(p_t * logp_t, seg, B)
    vloss = -(b["teacher_wdl"] * F.log_softmax(value.float(), -1)).sum(-1)
    agree = (segment_argmax(s, seg, B) == segment_argmax(b["teacher_logits"], seg, B)).float()
    loss = ce.mean() + value_weight * vloss.mean()
    return loss, dict(kl=(ce - ent).mean().detach(), value_ce=vloss.mean().detach(), top1_agree=agree.mean().detach())


# ------------------------------------------------------------------------------- bench
def make_bench(args):
    """Student-side copy of the teacher benchmark: our records + ply for the same positions."""
    from .shards import load_shards
    tb = np.load(args.teacher_bench)
    n = len(tb["offsets"]) - 1
    meta = json.load(open(os.path.join(args.sf_labels, "meta.json")))
    shards = load_shards(os.path.join(meta["data"], meta["split"]))
    recs, srcs = [], []
    for d in sorted(glob.glob(os.path.join(args.sf_labels, "chunk_*"))):
        recs.append(np.load(os.path.join(d, "records.npy")))
        srcs.append(np.load(os.path.join(d, "source.npy")))
    recs, srcs = np.concatenate(recs)[:n], np.concatenate(srcs)[:n]
    ply = np.array([int(i) - int(shards[s][0]["start"][np.searchsorted(shards[s][0]["start"], i, "right") - 1])
                    for s, i in srcs], np.int16)
    np.savez_compressed(args.out, records=recs, ply=ply, offsets=tb["offsets"], ours=tb["ours"],
                        sf_cp=tb["sf_cp"], played_local=tb["played_local"], mover_elo=tb["mover_elo"])
    print(f"wrote {args.out}: {n} positions")


@torch.no_grad()
def eval_strength(model, bench, device, elo, amp_dtype=None, batch=4096):
    """Expected-score loss (vs Stockfish's best move) of the student's top move and policy."""
    model.eval()
    off = torch.from_numpy(bench["offsets"]).to(device)
    ours = torch.from_numpy(bench["ours"].astype(np.int64)).to(device)
    cp = bench["sf_cp"].astype(np.float64)
    wp = torch.from_numpy(win_prob(np.clip(cp, -3000, 3000))).to(device)
    raw = torch.from_numpy(bench["records"]).to(device)
    ply = torch.from_numpy(bench["ply"].astype(np.int64)).to(device)
    played = torch.from_numpy(bench["played_local"]).to(device)
    n = len(raw)
    tot = dict(argmax_loss=0.0, sampling_loss=0.0, argmax_is_sf_best=0.0, human_move_top1=0.0)
    for s in range(0, n, batch):
        idx = torch.arange(s, min(n, s + batch), device=device)
        b = decode_records(raw[idx])
        b.pop("black")
        B = len(idx)
        b.update(self_elo=torch.full((B,), float(elo), device=device), tc=torch.full((B,), RAPID, device=device),
                 ply=ply[idx].clamp(0, 255))
        b["opp_elo"] = b["self_elo"]
        lens = off[idx + 1] - off[idx]
        flat, seg = csr_gather_index(off[idx], lens)
        first_local = torch.cumsum(lens, 0) - lens  # each position's first move in the flat batch
        with torch.autocast(device.type if isinstance(device, torch.device) else str(device).split(":")[0],
                            dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None):
            logits, _ = model(b)
        s_log = logits.float()[seg, ours[flat]]
        logp = segment_log_softmax(s_log, seg, B)
        best = torch.full((B,), -1.0, device=device, dtype=wp.dtype).scatter_reduce(0, seg, wp[flat], "amax")
        loss = best[seg] - wp[flat]
        top = segment_argmax(s_log, seg, B)
        tot["argmax_loss"] += loss[top].sum().item()
        tot["sampling_loss"] += segment_sum(logp.exp().double() * loss, seg, B).sum().item()
        tot["argmax_is_sf_best"] += (loss[top] == 0).sum().item()
        tot["human_move_top1"] += (top == first_local + played[idx]).sum().item()
    model.train()
    return {k: round(v / n, 5) for k, v in tot.items()}


# ------------------------------------------------------------------------------- train
def train(args):
    from .runtime import Dist, Rotating, fit, plan_data, shard_subset
    dist = Dist()  # rank / device (fit() reuses the same process group)
    all_parts = labelled_parts(args.parts, args.labels)[:args.max_parts]
    names = shard_subset(all_parts, dist)
    # this rank's parts, grouped so one group fits in memory (a group is swapped when used up)
    meta = {n: json.load(open(os.path.join(args.parts, n, "meta.json"))) for n in names}
    nbytes = [meta[n]["positions"] * 45 + meta[n]["legal_moves"] * 4 for n in names]
    where, groups = plan_data(names, nbytes, dist, args.store_device, args.data_ram_gb)
    data = Rotating(groups, [sum(meta[n]["positions"] for n in g) for g in groups],
                    lambda group: DistillStore(args.parts, args.labels, where, elo=args.elo, names=group,
                                               train_device=dist.device),
                    args.batch / dist.world, log=(lambda m: print(m, flush=True)) if dist.main else None)
    store = data.at(0)
    # the main rank holds out the last positions (whole games) of its first group for validation
    n_val = min(args.val_positions, store.n // 10) if dist.main else 0
    if dist.main:
        print(f"{len(all_parts)} labelled parts, {dist.world} rank(s); rank 0: {len(names)} parts in "
              f"{len(groups)} group(s) on {store.device}, first group {store.n:,} positions (val {n_val:,})",
              flush=True)
    bench = dict(np.load(args.bench)) if (args.bench and dist.main) else None
    val_idx = torch.arange(store.n - n_val, store.n)[:args.eval_positions]
    val_batches = [store.batch(val_idx[s:s + 4096]) for s in range(0, len(val_idx), 4096)]
    del store

    def make_batch(step, micro, size):
        store = data.at(step)
        g = torch.Generator().manual_seed(((args.seed * 1_000_003 + step) * 64 + micro) * 1024 + dist.rank)
        hold = n_val if data.current == 0 else 0
        return store.batch(torch.randint(0, store.n - hold, (size,), generator=g))

    def loss_fn(model, b):
        return distill_loss(model, b, args.tau, args.value_weight)

    def evaluate(model):
        acc, n = {}, 0
        with torch.no_grad():
            for b in val_batches:
                _, m = distill_loss(model, b, args.tau, args.value_weight)
                for k, v in m.items():
                    acc[k] = acc.get(k, 0.0) + float(v) * len(b["self_elo"])
                n += len(b["self_elo"])
        out = {k: round(v / max(n, 1), 5) for k, v in acc.items()}
        if bench is not None:
            out["strength"] = eval_strength(model, bench, dist.device, args.elo)
        return out

    model = build("player", json.loads(args.model))
    step = fit(model, args, make_batch, loss_fn, evaluate if n_val else None, kind="player",
               extra_state=dict(distill=dict(elo=args.elo, tau=args.tau)))
    if step < args.steps:  # stopped by SIGTERM before the last step: not a finished run
        raise SystemExit(75)


def main(argv=None):
    from .runtime import add_runtime_args
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("make-bench")
    p.add_argument("--sf-labels", required=True)
    p.add_argument("--teacher-bench", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("train")
    p.add_argument("--parts", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--bench", default=None)
    p.add_argument("--model", default='{"d_model": 256, "n_layers": 8, "n_heads": 8}')
    p.add_argument("--elo", type=float, default=3000.0, help="constant Elo input during distillation")
    p.add_argument("--tau", type=float, default=1.0, help="teacher softmax temperature")
    p.add_argument("--value-weight", type=float, default=0.5)
    p.add_argument("--max-parts", type=int, default=None)
    p.add_argument("--val-positions", type=int, default=200_000)
    p.add_argument("--eval-positions", type=int, default=50_000)
    p.add_argument("--store-device", default="auto", help="auto | cpu | cuda (where the labelled data lives)")
    add_runtime_args(p)
    args = ap.parse_args(argv)
    (make_bench if args.cmd == "make-bench" else train)(args)


if __name__ == "__main__":
    main()
