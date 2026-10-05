"""Train the player on human games (the "human/Elo phase").

    python -m fastelo.train --data data/rapid_v2 --run player_human \
        --init-from runs/distill/ckpt.pt --steps 20000 --batch 4096
    torchrun --nproc_per_node=4 -m fastelo.train --data data/rapid_v2 --run player_human ...

Runs on fastelo.runtime: any number of GPUs, automatic micro-batch, checkpoints mirrored to
--remote and resumed after a stop. Each rank loads its own share of the dataset shards.
Model hyperparameters: --model '{"d_model": 256, "n_layers": 8}' (ignored when resuming or
starting from --init-from, which carry their own config).
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from .features import REC, GameStore
from .models import build
from .runtime import (Dist, Rotating, add_runtime_args, fit, lr_at, make_optimizer,  # noqa: F401 (re-exported)
                      plan_data, shard_subset)

ELO_BINS = [0, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 4000]


def get_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="{}", help="JSON overrides for PlayerConfig")
    ap.add_argument("--elo-balance", type=float, default=0.0,
                    help="oversample rare ratings, sampling ~ p(elo)^(1-alpha): 0 natural mix, 1 flat over Elo")
    ap.add_argument("--eval-size", type=int, default=50000, help="validation positions")
    ap.add_argument("--max-shards", type=int, default=None, help="use only the first k train shards")
    ap.add_argument("--max-positions", type=int, default=None, help="cap on train positions (per rank)")
    ap.add_argument("--store-device", default="auto", help="auto | cpu | cuda: where the dataset tensors live")
    ap.add_argument("--train-split", default="train", help="split to train on ('val' for throughput benchmarks)")
    add_runtime_args(ap)
    return ap.parse_args(argv)


@torch.no_grad()
def evaluate_player(model, store, idx, device):
    sums = dict(policy_loss=0.0, top1=0.0, n=0)
    bins = np.zeros((len(ELO_BINS) - 1, 3))  # loss sum, top1 sum, count
    for s in range(0, len(idx), 4096):
        b = {k: v.to(device) for k, v in store.batch(idx[s:s + 4096]).items()}
        logits, _ = model(b)
        nll = torch.nn.functional.cross_entropy(logits.float(), b["move"], reduction="none")
        hit = (logits.argmax(-1) == b["move"]).float()
        sums["policy_loss"] += nll.sum().item()
        sums["top1"] += hit.sum().item()
        sums["n"] += len(nll)
        which = np.digitize(b["self_elo"].cpu().numpy(), ELO_BINS) - 1
        np.add.at(bins, (which, 0), nll.cpu().numpy())
        np.add.at(bins, (which, 1), hit.cpu().numpy())
        np.add.at(bins, (which, 2), 1)
    out = {k: v / sums["n"] for k, v in sums.items() if k != "n"}
    out["top1_by_elo"] = {f"{ELO_BINS[i]}-{ELO_BINS[i + 1]}": round(bins[i, 1] / bins[i, 2], 4)
                          for i in range(len(bins)) if bins[i, 2]}
    return out


def main(argv=None):
    args = get_args(argv)
    dist = Dist()
    t0 = time.time()
    def open_data(root):
        """This rank's shards of `root`, grouped so one group fits in memory (swapped when used up)."""
        dirs = shard_subset(GameStore.shard_dirs(root, args.train_split, args.max_shards), dist)
        nbytes = [os.path.getsize(os.path.join(d, "positions.npy")) for d in dirs]
        count = {d: n // REC for d, n in zip(dirs, nbytes)}
        ram = args.data_ram_gb or None
        where, groups = plan_data(dirs, nbytes, dist, args.store_device, ram)

        def load(group):
            s = GameStore(root, args.train_split, where, max_positions=args.max_positions, dirs=group)
            s.elo_weights = s.elo_balance_weights(args.elo_balance) if args.elo_balance > 0 else None
            return s

        rot = Rotating(groups, [sum(count[d] for d in g) for g in groups], load, args.batch / dist.world,
                       log=(lambda m: print(m, flush=True)) if dist.main else None)
        first = rot.at(0)
        if dist.main:
            print(f"rank 0: {root}: {len(dirs)} shards in {len(groups)} group(s) on {where}; first group "
                  f"{first.n_positions:,} positions / {first.n_games:,} games", flush=True)
        return rot

    data = open_data(args.data)
    val = GameStore(args.data, "val", "cpu") if dist.main else None
    if dist.main:
        print(f"val {val.n_positions:,} positions (loaded in {time.time() - t0:.0f}s)", flush=True)

    # model config: --init-from / resume carry their own; else --model
    overrides = json.loads(args.model)
    cfg = overrides
    src = os.path.join(args.runs_dir, args.run, "ckpt.pt")
    for path in (src, args.init_from):
        if path and os.path.exists(path):
            cfg = torch.load(path, map_location="cpu", weights_only=False)["model_cfg"]
            # training-only settings may change on top of an inherited architecture
            cfg.update({k: v for k, v in overrides.items() if k in ("elo_smooth", "value_weight", "dropout")})
            break
    model = build("player", cfg)

    def gen_for(step, micro):
        return torch.Generator().manual_seed(((args.seed * 1_000_003 + step) * 64 + micro) * 1024 + dist.rank)

    def to_dev(b):
        return {k: (v.to(dist.device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}

    def make_batch(step, micro, size):
        train = data.at(step)
        return to_dev(train.batch(train.sample_player_indices(size, gen_for(step, micro), train.elo_weights)))

    evaluate = None
    if dist.main:
        eval_gen = torch.Generator().manual_seed(1234)
        all_val = val.player_indices()
        eval_idx = all_val[torch.randperm(len(all_val), generator=eval_gen)[:args.eval_size]]
        evaluate = lambda m: evaluate_player(m, val, eval_idx, dist.device)

    # forward through whatever wrapper fit() applies (DDP / compile) so gradients are synchronised
    step = fit(model, args, make_batch, lambda m, b: model.loss_from_outputs(m(b), b), evaluate, kind="player")
    if step < args.steps:  # stopped by SIGTERM before the last step: not a finished run
        raise SystemExit(75)


if __name__ == "__main__":
    main()
