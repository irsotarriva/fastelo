"""Training runtime shared by the player and the distillation: any number of GPUs of any kind, and
runs that can be stopped and resumed.

Launch with `torchrun --nproc_per_node=<gpus> -m fastelo.<trainer> ...` (or plain `python -m` for
one device). Every rank loads only its own share of the data shards (shards are random subsets of
games), samples with a generator seeded by (seed, step, rank, micro-step) so a resumed run needs no
sampler state, and uses the largest micro-batch that fits on the *smallest* card, with gradient
accumulation keeping the global batch fixed whatever the hardware.

Checkpoints are written every `--ckpt-minutes` and uploaded in the background to `--remote`
(s3://bucket or file://dir); a fresh machine restores the latest one. SIGTERM
saves and uploads before exiting.
"""
import json
import math
import os
import signal
import time
from contextlib import nullcontext

import torch
import torch.distributed as torch_dist

from .cloud import BackgroundUploader, open_store


def add_runtime_args(p):
    g = p.add_argument_group("runtime")
    g.add_argument("--steps", type=int, default=3000, help="optimizer steps")
    g.add_argument("--batch", type=int, default=1024, help="global batch (positions or games) per optimizer step")
    g.add_argument("--micro-batch", default="auto", help="per-GPU micro-batch; 'auto' = largest that fits")
    g.add_argument("--micro-batch-max", type=int, default=8192)
    g.add_argument("--lr", type=float, default=1e-3)
    g.add_argument("--min-lr-frac", type=float, default=0.1)
    g.add_argument("--warmup", type=int, default=200)
    g.add_argument("--wd", type=float, default=0.05)
    g.add_argument("--optimizer", choices=["adamw", "muon"], default="adamw",
                   help="muon = Muon on transformer-block matrices + AdamW on the rest")
    g.add_argument("--clip", type=float, default=1.0)
    g.add_argument("--amp", choices=["auto", "bf16", "fp16", "none"], default="auto",
                   help="auto = bf16 where supported, else fp16 on GPU, none on CPU")
    g.add_argument("--compile", action="store_true")
    g.add_argument("--max-bad-frac", type=float, default=0.01)
    g.add_argument("--log-every", type=int, default=50)
    g.add_argument("--eval-every", type=int, default=500)
    g.add_argument("--ckpt-every", type=int, default=1000, help="also checkpoint every N steps")
    g.add_argument("--ckpt-minutes", type=float, default=10.0)
    g.add_argument("--run", required=True)
    g.add_argument("--runs-dir", default="runs")
    g.add_argument("--remote", default=None, help="s3://bucket or file://dir for checkpoints and logs")
    g.add_argument("--init-from", default=None, help="start from these model weights (new optimizer)")
    g.add_argument("--data-ram-gb", type=float, default=None,
                   help="host RAM for training data, all ranks together (default 45%% of what the box has)")
    g.add_argument("--threads", type=int, default=8)
    g.add_argument("--seed", type=int, default=0)


# ------------------------------------------------------------------------------ devices
class Dist:
    def __init__(self):
        self.world = int(os.environ.get("WORLD_SIZE", "1"))
        self.rank = int(os.environ.get("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device("cuda", self.local_rank)
        else:
            self.device = torch.device("cpu")
        if self.world > 1 and not torch_dist.is_initialized():
            torch_dist.init_process_group("nccl" if self.device.type == "cuda" else "gloo")
        self.main = self.rank == 0

    def all_min(self, x):
        return int(self._reduce(x, torch_dist.ReduceOp.MIN))

    def all_max(self, x):
        return float(self._reduce(x, torch_dist.ReduceOp.MAX))

    def _reduce(self, x, op):
        if self.world == 1:
            return x
        t = torch.tensor([float(x)], device=self.device)
        torch_dist.all_reduce(t, op=op)
        return t.item()

    def barrier(self):
        if self.world > 1:
            torch_dist.barrier()

    def close(self):
        if self.world > 1 and torch_dist.is_initialized():
            torch_dist.destroy_process_group()


def shard_subset(items, dist):
    """This rank's share of a list of data shards/parts (round robin)."""
    mine = items[dist.rank::dist.world]
    if not mine:
        raise ValueError(f"{len(items)} data shards cannot feed {dist.world} ranks")
    return mine


def host_ram_bytes():
    """RAM this job may use: the container limit if there is one, else the machine's total."""
    found = []
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            v = open(path).read().strip()
            if v.isdigit() and int(v) < 1 << 50:  # "max" / a huge number mean no limit
                found.append(int(v))
        except OSError:
            pass
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemTotal:"):
                found.append(int(line.split()[1]) * 1024)
    except OSError:
        pass
    return min(found) if found else 16 << 30


def plan_data(units, nbytes, dist, store_device="auto", ram_gb=None):
    """Where this rank's data units (shards/parts, `nbytes` each in memory) live, and how they are
    grouped so that one group fits. Returns (device, groups).

    auto: all of them on the training GPU if that takes < 35% of the card; otherwise host RAM, in
    groups of at most 45% of the usable RAM (`ram_gb` overrides the detected total) divided by the
    number of ranks. A box with little RAM then trains on one group at a time (see Rotating)."""
    if store_device == "auto":
        on_gpu = (dist.device.type == "cuda"
                  and sum(nbytes) < 0.35 * torch.cuda.get_device_properties(dist.device).total_memory)
        store_device = str(dist.device) if on_gpu else "cpu"
    if store_device != "cpu":
        return store_device, [list(units)]
    budget = (ram_gb * 1e9 if ram_gb else 0.45 * host_ram_bytes()) / dist.world
    groups, cur, size = [], [], 0
    for u, b in zip(units, nbytes):
        if cur and size + b > budget:
            groups.append(cur)
            cur, size = [], 0
        cur.append(u)
        size += b
    groups.append(cur)
    return "cpu", groups


class Rotating:
    """Keeps one group of data units in memory. `at(step)` returns the store for that step and
    swaps in the next group once the current one has been sampled about once, so the group is a
    function of the step alone and a resumed run needs no extra state."""

    def __init__(self, groups, sizes, load, per_step, log=None):
        """sizes[i]: samples (positions or games) in group i; per_step: samples this rank draws per
        optimizer step; load(units) -> store."""
        self.groups, self.load, self.log = groups, load, log
        ends, end = [], 0
        for s in sizes:
            end += max(1, round(s / per_step))
            ends.append(end)
        self.ends, self.current, self.store = ends, None, None

    def group_at(self, step):
        s = step % self.ends[-1]
        return next(i for i, e in enumerate(self.ends) if s < e)

    def at(self, step):
        g = self.group_at(step)
        if g != self.current:
            self.store = None  # release the old group before loading the next
            t0 = time.time()
            self.store, self.current = self.load(self.groups[g]), g
            if self.log and len(self.groups) > 1:
                self.log(f"step {step}: data group {g + 1}/{len(self.groups)} "
                         f"({len(self.groups[g])} units) loaded in {time.time() - t0:.0f}s")
        return self.store


def amp_dtype(args, device):
    if args.amp == "none" or (args.amp == "auto" and device.type != "cuda"):
        return None
    if args.amp == "auto":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return {"bf16": torch.bfloat16, "fp16": torch.float16}[args.amp]


def lr_at(step, args):
    if step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    t = min(1.0, (step - args.warmup) / max(1, args.steps - args.warmup))
    return args.lr * (args.min_lr_frac + (1 - args.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * t)))


def make_optimizers(model, args):
    """List of optimizers. Weight decay only on Linear / attention projection weights: embeddings
    (pieces, squares, Elo knots, ...) must not decay, or rarely seen rows are pulled back to zero.

    --optimizer muon: Muon (Newton-Schulz orthogonalised momentum, lr scaled to match AdamW's
    update RMS so the same schedule applies) on the 2D weight matrices *inside the transformer
    blocks*; AdamW on everything else (embeddings, Elo knots, policy/value heads, norms, biases),
    which is the usual split."""
    linear = {id(m.weight) for m in model.modules() if isinstance(m, torch.nn.Linear)}
    linear |= {id(p) for n, p in model.named_parameters() if n.endswith("in_proj_weight")}
    muon = set()
    if getattr(args, "optimizer", "adamw") == "muon":
        muon = {id(p) for n, p in model.named_parameters()
                if p.ndim == 2 and id(p) in linear and ".layers." in n}
    decay = [p for p in model.parameters() if id(p) in linear and id(p) not in muon]
    no_decay = [p for p in model.parameters() if id(p) not in linear]
    opts = [torch.optim.AdamW([{"params": decay, "weight_decay": args.wd},
                               {"params": no_decay, "weight_decay": 0.0}], lr=args.lr, betas=(0.9, 0.95))]
    if muon:
        opts.append(torch.optim.Muon([p for p in model.parameters() if id(p) in muon], lr=args.lr,
                                     weight_decay=args.wd, adjust_lr_fn="match_rms_adamw"))
    return opts


def make_optimizer(model, args):
    """Single AdamW (older call sites)."""
    return make_optimizers(model, argparse_namespace(args, optimizer="adamw"))[0]


def argparse_namespace(args, **override):
    import argparse
    return argparse.Namespace(**{**vars(args), **override})


# --------------------------------------------------------------------------- checkpoints
class Checkpointer:
    """Local checkpoint + log, mirrored to remote storage in the background."""

    def __init__(self, args, dist):
        self.dir = os.path.join(args.runs_dir, args.run)
        self.path = os.path.join(self.dir, "ckpt.pt")
        self.log_path = os.path.join(self.dir, "log.jsonl")
        self.dist = dist
        self.store = open_store(args.remote) if args.remote else None
        self.prefix = f"runs/{args.run}"
        self.uploader = BackgroundUploader(self.store) if (self.store and dist.main) else None
        self.last_save = time.time()
        os.makedirs(self.dir, exist_ok=True)

    def restore(self):
        """Latest checkpoint (local, else remote) as a dict, or None. Every rank reads the same file."""
        if self.store and self.dist.main and not os.path.exists(self.path):
            for name in ("ckpt.pt", "log.jsonl"):
                key = f"{self.prefix}/{name}"
                if self.store.exists(key):
                    self.store.get_file(key, os.path.join(self.dir, name))
        self.dist.barrier()
        if os.path.exists(self.path):
            return torch.load(self.path, map_location="cpu", weights_only=False)
        return None

    def log(self, rec):
        if not self.dist.main:
            return
        print(json.dumps(rec), flush=True)
        with open(self.log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def due(self, step, args):
        return step % args.ckpt_every == 0 or time.time() - self.last_save > 60 * args.ckpt_minutes

    def save(self, state):
        self.last_save = time.time()
        if not self.dist.main:
            return
        tmp = self.path + ".tmp"
        torch.save(state, tmp)
        os.replace(tmp, self.path)
        if self.uploader:
            self.uploader.submit(self.path, f"{self.prefix}/ckpt.pt")
            self.uploader.submit(self.log_path, f"{self.prefix}/log.jsonl")

    def upload_file(self, local, name):
        if self.uploader:
            self.uploader.submit(local, f"{self.prefix}/{name}")

    def flush(self):
        if self.uploader:
            self.uploader.flush()


# -------------------------------------------------------------------------------- loop
def _is_oom(e):
    return isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()


def find_micro_batch(model, make_batch, loss_fn, args, dist, autocast):
    """Largest per-GPU micro-batch (power of two) whose forward+backward fits, agreed across ranks."""
    cap = min(args.micro_batch_max, max(1, args.batch // dist.world))
    if args.micro_batch != "auto":
        size = min(int(args.micro_batch), cap)
    elif dist.device.type != "cuda":
        size = min(cap, 512)
    else:
        size = 1 << (cap.bit_length() - 1) if cap & (cap - 1) else cap
        while True:
            try:
                b = make_batch(0, 0, size)
                with autocast():
                    loss, _ = loss_fn(model, b)
                loss.backward()
                model.zero_grad(set_to_none=True)
                del b, loss
                torch.cuda.synchronize()
                break
            except RuntimeError as e:
                if not _is_oom(e) or size <= 8:
                    raise
                model.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                size //= 2
        # headroom: activations vary with the number of legal moves / game lengths
        size = max(8, int(size * 0.85))
        torch.cuda.empty_cache()
    return dist.all_min(size)


def fit(model, args, make_batch, loss_fn, evaluate=None, kind="player", extra_state=None):
    """Train `model`. make_batch(step, micro, size) -> dict on the device; loss_fn(model, batch) ->
    (loss, metrics); evaluate(model) -> dict (main rank only)."""
    dist = Dist()
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    ck = Checkpointer(args, dist)
    state = ck.restore()
    model.to(dist.device)
    raw_model = model
    opts = make_optimizers(model, args)
    step = 0
    if state:
        model.load_state_dict(state["model"])
        for o, s_ in zip(opts, state["opt"]):
            o.load_state_dict(s_)
        step = state["step"]
        if dist.main:
            print(f"resumed from step {step}", flush=True)
    elif args.init_from:
        init = torch.load(args.init_from, map_location="cpu", weights_only=False)
        model.load_state_dict(init["model"])
        if dist.main:
            print(f"initialised from {args.init_from} (step {init.get('step')})", flush=True)
    dtype = amp_dtype(args, dist.device)
    autocast = lambda: torch.autocast(dist.device.type, dtype=dtype or torch.float32, enabled=dtype is not None)
    scaler = torch.amp.GradScaler(enabled=dtype == torch.float16)
    if state and state.get("scaler") and scaler.is_enabled():
        scaler.load_state_dict(state["scaler"])

    micro = find_micro_batch(model, make_batch, loss_fn, args, dist, autocast)
    accum = max(1, math.ceil(args.batch / (micro * dist.world)))
    # shrink the micro-batch so that micro x accum x world is the requested global batch, not more
    micro = max(1, math.ceil(args.batch / (accum * dist.world)))
    if dist.world > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[dist.local_rank] if dist.device.type == "cuda" else None)
    if args.compile:
        model = torch.compile(model)
    if dist.main:
        n_params = sum(p.numel() for p in raw_model.parameters())
        print(f"{kind}: {n_params / 1e6:.2f}M params | {dist.world} x {dist.device.type} "
              f"({torch.cuda.get_device_name(dist.device) if dist.device.type == 'cuda' else 'cpu'}) | "
              f"micro-batch {micro} x accum {accum} x world {dist.world} = {micro * accum * dist.world} | amp {dtype}",
              flush=True)

    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))

    def save():
        ck.save(dict(model=raw_model.state_dict(), opt=[o.state_dict() for o in opts], step=step, kind=kind,
                     model_cfg=dict(raw_model.cfg.__dict__), scaler=scaler.state_dict() if scaler.is_enabled() else None,
                     args=vars(args), **(extra_state or {})))

    acc, n_good, n_bad, n_items, t_last, gnorm, n_dumped = {}, 0, 0, 0, time.time(), 0.0, 0
    model.train()
    while step < args.steps:
        # SIGTERM may reach only some ranks: stop together, or the others hang in the next all-reduce
        if dist.all_max(1.0 if stop["flag"] else 0.0) > 0:
            stop["flag"] = True
            break
        for o in opts:
            for g in o.param_groups:
                g["lr"] = lr_at(step, args)
            o.zero_grad(set_to_none=True)
        bad = False
        metrics_step = []
        for a in range(accum):
            b = make_batch(step, a, micro)
            sync = nullcontext() if (a == accum - 1 or dist.world == 1) else model.no_sync()
            with sync:
                with autocast():
                    loss, m = loss_fn(model, b)
                # every rank must take the same path, or the gradient all-reduce deadlocks
                if dist.all_max(0.0 if torch.isfinite(loss) else 1.0) > 0:
                    bad = True
                    if not torch.isfinite(loss) and n_dumped < 3:  # keep a few for offline replay
                        torch.save({k: v.cpu() if torch.is_tensor(v) else v for k, v in b.items()},
                                   os.path.join(ck.dir, f"nonfinite_batch_{step}_rank{dist.rank}.pt"))
                        n_dumped += 1
                    break
                scaler.scale(loss / accum).backward()
            metrics_step.append(m)
            n_items += micro * dist.world
        step += 1
        if bad:
            n_bad += 1
            for o in opts:
                o.zero_grad(set_to_none=True)
        else:
            for o in opts:
                scaler.unscale_(o)
            gnorm = float(torch.nn.utils.clip_grad_norm_(raw_model.parameters(), args.clip))
            for o in opts:
                scaler.step(o)
            scaler.update()
            n_good += 1
            for m in metrics_step:
                for k, v in m.items():
                    acc[k] = acc.get(k, 0.0) + float(v) / len(metrics_step)
        if step % args.log_every == 0:
            dt = time.time() - t_last
            rec = dict(step=step, split="train", lr=lr_at(step, args), grad_norm=round(gnorm, 3),
                       items_per_s=round(n_items / dt), **{k: round(v / max(n_good, 1), 4) for k, v in acc.items()},
                       nonfinite_batches=n_bad, loss_scale=scaler.get_scale() if scaler.is_enabled() else None)
            if dist.device.type == "cuda":
                rec["gpu_peak_gb"] = round(torch.cuda.max_memory_allocated(dist.device) / 2**30, 2)
            ck.log(rec)
            if n_bad > args.max_bad_frac * args.log_every or (scaler.is_enabled() and scaler.get_scale() < 1.0):
                save()
                ck.flush()
                raise RuntimeError(f"numerically unstable: {n_bad}/{args.log_every} non-finite steps")
            acc, n_good, n_bad, n_items, t_last = {}, 0, 0, 0, time.time()
        if step % args.eval_every == 0 or step == args.steps:
            # decided identically on every rank (only the main rank has an evaluate function), so the
            # barrier below always has all ranks
            if dist.main and evaluate is not None:
                raw_model.eval()
                ck.log(dict(step=step, split="val", **evaluate(raw_model)))
                raw_model.train()
            dist.barrier()
        if ck.due(step, args) or step == args.steps:
            save()
    if stop["flag"]:
        save()
        if dist.main:
            print(f"SIGTERM: saved step {step}, uploading before exit", flush=True)
    ck.flush()
    dist.close()
    return step
