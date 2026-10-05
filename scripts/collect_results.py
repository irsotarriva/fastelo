"""Copy the small result files of a set of runs into results/, the folder the report notebook reads.

    python scripts/collect_results.py --runs runs --out results

Checkpoints and datasets stay out of the repository; these are the measured numbers only.
"""
import argparse
import json
import os


def load(p):
    return json.load(open(p))


def save(obj, out, name):
    with open(os.path.join(out, name), "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=1)
    print(f"  {name} ({os.path.getsize(os.path.join(out, name)) / 1e3:.0f} kB)")


def log(path, train_every):
    """Validation rows, plus a thinned-out set of training rows."""
    rows = [json.loads(line) for line in open(path)]
    keep = [r for r in rows if r.get("split") == "val" or r["step"] % train_every == 0]
    drop = ("gpu_peak_gb", "nonfinite_batches", "loss_scale", "items_per_s", "grad_norm")
    return [{k: v for k, v in r.items() if k not in drop} for r in keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    R = lambda *p: os.path.join(args.runs, *p)
    os.makedirs(args.out, exist_ok=True)
    save(dict(distill=log(R("distill", "log.jsonl"), 2000), player=log(R("player", "log.jsonl"), 2000)),
         args.out, "training_logs.json")
    save(load(R("player", "eval_player", "summary.json")), args.out, "player_likelihood.json")
    save(load(R("player", "eval_strength", "summary.json")), args.out, "player_strength.json")
    cal = load(R("player", "calibration.json"))
    for k in ("player", "strong_player"):   # checkpoint paths
        cal.pop(k, None)
    save(cal, args.out, "calibration.json")
    save(load(R("player", "rating_response.json")), args.out, "rating_response.json")
    # results/rating_study.json (simulations behind the choice of the rating) and
    # results/synthetic_labels.json (fastelo.synth ladder) are copied by hand


if __name__ == "__main__":
    main()
