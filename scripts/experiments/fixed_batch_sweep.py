"""Fixed-batch sweep: memorise ONE batch of random tokens; also watch a DIFFERENT fresh random batch.

    python scripts/experiments/fixed_batch_sweep.py --config tiny
    python scripts/experiments/fixed_batch_sweep.py --config default      # ~2 min on a laptop CPU

For every (learning rate, seed, arm) it trains 30 steps on one fixed batch and, every 5 steps, measures the loss
on that batch ("fixed") and on another random batch that is never trained on ("fresh"). Random tokens carry no
signal, so a causal model's fresh loss must stay near ln(vocab) (about 9.70); a leak would pull it far below.
Deterministic given --seeds (model init and data both derive from the seed). Exploratory sanity check only:
see scripts/experiments/README.md.
"""
import argparse
import json
import math
import time

try:                      # imported as a package (tests) ...
    from scripts.experiments import _common as C
except ImportError:       # ... or run directly from the repo root
    import _common as C

import torch  # noqa: E402


def run_one(arm, cfg, seed, lr, steps, batch, seq_len, every, device):
    V = cfg.vocab_size
    model = C.build_seeded(arm, cfg, seed, device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    x, y = C.random_batch(V, batch, seq_len, 1000 + seed, device)     # fixed training batch
    fx, fy = C.random_batch(V, batch, seq_len, 2000 + seed, device)   # fresh batch, never trained on
    rec = []                                                          # (step, fixed loss, fresh loss)
    for step in range(steps + 1):
        if step % every == 0:
            rec.append((step, C.eval_loss(model, x, y, V), C.eval_loss(model, fx, fy, V)))
        if step < steps:
            C.train_step(model, opt, x, y, V)
    return rec


def at(rec, step, col):
    return next(r for r in rec if r[0] == step)[col]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", choices=["tiny", "default"], default="tiny")
    ap.add_argument("--arms", nargs="+", default=["T", "D", "P"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--lrs", type=float, nargs="+", default=[1e-4, 3e-4])
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--device", default="cpu", help="cpu (default), mps, cuda or auto")
    ap.add_argument("--out", default=None, help="optional JSON file for the raw per-step numbers")
    args = ap.parse_args(argv)

    device, cfg = C.pick_device(args.device), C.make_config(args.config)
    ln_v = math.log(cfg.vocab_size)
    print(f"config={args.config} device={device} ln(vocab)={ln_v:.3f} batch={args.batch}x{args.seq_len} steps={args.steps}")
    runs, t0 = {}, time.time()
    for lr in args.lrs:
        for seed in args.seeds:
            for arm in args.arms:
                rec = run_one(arm, cfg, seed, lr, args.steps, args.batch, args.seq_len, args.eval_every, device)
                runs[f"{arm}|{lr}|{seed}"] = rec
                print(f"lr={lr:g} seed={seed} {arm}: fixed {rec[0][1]:.2f}->{rec[-1][1]:.2f}  "
                      f"fresh {rec[0][2]:.2f}->{rec[-1][2]:.2f}  ({time.time() - t0:.0f}s)", flush=True)

    s20, s_end = min(20, args.steps), args.steps
    print(f"\nSUMMARY (mean over seeds {args.seeds}); fresh columns are (fresh loss - ln V)")
    print(f"{'lr':>8} {'arm':<4}{f'fixed@{s20}':>10}{f'fixed@{s_end}':>10}{f'fresh@{s_end} min..max':>24}{'worst fresh (any eval)':>26}")
    for lr in args.lrs:
        for arm in args.arms:
            recs = [runs[f"{arm}|{lr}|{s}"] for s in args.seeds]
            mean = lambda xs: sum(xs) / len(xs)
            fresh_end = [at(r, s_end, 2) - ln_v for r in recs]
            worst = max(x[2] for r in recs for x in r) - ln_v
            print(f"{lr:>8g} {arm:<4}{mean([at(r, s20, 1) for r in recs]):>10.2f}{mean([at(r, s_end, 1) for r in recs]):>10.2f}"
                  f"{min(fresh_end):>+14.3f}..{max(fresh_end):+.3f}{worst:>+26.3f}")
    lowest = min(x[2] for r in runs.values() for x in r) - ln_v
    print(f"\nlowest fresh-batch loss over all runs and evals: ln V {lowest:+.3f} "
          f"({'never below ln V: no sign of leakage' if lowest > -0.05 else 'BELOW ln V: investigate!'})")
    if args.out:
        json.dump({"ln_v": ln_v, "runs": runs}, open(args.out, "w"))
    return runs


if __name__ == "__main__":
    main()
