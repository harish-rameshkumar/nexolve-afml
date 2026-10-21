"""Lambda ablation for D on the fixed-batch task: what makes D memorise faster?

    python scripts/experiments/lambda_ablation.py                  # default config, 2 seeds, 20 steps (~1 min on CPU)
    python scripts/experiments/lambda_ablation.py --config tiny

Variants (all trained on one fixed batch of random tokens, final loss on that batch reported):
  T         standard attention
  D         DIFF V2 as designed (lambda = sigmoid(Linear(x)), learned, about 0.5 at init)
  D-lam0    D with lambda frozen near 0: the second attention map is ignored, so it behaves like plain attention
  D-lam1    D with lambda frozen near 1: the second map is subtracted (almost) fully
Frozen lambda replaces lambda_proj by a constant logit of -30 / +30 (sigmoid -> ~0 / ~1) with no parameters.
Exploratory sanity check only: see scripts/experiments/README.md.
"""
import argparse

try:                      # imported as a package (tests) ...
    from scripts.experiments import _common as C
except ImportError:       # ... or run directly from the repo root
    import _common as C

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

FROZEN_LOGIT = {"D-lam0": -30.0, "D-lam1": 30.0}


class FrozenLambdaLogit(nn.Module):
    """Stands in for lambda_proj: returns the same constant logit for every token and head, no parameters."""

    def __init__(self, n_heads: int, value: float):
        super().__init__()
        self.n_heads, self.value = n_heads, value

    def forward(self, x):                                   # (B, T, d) -> (B, T, H)
        return torch.full((*x.shape[:-1], self.n_heads), self.value, dtype=x.dtype, device=x.device)


def run_variant(variant, cfg, seed, lr, steps, batch, seq_len, device):
    V = cfg.vocab_size
    model = C.build_seeded("T" if variant == "T" else "D", cfg, seed, device)   # same init draws as the sweep
    if variant in FROZEN_LOGIT:
        for blk in model.blocks:
            blk.attn.lambda_proj = FrozenLambdaLogit(cfg.n_heads, FROZEN_LOGIT[variant])
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    x, y = C.random_batch(V, batch, seq_len, 1000 + seed, device)
    for _ in range(steps):
        C.train_step(model, opt, x, y, V)
    return C.eval_loss(model, x, y, V)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", choices=["tiny", "default"], default="default")
    ap.add_argument("--variants", nargs="+", default=["T", "D", "D-lam0", "D-lam1"],
                    choices=["T", "D", "D-lam0", "D-lam1"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--device", default="cpu", help="cpu (default), mps, cuda or auto")
    args = ap.parse_args(argv)

    device, cfg = C.pick_device(args.device), C.make_config(args.config)
    print(f"config={args.config} device={device} lr={args.lr:g} steps={args.steps} batch={args.batch}x{args.seq_len}")
    print(f"fixed-batch loss after {args.steps} steps, per seed {args.seeds}:")
    results = {}
    for v in args.variants:
        results[v] = [run_variant(v, cfg, s, args.lr, args.steps, args.batch, args.seq_len, device) for s in args.seeds]
        print(f"  {v:<7}{[round(x, 2) for x in results[v]]}", flush=True)
    return results


if __name__ == "__main__":
    main()
