"""Held-out leak detector with a positive control (the check behind item 4/5 of docs/leakage_check.md).

    python scripts/experiments/heldout_leak_control.py            # ~30 s on CPU

Train a small model on a STREAM of fresh random batches, then measure the loss on a held-out random batch.
Random tokens carry no signal, so a causal model must stay at ln(vocab). The positive control removes the
causal mask on purpose: such a model can read the next token, learns to copy it, and collapses far below
ln(vocab). Seeing both proves the detector can actually find a leak.
Exploratory sanity check only: see scripts/experiments/README.md.
"""
import argparse
import math

try:                      # imported as a package (tests) ...
    from scripts.experiments import _common as C
except ImportError:       # ... or run directly from the repo root
    import _common as C

import torch  # noqa: E402

import models.attention as attention_mod  # noqa: E402
import models.diff_attention as diff_mod  # noqa: E402
from configs.model_config import default_config  # noqa: E402


def held_out_loss(arm, steps, leaky, device, vocab=64, seq_len=24, batch=32, lr=3e-3, seed=0):
    cfg = default_config(vocab_size=vocab, context_len=seq_len, d_model=64, n_layers=2, n_heads=4,
                         ffn_hidden=176, attn_impl="einsum")   # einsum path: honours the (patched) mask
    saved = (attention_mod.causal_mask, diff_mod.causal_mask)
    if leaky:  # positive control: every query may attend to every key (including the future)
        allow_all = lambda n, dev: torch.ones(n, n, dtype=torch.bool, device=dev)
        attention_mod.causal_mask = diff_mod.causal_mask = allow_all
    try:
        model = C.build_seeded(arm, cfg, seed, device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr)
        stream = torch.Generator().manual_seed(1)                                  # one stream, a new batch per step
        for _ in range(steps):
            ids = torch.randint(0, vocab, (batch, seq_len + 1), generator=stream).to(device)
            C.train_step(model, opt, ids[:, :-1], ids[:, 1:], vocab)
        hx, hy = C.random_batch(vocab, 64, seq_len, 999, device)                  # held-out batch
        return C.eval_loss(model, hx, hy, vocab), math.log(vocab)
    finally:
        attention_mod.causal_mask, diff_mod.causal_mask = saved


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", nargs="+", default=["T", "D", "P"])
    ap.add_argument("--steps", type=int, nargs="+", default=[100, 200])
    ap.add_argument("--device", default="cpu", help="cpu (default), mps, cuda or auto")
    args = ap.parse_args(argv)
    device = C.pick_device(args.device)
    rows = []
    for steps in args.steps:
        for arm in args.arms:
            for leaky in (False, True):
                loss, ln_v = held_out_loss(arm, steps, leaky, device)
                rows.append((steps, arm, leaky, loss, ln_v))
                print(f"steps={steps} {arm} causal mask {'REMOVED' if leaky else 'intact '}: "
                      f"held-out loss {loss:.3f}  (ln V = {ln_v:.3f})", flush=True)
    return rows


if __name__ == "__main__":
    main()
