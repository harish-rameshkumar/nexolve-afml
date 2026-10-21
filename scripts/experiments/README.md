# Exploratory experiments (sanity checks, not results)

> **Read this first.** These scripts run on **random tokens**. Memorising one fixed batch of random tokens is **not**
> instruction learning, and the effect they show (D fits a fixed batch faster than T and P) **disappears at the tiny size**.
> The explanation we suggest for *why* subtraction speeds up memorisation is an **untested hypothesis**.
> They exist to rule out leakage/bugs and to describe a curiosity, not to support the project's research question.
> Background and conclusions: [docs/leakage_check.md](../../docs/leakage_check.md).

All commands are run from the repo root. Every script has `--help`, runs on `--device cpu` (default), `mps`, `cuda` or `auto`,
and is deterministic given its seeds (model initialisation and data both derive from the seed; data is drawn on the CPU, so
the tokens are identical on every device. On CPU the numbers reproduce bit for bit; on mps/cuda they can differ in the last digits).

| Script | What it does | Reproduces (in docs/leakage_check.md) | Command |
|---|---|---|---|
| `fixed_batch_sweep.py` | For each (lr, seed, arm): 30 AdamW steps on ONE fixed random batch; every 5 steps also the loss on a different fresh random batch. Prints a summary (mean fixed-batch loss at steps 20/30, fresh loss minus ln V) and the lowest fresh loss seen. | Fixed-batch table, **tiny** rows, and the "fresh-batch loss never drops below ln V" statement | `python scripts/experiments/fixed_batch_sweep.py --config tiny` (~15 s) |
| `fixed_batch_sweep.py` | Same, with the default (about 28M-parameter) config | Fixed-batch table, **default** rows | `python scripts/experiments/fixed_batch_sweep.py --config default` (~2 min on a laptop CPU) |
| `lambda_ablation.py` | Fixed-batch loss after 20 steps for T, D as designed, D with lambda frozen near 0 (second map ignored) and D with lambda frozen near 1 (full subtraction); 2 seeds, lr 3e-4. | "What explains the speed difference" paragraph | `python scripts/experiments/lambda_ablation.py` (~30 s on CPU) |
| `heldout_leak_control.py` | Train on a stream of fresh random batches and evaluate on a held-out random batch, with and without the causal mask. A causal model stays at ln V; a model whose mask was removed collapses. | Positive control: "ln(V) = 4.16 to about 0.2-1.0" | `python scripts/experiments/heldout_leak_control.py` (~30 s on CPU) |

Useful options: `--seeds 0 1 2`, `--lrs 1e-4 3e-4`, `--steps`, `--batch`, `--seq-len`, `--arms T D P`, `--out results.json` (sweep only, raw per-step numbers).

## What to expect (CPU, default settings)

- `fixed_batch_sweep.py --config default`: fixed-batch loss at step 20, lr 3e-4: T 0.76, D 0.22, P 0.66 (lr 1e-4: 1.62 / 0.65 / 1.39). Fresh loss minus ln V between +0.04 and +0.15, never below 0.
- `fixed_batch_sweep.py --config tiny`: all three arms within 0.02 of each other; fresh loss minus ln V between +0.004 and +0.032.
- `lambda_ablation.py`: T [0.83, 0.75], D [0.23, 0.23], D-lam0 [0.69, 0.69], D-lam1 [0.18, 0.18] (one value per seed).
- `heldout_leak_control.py`: mask intact stays at 4.15-4.17 (ln V = 4.159); mask removed 0.19-0.99.

The tables in the note are means over seeds rounded by hand, so a value printed by the script can differ from the note by 0.01.

## Why there is a held-out script as well as the fresh-batch check

The fresh-batch check inside the sweep is **too weak to prove "no leak"** on its own: with the causal mask removed, a model trained only
30 steps on one batch still has a fresh loss near ln V (it has not yet learned the copy shortcut). The held-out script trains long enough on
varied batches for a leaky model to learn it, so it can detect a leak; the same check is in `tests/test_no_leakage.py`.
