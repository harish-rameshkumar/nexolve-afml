# WS2: data and pretraining

One recipe for T, D and P. Only `--arm` and `--seed` change.

## Files

| File | What it does |
|---|---|
| `pretraining/prepare_data.py` | trains the tokenizer, builds `train.bin` / `val.bin` / `meta.json` (run once) |
| `pretraining/tokenizer.py` | byte-level BPE, vocab 16,384, reserved chat tokens |
| `pretraining/dataset.py` | seeded, step-addressable batches; frozen val loader |
| `pretraining/evaluate.py` | `val_loss(model, val, device)`: reused by WS3 for the post-SFT retention check |
| `pretraining/train.py` | the loop (fp16 + GradScaler, resume, checkpoints, logging) |
| `pretraining/config.py` | `PretrainConfig`: every hyper-parameter in one place |
| `scripts/plot_pretrain_curves.py` | loss curves + summary tables (the WS2 hand-off figures) |
| `tests/test_pretraining.py` | 19 CPU tests incl. crash-and-resume equivalence |

## Run it

```bash
pip install -r requirements.txt && pip install -r requirements-ws2.txt

# 1. ONCE, on one machine. Then upload data/pretrain/ to the team drive / a Kaggle dataset.
python -m pretraining.prepare_data --out data/pretrain

# 2. Per seed, per arm (same command again resumes after a crash; a finished run is skipped)
for arm in T D P; do python -m pretraining.train --arm $arm --seed 0; done

# 3. Figures + tables
python scripts/plot_pretrain_curves.py --runs outputs/pretrain --out outputs/pretrain_report
```
Kaggle with 2 GPUs: run two arms at once, `CUDA_VISIBLE_DEVICES=0 ... --arm T` and `CUDA_VISIBLE_DEVICES=1 ... --arm D`.
Every account must read byte-identical data: `meta.json` holds sizes + sha256; `train.py` checks sizes on start.

## The recipe (all arms)

| | |
|---|---|
| tokens | 4,600 steps x 65,536 = 301.5M (batch 64 seq x 1024; micro-batch 16 x accum 4) |
| optimiser | AdamW, betas (0.9, 0.95), wd 0.1 on >=2-D weights only (not norms / embedding), clip 1.0 |
| LR | peak 2e-3, 200 warm-up steps, cosine to 2e-4 |
| precision | fp16 autocast + GradScaler (fp32 master weights, fp32 residual stream and loss) |
| data | FineWeb-Edu `sample-10BT`, docs joined by `<|endoftext|>`, no cross-document mask |
| split | doc index % 100 == 0 -> val candidate (2M tokens kept); rest -> train (310M tokens built) |
| val | full frozen `val.bin` (2M tokens) every 250 steps |

**Seed** controls model init AND data order. Same seed = same batches for T, D and P (the "paired" in paired seeds).
Different seeds = different order of the same 310M-token pool.

## Outputs (per run, `outputs/pretrain/<arm>_s<seed>/`)

`final.pt` (fp32, loads with `models.load_checkpoint`; **the WS3 hand-off**), `step00250.pt ... step04500.pt`
(fp16 weights, for the loss-matched analysis; each row of `metrics.jsonl` with `type == "val"` gives the loss at that step),
`metrics.jsonl`, `run_config.json`, `done.json`. Disk: about 1.2 GB per run.

Logged: val loss, train loss, LR, grad norm, loss scale, tokens/s, estimated training FLOPs (PaLM convention
`6N + 12*L*H_q*hd*T`; D has 2x query heads), train GPU-hours and total GPU-hours (incl. eval/ckpt).

## GPU budget (from WS1's T4 speeds, 301.5M tokens per run)

| arm | tokens/s | hours/run |
|---|---|---|
| T | 48k | 1.74 |
| D | 37k | 2.26 |
| P | 45k | 1.86 |
| **one seed** | | **5.9** (+~0.3 eval/ckpt) |
| **3 seeds** | | **~18.5** |

Order runs seed-major (T, D, P for seed 0, then seed 1, ...): if the budget runs out you still have complete paired seeds (2 minimum).
Check your Kaggle/Colab quotas against ~18.5 h; a 2-GPU Kaggle session can do two arms at once.


## Results (measured, 2 seeds per arm)

Final frozen validation loss (4,600 steps, 301.5M tokens, LR 2e-3, one T4 GPU per run):

| Arm | Seed 0 / seed 1 val loss | Mean val loss | Training GPU-hours | Mean tokens/s | Training FLOPs |
|---|---|---|---|---|---|
| T | 3.5225 / 3.5191 | 3.5208 | 1.81 / 1.87 | 46.3k / 44.7k | 6.14e16 |
| D | 3.5152 / 3.5209 | 3.5181 | 2.34 / 2.24 | 35.8k / 37.4k | 7.57e16 |
| P | 3.5193 / 3.5058 | 3.5125 | 2.00 / 2.10 | 42.0k / 39.9k | 6.44e16 |

All six runs finished. Training time totals 12.36 GPU-hours, plus a few percent for evaluation and checkpoint saving (`total_gpu_h` in each `done.json`). The two seeds of one arm differ by up to 0.0135, as much as the gaps between arm means, so the arms ended at the same pretraining loss to within seed noise.

Five runs were trained on Google Colab and P seed 1 on Kaggle, using byte-identical data (SHA-256 of `train.bin`, `val.bin` and `tokenizer.json` matches `meta.json`). Figures and tables: `scripts/plot_pretrain_curves.py` writes `val_loss_vs_tokens.png`, `train_loss_vs_tokens.png`, `pretrain_summary.csv` and `pretrain_arm_means.csv`. Checkpoints are on the team drive under `nexolve/outputs/<arm>_s<seed>/`. Build the `final.pt` path from the folder name: the `final_checkpoint` field in `P_s1/done.json` holds a Kaggle path.



## For the other workstreams

* **WS3:** chat tokens are reserved with fixed ids: `<|endoftext|>`=0 (also end-of-response), `<|pad|>`=1, `<|system|>`=2,
  `<|user|>`=3, `<|assistant|>`=4, `<|end|>`=5, two spare. Pretraining never trains them. Use `pretraining.evaluate.val_loss`
  for the retention check.
* **WS4:** `val.bin` is the frozen pretraining split. Checkpoints every 250 steps carry their val loss in `extra["val_loss"]`.

## Decisions to confirm before the main runs

1. **LR 2e-3**, chosen from 500-step pilots on T (final val loss 4.93 at 5e-4, 4.68 at 1e-3, 4.58 at 2e-3) and confirmed stable on D and P (4.39 and 4.55 at 2e-3). Frozen for all arms. Note: 2e-3 was the highest value tried, so a higher value was not tested.
2. Seed controls init and data order (above).
3. fp16 path, fused AdamW and memory at micro-batch 16 are **untested on a GPU** (no GPU where this was built).
   If it runs out of memory: `--micro_batch 8 --grad_accum 8` gives the identical effective batch.
4. Packing without cross-document masking (nanoGPT style). Fine at this scale; say so in the report.
