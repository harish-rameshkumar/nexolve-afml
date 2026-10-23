# nexolve-afml

**Beyond Pretraining Loss: Does Differential Transformer V2 Learn Instructions More Efficiently?** (team Nexolve, AFML course project)

We train three ~28-29M-parameter Llama-style models from scratch with the same tokenizer, data, optimizer and schedule, then run full-parameter SFT and
measure instruction following against the **cumulative number of assistant-response tokens** seen. The question: does DIFF V2 attention learn instructions
more efficiently, and if so, is that the *mechanism* or just *extra parameters*?

| Arm | Attention | Params | vs T | FFN hidden |
|---|---|---|---|---|
| **T** | standard causal multi-head attention | 27,662,848 | - | 1408 |
| **D** | DIFF V2 (16 query heads, 8 K/V heads, token/head-specific lambda) | 29,260,288 | +5.77% | 1408 |
| **P** | standard attention, parameter-matched control (wider FFN) | 29,284,864 | +5.86% (+0.084% vs D) | 1584 |

Shared by all arms: vocab 16,384, context 1024, d_model 512, 6 layers, 8 heads (head dim 64), RoPE, pre-norm RMSNorm, SwiGLU, tied embeddings, no biases.
Why each choice: [docs/design_rationale.md](docs/design_rationale.md). Exact DIFF V2 spec notes: [docs/diff_v2_spec_notes.md](docs/diff_v2_spec_notes.md).

## Repo layout

```
configs/model_config.py   ModelConfig dataclass: the single source of truth for hyper-parameters
models/                   layers.py (RMSNorm, RoPE, SwiGLU), attention.py (T/P), diff_attention.py (D),
                          transformer.py, build.py (build_model, parameter_summary), checkpoint.py, diagnostics.py
scripts/                  param_summary.py, plot_attention.py, plot_lambda.py
tests/                    pytest suite (shapes, causality, naive DIFF reference, gradcheck, param matching, ...)
docs/                     design_rationale.md, diff_v2_spec_notes.md, diagnostics.md, viva_prep.md, figures/
```

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest                              # whole suite (~10 s on CPU)
python scripts/param_summary.py     # parameter table for T / D / P
```

## How to use the models (for teammates)

```python
import torch
from configs import default_config
from models import build_model, save_checkpoint, load_checkpoint, lambda_log_dict, attention_maps

cfg = default_config()                  # or default_config(n_layers=4, ...) to override
model = build_model("T", cfg)           # "T", "D" or "P"; same cfg object works for all three
logits = model(input_ids)               # input_ids: (B, T) long  ->  logits: (B, T, vocab). NO loss inside.

# loss is yours (SFT: mask prompt tokens, e.g. ignore_index=-100):
loss = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, cfg.vocab_size), input_ids[:, 1:].reshape(-1))

save_checkpoint("runs/t_step1000.pt", model, optimizer, step=1000)   # stores arm + config + weights
model, ckpt = load_checkpoint("runs/t_step1000.pt", device="cuda", optimizer=optimizer)  # rebuilds the right arm

# D only (SFT diagnostic): flat dict of floats, pass straight to your logger. Use a FIXED probe batch.
logger.log(lambda_log_dict(model, probe_ids), step=step)
maps = attention_maps(model, probe_ids)  # any arm; (layers, B, heads, T, T) on CPU; D's maps are signed
```
`model.lambda_values()` gives the raw per-token sigmoid(lambda) after any forward pass of a D model. Plots: `python scripts/plot_attention.py --ckpt-t T.pt --ckpt-d D.pt`,
`python scripts/plot_lambda.py --ckpt before.pt after.pt --labels before after`; how to read them: [docs/diagnostics.md](docs/diagnostics.md).

## Devices

Runs on **cpu, mps (Apple Silicon) and cuda**; there are no CUDA-only kernels. Attention uses PyTorch's `scaled_dot_product_attention` by default (`attn_impl="sdpa"`) and an explicit einsum version
whenever attention maps are requested (`attn_impl="einsum"` forces it); tests check the two agree. Tested here on CPU and MPS (forward/backward of D at the default size); CUDA and half precision were not tested in this repo,
so check them before a long run.

## Who owns what

Workstream 1 is defined; the hand-offs for 2-4 are *proposed* from the project plan, so confirm them with the owners.

| Workstream | Owns | Hands off |
|---|---|---|
| 1. Model architecture (this repo's `models/`) | T/D/P code, config, checkpoints format, diagnostics | `build_model`, `ModelConfig`, `save/load_checkpoint`, `lambda_log_dict`, `attention_maps` |
| 2. Data and pretraining | tokenizer (vocab 16,384), pretraining corpus, pretraining loop | tokenizer (`tokenizer.json`); `final.pt` for every arm and seed; `step*.pt` checkpoints every 250 steps; pretraining loss curves and summary tables. Shared Drive folder: `nexolve/outputs/<arm>_s<seed>/` |
| 3. SFT | instruction data, full-parameter SFT, logging vs cumulative assistant-response tokens | SFT checkpoints at each token budget; lambda logs for D |
| 4. Evaluation | instruction-following metrics, plots, statistics across arms/seeds | final comparison of T vs D vs P |

## Do not commit checkpoints or data

`.gitignore` excludes `*.pt`, `*.pth`, `*.ckpt`, `*.safetensors`, `data/` and `outputs/`. Keep weights and datasets out of git (share them through the team drive);
only small example figures in `docs/figures/` are tracked.
