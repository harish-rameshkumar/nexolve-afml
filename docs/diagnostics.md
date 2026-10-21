# Diagnostics: how to read the plots

Tools: `models/diagnostics.py` (functions), `scripts/plot_attention.py`, `scripts/plot_lambda.py`.

## Logging lambda during SFT (arm D only)

```python
from models import lambda_log_dict
# every N steps, on a fixed probe batch of token ids (B, T):
logger.log(lambda_log_dict(model, probe_ids), step=step)   # flat {name: float}
```
Keys: `lambda/layer{l}/{mean,std,min,max}` and `lambda/layer{l}/head{h}/mean`. Use the **same probe batch** every
time (lambda depends on the input tokens), otherwise changes you see may just be a different batch. For T and P
the call raises `ValueError`, since they have no lambda. `lambda_summary(model, ids)` returns the same numbers as tensors.
It runs under `no_grad`, on the model's device, and restores train/eval mode.

## Reading `plot_lambda.py`

Lambda is the fraction of the *second* attention map that gets subtracted from the first, separately for every
token and every head: `out = attn1 - sigmoid(lambda) * attn2`. The plots show `sigmoid(lambda)`, always between 0 and 1.

- **Near 0:** almost nothing is subtracted. The head behaves like an ordinary attention head (it just uses its first
  query). The differential mechanism is effectively switched off for that token/head.
- **Near 1:** the second map is subtracted almost fully, so whatever both maps attend to cancels. The head keeps
  only what the first map attends to *more* than the second. This is the strongest "noise cancelling" setting.
- **Near 0.5 everywhere:** what you see at random initialisation (we initialise the lambda projection near 0), i.e. the
  model has not decided anything yet. A **trained** model with a different spread (some heads high, some low, wider boxes) shows heads
  specialising; a narrow box means that head uses the same strength for every token, a wide one means lambda
  really depends on the token.
- **Before vs after SFT** (pass two checkpoints with `--ckpt before.pt after.pt`): shifts of whole layers or heads
  toward 0 or 1 show where SFT changed how aggressively each part of the network denoises. Compare the heatmaps
  on their shared 0..1 scale.

Left panel = distribution per layer (box = middle 50% of all token/head values, line = median). Heatmaps = mean per head.

## Reading `plot_attention.py`

Rows are (layer, head); **left is T (ordinary attention), right is D (differential)**. Query token on the vertical axis
(who is looking), key token on the horizontal axis (what it looks at). Only the lower triangle is non-zero: a token cannot see the future.

- **T:** every entry is a probability, >= 0, each row sums to 1. Darker = more attention.
- **D:** the plotted number is `A1 - sigmoid(lambda) * A2`, so it can be **negative**, hence the diverging scale centred on
  zero (red positive, blue negative, white zero). Rows sum to `1 - sigmoid(lambda)` instead of 1.
  - **Positive (red):** map 1 attends to this token more than the lambda-scaled map 2 does, so the token's value is *added* to the output.
  - **Negative (blue):** map 2 attends there more (after scaling), so the token's value is *subtracted*. The head is actively pushing that
    token's information *out* of the result, which ordinary softmax attention cannot do.
  - **White (about 0):** both maps agree, so the contribution cancels: that is the "noise" the subtraction removes.
- A clean differential head typically looks sparser than its T counterpart: a few strong red entries on a mostly-white background.
- **Random-init models look boring on purpose:** with lambda ≈ 0.5 and untrained queries, D's map is a faded copy of a flat softmax map
  with no blue at all. Judge trained checkpoints, not the random-init example images committed in `docs/figures/`.

Note the two colour scales are intentionally different (sequential for T, diverging for D) and each panel has its own limits, so compare *patterns*, not absolute colours.
Without a tokenizer, `--text` feeds UTF-8 bytes as token ids, which is not the project tokenizer; for real inputs pass `--ids` from the tokenizer.
