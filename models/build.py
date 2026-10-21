"""Entry point: build_model(arm, config), plus parameter-count helpers."""
from __future__ import annotations

import torch
import torch.nn as nn

from configs.model_config import ModelConfig, default_config
from models.attention import StandardAttention
from models.diff_attention import DiffAttentionV2
from models.transformer import Transformer


def diff_extra_params(cfg: ModelConfig) -> int:
    """Parameters D has on top of T (closed form).

    Per layer: the second set of query projections (d_model x H*hd, because D has
    2H query heads instead of H) plus lambda_proj (d_model x H). K, V and O are
    unchanged, as is everything outside attention.
    """
    per_layer = cfg.d_model * cfg.n_heads * cfg.head_dim + cfg.d_model * cfg.n_heads
    return cfg.n_layers * per_layer


def matched_ffn_hidden(cfg: ModelConfig) -> int:
    """FFN hidden size that makes P's parameter count (nearly) equal D's.

    A SwiGLU FFN has three d_model x hidden matrices, so one extra hidden unit costs
    3 * d_model parameters per layer, 3 * d_model * n_layers in total. P must absorb
    D's extra parameters through the hidden size:

        hidden_P = hidden_T + diff_extra_params / (3 * d_model * n_layers)

    With the default config that is 1408 + 1,597,440 / 9,216 = 1581.33, rounded to the
    nearest integer (1581, which leaves P 3,072 params = 0.0105% below D). Nothing
    forces the hidden size to be a multiple of 64; that is only a GPU-throughput habit
    and costs accuracy of the match (1600 would overshoot D by +0.59%). Override with
    cfg.ffn_hidden_p if you prefer a rounder number.
    """
    exact = cfg.ffn_hidden + diff_extra_params(cfg) / (3 * cfg.d_model * cfg.n_layers)
    return round(exact)


def build_model(arm: str, config: ModelConfig | None = None) -> nn.Module:
    """Build the model for arm "T" (standard), "D" (DIFF V2) or "P" (param-matched control)."""
    cfg = config or default_config()
    if arm == "T":
        return Transformer(cfg, StandardAttention, arm="T")
    if arm == "D":
        return Transformer(cfg, DiffAttentionV2, arm="D")
    if arm == "P":
        # P = T's attention module, unchanged, with a wider FFN (see matched_ffn_hidden).
        hidden = cfg.ffn_hidden_p if cfg.ffn_hidden_p is not None else matched_ffn_hidden(cfg)
        return Transformer(cfg, StandardAttention, ffn_hidden=hidden, arm="P")
    raise ValueError(f"unknown arm {arm!r}; expected 'T', 'D' or 'P'")


def parameter_summary(config: ModelConfig | None = None, verbose: bool = True) -> dict:
    """Parameter counts for all three arms built from one config.

    Returns {"T": n, "D": n, "P": n, "ffn_hidden": {"T":.., "D":.., "P":..}, "P_vs_D": ...}
    and prints a small table when verbose. Models are built on the "meta" device
    (shapes only, no memory), so this is instant.
    """
    cfg = config or default_config()
    counts, hidden = {}, {}
    with torch.device("meta"):
        for arm in ("T", "D", "P"):
            m = build_model(arm, cfg)
            counts[arm] = m.num_params()
            hidden[arm] = m.blocks[0].ffn.gate.out_features
    out = {**counts, "ffn_hidden": hidden,
           "P_minus_D": counts["P"] - counts["D"],
           "P_vs_D_pct": 100 * (counts["P"] - counts["D"]) / counts["D"]}
    if verbose:
        print(f"{'arm':<4}{'params':>14}{'vs T':>10}{'ffn hidden':>12}")
        for arm in ("T", "D", "P"):
            print(f"{arm:<4}{counts[arm]:>14,}{100 * (counts[arm] / counts['T'] - 1):>+9.2f}%{hidden[arm]:>12}")
        print(f"P - D = {out['P_minus_D']:+,} params ({out['P_vs_D_pct']:+.4f}%)")
    return out

