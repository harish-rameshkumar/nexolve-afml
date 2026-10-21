"""Entry point: build_model(arm, config)."""
from __future__ import annotations

import torch.nn as nn

from configs.model_config import ModelConfig, default_config
from models.attention import StandardAttention
from models.transformer import Transformer


def build_model(arm: str, config: ModelConfig | None = None) -> nn.Module:
    """Build the model for arm "T" (standard), "D" (DIFF V2) or "P" (param-matched control)."""
    cfg = config or default_config()
    if arm == "T":
        return Transformer(cfg, StandardAttention, arm="T")
    raise ValueError(f"unknown or not-yet-implemented arm {arm!r}; expected 'T', 'D' or 'P'")
