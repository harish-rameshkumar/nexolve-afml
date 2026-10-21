"""Llama-style decoder-only transformer shared by all arms.

Arms differ only in (a) which attention module each block uses and (b) the FFN
hidden size; everything else (embeddings, norms, RoPE, init) is identical so any
difference in results is attributable to attention.
"""
from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn as nn

from configs.model_config import ModelConfig
from models.layers import RMSNorm, SwiGLU


class Block(nn.Module):
    """Pre-norm block: x = x + Attn(Norm(x)); x = x + FFN(Norm(x))."""

    def __init__(self, cfg: ModelConfig, attn: nn.Module, ffn_hidden: int):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = attn
        self.ffn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.ffn = SwiGLU(cfg.d_model, ffn_hidden)

    def forward(self, x, return_attn: bool = False):
        a, attn_map = self.attn(self.attn_norm(x), return_attn=return_attn)
        x = x + a
        x = x + self.ffn(self.ffn_norm(x))
        return x, attn_map


class Transformer(nn.Module):
    def __init__(self, cfg: ModelConfig, attn_factory: Callable[[ModelConfig], nn.Module],
                 ffn_hidden: int | None = None, arm: str = "T"):
        super().__init__()
        self.cfg, self.arm = cfg, arm
        ffn_hidden = ffn_hidden or cfg.ffn_hidden
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(
            Block(cfg, attn_factory(cfg), ffn_hidden) for _ in range(cfg.n_layers)
        )
        self.final_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight  # one (V, d) matrix used twice
        self.apply(self._init_weights)
        # GPT-2 style: scale the layers that write into the residual stream by
        # 1/sqrt(2*n_layers) so the stream's variance doesn't grow with depth.
        for name, p in self.named_parameters():
            if name.endswith(("o_proj.weight", "ffn.down.weight")):
                nn.init.normal_(p, std=cfg.init_std / math.sqrt(2 * cfg.n_layers))

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=self.cfg.init_std)

    def forward(self, input_ids: torch.Tensor, return_attn: bool = False):
        """input_ids: (B, T) long -> logits (B, T, V).

        With return_attn=True returns (logits, [attn per layer, each (B, H, T, T)]).
        Loss is left to the caller (pretraining and SFT mask tokens differently).
        """
        assert input_ids.shape[1] <= self.cfg.context_len, "sequence longer than context_len"
        x = self.embed(input_ids)                                 # (B, T, d)
        maps = []
        for blk in self.blocks:
            x, m = blk(x, return_attn=return_attn)
            maps.append(m)
        logits = self.lm_head(self.final_norm(x))                 # (B, T, V)
        return (logits, maps) if return_attn else logits

    def num_params(self) -> int:
        """Unique parameters (tied embedding counted once)."""
        return sum(p.numel() for p in self.parameters())
