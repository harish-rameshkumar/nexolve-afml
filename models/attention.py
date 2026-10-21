"""Standard causal multi-head attention (arm T, and arm P's attention).

Tensor-shape legend used in all comments:
    B = batch, T = sequence length (queries), S = sequence length (keys; S == T
    here, named differently so einsum letters stay distinct), H = heads,
    hd = head dim, d = d_model.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.model_config import ModelConfig
from models.layers import RotaryEmbedding


def stable_softmax(scores: torch.Tensor) -> torch.Tensor:
    """Softmax over the last dim, computed in at least float32.

    Half-precision scores are upcast for accuracy; float64 stays float64 (needed
    for gradcheck). The result is cast back to the input dtype.
    """
    acc = torch.promote_types(scores.dtype, torch.float32)
    return torch.softmax(scores, dim=-1, dtype=acc).to(scores.dtype)


def causal_mask(T: int, device) -> torch.Tensor:
    """(T, T) bool, True where query t may attend to key s (s <= t)."""
    return torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))


class StandardAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads, self.head_dim, self.impl = cfg.n_heads, cfg.head_dim, cfg.attn_impl
        d, hd, H = cfg.d_model, cfg.head_dim, cfg.n_heads
        self.q_proj = nn.Linear(d, H * hd, bias=False)
        self.k_proj = nn.Linear(d, H * hd, bias=False)
        self.v_proj = nn.Linear(d, H * hd, bias=False)
        self.o_proj = nn.Linear(H * hd, d, bias=False)
        self.rope = RotaryEmbedding(hd, cfg.context_len, cfg.rope_theta)

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        """x: (B, T, d) -> (out (B, T, d), attn (B, H, T, T) or None)."""
        B, T, _ = x.shape
        H, hd = self.n_heads, self.head_dim

        # Project and split into heads: (B, T, H*hd) -> (B, T, H, hd)
        q = self.q_proj(x).view(B, T, H, hd)
        k = self.k_proj(x).view(B, T, H, hd)
        v = self.v_proj(x).view(B, T, H, hd)
        q, k = self.rope(q), self.rope(k)  # position enters only through q and k

        if self.impl == "sdpa" and not return_attn:
            # Fused kernel; is_causal=True applies the same lower-triangular mask.
            o = F.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True
            ).transpose(1, 2)                                   # (B, T, H, hd)
            attn = None
        else:
            o, attn = self.explicit_attention(q, k, v)
        return self.o_proj(o.reshape(B, T, H * hd)), attn

    @staticmethod
    def explicit_attention(q, k, v):
        """Reference implementation with einsum. q, k, v: (B, T, H, hd)."""
        B, T, H, hd = q.shape
        # scores[b,h,t,s] = sum_e q[b,t,h,e] * k[b,s,h,e] / sqrt(hd)
        scores = torch.einsum("bthe,bshe->bhts", q, k) / math.sqrt(hd)   # (B, H, T, S)
        # Causal mask: query t sees keys s <= t only. -inf -> weight exactly 0.
        mask = causal_mask(T, q.device)                                   # (T, S)
        scores = scores.masked_fill(~mask, float("-inf"))
        attn = stable_softmax(scores)                                     # rows sum to 1
        # out[b,t,h,e] = sum_s attn[b,h,t,s] * v[b,s,h,e]
        o = torch.einsum("bhts,bshe->bthe", attn, v)                      # (B, T, H, hd)
        return o, attn
