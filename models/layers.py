"""Small building blocks shared by every arm: RMSNorm, RoPE, SwiGLU."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """y = x / rms(x) * g, with rms(x) = sqrt(mean(x^2) + eps).

    Like LayerNorm without mean-centering and bias: cheaper and just as stable
    in practice. The statistic is computed in float32 for numerical safety.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (..., dim)
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return xf.to(x.dtype) * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary position embedding (RoPE), "rotate-half" convention.

    Each head vector (size hd) is split into two halves (x1, x2). Pair i is
    rotated by angle pos * theta^(-2i/hd):
        [x1', x2'] = [x1*cos - x2*sin,  x2*cos + x1*sin]
    so q.k depends only on the *relative* offset between positions.
    """

    def __init__(self, head_dim: int, max_len: int, theta: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / theta ** (torch.arange(0, head_dim, 2).float() / head_dim)  # (hd/2,)
        pos = torch.arange(max_len).float()                                          # (T,)
        angles = torch.einsum("t,f->tf", pos, inv_freq)                              # (T, hd/2)
        # non-persistent: recomputed on construction, not stored in checkpoints
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, H, hd) -> same shape, rotated by position."""
        T = x.shape[1]
        cos = self.cos[:T, None, :].to(x.dtype)  # (T, 1, hd/2), broadcast over B and H
        sin = self.sin[:T, None, :].to(x.dtype)
        x1, x2 = x.chunk(2, dim=-1)              # each (B, T, H, hd/2)
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class SwiGLU(nn.Module):
    """FFN(x) = W_down( silu(W_gate x) * W_up x ), no biases.

    The gate multiplies the up-projection elementwise, letting the network
    switch hidden features on/off per token. Costs 3 matrices instead of 2.
    """

    def __init__(self, d_model: int, hidden: int):
        super().__init__()
        self.gate = nn.Linear(d_model, hidden, bias=False)
        self.up = nn.Linear(d_model, hidden, bias=False)
        self.down = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, T, d) -> (B, T, d)
        return self.down(F.silu(self.gate(x)) * self.up(x))
