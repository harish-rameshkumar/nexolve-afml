"""Differential attention V2 (DIFF V2), arm D.

Spec: docs/diff_v2_spec_notes.md (blog + official code `multihead_flashdiffv2.py`).

Plain-language summary of the mechanism
---------------------------------------
Ordinary attention gives every query head ONE softmax map over the context, and
softmax can never output exactly zero, so a little "noise" attention always
leaks onto irrelevant tokens. DIFF V2 gives every head TWO query vectors
(q1, q2) that look at the SAME keys and values. It computes two attention
outputs and subtracts a fraction of the second from the first:

        out = attn1 - sigmoid(lambda) * attn2

Whatever attention both maps agree on (the shared "noise") cancels; what only
map 1 attends to survives. `lambda` is predicted per token and per head.

Shape legend: B batch, T queries, S keys (S == T), H heads (the number of
OUTPUT heads, same as T's n_heads), P = 2 (the pair: 0 -> attn1, 1 -> attn2),
hd head dim, d d_model.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.model_config import ModelConfig
from models.attention import causal_mask, stable_softmax
from models.layers import RotaryEmbedding


class DiffAttentionV2(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads, self.head_dim, self.impl = cfg.n_heads, cfg.head_dim, cfg.attn_impl
        d, hd, H = cfg.d_model, cfg.head_dim, cfg.n_heads

        # DESIGN: 2H query heads but only H key/value heads. Each K/V head serves
        # exactly one pair of query heads (a GQA group of size 2). Sharing K and V
        # inside the pair is what makes the subtraction meaningful: the two maps
        # differ only in their queries, so common-mode noise cancels. (The blog's
        # "Ablation 1" subtracts heads with different K/V and trains worse.)
        self.q_proj = nn.Linear(d, 2 * H * hd, bias=False)   # twice T's query width
        self.k_proj = nn.Linear(d, H * hd, bias=False)       # same as T
        self.v_proj = nn.Linear(d, H * hd, bias=False)       # same as T (V is NOT doubled, unlike V1)

        # DESIGN: lambda is predicted from the token itself, one number per output
        # head, instead of one learned scalar per layer (V1). A token that needs a
        # sharp, clean lookup can subtract a lot; one that needs a broad average can
        # subtract little. sigmoid keeps lambda in (0, 1), which bounds the output
        # size (blog ablation 3: without sigmoid training is less stable).
        self.lambda_proj = nn.Linear(d, H, bias=False)

        # DESIGN: after the subtraction we are back to H heads of width hd, so the
        # output projection is exactly T's (H*hd -> d). No per-head norm here (V1
        # had one; V2 removes it because it hurt stability at scale).
        self.o_proj = nn.Linear(H * hd, d, bias=False)

        # DESIGN: same RoPE module as T, so position handling is not a variable.
        self.rope = RotaryEmbedding(hd, cfg.context_len, cfg.rope_theta)

        # Diagnostic: sigmoid(lambda) of the latest forward pass, shape (B, T, H),
        # detached. Read via Transformer.lambda_values().
        self.last_lambda: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        """x: (B, T, d) -> (out (B, T, d), diff attention map (B, H, T, T) or None).

        The returned map is A1 - sigmoid(lambda) * A2. Unlike a softmax map it can
        contain negative entries and its rows sum to 1 - sigmoid(lambda), not 1.
        """
        B, T, _ = x.shape
        H, hd = self.n_heads, self.head_dim

        # Project. q has 2H heads; k and v have H heads.
        q = self.q_proj(x).view(B, T, 2 * H, hd)           # (B, T, 2H, hd)
        k = self.k_proj(x).view(B, T, H, hd)               # (B, T, H, hd)
        v = self.v_proj(x).view(B, T, H, hd)               # (B, T, H, hd)
        q, k = self.rope(q), self.rope(k)                  # rotate by position

        # lam[b,t,h] in (0,1): how much of the second map to subtract for token t, head h.
        lam = torch.sigmoid(self.lambda_proj(x))           # (B, T, H)
        self.last_lambda = lam.detach()

        # DESIGN (head pairing): query head 2j and 2j+1 form output head j and share
        # K/V head j. Viewing the 2H axis as (H, 2) puts the pair index last, i.e.
        # q_pair[..., j, 0, :] is head 2j (attn1) and [..., j, 1, :] is head 2j+1
        # (attn2). This is the official `attn[:, 0::2]` / `attn[:, 1::2]` split.
        # NOT the first-half/second-half split, which pairs heads with different K/V.
        q_pair = q.view(B, T, H, 2, hd)                    # (B, T, H, P, hd)

        if self.impl == "sdpa" and not return_attn:
            # Fused path. Repeat each K/V head twice (head j -> slots 2j, 2j+1) so
            # slot i uses K/V head i // 2, exactly the GQA mapping, then run ordinary
            # causal attention on 2H heads and split the result as above.
            k2 = k.repeat_interleave(2, dim=2)             # (B, S, 2H, hd)
            v2 = v.repeat_interleave(2, dim=2)             # (B, S, 2H, hd)
            o = F.scaled_dot_product_attention(
                q.transpose(1, 2), k2.transpose(1, 2), v2.transpose(1, 2), is_causal=True
            ).transpose(1, 2)                              # (B, T, 2H, hd)
            o = o.reshape(B, T, H, 2, hd)                  # (B, T, H, P, hd)
            o1, o2 = o[:, :, :, 0], o[:, :, :, 1]          # each (B, T, H, hd)
            diff = o1 - lam.unsqueeze(-1) * o2
            attn = None
        else:
            diff, attn = self.explicit_attention(q_pair, k, v, lam)
        return self.o_proj(diff.reshape(B, T, H * hd)), attn

    @staticmethod
    def explicit_attention(q_pair, k, v, lam):
        """einsum reference. q_pair (B,T,H,2,hd); k, v (B,S,H,hd); lam (B,T,H)."""
        hd = q_pair.shape[-1]
        T = q_pair.shape[1]
        # scores[b,h,p,t,s] = sum_e q[b,t,h,p,e] * k[b,s,h,e] / sqrt(hd)
        # Both members p of a pair use the SAME k[b,s,h,:] (that is the sharing).
        scores = torch.einsum("bthpe,bshe->bhpts", q_pair, k) / math.sqrt(hd)  # (B, H, P, T, S)
        scores = scores.masked_fill(~causal_mask(T, q_pair.device), float("-inf"))  # causal
        a = stable_softmax(scores)                                              # (B, H, P, T, S)
        # Each map multiplies the shared V: o[b,t,h,p,e] = sum_s a[b,h,p,t,s] v[b,s,h,e]
        o = torch.einsum("bhpts,bshe->bthpe", a, v)                             # (B, T, H, P, hd)
        # DESIGN: subtract AFTER multiplying by V. Since both maps share V this equals
        # (A1 - lam*A2) V, but it keeps the fused-kernel path and this path identical.
        diff = o[:, :, :, 0] - lam.unsqueeze(-1) * o[:, :, :, 1]                # (B, T, H, hd)
        # Effective (signed) attention map, for plotting/diagnostics only.
        lam_map = lam.permute(0, 2, 1).unsqueeze(-1)                            # (B, H, T, 1)
        attn = a[:, :, 0] - lam_map * a[:, :, 1]                                # (B, H, T, S)
        return diff, attn
