"""Naive, slow DIFF V2 attention written straight from the spec (one head, one query
row at a time, python loops, no einsum, no fused kernels). Used only by tests.

It reads the weights of a DiffAttentionV2 module so outputs can be compared.
`pairing="interleaved"` is the spec (heads 2j and 2j+1 are subtracted);
`pairing="halves"` is the blog's "wrong" Ablation 1 (head i minus head i+H).
"""
import math

import torch


def naive_diff_attention(mod, x, pairing="interleaved"):
    B, T, _ = x.shape
    H, hd = mod.n_heads, mod.head_dim
    q = mod.rope(mod.q_proj(x).view(B, T, 2 * H, hd))   # 2H query heads
    k = mod.rope(mod.k_proj(x).view(B, T, H, hd))        # H key heads
    v = mod.v_proj(x).view(B, T, H, hd)
    lam = torch.sigmoid(mod.lambda_proj(x))              # (B, T, H)

    # Step 1: ordinary causal softmax attention for each of the 2H query heads.
    # GQA mapping (what FlashAttention does): query head i reads K/V head i // 2.
    outs, maps = [], []
    for i in range(2 * H):
        kv = i // 2
        o_i = torch.zeros(B, T, hd, dtype=x.dtype)
        a_i = torch.zeros(B, T, T, dtype=x.dtype)
        for b in range(B):
            for t in range(T):
                logits = torch.stack([(q[b, t, i] * k[b, s, kv]).sum() / math.sqrt(hd)
                                      for s in range(t + 1)])        # only s <= t
                w = torch.softmax(logits, dim=0)
                a_i[b, t, : t + 1] = w
                o_i[b, t] = sum(w[s] * v[b, s, kv] for s in range(t + 1))
        outs.append(o_i)
        maps.append(a_i)

    # Step 2: subtract. Output head j = attn_first - sigmoid(lam) * attn_second.
    heads, diff_maps = [], []
    for j in range(H):
        first, second = (2 * j, 2 * j + 1) if pairing == "interleaved" else (j, j + H)
        heads.append(outs[first] - lam[:, :, j:j + 1] * outs[second])
        diff_maps.append(maps[first] - lam[:, :, j:j + 1] * maps[second])
    out = torch.cat(heads, dim=-1)                       # (B, T, H*hd)
    return mod.o_proj(out), torch.stack(diff_maps, dim=1)  # map: (B, H, T, T)
