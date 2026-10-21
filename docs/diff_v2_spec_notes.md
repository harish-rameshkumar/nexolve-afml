# DIFF V2 spec notes (read BEFORE writing the D model)

Sources, all fetched and read on 2026-10-21:

- **[Blog]** "Differential Transformer V2", Ye/Dong/Sun/Wei (Microsoft),
  https://huggingface.co/blog/microsoft/diff-attn-v2
- **[Code]** the official reference implementation the blog links to:
  `microsoft/unilm`, `Diff-Transformer/Diff-Transformer-V2/multihead_flashdiffv2.py`
- **[Paper]** Differential Transformer (V1), arXiv:2410.05258 (HTML v2)

Quotes are verbatim. Where the blog and the code give the same fact, both are shown.

## 1. Head counts: query heads vs K/V heads

[Blog] "DIFF V2 doubles number of query heads while maintaining number of key value heads, and the extra dimension is reduced back to h*d after the differential operation so the W_O projection remains the same as baseline Transformer."

[Blog] pseudo-code docstring:
```
q:   (N, 2h, d)
k:   (N, h_kv, d)
v:   (N, h_kv, d)
lam: (N, h, 1)
```
[Code] `self.num_q_heads = 2 * self.num_heads if self.use_diff_v2 else self.num_heads`;
`num_kv_heads` defaults to `num_heads` when not given.

**For us:** h = 8 (the T head count), so D has 16 query heads, and 8 K/V heads (h_kv = h = 8,
i.e. plain multi-head, not grouped, because T uses 8 K/V heads and `num_kv_heads=None` means h).
Head dim d = 64 everywhere, V has the same head dim as K (no doubled V as in V1).

## 2. How Q/K/V are shaped and paired

[Blog] "Note DIFF V2 subtracts two heads that are in the same GQA group, which means they share the same key and value. This is crucial to performance."

[Blog] "(... Heads belonging to the same GQA group are arranged contiguously in the output)"

[Blog, code] the correct pairing is the interleaved one:
```
attn = flash_attn_func(q, k, v)
attn1, attn2 = (attn[:, 0::2],
                attn[:, 1::2])
```
and the explicitly **wrong** one (Ablation 1): `attn[:, :nh//2], attn[:, nh//2:]`, which gives
"obvious training instability ... and higher loss".

[Code] `q = q.view(bsz, tgt_len, self.num_q_heads, self.head_dim)` ... `attn1, attn2 = attn[:, :, 0::2], attn[:, :, 1::2]`.

**Pairing, spelled out.** FlashAttention's GQA maps query head `i` to K/V head `i // (num_q_heads / num_kv_heads)`.
With 16 query heads and 8 K/V heads the group size is 2, so query heads (0,1) share K/V head 0,
(2,3) share K/V head 1, and so on. Even head 2j is attn1 of output head j; odd head 2j+1 is attn2.
Equivalent view that we will use: reshape q to `(B, T, h, 2, d)`; index `[..., 0, :]` is attn1, `[..., 1, :]` is attn2.

## 3. Lambda (token- and head-specific)

[Blog] "We replace the globally shared λ with a token-specific, head-wise projected λ. This eliminates the exponential re-parameterization and initialization of λ."

[Blog] "The λ in DIFF V2 is projected from X for each token each head."

[Code] `self.lambda_proj = nn.Linear(self.d_model, self.num_heads, bias=False)`; in forward `lambda_val = self.lambda_proj(x)`.
So lambda is one linear map d_model -> h from the (normed) layer input, no bias, shape `(B, T, h)`: one value per token per **output** head (h = 8, not 16).

## 4. The differential subtraction

[Blog] pseudo-code:
```
lam_val = sigmoid(lam)
attn = attn1 - lam_val * attn2
```
[Blog] in formula form: c_i = sum_j ( Softmax(z_ij^1) - sigmoid(lambda_i) * Softmax(z_ij^2) ) v_j, "RMS(c_i) in (0, sqrt 2)".

[Code] `return attn1 - torch.sigmoid(lambda_val).unsqueeze(-1) * attn2`.

**Important detail:** the subtraction is applied to the two heads' *outputs*
(after multiplying each softmax map by the shared V). Because both maps multiply the same V, this is
algebraically identical to `(A1 - sigmoid(lam) * A2) V`, the "difference of attention maps" form. We rely on that identity
in the naive reference test. Note lambda is per query-token `i`, so it scales the whole row of A2 for that token.

Ablations in the blog (why each piece is there):
- no lambda scaling -> "excessively small context RMS at initialization"
- lambda without sigmoid -> "context RMS is unbounded from above", less stable
- subtracting non-group heads -> unstable and worse (Ablation 1)
- Transformer with 1.5*h heads (parameter-aligned control) -> "higher training loss comparing to DIFF V2" (this is our arm P's motivation)

## 5. Normalization

[Blog] "We remove the per-head RMSNorm after differential attention. We find the per-head RMSNorm can lead to instability in later stages of large-scale pretraining of LLM."

[Blog] "The (1 - lambda_init) post-multiplier of V1 is also gone": the V2 code (Section 1 pseudo-code and [Code]) has no norm and no scale after the subtraction.
(The block-level pre-norm RMSNorm before attention is unchanged: it is not part of the attention spec.)

## 6. Output projection

[Blog] "... the extra dimension is reduced back to h*d after the differential operation so the W_O projection remains the same as baseline Transformer."

[Code] `self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.d_model, bias=False)` and
`attn.reshape(bsz, tgt_len, self.num_heads * self.head_dim)` before it.
So W_O is (h*d -> d_model) = (512 -> 512), identical to T.

## 7. Parameter consequences (default config, per layer)

|                         | T            | D (V2)                          |
|-------------------------|--------------|----------------------------------|
| W_Q                     | 512 x 512    | 512 x **1024** (+262,144)        |
| W_K, W_V                | 512 x 512 each | same                           |
| W_O                     | 512 x 512    | same                             |
| lambda_proj             | -            | 512 x 8 (+4,096)                 |

Extra per layer: 266,240; over 6 layers: 1,597,440. Totals: T = 27,662,848, **D = 29,260,288 (+5.77%)**.

## 8. Ambiguities, and V1 vs V2 differences

| Topic | V1 (paper) | V2 (blog + code) |
|---|---|---|
| Q/K split | Q, K have 2d per head, split into (Q1,Q2),(K1,K2); two **different** key sets | 2h query heads, h_kv key heads; the pair **shares** K and V |
| V head dim | 2d | d |
| lambda | one learnable scalar per layer (from 4 vectors, exponentiated) + `lambda_init = 0.8 - 0.6 exp(-0.3 (l-1))` | `sigmoid(Linear(x))`, per token, per head; no special init |
| Post-attn norm | per-head GroupNorm/RMSNorm, then x(1 - lambda_init) | **none** |
| Attention-parameter budget | matched to the baseline Transformer | **larger** (extra Q2, plus lambda_proj) |
| W_O | 2d x d_model per head | h*d -> d_model, same as baseline |
| Kernel | custom | plain FlashAttention/SDPA, then subtract |

Ambiguities / things the spec does NOT pin down (our decisions, all flagged):

1. **Initialisation of lambda_proj.** [Code] uses PyTorch's default `nn.Linear` init; the blog says nothing. We use the shared
   `N(0, 0.02)` init like every other matrix, so `sigmoid(lambda)` starts near 0.5 (the second map is subtracted at half strength).
2. **RoPE convention.** [Code] uses `interleaved=True` rotary. We reuse T's rotate-half RoPE. Both are valid RoPE (a fixed permutation of
   the head dimensions) and the head-dim permutation is irrelevant for a model trained from scratch; sharing T's RoPE keeps the only T/D difference in attention.
3. **Which input feeds lambda_proj.** [Code] passes `x`, the same tensor given to q/k/v projections. In a pre-norm block that is the
   RMS-normed layer input, which is what we do.
4. **Both paired query heads get the same RoPE'd K and same V; they differ only in their query projection.** The spec implies this
   (shared K/V) but does not say how the pair should be *initialised* differently; random init already breaks the symmetry.
5. **h_kv.** The blog's examples leave h_kv free (GQA). We follow the code default `num_kv_heads=None -> h`, which matches T (8 K/V heads).
   This is the choice that makes "only extra Q2 and lambda" the sole difference from T.
6. **Blog vs "parameter reallocation".** The blog says extra parameters are "borrowed from other parts of the model" and that ~25% of
   attention parameters can be saved. We do not reallocate for D: D keeps T's FFN and simply has more parameters. The size difference is
   then absorbed by the control P (project plan), not by shrinking D.
7. The blog's results are "still running" and no formal report exists; the claims above are the authors' preliminary observations.
