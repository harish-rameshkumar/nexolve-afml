# Viva prep: 20 likely questions about the model

Each answer is short and points to where it is shown in code. Longer explanations: `docs/design_rationale.md`;
spec quotes: `docs/diff_v2_spec_notes.md`. "Test" = a pytest function; run `pytest -v` to see them.

## Architecture

**1. What exactly are we training?** Three ~28-29M-parameter decoder-only (Llama-style) transformers from scratch: 6 layers, d_model 512, 8 heads of size 64,
vocab 16,384, context 1024. Everything is in `configs/model_config.py`; assembly in `models/transformer.py`, entry point `build_model` in `models/build.py`.

**2. Walk me through one layer.** `x = x + Attn(RMSNorm(x))`, then `x = x + SwiGLU(RMSNorm(x))`. See `Block.forward` in `models/transformer.py`. Pre-norm keeps a clean residual path.

**3. Why pre-norm and RMSNorm?** Stable gradients without warm-up tricks, cheaper than LayerNorm (no mean, no bias). `RMSNorm` in `models/layers.py`; rationale section 1.

**4. How does the model know word order?** RoPE: queries and keys are rotated by a position-dependent angle so q.k depends only on relative distance. `RotaryEmbedding` in `models/layers.py` (shape comments inside).
The same module is used by T, D and P, so position handling is not a variable.

**5. Why SwiGLU and why hidden size 1408?** Gated feed-forward (`SwiGLU` in `models/layers.py`). It has three matrices, so hidden is ~8/3 x 512 = 1365, rounded up to 1408 (a multiple of 64), which keeps the parameters close to a 4x GELU MLP.

**6. What does tying the embeddings do, and how do you know it is on?** Input table and output layer are the same 16,384 x 512 matrix (~8.4M params, ~30% of T). Test: `test_t_model.py::test_tied_embeddings`; parameters are counted once in `Transformer.num_params`.

**7. Show me that the mask is causal.** `causal_mask` + `explicit_attention` in `models/attention.py` (`masked_fill(~mask, -inf)` before softmax). Test: `test_causality` in each of `test_t_model.py`, `test_d_model.py`, `test_p_model.py`
changes a future token and requires identical earlier logits, for both attention implementations. `test_attention_maps_valid` checks the upper triangle is exactly 0 and rows sum to 1.

**8. What is teacher forcing and what loss do you use?** Feed true previous tokens, predict the next token at every position at once; loss = cross-entropy `-log p(true next token)`. The model returns logits only (`Transformer.forward`); the loss is computed by the pretraining/SFT code
(SFT scores only assistant tokens). Overfit tests (`test_overfit_tiny_batch`) show the logits/loss pipeline can fit data.

**9. What are the two attention implementations and how do you know they agree?** An explicit einsum version (readable, with shape comments; used when attention maps are requested) and the fused `scaled_dot_product_attention` (default, fast, works on CPU/MPS/CUDA).
Tests: `test_sdpa_matches_einsum` for T (`test_t_model.py`) and D (`test_d_model.py`).

## DIFF V2 (arm D)

**10. What does DIFF V2 change compared with standard attention?** Two queries per head over the same keys/values; output = `attn1 - sigmoid(lambda) * attn2`. 16 query heads, 8 K/V heads, output projection unchanged. `models/diff_attention.py`;
shapes asserted in `test_d_model.py::test_q_k_v_head_counts`.

**11. How is V2 different from the original paper (V1)?** V1: keys also split in two (no sharing), V twice as wide, one scalar lambda per layer with a special init, per-head norm after the subtraction, attention parameters matched to the baseline.
V2: paired heads share K/V, V is normal width, lambda is projected from each token (sigmoid), no post-subtraction norm, extra Q parameters accepted. Table in `docs/diff_v2_spec_notes.md` section 8.

**12. Why must the two subtracted heads share keys and values, and why 0::2 / 1::2?** Only then do the two maps differ solely in their queries so shared noise cancels. Interleaved pairs match the GQA group layout. The blog's wrong half/half split trains worse.
Test: `test_d_model.py::test_wrong_half_half_pairing_differs` (also `reference_diff.py` implements both pairings).

**13. How is lambda computed and what does its value mean?** `lambda_proj = Linear(512, 8)` applied to each token's (normed) hidden state, then sigmoid: one value in (0,1) per token per head. Near 0: head behaves like normal attention; near 1: second map nearly fully subtracted.
Test: `test_lambda_exposed_per_layer_and_head`; reading guide: `docs/diagnostics.md`.

**14. How do you know your D is what the spec says, not just something plausible?** A deliberately naive, loop-based reference written from the spec (`tests/reference_diff.py`) must match the module's output and its signed attention map: `test_matches_naive_reference`.
A float64 `torch.autograd.gradcheck` confirms the gradients (also w.r.t. `lambda_proj`): `test_gradcheck_double`. The spec itself is quoted in `docs/diff_v2_spec_notes.md`.

**15. Can attention weights be negative in D? What do rows sum to?** Yes: the effective map `A1 - lambda*A2` can be negative; each row sums to `1 - sigmoid(lambda)`. Test: `test_diff_map_rows_sum_to_one_minus_lambda`. Plots: `scripts/plot_attention.py` (diverging colour scale).

**16. Why no normalisation after the subtraction?** V2 removed V1's per-head norm: a near-uniform head has tiny output and the norm would amplify it by ~sqrt(n) (about 100x at 8192 tokens), causing large gradients. The bounded, token-specific lambda controls magnitude instead. Rationale doc, "no normalisation" section.
Caveat to state: the instability was reported at large scale.

## Parameters and the control

**17. Why do T, D and P have different parameter counts?** T 27,662,848; D 29,260,288 (+1,597,440: second Q projection 512x512 plus lambda 512x8, per layer, x6); P 29,284,864 (T with FFN hidden 1584 instead of 1408).
`parameter_summary` in `models/build.py`, `python scripts/param_summary.py`; tests `test_param_count_vs_t` and `test_p_matches_d_within_one_percent` (P is +24,576 params, +0.084%, vs D).

**18. Why does P exist and why widen the FFN?** D gets extra parameters from its second query set, so D > T alone would not prove the *mechanism* matters. P has the same extra parameters but plain attention: D vs P isolates the mechanism from raw capacity.
The FFN is the cleanest place to spend them because attention stays identical to T's (`test_p_attention_identical_to_t`, `test_only_ffn_width_differs_from_t`). Outcome table in the rationale doc.
Say honestly: parameters are matched but FLOPs are not.

**19. Why 1584 and not the exact 1581?** Exact match needs 1581.33; we round to a multiple of 16 for GPU efficiency, +0.084% vs D (`matched_ffn_hidden` in `models/build.py`; `test_matched_hidden_formula`). It stays "auto" so P remains matched if the config changes.

## Experiment framing

**20. Why "sample efficiency" and not "compute efficiency"?** Our metric is instruction-following versus *cumulative assistant-response tokens seen in SFT*: how much it learns per training example. That is a different question from how much it learns per FLOP or per second.
D does about twice T's query-key work and has more parameters, so it may be costlier per step even if better per token; we compare at equal tokens, with equal data, optimizer and schedule, and we report that D/P differ in compute. Also note the V2 authors' own results are preliminary
("experiments are still running"), and we are not claiming to reproduce them at 28M parameters.
