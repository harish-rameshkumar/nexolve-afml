# Design rationale

Plain-language notes on why each piece of the model is the way it is. Each section says what the piece is,
why we use it, and what would go wrong without it. Code pointers are in `docs/viva_prep.md`;
the exact DIFF V2 specification (with quotes) is in `docs/diff_v2_spec_notes.md`.

Notation: *token* = one piece of text the model reads; *logit* = the model's raw score for each possible next token;
*residual stream* = the running vector each token carries through the layers, to which every layer adds its result.

# Part 1: shared by all three models (T, D, P)

## Pre-norm with RMSNorm (instead of post-norm / LayerNorm)

**What it is.** Before each attention block and each feed-forward block we normalise the input
(`x = x + Attn(Norm(x))`, `x = x + FFN(Norm(x))`) and add the result back onto the residual stream.
RMSNorm rescales a vector by its root-mean-square size and a learned per-dimension gain; unlike LayerNorm it does not subtract the mean and has no bias.
**Why.** In pre-norm the residual path is never normalised, so the gradient has a clean route from the loss back to the first layer,
which keeps training stable without careful warm-up. RMSNorm gives about the same stability as LayerNorm with less computation and fewer parameters.
**Without it.** Post-norm (normalising after the addition) trains less stably as depth grows, and with no normalisation at all activations
can grow layer after layer until the numbers overflow or the loss spikes.

## RoPE (instead of learned or sinusoidal position embeddings)

**What it is.** Attention by itself has no idea of word order. RoPE ("rotary position embedding") injects order by rotating each query and key vector,
in pairs of dimensions, by an angle that grows with the token's position.
**Why.** When two rotated vectors are multiplied, the result depends only on how far apart the two tokens are, not on where they are. So the model
directly sees *relative* distance, which is what language mostly cares about, and RoPE adds no parameters. Learned position tables
are limited to the positions seen in training, and sinusoidal embeddings added at the input mix position into the content before attention.
**Without it.** Attention itself is order-blind (the causal mask leaks only weak, indirect position cues), so the model would struggle to tell "dog bites man" from "man bites dog". RoPE only rotates q and k, never v,
so position affects *where to look* but not *what is read*. Note we only train at length 1024; RoPE does not by itself guarantee good behaviour on longer texts.

## SwiGLU feed-forward (instead of a ReLU/GELU MLP)

**What it is.** The feed-forward block is applied to each token separately and holds most of the model's parameters.
SwiGLU computes two projections of the token: one passes through a smooth switch (SiLU) and multiplies the other one element by element,
`down( silu(gate(x)) * up(x) )`.
**Why.** The multiplication lets the network switch individual features on or off depending on the token ("gating"), which in published comparisons gives lower loss
than a plain MLP at the same size. It uses three matrices instead of two, so the hidden width is reduced to about 8/3 of d_model (1365, rounded up to 1408 = 22 x 64) to keep the parameter count
close to a standard 4x GELU MLP.
**Without it.** A plain MLP would also work; we would just get slightly worse loss per parameter. This is a quality choice, not a correctness one.

## Tied input/output embeddings

**What it is.** The embedding table turns a token id into a vector; the output layer turns the final vector into a score for each of the 16,384 tokens.
We use the *same* matrix for both (the output layer is its transpose).
**Why.** The table is 16,384 x 512 = 8.4M numbers, about 30% of T's 27.7M parameters. Sharing it removes a second copy, and it makes sense: a token's input
vector and the direction that predicts it at the output are about the same thing. In a small model the saved parameters are better spent on layers.
**Without it.** The model would be about 36M parameters, with about a quarter of them in a second, separately learned table.

## No biases

**What it is.** Linear layers compute `W x` and not `W x + b`; RMSNorm has no bias either.
**Why.** Biases add few parameters and little modelling power next to the normalisation layers, which already control the scale and
(through the gains) the shape of the activations. Dropping them matches modern Llama-style models and keeps the parameter formulas simple (e.g. D's extra parameters are exactly Q2 and lambda). A test checks that no bias parameter exists.
**Without it.** Nothing breaks; adding biases would just slightly change the parameter counts we are carefully matching.

## Causal masking and teacher forcing

**What it is.** The model must predict each token from the tokens *before* it. The causal mask makes position t unable to attend to positions after t (those scores are set to -infinity before the softmax, so their weight is exactly 0).
Teacher forcing means that during training we always feed the *true* previous tokens (not the model's own guesses), and predict the next token at every position in parallel.
**Why.** One forward pass over a 1024-token text produces 1024 training examples at once, and the mask guarantees each one is a fair question.
**Without the mask.** The model could read the answer (the next token) in its own input; training loss would fall to almost zero while the model learns nothing useful, and it would fail at generation time where the future does not exist.
Our test changes a future token and checks that all earlier logits stay identical.

## Cross-entropy as the training loss

**What it is.** For each position, the loss is `-log p(true next token)`, where p comes from a softmax over the logits. Minimising it is maximum-likelihood training; perplexity is `exp(loss)`.
**Why.** It rewards putting high probability on what really came next, penalises confident mistakes heavily, and has simple, well-behaved gradients.
**Without it** (e.g. plain accuracy) there is no smooth signal telling the model *how* wrong it was. Our models return logits only; the loss is computed by the caller because pretraining uses all positions
while SFT ignores the prompt tokens (only assistant-response tokens are scored).

# Part 2: Differential attention (D) and the control (P)

## Why subtracting two softmax maps cancels attention noise

**What it is.** In ordinary attention a token spreads weight over earlier tokens with a softmax, and a softmax can never give an exact zero: every irrelevant token still receives a little weight,
and in a long context those little bits add up ("attention noise"). In DIFF each head has two queries looking at the *same* keys and values, giving two attention maps A1 and A2, and the output is `A1 V - lambda * A2 V`.
**Why it helps (the idea).** Think of noise-cancelling headphones: the part of the pattern that both maps share (a broad, low-level spread over everything) is subtracted away, while what map 1 attends to
*more* than map 2 survives. The result can be sparser and can even put negative weight on a token, which plain attention cannot do.
**Without the subtraction.** The model has only positive, always-nonzero weights and has to learn to ignore the leftover noise in later layers. Caveat: this is the authors' motivation and early evidence (their V2 results are still preliminary);
whether it helps *instruction following in our small models* is exactly what the experiment tests.

## Why lambda is token- and head-specific in V2

**What it is.** `lambda` says how much of map 2 to subtract. V1 used one learned number for the whole layer, with a hand-designed starting value. V2 computes it from the token itself with a small linear layer (one value per head),
passed through a sigmoid so it always lies between 0 and 1.
**Why.** Different tokens need different amounts of cleaning: a token doing a sharp lookup wants strong subtraction, a token averaging over context wants little. A sigmoid keeps the output size bounded, and the blog reports that
removing the sigmoid makes training less stable and removing the lambda scale entirely makes the output too small at the start (their ablations 2 and 3). It also removes V1's special initialisation.
**Without it.** One fixed number per layer cannot adapt to the token and cannot be switched off for heads that do not need it. In our model, lambda also gives us a diagnostic: we can log it during SFT and see which heads change.

## Why the paired heads are interleaved (0::2 / 1::2) and share K/V

**What it is.** DIFF V2 has 16 query heads but only 8 key/value heads. Query heads 2j and 2j+1 form pair j and both read key/value head j; the output of head 2j+1 (scaled by lambda) is subtracted from head 2j.
**Why.** Subtraction only cancels *common* structure if the two maps see the same keys and values; then they differ only in their queries. Heads 2j and 2j+1 are neighbours in the layout that grouped-query attention uses, so they share K/V automatically.
**Without it.** If instead we paired the first half of the heads with the second half (head i with head i+8), the two heads would read different keys/values, so there is no shared noise to cancel; the blog's "Ablation 1" shows this trains
unstably and to a worse loss. A test (`test_wrong_half_half_pairing_differs`) checks that our code uses the interleaved pairing and that the wrong pairing gives different outputs.

## Why there is no normalisation after the subtraction (V2)

**What it is.** V1 normalised each head's output right after the subtraction (and multiplied by `1 - lambda_init`). V2 removes that.
**Why.** If a head attends almost uniformly over n tokens its output is tiny (about 1/sqrt(n) of normal size), and a per-head norm then blows it back up by about sqrt(n): the blog notes roughly a 100x magnification at 8192 tokens, which led to large gradients and
instability late in large-scale training. In V2 the token-specific, sigmoid-bounded lambda already controls the output size, so the norm is no longer needed.
**Without this change.** We would carry V1's instability risk. Honest caveat: that instability was reported at large scale; at our ~29M scale it may never show up, which is one more reason to follow the official V2 spec rather than improvise.

## Why P is the right control

**The confound.** DIFF V2 is not "T plus a trick": to run the subtraction it needs a *second set of query
projections* (16 query heads instead of 8) and a small `lambda` projection. With the default config that is
1,597,440 extra parameters (+5.77%) in the attention modules. If D learned instructions faster than T, there
would be two explanations we could not tell apart:

1. the **subtraction** (differential attention) does something useful, or
2. D simply has **more parameters**, so any model with ~5.8% extra capacity would have done as well.

**What P is.** P is exactly T (same attention module, same everything) except that its SwiGLU FFN is
wider (hidden **1584** instead of 1408), chosen so P has the same total parameter count as D to within 0.084%
(29,284,864 vs 29,260,288, a gap of +24,576 parameters).

| arm | params | vs T | FFN hidden |
|---|---|---|---|
| T | 27,662,848 | +0.00% | 1408 |
| D | 29,260,288 | +5.77% | 1408 |
| P | 29,284,864 | +5.86% | 1584 |

*Why 1584.* One extra hidden unit costs 3 x 512 x 6 = 9,216 parameters, so matching D exactly would need
1408 + 1,597,440 / 9,216 = 1581.33. We round to the nearest multiple of 16 because GPU tensor cores run
fastest on widths that are multiples of 8/16; the price is a +0.084% gap to D (the exact integer 1581 would
be -0.011%, but is an awkward width). A multiple of 64 (1600) would be +0.59%; all of these are far inside the 1% tolerance. So P gets the extra capacity but has *no* differential mechanism.

**What question P answers.** "Is any gain of D over T explained by extra parameters alone?" Comparing D with
P holds capacity fixed, so a difference between them is attributable to the mechanism, not the size.

**Why widen the FFN (and not something else).** The extra parameters have to go somewhere. The FFN is where
a standard transformer keeps most of its parameters, widening it is the most common way to scale a model, and
it leaves the attention module identical to T's, so the only T-to-P change is the FFN width. We considered
spending them on attention instead (e.g. more heads, the blog's "1.5h heads" ablation); that makes the
control attention-capacity-matched but changes the attention structure, so the comparison between D and P would
no longer be "same attention, different mechanism". A wider FFN is the cleaner "extra capacity, nothing else"
control. (The blog's own 1.5h-heads ablation reports higher loss than DIFF V2; we can still add it as an extra
arm later if time permits.)

**Reading the results (three outcomes).** Let "better" mean lower loss / better instruction-following at the
same number of SFT tokens.

| Outcome | What we would see | What it means |
|---|---|---|
| **D > T and D > P** | D beats both; P is about the same as T (or only slightly better) | The subtraction itself helps. Extra parameters are not the explanation. This is the strongest evidence for DIFF V2. |
| **D ≈ P > T** | D and P are close to each other and both beat T | The gain is explained by capacity. DIFF V2 is not doing anything a plain wider model does not. A valid, honest negative result for the mechanism. |
| **Overlapping** | Differences between arms are within seed-to-seed noise | We cannot conclude either way at this scale; report confidence intervals, and say that more seeds or a larger setting would be needed. Do not over-read small gaps. |

In all three cases the comparison is informative only if the arms share the tokenizer, data, optimizer and
schedule, which is why those are fixed across T, D and P.

**Caveat.** Matching parameters does not match compute: D's attention does about twice the query-key work of T
and P's wider FFN does more FFN work, so FLOPs differ slightly between D and P. We match parameters because
that is the confound we set out to control; we note the FLOPs difference rather than hide it.
