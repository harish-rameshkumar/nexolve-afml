# Leakage check: why D memorises a fixed random batch faster

**Observation (Colab T4, default config, batch 8 x 1024, 20 steps on one fixed batch of random tokens):** D's loss falls
from 9.81 to 1.57; T and P only to about 4.0 and 3.7. Before treating this as anything but a curiosity we checked whether D (or any arm) can see
future tokens.

## Verdict: no leakage in any arm

`tests/test_no_leakage.py` (21 tests, about 7 s) checks, for T, D and P on both attention paths (SDPA and einsum):

1. **Per-position losses, not only logits:** changing tokens from position c onwards leaves the losses at positions 0..c-2 unchanged (to 1e-5). Position c-1 is expected to change because its *target* is token c.
2. **Gradients:** the gradient of the loss at position t with respect to the embeddings of every later token is exactly zero, while it is non-zero for earlier tokens.
3. **Batch independence:** changing one sample in a batch leaves the other samples' losses unchanged.
4. **Held-out random batch:** train on a stream of fresh random batches, then evaluate on a held-out random batch. Random tokens carry no signal, so a causal model must stay at ln(V). T, D and P stay within 0.05 of ln(V); a model that can read the next token collapses.
5. **Positive controls:** with the causal mask deliberately removed, checks 1 and 4 fail (held-out loss drops from ln(V) = 4.16 to about 0.2-1.0). So the checks can actually detect a leak.

## Fixed-batch experiment (exploratory script, not committed)

Fixed batch 4 x 64, 30 steps, 3 seeds, lr in {1e-4, 3e-4}; a different fresh random batch evaluated every 5 steps. ln(V) = 9.704 for vocab 16,384.

| Config | lr | Fixed-batch loss at step 20 (T / D / P) | Fresh-batch loss at step 30 minus ln(V) |
|---|---|---|---|
| tiny (d_model 64, 2 layers) | 1e-4 | 9.44 / 9.46 / 9.43 | +0.03 for all arms |
| tiny | 3e-4 | 8.95 / 8.95 / 8.94 | +0.03 for all arms |
| default (d_model 512, 6 layers) | 1e-4 | 1.62 / 0.66 / 1.39 | +0.05 to +0.14 |
| default | 3e-4 | 0.76 / 0.23 / 0.66 | +0.05 to +0.15 |

(Means over three seeds; the seed-to-seed spread is small: for D at lr 3e-4 the three seeds give 0.22-0.23.)

- **Fresh-batch loss never drops below ln(V)** for any arm; it rises slightly (up to +0.15), which is what memorising one batch does. A leak would push it far *below* ln(V).
- **D's speed advantage on the fixed batch is consistent** across all 3 seeds and both learning rates at the default size, and **absent in the tiny config**.

## What explains the speed difference (exploratory, 2 seeds)

Default config, lr 3e-4, 20 steps, fixed-batch loss: T 0.75-0.83; D with lambda frozen at about 0 (second map ignored, so it behaves like plain attention) 0.69;
D as designed (lambda about 0.5 at init, learned) 0.23; D with lambda frozen at about 1 (full subtraction) 0.18. The faster memorisation follows the amount of
subtraction, so it comes from the differential mechanism itself and not from a bug or from the extra query parameters alone (P has them too, via its FFN, and is only slightly faster than T).
Why subtraction helps memorisation is a hypothesis we have not tested (for example, subtracting a shared average may let positions keep their token-specific signal instead of blending it into a mean).

## What this does and does not show

- It rules out the bug explanation: all three arms are causal, to the level of exact per-position losses and gradients.
- It does **not** show that D learns *instructions* more efficiently. Memorising one batch of random tokens is a different task from instruction following, and the effect is absent at the tiny size.
  Treat it as a sanity observation, not as evidence for the research question.
