# Design rationale

(Sections for the shared components and DIFF's mechanism are added in step (f).
This section is written first because it explains the experimental design.)

## Why P is the right control

**The confound.** DIFF V2 is not "T plus a trick": to run the subtraction it needs a *second set of query
projections* (16 query heads instead of 8) and a small `lambda` projection. With the default config that is
1,597,440 extra parameters (+5.77%) in the attention modules. If D learned instructions faster than T, there
would be two explanations we could not tell apart:

1. the **subtraction** (differential attention) does something useful, or
2. D simply has **more parameters**, so any model with ~5.8% extra capacity would have done as well.

**What P is.** P is exactly T (same attention module, same everything) except that its SwiGLU FFN is
wider (hidden 1581 instead of 1408), chosen so P has the same total parameter count as D to within 0.011%
(29,257,216 vs 29,260,288). So P gets the extra capacity but has *no* differential mechanism.

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
