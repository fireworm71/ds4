# T11: which layers to promote does not matter. The count does.

Measured 2026-09-18. Answers "can we find which layers have impact, then take
the best?" -- the idea being to pick the highest-impact k layers for a Q3
splice instead of simply taking the last k.

**Result: there is a weak real per-layer signal, and it is ~13x smaller than
the noise floor of the scoring we would use to exploit it. Layer selection is
not a lever. Keep taking the last k.**

## Method

For each of the 40 layers, relative Frobenius divergence between the Q2 and Q4
routed-expert weights, `||W_q2 - W_q4|| / ||W_q4||`, weighted across
gate/up/down by tensor size.

Why Q2-vs-Q4 rather than each against the source: `config.json` gives
`expert_dtype: fp4`, so the original routed experts are already 4-bit. Q4_K is
a ~4.5-bit format and therefore sits very close to the source, which makes the
Q2<->Q4 gap a good proxy for Q2's own error -- and avoids parsing 476 GiB of
safetensors.

Dequantizers for all three formats (IQ2_XXS and Q2_K on the Q2 side, Q4_K on
the Q4 side) were written against the engine's own tables, extracted from
`ds4.c:1018-1033` (`kmask_iq2xs`, `ksigns_iq2xs`, `iq2xxs_grid`) so they match
what actually runs. Rows are sampled rather than reading all 427 GiB.

**Validation before use:** dequantized Q2 and Q4 rows of the same tensor
correlate **0.935-0.950** with matching mean and standard deviation -- the
signature of two quantizations of one weight set. A dequantizer bug would have
shown up as noise here.

Two independent runs: 24 rows/tensor (seed 20260918) and 48 rows/tensor
(seed 1).

## The distribution is flat

| run | mean | std | spread |
|---|---|---|---|
| A, 24 rows | 0.33848 | 0.00248 | 3.13% |
| B, 48 rows | 0.33840 | 0.00204 | 2.28% |

No front/back gradient, no outlier layers, nothing structural. Every layer is
quantized about equally badly by Q2.

## The signal is real but negligible

Spearman rank correlation between the two independent samples: **+0.450**.
That is not zero, so there is genuine per-layer variation underneath the
sampling noise (a Spearman-Brown estimate puts each run's correlation with the
truth around 0.67). Top-10 overlap 4/10, bottom-10 5/10.

So the ranking is not pure noise -- but the magnitude decides it:

| quantity | value |
|---|---|
| mean divergence, all layers | 0.33844 |
| last 7 (33-39, what the 7L splice uses) | 0.33826 |
| best 7 by divergence (0, 2, 12, 14, 18, 19, 36) | 0.34140 |
| advantage of choosing optimally | **+0.93%** |
| 7L measured capture of the Q2->Q4 gap | 18.0% |
| projected capture if optimally chosen | **18.2%** (+0.17 pp) |
| projected NLL gain | **+0.000199** |
| scoring noise floor (8L vs 7L difference) | ~0.0026 |

**The best possible reordering is 13x smaller than the noise in the measurement
we would use to detect it.** It could not be confirmed even if we built the
splice.

## Two premises this retires

1. **`gguf-tools/mixed/README.md` says the last layers matter most** -- the
   "Last Six Layers Q4 Experiment" is justified on that basis. It does not hold
   here: the last 7 layers score 0.33826 against an all-layer mean of 0.33844,
   i.e. *very slightly below average*. The last-k choice is fine, but because
   it is arbitrary and contiguous, not because those layers are special.

2. **"Layer 32 is worth nothing"** (from T10, where 8L did not beat 7L) was
   over-read. Layer 32's divergence is 0.3393/0.3363 across the two runs --
   ordinary, mid-pack. The 8L-vs-7L difference of 0.000139 NLL was scoring
   noise, not evidence about that layer. Both methods now agree layers
   contribute equally; the earlier anomaly was never real.

## What this means for building splices

Choose k by the memory ceiling and the context target, then take any k layers.
Per T9/T10 the return is close to linear in k (7 layers -> 18.0% of the gap,
against 17.5% for a uniform model), so:

* 256K context -> k=7, planned 100.06 GiB/rank. **This is the built and
  measured configuration.**
* 128K -> k=9, 32K -> k=10, by the same arithmetic.

Do not spend a splice-and-transfer cycle on layer selection.

## What was not tested

Weight-space divergence is not output impact. A layer whose weights differ
substantially but whose experts rarely activate matters less than its score
suggests; an activation-weighted metric (imatrix-style) could in principle
surface structure this misses. Given the measured effect is 13x under the noise
floor, that structure would have to be far larger than anything seen here to
change the recommendation -- but the flatness reported above is a statement
about weights, not about routing.
