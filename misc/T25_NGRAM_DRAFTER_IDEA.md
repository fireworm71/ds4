# T25: a zero-cost drafter, and what else the other codebases still offer

## Residency, confirmed

Every measurement in T22-T24 was fully resident. From the run logs:

```
memory: KV 0.13 GiB + buffers 6.77 GiB + resident model 80.56 GiB = 87.46 GiB planned
```

No expert-cache line, no `--ssd-streaming`, zero mentions of streaming, against
~113 GiB available per rank. Nothing here is disk-bound.

## The wall, restated

T23 closed every engine-side lever and landed on: speculation is limited by
**drafter hit rate**, not engine overhead. The precise shape of that wall is

```
propose costs             40 ms
a successful block saves  ~63 ms
so propose is +EV only if P(success) > 63%;  measured 17%
```

Every idea so far tried to cut the 63 ms or raise the 17%. **There is a third
term nobody attacked: the 40 ms.** Not by making DSpark cheaper -- by not using
DSpark.

## Idea 1: an n-gram / prompt-lookup drafter (propose ~= 0)

vLLM ships this as its `ngram` speculative method; llama.cpp ships
`examples/lookup` with a full `common/ngram-cache.{h,cpp}`:

```c
#define LLAMA_NGRAM_MIN    1
#define LLAMA_NGRAM_MAX    4
/* Data structures to map n-grams to empirical token probabilities */
```

It is a hash map from the last 1-4 tokens to what empirically followed them,
maintained over the prompt and the generated text. **Drafting is a hash lookup:
no GPU work, no model, no collective.** ds4 has no such path -- `grep -c
'prompt.lookup|ngram.draft|lookup.decod' ds4.c` returns 0. (Engram is unrelated:
it is n-gram *embeddings* feeding layers 1 and 14 of the model itself.)

With propose at ~0 the arithmetic changes completely. Using the post-Stage-1
fit `verify(K) ~= 60 + 17K` against 44 ms/token:

| block | verify | break-even accepted tokens |
|---|---|---|
| K=2 | 94 ms | **E[a] > 1.14** |
| K=3 | 111 ms | E[a] > 1.52 |
| K=4 | 128 ms | E[a] > 1.91 |

No `P(success) > 63%` gate at all, because a failed draft costs nothing to
produce. The only cost is the verify, and it is gated on match length -- the
same shape as the DSpark confidence gate, at zero price.

**Why this fits ds4 specifically:** the whole downstream path already works.
Verify is byte-exact, both ranks stay in lockstep through partial commits
(`DS4_DS41_VERIFY_POSLOG` confirms identical fingerprints), rollback is correct,
and the TP protocol is wired. An n-gram drafter only has to *produce drafts* --
it slots in where `ds4_session_prepare_dspark_draft_impl` does, and everything
after is unchanged.

**Where it wins and loses:** n-gram drafting is strong on repetitive and
structured text -- code, JSON, tables, long quotations, anything that repeats
the prompt -- and weak on novel prose. That is roughly the inverse of nothing:
it is pure addition, since it can be gated to only fire on a long match and
costs nothing when it does not.

It also composes with DSpark rather than replacing it: try the n-gram first
(free), fall back to DSpark only when there is no match and the scheduler
thinks it is worth 40 ms.

## Idea 2: trim the drafter's vocabulary projections (~6 ms of the 40)

`prop_logits` 3.3 ms and `prop_markov` 2.7 ms per propose are full
**256,960-entry** projections when the drafter needs only the top few
candidates. "Speculative Decoding with a Speculative Vocabulary"
(arXiv 2602.13836) is exactly this. Small, but it is 15% of propose and it is
self-contained.

## Idea 3: adaptive verification for MoE

"Making Every Verified Token Count: Adaptive Verification for MoE Speculative
Decoding" (arXiv 2605.00342) targets our exact configuration -- speculative
decode where the verify cost is dominated by sparse-MoE expert traffic. Worth
reading before any further verify-side work; it is the one paper found that
addresses the cost structure T23 measured rather than the generic case.

## Idea 4: lookahead decoding (noted, probably not for us)

llama.cpp's `examples/lookahead` generates n-grams by Jacobi iteration on the
target model itself, needing no drafter. It costs *target* compute, and T23
showed this pair is already 77% GPU-busy during decode, so there is no capacity
to spend. Listed for completeness.

## Recommendation

**Idea 1 is the one worth building.** It is the only proposal that removes the
40 ms term rather than shaving it, it reuses a verify path that is already
correct and lockstep-safe, it has two reference implementations to copy, and it
can be gated so it never costs anything when it does not fire.

Expected outcome is workload-dependent by construction: a clear win on code and
repetitive generation, neutral on prose. That is a better risk profile than
anything left on the engine side, all of which was measured to parity at best.

---

# BUILT AND MEASURED: it loses, and the reason is a cap nobody had found

`DS4_DSPARK_NGRAM_DRAFT=1` implements prompt-lookup drafting in
`ds4_session_prepare_dspark_draft`, ahead of the DSpark chain, with
`DS4_DSPARK_NGRAM_MIN/MAX` controlling match length and
`DS4_DSPARK_NGRAM_DRAFT_MAX` the draft cap. A miss falls through to the model.
It works mechanically and costs nothing to run.

## Results

| configuration | tg | blocks | accepted | verify total |
|---|---|---|---|---|
| baseline (no spec) | **22.01** | -- | -- | -- |
| ngram, min=3, counting prompt | 8.21 | 65 | 9 | 14.7 s |
| ngram, min=8, verbatim-repeat prompt | 13.73 | 26 | 19 | 5.6 s |

Both lose badly. Two separate lessons.

**My first prompt was wrong.** "Item 1: alpha. Item 2: alpha..." is a *counting*
pattern: the suffix `alpha.\nItem` matches, and lookup then predicts the number
that followed last time, which is always wrong. Lookup fails precisely at the
varying token. A min match of 3 also fires on meaningless prose matches -- 65
verifies for 9 accepted tokens.

**Raising the gate helps but does not fix it.** min=8 on verbatim-repeating
output cut verifies from 65 to 26 and roughly doubled acceptance per block, but
still only **0.73 accepted tokens per verify** against the 1.14 break-even.

## The real blocker: the 8-row verify cap

Propose is now free, so the binding constraint moved to **accepted tokens per
verify**. And the economics of a *long* accepted run are excellent:

```
verify(K) ~= 60 + 17K          (post Stage 1)
K=7  -> 179 ms for 8 tokens  = 22.4 ms/token  = 45 t/s
```

Lookup decoding earns its keep on long verbatim spans -- ten or twenty tokens
copied from the prompt -- which is exactly where those numbers land. **But the
draft can never be that long here:** `DS4_TP_BATCH_MAX_ROWS = 8` caps verify
rows, so at most ~7 draft tokens can ever be checked in one block, and
`ds4_tp_batch_block_begin` rejects anything above it.

So the configuration permits only the short drafts at which lookup is weakest,
and forbids the long ones at which it is strongest. That cap is a small
compile-time constant, but it sizes the RDMA slab on both ranks, so raising it
is a real change rather than a knob.

## What would make this line work

1. **Raise `DS4_TP_BATCH_MAX_ROWS`** to 16 or 32 and re-measure. This is the
   precondition; without it lookup cannot express the drafts that pay.
2. **Test on the actual use case.** Neither prompt here was it. Lookup decoding
   is for regenerating text that largely exists in the context -- "rewrite this
   function with X changed", diff-style edits, long quotation. Synthetic
   repetition with a counter is close to a worst case.
3. Keep the strict gate: min=8 was clearly better than min=3, and the cost of a
   wrong draft is a whole verify.

## Status

Implemented, off by default, measured negative at the current row cap. The code
is small and self-contained; the finding that matters is that **the 8-row verify
cap, not propose cost, is what stops a free drafter from paying.**

---

# RESULT: the first configuration that beats baseline

## The earlier measurements were my bug, not the idea

All three n-gram runs above were invalid. The caller prepends the anchor
itself:

```c
memmove(s->dspark_draft_tokens + 1, s->dspark_draft_tokens, ...);
s->dspark_draft_tokens[0] = first_token;
s->dspark_draft_len++;
```

My lookup also wrote `token` at `out[0]`, so every draft carried a **duplicated
anchor** and mismatched at row 1 by construction. That is the whole explanation
for `accepted_len_hist` being almost entirely zeros in those runs. The drafter
must return the continuation only.

With that fixed, `miss_first` goes from nonzero to **0**.

## Measured, matched prompts, same binary and config

| workload | baseline | + lookup drafting | delta |
|---|---|---|---|
| verbatim copy (260 tok) | 22.45 | **24.43** | **+8.8%** |
| prose (128 tok) | 23.14 | 21.72 | -6% |

On the copy task:

```
cycles=124 for 260 tokens = 2.10 tokens per target step
miss_first=0
draft_len_hist    14:11      (11 blocks at the 14-token cap)
accepted_len_hist 14:6       (6 blocks accepted ALL 14)
accepted_draft=131 over 15 blocks = 8.7 accepted per block
```

**Six blocks accepted fourteen tokens each.** That is the regime the whole
analysis said was needed and that nothing else reached.

## Raising the row cap was a real precondition

`DS4_TP_BATCH_MAX_ROWS` 8 -> 16 (and a hardcoded `rows > 8` in ds4_cuda.cu that
shadowed it, now a named constant). At 8 rows the 14-token drafts above cannot
exist; the feature could only express the short drafts where lookup is weakest.
The cap sizes the registered slab -- 40 layers x rows x 20 KiB x 2, so 26 MiB at
16 rows -- and is otherwise free.

## A miss must decline, not fall through

Falling through to DSpark after a lookup miss costs the model's ~40 ms propose
on top of a lookup that already said no: prose measured **19.28** that way. With
a miss declining the cycle outright, prose recovers to **21.72**.

The residual 6% is one mispredicted 14-row block costing 337 ms in a 5.9 s run.
A stricter gate (longer minimum match, or a shorter draft when the match is
marginal) should close most of it; `DS4_DSPARK_NGRAM_HYBRID=1` restores the old
fallthrough for comparison.

## Status and recommendation

Off by default. Enable with:

```
DS4_DSPARK_NGRAM_DRAFT=1 DS4_DSPARK_NGRAM_MIN=8 DS4_DSPARK_NGRAM_MAX=16 \
DS4_DSPARK_NGRAM_DRAFT_MAX=14 DS4_DSPARK_MIN_VERIFY_DRAFTS=4
```

This is worth having for copy-heavy work -- regenerating a file with edits,
reproducing quoted text, structured output that echoes the prompt -- and should
stay off for open-ended prose until the gate is tightened. It is also the only
thing in T22-T25 that beat the baseline at all, which is a reasonable argument
for tightening the gate rather than shelving it.
