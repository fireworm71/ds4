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
