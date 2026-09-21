# T28: review of the four-track speculative plan

Reviewing `implementation_plan.md`. Two tracks are sound, one rests on an
arithmetic error, and one is sequenced dangerously. Written while the plan is
being executed, so the Track 3 item is the urgent one.

## Track 3 is off by ~3x -- do not spend a kernel project on it

The plan states:

> `prop_logits` (3.3 ms) and `prop_markov` (2.7 ms) calculate dot products over
> the full 256,960 vocabulary. [...] This drops propose latency from ~40 ms down
> to ~20 ms, lowering the break-even accept threshold from 63% to ~35%.

**The two numbers it quotes sum to 6.0 ms, not 20 ms.** T23 profiled propose
with `DS4_DSPARK_STAGE_PROFILE=1` over 9 stage executions:

| part | share of chain |
|---|---|
| **ffn** (128-expert MoE) | 55% |
| **q_path** (Q projection) | 35% |
| attn_output_hc | 10% |
| attention, norms, next_input | <1% |

The ~40 ms of propose is ~30 ms of stage chain (dominated by ffn and q_path --
pure matmul width) plus `prop_logits` 3.3 and `prop_markov` 2.7. **Trimming the
vocabulary projections can only touch that 6 ms.** T23 sized this item
explicitly: "Stage 4b -- top-k the vocabulary projections (cheap) ... Expect
-5 ms."

So Track 3 delivers propose 40 -> ~34-35 ms, not 40 -> 20 ms. The derived claim
that break-even falls from 63% to ~35% does not follow; it moves to roughly 55%.

Track 3 is still worth doing as a cheap win. It is **not** worth doing as a
kernel project with a 2x latency target, and nothing downstream should be
planned against a 20 ms propose.

## Track 4 must not precede Track 1's proof

Track 4 sets `DS4_DSPARK_NGRAM_DRAFT_MAX` to a default of **48**, and Phase 1
runs Tracks 1 and 4 together. The walkthrough's own Track 3 benchmarks argue
against this:

```
C refactor   cap=64 -> 21.14 t/s     cap=32 -> 22.25 t/s     baseline 22.08
JSON extract cap=32 -> 15.00 t/s                             baseline 21.60
```

Long drafts are already measured as *negative* on code at cap=64 and heavily
negative on JSON. Defaulting to 48 ships that regression to every user.

Sequence it the other way: land Track 1 (confidence gating), prove the JSON case
is no longer a regression, and only then raise the default. Track 1 is the fix;
Track 4 is the victory lap.

Same caution for `DS4_DSPARK_NGRAM_HYBRID` defaulting to `auto`: T25 measured
hybrid fallthrough at **-17% on prose** (19.28 against 23.14). Whatever `auto`
resolves to must not be "on" for prose.

## The verification plan is missing its most important case

Benchmark 3 covers prose downside protection. **The known regression is JSON,
not prose** -- prose is already neutral at 21.86. Add the JSON extraction case
as a gate: Track 1 succeeds only if `json_extract_chat.txt` returns to >= 21.60
t/s. That is the number that decides whether confidence gating works.

## Two process items

1. **Disable the dynamic n-gram tier for every benchmark.** It is currently
   enabled implicitly whenever `DS4_DSPARK_NGRAM_DRAFT` is set (ds4.c:58165) and
   persists to `~/.cache/ds4/ngram_cache.bin` via `atexit`. Runs therefore
   inherit state from previous runs and are not reproducible. Set
   `DS4_NGRAM_DYNAMIC=0` in all A/B measurements, or clear the file between runs.
2. **Re-fit the verify cost model.** The plan assumes verify(K=48) ~= 440 ms.
   The T23 model (60 fixed + 17/row) predicts ~876 ms. Direct RDMA slab staging
   plausibly cut the per-row term, but the gap is 2x and every economic
   argument in the plan rests on it. T23's protocol requires re-fitting
   `fixed + per-row` after each stage; that has not been done since Stage 2.

## What is good

**Track 1 is the right first move** and is well specified. Sizing draft length by
continuation confidence rather than prefix depth alone is exactly what the
current code lacks -- `ds4_session_ngram_draft` modulates `allowed` by `depth`
and streak state, neither of which knows whether the n-gram table held one
continuation or five. The early-stop on `top_count/total_count < 0.60` is the
mechanism the JSON case needs.

**The cost-asymmetry framing is correct** and is the right mental model:
a K=48 block is ~75 t/s effective when it commits 33 and ~2.3 t/s when it
commits 1. Matching depth to certainty is the whole game.

**Track 2 is plausible but rank it last.** A static corpus only pays if the model
emits that exact boilerplate; otherwise it raises attempt rate into verify time
with no acceptance, which is the JSON failure mode again. It also must be
tokenized with ds4's own tokenizer -- a generic tokenizer produces token ids that
are silently wrong for this vocabulary.

## Suggested order

1. Track 1, gated on JSON >= 21.60 t/s.
2. Track 3, budgeted at -5 ms, not -20 ms.
3. Track 4 defaults, only after 1 passes.
4. Track 2.

---

# URGENT: the batch-invariance bug is known, and it invalidates the benchmark series

The `ROUTER_MAX_QUEues` divergence just found is **T21 Phase 16/17**, already
diagnosed and recorded. Two consequences follow, and they matter more than the
bug itself.

## 1. SERIAL_VERIFY makes it correct and pointless

T21, verbatim:

> With an invariant kernel, `SERIAL_VERIFY` can be dropped, the verify batches
> again, and the measured 1.23 tokens per target step turns into real
> throughput. **Without it, V4.1 speculative decode is correct but pointless**,
> and target-only at 21.55 tg/s remains the answer.

Measured in that mode:

```
cycles=104  proposed=90  accepted_draft=4  full=17  partial=3
greedy output: BYTE-IDENTICAL to no-speculation
```

**4 accepted draft tokens over 104 cycles.** Serial verify sweeps one row at a
time, so each row costs a decode-equivalent pass and the batching that made
verify cheap (0.317 ms/gate-row against decode's 0.426) is gone. Correctness and
the speedup are, today, mutually exclusive.

## 2. Therefore the 24.43 -> 29.75 series was measured in the divergent mode

`DS4_DS41_SERIAL_VERIFY` is off by default (ds4.c:41915), and the report above
states the TP implementation of that path was broken in three ways
(`row_preds` never assigned, `logits_half` never recorded). A broken path cannot
have been enabled during the benchmark series. So every speculative number from
24.43 through 29.75 -- **including my own T25 lookup-drafter results** -- was
produced by the batched verifier that commits its own non-invariant argmax.

`ROUTER_MAX_QUEues` is what that looks like when you read the output instead of
the throughput counter. The walkthrough's "greedy bit-exact decoding" and
"bit-for-bit identical in reasoning" claims were not tested against a
target-only baseline.

**Every speculative throughput number needs re-measuring once correctness is
enforced, and should be expected to collapse toward target-only.**

## 3. Do not reach for CORE_INVARIANT as a cheaper fix

The obvious idea -- keep MoE and HC batched, use the per-row attention fallback
(`DS4_DS41_CORE_INVARIANT=1`) -- was already tried and does not work. T21: the
serial path "unlike the per-row attention fallback [...] keeps
`ds41_attention_publish_batch`, which is where the ratio-2 carry snapshot is
taken," and "today only the serial path reaches zero."

## 4. The real remaining task

Make `ds41_attention_batch` batch-size invariant. T21 calls it "a bounded CUDA
question -- likely GEMM tiling or reduction order in the attention batch path --
and it is the only thing left between here and a working feature."

There is already a one-run differential harness for it, single box, no drafter
and no second machine:

```
DS4_DS41_UNIFY_DECODE=1 DS4_DS41_VERIFY_STATE_DIFF=3     # must print arrays_differing=0
                                                         # WITHOUT SERIAL_VERIFY
```

That is the test any candidate kernel fix must pass. It is also far cheaper than
another cluster benchmark round.

## 5. Track 5 is unaffected

The argmax-only logits exchange is orthogonal and still valid -- it is a
transport win worth ~65 ms on a K=48 verify (~7% of the T23 model's ~876 ms),
independent of which verify path runs. Worth finishing. Just do not expect it to
matter until the kernel question above is settled.
