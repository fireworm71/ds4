# T22: why speculative decode is slower than baseline on the pair

Measured 2026-09-19/20 on the Spark pair, Q2 TP2 resident, 46-token prompt,
128 generated tokens, greedy. Continues antigravity's work
(`7f7e437..5a7985f`).

## 1. Where we are

```
target-only baseline          23.14 t/s
--dspark, confidence 0.7      20.04 t/s      (-13%)
```

Reproduces antigravity's 20.35. Speculation is a net loss.

## 2. It is not a correctness problem

I previously called the confidence-0.3 divergence "corruption". **That was
wrong.** The text is fluent and on-topic; it diverges from the baseline only
where a near-tie flips, which is the documented behaviour of non-strict
direct-commit mode ("output may differ from one-token decode due to batched
floating-point operation order").

Two checks back this up:

* `DS4_DS41_VERIFY_POSLOG=1` on both ranks: every commit, including partial
  ones, reports identical `pos0/keep/rows/pos` and an identical KV fingerprint
  on rank 0 and rank 1. The pair stays in lockstep through rollbacks.
* The single-box rollback invariant still passes on current code:
  `arrays_differing=0 of 56` for a 5-row verify committing one row.

## 3. The confidence sweep: 0.7 is already optimal

| confidence | tg | blocks verified | accepted | accepted/block | verify each |
|---|---|---|---|---|---|
| 0.7 (default) | **20.04** | 2 | 7 | **3.5** | 196 ms |
| 0.5, min-verify 4 | 16.37 | 10 | 14 | 1.4 | 235 ms |
| 0.3 | 14.78 | 15 | 16 | 1.1 | 222 ms |

The drafter's confidence is **well calibrated**: at 0.7 it is right 3.5 times
out of 4, and every relaxation buys blocks that are mostly rejected. There is
no threshold that produces more *useful* drafts. Loosening it is strictly
worse.

So the drafter delivers roughly two usable blocks per 128 tokens, and that is
what it has to give.

## 4. The verify sweep is the structural problem

`DS4_DSPARK_VERIFY_TIMING=1`, rank 0, at confidence 0.3 (15 samples):

```
count=4  total=169.25  sweep=157.59  proj=9.74   logits_rdma=1.93
count=4  total=148.94  sweep=137.87  proj=9.67   logits_rdma=1.41
count=5  total=181.66  sweep=166.21  proj=11.15  logits_rdma=4.30
count=6  total=194.37  sweep=179.89  proj=12.74  logits_rdma=1.74
```

The sweep is 90% of it. Fitting: **~100 ms fixed + ~15 ms per row.**

Against a CUDA-graphed decode at **43.5 ms/token**, that is the whole story:

| block | verify cost | cost to just decode those tokens | saving |
|---|---|---|---|
| K=1 (2 tokens) | ~130 ms | 87 ms | **-43 ms (worse)** |
| K=2 (3 tokens) | ~140 ms | 131 ms | -9 ms (worse) |
| K=4 (5 tokens) | ~166 ms | 217 ms | +51 ms |

A batched verify is *barely* cheaper than decoding the same tokens
sequentially, and for short blocks it is more expensive. That is why
`DS4_DSPARK_MIN_VERIFY_DRAFTS` defaults to 3 -- and why the drafter's abundant
1- and 2-token drafts are thrown away instead of being wins.

## 5. The ceiling, with today's sweep

Even with perfect acceptance on every block, a fully accepted 5-token block
saves ~51 ms against 217 ms of decode. Over 128 tokens that is at most about
**29 t/s**, and only if every cycle produced a full block -- which the drafter
does not do.

With the drafter's actual output (2 usable blocks / 128 tokens), the arithmetic
is unforgiving:

```
2 blocks save   ~305 ms of decode
2 verifies cost  ~391 ms
110 cycles of wasted propose (5.7 ms each) cost ~627 ms
```

Speculation loses before it starts. Removing **all** propose waste still only
gets to ~22.2 t/s, below the 23.14 baseline, because the two verifies alone
cost more than the decode they replace.

## 6. What would actually change this

**Give the verify sweep the decode path's launch profile.** Decode runs 40
layers in 43.5 ms because `ds41_decode_island` captures them into CUDA graphs;
the sweep runs the same 40 layers eagerly and pays ~100 ms of fixed launch
overhead -- about 1.75 ms per layer that the graph does not pay. Antigravity
already removed the synchronisation component (80 drains -> 3, commit
`a1cbaea`); what remains is per-kernel launch cost.

If the sweep's fixed cost approached the graphed path's, then:

```
K=1 (2 tokens)  ~60 ms   vs 87 ms decode    -> profitable
K=4 (5 tokens)  ~105 ms  vs 217 ms decode   -> 2x
```

Every draft the drafter produces becomes profitable, `MIN_VERIFY_DRAFTS` can
drop to 1, and the drafter's frequent short drafts stop being discarded. That
is the difference between "speculation cannot pay here" and "speculation pays
on most cycles".

**This is the one lever worth funding.** Threshold tuning is exhausted (section
3), propose-waste removal alone is insufficient (section 5), and the drafter
itself is well calibrated.

## 7. Secondary, already banked

`cbad47a` fixes short-prompt prefill on a resident pair: a prompt under 256
tokens was cut into 8-row micro-batches, costing `ceil(n/8)` RDMA expert
exchanges per layer instead of one. 192-token prefill **23.56 -> 37.16 t/s**,
output byte-identical. Only matters below 256 tokens; prefill above that
already scales.

## 8. Operational note

Any flag that changes batch shape -- `DS4_METAL_DISABLE_V41_TP_SMALL_PREFILL`
among them -- must be set on **both** ranks. Setting it on one side only
desynchronises the pair and fails with `tp: prefill synchronization failed`.
