# T17: per-layer partitioning takes the decode hit rate from 63.5% to 95.9% and buys no speed at all

Measured 2026-09-18, Q4 TP2 streaming over RoCE RDMA, 16K/512, `ctx-alloc
32768`, one binary, both ranks on matching sha256, `sm_121a` 9/9. Probe
disabled in both arms so partitioning is measured alone.

## The diagnosis

Decode walks layers 0..39 for every token, so expert access is **cyclic over a
working set larger than the cache**: Q4 TP2 has 7,680 owned experts competing
for 3,940 slots in one global pool. Cyclic access over capacity is the pattern
LRU handles worst -- by the time a token returns to layer 0, the other 39
layers' traffic has evicted its experts, and every layer keeps paying for the
others. The engine's own hot-set report sized the loss: a 61-expert working set
per layer would cover 86.8% of decode selections where global LRU achieved
63.5%.

## The change

`DS4_CUDA_EXPERT_PARTITION` bounds each layer to `slots / 40` = 98 slots, so a
layer's experts compete only with their own layer across tokens. A slot is a
legal victim when partitioning is off, the slot is empty (refusing free
capacity would be strictly worse), the layer is under quota, or the slot
belongs to that layer. Reuses `slot.layer` and `g_layer_resident_count[]`,
which the probe work had already added.

A mandatory unrestricted fallback runs if the partition leaves nothing takeable
-- every slot of that layer resolved as a hit this call -- counted in
`g_partition_fallbacks`. The pre-existing principle in this code is that a
policy must never be the reason a layer fails where plain LRU would have
succeeded, and that is preserved.

## Result

| metric | control | partitioned | change |
|---|---|---|---|
| decode hit rate | 63.5% | **95.9%** | **+32.4 pp** |
| decode fetched | 54.71 GiB | **45.74 GiB** | -16.4% |
| decode load time | 10.019 s | **7.999 s** | **-2.02 s** |
| decode per_call | 4.140 ms | 0.391 ms | -91% |
| prefill pp | 167.94 | 168.43 | +0.3% |
| **decode tg** | **10.15** | **10.06** | **-0.9%** |

The cache policy result is emphatic and the throughput result is nil. Hit rate
also beat the hot-set bound that motivated the work: partitioning reached 95.9%
where the report predicted 86.8% for a 61-expert set, because a 98-slot
partition holds more than the top 61.

Prefill was not harmed, which was the stated risk -- a prefill chunk touches
most of a layer's 192 experts and a 98-slot quota could have thrashed within
the partition. It did not show up in pp.

## Why the speed did not follow, and what that closes

2.02 s saved from a 50.9 s decode is 4% that never appeared. The only
consistent reading is that **decode expert fetching is already overlapped with
compute** -- the fetch threads run concurrently and are not on the critical
path. Removing I/O from a path that was not waiting on I/O buys nothing.

Combined with T16, two independent attacks on two different suspected
bottlenecks both produced nothing:

* **T16** removed the per-layer host sync on 88% of calls (device-side probe).
  Result: -8.2%, and the sync it removed was mostly the host *observing*
  attention work that happens regardless.
* **T17** removed 16% of decode disk traffic by fixing the eviction policy.
  Result: nil, because that traffic was already overlapped.

So Q4 TP2 decode at ~10 t/s is bound by **GPU compute plus the unavoidable
layer-serial structure**, not by expert-cache machinery and not by expert
fetching. There is no per-call lever left here. T13's framing ("the lever is
hit rate") is now also refuted -- by fixing the hit rate and observing nothing.

## Where that leaves the campaign

The original `PLAN_TG_NEXT_LEVERS` was right that the only remaining attacks
are the ones that **amortise** the fixed per-token cost rather than reduce it:

* **Lever 1, speculative decoding.** Verifies k drafted tokens per cycle, so
  the 40 layer-serial steps are paid once for k tokens instead of once each.
  This is the only lever that changes the structure rather than the constant.
  Its prerequisites are already written up: the drafter ring-capture fix and
  the state oracle (see the T-series notes on `b0cbc30`).
* **Lever 4, batched serving.** Amortises the same cost across concurrent
  sessions. Upstream's `session_concurrency_bench` (`147b263`) is the harness
  and it now links after the `quality-score` fix.

## Disposition

**Keep partitioning, default it on, but claim only what it delivers.** It is a
large, clean win on cache behaviour -- 95.9% versus 63.5% hit rate, 16% less
decode I/O, 91% less per-call fetch time -- and it costs nothing measurable in
either phase. That matters for disk wear, for power, and for any configuration
whose fetching is *not* overlapped (single box, or a slower disk). It is not a
throughput optimisation on this pair and should not be described as one.

**Do not spend further effort on the streaming expert cache.** Two measured
nulls bracket it: machinery is 4 us on a hit (T13) and I/O is overlapped
(T17).


---

# CORRECTION (same day): two claims in this note were wrong

## 1. "Default it on" -- it is not, and it should not be

The Disposition above says to default partitioning on. The code does not: the
quota is opt-in via `getenv("DS4_CUDA_EXPERT_PARTITION")`. The commit message
for this work described a disposition the code never implemented. Leaving it
opt-in is now the *correct* answer for the reason below, so the code stays as
written and this note is the correction.

## 2. "Helps any configuration whose fetching is not overlapped" -- refuted

That sentence was reasoning, not measurement. Tested on Q4 **single box**
streaming, 16K/512, fetch threads 8 -- the most favourable case available,
because decode I/O there is **43.4 s of an 88.9 s decode (49%)** against only
20% under TP2:

| metric | control | partitioned | change |
|---|---|---|---|
| decode hit rate | 84.5% | **84.1%** | **-0.4 pp** |
| decode fetched | 353.73 GiB | **363.13 GiB** | **+2.7%** |
| decode load time | 43.378 s | 43.445 s | +0.2% |
| pp | 147.63 | 146.86 | -0.5% |
| tg | 5.76 | 5.76 | 0.0% |

Partitioning made the hit rate marginally **worse** and moved slightly **more**
bytes. The claim does not hold.

## Why, and the rule that actually governs it

Partitioning pays only when the per-layer quota is at least the layer's hot-set
size, and costs a little when it is not:

| configuration | experts/layer owned | quota | selections/call | outcome |
|---|---|---|---|---|
| Q4 **TP2** | 192 | 98 | ~3 (rank owns half the 6) | 63.5% -> **95.9%** |
| Q4 **single box** | 384 | 99 | 6 | 84.5% -> 84.1% |

Under TP2 the ownership filter halves both the expert population and the
per-call demand, so a 98-slot quota comfortably covers the layer's working set
(the hot-set report put the top 61 at ~94% coverage). On a single box the same
99 slots face 384 experts and twice the per-call demand, so the quota binds
tighter than the hot set and simply removes the flexibility global LRU had to
let a busy layer borrow slots from an idle one.

So the honest scope is narrow: **partitioning is a large cache-behaviour win
for network TP specifically, worth nothing for throughput anywhere measured,
and mildly counterproductive single box.** Opt-in is right.

## What this test does *not* show

It does not confirm or deny the overlap explanation for T17's null. The two
single-box arms had nearly identical hit rates (84.5 vs 84.1), so there was no
I/O change for throughput to respond to. Whether decode fetching is overlapped
with compute remains inferred from the TP2 result alone, and the direct
measurement -- GPU kernel time via `DS4_METAL_DECODE_STAGE_PROFILE` -- is still
owed. The compute-bound conclusion in T16/T17 rests on two negatives and should
be treated as provisional until someone measures the positive.
