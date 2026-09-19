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
