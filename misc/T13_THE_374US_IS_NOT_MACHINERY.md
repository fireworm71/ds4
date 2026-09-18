# T13: the streaming tax is disk I/O, not a host round trip. Tiers 2 and 3 are cancelled.

Measured 2026-09-18, Q2 TP2 streaming, 16K/512, ctx-alloc 32768, RoCE RDMA,
`DS4_CUDA_EXPERT_CACHE_STATS=1` with per-phase timers added to
`cuda_stream_selected_cache_begin_load` and `ds4_gpu_tensor_read`.

## The claim under test

`PLAN_TG_NEXT_LEVERS` opens with a structural fact inherited from prep §26/§27:

> The streaming machinery costs **~374 us per layer per token** regardless of
> hit rate -- it is the host<->GPU round trip for the router's expert ids, not
> transfer overhead. ... Per-call optimisation is CLOSED.

T12 built the first attack on that round trip (making the `begin_load` stream
sync conditional) and measured a null. This is the follow-up that asks where
the time actually goes instead of arguing about it.

## Measured attribution, per layer per token

Decode ran at 16.13 tg = 61.7 ms/token over 40 layers = **1.543 ms per layer
per token**. Against that budget:

| component | ms/layer/token | share |
|---|---|---|
| `begin_load` machinery (pure-hit call) | **0.004** | **0.26%** |
| `ds4_gpu_tensor_read` D2H, x2 per layer | 0.026 | 1.69% |
| `begin_load` stream sync | **0.000** | **0.00%** |
| *prep §26/§27 claimed round trip* | *0.374* | *24.24%* |

The two candidate drains together are **30 us**, not 374. There is no per-call
machinery tax to remove.

## Where the time actually is

```
begin_load split: pure-hit calls=19971  0.004 ms/call
                  fetching calls=  428 30.623 ms/call
```

* 97.9% of calls are pure hits and cost **4 us**.
* 2.1% of calls fetch, and cost **30.6 ms** each.
* Weighted mean 0.646 ms/call, which reproduces the measured body exactly.

Those 428 calls account for **13.1 s of the 31.7 s decode -- 41% of it**. Each
moves 103 MiB at an effective **3.52 GB/s**, which is disk bandwidth. The cost
is bytes moved, full stop.

## Consequences

**Tier 2 is cancelled.** Keeping the router ids on-device via a probe kernel
and a device-side slot table would remove the 0.026 ms D2H: **1.7%** of a
decode step, and only if the rest were free. It was aimed at a cost that is not
there.

**Tier 3 is cancelled** for the same reason -- it targeted the same round trip.

**Tier 1 is kept and explained.** T12's conditional sync measured null because
the sync is 0.000 ms. It remains correct (logits byte-identical) and free, but
it is bookkeeping, not a win.

**The lever is hit rate, which is Lever 2 of the original plan.** Every avoided
miss is worth ~30.6 ms of fetching amortised across the calls that would have
performed it. Lever 2 (cache policy: `DS4_CUDA_EXPERT_EVICT`, `_LIVES`, CLOCK
vs LRU) was retired against a +2.0pp hit-rate gate on the premise that decode
misses never repeat -- a premise §24 later disproved by measuring 2.38 misses
per distinct expert. It should be re-run, and it is now the only per-call lever
with measured headroom.

**One earlier statement of mine needs correcting.** I said faster storage would
not help because the penalty was machinery. On this measurement it plainly
would: the fetching path is disk-bandwidth-bound at 3.52 GB/s.

## The tension with §26, stated honestly rather than resolved

§26 measured Q2 TP2 streaming at a **99.3%** hit rate touching disk for only
**1.2 s** across 512 tokens, and still saw a ~25% penalty against resident. In
that regime fetching cannot explain the gap, and my instrument says the two
drains are 30 us, so it is not those either. Both cannot be complete.

Candidate explanations, none verified here:

* different cache warmth -- this run averaged 92.7% hit including the cold fill
  and moved 43 GiB, a materially more miss-heavy regime than §26's;
* §26 and §27 predate the sm_75 build defect being found, and the campaign has
  already voided one day of measurements to that cause;
* the 1.2 s disk figure may count something narrower than the 13.1 s of
  wall-clock these fetching calls occupy.

What is not in doubt is this run: **at 92.7% hit, machinery is 4 us and disk is
41% of decode.** The claim that should not survive is the categorical one --
"~374 us per layer per token regardless of hit rate" -- because the measured
per-call cost is emphatically *not* independent of hit rate. It is ~0 on a hit
and ~30.6 ms on a fetch.

## Instrumentation

Left in place behind the existing `DS4_CUDA_EXPERT_CACHE_STATS` flag:
`begin_load` body/sync timing, the hit/fetch split, and `ds4_gpu_tensor_read`
D2H timing. Zero cost when the flag is unset. Re-running the §26 configuration
with these counters is the cheapest way to close the tension above.
