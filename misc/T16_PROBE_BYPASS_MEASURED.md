# T16: the device-side probe works, is correct, and costs 8% of Q4 decode

Measured 2026-09-18 on the Spark pair (`spark-0fb3` coordinator 10.99.0.1,
`promaxgb10-493d` worker 10.99.0.2), Q4 TP2 streaming over RoCE RDMA, 16K/512,
`ctx-alloc 32768`, one binary for every arm with both ranks on matching
sha256, `sm_121a` verified 9/9 after each build.

Q4 TP2 is the right target: at 151.17 GiB/rank it cannot run resident, so
Lever 5 (T15) cannot promote it and streaming is a physical requirement.

## Result

| arm | pp | tg steady | bypass rate |
|---|---|---|---|
| probe ON, hot spin | 168.20 | 9.30 | 88.18% |
| probe ON, hot spin (repeat) | 167.54 | 9.34 | -- |
| probe ON, spin+`sched_yield` | 168.49 | 9.31 | 88.18% |
| **probe OFF (`DS4_CUDA_DISABLE_PROBE=1`)** | 167.94 | **10.15** | -- |

Probe mean **9.32** against a control of **10.15**: **-8.2% decode**, with
0.4% drift across the three probe arms. Prefill is unchanged (probe is gated
on `n_tokens == 1`), so the effect is decode-only and the sign is unambiguous.

The control validates against the campaign's recorded Q4 TP2 figure of
10.31 tg (-1.6%), the usual offset for builds from this branch, so the
comparison is against a sound baseline rather than a broken one.

**Correctness gate passed.** Q4 TP2, 2048/8, frontier logit dumps with the
probe on and off are **byte-identical**. The probe computes exactly what the
host path computes; this is a performance result, not a correctness one.

## The mechanism works -- that is what makes the result interesting

The bypass fired on **88.18%** of decode calls (18,061 of 20,481). It did
precisely what it was designed to do: skip the `cudaStreamSynchronize` at
`ds4_cuda.cu:25075`, skip the D2H of selected ids, skip the host hash lookup,
skip the H2D of the remap. And decode got **slower**.

An earlier estimate in review put the bypass rate near 7%, reasoning that a
63.5% per-expert hit rate makes all six hitting unlikely. That was wrong: the
63.5% figure is per unique lookup across the whole run including the cold
fill, and the `g_layer_all_resident[]` fast path carries many calls that the
per-expert arithmetic misses. The design's central assumption held.

## The spin was the obvious suspect and it is not the cause

On aarch64 the poll loop used `asm volatile("yield")`, which is only an SMT
hint to the core and does **not** release the CPU to the scheduler. Q4 decode
fetches 54.71 GiB and needs cores for `DS4_CUDA_EXPERT_FETCH_THREADS` and RDMA
progress, so a held core was a plausible explanation -- `cudaStreamSynchronize`
by contrast lets the driver block and frees the core.

Replaced with a hybrid: spin for a tunable budget (`DS4_CUDA_PROBE_SPIN`,
default 2048 iterations) to catch the already-done case, then `sched_yield()`.

Result: **9.31 tg, unchanged.** The hypothesis is refuted and the hybrid is
kept only because it is strictly better behaviour under contention.

## Why the win was always smaller than advertised

T14 reported the sync as 1.032 ms/layer and framed it as the recoverable
prize. But T14's own comparison table puts the whole resident-vs-streaming
penalty at **582 us/layer** (1.78 vs 1.19 ms/layer). Both cannot be overhead:
1.032 > 0.582. Most of the sync wait is the host *observing* GPU work -- the
attention pipeline -- that happens either way. The genuinely recoverable
bubble was at most 582 us/layer, and whatever the probe adds exceeds it.

## Candidates not yet tested

* The probe adds a kernel launch per layer per token (`<<<1,32>>>`), 40 per
  token, on the critical path.
* The 11.8% of calls that miss now pay the probe **and** the sync they were
  meant to replace, so the fallback is strictly more expensive than before.
* Bypassing `begin_load` also skips whatever it does beyond slot lookup --
  `cuda_stream_compact_prefill` and the `g_stream_prefill_ids`/`slots` publish
  among them. If a downstream MoE kernel benefits from that shaping, the hit
  path may be dispatching a slower variant.

The third is the most promising and the cheapest to check: instrument the MoE
kernel time on the bypass path against the host path.

## Disposition

**Default the probe off on Q4.** `DS4_CUDA_DISABLE_PROBE=1` is currently the
faster configuration on the only model that requires streaming, and the
machinery should not ship on by default while that is true.

**Keep the machinery.** It is correct (byte-identical logits), it is the right
architecture, and three of its parts are independently sound and reusable: the
device slot table, the eviction invalidation of that table, and the probe
kernel itself. If the third candidate above turns out to be the cause, the fix
is small and the win becomes real.

**Do not pursue this ahead of Lever 2.** T13 established that hit rate governs
streaming cost; Q4 TP2 decode runs at a 63.5% unique-lookup hit rate and
fetched 54.71 GiB during decode alone. Cache policy has measured headroom that
this path does not.
