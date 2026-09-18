# Plan: remaining tg levers for V4.1 on the Spark pair

Handoff document, self-contained. Written 2026-09-17, after the Q4 TP2
streaming campaign (`misc/Q4_TP2_STREAMING_PREP.md`, sections 1-27) and the
corrected RDMA tier campaign. Everything here assumes the **correct build**:
`make cuda-spark` or explicit `CUDA_ARCH=sm_121`, verified with
`cuobjdump -lelf ds4-bench | grep -c sm_121a` after EVERY rebuild, both boxes.

## Where things stand (all measured, correct build, 16K prompt / 512 decode)

| configuration | pp | tg | notes |
|---|---|---|---|
| Q4 single box, threads 8 | 147 | 5.95 | streams 351 GiB/run of decode reads |
| Q4 single box + RDMA tier, t8 | 153.81 | 6.36 | tier cell provisional (single arm) |
| **Q4 TP2 streaming** | **174.12** (t8) | **10.31** (t1) | the capacity win; each rank streams its half |
| Q2 TP2 streaming | 319.23 | 16.54 | 99.3% hit -- wrong configuration, see next |
| **Q2 TP2 resident** | **412.87** | **21.94** | drop `--ssd-streaming`; shard fits |

Structural facts that shape everything below:

* The streaming machinery costs **~374 us per layer per token** regardless of
  hit rate -- it is the host<->GPU round trip for the router's expert ids, not
  transfer overhead. Three optimisation attempts (lazy sync, allocation reuse,
  pinned async both directions) were all **neutral**, verified with an
  interleaved control (section 27). Per-call optimisation is CLOSED.
* Decode misses repeat: **2.38 misses per distinct expert** at Q4 (section 24).
  Every "no reuse" null in the old campaign was measured on the broken sm_75
  build and is void.
* Fetch depth: knee at **8** for prefill at both quants; decode wants **1**
  under TP2. Prefetchers (recurrence AND router-prediction) are null. CLOSED.
* tg drift within a warm session is ~2.2%; pp drift ~0.3%. Interleave arms and
  repeat the opening arm, always.

## DEAD -- do not re-try without new information

* ~~Q3splice TP2 resident~~ -- the Q3splice GGUF is broken (Q2 and Q4 experts
  shuffled together, per Jason 2026-09-17). A real Q3 quant would make this
  the best quality/speed point on the pair (per-rank shard ~91 GiB, fits
  resident), but no valid artifact exists. Revisit ONLY if a clean Q3 is
  built -- and building one may be worth it for exactly this reason.
* begin_load / slot-machinery micro-optimisation (section 27, structural).
* Fetch-thread tuning beyond {prefill 8, decode 1}; any prefetcher.
* dm-cache / rambox block composites (6-8x slower, prior campaign).

## Lever 1 -- Speculative decoding (DSpark/MTP) under TP2

**The only lever that attacks the 374 us structurally.** The cost is 40 host
syncs per TOKEN; spec decode verifies k drafted tokens per cycle, dividing the
fixed per-token cost by the acceptance rate. Even the current measured
acceptance (1.18 tokens/cycle on the spark-opt DSpark work) would be ~+15% tg
from amortisation alone -- and verify batches also route 6 experts x k tokens
per layer in ONE begin_load call, so the streaming overhead amortises too.

* `ds4_tp.c` explicitly permits drafting on the leader: the verify block is
  mirrored to the worker via `DS4_TP_FRAME_VERIFY`.
* The V4.1 DSpark chain exists on `spark-opt` (drafts accepted and committed,
  verify path ported); it has never been run under TP2 or with streaming.
* CONFLICT TO RESOLVE FIRST: docs/DISTRIBUTED.md says "Vision, DSpark and
  other model/quant layouts are not supported by CUDA network TP", while the
  ds4_tp.c comment says drafting is allowed on the leader. One of them is
  stale; find out which before building anything.
* Risk: acceptance measured on single box may drop under TP2 gate latency.

Measure: Q4 TP2, story 16K/512, spec on/off interleaved. Success = tg > 11.5.

## Lever 2 -- Re-run the cache-policy campaign on the correct build

Eleven policies were retired against a +2.0pp hit-rate gate on the premise
that misses never repeat. That premise is now known false (2.38x reuse). CLOCK
lives=3 reached +0.56pp on the broken build; with real recurrence it may
clear the gate.

* Each +1pp of Q4 decode hit rate is ~1.9 GiB fewer reads per 512 tokens.
* Patch surface already exists (`DS4_CUDA_EXPERT_EVICT`, `_LIVES`).
* Run at Q4 single-box (84.5% hit, most headroom) AND Q4 TP2 (96.2%).
* Keep the pre-registered gate discipline: state the threshold before running.

Measure: story 16K/512, LRU vs CLOCK lives={1,3}, interleaved, 2 repeats.

## Lever 3 -- Small cross-tier from TP2's spare RAM

Each Q4 TP2 rank plans ~99 GiB of 121, leaving ~15 GiB idle per box. The
section-24 coverage curve says 20 GiB ideally-placed covers 24.4% of misses;
15 GiB ~19%. The old plan ruled out tier+TP2 for a 104 GiB region and never
considered a small one. Each rank serves its OWN half's hot misses from the
peer's spare RAM at 21.9 GB/s against 7 GB/s local.

* Requires the region server to accept a HOT-SET region (ideal placement),
  not a file prefix -- this is the section-24 "placement is the lever" item.
* Ceiling ~3-4% tg at Q4 TP2. Do after levers 1-2; skip if either lands big.

## Lever 4 -- Batched serving (throughput per box, not latency)

`ds4-server --batched-session 8` amortises the per-layer host sync across
sessions; with >=5 ready sessions the server batches decode across both GPUs
under TP. If the deployment goal is tokens/sec/box rather than single-stream
latency, this sidesteps the structural cost instead of fighting it. Unmeasured
under TP2 streaming. Measure: aggregate tg at batch {1,4,8}, Q4 TP2.

## Lever 5 -- Resident fast path for fitting shards (engineering, not research)

Q2 TP2 streaming pays 33% over resident for nothing (section 26). A load-time
check -- "does this rank's shard fit the cache budget? then map it resident
and skip the streaming path" -- turns the Q2 footgun into the fast
configuration automatically and removes a whole class of misconfiguration. No
new mechanism; it is routing between two paths that both already work.

## Also owed, independent of tg

* The Makefile arch guard: a bare `make ds4-bench` must hard-fail instead of
  silently building sm_75 with MXFP4 compiled out. Ten minutes; prevents the
  defect that voided a day of measurements.
* Remove `DS4_CUDA_TP_STREAMING` once the TP2 streaming path is trusted --
  the flag was an opt-in for landing unvalidated code, and it is now the
  199th env knob. Complexity-minimisation applies.

## Suggested order

1. Lever 5 (small, removes a footgun permanently)
2. Lever 1 (biggest structural ceiling)
3. Lever 2 (cheap, correction of a void result)
4. Levers 3-4 (conditional on the above)
