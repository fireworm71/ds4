# T12: make the per-layer expert sync conditional -- PLAN, then MEASURED NULL

## The cost, restated

Streaming decode pays a fixed toll of **~374 us per layer per token**, which at
40 layers is 14.96 ms/token. That accounts for essentially the whole measured
streaming penalty:

```
Q2 TP2 resident   21.94 tg = 45.58 ms/token
Q2 TP2 streaming  16.54 tg = 60.46 ms/token
                  difference 14.88 ms/token
374 us x 40 layers         = 14.96 ms/token   -> 100.5% of the gap
```

It is **not** bandwidth. Three independent results say so: the streamed run
held a 99.3% hit rate and touched disk for 1.2 s across 512 tokens (prep §26);
lazy sync, allocation reuse and pinned-async-both-directions were all neutral
(prep §27); and adding read-ahead parallelism on top of a depth-8 foreground
fetch *hurt* by 9.6-14.1% (T-readahead). GB10 is an integrated, unified-memory
part, so the "H2D/D2H" crosses no bus at all -- it is pure serialization.

## Where it actually happens

Two drains per layer per token:

1. `ds4.c:22331` -- `ds4_gpu_tensor_read(metal_graph_router_selected(g), ...)`
   pulls 24 bytes of router ids to the host with a blocking `cudaMemcpy`
   (`ds4_cuda.cu:3513`).
2. `ds4_cuda.cu:29637` -- `cudaStreamSynchronize(cuda_decode_stream())` inside
   `cuda_stream_selected_cache_begin_load`, labelled "stream expert reuse wait".

Plus a latent third: `ds4_gpu_tensor_read_after_selected_event`
(`ds4_cuda.cu:36091`) accepts an `event_value`, discards it (`(void)event_value;`)
and calls `cudaDeviceSynchronize()` -- a device-wide drain, heavier than a
stream sync. The fine-grained wait its name promises was never implemented.

## What drain #2 guards, and why it is usually unnecessary

Its label is the answer: slot **reuse**. The host may pick a victim slot and
overwrite it while a previously launched kernel is still reading that slot, so
the stream must be drained before eviction.

But eviction only happens on a miss. Reading `begin_load` end to end:

* ids are validated and folded into `unique` + `remap` -- pure host work on
  `selected_ids`, which the host already holds;
* a resize check may release and reallocate the cache;
* a hit loop resolves each unique expert through `g_stream_expert_by_gate`;
* a **miss** loop picks a victim and calls `cuda_model_copy_to_device_streamed`
  -- the only device writes;
* `uploads.finish()` is a no-op when `uploads.active` was never set.

So when every selected expert is already resident and the cache needs no
resize, **nothing is evicted and nothing is uploaded** -- the drain has no work
to guard. At the hit rates this cache sustains (99.3% at Q2, 96.2% at Q4) that
is almost every call.

## Tier 1 -- the change (this document)

Make drain #2 conditional. Before synchronizing, test whether the call will
take a slot:

```c
static bool cuda_stream_selected_all_resident(table, selected_ids, slot_count);
```

returning true only when the cache config matches (same `model_map`, same
gate/down expert bytes, non-empty slot table) and every non-negative selected
id is already a key in `g_stream_expert_by_gate`. Negative ids are skipped --
under network TP they mark the peer's experts and are carried through the remap
untouched. Then:

```c
if (!cuda_stream_selected_all_resident(table, selected_ids, slot_count)) {
    if (!cuda_ok(cudaStreamSynchronize(cuda_decode_stream()), "stream expert reuse wait"))
        return 0;
}
```

Nothing else moves. The function still rebuilds `unique`/`remap` below (six
elements, trivially cheap), so the fast path adds one hash lookup per selected
expert and removes one full stream drain.

`DS4_CUDA_STRICT_EXPERT_SYNC=1` forces the old unconditional behaviour, so the
arms can be interleaved on one binary -- the same discipline the read-ahead
measurement used.

### Why this is safe

* No eviction on the fast path, so no slot a launched kernel could still be
  reading is reused.
* Slot bookkeeping (`used = stamp`, the CLOCK hand) is host-side only.
* `uploads.finish()` and `cuda_ppf_flush()` are no-ops with nothing queued.
* Ordering of the published mapping is unchanged -- it is issued on the same
  stream, so it still follows the consuming kernels in stream order.

### Residual risk to check by measurement, not assertion

The read-ahead reader (`cuda_stream_prefetch_before_load`) and the decode
predictor can both touch slots outside the foreground path. Both are off by
default, and T-readahead recommended against enabling read-ahead at all, but
the fast path must be verified with them off and the change should be
considered untested with them on.

## Expected result

If drain #2 is roughly half the toll, Q2 TP2 streaming should move from 16.54
toward ~18.8 tg. Removing both drains (Tier 2) would put it at 45.50 ms/token
= **21.98 tg against resident's 21.94** -- parity, which would dissolve the
resident/streaming cliff entirely.

Pre-registered gate for Tier 1: **Q2 TP2 streaming tg > 17.5**, interleaved
against `DS4_CUDA_STRICT_EXPERT_SYNC=1`, opening arm repeated. Below that and
the drain is not where the time goes and Tier 2 needs rethinking rather than
building.

## Correctness gate

Speed without correctness is worthless here. The fast path must not change what
the model emits: same shape, same prompt, compare generated tokens against the
strict-sync arm. A mismatch means a slot was reused under a running kernel,
which is exactly the bug the drain exists to prevent.

## Tiers 2 and 3, not in scope here

* **Tier 2** removes drain #1: keep a device-side id->slot table, resolve the
  mapping in a probe kernel reading the router ids *in device memory*, and set
  a miss flag plus miss list in pinned memory. All-hit needs no host round trip
  at all; a miss is picked up by polling the flag without draining. CUDA 13.0
  is installed, so conditional graph nodes can branch on the device predicate.
* **Tier 3** is structural and specific to this hardware: host and device share
  coherent DRAM, so for a page already in RAM there is nothing to copy. The
  cache could hold pointers into the mapped model instead of copies, making a
  hit a pointer lookup and reducing a true miss to "page not resident" -- a
  host paging concern rather than a per-layer GPU one. Resident mode already
  proves the GPU can consume expert weights this way: 93.02 GiB of shard plus
  5.06 GiB of buffers on a box with ~113 GiB usable only fits because resident
  is zero-copy direct addressing, not a device-side duplicate.


---

# RESULT (measured 2026-09-18): null. Drain #2 is the free half, not the larger half.

Q2 TP2 streaming, 16K/512, ctx-alloc 32768, one binary, both ranks matched
(`sha256 650d0e7c...`), arms interleaved with the opening arm repeated:

| arm | pp | tg steady |
|---|---|---|
| conditional sync | 292.34 | 16.27 |
| **strict sync (control)** | **292.50** | **16.08** |
| conditional sync, repeat | 299.25 | **16.08** |

The repeat lands exactly on the control. The +1.2% in the opening arm was
drift; the honest reading is **no measurable change**. Pre-registered gate was
tg > 17.5 and it is missed decisively.

**Correctness gate passed.** Single box, Q2 streaming, 2048/16 with
`--dump-frontier-logits-dir` under both settings: the frontier logit dumps are
**byte-identical** (1,614,134 bytes each). Skipping the drain on the
all-resident path does not change what the model computes, which is what the
drain existed to protect.

## Why it is null, and why that is worth knowing

The plan asserted drain #2 was "the larger half" of the ~374 us. **That was
wrong.** Drain #2 costs essentially nothing *because drain #1 has already paid
for it*: `ds4_gpu_tensor_read` (`ds4.c:22331`) does a blocking D2H `cudaMemcpy`
for the router ids, which waits on the same decode-stream work. By the time
`begin_load` reaches its `cudaStreamSynchronize`, the stream is already
drained, so there is nothing left to wait for and removing the wait recovers
nothing.

So the whole 374 us per layer per token sits on **drain #1**. That does not
weaken Tier 2, it sharpens it: the fix is not "synchronise less", it is **the
router ids must never leave device memory**. Tier 2 -- resolving the mapping in
a probe kernel against a device-side slot table, with a pinned miss flag --
attacks the only serialization that actually costs. And it now has to deliver
the full win on its own; nothing was banked here.

## Disposition of the change

**Kept, as a prerequisite rather than a win.** It is correct (logits
byte-identical), free (+0.0% within drift), and it removes a drain that is
genuinely redundant today but would start costing the moment drain #1 is
removed -- at which point `begin_load`'s sync becomes the *first* drain rather
than the second. Landing it now means Tier 2 does not have to land two changes
at once.

`DS4_CUDA_STRICT_EXPERT_SYNC=1` is retained so the arms stay reproducible.

## Note on the baselines

Against prep §26's recorded 16.54 tg these arms read ~2% low, consistent with
every other cross-build comparison in this campaign -- which is exactly why the
control was run on the same binary rather than against the record. Comparing
arm 1 to §26 alone would have shown a 1.6% *regression* from a change that is
provably inert.
