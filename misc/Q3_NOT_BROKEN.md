# Q3 is NOT a broken GGUF. The artifact is byte-clean; the streaming expert cache cannot hold two expert sizes.

Measured 2026-09-17 on spark-0fb3. This reverses the `PLAN_TG_NEXT_LEVERS`
DEAD entry, which reads:

> ~~Q3splice TP2 resident~~ -- the Q3splice GGUF is broken (Q2 and Q4 experts
> shuffled together, per Jason 2026-09-17). ... no valid artifact exists.
> Revisit ONLY if a clean Q3 is built.

**A valid artifact already exists and no rebuild is needed** -- which matters,
because with 38 GiB free on a 3.6 TB disk a 362 GiB rebuild was impossible
anyway.

## 1. The artifact is coherent, and its bytes come from where they should

`DeepSeek-V4.1-Flash-Q3splice.gguf`, 362.0 GiB, 1046 tensors (same count as
both sources). Read its header and compared every tensor against **both** the
Q2 and Q4 files by declared type, byte length, and sampled payload (four 8 KiB
windows per tensor, at the same relative offsets in source and splice):

| group | result |
|---|---|
| routed experts, layers 0-33 (102 tensors) | byte-identical to **Q2** at every sampled window |
| routed experts, layers 34-39 (18 tensors) | byte-identical to **Q4** at every sampled window |
| all 926 non-expert tensors | byte-identical to the **Q2** base |
| KV metadata blob | byte-identical to the Q2 base (5,762,897 B) |
| sum of tensor bytes | 361.9 GiB against a 362.0 GiB file |

No layer mixes sources within itself. Nothing is shuffled. This is exactly the
documented design in `gguf-tools/mixed/README.md` -- the Q2 base with the last
six layers' routed experts taken from Q4 -- applied to V4.1's 40 layers, so
34-39 rather than the README's 37-42 for the older 43-layer V4 Flash.

## 2. It fits resident per rank under TP2, and the arithmetic validates

Splitting routed experts 50/50 and replicating the rest. Engram is 189.4 GiB of
the file and is disk-only (`Engram disk-only` in every run log), so it is not
part of the resident footprint:

| quant | routed | resident non-routed | TP2 per-rank | fits 121 GiB? |
|---|---|---|---|---|
| Q2 | 142.4 | 8.78 | **80.0** | yes |
| **Q3splice** | 163.7 | 8.78 | **90.7** | **yes** |
| Q4 | 284.8 | 8.78 | **151.2** | no |

Two independent checks that this model is right: the Q2 row lands on 80.0 GiB
against the **80.56 GiB/rank measured** in prep §26, and the Q3 row lands on
90.7 GiB against the **~91 GiB** the plan itself predicted. The Q4 row explains
why Q4 must stream.

So the prize the plan described is real: Q3 quality at a footprint that runs
**resident** on the pair, where prep §26's rule ("stream only when the shard
does not fit") says resident TP2 is worth +29% pp and +33% tg over streaming
the same shard. Against Q4 TP2 streaming (174.12 / 10.31) the reference point
is Q2 TP2 resident at **412.87 / 21.94**.

## 3. What Jason actually saw: ~81 GiB of expert cache reallocated twice per token

Single box, `--ssd-streaming`, 2048-token prefill, 4 decode tokens. The expert
cache oscillates between two allocations for the whole run:

```
ds4:   routed expert size: 9.49 MiB
ds4: CUDA SSD expert cache: 8723 slots, 80.86 GiB   <- Q2-sized experts
ds4: CUDA SSD expert cache: 4361 slots, 80.85 GiB   <- Q4-sized experts
ds4: CUDA SSD expert cache: 8723 slots, 80.86 GiB
ds4: CUDA SSD expert cache: 4361 slots, 80.85 GiB
   ... 10 reallocations for 2048 prefill tokens + 4 decode tokens
2048,2048,71.19,4,0.13,7639.930,3,0.13,0
```

**tg 0.13 tok/s. First token 7.6 seconds.** A 16K/512 run at the shape Q2 and
Q4 both finish in ~6 minutes was killed at 30+ minutes, unfinished.

That is indistinguishable from "the model is broken" at the prompt, and the
conclusion was a fair one to draw. But the cause is in the engine.

### The exact line

`ds4_cuda.cu:29730`, in `cuda_stream_selected_cache_begin_load`:

```c
if (cache.model_map != table->model_map ||
    cache.gate_expert_bytes != table->gate_expert_bytes ||
    cache.down_expert_bytes != table->down_expert_bytes ||
    g_stream_expert_slots.size() < unique.size()) {
    cuda_stream_selected_cache_release();   /* frees and clears everything */
    ...                                     /* then reallocates ~81 GiB */
```

The streaming expert cache is **one shared slab, strided for a single layer's
expert size**. `begin_load` runs per layer per token. The splice changes expert
size at the 33->34 boundary and back at the 39->0 wrap, so the cache is torn
down and rebuilt twice per token, and no cached expert ever survives -- the hit
rate collapses to zero on top of the allocation cost.

This assumption is confined to that slab. Everywhere else the engine already
sizes experts per layer (`routed_expert_row_bytes(layer->ffn_gate_exps)` at
`ds4.c:19597`, `:22103`, `:23220`, ...), which is why a mixed-quant file parses,
loads, and produces output at all rather than failing.

## 4. Why this is a small fix, and why resident may need no fix

**Streaming fix:** size the slab for the **maximum** per-layer expert bytes at
cache-creation time and stride slots at that width, instead of re-deriving it
from whichever layer called last. Costs slots for the small layers (4361 rather
than 8723 here) but removes every reallocation. No policy change; the eviction
policies already key on slots, not bytes.

**Resident:** the resident path does not use this slab at all, and per-layer
sizing is already correct elsewhere. So Q3 resident TP2 -- the configuration
the plan actually wanted -- plausibly needs **nothing**. That is the cheap
experiment, and it should be run before anyone writes the streaming fix.

## 5. Blocker, and what it needs from Jason

Q3 resident TP2 needs both boxes, and I cannot start the peer's worker:
`ssh 10.99.0.2` fails with `Permission denied (publickey,password)` from
spark-0fb3 -- the same gap prep §6 recorded. Also unverified from here:
whether promax holds a copy of the Q3splice, and whether its bytes match this
one (the ranks need identical model bytes, per prep §7).

To run it:

1. Confirm promax has `DeepSeek-V4.1-Flash-Q3splice.gguf` and that it matches
   (a sampled-window compare like the one in §1 is enough; a full hash of
   362 GiB is not needed).
2. Worker on promax: `./ds4 --tensor-parallel --role worker` with `--ctx 32768`
   and **no** `--ssd-streaming`.
3. Coordinator here: `ds4-bench --tensor-parallel --role coordinator`,
   `--ctx-alloc 32768`, **no** `--ssd-streaming` -- the whole point is that the
   90.7 GiB shard is resident.
4. Same shape as every other cell: `tests/long_context_story_prompt.txt`,
   16384/512, arms interleaved.

Pre-registered expectation, stated before running: pp and tg between Q2 TP2
resident (412.87 / 21.94) and Q4 TP2 streaming (174.12 / 10.31), and much
nearer the Q2 end, because the shard is resident and only 13% larger than
Q2's. **Success = tg > 15**, which would beat Q4 TP2 streaming by >45% at
materially better quality than Q2.

## 6. What is established and what is not

Established: the artifact is sound; it fits resident per-rank at 90.7 GiB; the
streaming failure is the shared-slab size assumption, reproduced and located.

Not established: Q3 resident TP2 numbers (needs the peer); output *quality* of
the splice, as opposed to its byte provenance -- the README's claim that
last-six-layers-Q4 scores closer to full Q4 than to Q2 was measured on the
older V4 Flash, not V4.1, and deserves its own scoring run once it is fast
enough to score.
