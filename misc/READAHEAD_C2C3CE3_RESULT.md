# RESULT: upstream's prefill read-ahead (c2c3ce3) is a SUBSTITUTE for the campaign's multi-threaded fetch, not a complement -- and the worse one

Measured 2026-09-17 on spark-0fb3 (GB10, sm_121a verified by `cuobjdump` on
both binaries), single box, `--ssd-streaming`, 16384-token prompt from
`tests/long_context_story_prompt.txt`, 512 decode tokens, `--ctx-alloc 32768`.
Arms interleaved; every ON arm repeated. 14 runs.

Branch `readahead-c2c3ce3` = `q4tp2-stream` (3a297a9) + cherry-pick of
`c2c3ce3`. Vanilla = `origin/main` @ `8db1d1d`, unmodified.

## This overturns PLAN_TG_NEXT_LEVERS_V2 Step 0

That plan said c2c3ce3 was "complementary, not duplicative: the campaign's
depth-8 knee is depth *within* a layer's fetch; this is overlap *across*
layers." **That reasoning was wrong.** Both mechanisms buy the same thing --
NVMe read parallelism during prefill -- out of the same fixed budget. They do
not stack, and the campaign's version is strictly better.

**Recommendation: do not take c2c3ce3.**

## The grid (prefill tok/s)

Branch `readahead-c2c3ce3`, Q4:

| fetch depth | read-ahead ON | read-ahead OFF | delta |
|---|---|---|---|
| 1 | 131.14 | 132.44 | -1.0% |
| **8** | 132.58 *(131.09, 134.06)* | **146.68** | **-9.6%** |

Branch `readahead-c2c3ce3`, Q2:

| fetch depth | read-ahead ON | read-ahead OFF | delta |
|---|---|---|---|
| 1 | 314.49 | 283.10 | **+11.1%** |
| **8** | 310.30 *(303.10, 317.50)* | **361.41** | **-14.1%** |

Vanilla `origin/main` (no `DS4_CUDA_EXPERT_FETCH_THREADS` knob exists there,
so it runs at the equivalent of depth 1):

| quant | read-ahead ON | read-ahead OFF | delta |
|---|---|---|---|
| Q4 | 133.39 *(133.58, 133.19)* | 133.36 | +0.0% |
| Q2 | 299.74 *(298.46, 301.01)* | 279.26 | **+7.3%** |

Drift on the repeated ON arms: Q4 +2.3%, Q2 +4.8% (branch), Q4 -0.3%,
Q2 +0.9% (vanilla). Every effect called significant above is outside it; the
Q4 depth-1 cell (-1.0%) is not, and is reported as null.

## Answering it directly: is the read-ahead better than our multi-fetch?

No, at either quant. Best read-ahead configuration vs multi-fetch alone:

| quant | read-ahead, depth 1 | multi-fetch (depth 8), read-ahead off | multi-fetch wins by |
|---|---|---|---|
| Q4 | 131.14 | **146.68** | **+11.9%** |
| Q2 | 314.49 | **361.41** | **+15.0%** |

## The mechanism, tested rather than assumed

The same binary flips the sign of the effect when only the foreground fetch
depth changes: Q2 read-ahead is **+11.1%** at depth 1 and **-14.1%** at depth
8. That is the signature of two mechanisms competing for one resource, and it
rules out a code-interaction or merge artifact -- nothing about the build
differs between those two cells.

`DS4_CUDA_SSD_PREFETCH_PROFILE` shows why directly. At Q4 the reader moves a
**full layer** per step -- 384 experts, 7.12 GiB -- taking 1.79-2.28 s, while
the foreground reports `wait=0.48-1.23 s` blocked on it at every one of the 40
layers. The read-ahead is not overlapping with spare bandwidth; it is taking
bandwidth the foreground fetch wanted, and at Q4 (18.98 MiB experts, 2x Q2's)
it cannot finish a layer in time no matter what, so it only ever subtracts.

This is consistent with the campaign's own null prefetch results (prep §22)
rather than in tension with them: every prefetcher this campaign tried was
also competing for the same saturated NVMe.

## Incidental: what the campaign's fetch work is worth

Both arms read-ahead OFF, depth 8 vs vanilla, same shape and box:

| quant | pp vanilla -> branch | tg vanilla -> branch |
|---|---|---|
| Q4 | 133.36 -> **146.68** (+10.0%) | 4.86 -> **5.84** (+20.2%) |
| Q2 | 279.26 -> **361.41** (+29.4%) | 10.69 -> **11.31** (+5.8%) |

The branch's depth-8 OFF cell (146.68) also reproduces the campaign's recorded
single-box Q4 reference of 147 pp, which is independent evidence that the
cherry-pick and its conflict resolution did not perturb the baseline.

## tg is null everywhere, as designed

The read-ahead only runs on prefill batches >= 2048 tokens, so decode should be
untouched, and it is: Q4 5.835 ON vs 5.84 OFF, Q2 11.305 ON vs 11.31 OFF at
depth 8. No tg cell in any of the 14 runs moved outside the campaign's
documented ~2.2% tg drift.

## If it is ever taken anyway, the merge is already done and one bug is fixed

The cherry-pick conflicts in six hunks of `ds4_cuda.cu`, all resolved on this
branch (compiles clean, sm_121a verified, baseline reproduced). One of them is
a real defect and not a cosmetic merge:

* Upstream's `cuda_stream_prefetch_protects()` keeps the reader's in-flight
  slots from being evicted, and upstream added that guard to the **only**
  eviction policy it has (plain LRU). The campaign has two more -- CLOCK with
  lives (`DS4_CUDA_EXPERT_EVICT`/`_LIVES`) and the tier-aware LRU -- and
  neither would have honoured it. All three scans now carry the guard;
  without it the foreground can evict a slot the reader is copying into.
* `cuda_model_stage_read_from()` is kept as upstream's plain parameterised
  read with no tier lookup and no counters, because the tier readers increment
  several non-atomic globals and the reader thread runs concurrently with the
  foreground. Read-ahead bytes therefore do not appear in `DS4_FETCH_STATS`
  disk totals; `DS4_CUDA_SSD_PREFETCH_PROFILE` accounts for them separately.
* The expert clock starts at 1, not 0, so `used == 0` keeps meaning "never
  filled" -- which is what both upstream's cold-read-ahead property and the
  campaign's CLOCK `if (!c.used)` case depend on.

## What this does and does not settle

Settled: c2c3ce3 should not be taken while the campaign's fetch depth is >1,
at either quant, single box. Its own gate (`total_count >= 2048`) is not the
relevant one; foreground fetch depth is.

Not settled, and not worth doing unless something changes:

* TP2. Untested -- under TP2 the read-ahead would additionally need the
  ownership filter so a rank reads ahead only its own half. Moot given the
  single-box result, and it is the reason this was never going to be a free
  pick.
* Whether a *bounded* read-ahead (a few experts rather than a full layer,
  sharing the depth-8 budget instead of adding a 41st reader) could beat
  depth 8 alone. That is a different mechanism from the one measured here.
* Q2 resident. Read-ahead only exists under `--ssd-streaming`, so it cannot
  act there at all; per prep §26 that is the configuration Q2 should actually
  run in.
