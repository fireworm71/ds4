# Next steps: revision of PLAN_TG_NEXT_LEVERS

Written 2026-09-17, after reading `misc/PLAN_TG_NEXT_LEVERS.md` and checking it
against the code and against upstream. The original's measurements and its DEAD
list stand unchanged. What follows revises the *order*, resolves the one
question it left open, and adds two prerequisites and one external event it
could not have known about.

## Summary of what this revision changes

1. **Upstream moved 54 commits** while the campaign ran, and one of them
   (`c2c3ce3`) is V4.1 CUDA SSD prefill read-ahead, validated on the Sparks, in
   the same expert cache this campaign rewrote. It is not in any spark branch.
2. **Lever 1's "CONFLICT TO RESOLVE FIRST" is not a conflict.** Details below;
   it costs a one-line doc edit, not an investigation.
3. **Lever 1 has two prerequisites** named in `b0cbc30`'s own commit message
   that the plan does not carry forward. Measuring TP2 spec decode before them
   would produce an uninterpretable number.
4. **Lever 4 got much cheaper**: upstream just built its measurement harness.
5. The whole campaign record is untracked in a deletable worktree.

---

## Step 0 -- upstream integration (decide first; it prices everything else)

Measured state:

| branch | base | ahead | behind origin/main | clean? |
|---|---|---|---|---|
| `main` | -- | 0 | **54** | clean |
| `spark-opt` (worktree `ds41f-spark`) | `a04f46f` | 44 | 53 | **dirty** |
| `q4tp2-stream` (worktree `q4tp2`) | `a04f46f` | 45 | 53 | clean -- **campaign tip** |
| `spark-prefetch` (worktree `ds41f-prefetch`) | `a04f46f` | 3 | 53 | clean |
| `ds41f-spark` (branch, not the worktree) | `main` | 15 | 54 | -- |

Trap: the worktree named `ds41f-spark` is checked out on branch **`spark-opt`**,
and a *different*, older branch is also named `ds41f-spark`. Work from
`q4tp2-stream` in the `q4tp2` worktree; that is the tip.

`main` is a pure upstream mirror with nothing local on it -- fast-forward it,
it is free.

Rebase surface for the spark branches, `a04f46f..origin/main` vs
`a04f46f..q4tp2-stream`:

| file | upstream | spark |
|---|---|---|
| `ds4_tp.c` | **0** | +13 -2 |
| `ds4_ssd.c` | **0** | 0 |
| `ds4_gpu.h` | +312 | +8 |
| `ds4_cuda.cu` | +285 -17 | **+2786 -24** |
| `ds4.c` | +6490 -41 | +1005 -44 |

> **SUPERSEDED 2026-09-17 for `c2c3ce3` -- see
> `misc/READAHEAD_C2C3CE3_RESULT.md`.** It was cherry-picked, the merge was
> resolved, and it was measured at both quants against vanilla `origin/main`.
> The "complementary, not duplicative" reasoning below is **wrong**: the
> read-ahead and the campaign's multi-threaded fetch buy the same NVMe read
> parallelism out of the same budget. At depth 8 it costs **-9.6% pp (Q4)** and
> **-14.1% pp (Q2)**; the campaign's multi-fetch alone beats the best
> read-ahead configuration by **+11.9% / +15.0%**. **Do not take `c2c3ce3`.**
> `147b263` (the bench harness) is unaffected and still worth picking.

**Recommendation: cherry-pick two commits now, defer the 53-commit rebase.**

* `c2c3ce3` "Overlap CUDA SSD expert reads with V4.1 prefill" -- reads the next
  layer into reserved slots of the admitted expert cache behind two 8 MiB
  staging buffers, enabled from 2K tokens. This is **complementary, not
  duplicative**: the campaign's depth-8 knee (§22) is depth *within* a layer's
  fetch; this is overlap *across* layers. The campaign's null prefetch results
  were all decode-side; this is prefill-side. It is the one upstream change
  that plausibly moves a cell in the headline table.
  Caveat: it must be taught the TP2 ownership filter before it is correct under
  TP2 streaming -- a rank must read ahead only its own half, the same rebasing
  the owned dispatch already does for the on-demand path. Budget for that, and
  assume the pick conflicts in `ds4_cuda.cu`.
* `147b263` "Add a session-concurrency benchmark for the engine and the server"
  -- `speed-bench/session_concurrency_bench.c`, `concurrency_sweep.sh`,
  `serve_concurrency_bench.py`. Additive plus ~12 Makefile lines; near-zero
  conflict. This is Lever 4's harness and part of Lever 1's gate (see below).

Defer the rest: upstream's +6490 in `ds4.c` is overwhelmingly Qwen3.8 Flash
Next, a model family this campaign does not touch. Re-attempt the full rebase
after Lever 1 lands, and do not let it slip further than that -- `ds4_cuda.cu`
divergence is already 10:1 in the campaign's favour and only grows.

**Cost to book honestly:** `c2c3ce3` changes prefill, so the *pp* cells of the
headline table (174.12 Q4 TP2, 412.87 Q2 TP2 resident) must be re-measured
after the pick. tg should be untouched -- prove it with one interleaved
control rather than assuming it.

## Step 1 -- the Makefile arch guard (30 minutes; do it today)

Root cause, precisely: `Makefile:40` sets `CUDA_ARCH ?=` empty, and the whole
`ifneq ($(strip $(CUDA_ARCH)),)` block at `:41-49` is then skipped, so
`NVCC_ARCH_FLAGS` stays empty and **`-DDS4_CUDA_HAVE_MXF4=1` is never defined**.
nvcc falls back to its built-in default arch and MXFP4 compiles out, silently.

`make cuda` already guards (`Makefile:262-266`), and `cuda-spark` (`:255`) and
`cuda-generic` (`:259`) set the variable. What has no guard is the direct
targets -- `ds4-bench` (`Makefile:306`), `ds4` (`:300`), `ds4-server` (`:303`),
each of which links `$(CORE_OBJS)` and so compiles `ds4_cuda.o` with an empty
`NVCC_ARCH_FLAGS`. That is exactly how the defect that voided a day of
measurements got in.

Confirmed identical upstream (`origin/main:Makefile`, same empty default, same
lone guard inside the `cuda:` target), so this is worth a PR, not just a local
patch.

Fix: give the `ifneq` an `else` branch that hard-fails, or make the CUDA
compile rules depend on a guard target. Keep it inside the existing non-Apple
CUDA branch so CPU, Metal and ROCm builds are untouched.

Small, self-contained, and it protects every Spark user, not just this
campaign.

## Step 2 -- Lever 5, resident fast path (unchanged, still first among levers)

Two distinct items hide under one heading; do the cheap one.

* **(a) Load-time routing** -- "does this rank's shard fit the cache budget?
  then map it resident and skip the streaming path." Captures the whole 33%
  for the fits case (§26: Q2 TP2 16.54 -> 21.94 tg, 319 -> 413 pp, first token
  154-167 ms -> 53 ms) with no new mechanism, only routing between two paths
  that both already work.
* **(b) Per-access bypass** (§26's "worth building") -- skip the cache
  machinery per expert access when the shard is fully resident. Only matters
  for quants that fit but only just. Defer.

Fold in the owed flag removal here: once the router chooses the path itself,
`DS4_CUDA_TP_STREAMING` has no job. Sites: `ds4_tp.c:631-637`,
`ds4_cuda.cu:24975`, `docs/DISTRIBUTED.md:108`.

## Step 3 -- Lever 1, speculative decoding, re-scoped

### The conflict, resolved -- it was never a conflict

* `ds4_tp.c:642` (on `q4tp2-stream`) is about the **transport**: drafting
  is allowed on the leader, the verify block is mirrored via
  `DS4_TP_FRAME_VERIFY`. That machinery is real and dispatched --
  `ds4_tp.c:3156` -> `ds4_session_tp_spec_cycle` (`ds4.c:75716`), with the
  verify window helpers at `ds4_tp.c:1365-2242` and the leader sending at
  `ds4.c:74942` / `ds4.c:75490`.
* `docs/DISTRIBUTED.md:92-94` ("V4.1 supports vision but not speculative
  decoding") is about the **V4.1 model**, not about TP. It is true on `main`
  and stale on `spark-opt`, where the chain exists behind
  `DS4_V41_DSPARK_ENABLE`.

Both were accurate when written; they are scoped to different layers. Action is
a one-line doc edit when the V4.1 DSpark chain lands on `main`, not an
investigation. Bonus: **upstream has not touched `ds4_tp.c` since `a04f46f`**,
so Lever 1's transport has zero rebase exposure.

### The two prerequisites the plan omits

Both are stated in `b0cbc30`'s own message and neither has landed -- every
commit after it on `spark-opt` is Engram readahead or the RDMA T1-T4 series.

**1a. Drive the drafter's context capture from the verify batch.**
The drafter's context ring is fed by the single-token decode capture, which a
batch commit never runs, so after every verified block the drafter conditions
on a stale ring. Shadow mode measured **26%** acceptance; the live histogram is
`accepted_len 0:26, 1:4, 2:1` over 48 tokens -- i.e. ~1.19 tokens/cycle.

This matters for how the plan is read: its "+15% tg even at the current 1.18
tokens/cycle" is arithmetic on **a bug artifact, not a floor**. Fix 1a,
re-measure acceptance on a single box, and only then pre-register the TP2 gate.
Running TP2 first risks a null that means nothing.

**1b. Build the state oracle.**
`b0cbc30` names this as the outstanding correctness milestone: compare
post-commit KV, carry, history and pos against a serial run. Byte-identity of
logits is the wrong gate -- the batch path's FP reduction order differs from
single-token decode, which this engine already documents, and the divergence
survives both `DS4_DS41_VERIFY_COMMIT1=1` and a single-row batch, so it is
neither the accept rule nor the rollback.

Upstream's new bench gives you most of this for free:
`session_concurrency_bench --verify` checks "speculative == plain greedy" (and
"batched == sequential"), printing the logit gap on mismatch
(`speed-bench/session_concurrency_bench.c:468-545, 739-741`). Adapt it rather
than writing one. Do **not** assume it passes for V4.1 unmodified -- token
equality can still flip on near-ties under the same reduction-order difference,
so expect to need a tolerance rule on logits plus exact assertions on state.

**1c. Only then measure.** Q4 TP2, story 16K/512, spec on/off interleaved,
repeat the opening arm (tg drift ~2.2%). Keep the plan's pre-registered
`tg > 11.5`, but restate it after 1a, with the acceptance you actually have.
The plan's own risk -- acceptance may fall under TP2 gate latency -- stands.

## Step 4 -- Lever 2, cache policy re-run (unchanged, cheap, still right)

Eleven policies were retired against a +2.0pp gate on the now-false premise
that misses never repeat (§24: 2.38 misses per distinct expert). Re-run LRU vs
CLOCK lives={1,3} on `DS4_CUDA_EXPERT_EVICT` / `_LIVES`, interleaved, two
repeats, at Q4 single-box (84.5% hit) and Q4 TP2 (96.2%).

One sequencing constraint added: run this **after** the `c2c3ce3` pick, not
before. Prefill read-ahead changes cache occupancy at the moment decode starts,
which is the hit rate the policies are being graded on. Keep the
pre-registered-gate discipline.

## Step 5 -- Levers 3 and 4, conditional as before

**Lever 4** is now much cheaper to attempt and the honest scope is narrower.
The harness arrives with `147b263`. But upstream's *batched speculation* is
Metal-first -- `d569d05` puts the batched recurrent/attention rows in
`metal/qwen4.metal` and `ds4_metal.m`; CUDA gets Qwen inference (`df6eedd`) but
not those row kernels. On the Sparks, Lever 4 therefore means batched decode
**without** batched spec unless someone ports them.

The upstream series is still the design precedent to follow if Lever 1 and
Lever 4 are ever combined for V4.1: batch the predictor across sessions
(`e2cba0b`), bound the batch and invalidate failed predictor state (`548cdaf`),
decide per cycle whether to draft (`3077786`), price the cycle kinds after
eight drafted cycles (`9ba464d`), and serve it with bounded session state
(`ab38bc6`, which also touches `ds4_server.c`).

**Lever 3** (small cross-tier from TP2's spare RAM, ceiling 3-4%) is unchanged
and stays last.

## Revised order

0. Fast-forward `main`; cherry-pick **`147b263` only** onto `q4tp2-stream`.
   `c2c3ce3` is measured and rejected (`READAHEAD_C2C3CE3_RESULT.md`); the pp
   baselines therefore do not move and need no re-measurement.
1. Makefile arch guard (and send it upstream).
2. Lever 5(a) + delete `DS4_CUDA_TP_STREAMING`.
3. Lever 1 as 1a -> 1b -> 1c, in that order.
4. Lever 2.
5. Levers 4 then 3, conditional.

Steps 1 and 2 are independent of step 0 and can go in parallel.

## Housekeeping, and one real risk

**The campaign record is one `git worktree remove` from gone.** `.gitignore:81`
ignores `/misc/`, and only four `misc/*.md` files are tracked (they predate the
ignore). Everything this campaign produced -- `PLAN_TG_NEXT_LEVERS.md`, the
73 KB `Q4_TP2_STREAMING_PREP.md` with all 27 sections, the 228 KB
`SPARK_OPT_STATE.md`, every `T*_RESULT.md` and log -- exists only as untracked
files in the `ds41f-spark` worktree. There is precedent for the fix:
`git add -f misc/*.md` on the campaign branch, exactly as the other four got
there. Do it before anything else touches these worktrees.

**Uncommitted work in `ds41f-spark`:** ~24 lines adding block-device support to
`model_open` (`ds4.c`) and `ds4_engram_table_open` (`ds4_engram.c`) -- sizing
via `lseek` when `st_size` is 0 and `S_ISBLK`. Leftover from the dm-cache /
rambox composite line the plan lists as DEAD. It is self-contained and
defensible on its own merits (serving a GGUF from a block device), but it
belongs to a closed line of work: commit it as a standalone change or discard
it. Do not let it ride along into a lever branch.
