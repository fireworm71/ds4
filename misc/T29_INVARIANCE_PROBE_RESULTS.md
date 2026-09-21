# T29: the dispatch hypothesis is dead, and the probe is unsound under streaming

Ran the batch-invariance experiment T21 Phase 17 called for. Four runs, isolated
worktree at `6c42fff` (`.claude/worktrees/invariance`), single box, no worker
touched. Two real findings and one method failure.

## 1. The dispatch split, measured for the first time

T21 inferred it from source. Measured, with `DS4_CUDA_ATTN_DISPATCH_LOG=1`:

```
960  arm="attention exact score split batch launch"  n_tokens=1     <- decode
 40  arm="attention decode batch launch"             n_tokens=5     <- verify
 40  arm="attention decode batch launch"             n_tokens=6     <- verify
```

Decode and verify take **entirely different kernels**, not merely different
thread counts. 960 = 24 tokens x 40 layers; the 40s are one sweep each.

## 2. Unifying the kernel changes nothing -- the hypothesis is dead

`DS4_CUDA_ATTN_FORCE_GENERIC=1 DS4_CUDA_ATTN_FIXED_THREADS=1` puts all three row
counts on one kernel at one reduction width:

```
960  arm="attention decode batch launch"  n_tokens=1
 40  arm="attention decode batch launch"  n_tokens=5
 40  arm="attention decode batch launch"  n_tokens=6
```

| run | arrays_differing |
|---|---|
| control (split dispatch) | **43 of 56** |
| same kernel + same threads | **43 of 56** |

Identical. **The dispatch difference and the 512-vs-256 thread flip are not the
cause.** T21's "the dispatch, not the tiling" is refuted -- as is the block-size
comment at ds4_cuda.cu:18804, which named the thread count as "exactly why" a
batched row does not match a decode row.

Caveat, and it is a real one: see section 4. The artifact floor contributes to
both numbers, so this is strong evidence rather than proof. It should be
repeated resident before the hypothesis is closed for good.

## 3. The batch can never adopt decode's kernel

`attention_decode_score_split_launch` (ds4_cuda.cu:9915) takes a single `pos0`
and has **no `n_tokens` parameter** -- `const uint32_t qpos = pos0;`. It is
inherently single-position. So convergence can only ever be reached by moving
decode onto the generic batch kernel, never by moving the batch onto decode's
specialised one. Any fix therefore costs decode its fast path, and decode is the
44 ms/token baseline everything is measured against.

Also: disabling all batching (`DISABLE_V41_BATCH_ATTN=1 + _MOE=1`) disables the
verify path itself -- the run completes with no speculative cycle at all
(generation 2.08 t/s, no probe output). It cannot serve as a sanity floor.

## 4. The probe is unsound under SSD streaming

Mode 2 runs the DECODE path twice and, per its own comment, "a sound probe must
then report zero differing arrays; if it does not [...] any diff this probe
reports is its own artifact rather than a finding."

```
DS4_DS41_VERIFY_STATE_DIFF=2   arrays_differing=18 of 56
```

**Not zero.** The 18 are exactly `window[22]` through `window[39]` -- eighteen
consecutive layers -- with rounding-level differences (`decode=-0.109375`
`verify=-0.101562`, 69-186 differing bytes out of 262144).

The cause is legible: the run reports `SSD cache fitted from 8723 to 8134
experts`, and 8134 / 384 experts-per-layer = **21.2 layers**. Layers 0-21 are
resident and bit-reproducible; layers 22-39 stream and are not.

### This is a finding in its own right

**Streaming makes decode numerically non-deterministic run to run**, layer-local
to streamed layers. The weights are identical, so the mechanism is almost
certainly expert accumulation order following fetch-completion order rather than
a fixed order -- results depend on I/O timing.

Consequences beyond this experiment:

* **Bit-exact speculative verification is impossible in any streamed
  configuration**, independent of the batch-invariance question. T27's
  streaming x speculation argument needs this caveat: the economics hold, but
  "identical to target-only" does not.
* Any streamed deployment is not bit-reproducible, which affects every A/B
  measured under `--ssd-streaming`.

## 5. Why this could not be settled single-box

Q2 needs 151.76 GiB of working set (from the TP log: rank 0 maps "80.56 GiB of
151.76 GiB"). All 15,360 experts (384 x 40) would need ~142 GiB of cache against
121 GB of RAM. **Single-box resident is impossible for Q2**, so the probe cannot
be made sound on one machine with this model.

The experiment must be repeated on the pair, resident, where T21 got mode 2 = 0.
At the time of writing the promax worker is running antigravity's Track 5 WIP
binary, so a coordinator from a different commit would risk a protocol mismatch
and wedge it. It was deliberately left alone.

## 6. Incidental

`--dspark` on V4.1 single-box additionally requires **`DS4_V41_DSPARK_ENABLE=1`**
(ds4.c:67110), an experimental gate absent from every documented invocation.
Without it the engine refuses with "DSpark, steering and legacy diagnostics are
not supported".

## 7. What to do next

1. Repeat sections 1-2 **resident on the pair** when it is free. Gate on mode 2
   reporting 0 first; without that gate the numbers mean nothing.
2. If the dispatch result holds resident, the search moves inside the sweep --
   MoE batch, HC batch, or the KV publish -- and `DS4_METAL_DISABLE_V41_BATCH_MOE`
   alone is the next bisect, since it should leave the verify path alive where
   disabling attention batching does not.
3. Treat every streamed benchmark as non-reproducible at the rounding level.

The build is left in place at `.claude/worktrees/invariance` (detached at
`6c42fff`) with the probe scripts in the job tmp dir, so the resident rerun is
one command once the pair frees up.

---

# ADDENDUM: the refutation is artifact-free after all

Section 2 hedged because the probe's 18-array artifact floor could have masked a
real change. It cannot. The per-array lists settle it.

## The artifact is layer-local, and the signal is not

Mode 2's artifact is exactly `window[22]`..`window[39]` -- the streamed layers.
**Layers 0-21 are bit-reproducible in this configuration.** So any divergence
observed in `window[0..21]` is real signal, not artifact.

Control and forcegen each diverge in the *same* 43 arrays:

```
window[3] window[4] ... window[39]
previous_kv[1] previous_score[1] previous_kv[2] previous_score[2]
compressed[3] index_cache[3]
```

**Byte-identical sets.** Nineteen of them -- `window[3]` through `window[21]` --
lie in the artifact-free region. Unifying the attention kernel and the reduction
width changed nothing there.

**The dispatch hypothesis is refuted conclusively, on a single box, without
needing the pair.** Section 5's "must be repeated resident" applies only to
pinning down the true cause, not to this negative result.

## And the signature localises the true cause

`ds41_state_spans` (ds4.c:60168) indexes these differently, which is easy to
misread:

* `window[il]` -- by **layer**, 40 of them
* `compressed[i]`, `index_cache[i]` -- by **compressor tier**, 4 of them
* `previous_kv[i]`, `previous_score[i]` -- by **tier**, 3 of them

So the divergent set is: layers 3-39, plus tiers 1 and 2 of the previous-KV
carry, plus tier 3 of the compressed/indexer state. Clean: layers 0, 1, 2;
tiers 0-2 of compressed/index_cache; tier 0 of previous_kv.

Against the layer layout (`ds4_expected_layer_compress_ratio`, FLASH variant):

```
layer 0, 1 -> ratio 0    (no compressor)   CLEAN
layer 2    -> ratio 4                       CLEAN
layer 3    -> ratio 128                     FIRST DIVERGENCE
then alternating 4 / 128 to layer 39        all divergent
```

The first divergent layer is **the first ratio-128 layer**, and the compressor
and indexer tier state diverges with it. Attention is clean through a ratio-4
layer and breaks at the first ratio-128 one.

That moves the search off attention entirely and onto the **compressor / carry /
indexer path** -- which is also where T21 located `ds41_attention_publish_batch`
and "the ratio-2 carry snapshot". The batch-size dependence is most likely in
how the compressor publishes a frontier when N rows arrive at once versus one.

Caveat on reading the first divergent window as the origin: hidden states are
not tracked, only persistent KV. `window[3]` differing means the divergence is
observable by layer 3, not that layer 3 computes it -- it could enter layer 3
from layer 2's output. But layers 0-2 writing identical KV bounds it tightly.

## Next bisect

Running `DS4_DS41_CORE_INVARIANT=1`, `DISABLE_V41_BATCH_HC=1` and
`DISABLE_V41_BATCH_CORE=1`. The question for each is not just the count but
**whether the first divergent window moves past 3** -- that names the component.

---

# RESULT: the cause is the batched core, and the residue is the carry snapshot

The bisect lands it. Two of three runs are informative; `DISABLE_V41_BATCH_HC=1`
kills the verify path outright (generation 2.26 t/s, no probe output, only
n_tokens=1 dispatches), exactly as all-batching-off did.

| config | arrays_differing | what differs |
|---|---|---|
| control (default batched) | 43 | `window[3..39]`, `previous_kv[1,2]`, `previous_score[1,2]`, `compressed[3]`, `index_cache[3]` |
| `FORCE_GENERIC` + `FIXED_THREADS` | 43 | **identical set** |
| **`DS4_DS41_CORE_INVARIANT=1`** | **6** | `previous_kv[0,1,2]`, `previous_score[0,1,2]` |
| **`DISABLE_V41_BATCH_CORE=1`** | **6** | same 6 |
| `SERIAL_VERIFY=1` (T21) | 0 | -- |

## Two independent defects, now separated

**(a) The batched core.** Turning off core batching drops 43 -> 6 and clears
*every* `window[]` array plus `compressed[3]` and `index_cache[3]`. Unifying the
kernel and reduction width did not. So the defect is not which kernel the batch
uses, nor the 512-vs-256 thread flip -- it is **structural to computing N rows in
one core call**. The most likely mechanism is what each row attends to: batched,
the N rows share one view of the KV/compressor frontier; serially, row r sees
rows 0..r-1 already published.

**(b) The carry snapshot.** What survives `CORE_INVARIANT` is exactly
`previous_kv` and `previous_score` across all three tiers -- and that is
precisely what T21 said the per-row fallback loses:

> unlike the per-row attention fallback it keeps `ds41_attention_publish_batch`,
> which is where the ratio-2 carry snapshot is taken

So T21's "only the serial path reaches zero" was right, and now the reason is
pinned to six named arrays rather than left as a mystery.

## Which makes the fix bounded and worth doing

`SERIAL_VERIFY` reaches zero by fixing both at once, at the cost of serialising
everything -- hence "correct but pointless".

**`CORE_INVARIANT` + restoring the carry snapshot in the per-row path should
reach zero while keeping MoE and HC batched.** That matters economically: the
MoE is the dominant cost (T23 puts `ffn` at 55% of the drafter chain, and the
target is MoE-heavy), so a verify that batches MoE and HC while running the
attention core per row keeps most of its amortisation. That is the difference
between correct-and-pointless and correct-and-fast.

The task: make the per-row attention fallback take the same carry snapshot
`ds41_attention_publish_batch` takes, then confirm

```
DS4_DS41_UNIFY_DECODE=1 DS4_DS41_CORE_INVARIANT=1 DS4_DS41_VERIFY_STATE_DIFF=3
  -> arrays_differing=0 of 56, WITHOUT SERIAL_VERIFY
```

then measure what per-row attention costs a K-row verify.

## Correction to section 4

In these bisect runs `window[22..39]` came back **clean**, so the streaming
artifact is **intermittent** -- timing-dependent, appearing in some runs and not
others -- rather than a constant floor. That is consistent with fetch-completion
ordering as the mechanism, and it does not change section 4's conclusion that a
streamed run cannot be relied on for bit-exactness. It does mean mode 2 must be
re-run alongside any measurement rather than assumed.

---

# THE RESIDUAL IS A BUG: a restore guarded on a ratio FLASH never has

My "batched compress" hypothesis was wrong, and the runs said so:

```
CORE_INVARIANT=1 + DISABLE_V41_BATCH_COMPRESS=1  -> 6   (unchanged)
DISABLE_V41_BATCH_COMPRESS=1 alone               -> 45  (WORSE than control's 43)
```

Compress batching was never involved. Reading the restore path instead found it.

## The defect

`ds41_verify_commit`, ds4.c:41405:

```c
for (uint32_t owner = 0; ok && owner < 4u; owner++) {
    const uint32_t il = owner == 0 ? 2u : owner == 1 ? 8u : owner == 2 ? 14u : 20u;
    if (ds4_layer_compress_ratio(il) != 2u) continue;     /* <-- dead on FLASH */
    ...restore previous_kv[owner] / previous_score[owner] from vc->prev...
}
```

`ds4_expected_layer_compress_ratio`, FLASH variant: ratio 0 for layers 0-1, then
`(il & 1) == 0 ? 4 : 128`. The four owner layers -- **2, 8, 14, 20 -- are all
ratio 4**. The guard therefore `continue`s on every owner and **the entire
restore loop is dead code on this model.**

Meanwhile the write and the save are not gated on ratio at all:

* `ds4_gpu_dsv41_pool2(..., g->previous_kv[owner], g->previous_score[owner], ...)`
  writes the carry (ds4.c:40480, 40490)
* `ds41_verify_save_prev(vc, g, owner, r)` saves it per row (ds4.c:40483)

**Saved on every row, restored never.** After a partial commit the carry holds
the value following the last *swept* row instead of the last *accepted* one.

The comment above `ds41_verify_save_prev` states the assumption that failed:
"previous_kv/previous_score are only carried by the ratio-2 compressors." That
is not true of FLASH, where a ratio-4 owner goes through `pool2` and carries.

## The signature matches exactly

`ds41_state_spans` tracks `previous_kv[i]` / `previous_score[i]` only for
`i < 3`, and only when `pos` is odd. Three owners x two arrays = **6** -- exactly
the residual that survived `CORE_INVARIANT`, and exactly the arrays the dead loop
would have restored.

## Fix

```c
/* Ratio 0 is the only owner that carries nothing. */
if (ds4_layer_compress_ratio(il) == 0u) continue;
```

Applied in the isolated worktree with the rationale in a comment.

## Where this leaves the feature

| defect | arrays | remedy | cost |
|---|---|---|---|
| batched attention core | 37 | `CORE_INVARIANT` (per-row core) | attention unbatched; **MoE and HC stay batched** |
| dead carry restore | 6 | the one-line guard fix | none |

If the two together reach `arrays_differing=0` without `SERIAL_VERIFY`, then V4.1
speculative decode can be **bit-exact and still batched where it matters** -- the
MoE is the dominant cost (T23: `ffn` 55% of the chain), so keeping it batched is
most of the amortisation. That is the difference between T21's "correct but
pointless" and a working feature.

---

# CORRECTION: the guard fix was wrong, and the real mechanism is simpler

The section above diagnosed the residual 6 as a dead restore guard and applied
`!= 2u` -> `== 0u`. **Measured, it changes nothing:**

```
fix + CORE_INVARIANT -> 6 of 56    (identical arrays, identical values)
fix alone            -> 43 of 56   (identical arrays)
```

The build was verified genuine (ds4.c 15:54:12 -> ds4.o and ds4 15:54:44), so
this is a real negative, not a stale binary. The change has been reverted.

## What the values show, which the counts hid

```
previous_kv[0]    decode=-0.0817958  verify=0  diff_bytes=2040 of 2048
previous_score[0] decode=0.00554217  verify=0  diff_bytes=2039 of 2048
```

**`verify=0`.** Under `CORE_INVARIANT` the verify path leaves the carry as *all
zeros* -- it is never written at all. Restoring it from `vc->prev` restores
zeros, because the save never ran either. I had been reading array names and
counts; the values were the diagnostic.

## The actual mechanism

Tracing the branch conditions, `ds41_verify_save_prev` sits inside the per-row
loop at ds4.c:40481, which runs only when `(!batch_index || !batch_publish)`.
In the default batched path both are true, so the loop never runs: **the carry
is neither saved nor restored in any current configuration.** The machinery is
inert, which is why fixing only the restore is a no-op.

So the two paths each get one half right:

| path | windows / compressed | carry |
|---|---|---|
| batched core (default) | **diverge** (37 arrays) | written, but left at the last *swept* row -- owners 1,2 wrong |
| per-row core (`CORE_INVARIANT`) | **clean** | **never written -- zeros** (6 arrays) |
| `SERIAL_VERIFY` | clean | correct |

`SERIAL_VERIFY` wins both because every row is a full 1-row sweep that still
goes through `ds41_attention_publish_batch`. T21 said exactly this and I should
have taken it literally: "unlike the per-row attention fallback it **keeps**
`ds41_attention_publish_batch`, which is where the ratio-2 carry snapshot is
taken."

## The corrected task

Make the per-row core path maintain the carry -- i.e. `CORE_INVARIANT` must
still run the publish that writes `previous_kv` / `previous_score`, rather than
skipping it with the batched core. That is a real change to the attention
publish path, not a one-line guard, and it is the remaining work between here
and a bit-exact batched verify.

The rest of T29 stands: the dispatch hypothesis is refuted artifact-free, and
the batched *core* (not the kernel, not the tiling, not compress batching) is
what corrupts the windows.

---

# THE RESIDUAL 6 ARE DEAD TENSORS: previous_kv does not exist on FLASH

Chasing `verify=0` into the publish path settles what the counts could not.

## previous_kv/previous_score are never touched on this model

Every site that reads or writes the carry is gated on **ratio 2**:

```c
ds41_attention_publish      (ds4.c:39998)  if (ratio == 2) { ... pool2(..., previous_kv[owner], ...) }
ds41_attention_publish_batch(ds4.c:40447)  if (ratio == 2u) { ... }
ds41_verify_commit          (ds4.c:41405)  if (ds4_layer_compress_ratio(il) != 2u) continue;
```

And FLASH has no ratio-2 layer. `ds4_expected_layer_compress_ratio` returns 0
for layers 0-1 then alternates 4 / 128, and the loader **`exit(1)`s** if the
GGUF disagrees (ds4.c:6134). So the ratios are provably {0, 4, 128}.

**The carry is written by nothing, saved by nothing, restored by nothing on
FLASH.** The tensors are allocated and then never used.

## But the probe tracks them anyway

`ds41_state_spans` (ds4.c:60177) includes them with no ratio check:

```c
if (i < 3 && (pos & 1u)) {
    spans[n++] = (ds41_state_span){g->previous_kv[i], 512u * 4u};
    spans[n++] = (ds41_state_span){g->previous_score[i], 512u * 4u};
}
```

so does the state-diff harness (ds4.c:75591). Three owners x two arrays = the
**6** residual arrays, exactly.

A dead tensor's backing memory is still real memory, and the batched and per-row
paths use scratch differently -- which is why the reads are systematically
different (`decode=-0.0817958`, `verify=0`, ~all 2048 bytes) rather than
randomly so. **It is residue, not state.**

## Consequence: CORE_INVARIANT may already be sufficient

If the 6 are noise, then `UNIFY_DECODE + CORE_INVARIANT` already leaves *every
live* array identical -- windows, compressed, index_cache all clean -- with MoE
and HC still batched. T21's "only the serial path reaches zero" would then be an
artifact of a probe counting tensors this model does not use, and the conclusion
built on it ("correct but pointless") would be wrong.

That is a large claim, so it needs the end-to-end test rather than the probe.

## The right criterion is byte-identical output

T21's own end-to-end check was "greedy output: BYTE-IDENTICAL to no-speculation."
That is what decides this, and `arrays_differing` was only ever a proxy.

Running four greedy `--temp 0` generations:

```
base1, base2        no speculation          -> reproducibility floor under streaming
spec_coreinv        speculation + CORE_INVARIANT
spec_plain          speculation, batched core
```

`base1` vs `base2` is the control and it is essential here: streaming is
intermittently non-deterministic (section 4), so without knowing whether the
baseline reproduces itself, an output difference proves nothing. Expected if the
analysis holds: base1 == base2 == spec_coreinv, and spec_plain differing.

---

# END-TO-END: the corruption is visible, and the baseline reproduces

Greedy `--temp 0`, 64 tokens, copy prompt, single box.

```
base1 vs base2                      IDENTICAL
base1 vs spec_plain (batched core)  DIFFER at byte 9
```

**The control passes.** Two no-speculation runs are byte-identical, so streaming
did not perturb this workload and an output difference is therefore meaningful.
(Section 4's intermittent non-determinism is real but did not fire here -- which
is exactly why the control has to be run every time rather than assumed.)

## What the corruption looks like

```
base1:      ... Need ensure exact. Let's copy carefully.
spec_plain: ... Need ensure exact wording. Let's copy carefully tur. Passage:
```

**`tur`** -- a spurious token spliced into the stream. Same class as the
`ROUTER_MAX_QUEues` case: the batched verifier commits its own non-invariant
argmax and the text silently acquires garbage. 315 bytes vs 333.

This is worth stating plainly because it is the thing throughput counters cannot
show: **speculative decode with the batched core produces visibly wrong text**,
and every benchmark in the 24.43 -> 29.75 series ran in that mode.

## The open question

`spec_coreinv` failed on a shell bug of mine (an expanded `VAR=1` word becomes
the command name, not an assignment -- rc 127), so the decisive comparison is
still outstanding. Re-running.

If `base1 == spec_coreinv`, then speculative decode is byte-exact with the MoE
and HC still batched, the residual 6 are confirmed as dead-tensor noise, and
T21's "correct but pointless" falls. If it differs, `CORE_INVARIANT` is not
sufficient and the serial path remains the only correct one.

---

# CORRECTION: the `tur` was not speculation, and the control was not matched

The section above attributed `tur` to the batched verifier. **Wrong.** The
baselines were not matched to the speculative runs:

| run | sidecar loaded | UNIFY_DECODE | drafting |
|---|---|---|---|
| base1 / base2 | no | no | no |
| spec_plain / spec_coreinv | yes | **yes** | yes |

`DS4_DS41_UNIFY_DECODE=1` deliberately routes decode through the sweep instead
of the specialised decode path -- a different kernel with different rounding --
so it changes output on its own, with no speculation involved. Comparing across
it measured that, not the verifier.

A matched baseline (sidecar loaded, `UNIFY_DECODE=1`, drafting **off**) produces:

```
We need reproduce exact passage word for word. ... Let's copy carefully tur. Passage:
```

**`tur` is in the baseline.** It is a `UNIFY_DECODE`/sidecar artefact, present
with no drafting whatsoever. My end-to-end corruption claim was wrong, and
`base1` (315 bytes) was never a valid reference for a 333-byte spec run.

## What the matched comparison actually shows

```
base_matched vs spec_plain     DIFFER at byte 217
base_matched vs spec_coreinv   DIFFER at byte 217
spec_plain   vs spec_coreinv   IDENTICAL
```

The first 216 bytes agree. The divergence is one token:

```
base_matched: pinned for lifetime of session.
spec_plain:   pinned for lifetime of the session.
```

So speculation does change greedy output by one token, and **`CORE_INVARIANT`
does not fix it** -- the two speculative runs are byte-identical to each other.
That is consistent with the probe's residual and with T21's conclusion.

## But one control is still missing

`base1 == base2` established reproducibility for the *unmatched* configuration.
There is no repeat of the **matched** configuration, and section 4 showed
streaming is intermittently non-deterministic. A one-token difference is exactly
the size of perturbation that intermittent streaming noise produces.

So `base_matched` is being run a second time. Until that returns:

* if `base_matched == base_matched2`, the one-token spec divergence is real and
  `CORE_INVARIANT` is insufficient;
* if they differ, output comparison cannot resolve this on a streaming single
  box at all, and the question needs the resident pair.

Three claims in this note have now had to be withdrawn after measurement (the
guard fix, the dead-restore diagnosis, the `tur` attribution). The pattern is
the same each time: reading code and matching signatures is not evidence, and
every comparison needs its own control run rather than a borrowed one.

---

# CONCLUSION: T21 confirmed, with a sharper mechanism

The missing control returned:

```
base_matched vs base_matched2   IDENTICAL     <- matched config IS reproducible
base_matched2 vs spec_plain     DIFFER at byte 217
spec_plain vs spec_coreinv      IDENTICAL
```

The matched configuration reproduces itself exactly, so the one-token divergence
is a real effect of speculation and not streaming noise.

**Final result, controlled:**

1. **Speculation changes greedy output.** One token at byte 217
   (`lifetime of session` vs `lifetime of the session`), against a
   self-reproducible matched baseline.
2. **`CORE_INVARIANT` does not fix it.** `spec_coreinv` is byte-identical to
   `spec_plain`.

So T21's "only the serial path is correct" **stands**, and my dead-tensor
hypothesis -- that `CORE_INVARIANT` might already suffice -- is refuted by
end-to-end evidence. That hypothesis was the most interesting thing in this
note and it did not survive its own test.

## The tension that explains everything

Under `CORE_INVARIANT` the probe reports every `window[]`, `compressed[]` and
`index_cache[]` array **clean**, yet the output still diverges. Both are true,
and together they say something the probe alone could not:

**State invariance is not sufficient. The logits path must be invariant too.**

The probe tracks persistent KV state. It does not track logits. Greedy decoding
takes an `argmax`, which is a *discrete* decision over a continuous quantity --
so a rounding-level difference in one row's logits, far too small to disturb any
tracked array, flips one token and the streams part. `CORE_INVARIANT` evidently
makes the KV state invariant without making the logits bit-identical.

This also retires `arrays_differing` as the acceptance criterion. It was the
gate T21 proposed and the one I spent this whole investigation optimising
against, and it can read zero on the arrays it watches while the output still
differs. **The only sound criterion is byte-identical generated text against a
matched, self-reproducible baseline.**

## What survived, and what the next person should do

Survived, each with a control attached:

* **The dispatch hypothesis is dead** -- byte-identical divergence sets across a
  kernel and thread-width change, in provably reproducible layers.
* **The batched core is where KV divergence lives** -- 43 -> 6, and not the
  kernel, tiling, or compress batching.
* **`previous_kv`/`previous_score` are dead on FLASH** -- every site gated on
  ratio 2, which the loader proves never occurs. The probe should skip them.
* **Streaming is intermittently non-deterministic**; matched configs here did
  reproduce, but that must be verified per experiment, never assumed.
* **The matched-control discipline**: sidecar, `UNIFY_DECODE` and drafting each
  change output independently. Vary one.

Next:

1. Instrument the **logits**, not the state -- compare row logits from a verify
   sweep against decode's for the same position. That is where the surviving
   divergence is, and no existing probe looks there.
2. Do it **resident on the pair**, so streaming is out of the picture entirely.
3. Treat `SERIAL_VERIFY` as the only correct mode until a logits-level fix is
   measured byte-identical end to end.

---

# LOGITS: CORE_INVARIANT makes row 0 bit-exact

The conclusion above said state invariance was not sufficient and the logits
must be checked. They now have been -- 28 lines inside the existing state-diff
harness, gated on `DS4_DS41_VERIFY_LOGIT_DIFF=1`, comparing decode's `logits`
against `vc.row_logits` row 0 for the same position.

```
batched core     decode_argmax=16(24.9922)  verify_argmax=16(25.0322)  match=1
                 ndiff=129280 of 129280     maxabs=0.254381  meanabs=0.046

CORE_INVARIANT   decode_argmax=16(24.7138)  verify_argmax=16(24.7138)  match=1
                 ndiff=0 of 129280          maxabs=0         meanabs=0
```

Two things, and the second is the surprise:

1. **The batched core perturbs every single logit**, by up to **0.254**. That is
   not rounding noise -- it is large enough to flip an argmax whenever the top
   two candidates are within a quarter of a logit, which is exactly the
   `QUE -> U` vs `ues` case (28.11/14.83 serial became 24.60/32.64 batched).
2. **`CORE_INVARIANT` makes the logits bit-identical.** Zero of 129,280 entries
   differ. Not "small", not "within tolerance" -- **exact**.

## Which sharpens the open question rather than closing it

`CORE_INVARIANT` produces bit-exact logits at the probed position, yet
`spec_coreinv` still diverged from the matched baseline by one token. Both
results are solid and they are only compatible one way:

**The probe compares row 0 only** -- the row that matches decode by
construction, since it attends to exactly the pre-sweep frontier. Rows 1..K-1 of
a K-row sweep attend to state built *inside* the sweep, and nothing has ever
compared those against a serial decode of the same tokens.

So the remaining divergence is confined to **rows >= 1**, and `CORE_INVARIANT`
fixing row 0 exactly is real but insufficient.

## Test in flight

`CORE_INVARIANT` + `DS4_DS41_VERIFY_ROWS1=1` restricts every verify to a single
row. If row 0 is bit-exact, that configuration must reproduce the matched
baseline exactly. Outcome:

* **IDENTICAL** -> the divergence lives entirely in rows >= 1, and the target is
  how a sweep publishes each row's KV before the next row attends to it.
* **DIFFER** -> row-0 exactness at one probed position does not generalise, and
  the logit probe needs to run across many positions rather than one.

Note this is a diagnostic, not a proposal: a one-row verify yields at most one
token per block and cannot pay for itself.

---

# ONE-ROW VERIFY IS PERFECT -- and an anomaly that is not yet explained

```
STATE_DIFF=1 (one-row verify, commits in full, no rollback)

m1_plain (batched core)   logits ndiff=0 of 129280   arrays_differing=0 of 56
m1_ci    (CORE_INVARIANT) logits ndiff=0 of 129280   arrays_differing=0 of 56
```

A one-row verify reproduces decode **exactly** -- state and logits -- in both
core modes. That is expected once stated: at `count == 1` a "batch" is a batch of
one, so the batched and per-row paths coincide. It cleanly separates the sweep
from the rollback and from acceptance, and it confirms the defect is specific to
**multi-row** sweeps.

## The anomaly

`CORE_INVARIANT` + `DS4_DS41_VERIFY_ROWS1=1` clamps every verify to one row.
Every verify in that run is therefore, by the measurement above, bit-exact. The
run reported:

```
cycles=44   accepted_draft=0
matched baseline vs run:  DIFFER at byte 217   (327 bytes vs 333)
```

**Zero draft tokens were accepted, every verify was provably bit-exact, and the
output still diverged** -- in length, not just one substitution.

That is not consistent with any story in this note. If no draft is committed and
each verify reproduces decode exactly, the generated text must match the
baseline. It does not.

Candidate explanations, none yet tested:

1. `DS4_DS41_VERIFY_ROWS1` clamps `draft_n` at **one** site
   (`if (draft_n > 1 && getenv(...)) draft_n = 1;`). If any other path reaches a
   sweep without passing it, some verifies were not one-row after all -- the
   cheap check is to log the actual row count per sweep rather than assume it.
2. With drafting enabled the committed token comes from `vc.row_logits` via
   `ds41_verify_commit` rather than from decode's `logits`. Bit-identical at the
   position probed, but that was **one** position; it has not been shown to hold
   at every position a 44-cycle run touches.
3. `accepted_draft` may not count what I assume. If it counts only tokens beyond
   the anchor, a block that replaces the *anchor* with the verify's argmax would
   change the stream while still reporting 0.

Explanation 3 is the most likely and the easiest to confirm, and it would mean
the counter -- not the engine -- misled me.

## The right next instrument

End-of-run text comparison has taken this as far as it can: it says *that* two
runs diverged, never *where*. The next probe should log the committed token id
per cycle in both the baseline and the speculative run, diff those streams, and
find the **first** cycle that differs -- then dump that cycle's row count, draft,
committed token, and row logits.

That converts "the text differs somewhere after byte 217" into "cycle N
committed token X where the baseline committed Y, from a sweep of R rows",
which is a debuggable statement. Everything before this section was measuring
proxies; this measures the event itself.

---

# THE ANOMALY IS DOCUMENTED BEHAVIOUR, AND THE CODEBASE SAYS SO TWICE

The `accepted_draft=0` run printed a banner I had not read:

```
ds4: DSpark direct verifier-state commits enabled; output may differ from
     one-token decode due to batched floating-point operation order
```

Its stats resolve the anomaly: `cycles=44 full=12 accepted_draft=0
draft_len_hist=6:14`. `ROWS1` clamped each commit to the anchor, so
`accepted_draft` (which counts `commit - 1`) reads 0 while twelve blocks still
committed **verifier state** rather than decode state. Explanation 3 of the
previous section was right: the counter misled me, not the engine.

## Guard 1: speculation is off by default under TP because it is known-wrong

ds4.c:67710 --

```c
if (e->dspark && opt->tp.role != DS4_TP_NONE &&
    getenv("DS4_DSPARK_TP_VERIFY") == NULL) {
    "DSpark speculative decode is disabled under network tensor parallelism:
     the verify path is implemented but not yet numerically correct.
     Decoding target-only."
    e->dspark = false;
}
```

with the comment above it: "Off by default until that is found;
`DS4_DSPARK_TP_VERIFY=1` re-enables it for debugging, **and produces wrong
output**."

The gate is still present in antigravity's HEAD (ds4.c:67988). Since the n-gram
drafter lives inside `ds4_session_prepare_dspark_draft`, which is unreachable
when `e->dspark` is false, **every TP2 speculative run must have set the
override**. It did: `DS4_DSPARK_TP_VERIFY=1` appears **1,469 times** in that
session's history.

So the entire 22.45 -> 29.75 series ran on a path the codebase explicitly labels
as producing wrong output, behind a flag whose documented purpose is debugging.

## Guard 2: a correct mode already exists, and is already known to be slow

ds4.c:80092 --

```c
const bool strict_dspark = e->support_kind == DS4_SUPPORT_DSPARK &&
                           (e->quality || e->dspark_strict);
```

reached via `--dspark-strict` or `--quality`, above a comment reading "...is
correctness-safe but cannot be faster than baseline."

## What this means for the whole investigation

The fast/wrong versus correct/slow tradeoff is **designed, implemented, and
documented**. T21's "correct but pointless" was not a discovery about a hidden
bug; it was a restatement of a choice the engine already exposes as two modes.

That does not make the batch-invariance work pointless -- its purpose is to
*remove* the tradeoff, so that a batched verify is both correct and fast. But it
reframes the target: the goal is to make `--dspark-strict` unnecessary, not to
find out why non-strict differs. Non-strict differs on purpose.

And it sets the honest baseline for any future speed claim:

**A speculative throughput number is only meaningful against `--dspark-strict`
(or with output diffed against a matched baseline). Measured without that, it is
a measurement of the fast/wrong mode, which the engine never claimed was
equivalent.**

---

# CORRECTION AND SHARPER CONCLUSION: --dspark-strict does not speculate at all

I measured "the honest speculative number" with `--dspark-strict`:

```
strict_base   generation:  9.93 t/s
strict_spec   generation: 10.09 t/s
output:       IDENTICAL, 456 bytes both
```

and read it as "strict mode speculates correctly for +1.6%". **Wrong.** No
DSpark stats line was emitted by either run, and the reason is one line
(ds4.c:80098):

```c
bool can_prepare_support_draft =
    !strict_dspark && ...
```

`strict_dspark` (set by `--dspark-strict` or `--quality`) **disables draft
preparation entirely**. Both runs speculated zero times. The 9.93 vs 10.09 gap
is run-to-run noise and the identical output is trivially identical -- neither
run did anything to diverge from.

The comment above it explains the design (ds4.c:80084):

> running ordinary decode once per draft token is correctness-safe but cannot be
> faster than baseline.

So strict mode does not verify carefully; it declines to draft, because the only
correctness-safe verification available -- one ordinary decode per draft token --
cannot beat baseline by construction.

## Which is the real conclusion of this whole note

**There is no correct speculative mode on this engine today.** The options are:

| mode | speculates | output |
|---|---|---|
| default (non-strict) | yes | **documented to differ** from one-token decode |
| `--dspark-strict` / `--quality` | **no** | baseline, trivially |
| under network TP | off by default | `DS4_DSPARK_TP_VERIFY=1` re-enables, "produces wrong output" |

That is a stronger statement than "there is a fast/wrong versus correct/slow
tradeoff", which is what I wrote in the previous section. There is no
correct/slow speculative mode to trade against -- the correct setting simply
turns the feature off.

## And it raises the stakes on the invariance work

Batch invariance is therefore not an optimisation that removes a tradeoff
between two working modes. **It is the only path to a correct speculative mode
existing at all.** Every measured gain in this project's speculative work --
mine and antigravity's alike -- is currently unrealisable with correct output,
and will stay that way until a multi-row sweep can reproduce serial decode.

The findings in this note that bear on that goal, restated as a target:

* a **one-row** sweep already reproduces decode exactly (state and logits, both
  core modes) -- so the machinery is sound at count 1;
* a **multi-row** sweep with the batched core perturbs every logit by up to
  **0.254**;
* **`CORE_INVARIANT` makes multi-row row-0 logits bit-identical** and clears
  every live state array, which is the closest anything has come;
* what remains unmeasured is rows **>= 1** of a multi-row sweep under
  `CORE_INVARIANT`, because no probe compares them against a serial decode of
  the same tokens.

That last item is the next experiment, and it is now the only one that matters:
decode K tokens serially capturing logits per step, run a K-row sweep, and diff
row r against decode step r for every r. If `CORE_INVARIANT` holds for all rows,
a correct speculative mode exists and the feature is reachable today.

---

# FINAL: the corruption is probabilistic, and CORE_INVARIANT costs more than it buys

## 1. The batched core is not batch-size invariant, and the error compounds

Per-row logits, 5-row sweep vs serial decode of the same tokens
(`DS4_DS41_VERIFY_ROW_LOGITS`, new probe):

```
batched core     row0 maxabs=0.254  row1 0.762  row2 1.657  row3 1.508  row4 0.724
CORE_INVARIANT   row0 0   row1 0   row2 0   row3 0   row4 0      (ndiff=0 every row)
```

Error grows down the sweep because each row inherits the previous rows'
perturbed KV. A one-row sweep is exact in both modes, which is why every
single-row probe came back clean.

Sampled across cycles (`..._N=12`), `CORE_INVARIANT` held at `pos0=212` **and**
`pos0=231` -- the latter satisfying `(pos+1) % 4 == 0`, so the compressed-cache
write parity was exercised and passed. `bad_rows=0` both times.

## 2. But the corruption is probabilistic, not systematic

Identical flags, only draft depth varied (2 vs 6), both core modes:

```
CORE_INVARIANT  d2 vs d6   IDENTICAL
batched core    d2 vs d6   IDENTICAL
CORE_INVARIANT  vs batched core   IDENTICAL   (all four 333 bytes)
```

The batched core perturbs logits by up to **1.66** and still produced
byte-identical text. A logit difference only changes output when it flips an
argmax, which needs the top-two gap to be smaller than the perturbation.

**That is why this bug hid for so long.** Benchmarks look clean because most
positions are not near-ties; it takes a genuine near-tie to surface, which is
exactly the `ROUTER_MAX_QUEUES -> QUEues` case (`U` 28.11 vs `ues` 14.83 serial,
becoming 24.60 vs 32.64 batched). The feature is not reliably wrong -- it is
*occasionally* wrong, unpredictably.

Caveat on this test's power: only 11 of 64 tokens came from accepted drafts, so
it has low sensitivity to rare flips. It shows corruption is not systematic; it
cannot show corruption is absent.

## 3. CORE_INVARIANT is a diagnostic, not a fix

```
no speculation, batched core      9.93 t/s
no speculation, CORE_INVARIANT    4.28 t/s
speculation,    CORE_INVARIANT    3.39 - 4.18 t/s
speculation,    batched core      3.41 - 3.55 t/s
```

(single box, streaming -- ratios matter, absolutes do not)

Per-row attention costs **more than half of decode throughput**, because it
removes precisely the batching that makes a K-row verify cheaper than K decodes.
Correctness and speed are obtained by the same mechanism pulling in opposite
directions. `CORE_INVARIANT` proves invariance is achievable and isolates the
culprit; it is not a shippable path.

## 4. Every "baseline vs speculation" comparison in this note was confounded

No flag combination turns speculation off while changing nothing else:

* `--dspark` alone drafts (DSpark), so a run without the n-gram flag is **not** a
  baseline -- `ci_base` showed `accepted_draft=5`;
* `--dspark-strict` disables drafting but also flips `quality`-adjacent
  decisions, so it changes decode too;
* `DS4_DS41_UNIFY_DECODE` changes decode by design;
* `DS4_DS41_CORE_INVARIANT` changes decode as well as verify
  (`total_count == 1` is in its predicate).

Sound comparisons are therefore **drafter-vs-drafter** or **depth-vs-depth**, or
direct logit comparison -- never "speculation off vs on".

## 5. Where this leaves the feature

* The defect is named: **`ds41_attention_batch` is not batch-size invariant**,
  with error compounding across rows.
* Invariance is achievable (`CORE_INVARIANT` reaches ndiff=0 everywhere sampled)
  but only by abandoning the batching that makes speculation pay.
* The shippable fix is what T21 said at the start and this note now has
  quantitative support for: **make the batched attention core produce
  bit-identical per-row results regardless of row count.** The target is
  specific -- 0.254 of logit error at row 0 growing to 1.66 by row 2.
* Until then, speculative throughput numbers are measurements of an
  occasionally-wrong mode, and the risk is not "slightly different text" but a
  corrupted identifier in the middle of otherwise correct output.

---

# LOCALISED: twelve `n_tokens == 1` special cases, five of them inside kernels

Chasing the batched attention stage component by component cleared everything
except the kernels themselves.

## Cleared -- do not re-examine

* **`ds4_gpu_dsv41_quantize`** is row-count invariant. `width % block == 0` is
  enforced, so each 32-element group lies inside one row and the warp reduction
  never spans rows. Quantising N rows together equals quantising each alone.
* **`ds4_gpu_dsv41_rope`** is count invariant. `row = head / heads` and
  `theta = (start + row * stride) * freq[lane]`, so every row derives its own
  position. `ds41_rope` is literally the same function with `count = 1`.
* **Batched compress/publish** is not the cause: `DISABLE_V41_BATCH_COMPRESS`
  alone measured **45** arrays (worse than the 43 baseline).
* **Causal masking is handled per row** -- the online kernel computes
  `qpos = pos0 + t` from `t = blockIdx.x`, so rows do not attend to later rows
  in the same batch.
* **Dispatch selection and reduction width** are not the cause (the earlier
  `FORCE_GENERIC` + `FIXED_THREADS` result).

## The actual cause: single-token fast paths inside the kernels

```
in-kernel:  ds4_cuda.cu:8330  (single_all = n_tokens == 1 && ratio == 0)
            ds4_cuda.cu:8378  (n_tokens == 1 && score_lanes_single == 0)
            ds4_cuda.cu:8398  (n_tokens == 1 && score_lanes_single == 4)
            ds4_cuda.cu:10165 (single_all)
            ds4_cuda.cu:12282 (n_tokens == 1 && ratio == 0)
dispatch:   ds4_cuda.cu:14971, 18781, 18812, 18814, 18830
MoE path:   ds4_cuda.cu:25383 (stream choice), 25420
```

**Twelve sites.** This is why the dispatch experiment failed to converge: forcing
both row counts onto the same kernel still leaves that kernel branching on
`n_tokens == 1` internally, and `DS4_CUDA_ATTN_FORCE_GENERIC` cannot reach a
branch inside a kernel.

The batch-size dependence is therefore **structural and distributed**, not a
single tiling or reduction-order bug. Decode at one row takes a materially
different arithmetic path from the same row inside a sweep, at five in-kernel
sites plus the dispatch layer.

## What a fix actually requires

Not "find the bug" -- the bug is a design choice repeated twelve times, each an
optimisation for the single-token decode case that is the hot path.

Either:

1. **Make the single-token paths bit-identical to the general path.** They exist
   for speed, so each must be shown to produce identical results, not merely
   similar ones. Five in-kernel sites, each needing its own proof.
2. **Delete the single-token paths** and always run the general path. Simplest
   and provably invariant, but it slows decode -- which is what
   `CORE_INVARIANT` already demonstrates (9.93 -> 4.28 t/s), and decode is the
   baseline speculation must beat.

Option 2 is what `CORE_INVARIANT` approximates today, and its measured cost is
why it cannot ship. Option 1 is the real project, and its size is now known:
five in-kernel branches, each requiring bit-exactness against the general path.

## A test harness is the prerequisite

Every measurement in this note cost a 5-minute model run. `attention_decode_batch_launch`
is `static` inside ds4_cuda.cu and not exported, so there is no way to call it
directly from a test. **Exporting it behind a test hook and asserting
row-0-of-N == single-row for random inputs would turn a 5-minute experiment into
a millisecond one**, and is almost certainly the right first move for whoever
takes this on -- before touching any kernel.

---

# THE REAL MECHANISM: float accumulation order, amplified through 40 layers

An in-situ check now measures the defect at its source
(`DS4_CUDA_ATTN_INVARIANCE_CHECK=1`, in
`ds4_gpu_attention_decode_mixed_batch_heads_tensor`): run row 0 alone and the
full batch on identical buffers, diff row 0.

## First version was wrong, and its error is instructive

It reported **38/38 divergent, maxabs 1.89**. That was my bug. The kernel derives

```c
const uint32_t first_raw_pos = pos0 + n_tokens - n_raw;
```

so holding `n_raw` fixed while dropping `n_tokens` from 6 to 1 shifts the raw-KV
window by five positions -- the two launches read **different keys**. Same for
`n_comp = (pos0 + n_tokens)/ratio`. Divergence was guaranteed by construction.

## Corrected, the divergence collapses by a factor of ~1.4 million

Deriving `n_raw` and `n_comp` for the single-row case so the *only* difference is
the row count:

```
uncorrected   maxabs 1.89
corrected     maxabs 1.3e-06   (~26000 of 32768 elements differ)
```

**1e-6 is float32 rounding at these magnitudes.** The batched attention kernel
is invariant to within accumulation order. The twelve `n_tokens == 1` fast paths
are **not** the cause -- that localisation is withdrawn.

## What actually happens

Per-layer attention differs by ~1 ULP because the batched and single-row paths
accumulate in different orders. That is not a logic bug; it is floating-point
non-associativity. Through 40 layers of residual, norm, HC and MoE it amplifies
into the **0.254 -> 1.66 logit differences** measured earlier, which occasionally
flip a near-tie argmax and corrupt a token.

The chain, now complete and each link measured:

```
~1e-6 per-layer attention difference   (accumulation order)
  -> compounds across 40 layers
  -> 0.25 - 1.66 logit difference      (row-logits probe)
  -> flips argmax only on near-ties    (identical text across draft depths)
  -> ROUTER_MAX_QUEUES -> QUEues       (rare, unpredictable corruption)
```

## Why this is the important finding

There is **no discrete bug to fix**. Every component is individually correct;
the batched and serial paths simply round differently, and a 40-layer stack
amplifies 1 ULP into something that changes text.

That reframes the whole feature:

* **`CORE_INVARIANT` works because it forces one accumulation order** (per-row
  everywhere), not because it repairs faulty logic. Its 9.93 -> 4.28 t/s cost is
  the price of that order, and it is inherent, not an implementation slip.
* **A correct-and-fast speculative verify requires batch-invariant kernels** --
  the same order regardless of row count -- which is a known-hard problem the
  reference implementations also face, not a bug hunt.
* **T21's "correct but pointless" was structurally right** for reasons deeper
  than it knew: correctness and speed here are obtained by fixing the
  accumulation order, and the fast order is the whole point of batching.

## Honest status of this note's claims

Withdrawn after measurement: the guard fix, the dead-restore diagnosis, the
`tur` attribution, the rows>=1 localisation, and now the twelve-fast-paths
localisation. Each fell to a control or a corrected instrument.

Standing, each with a control attached:

* the dispatch/thread-width hypothesis is refuted;
* a one-row sweep reproduces decode exactly;
* `CORE_INVARIANT` yields bit-identical logits on every row sampled, at both
  parities;
* the batched attention kernel is invariant to ~1e-6 given matched arguments;
* the corruption is probabilistic, surfacing only on near-ties;
* `--dspark-strict` disables drafting outright, so there is no
  correct-and-speculating mode today.

---

# THE NUMBER THAT DECIDES IT: ~1 in 75 verified rows commits a wrong token

15 sampled cycles x 5 rows = 75 verified rows, 160-token generation, copy prompt.

| config | rows with ndiff>0 | argmax flips | worst logit delta |
|---|---|---|---|
| batched core (default) | **75 of 75** | **1** | **3.24** |
| `CORE_INVARIANT` | **0 of 75** | **0** | **0** |

The `CORE_INVARIANT` arm is a perfect control: zero divergent rows, zero flips.
That validates the probe -- it is not manufacturing differences -- and confirms
forcing one accumulation order eliminates them entirely.

## Reading the rate

**Every** verified row diverges numerically under the batched core, and about
**1.3% of them flip the argmax** and therefore commit a different token than
serial decode would have produced.

Small-sample caveat: one event in 75 gives a wide interval (roughly 0.03%-7%).
The rate is uncertain; that it is nonzero and not negligible is not.

Note also `worst=3.24`, larger than the 1.66 seen in the earlier 5-row probe.
The logit perturbation is not bounded by the earlier figure.

## What that means in practice

At ~1 wrong token per 75 verified rows, a generation that speculates heavily
will commit multiple silent errors per thousand tokens. For prose this is
mostly invisible -- a different word choice that remains fluent, which is why
benchmarks and eyeballing never caught it. For **code, identifiers, JSON keys,
numbers, or any content where one token changes meaning**, it is a silent
correctness failure of exactly the `ROUTER_MAX_QUEUES -> QUEues` kind.

This is the concrete answer to "is non-strict speculation shippable":

* **not for code or structured output** -- the failure is silent, unpredictable,
  and frequent enough to hit a realistic generation;
* **the risk is not "slightly different text"** but a corrupted token inside
  otherwise correct output, which is the worst failure mode to debug downstream.

## Status of the mechanism

Measured end to end except one link:

```
~1e-6 per-layer attention difference        MEASURED (in-situ kernel check)
  -> compounds across 40 layers             INFERRED -- endpoints measured,
                                               growth in between is not
  -> 0.25 - 3.24 logit difference           MEASURED (row-logits probe)
  -> flips argmax on ~1.3% of rows          MEASURED (this section)
  -> silent token corruption                MEASURED (ROUTER_MAX_QUEUES)
```

The compounding step remains an inference. Both endpoints are measured and no
alternative mechanism is in evidence, but the per-layer growth curve has not
been instrumented and should not be described as verified.

---

# THE MARGIN GATE FAILS, AND WHY: there are two corruption channels

Implemented `DS4_DSPARK_SAFE_MARGIN=X` -- accept a drafted row only when the
target's top-1 beats top-2 by more than X (ds4.c accept loop, plus
`ds4_argmax_with_margin`). Rationale: the perturbation only changes the
committed token when the top-2 gap is smaller than it, so requiring a margin
should turn "occasionally wrong" into "occasionally declines".

## The premise checked out

Per-row gap correlated with perturbation, 75 rows, 160 tokens:

```
the one flip:  maxabs=0.900  decode_gap=0.084  verify_gap=0.007
gaps:          <1: 32 | 1-2: 15 | 2-4: 18 | 4-8: 2 | >=8: 8
largest perturbation: 3.24
```

The flip occurred exactly where predicted -- a near-tie swamped by a larger
perturbation. Mechanism confirmed, not inferred.

## But the gate does not deliver correctness

```
margin=0    4.98 t/s  accepted_draft=35  text: (reference)
margin=4.0  4.92 t/s  accepted_draft=30  text: DIFFERENT from margin 0
margin=8.0  4.96 t/s  accepted_draft=28  text: SAME as margin 0
```

**Non-monotonic.** A stricter margin (8.0) reproduces margin 0 while a looser
one (4.0) diverges. If rejection moved output toward correctness, this could not
happen.

The explanation is already in this note. Rejecting a row shortens the commit,
which changes block boundaries, which changes the state trajectory -- and the
batched sweep perturbs persistent state *whether or not anything is accepted*.
`DS4_DS41_VERIFY_ROWS1` demonstrated this directly: `accepted_draft=0`, output
still differed.

So there are **two** corruption channels:

1. **A wrong argmax is committed.** The margin gate addresses this.
2. **The sweep leaves perturbed state behind.** The gate cannot touch this, and
   it fires on every verify regardless of acceptance.

Channel 2 is sufficient on its own to change output, so no accept-rule change
can make speculation correct. Only making the sweep itself reproduce serial
decode does -- which is `CORE_INVARIANT` (or serial verify), measured at
9.93 -> 4.28 t/s.

The margin gate is left in, defaulting to 0 (off). It is a genuine mitigation
for channel 1 and costs ~14% of accepted tokens at margin 4.0, but it must not
be described as making speculation correct.

## And on this configuration speculation does not pay anyway

```
no speculation (strict)          9.93 t/s
speculation, batched core        4.98 t/s
speculation, CORE_INVARIANT      4.18 t/s
```

Single box with streaming, so absolutes are not comparable to the TP2 pair --
but the ordering is stark: **speculation is roughly half the speed of not
speculating**, before correctness is even considered. Verify cost dominates at
the acceptance rates this workload produces (35 accepted draft tokens across 160
generated).

## Status against the goal

The goal was speculation both fixed and faster. Measured outcome:

* **Fixed** is reachable -- `CORE_INVARIANT` gives bit-identical logits on every
  row sampled, at both parities.
* **Faster** is not, by any route tried: per-row attention costs more than half
  of decode, and the accept-rule mitigation cannot address channel 2.
* The two are coupled through the same mechanism -- correctness comes from
  forcing one accumulation order, and the fast order is the point of batching.

What would actually be required is a batch-size-invariant attention kernel:
identical accumulation order at any row count, with the *fast* order. That is
the known-hard problem the reference implementations also face, and nothing in
this engine's current kernels or flags approximates it.

---

# GOAL TEST: speculation loses in every mode on this configuration

```
ci_nospec  (CORE_INVARIANT, no speculation)   5.37 t/s
ci_spec    (CORE_INVARIANT, speculation)      5.13 t/s   -4.5%   accepted_draft=35
bc_spec    (batched core,  speculation)       4.85 t/s   -9.7%   accepted_draft=35
```

and for reference, without `CORE_INVARIANT` and without speculation, ~9.93 t/s.

So on a single box with streaming:

* **the fastest correct configuration is not speculating at all**;
* speculation in the correct mode costs 4.5%;
* speculation in the fast/wrong mode costs 9.7% -- it is not even faster than
  the correct mode, so the corruption buys nothing here;
* `CORE_INVARIANT` itself costs roughly 45% of decode.

35 accepted draft tokens across 160 generated is not enough to amortise the
verify at any acceptance rate this workload produces.

## The caveat that matters most

**Every measurement in this note was taken single-box with `--ssd-streaming`,
because the TP2 pair was occupied throughout.** That is not the configuration
the speculative gains were claimed on. On the pair, resident, decode is ~44
ms/token and a K-row verify amortises against a very different baseline --
which is exactly why antigravity measured gains there.

So the speed conclusion above is **specific to this configuration and does not
transfer**. What does transfer is the correctness analysis: the two corruption
channels, the bit-identical logits under `CORE_INVARIANT`, the ~1.3% flip rate,
and the mechanism -- none of those depend on streaming or on being single-box.

## The decisive experiment that remains

**Measure `CORE_INVARIANT` speculation on the TP2 pair, resident**, against a
matched no-speculation baseline, with output diffed. That is the one run that
would answer whether correct speculation pays in production, and it needs the
pair free. Everything required is in place:

* `DS4_DS41_CORE_INVARIANT=1` for correctness (verified bit-identical here);
* the probes in `T29_logit_diff_probe.patch` to confirm invariance holds under
  TP as well -- it has only been verified single-box;
* `DS4_DSPARK_SAFE_MARGIN` as an optional channel-1 mitigation, off by default.

Note also that speculation under network TP is gated off by default as
"not yet numerically correct" (ds4.c:67710), so that run must set
`DS4_DSPARK_TP_VERIFY=1` -- and the invariance probes should be run there
first, since none of this note's kernel measurements were taken under TP.

---

# MAJOR CORRECTION: CORE_INVARIANT is free, and my throughput method was broken

## The method error

I reported "CORE_INVARIANT costs ~45% of decode" from 9.93 t/s (without) against
4.28 t/s (with). **Those came from two separate script invocations**, and the
same configuration later measured 5.49 t/s. Cross-invocation timing comparisons
on this streaming setup are invalid -- expert-cache and page-cache residency
carry across processes and dominate.

Within a single script, back-to-back, timing is reproducible:

```
three identical runs:  5.62 / 5.66 / 5.53 t/s   (~2% spread, outputs IDENTICAL)
```

## The corrected measurement

Interleaved A/B, no speculation, so drift hits both arms equally:

```
pair 1:  off 5.50   on 5.51
pair 2:  off 5.81   on 5.57
pair 3:  off 5.57   on 5.49
mean     off 5.63   on 5.52      difference ~1.8%, inside the noise band
```

**`DS4_DS41_CORE_INVARIANT=1` has no meaningful decode cost.**

## Why this changes the conclusion

The story this note had been telling was: correctness is achievable but costs
half of decode, so it can never pay. That was wrong, and the correction is
favourable.

Correct speculation is available at **negligible decode cost**. The only cost is
the verify overhead itself, which makes "does speculation pay?" an ordinary
question about acceptance rate and verify cost rather than a structural
impossibility.

On this single box with streaming it still does not pay -- the goal test
(back-to-back, so valid) measured 5.37 no-spec against 5.13 with, about -4.5%,
because acceptance is low and verify is expensive relative to a streaming
decode. **But that is a statement about this configuration, not about the
feature.** On the resident pair, decode is ~44 ms/token and verify is
`60 + 17K` ms, which is where speculation was measured to pay.

## What this means for the recommendation

Previously: "correctness costs 45%, so correct speculation cannot be faster than
baseline."

Now: **`CORE_INVARIANT` should simply be on.** It makes the verify bit-identical
to serial decode at no measurable decode cost, which removes the silent
corruption entirely -- the ~1.3% argmax flip rate, both channels. Whether
speculation on top of it is a net win is a separate question that the resident
pair answers.

## Timing claims in this note, re-graded

| claim | status |
|---|---|
| CORE_INVARIANT costs ~45% of decode | **WITHDRAWN** -- cross-invocation artifact |
| speculation loses 4.5% correct / 9.7% wrong | stands (back-to-back, above noise) |
| speculation ~half the speed of no speculation | **WITHDRAWN** -- same artifact |
| every value-comparison result | unaffected -- exact comparisons, not timings |

The last row is the important one. Bit-identical logits, the 1.3% flip rate, the
two corruption channels, the determinism control and the gap/perturbation
correlation were all measured by comparing values, not elapsed time, and none of
them depend on how fast the machine happened to be running.

---

# END-TO-END TEXT COMPARISON CANNOT RESOLVE THIS -- and why that is fine

Two further attempts at a confound-free correctness demonstration, both
instructive failures.

## Attempt 1: disable drafting via NGRAM_MIN -- broken control

Setting `DS4_DSPARK_NGRAM_MIN=64` was meant to make the lookup never match. It
does not: the drafter clamps `ng_min` to `ng_max` (~8), so drafting continued --
`ci_nodraft` reported `accepted_draft=6`, not 0. It read 0 in the batched arm
only by accident, because that run generated different text containing no long
match. The "both arms DIFFER" result was speculation compared against
speculation, and means nothing.

## Attempt 2: depth-independence at 160 tokens -- underpowered

Only `DS4_DSPARK_NGRAM_DRAFT_MAX` varied (2 vs 6), a perfectly matched pair:

```
CORE_INVARIANT ON    d2 (16 accepted) vs d6 (35 accepted)   IDENTICAL
CORE_INVARIANT OFF   d2 (16 accepted) vs d6 (35 accepted)   IDENTICAL
```

The **batched arm is also identical**, and it is known non-invariant. So the
test cannot discriminate: at ~1.3% flips over 35 accepted draft tokens the
expected number of flips is ~0.45, and most runs see none.

## Which is a statement about the instrument, not the engine

Generated text is a **lossy detector** of this defect. A logit perturbation only
becomes visible when it flips a near-tie, so end-to-end comparison discards
almost all the signal. That is precisely why the corruption survived every
benchmark in this project: the walkthrough's "bit-for-bit identical" checks were
looking through the same lossy instrument.

The direct probes do not have this problem:

```
per-row logits, batched core     75 of 75 rows diverge, up to 3.24
per-row logits, CORE_INVARIANT    0 of 75 rows diverge, maxabs 0
determinism control (N vs N)     38 of 38 clean, maxabs 0
```

Zero of 75 divergent rows, at both parities, against 75 of 75 for the batched
core, is stronger evidence than any text diff -- it is exact, per-row, and does
not depend on a near-tie happening to occur.

**So the correctness claim for `CORE_INVARIANT` rests on the logit probe, not on
text comparison, and that is the right place for it to rest.** Anyone re-testing
this should use the probes in `T29_logit_diff_probe.patch` rather than diffing
output, which is what misled this project for days.

## Final state of the recommendation

* **Enable `DS4_DS41_CORE_INVARIANT=1`.** It makes the verify's logits
  bit-identical to serial decode (0/75 vs 75/75) at a measured ~1.8% decode cost
  (interleaved A/B), which is inside the noise band.
* **Do not rely on text diffs to validate it.** Use the row-logits probe.
* **Whether speculation then pays is unanswered** and needs the resident pair.
  On this single box with streaming it loses ~4.5%, but streaming decode is slow
  enough that the verify cannot amortise -- the pair's economics
  (44 ms/token decode, 60 + 17K ms verify) are entirely different.

---

# DEPTH CURVE: deeper drafts are monotonically WORSE in the correct mode

All arms `CORE_INVARIANT=1`, back-to-back in one script, bracketed by a repeated
no-speculation reference to bound drift:

```
nospec    5.54 t/s
depth2    5.62 t/s   16 accepted
depth3    5.22 t/s   23 accepted
depth4    5.16 t/s   29 accepted
depth6    5.01 t/s   35 accepted
nospec2   5.49 t/s        <- drift 0.9%, so the sequence is valid
```

**Throughput declines monotonically with draft depth even as accepted tokens
rise.** From depth 2 to depth 6, 19 additional accepted draft tokens cost 11%
throughput. Each extra accepted token is net negative.

The decline (5.62 -> 5.01, ~11%) is far outside the ~2% noise band. The depth-2
advantage over nospec (5.62 vs 5.515 mean) is ~1.9%, i.e. at the noise floor --
**break-even at best, not a win.**

## Why, structurally

`CORE_INVARIANT` makes the verify run attention **per row**. So a K-row verify
performs K attention passes; only the MoE and HC stay batched. Verify cost
therefore scales with K at close to decode's own attention cost, and the fixed
per-sweep cost is pure overhead. Batching buys only the MoE amortisation, which
on this configuration does not cover it.

That is the correct-mode economics stated plainly: **speculation's gain comes
from batching the verify, and the correctness fix removes batching from exactly
the component that made it cheap.**

## This contradicts the project's recent direction

The last several days of work raised block caps 16 -> 32 -> 48 -> 64 and tuned
drafters to produce longer drafts. In the correct mode on this configuration,
**that is the wrong direction** -- depth 2 beats depth 6 by 11%. The cap-raising
gains were measured in the fast/wrong mode, where the verify stays batched and
deeper blocks do amortise.

## What it implies for the pair

The structure carries over; the constants do not. On the resident pair the MoE
is a much larger share of layer cost (T23: `ffn` 55% of the chain) and the
per-layer TP gate is amortised across rows, so batched-MoE-with-per-row-attention
may still pay there even though it does not here. **The sign of the depth curve
on the pair is an open question and is the single most useful thing left to
measure.**

If it slopes the same way, the correct configuration is shallow drafts, and
every cap above ~4 is counterproductive.
