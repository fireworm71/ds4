# T21: plan -- what llama.cpp's DSpark teaches us, and how to get spec decode on the pair

Written 2026-09-19, after reading llama.cpp's DSpark implementation
(`/home/jason/llama.cpp`, 4d19b2876, 2026-08-26) and our own TP verify path.

## 1. What llama.cpp actually does

DSpark is implemented as a variant of DFlash -- a **block denoising**
speculator, matching the paper's "semi-autoregressive generation".

```c
// common/speculative.cpp
const int32_t n_block_tokens = n_draft + (is_dspark && sample_from_anchor ? 0 : 1);
for (int32_t i = 0; i < n_block_tokens; ++i) {
    common_batch_add(batch, i == 0 ? dp.id_last : mask_token_id, n + i, { seq_id }, true);
}
int ret = llama_decode(ctx_dft, batch);          // the WHOLE block, one pass
```

Then confidence-truncate and sample each position:

```c
const float * conf = params.p_min > 0.0f ? llama_get_embeddings_nextn(ctx_dft) : nullptr;
for (int32_t i = i_draft_beg; i < n_block_tokens; ++i) {
    if (conf && conf[(size_t) idx * n_embd_dec] < params.p_min) break;
    common_sampler_sample(smpl, ctx_dft, idx, true);
    ...
}
```

Points of agreement with us, which is reassuring:

| | llama.cpp | ours |
|---|---|---|
| target features | concat hidden at `target_layer_ids` | same |
| `n_embd_enc` | `target_layer_ids_n * n_embd_tgt` | `in_dim = 3 * 5120 = 15360` |
| block size | from `dflash.block_size` metadata | `dspark_block_size` = 5 |
| confidence | truncate below `p_min` | truncate below threshold |

**Points of difference.** llama.cpp draws the whole block in one drafter
decode. We run a three-stage chain plus a separate confidence-head pass
(`prop_chain` + `prop_conf0`). That is a real architectural difference, but it
is *not* our bottleneck -- see the economics in section 4 -- so it is not on
this plan.

**What is transferable:**

* `convert_hf_to_gguf.py --dspark` exports the DSpark draft tensors from
  `DeepseekV4ForCausalLM` as a standalone GGUF. This is an independent producer
  of the sidecar we currently have exactly one good copy of.
* Its confidence-truncation semantics are a cross-check on ours.

**What is not transferable:** its verify is an ordinary `llama_decode` on the
target, with multi-device placement handled beneath it in ggml. It has no
concept of a TP-restricted verify because it never needs one. Our network TP is
two processes in lockstep over an explicit wire protocol, so "just batch it"
is not available to us. Do not read llama.cpp's silence here as evidence that
our problem is easy.

## 2. The finding that actually changes the plan

**Our TP speculative-verify protocol already exists, and it is symmetric.**

```c
// ds4_tp.h
DS4_TP_FRAME_VERIFY = 11,
DS4_TP_FRAME_VERIFY_COMMIT = 12,

DS4_TP_VERIFY_ROLLBACK_REPLAY = 0,
DS4_TP_VERIFY_COMMIT_FULL     = 1,
DS4_TP_VERIFY_COMMIT_PREFIX   = 2,
```

Leader side is `ds4_session_eval_dspark_speculative_argmax` (ds4.c:75251+):
announce the block, mutate, decide, send the commit mode. Worker side is
`ds4_session_tp_spec_cycle` (ds4.c:76025): snapshot the frontier, apply the
drafts, verify, wait for the commit frame, apply FULL / PREFIX / ROLLBACK.

**But both ends verify with `metal_graph_verify_suffix_tops` on `s->graph`**,
and for V4.1 that graph holds only the drafter's scratch. Our own comment says
so:

```c
/* V4.1 verifies on its own graph. Everything below this point -- frontier
 * snapshot, suffix verify, replay -- operates on s->graph, which for V4.1
 * holds only the drafter's scratch, so it can never verify a V4.1 draft. */
if (ds4_session_is_ds41(s)) { ... return rc; }   /* returns before the TP path */
```

So V4.1 returns before ever reaching the working TP machinery. The protocol is
built; the V4.1 verify is simply not wired to it.

### The two hard parts are already done

* **Running N rows under TP.** `ds41_graph_verify_rows` runs its rows through
  `ds41_graph_prefill_sweep`, which **already supports `tp_world == 2`** --
  it gates on `ds41_tp_batch_enabled(g)`, true for TP2 unless an ablation env
  var is set. This is the same machinery TP2 prefill chunks use every run.
* **Rolling back.** `ds41_verify_commit` touches only `g->window[il]`,
  `g->previous_kv/score[owner]`, `g->history` and `g->pos` -- all rank-local
  GPU state. No messaging required; each rank rolls back its own.

Sizes fit: `DS4_TP_BATCH_MAX_ROWS = 8`, block size 5, `DS4_SPEC_PREFIX_SLOTS = 5`.

**What is missing is lockstep**, not capability: both ranks must run the same
rows and apply the same keep count. That is exactly what the existing commit
protocol carries.

This makes the `g->tp_world == 2` term in `ds41_graph_verify_rows` look
conservative rather than load-bearing. 24f087d called it "the ported design
asserts the same restriction", which is a statement about the source design,
not a proof that our TP2 cannot satisfy it.

## 3. The plan

### Phase 0 -- close the correctness hole (small)

Move the TP refusal into `ds4_session_ds41_dspark_verify`'s **entry** guard so
it returns `n_accept` ("nothing mutated: stay serial") instead of falling into
the `-1` path that invalidates the frontier.

Today the load-time decline (`367ffcf`) keeps this unreachable, but that is a
policy, not an invariant: anything that re-enables DSpark under TP resurrects a
run-killing abort. The entry guard is the invariant and is worth having whether
or not we do the rest.

### Phase 1 -- a second clean sidecar (independent, do first)

Regenerate the DSpark GGUF with llama.cpp's converter, scan it with the new
load-time guard, and diff its tensor inventory against `q4v3`.

We currently have **one** good copy of an artifact whose other two copies are
corrupt, and no independent way to make another. That is the single largest
fragility in this whole area and it is cheap to remove.

### Phase 2 -- GATE: measure acceptance before building anything

**Do not start Phase 3 until this number exists.**

The only clean acceptance sample we have is 3 accepted of 16 proposed
(18.75%) at 2048 ctx on a single box. If that holds at realistic context, the
TP verify is not worth building. vLLM reports 78% on structured text and 34%
on prose for the same model family, so our 18.75% is either a small-sample
artifact, a threshold we have set badly, or a real gap in our drafter.

Acceptance is a property of the model and the drafter, **not of TP** -- so it
can be measured single-box, under streaming, with no TP work at all. Run it at
16K context over a few hundred tokens on both prose and code prompts and report
`accept_rate` and `accepted_len_hist`.

If acceptance lands near vLLM's range, Phase 3 pays for itself (section 4). If
it lands near 18%, the next work is the drafter, not the verify.

### Phase 3 -- wire the V4.1 verify to the TP protocol

Three edits, in this order:

**3a. Worker** (`ds4_session_tp_spec_cycle`): branch on `ds4_session_is_ds41(s)`
and verify on `s->ds41_graph` via `ds41_graph_verify_rows` instead of
`metal_graph_verify_suffix_tops`. On the commit frame, apply
`ds41_verify_commit(g, &vc, keep)` for FULL/PREFIX and a full rewind for
ROLLBACK_REPLAY. Keep the existing divergence check -- "the leader committed a
block our verify failed to apply" must stay fatal.

**3b. Leader** (`ds4_session_ds41_dspark_verify`): when `tp_world == 2`, call
`ds4_tp_send_verify()` **before** mutating anything, run the rows locally,
compute `keep` from the per-row logits exactly as the single-box path does,
send `ds4_tp_send_verify_commit(mode, keep)`, then `ds41_verify_commit(keep)`.
The ordering constraint is already documented in the non-V4.1 leader code:
announce the block first, because the worker runs its half and then blocks on
the commit decision.

**3c.** Drop `g->tp_world == 2` from `ds41_graph_verify_rows`'s entry guard.

Then re-enable DSpark under TP by removing the load-time decline from `367ffcf`.

### Phase 4 -- validate, then measure

* **Byte-identical greedy output** with DSpark on versus target-only, same
  prompt. This is the invariant 24f087d explicitly protected ("greedy output
  stays byte-identical to the pre-port baseline") and it is the only way to
  know the two ranks stayed in lockstep. A speed number without it is
  worthless.
* A deliberate mid-block rejection, to exercise COMMIT_PREFIX and ROLLBACK on
  both ranks rather than only the happy path.
* Only then: tg against the **21.44 tg/s** control (Q2 TP2 resident, 16K, 256
  tokens, measured 2026-09-19).

## 4. Economics -- why this is worth doing at all

Measured on the TP2 run that reached the verify: `propose=26.930 ms` over 5
cycles, about **5.4 ms per cycle**. Target decode is 1/21.44 = **46.7 ms per
token**. So drafting costs roughly 11% of one token's decode.

With a 5-token block, breakeven is an average acceptance of well under one
token per cycle. At vLLM's reported 34% (prose) the block yields ~1.7 tokens
per cycle; at 78% (structured) it is far more. Either would be a large win.

The risk is entirely in the acceptance rate, which is why Phase 2 is a gate and
not a formality.

## 5. What this plan deliberately does not do

* **Restructure the drafter** into llama.cpp's single-pass block decode. Our
  propose cost is ~11% of a token; even eliminating it entirely is a smaller
  prize than getting any speculation at all working. Revisit only if Phase 2
  shows the three-stage chain is also hurting acceptance.
* **Chase vLLM's "adaptive verification".** Fixed 5-token blocks first.
* **Compare our tg against vLLM's 84 tok/s or SGLang's 37.9 tok/s.** Different
  quantisations, different KV paths, different model variants, different box
  counts. Those numbers say speculation works on this hardware; they say
  nothing about our delta.

---

# PHASE 2 RESULT: the gate passes -- but `accept_rate` was the wrong metric

Measured 2026-09-19, Q2 single box, SSD streaming, 16K prompt, 256 generated
tokens, clean `q4v3` sidecar.

| run | cycles for 256 tok | accept_rate | avg_accept | blocks drafted | miss_first |
|---|---|---|---|---|---|
| prose, scheduler on | 180 | 20.22% | 0.200 | 33 | 2 / 180 |
| code, scheduler on | 222 | 10.87% | 0.045 | 21 | -- |
| prose, `DS4_DSPARK_SCHEDULER=0` | **142** | 17.70% | 0.401 | 64 | 8 / 142 |

## The metric that matters is tokens per target step, not accept_rate

`accept_rate` is `accepted_draft / proposed`, so proposing 5 and accepting 2
scores 40% while *doubling* throughput. It is the wrong gate. The number that
determines speedup is how many tokens come out per target step:

```
scheduler on   256 tokens / 180 cycles = 1.42 tokens per target step
scheduler off  256 tokens / 142 cycles = 1.80 tokens per target step
```

**1.80 tokens per target step is a 44% reduction in target work.** The earlier
"18.75%, probably not worth it" read was an artifact of the metric, not of the
drafter.

## The drafter is not defective

`miss_first = 2 / 180`: the drafter's first token matches the target's own
argmax 99% of the time. That is not a broken drafter. The low `accept_rate`
comes from later block positions, which is expected of a semi-autoregressive
block draft and is exactly what confidence truncation is for.

The code prompt scoring *worse* than prose (10.87% vs 20.22%) initially looked
like an inverted result against vLLM's 78%-on-structured claim. With the
scheduler in play it is not comparable: the code run drafted only 21 blocks in
222 cycles because the backoff had latched. Content sensitivity should be
re-measured with the scheduler off before drawing any conclusion.

## The scheduler backoff is the largest single lever

With the scheduler on, **113 of 180 cycles were skipped before drafting**. Its
`many_no_draft` term (`no_draft * 2 >= cycles`) is self-reinforcing: skipped
cycles count as no-draft, which keeps the condition true. Turning it off took
cycles from 180 to 142 for the same 256 tokens.

This is tuning, not redesign, and it is worth doing on its own -- it costs
nothing and applies to single-box decode today.

## Revised economics for TP2

Decode at Q2 TP2 resident is weight-bandwidth-bound: 46.7 ms/token against a
prefill rate of 422 tok/s. A 5-row verify batch reads the same weights once, so
it should cost close to a single decode rather than five. Taking verify ~= one
decode and propose at the measured 5.4 ms/cycle:

```
baseline     256 tokens * 46.7 ms                  = 11,955 ms
speculative  142 cycles * (46.7 + 5.4) ms          =  7,398 ms   -> ~1.6x
```

Break-even is a verify batch costing under ~1.7x a single-token decode. That is
a low bar for a batched path that already exists.

**Verdict: proceed to Phase 3.** The residual risk is concentrated in one
measurable quantity -- the cost of a 5-row verify batch under TP2 -- and not in
whether the drafter works.

---

# PHASE 3/4 RESULT: TP lockstep works; the batched verify does not

## Phase 3 is implemented and the protocol is sound

The V4.1 verify is now wired to the existing TP frames: the leader announces
the block with `DS4_TP_FRAME_VERIFY` before touching anything, both ranks run
`ds41_graph_verify_rows` and meet inside the MoE exchange, the leader sends
FULL or PREFIX with the keep count, and both apply `ds41_verify_commit`.

On the pair this **runs to completion with no divergence, no aborts and no
errors** -- 256 tokens at 16K context, `full=32 partial=7`, worker log clean.
The `tp_world == 2` guard really was liftable: `ds41_graph_prefill_sweep`
handles the rows and `ds41_verify_commit` is rank-local, exactly as section 2
predicted.

## Phase 4 failed, and found something bigger

Greedy output (`--temp 0`, 128 tokens, identical prompt) is **not** preserved:

```
TP2      common prefix 14 bytes, then: "...CYP? Need detailed technical
         explanation of how tensor parallelism split MoE layer<|begin_of"
single   common prefix 16 bytes, then: "...mixture-of-exper MoE layer across
         two machines."
```

Stray tokens and spliced word fragments -- well outside the "batched
floating-point operation order" tolerance the engine warns about.

**Single box diverges too.** This is not a TP bug.

### The engine's own diagnostic localises it exactly

`DS4_DS41_VERIFY_COMMIT1=1` accepts only `drafts[0]`, which the caller has
already matched against the target's own logits. Its comment states the
contract:

> Any divergence from serial output under that setting is the batch's effect
> on state, not the accept rule -- a diagnostic separation, not a mode.

Run with it, single box: **diverges at the same 16-byte prefix, into the same
"mixture-of-exper MoE" corruption** as the full speculative run.

**So the fault is the verify batch's effect on graph state, not the accept
rule, not the row logits, and not anything added for TP.** Running `count` rows
through `ds41_graph_prefill_sweep` with the verify context armed and then
rolling back does not leave the graph where serial decode would have left it.

This is what 24f087d set out to avoid -- "the rows go through the real sweep
rather than a narrowed copy, because the verifier has to agree with what
prefill/decode would compute" -- and it does not agree.

### Why this explains every earlier number

* Low acceptance: the per-row logits come from a batch whose state is already
  wrong, so later rows rarely match the drafter.
* Worse acceptance under TP (max 1 accepted vs 4 single box) and a shorter
  draft-length profile: the batch path diverges further under TP, and the
  drafter's captured hidden comes from that same state.
* 14.81 tg/s against the 21.44 control: paying propose + verify for almost
  nothing.

## Current state

`--dspark` is **disabled under network TP by default**; `DS4_DSPARK_TP_VERIFY=1`
re-enables the new path for debugging and produces wrong output. The Phase 3
code is kept because the lockstep half of it is correct and worth preserving.

**Single-box `--dspark` on V4.1 also corrupts output** and is left as-is: it is
double opt-in (`DS4_V41_DSPARK_ENABLE=1` plus `--dspark`) and labelled
experimental, and disabling it is a policy call rather than part of this work.
It should not be used until the batch-state defect is fixed.

## What to do next

The target is now one specific, single-box-reproducible question: **why does
`ds41_graph_verify_rows` leave the graph in a different state than serial
decode?** No TP, no drafter, no artifacts involved -- it reproduces with
`DS4_DS41_VERIFY_COMMIT1=1` on one box, which makes it cheap to bisect.

Prime suspects, in order:

1. **What the verify snapshot does not save.** `ds41_verify_ctx` captures
   `window[il]` and four `previous_kv/score` owners. If the sweep mutates any
   other rolled-forward state -- Engram, carry buffers, the compressed-KV
   ring's own bookkeeping -- the rollback silently leaves it advanced.
2. **The `previous_kv/score` owner set.** `ds41_verify_commit` only restores
   owners whose `ds4_layer_compress_ratio(il) == 2`, hard-coded to layers
   2/8/14/20. If any other layer carries that state, it is never restored.
3. **Batch vs single-row arithmetic in the sweep.** Compare a 1-row verify
   against a plain decode step first: if even `count == 1` diverges, the
   problem is the sweep, not the rollback.

That last check is the cheapest and should come first -- it splits "the batch
is wrong" from "the rollback is incomplete" in one run.

---

# PHASE 5: the defect is state, not arithmetic

Three more measurements narrow it to one statement.

## The rollback is exonerated

`DS4_DS41_VERIFY_ROWS1=1` (added here) clamps the verify to a single row.
`ds41_verify_commit` only restores the rings when `keep < rows`, so a one-row
block that commits in full **rolls nothing back**. Paired with
`DS4_DS41_VERIFY_COMMIT1=1` the accept rule is out of the picture too.

Result: **still diverges.** Nothing was rolled back and nothing beyond
`drafts[0]` was accepted, so neither the rollback nor the accept rule can be
the cause.

## The sweep's arithmetic is exonerated

```
ds4 --decode-consistency 16
ds4: decode-consistency compared prefix_tokens=71 vocab=129280
     max_abs=0 at token=0 live=3.53901505 fresh=3.53901505 rms=0
```

**Bit-identical.** A fresh full prefill and incremental decode produce exactly
the same logits, so the prefill sweep and the decode path compute the same
thing. The verify is not doing bad arithmetic.

## What is left

Both paths are individually correct and produce identical logits, yet appending
one token through the sweep and then continuing to decode diverges. The only
thing that survives is that **the two paths leave different auxiliary state
behind**, so a session that mixes them is corrupted from that point on.

`--decode-consistency` cannot see this: it compares a fresh prefill of a whole
prefix against decode of that prefix. It never decodes, appends via the sweep,
and then decodes again -- which is precisely what the verify does.

Candidate state, in order:

1. **The ratio-2 carry.** `ds4_gpu_dsv41_pool2` threads
   `previous_kv/previous_score` position to position. Decode calls it at
   ds4.c:39996 with `count=1`; the sweep calls it batched. `n_head_dim` is 512
   for FLASH41 so the literal `512` there is not a mismatch, but whether the
   carry ends in the same place after a sweep append as after a decode append
   is untested.
2. **Engram history.** `ds41_graph_verify_rows` advances a *local* copy of
   `g->history` per row and stores it in `vc->history[]`, then
   `ds41_verify_commit` assigns `g->history = vc->history[keep-1]`. If the
   sweep also advances `g->history` internally, the assignment either
   double-advances or reverts it.
3. **Window ring bookkeeping** for the non-ratio-2 layers, which
   `ds41_verify_ctx` snapshots but which the sweep may index differently from
   decode.

## A note on the `vc`-armed publish

`ds41_attention_publish_batch` changes its forward computation when `vc` is
non-NULL: it pools **row by row** instead of in one batched call, so the
verify's numerics differ from every other caller's by construction. At
`count == 1` the two are equivalent, so this does not explain the one-row
divergence -- but it is a second, independent correctness hazard for
`count > 1` and should not be left standing once the state bug is fixed.

## The cheap fix worth trying first

Rather than making the sweep leave decode-identical state, **replay the
accepted tokens through the decode path after committing**. That is exactly
what the non-V4.1 TP worker already does for `ROLLBACK_REPLAY`. It costs one
decode step per accepted token, which eats much of the speedup, but it would
prove the diagnosis and give a correct baseline to optimise from.

## One avenue that does not work, recorded so it is not retried

`--decode-consistency N` with `--dspark` looks like the ideal probe -- decode
through the speculative path, then compare against a fresh prefill -- but it
does not exercise it. The run emits **no DSpark stats at all** and reports the
same `live=3.53901505 fresh=3.53901505 max_abs=0` as the plain run, byte for
byte: `--decode-consistency` drives its own decode loop, which never enters the
speculative path.

Confirming the state hypothesis therefore needs either a probe placed inside
the speculative loop, or the replay experiment above.

---

# PHASE 6: the verify snapshot is missing two of the four persistent arrays

`ds41_state_spans` is the engine's own authority on what V4.1 graph state
persists -- it is what session save/restore serialises, and its comment is
explicit: "Only owner caches, windows and unfinished compression pairs
persist."

```c
static uint32_t ds41_state_spans(ds41_gpu_graph *g, uint32_t pos, ...) {
    for (uint32_t il = 0; il < 40; il++)
        spans[n++] = {g->window[il],        raw * 512 * 4};
    for (uint32_t i = 0; i < 4; i++) {
        spans[n++] = {g->compressed[i],     live * 512 * 4};
        spans[n++] = {g->index_cache[i],    live * 128 * 4};
        if (i < 3 && (pos & 1u)) {
            spans[n++] = {g->previous_kv[i],    512 * 4};
            spans[n++] = {g->previous_score[i], 512 * 4};
        }
    }
}
```

Against what the verify snapshots and restores:

| persistent state | in `ds41_state_spans` | saved by `ds41_verify_ctx` | restored by `ds41_verify_commit` |
|---|---|---|---|
| `window[il]` x40 | yes | yes (`vc->win`) | yes |
| `previous_kv[i]` / `previous_score[i]` | yes | yes (`vc->prev`) | yes |
| **`compressed[i]` x4** | **yes** | **no** | **no** |
| **`index_cache[i]` x4** | **yes** | **no** | **no** |

**The compressed-KV rings and the index caches are never snapshotted and never
restored.** On any partial accept -- `keep < rows` -- the verify rolls back the
window and the ratio-2 carry, then leaves `compressed[]` and `index_cache[]`
advanced past the accepted prefix, permanently out of step with `g->pos`.

Partial accepts are not an edge case: the runs above recorded `partial=13`
(single box), `partial=7` (TP2, scheduler off) and `partial=2` (TP2, scheduler
on). Every one of those corrupted the session.

This is a definite defect, independently of the one-row divergence. It is worth
stating that the two are separate:

* **Incomplete rollback** (this section) -- explains corruption on every
  partial accept. Precisely identified.
* **Sweep-append divergence** (Phase 5) -- `DS4_DS41_VERIFY_ROWS1` commits in
  full and rolls nothing back, and still diverges. Not explained by this.

Fixing only the first will not produce correct output, which is why it is left
here as a described defect rather than a patch: the fix needs the second bug
resolved alongside it so that greedy byte-identity can actually validate it.
Shipping an unverifiable change into a distributed lockstep path is worse than
naming it precisely.

## A smaller discrepancy in the same area

`ds41_state_spans` carries `previous_kv/previous_score` for **`i < 3`** only,
while `ds41_verify_ctx` allocates and `ds41_verify_commit` restores **4**
owners (layers 2/8/14/20, guarded by `ds4_layer_compress_ratio(il) == 2`). One
of the two is wrong about owner 3. Worth resolving while in this code, though
it is not implicated in either divergence above.

## Where a fix should start

1. Mirror `ds41_verify_ring` for `compressed[i]` and `index_cache[i]`, saving
   only the rows the block touches (`pos0/ratio .. (pos0+rows)/ratio`) rather
   than the whole ring -- the full rings are ~16 MiB per owner at 16K context
   and cannot be copied per cycle.
2. Resolve the `i < 3` versus 4-owner disagreement.
3. Only then chase the one-row divergence, with byte-identity as the test.

---

# PHASE 7: the sweep is fine at every row count; mixing paths is the problem

Two more eliminations, both cheap.

## The one-row sweep is bit-identical to decode

`DS4_METAL_DISABLE_V41_LAYER_PREFILL=1` forces `ds41_prefill_count` to 1, so
the whole prefix is built one row at a time through the same sweep the verify
uses:

```
ds4: decode-consistency compared prefix_tokens=71 ... max_abs=0 rms=0
```

Identical to the chunked run. **The one-row sweep computes exactly what decode
computes.** Combined with Phase 5, the sweep is now exonerated at every row
count tested.

## `ds41_hash_tokens` has no side effects on the graph

`ds41_graph_verify_rows` hashes each row itself, before the sweep, to fill
`vc->history[]` -- and the sweep hashes the same tokens again. That looked like
a double-apply, but `ds4_engram_hash` takes the layout **`const`** and mutates
only the `ds4_engram_history` it is handed, which here is a local copy.
`g->pos` is saved and restored around the call. The loop is pure.

## What that leaves

Every component is individually correct:

| candidate | verdict |
|---|---|
| accept rule | exonerated -- `DS4_DS41_VERIFY_COMMIT1` diverges identically |
| rollback | exonerated -- `DS4_DS41_VERIFY_ROWS1` rolls nothing back, still diverges |
| sweep arithmetic, chunked | exonerated -- `max_abs=0` vs decode |
| sweep arithmetic, one row | exonerated -- `max_abs=0` vs decode |
| `vc` save helpers | exonerated -- pure copy-out |
| row-by-row vs batched pool2 | equivalent at `count == 1` |
| `ds41_hash_tokens` side effects | exonerated -- const layout, local history |
| `n_head_dim` literal `512` | matches FLASH41 |

So the defect is not in any of these in isolation. It is in **appending
through the sweep onto state that decode built** -- the one thing nothing else
in the engine does, and the one thing `--decode-consistency` structurally
cannot test, because it only ever builds a prefix by a single method and
compares the result.

## The next step needs instrumentation, not another flag

Snapshot the four arrays `ds41_state_spans` names -- `window[]`,
`compressed[]`, `index_cache[]`, `previous_kv/score[]` -- after appending one
token two ways from the same starting state:

1. through the decode path, and
2. through `ds41_graph_verify_rows` with `count = 1`,

then diff them. Whichever array differs is the bug, and the diff will say
whether it is a wrong value or a wrong ring index. Everything above has
narrowed the search to those four arrays and one token; this is now a bounded
comparison rather than an investigation.

---

# PHASE 8: ROOT CAUSE -- the two append paths are not bit-reproducible

`DS4_DS41_VERIFY_STATE_DIFF=1` (added here) appends the same token twice from
identical state -- once through `ds41_graph_step` (decode), once through
`ds41_graph_verify_rows` with `count = 1` -- restoring the full state in
between, and diffs the four arrays `ds41_state_spans` names.

```
state-diff token=16 pos0=58 decode_pos=59 verify_pos=59 ok=1
  window[9]  bytes=262144 first_diff_at=118814 (float 29703) decode=0.0703125  verify=0.078125   diff_bytes=86
  window[10] bytes=262144 first_diff_at=118786 (float 29696) decode=-0.21875   verify=-0.234375  diff_bytes=157
  window[11] ...                                             decode=-0.109375  verify=-0.125     diff_bytes=185
  window[14] ...                                             decode=0.0703125  verify=0.0625     diff_bytes=382
  window[16] ...                                             decode=-0.28125   verify=-0.21875   diff_bytes=378
state-diff arrays_differing=35 of 56
```

Three things fall out of this immediately.

**The position is right.** `float 29696 / 512 = slot 58`, and `pos0 = 58`. Both
paths write exactly the row they should, and both leave `pos = 59`. Nothing is
misindexed and nothing is written to the wrong slot.

**The values are close, not wrong.** `0.0703125` vs `0.078125`, `-0.21875` vs
`-0.234375`, `-0.109375` vs `-0.125` -- neighbouring values one quantisation
step apart, not garbage. Only 86-382 bytes of each 2048-byte row differ; most
of the row is bit-identical.

**It accumulates with depth.** Layers 0-8 agree exactly; `window[9]` is the
first to differ, and 35 of the 56 arrays differ in total.

## What this means

The defect is not missing state, a bad index, or a lost rollback. **The decode
path and the batch/verify path do not produce bit-identical keys for the same
token at the same position from the same prior state.** They agree to within a
quantisation step, and the difference compounds through depth.

That is enough to change an argmax, which is why greedy output diverges after
~15 bytes, and why the verify's per-row logits so rarely match the drafter:
they are computed from a KV state that decode would never have produced.

This also dissolves the apparent contradiction with `--decode-consistency`.
That check builds a prefix by *one* method and compares the result, so it never
sees a cross-path append. Both paths are internally self-consistent -- which is
exactly what it measured, bit-identically, and exactly why it could not find
this.

## Consequence for the design

A verify that appends through the batch path can never leave the graph in the
state serial decode would have, no matter how complete the snapshot is. So the
snapshot-and-rollback design cannot be made correct by extending the snapshot;
the Phase 6 gap in `compressed[]`/`index_cache[]` is real and still worth
fixing, but fixing it would not have produced correct output.

**Replay is now the clear design, not the fallback.** After the accept
decision, roll back to `pos0` and re-run the accepted tokens through
`ds41_graph_step`, so committed state is decode-built by construction. It costs
one decode step per accepted token, which caps the speedup at roughly the ratio
of a batched verify to N serial decodes -- much less than the 1.6x Phase 2
projected, but correct. The non-V4.1 TP worker already does exactly this for
`ROLLBACK_REPLAY`, so the pattern and its plumbing exist.

The cheaper alternative worth measuring first: find out *why* the paths round
differently. If it is a single kernel choosing a different accumulation order
or intermediate precision at `count == 1`, making the verify use the decode
kernel for its rows would preserve the full speedup. The diff above localises
the search to whatever writes `window[]` between layers 8 and 9.
