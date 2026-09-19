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
