# T19: the DSpark support models were never incompatible -- and the drafter still proposes nothing

Measured 2026-09-18/19 on the Spark pair. Supersedes T18's conclusion that
Lever 1 was artifact-blocked.

## 1. T18 was wrong: the artifacts are correct

T18 reported the support model as incompatible -- `invalid=15`, MoE tensors
carrying 128 experts "where the target has 384" -- and concluded no valid
DSpark model existed on the machine. That conclusion was wrong, and the
checkpoint config says so plainly (`dsv41-flash/config.json`, `text_config`):

```
n_routed_experts           = 384     <- main model
dspark_n_routed_experts    = 128     <- DSpark stages, by design
num_experts_per_tok        = 6
dspark_num_experts_per_tok = 3
num_nextn_predict_layers   = 3       <- matches the reported stages=3
```

**128 is correct for the drafter.** The engine's own struct comments already
said so: `uint32_t n_experts; /* drafter routed-expert count; V4.1
DSpark=128 */`.

## 2. The real bug: a metadata gap plus a wrong fallback

`model_dspark_summary` looks for a drafter expert count under
`deepseek4.dspark.n_routed_experts`, `deepseek4.dspark_n_routed_experts` or
`dspark.n_routed_experts`. The GGUF carries none of them -- dumping its header
shows only `dspark_block_size`, `dspark_markov_rank`, `dspark_noise_token_id`
and `dspark_target_layer_ids`. The back-compat fallback then assumed the
backbone's count:

```c
} else {
    /* Back-compat: the vision-exp DSpark checkpoint shares the backbone's
     * expert count and stamps no drafter-specific key. */
    s.n_experts = DS4_N_EXPERT;      /* 384 */
}
```

So every routed drafter tensor was validated against 384, all 15 failed, and
drafting was silently disabled while the propose chain still ran -- 2.1 ms per
token for zero proposals.

**Fix:** infer the count from the tensor that carries it. A routed drafter
tensor's last dimension *is* the drafter's expert count, so the scan that
already walks MTP tensors records it, and the summary uses it when no metadata
key is present. Metadata still wins where a writer stamps it. Top-k follows the
same path (`128 != 384` -> 3, matching `dspark_num_experts_per_tok`).

**Verified:** `invalid=15` -> **`invalid=0`**, on both `ds41f-dspark-q4` and
`ds41f-dspark-q4v3`. No artifact rewrite needed; all three variants on disk are
sound.

This also dissolves the `b0cbc30` discrepancy noted in T18. That work reported
real drafts (`accepted_draft=6`, `accepted_len_hist 0:26,1:4,2:1`) because it
predates this validation path, not because it used a different model or target.
Nothing needs rebuilding or recovering.

## 3. Drafting is still silent, and the blocker is now precise

With `invalid=0` and drafting enabled, Q2 TP2 resident, 16K prompt:

```
cycles=247 proposed=0 accepted_draft=0 accept_rate=0.00%
no_draft=247 no_room=0 invalid=0 errors=0 verifier_unavailable=0
scheduler_skips=223 tail_skips=8
propose=478.358 ms  prop_chain=230.750  prop_conf0=122.723
net_saved=-478.358  accepted_len_hist=0:247
```

`DS4_DSPARK_SPEC_LOG` gives the reason:

```
DSpark scheduler no-draft pause ... confidence0=n/a:0.000   <- first cycle
DSpark scheduler no-draft pause ... confidence0=nan         <- every cycle after
```

**The drafter's confidence head yields no usable value** -- `nan` under TP2
after the first cycle, and `n/a` (never computed) in a single-box run. Every
draft is therefore rejected: a NaN fails every threshold comparison.

### Ruled out by measurement, not assumption

| candidate | verdict |
|---|---|
| incompatible artifact | **real bug, fixed** (section 2) |
| confidence threshold too high | ruled out -- `--dspark-confidence 0.15` behaves exactly as the 0.7 default, which is what a NaN predicts |
| ROCm bypass latch | ruled out -- `dspark_sched_bypass` requires `ds4_session_dspark_rocm_gfx1151_fast_path`, unreachable on CUDA |
| scheduler backoff | symptom, not cause -- `many_no_draft` (`no_draft*2 >= cycles`) re-arms the skip countdown *because* nothing drafts |
| checkpoint-specific | ruled out -- q4 and q4v3 behave the same |

## 4. Cost of the current state

`--dspark` on Q2 TP2 resident: **20.61 tg against a 21.83 control, -5.6%**, for
a drafter that proposes nothing. Note this is *worse* than before the
expert-count fix in one narrow sense: the incompatibility bail-out added
alongside it no longer fires, because the model is now correctly recognised as
compatible. That bail-out remains right for genuinely mismatched models.

**Do not enable `--dspark` on this pair** until the confidence NaN is resolved.

## 5. What is owed next

The blocker is now one specific question: **why does the drafter's confidence
head produce NaN (TP2) or nothing (single box)?** That is a numerical fault in
the drafter path, not a configuration or artifact problem, and it is a fresh
piece of work rather than a continuation.

Two leads worth taking first, both cheap:

* The expert-count gap proves drafter configuration can silently default wrong.
  Audit the other drafter parameters the GGUF does not stamp -- FF size, head
  dims, markov rank wiring -- against `text_config`'s `dspark_*` keys the same
  way.
* `b0cbc30` documents a *separate* known defect: the drafter's context ring is
  fed by the single-token decode capture, which a batch commit never runs, so
  the drafter conditions on a stale ring. A stale or uninitialised ring is a
  plausible NaN source, and its named fix (drive the capture from the verify
  batch) was never implemented.

Reproduction is one command with `DS4_DSPARK_SPEC_LOG=1 DS4_DSPARK_STATS=1`;
the confidence value in the scheduler pause line is the signal to watch.
