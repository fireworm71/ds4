# T18: Q2 TP2 spec decode fits memory, never drafts, and costs 5.6% -- the support model is incompatible

Measured 2026-09-18, Q2 TP2 **resident** over RoCE RDMA, 16K prompt,
`ctx-alloc 32768`, `--dspark --mtp-model ds41f-dspark-q4.gguf`,
`DS4_V41_DSPARK_ENABLE=1`, one binary, both ranks carrying the draft model
(sha256 `da7cf160...`, shipped to promax and hash-verified).

## 1. Memory: it fits, which was the question

| component | GiB/rank |
|---|---|
| Q2 TP2 resident shard | 80.56 |
| context/buffers @32K, 4K chunk | 5.06 |
| DSpark support model | 7.81 |
| **total** | **93.43** |

Against the proven-good ceiling of 100.72 that leaves **~7.3 GiB spare**, and
TTFT actually improved (52.4 ms against the control's 54.7). Memory is not the
obstacle for Q2.

**No Q3 splice fits with the draft model**, at any context: 7L needs 105.89 at
32K and 107.88 at 256K; even the 6L splice misses at 104.11. With spec decode
the Q4-layer budget collapses to k=4 at 32K, k=3 at 128K, k=2 at 256K, against
k=7 without it. So spec decode and the Q3 quality upgrade are **mutually
exclusive on this pair** -- roughly half to three-quarters of the Q3 quality
gain is the price of the draft model.

**Hazard:** the admit check (`ds4.c:66275`) adds `e->vision_model.size` to the
weights total but has **no term for the DSpark support model**. Its 7.81 GiB is
unbudgeted, and the real ceiling is already 5.8 GiB below what the check
permits (106.48). A Q3-plus-draft configuration would therefore pass admission
and hit the documented silent hang rather than being refused. The owed
admit-check fix should count this model too.

## 2. Speed: a 5.6% regression

| arm | pp | tg steady |
|---|---|---|
| Q2 TP2 resident (control, 8K chunk) | 410.47 | **21.83** |
| + spec decode, draft width 5 | 407.72 | **20.61** |
| delta | -0.7% | **-5.6%** |

## 3. Why: it never proposed a single draft

`DS4_DSPARK_STATS=1`, 256-token run:

```
cycles=247 first_tokens=247 proposed=0 accepted_draft=0 accept_rate=0.00%
avg_accept=0.000 no_draft=247 scheduler_skips=223 tail_skips=8
time_ms propose=548.013 prop_chain=305.078 prop_conf0=116.736
         verify=0.000 saved=0.000 net_saved=-548.013
draft_len_hist=none accepted_len_hist=0:247
```

Every one of 247 cycles produced **no draft**, yet the drafter chain ran for
**548 ms** (2.1 ms per token over the run). `net_saved` is exactly the negative
of the propose time. The measured -5.6% is pure drafter overhead with zero
drafts to pay for it -- verify never executed at all.

## 4. The root cause is the artifact, not the code

From the load log:

```
DSpark tensor mtp.1.ffn_up_exps.weight (ffn_up_exps) has dim[2]=128, expected 384
DSpark tensor mtp.2.ffn_gate_inp.weight (ffn_gate_inp) has dim[1]=128, expected 384
...
DSpark support model detected: ds41f-dspark-q4.gguf
  (stages=3 block=5 markov_rank=256 tensors=81 missing=0 invalid=15 metadata_errors=0)
```

The support model's MoE tensors carry **128 experts where the target has 384**,
so 15 tensors are invalid and the drafter cannot run. Checked all three
variants on disk -- `ds41f-dspark-q4.gguf`, `-q4v2`, `-q4v3` -- and **every one
is 128-expert**. There is no compatible DSpark support model on this machine.

## 5. What this means for Lever 1

Lever 1 is blocked **one level below** where the plan thought. The documented
prerequisites were the drafter ring-capture fix and the state oracle; both are
real, but neither can be worked on until a support model matching the
384-expert target exists. Nothing measured here says anything about whether
speculative decoding helps on this hardware -- the mechanism never engaged.

A discrepancy worth resolving before rebuilding anything: `b0cbc30` reported
`accepted_draft=6, partial=10, accepted_len_hist 0:26,1:4,2:1` over a 48-token
run, so drafts demonstrably *did* happen during that work. Either it ran
against a 128-expert V4.1 target, or against a support model that is no longer
on disk. Finding out which determines whether a new support model must be
built or an existing one recovered.

## 6. Disposition

* **Do not enable `--dspark` on this pair.** It is a 5.6% loss with no
  possibility of gain while the support model mismatches.
* **Lever 1 is artifact-blocked**, not code-blocked. Resolve the support model
  first; the ring-capture fix and state oracle stay owed behind it.
* **Q2 is the only target that could host spec decode** if a valid model
  appears, and adopting it means giving up the Q3 quality upgrade.
* The engine behaved correctly throughout: it detected the mismatch, reported
  `invalid=15`, declined to draft, and produced correct output. The only fault
  is that it charges 2.1 ms/token for a drafter it has already determined
  cannot propose -- worth an early bail-out so a mismatched model costs
  nothing rather than 5.6%.
