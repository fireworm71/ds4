# T30: speculative decode on V4.1 -- current state, in one page

T29 is the working record (1500 lines, including six claims I withdrew after
measurement). This is what survived, what to do, and the one thing left to test.

## 1. There is a real, silent correctness bug in shipped speculation

The batched verify is **not batch-size invariant**. A row computed inside an
N-row sweep does not equal the same row computed alone.

```
per-row logits, batched core      75 of 75 rows diverge, up to 3.24 logits
argmax flips                      ~1.3% of verified rows
```

A flip commits a *different token* than serial decode would. Measured cause: the
top-2 gap being smaller than the perturbation (the observed flip had gap 0.084
against a 0.90 perturbation).

**The failure mode is the dangerous one**: not garbled text, but one wrong token
inside otherwise correct output -- `ROUTER_MAX_QUEUES` became `ROUTER_MAX_QUEues`.
Invisible in prose, silent in code, JSON and identifiers.

## 2. The fix exists, is one flag, and is free

```
DS4_DS41_UNIFY_DECODE=1 DS4_DS41_CORE_INVARIANT=1
```

```
per-row logits, CORE_INVARIANT     0 of 75 rows diverge, maxabs 0
decode cost (interleaved A/B)      ~1.8%, inside the noise band
```

Verified at both compressed-cache parities. **Enable it.** It removes the silent
corruption at no measurable decode cost, independently of whether speculation
itself is worth running.

## 3. But speculation does not currently pay in the correct mode

Depth curve, all arms `CORE_INVARIANT`, back-to-back, drift-bracketed:

```
nospec  5.54 |  depth2 5.62 |  depth3 5.22 |  depth4 5.16 |  depth6 5.01
```

Monotonically worse with depth. Depth 2 over nospec is ~1.9%, i.e. noise-floor
break-even.

Structural reason: `CORE_INVARIANT` runs the verify's attention **per row**, so
only MoE and HC stay batched. Speculation's gain comes from batching the verify,
and the correctness fix removes batching from the component that made it cheap.

## 4. Consequence: recent cap-raising work points the wrong way

Block caps were raised 16 -> 32 -> 48 -> 64 and drafters tuned for longer drafts.
Those gains were measured in the **fast/wrong** mode, where the verify stays
batched. In the correct mode depth 2 beats depth 6 by 11%. Any correct
configuration wants *shallow* drafts.

## 5. Validate with the probe, never with text diffs

Text comparison is a **lossy detector**: a perturbation only shows up when it
flips a near-tie, so most of the signal is discarded. A depth-independence test
at 160 tokens passed in *both* core modes, including the known-broken one. This
is why the corruption survived every benchmark in this project.

Use `T29_logit_diff_probe.patch`:

```
DS4_DS41_VERIFY_ROW_LOGITS=5 DS4_DS41_VERIFY_ROW_LOGITS_N=25   # per-row logits vs serial
DS4_CUDA_ATTN_INVARIANCE_CHECK=1                               # in-situ kernel 1-vs-N
DS4_CUDA_ATTN_INVARIANCE_SELF=1                                # determinism control
```

## 6. The one open question

**Does the depth curve slope the same way on the resident TP2 pair?**

Everything above was measured single-box with `--ssd-streaming`, because the
pair was occupied throughout. The structure carries over; the constants do not.
On the pair MoE is a far larger share of layer cost (T23: `ffn` 55% of the
chain) and the per-layer TP gate amortises across rows, so batched-MoE with
per-row attention may pay there even though it does not here.

To run it:

* speculation is gated off under TP as "not yet numerically correct"
  (ds4.c:67710) -- that run must set `DS4_DSPARK_TP_VERIFY=1`;
* re-run the invariance probe **under TP first** -- no kernel measurement in
  T29 was taken under TP;
* compare depths 2/4/6 against nospec, back-to-back in one script, bracketed by
  a repeated reference (cross-invocation timing is invalid on this setup).

If the curve slopes the same way, correct speculation does not pay on this
engine and the honest move is target-only decode until the attention core is
made batch-size invariant. If it slopes the other way, the correct
configuration is `CORE_INVARIANT` with shallow drafts.

## 7. Measurement hygiene, learned the hard way

* **Timings are only comparable back-to-back within one script.** The same
  config measured 9.93 and 5.49 t/s in different invocations; a "45% penalty"
  claim died of this.
* **No flag turns speculation off cleanly** -- `--dspark` drafts on its own,
  `--dspark-strict` also changes decode, `UNIFY_DECODE` and `CORE_INVARIANT`
  change decode. Compare depth-vs-depth, or use the probes.
* `DS4_NGRAM_DYNAMIC=0` on every benchmark, or the persistent n-gram cache makes
  runs depend on what ran before.
* `DS4_V41_DSPARK_ENABLE=1` is required for single-box V4.1 DSpark.

---

## 8. Verify economics, measured -- the binding constraint is acceptance

`DS4_DSPARK_VERIFY_TIMING=1`, 6-row drafts, single box:

```
CORE_INVARIANT   verify_rows(6) = 600 ms   (sweep 580, proj 19, logits_rdma 0.07)
batched core     verify_rows(6) = 550 ms   (sweep 530, proj 19, logits_rdma 0.07)
```

**`CORE_INVARIANT` costs ~9% of verify time**, not the catastrophic penalty
earlier sections assumed. Correctness is cheap in the verify as well as in
decode (~1.8%).

Against a decode step of ~195 ms (5.14 t/s), a 6-row verify costs **3.1
decode-equivalents**. So a 6-row block must accept **more than ~3 tokens on
average** to pay for itself.

Measured on this run:

```
51 cycles, 108 proposed, 12 accepted draft tokens
accept_rate 11.1%, avg_accept 0.235 tokens per verify
```

Each verify costs ~600 ms and saves 0.235 x 195 = **46 ms**. Net loss ~554 ms
per verify. That, not the verify cost structure, is why the depth curve slopes
down: deeper blocks multiply a cost that acceptance never repays.

### Which restates T23's conclusion, now in the correct mode

T23 concluded speculation is "limited by drafter hit rate, not engine overhead".
That holds in the correct mode too, and the numbers are sharper: verify(6) is
*already cheaper* than 6 decodes (600 ms vs 1170 ms). The machinery works. It
loses because it accepts 0.235 tokens per attempt instead of the ~3 it needs.

### Caveat on this particular run

At `-n 80` the model is still writing its reasoning preamble ("We need reproduce
exact passage...") and has not reached the passage it was asked to copy. The
lookup drafter has little to match against, so 0.235 is a preamble-regime
number, not a copy-regime one. Longer generations reach the repetitive region
where T25 measured 30-token accepted blocks.

**The honest statement is therefore conditional**: with `CORE_INVARIANT` on, a
6-row block needs ~3 accepted tokens to break even. Whether a workload clears
that bar is a drafter/workload question, and the copy regime plausibly does
while a reasoning preamble plainly does not.
