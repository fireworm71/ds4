# T23: plan -- cut the speculative cycle below break-even

## The number to beat

At default confidence with `MIN_VERIFY_DRAFTS=2`, a speculative cycle yields
**~3.25 tokens** and costs:

```
propose   ~40 ms      (prop_chain ~30 ms)
verify   ~134 ms      (sweep = 76.3 fixed + 16.8/row)
total    ~174 ms      =  53 ms/token
baseline               43.5 ms/token
```

Break-even is `3.25 x 43.5 = 141 ms`. **We need ~35 ms. A win needs ~76 ms.**

## Where the sweep's time actually goes

`DS4_TP_BIG_GATE_DEBUG=2`, 560 gates sampled on the pair:

| component | per gate | x40 layers | share |
|---|---|---|---|
| since-last-release (layer GPU compute + `cudaStreamSynchronize`) | 3.005 ms | 120 ms | 79% |
| **handshake** (TCP header round trip + 2 `setsockopt`) | **0.426 ms** | **17.0 ms** | 11% |
| **transfer** (RDMA exchange, ~102 KB each way) | **0.388 ms** | **15.5 ms** | 10% |

Sum 3.82 ms/gate x 40 = 153 ms, which matches the measured sweep.

**The addressable overhead is handshake + transfer = 32.5 ms per sweep** -- almost
exactly the 35 ms break-even gap. The remaining 79% is genuine layer compute
that scales with rows, and is not overhead to remove.

## Stage 1 -- delete the per-layer TCP handshake (17 ms/sweep)

`ds4_tp_big_gate_exchange` does this *before every layer's* RDMA:

```c
tp_socket_set_gate_timeout(tp->data_fd, tp->gate_timeout_ms + 2000u);   /* setsockopt */
tp_write_full(fd, &h, sizeof h); tp_read_full(fd, &ph, sizeof ph);       /* TCP round trip */
tp_socket_set_gate_timeout(tp->data_fd, tp->gate_timeout_ms);            /* setsockopt */
```

It exists to detect desync (`magic/layer/gate/seq`) and to widen the socket
timeout for a cold peer. Both can be kept without a round trip:

1. **Carry the header in the RDMA payload.** Prepend the 16-byte
   `ds4_tp_gate_header` to the staged buffer and validate it on arrival. Desync
   detection is preserved exactly; the TCP round trip disappears.
2. **Hoist the timeout.** Set the widened timeout once when the sweep begins and
   restore it once at the end, rather than twice per layer.

Expected: **-17 ms per sweep**, no protocol semantics lost.

Risk: the handshake currently also *synchronises* the ranks before the RDMA
window. `tp_rdma_drain_decode_window` already runs after it; whether the
handshake is load-bearing for window ordering must be confirmed by reading
`tp_rdma_big_gate_exchange`, not assumed. If it is, fall back to option (1)
only, which keeps a rendezvous inside the RDMA path.

## Stage 2 -- stop copying through host staging (up to 15 ms/sweep)

`ds4_gpu_tp_big_gate_encode` currently does:

```c
cudaStreamSynchronize(...);
ds4_gpu_tensor_read (out_t, staging);      /* D2H 102 KB */
big(..., staging, peer, bytes);            /* RDMA        */
ds4_gpu_tensor_write(in_t,  peer);         /* H2D 102 KB */
```

GB10 has **unified coherent memory**: host and device are the same physical
RAM, so both copies are memcpys that exist only to reach a
`cudaHostAlloc` buffer the RDMA is registered against. Registering the tensor
memory directly and exchanging in place removes them.

Expected: a large part of the 15.5 ms transfer term. Lower confidence than
Stage 1 -- the 0.388 ms may be dominated by RoCE latency rather than the
copies, in which case this stage yields little. **Stage 1 first.**

## Stage 3 -- only if Stages 1-2 are not enough: the sync

`cudaStreamSynchronize(cuda_decode_stream())` drains the whole stream. A
`cudaEvent` recorded straight after the MoE partial would wait only on what the
gate needs. The measurement says this is mostly real compute, so expect little
-- this stage is a fallback, not a plan.

## Stage 4 -- propose (~40 ms, prop_chain ~30 ms)

The drafter's 3 stages run with TP suspended (`g->tp_world = 0`) and
un-graphed. 3 layers costing 30 ms against the target's 40 layers costing
~153 ms at the same row count is disproportionate; there is no gate overhead
there, so it is eager launch plus a 128-expert MoE.

Two sub-items, cheapest first:
* `prop_logits` 3.3 ms and `prop_markov` 2.7 ms per propose are full
  256,960-entry vocabulary projections. The drafter only needs the top few
  candidates; a top-k or truncated projection should cut most of this.
* Investigate whether the drafter's stage chain can reuse the decode island's
  CUDA-graph capture. It is single-token-shaped work with stable pointers,
  which is what the island requires.

## Verification protocol -- after every stage

This area has produced four wrong conclusions already; none of these are
optional.

1. `DS4_DS41_VERIFY_POSLOG=1` on **both** ranks -- every commit must report
   identical `pos0/keep/rows/pos` and KV fingerprint.
2. Greedy output vs the target-only baseline, same prompt, `--temp 0`.
3. Re-fit the sweep: `DS4_DSPARK_VERIFY_TIMING=1`, several blocks, report
   `fixed + per-row`.
4. tg against **23.14 t/s**, 128 tokens.
5. `DS4_TP_BIG_GATE_DEBUG=2` to confirm the term actually moved.

## Go / no-go

* After Stage 1: sweep fixed cost should fall ~17 ms. If it does not, the
  handshake was not on the critical path and the model of the system is wrong
  -- stop and re-measure rather than continuing to Stage 2.
* After Stages 1+2: cycle cost should be **~141 ms or below**, i.e. at least
  break-even. If it is still above 150 ms, speculation cannot pay on this
  engine and the honest move is to close it out and keep target-only at 23.14
  t/s.
* Stage 4 is what turns break-even into a win: propose 40 -> ~25 ms puts the
  cycle near 119 ms, about **27 t/s**.

## What is explicitly not worth doing

* **Confidence tuning.** Swept: 0.7 is optimal, the drafter is well calibrated,
  and every relaxation buys blocks that are mostly rejected.
* **min-verify tuning.** 2 and 3 measure the same (20.0 t/s); 4 is worse.
* **Scheduler feedback (plan Components 2/3).** `scheduler_skips=86` of 110 --
  the scheduler already suppresses drafting on cycles that would waste propose.

---

# STAGE 1 RESULT and the revised critical path

## Stage 1 delivered, and is measured

`DS4_TP_GATE_HANDSHAKE_EVERY=N` (commit `8bf79a1`) performs the big-gate TCP
rendezvous only where `layer % N == 0`. At N=40:

```
rows=4   143.7 -> 128.8 ms
rows=5   159.9 -> 143.1 ms
rows=6   177.4 -> 160.8 ms
```

**-16 ms at every row count**, as predicted, with zero desync reports and zero
RNR stalls. tg at confidence 0.3: 14.78 -> 17.78 t/s.

At default confidence tg is unchanged (20.01 -> 20.10): only four blocks reach
verify in 128 tokens, so 16 ms/sweep is ~1% of runtime. **The saving scales
with how often verify runs.**

## Which moves the constraint off verify entirely

With Stage 1, `verify(K=3) ~= 111 ms` for 4 tokens -- **27.8 ms/token against a
43.5 ms baseline**. Speculation is now strongly profitable *per block*. What
stops it is the decision to propose at all:

```
propose costs             40 ms
a successful block saves  ~63 ms
so propose is +EV only if P(success) > 63%
measured P(success)       ~17%   (4 usable blocks in ~24 attempts)
```

`scheduler_skips=86 of 110` is therefore **correct behaviour**, not a bug to
fix. Plan Components 2/3 would have made this worse.

## Why propose costs 40 ms

`DS4_DSPARK_STAGE_PROFILE=1`, 9 stage executions:

| part | share of chain |
|---|---|
| **ffn** (128-expert MoE) | 55% |
| **q_path** (Q projection) | 35% |
| attn_output_hc | 10% |
| attention, norms, next_input | <1% |

Both dominant parts are pure matmul width -- and the drafter runs them
**un-parallelised**. `metal_graph_eval_dspark_stage_chain` sets `g->tp_world = 0`
(ds4.c:34767) because:

> The support model runs only on the coordinator. Its generic layer helpers
> share the base graph object, so temporarily disarm TP or they would encode
> expert gates that the worker can never reach.

So rank 0 does 100% of the drafter's work while rank 1 idles. The target splits
both the Q projection (`q_dim = N_HEAD / tp_world * N_HEAD_DIM`) and the experts
across ranks; the drafter splits neither.

## Revised plan

**Stage 4a -- run the drafter on both ranks (largest lever).** The worker
already loads the DSpark model (`--mtp-model` is passed to it). Driving its
stage chain from a TP frame, exactly as `DS4_TP_FRAME_VERIFY` drives the
worker's verify, would let `tp_world` stay 2 through the chain and split `ffn`
and `q_path`. Expect chain 30 -> ~15-18 ms, propose 40 -> ~22 ms.

**Stage 4b -- top-k the vocabulary projections (cheap).** `prop_logits` 3.3 ms
and `prop_markov` 2.7 ms are full 256,960-entry projections; the drafter needs
only the top few candidates. Expect -5 ms.

**Not available: an early confidence short-circuit.** Plan Component 3 assumed
stage-0 confidence could gate stages 1..4. It cannot: `confidence0` is computed
from `batch_ffn_norm`, which is the *output* of the full chain, so the 30 ms is
spent before the 0.06 ms check can run. That is also why `fuse_final_hidden`
exists.

## Honest arithmetic on where this lands

With Stages 1 + 2 + 4a + 4b:

```
propose ~17 ms,  verify(K=3) ~96 ms
save per success = 4 x 43.5 - 96 = 78 ms
+EV threshold    = 17 / 78 = 22%
measured P(success)            17%
```

**The full programme lands just short of the threshold**, close enough that it
turns on acceptance rate and on how much Stage 2 actually yields. This is worth
saying plainly before more effort goes in: the remaining work is real
engineering (a worker-side drafter driven over the TP protocol), and the
expected outcome is *parity to modest gain*, not the 28-34 t/s the original
plan targeted.

The case for doing it anyway is that Stage 4a also halves drafter latency for
single-box use, where there is no verify cost to amortise against.

---

# STAGE 1, REVISED: the intended mechanism already exists and is unreachable

Stage 1 (`DS4_TP_GATE_HANDSHAKE_EVERY`) removes the per-layer TCP rendezvous by
skipping it. That works (-16 ms, no RNR observed) but it is a workaround, and
the codebase already contains the principled version.

## `ds4_tp_batch_block_begin` / `_end`

`ds4_tp.c:2197`, with a comment that describes precisely the optimisation Stage 1
gropes at:

> Verify-block window: called on both ranks right before a speculative verify
> block whose every layer ends in one batch gate of `rows` rows. Drains the
> decode receive window, posts the first layer's receives, crosses **one
> control-byte barrier** (so both sides are posted before any send), and from
> then on **each gate posts the next layer's receives and sends its rows without
> any handshake.**

It is strictly better than Stage 1: one barrier per block instead of none, so
both ranks are provably posted before any send (no reliance on RNR retry), and
each gate pre-posts the *next* layer's receives, so the exchange is pipelined
rather than merely handshake-free.

## Why it never runs on our pair

Two independent reasons, both structural:

1. **The call site is Metal-only.** `ds4.c:37868` wraps it in
   `#if defined(__APPLE__)`, so on CUDA it is not even compiled.
2. **It is attached to the wrong verify path.** The call sits in
   `metal_graph_verify_suffix_tops_impl` -- the pre-V4.1 verify. V4.1 uses
   `ds41_graph_verify_rows`, which never touches it.

## And V4.1 uses a different gate function

There are two batch gate paths in the transport:

| function | buffers | rendezvous |
|---|---|---|
| `ds4_tp_batch_gate_exchange` | fixed slab offsets per layer | **windowed** (`block_active`) |
| `ds4_tp_big_gate_exchange` | arbitrary pointers | **per-gate TCP handshake** |

V4.1's `ds41_sum_partial_batch` calls `ds4_gpu_tp_big_gate_encode`, i.e. the
**big** path, because its buffers (`x` and `g->batch.q`) are not in the
registered slab. So even with the Apple guard removed, the window would not
apply -- `block_active` is only honoured by the batch path.

## The real Stage 1

Extend the window to the big-gate path: when `r->block_active`, have
`ds4_tp_big_gate_exchange` skip the handshake and consume pre-posted receives,
exactly as `ds4_tp_batch_gate_exchange` does. Then call
`ds4_tp_batch_block_begin/_end` around `ds41_graph_verify_rows` on all
platforms, not just Apple.

This is RDMA-layer work -- receive posting, completion accounting, and the
`recv_window_active` / `block_active` interaction in
`tp_rdma_big_gate_exchange`, which currently *refuses* to run while a receive
window is active. It should land the same ~16 ms as Stage 1 with a proper
barrier, and it removes a `getenv`-gated shortcut from the hot path.

Until then `DS4_TP_GATE_HANDSHAKE_EVERY` stands as an interim measure, off by
default.

## One thing Stage 1 does not affect, correctly

Plain decode is untouched. `ds4_tp_gate_exchange` (the row gate, used by
single-token decode) goes straight to `tp_rdma_gate_exchange` under RDMA and
**never does a TCP handshake**. Only the big gate did, so only the verify path
ever paid it -- which matches the measurement that decode tg was unchanged by
Stage 1.
