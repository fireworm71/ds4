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
