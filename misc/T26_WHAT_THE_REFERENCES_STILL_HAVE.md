# T26: tg and pp proposals, against vLLM / llama.cpp / SGLang

T23 closed every engine-side lever and concluded: "Further gains need a better
drafter -- more usable blocks per propose -- which is a model question, not a
kernel or transport one." T25 acted on that with prompt-lookup drafting.

This note asks what the reference implementations still have that we do not,
after the lookup drafter. It starts from a measurement that sharpens T23's
conclusion considerably.

## 0. The measurement that sets the agenda

30-token lookup drafts, copy-heavy prompt, 26.94 t/s:

```
cycles=127   drafts=8   accepted_draft=131   full=3   partial=5
draft_len_hist=30:8     accepted_len_hist=...,30:3
```

**8 attempts in 127 cycles (6.3%) produced 131 of ~260 tokens.** Half the output
came from eight blocks. Mean yield per attempt is 16.4 tokens.

So the drafter is not inaccurate -- it is silent. The economics per attempt are
already excellent:

```
verify(K) = ~60 fixed + 16.8/row        (post Stage 1, 8bf79a1)
cost per token if fully accepted = (60 + 17K)/(K+1)
  K=6  -> 26.0 ms      K=30 -> 18.4 ms      baseline decode 44.0 ms
```

A fully accepted 30-block runs at 18.4 ms/token, **2.4x baseline**. We reach
only 26.94 t/s because 94% of cycles never attempt.

**The lever is attempt rate.** Every proposal below is ranked by how much it
raises the fraction of cycles that produce a usable block.

## 1. TG-1: llama.cpp's three-tier n-gram cache (highest value)

`common/ngram-cache.{h,cpp}` plus `examples/lookup`. llama.cpp keeps **three**
caches and consults them in priority order:

| cache | source | lifetime |
|---|---|---|
| context | the current generation | this request |
| dynamic | accumulated across runs, persisted to disk | permanent |
| static  | precomputed from a corpus (`llama-lookup-create`) | permanent |

Each is a hash map from an n-gram (they use n = 1..4) to a *token -> count*
table, so a lookup returns ranked candidates with frequencies, not one match.

**Ours is the context tier only, and as a linear backward scan.** That is why
attempt rate is 6.3%: a draft requires a >=3-deep literal match inside the
current context, which essentially only happens when the model is copying text
it has already emitted.

Proposal:

* Replace the backward scan with a hash map keyed on 2/3/4-grams -> candidate
  token + count. O(1) lookup instead of O(n) scan, and it returns *ranked
  alternatives*, which TG-2 needs.
* Add a **dynamic** tier persisted to disk, accumulated from this model's own
  past outputs. Its idiolect (phrasing, formatting, reasoning preambles) is
  highly repetitive across runs -- exactly the regime where the current context
  tier has nothing yet.
* Optionally a **static** tier built from a corpus.

Expected: attempt rate from 6.3% toward llama.cpp-lookup territory. Even a
modest rise is large here, because yield per attempt is 16.4 tokens. Going from
6.3% to 20% of cycles at the same yield roughly triples the drafted share of
output. Pure CPU/host work, no GPU or transport change, and it composes with
everything already committed.

Risk: none to correctness -- a wrong draft is rejected by verify. The cost of a
bad attempt is one sweep, which antigravity's miss-streak backoff already damps.

## 2. TG-2: tree drafting -- spend rows on breadth, not length

vLLM (`tree_attn`, EAGLE-2), SGLang, and SpecInfer verify a **tree** of
candidates in one forward pass using a custom attention mask, rather than one
linear chain.

This fits our cost curve unusually well. Our verify is **60 ms fixed + 17 ms per
row**. Fixed cost dominates at small K, so additional rows are cheap relative to
starting a sweep at all. A linear K-draft spends all K rows on a single
hypothesis that fails entirely if token 1 is wrong. A tree spends the same rows
across several hypotheses.

Concretely, with the ranked candidates TG-1 produces: branch at the first two
positions (top-2 each), then continue the best path linearly. A 4-wide, depth-8
tree is ~20 rows -- about 400 ms -- and covers four independent continuations
where today 20 rows cover one.

It also solves a structural wart: today a lookup hit *excludes* DSpark, and a
lookup miss declines the cycle. As a tree, the lookup continuation and the
DSpark chain are simply two branches verified in the same sweep, for barely more
than the cost of verifying either alone.

Cost: this is the one item here that touches `ds41_graph_verify_rows`. Rows stop
being a linear sequence, so each row needs a parent link for the causal mask and
position id, and commit must select the best accepted *path* rather than a
prefix. The row machinery already handles arbitrary row counts and a `keep`
prefix commit, so the change is the mask and the commit selection, not the
transport.

## 3. TG-3: SGLang's DP attention -- one gate per layer instead of two

SGLang runs **data-parallel attention** with tensor-parallel MoE: attention is
replicated per rank (each keeps full KV), only the MoE is split. With MLA the
KV is a compressed latent, so replication is cheap in memory.

For us the interesting consequence is not raw speed but **gate count**. V4.1
issues two `ds41_sum_partial_batch` gates per layer -- attention output and MoE
output. T23 ruled out *merging* them (true data dependency) and therefore
treated 80 gates/sweep as fixed. DP attention does not merge them; it
**eliminates** the attention one, because a replicated attention output needs no
reduction.

That matters for three reasons:

1. `ds4_gpu_tp_big_gate_encode` opens with `cudaStreamSynchronize`. Halving
   gates halves the per-layer pipeline drains, and T23's own breakdown puts
   73-79% of sweep time in "release" = compute + drain. Some of that is
   drain-induced bubble, not compute.
2. **It unblocks the verify window.** T23's remaining transport item needed
   "one batch gate per layer"; the window was abandoned precisely because V4.1
   issues two and the slab is layer-indexed. DP attention makes the existing
   `ds4_tp_batch_block_begin/_end` applicable as written.
3. It removes 40 exchanges/token from decode (~2.7 ms at the measured
   0.067 ms/exchange).

Honest accounting: replicating attention duplicates work that is currently
split, adding roughly what the removed exchanges save. **Treat DP attention as a
correctness/structure win that unblocks the window, not as a speed win on its
own.** Its value is that it converts a closed item back into an open one.

## 4. PP-1: sweep the prefill chunk size at long context (cheapest)

Prefill batches through `ds41_prefill_count`. Per-gate handshake is fixed cost
amortised over rows, and transfer scales with rows (20480 B/row), so pp should
improve with chunk size until the RDMA transfer saturates. That knee has never
been measured at 16k.

This is a pure env/parameter sweep with no code risk and it directly addresses
"pp should scale up at 16k". Do it before any pp engineering.

## 5. PP-2: overlap the reduction with compute by splitting rows

vLLM and SGLang both overlap collectives with compute. T23 correctly ruled this
out *for decode* -- at one row there is nothing to split, and the pair is 77%
busy on its own GPU work, so there is no bubble.

**At prefill scale the argument inverts.** With a 256-row chunk each gate moves
5.2 MB and takes real milliseconds, while compute per layer is large. Split the
chunk in half: reduce half A while half B computes, then swap. The transport
already supports arbitrary row counts; this is scheduling, not protocol.

The same trick applies to verify sweeps now that drafts are 30+ rows
(614 KB/gate), which is a tg win as well as a pp one.

## 6. PP-3: a prefix cache pool (SGLang RadixAttention)

`ds4_session_common_prefix` matches a prompt against **the one live checkpoint**,
linearly. SGLang keeps a radix tree of many cached prefixes with LRU eviction,
so branching conversations, retries, and parallel requests sharing a system
prompt all skip prefill.

ds4 already has the hard part: KV checkpoint save/restore. What is missing is a
small pool keyed by prefix hash instead of a single live checkpoint. For agent
loops -- which re-send a growing conversation and frequently branch or retry --
this removes prefill entirely on a hit, which beats any pp t/s improvement.

## 7. Ranking

| # | item | attacks | risk | expected |
|---|---|---|---|---|
| 1 | n-gram cache, 3 tiers + hash map | attempt rate (6.3%) | low | large, and it is the measured limiter |
| 2 | tree drafting | attempt yield, unifies lookup+DSpark | high | large, structural |
| 3 | prefill chunk sweep at 16k | pp scaling | none | unknown, cheap to find out |
| 4 | row-split comm/compute overlap | pp, and 30-row sweeps | medium | moderate |
| 5 | prefix cache pool | repeat prefill | medium | removes pp entirely on a hit |
| 6 | DP attention | gate count, unblocks window | high | structural, not speed |

Start with 1 and 3: one is the measured limiter, the other is free.

## Explicitly still closed (do not revisit -- see T23/T24)

confidence and min-verify tuning; gate handshake beyond Stage 1; merging the two
per-layer gates; overlapping propose with decode; drafter tensor parallelism;
piecewise CUDA graphs around the sweep.

---

# ADDENDUM: state at commit 6c42fff

Written after this note; it changes two of the premises above.

`6c42fff` landed **T23 Stage 2** -- direct RDMA slab staging in
`ds4_gpu_tp_big_gate_encode` via `ds4_gpu_tp_set_big_staging`, removing the host
bounce buffers. Plus 64-row verification, an AIMD streak manager, and a
depth*2 + min(avail, max_draft) candidate score. Reported:

```
copy-heavy decode   29.75 t/s   (22.45 baseline, +32.5%)
short prefill      122.40 t/s   (41.70 baseline)
prose              21.86 t/s    (308 ms total verify -- protected)
one 61-draft block accepted whole
```

Consequences for the ranking:

* **PP-2 (row-split comm/compute overlap) is now worth more, not less.** Stage 2
  removed the memcpy component of the transfer term, so what remains is RoCE
  latency and bandwidth -- which is exactly what overlapping hides. The two
  compose.
* **TG-1 (n-gram cache) is unchanged and still the top item.** Everything in
  6c42fff improves what happens *once a draft exists*: longer windows, better
  candidate ranking, streak adaptation. None of it raises the 6.3% attempt rate,
  because the context tier still requires a literal match in the current
  context. The 61-token block is the same phenomenon as the 30-token blocks --
  enormous yield on the rare cycle that fires.
* Prose at 21.86 t/s with 308 ms of verify confirms the backoff makes misses
  nearly free. That is the precondition for TG-1: raising attempt rate is only
  safe because a wrong attempt now costs little.

The gap between 29.75 (copy) and 21.86 (prose) is the attempt-rate gap, and it
is the whole remaining prize.
