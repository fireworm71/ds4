# T24: how llama.cpp, vLLM and SGLang make speculative decoding pay

Written after measuring why it does not pay in ds4 on the Spark pair (T22, T23).
This corrects one of my own recommendations.

## 0. First, a negative result

`MIN_VERIFY_DRAFTS=1` with Stage 1 gives **16.63 t/s**, worse than 2 (20.10) or
3 (20.04). I predicted it would win, reasoning that Stage 1 made a K=1 verify
cost ~77 ms for 2 tokens. Measured verify averaged **149 ms** per block at a
mean K of 2.06, so the sweep model underestimates at low K. The tuning space is
closed: 1 -> 16.63, 2 -> 20.10, 3 -> 20.04, 4 -> 16.37.

## 1. llama.cpp: one CUDA graph for the whole forward pass

`ggml-cuda.cu` captures the **entire ggml graph** -- every layer -- into a single
CUDA graph and replays it, using `cudaGraphExecUpdate` to patch kernel
parameters between calls instead of re-capturing. One launch per `llama_decode`,
at any batch size. A speculative verify of K+1 tokens is therefore one graph
launch with no per-layer host interaction.

Most relevant to us, `ggml_cuda_graph_check_compability` disables graphs for
exactly one reason:

```c
if (node->op == GGML_OP_MUL_MAT_ID) {
    if (ggml_cuda_mul_mat_id_needs_sync(node, cc)) {
        // the mul_mat_id fallback path synchronizes the stream, so we cannot use CUDA graphs
        use_cuda_graph = false;
```

**The MoE is the one op that can break graph capture, because its fallback
synchronises the stream.** They treated that as a bug to fix in the kernel
rather than a reason to give up graphs. ds4's per-layer
`cudaStreamSynchronize` inside the TP gate is the same class of blocker.

## 2. vLLM: piecewise graphs, and it hits our exact hardware problem

vLLM does not require the whole forward to be capturable. It uses **piecewise
capture**: graph everything except ops that cannot be captured (principally
attention), so launch overhead is removed from the rest.

And vLLM has an open issue that is *specifically about GB10*:

> Cross-node CUDA graph capture fails (illegal memory access at capture_end) --
> **host-staged NCCL all-reduce** captured by `breakable_cudagraph` on **GB10
> (no GPUDirect)**

Same hardware, same root cause as ds4's gate: **without GPUDirect the collective
must stage through host memory**, and a host-staged collective cannot be
captured. Their workaround is to run `tensor_model_parallel_all_reduce`
**eagerly, writing results in place**, so piecewise capture still succeeds
around it.

Others avoid the problem by changing the collective: a custom **all-gather then
local summation** is capturable where standard NCCL `all_reduce` is not.
TensorRT-LLM's "one-model" paradigm goes furthest, capturing the entire
drafting *and* verification loop in a single graph.

## 3. SGLang: the drafter stays at TP=1 -- which kills my Stage 4a

SGLang runs draft and target in the same process and GPU group, one scheduler
driving a tight draft -> verify -> accept loop. Its documented guidance:

> the draft model should keep `--speculative-draft-tensor-parallel-size 1` even
> when the target uses tensor parallelism

**That directly contradicts T23 Stage 4a**, where I proposed running ds4's
drafter on both ranks to split `ffn` and `q_path`. The reference implementations
deliberately do the opposite: a drafter is small enough that a per-layer
collective costs more than the compute it saves. On our pair a gate is ~3.8 ms
and the drafter has 3 layers, so TP-parallelising it would add ~11 ms of gate
cost to save ~15 ms of matmul -- a wash at best, and likely worse once rank skew
is counted. **ds4 suspending TP for the drafter (`g->tp_world = 0`) is correct,
and Stage 4a should be dropped.**

What SGLang does instead:

* **P-EAGLE / parallel drafting** -- generate all K draft tokens in **a single
  forward pass** rather than a sequential chain. ds4 runs 3 sequential stages
  (`prop_chain`, 30 ms); this is the structural alternative.
* **CUDA graph range extended by `K x max_num_seqs`** so the verify shapes are
  captured too.
* **Asynchronous overlap** is on their roadmap explicitly, with metrics named
  for it: hit rate, fallback rate by cause, drafter-ahead, and **verify-GPU
  bubble**, plus a receive daemon, dedicated copy stream and double-buffered GPU
  buffers.

## 4. What this means for ds4, revised

Our measured problem is two numbers: **propose 40 ms** and **verify fixed
~60 ms (post-Stage 1)**, against a 43.5 ms graphed decode. The reference
implementations address both structurally, and none of their answers is
threshold tuning.

**Ranked by expected value for us:**

1. **Overlap propose with the target's decode** (SwiftSpec; SGLang's async
   roadmap). ds4 runs propose and decode strictly serially, so propose is pure
   added latency. Overlapped, it approaches free -- and the whole +EV problem
   dissolves: the threshold `P(success) > propose / saving` collapses toward
   zero. This is the single highest-value structural change and it does not
   require touching kernels.
2. **Piecewise CUDA-graph the verify sweep**, leaving the gate eager -- exactly
   vLLM's GB10 workaround. Removes eager launch cost from the 40 layers of
   compute while keeping the host-staged collective that GB10 forces on us.
3. **Single-pass parallel drafting** instead of a 3-stage chain, following
   P-EAGLE. Cuts `prop_chain` structurally rather than by fractions.
4. **Drop Stage 4a.** Superseded by SGLang's explicit guidance above.

## 5. What we already share with them

* Host-staged collective on GB10 because there is no GPUDirect -- vLLM
  issue #46253 is the same wall.
* MoE as the op that forces a stream sync and blocks graph capture -- llama.cpp
  fixed this in `mul_mat_id`.
* A drafter kept off the tensor-parallel path.

So ds4's architecture is not unusual. What is missing relative to the references
is **graph capture around the eager collective**, and **overlap between drafting
and target execution**.

## Sources

* llama.cpp `ggml/src/ggml-cuda/ggml-cuda.cu` (local checkout, 4d19b2876)
* vLLM issue #46253, cross-node CUDA graph capture on GB10
* vLLM CUDA graph design docs
* SGLang speculative decoding docs; P-EAGLE issue #23171; parallel spec-decode
  roadmap issue #27462
* SwiftSpec, arXiv 2506.11309 (asynchronous speculative decoding)
* Adaptive Verification for MoE Speculative Decoding, arXiv 2605.00342
