# T15: Lever 5 Auto-Promotion -- Recovering Resident Speed for Fitting Shards

Measured 2026-09-18 on dual DGX Spark pair (`spark-0fb3` coordinator 10.99.0.1, `promaxgb10-493d` worker 10.99.0.2) over RoCE RDMA, TP2, branch `q3-mixed-quant-tp` (`62a53be`).

## 1. Background & Motivation
In T14, we demonstrated that the ~23.3 ms/token penalty in TP2 streaming mode was caused by:
1. Synchronous `cudaStreamSynchronize(cuda_decode_stream())` on the host CPU in `routed_moe_launch` (`ds4_cuda.cu:25075`), consuming **1.032 ms per layer (41.28 ms per token across 40 MoE layers)**.
2. Invalidation of CUDA Graphs (`ds41_decode_island`), falling back to eager per-kernel dispatch.

For any model shard that physically fits in available non-movable RAM, streaming is completely unnecessary. Forcing or defaulting to streaming degrades decode throughput from ~21.5 t/s down to ~14 t/s.

## 2. Implementation: Lever 5 Auto-Promotion
In `ds4.c:66615`, when network TP is requested with `--ssd-streaming`:
- The engine computes the rank's exact shard size (`weights_model_map_sharded_spans`).
- Computes static context buffers and compares against the safe non-movable budget (`host / 8 * 7 = 106.42 GiB`).
- If `fixed < budget`, the engine dynamically clears `e->ssd_streaming = false`, outputs an auto-promotion notice, constructs aligned CUDA artifacts, and enables CUDA graph execution.
- If true streaming is explicitly needed for testing, setting `DS4_FORCE_SSD_STREAMING=1` bypasses Lever 5.

## 3. Measured Headline Results

### Test A: Q2 TP2 Resident Bypass (passing `--ssd-streaming`)
* **Detection Log**:
  `ds4: TP expert shard (rank 0, 80.56 GiB) fits memory budget (90.38 GiB planned of 106.48 GiB safe); promoting to resident mode`
* **Prefill**: **412.96 t/s** (16,384 tokens)
* **First Token Latency (TTFT)**: **52.57 ms**
* **Steady Decode Speed**: **21.45 t/s** (46.62 ms/token)
* **Result**: **+52.4% decode throughput increase** over stock streaming (14.07 t/s).

### Test B: Q3splice 7L TP2 at 256K Context Allocation (passing `--ssd-streaming`)
* **Model**: `DeepSeek-V4.1-Flash-Q3splice-7L.gguf` (366 GiB on disk; layers 33–39 Q4_K).
* **Detection Log**:
  `ds4: TP expert shard (rank 0, 93.02 GiB) fits memory budget (102.08 GiB planned of 106.48 GiB safe); promoting to resident mode`
* **Planned Footprint**: **100.08 GiB planned** per rank.
* **Watchdog Min Free Nonmovable RAM**: Spark **6,947 MiB**, Promax **3,337 MiB** (safely above 2,500 MiB limit).
* **Prefill**: **385.57 t/s** (16,384 tokens)
* **First Token Latency (TTFT)**: **61.18 ms**
* **Steady Decode Speed**: **19.84 t/s** (50.40 ms/token)
* **Quality**: NLL 0.343660 (-5.8% NLL improvement vs Q2, winning 69/100 official test cases).
* **Result**: Delivers a **+92.4% speedup** over the Q4 streaming baseline (10.31 t/s) with full 256K context allocation.
