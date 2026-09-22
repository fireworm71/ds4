# T31: the TP2 depth curve -- T30's open question, answered

T30 section 6 left one open question: does the `CORE_INVARIANT` depth curve
slope the same way on the resident TP2 pair as it does single-box? It does.
Measured on the pair; the correct-speculation track is closed on this engine.

## 1. Setup

```
branch        spec-tp2-consolidated (6bd6218)
binary        SHA256 62d55faf53ab... identical on Spark (coordinator) and Promax (worker)
transport     RoCEv2 RDMA, rocep1s0f1, 50/50 expert split
              DS4_TP_GATE_HANDSHAKE_EVERY=61
model         DeepSeek-V4.1-Flash-Q4.gguf (483 GiB), SSD streaming on both ranks
sidecar       DeepSeek-V4.1-Flash-DSpark-Q4v3.gguf (7.90 GiB)
workload      /tmp/code_refactor_chat.txt, 741 in / 150 gen, --temp 0
              all arms back-to-back in one script, bracketed by nospec_A/nospec_B
```

## 2. The TP2 CORE_INVARIANT depth curve

All speculative arms run `DS4_DSPARK_TP_VERIFY=1` with `CORE_INVARIANT=1`
(per-row attention, batched MoE/HC -- the numerically correct verify mode).

| label     | K  | prefill   | generation  | vs nospec | SHA256 vs nospec_A |
|-----------|----|-----------|-------------|-----------|--------------------|
| nospec_A  | 0  | 24.13 t/s | 6.88 t/s    | reference | 5f6109ac           |
| depth2    | 2  | 23.53 t/s | 5.72 t/s    | -16.9%    | diverges (65b3ba02)|
| depth4    | 4  | 23.50 t/s | 5.22 t/s    | -24.1%    | diverges (6254c6a8)|
| depth6    | 6  | 23.26 t/s | 5.90 t/s    | -14.2%    | diverges (4cb15237)|
| depth8    | 8  | 23.50 t/s | 5.61 t/s    | -18.5%    | diverges (abb5a260)|
| depth12   | 12 | 23.48 t/s | 5.19 t/s    | -24.6%    | diverges (9f63a89b)|
| nospec_B  | 0  | 24.28 t/s | 6.61 t/s    | bracket   | exact match (5f6109ac) |

Every draft depth loses to serial decode, -14% to -25%. The single-box
structure carries over to the pair: per-row attention removes the batching
that paid for the verify, and the TP2-specific hope from T30 (MoE a larger
share of layer cost, gate amortising across rows) does not compensate.

Note the depth outputs also diverge from each other and from nospec in the
correct mode -- expected: depth changes which rows commit, and per-row
attention is invariant to batch size, not to draft path. The nospec bracket
matching exactly (5f6109ac both ends) is the drift control.

## 3. Decision (T30's rule, applied)

T30: "If the curve slopes the same way, correct speculation does not pay on
this engine and the honest move is target-only decode until the attention
core is made batch-size invariant."

It slopes the same way. Therefore:

* **Default stays target-only decode.** `DS4_DSPARK_TP_VERIFY` remains gated
  off by default under TP.
* The correct mode (`CORE_INVARIANT`) is kept for probe/validation work, not
  for throughput.
* The only path to speculation that is both fast and correct is making the
  batched attention core batch-size invariant. That is now the single blocker
  on the entire track.

## 4. Fast verify mode on TP2 (batched attention + batched MoE)

Measured for completeness, with the T30 section 5 caveat attached: text
parity is a lossy detector. "Bit-exact" below means no near-tie was
encountered in these windows, not that the mode is safe.

| workload                    | in/gen   | serial tg      | spec tg          | delta          | text parity   |
|-----------------------------|----------|----------------|------------------|----------------|---------------|
| C code refactor             | 741/150  | 7.62-7.72 t/s  | 9.94-10.26 t/s   | +30% to +35%   | 0 diff        |
| JSON extraction             | 312/150  | 5.23-5.35 t/s  | 5.37-5.56 t/s    | +3% to +4%     | 0 diff        |
| novel prose, HYBRID=0       | 100/100  | 8.42-8.71 t/s  | 8.75 t/s         | 0.0%           | 0 diff        |
| novel prose, hybrid fallback| 100/100  | 8.42-8.71 t/s  | 4.89 t/s         | -44%           | diverges @tok65|

## 5. Token 65 is the predicted near-tie flip

`DS4_DSPARK_SPEC_LOG=1` trace of the hybrid-fallback prose run:

1. Tokens 1-64 are identical between serial and speculative decode.
2. At token 65 ("...Sentence 1: A binary search tree stores...") the top-2
   candidates ("keys" vs "each node") sit in an argmax near-tie.
3. The ~1.3e-6 float accumulation variation of batched attention tips the
   continuous logit across the boundary. Both continuations are fluent; they
   diverge from that token on.

This is exactly the failure mode T29/T30 characterised (~1.3% of verified
rows flip on near-ties, invisible in prose, silent in code). It is included
here as the standing reminder that the 0-diff rows in section 4 are workload
luck, not correctness.

## 6. Hybrid fallback disposition

`DS4_DSPARK_NGRAM_HYBRID=0` (pure n-gram, no dspark fallback) skips verify
sweeps when the cache has no match (<250ns), recovering 8.75 t/s with 0%
penalty and text parity on prose. 6bd6218 makes hybrid fallthrough default
to 0 under SSD streaming. That default is validated by this data and stays.

The -44% hybrid-fallback regression is recorded here as the rationale; no
further work is planned on the fallback path while the attention core remains
non-invariant.
