# T27: hypothesis -- MoE streaming and speculation are complementary, not antagonistic

Question: single node, Q2, DSpark, MoE streaming. Does it improve perf, or does
`verify()` force an MoE load per spec batch?

**Answer: verify does force an MoE load per batch, but it is the expert
*union*, which grows sublinearly in K, while the cost it displaces is fixed per
forward pass. The trade is strongly favourable -- more favourable than
speculation in resident mode.**

## 1. The fact that decides it (T14)

Q2 TP2 streaming decode, 71.07 ms/token:

| component | ms/token | share |
|---|---|---|
| **`owned filter wait` (per-layer stream sync)** | **41.28** | **58.1%** |
| GPU kernels | 20.75 | 29.2% |
| decode disk fetching | 8.40 | 11.8% |
| D2H + begin_load machinery | 0.64 | 0.9% |

The streaming penalty is **not disk I/O**. It is a per-MoE-layer wait on the
host probe thread (`g_host_probe_ctrl->token_seq != probe_seq`,
ds4_cuda.cu:25292), 1.032 ms x 40 layers.

That cost is **per forward pass**, not per token. A K-row verify is one forward
pass over 40 layers, so it pays 41.28 ms once for up to K+1 tokens:

```
streaming decode, 30 tokens : 30 x 41.28 = 1238 ms of sync
streaming verify, K=30      :      41.28 =   41 ms of sync
```

**Amortised over an accepted 30-block the dominant term falls from 41.28 to
~1.3 ms/token.** That is more than half the streaming decode budget removed.

## 2. Why this flips the resident-mode result

In resident mode block size is neutral: measured 27.50 t/s at K=16 against
26.94 t/s at K=30 -- a wash, because resident verify is compute-bound and has no
per-pass fixed cost worth amortising.

Streaming introduces a 41 ms per-pass fixed cost. **So the optimal K under
streaming is much larger than under resident**, and speculation is worth more
there. The mode inverts the tuning, which is not obvious from the resident
numbers alone.

## 3. The counter-pressure: the expert union

Verify must have every expert any of its K rows selects, per layer. Under
independent uniform routing (top-6 of 384):

```
U(K) = 384 x (1 - (1 - 6/384)^K)
K=1 -> 6      K=6 -> 34      K=30 -> 145      K=61 -> 237
```

At 366 GB / 40 layers / 384 experts ~= **24 MB per expert**, K=30 touches
~3.5 GB per layer, ~139 GB per verify -- more than the 121 GB of node RAM. Taken
at face value a single large verify would sweep the entire expert cache and
leave the following tokens cold.

**But independent uniform routing is the wrong model for the accepted case.**
Speculation now fires on verbatim-copy regions (T25/T26: 8 drafts in 127 cycles,
all in repeated text). Those tokens were processed before, so they route to
largely the same experts, which are already hot. The worst case above does not
describe the regime where blocks are actually accepted.

This is directly measurable with the existing `DS4_CUDA_EXPERT_CACHE_STATS=1`:
compare distinct-expert count and hit rate for a verify batch in a copy region
against the U(K) curve. **That measurement is the go/no-go for this whole
hypothesis** and it is cheap.

## 4. A concrete blocker in the code

`ds4_cuda.cu:25384`:

```c
if (owned_filtered && n_tokens <= 8u && (n_expert == 3u || n_expert == 6u)) {
    /* Keep TP decode and short appends on their Q8_K arithmetic. */
```

The aligned Q2_K MoE fast path is gated at **n_tokens <= 8**. `n_expert == 3` is
the TP2 case (top-6 split across ranks), `n_expert == 6` the single-node case.

**A 30- or 64-row verify falls off this path onto the generic one.** So on Q2
today, exactly the large blocks that would collect the amortisation lose the
specialised kernel. This cliff must be lifted before any of section 1 can be
measured honestly.

## 5. Single node vs the pair

For Q2 (366 GB file, 121 GB per node), streaming is mandatory -- it does not fit
resident on one node or across the pair.

Single node gains:
* all gate and transport cost disappears (~5.4 ms/token of exchange across 80
  gates in decode, plus the sweep's gate overhead)
* no peer wait, no rank skew, no handshake

Single node loses:
* each node must cover the whole 366 GB expert set rather than its half, so
  cache coverage per node roughly halves and per-node disk traffic roughly
  doubles. The 8.4 ms/token disk term is small *because* TP2 halves it.
* the streaming path is built around slice ownership --
  `owned_streaming = owned_filtered && cuda_tp_streaming_enabled() && g_n_gpus == 1`

**Net: TP2 is what makes Q2 streaming tractable. Single node trades ~5 ms/token
of gates for a materially worse hit rate on the disk term.** It will not beat
the pair.

But single node is where speculation matters *most*: the 41 ms/pass fixed cost
is identical, and there is no gate overhead complicating the amortisation. So
single-node Q2 + streaming + lookup drafting is a more interesting configuration
than its raw baseline suggests -- speculation is worth the most exactly where
the baseline is worst.

## 6. The design this suggests

Extend the n-gram cache (T26 TG-1) to record **expert routing** alongside tokens.
On a lookup hit the cache then knows which experts those tokens used last time,
which gives two things nothing else can:

1. **Size K by predicted expert union rather than by rows.** Grow the block
   until the predicted union approaches the cache budget, then stop. This is the
   right control variable under streaming, and it is not row count.
2. **Prefetch.** Hand the host probe thread the expert list before the forward
   pass begins. This attacks the 41.28 ms sync *directly* rather than merely
   amortising it -- the wait exists because the probe has not finished deciding
   what to fetch, and a lookup hit already knows.

Note this contradicts the current AIMD, which expands the window toward 64 on
verbatim streaks with no notion of memory. Under streaming that is the wrong
variable: a verbatim streak is exactly when the union is small, but the AIMD
does not know that, and on a novel-text block at K=64 it would thrash.

## 7. Test plan, cheapest first

1. `DS4_CUDA_EXPERT_CACHE_STATS=1` on a copy-region verify: measure distinct
   experts per layer at K=6/16/30 against U(K). Confirms or kills section 3.
2. Lift the `n_tokens <= 8` gate (section 4) and re-measure.
3. Q2 TP2 streaming, lookup drafting on, K swept 6..48. If section 1 holds, tg
   should rise with K far past where resident mode went flat.
4. Only then consider single node.

---

# CORRECTION to section 5: Q2 IS resident on the pair

Section 5 asserted Q2 "does not fit resident on one node or across the pair,"
reasoning from the 366 GB file against 121 GB/node. **That is wrong, and T14's
own table contradicts it** -- it has a measured "Q2 TP2 Resident" column at
20.93 t/s, which I read before writing the claim.

Antigravity's walkthrough gives the number: **~80.56 GiB resident per rank**,
i.e. ~161 GiB of working set across the pair, not 341 GiB. The file is much
larger than the tensors a given run actually maps (this is the mixed-quant
branch).

Consequences for the single-node question, which get substantially better:

```
working set      ~161 GiB
single node RAM   121 GiB
deficit            ~40 GiB  ->  ~75% of the working set is resident
```

So single-node Q2 does not stream a 245 GB shortfall; it streams roughly the
coldest quarter of the experts. With the usage skew the expert cache already
exploits (decode disk was only 8.40 ms/token even under full streaming), the
resident 75% should capture well over 75% of accesses.

**Single-node Q2 is therefore far more viable than section 5 concluded.** The
streaming x speculation synergy in sections 1-3 is unchanged and still applies
to the streamed remainder -- and it matters more there, because the per-pass
41.28 ms probe sync is paid whenever *any* layer streams, regardless of how
small the streamed fraction is.

The rest of section 5 stands: TP2 still halves per-node traffic and doubles
cache coverage, so the pair remains better. But the gap is 1.3x of working set,
not 3x of file, and single node is a reasonable target rather than a stretch.
