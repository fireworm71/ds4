# T9: the Q3 8L splice is measurably better than Q2, and it costs 7% of decode

Measured 2026-09-18 on the Spark pair (`spark-0fb3` coordinator 10.99.0.1,
`promaxgb10-493d` worker 10.99.0.2) over RoCE RDMA, TP2 **resident**, branch
`q3-mixed-quant-tp` (`d0d459e` + `67bcb60`), both ranks running byte-identical
`ds4`/`ds4-bench` built at `sm_121a` (9/9 sections, verified after every build).

This answers the question the whole Q3 line rested on and never had: **is the
splice actually better than Q2?** Yes, and by more than a rounding error.

## The artifact

`DeepSeek-V4.1-Flash-Q3splice-8L.gguf`, built here with
`splice_mixed_expert_layers_gguf.py --q4-layers 32-39`. 369.07 GiB,
`sha256 b3a577e626047f5c3866c4b2bb076c8debaf1f8438ccf0e216e93c0353415acb`,
identical on both boxes. Verified per-layer against both sources by declared
type, byte length and sampled payload: layers 0-31 routed experts byte-identical
to Q2, layers 32-39 to Q4, no layer internally mixed, all 926 non-expert tensors
and the KV blob identical to the Q2 base.

## Quality: 100-case official-continuation scoring

`gguf-tools/quality-testing/score_official` against
`deepseek-v4.1-flash-20260910-general/manifest.tsv`, ctx 4096, both models
scored TP2-resident over RDMA with the same binary. Metric is target-token
negative log likelihood against official continuations -- it judges the whole
distribution, not one sampled answer.

| metric | Q2 | **Q3 8L** | change |
|---|---|---|---|
| **avg NLL** (lower better) | 0.364937 | **0.343799** | **-5.79%** |
| cases won | 31 | **69** | 69/100, zero ties |
| first-token matches | 80 | 81 | +1 |
| top-1 rate vs API | 0.905478 | **0.917168** | +1.17 pp |
| top-n recall vs API | 0.780035 | **0.820698** | +4.07 pp |
| pairwise agreement | 0.818495 | **0.842590** | +2.41 pp |
| target-token MAE | 0.226987 | **0.202049** | -11.0% |
| top-logprob MAE | 2.080949 | **1.584871** | -23.8% |

Every aggregate moves the same way, and the win is broad rather than a few
outliers: 69 of 100 cases improve, with the largest single-case gains
(`delta_nll` -6.31, -6.31, -4.81) far exceeding the largest regressions
(+4.96, +2.97, +2.71). Upgrading 8 of 40 layers -- 20% of the layers, 12.1 GiB
per rank -- buys a 5.8% NLL reduction and a 23.8% cut in distributional error
against the reference logprobs.

Raw data: `T9_score_q2_general100.tsv`, `T9_score_q3_8L_general100.tsv`,
`T9_compare_q2_vs_8L.txt`.

## Speed: matched configuration, 16K/512

Both arms `--ctx-alloc 32768`, `DS4_METAL_DISABLE_V41_8K_CHUNK=1` (4K prefill
chunk), `tests/long_context_story_prompt.txt`, same binary, same session. The
matched chunk matters: 8L *requires* the 4K chunk (see the ceiling below), so
comparing it against Q2 on the stock 8K chunk would confound the chunk with the
model.

| config | pp | tg steady | TTFT | planned/rank |
|---|---|---|---|---|
| Q4 TP2 **streaming** (reference) | 174.12 | 10.31 | ~154 ms | 99.11 |
| Q2 TP2 resident, 4K chunk | **404.48** | **21.57** | 54.8 ms | 88.00 |
| **Q3 8L TP2 resident, 4K chunk** | **398.05** | **20.06** | 57.4 ms | **100.07** |
| delta 8L vs Q2 | -1.6% | **-7.0%** | +4.7% | +12.07 |

Against the configuration Q4 actually has to use on this pair -- streaming --
8L is **+129% prefill and +95% decode**, at better-than-Q2 quality.

## The memory ceiling, mapped

8L is not fragile; it is close to a hard edge that the engine does not enforce.
The admit check allows `host/8*7` = **106.48 GiB** (`ds4.c:66261`), but this
pair stops working around **100 GiB planned per rank**:

| config | prefill chunk | planned/rank | result |
|---|---|---|---|
| 8L @ ctx 4096 | 2048 | ~96.6 | works (192.29 / 18.61) |
| 8L @ ctx 8192 | 4096 | ~98.2 | works (194.98 / 19.24) |
| **8L @ ctx 32768, 4K chunk** | 4096 | **100.07** | **works (398.05 / 20.06)** |
| 8L @ ctx 32768, stock chunk | 8192 | 102.61 | **fails** |
| 8L @ ctx 262144, 4K chunk | 4096 | 101.86 | **fails** |
| 10L @ ctx 262144, 4K chunk | 4096 | 107.42 | refused at admit |

So the working ceiling is between **100.07 (works)** and **101.86 (fails)**.
That also explains the whole series: Q2 at 88.00 is comfortable, the 6L splice
at 99.05 squeaks through, 8L at 100.07 is the largest shard that fits, and 10L
at 108.61 never had a chance. **8L is at the edge of what this pair can hold.**

### Max Q4 layers per context, from the measured ceiling

Shard cost is exactly **1.78 GiB per Q4 layer**: `shard(k) = 80.56 + 1.78k`,
validated against every measured point (k=0 -> 80.56, k=6 -> 91.24,
k=8 -> 94.80, k=10 -> 98.36). Adding the measured context term at the 4K
prefill chunk gives `planned(k) = shard(k) + ctx_term`.

Bracket after probing with the 8L at an intermediate context:

| point | planned | result |
|---|---|---|
| 8L @ ctx 32768 | 99.86 | works |
| **8L @ ctx 131072** | **100.72** | **works** (403.21 pp / 19.99 tg) |
| 8L @ ctx 262144 | 101.86 | fails |

So the ceiling sits between **100.72 (good)** and **101.86 (bad)** -- a 1.14 GiB
window, narrower than one layer.

Context terms measured at the 4K chunk: 32K -> 5.07, 128K -> 5.92,
256K -> 7.05 GiB.

**Max Q4 layers by context:**

| ctx-alloc | max k | planned at max k | margin below 100.72 |
|---|---|---|---|
| 32768 | **10** | 99.63 | 1.09 |
| 131072 | **9** | 98.44 | 2.28 |
| **262144 (256K)** | **7** | **100.07** | **0.65** |

At 256K, k=7 plans 100.07 GiB -- **0.65 GiB below a proven-good point** and
1.79 GiB below the nearest known failure, so it has real margin. k=8 at 256K is
the 101.86 that fails. **7 is the answer at 256K**; 8 layers is capped at 128K.

Not yet built or run: a 7-layer splice. The figure is arithmetic from a model
validated at four points plus a proven-good point above it, not a measurement.
Confirming it costs a ~15 min splice and a ~19 min transfer.

### The failure mode is silent, and that is the dangerous part

Past ~101 GiB the run passes admission, both ranks bind, and then the worker
stalls at the layer-0 big gate without logging anything; the coordinator times
out with `big gate header exchange failed or peer closed ... Resource
temporarily unavailable`. The worker stays alive. Pushed to ~102 GiB, promax
went into memory pressure deep enough that **sshd itself stalled for minutes**,
without ever invoking the OOM killer.

Ruled out by single-variable tests, each with a confirmed worker log:

* **Not admission** -- both ranks print their plan and bind.
* **Not RDMA or pinning** -- identical failure over `--transport tcp`.
  (Noted in passing: promax has `ulimit -l` 15.2 GiB, spark is unlimited.
  Real asymmetry, not this cause.)
* **Not the gate timeout** -- unchanged at `DS4_TP_GATE_TIMEOUT_MS=30000`
  against the 750 ms default.
* **Not chunk count** -- a single-chunk 8192-token prefill at ctx 32768 fails,
  while a two-chunk prefill at ctx 8192 succeeds. It tracks total planned
  memory, nothing else.

**Owed:** the admit check should bound itself to what the host can actually
hold, not `host*7/8`. A guard that refused at ~101 GiB would have turned three
silent multi-minute hangs into one clear error. This is the same lesson as the
Makefile arch guard: the defect was not the failure, it was that nothing caught
it.

## Recommendation

Adopt **Q3 8L TP2 resident at ctx 32768 with the 4K prefill chunk** as the
pair's configuration when quality matters, and keep Q2 resident when decode
latency matters more than 5.8% NLL. 8L is strictly better than the 6L splice --
more Q4 layers, and 6L was never quality-scored at all.

Do **not** pursue 10L: refused at admit, and the 100 GiB ceiling leaves no room.

## What is still not established

* **Full Q4 as the upper reference.** We know 8L closes part of the Q2->Q4
  quality gap but not what fraction, because Q4 cannot run resident on this
  pair (151.17 GiB/rank) and would have to be scored streaming. Without it,
  "8L captures most of Q4's quality for 2/3 of the memory" is an assumption,
  not a measurement. This is the single most useful follow-up.
* **6L quality**, for the same reason -- it would say whether layers 32-33 earn
  their 3.56 GiB.
* **Long-context quality.** Scoring ran at ctx 4096; the `-long` and
  `-extended` manifests exercise sparse attention and were not run.
* **Output correctness beyond scoring.** NLL agreement is strong evidence the
  mixed path is computing correctly, but no byte-level or state oracle was run
  against a single-box reference.
