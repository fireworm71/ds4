# T20: the DSpark confidence NaN was a corrupt support artifact

Measured 2026-09-19 on the Spark pair. Closes the blocker T19 left open and
corrects T19's conclusion.

## 1. What was actually wrong

`ds41f-dspark-q4.gguf` holds non-finite weights in **37 of its 84 tensors**.

```
mtp.0.main_proj.weight   Q8_0  n=78643200  119541 NaN scales (4.8%)
mtp.0.main_norm.weight   F32   n=5120          36 NaN, absmax 3.2e38
mtp.0.attn_q_a.weight    Q8_0  n=6553600     9875 NaN scales
... 34 more
```

An RMS-norm weight should sit around 1.0; `main_norm` ranges to 3.2e38. Roughly
4.8% of every Q8_0 tensor's F16 scales are NaN -- the signature of random bytes,
not of a numerical blow-up.

The engine's validation is shape- and type-only, so the file passed with
`invalid=0`. The single visible symptom was a drafter that proposed nothing.

## 2. How it was localised

Bisecting the propose path, one link at a time, rather than guessing:

| stage | result |
|---|---|
| captured hidden (`dspark_target_hidden`) | **finite**, 0 NaN of 15360 |
| `stage0_proj = matmul(main_proj, hidden)` | **5120/5120 NaN** <- enters here |
| `main_x = rms_norm(stage0_proj, main_norm)` | NaN |
| stage input row 0 | NaN (rows 1..5, the draft embeddings, stay finite) |
| every stage output | NaN |
| `confidence0` | NaN -> fails every threshold comparison |

The decisive line is the engine's own `stagein` probe:

```
ds4: dbg stagein stage0_proj |.|=0.0000 nan=5120/5120  target_hidden nan=0/15360
ds4: dbg stagein main_x=nan target_hidden=18.7997
```

Clean input, all-NaN output, across a single matmul. That points at the weight,
and scanning the GGUF confirmed it.

## 3. The clean artifact drafts

`ds41f-dspark-q4v3.gguf` scans clean -- **0 non-finite tensors** -- with sane
magnitudes (`main_norm` absmax 0.084, `main_proj` scales absmax 0.0030). Same
binary, same target, same command, only the support model swapped:

```
                        q4 (corrupt)     q4v3 (clean)
confidence0             nan              1.887597
proposed                0                16
accepted_draft          0                3
draft_len_hist          none             1:1,2:2,4:1
```

Q2 single box, 2048 ctx, 32 generated tokens. The drafter works.

## 4. Why T19 got this wrong

T19 recorded "q4 and q4v3 behave the same -- ruled out, checkpoint-specific"
and concluded the fault was inside the drafter's forward pass on CUDA. That
comparison was invalid.

The DSpark scheduler refuses to draft near the end of a generation:

```
ds4: DSpark scheduler tail skip max=4 min=10
```

With `--gen-tokens` below that floor the propose path **never executes**. Both
models looked equally silent because neither ran. The lesson is narrow and
practical: a null result from a run that never reached the code under test is
not evidence, and "both behave the same" deserves a check that either behaved
at all.

## 5. The durable fix

A corrupt-but-well-shaped artifact cost several sessions. Shape validation now
has a value counterpart: `dspark_weights_validate_finite` walks the drafter's
F32 and F16 tensors at load, names each offending tensor, and disables drafting
with an explicit message instead of paying for a propose chain that cannot
work.

```
ds4: DSpark tensor mtp.0.main_norm.weight holds 39 non-finite of 5120 values; the support GGUF is corrupt
ds4: DSpark: 12 tensors hold non-finite values (only the first 8 listed)
ds4: DSpark support model detected: ... invalid=0 nonfinite=12 ...
ds4: DSpark support model is corrupt (12 tensors hold non-finite weights); drafting disabled, decoding target-only. Re-convert the support GGUF.
```

`q4v3` reports `nonfinite=0` and drafting stays enabled.

F32 and F16 only: together a few hundred MiB at worst, and every corrupt tensor
observed had non-finite values among them. Covering the Q8_0 bulk would mean
reading the whole 8 GiB file on every load, for a check that the cheap scan
already passes.

The scan counts 12 tensors where the offline scan counts 37 because the engine
only binds the tensors the drafter actually uses, and only F32/F16 among those.
Detection is what matters, not the census.

## 6. Status of the artifacts on disk

| file | verdict |
|---|---|
| `ds41f-dspark-q4.gguf` | **corrupt** -- 37 tensors non-finite; do not use |
| `ds41f-dspark-q4v2.gguf` | **corrupt** -- 37 tensors non-finite; do not use |
| `ds41f-dspark-q4v3.gguf` | **clean** -- 0 non-finite; this is the one to use |

Two of the three support artifacts on disk are corrupt, in the same way and to
the same extent (37 tensors, ~4.8% of Q8_0 scales). That is consistent with a
conversion step that was fixed between v2 and v3, and it means the surviving
good file should be treated as the only one -- the other two are worth deleting
rather than leaving as traps.

The peer box carried only the corrupt `q4`, so every TP2 run that named `q4v3`
was still loading corrupt weights on the worker side. The clean file has been
copied across.

T19 section 2's conclusion stands unchanged and separately: the drafter's
expert count (128, not the backbone's 384) really was mis-inferred, and that
fix is still required. It was necessary but not sufficient.
