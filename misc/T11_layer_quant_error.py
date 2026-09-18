"""Per-layer Q2-vs-Q4 quantization divergence for V4.1 routed experts.

The source experts are FP4 (config.json expert_dtype=fp4) and Q4_K is a ~4.5-bit
format, so Q4 sits close to the source and ||Q2-Q4|| is a good proxy for Q2's
own error. Sampling rows rather than reading 427 GiB keeps this to minutes.
"""
import sys, json, math
import numpy as np
sys.path.insert(0, "/home/jason/ds4-41flash/.claude/worktrees/q3-tp2/gguf-tools/mixed")
from splice_mixed_expert_layers_gguf import parse_gguf, qtype_name
from pathlib import Path

T = json.load(open("/home/jason/.claude/jobs/882780ed/tmp/iq2_tables.json"))
KMASK = np.array(T["kmask_iq2xs"], dtype=np.uint8)
KSIGNS = np.array(T["ksigns_iq2xs"], dtype=np.uint8)
GRID = np.array(T["iq2xxs_grid"], dtype=np.uint64)
GRID_BYTES = GRID.view(np.uint8).reshape(256, 8).astype(np.int32)  # magnitudes

QK = 256
TSIZE = {"Q4_K": 144, "Q2_K": 84, "IQ2_XXS": 66}


def f16(buf, off):
    return float(np.frombuffer(buf[off:off + 2], dtype=np.float16)[0])


def deq_q4k(b):
    """b: one 144-byte block -> 256 floats"""
    d, dmin = f16(b, 0), f16(b, 2)
    sc = np.frombuffer(b[4:16], dtype=np.uint8)
    q = np.frombuffer(b[16:144], dtype=np.uint8)
    out = np.empty(QK, dtype=np.float32)
    def sm(j):
        if j < 4:
            return (sc[j] & 63), (sc[j + 4] & 63)
        return ((sc[j + 4] & 0xF) | ((sc[j - 4] >> 6) << 4),
                (sc[j + 4] >> 4) | ((sc[j] >> 6) << 4))
    o = 0
    for n in range(0, QK, 64):
        j = n // 32
        s1, m1 = sm(j); s2, m2 = sm(j + 1)
        blk = q[o:o + 32].astype(np.int32)
        out[n:n + 32] = d * float(s1) * (blk & 0xF) - dmin * float(m1)
        out[n + 32:n + 64] = d * float(s2) * (blk >> 4) - dmin * float(m2)
        o += 32
    return out


def deq_q2k(b):
    """b: one 84-byte block -> 256 floats"""
    sc = np.frombuffer(b[0:16], dtype=np.uint8)
    q = np.frombuffer(b[16:80], dtype=np.uint8)
    d, dmin = f16(b, 80), f16(b, 82)
    out = np.empty(QK, dtype=np.float32)
    y = 0; is_ = 0; o = 0
    for n in range(0, QK, 128):
        shift = 0
        for _ in range(4):
            s = sc[is_]; is_ += 1
            dl, ml = d * float(s & 0xF), dmin * float(s >> 4)
            blk = q[o:o + 16].astype(np.int32)
            out[y:y + 16] = dl * ((blk >> shift) & 3) - ml
            y += 16
            s = sc[is_]; is_ += 1
            dl, ml = d * float(s & 0xF), dmin * float(s >> 4)
            blk = q[o + 16:o + 32].astype(np.int32)
            out[y:y + 16] = dl * ((blk >> shift) & 3) - ml
            y += 16
            shift += 2
        o += 32
    return out


def deq_iq2xxs(b):
    """b: one 66-byte block -> 256 floats"""
    d = f16(b, 0)
    qs = np.frombuffer(b[2:66], dtype=np.uint16)
    out = np.empty(QK, dtype=np.float32)
    for ib in range(8):
        a = qs[4 * ib:4 * ib + 4].view(np.uint32) if False else None
        a32 = np.frombuffer(qs[4 * ib:4 * ib + 4].tobytes(), dtype=np.uint32)
        aux0, aux1 = int(a32[0]), int(a32[1])
        db = d * (0.5 + (aux1 >> 28)) * 0.25
        a8 = [(aux0 >> (8 * k)) & 0xFF for k in range(4)]
        for l in range(4):
            g = GRID_BYTES[a8[l]]
            signs = int(KSIGNS[(aux1 >> (7 * l)) & 127])
            sgn = np.where((signs & KMASK) != 0, -1.0, 1.0)
            out[ib * 32 + l * 8:ib * 32 + l * 8 + 8] = db * g * sgn
    return out


DEQ = {"Q4_K": deq_q4k, "Q2_K": deq_q2k, "IQ2_XXS": deq_iq2xxs}


def deq_row(fh, base, ne, tname):
    nb = ne // QK
    raw = np.frombuffer(read_at(fh, base, nb * TSIZE[tname]), dtype=np.uint8)
    f = DEQ[tname]
    return np.concatenate([f(raw[i * TSIZE[tname]:(i + 1) * TSIZE[tname]].tobytes())
                           for i in range(nb)])


def read_at(fh, off, n):
    fh.seek(off); b = fh.read(n); assert len(b) == n, (off, n, len(b)); return b


Q2P = "/home/jason/models/ds41f-q2/DeepSeek-V4.1-Flash-Q2.gguf"
Q4P = "/home/jason/models/ds41f-q4/DeepSeek-V4.1-Flash-Q4.gguf"
g2, g4 = parse_gguf(Path(Q2P)), parse_gguf(Path(Q4P))
f2, f4 = open(Q2P, "rb", buffering=0), open(Q4P, "rb", buffering=0)

N_LAYER = 40
ROWS_PER_TENSOR = int(sys.argv[1]) if len(sys.argv) > 1 else 24
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 20260918
rng = np.random.default_rng(SEED)

print(f"sampling {ROWS_PER_TENSOR} rows per tensor per layer\n")
print(f"{'layer':>5} {'gate':>9} {'up':>9} {'down':>9} {'weighted':>9}")
results = {}
for L in range(N_LAYER):
    per = {}
    for part in ("gate", "up", "down"):
        nm = f"blk.{L}.ffn_{part}_exps.weight"
        t2, t4 = g2.tensor_by_name[nm], g4.tensor_by_name[nm]
        ne = int(t2.dims[0])
        nrows = 1
        for d in t2.dims[1:]:
            nrows *= int(d)
        n2, n4 = qtype_name(t2.ggml_type), qtype_name(t4.ggml_type)
        rs2, rs4 = (ne // QK) * TSIZE[n2], (ne // QK) * TSIZE[n4]
        idx = rng.choice(nrows, size=min(ROWS_PER_TENSOR, nrows), replace=False)
        num = den = 0.0
        for r in idx:
            w2 = deq_row(f2, t2.data_offset + int(r) * rs2, ne, n2)
            w4 = deq_row(f4, t4.data_offset + int(r) * rs4, ne, n4)
            num += float(np.sum((w2 - w4) ** 2))
            den += float(np.sum(w4 ** 2))
        per[part] = math.sqrt(num / den) if den > 0 else float("nan")
        per[part + "_bytes"] = t4.n_bytes
    tot = sum(per[p + "_bytes"] for p in ("gate", "up", "down"))
    wgt = sum(per[p] * per[p + "_bytes"] for p in ("gate", "up", "down")) / tot
    results[L] = {**{p: per[p] for p in ("gate", "up", "down")}, "weighted": wgt}
    print(f"{L:5d} {per['gate']:9.5f} {per['up']:9.5f} {per['down']:9.5f} {wgt:9.5f}")

json.dump(results, open(f"/home/jason/.claude/jobs/882780ed/tmp/layer_error_{SEED}.json", "w"), indent=1)
order = sorted(results, key=lambda L: -results[L]["weighted"])
print("\nlayers ranked by relative Q2-vs-Q4 divergence (highest first):")
print(" ".join(str(L) for L in order))
print("\ntop 10:", ", ".join(f"{L}({results[L]['weighted']:.4f})" for L in order[:10]))
print("bottom 10:", ", ".join(f"{L}({results[L]['weighted']:.4f})" for L in order[-10:]))
