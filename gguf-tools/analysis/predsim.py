"""Cross-layer routing predictability: can the state at layer L name layer
L+k's experts?

Controls before any claim:
  A. replica-select(dumped logits[L] , bias[L]) must equal the actual
     selection from the routing dump for ~every (token, layer). Validates the
     selection replica (sqrt(softplus) + bias, top-6).
  B. gate_inp[L] @ norm[L] must reproduce the dumped logits (cosine ~1).
     Validates tensor extraction, orientation, and the norm-input pairing.

Prediction: logits_pred = gate_inp[L+k] @ (norm[L] / w_norm[L] * w_norm[L+k]);
top-6 with bias[L+k]; overlap with the actual selection at (token, L+k).
Baselines: chance = 6/384 = 1.56%; the identity predictor (same expert ids as
layer L) measured 1.4%.
"""
import sys, glob, os
import numpy as np
sys.path.insert(0, '/home/jason/.claude/jobs/e839ef4c/tmp')
from ggufx import GGUF

T = '/home/jason/.claude/jobs/e839ef4c/tmp'
MODEL = '/home/jason/models/ds41f-q4/DeepSeek-V4.1-Flash-Q4.gguf'
N_LAYER, N_EXPERT, N_USED = 40, 384, 6

def softplus(x):
    return np.where(x > 20, x, np.log1p(np.exp(np.minimum(x, 20.0))))

def select6(logits, bias):
    score = np.sqrt(softplus(logits)) + bias
    return set(np.argsort(-score, kind='stable')[:N_USED].tolist())

# ---- dumps ----
def load_dumps(prefix, name, n):
    out = {}
    for p in glob.glob(os.path.join(T, 'hcap', f'h_{name}-*_pos*.bin')):
        base = os.path.basename(p)
        il = int(base.split(f'{name}-')[1].split('_pos')[0])
        pos = int(base.split('_pos')[1].split('.bin')[0])
        v = np.fromfile(p, dtype='<f4').astype(np.float64)
        assert v.size == n, (p, v.size)
        out[(pos, il)] = v
    return out

norms = load_dumps(T, 'v41_ffn_norm', 5120)
logits_d = load_dumps(T, 'v41_route_logits', N_EXPERT)
positions = sorted({p for p, _ in norms})
print('dumped: %d positions x layers (norm %d, logits %d entries)'
      % (len(positions), len(norms), len(logits_d)))

# ---- actual selections from the routing dump, aligned by order ----
actual = {}     # (token_idx, layer) -> set of 6
tok_seq = []
for line in open(os.path.join(T, 'route_hcap.txt')):
    f = line.split()
    if f[0] == 'p' or len(f) < 3:
        continue
    t, l = int(f[0]), int(f[1])
    actual[(t, l)] = set(int(x) for x in f[2:])
    if not tok_seq or tok_seq[-1] != t:
        tok_seq.append(t)
# dump pos <-> route token: both are dense sequences in decode order
pos_of_token = {t: positions[i] for i, t in enumerate(tok_seq) if i < len(positions)}

# ---- weights ----
g = GGUF(MODEL)
gate_inp, wnorm, bias = {}, {}, {}
for il in range(N_LAYER):
    gate_inp[il] = g.read(f'blk.{il}.ffn_gate_inp.weight')       # (384, 7168)
    wnorm[il] = g.read(f'blk.{il}.ffn_norm.weight')              # (7168,)
    bias[il] = g.read(f'blk.{il}.exp_probs_b.bias')        # (384,)
print('weights: gate_inp %s wnorm %s bias %s'
      % (gate_inp[0].shape, wnorm[0].shape, bias[0].shape))

# ---- control A: selection replica on dumped logits ----
okA = totA = 0
for (t, l), sel in actual.items():
    p = pos_of_token.get(t)
    if p is None or (p, l) not in logits_d:
        continue
    rep = select6(logits_d[(p, l)], bias[l])
    okA += len(rep & sel)
    totA += N_USED
print('CONTROL A (selection replica vs actual): %.2f%% of expert picks match'
      % (100.0 * okA / totA))

# ---- control B: recomputed logits vs dumped ----
cos, okB, totB = [], 0, 0
for (t, l), sel in list(actual.items())[:400]:
    p = pos_of_token.get(t)
    if p is None or (p, l) not in norms:
        continue
    lg = gate_inp[l] @ norms[(p, l)]
    d = logits_d[(p, l)]
    cos.append(float(lg @ d / (np.linalg.norm(lg) * np.linalg.norm(d) + 1e-30)))
    okB += len(select6(lg, bias[l]) & sel)
    totB += N_USED
print('CONTROL B (gate_inp @ dumped norm): cosine mean %.6f min %.6f; '
      'selection match %.2f%%'
      % (np.mean(cos), np.min(cos), 100.0 * okB / totB))

# ---- prediction curve ----
print()
print('prediction: top-6 of router(L+k) applied to state at L')
print('%4s %10s %12s %10s' % ('k', 'overlap%', 'tokens*layers', 'full-6%'))
for k in (1, 2, 4):
    hit = tot = full = cases = 0
    for ti, t in enumerate(tok_seq):
        p = pos_of_token.get(t)
        if p is None:
            continue
        for l in range(N_LAYER - k):
            if (p, l) not in norms or (t, l + k) not in actual:
                continue
            u = norms[(p, l)] / np.where(np.abs(wnorm[l]) < 1e-12, 1e-12, wnorm[l])
            x = u * wnorm[l + k]
            pred = select6(gate_inp[l + k] @ x, bias[l + k])
            inter = len(pred & actual[(t, l + k)])
            hit += inter
            tot += N_USED
            full += (inter == N_USED)
            cases += 1
    if tot:
        print('%4d %9.1f%% %12d %9.1f%%'
              % (k, 100.0 * hit / tot, cases, 100.0 * full / cases))
print()
print('baselines: chance 1.6%; identity (same ids as L) measured 1.4%')
