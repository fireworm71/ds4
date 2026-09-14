"""Miss-conditional prediction coverage: of the experts that actually MISS at
layer L+k, what fraction did the computed router-at-L prediction name?

This is the number the prefetch design lives on: resident experts need no
prefetch, so the 68.5% all-expert overlap only matters insofar as it covers
the ~1.5 missing experts per stalling layer. Misses are identified by an LRU
replay over the same run's own prefill+decode routing (the replay method that
reproduced hardware bit-exactly on the full traces).
"""
import sys, collections
sys.path.insert(0, '/home/jason/.claude/jobs/e839ef4c/tmp')
import numpy as np
from predsim import (norms, actual, tok_seq, pos_of_token, gate_inp, wnorm,
                     bias, select6, N_LAYER, N_USED)

T = '/home/jason/.claude/jobs/e839ef4c/tmp'
SLOTS = 3977
used = collections.OrderedDict()
miss_at = collections.defaultdict(set)
pp, dec = [], []
for line in open(T + '/route_hcap.txt'):
    f = line.split()
    if not f:
        continue
    if f[0] == 'p':
        pp.append((int(f[2]), [int(x) for x in f[3:]]))
    elif len(f) >= 3:
        dec.append((int(f[0]), int(f[1]), [int(x) for x in f[2:]]))
for l, ex in pp:
    for e in ex:
        k = (l, e)
        if k in used:
            used.move_to_end(k)
        else:
            if len(used) >= SLOTS:
                used.popitem(last=False)
            used[k] = 1
for t, l, ex in dec:
    for e in dict.fromkeys(ex):
        k = (l, e)
        if k in used:
            used.move_to_end(k)
        else:
            miss_at[(t, l)].add(e)
            if len(used) >= SLOTS:
                used.popitem(last=False)
            used[k] = 1

for k in (1, 2):
    mh = mt = 0
    for t in tok_seq:
        p = pos_of_token.get(t)
        if p is None:
            continue
        for l in range(N_LAYER - k):
            key = (t, l + k)
            if (p, l) not in norms or key not in miss_at or not miss_at[key]:
                continue
            u = norms[(p, l)] / np.where(np.abs(wnorm[l]) < 1e-12, 1e-12, wnorm[l])
            pred = select6(gate_inp[l + k] @ (u * wnorm[l + k]), bias[l + k])
            mh += len(pred & miss_at[key])
            mt += len(miss_at[key])
    print('k=%d: MISS-conditional coverage %.1f%%  (%d of %d missed experts predicted)'
          % (k, 100.0 * mh / mt, mh, mt))

total_miss = sum(len(v) for v in miss_at.values())
total_demand = sum(len(set(ex)) for _, _, ex in dec)
print('replay: %d decode misses / %d demands (%.1f%% miss rate)'
      % (total_miss, total_demand, 100.0 * total_miss / total_demand))
