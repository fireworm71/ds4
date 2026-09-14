"""Fixed per-layer resident tables: does pinning a static set help?

Jason's idea: precompute a per-layer table of experts to keep resident, rather
than letting a policy decide dynamically.

Tested here in its STRONGEST possible form. The table is built with full
knowledge of the future -- top-K experts per layer by actual access count over
the whole trace. No online method (popularity estimation, Belady-residency
mining, anything) can beat an oracle static table, because that IS the optimal
static set for the sequence. So if the oracle table loses, every way of
computing such a table loses, and the idea is closed rather than merely
untuned.

Design: K slots per layer are pinned and never evicted; the remaining
(slots - 40*K) are a shared LRU pool over everything else. K=0 is exactly the
current policy, so the sweep includes its own control.

Also prices the "keep whole layers resident in a rotating window, drop L-2 and
read L+2" variant, which is a bandwidth question rather than a policy one.
"""
import sys, collections
sys.path.insert(0, '/home/jason/.claude/jobs/e839ef4c/tmp')
from policysim import load, flatten, next_use_table, run_lru, run_belady

def oracle_tables(seq, k):
    """Top-k (layer, expert) keys per layer by total access count in the trace.
    Future knowledge on purpose: this is the ceiling for static tables."""
    per_layer = collections.defaultdict(collections.Counter)
    for layer, e in seq:
        per_layer[layer][e] += 1
    pinned = set()
    for layer, ctr in per_layer.items():
        for e, _ in ctr.most_common(k):
            pinned.add((layer, e))
    return pinned

def run_pinned(seq, score_from, slots, pinned):
    """Pinned keys occupy slots permanently; everything else shares LRU over
    what is left. A pinned key never misses after its first touch."""
    budget = slots - len(pinned)
    if budget < 0:
        return None
    lru = collections.OrderedDict()
    warm = set()
    hits = misses = 0
    for i, k in enumerate(seq):
        score = i >= score_from
        if k in pinned:
            if k in warm:
                if score: hits += 1
            else:
                if score: misses += 1
                warm.add(k)
            continue
        if k in lru:
            if score: hits += 1
            lru.move_to_end(k)
        else:
            if score: misses += 1
            if len(lru) >= budget:
                lru.popitem(last=False)
            lru[k] = 1
    return hits, misses

if __name__ == '__main__':
    path = sys.argv[1]
    slots = int(sys.argv[2]) if len(sys.argv) > 2 else 3977
    trace = load(path)
    seq, score_from = flatten(trace)
    nxt = next_use_table(seq)
    n_layers = len({l for l, _ in seq})

    bh, bm, _ = run_lru(seq, score_from, nxt, slots)
    base = 100.0 * bh / (bh + bm)
    obh, obm, _ = run_belady(seq, score_from, nxt, slots)
    bel = 100.0 * obh / (obh + obm)
    print('slots %d over %d layers = %.1f per layer' % (slots, n_layers, slots / float(n_layers)))
    print('LRU     %.2f%%   Belady %.2f%% (+%.2fpp)' % (base, bel, bel - base))
    print()
    print('ORACLE static per-layer tables (built with full future knowledge --')
    print('no online table-builder can beat these):')
    print('%6s %8s %10s %9s' % ('K/layer', 'pinned', 'hit%', 'delta pp'))
    for k in (0, 5, 10, 25, 50, 75, 99):
        pinned = oracle_tables(seq, k)
        if len(pinned) >= slots:
            print('%6d  (pins exceed slot budget, skipped)' % k)
            continue
        r = run_pinned(seq, score_from, slots, pinned)
        if r is None:
            continue
        h, m = r
        rate = 100.0 * h / (h + m)
        print('%6d %8d %9.2f%% %+9.2f' % (k, len(pinned), rate, rate - base))

    print()
    print('=== "drop L-2, read L+2" as a whole-layer window: bandwidth check ===')
    experts = len({e for _, e in seq})
    per_expert_mib = 18.98
    layer_gib = experts * per_expert_mib / 1024.0
    print('a full layer is %d experts x %.2f MiB = %.1f GiB' % (experts, per_expert_mib, layer_gib))
    print('advancing the window one layer per layer-step (~3.8 ms) needs')
    print('  %.1f GiB / 3.8 ms = %.0f GiB/s sustained' % (layer_gib, layer_gib / 0.0038))
    print('measured device read rate is ~10 GiB/s, so this is ~%.0fx over budget.'
          % ((layer_gib / 0.0038) / 10.0))
    print('per token that would be %.0f GiB against the %.2f GiB actually moved.'
          % (layer_gib * 40, 0.56))
