"""Candidate online policies, run against the pre-registered gate.

GATE, registered before any of these were run: an ONLINE policy (no future
knowledge) must reach >= +2.0pp hit rate over LRU on this replay before it earns
any GPU time. Derivation: CLOCK's +0.55pp on replay delivered +1.4% throughput
on hardware at ~1.5 sigma, so +2.0pp is roughly a 16% miss cut and ~+5%
throughput -- the smallest gain worth validating against a 2.6% run-to-run
spread. Belady sits at +5.98pp, so the gate asks for a third of the bound.

The campaign prior (LFU, SLRU and static pinning all within +/-1% of LRU in
earlier work) says nothing here will clear it. Reporting the null and stopping
is the committed outcome if so -- no promoting the best loser.
"""
import sys, collections, math
sys.path.insert(0, '/home/jason/.claude/jobs/e839ef4c/tmp')
from policysim import load, flatten, next_use_table, run_lru, run_clock, run_belady


def run_lru_k(seq, score_from, nxt, slots, K=2):
    """LRU-K: order by the Kth-most-recent access instead of the most recent.
    This is the classic estimator of REUSE DISTANCE rather than recency, which
    matters because the eviction diagnostic showed LRU's failure is exactly a
    ranking failure on next-use distance (median 85 tokens vs Belady's 288).
    Entries with fewer than K accesses are evicted first -- unknown distance
    treated as infinite, per the standard formulation."""
    hist = collections.defaultdict(collections.deque)
    resident = collections.OrderedDict()   # insertion order is irrelevant, used as a set
    hits = misses = 0
    for i, k in enumerate(seq):
        h = hist[k]
        h.append(i)
        if len(h) > K:
            h.popleft()
        score = i >= score_from
        if k in resident:
            if score: hits += 1
            continue
        if score: misses += 1
        if len(resident) >= slots:
            victim = min(resident, key=lambda x: (len(hist[x]) >= K, hist[x][0]))
            del resident[victim]
        resident[k] = 1
    return hits, misses, []


def run_lfu_decay(seq, score_from, nxt, slots, halflife_tokens=64):
    """Decayed LFU. Frequency estimates selection probability p, and expected
    next use is ~1/p, so this is a direct if noisy estimate of what Belady
    sorts on. Half-life in ACCESSES (240 per token)."""
    hl = halflife_tokens * 240.0
    lam = math.log(2.0) / hl
    val, last = {}, {}
    resident = {}
    hits = misses = 0
    for i, k in enumerate(seq):
        if k in val:
            val[k] = val[k] * math.exp(-lam * (i - last[k])) + 1.0
        else:
            val[k] = 1.0
        last[k] = i
        score = i >= score_from
        if k in resident:
            if score: hits += 1
            continue
        if score: misses += 1
        if len(resident) >= slots:
            victim = min(resident, key=lambda x: val[x] * math.exp(-lam * (i - last[x])))
            del resident[victim]
        resident[k] = 1
    return hits, misses, []


def run_slru(seq, score_from, nxt, slots, prot_frac=0.5):
    """Segmented LRU: first touch lands in probation, a second hit promotes to
    protected, demotions fall back to probation. Aimed at the 61% of loads that
    are never reused."""
    prot_cap = int(slots * prot_frac)
    prob_cap = slots - prot_cap
    prot, prob = collections.OrderedDict(), collections.OrderedDict()
    hits = misses = 0
    for i, k in enumerate(seq):
        score = i >= score_from
        if k in prot:
            if score: hits += 1
            prot.move_to_end(k)
            continue
        if k in prob:
            if score: hits += 1
            del prob[k]
            prot[k] = 1
            if len(prot) > prot_cap:
                dk, _ = prot.popitem(last=False)
                prob[dk] = 1
                if len(prob) > prob_cap:
                    prob.popitem(last=False)
            continue
        if score: misses += 1
        prob[k] = 1
        if len(prob) > prob_cap:
            prob.popitem(last=False)
    return hits, misses, []


def run_per_layer_lru(seq, score_from, nxt, slots, n_layers=40):
    """Equal per-layer pools. Tests whether the shared global pool lets some
    layers starve others, given every layer needs exactly 6 experts per token."""
    per = max(1, slots // n_layers)
    pools = collections.defaultdict(collections.OrderedDict)
    hits = misses = 0
    for i, k in enumerate(seq):
        score = i >= score_from
        pool = pools[k[0]]
        if k in pool:
            if score: hits += 1
            pool.move_to_end(k)
            continue
        if score: misses += 1
        pool[k] = 1
        if len(pool) > per:
            pool.popitem(last=False)
    return hits, misses, []


if __name__ == '__main__':
    path = sys.argv[1]
    slots = int(sys.argv[2]) if len(sys.argv) > 2 else 3977
    trace = load(path)
    seq, score_from = flatten(trace)
    nxt = next_use_table(seq)

    base_h, base_m, _ = run_lru(seq, score_from, nxt, slots)
    base_rate = 100.0 * base_h / (base_h + base_m)
    bh, bm, _ = run_belady(seq, score_from, nxt, slots)
    b_rate = 100.0 * bh / (bh + bm)

    print('gate: online policy must reach >= LRU +2.0pp on replay')
    print('LRU      %.2f%%  (baseline)' % base_rate)
    print('Belady   %.2f%%  (+%.2fpp, bound, not a target)' % (b_rate, b_rate - base_rate))
    print()
    print('%-22s %8s %9s %8s  %s' % ('policy', 'hit%', 'delta pp', 'misses', 'gate'))

    cands = [
        ('CLOCK lives=3', lambda: run_clock(seq, score_from, nxt, slots, 3)),
        ('CLOCK lives=1', lambda: run_clock(seq, score_from, nxt, slots, 1)),
        ('CLOCK lives=7', lambda: run_clock(seq, score_from, nxt, slots, 7)),
        ('LRU-2', lambda: run_lru_k(seq, score_from, nxt, slots, 2)),
        ('LRU-3', lambda: run_lru_k(seq, score_from, nxt, slots, 3)),
        ('LFU decay hl=16tok', lambda: run_lfu_decay(seq, score_from, nxt, slots, 16)),
        ('LFU decay hl=64tok', lambda: run_lfu_decay(seq, score_from, nxt, slots, 64)),
        ('LFU decay hl=256tok', lambda: run_lfu_decay(seq, score_from, nxt, slots, 256)),
        ('SLRU 50/50', lambda: run_slru(seq, score_from, nxt, slots, 0.5)),
        ('SLRU 75/25', lambda: run_slru(seq, score_from, nxt, slots, 0.75)),
        ('per-layer LRU', lambda: run_per_layer_lru(seq, score_from, nxt, slots)),
    ]
    best = None
    for name, fn in cands:
        h, m, _ = fn()
        r = 100.0 * h / (h + m)
        d = r - base_rate
        mark = 'PASS' if d >= 2.0 else ''
        print('%-22s %7.2f%% %+8.2f %8d  %s' % (name, r, d, m, mark))
        if best is None or d > best[1]:
            best = (name, d)
    print()
    print('best online candidate: %s at %+.2fpp' % best)
    if best[1] < 2.0:
        print('GATE NOT MET. Committed outcome: report the null and stop.')
        print('No GPU validation time is spent, and the best loser is not promoted.')
