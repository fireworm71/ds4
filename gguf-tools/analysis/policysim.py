"""Rank cache policies offline against the Belady bound.

Validation first, twice. The trace replay must reproduce BOTH hardware points:
LRU at 87.4% and CLOCK at 88.0% (hardware miss counts 25,635 and 24,507, which
were bit-identical across repeated runs, so they are exact targets rather than
noisy ones). A simulator that gets LRU right by luck but CLOCK wrong cannot be
trusted to rank anything.

Then the question the peer session asked and that I want answered too: what is
Belady exploiting that CLOCK is not? Both policies see the same misses; the
difference is entirely in WHICH resident entry they throw away. So the
diagnostic is the next-use distance of evicted entries -- the ideal policy
evicts things that are far away, and any online policy's loss is the mass it
evicts that turns out to be near.

Approximation worth naming: the hardware protects all six experts of the
current call from eviction by stamping them before any miss is serviced. This
replay processes accesses one at a time, so a just-inserted entry could in
principle be evicted by a later access within the same call. With 3,977 slots
and six accesses the odds are negligible, but it is an approximation, not an
identity.
"""
import sys, collections, heapq

WARMUP = 20

def load(path):
    out = []
    for line in open(path):
        f = line.split()
        if len(f) >= 3 and f[0] != 'p':
            out.append((int(f[0]), int(f[1]), [int(x) for x in f[2:]]))
    return out

def flatten(trace):
    seq, score_from = [], 0
    for token, layer, experts in trace:
        for e in dict.fromkeys(experts):
            if token <= WARMUP:
                score_from = len(seq) + 1
            seq.append((layer, e))
    return seq, score_from

def next_use_table(seq):
    nxt = [len(seq)] * len(seq)
    last = {}
    for i in range(len(seq) - 1, -1, -1):
        k = seq[i]
        nxt[i] = last.get(k, len(seq))
        last[k] = i
    return nxt

def run_lru(seq, score_from, nxt, slots):
    used = collections.OrderedDict()
    hits = misses = 0
    eplen = []
    for i, k in enumerate(seq):
        score = i >= score_from
        if k in used:
            if score: hits += 1
            used.move_to_end(k)
            used[k] = nxt[i]     # refresh: stale next_use gave negative distances
        else:
            if score: misses += 1
            if len(used) >= slots:
                vk, vi = used.popitem(last=False)
                if score: eplen.append(vi - i)
            used[k] = nxt[i]
    return hits, misses, eplen

def run_clock(seq, score_from, nxt, slots, lives_cap=3):
    """CLOCK second chance, mirroring DS4_CUDA_EXPERT_EVICT=1."""
    slot_key = [None] * slots
    slot_lives = [0] * slots
    slot_nxt = [0] * slots
    where = {}
    hand = 0
    n_filled = 0
    hits = misses = 0
    eplen = []
    for i, k in enumerate(seq):
        score = i >= score_from
        j = where.get(k)
        if j is not None:
            if score: hits += 1
            if slot_lives[j] < lives_cap:
                slot_lives[j] += 1
            slot_nxt[j] = nxt[i]  # refresh, same reason as LRU above
            continue
        if score: misses += 1
        if n_filled < slots:
            j = n_filled
            n_filled += 1
        else:
            while True:
                if slot_lives[hand] == 0:
                    j = hand
                    hand = (hand + 1) % slots
                    break
                slot_lives[hand] -= 1
                hand = (hand + 1) % slots
            if score: eplen.append(slot_nxt[j] - i)
            del where[slot_key[j]]
        slot_key[j] = k
        slot_lives[j] = 0
        slot_nxt[j] = nxt[i]
        where[k] = j
    return hits, misses, eplen

def run_belady(seq, score_from, nxt, slots):
    resident, heap = {}, []
    hits = misses = 0
    eplen = []
    for i, k in enumerate(seq):
        score = i >= score_from
        if k in resident:
            if score: hits += 1
        else:
            if score: misses += 1
            if len(resident) >= slots:
                while heap:
                    neg, vk = heapq.heappop(heap)
                    if vk in resident and resident[vk] == -neg:
                        del resident[vk]
                        if score: eplen.append(-neg - i)
                        break
        resident[k] = nxt[i]
        heapq.heappush(heap, (-nxt[i], k))
    return hits, misses, eplen

def pct(v, q):
    if not v: return float('nan')
    s = sorted(v)
    return s[min(len(s) - 1, int(q * len(s)))]

if __name__ == '__main__':
    path = sys.argv[1]
    slots = int(sys.argv[2]) if len(sys.argv) > 2 else 3977
    trace = load(path)
    seq, score_from = flatten(trace)
    nxt = next_use_table(seq)
    print('trace %d rows -> %d expert accesses, scoring from %d'
          % (len(trace), len(seq), score_from))
    print('accesses per token = %d, so %d slots = %.1f tokens of horizon'
          % (240, slots, slots / 240.0))

    results = []
    for name, fn, target in (('LRU', run_lru, 25635), ('CLOCK', run_clock, 24507),
                             ('Belady', run_belady, None)):
        h, m, ep = fn(seq, score_from, nxt, slots)
        rate = 100.0 * h / (h + m)
        results.append((name, rate, m, ep))
        tgt = ''
        if target:
            tgt = '   hardware %d misses, delta %+.1f%%' % (
                target, 100.0 * (m - target) / target)
        print('%-7s hit %.2f%%  misses %6d%s' % (name, rate, m, tgt))

    print()
    print('=== what does each policy throw away? ===')
    print('next-use distance of evicted entries, in EXPERT ACCESSES (240/token)')
    print('%-7s %10s %10s %10s %10s' % ('policy', 'p10', 'median', 'p90', 'n'))
    for name, rate, m, ep in results:
        print('%-7s %10d %10d %10d %10d'
              % (name, pct(ep, 0.10), pct(ep, 0.50), pct(ep, 0.90), len(ep)))
    print()
    print('An eviction whose next use is beyond the horizon (%d accesses) was'
          % slots)
    print('free. One inside it cost a future miss. That share is the loss:')
    for name, rate, m, ep in results:
        if not ep: continue
        bad = sum(1 for d in ep if d < slots)
        print('  %-7s %5.1f%% of evictions were reused within the horizon'
              % (name, 100.0 * bad / len(ep)))
