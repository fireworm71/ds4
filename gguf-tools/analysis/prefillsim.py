"""Prefill cache-policy replay: is the cyclic-reuse money real, and reachable?

Prefill's access pattern is chunk-major: layers 0..39 for chunk 1, then 0..39
for chunk 2. The deep run measured 43.5% hits where a naive cyclic-LRU model
predicts ~0 -- so the simple model is wrong somewhere, and the first job of
this replay is to VALIDATE against the measured number before trusting any
policy comparison (same discipline as the decode work: the decode replay had
to reproduce 87.4% and the LRU->CLOCK delta before its rankings were
believed).

Policies:
  LRU            validate against measured
  Belady         capacity-bounded ceiling
  MRU            evict most-recently-used: the textbook cyclic fix
  freeze         first fill wins, then bypass on miss (optimal STABLE subset
                 for a pure cycle; also the cheapest to implement in-engine)
  LRU+reserve    LRU at the +2-layer-reclaimed slot count, for reference

Stats first: chunk structure, per-layer overlap between consecutive chunks
(the analog of decode's cross-token overlap), and the demand mix.
"""
import sys, collections, heapq

def load_pp(path):
    """Returns (pp, dec): pp = [(call, layer, [uniques])], dec = decode rows."""
    pp, dec = [], []
    for line in open(path):
        f = line.split()
        if not f:
            continue
        if f[0] == 'p':
            pp.append((int(f[1]), int(f[2]), [int(x) for x in f[3:]]))
        elif len(f) >= 3:
            dec.append((int(f[0]), int(f[1]), [int(x) for x in f[2:]]))
    return pp, dec

def pp_stats(pp):
    layers = sorted({l for _, l, _ in pp})
    n_layers = len(layers)
    calls = len(pp)
    chunks = collections.defaultdict(dict)     # chunk index by layer-0 boundaries
    ci = -1
    for call, layer, ex in pp:
        if layer == layers[0]:
            ci += 1
        chunks[ci][layer] = set(ex)
    n_chunks = ci + 1
    total_demand = sum(len(ex) for _, _, ex in pp)
    uniq = len({(l, e) for _, l, ex in pp for e in ex})
    print('prefill: %d calls, %d layers, %d chunks, %d demanded, %d distinct (layer,expert)'
          % (calls, n_layers, n_chunks, total_demand, uniq))
    if n_chunks > 1:
        ov = []
        for c in range(1, n_chunks):
            for L in chunks[c]:
                if L in chunks[c - 1]:
                    a, b = chunks[c][L], chunks[c - 1][L]
                    if a:
                        ov.append(len(a & b) / float(len(a)))
        print('same layer, previous chunk overlap: %.1f%% (n=%d layer-pairs)'
              % (100.0 * sum(ov) / len(ov), len(ov)))
        sizes = [len(s) for c in chunks.values() for s in c.values()]
        print('uniques per (chunk, layer): mean %.0f  min %d  max %d'
              % (sum(sizes) / len(sizes), min(sizes), max(sizes)))
    return total_demand

def seq_of(pp):
    seq = []
    for call, layer, ex in pp:
        for e in ex:
            seq.append((layer, e))
    return seq

def sim_lru(seq, slots, mru=False):
    used = collections.OrderedDict()
    hits = misses = 0
    for k in seq:
        if k in used:
            hits += 1
            used.move_to_end(k)
        else:
            misses += 1
            if len(used) >= slots:
                # OrderedDict front = least recently used (hits move_to_end).
                # last=False pops the front = LRU; last=True pops the newest =
                # MRU. The first version had these inverted, which swapped the
                # LRU and MRU rows and briefly made MRU look like the winner.
                used.popitem(last=mru)
            used[k] = 1
    return hits, misses

def sim_freeze(seq, slots):
    resident = set()
    hits = misses = 0
    for k in seq:
        if k in resident:
            hits += 1
        else:
            misses += 1
            if len(resident) < slots:
                resident.add(k)
    return hits, misses

def sim_belady(seq, slots):
    n = len(seq)
    nxt = [n] * n
    last = {}
    for i in range(n - 1, -1, -1):
        nxt[i] = last.get(seq[i], n)
        last[seq[i]] = i
    resident, heap = {}, []
    hits = misses = 0
    for i, k in enumerate(seq):
        if k in resident:
            hits += 1
        else:
            misses += 1
            if len(resident) >= slots:
                while heap:
                    neg, vk = heapq.heappop(heap)
                    if vk in resident and resident[vk] == -neg:
                        del resident[vk]
                        break
        resident[k] = nxt[i]
        heapq.heappush(heap, (-nxt[i], k))
    return hits, misses

if __name__ == '__main__':
    path = sys.argv[1]
    slots = int(sys.argv[2]) if len(sys.argv) > 2 else 3977
    expect = sys.argv[3] if len(sys.argv) > 3 else None
    pp, dec = load_pp(path)
    if not pp:
        print('no prefill lines in', path)
        sys.exit(1)
    total = pp_stats(pp)
    seq = seq_of(pp)
    print()
    print('%-16s %8s %9s %11s' % ('policy', 'hit%', 'misses', 'fetch GiB'))
    rows = [
        ('LRU', lambda: sim_lru(seq, slots)),
        ('MRU', lambda: sim_lru(seq, slots, mru=True)),
        ('freeze', lambda: sim_freeze(seq, slots)),
        ('LRU +reserve', lambda: sim_lru(seq, slots + 768)),
        ('Belady', lambda: sim_belady(seq, slots)),
    ]
    for name, fn in rows:
        h, m = fn()
        print('%-16s %7.2f%% %8d %10.1f' % (name, 100.0 * h / total, m, m * 18.98 / 1024.0))
    if expect:
        print()
        print('measured on hardware for this workload: %s' % expect)
