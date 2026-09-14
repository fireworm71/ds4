"""Offline analysis of the per-(token, layer) routing trace.

The important part is the SIMULATOR. Every cache-policy question so far has cost
a 4-minute GPU arm, which is why only a handful of policies have been tried. The
routing trace is the only input a cache policy actually consumes, so given the
trace, any policy can be evaluated offline in seconds.

That is only trustworthy if the simulator reproduces reality first. Step 1
replays plain LRU at the real slot count and must land on the measured 87.4%
decode hit rate for Q4 at 3,977 slots. If it does not, the simulator is wrong
and nothing below it means anything -- so it prints the comparison and says so
rather than proceeding quietly.

Then the two questions worth asking:
  (a) how much does the selected set move between ADJACENT LAYERS within one
      token -- the axis nothing has measured;
  (b) how much does it move for the SAME LAYER across adjacent tokens -- the
      axis the failed predictors used.
"""
import sys, collections, heapq

def load(path):
    trace = []   # (token, layer, [experts])
    for line in open(path):
        f = line.split()
        if len(f) < 3 or f[0] == 'p':
            continue
        trace.append((int(f[0]), int(f[1]), [int(x) for x in f[2:]]))
    return trace

WARMUP_TOKENS = 20   # just past the 3977/240 = 16.6-token cache horizon

def simulate_lru(trace, slots, warmup=WARMUP_TOKENS):
    """Global pool keyed by (layer, expert), exact LRU by access order.

    Accesses during warmup still populate the cache but are not scored. The
    hardware enters decode with a cache already full from prefill, while this
    replay starts empty; without skipping the fill-up the simulator would show
    roughly 2pp worse than reality for reasons that have nothing to do with the
    policy, and the same bias would distort the Belady comparison."""
    # OrderedDict gives O(1) LRU; a min() scan over 3,977 slots on each of
    # ~25k misses is 100M dict probes and turns a seconds-long experiment into
    # a minutes-long one, which defeats the purpose of having a simulator.
    used = collections.OrderedDict()
    hits = misses = 0
    for token, layer, experts in trace:
        score = token > warmup
        for e in dict.fromkeys(experts):     # dedupe, preserve order
            key = (layer, e)
            if key in used:
                if score: hits += 1
                used.move_to_end(key)
            else:
                if score: misses += 1
                if len(used) >= slots:
                    used.popitem(last=False)
                used[key] = 1
    return hits, misses

def simulate_belady(trace, slots):
    """Optimal offline replacement: evict whatever is used furthest in the
    future. No online policy can beat this, so the gap between it and LRU is
    the ENTIRE budget available to cache-policy work -- speculation, eviction
    order, protection, all of it. If that gap is small, every remaining policy
    idea is capped at it and the effort belongs elsewhere."""
    seq = []
    score_from = 0
    for token, layer, experts in trace:
        for e in dict.fromkeys(experts):
            if token <= WARMUP_TOKENS:
                score_from = len(seq) + 1
            seq.append((layer, e))
    # next_use[i] = index of the next access to the same key, or +inf
    next_use = [len(seq)] * len(seq)
    last = {}
    for i in range(len(seq) - 1, -1, -1):
        k = seq[i]
        next_use[i] = last.get(k, len(seq))
        last[k] = i

    # Max-heap on next_use with lazy deletion: entries whose recorded next_use
    # is stale (the key was touched again since) are discarded when popped.
    resident = {}      # key -> current next_use index
    heap = []          # (-next_use, key)
    hits = misses = 0
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
                        break
        resident[k] = next_use[i]
        heapq.heappush(heap, (-next_use[i], k))
    return hits, misses

def overlap_stats(trace):
    by_token = collections.defaultdict(dict)
    for token, layer, experts in trace:
        by_token[token][layer] = set(experts)

    cross_layer = []     # layer L vs layer L-1, same token
    cross_token = []     # layer L at token t vs t-1
    tokens = sorted(by_token)
    for ti, t in enumerate(tokens):
        layers = by_token[t]
        for L in sorted(layers):
            if L - 1 in layers:
                a, b = layers[L], layers[L - 1]
                cross_layer.append(len(a & b) / float(len(a)))
            if ti > 0:
                prev = by_token[tokens[ti - 1]]
                if L in prev:
                    a, b = layers[L], prev[L]
                    cross_token.append(len(a & b) / float(len(a)))
    return cross_layer, cross_token

def mean(v):
    return sum(v) / float(len(v)) if v else float('nan')

if __name__ == '__main__':
    path = sys.argv[1]
    slots = int(sys.argv[2]) if len(sys.argv) > 2 else 3977
    expect = float(sys.argv[3]) if len(sys.argv) > 3 else 87.4
    trace = load(path)
    tokens = len({t for t, _, _ in trace})
    layers = len({l for _, l, _ in trace})
    print('trace: %d rows, %d tokens, %d layers' % (len(trace), tokens, layers))

    hits, misses = simulate_lru(trace, slots)
    rate = 100.0 * hits / (hits + misses)
    print()
    print('=== simulator validation (must match the real run) ===')
    print('LRU @ %d slots: hit %.1f%%  (%d hits, %d misses)' % (slots, rate, hits, misses))
    print('measured on hardware: %.1f%%' % expect)
    if abs(rate - expect) > 1.5:
        print('MISMATCH > 1.5pp -- the simulator does not model the real cache.')
        print('Do not trust any policy comparison until this is reconciled.')
    else:
        print('within 1.5pp: simulator is faithful enough to compare policies')

    bh, bm = simulate_belady(trace, slots)
    brate = 100.0 * bh / (bh + bm)
    print()
    print('=== how much room is there for ANY policy? ===')
    print('LRU      %.1f%%   misses %d' % (rate, misses))
    print('Belady   %.1f%%   misses %d   (optimal, unachievable online)' % (brate, bm))
    print('headroom %.1fpp  -- every policy idea, including speculation, is'
          % (brate - rate))
    print('           capped by this. Decode load is ~42%% of the generation')
    print('           phase, so the ceiling on throughput from cache policy')
    print('           alone is about %.1f%%.'
          % (0.423 * 100.0 * (bm and (misses - bm) / float(misses) or 0.0)))

    cl, ct = overlap_stats(trace)
    print()
    print('=== how much does routing move? (share of a layer\'s experts also in...) ===')
    print('previous LAYER, same token : %.1f%%  (n=%d)' % (100 * mean(cl), len(cl)))
    print('same layer, previous TOKEN : %.1f%%  (n=%d)' % (100 * mean(ct), len(ct)))
