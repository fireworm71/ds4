"""Warm-start experiments: what is the initial cache state worth?

Arms, all replayed on the captured decode trace, scored from token 0 because
the transient IS the object of study (the +2.0pp gate work scored past a
warmup; these numbers are therefore not comparable to that table):

  cold          empty cache (worst case; hardware actually starts with the
                prefill tail, so reality sits between cold and final-state)
  final-state   "cache the final result": initial state = the resident set at
                the END of this same trace, with its final recency order.
                Models persisting the cache across a restart / across turns,
                as an optimistic bound (self-overlap >= next-turn overlap).
  oracle-first  "oracle of the response": preload the 3,977 keys the response
                uses EARLIEST (soonest-needed kept most-recent so LRU does not
                evict them before use). LRU thereafter.
  oracle-freq   same, but top 3,977 by total use count.
  oracle+Belady oracle-first initial AND Belady management: the true ceiling
                of any warm start bounded by the slot budget.

Also prints the RAM arithmetic for "hold the whole response's working set".
"""
import sys, collections, heapq
sys.path.insert(0, '/home/jason/.claude/jobs/e839ef4c/tmp')
sys.path.insert(0, '/home/jason/ds4-41flash/.claude/worktrees/ds41f-spark/gguf-tools/analysis')
from policysim import load

def flatten_tok(trace):
    seq = []
    for token, layer, experts in trace:
        for e in dict.fromkeys(experts):
            seq.append((token, (layer, e)))
    return seq

def run_lru(seq, slots, initial=None, transient_tokens=50):
    """initial: iterable of keys, FIRST = FIRST EVICTED. Scores everything."""
    used = collections.OrderedDict()
    if initial:
        for k in initial:
            used[k] = 1
    hits = misses = 0
    early_miss = late_miss = 0
    t0 = seq[0][0]
    for token, k in seq:
        early = (token - t0) < transient_tokens
        if k in used:
            hits += 1
            used.move_to_end(k)
        else:
            misses += 1
            if early: early_miss += 1
            else: late_miss += 1
            if len(used) >= slots:
                used.popitem(last=False)
            used[k] = 1
    return hits, misses, early_miss, late_miss, list(used.keys())

def run_belady(seq, slots, initial=None):
    keys = [k for _, k in seq]
    n = len(keys)
    nxt = [n] * n
    last = {}
    for i in range(n - 1, -1, -1):
        nxt[i] = last.get(keys[i], n)
        last[keys[i]] = i
    first_use = last                       # after the loop: first occurrence
    resident, heap = {}, []
    if initial:
        for k in initial:
            fu = first_use.get(k, n)
            resident[k] = fu
            heapq.heappush(heap, (-fu, k))
    hits = misses = 0
    for i, k in enumerate(keys):
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
    trace = load(path)
    seq = flatten_tok(trace)
    total = len(seq)
    tokens = len({t for t, _ in seq})

    first_seen = {}
    counts = collections.Counter()
    for i, (t, k) in enumerate(seq):
        if k not in first_seen:
            first_seen[k] = i
        counts[k] += 1
    uniques = len(first_seen)

    print('=== the RAM arithmetic first ===')
    for name, mib, budget_slots, budget_gib in (
            ('Q4', 18.98, 3977, 73.73), ('Q2', 9.49, 8723, 80.86)):
        ws = uniques * mib / 1024.0
        print('%s: response working set %d experts x %.2f MiB = %.1f GiB '
              '(cache %.1f GiB / %d slots; box RAM 121 GiB)'
              % (name, uniques, mib, ws, budget_gib, budget_slots))
    print('The %d-token response touches %d distinct (layer,expert) keys;' % (tokens, uniques))
    print('%.1fx the Q4 slot budget and 1.6x the MACHINE at Q4 size.' % (uniques / slots))

    print()
    print('=== warm-start arms, LRU management unless stated (scored from token 0) ===')
    print('%-26s %8s %9s %12s %12s' % ('arm', 'hit%', 'misses', 'first-50-tok', 'later'))

    h, m, em, lm, final_state = run_lru(seq, slots)
    base_m = m
    print('%-26s %7.2f%% %8d %12d %12d' % ('cold', 100.0 * h / total, m, em, lm))

    h, m, em, lm, _ = run_lru(seq, slots, initial=final_state)
    print('%-26s %7.2f%% %8d %12d %12d' % ('final-state (persist)', 100.0 * h / total, m, em, lm))

    by_first = sorted(first_seen, key=first_seen.get)
    preload_first = by_first[:slots]
    h, m, em, lm, _ = run_lru(seq, slots, initial=list(reversed(preload_first)))
    print('%-26s %7.2f%% %8d %12d %12d' % ('oracle-first-use', 100.0 * h / total, m, em, lm))

    preload_freq = [k for k, _ in counts.most_common(slots)]
    h, m, em, lm, _ = run_lru(seq, slots, initial=list(reversed(preload_freq)))
    print('%-26s %7.2f%% %8d %12d %12d' % ('oracle-most-used', 100.0 * h / total, m, em, lm))

    h, m = run_belady(seq, slots)
    print('%-26s %7.2f%% %8d %12s %12s' % ('Belady cold (ref)', 100.0 * h / total, m, '-', '-'))
    h, m = run_belady(seq, slots, initial=preload_first)
    print('%-26s %7.2f%% %8d %12s %12s' % ('oracle + Belady (ceiling)', 100.0 * h / total, m, '-', '-'))

    print()
    print('deltas are against cold = %d misses.' % base_m)
