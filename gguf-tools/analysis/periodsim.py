"""Test the PERIODICITY hypothesis: expert demand is timed, not just ordered.

Every policy in the null sweep ranked by ORDER statistics -- recency, frequency,
Kth-recency. Belady ranks by TIME: when does this key return. Jason's
hypothesis is that (layer, expert) demand is periodic, in which case an online
policy can estimate each key's return time from its own interarrival history,
evict the key whose predicted return is furthest, and prefetch keys whose
predicted return is imminent.

If the hypothesis is right, LRU's failure mode is specific: among quiet keys a
periodic process means the LONGEST-absent key is the CLOSEST to returning, and
LRU evicts exactly that one first. That would explain the 65-256-token miss
band and why LRU-2 (even more aggressive against old keys) went negative.

Three parts, in order of decisiveness:
  1. MEASURE the hazard function from the raw trace, unconditioned on any
     cache: P(key returns at age a | absent until a). Rising hazard beyond the
     burst = periodic, anti-LRU is right post-burst, the idea lives.
     Falling hazard = heavy-tailed, LRU is the right shape, the idea dies here.
  2. Per-key interarrival regularity (CV) and the miss-predictability budget:
     what share of misses even HAVE enough history to predict from.
  3. Policies against the pre-registered +2.0pp gate:
       period-oracle  T = key's true mean interarrival over the whole trace
                      (phase-free future knowledge: the CEILING for every
                      period-estimating method; if this fails, they all do)
       period-online  T = mean of gaps observed so far
       each with and without timed prefetch (budget/token, 2-token lead).
"""
import sys, collections, heapq, math
sys.path.insert(0, '/home/jason/.claude/jobs/e839ef4c/tmp')
sys.path.insert(0, '/home/jason/ds4-41flash/.claude/worktrees/ds41f-spark/gguf-tools/analysis')
from policysim import load, WARMUP

BIG = float('inf')

def flatten_tok(trace):
    """[(token, key)] deduped within a (token, layer) row, plus score_from."""
    seq, score_from = [], 0
    for token, layer, experts in trace:
        for e in dict.fromkeys(experts):
            if token <= WARMUP:
                score_from = len(seq) + 1
            seq.append((token, (layer, e)))
    return seq, score_from

# ---------------------------------------------------------------- part 1+2
def demand_stats(seq):
    occ = collections.defaultdict(list)          # key -> [token, ...]
    for token, k in seq:
        if not occ[k] or occ[k][-1] != token:
            occ[k].append(token)
    gaps = []
    per_key_gaps = {}
    for k, ts in occ.items():
        g = [b - a for a, b in zip(ts, ts[1:])]
        if g:
            per_key_gaps[k] = g
            gaps.extend(g)

    print('=== 1. hazard of return, from the raw trace (no cache involved) ===')
    print('keys %d, reuse gaps %d, gap mean %.1f median %d tokens'
          % (len(occ), len(gaps), sum(gaps) / len(gaps),
             sorted(gaps)[len(gaps) // 2]))
    buckets = [(1, 2), (3, 4), (5, 8), (9, 16), (17, 32), (33, 64),
               (65, 128), (129, 256), (257, 10 ** 9)]
    hist = collections.Counter()
    for g in gaps:
        for lo, hi in buckets:
            if lo <= g <= hi:
                hist[(lo, hi)] += 1
                break
    total = len(gaps)
    surv = total
    print('%12s %9s %10s %12s %14s' % ('age', 'returns', 'share', 'bucket haz', 'PER-TOKEN haz'))
    for lo, hi in buckets:
        n = hist[(lo, hi)]
        h = n / surv if surv else 0.0
        width = min(hi, 512) - lo + 1
        # Buckets double in width, so the bucket hazard rises even for a
        # memoryless process. The per-token rate is the statistic that decides
        # periodic-vs-heavy-tailed; reading the unnormalised column nearly
        # produced the wrong verdict here.
        pt = 1.0 - (1.0 - h) ** (1.0 / width) if h < 1.0 else 1.0
        label = '%d-%d' % (lo, hi) if hi < 10 ** 9 else '>%d' % (lo - 1)
        print('%12s %9d %9.1f%% %11.1f%% %13.2f%%'
              % (label, n, 100.0 * n / total, 100.0 * h, 100.0 * pt))
        surv -= n
    print('PER-TOKEN hazard rising past the burst = periodic (anti-LRU right).')
    print('Falling = heavy-tailed: the longer absent, the less likely to')
    print('return per token, so LRU evicting the oldest is the right shape.')

    print()
    print('=== 2. per-key regularity and the predictability budget ===')
    cvs = []
    for k, g in per_key_gaps.items():
        if len(g) >= 3:
            m = sum(g) / len(g)
            var = sum((x - m) ** 2 for x in g) / len(g)
            cvs.append((math.sqrt(var) / m if m else 0.0, len(g)))
    if cvs:
        w = sum(n for _, n in cvs)
        wm = sum(cv * n for cv, n in cvs) / w
        lo = sum(n for cv, n in cvs if cv < 0.4)
        mid = sum(n for cv, n in cvs if 0.4 <= cv < 0.8)
        hi = sum(n for cv, n in cvs if cv >= 0.8)
        print('keys with >=3 gaps: %d (weighted by gap count: CV<0.4 %.0f%%, '
              '0.4-0.8 %.0f%%, >=0.8 %.0f%%; weighted mean CV %.2f)'
              % (len(cvs), 100.0 * lo / w, 100.0 * mid / w, 100.0 * hi / w, wm))
        print('CV ~0 = clockwork, ~1 = Poisson/random, >1 = bursty.')
    return occ

# ---------------------------------------------------------------- part 3
def run_period(seq, score_from, slots, oracle_T=None,
               prefetch_budget=0, lead_tokens=2, fallback='inf'):
    """Evict max predicted-next-use; predicted = last_use + T_hat (tokens).
    T_hat: oracle_T[key] if given, else mean of gaps seen so far.
    First-sight keys: T_hat = inf under fallback='inf' (evict-first, matching
    the 61%-never-reused population).
    Optional prefetch: at each token boundary, insert up to budget absent keys
    whose predicted return lies within lead_tokens, nearest first."""
    resident = {}                 # key -> predicted next use (tokens)
    heap = []                     # (-pred, key, stamp); lazy deletion
    stamp = collections.Counter()
    last = {}
    gsum = collections.Counter()
    gcnt = collections.Counter()
    absent_pred = {}              # evicted/quiet keys -> predicted return
    was_prefetched = set()
    hits = misses = issued = used = 0
    cur_token = None

    def predict(k, now):
        if oracle_T is not None:
            T = oracle_T.get(k)
        elif gcnt[k]:
            T = gsum[k] / gcnt[k]
        else:
            T = None
        if T is None:
            return BIG if fallback == 'inf' else now + 512.0
        return now + T

    def insert(k, pred, now):
        while len(resident) >= slots:
            negp, vk, vs = heapq.heappop(heap)
            if vk in resident and stamp[vk] == vs and resident[vk] == -negp:
                del resident[vk]
                absent_pred[vk] = -negp
                break
        resident[k] = pred
        stamp[k] += 1
        heapq.heappush(heap, (-pred, k, stamp[k]))
        absent_pred.pop(k, None)

    for i, (token, k) in enumerate(seq):
        score = i >= score_from
        if token != cur_token:
            cur_token = token
            if prefetch_budget:
                cands = [(p, ak) for ak, p in absent_pred.items()
                         if token <= p <= token + lead_tokens]
                cands.sort()
                for p, ak in cands[:prefetch_budget]:
                    if score: issued += 1
                    insert(ak, predict(ak, token), token)
                    was_prefetched.add(ak)
        if k in last:
            gsum[k] += token - last[k]
            gcnt[k] += 1
        last[k] = token
        pred = predict(k, token)
        if k in resident:
            if score:
                hits += 1
                if k in was_prefetched:
                    used += 1
            was_prefetched.discard(k)
            resident[k] = pred
            stamp[k] += 1
            heapq.heappush(heap, (-pred, k, stamp[k]))
        else:
            if score: misses += 1
            insert(k, pred, token)
    return hits, misses, issued, used

if __name__ == '__main__':
    path = sys.argv[1]
    slots = int(sys.argv[2]) if len(sys.argv) > 2 else 3977
    trace = load(path)
    seq, score_from = flatten_tok(trace)
    occ = demand_stats(seq)

    oracle_T = {}
    for k, ts in occ.items():
        g = [b - a for a, b in zip(ts, ts[1:])]
        if g:
            oracle_T[k] = sum(g) / len(g)

    print()
    print('=== 3. timed policies vs the pre-registered gate (LRU 87.79%, '
          'Belady 93.77%, gate = LRU +2.0pp) ===')
    print('%-34s %8s %9s %8s %8s %s' %
          ('policy', 'hit%', 'delta pp', 'issued', 'used', 'gate'))
    base = 87.79
    arms = [
        ('period-oracle evict only', dict(oracle_T=oracle_T)),
        ('period-oracle + prefetch b=8', dict(oracle_T=oracle_T, prefetch_budget=8)),
        ('period-oracle + prefetch b=32', dict(oracle_T=oracle_T, prefetch_budget=32)),
        ('period-online evict only', dict()),
        ('period-online + prefetch b=8', dict(prefetch_budget=8)),
        ('period-online + prefetch b=32', dict(prefetch_budget=32)),
    ]
    for name, kw in arms:
        h, m, iss, use = run_period(seq, score_from, slots, **kw)
        r = 100.0 * h / (h + m)
        d = r - base
        prec = ' prec %.0f%%' % (100.0 * use / iss) if iss else ''
        print('%-34s %7.2f%% %+8.2f %8d %8d %s%s' %
              (name, r, d, iss, use, 'PASS' if d >= 2.0 else '', prec))
    print()
    print('period-oracle is the CEILING for every period-based method: it knows')
    print('each key\'s true mean interarrival. If it fails the gate, no online')
    print('period estimator can pass it, and the Belady gap is phase knowledge')
    print('-- WHICH visit happens next, not how often -- unreachable online.')
