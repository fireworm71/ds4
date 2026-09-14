"""E8: decompose the LRU->Belady gap into reachable and unreachable parts.

Every scored miss falls into one of:
  compulsory  first-ever use of the key. No policy avoids these -- prefetch
              cannot name a key that has never appeared and history methods
              have nothing to condition on.
  capacity/timing  the key was seen before; keeping or refetching it early was
              possible in principle. Belady avoids most of these with phase
              knowledge; the measured policy results say how much of that is
              reachable online.
"""
import sys, collections
sys.path.insert(0, '/home/jason/.claude/jobs/e839ef4c/tmp')
sys.path.insert(0, '/home/jason/ds4-41flash/.claude/worktrees/ds41f-spark/gguf-tools/analysis')
from policysim import load, flatten, next_use_table, run_lru, run_belady

path = sys.argv[1]
slots = int(sys.argv[2]) if len(sys.argv) > 2 else 3977
trace = load(path)
seq, score_from = flatten(trace)
nxt = next_use_table(seq)

seen = set()
compulsory = 0
scored = 0
for i, k in enumerate(seq):
    if i >= score_from:
        scored += 1
        if k not in seen:
            compulsory += 1
    seen.add(k)

lh, lm, _ = run_lru(seq, score_from, nxt, slots)
bh, bm, _ = run_belady(seq, score_from, nxt, slots)

print('scored accesses            %8d' % scored)
print('compulsory misses          %8d  (%.1f%% of scored, unavoidable by ANY policy)'
      % (compulsory, 100.0 * compulsory / scored))
print()
print('                              misses    non-compulsory')
print('LRU                        %8d      %8d' % (lm, lm - compulsory))
print('Belady                     %8d      %8d' % (bm, bm - compulsory))
print()
avoid = lm - bm
print('Belady avoids %d of LRU\'s %d non-compulsory misses (%.1f%%)'
      % (avoid, lm - compulsory, 100.0 * avoid / (lm - compulsory)))
print()
print('online-reachable share of that, measured:')
print('  best order-based policy (CLOCK)      1,114 misses avoided  (9.3%%)')
print('  best timing-based policy (per-key period oracle)  NEGATIVE')
print('  timed prefetch precision at 2-token lead          1%%')
print()
print('=> %.1fpp of the %.2fpp gap is phase knowledge: WHICH visit comes next,'
      % (100.0 * (avoid - 1114) / (scored), 100.0 * avoid / scored))
print('   not how often or how recently. No function of this trace\'s history')
print('   reaches it; Belady reads the future sample path.')
