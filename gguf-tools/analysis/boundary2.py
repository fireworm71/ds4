"""pp->tg boundary replay with REAL prefill contents (needs a trace captured
with the prefill-inclusive dump).

Arms:
  keep    prefill inserts into the shared LRU, decode continues on it --
          the hardware's current behaviour. Validation arm: its prefill and
          decode hit rates must land near the measured 0.0% / 87.4%.
  bypass  prefill reads without inserting; decode starts cold. The serve-mode
          proposal (DS4_CUDA_PREFILL_NO_CACHE_INSERT): pp stops wiping tg's
          working set. Single-shot cost is what this measures; multi-turn
          benefit needs a multi-turn trace and is NOT claimed here.
  tail-k  prefill inserts only for the LAST k layers -- a middle ground where
          the warm tail decode actually uses survives but early-layer churn is
          skipped. Costs nothing at run time (layer index is known).
"""
import sys, collections
sys.path.insert(0, '/home/jason/.claude/jobs/e839ef4c/tmp')
from prefillsim import load_pp

def run(pp, dec, slots, insert_pp=True, tail_from=None, transient_tokens=50):
    used = collections.OrderedDict()
    pph = ppm = 0
    for call, layer, ex in pp:
        for e in ex:
            k = (layer, e)
            if k in used:
                pph += 1
                used.move_to_end(k)
            else:
                ppm += 1
                take = insert_pp and (tail_from is None or layer >= tail_from)
                if take:
                    if len(used) >= slots:
                        used.popitem(last=False)
                    used[k] = 1
    dh = dm = early = 0
    t0 = dec[0][0] if dec else 0
    for token, layer, ex in dec:
        for e in dict.fromkeys(ex):
            k = (layer, e)
            if k in used:
                dh += 1
                used.move_to_end(k)
            else:
                dm += 1
                if token - t0 < transient_tokens:
                    early += 1
                if len(used) >= slots:
                    used.popitem(last=False)
                used[k] = 1
    return pph, ppm, dh, dm, early

if __name__ == '__main__':
    path = sys.argv[1]
    slots = int(sys.argv[2]) if len(sys.argv) > 2 else 3977
    pp, dec = load_pp(path)
    print('trace: %d prefill calls, %d decode rows' % (len(pp), len(dec)))
    print('%-14s %10s %10s %10s %12s' %
          ('arm', 'pp hit%', 'tg hit%', 'tg misses', 'first-50-tok'))
    arms = [
        ('keep (stock)', dict()),
        ('bypass', dict(insert_pp=False)),
        ('tail-14', dict(tail_from=26)),
        ('tail-27', dict(tail_from=13)),
    ]
    for name, kw in arms:
        pph, ppm, dh, dm, early = run(pp, dec, slots, **kw)
        ppr = 100.0 * pph / (pph + ppm) if pph + ppm else 0.0
        dr = 100.0 * dh / (dh + dm) if dh + dm else 0.0
        print('%-14s %9.2f%% %9.2f%% %10d %12d' % (name, ppr, dr, dm, early))
    print()
    print('validation: keep-arm rates should land near the measured hardware')
    print('numbers for this workload (shallow: pp 0.0%, tg 87.4%).')
