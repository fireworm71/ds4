"""Would cross-chunk prefetch hide deep-prefill's fetch stall?

Prefill's fetch is serial with compute (measured: deep wall 216 s = ~175 s
compute + 40.9 s load; the decode-side async overlap machinery does not run in
prefill). During each layer call's ~1.2 s of compute the disk is idle, and the
NEXT layer's demand is predictable across chunks: chunk C's layer-L set
overlaps chunk C-1's by a measured 87.4%.

So: replay the deep trace under the engine's (bit-exact-validated) LRU, and
for every miss ask whether it could have been staged during the previous
call's compute window -- i.e. whether the key was in the previous chunk's set
for that layer. Bandwidth check included: hideable bytes per call must fit the
window (compute_per_call x disk rate), and the unpredictable remainder stays
stalled.

This changes NO cache decision -- the same keys load at the same begin_load
points, so output would be byte-identical; the only difference is whether the
pread happens during the stall or during the preceding compute.
"""
import sys, collections
sys.path.insert(0, '/home/jason/.claude/jobs/e839ef4c/tmp')
from prefillsim import load_pp

EXPERT_MIB = 18.98
DISK_GIBS = 10.4          # measured effective prefill load rate
COMPUTE_S = 1.23          # (216 s - 40.9 s) / 142 calls
LOAD_WALL_S = 40.9
TOTAL_WALL_S = 216.0

path = sys.argv[1]
slots = int(sys.argv[2]) if len(sys.argv) > 2 else 3977
pp, _ = load_pp(path)

layer0 = min(l for _, l, _ in pp)
prev_chunk = {}
cur_chunk = {}
used = collections.OrderedDict()
stall_b = hide_b = 0.0
window_b = COMPUTE_S * DISK_GIBS * 1024.0   # MiB stageable per call
n_hits = n_miss = n_hidden = 0

for call, layer, ex in pp:
    if layer == layer0 and cur_chunk:
        prev_chunk = cur_chunk
        cur_chunk = {}
    cur_chunk[layer] = set(ex)
    pred = prev_chunk.get(layer, set())
    call_hide = 0.0
    for e in ex:
        k = (layer, e)
        if k in used:
            n_hits += 1
            used.move_to_end(k)
            continue
        n_miss += 1
        if e in pred and call_hide + EXPERT_MIB <= window_b:
            call_hide += EXPERT_MIB
            n_hidden += 1
            hide_b += EXPERT_MIB
        else:
            stall_b += EXPERT_MIB
        if len(used) >= slots:
            used.popitem(last=False)
        used[k] = 1

total_b = stall_b + hide_b
print('misses %d (validates: engine measured 22,906)' % n_miss)
print('fetched %.1f GiB total; hidden %.1f GiB (%d misses, %.1f%%), still stalled %.1f GiB'
      % (total_b / 1024, hide_b / 1024, n_hidden, 100.0 * n_hidden / n_miss, stall_b / 1024))
new_load = LOAD_WALL_S * (stall_b / total_b)
print()
print('stalled load: %.1f s -> %.1f s' % (LOAD_WALL_S, new_load))
print('deep prefill wall: %.0f s -> %.0f s  (%.1f%% faster, %.0f -> %.0f t/s)'
      % (TOTAL_WALL_S, TOTAL_WALL_S - LOAD_WALL_S + new_load,
         100.0 * (LOAD_WALL_S - new_load) / TOTAL_WALL_S,
         30474 / TOTAL_WALL_S, 30474 / (TOTAL_WALL_S - LOAD_WALL_S + new_load)))
print()
print('per-call staging budget %.1f GiB vs worst-case need %.1f GiB'
      % (window_b / 1024, max(len(ex) for _, _, ex in pp) * EXPERT_MIB / 1024))
