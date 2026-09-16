# T1: standalone one-sided RDMA READ expert transport (prototype)

Task T1 of the RDMA expert transport plan (`misc/RDMA_EXPERT_TRANSPORT_PLAN.md`
in the working tree; `/misc/` is gitignored, so it is not in the repo).
No ds4 code involved.
`t1_server` registers a region and goes passive; `t1_client` fetches an
expert-sized span by RDMA READ striped across two QPs on two NICs, and
verifies every byte.

    make            # needs libibverbs-dev; builds clean under -Wall -Wextra

## Status

**T1 PASSES.** Measured 2026-09-16, spark-0fb3 ↔ promaxgb10-493d, 200 iters
per arm, byte verification on:

| span | single leg | **striped** | speedup |
|---|---|---|---|
| Q2 9.49 MiB | 756.8 µs (13.15 GB/s) | **433.3 µs (22.96 GB/s)** | 1.75× |
| Q4 18.98 MiB | 1488.0 µs (13.38 GB/s) | **829.1 µs (24.00 GB/s)** | 1.79× |

(medians; min-to-min is 1.80× for both). Against local NVMe that is 2.74× /
2.34×. **ALL BYTES MATCH in all four arms**, striped arms split exactly
50.0/50.0 by per-NIC counter, and single-leg arms leave the idle NIC at 0.00 GB.

Mins land within 0.6% of the `ib_read_bw` figures T0 predicted; medians run
1–5% higher, which is the expected cost of a per-operation QD1 median against
a QD16 throughput average. Full detail in `misc/T1_RESULT.md` (untracked).

## Running it for real

**1. On promax — start the server** (Jason or the promax agent; no sudo needed).
Check the device names there first, they need not match spark's:

    ibv_devices                      # pick the two whose ports are ACTIVE
    ibv_devinfo -d <dev> | grep -E 'state|active_mtu'

    ./t1_server --dev-a <promax_dev_for_10.99.0.2> \
                --dev-b <promax_dev_for_10.99.2.2> \
                --port-a 19515 --port-b 19516 --size $((1024*1024*1024))

It prints both devices' node GUIDs and refuses to start if they are the same.
It serves clients in a loop and outlives each one, so it does not need
restarting between arms (unlike the perftest tools — plan §7).

**2. On spark — run the two arms.** Baseline first, then striped:

    ./t1_client --host-a 10.99.0.2 --host-b 10.99.2.2 --span 9950986 --iters 200 --single
    ./t1_client --host-a 10.99.0.2 --host-b 10.99.2.2 --span 9950986 --iters 200
    ./t1_client --host-a 10.99.0.2 --host-b 10.99.2.2 --span 19901972 --iters 200 --single
    ./t1_client --host-a 10.99.0.2 --host-b 10.99.2.2 --span 19901972 --iters 200

**T1 passes when**, against the T0 measurements:

| span | `--single` should be ≈ | striped should be ≈ |
|---|---|---|
| 9950986 (Q2, 9.49 MiB) | 739 µs | **413 µs** |
| 19901972 (Q4, 18.98 MiB) | 1479 µs | **819 µs** |

and `verify` reads `ALL BYTES MATCH` in every arm. A striped run that lands at
~739/~1479 µs instead — i.e. no better than one NIC — means the two legs are
on one remote device; see below.

## The trap this prototype is built to refuse

Two QPs on different **local** NICs is not enough: they must also land on
different **remote** devices. The remote device is the responder, so one
remote device is one ~13 GB/s sender no matter how many local NICs pull, and
the two legs simply split it 50/50 — striping then measures as worth exactly
nothing. T0 hit this false negative first (plan §6).

Three guards exist, and none of them should be removed:

* the server refuses `--dev-a` == `--dev-b`, and also refuses two device names
  that report the same `node_guid`;
* the client aborts if the two legs' **remote** GUIDs match, before timing
  anything;
* every run prints the per-NIC byte split and flags anything outside 40–60%.

Do not judge the split from throughput alone — read the counters, on both
boxes. promax can read its own unprivileged at
`/sys/class/infiniband/<dev>/ports/1/counters/port_rcv_data` (4-byte words).

## Notes

* `--iters` walks the region so no two iterations read the same bytes
  (methodology rule 2); the region must therefore be ≥ a few spans.
* Byte verification is **on by default** and costs CPU; the reported per-span
  times exclude it, and the aggregate is computed from those times, not wall.
* The handshake carries a version tag — a stale binary on one side fails
  loudly rather than reading a garbage rkey.
* The server `mlock`s the region and warns if that fails; an unpinned region
  makes the numbers lie.
* Handshake structs are exchanged raw. Both boxes are aarch64 LE, which is
  fine for a prototype but is not a wire format.
