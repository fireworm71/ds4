# T32: wiring vivo into the fleet, and what it can actually do for TP2 Q4

Written 2026-10-04 from spark-0fb3. Both boxes' facts below are read out of
`/sys`, the kernel log and `ibv_devinfo` today (vivo over SSH, unprivileged);
the throughput numbers are the earlier T1/T31 measurements. Neither box grants
passwordless root, so the link itself is still unconfigured.

## 1. The fleet as of today

| box | role today | link |
|---|---|---|
| spark-0fb3 (GB10, 121 GB) | TP2 coordinator, rank 0 | cage p1 to promax, 200G, up |
| promaxgb10-493d (GB10) | TP2 worker, rank 1, `10.99.0.2` / `10.99.2.2` | same cable, two PCIe paths |
| vivo (`192.168.0.248`, Ryzen 5 5600X, 62 GB, RTX 3090, Ubuntu 24.04) | nothing yet | 1 GbE LAN; ConnectX-6 cabled, port admin-down |

vivo, measured today: **62 GB RAM** (60 free), 12 threads, one **ConnectX-6
MT28908, single port** (`enp5s0np0` / `mlx5_0`, fw 20.43.8004), port capability
**100 Gb/s (2X HDR)**, card on a **Gen3 x16 link — 126.016 Gb/s, i.e. 15.75
GB/s**, so the wire and not the slot is the limit (the slot is Gen4-capable;
the BIOS trained it at Gen3, which costs nothing at 100G). Storage is one
4 TB Crucial P310: 27 GiB free on `/`, 219 GiB free on `/mnt/data` (exfat).
`memlock` is 7.83 GiB today, which the setup script fixes. The 3090 is
**currently unusable** — `nvidia-driver-580-server-open` is installed but no
nvidia module is loaded, so the DKMS build or a reboot is outstanding. None of
the roles below need the GPU.

Both ends of the new cable are confirmed plugged, from the two kernel logs:
module 0 "Cable plugged" on the Spark's cage p0 (Oct 03 18:47) and on vivo's
ConnectX-6 (Oct 04 02:40, at boot). vivo's netdev has simply never been brought
up — `oper=down`, verbs port `state=1: DOWN phys=3: Disabled` — which is the
whole reason the Spark reports "No partner detected".

The Spark's ConnectX-7 is one chip (`phys_switch_id b40f00000347bb4c`) with two
QSFP cages, each exposed through two PCIe root complexes:

| cage | netdevs | verbs devices | state |
|---|---|---|---|
| p1 | `enp1s0f1np1`, `enP2p1s0f1np1` | `rocep1s0f1`, `roceP2p1s0f1` | **up, 200G** — the promax TP link |
| p0 | `enp1s0f0np0`, `enP2p1s0f0np0` | `rocep1s0f0`, `roceP2p1s0f0` | module plugged, **no partner** |

Two netdevs per cage is why T1 could stripe two legs over one cable and still
get 1.79x: each PCIe path is only 126.028 Gb/s (32 GT/s x4, from the mlx5 probe
line), so one leg cannot fill a 200G port but two can.

Cage p0 took a module at `Oct 03 18:47:44` ("Port module event: module 0, Cable
plugged" on both `0000:01:00.0` and `0002:01:00.0` — one physical event, both
PCIe paths reporting). `ethtool` still says `Link detected: no (Autoneg, No
partner detected)` and the verbs port is `state=1: DOWN phys=3: Disabled`. So
the cable is in the Spark but the far end is not yet alive: expected, since
vivo's card has never been configured.

## 2. The link

Addressing, following the existing convention that the Spark is always `.1`:

| | spark-0fb3 | vivo |
|---|---|---|
| netdev | `enp1s0f0np0` (cage p0) | the ConnectX-6 port with the cable |
| address | `10.99.4.1/30` | `10.99.4.2/30` |
| MTU | 9000 | 9000 |

Both ends must sit in the same /24: `ds4_rdma_tier.c` picks its local device by
matching the peer address against each device's RoCEv2 GID on the same /24
(`rt_open_dev`). `10.99.0.0/30` and `10.99.2.0/30` are the promax legs, hence
`10.99.4.0/30` here.

`scripts/rdma_link_setup.sh` does the root-only half on either end:

    sudo ./scripts/rdma_link_setup.sh --role spark            # on spark-0fb3
    sudo ./scripts/rdma_link_setup.sh --role vivo --install    # on vivo

It brings the link up, sets the MTU, assigns the address, writes the memlock
limit the region server needs, optionally persists the address with netplan,
and then prints the link state, the port state, the GID table with **the
RoCEv2 GID index for this link's IPv4 marked** — do not trust the tools'
default of 3, it is right only by coincidence — and a peer ping.
`--check` runs the read-only half and needs no root.

If carrier stays 0 with the cable in both ends, the likely cause is a 100G
ConnectX-6 facing a 200G-rated cage that will not autoneg: pass
`--speed 100000` on **both** ends.

`scripts/rdma_inventory.sh` needs no root at all and prints the facts section 7
still lists as unknown.

### 2.1 Found while checking: the promax link lost its jumbo MTU

Both halves of the live TP link are at netdev MTU 1500 today, so
`ibv_devinfo` reports `active_mtu: 1024` against `max_mtu: 4096`. T1 was
measured with 4096 (`t1_common.h`: "4096 here, not ds4_tp.c's 1024"), so the
setting did not survive the 2026-09-21 reboot -- nothing persists it. Every
RDMA message on that link is now cut into 4x as many packets as when the
transport was benchmarked. Restoring it is one command per netdev on each
Spark, and promax needs the same:

    sudo ip link set dev enp1s0f1np1  mtu 9000
    sudo ip link set dev enP2p1s0f1np1 mtu 9000

Worth doing before any TP2 Q4 number is taken as a baseline, and worth
re-checking after every reboot until it is persisted (`--persist` writes the
netplan file for the new link; the promax link has no such file yet).

## 3. What vivo can do for TP2 Q4, ranked

Current baseline to beat (T31, `DeepSeek-V4.1-Flash-Q4.gguf`, 518,596,067,328 B
= 483.0 GiB, SSD streaming on both ranks): **prefill 24.13 t/s, decode
6.88 t/s**.

### 3.1 It cannot be a TP rank

TP2 is a 50/50 expert split with exactly one worker, and there is no ratio
knob in the tree. A rank also has to hold its non-expert weights, KV and
expert cache resident, and the pair's measured per-rank planning ceiling is
~100 GiB (100.07 GiB works, 101.86 fails). The 3090 has 24 GB. Separately, `docs/DISTRIBUTED.md` scopes CUDA
network TP to one GPU per rank with matching model/quant layouts, and the two
boxes would be sm_86 against sm_121. This is closed, not a tuning problem.

### 3.2 Remote-RAM expert tier peer (the purpose-built path)

`ds4_rdma_tier.c` + `tests/rdma_t1/ds4_region_server.c` already exist for
exactly this: a peer box holds the model's leading bytes in mlocked RAM and
ds4 fetches spans from it by one-sided RDMA READ, falling through to the model
file on any doubt. The interesting part is that **vivo is the only box that can
play this role while both Sparks are busy being TP ranks** — a rank has no RAM
to spare for someone else's region.

Measured transport, from T1 (spark ↔ promax, Q4-sized 18.98 MiB span): one leg
13.38 GB/s, two legs striped 24.00 GB/s, local NVMe ~8.4 GB/s. vivo gets **one
leg**, because one cable to a one-port card is one remote device, and two QPs on
one remote device split it 50/50 and stripe to nothing (the T0 trap). Its port
is 100G, so the wire caps the leg at 12.5 GB/s and ~11.5 GB/s is the realistic
figure: **~1.4x the Spark's local NVMe** for covered spans.

The honest arithmetic, and it is why this is not a headline number. The tier
covers a prefix `[0, N)` of the file, so the hit rate is about `N / 483 GiB`.
A hot-set region would not do better: promotion already measured null because
"every span that reaches disk is a first touch", i.e. the expert cache absorbs
the reuse and what reaches disk is spread across the file. With decode disk I/O
measured at 8.4 ms/token against a 145 ms/token TP2 Q4 decode budget, the
saving is `8.4 ms x coverage x (1 - 1/speedup)`:

| vivo RAM for the region | coverage | decode saving (estimate) |
|---|---|---|
| 50 GiB (what 62 GB of RAM allows) | 10.4% | ~0.25 ms/token, **~0.2%** |
| 112 GiB (hypothetical) | 23% | ~0.45 ms/token, ~0.3% |

So on decode this is noise. It is worth running only for the prefill arm and
only because the cost is low.

Prefill is the arm worth measuring instead: it is read-heavy (24.13 t/s =
41.4 ms/token of work) and we have never measured what share of it sits in
`cuda_model_stage_read()` on the pair. `DS4_FETCH_STATS=1` answers that
directly — it reports the tier against local disk in bytes *and* nanoseconds,
so a tier that moves many bytes and saves no time shows up as exactly that.

Run order, once the link is up:

1. On vivo, make the prefix file. The region server reads a file's leading
   bytes, so vivo needs only as many bytes as it will hold, not 483 GiB:
   `head -c 48G /path/DeepSeek-V4.1-Flash-Q4.gguf > q4.prefix` — shipped over
   the new link, not the 1 GbE.
2. On vivo: `./ds4_region_server --file q4.prefix --single --dev-a <dev>
   --gid-index <from the setup script> --port-a 19515`
3. On spark-0fb3, rank 0 only:
   `DS4_EXPERT_TIER_RDMA=10.99.4.2:19515 DS4_EXPERT_TIER_RDMA_GID=<idx>
   DS4_FETCH_STATS=1 ./ds4 ...` plus the usual Q4 TP2 streaming flags.
   The tier verifies five sampled offsets against ds4's own model fd before it
   activates, so a wrong or over-large prefix file refuses rather than lies.
4. Read the stats. Decide from the nanoseconds, not the bytes.

Two things to know before spending a day on it: promotion of uncovered spans
was already measured null ("every span that reaches disk is a first touch,
because the expert cache already absorbs the reuse"), and rank 1 cannot reach
vivo at all over RDMA — the Spark's only free cage is the one now pointing at
vivo. If vivo's ConnectX-6 turns out to be dual-port, a second cable into
promax's free cage p0 would give rank 1 its own leg, and the region server
would then need a leg-independent mode (one region, two devices, each serving
single-leg clients; it currently insists a client connect both legs).

### 3.3 Capacity: vivo as storage behind the streaming path

The Spark's root NVMe is at 94% — 226 GiB free with 1,559 GiB of model files
already on it. That, not throughput, is the constraint that bites first: there
is no room for another Q4-sized artifact. vivo over NVMe-oF/RDMA is the pattern
already used with promax's `rambox` target, and its Crucial P310 served over a
100G link would land in the same ballpark as the Spark's local NVMe.

**But vivo has no room either**: 219 GiB free on `/mnt/data` and 27 GiB on `/`,
against 483 GiB for Q4 and 341-369 GiB for the Q2/Q3 artifacts. Nothing
model-sized fits. This option is closed until someone frees ~500 GiB on vivo,
and `/mnt/data` being exfat is a second reason not to put a streamed model
there. It stays listed because the arithmetic changes the moment a disk is
added, not because it is actionable today.

### 3.4 Offload the side work

The 3090 would be a perfectly good box for the things that currently steal the
pair: n-gram corpus builds (`--build-ngram-corpus`), quality scoring runs,
imatrix work. Zero coupling, no protocol risk, and it keeps the pair free for
the measurements that need both ranks. Blocked on vivo's nvidia driver not
loading (§1), which is a separate and much smaller problem than anything else
here.

## 4. The one code change this needed

`ds4_region_server` refused to start on a box with a single RoCE device: it
requires `--dev-a` and `--dev-b` to be distinct, which is right for a Spark
(that guard is what stopped T0's false negative) but makes a one-port
ConnectX-6 unable to serve at all. Added `--single`: serve leg A only, bind one
port, advertise one leg. The two-leg path is untouched and still refuses two
names that resolve to one device, now with a pointer to `--single`.

Verified on spark-0fb3: clean build, single-leg start binds only 19515 and
prints `leg B disabled`, two-leg start is byte-for-byte the same behaviour as
before (both ports bound, distinct devices confirmed), and same-device without
`--single` still dies. **The single-leg data path is not yet exercised on the
wire** — that needs the link up, because the tier client requires a RoCEv2 GID
carrying an IPv4 address on the peer's /24 and no interface here has one yet.

## 5. Order of work

1. ~~Install the key, run the inventory on vivo~~ — done 2026-10-04, §1.
2. `sudo scripts/rdma_link_setup.sh` on both ends; get carrier and a ping.
3. `ib_read_bw` both directions for the real one-leg ceiling.
4. `test_rdma_tier` against a `--single` region server over the live link:
   correctness first, byte-exactness at odd offsets.
5. Then, and only then, the §3.2 measurement with `DS4_FETCH_STATS=1`.
6. In parallel, §3.4 — it needs nothing but SSH.

## 6. Access

A dedicated key was generated on spark-0fb3 for this:
`~/.ssh/id_ed25519_vivo`, with `Host vivo` (192.168.0.248, over the LAN) and
`Host vivo-rdma` (10.99.4.2, once the link is up) in `~/.ssh/config`. Installed
and verified 2026-10-04: `ssh vivo` reaches `jason@vivo`, and `~/vivo-cx6/` on
the Spark is the self-contained bundle shipped there (scripts, region-server
sources, a flat Makefile, this note).

## 7. Answered, and what is left

Everything §7 originally asked is now measured and folded into §1: 62 GB of
RAM, a single-port 100G ConnectX-6 on a Gen3 x16 link, the cable confirmed in
both ends, 219 GiB free on vivo, `jason` is the account. What is left needs
somebody's root password:

1. `sudo rdma_link_setup.sh --role vivo --install` on vivo and
   `--role spark` here, then carrier and a ping.
2. MTU 9000 on the two promax netdevs and on promax itself (§2.1) before any
   baseline is taken.
3. Only if §3.2 is to be tried: re-login after the memlock file lands, then the
   prefix file and the run order in §3.2.
