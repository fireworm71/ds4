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
| vivo (`192.168.0.248`, Ryzen 5 5600X, 62 GB, RTX 3090, Ubuntu 24.04) | tier peer candidate for rank 1 | `10.99.4.2/30` to promax, 100G RoCEv2, **up and measured** |

Final link map, after the 2026-10-04 bring-up:

    spark-0fb3 cage p1  <--200G 4X-->  promax cage p1    10.99.0.1/.2 + 10.99.2.1/.2
    vivo CX6 (wide end) <--100G 2X-->  promax cage p0    10.99.4.1/.2
    vivo CX6 (branch B) <---dark---->  spark-0fb3 cage p0  (see §2.2)

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

### 2.2 First bring-up: vivo is cabled to promax, not to the Spark

**The asymmetry below is not a PHY problem. The cable runs from vivo to
*promax*.** promax's own cage p0 is up at 100000 Mb/s (`rocep1s0f0` and
`roceP2p1s0f0` both `PORT_ACTIVE`, 100 Gb/s 2X HDR) since `Oct 03 19:40:22`,
and an ICMPv6 link-local multicast ping out of promax's `enp1s0f0np0` is
answered by `fe80::202:c9ff:fe01:7513` in 0.787 ms — which is exactly vivo's
ConnectX-6 GID. vivo then shows `rx_packets` and a neighbour entry for
promax's MAC `fc:4c:ea:f9:49:3d`. The link is healthy and carries frames; it
simply has the wrong box on the far end for the plan as written.

**The cable is a bifurcated one, and vivo holds its wide end** (Jason,
2026-10-04): the two 100G branches go into the Spark and into promax. A
200G-to-2x100G breakout DAC wires lanes 0-1 of the wide end to one branch and
lanes 2-3 to the other, and it is passive copper, so the wiring is

    vivo ConnectX-6 ==(200G end)==>  branch A --> promax cage p0  (up, 100G)
                                     branch B --> Spark cage p0   (dark)

vivo's card is a **ConnectX-6 VPI, HDR IB (200Gb/s) and 200GbE, single-port
QSFP56, PCIe4.0 x16** (its own `mstconfig` description), and `ethtool` lists
`200000baseCR4` among its supported modes -- a real four-lane port. But one
physical port trains **one** link: it came up at `active_width 2X /
active_speed 50 Gbps` on the two lanes going to promax and leaves the other two
dark. Lighting both branches at once would need a 2x100G port split, and
`mstconfig -d 0000:05:00.0 q` offers no split knob on this card and firmware
(only `MULTI_PORT_VHCA_EN`, `PORT_OWNER` and friends). **So the intended
vivo-spark-promax triangle is not available: vivo talks to exactly one Spark at
a time, and today that is promax.**

The lane counts confirm it without touching anything: the Spark-promax cable
reports `200 Gb/sec (4X HDR)` on p1 at both ends — four lanes — while promax's
p0 and vivo both report `100 Gb/sec (2X HDR)`, two lanes. The Spark's cage p0
therefore sees a module and no signal whatsoever: every counter zero, "No
partner detected". Nothing is faulty and nothing needs re-seating.

The consequence matters more than the diagnosis: **the Spark cannot reach vivo
through this cable, at any speed**, and no amount of configuration here changes
that — the lanes it would need are physically terminated in a port that is
already using the other two for promax.

If the tier must sit on rank 0 instead, move vivo's wide end so that branch A
lands in the Spark's cage p0 — promax then loses the vivo link, since only one
branch of a breakout can be live. Neither is needed for the measurement;
§2.4 is.

**The upgrade worth knowing about**: vivo's port is HDR 200G and its slot
carries 126.016 Gb/s (Gen3 x16, and the card is a PCIe4.0 x16 part). Replacing
the breakout with a straight 200G QSFP56 DAC into one Spark cage would run the
leg at the PCIe ceiling, **~15.7 GB/s instead of today's 12.26** — +28%, and
1.87x the Spark's local NVMe. If the slot can also be trained at Gen4 the
ceiling moves to the 200G wire, ~25 GB/s. That is the single cheapest way to
make this path meaningfully faster, and it costs one cable.

For the record, the kernel logs read as that cable going in at 18:47 on Oct 03
(both the Spark's and promax's ends, 14 s apart), promax's end training at
18:51 and dropping at 18:58, then coming up at 100G at 19:40 once vivo's card
was on branch A.

**My first reading of this was wrong.** I called it a lane/rate mismatch — a
200G 4-lane cage against a 100G 2-lane card, with vivo locking onto 2 lanes of
garbage — because vivo's PHY counters were climbing (24,341 frames,
370 million corrected bits) while the Spark's stayed at zero. Those frames were
promax talking to vivo, and vivo's `rx_corrected_bits_phy` is a lifetime FEC
counter that includes the mis-trained window on Oct 03; `rx_symbol_err_phy` was
0 the whole time, which should have stopped me. Do not force link speeds and do
not buy a different cable on the strength of that paragraph. The one test that
settles which boxes a cable joins is a link-local multicast ping out of the
interface, which needs no root at either end.

### 2.3 What the counters said (kept for the signature)

Both scripts ran 2026-10-04 17:5x. Addresses, MTU 9000 and the RoCEv2 GID index
(3, on both ends) are in place. The link is not:

| | spark-0fb3 `enp1s0f0np0` | vivo `enp5s0np0` |
|---|---|---|
| ethtool | `Link detected: no (Autoneg, No partner detected)` | **`Link detected: yes`, 100000 Mb/s** |
| verbs port | `DOWN / phys Disabled` | **`ACTIVE / phys LinkUp`, 100 Gb/s (2X HDR)** |
| `rx_packets_phy` | 0 | **24,341** |
| `rx_packets` (to the stack) | 0 | 0 |
| `rx_corrected_bits_phy` | 0 | **370,387,351** |

### 2.4 Fixing it: move the address, not the cable

The cheapest repair touches no hardware. vivo keeps everything it has
(`10.99.4.2/30`, MTU 9000, GID index 3); the `.1` end of the /30 moves from the
Spark's dead cage to promax's live one, and the tier then serves **rank 1**
instead of rank 0 — which costs nothing, because the two ranks stream
symmetric halves of the experts.

    # spark-0fb3: the address is on a port with nothing on the other end
    sudo ip addr del 10.99.4.1/30 dev enp1s0f0np0

    # promax: adopt it on the cage that actually reaches vivo
    sudo ./rdma_link_setup.sh --iface enp1s0f0np0 --addr 10.99.4.1/30 --peer 10.99.4.2

The alternative is physical and, because the cable is a breakout (§2.2), it
costs promax the vivo link: move the 200G end from promax's cage p0 into the
Spark's, leave vivo on branch A, and everything already configured here works
unchanged with the tier on rank 0. A straight 100G DAC from the Spark's p0 to
vivo does the same without taking anything away. Worth it only if rank 0
specifically matters.

### 2.5 Two more things the bring-up turned up

* **The Spark has no TP addresses.** promax still holds `10.99.0.2/30` and
  `10.99.2.2/30`, but `10.99.0.1` and `10.99.2.1` are absent on this box —
  lost in the 2026-09-21 reboot along with the MTU. TP2 cannot run over RDMA
  until they are back:

      sudo ip addr replace 10.99.0.1/30 dev enp1s0f1np1
      sudo ip addr replace 10.99.2.1/30 dev enP2p1s0f1np1

* **The MTU fix is only half applied, and parity is mandatory.** The Spark's p1
  netdevs are at 9000 and report `active_mtu 4096`; *all four* of promax's are
  still at 1500, so every port there reports 1024. This is not a performance
  footnote: `ds4_rdma_tier.c:398` and `t1_common.h:223` both program
  `path_mtu` from their **own** port, and the handshake carries no MTU field,
  so a 4096 end talking to a 1024 end does not negotiate down — it fails at
  the RTR transition. Measured today: `ib_read_bw` between promax (1024) and
  vivo (4096) died with "Failed to modify QP to RTR / Unable to Connect the
  HCA's through the link", and the same run with `-m 1024` on both ends worked.
  promax needs MTU 9000 on `enp1s0f1np1`, `enP2p1s0f1np1` and `enp1s0f0np0`.

### 2.6 The leg, measured twice: 91.60 then 98.05 Gb/s

`ib_read_bw` promax -> vivo (the direction the tier reads in), 1 MiB messages,
4 QPs:

| arm | GIDs | path MTU | BW average |
|---|---|---|---|
| before the MTU repair | RoCE v1 link-local, no IPv4 yet | 1024 | 91.60 Gb/s |
| after §2.5 was applied | **RoCE v2, IPv4, index 3 both ends** | **4096** | **98.05 Gb/s** |

**12.26 GB/s on a 100G link — 98% of line rate.** The MTU parity fix is worth
+7% on its own. Against the Spark's ~8.4 GB/s local NVMe that is **1.46x**,
which lands where §3.2 predicted and settles that there is no PHY, PCIe or
cable problem anywhere in this path. What the tier is worth is now purely a
question about hit rate and the `max(rank0, rank1)` gate, not about the fabric.

### 2.7 The tier itself: proven end to end, byte-exact

`ds4_region_server --single` on vivo (x86_64, built from this branch) serving a
512 MiB file, `test_rdma_tier` on promax over the live leg, same file
byte-identical on both ends (`sha256 3a701333...`):

    ds4: rdma expert tier active: 0.50 GiB from 10.99.4.2, 1 leg x 8 pairs, verified
    -- single-threaded, 16 offsets --
       unaligned offset + odd length: OK
       byte-exact
    -- 8 threads x 12 spans --
       113 reads, 1.05 GiB, 11.73 GB/s aggregate, 4042.0 us/span mean
    ALL CHECKS PASSED

So the `--single` patch works on the wire and not only at startup: the client
accepts a one-leg peer, verifies it, and reads byte-exact at unaligned offsets
and odd lengths from eight threads at **11.73 GB/s aggregate — 96% of the
`ib_read_bw` ceiling**. Everything between a region of RAM on vivo and ds4's
`cuda_model_stage_read()` is now demonstrated except ds4 itself.

One footnote for anyone adding a second leg later: vivo's ConnectX-6 reports
`node_guid 0x0000000000000000`. With two such devices the client's T0-trap
check would see two equal GUIDs and drop to one leg -- correctly by its own
logic, wrongly in fact. Single leg is unaffected; nothing compares.

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
3. On the rank that is actually cabled to vivo — **promax, rank 1**, as §2.2
   establishes: `DS4_EXPERT_TIER_RDMA=10.99.4.2:19515
   DS4_EXPERT_TIER_RDMA_GID=<idx> DS4_FETCH_STATS=1 ./ds4 ...` plus the usual
   Q4 TP2 streaming flags, on the worker command line. The tier verifies five
   sampled offsets against ds4's own model fd before it activates, so a wrong
   or over-large prefix file refuses rather than lies.
4. Read the stats out of the worker's log. Decide from the nanoseconds, not the
   bytes.

Two things to know before spending a day on it. Promotion of uncovered spans
was already measured null ("every span that reaches disk is a first touch,
because the expert cache already absorbs the reuse"). And only **one** rank can
have the tier: vivo's card is single-port, so one cable, one box. Giving the
other rank a leg too would need a second port in vivo plus a cable into the
remaining free cage, and then a leg-independent mode in the region server (one
region, two devices, each serving single-leg clients; it currently insists one
client connect both legs). An asymmetric pair is also a measurement hazard —
accelerate one rank's fetch path and the other becomes the straggler that sets
the token time, which caps any gain at roughly what the slower rank still
spends on disk.

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
2. ~~Scripts on both ends~~ — done; addresses, MTU 9000 and GID index 3 are in
   place, and the link turned out to land on promax (§2.2).
3. ~~Repair the addressing, the TP addresses and promax's MTU~~ — done
   2026-10-04. promax holds `10.99.4.1/30` on the vivo leg, this box has
   `10.99.0.1`/`10.99.2.1` back, and every port on both Sparks and vivo now
   reports `active_mtu 4096`.
4. ~~`ib_read_bw` and `test_rdma_tier` over the live leg~~ — done: 98.05 Gb/s
   and ALL CHECKS PASSED, §2.6 and §2.7.
5. §3.2 only if the prefill question is worth a day. It is a one-rank,
   10%-coverage, 1.4x change behind a `max(rank0, rank1)` gate, so the decode
   arm can be predicted at zero and skipped.
6. §3.4 whenever vivo's nvidia driver is fixed — unblocks real work and needs
   nothing from the fabric.

Meanwhile the link is already worth having for what it moves: artifacts between
promax and vivo at ~11 GB/s instead of the 1 GbE's ~110 MB/s. A 483 GiB model
is 12 hours over the LAN and 12 minutes over this.

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
