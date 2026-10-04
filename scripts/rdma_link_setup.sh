#!/usr/bin/env bash
# rdma_link_setup.sh -- bring up one direct RoCEv2 leg between two ds4 boxes.
#
# Everything in here needs root, which is why it is a script and not a session:
# link up, MTU, address, the memlock limit the region server mlocks against,
# and (optionally) persistence. The checks at the end need no privileges and
# are also available on their own with --check.
#
#   # on the Spark (coordinator side, free QSFP cage p0):
#   sudo ./scripts/rdma_link_setup.sh --role spark
#
#   # on vivo (the new ConnectX-6 box):
#   sudo ./scripts/rdma_link_setup.sh --role vivo --install
#
#   # verification only, no changes, no root needed:
#   ./scripts/rdma_link_setup.sh --role spark --check
#
# Defaults put the link on 10.99.4.0/30 -- spark .1, vivo .2 -- because
# 10.99.0.0/30 and 10.99.2.0/30 are already the two legs to promax. Both ends
# must share a /24: the ds4 RDMA tier picks its local device by matching the
# peer address against each device's RoCEv2 GID on the same /24.
#
# Options:
#   --role spark|vivo   which end this is (required unless --iface and --addr)
#   --iface NAME        netdev to configure (default: role-dependent/auto)
#   --addr A.B.C.D/P    local address (default: role-dependent)
#   --peer A.B.C.D      peer address to ping (default: the other end)
#   --mtu N             link MTU, default 9000 (RoCE then negotiates 4096)
#   --speed N           force port speed in Mb/s and disable autoneg, e.g.
#                       100000 for a ConnectX-6 that will not autoneg to a
#                       200G-rated cage. Only when the link stays down.
#   --install           apt-get the RDMA userspace + build tools (vivo side)
#   --persist           also write a netplan file so the address survives reboot
#   --no-memlock        skip the /etc/security/limits.d memlock entry
#   --force             configure an interface that already has an IPv4 address,
#                       or one this script otherwise refuses to touch
#   --check             run the verification block only, change nothing

set -uo pipefail

role=""; iface=""; addr=""; peer=""; mtu=9000; speed=""
do_install=0; do_persist=0; do_memlock=1; force=0; check_only=0

while [ $# -gt 0 ]; do
    case "$1" in
        --role) role="${2:?}"; shift 2 ;;
        --iface) iface="${2:?}"; shift 2 ;;
        --addr) addr="${2:?}"; shift 2 ;;
        --peer) peer="${2:?}"; shift 2 ;;
        --mtu) mtu="${2:?}"; shift 2 ;;
        --speed) speed="${2:?}"; shift 2 ;;
        --install) do_install=1; shift ;;
        --persist) do_persist=1; shift ;;
        --no-memlock) do_memlock=0; shift ;;
        --force) force=1; shift ;;
        --check) check_only=1; shift ;;
        -h|--help) sed -n '2,42p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 1 ;;
    esac
done

die()  { echo "rdma-setup: $*" >&2; exit 1; }
note() { echo "rdma-setup: $*"; }
run()  { echo "  + $*"; "$@" || die "failed: $*"; }

case "$role" in
    spark)
        # Cage p0 is the free one: p1 (…f1np1) carries the TP link to promax.
        [ -n "$iface" ] || iface=enp1s0f0np0
        [ -n "$addr" ]  || addr=10.99.4.1/30
        [ -n "$peer" ]  || peer=10.99.4.2
        ;;
    vivo)
        [ -n "$addr" ] || addr=10.99.4.2/30
        [ -n "$peer" ] || peer=10.99.4.1
        ;;
    "")
        [ -n "$iface" ] && [ -n "$addr" ] || die "need --role spark|vivo (or both --iface and --addr)"
        ;;
    *) die "unknown role '$role' (spark|vivo)" ;;
esac

# ---------------------------------------------------------------- interface

# Auto-pick a mlx5 netdev when none was named: prefer one that already has
# carrier, else the only one, else refuse and list the candidates.
if [ -z "$iface" ]; then
    cands=()
    for n in /sys/class/net/*; do
        i=$(basename "$n")
        [ "$(basename "$(readlink -f "$n/device/driver" 2>/dev/null)" 2>/dev/null)" = mlx5_core ] || continue
        [ -d "$n/device/infiniband" ] || continue
        cands+=("$i")
    done
    [ ${#cands[@]} -gt 0 ] || die "no mlx5_core interface with an infiniband device found.
  Is the card seated and the driver loaded?  'lspci -d 15b3:' and 'dmesg | grep mlx5' tell you which."
    for i in "${cands[@]}"; do
        [ "$(cat "/sys/class/net/$i/carrier" 2>/dev/null)" = 1 ] && iface="$i" && break
    done
    if [ -z "$iface" ]; then
        if [ ${#cands[@]} -eq 1 ]; then iface="${cands[0]}"
        else die "several mlx5 interfaces and none has carrier: ${cands[*]}
  Pick the cabled one with --iface."; fi
    fi
    note "auto-selected interface $iface"
fi

[ -d "/sys/class/net/$iface" ] || die "no such interface: $iface"
ibdev=$(ls "/sys/class/net/$iface/device/infiniband" 2>/dev/null | head -1)

if [ "$check_only" = 0 ]; then
    [ "$(id -u)" = 0 ] || die "needs root: re-run with sudo (or pass --check)"
    case "$iface" in
        *f1np1)
            [ "$force" = 1 ] || die "$iface is the cage already carrying the promax TP link.
  Use cage p0 (…f0np0), or --force if you really mean to touch it." ;;
    esac
    existing=$(ip -o -4 addr show dev "$iface" | awk '{print $4}')
    if [ -n "$existing" ] && [ "$existing" != "$addr" ] && [ "$force" = 0 ]; then
        die "$iface already holds $existing; pass --force to replace it with $addr"
    fi
fi

# ---------------------------------------------------------------- changes

if [ "$check_only" = 0 ]; then
    if [ "$do_install" = 1 ]; then
        note "installing RDMA userspace and build tools"
        export DEBIAN_FRONTEND=noninteractive
        run apt-get update
        run apt-get install -y rdma-core libibverbs-dev libibverbs1 ibverbs-utils \
                               ibverbs-providers perftest build-essential
    fi

    note "loading modules"
    modprobe mlx5_ib 2>/dev/null || true
    modprobe ib_uverbs 2>/dev/null || true

    # NetworkManager will happily drop a hand-assigned address on a device it
    # manages. Hand the cable interface over to us for this boot.
    if command -v nmcli >/dev/null 2>&1; then
        if [ "$(nmcli -g GENERAL.STATE device show "$iface" 2>/dev/null | head -1)" != "" ]; then
            nmcli device set "$iface" managed no >/dev/null 2>&1 \
                && note "NetworkManager: $iface set unmanaged for this boot"
        fi
    fi

    note "configuring $iface -> $addr mtu $mtu"
    run ip link set dev "$iface" up
    ip link set dev "$iface" mtu "$mtu" || note "WARNING: MTU $mtu refused; RoCE will negotiate a smaller path MTU"
    run ip addr replace "$addr" dev "$iface"

    if [ -n "$speed" ]; then
        note "forcing speed ${speed}Mb/s, autoneg off"
        ethtool -s "$iface" speed "$speed" autoneg off \
            || note "WARNING: the driver refused the forced speed"
        sleep 3
    fi

    if [ "$do_memlock" = 1 ]; then
        f=/etc/security/limits.d/95-ds4-rdma.conf
        note "memlock limit -> $f (takes effect on next login)"
        printf '* soft memlock unlimited\n* hard memlock unlimited\nroot soft memlock unlimited\nroot hard memlock unlimited\n' > "$f"
    fi

    if [ "$do_persist" = 1 ]; then
        f="/etc/netplan/90-ds4-rdma-$iface.yaml"
        note "writing $f"
        {
            echo "network:"
            echo "  version: 2"
            echo "  ethernets:"
            echo "    $iface:"
            echo "      addresses: [$addr]"
            echo "      mtu: $mtu"
            echo "      dhcp4: false"
            echo "      dhcp6: false"
            echo "      optional: true"
        } > "$f"
        chmod 600 "$f"
        netplan generate 2>&1 | sed 's/^/  /' || note "WARNING: netplan generate complained; review $f"
        note "review it, then 'sudo netplan apply' -- not applied automatically"
    fi

    # The link needs a moment after the peer end comes up.
    for _ in 1 2 3 4 5 6 7 8 9 10; do
        [ "$(cat "/sys/class/net/$iface/carrier" 2>/dev/null)" = 1 ] && break
        sleep 1
    done
fi

# ---------------------------------------------------------------- checks

echo
echo "== link =="
printf '  %-14s carrier=%s speed=%s mtu=%s ipv4=%s\n' "$iface" \
    "$(cat "/sys/class/net/$iface/carrier" 2>/dev/null)" \
    "$(cat "/sys/class/net/$iface/speed" 2>/dev/null)" \
    "$(cat "/sys/class/net/$iface/mtu" 2>/dev/null)" \
    "$(ip -o -4 addr show dev "$iface" | awk '{printf "%s ", $4}')"
command -v ethtool >/dev/null 2>&1 && ethtool "$iface" 2>/dev/null | \
    grep -E "Speed|Duplex|Port|Link detected|Auto-neg" | sed 's/^/  /'

# PHY-level counters, because they separate "nothing on the wire" from "signal
# arrives but will not train". rx_packets_phy climbing while rx_packets stays 0,
# with a large rx_corrected_bits_phy, is a lane/rate mismatch and not a cabling
# fault: a 200G 4-lane port facing a 100G 2-lane one can leave the narrow end
# claiming LinkUp while the wide end reports "No partner detected". Force BOTH
# ends to the lower rate (--speed 100000) rather than re-seating cables.
if command -v ethtool >/dev/null 2>&1; then
    phy=$(ethtool -S "$iface" 2>/dev/null | grep -E \
        "^ *(rx_packets|rx_packets_phy|rx_bytes_phy|rx_corrected_bits_phy|rx_symbol_err_phy|link_down_events_phy):")
    [ -n "$phy" ] && { echo "  phy counters:"; echo "$phy" | sed 's/^ */    /'; }
fi

echo
echo "== verbs =="
if [ -z "$ibdev" ]; then
    echo "  $iface has no infiniband device: the RDMA half of the driver is not loaded."
    echo "  Try 'sudo modprobe mlx5_ib' and 'dmesg | grep -i mlx5' for the reason."
else
    p="/sys/class/infiniband/$ibdev/ports/1"
    printf '  %s port 1: state=%s phys=%s rate=%s\n' "$ibdev" \
        "$(cat "$p/state" 2>/dev/null)" "$(cat "$p/phys_state" 2>/dev/null)" \
        "$(cat "$p/rate" 2>/dev/null)"
    printf '  fw=%s node_guid=%s\n' \
        "$(cat "/sys/class/infiniband/$ibdev/fw_ver" 2>/dev/null)" \
        "$(cat "/sys/class/infiniband/$ibdev/node_guid" 2>/dev/null)"

    # The RoCEv2 GID index for this link's IPv4. The t1 tools default to 3,
    # which is only right by coincidence: pass what is printed here.
    ip4=${addr%%/*}
    hex=$(printf '%02x%02x:%02x%02x' $(echo "$ip4" | tr '.' ' '))
    want="0000:0000:0000:0000:0000:ffff:$hex"
    gid_index=""
    for g in "$p"/gids/*; do
        [ -e "$g" ] || continue
        v=$(cat "$g" 2>/dev/null)
        [ "$v" = "0000:0000:0000:0000:0000:0000:0000:0000" ] && continue
        idx=$(basename "$g")
        t=$(cat "$p/gid_attrs/types/$idx" 2>/dev/null)
        nd=$(cat "$p/gid_attrs/ndevs/$idx" 2>/dev/null)
        mark=" "
        if [ "$v" = "$want" ] && [ "$t" = "RoCE v2" ]; then mark="*"; gid_index="$idx"; fi
        printf '  %sgid[%2s] %s  %-10s %s\n' "$mark" "$idx" "$v" "${t:-?}" "${nd:-}"
    done
    if [ -n "$gid_index" ]; then
        echo "  -> RoCEv2 GID index for $ip4 is $gid_index   (--gid-index / DS4_EXPERT_TIER_RDMA_GID)"
    else
        echo "  -> no RoCEv2 GID carries $ip4 yet. The address must be on THIS netdev"
        echo "     and the port must be ACTIVE before the GID table gets the entry."
    fi
fi

echo
echo "== peer =="
if [ -n "$peer" ]; then
    if ping -c 3 -W 2 -I "$iface" "$peer" >/tmp/.rdma_ping.$$ 2>&1; then
        tail -2 /tmp/.rdma_ping.$$ | sed 's/^/  /'
        echo
        echo "  Both ends are up. Next, from the ds4 (client) box:"
        echo "    ib_read_bw -d <local ib dev> -x <gid index> -F --report_gbits \\"
        echo "               -s 1048576 -q 8 $peer"
        echo "  with 'ib_read_bw -d <peer ib dev> -x <peer gid index> -F' running on the peer."
    else
        tail -3 /tmp/.rdma_ping.$$ | sed 's/^/  /'
        echo "  Peer $peer does not answer yet. Normal until the other end runs this"
        echo "  script too. If carrier is 0 on a cabled port: check the cable is in the"
        echo "  right cage, then try --speed 100000 (ConnectX-6 against a 200G cage)."
    fi
    rm -f /tmp/.rdma_ping.$$
fi
echo
