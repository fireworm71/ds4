#!/bin/bash
# Start the T1 RDMA server on promax with the right devices, auto-detected.
#
# The two legs MUST sit on two different verbs devices. Two QPs on one device
# split that device's ~13 GB/s 50/50 and striping measures as worth nothing --
# that false negative is the whole reason this script picks the devices for you
# instead of letting you type them.
set -u
cd "$(dirname "$0")"

SIZE=${SIZE:-$((1024*1024*1024))}
PORT_A=${PORT_A:-19515}
PORT_B=${PORT_B:-19516}

if [ ! -x ./t1_server ]; then
    echo "== building =="
    make || { echo "build failed -- need libibverbs-dev and a C compiler"; exit 1; }
fi

dev_a=""; dev_b=""
for d in /sys/class/infiniband/*; do
    [ -d "$d" ] || continue
    dev=$(basename "$d")
    state=$(cat "$d/ports/1/state" 2>/dev/null || echo "")
    case "$state" in *ACTIVE*) ;; *) continue ;; esac
    netdev=$(ls "$d/device/net" 2>/dev/null | head -1)
    [ -n "$netdev" ] || continue
    ip4=$(ip -o -4 addr show "$netdev" 2>/dev/null | awk '{print $4}' | cut -d/ -f1 | head -1)
    [ -n "$ip4" ] || continue
    echo "  found $dev ($netdev) $ip4  state=$state"
    case "$ip4" in
        10.99.0.*) dev_a="$dev" ;;
        10.99.2.*) dev_b="$dev" ;;
    esac
done

if [ -z "$dev_a" ] || [ -z "$dev_b" ]; then
    echo
    echo "ERROR: could not find one ACTIVE device on 10.99.0.x AND one on 10.99.2.x."
    echo "       found: dev_a='${dev_a:-none}' dev_b='${dev_b:-none}'"
    echo "  - both addresses present?          ip -4 addr | grep 10.99"
    echo "  - both ports ACTIVE?               ibv_devinfo | grep -E 'hca_id|state'"
    echo "  - arp_ignore clear? (must be 0)    sysctl net.ipv4.conf.all.arp_ignore"
    echo "    it is max(all,<iface>), so a per-interface setting reads back 0 and"
    echo "    does nothing; with 1 the second NIC is silently unusable."
    exit 1
fi
if [ "$dev_a" = "$dev_b" ]; then
    echo "ERROR: both addresses are on the same device ($dev_a). Cannot stripe."
    exit 1
fi

echo
echo "== starting T1 server =="
echo "   leg A $dev_a (10.99.0.2) tcp :$PORT_A"
echo "   leg B $dev_b (10.99.2.2) tcp :$PORT_B"
echo "   region $((SIZE/1024/1024)) MiB -- it is mlocked, so keep an eye on free RAM"
echo "   runs in a loop and outlives each client; Ctrl-C to stop."
echo
exec ./t1_server --dev-a "$dev_a" --dev-b "$dev_b" \
                 --port-a "$PORT_A" --port-b "$PORT_B" --size "$SIZE"
