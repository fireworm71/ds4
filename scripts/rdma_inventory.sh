#!/usr/bin/env bash
# rdma_inventory.sh -- facts needed before wiring a box into a ds4 RDMA setup.
#
# No sudo, no writes outside --io, no package installs: it reads /sys, /proc and
# the unprivileged verbs tools. Run it on the new box and paste the output back.
#
#   ./scripts/rdma_inventory.sh              # inventory only
#   ./scripts/rdma_inventory.sh --io PATH    # also time a 2 GiB O_DIRECT read
#
# The --io arm writes a 2 GiB temp file under PATH, reads it back with O_DIRECT
# and deletes it. Skip it on a full filesystem.

set -uo pipefail

io_path=""
while [ $# -gt 0 ]; do
    case "$1" in
        --io) io_path="${2:?--io needs a directory}"; shift 2 ;;
        -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 1 ;;
    esac
done

hr() { printf '\n== %s ==\n' "$1"; }
have() { command -v "$1" >/dev/null 2>&1; }

hr "box"
echo "hostname : $(hostname)"
echo "arch     : $(uname -m)"
echo "kernel   : $(uname -r)"
have lsb_release && echo "os       : $(lsb_release -ds 2>/dev/null)"
echo "cpu      : $(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2- | sed 's/^ //')"
echo "cores    : $(nproc)"
echo "memlock  : soft=$(ulimit -Sl) hard=$(ulimit -Hl) KiB   <- the region-server mlock budget"

hr "memory"
free -g | sed 's/^/  /'

hr "gpu"
if have nvidia-smi; then
    nvidia-smi --query-gpu=name,memory.total,driver_version,pcie.link.gen.current,pcie.link.width.current \
               --format=csv,noheader 2>/dev/null | sed 's/^/  /'
else
    echo "  no nvidia-smi"
fi

hr "storage"
df -hT -x tmpfs -x devtmpfs -x squashfs -x overlay 2>/dev/null | sed 's/^/  /'
echo
lsblk -d -o NAME,SIZE,ROTA,MODEL 2>/dev/null | grep -vE '^loop' | sed 's/^/  /'

hr "mellanox pci"
found=0
for d in /sys/bus/pci/devices/*; do
    [ "$(cat "$d/vendor" 2>/dev/null)" = "0x15b3" ] || continue
    found=1
    slot=$(basename "$d")
    printf '  %s  dev=%s  link now: %s x%s   max: %s x%s   numa=%s\n' \
        "$slot" "$(cat "$d/device" 2>/dev/null)" \
        "$(cat "$d/current_link_speed" 2>/dev/null)" "$(cat "$d/current_link_width" 2>/dev/null)" \
        "$(cat "$d/max_link_speed" 2>/dev/null)" "$(cat "$d/max_link_width" 2>/dev/null)" \
        "$(cat "$d/numa_node" 2>/dev/null)"
    have lspci && lspci -s "$slot" 2>/dev/null | sed 's/^/      /'
done
[ "$found" = 1 ] || echo "  no Mellanox/NVIDIA networking PCI device found (vendor 0x15b3)"

hr "net interfaces"
for n in /sys/class/net/*; do
    i=$(basename "$n")
    [ "$i" = lo ] && continue
    drv=$(basename "$(readlink -f "$n/device/driver" 2>/dev/null)" 2>/dev/null)
    ib=$(ls "$n/device/infiniband" 2>/dev/null | tr '\n' ' ')
    addr=$(ip -o -4 addr show dev "$i" 2>/dev/null | awk '{printf "%s ", $4}')
    printf '  %-16s drv=%-10s oper=%-6s carrier=%-2s speed=%-8s mtu=%-5s ib=%-14s ipv4=%s\n' \
        "$i" "${drv:--}" "$(cat "$n/operstate" 2>/dev/null)" \
        "$(cat "$n/carrier" 2>/dev/null)" "$(cat "$n/speed" 2>/dev/null)" \
        "$(cat "$n/mtu" 2>/dev/null)" "${ib:--}" "${addr:--}"
done

hr "verbs devices"
if have ibv_devices; then
    ibv_devices 2>/dev/null | sed 's/^/  /'
else
    echo "  ibv_devices missing (install ibverbs-utils)"
fi
for p in /sys/class/infiniband/*/ports/1; do
    [ -e "$p" ] || continue
    d=$(basename "$(dirname "$(dirname "$p")")")
    amtu=""
    have ibv_devinfo && amtu=$(ibv_devinfo -d "$d" 2>/dev/null | awk '/active_mtu/{print $2; exit}')
    echo
    printf '  %s port 1: state=%s phys=%s rate=%s active_mtu=%s\n' "$d" \
        "$(cat "$p/state" 2>/dev/null)" "$(cat "$p/phys_state" 2>/dev/null)" \
        "$(cat "$p/rate" 2>/dev/null)" "${amtu:-?}"
    printf '      fw=%s node_guid=%s\n' \
        "$(cat /sys/class/infiniband/"$d"/fw_ver 2>/dev/null)" \
        "$(cat /sys/class/infiniband/"$d"/node_guid 2>/dev/null)"
    # Non-zero GIDs with type and backing netdev. This is the table the ds4 tier
    # and the t1 tools index into (--gid-index / DS4_EXPERT_TIER_RDMA_GID): the
    # entry you want is the RoCE v2 one carrying the link's IPv4 address.
    for g in "$p"/gids/*; do
        [ -e "$g" ] || continue
        v=$(cat "$g" 2>/dev/null)
        [ "$v" = "0000:0000:0000:0000:0000:0000:0000:0000" ] && continue
        idx=$(basename "$g")
        printf '      gid[%2s] %s  %-10s %s\n' "$idx" "$v" \
            "$(cat "$p/gid_attrs/types/$idx" 2>/dev/null)" \
            "$(cat "$p/gid_attrs/ndevs/$idx" 2>/dev/null)"
    done
done

hr "toolchain"
if have dpkg; then
    dpkg -l 2>/dev/null | awk '/ (libibverbs1|libibverbs-dev|rdma-core|ibverbs-utils|ibverbs-providers|perftest|build-essential) /{printf "  %-22s %s\n", $2, $3}'
fi
for t in gcc make ibv_devinfo ib_read_bw; do
    printf '  %-12s %s\n' "$t" "$(command -v "$t" || echo MISSING)"
done

if [ -n "$io_path" ]; then
    hr "disk read, O_DIRECT, 2 GiB"
    f="$io_path/.rdma_inventory_io.$$"
    if dd if=/dev/zero of="$f" bs=1M count=2048 oflag=direct status=none 2>/dev/null; then
        sync
        echo "  read : $(dd if="$f" of=/dev/null bs=1M iflag=direct 2>&1 | tail -1)"
    else
        echo "  could not write $f (space? permissions? O_DIRECT unsupported?)"
    fi
    rm -f "$f"
fi

echo
echo "done."
