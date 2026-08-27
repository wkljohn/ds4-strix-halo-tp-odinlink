#!/usr/bin/env bash
set -euo pipefail

# Read-only GLM-5.3 TP=2 platform gate. A large GPUVM aperture is necessary
# but not sufficient on a UMA APU: the resident allocation needs real backing.
host1=${DS4_TP_NODE1:-node1}
host2=${DS4_TP_NODE2:-node2}
dev1=${DS4_TP_RDMA_DEVICE1:-mlx5_0}
dev2=${DS4_TP_RDMA_DEVICE2:-mlx5_1}
target_gib=${DS4_GTT_TARGET_GIB:-112}
required_resident=${DS4_GTT_REQUIRED_RESIDENT_BYTES:-103750315384}
reserve_bytes=${DS4_GTT_HOST_RESERVE_BYTES:-3221225472}

[[ $target_gib =~ ^[1-9][0-9]*$ ]] || {
    echo "error: DS4_GTT_TARGET_GIB must be a positive integer" >&2
    exit 2
}
[[ $required_resident =~ ^[1-9][0-9]*$ ]] || {
    echo "error: DS4_GTT_REQUIRED_RESIDENT_BYTES must be positive" >&2
    exit 2
}
[[ $reserve_bytes =~ ^[0-9]+$ ]] || {
    echo "error: DS4_GTT_HOST_RESERVE_BYTES must be an integer" >&2
    exit 2
}

target_bytes=$((target_gib * 1024 * 1024 * 1024))
target_pages=$((target_bytes / 4096))

host_check_script=$(cat <<'REMOTE'
set -euo pipefail
dev=$1
target_bytes=$2
target_pages=$3
required_resident=$4
reserve_bytes=$5

fail=0
host=$(hostname -s)
cmdline=$(cat /proc/cmdline)
mem_total_kib=$(awk '/^MemTotal:/{print $2}' /proc/meminfo)
mem_available_kib=$(awk '/^MemAvailable:/{print $2}' /proc/meminfo)
mem_total_bytes=$((mem_total_kib * 1024))
mem_available_bytes=$((mem_available_kib * 1024))
gtt_total=$(for f in /sys/class/drm/card*/device/mem_info_gtt_total; do
    [ -r "$f" ] && cat "$f"
done | sort -nr | head -1)
pages=$(cat /sys/module/ttm/parameters/pages_limit)
pool=$(cat /sys/module/ttm/parameters/page_pool_size)
iommu_groups=$(find /sys/kernel/iommu_groups -mindepth 1 -maxdepth 1 -type d | wc -l)
local_mem=$(awk '/^local_mem_size /{print $2; exit}' \
    /sys/class/kfd/kfd/topology/nodes/*/properties 2>/dev/null || true)

printf 'host=%s\n' "$host"
printf 'kernel=%s\n' "$(uname -r)"
printf 'cmdline=%s\n' "$cmdline"
printf 'mem_total_bytes=%s\n' "$mem_total_bytes"
printf 'mem_available_bytes=%s\n' "$mem_available_bytes"
printf 'kfd_local_mem_size=%s\n' "${local_mem:-unknown}"
printf 'gtt_aperture_bytes=%s\n' "${gtt_total:-0}"
printf 'ttm_pages_limit=%s\n' "$pages"
printf 'ttm_page_pool_size=%s\n' "$pool"
printf 'iommu_groups=%s\n' "$iommu_groups"

if [ -z "$gtt_total" ] || [ "$gtt_total" -lt "$target_bytes" ]; then
    echo 'gtt_aperture=FAIL'
    fail=1
else
    echo 'gtt_aperture=PASS'
fi
if [ "$pages" -lt "$target_pages" ] || [ "$pool" -lt "$target_pages" ]; then
    echo 'ttm_limits=FAIL'
    fail=1
else
    echo 'ttm_limits=PASS'
fi
if [ "$mem_total_bytes" -lt "$target_bytes" ] ||
   [ "$mem_available_bytes" -lt $((required_resident + reserve_bytes)) ]; then
    echo 'resident_backing=FAIL'
    fail=1
else
    echo 'resident_backing=PASS'
fi
if [ "$iommu_groups" -eq 0 ]; then
    echo 'translated_iommu=FAIL'
    fail=1
else
    echo 'translated_iommu=PASS'
fi

rdma_line=$(rdma link show 2>/dev/null | grep -F "link $dev/1 " || true)
if [[ $rdma_line != *'state ACTIVE'* ]] || [[ $rdma_line != *'physical_state LINK_UP'* ]]; then
    echo 'rdma_link=FAIL'
    fail=1
else
    echo 'rdma_link=PASS'
fi

rocev2_gid=''
for type_path in /sys/class/infiniband/"$dev"/ports/1/gid_attrs/types/*; do
    [ -r "$type_path" ] || continue
    idx=${type_path##*/}
    type=$(cat "$type_path" 2>/dev/null || true)
    gid=$(cat "/sys/class/infiniband/$dev/ports/1/gids/$idx" 2>/dev/null || true)
    ndev=$(cat "/sys/class/infiniband/$dev/ports/1/gid_attrs/ndevs/$idx" 2>/dev/null || true)
    if [ "$type" = 'RoCE v2' ] &&
       [ "$gid" != '0000:0000:0000:0000:0000:0000:0000:0000' ] &&
       [ -n "$ndev" ]; then
        rocev2_gid=$idx
    fi
done
printf 'rocev2_gid_index=%s\n' "${rocev2_gid:-none}"
if [ -z "$rocev2_gid" ]; then
    echo 'rocev2_gid=FAIL'
    fail=1
else
    echo 'rocev2_gid=PASS'
fi

exit "$fail"
REMOTE
)

run_host() {
    local host=$1 dev=$2
    if [ "$host" = local ] || [ "$host" = localhost ] ||
       [ "$host" = 127.0.0.1 ]; then
        bash -s -- "$dev" "$target_bytes" "$target_pages" \
            "$required_resident" "$reserve_bytes" <<<"$host_check_script"
    else
        ssh -o BatchMode=yes -o ConnectTimeout=5 "$host" \
            "bash -s -- '$dev' '$target_bytes' '$target_pages' \
             '$required_resident' '$reserve_bytes'" <<<"$host_check_script"
    fi
}

printf 'target_gtt_bytes=%s\n' "$target_bytes"
printf 'required_resident_bytes=%s\n' "$required_resident"
printf 'host_reserve_bytes=%s\n' "$reserve_bytes"
fail=0
for pair in "$host1:$dev1" "$host2:$dev2"; do
    host=${pair%%:*}
    dev=${pair#*:}
    echo "--- $host ($dev) ---"
    if run_host "$host" "$dev"; then
        echo 'host_gate=PASS'
    else
        echo 'host_gate=FAIL'
        fail=1
    fi
done

if [ "$fail" -ne 0 ]; then
    echo 'status=FAIL'
    echo 'note=aperture/link success does not prove backed resident capacity'
    exit 1
fi
echo 'status=PASS'
echo 'next=run the mapped-slab MR probe and a production-shaped allocation/touch gate'
