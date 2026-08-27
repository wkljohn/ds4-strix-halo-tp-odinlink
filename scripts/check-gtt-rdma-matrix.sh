#!/usr/bin/env bash
set -euo pipefail

# Read-only prerequisite check for the GLM-5.3 TP=2 bring-up.  It never
# changes boot parameters, restarts a host, or starts DS4.  Run it once per
# boot on both independent nodes before attempting a mapped-slab probe.
host1=${DS4_TP_NODE1:-node1}
host2=${DS4_TP_NODE2:-node2}
dev1=${DS4_TP_RDMA_DEVICE1:-mlx5_0}
dev2=${DS4_TP_RDMA_DEVICE2:-mlx5_1}

remote_check() {
    local host=$1 dev=$2
    ssh -o BatchMode=yes -o ConnectTimeout=4 "$host" "bash -s -- '$dev'" <<'REMOTE'
set -euo pipefail
dev=$1
cmdline=$(cat /proc/cmdline)
vram=$(cat /sys/class/drm/card*/device/mem_info_vram_total 2>/dev/null | head -1 || true)
gtt=$(cat /sys/module/amdgpu/parameters/gttsize 2>/dev/null || true)
pages=$(cat /sys/module/ttm/parameters/pages_limit 2>/dev/null || true)
pool=$(cat /sys/module/ttm/parameters/page_pool_size 2>/dev/null || true)
printf 'host=%s\n' "$(hostname -s)"
printf 'kernel=%s\n' "$(uname -r)"
printf 'cmdline=%s\n' "$cmdline"
printf 'vram_total_kib=%s\n' "$vram"
printf 'gttsize_mib=%s\n' "$gtt"
printf 'ttm_pages_limit=%s\n' "$pages"
printf 'ttm_page_pool_size=%s\n' "$pool"
if command -v ibv_devinfo >/dev/null 2>&1; then
    ibv_devinfo -d "$dev" -v 2>&1 | awk '/hca_id:|transport:|state:|link_layer:|gid_tbl_len:/{print}' | head -12
else
    echo 'ibv_devinfo=missing'
fi
REMOTE
}

echo "required_cmdline=amd_iommu=off amdgpu.gttsize=126976 ttm.pages_limit=32505856 ttm.page_pool_size=32505856"
for pair in "$host1:$dev1" "$host2:$dev2"; do
    host=${pair%%:*}; dev=${pair#*:}
    echo "--- $host ($dev) ---"
    if ! remote_check "$host" "$dev"; then
        echo "connection=FAIL"
        continue
    fi
    echo "connection=PASS"
done

echo "status=READ_ONLY; exact 124-GiB GTT compatibility requires a reboot with the required cmdline on both hosts"
echo "next=make tests/roce_v2_mr_probe && run ./tests/roce_v2_mr_probe <device> separately on each host after production is stopped"
