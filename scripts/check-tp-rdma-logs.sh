#!/bin/bash
# Prove that both ranks used the requested RDMA provider without transport or
# kernel failures. Shared by timed results and pre-timing diagnostic gates.
set -euo pipefail

COORD_LOG=${1:?usage: check-tp-rdma-logs.sh COORD_LOG WORKER_LOG RDMA_PROFILE}
WORKER_LOG=${2:?missing worker log}
RDMA_PROFILE=${3:-odinlink}

for path in "$COORD_LOG" "$WORKER_LOG"; do
  [[ -r $path ]] || { echo "error: missing TP log: $path" >&2; exit 1; }
done

grep -q 'worker connected, transport=rdma' "$COORD_LOG" || {
  echo "error: diagnostic did not use coordinator RDMA" >&2; exit 1;
}
grep -q 'leader connected, transport=rdma' "$WORKER_LOG" || {
  echo "error: diagnostic did not use worker RDMA" >&2; exit 1;
}

case $RDMA_PROFILE in
  odinlink)
    grep -q '"fallback_calls":0' "$COORD_LOG" || {
      echo "error: coordinator OdinLink provider reported fallback traffic" >&2; exit 1;
    }
    grep -q '"fallback_calls":0' "$WORKER_LOG" || {
      echo "error: worker OdinLink provider reported fallback traffic" >&2; exit 1;
    }
    ;;
  roce-v2)
    for log in "$COORD_LOG" "$WORKER_LOG"; do
      grep -q 'rdma device mlx5_' "$log" &&
      grep -q 'rdma GID index .* (RoCE v2)' "$log" &&
      grep -q 'mlx5 queue pair uses RC' "$log" &&
      grep -q 'mlx5 registered host slab as 3 MRs' "$log" || {
        echo "error: log does not prove mlx5 RoCE v2 RC with segmented MR: $log" >&2
        exit 1
      }
      ! grep -q 'rdma device odl_tb5_' "$log" || {
        echo "error: RoCE run unexpectedly used OdinLink: $log" >&2
        exit 1
      }
    done
    ;;
  *)
    echo "error: unknown RDMA profile: $RDMA_PROFILE" >&2
    exit 2
    ;;
esac

if grep -Eqi 'timeout waiting|transport failed|decode .* failed|kernel (launch )?failed|nan detected' \
     "$COORD_LOG" "$WORKER_LOG"; then
  echo "error: TP logs contain a transport, decode, or kernel failure" >&2
  exit 1
fi

echo "validated_rdma_profile=$RDMA_PROFILE"
