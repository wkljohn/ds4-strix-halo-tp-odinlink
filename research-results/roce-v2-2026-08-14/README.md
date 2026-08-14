# RoCE v2 transport bring-up — 2026-08-14

This folder records the first DS4 TP=2 Q4_K validation over the two-node
ConnectX-4 Lx link. The goal was a second production RDMA path without changing
OdinLink's provider loading, wire frame, QP transition masks, or optimized
message policy.

## Reference link

| Rank | Device | Netdev/address | GID |
|---|---|---|---|
| coordinator | `mlx5_0` | `ens1f0np0`, `192.168.99.1/24` | index 3, RoCE v2 |
| worker | `mlx5_1` | `ens1f1np1`, `192.168.99.2/24` | index 3, RoCE v2 |

The link had already measured 22.75 Gb/s RC write bandwidth and 2.61 us
average write latency with perftest.

## Critical allocator finding

mlx5 rejects DS4's normal gfx1151 `hipMalloc` slab with `EFAULT`. A complete
481 MiB `hipHostMallocMapped` replacement also fails registration with
`ENOMEM`. Both nodes pass when the RoCE profile:

- replaces, rather than duplicates, the TP slab with mapped pinned host memory;
- caps its direct-prefill capacity at 2,048 rows;
- registers the natural core, big-out, and big-in regions separately.

The resulting Q4_K slab is about 77.4 MiB for the 2,048-row benchmark: about
13.4 MiB core plus two 32 MiB direct regions. OdinLink continues to allocate
its original `hipMalloc` slab and retains the 8,192-row default.

Re-run the exact allocator gate on each node:

```sh
make tests/roce_v2_mr_probe
./tests/roce_v2_mr_probe mlx5_0       # coordinator
./tests/roce_v2_mr_probe mlx5_1       # worker
```

The probe deliberately reports the expected `hipMalloc` failure before proving
the mapped-host, three-MR layout.

For production, do not launch the coordinator from a transient *user* service
whose manager inherited the common 8 MiB locked-memory limit. Setting
`LimitMEMLOCK=infinity` on such a unit only records the requested value; the
child still reports 8 MiB in `/proc/<pid>/limits` and mlx5 returns `ENOMEM` on
the first 13.4 MiB MR. The deployment launcher therefore uses a system
transient unit for RoCE, sets the limit there, and validates the child's
effective limit. Its OdinLink launch path remains a user service.

## Transport behavior

The mlx5 path:

- selects an IPv4-mapped RoCE v2 GID using `_ibv_query_gid_ex`, with sysfs
  type lookup as the fallback;
- fails closed if an explicitly selected mlx5 GID is not RoCE v2;
- always creates RC, with infinite RNR retry and normal RC retry attributes;
- retains the conservative generic 16 KiB decode message policy;
- uses system `libibverbs`, never `DS4_TP_VERBS_LIB` or OdinLink environment.

OdinLink retains its existing UC-attempt/RC-fallback construction, minimal
INIT/RTR/RTS masks, 128 KiB message policy, async feature gate, and explicit
provider library.

## Reproduction

Both filesystems need the same binary and model at the same absolute paths.
The launcher verifies their hashes and rejects logs that do not prove mlx5,
RoCE v2, RC, segmented MRs, full token count, and clean RDMA completion.

```sh
DS4_BENCH_RDMA_PROFILE=roce-v2 \
DS4_BENCH_OUT="$PWD/research-results/roce-v2-2026-08-14" \
  ./run-tp-ds4-bench.sh roce-q4-2048x300-r1 \
  /absolute/path/DeepSeek-V4-Flash-Q4_K-0731.gguf
```

When the RoCE profile is selected, the launcher defaults to a 2,048-token
prefill chunk to stay within the proven mlx5 registration budget.

## Results

Every accepted full run generated 300/300 tokens with fingerprint
`5f8a983422299d76`.

| Transport | Prefill chunk | Prefill | Decode |
|---|---:|---:|---:|
| OdinLink, pre-change control | 4,096 | 195.13 t/s | 14.67 t/s |
| OdinLink, matched control r1 | 2,048 | 198.55 t/s | 14.88 t/s |
| OdinLink, matched control r2 | 2,048 | 192.63 t/s | 14.84 t/s |
| RoCE v2 r1 | 2,048 | 224.05 t/s | 16.98 t/s |
| RoCE v2 r2 | 2,048 | 221.47 t/s | 17.18 t/s |

The two-run midpoint is 195.59/14.86 t/s for OdinLink and 222.76/17.08 t/s
for RoCE v2: **+13.9% prefill and +14.9% decode**.

Raw CSV and rank logs in this directory are intentionally git-excluded local
research evidence; this README is committed.
