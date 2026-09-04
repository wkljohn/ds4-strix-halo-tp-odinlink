# OdinLink tensor-parallel transport

DS4 can run two-node tensor parallelism over the standalone OdinLink verbs
provider. The nodes may have independent filesystems: copy the DS4 and provider
binaries explicitly and compare their SHA-256 hashes before every benchmark.

## Provider isolation

OdinLink is selected explicitly:

```sh
export DS4_TP_VERBS_LIB=/path/to/libodl_tb5_verbs.so.0.1.0
export LD_LIBRARY_PATH=/path/to/odinlink/lib
```

Without `DS4_TP_VERBS_LIB`, DS4 loads the system `libibverbs` directly. This
keeps native providers such as Mellanox `mlx5` and `mlx4` outside the OdinLink
shim and preserves the existing generic 16 KiB decode-message framing.
InfiniBand ports without an IPv4-mapped GID fall back to GID 0 automatically.
OdinLink devices
named `odl_tb5_*` negotiate a 128 KiB limit, allowing a normal 28,672-byte
decode vector to use one message. `DS4_TP_RDMA_DECODE_MAX_MSG` can override the
local advertised limit; the peers use the lower value.

## Recommended Strix Halo settings

Use the same provider paths on both ranks:

```sh
export DS4_TP_VERBS_LIB=/path/to/libodl_tb5_verbs.so.0.1.0
export LD_LIBRARY_PATH=/path/to/odinlink/lib
```

The validated Strix Halo settings are defaults in this fork. `ODL_VERBS_WC_STREAM_COPY`
belongs to the OdinLink provider and defaults on when DS4 explicitly loads an
OdinLink provider. It accelerates
reads from GPU-written, write-combined host memory with an ordinary unaligned
prefix and AVX-512 streaming loads for the aligned body. It requires provider
commit `d0b54fc` or later. Set `ODL_VERBS_WC_STREAM_COPY=0` to run a control
arm. Keep any override identical on both ranks.

The gfx1151 packed Q8 attention-output low kernel is also a Strix Halo TP=2
default. It is a compute-only, exact-shape dispatch and does not inspect the
RDMA provider, so system `libibverbs` and Mellanox paths are unaffected. Set
`DS4_ROCM_DISABLE_ATTN_OUT_LOW_PACK4=1` on both ranks for its control arm. The
matching packed expansion kernel has the same provider isolation and uses
`DS4_ROCM_DISABLE_ATTN_OUT_EXPAND_PACK4=1` as its control. The packed Q-B
decode kernel is likewise transport-neutral and uses
`DS4_ROCM_DISABLE_ATTN_Q_B_PACK4=1`.

Launch both model loads concurrently. The worker may begin connection retries
before the coordinator listens; if it exhausts retries, relaunch only the
worker after the coordinator is ready. Serial model loading wastes time and is
unnecessary when the filesystems are independent.

```sh
# Worker
./ds4 -m model.gguf --rocm --tensor-parallel --role worker \
  --coordinator 10.4.0.1 5599 --transport rdma -c 4096

# Coordinator
./ds4 -m model.gguf --rocm --tensor-parallel --role coordinator \
  --listen 10.4.0.1 5599 --transport rdma -c 4096 \
  --temp 0 --seed 42 -n 30 --prompt-file prompt.txt
```

## Validated streaming-copy result

The 2026-08-06 A/B used two Ryzen AI MAX+ 395 systems, DeepSeek V4 Flash
Q4_K, a 9,881-byte prompt, a 4,096-token context, and provider commit
`d0b54fc`.

| Arm | Prefill | Generation |
|---|---:|---:|
| Default copy, run 1 | 95.97 t/s | 10.12 t/s |
| Streaming copy, run 1 | 138.85 t/s | 11.33 t/s |
| Streaming copy, run 2 | 138.71 t/s | 11.12 t/s |
| Default copy, run 2 | 95.80 t/s | 9.79 t/s |
| Default median | 95.89 t/s | 9.96 t/s |
| Streaming median | 138.78 t/s | 11.23 t/s |
| Median change | **+44.7%** | **+12.8%** |

All four generated stdout files were byte-identical. Each optimized rank
reported 11,696 streaming calls and 1,241,465,728 streaming bytes, with zero
full-request fallbacks. Coordinator RSS remained approximately 26.2-26.5 GiB.

A valid optimized run must emit an `odl_wc_stream_copy_summary` JSON record on
both ranks with `"enabled":true`, nonzero `stream_calls`, and explained
fallback traffic. Always compare greedy output bytes as well as throughput.

The provider retains one reusable bounce allocation per send-queue slot. Its
memory high-water mark therefore depends on queue depth and the largest send
that occupies each slot; monitor RSS for substantially larger workloads.

## Terminal callback failure

An RDMA callback failure is terminal for a lockstep TP session. The GPU may
already have many wait-value gates queued when a callback times out; releasing
only the failed sequence lets the service thread enter another network wait
while both GPUs remain parked.

DS4 now handles this in the transport-neutral ROCm gate runtime: it latches the
failure, publishes a terminal release value to both gate channels, prevents a
smaller per-request release from overwriting it, disables further callbacks,
and stops the service loop. This changes no successful gate or provider path,
so OdinLink and generic/Mellanox verbs retain identical normal behavior. No
OdinLink provider or kernel-module change is required for this failure mode.

Do not abruptly kill either rank while its GPU is waiting at a TP gate. Allow
the terminal drain to return both processes; verify zero GPU activity and VRAM
before changing modules, devices, or providers.
