# Range-based sharding fixed the VRAM leak; now a TP gate deadlock

## Fixed: expert sharding by range (patch 12, replaces patch 9's masking)

| | before (mask) | after (range) |
|---|---|---|
| peak VRAM | 95.52 GiB | **85.65 GiB** |
| outcome | arena alloc failed (1152 MiB), illegal memory access | no OOM, prefill entered |

The launcher now receives gate/up/down offsets shifted by `base*expert_bytes`
and `n_total_expert = count`, so its span is `count*expert_bytes` = 576 MiB (the
resident half) instead of 1152 MiB (the whole layer). Selection indices are
rebased `e-lo`; unowned pairs get index 0 and weight 0, so partial sums still
recombine exactly.

Two things worth recording:

- **The batch path had NO sharding at all.** Patch 9 only touched
  `ds4_gpu_routed_moe_one_tensor`; `ds4_gpu_routed_moe_batch_tensor` - which is
  the *prefill* path, i.e. the one that was failing - went unsharded. Without
  patch 12 it both double-counted experts across ranks and paged in the unowned
  half. Applying the fix to only the one-token entry point would have left
  prefill silently wrong.
- **`cuda_tmp_alloc` returns a single shared global buffer**, so the two scratch
  regions (int32 selection + float weights) must come from ONE allocation that
  is split. Two calls would have aliased and corrupted the selection.

## Now blocked: both ranks spin, no gate traffic

Both processes stay alive with **both GPUs pinned at 100% for 16+ minutes** and
produce nothing. The TCP connection carries **204 bytes sent / 92 received and
then nothing** - that is the hello handshake and no gate exchange at all. So it
is deadlocked at or before the first gate, not merely slow.

### Prime suspect: the signal-memory fallback

Both ranks log:

    ds4: ROCm tp_init: hipMallocSignalMemory unavailable, using host memory

The gate runtime waits on the GPU with `hipStreamWaitValue64` against a flag the
CPU service thread writes. With `hipMallocSignalMemory` unavailable that flag
lives in ordinary host memory, and a stream wait-value may simply never observe
host-side stores - the wait is designed for device-visible signal memory.

That fits the evidence exactly: GPU busy at 100% (spinning in the wait), CPU
never asked to send anything, zero bytes on the wire.

The earlier T0/T2 probes do NOT cover this. T0 showed `hipMalloc` memory is
host-*reachable*, and T2 showed flag-arrival implies payload visibility - both
tested the CPU observing GPU writes, which is the opposite direction from the
GPU observing CPU writes through a stream wait-value.

### Next steps, cheapest first

1. Probe `hipStreamWaitValue64` against host memory directly (tiny, no model):
   does a stream wait ever observe a CPU-side store? If not, this is confirmed.
2. Find out why `hipMallocSignalMemory` is unavailable on this ROCm 7.2.0 /
   gfx1151 build - it may need `hipExtMallocWithFlags` with an uncached/
   fine-grained flag instead.
3. Fallback if signal memory cannot be had: drive the gate from the host side
   (stream-callback or explicit sync per gate) rather than a device wait-value.
   Slower per gate, but correctness first - and decode is gate-bound, so this
   would also give the first honest t/s number.
