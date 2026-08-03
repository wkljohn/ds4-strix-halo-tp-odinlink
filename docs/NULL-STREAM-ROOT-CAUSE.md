# The gate deadlock: hipStreamWriteValue64 does not work on the null stream

Two independent defects, both now fixed. Prefill completes and the gate flows.

## Defect 1 (mine, patch 4): the null stream

`scripts/t4_null_stream_gate_probe.cpp`, gfx1151 / ROCm 7.2.0:

    A. created stream, no prior kernel, 1 pair            : ARRIVAL SEEN (0.00 s)
    B. NULL stream,    no prior kernel, 1 pair            : *** NEVER SEEN *** (12.00 s)
    C. NULL stream,    long kernel queued first, 1 pair   : *** NEVER SEEN ***
    D. NULL stream,    long kernel + 44 pairs             : *** NEVER SEEN ***

Case B is the control: identical code, nothing queued ahead, single pair. The
ONLY difference from A is stream 0 versus `hipStreamCreate`. So
`hipStreamWriteValue64` simply never lands on the null stream here. Nothing to
do with kernels, ordering, or queue depth.

Patch 4 chose the null stream deliberately, with a comment justifying it:
"WHY THE NULL STREAM. Every compute kernel launches <<<grid,block,shmem>>> with
no stream argument, so nothing else orders against them." The ordering reasoning
was sound; the assumption that stream-memory ops function there was never tested.

**T3 missed this entirely because it used `hipStreamCreate`** - it validated the
primitive in a configuration ds4 does not use, and I reported it as clearing the
gate mechanism. Probes must reproduce the real configuration, not a convenient one.

Fix (patch 13): dedicated `g_tp_stream` from `hipStreamCreate`, used for both the
write and the wait. `scripts/t5_gate_stream_fix_probe.cpp` verified this is safe
BEFORE changing ds4:

    E1 gate on created stream (compute on null) : ARRIVAL SEEN (0.00 s)
    E2 stamp BEFORE release (want 0)            : 0  OK - ordering holds
    E2 stamp AFTER release  (want 42)           : 42 - released correctly

i.e. HIP's legacy null-stream implicit sync still makes null-stream kernels wait
for the gate, so the blocking semantics the design depends on are preserved.

## Defect 2 (mine, patch 12): scratch lifetime - found by review, not by the hang

`cuda_tmp_alloc` (`rocm/ds4_rocm_runtime.cuh:567`) is a ONE-SLOT global
allocator: it hands back `g_cuda_tmp` if large enough, else `cudaFree`s it and
`cudaMalloc`s a new one. Patch 12 put the remapped selection/weights there, and
`routed_moe_launch` calls the SAME allocator again at `moe_launch.cuh:919` and
`:1771` while those pointers are still live. The Q2_K/WMMA request at `:1771` is
much larger, so the remap buffer is freed out from under `use_selected->ptr`; on
the `:919` path the sizes can alias exactly and the `cudaMemset` of `counts`
zeroes the selection before the count kernel reads it.

Either way `counts[]` is garbage, and `rocm/ds4_rocm_moe.cuh:3519` runs
`for (uint32_t p0 = 0; p0 < count; p0 += PAIR_TILE)` - an unbounded loop. 100%
GPU forever, no fault (the 96 GiB carve-out means the OOB reads land in mapped
memory instead of faulting).

I had guarded against my own two scratch regions aliasing EACH OTHER and never
considered the callee re-entering the same allocator. Fix: a dedicated grow-only
`cudaMalloc` buffer for the remap.

## Result

    encode count: 44      pump count: 44        (was 44 / 0)
    43 big gates (ch=1) through prefill, then ch=0 row gate at layer 0 pumped

Prefill now COMPLETES and decode starts. New failure, further along:

    ds4: decode failed: rocm decode failed

## Method note

My "GPU stuck in a kernel before the first gate" conclusion happened to be true
of defect 2, but was NOT forced by the evidence I cited. As the review pointed
out, `rocm-smi` utilization cannot distinguish a running kernel from a queue
parked on a barrier packet - `--showpower`/`--showclocks` can, and I never
measured them. Two of my last three hypotheses here were wrong; the ones that
held were the ones I probed rather than argued.
