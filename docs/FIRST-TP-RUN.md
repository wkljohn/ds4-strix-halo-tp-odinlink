# First ds4 tensor-parallel run on ROCm — TP binds, blocked at session backend

Date: 2026-08-03. Both nodes, DeepSeek-V4-Flash-Q4_K-0731 (164.6 GB GGUF),
TCP transport (RDMA deliberately not yet enabled).

## What worked — TP is live on ROCm

Upstream refuses TP on any non-Metal backend. With patches 1-11 both ranks
reached:

    ds4-tp: worker connected, transport=tcp
    ds4: ROCm TP rank 0 ready (2 channels, HIP stream wait-value)
    tensor parallelism bound: rank 0, 50/50 expert split, tcp transport

and symmetrically rank 1. **No "tensor parallelism is Metal-only" message
anywhere** - the gate runtime, the backend gate, the engine guards, the QP
fallback and the compute kernels all carried.

## The residency prediction was exact

Computed offline from the GGUF header *before* loading, from the rule in
`ds4.c:55615` ("replicated dense weights plus its expert shard"):

| | predicted | ds4 reported |
|---|---|---|
| total weights | 153.32 GiB | `153.32 GiB` |
| per-rank shard | 80.76 GiB | `mapping 219 spans, 80.76 GiB of 153.32 GiB` |

Routed experts are 145.12 GiB (94.7% of weights) and dense is only 8.20 GiB,
which is why halving experts is nearly the whole win. 80.76 GiB against 96 GiB
VRAM leaves ~15 GiB - the model fits two nodes with room, and would not fit one.

Both shards warmed with distinct checksums (rank0 2516722070, rank1 2516504328),
as expected for different halves.

## Blocker: ds4_gpu_tensor_fill_f32 launch fails

    ds4: ROCm tensor fill f32 launch failed: invalid argument
    ds4: sampled CLI generation requires a session backend   (ds4_cli.c:524)

`rocm/ds4_rocm_runtime.cuh:5982`:

    extern "C" int ds4_gpu_tensor_fill_f32(ds4_gpu_tensor *tensor, float value, uint64_t count) {
        if (!tensor || count > tensor->bytes / sizeof(float)) return 0;
        if (count == 0) return 1;
        fill_f32_kernel<<<(count + 255u) / 256u, 256>>>((float *)tensor->ptr, count, value);
        return cuda_ok(cudaGetLastError(), "tensor fill f32 launch");
    }

Not yet root-caused. Candidates, in order:

1. **`cudaGetLastError()` reports a STICKY earlier error, not this launch.** It
   returns and clears the last error from anywhere, so a prior failed op is
   attributed here. This is the most likely explanation and must be excluded
   first by clearing the error immediately before the launch.
2. A null/failed `tensor->ptr` reaching the launch (run 1 showed real
   `tensor alloc failed: out of memory` right after this same line).
3. A grid dimension out of range - needs `count` ~5.5e11 to overflow x-dim, so
   unlikely but cheap to assert.

## Memory note (run 1 vs run 2)

Run 1 at default ctx=32768 OOM'd:

    q8 fp16 cache ... cached=9.91 GiB free=4.81 GiB reserve=4.80 GiB total=96.00 GiB
    ds4: ROCm tensor alloc failed: out of memory   (x2)

Run 2 with `-c 8192` removed the OOM (context 1.03 -> 0.53 GiB) and the alloc
failures disappeared, leaving only the fill failure.

`DS4_ROCM_STREAM_Q8_F16_CACHE_GB=2` had **no effect** - the cache still reported
9.91 GiB, so that env var does not gate this allocation path (it appears to
apply to ssd-streaming mode). `DS4_CUDA_NO_Q8_F16_CACHE` is the other documented
lever (`ds4_rocm_runtime.cuh:4976`) and is untried. The reserve arithmetic
itself is correct: 96 GiB / 20 = 4.8 GiB, matching the reported reserve.

## Not yet reached

Decode throughput - no token has been generated yet, so there is still **no
t/s number** for ds4 TP. The llama.cpp reference on this hardware is 15 t/s.
