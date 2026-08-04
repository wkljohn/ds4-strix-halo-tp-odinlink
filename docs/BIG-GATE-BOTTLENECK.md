# The FFN all-reduce is 35% of prefill, and 64% of THAT is a CPU memcpy at 200 MB/s

## What `hc_post` actually is

The stage profile put `ffn hc_post` at 35.2% of prefill. It is **not**
hyper-connection math. Between the `shared_down` boundary (ds4.c:28826) and the
`hc_post` boundary (ds4.c:29144) sits the **TP FFN all-reduce** (ds4.c:29071-29093):
`ds4_gpu_tp_big_gate_encode` of the full routed partial - 2186 x 4096 x 4 B =
**35.8 MB each way, per layer**. The actual hc expand is ~2 ms: the
attention-side `hc_post`, which is the same expand *without* an exchange,
measures 1.45 ms.

## Measured split (DS4_TP_BIGGATE_PROFILE=1)

    16 gates direct=0 | staging-copy 5721.6 ms (64%) | wire+wait 3274.2 ms (36%)
    573.0 MB moved -> copy 200 MB/s, effective 64 MB/s

**64% of big-gate time is a CPU memcpy running at 200 MB/s.**

Mechanism: `tp_rdma_big_gate_exchange`'s `direct` fast path (ds4_tp.c:1148-1150)
requires BOTH buffers inside the ibv-registered slab. `batch_routed_out` is an
ordinary graph tensor, so **`direct` is always 0 for prefill** and every 1 MiB
round does `memcpy(stage_send, out+off, ...)` (ds4_tp.c:1171) plus a mirror copy
on receive (ds4_tp.c:1270). Those are **CPU reads of hipMalloc device memory** -
host-reachable on this UMA APU but write-combining, hence ~200 MB/s against the
10-20 GB/s a normal host memcpy achieves.

**This retroactively explains the RDMA-vs-TCP result** in RDMA-WORKING.md: they
measured at parity (8.29 vs 8.57 t/s decode) because both transports share this
host-copy bottleneck. The wire was never the limiter.

## Patch 21: device-DMA staging - IMPLEMENTED, DISABLED, DOES NOT WORK YET

`ds4_tp_set_devcopy()` (ds4_tp.h) lets the backend register a device-side copy so
the GPU's DMA engines do the staging - they are idle, since the GPU is parked in
`hipStreamWaitValue64`. ds4_rocm.cu implements it with `hipMemcpyAsync` on a
**non-blocking** stream (a null-stream copy would implicitly wait on
`g_tp_stream`, which is parked on the very gate being serviced - deadlock).

**It breaks the transfer:**

    ds4-tp: timeout waiting for bulk RDMA round (33/64 recvs, send=1)

The GPU's DMA writes into the registered slab are not reliably visible to the
OdinLink NIC. `hipStreamSynchronize` orders the copy but does not make it
externally visible to a third-party device. This is a coherence problem, not a
correctness bug in the copy itself.

Gated behind `DS4_TP_DEVCOPY=1`, **off by default**; verified no regression with
it off (prefill 29.91 t/s, coherent output).

## The better fix: make `direct` true

Rather than fight coherence, give the slab dedicated big-gate regions sized for
the max prefill chunk and have ds4.c copy into them **on-stream** before the
gate. Then `direct` is true, there is no staging copy at all in either
direction, and the peer partial can be consumed straight from a slab view. Risk
to retire first: whether OdinLink's `ibv_reg_mr` accepts a ~150 MB registration
(today's slab is ~25 MB).

Expected if the copy disappears: gate 1080 ms -> ~50 ms per layer-chunk, i.e.
roughly **30-35% of prefill wall**, correctness-neutral because it is pure byte
transport. It would also speed decode's gates and any future attention split.
