# DSpark verifier latency and RoCE v2 bottleneck profile — 2026-08-21

This is a diagnostic record, not a README performance result. It profiles the
current width-5, slab-direct attention candidate on the fixed Antirez Q4
2,048+300-token workload. Both ranks used matched `PROFILE=1` binaries,
ConnectX-4 Lx RC/RoCE v2, the 118/138 DSpark expert split, and no expanded
weight cache. The generated sequence retained the exact known fingerprint
`7174e214e05fd83e`.

## End-to-end control

| Run | Prefill | Decode | Verifier calls | Speculative cycles | Fingerprint |
|---|---:|---:|---:|---:|---|
| pooled HIP stage events + host sync | 202.67 t/s | 11.16 t/s | 83 | 143 | exact |
| transport phase counters only | 201.39 t/s | 11.35 t/s | 83 | 143 | exact |

The second run reproduces the uninstrumented first slab-direct result of
11.35 t/s. Profiling results must not be promoted as benchmark candidates.

## One target verifier invocation

There are 43 target layers per verifier call. Pooled HIP events observed 3,569
layer samples, exactly `83 * 43`, with no dropped samples.

| Component | Rank 0 per layer | Rank 1 per layer | Rank 0 per verifier | Rank 1 per verifier |
|---|---:|---:|---:|---:|
| Attention, including its TP combine | 2.369 ms | 2.401 ms | 101.87 ms | 103.24 ms |
| Routed MoE compute, before its TP gate | 2.182 ms | 2.360 ms | 93.83 ms | 101.48 ms |
| FFN gate/combine plus residual tail | 0.737 ms | 0.554 ms | 31.69 ms | 23.82 ms |
| Whole layer | 5.287 ms | 5.316 ms | 227.34 ms | 228.59 ms |

The coordinator's DSpark wall counters measured 20,418.1 ms across 83
verifiers, or 246.0 ms per invocation. Its GPU layer interval was 20,169.6 ms,
or 243.0 ms per invocation. The pooled-event sum is slightly lower because it
does not include verifier upload/head/read bookkeeping and event boundaries do
not cover all host-side gaps.

Attention on rank 0 divides further into approximately 9.72 ms front end,
45.75 ms QKV-to-core, 34.36 ms output projection, and 12.04 ms post/combine per
verifier. The corresponding rank-1 values are 9.93, 48.25, 35.52, and 9.55 ms.
The attention and routed-MoE families together occupy about 86% of the measured
GPU-layer interval.

## Proposal and target-anchor cost

| Component | Total | Normalized |
|---|---:|---:|
| Q8 DSpark proposal | 2,088.4 ms | 14.60 ms/speculative cycle |
| Target anchor/ordinary work | 4,264.8 ms | 29.82 ms/speculative cycle |
| Batch verification | 20,418.1 ms | 246.0 ms/verifier invocation |

The proposal is not the leading cost. Batch verification dominates even after
the transport fixes.

## Verifier transport breakdown

Each verifier layer performs two five-row, 100 KiB registered-slab exchanges:
attention then FFN. The diagnostic timestamps the receive posting, 16-byte
RDMA-WRITE readiness cookie, linked row SEND posting, and CQ completion. Values
below are means over 3,569 calls of each phase on each rank.

| Rank / phase | Ready wait | Row SEND/CQ wait | Local verbs work | Total callback |
|---|---:|---:|---:|---:|
| Rank 0 attention | 142.32 us | 61.58 us | 0.59 us | 204.48 us |
| Rank 0 FFN | 339.48 us | 66.62 us | 1.89 us | 407.99 us |
| Rank 1 attention | 42.48 us | 67.29 us | 3.70 us | 113.46 us |
| Rank 1 FFN | 135.49 us | 68.91 us | 3.50 us | 207.90 us |

`Ready wait` is not wire latency. It starts after this rank publishes its
cookie and therefore includes how early it reached the gate relative to its
peer. Rank 0 consistently waits longer, showing that rank 1 is the compute
straggler, especially before the FFN exchange. The actual 100 KiB row transfer
and completion is about 62–69 us. Receive posting, locks, and doorbell setup
are only a few microseconds.

The critical-rank callback cost is about 0.612 ms per layer, or 2.18 seconds
over the run. Of that, only about 0.46–0.49 seconds is the aligned payload/CQ
floor measured by the isolated 10,000-call test (66.6–69.2 us per exchange).
Most of the apparent “network time” is synchronization caused by rank-1
compute skew, not RoCE bandwidth or libibverbs overhead.

## Conclusion

The next large gain is not another transport protocol rewrite. Even perfect
zero-latency transport could remove only roughly 2.2 seconds from this run,
while target verification consumes about 20 seconds. The priority order from
this profile is:

1. reduce rank-1 routed-MoE work or rebalance the DSpark expert ownership;
2. make attention and routed Q4 compute reuse verifier rows more efficiently;
3. then reduce the readiness round trip or overlap it, after compute skew has
   been reduced;
4. retain the slab-direct registered transport, whose host/verbs overhead is
   already negligible relative to verifier kernels.

Raw evidence:

- `runs/coordinator-antirez-q4-dspark-w5-attn-slab-profile-r1.log`
- `runs/worker-antirez-q4-dspark-w5-attn-slab-profile-r1.log`
- `runs/coordinator-antirez-q4-dspark-w5-attn-slab-net-profile-r1.log`
- `runs/worker-antirez-q4-dspark-w5-attn-slab-net-profile-r1.log`
