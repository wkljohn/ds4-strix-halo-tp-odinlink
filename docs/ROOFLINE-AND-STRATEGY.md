# What actually limits this, and why the attention split is the whole game

Derived from a measured decomposition of active bytes per decoded token for
DeepSeek-V4-Flash (recorded independently of this port), combined with the
measured effective bandwidth of one Strix Halo node (~210 GB/s = 195.6 GiB/s,
LPDDR5-8000, 256-bit).

## Active weights streamed per decoded token: 16.90 GiB

| component | GiB | share |
|---|---|---|
| always-read (dense/attention/norms/hc) | 13.69 | 81% |
| ...of which ATTENTION | 9.45 | 56% |
| 6 active routed experts | 3.21 | 19% |

**The 6 routed experts are only 19% of the bytes.** This is the single most
counter-intuitive fact about optimising this model and it invalidates the
instinct to focus on MoE placement.

## Roofline by TP configuration

| config | bytes/node | ceiling |
|---|---|---|
| no TP, one node | 16.90 GiB | 11.6 t/s |
| **experts sharded only (what we run today)** | 15.29 GiB | **12.8 t/s** |
| experts + attention split | 10.57 GiB | **18.5 t/s** |

The 11.6 t/s row independently reproduces a separately measured
"non-speculative ceiling ~11.6 tok/s" for this model on this hardware, which is
a useful check that this arithmetic is not fantasy.

Measured today: **6.5 t/s = 51% of our own 12.8 t/s ceiling.**

## Three consequences

1. **Expert sharding buys almost nothing for speed** (11.6 -> 12.8, +10%). Its
   value was making the model FIT (80.76 GiB/node vs 96), not making it fast.
   All the effort in patches 9/12 was a residency fix wearing a performance
   costume.

2. **The attention head split is the only structural lever above llama.cpp.**
   It moves the ceiling to 18.5 t/s, past llama.cpp's measured 15. And it is
   currently a NO-OP on ROCm: `g_tp_attn_head_split` is written at
   ds4_rocm.cu:476 and never read, so both ranks stream all 9.45 GiB of
   attention weights every token. Metal implements it in 4 batch kernels
   (ds4_metal.m:30915, 31050, 31991, 32128) via
   `ds4_gpu_tp_attn_head_range(n_head, group=8, &head_base, &head_count)`.

3. **This is why ds4 TP can beat llama.cpp at all.** llama.cpp's `-ts` is
   PIPELINE parallelism - for one token node A runs its layers, then node B runs
   its, so bandwidth is NOT additive and both GPUs sit ~44% busy. ds4 does real
   tensor parallelism, where the two nodes' bandwidth IS additive. That is the
   entire architectural argument for this port, and it only pays off once the
   dominant term (attention) is actually split.

## Ordering

Correctness first - 6.5 t/s of wrong tokens is worth nothing. But note the
attention head split is BOTH the top performance lever AND a latent correctness
bug (the one-per-chunk zeroing at ds4.c:44002 assumes kernels write only owned
heads). Implementing it properly addresses both, and should be done once,
carefully, rather than twice.

Transport is NOT the barrier: the gate pattern is point-to-point with no
collectives, and an independent estimate put comms at 2.5-6.3 ms/token against
~21.5 ms of halved compute. Do not start with RDMA.
