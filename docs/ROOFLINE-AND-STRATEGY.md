# CORRECTED - read the correction block first

The original version of this document used active-byte figures measured for the
**UD-Q8_K_XL** checkpoint (161.9 GB) and applied them to the **Q4_K** checkpoint
we actually run. Different quantisation, different byte counts, different
conclusion. Recomputed directly from our own GGUF header below; the components
sum exactly to the 153.32 GiB file total, so they are self-consistent.

## Corrected numbers - Huihui-DeepSeek-V4-Flash-0731-abliterated Q4_K

| component | total | streamed per token |
|---|---|---|
| routed experts | 145.12 GiB | 3.40 GiB (6 of 256) |
| attention + indexer + compressor | 5.49 GiB | 5.49 GiB (all) |
| other dense (shexp, norms, hc) | 1.20 GiB | 1.20 GiB (all) |
| embeddings / output | 1.51 GiB | partial |
| **active per token** | | **10.09 GiB** |

At 195.6 GiB/s effective per node:

| config | bytes/node | ceiling |
|---|---|---|
| no TP, one node | 10.09 GiB | 19.4 t/s |
| **experts sharded only (today)** | 8.39 GiB | **23.3 t/s** |
| experts + attention split | 5.64 GiB | 34.7 t/s |

## What this changes

**Measured 6.5 t/s is 28% of our own ceiling, not 51%.**

- The gap to the CURRENT ceiling is **3.6x**. The attention split is worth
  **1.5x**. The inefficiency is the bigger prize by more than a factor of two.
- **Our current configuration already ceilings at 23.3 t/s, above llama.cpp's
  measured 15.** We do NOT need the attention split to beat the reference. It
  remains a genuine latent correctness bug and a later performance lever, but it
  is NOT the priority.
- The previous version of this file concluded the opposite and said "the
  attention split is the whole game". That was wrong, and wrong for an avoidable
  reason: transferring a measured decomposition across checkpoints without
  re-deriving it.

## Revised priority

1. **Correctness.** 6.5 t/s of wrong tokens is worth zero.
2. **Find the 3.6x.** We are leaving ~17 t/s on the table inside the current
   design. Measure where the decode step actually goes - transport vs GPU
   kernels vs host dispatch - before changing anything. 86 synchronous gates per
   token is the obvious suspect, but suspicion is not measurement.
3. Attention head split (1.5x, ceiling 34.7) - also fixes a latent correctness
   bug, so worth doing eventually, but not first.
4. DSpark/MTP - multiplicative on top, blocked by an unavailable-stub kernel.

Transport is still not the barrier: point-to-point gates, no collectives.

---

# (original, superseded - retained for the record)

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
