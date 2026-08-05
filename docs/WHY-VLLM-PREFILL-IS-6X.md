# vLLM does 198.8 tok/s prefill on this hardware; we do ~30. Why.

Measured on the SAME two Strix Halo gfx1151 nodes, same model family:

| engine | prefill | decode |
|---|---|---|
| **vLLM TP=2** | **198.8 tok/s** | 3.28 t/s |
| **llama.cpp (pipeline)** | **80-95 tok/s** | 9.42 no-draft |
| **ds4 TP=2 (this port)** | **~30 t/s** | **~10.5 t/s** |

**The llama.cpp 80-95 t/s figure is GOLDEN EVIDENCE**: directly witnessed
firsthand at the node by the operator running this project, not a
secondhand or reconstructed number. Do not treat this figure as unverified
or lacking provenance in any future analysis (an earlier automated
research pass flagged it as "no benchmark command or model provenance
attached" - that flag is WITHDRAWN; the figure stands as confirmed).
Separately confirmed via source-code analysis
(`LLAMACPP-KERNEL-LIFT.md`'s "RESOLVED" section): llama.cpp's own dispatch
rule forces MMQ for `n_experts >= 64`, and MMQ on gfx1151/RDNA3.5 uses
WMMA, not DP4A - so this 80-95 t/s figure is llama.cpp's MMQ-WMMA path on
this exact hardware, matching what "should" produce a large advantage over
ds4's pre-fix DP4A-only prefill path.

ds4 is the BEST of the three at decode and **6.6x worse than vLLM at prefill**.
The engines have opposite optimisation profiles, but 6.6x is far more than my
identified bottlenecks explain.

## The MoE kernel is running at 6-12% of peak

`routed_moe` is 46.6% of prefill, ~714 ms per layer-chunk at 2186 tokens.

    FLOPs = 3 (gate,up,down) x 4096 x 2048 x 2 x 2186 tokens x 6 experts
          = 660 GFLOP
    achieved = 660 GFLOP / 714 ms = 925 GFLOP/s

**CORRECTED**: I first compared against fp32 peak (~14.8 TFLOP/s). Wrong
denominator - the kernel uses `v_dot4_i32_iu8`, whose peak on gfx1151 is
**59.4 TOPS** (full-rate VALU, 64 lanes/CU/clk x 4 MAC x 2, 40 CU @ 2.9 GHz).

    -> ds4's MoE achieves 925 GFLOP/s = 1.56% of DP4A peak, i.e. ~64x off.

Independent ISA analysis decomposes that into two ~8x factors: the 8-token tile
gives an arithmetic intensity of only 26.4 flop/byte (a 5.9 TFLOP/s ceiling for
that shape, of which the kernel reaches 16%), and the inner loop runs ~21
instructions per dot4 - 116 `v_movrel` from a non-unrolled runtime-trip-count
loop, 195 and/lshr from re-doing the Q4_K nibble unpack per token, and 0
`ds_read` despite 37 KB of LDS (a generic pointer forces flat loads).

And it is not bandwidth-bound either: touching all 128 owned experts is
1.8 GB per layer = **2.5 GB/s** against the 240 GB/s measured device read
bandwidth, i.e. ~1%.

**Neither compute-bound nor bandwidth-bound at 6% and 1% respectively** means
the kernel is latency/occupancy-starved. A well-tuned grouped GEMM reaches
40-60% of peak. `moe_gate_up_mid_q4K_expert_tile8_row32_kernel`
(rocm/ds4_rocm_moe.cuh:1754) and its down twin (moe.cuh:2374) are therefore
roughly **5-10x off**, which accounts for the whole 6.6x gap by itself.

## Consequence: my ranked fix list was the wrong SIZE

- the unowned-pair skip removes ~2x of *wasted* work in a kernel that is already
  5-10x off. It is real, but it is a constant factor on a bad constant.
- the big-gate host copy is 35% of prefill and worth ~1.4x. Also real, also not
  the dominant term.
- neither touches a kernel at 6% of peak.

vLLM's MoE is a grouped/ragged batched GEMM (two `matmul_ogs` calls per layer,
no per-expert loop) through tuned Triton/AITER kernels. ds4 hand-writes
per-expert tile kernels over sorted pairs. That difference - not parallelism,
not the transport, not the masking waste - is where the 6.6x lives.

## Open question this raises

Is ds4's prefill under-engineered, or does its design foreclose the fix? ds4
keeps weights resident and quantised and never dequantises a whole expert, which
is exactly what makes its DECODE the best of the three engines (10.5 t/s vs
llama.cpp's 9.42 and vLLM's 3.28). A dequantise-to-f16-then-batched-GEMM prefill
strategy may be incompatible with that residency design, or may simply be
absent. That is the question to answer before investing in MoE kernel work.


## The vLLM comparison is VALID (checked)

`V026-DECISION.md:287` records "Everything here was sequential and idle", so
198.8 tok/s (5.03 ms/prompt-token) is single-stream prefill, directly comparable
to our ~30. It is NOT the batched-concurrency number - that is a separate
162-181 tok/s figure from a different benchmark.

## MEASURED: narrowing the tile makes it WORSE

`DS4_ROCM_EXPERT_TILE_M` (patch 22) makes the width selectable. Both kernels
already exist in-tree; the launch branch is compile-time so only one is
instantiated.

| tile_m | LDS | occupancy | prefill |
|---|---|---|---|
| 4 | 18688 B | 75% | **27.10 t/s** |
| 8 (default) | 37376 B | 37.5% | **30.00 t/s** |

**Occupancy is not the binding constraint.** tile4 doubles the weight
re-streaming and that cost exceeds the occupancy gain. The kernel is
weight-streaming bound at the tile level.

Consequence: the fix is to go **wider**, not narrower - more tokens amortised
per weight read - which requires K-tiling to keep LDS bounded (staging the whole
K=4096 for the tile is what forces 37 KB today). Going wider without K-tiling
would blow LDS entirely.
