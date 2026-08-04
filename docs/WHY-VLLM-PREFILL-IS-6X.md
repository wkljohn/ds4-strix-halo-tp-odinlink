# vLLM does 198.8 tok/s prefill on this hardware; we do ~30. Why.

Measured on the SAME two Strix Halo gfx1151 nodes, same model family:

| engine | prefill | decode |
|---|---|---|
| **vLLM TP=2** | **198.8 tok/s** | 3.28 t/s |
| llama.cpp (pipeline) | 80-95 tok/s | 9.42 no-draft |
| **ds4 TP=2 (this port)** | **~30 t/s** | **~10.5 t/s** |

ds4 is the BEST of the three at decode and **6.6x worse than vLLM at prefill**.
The engines have opposite optimisation profiles, but 6.6x is far more than my
identified bottlenecks explain.

## The MoE kernel is running at 6-12% of peak

`routed_moe` is 46.6% of prefill, ~714 ms per layer-chunk at 2186 tokens.

    FLOPs = 3 (gate,up,down) x 4096 x 2048 x 2 x 2186 tokens x 6 experts
          = 660 GFLOP
    achieved = 660 GFLOP / 714 ms = 925 GFLOP/s

gfx1151 fp32 peak is ~14.8 TFLOP/s at 40 CU / 2.9 GHz (~7.4 if the probe's
`multiProcessorCount = 20` is CUs rather than WGPs - the uncertainty does not
change the conclusion).

    -> ds4's MoE achieves 6% (40 CU) to 12% (20 CU) of fp32 peak.

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
