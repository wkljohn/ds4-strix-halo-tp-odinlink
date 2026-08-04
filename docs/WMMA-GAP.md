# ds4 already has WMMA MoE kernels — just not for Q4_K

Found while the comparative review was running. This is the most concrete
statement of what our prefill lacks.

## The kernels that exist

    rocm/ds4_rocm_moe.cuh:3759  moe_gate_up_mid_iq2_hotlist_wmma_n2_kernel
    rocm/ds4_rocm_moe.cuh:3905  moe_gate_up_mid_q2K_hotlist_wmma_n2_kernel
    rocm/ds4_rocm_moe.cuh:4065  moe_down_q2K_hotlist_wmma_kernel
    rocm/ds4_rocm_moe.cuh:4144  moe_down_q2K_hotlist_wmma_n2_kernel

| quant | WMMA MoE kernels |
|---|---|
| IQ2 | 1 |
| Q2_K | 3 |
| **Q4_K (what our model uses)** | **0** |
| Q8_0 | 0 |

They are properly templated on real matrix-core tiles - launched as
`<4,16,16,16,...>` and `<16,16,16,16,...>` (moe_launch.cuh:314-347).

## The gate is NOT decode-only

    const int use_wmma_hot = hot_experts_dev &&
        !g_quality_mode &&
        (expert_mid_dim % 16u) == 0u && (out_dim % 16u) == 0u;   // moe_launch.cuh:241

**No `n_tokens` condition.** It needs a hot-expert list and 16-alignment, and our
model satisfies the alignment (expert_mid_dim 2048, out_dim 4096). So the WMMA
machinery is batch-agnostic and dimensionally compatible with prefill; the only
thing missing is a Q4_K implementation.

## Why this matters

Our Q4_K prefill kernel `moe_gate_up_mid_q4K_expert_tile8_row32_kernel` uses
`v_dot4_i32_iu8` (DP4A) and reaches 925 GFLOP/s = **1.56% of the 59.4 TOPS DP4A
peak**, with 4.8% of its instructions doing useful arithmetic. WMMA-FP16 peak on
gfx1151 is ~59.4 TFLOPS and WMMA-INT8 ~118.8 TOPS - i.e. the matrix cores are
2-4x the vector path even before accounting for the instruction overhead we
measured.

Both faster engines use matrix cores for prefill: vLLM through Triton/AITER,
llama.cpp through its WMMA-MMQ path on RDNA3. **ds4 uses them for Q2_K/IQ2 and
not for Q4_K.** That, not parallelism and not the transport, is the most likely
single explanation for prefill being 29.5 vs 80-95 vs 198.8 t/s.

## Caveats before treating this as the plan

- The existing kernels are "hotlist" shaped - they consume `hot_experts_dev` and
  a `hot_count`, a different dispatch from the sorted-pair/expert-tile route our
  Q4_K path takes. Porting is not a copy; the dispatch has to be reconciled.
- Q4_K's superblock layout (144 B per 256 weights, per-sub-block scales/mins) is
  more complex to feed into a 16x16x16 tile than Q2_K's. The dequant has to
  happen in-register per tile, which is exactly what our current kernel fails to
  do (it re-unpacks per token).
- Two kernel-level predictions have already failed in the wrong direction here
  (tile narrowing, loop templating). This one should be prototyped and MEASURED
  before being believed.
