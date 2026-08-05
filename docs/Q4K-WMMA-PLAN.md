# Q4_K WMMA lift for ds4 — plan, corrected, ready to execute

Supersedes the "stop, don't port" conclusion from the first Stage -1 pass.
That conclusion was wrong - see "Why Stage -1 reversed" below. This doc
consolidates: the original llama.cpp-grounded staged plan, an independent
Opus review's integration/testing corrections, and the corrected Stage -1
evidence. Read alongside [[dp4a-wmma-proxy-model-trap]] and
docs/WMMA-GAP.md, docs/LLAMACPP-KERNEL-LIFT.md, docs/MOE-KERNEL-ISA.md.

## Why Stage -1 reversed

First pass tested WMMA-vs-DP4A for Q4_K using a small dense proxy model
(Qwen2.5-0.5B, `mul_mat_q`, pp512/pp2048) on a llama.cpp checkout. Result:
DP4A 45-62% faster. Conclusion at the time: don't port WMMA into ds4.

That proxy was the wrong test. A sibling deployment (llama.cpp
`run-dsv4-huihui-q4k-2node.sh`) re-ran the same question on the REAL shape -
`test-backend-ops perf -o MUL_MAT_ID`, k=4096, m=2048, 256 experts, 6 used,
swept at batch {1,4,5,6,945,2048} (decode, all three DSpark verify depths,
both prefill chunk sizes) - no full model load, single node, isolated.

Result: **WMMA won at every batch size, growing to 2.5x at prefill-relevant
sizes (945, 2048)** - the opposite direction from the proxy, worst exactly
where the proxy predicted DP4A's biggest win. Mechanism: WMMA amortizes a
loaded weight tile across many output columns, so its advantage scales UP
with batch; a dense model's `mul_mat_q` shapes don't predict a 256-expert
`mul_mat_id` MoE's. Full numbers in [[dp4a-wmma-proxy-model-trap]].

ds4's Q4_K MoE kernel is the same shape family - same checkpoint
(DeepSeek-V4-Flash), same 256-expert/6-used routing, same K=4096 gate/up
dimension. **This is real supporting evidence that integer WMMA is
favorable for this model/hardware/shape family - but it does NOT transfer
quantitatively.** ds4's DP4A kernel is not a generic dense `mul_mat_q`
baseline: `moe_gate_up_mid_q4K_expert_tile8_row32_kernel` already shares
one weight-row read across up to 8 routed tokens and stages all 8
activations in LDS before the dot product (ds4_rocm_moe.cuh:1784,1798-1806)
- i.e. it already captures *some* of the same batch-reuse concept that
makes WMMA scale up with batch, just capped at 8 tokens instead of
llama.cpp's J-up-to-128 matrix tile. At typical prefill loads (945x6/256
~=22, 2048x6/256~=48 routed pairs per populated expert under uniform
routing - real distribution is skewed, confirm before trusting these), ds4
re-reads the weight row across roughly 3-6 tile8 workgroups where
llama.cpp's WMMA tile would cover it in one pass, so there is still
plausible large headroom - but **do not carry the measured 2.5x forward as
an expected ds4 result.** Same direction, uncertain magnitude - see "First
concrete action" below for the ds4-specific measurement this requires.

## What ds4 already has (precedent, not a template)

- Q4_K MoE: DP4A only, `moe_gate_up_mid_q4K_expert_tile8_row32_kernel`
  (ds4_rocm_moe.cuh:1754), calling `dev_dot_q4_K_q8_K_block8` (:321).
  Measured 1.56% of DP4A peak. Separately diagnosed root cause: activation
  pointers are generically typed, so LDS gets written but every read still
  compiles to `flat_load` instead of `ds_read` - zero `ds_read` in the
  disassembly despite 37,376 B of static LDS allocated and live (confirmed
  the caching branch executes for DS4_N_EMBD=4096, xq_blocks=16). Fix
  designed (compile-time `bool CACHE_XQ` kernel specialization, address a
  separate track from this WMMA port, expected gain modest: 0-10% kernel,
  low single digits end-to-end - it does not by itself explain the 1.56%
  gap). Track in parallel or first; it changes the DP4A baseline this
  port's gates should be measured against (see Opus corrections below).
- Q2_K / IQ2_K MoE: real rocWMMA **fp16** kernels
  (ds4_rocm_moe.cuh:3759,3905,4065,4144) - `v_wmma_f32_16x16x16_f16`,
  confirmed via disassembly, 83 instances, matrix_a/matrix_b fp16 fragments,
  float accumulator. This proves ds4's HIP build toolchain and launch/error
  conventions support WMMA kernels end-to-end. It does NOT prove the
  specific instruction family needed here: llama.cpp's Q4_K MMA path uses
  **integer** WMMA (`v_wmma_i32_16x16x16_iu8`), confirmed present and
  assembleable on this exact toolchain (ROCm 7.2.0, gfx1151) via a direct
  `llvm-mc` probe, but ds4 has zero instances of it today. Do not assume
  the fp16 kernels de-risk the integer path - they don't share fragment
  layout, LDS stride, or hazard handling.

## The port target: llama.cpp's real, hardware-tuned Q4_K MMA code

Confirmed by direct read of the local checkout
(~/Desktop/cc/models/llama.cpp-hyv3/ggml/src/ggml-cuda/):

- gfx1151 hits `amd_wmma_available()` (common.cuh:340, RDNA3/RDNA4 gate).
- RDNA3/4 dispatch selects the "rdna4" MMQ config table
  (mmq.cuh:225,244; table at mmq-config-rdna4.cuh:108). For Q4_K:
  **I=128** output rows, **J up to 128** activation columns (fallback
  specializations at J={16,32,64,128}), **256 threads / 8 wave32 warps**,
  **K=256** iteration (`MMQ_ITER_K`), Q8_1 SRAM layout, no stream-K.
- Q4_K loader `ggml_cuda_mmq_load_tiles_q4_K` (mmq-load-tiles.cuh:622):
  builds a row-strided Q8_1-compatible LDS tile; `x_qs` at tile base,
  `x_dm` after 2*MMQ_TILE_NE_K=64 int32 (mmq-load-tiles.cuh:629); packed
  Q4 nibbles split via `0x0F0F0F0F` into two eight-int halves (:651); 6-bit
  scale/min unpacked via `unpack_scales_q45_K`, stored as
  `half2(d*scale, -dmin*min)` (:680).
- Q8_1 SRAM stride = 2*32 quant ints + 2*(32/8) scale/min ints + 4 padding
  = 76 int32 (mmq.cuh:128,152). At I=128: X tile alone = 128*76*4 =
  38,912 B. Total dynamic LDS ~42-58 KiB depending on J (formula at
  mmq.cuh:1354).
- Shared dot `ggml_cuda_mmq_vec_dot_q8_1_q8_1_mma` (mmq.cuh:734, body at
  mmq-vec-dot.cuh:315,337): A-fragment `tile<16,8,int,input_layout>`,
  B-fragment same, C-fragment `tile<16,16,int,DATA_LAYOUT_J_MAJOR>`.
  **RDNA3-specific: `input_layout` is `DATA_LAYOUT_I_MAJOR_MIRRORED`**, not
  row-major (mma.cuh:90) - a mirrored 16x8 int fragment holds 8 int32/lane
  (mma.cuh:530); RDNA3's 16x8 loader does two 16-byte LDS copies per lane
  (mma.cuh:829), vs one on RDNA4 - i.e. **16 VGPRs/fragment-set on RDNA3
  vs 12 on RDNA4**, a real register-pressure premium already baked into
  the tuned config. `rows_per_warp`=16 fixed on AMD WMMA (mmq.cuh:179); at
  I=128/8 warps, each warp owns one 16-row output band.
- K=256 handled as two 128-value Q8_1 activation blocks: load, dot at
  K-offset 0, reload, dot at K-offset 32 int32 units (mmq.cuh:875).

vLLM's GGUF plugin (~/Desktop/cc/vllm-gguf-plugin, Apache-2.0, cloned
locally) is **not** a WMMA reference - confirmed zero WMMA instructions,
its own header says copied from llama.cpp commit b2899 (pre-WMMA). Useful
for two things only: an independent Q4_K dequant cross-check
(`vec_dot_q4_K_q8_1_impl_mmq`, vecdotq.cuh:353; raw indexing, :1159), and
a Stage 2 routing-metadata reference (`moe.cuh:558,573` - 8 warps, 8
routed columns, 128-row tile on ROCm). Also a second real precedent
(beyond ds4's own ggml attribution) that porting this exact kernel family
across the MIT/Apache-2.0 boundary with attribution is normal practice.

Licensing: llama.cpp MIT, ds4's LICENSE already carries ggml copyright
attribution (README.md:53-56 already documents adapting ggml quant
layouts/kernels). Porting more of the same family is precedented, not a
new question.

## Staged plan (Stage 0-5), with the Opus review's corrections folded in

**Recommendation, decisive: port/adapt llama.cpp's real Q4_K MMA code.**
Do not use ds4's Q2_K WMMA kernel as the primary structural template - it's
useful only for HIP build/launch conventions. Reusing llama.cpp's actual
tuned tile/warp/K parameters removes tile shape, warp count, K-iteration,
LDS padding, fragment layout, and scale/min staging as simultaneous
unknowns - the dominant source of risk in a from-scratch reimplementation.

### Integration facts (must hold before Stage 0 writes code)

1. **`cuda_block_q4_K`/`cuda_block_q8_K` are file-local typedefs inside
   `ds4_rocm.cu`** (lines ~65-83), not in a header - the `rocm/*.cuh`
   files are textual includes that depend on that translation-unit
   context (`ds4_rocm_moe.cuh:1-4` says so explicitly). A standalone
   Stage-0 bench file CANNOT just `#include` its way to the real struct -
   it must replicate the layout. Add `static_assert(sizeof(...)==144)`
   etc. on BOTH the bench copy and (at Stage 1) inside `ds4_rocm.cu` next
   to the real typedefs, so a future struct change is a build break, not
   a silent divergence.
2. **Build with the exact production `ROCM_CFLAGS`**
   (Makefile:43: `-O3 -ffast-math -g -fno-finite-math-only -pthread
   -D__HIP_PLATFORM_AMD__ --offload-arch=gfx1151`), plus `-Wall -Wextra`
   on top (missing from `ROCM_CFLAGS` but required by
   `QA_BEFORE_RELEASES.md`'s warning-free release gate). A bench built
   without `-ffast-math` measures a different FP-contraction regime than
   production and any tolerance derived from it is wrong.
3. **The activation format is a real third component, not a cosmetic
   adaptation.** ds4 uses Q8_K (per-256 scale, exact int16 `bsums`);
   llama.cpp's MMQ needs Q8_1 (per-32 fp16 scale+sum,
   `block_q8_1_mmq`/`MMQ_Q8_1_DS_LAYOUT_DS4`, mmq.cuh:27-46,81-83).
   Porting requires implementing `quantize_mmq_q8_1` (quantize.cu:458-549)
   as an explicit, separately-costed component. It also introduces a real
   precision-loss mechanism worth testing for: fp16 can't exactly
   represent a 32-element int8 sum above 2048, so add activation-extrema
   edge cases (all +127, all -127, alternating ±127) alongside the
   existing weight-extrema ones.
4. **The production kernel is fused, not a bare GEMM.**
   `moe_gate_up_mid_q4K_expert_tile8_row32_kernel` does gate+up+SwiGLU+
   routing-weight-multiply+scatter-write+quarter-warp-reduction in one
   launch. A fused gate+up WMMA kernel needs **two** accumulator sets
   live at once: at I=128/J=128, `tile<16,16,int,J_MAJOR>` = 8 int32/lane
   per matrix, `rows_per_warp`=16 -> 8 C-tiles/warp = **64 VGPRs for one
   matrix, 128 for gate+up fused** - before A/B fragments (16 VGPRs/set on
   RDNA3) and the epilogue. This is close to the register-pressure cliff.
   **Stage 0 must benchmark the fused dual-accumulator variant, not just
   a single GEMM** - a single-GEMM Stage 0 can pass its no-spill gate
   while the actually-needed fused kernel spills, and you won't find out
   until Stage 3.
5. **Occupancy desk-check (computable now, no code needed):** gfx1151 has
   128 KB LDS/WGP. ds4's current tile8 DP4A kernel uses 37,376 B -> 3
   workgroups/WGP -> 24/64 wave32 = 37.5% occupancy baseline
   (moe_launch.cuh:753-761). The llama.cpp-tuned config at I=128 needs
   38,912-57,856 B depending on J -> pinned to **2 workgroups/WGP = 25%
   occupancy at every J** - WORSE than today's baseline. Don't gate on
   "achieved >=2 workgroups" (that's always true here and would falsely
   PASS a regression) - gate on occupancy vs the 37.5% baseline, or a
   measured arithmetic-intensity increase that justifies going below it
   (AI goes from ~26 op/B today to ~455 op/B at I=128/J=128, a ~17x
   increase - genuinely the strongest argument for I=128 despite the
   occupancy hit, but state it as a real tradeoff, not a free win).
   Include I=64 as a first-class Stage-0 arm (not a fallback) - it hits
   50% occupancy at J<=64 per the same math.
6. **Correctness-check circularity**: llama.cpp and vLLM's Q4_K math are
   the same lineage (vLLM's header says copied from llama.cpp); ds4's own
   `dev_dot_q4_K_q8_K_block8` is also a ggml transcription. Three
   references from one lineage prove transcription fidelity, not
   correctness. Independent ground truth already exists in-repo:
   `tests/test_q4k_dot.c`'s `ref_dot()` - upgrade it from float to
   double, add a Q8_1 arm, tighten tolerance from 1% to something that
   would actually catch a bug. **Decisive missing test**: a
   scale-equalized equivalence mode - build Q8_1 activation blocks from
   ds4's existing Q8_K blocks with all four per-32 `d` forced equal to
   the Q8_K block `d`, sum computed exactly then fp16-rounded. In that
   mode WMMA and DP4A are algebraically identical up to float
   association, so max-rel-error should be <1e-5 - without this, a real
   bug hides inside the "expected" quantization-format divergence. Use a
   combined absolute+relative criterion (`|a-b| <= atol + rtol*|b|`), not
   a bare relative error - a raw relative gate is undefined/unstable near
   zero and several accumulator terms here legitimately approach zero.
7. **Gate math must derive from measured shares, using the correct
   formula.** Fractional end-to-end time saved by a component speedup is
   `share * (1 - 1/speedup)`, NOT `share * (speedup - 1)`. `PREFILL-PROFILE.md`
   measures routed_moe at 46.6% of prefill: to hit +5% e2e you need
   routed_moe >=1.12x (`1/(1-0.05/0.466)`), not 1.08x (which only yields
   +3.6% and is a hard fail against a 5% Stage 4 target). If only gate+up
   ships (down stays DP4A, ~31.1% of prefill), gate+up needs >=1.20x
   (`1/(1-0.05/0.311)`), not 1.18x. Set Stage 2/3 gates from these
   corrected numbers.
8. **DP4A baseline may move underneath this work, AND is not a naive
   baseline to begin with.** Two separate cautions: (a) the LDS-pointer
   bug fix (see "What ds4 already has" above) is a real, separate change
   to the comparison baseline - report Stage 0 speedup against BOTH
   shipping DP4A and a best-effort-fixed DP4A (unpack hoisted, LDS
   pointer correctly typed). (b) ds4's `tile8_row32` DP4A kernel already
   shares one weight-row read across up to 8 tokens and stages them in
   LDS (ds4_rocm_moe.cuh:1784,1798-1806) - this is NOT a generic dense
   `mul_mat_q` baseline like the one that showed a naive 2.5x WMMA win on
   a different (dense, non-MoE) model. **Do not assume Stage 0's WMMA
   margin will resemble that 2.5x figure** - same direction is supported
   by evidence, magnitude is not. Compare tile4 vs tile8 DP4A first
   (Stage -0A below) to know how strong the real baseline actually is
   before setting expectations.
9. **ISA gates must be mechanical, not "seems reasonable"**: per
   specialization, specify and check exactly: max VGPR/SGPR count, zero
   scratch/spill bytes, static+dynamic LDS bytes, predicted
   workgroups/WGP, the exact expected `v_wmma_i32_16x16x16_iu8` count per
   unrolled K-iteration and total per output tile (not just ">0" - an
   unexpectedly low count silently means partial scalar fallback), and a
   positive `ds_read`/`ds_load` count **in the kernel body specifically**
   (not merely present somewhere in the compiled object).
10. **Stage -0C - toolchain sanity check, done right, and done FIRST,
    separate from Stage 0's performance work**: do NOT use ds4's existing
    Q2_K WMMA kernel as a "does WMMA work at all" smoke test - it's fp16
    WMMA, a different instruction family, and would give a false green
    light. Instead: a ~50-line standalone integer-WMMA fragment-identity
    test (fill A with a ramp, B with identity, load via llama.cpp's
    mirrored-layout helpers verbatim, call
    `__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32`, assert C==A elementwise,
    disassemble and assert instruction count) - catches a misunderstood
    mirrored layout, wrong LDS stride, or silent scalarization before any
    Q4_K code exists. One afternoon. This is a correctness/toolchain
    check, not a real-shape performance measurement - keep it out of
    Stage 0's go/no-go numbers.

### Stage 0 - standalone faithful port

Branch `q4k-wmma-stage0-llamacpp-port`. New file only:
`ds4-strix-halo-tp/scripts/q4k_wmma_mmq_bench.cu`. Do not touch
`ds4-upstream/` production dispatch (ds4-upstream itself is an
uncommitted git-apply patch stack on top of pinned `54b36ed` - see
[[dp4a-wmma-proxy-model-trap]]'s git-checkout warning; back up any file
before editing, never `git checkout --` to undo).

Contents: narrowly-adapted copy of llama.cpp's Q4_K MMA loader, shared
Q8_1xQ8_1 dot, RDNA3 mirrored-fragment helpers, scale-unpack helper, with a
comment naming source checkout/commit/files/MIT license. Fixed Stage-0
specializations at llama.cpp's real RDNA3 config (I=128 AND I=64 as
first-class arms). DP4A baseline using the same harness (both shipping and
unpack-hoisted variants, correction 8). vLLM-derived scalar reference kept
separate (correction 6). Fused dual-accumulator gate+up variant benchmarked
alongside the single GEMM (correction 4). Scale-equalized equivalence mode
(correction 6) plus weight- and activation-extrema edge cases (corrections
3, and the existing weight-extrema list: all-zero nibbles, all-15, 6-bit
scale extrema, zero d, zero dmin). Non-power-of-two bucket sizes (17, 33,
96, 129) alongside M={8,16,32,64,128,256} - dump the real per-expert
`sorted_counts` histogram at the production prefill chunk size first to
confirm the tested M range matches actual routing before trusting the
benchmark shape.

Report TWO metrics, not one: **core** (fused gate+up kernel only) and
**pipeline** (Q8_1 quantization + sorting/tiling + fused gate+up). The
go/no-go decision uses the pipeline metric; core timing is diagnostic
(explains why pipeline passed or failed, e.g. quantization overhead
eating the kernel win). M means routed pairs per expert, not global token
batch - state this explicitly in the results table.

Go: correctness passes at all three levels including the scale-equalized
1e-5 (combined abs+rel) check; no spills in the fused dual-accumulator
variant; wave occupancy justified per correction 5; ISA gates per
correction 9; pipeline metric beats BOTH DP4A baselines (correction 8) by
>=15% geomean across M=16-256, >=1.25x at EACH of M=32/64/128 individually
(a miss at any one of the three is Stop, not partial credit).
Stop: <5% or negative vs either baseline at the pipeline metric.
Investigate-once: 5-15%, don't over-invest even if hit (two prior kernel
bets in this project regressed despite looking right on paper at this
stage).

### Stage 1 - ds4-native kernel shell (no routing yet)
Branch `q4k-wmma-stage1-ds4-shell`. Move the proven Stage-0 core into a
ds4-style header, preserve attribution/algorithm, adapt pointer types,
stream handling, static_asserts into `ds4_rocm.cu` (correction 1). State
tolerance up front (not "bitwise identical" - `-ffast-math` reassociation
will differ between a standalone TU and ds4's TU). Within 5% of the
corresponding Stage-0 I/J specialization's perf (not vs shipping DP4A -
I=128 is already expected to drop from 37.5% to 25% occupancy, so "same
occupancy as DP4A" is not the right comparison here), same no-spill
condition. This stage adds a test-only dispatch shell to exercise the
ds4-native code paths; the `DS4_ROCM_DISABLE_Q4K_WMMA=1` kill switch and
real production dispatch begin at Stage 3, not here - Stage 1 has no
routing to switch between yet.

### Stage 2 - routed expert-tile microbenchmark
Branch `q4k-wmma-stage2-routed`. This is the first production-shape stage;
keep it in the harness until the routing contract and crossover are proven.
Implement in this order:

**Items 1-2 status (2026-08-04): DONE, validated on real hardware.** New
harness `ds4-strix-halo-tp/scripts/q4k_wmma_routed_bench.cu` copies ds4's
real CSR routing builders verbatim (`moe_count_sorted_pairs_kernel`,
`moe_prefix_sorted_pairs_kernel`, `moe_build_expert_tile_offsets_kernel`,
`moe_build_expert_tiles_kernel`, plus the deterministic-scatter variant
ds4 actually launches - `moe_rocm_moe.cuh:988-1082`), does one Q8_1
conversion per unique token, and dispatches the validated WMMA gate/up
GEMMs across the full multi-expert tile stream (not one bucket at a
time). Validated against the shipping `moe_gate_up_mid_q4K_expert_tile8_row32_kernel`
across 4 routing patterns (balanced, skewed, tiny-tail, single-expert,
up to 256 experts): routing+guards exact match, GEMM-only max_abs
0.019-0.029 (matching the already-validated single-expert noise floor),
fused epilogue (clamp+SwiGLU+routing-weight+pair-major scatter) passes
with margins 0.046-0.050 using a closed-form tolerance derived from
SwiGLU's actual error-propagation math (not a flat guess - see the
harness source for the derivation: `|ΔM| <= |w*SiLU(g_w)|*|Δu| +
|w*u_r|*|SiLU(g_w)-SiLU(g_r)|`, floored at the baseline atol so no
element loses noise budget). One real bug was found and fixed along the
way (a test-harness clamp mismatch between the WMMA and reference
launches, not a kernel bug), plus two iterations to get the epilogue
tolerance derivation right (first pass under-covered near-zero-gate
elements by scaling atol to zero instead of flooring it).

**Item 3 status (2026-08-04): DONE, validated on real hardware.** Added
`wmma_min_count` early-return to the routed WMMA kernel and a new
harness-only `q4k_dp4a_cold_tile16_kernel` (verbatim shipping kernel left
untouched) with the complementary threshold check, both launched
unconditionally over the full tile stream every time - no host branching,
no D2H counts copy. Validated with per-output double-write markers (WMMA
adds 1, DP4A adds 2 - catches missing writes, cross-path double writes,
and duplicate same-path writes) and device-side skip/process counters
checked against independently-computed expected hot/cold tile counts, on
a 5th "mixed" routing case. The original item-3 validation used expert
counts {1,3,4,8,16,40} at the then-provisional threshold=4: counters
exactly matched expectation (6 hot/2 cold), zero marker
violations. Measured dual-unconditional-launch overhead vs an idealized
pre-compacted dispatch at thresholds {2,4,8,16}: overhead is ~0%
(-0.2 to -0.6%, i.e. within noise) - the device-side compaction/prefix-sum
optimization this item's text flagged as a possible follow-up is not
needed. The provisional `wmma_min_count=4` guess came from Stage-0
isolated-GEMM measurements (bucket 4 cleared 2.7x there); it is explicitly
superseded by the full routed-pipeline result below.

**Item 4 status (2026-08-04): DONE, validated on real hardware.** The 13-case
correctness sweep covers balanced/ad-hoc skew, the explicitly synthetic
Zipf `p(i) proportional to 1/i^1.2` proxy (not production telemetry), empty
experts, exact 1/7/8/9/15/16/17 boundaries, repeated selections across a
WMMA tile boundary, and all 256 experts active. Routing, guards, independent
CPU gate/up/mid references, crossover ownership markers, and process/skip
counters all pass.

The validated gate/up crossover threshold is `wmma_min_count=6`, superseding
the earlier provisional value 4. The five-launch routed pipeline (one Q8_1
conversion, gate WMMA, up WMMA, DP4A fallback, and epilogue) has an observed
~30.5 us fixed overhead floor from buckets 4-9. That cost is not amortized at
4-5 pairs/expert even though those GEMMs looked favorable in isolation. The
timings below are the threshold=4 measurement sweep that located the break-even;
the final column applies the selected threshold=6 retention rule:

| pairs/expert | crossover us | shipping DP4A us | speedup | retained at 6 |
|---:|---:|---:|---:|:---:|
| 1  | 27.519 | 11.480 | 0.4172x | no |
| 4  | 30.519 | 23.320 | 0.7641x | no |
| 5  | 30.480 | 27.279 | 0.8950x | no |
| 6  | 30.559 | 31.159 | 1.0196x | yes |
| 7  | 30.599 | 35.159 | 1.1490x | yes |
| 8  | 31.399 | 39.079 | 1.2446x | yes |
| 9  | 31.680 | 41.719 | 1.3169x | yes |
| 15 | 32.959 | 42.239 | 1.2816x | yes |
| 16 | 33.160 | 42.639 | 1.2859x | yes |
| 17 | 33.239 | 42.759 | 1.2864x | yes |
| 32 | 33.360 | 48.399 | 1.4508x | yes |
| 64 | 34.800 | 93.558 | 2.6884x | yes |

Bucket 6 is the empirical break-even point: its 1.0196x result is not a
greater-than-5% regression, and all larger measured buckets improve from
1.1490x through 2.6884x. Buckets 4-5 now correctly remain on DP4A and are
excluded from the retained-WMMA geomean. The down-projection track has since
closed below with its own independently measured crossover threshold.

**STAGE 2 GATE: PASS (2026-08-04, confirmed on real hardware at
`wmma_min_count=6`).** Final retained-bucket sweep {6,7,8,9,15,16,17,32,64}:
`retained_geomean=1.3616x` (required >=1.20x), `per_bucket_regression=PASS`
(zero retained buckets exceed the 5% regression bound), full correctness
suite `PASS` across all 13 cases. This closes Stage 2's own go/no-go
decision for gate/up. The independent down-projection gate is recorded below;
with both gates passed, Stage 3 is the first step that touches ds4-upstream/
itself.

1. Reproduce ds4's existing device-built CSR contract exactly, rather than
   inventing a WMMA-only routing format: `sorted_pairs` contains
   `token*n_expert+slot`, `offsets[expert]` starts the expert bucket,
   `counts[expert]` is its live length, and `tile_experts[tile]` plus
   `tile_starts[tile]` identifies a tile within that bucket. The production
   builders are `moe_count_sorted_pairs_kernel`,
   `moe_prefix_sorted_pairs_kernel`,
   `moe_scatter_sorted_pairs_deterministic_kernel` (the atomic
   `moe_scatter_sorted_pairs_kernel` remains defined but is not launched),
   `moe_build_expert_tile_offsets_kernel`, and
   `moe_build_expert_tiles_kernel` in `rocm/ds4_rocm_moe.cuh`; their launch
   sequence is in `rocm/ds4_rocm_moe_launch.cuh:955-994`. The shipping Q4_K
   gate/up consumer is `moe_gate_up_mid_q4K_expert_tile8_row32_kernel`
   (`ds4_rocm_moe.cuh:1754-1839`). Preserve its pair decoding, routing-weight
   lookup, clamp, SwiGLU, and pair-major `mid_out` scatter semantics.
2. Add a routed Q8_1 conversion over the live input-token rows once, then
   launch the two already-measured integer-WMMA GEMMs (gate and up) against
   the same converted activation. Add the production epilogue only after
   both routed GEMMs match independently. The measured two-launch pipeline is
   the Stage-2 target; do **not** block integration on a new dual-accumulator
   kernel.
3. Make crossover entirely device-resident. Build tiles for all experts as
   today, pass `counts` and a measured `wmma_min_count` to both paths, and
   have the WMMA tile return when `counts[expert] < wmma_min_count` while a
   DP4A tile specialization returns when it is at or above the threshold.
   Thus two fixed launches partition the same `tile_experts`/`tile_starts`
   stream without a host decision or compacted hotlist. If the extra empty
   workgroups matter, add a device compaction/prefix-sum as a later
   optimization, still without D2H. Do not copy the Q2 hotlist selection:
   `routed_moe_q2_float_down_launch` synchronously copies `counts` D2H at
   `moe_launch.cuh:225`, and the IQ2 gate hotlist repeats that pattern at
   `moe_launch.cuh:1008` before copying a host-built list H2D at :1020. That
   per-layer synchronization is unacceptable in the 86-gate TP loop.
4. Sweep balanced, Zipf/skewed, and recorded-production histograms across
   0-256 pairs/expert, including empty experts, 1/7/8/9/15/16/17-sized
   buckets, final partial WMMA tiles, repeated token/expert selections, and
   all 256 experts. Compare routed IDs, routing weights, gate, up, SwiGLU
   mid, and guard regions against the shipping DP4A path and the independent
   CPU reference. Measure conversion + both GEMMs + epilogue + fallback
   launch overhead, not isolated WMMA time.

Start Q4_K down-projection after item 1 freezes the shared CSR/tile contract;
that prerequisite is now satisfied by the Stage-2 gate/up PASS. Implement and
measure it in a separate `scripts/q4k_wmma_down_routed_bench.cu`, reusing the
validated builders and integer-WMMA core without changing the gate/up harness:

1. Fix the Flash track's GEMM shape at **M = expert-bucket pair count,
   K = 2048, N = 4096**. These are not inferred from the gate/up shape:
   `DS4_SHAPE_FLASH.n_ff_exp` is 2048 and `.n_embd` is 4096
   (`ds4.c:535-554`), while the actual layer derives `expert_mid_dim` from
   `ffn_gate_exps->dim[1]` and the routed output dimension from
   `ffn_down_exps->dim[1]` (`ds4.c:21531-21539`). Keep the harness constants
   explicit and assert the production weight layout is down rows of K=2048,
   N=4096. Thus each pair has 8 Q8_K blocks and, after repacking, 16 Q8_1
   128-value blocks; the integer-WMMA K loop is half the gate/up K=4096 loop,
   while the N sweep is 4096 output rows.
2. Quantize the **pair-major FP32 post-SwiGLU `mid`**, not the original token
   input. The shipping gate/up kernel writes
   `mid_out[pair*expert_mid_dim+row] = SiLU(gate)*up*routing_weight`
   (`ds4_rocm_moe.cuh:1831-1837`). Production then launches
   `q8_K_quantize_kernel(midq, mid, expert_mid_dim, pair_count)`
   (`ds4_rocm_moe_launch.cuh:1387-1390`); that quantizer consumes one FP32 row
   and produces ordinary Q8_K blocks with `d`, `qs`, and `bsums`
   (`ds4_rocm_moe.cuh:483-529`). The new harness must reproduce that FP32 ->
   Q8_K step verbatim, then reuse the proven Q8_K -> Q8_1 conversion. This is
   a two-kernel conversion pipeline, not a direct conversion of token Q8_K.
3. Preserve both pair output and final-combine semantics. The shipping tile8
   fallback indexes `midq` by `pair`, selects weights by the CSR tile's expert,
   and either writes `down[pair*out_dim+row]` or atomically adds it to
   `out[token*out_dim+row]` (`ds4_rocm_moe.cuh:2390-2433`). Routing weights are
   already folded into `mid`; down must not multiply them again. In the normal
   non-atomic path production follows the down launch with `moe_sum_kernel`
   over the six pair rows (`ds4_rocm_moe_launch.cuh:1620-1624`); for
   `n_tokens >= 128`, `use_atomic_down` is selected
   (`ds4_rocm_moe_launch.cuh:771`) and the output is zeroed before atomic
   accumulation (`ds4_rocm_moe_launch.cuh:1483-1486`). First validate
   pair-major WMMA and DP4A results independently against a scalar CPU
   Q4_K-by-quantized-mid reference, then validate the six-expert token sum.
   Cover the atomic combine as a separate semantic case; do not mix atomic and
   pair-major writers in one crossover run.
4. Use the shared device-resident `sorted_pairs`/`offsets`/`counts` and
   `tile_experts`/`tile_starts` stream. The shipping tile8 consumer is
   `moe_down_q4K_expert_tile8_row32_kernel`
   (`ds4_rocm_moe.cuh:2374-2435`), launched at
   `ds4_rocm_moe_launch.cuh:1490-1505`. The structural fp16 precedents remain
   `moe_down_q2K_hotlist_wmma_kernel` and
   `moe_down_q2K_hotlist_wmma_n2_kernel`
   (`ds4_rocm_moe.cuh:4064-4141,4143-4278`), with current launches at
   `ds4_rocm_moe_launch.cuh:314-420`; use their bucket/tail and paired-N store
   organization only. Take Q4_K unpacking, Q8_1 layout, RDNA3 mirrored
   fragments, and scale/min correction from `q4k_wmma_routed_kernel`.
5. Give down its own `down_wmma_min_count`, initialized only as an unevaluated
   sweep variable--never to gate/up's 6. Sweep thresholds at least
   `{1,2,4,6,8,12,16,24,32}` over the same balanced, Zipf/skewed, recorded,
   empty, boundary, tail, repeated-selection, and all-expert cases. Time the
   complete candidate pipeline: FP32-mid -> Q8_K, Q8_K -> Q8_1, one down WMMA
   GEMM, complementary DP4A launch, and pair-to-token sum (or zero + atomic
   combine in its separate case). Compare against FP32-mid -> Q8_K, the
   unmodified shipping tile8 DP4A launch, and the same combine. Report every
   bucket, retained geomean, any >5% retained-bucket regression, and the
   unconditional dual-launch versus ideal-compacted overhead exactly as in
   gate/up; select the lowest threshold that passes the existing >=1.20x
   geomean/no-regression gate. K=2048, N=4096, one GEMM, the extra FP32 -> Q8_K
   launch, and combine cost make this a new empirical breakeven.

**Down-projection status (2026-08-04): DONE, validated on real hardware.**
The independent scalar CPU Q4_K-by-Q8_1 reference accumulates 68 floating-point
terms per output (64 block products plus 4 scale/min correction terms). Its
forward-error comparison uses the standard cancellation-aware bound
`gamma_67 = 67*u/(1-67*u)`, where `u` is the binary32 unit roundoff, and accepts
`|gpu-cpu| <= 2*gamma_67*sum_abs_terms`. Here `sum_abs_terms` is accumulated in
double by `down_host_q4k_q81`; the factor 2 covers the two independently rounded
evaluation paths without replacing cancellation sensitivity with a flat
absolute tolerance. On failure, the comparator reports pair, expert, row, GPU
and CPU values, absolute difference, ULP distance, `sum_abs_terms`, and bound
ratio. Real gfx1151 validation produced zero mismatches and full `PASS` across
all 13 cases.

The selected independent dispatch threshold is `down_wmma_min_count=1`. It is
the lowest tested threshold and retains the most measured buckets (9), with a
`10.7201x` retained geomean and no retained bucket regressing by more than 5%.
Every higher tested threshold also passes, but retains fewer buckets; its larger
geomean merely reflects filtering out smaller wins rather than a better dispatch
policy. There is no empirical reason to raise the threshold for safety: bucket
1 itself is measured and passes, and no smaller nonempty expert bucket exists.
Real routing and integrated single-layer behavior remain Stage-3 validation
concerns, not grounds to discard a demonstrated Stage-2 win.

| `down_wmma_min_count` | retained buckets | retained geomean | regression >5% | gate |
|---:|---:|---:|:---:|:---:|
| 1  | 9 | 10.7201x | no | PASS |
| 2  | 8 | 12.7086x | no | PASS |
| 4  | 8 | 12.6612x | no | PASS |
| 6  | 7 | 13.7636x | no | PASS |
| 8  | 6 | 14.5784x | no | PASS |
| 12 | 5 | 15.3713x | no | PASS |
| 16 | 4 | 15.4989x | no | PASS |
| 24 | 3 | 15.0178x | no | PASS |
| 32 | 2 | 16.3214x | no | PASS |

**STAGE 2 DOWN-PROJECTION GATE: PASS (2026-08-04, confirmed on real
hardware at `down_wmma_min_count=1`).** The threshold-1 sweep retains 9
buckets with `retained_geomean=10.7201x` (required >=1.20x),
`per_bucket_regression=PASS` (zero retained buckets exceed the 5% regression
bound), and the full 13-case correctness suite passes with zero gamma-bound
mismatches. This is a separate down-projection go/no-go decision; it neither
inherits nor changes the gate/up threshold or result.

There is no format or routing blocker: production confirms Q8_K pair-major
mid activation and CSR-compatible pair outputs. The useful Q2_K precedent is
structural only because it dequantizes Q2_K to half and uses fp16 rocWMMA.
Keep gate/up and down enable bits and thresholds independent.

Gate: exact routing/scatter semantics, no padded or cross-expert writes,
per-GEMM correctness at the Stage-0 tolerance and fused-path tolerance no
worse than the validated harness, gate/up geomean >=1.20x for buckets retained
on WMMA, and <=5% regression for any retained bucket. Down has its own
go/no-go table and does not inherit gate/up's result. Both independent Stage-2
gates are now closed PASS; Stage 2 as a whole is complete.

### Stage 3 - single-layer MoE integration
Branch `q4k-wmma-stage3-single-layer`. Port only the Stage-2 winners into
`rocm/ds4_rocm_moe.cuh` (integer-WMMA helpers/kernels) and wire their launches,
Q8_1 scratch sizing, and device-side threshold predicates into
`rocm/ds4_rocm_moe_launch.cuh`. Reuse the existing routing builders and scratch
arrays; do not add a second host routing pass. Land gate/up first. Down can
land in the same stage independently once its Stage-2 gate passes; failure of
down must leave the faster gate/up-only configuration shippable, and each
projection must have a separately logged enable bit and threshold.

Dispatch policy, in order: supported gfx1151 integer-WMMA build and exact
shape/alignment; `DS4_ROCM_Q4K_WMMA=1`; no
`DS4_ROCM_DISABLE_Q4K_WMMA=1` (disable wins); `!g_quality_mode`; negotiated
TP feature enabled; then the per-expert device-side crossover. This follows
the existing quality convention actually visible in the Q2 paths:
`routed_moe_q2_float_down_launch` includes `!g_quality_mode` at
`moe_launch.cuh:242`, while IQ2 gate hotlist eligibility and its f16-mid path
check it at :1005 and :1030. Unsupported shapes and `--quality` take the
shipping Q4_K DP4A path without changing output layout.

TP negotiation has an existing mechanism to extend; it is **not** a new
transport. `ds4_tp_identity` in `ds4_tp.h:39-56` is exchanged before inference
by `tp_hello_exchange` (`ds4_tp.c:1405-1455`), whose fixed hello already
fail-closes on protocol/model mismatch. Add a `runtime_features` bitmask to
`ds4_tp_identity` and `ds4_tp_hello_fixed`, include a
`DS4_TP_FEATURE_Q4K_WMMA` bit representing the final effective state after
env/quality/hardware checks, compare it for equality in `tp_hello_exchange`,
and bump `DS4_TP_PROTOCOL_VERSION` (currently 7 at `ds4_tp.c:48`). Populate
the bit in both leader construction (`ds4_cli.c:2131`) and worker construction
(`ds4_tp.c:2194`) through one backend query, before `ds4_tp_create`. A mismatch
must abort TP startup with both bitmasks in the error; it must never silently
choose per-rank paths. Log rank, negotiated gate/up/down bits, thresholds,
quality state, and kill-switch state once at startup, not per layer. Add hello
unit tests for equal-enabled, equal-disabled, and mismatched feature masks.

Validate one real Q4_K layer on each rank and after the TP sum: exact routed
IDs/weights, shard boundaries, empty/tail experts, gate/up/mid, optional down,
and final output versus forced-DP4A. Test opt-in, quality fallback, kill switch,
unsupported shape, and deliberately mismatched rank environments. Gate:
MoE-layer speedup >=1.12x when gate/up+down ship together; if only gate/up
ships, require its corrected >=1.20x component target and show >=5% projected
e2e saving from measured shares. Decode stays within ±3%, no OOM/scratch
growth failure, and the kill switch restores the old kernels without rebuild.

The optional single-kernel dual-accumulator gate+up refinement comes **after
Stages 2 and 3**, and may be deferred indefinitely. Two launches sharing one
Q8_1 conversion already measured 6.6x at bucket 22 and 9.9x at bucket 48,
comfortably clearing the integration gates. Fusing earlier would reintroduce
the known 128-accumulator-VGPR pressure/spill risk while routing, TP safety,
and Q4_K down remain the real unimplemented work. Revisit only if integrated
profiles show launch overhead or duplicate weight-side setup materially limits
end-to-end gain; it must then beat the two-launch production path and pass the
same correctness/ISA gates.

## CRITICAL: DS4_ROCM_Q4K_WMMA=1 produced corrupted output, measured 2026-08-05

First real end-to-end test since Stage 3 landed (`e906a14`). Full 2-node TP,
OdinLink RDMA, `DS4_TP_BIG_DIRECT=1`, `DS4_ROCM_Q4K_WMMA=1`:

    ds4: ROCm Q4_K WMMA startup rank=0 negotiated=0x00000001 gate=1 up=1 down=0 ...
    ds4: prefill: 36.54 t/s, generation: 11.38 t/s

**Both throughput numbers look healthy - prefill comparable to the best
BIG_DIRECT-only run, decode even slightly above the ~10.5 baseline.** The
actual generated text was `<｜begin▁of▁sentence｜>` repeated for the entire
output, from the very first generated token - no coherent prefix at all. This
is the historical corruption signature (`CORRUPTION-BISECT.md`: "first token
already wrong"), not a degenerate-but-coherent repetition loop like the
Chinese/English loops seen elsewhere in this session's testing.

**This is the dangerous case: if you only checked t/s, this reads as a clean
win.** It is not. Do not trust throughput numbers alone for this feature.

**Isolated: this is WMMA-specific, not a BIG_DIRECT/RDMA interaction.** Same
test over plain `--transport tcp`, no BIG_DIRECT, same `DS4_ROCM_Q4K_WMMA=1`:

    ds4: ROCm Q4_K WMMA startup rank=0 negotiated=0x00000001 gate=1 up=1 down=0 ...
    ds4: prefill: 39.32 t/s, generation: 10.46 t/s

Identical corruption (`<｜begin▁of▁sentence｜>` repeated from the first
generated token, same negotiated `gate=1 up=1`), again with healthy-looking
throughput on both axes. Do not enable
`DS4_ROCM_Q4K_WMMA=1` in any deployment until this is root-caused and a
correctness check (exact output match against forced-DP4A, not just "does it
run") passes. The kill switch (`DS4_ROCM_DISABLE_Q4K_WMMA=1`) is confirmed
present and is the safe state until this is resolved.

Next diagnostic: the WMMA kernel likely has a real numerics/fragment-layout
bug independent of transport - check the standalone Stage 2 correctness
harness (`q4k_correctness_test.cu` per docs history) still passes in
isolation; if it does, the bug is specifically in the Stage 3 production
integration (routing/scratch/dispatch wiring), not the kernel itself.

### Stage 4 - opt-in end-to-end prefill
Branch `q4k-wmma-stage4-prefill-optin`. Run only with the negotiated opt-in
enabled on both ranks. Profile gate/up conversion, both WMMA launches,
epilogue, down conversion/kernel (when enabled), DP4A fallback buckets, and TP
wait separately on **both** ranks. Use the >=3306-token prompt, production
945/2048 prefill chunks and verify depths, >=3 samples per arm, identical
inputs, warmups, and non-overlapping distributions. Record real per-expert
`counts` histograms and confirm each measured threshold against them; compare
gate/up-only, gate/up+down, and forced-DP4A arms so down's incremental value is
not hidden. Also run decode, long repeated sessions, `--quality`, kill switch,
and a negative TP feature-mismatch startup test.

Gate: >=5% end-to-end prefill, >=8% aggregate routed-MoE reduction, no decode
regression outside the Stage-3 ±3% band, no numerical/stability regression,
and no rank-asymmetric path or new host synchronization in traces. A down path
that misses its incremental gate is disabled without delaying gate/up.

### Stage 5 - default-on decision
Default-on only after two independent gfx1151 TP=2 machines reproduce Stage
4, including skewed real routing distributions and repeated long runs. Change
the negotiated default bit, not merely a local dispatch default. Keep
`DS4_ROCM_DISABLE_Q4K_WMMA=1` as the permanent fail-closed rollback and retain
`--quality` as forced DP4A. Non-gfx1151 and unsupported shapes must compile and
fall back safely.

Release/CI matrix: single-rank and TP=2; equal enable/equal disable/mismatched
feature hello; gate/up-only and gate/up+down; thresholds on both sides of the
crossover; empty/tail/skewed experts; quality and kill switch; numerical parity
and guard regions; scratch/OOM failure fallback; and a check that the routed
path contains no D2H counts copy. Review llama.cpp attribution and the integer
WMMA ISA/no-spill gates. Freeze separate gate/up and down thresholds from
measured **expert bucket counts**, never global batch size, and document the
two-launch gate/up kernel as the supported default unless the optional fused
variant later proves an end-to-end win.

## First concrete action - split into two (do this before Stage 0 code)

The original single "reproduce llama.cpp's test on ds4's shape" action
conflated two different questions. Confirmed by checking ds4's actual
repo: no existing tool substitutes for either. `q4k-dot-test`
(Makefile:400) is a CPU-only correctness test, compiled with the host
compiler - it cannot time the ROCm kernel (test_q4k_dot.c:138,175).
`ds4-bench` (Makefile:70) opens the full model/engine
(ds4_bench.c:577) and times whole-session prefill around sync points
(:683) - not an isolated kernel harness. The MoE event profiler
(`DS4_ROCM_MOE_DECODE_PROFILE`) only fires at `n_tokens==1`
(moe_launch.cuh:1980) - decode-only, no prefill bucket sweep. The kernel
itself is `static` inside a textual include tied to the production
translation unit (ds4_rocm_moe.cuh:1) - a standalone harness cannot link
to it directly and must copy it with layout `static_assert`s (matching
Integration fact 1's guidance).

**Stage -0A - ds4 baseline characterization (do this first, cheapest,
answers the ds4-specific question the reversal doesn't resolve by
itself):** a minimal synthetic harness
(`ds4-strix-halo-tp/scripts/q4k_dp4a_baseline_bench.cu`) that copies ONLY
the production Q4_K/Q8_K layouts (with static_asserts) and the current
tile4/tile8 DP4A kernel + helpers - no WMMA yet. Measure kernel-only and
full-routed-pipeline time separately, at buckets
{1,4,5,6,8,16,17,22,32,33,48,64,96,128}, both tile4 and tile8, K=4096,
2048 output rows, balanced and skewed 256-expert/6-used routing, against
both shipping DP4A and (once it exists) the LDS-pointer-fixed DP4A. This
answers: how much reuse does tile8 already capture, where does it
saturate, how strong is the real baseline - directly resolving the
"same direction, uncertain magnitude" question from Integration fact 8,
before spending effort on a WMMA implementation.

Build once GPU access is confirmed:
```
/opt/rocm/bin/hipcc -O3 -ffast-math -g -fno-finite-math-only -pthread \
  -D__HIP_PLATFORM_AMD__ -Wno-unused-command-line-argument \
  --offload-arch=gfx1151 -Wall -Wextra \
  ds4-strix-halo-tp/scripts/q4k_dp4a_baseline_bench.cu \
  -o /tmp/q4k_dp4a_baseline_bench -lm -pthread
```
(exact production flags, Makefile:43, plus -Wall -Wextra per Integration
fact 2). CLI/sweep controls to be designed into the file before running.

**Stage -0A: DONE, measured 2026-08-04.** Harness built at
`ds4-strix-halo-tp/scripts/q4k_dp4a_baseline_bench.cu`, compiled clean with
the exact production flags above, run on real gfx1151 hardware. Results
(median of 20 timed iterations after 5 warmup, HIP events, one populated
expert isolating pure kernel cost):

| bucket | tile4 (us) | tile8 (us) | tile4/tile8 |
|---|---|---|---|
| 1 | 81.80 | 76.12 | 1.075x |
| 4 | 227.67 | 190.43 | 1.196x |
| 5 | 382.03 | 231.07 | 1.653x |
| 6 | 475.94 | 272.39 | 1.747x |
| 8 | 307.43 | 356.71 | 0.862x (tile4 wins - anomaly, not yet explained) |
| 16 | 724.22 | 651.78 | 1.111x |
| 17 | 779.98 | 683.70 | 1.141x |
| **22** (~945x6/256, prod bucket) | 1134.93 | **830.21** | **1.367x** |
| 32 | 1838.18 | 1170.48 | 1.570x |
| 33 | 1922.82 | 1200.64 | 1.601x |
| **48** (~2048x6/256, prod bucket) | 2817.27 | **1691.63** | **1.665x** |
| 64 | 3926.68 | 2189.05 | 1.794x |
| 96 | 5770.94 | 3142.90 | 1.836x |
| 128 | 7698.16 | 4132.31 | 1.863x |

At the two production-realistic bucket sizes, tile8 already delivers a
real 1.37-1.66x speedup over tile4 purely from LDS-staged reuse, growing
with bucket size. **Confirms Integration fact 8's caution empirically**:
ds4's shipping DP4A baseline is genuinely stronger than a naive
mul_mat_q-style kernel, so Stage 0/-0B's WMMA margin should be judged
against THIS baseline, not extrapolated from the 2.5x seen on llama.cpp's
plain dense-proxy comparison. The bucket=8 anomaly (tile4 beats tile8)
is unexplained and worth a note if it recurs - possibly a tile-boundary
effect (8 pairs = exactly 1 tile8 workgroup vs 2 tile4 workgroups) rather
than noise, but not investigated further here since it's off the
production-relevant bucket path.

Not yet done from the full Stage -0A spec: skewed-routing variant, and
comparison against the LDS-pointer-fixed DP4A (that fix is designed but
not yet implemented - see "What ds4 already has" above). Add both before
treating this baseline as final.

**Stage -0B - first real WMMA-vs-DP4A comparison:** NOT achievable with a
DP4A-only harness - requires the smallest faithful fused WMMA arm. This is
effectively the early portion of Stage 0 itself, not a separate step
before it. Stage -0A informs how much margin Stage -0B/Stage 0 needs to
clear to be meaningful (if tile8 already captures most of the reuse
benefit, the WMMA bar is higher than if it captures little).

Zero-GPU-access prep available right now: extract the exact dependency
closure for `dev_dot_q4_K_q8_K_block8` and the tile4/tile8 kernels; draft
(don't compile) the Stage -0A harness structure and result schema;
precompute expected bucket distributions from 945x6 and 2048x6 routing
(clearly marked as uniform-distribution estimates, not measured); finish
the per-I/J LDS/VGPR/expected-WMMA-count/occupancy table from
Integration fact 5; decide whether the LDS-pointer DP4A fix lands before
or after this benchmarking (Integration fact 8 requires both baselines
either way - don't imply the fixed one already exists if it doesn't yet).

**Stage -0B status (2026-08-04, updated): kernel written, compiles,
correctness now CONFIRMED after one Codex debug round.** Codex found the
bug: the RDNA3 `DATA_LAYOUT_J_MAJOR` accumulator decode had row/token
coordinates reversed when selecting weight/activation metadata and
writing the result - every WMMA value was being scaled and stored to the
wrong output element. Fix: `row = wave*16 + 2*l + lane/16`,
`token = j0 + lane%16` (previously swapped). After the fix, rerun on real
hardware across all 14 buckets:
- Realistic per-block-scale mode (atol=5e-2, rtol=5e-3, appropriate for
  real Q8_K->Q8_1 fp16 format conversion noise): 8 bad out of ~1.06M
  compared elements (>99.999% pass), max_abs 0.03-0.07.
- Scale-equalized mode showed a much higher nominal fail count at first
  (e.g. bucket=1: 1563/2048), which looked alarming, but inspecting
  actual (wmma, ref) value pairs showed errors of the same ~0.01-0.03
  absolute magnitude as the realistic-mode pass - i.e. the SAME small,
  expected fp16-rounding noise, just measured against too tight a
  tolerance (1e-4). Equal Q8_1 sub-block scales do not make the (d, d*sum)
  fp16-pair encoding bit-exact, since the *sum* term still needs to round
  to fp16 precision regardless of how "nice" the scale is - this was a
  flawed assumption in the test harness's equalized-mode tolerance, not a
  kernel bug. Confirmed by direct value inspection, not just aggregate
  counts.
- Timing is essentially unchanged from the buggy version (the fix only
  changed output attribution, not the amount of WMMA/LDS work done), so
  the previously-reported core-timing ratios are now a genuinely
  validated result: 12-25x faster than shipping DP4A tile8 core time
  across buckets 4-128, ~10-11x even including full Q8_K->Q8_1 conversion
  overhead at production buckets 22/48. Still "diagnostic, not
  apples-to-apples" per the harness's own banner (WMMA computes one GEMM;
  tile8 is fused gate+up+SwiGLU) - real end-to-end gain will be smaller,
  but this clears the Stage 0 gate thresholds (>=1.20x gate+up-only,
  >=1.12x full routed-MoE share) by a wide margin.
- Not yet done: I=128 (only I=64 tried; J=64/128 spilled registers and
  were dropped from the sweep), and real (non-uniform, non-random)
  routing-derived activation data.

**Fused gate+up+SwiGLU apples-to-apples result (2026-08-04):** added a
`time_wmma_pipeline()` arm - two independent-weight WMMA GEMMs (gate, up)
sharing one Q8_1 conversion, combined with the same SwiGLU formula tile8
applies internally (`swiglu_weight_kernel`), timed as one unit and
compared against tile8's single fused launch (the thing that actually
ships). Real hardware result across all 14 buckets:

| bucket | dp4a8_us | pipe2_us (2xWMMA+conv+swiglu) | dp4a8/pipe2 |
|---|---|---|---|
| 1 | 76.20 | 69.72 | 1.093x |
| 4 | 190.35 | 70.20 | 2.712x |
| 5 | 229.83 | 70.92 | 3.241x |
| 6 | 272.99 | 71.12 | 3.839x |
| 8 | 358.87 | 75.68 | 4.742x |
| 16 | 659.34 | 82.56 | 7.986x |
| 17 | 683.58 | 115.48 | 5.920x |
| 22 (prod) | 823.26 | 124.16 | 6.631x |
| 32 | 1175.52 | 133.76 | 8.789x |
| 33 | 1203.60 | 167.63 | 7.180x |
| 48 (prod) | 1690.75 | 171.59 | 9.853x |
| 64 | 2205.29 | 181.55 | 12.147x |
| 96 | 3192.98 | 310.59 | 10.280x |
| 128 | 4161.87 | 372.99 | 11.158x |

This clears the Stage 0 gate thresholds (>=1.20x gate+up-only share,
>=1.12x full routed-MoE share) by a wide margin at both production
buckets (22, 48). Correctness on the fused path: 18,020 bad out of
983,040 compared elements (~1.83% fail rate) at atol=5e-2/rtol=1e-2 -
much better than the pre-fix ~98% failure, but not as clean as the
per-GEMM check's >99.999%. Likely SwiGLU's nonlinearity occasionally
amplifying already-tiny (~0.03-0.07 abs) per-GEMM noise past the chosen
tolerance, not a new logic bug (the two underlying GEMMs are
independently validated), but not yet root-caused - treat the fused
timing number as trustworthy and the fused correctness margin as "good
but not yet fully explained" pending a closer look.

Caveats unchanged: synthetic/uniform routing data, I=64 not I=128,
J=64/128 unavailable (register spill).

Superseded status text below (kept for the record of what looked wrong
and why, per the correctness-discipline standing constraint - do not
delete debugging history that explains a reversal):
Codex wrote a real I=64 integer-WMMA arm (`q4k_dp4a_baseline_bench.cu`,
ported from llama.cpp's mmq-load-tiles.cuh/mma.cuh RDNA3
`DATA_LAYOUT_I_MAJOR_MIRRORED` fragment scheme) plus a same-input
correctness check against the shipping DP4A kernel. First run reported
`bad=0, max_abs=0` across every bucket - a false positive: a shared bug
(`uint16_t d = __float2half_rn(f)` performs a VALUE conversion, not a bit
reinterpret, silently zeroing every synthetic q4_K block's d/dmin in
BOTH the WMMA harness and the separately-written correctness-test file,
`q4k_correctness_test.cu`) made every dot product compute 0 on both
arms, so the "exact match" was 0 vs 0. Fixed in both files with a
`f16_bits()` helper (half -> memcpy -> uint16_t). After the fix:
- `q4k_correctness_test.cu` (independent double-precision CPU reference
  vs the single-pair `dev_dot_q4_K_q8_K_block`): genuinely passes 210/210
  with real nonzero values agreeing to ~6 significant digits - ds4's
  shipping DP4A dot product is confirmed correct.
- The WMMA arm in `q4k_dp4a_baseline_bench.cu`: badly wrong. ~95-99% of
  output elements fail tolerance at every bucket, in BOTH the
  scale-equalized exact-match mode (max_abs ~120-265, should be ~0) and
  the realistic-scale mode. This is a real logic bug in the fragment
  load / RDNA3 mirrored-layout / scale-unpacking code, not a precision
  artifact - Codex flagged exactly this risk ("I could not verify the
  resulting lane-to-matrix mapping numerically without GPU execution").
  The dp4a4_us/dp4a8_us timing columns are unaffected (kernel timing
  doesn't depend on data correctness, same instruction shape either way)
  but the WMMA "14-25x" core-timing ratio is NOT yet meaningful, since
  it's timing a kernel that isn't computing the right answer - could
  still change once correctness is fixed (more or fewer WMMA/LDS ops
  needed). Next step: debug the WMMA kernel's fragment/scale layout
  against the now-fixed, genuinely-working oracle in
  `q4k_correctness_test.cu` before trusting any WMMA timing number.

## Probability estimate

Two separate numbers, not one - the reversal resolves different amounts
of risk for each:

- **Stage 0 correctness + core-performance gate passes: 65-80%**
  (up from the original ~40-65% estimate). Raised materially because the
  same model/hardware/real-MoE-shape has now demonstrated a decisive WMMA
  advantage - not just "resolved in principle" but measured on this exact
  checkpoint and hardware family. Not raised further because ds4's
  stronger tile8 DP4A baseline, the fused dual-accumulator register
  pressure, the Q8_K->Q8_1 conversion cost, and the I=128 occupancy drop
  are all separate, ds4-specific risks the llama.cpp measurement says
  nothing about.
- **Reaching a default-on, >=5% end-to-end result (Stage 4/5): 35-55%.**
  Stays meaningfully lower than the Stage 0 number because a kernel-level
  win must still survive routing, quantization, tail buckets, device-side
  crossover dispatch, the fused epilogue, TP-handshake consistency across
  two ranks, and Amdahl's-law dilution from routed_moe being only 46.6%
  of prefill (corrected gate math, Integration fact 7).

Revise the first number after Stage -0A/-0B; revise the second only once
Stage 2's pipeline-metric timing exists - Stage 0 alone cannot inform it.
