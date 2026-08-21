# External Q4_K / DeepSeek V4 decode survey for DS4 TP=2 gfx1151

Date: 2026-08-20
Scope: ordinary greedy decode, **no DSpark** on the headline. DSpark/DGX-Spark papers are survey material only.
Hardware target: 2× Ryzen AI MAX+ 395 (gfx1151 / RDNA3.5), TP=2, cache-free GGUF Q4_K, OdinLink or mlx5 RoCE v2.

Pinned local trees inspected:

- DS4: `ds4-release-main-20260820` @ `2e7210a`
- llama.cpp: `llama.cpp-upstream-latest` (`ggml/src/ggml-cuda/mmq-config-rdna3-5.cuh`, `vendors/hip.h` RDNA3_5)
- vLLM blog + recipes: DeepSeek V4 2026-04-24
- AITER/ATOM: https://github.com/ROCm/aiter and https://github.com/ROCm/ATOM (2026-08 README)
- Prior DS4 research: `docs/DECODE-RESEARCH-AND-PLAN-2026-08-06.md`, `research-results/q4k-step0-2026-08-17/` (Steps 0–26), `docs/Q4K-WMMA-PLAN.md`

## 1. llama.cpp — GGUF Q4_K on RDNA3 / RDNA3.5 / gfx1151

### What they actually run

llama.cpp has a separate **MMVQ** (one-token matvec) and **MMQ** (batched / prefill) family. gfx1151 is `RDNA3_5`, not RDNA3_0. Tile tables live in `mmq-config-rdna3-5.cuh`. Q4_K uses `GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1` and `MMQ_ITER_K` with I/J/nwarps combinations such as 128×2×64×16 and 256×2×128×64/128.

Native packed integer dots use `__builtin_amdgcn_sudot4` (`common.cuh`). Integer WMMA `v_wmma_i32_16x16x16_iu8` is assembleable on gfx1151 (DS4 already proved this for Q4_K **prefill** WMMA).

### gfx1151-specific findings (do not copy RDNA4 tables)

- Issue [#21284](https://github.com/ggml-org/llama.cpp/issues/21284): inefficient MMQ defaults on Strix Halo Q4_K. Extra warps that help discrete RDNA3/RDNA4 **regress gfx1151** because LPDDR5X saturates and reduction overhead grows.
- PR [#19478](https://github.com/ggml-org/llama.cpp/pull/19478): architecture-specific MMVQ warp tables; gfx1151 often wants **one wave**.
- PR [#26301](https://github.com/ggml-org/llama.cpp/pull/26301) **MMVDQ**: dequant Q4_K/Q5_K/Q6_K weights to float, keep activations F32, packed `float2`/`float4` loads. Defaults on for RDNA3.5. Mean +3.09% / median +2.78% over 38 models. **Large MoE is ~+0.2%** (Qwen3.5-35B-A3B). Gains shrink with model size; this is a small-model bandwidth play, not a 284B MoE lever.
- Vulkan vs ROCm on gfx1151 (r/LocalLLaMA, Qwen3.6-35B-A3B Q6_K): Vulkan tg128 51.2 vs ROCm 42.3. Backend observation; DS4 HIP kernels do not inherit it.
- AMD-validated llama.cpp binaries exist for gfx1150/gfx1151 (ROCm 7.1.1). rocWMMA long-context crash guards (VEC fallback for missing tiles) matter for **long context prefill**, not 2K greedy decode.

### Transfer to DS4

Transferable: shape-gated packed sudot4/WMMA, F32 activation reuse, **do not add waves**. Not transferable: MMQ dispatcher, `mmq_x/mmq_y/nwarps` tables as-is, Vulkan, MMVDQ as the first 25 t/s project.

## 2. vLLM — DeepSeek V4 (CUDA-first; ROCm is Instinct)

vLLM's DeepSeek V4 path ([blog 2026-04-24](https://vllm.ai/blog/2026-04-24-deepseek-v4), PR 40760) is the best public decode-graph description:

1. **Fusions (elementwise, 1.4–20× on those kernels):** compressor+RMSNorm+RoPE+cache insert; inverse RoPE+fp8 quant; fused Q-norm+KV RoPE+K insert. DS4's matching stages are already 3–11 µs; fusion here is residual 1% unless a new trace shows otherwise.
2. **Multi-stream overlap** of indexer vs compressor vs SWA insert: **5–6% e2e at low batch on CUDA**. ROCm auxiliary-stream PRs were closed without gfx1151-class validation. DS4's shared/routed overlap oracle was **neutral** (same UMA channel).
3. **CUDA graphs / FULL_AND_PIECEWISE.** DS4 gfx1151 HIP-graph probe: legacy default stream cannot capture; optimistic 32-node replay saved <2 µs/chain.
4. **Hybrid KV / c4a / c128a / DSA top-k.** Long-context cost control, not 2K headline. DS4 already has temporal compressor batching and online softmax.
5. **FP4/FP8 MoE, FlashMLA, FlashInfer, MegaMoE.** Wrong tensor format vs GGUF Q4_K.
6. **HIP W4A16** (PRs 41394, 44075): skinny-decode packed vector kernels for GPTQ/AWQ. Design evidence (separate decode kernel family, wave-local reduction, shape dispatch). Layout is not GGUF Q4_K.
7. **QuickReduce quantized collectives:** low-concurrency regressions when forced. DS4 transport is already ~11% of an older gate interval and posting is ≤0.2 µs today.

ROCm vLLM on gfx1151: community patches exist (kyuz0 toolboxes, `on_gfx9()` gates, AITER shuffle KV missing gfx1151 tables). vLLM issue 37151: gfx1151 HSA segfaults. Not a drop-in decode stack for this GGUF TP=2 service.

## 3. AITER / ATOM (Instinct gfx942/gfx950; experimental RDNA4)

[ROCm/aiter](https://github.com/ROCm/aiter) production kernels: fused MoE, FlashAttention, block-scaled FP8/FP4 GEMM, ASM/CK/Triton. Target MI355X/gfx950. AMD-fork gfx1151 enablement is vLLM FP8/AWQ, **not GGUF Q4_K**.

[ROCm/ATOM](https://github.com/ROCm/ATOM) (2026-08):

- DP Attention + TP MoE + Two-Batch Overlap (AG/RS) for DeepSeek V4 on **MI355X**.
- Piecewise CUDA graphs; FP8 KV; MORI all-to-all EP.
- **DeepSeek-V4-Pro DSpark** (2026-07): semi-autoregressive block drafter, confidence-scheduled ragged verification, FP8 KV, DP attention. **Headline of this campaign excludes DSpark.**
- Experimental **Navi4 / gfx1201** (RDNA4) only, not gfx1151.
- Prefill/decode disaggregation via Mooncake RDMA — different problem than 86 lockstep TP gates on 2 APUs.

Transfer: TBO/microbatch overlap is a **batch>1** idea. DS4 batch-1 UMA overlap already failed. Keep ATOM's "retain verifier hidden rows in a position-addressed cache" note for a later DSpark campaign only.

## 4. DGX Spark / GB10 (NVIDIA-only; survey, not a port)

Public DGX Spark DeepSeek V4 Flash reports (2026-07/08):

- 2× DGX Spark + DSpark: ~**49 tok/s** agent traffic, 1M ctx, 6-way concurrency (vLLM + B12X **MXFP4** fused MoE).
- Single GB10: 2-bit ~81 GiB + DSpark.
- vLLM recipes: DSpark is a **speculative module**, not new target weights. GB10 overlay, NVFP4, CUDA graphs, EP.

None of this maps to gfx1151 GGUF Q4_K. Useful only as: (a) proof that 25+ t/s on Flash is possible **with NVFP4 + speculation + datacenter interconnect**, (b) reminder that DSpark's verifier hidden-state contract is the hard part (DS4 already measured batch vs decode RMS 2.29). **Do not import Spark kernels.**

## 5. RDNA 3.x (discrete gfx110X) vs RDNA 3.5 (gfx1151)

| Lever | Discrete RDNA3 | gfx1151 Strix Halo |
|---|---|---|
| Extra MMVQ/MMQ warps | Often faster | Measured regression (shared LPDDR5X) |
| WMMA integer iu8 | Yes | Yes; DS4 uses it for Q4_K **prefill** |
| sudot4 | Yes | Yes; DS4 Q8 decode pack4 / DP4A |
| HIP graphs | Sometimes | Capture refused on DS4 legacy default stream; launch-only <2 µs |
| Aux streams | Mixed | DS4 UMA contention → zero net |
| Bandwidth | GDDR, higher | ~195.6 GiB/s effective LPDDR5X |

Copying RDNA3/RDNA4 "more waves / more tiles" is a known gfx1151 footgun.

## 6. Strix Halo (this machine) — DS4 current position

Already in production (cache-free, no DSpark): packed Q8 attn-out low/expand/Q-B; greedy top-2; staged Q4_K activations; host-callback row gates; temporal compressor batching; Q4_K WMMA **prefill**; IQ2 I8 WMMA; skip-unowned tiles; online softmax; RoCE-only prefill wavefront (gated off on OdinLink).

Measured decode stages (post-temporal, one token, ~19.4 t/s era):

| Stage | Time |
|---|---|
| Q/KV projection | 40–41 µs |
| Q_B | 85 µs |
| indexed mixed attention | 78–79 µs |
| attn-out low / expand | 90–91 / 99–100 µs |
| routed gate/up + epilogue | 140–156 µs |
| routed down | 74–81 µs |
| FFN gate wait skew (mean paired max) | 138 µs (expert-count imbalance, r=0.984) |

86 real graph boundaries. Transport floor ~6 µs paired min. Wire is not the 12 ms gap.

## 7. Prior DS4 attempts — what can flip

| Attempt | Result | Flip now? |
|---|---|---|
| Packed Q8 pack4 | Shipped, +5–8% each | Done |
| Activation DP4A (once rejected) | Later shipped for Q8 | Done; do not reverse |
| Temporal compressor batch | Shipped, 17.86→19.37 | Done |
| Temporal side-stream overlap | Same-token consumer; closed | **No** |
| HIP graphs | <2 µs; capture refused | **No** |
| Extra `MOE_MID` row-shard | Exact, extra callback ate the win (210.59/19.29 vs 219.56/19.37) | **Arithmetic yes / extra gate no** |
| Shared-expert half move | +0.03 t/s | **No** |
| Q_B prefetch / shared-routed overlap / Q-B fusion | Neutral or fingerprint break | **No** |
| MMVDQ (llama.cpp 26301) | Large-MoE +0.2% | **No as first project** |
| More waves | gfx1151-negative | **No** |
| AITER/vLLM kernel ports | Wrong GPU/format | **No** |
| Persistent Q8→F16 cache | Decode-neutral, 10 GiB | **No** |
| Transport pollers / opcode | Neutral/negative | **No** |
| One-token Q4_K MoE kernel (Fable fallback) | Never given a post-19.37 isolated 10% oracle | **Open** |
| Row-shard without 3rd host callback | Open design | **Open if kernel cannot close 25** |
| Long-context WMMA / grouped indexed attn | Grouped 4-head was 3.7× slower | **Open only at ≥32K** |
| Device-resident routing / D2H `h_counts` | Residual | After a fresh trace |

## 8. Long context (required measurement, not the 2K headline)

vLLM/ATOM long-ctx wins are: packed hybrid KV, fused compressor epilogue, DSA top-k, FP8 KV. DS4 already batches temporal compressor at ratio-4/128. Remaining long-ctx lever is **indexed attention core** (78 µs at 2K; grows with selected compressed tokens). llama.cpp RDNA3.5 WMMA flash-attention tiling and AOTriton gfx1151 FA are algorithm references, not drop-ins (DS4 has custom CSA/HCA/indexer/sink contract). A 4-head staged KV candidate was 3.7× slower and non-bitwise — do not revive that geometry.

## 9. Arithmetic toward 25 t/s

19.22 t/s → 25.0 t/s needs **12.0 ms/token**.

- Experts-only roofline 23.3 t/s = 42.9 ms. 25 t/s = 40.0 ms, **slightly above** that ceiling.
- FFN imbalance pool ~5.9 ms/token (max recoverable without new bytes).
- One-token MoE ~220 µs/layer × 43 ≈ 9.5 ms. A 20% kernel cut is ~1.9 ms.
- Even stacking kernel + all jitter (without extra gates) is ~7–8 ms, landing near **22–23 t/s**, not 25, unless a new byte division is real.

Honest combination (to be approved by Codex before any arithmetic commit):

1. Isolated one-token Q4_K gate/up/down oracles vs current production kernel (sudot4/sub-wave pack, RDNA3.5 tile geometry on existing WMMA/DP4A bodies; MMVDQ control allowed to lose).
2. Only if (1) cannot close the gap: row-shard **without** a third `hipLaunchHostFunc` gate (async mid write, wait at existing FFN gate).
3. Long-context: measure ≥32K decode on the same commit; do not change 2K arithmetic for FA tiling unless the 2K fingerprint still matches.

Sources: llama.cpp PRs 19478/26301, issue 21284; vLLM DeepSeek V4 blog/PR 40760, W4A16 PRs 41394/44075; ROCm AITER; ROCm ATOM README 2026-08; NVIDIA DGX Spark public logs 2026-07; DS4 step0 Fable reviews 1–6.

## 10. Scope addendum after Fable review — Q2_K, CK/hipBLASLt, Qwen3.8, ds4-on-spark

Date: 2026-08-21. This addendum closes the explicit survey gaps called out by
the user objective and Fable's follow-up review. It does not approve a new
implementation stage.

### Q2_K / IQ2 decode prior art

DS4's current Q2 path is not "just fewer bytes" Q4_K. The production fork
already has `docs/Q2-I8-WMMA.md`: compact IQ2/Q2 gate/up codebook groups expand
directly into tile-local INT8, reuse the dynamic Q8_1 activation stream, and use
native gfx1151 integer WMMA. That is why STAGE0 already records Q2_K RoCE decode
near Q4_K (`19.14 t/s`, fingerprint `f9cb3a8a17e95c71`) rather than a dramatic
decode jump from the smaller file.

Transfer from llama.cpp Q2_K is likely smaller than for Q4_K:

- llama.cpp's RDNA3.5 public optimization wave is dominated by Q4_K/Q5_K/Q6_K
  MMQ/MMVDQ and MoE tiling notes, not a visible gfx1151 Q2_K one-token decode
  breakthrough.
- Generic reports that Q2_K is not always faster than Q4_K are consistent with
  DS4's measurements: decode remains shaped by activation movement, expert
  scheduling, launch/gate cadence, and extra unpack/codebook work, not just model
  byte count.
- Therefore Q2_K is a regression-gate and measurement target for this campaign,
  not the first place to seek 21 t/s for Q4_K.

Required before any future promotion: re-run Q2_K on RoCE v2 and OdinLink with
candidate mode and fingerprint `f9cb3a8a17e95c71`. The current branch's default-
off changes have been validated mostly on Q4_K+RoCE; Q2_K and OdinLink are still
measurement gaps.

### Composable Kernel / CK

Primary source: <https://github.com/ROCm/composable_kernel> and ROCm's CK
documentation describe CK as a general framework for performance-critical HIP
C++ kernels. Current CK changelog material includes int8 GEMM support and richer
GEMM epilogues.

Transfer verdict for DS4 no-DSpark TP=2 decode: low.

- CK is useful as a design reference for generated GEMM, tensor coordinate
  transforms, epilogue composition, and int8 GEMM coverage.
- It does not provide a drop-in GGUF `Q4_K` / `Q2_K` routed-MoE one-token kernel
  with DS4's block scales/mins, SwiGLU association, TP expert ownership, and
  fingerprint-exact arithmetic.
- Batch-1 decode is a GEMV/MoE scheduling problem on Strix Halo UMA; a generic
  GEMM library path would first require repacking or dequantization buffers, which
  violates the campaign's "no persistent expanded cache" default unless proved
  tile-local and fingerprint-safe.

Reopen condition: only if a small standalone CK int8/GEMM microbench on gfx1151
beats the current DS4 tile-local kernel at the exact routed-MoE shapes and can
express Q4_K/Q2_K scales without a persistent conversion copy.

### hipBLASLt

Primary sources: <https://github.com/ROCm/hipBLASLt> and ROCm hipBLASLt docs
describe a flexible GEMM API centered on `hipblasLtMatmul`. Separate ROCm issue
reports also show gfx1151 hipBLASLt support/selection has been fragile across
ROCm versions.

Transfer verdict for DS4 no-DSpark TP=2 decode: near zero for the current goal.

- hipBLASLt is a dense GEMM library. DS4's bottleneck is cache-free GGUF
  one-token routed MoE with quantized expert tensors, sparse expert routing, TP
  rank ownership, and exact self-regression fingerprints.
- Using hipBLASLt directly would require either expanded F16/BF16/INT8 matrices or
  a custom prepack step. A persistent expanded-weight path is explicitly banned
  as a default because it previously consumed about 10 GiB per rank and did not
  improve decode.
- hipBLASLt remains relevant to dense BF16 drafter or future non-GGUF baseline
  tests, not to Q4_K/Q2_K production decode.

### Qwen3.8-27B optimization work

Primary sources: vLLM recipe page for `Qwen/Qwen3.8-27B`, Qwen's model card, and
community Strix Halo/DGX Spark reports. The public story is dominated by:

- dense 27B, hybrid/linear attention, built-in MTP draft head;
- NVFP4 / mixed-int4 / Blackwell-oriented vLLM recipes;
- Strix Halo reports where MTP can multiply apparent generation speed when the
  draft head has high acceptance.

Transfer verdict for this no-DSpark DeepSeek V4 Q4_K/Q2_K TP=2 goal: low to
moderate, mostly as a *separate baseline*.

- Qwen3.8's dense architecture and draft-head behavior are not DS4's sparse
  DeepSeek V4 routed-MoE verifier path.
- The most attractive Qwen speedups are MTP/speculative or NVFP4 format wins.
  The current campaign explicitly excludes DSpark/speculation from the headline
  and uses GGUF Q4_K/Q2_K.
- The useful transfer is methodological: benchmark with a fixed harness, publish
  acceptance/fingerprint gates, and keep MTP/speculation in a separate lane from
  ordinary greedy decode.

Required for the user's "three models" research request: create a later
baseline-only matrix for DeepSeek V4 Q4_K/Q2_K plus Qwen3.8-27B on the same
hardware, but do not use Qwen MTP numbers as evidence for no-DSpark DS4 decode.

### ds4-on-spark / DGX Spark details

Primary sources: <https://github.com/Entrpi/ds4-on-spark>, NVIDIA DGX Spark forum
threads, and DGX Spark community recipes. ds4-on-spark is valuable prior art, but
its speed path differs materially from this fork:

- Blackwell CUDA, not gfx1151 ROCm;
- serving engine work: continuous batching, prefix caching, warm starts, fork-by-
  copy for agent branches, disk KV persistence, OpenAI-compatible server;
- DSpark/speculative decode and Blackwell/NVFP4 or B12X-style low-precision
  kernels;
- single-node or high-bandwidth NVIDIA/RDMA assumptions, not two Strix Halo APUs
  with GGUF Q4_K/Q2_K over OdinLink/RoCE TP gates.

Transfer verdict:

1. High for benchmark presentation and serving ergonomics.
2. Medium for future optional DSpark lane: acceptance-aware verifier scheduling,
   prefix/KV lifecycle, and "cannot make slower" gating.
3. Low for current no-DSpark Q4_K/Q2_K kernel work; do not port CUDA kernels or
   publish speculative numbers as ordinary greedy decode.

## 11. Current honest next action

Do not implement another production optimization on this branch until a new
mechanism has a demonstrated ceiling and a fresh gate approves it. The next
work is measurement-only:

| Required measurement | Why |
|---|---|
| Q4_K RoCE v2 control on this branch | Confirms default-off commits remain inert vs main-era `19.22 t/s` |
| Q4_K OdinLink control | Required by objective: RoCE improvements must not regress OdinLink |
| Q2_K RoCE v2 control | Required by objective: preserve Q2_K signature and performance |
| Q2_K OdinLink control | Missing baseline; needed for transport-neutral handover |
| Optional Q4_K 32K long-context repeat | Keeps the most important long-context use case visible |

Only after that matrix passes should a future gate consider reopening code work.
