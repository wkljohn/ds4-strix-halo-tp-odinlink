# Evidence packet for two Fable framework-breakthrough reviews

This packet is evidence collected by the network-enabled main agent on
2026-08-21. Fable is asked to reason from these sources, not to claim direct
network access.

## Current exact baselines

- Q4_K RoCE v2 median: 257.87 prefill / 19.32 decode t/s, exact FNV64
  `5f8a983422299d76`.
- Q2_K RoCE v2 median: 202.86 / 19.13 t/s, exact FNV64
  `f9cb3a8a17e95c71`.
- Q4_K OdinLink: 218.89 / 18.89 t/s, exact Q4 fingerprint.
- Q2_K OdinLink: 178.06 / 18.69 t/s, exact Q2 fingerprint.
- Q4 target is at least 21 t/s without DSpark or persistent expanded-weight
  caches. At 19.32 t/s this requires reducing 51.76 ms/token to 47.62, a
  4.14 ms/token improvement.

## Locally verified Entrpi/ds4 commit objects

The `spark` remote is `https://github.com/Entrpi/ds4.git`; all objects below
resolve in this repository:

- `135ab507c7b84c67744eb34ec9f92eae51fe0fca` (2026-07-04), fused
  cooperative HC-stage kernel, commit claim -2.0 ms/token.
- `69a5e83df17f8c94863944b7572ce5e3b0b32de5` (2026-07-04), fused QKV
  post kernels: head RMS+Q RoPE and KV RoPE+FP8+store.
- `02f6914f1aed06bf8b7f7ab50b57ae90d680ed93` (2026-07-05), fused F16
  router projection+combine+top-6.
- `fd71740e62dbbca3736ba203e3f99c8a9ab24d21` (2026-07-05), fused
  compressor pair+store.
- `e221241c6ea8f2f77105cf21ff50a3990dfa59e1` (2026-07-05), Q2_K aligned
  row-pair SoA repack, bit-exact decode twin, commit claim -0.6/-0.9 ms/token.

Current DS4 inspection finds its ROCm graph still performs RMS, F16 projection,
and fused tail as three launches twice per layer. The Q side of `69a5e83` is
already represented; KV quantize/store remains separate. Current router was
measured near 3 us/layer. The paired compressor/store API is a stub in the
current ROCm implementation and temporal batching bypasses it often.

## Newly found primary-source framework work

1. AMD-Ecosystem/llama.cpp PR 67, July 2026:
   https://github.com/AMD-Ecosystem/llama.cpp/pull/67
   It folds decode GEMV tails (activation, activation*mul, residual-add through
   a view) into MMVQ epilogues. On gfx1151 Qwen3.6-35B-A3B Q4_K_M it reports
   58.73 -> 59.53 t/s, with launches/token 1263 -> 1103 in the associated
   upstream summary.
2. AMD-Ecosystem/llama.cpp PR 59:
   https://github.com/AMD-Ecosystem/llama.cpp/pull/59
   It concatenates K/V weights so two N=1024 MMVQ dispatches become one N=2048
   dispatch on gfx1151, raising occupancy 12.8 -> 16 waves/SIMD. It is flat on
   large models and initially duplicates K/V weights, so the direct design is
   disallowed by our cache-free requirement; a replacement-layout analogue is
   still open as an oracle.
3. llama.cpp AMD PR-set 2 summary:
   https://github.com/ggml-org/llama.cpp/discussions/26349
   New items include MoE actual-token compacted tiling, MMVDQ for n=1, and the
   gfx1151 MMQ device table. Our prior tests already reject MMVDQ as a first
   project and compacted MoE prefill tiling does not address one-token decode.
4. llama.cpp AMD PR-set 4 summary:
   https://github.com/ggml-org/llama.cpp/discussions/26378
   Decode focus is GEMV epilogue fusion and K/V dispatch fusion.
5. llama.cpp AMD PR-set 6 summary:
   https://github.com/ggml-org/llama.cpp/discussions/26380
   Shared-expert auxiliary-stream overlap reports +7.4% tg128, but DS4's
   temporal side-stream/UMA overlap attempts are already rejected and TP rank
   skew makes this non-transferable without a new proof.
6. AITER Q2 2026 roadmap:
   https://github.com/ROCm/aiter/issues/3442
   Includes fused DeepSeek-V4 compressor attention, fused qk norm+RoPE+quant,
   persistent MLA for gfx950, fused qknorm+allreduce, and two-stage fused MoE.
   AITER itself labels gfx1151 experimental; most CK/ASM kernels are CDNA-only.
7. AITER fused qk norm+RoPE+quant PR 3320:
   https://github.com/ROCm/aiter/pull/3320
   One wave64/block, one launch for RMSNorm+GPT-J RoPE+optional FP8 quant,
   targeting DeepSeek-V4-Pro TP=8 on gfx950. Its fusion boundary is relevant;
   its exact arithmetic and hardware specialization are not directly portable.
8. vLLM ROCm Q2 2026 roadmap:
   https://github.com/vllm-project/vllm/issues/44092
   DeepSeek-V4 work includes AITER MLA decode and CSA multistream decode. These
   are largely gfx950/TP8/batch-serving work and need architecture filtering.

## Already tested or excluded

Do not merely rename these: DSpark/speculation; persistent Q8-to-F16 cache;
extra MOE_MID host callback; slot stealing; STAGE_MIDQ; compressor/indexer TP
split at 2K; K-shard down with changed reduction tree; HIP graphs (<2 us);
temporal auxiliary stream/UMA overlap; Q_B prefetch/fusion; transport pollers,
opcode or MTU work; more waves; MMVDQ as first project; grouped-head attention
at 2K; direct AITER/vLLM kernel port; duplicated resident repacks.

The strongest current local plan is to profile the remaining launch chains,
then test an exact two-kernel HC fusion: kernel A recomputes the same RMS
reduction per matmul block and explicitly rounds the scaled input to F32;
kernel B preserves the existing fused tail. If that is insufficient, a
device-visible completion-word ordering probe may enable gate-free FFN row
sharding without a third callback. Any production candidate must pass Q4/Q2,
RoCE/OdinLink, exact fingerprint and semantic gates.
