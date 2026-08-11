# Cache-free gfx1151 TP decode closeout

Date: 2026-08-10

This closes the ordinary Q4_K/Q2_K decode regression investigation using the
fixed `ds4-bench-tp` protocol: two gfx1151 nodes, TP=2, explicit OdinLink RDMA,
2,048 prompt tokens, 300 forced greedy generation tokens, context 4,096,
cache-free main weights, three independent process/model loads, and medians.

## External implementation review

Three passes compared the DS4 hot path with current llama.cpp, vLLM and ROCm
AITER before implementation.

- llama.cpp already dynamically quantizes activations and uses packed integer
  dots in its MMQ/MMVQ family. Its gfx1151 investigation identified native
  `sudot4`, activation-rounding cost, MMQ geometry and VGPR pressure as the
  useful RDNA3.5 levers. DS4 does not share llama.cpp's MMQ tile dispatcher,
  so copying `mmq_x/mmq_y/nwarps` would not address DS4's one-token kernels.
  The transferable mechanism was native packed DP4A with shape-specific
  dispatch. Source: <https://github.com/ggml-org/llama.cpp/issues/21284>.
- vLLM's newer AMD W4A16 work confirms that skinny decode needs a separate
  kernel family, packed vector operations, wave-local reduction and
  shape-specific dispatch. Its GPTQ/AWQ layouts are not GGUF Q4_K/Q2_K and
  are therefore design evidence rather than drop-in code. Relevant upstream
  changes include vLLM PRs 41394 and 44075.
- Current AITER production support and release wheels target Instinct gfx942
  and gfx950. Its block-scaled FP4/FP8 GEMM and MoE kernels do not provide a
  directly usable GGUF Q4_K/Q2_K gfx1151 decode path. Source:
  <https://github.com/ROCm/aiter>.

This updates the earlier conclusion in
`DECODE-RESEARCH-AND-PLAN-2026-08-06.md`. Activation quantization was initially
deprioritized because upstream aggregate evidence favored F32 activation
loads. The exact DS4 projection shapes changed that decision: the isolated
Q8_0 DP4A harness measured 2.04--3.88x kernel speedups with about 0.0037--0.0038
NRMSE and no persistent allocation. End-to-end testing then proved a useful
ordinary-decode gain.

## Regression audit

An exact-source comparison against the pre-DSpark checkpoint did not reproduce
a code regression in ordinary decode. The apparent drop came from comparing
the former 13+ t/s short/different-context observations with the fixed
2,048+300 workload, whose pre-candidate Q4_K median was 9.92 t/s. The original
historical `4b35010` code measured 7.58 t/s under that same fixed workload.
Consequently all promotion decisions below use only like-for-like
`ds4-bench-tp` runs; no interactive or short-generation number is substituted.

> **2026-08-11 validation update:** the 14.31 t/s ordinary and 14.92 t/s
> DSpark rows below were produced by configurations that no longer pass the
> repaired token-fingerprint gate. They are retained as historical research,
> not production claims. The valid ordinary full-head baseline is 165.16 t/s
> prefill and 12.40 t/s decode; the repaired exact greedy-only top-2 candidate
> measured 154.40/12.54 t/s and fingerprint `752795b2398ef49c`. Its paired
> final full-head control measured 154.88/12.32 t/s, showing no meaningful
> prefill change and a 1.79% decode gain in the current system state.

## Implemented mechanisms

1. **Exact greedy top-2 TP exchange.** Rank 1 sends its best two `(id,value)`
   pairs instead of a 256 KiB FP32 vocabulary half. Two candidates are enough
   to preserve both ordinary argmax and argmax excluding one token. The
   feature is negotiated and automatically disabled for speculative sessions.
   Greedy sampling uses the TP-aware candidate merge; temperature sampling and
   full-logit/logprob consumers fail closed because the complete logits row is
   intentionally not materialized.
2. **Staged Q4_K activation.** The one-token gate/up workgroup stages sixteen
   Q8_K activation blocks once in about 4.6 KiB LDS and reuses them across its
   output rows. Dot and reduction order remain unchanged.
3. **Paired Q8_0 DP4A.** Eligible one-token projection pairs dynamically
   quantize their shared activation once and execute native packed integer
   dots. Exact production shapes choose the measured 8- or 16-wave launch.
   This adds no weight copy or persistent cache.
4. **Production profiling gate.** `make rocm` compiles hot per-gate,
   per-layer, event and graph-token profiling branches out. Diagnostic builds
   use `make rocm PROFILE=1`. Correctness checks and error reporting remain.

These remain explicit inference switches. Current candidate validation keeps
paired DP4A disabled because it changed trajectories; top-2 is greedy-only:

```sh
export DS4_TP_GREEDY_TOP2=1
export DS4_ROCM_Q4K_DECODE_STAGE_XQ=1
export DS4_ROCM_Q8_DECODE_PAIR_DP4A=0
```

## Fixed-workload results

| Configuration | Prefill median | Decode median | Steady decode median |
|---|---:|---:|---:|
| Original Q4_K at `4b35010` | 30.38 t/s | 7.58 t/s | 7.59 t/s |
| Current Q2_K | 118.24 t/s | 13.25 t/s | 13.27 t/s |
| Current Q4_K, ordinary | 166.56 t/s | 14.31 t/s | 14.34 t/s |
| Current Q4_K + DSpark, exact target | 166.92 t/s | 14.92 t/s | 15.00 t/s |

Ordinary Q4_K decode runs were 13.12/14.49/14.31 t/s. Q2_K runs were
12.23/13.33/13.25 t/s. DSpark exact-target runs were
14.92/15.05/14.78 t/s with identical 51.43% acceptance. Every accepted run
used explicit RDMA and reported zero provider fallback calls.

After the final profiling gate and documentation commits, the plain launcher
command was rerun with synchronized binary SHA-256
`f9b43ed5312552c1bcf647a7665a58e79dc3c1f3a148a503e03d8933a7ebb631`.
It measured 167.67 prefill, 13.31 overall decode and 13.33 steady decode t/s
with 300 generated tokens and zero provider fallback calls. This is a
corroborating final-build run; the table retains the less selective three-run
medians.

## DSpark incompatibility caught by A/B

Applying paired one-token DP4A only to committed target decode changed its
numerical trajectory relative to the five-row verifier. A three-run A/B
measured 11.95 t/s median and 29.83% acceptance, versus 14.92 t/s and 51.43%
for the exact target. Production now automatically disables that kernel in a
speculative process, even if its ordinary switch is set. A deliberate research
arm requires `DS4_ROCM_Q8_DECODE_PAIR_DP4A_SPECULATIVE=1`.

The guard was tested on the final code by requesting the incompatible switch:
both ranks printed the guard warning, acceptance remained 51.43%, decode was
14.54 t/s, and provider fallback remained zero. A future five-row paired-DP4A
kernel may recover the speed only after one-row and 2--5-row tests show that
verifier and committed target arithmetic agree.

## Memory and portability

The ordinary mechanisms add negligible persistent VRAM. They do not recreate
the roughly 9.85--9.91 GiB/rank Q8-to-F16 cache. Kernel dispatch is independent
of the verbs provider; OdinLink-specific transport behavior remains capability
gated, so Mellanox/generic verbs paths retain their existing protocol. Raw CSV
and rank logs are preserved under
`research-results/quant-comparison-2026-08-10/` on the test machine.
