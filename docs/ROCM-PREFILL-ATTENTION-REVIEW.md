# ROCm prefill-attention review

Date: 2026-08-06

This review asks whether DeepSeek V4 prefill attention still contains a
credible **5% or larger end-to-end TP=2 gain** on the two-node Strix Halo
OdinLink reference setup. It covers the current DS4 implementation, current
upstream implementations, a non-synchronizing GPU trace, and rejected
hardware experiments.

## DS4 contract and current path

The relevant DS4 shape is FP32 Q plus one shared FP32 latent K/V vector per
row, 16 query heads, head dimension 512, a 256-row raw sliding window,
ratio-4 compressed rows, causal visibility, and learned attention sinks.
The output is FP32. Static compressed attention and raw attention use
`attention_static_mixed_heads8_online_kernel` while their visible raw plus
compressed key count is at most 768. Longer shapes fall back to the hipBLAS
score-matrix path.

The earlier TP row split is implemented and correct, including absolute
causal offsets and stable temporary-buffer sizing. Its controlled hardware
result was effectively flat: 99.25 t/s replicated versus 98.75 t/s split.
The extra TP output exchange consumed the saved replicated work, so it
remains off by default.

## Current primary-source survey

- [vLLM at `7b4ed496`](https://github.com/vllm-project/vllm/blob/7b4ed49628abd7860a435d6798feef76a944cb02/vllm/v1/attention/ops/rocm_aiter_mla_sparse.py)
  has a DeepSeek V4 sparse-prefill Triton kernel for the exact 512-wide
  latent. It uses tiled QK/PV, online softmax, attention sinks, indirect KV
  rows, and index prefetch. Commit
  [`3c1bc1fc0`](https://github.com/vllm-project/vllm/commit/3c1bc1fc0)
  reduced the kernel from eight to four warps and prefetched the next index.
- [AITER at `00b271b9`](https://github.com/ROCm/aiter/tree/00b271b95f8a3405fa211e509c63441040557305)
  contains both a general Triton flash-prefill implementation and a DSV4
  sparse-attention implementation. The general kernel has an explicit
  gfx1151 preset (`BLOCK_M=128`, `BLOCK_N=64`, eight warps, two waves/EU),
  although that preset was tuned for a 72-wide vision-attention shape. The
  DSV4 kernel autotunes head tiles 32/64 and key tiles 16/32/64 with FP32
  online-softmax state. These are strong algorithm and geometry references,
  but are Python/Triton kernels rather than linkable DS4 HIP entry points.
- [llama.cpp at `6a32c29a`](https://github.com/ggml-org/llama.cpp/blob/6a32c29a746a2e44de463de647f9f6661eb5086b/ggml/src/ggml-cuda/fattn-tile.cuh)
  instantiates a native HIP tile for the exact 512/512 QK/V shape, including
  RDNA configurations and attention sinks. Its matrix path converts FP32
  inputs to FP16. DS4 cannot inherit that numerical choice without a separate
  accuracy decision, and its compressed-row visibility/layout still needs an
  adapter.
- [Composable Kernel at `15db95b1`](https://github.com/ROCm/composable_kernel/tree/15db95b18519efab1cf1cc2e6d2de4468ba58187)
  and [ROCm FlashAttention at `77aacb68`](https://github.com/ROCm/flash-attention/tree/77aacb68d194ba9af1010eda5eac3e7c0df8e6f6)
  provide useful FMHA infrastructure, but the surveyed entry points do not
  provide a directly usable gfx1151, FP32, 512-wide shared-latent kernel with
  DS4's compressed visibility and sink contract.

The reusable idea is therefore tiled/online softmax and KV reuse—not a
drop-in external kernel.

## Measured opportunity bound

`rocprofv3 --kernel-trace --stats` was used instead of the old per-stage
synchronizing profiler. Both ranks loaded concurrently and used byte-identical
binaries. On the 9,881-byte reference prompt:

- prefill was 138.23 t/s under tracing;
- the 43 static mixed-attention launches consumed 1,518.814 ms total;
- the complete prefill took about 12.32 seconds;
- attention was therefore about 12.3% of prefill wall time.

An end-to-end 5% speedup requires reducing this attention cost by roughly 41%
if everything else is unchanged. The opportunity is real, but only a large
kernel improvement can clear the project threshold.

## Rejected hardware experiments

All trials used the same 9,881-byte prompt, TP=2, OdinLink provider and
runtime settings on both independent node filesystems. Candidate binaries
were copied to both nodes and SHA-256 matched before launch.

| Experiment | Baseline | Candidate | Result | Correctness |
|---|---:|---:|---:|---|
| Full dense FP16 QK/PV | — | — | 1,385.94 MiB scratch allocation failed | no computation |
| 128-query block-sparse FP16 hipBLAS | 152.11 t/s | 146.02 t/s | -4.0% | greedy output diverged immediately |
| FP32 adjacent-query reuse with 57 KiB LDS scores | 152.11 t/s | 135.54 t/s | -10.9% | greedy output identical |
| FP32 adjacent-query reuse with QK recomputation | 152.11 t/s | 119.31 t/s | -21.6% | performance gate failed |

The FP16 experiment retained FP32 scores, masking, softmax, sinks and output,
but rounded Q/K and normalized probabilities for matrix-engine GEMMs. Even
after limiting each tile to the union of its raw window and visible compressed
prefix, conversion/launch costs outweighed matrix throughput and the numerical
change crossed the first greedy decision.

The exact FP32 reuse form halved K/V fetches but its score tile raised LDS from
about 32 KiB to 57 KiB and lost occupancy. Removing the score tile restored
occupancy but recomputing 512-wide QK dots cost much more than the traffic it
saved. Neither structure is a production candidate.

## Current FFN transport bound

The optimized provider was also profiled after the attention review. An
interrupted run reached 16 of 43 gates and moved 447.0 MB in 436.9 ms (1,023
MB/s), with zero staging-copy time. At the same per-gate cost, all 43 layers
account for about 1.17 seconds, or roughly 10% of an 11--12 second prefill.

The internal breakdown was 112 rounds: 1.3 us/round in `post_recv`, 571.9
us/round in `post_send`, 37.8 us to first CQE, 3,236.6 us to last receive, and
696.2 us to last send. Even eliminating all provider submit time would save
only about 0.17 seconds per prefill, below the 5% end-to-end threshold. The
remaining large component is payload arrival, so a half-width wire format was
the only transport candidate still capable of clearing the threshold.

Real-tensor telemetry supported trying F16: across 300,298,240 reference
prefill elements, maximum absolute magnitude was 17,560.76, with no values over
32,752 and no NaN or infinity. An opt-in implementation packed both local
partials symmetrically, exchanged half as many bytes, and unpacked both sides
on the ordered TP stream. It reused idle tails of the registered slab and
required no extra allocation. Provider counters confirmed the traffic
reduction (642.6 MB versus 1,242.9 MB for the complete test session).

It failed the accuracy gate. With identical prompt, seed and 30-step greedy
generation, the first selected-token divergence occurred at step 15 and 6 of
30 selected tokens differed. The experimental implementation was removed
before performance A/B and is not present in production source. BF16 has the
same two-byte payload but less significand precision, so it is not a credible
accuracy improvement over this rejected F16 result.

## Decision

Do not enable TP attention row splitting, FP16 hipBLAS attention, adjacent
query reuse, or F16/BF16 FFN wire compression. The only attention design with
a theoretical 5% end-to-end bound is a true fused tiled kernel in the
vLLM/AITER/llama.cpp family: multiple query rows per workgroup,
matrix-friendly QK/PV, online FP32 softmax, no quadratic score matrix, and no
excessive LDS score tile.

On gfx1151, the available matrix path requires narrow inputs; the tested
narrow attention and FFN-wire paths both changed greedy decisions. Exact FP32
adjacent-query reuse was substantially slower. Consequently there is no
remaining correctness-preserving prefill candidate with a credible 5% gain in
the reviewed paths. A future approximate fused design should not become the
default unless it passes layer-level tensor/logit validation across long,
short, window-boundary, compression-boundary, and near-tie prompts.
