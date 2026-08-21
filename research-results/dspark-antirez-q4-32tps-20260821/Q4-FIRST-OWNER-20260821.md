# Q4_K DSpark verifier first-owner result (2026-08-21)

## Scope

This stage optimizes the Antirez DeepSeek V4 Flash 0731 Q4_K routed-MoE
verifier on two gfx1151 ranks. It does not change ordinary width-one decode,
prefill, the model representation, or RDMA traffic, and adds no persistent
weight cache.

The fixed validation workload is `ds4-bench-tp`, TP=2, RoCE v2, 2,048 prompt
tokens plus 300 generated tokens, DSpark width five, 118/138 expert placement,
and token fingerprint `7174e214e05fd83e`.

## Design

The verifier routes 2--5 rows through six experts per row. The old batch path
re-read a selected expert's Q4_K gate/up weights independently for every row.
The new fixed pair-major grid lets only the first live occurrence of an expert
execute; that CTA gathers later live rows selecting the same expert, stages
their Q8_K activations, and applies the same scalar Q4_K dot/reduction to each
row. It is sort-free and preserves the established launch geometry.

TP remapping changes peer-owned routes to local expert 0 with weight zero.
Owner election and matching therefore ignore zero-weight routes, while every
inactive output is freshly zeroed. This prevents peer routes from colliding
with a real local expert 0 or leaving stale scratch.

Gate/up projections were bit-exact, but LLVM `-ffast-math` reassociated the
duplicate-row SwiGLU numerator. Generated gfx1151 ISA showed:

- serial: `((route_weight * gate) * up) * reciprocal`
- first attempt: `((up * gate) * route_weight) * reciprocal`

A tiny pair-major epilogue now pins the serial multiplication order with VGPR
compiler barriers. Gate/up scratch is then reused for mid Q8_K quantization;
there is no persistent allocation.

The path requires all of the following: DSpark TP verifier mode, Q4_K
4096x2048x4096 routed tensors, six slots, width 2--5, 16 input Q8_K blocks,
eight mid Q8_K blocks, canonical direct-sum6 down, and a negotiated TP feature
bit. A transient marker prevents unrelated small Q4 batches or tiny prefill
from entering the kernel. A rank setting mismatch fails in TP hello.

## Isolated correctness and speed

All comparisons are `memcmp`; inactive TP-remapped rows additionally require
canonical positive zero.

| Width | Live up mismatch | Mid mismatch | Routed output mismatch | Zero-route failure | Speedup vs serial rows |
|---:|---:|---:|---:|---:|---:|
| 2 | 0 / 24,576 | 0 / 24,576 | 0 / 8,192 | 0 | 1.52x |
| 3 | 0 / 36,864 | 0 / 36,864 | 0 / 12,288 | 0 | 1.76x |
| 4 | 0 / 49,152 | 0 / 49,152 | 0 / 16,384 | 0 | 1.88x stable run |
| 5 | 0 / 61,440 | 0 / 61,440 | 0 / 20,480 | 0 | 2.13x |

The swapped gate/up width-two run also passes, proving both raw projections
rather than only the tensor occupying the `up` auxiliary.

Run the isolated gate with:

```sh
make test-rocm-q4k-verify-batch-oracle
make test-tp-hello
```

## End-to-end attribution

All rows below use the same model, drafter, workload, transport, fingerprint,
and acceptance histogram.

| Configuration | Prefill | Decode | Exact fingerprint |
|---|---:|---:|---|
| Prior serial-row MoE, best matched run | 189.89 t/s | 11.35 t/s | yes |
| Batch MoE + canonical direct-sum6 control | 188.34 t/s | 12.19 t/s | yes |
| First-owner prototype r1 | 188.87 t/s | 13.01 t/s | yes |
| First-owner prototype r2 | 189.74 t/s | 13.07 t/s | yes |
| First-owner prototype r3 | 189.46 t/s | 13.00 t/s | yes |
| Negotiated/transient-gated final binary r1 | 190.40 t/s | 13.10 t/s | yes |
| Negotiated/transient-gated final binary r2 | 189.86 t/s | 13.02 t/s | yes |
| Negotiated/transient-gated final binary r3 | 190.27 t/s | 13.03 t/s | yes |

The final hardened three-run median is 190.27 prefill and 13.03 decode t/s.
The complete exact Q4 batch stage improved decode by 14.8% over the best
matched serial-row run; first-owner improved the batch/direct-sum control by
6.9%. The prototype median is retained only as development evidence and is not
mixed into the final result.

Ordinary balanced 128/128 decode passed fingerprint `b7694f9d11a3760e` at
258.60 prefill / 19.57 decode t/s.

After the backend dispatch was hardened to the exact six-route, 4096/2048/4096
Q4_K verifier geometry and tensor strides, a fresh full candidate run measured
189.95 prefill / 13.09 decode t/s. It retained fingerprint
`7174e214e05fd83e` and the identical acceptance histogram, so it validates the
shape-gated code without replacing the three-run median above.

## Reproduce

```sh
DS4_BENCH_DSPARK=1 \
DS4_BENCH_MTP=/absolute/path/DeepSeek-V4-Flash-DSpark-support-0731.gguf \
DS4_BENCH_CANDIDATE=1 \
DS4_BENCH_EXPECT_FNV64=7174e214e05fd83e \
./run-tp-ds4-bench.sh q4-dspark-first-owner-r1 \
  /absolute/path/DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix-0731.gguf \
  DS4_ROCM_Q8_REUSE_QUANT=1 \
  DS4_TP_BATCH_ATTN_HEAD_SPLIT=1 \
  DS4_DSPARK_VERIFY_BATCH_MOE=1 \
  DS4_DSPARK_VERIFY_TP_SHARED_SPLIT=1 \
  DS4_TP_VERIFY_ROW_BATCH=1 \
  DS4_TP_VERIFY_ATTN_SLAB=1 \
  DS4_ROCM_Q4K_VERIFY_DIRECT_SUM6=1 \
  DS4_ROCM_Q4K_VERIFY_FIRST_OWNER=1
```

## Rejected nearby paths

- The existing sorted expert tile4/tile8 kernels were 21--47% slower than
  serial rows at verifier widths and changed thousands of mid values.
- A dynamic non-unrolled match loop reduced the first-owner gain to about
  1.18x and did not repair the epilogue rounding.
- A generic linear epilogue preserved the wrong multiplication association.
- Recomputing Q4 dots as a correction lane is unnecessary: swapped raw
  gate/up projections are already exact.

The next kernel target is exact weight-outer Q8 attention projection reuse at
width two, then widths three through five.
