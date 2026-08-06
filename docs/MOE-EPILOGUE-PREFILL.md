# Coalesced Q4_K MoE prefill epilogue on gfx1151

This note records the staged validation of DS4's row-parallel routed-MoE
gate/up epilogue. Raw two-node logs and profiler CSVs are retained locally in
the ignored `research-results/2026-08-06/raw/moe-epilogue-research/` tree.

## Result

| Cache-free TP=2, same final binary | Prefill | Decode sample |
|---|---:|---:|
| Diagnostic rollback | 138.24 t/s | 13.40 t/s |
| Default coalesced epilogue | **167.73 t/s** | 12.82 t/s |
| Change | **+21.3%** | not attributed; see below |

Both arms used the same 10,093-byte prompt, generated 300 tokens, made 41,022
provider calls per rank, moved 2,423,246,080 provider bytes per rank, and
reported zero WC-stream-copy fallbacks. Their generated outputs have the same
SHA-256, `670b451dfa8be54353e7187636cfa7c3750f7f5cddba68e45aa2ed9d68a255d2`.

The candidate is selected only for `n_tokens >= 8`; single-token decode calls
the original kernel. It therefore cannot explain the small decode-rate
difference in this one A/B pair, which is retained rather than normalized
away. The independent 300-token default decode reference is 13.83 t/s.

Planned memory remains 81.18 GiB per rank. The kernel uses only the existing
gate, up, mid, routing, and weight buffers and adds no persistent allocation.
The Q8-to-F16 cache was unset in every result above.

## Why the old layout was slow

The WMMA gate and up projections produce pair-major arrays. The former
epilogue launched 16 threads per expert tile, assigned one routed pair to each
thread, and made every thread walk all 2,048 expert rows serially. Threads in
a wave therefore accessed different pair-sized regions rather than adjacent
rows.

The new launch uses 256 threads and a two-dimensional grid. One thread owns
one row and loops across the at-most-16 pairs in its expert tile. Neighboring
threads consequently read and write neighboring rows for each pair, while
each row is still computed exactly once.

## Small-step validation

The standalone harness is `scripts/moe_epilogue_layout_bench.cu`. Build and
run it before integrating further epilogue changes:

```sh
/opt/rocm/bin/hipcc -O3 -ffast-math -fno-finite-math-only \
  --offload-arch=gfx1151 scripts/moe_epilogue_layout_bench.cu \
  -o /tmp/moe_epilogue_layout_bench
/tmp/moe_epilogue_layout_bench
```

It first checks a deliberately non-aligned 35-pair by 2,051-row case with
clamping and gate/up writeback. All 215,355 gate, up, and mid floats matched
bit-for-bit. It then times the 22,720-pair by 2,048-row production-scale
layout:

```text
serial rows:    210.0272 ms
coalesced rows:   2.7019 ms
change:          -98.7%
```

The next rung was a one-token full-model smoke test, followed by a 30-token
exact-output test. Only after those passed was the candidate enabled by
default with the decode gate and tested for 300 generated tokens on both
ranks.

## Integrated trace

`rocprofv3 -f csv --kernel-trace --stats` captured one 43-layer prefill. The
comparison uses the earlier compact-Q8 trace as the old-layout reference:

| Coordinator kernel | Old layout | Coalesced layout |
|---|---:|---:|
| Routed gate/up/mid epilogue | 3.779 s (18.42%) | 0.086 s (0.32%) |
| Calls | 43 | 43 |
| Per call | 87.874 ms | 1.999 ms |
| Kernel-time change | | **-97.7%** |

The profiler run itself measured 104.87 t/s and contains large ROCm stream
wait outliers, so it is used only for kernel attribution. The unprofiled,
matched full-model A/B above is the throughput result.

## Controls and portability

The optimized path is the default only for the ROCm Q4_K WMMA gate/up route
and batches of at least eight tokens. Decode, smaller batches, other
quantizations, non-WMMA shapes, and all fallback paths retain the old
epilogue. Set the following on both ranks for a diagnostic rollback:

```sh
export DS4_ROCM_MOE_GATE_UP_EPILOGUE_COALESCED=0
```

This is a compute-only dispatch change. It does not branch on, alter, or
allocate transport state, so OdinLink and Mellanox verbs use the same
respective communication paths as before.
