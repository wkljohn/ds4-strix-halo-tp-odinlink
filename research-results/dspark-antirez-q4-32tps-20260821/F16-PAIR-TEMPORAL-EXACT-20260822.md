# Exact temporal F16-pair reuse in the DSpark verifier

## Result

The exact TP verifier previously evaluated paired F16 attention-compressor and
indexer-compressor projections one token row at a time. The accepted candidate
uses the existing exact temporal kernel for rows two through four and decomposes
width five as 4+1. Each row retains the one-token FP32 accumulator, lane-K
order, FMA sequence, and wave reduction while the compact F16 weight streams
are read once per temporal group.

The fixed 2,048+300 Antirez Q4 workload produced:

| Run | Provider / mode | Prefill | Decode | FNV64 |
| --- | --- | ---: | ---: | --- |
| r1 | RoCE v2 + DSpark | 190.59 t/s | 14.09 t/s | `2aa153138c195efc` |
| r2 | RoCE v2 + DSpark | 190.65 t/s | 14.00 t/s | `2aa153138c195efc` |
| r3 | RoCE v2 + DSpark | 190.90 t/s | 14.08 t/s | `2aa153138c195efc` |
| Cross-provider | OdinLink + DSpark | 182.44 t/s | 14.19 t/s | `2aa153138c195efc` |
| Ordinary control | RoCE v2, no DSpark | 258.79 t/s | 19.52 t/s | `b7694f9d11a3760e` |

The three-run RoCE v2 median is **190.65 prefill / 14.08 decode t/s**.
Against the corrected exact-output-head reference at 13.73 t/s, decode improves
2.6%. All DSpark runs retain 153 cycles, 232 accepted drafts, 1.961 final
tokens/cycle, and the same acceptance histogram. The ordinary path does not
negotiate the DSpark exact-F16 feature and remains at its production reference.

Across the same 89 full verifier calls, measured verifier time falls from
15,012.443 ms to a three-run median 14,481.764 ms: **5.96 ms saved per full
verifier**, or 3.5%. The candidate adds no persistent allocation, repacked
weights, or expanded-weight cache.

## Isolated arithmetic and performance gate

`tests/test_rocm_f16_pair_temporal_exact.cu` uses signed full-mantissa F32
activations and nontrivial F16 weights. It compares both paired outputs bitwise
against the repeated one-row production dispatch and then measures GPU event
time. The real model uses output widths 128 and 256; 512 and 1024 remain covered
because the same public temporal primitive supports them.

| Rows | Output width | Serial exact | Temporal exact | Speedup | Exact |
| ---: | ---: | ---: | ---: | ---: | :---: |
| 2 | 1024 | 0.0660 ms | 0.0330 ms | 2.00x | yes |
| 3 | 1024 | 0.0907 ms | 0.0317 ms | 2.86x | yes |
| 4 | 1024 | 0.1207 ms | 0.0326 ms | 3.70x | yes |
| 5 | 128 | 0.0431 ms | 0.0198 ms | 2.17x | yes |
| 5 | 256 | 0.0503 ms | 0.0226 ms | 2.23x | yes |
| 5 | 512 | 0.0796 ms | 0.0338 ms | 2.36x | yes |
| 5 | 1024 | 0.1510 ms | 0.0649 ms | 2.33x | yes |

Four rows consume exactly 64 KiB of gfx1151 LDS
(`4 * 4096 * sizeof(float)`). Width five therefore stays 4+1. The kernel uses
no additional LDS and the shape gate rejects unsupported dimensions.

Run the isolated gate with:

```sh
make test-rocm-f16-pair-temporal-exact
```

## Dispatch and isolation

The optimization remains behind the existing
`DS4_TP_FEATURE_DSPARK_F16_PAIR_ROWS_EXACT` negotiation and the
`DS4_ROCM_DSPARK_F16_PAIR_ROWS_EXACT=1` runtime request. Both independent ranks
must advertise the same feature. The wrapper is selected only for the active
two-to-five-row DSpark batch verifier; ordinary decode, prompt prefill, other
backends, and non-TP execution keep their previous dispatches.

Both live call sites pass dense row-major F32 buffers sized as
`n_tokens * in_dim` and `n_tokens * out_dim`. Tensor views are bounds checked
and null-safe to free. Any view or temporal dispatch failure returns false and
aborts the layer/verifier; no partially written output is accepted or hashed.

## Evidence

Durable CSV, manifests, and coordinator/worker logs are under `runs/` with
tags:

```text
antirez-q4-dspark-f16temporal300-r1
antirez-q4-dspark-f16temporal300-r2
antirez-q4-dspark-f16temporal300-r3
antirez-q4-dspark-f16temporal300-odin-r1
antirez-q4-ordinary-f16temporal-control-r1
```

The validated binary hashes on both independent filesystems were:

```text
ds4          2f9c66948f85c46aa038a8bbf53717133b8a711e2d385a3dc9eb36fb85cb7889
ds4-bench-tp 6fbe6b379fbda64bfaaffaae5c6dcdc212f7428793e2acbd58134f6a8a3bd1e8
```
