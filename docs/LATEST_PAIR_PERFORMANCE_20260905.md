# Latest active-pair performance record

This record covers only the current pair (`10.4.0.1` and `10.4.0.2`). Node
182 was not contacted, rebooted, or used. The two checkouts were clean at
commit `8f756592586d1274cc9e1849a0062d58baf3c39e` before the run.

## DeepSeek Q4_K probes

The run used `ds4-bench-tp`, the Antirez DeepSeek V4 Flash Q4_K model, RoCE
v2 (`mlx5_0` / `mlx5_1`, GID 3), 2,048 prompt tokens plus 300 generated
tokens, and a 2,048-token prefill chunk.

The original single probe used an older ROCm 7.2 binary. It measured
301.31/19.57 t/s and remains valid for that artifact, but it is not a sample of
the current ROCm 7.14 build. After rebuilding, three source-clean ROCm 7.14
runs produced:

| Run | Prefill | Decode | Fingerprint |
|---|---:|---:|---|
| 1 | 305.02 t/s | 21.12 t/s | `0163c44015591445` |
| 2 | 333.22 t/s | 21.18 t/s | `0163c44015591445` |
| 3 | 334.75 t/s | 21.17 t/s | `0163c44015591445` |
| **Median** | **333.22 t/s** | **21.17 t/s** | exact match |

Raw evidence is retained as
`$DS4_RESEARCH_ROOT/bench-runs/warm-repeat-fixed-q4-r{2,3,4}-20260905.{csv,manifest}`.
The manifests bind source commit `d97abb5...`, binary
`c71bd76f...`, ROCm 7.14 toolchain fingerprint `d2346fc4...`, matching model
and prompt artifacts, and the exact workload. The excluded ROCm 7.2 probe is
kept at `latest-pair-deepseek-q4-20260905.{csv,manifest}` for provenance.

## GLM-5.3 Flash Q4_K probe

The executable was rebuilt from this same commit with the pinned ROCm
7.14/gfx1151 toolchain and synchronized to the pair. The valid rerun used the
staged ordinary GLM TP path, RoCE v2, batch 256, and the 4,096-token diverse
prefill plus 300-token decode workload. Both ranks engaged the batched path,
Q4_K WMMA, sparse NoPE F16 GEMM, and BF16 WMMA QKV/output kernels; the exact
fingerprint and semantic checks passed.

| Prefill | Decode | Fingerprint |
|---:|---:|---|
| **74.81 t/s** | **10.11 t/s** (10.12 steady) | `9012bd4d7c5ce422` |

Raw evidence is archived at
`$DS4_RESEARCH_ROOT/bench-runs/latest-pair-glm-q4-staged-20260905.{csv,manifest}`.
The earlier 95.42/9.95 result came from a different source/binary build and is
kept as historical comparison at
`$DS4_RESEARCH_ROOT/bench-runs/glm5-q4-mainmerge-r1-20260904.csv`.

## Promotion state

This branch contains documentation only; no production default or README
promoted row was changed. The earlier minimal GLM probe was invalid and
produced no row; the staged rerun above is the current valid measurement. The
DeepSeek repeat series confirms the existing `0163c44015591445` arithmetic
path and does not require a fingerprint promotion.
