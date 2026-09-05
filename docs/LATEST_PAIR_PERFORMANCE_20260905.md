# Latest active-pair performance record

This record covers only the current pair (`10.4.0.1` and `10.4.0.2`). Node
182 was not contacted, rebooted, or used. The two checkouts were clean at
commit `8f756592586d1274cc9e1849a0062d58baf3c39e` before the run.

## Fresh DeepSeek Q4_K probe

The run used `ds4-bench-tp`, the Antirez DeepSeek V4 Flash Q4_K model, RoCE
v2 (`mlx5_0` / `mlx5_1`, GID 3), 2,048 prompt tokens plus 300 generated
tokens, and a 2,048-token prefill chunk.

| Prefill | Decode | Fingerprint | Result |
|---:|---:|---|---|
| 301.31 t/s | 19.57 t/s (19.60 steady) | `b7694f9d11a3760e` | valid single run; zero fallback; `kvcache_bytes=0` |

Raw evidence is retained at
`$DS4_RESEARCH_ROOT/bench-runs/latest-pair-deepseek-q4-20260905.{csv,manifest}`.
The manifest records source commit `8f756592...`, matching binary and model
artifacts, and the exact workload. This fingerprint differs from the older
promoted `0163c44015591445` row, so it is evidence for revalidation, not an
automatic replacement of a promoted table row.

## GLM-5.3 Flash Q4_K probe

The executable was rebuilt from this same commit with the pinned ROCm
7.14/gfx1151 toolchain and synchronized to the pair. The run used the staged
ordinary GLM TP path, RoCE v2, batch 256, and the 4,096-token diverse prefill
plus 300-token decode workload. Both ranks initialized correctly, but the
coordinator produced no CSV row after approximately 11 minutes and the run
was stopped. It is not a performance result and must not be published as one.

The last valid GLM Q4_K source-clean record remains:

| Prefill | Decode | Fingerprint |
|---:|---:|---|
| 95.42 t/s | 9.95 t/s | `9012bd4d7c5ce422` |

That record is archived at
`$DS4_RESEARCH_ROOT/bench-runs/glm5-q4-mainmerge-r1-20260904.csv`.

## Promotion state

This branch contains documentation only; no production default or README
promoted row was changed. Before merging new performance numbers, repeat the
DeepSeek probe enough times for the applicable gate and diagnose the current
GLM 7.14 hang. The failed GLM run left no inference process on either active
node.
