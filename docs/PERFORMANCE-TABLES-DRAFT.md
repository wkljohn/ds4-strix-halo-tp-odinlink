# Ordinary inference performance tables (draft)

These tables are the draft presentation for the workload-diversity gate. They
cover ordinary cache-free TP=2 inference only; DSpark is intentionally absent.
The model is Antirez DeepSeek V4 Flash 0731 Q4_K and RDMA is mandatory.

## Validated production anchor

| Transport | Workload | Prefill | Decode | Runs | Fingerprint |
|---|---|---:|---:|---:|---|
| OdinLink RDMA | 2,048 prompt + 300 decode | 231.46 t/s | 19.13 t/s | three-run median | `b7694f9d11a3760e` |
| ConnectX-4 Lx RoCE v2 | 2,048 prompt + 300 decode | 272.51 t/s | 19.54 t/s | three-run median | `b7694f9d11a3760e` |

## Cross-disciplinary long-context screen

| Transport | Workload | Prefill | Decode | Steady decode | Runs | Fingerprint |
|---|---|---:|---:|---:|---:|---|
| ConnectX-4 Lx RoCE v2 | cross-disciplinary v1, 4,096 prompt + 300 decode | 248.70 t/s | 18.15 t/s | 18.18 t/s | one draft run | `37c5c76993e1d15a` |

The cross-disciplinary prompt alternates software debugging, quantitative
science, policy/document retrieval, and structured-data analysis. It uses the
same model and validated binaries as the production anchor, balanced 128/128
experts, a 2,048-token prefill chunk, no DSpark, and no expanded-weight cache.

The 4K result is 8.7% lower in prefill and 7.1% lower in decode than the 2K
RoCE production median. Because prompt length and content both differ, this is
a workload-sensitivity observation, not an attribution to either variable and
not evidence of a code regression by itself. It demonstrates why the 2K coding
anchor cannot be the sole promotion test.

This `candidate=0` run discovers the draft fingerprint; it does not count as a
frozen baseline. Before this becomes a release gate, repeat three independent
unchanged-stack runs in candidate-validation mode with that expected
fingerprint, freeze their median and variance, then compare each candidate with
`scripts/diverse-bench-gate.py`. The gate rejects an individual prefill or
decode regression; it does not average the long result into the headline
production number.

## Reproduce the draft long screen

```bash
./scripts/run-tp-diverse-bench.sh ordinary-diverse-roce-r1 \
  /absolute/path/to/DeepSeek-V4-Flash-Q4_K.gguf
```

The launcher generates the versioned prompt under `DS4_RESEARCH_ROOT`, loads
the model once, requires explicit RDMA, and prints this table directly from the
validated CSV.
