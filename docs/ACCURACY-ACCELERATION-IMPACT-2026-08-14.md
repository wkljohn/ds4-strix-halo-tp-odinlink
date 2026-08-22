# Accuracy impact of the gfx1151 acceleration paths

Date: 2026-08-14

Branch: `research/accuracy-accel-deviation`

Base commit: `345b7aa`

## Decision

Keep both acceleration paths enabled by default.

- The Q2 fused IQ2_XXS integer-WMMA path changes floating-point results, but
  its measured quality movement is small and statistically inconclusive on the
  100-case official suite. It provides an 8.6% prefill improvement over the
  immediately preceding mixed-Q4 path.
- Q4 WMMA is exactly neutral on all 2,289 official continuation tokens in the
  short-prompt suite. At the 2,048-token production frontier it preserves the
  argmax and first generated token while producing an extremely small change in
  normalized probability, despite materially different raw logits.

Use the kill switches when exact reproduction of the preceding arithmetic is
more important than throughput:

```sh
DS4_ROCM_DISABLE_IQ2_I8_WMMA=1   # hybrid Q2 only
DS4_ROCM_DISABLE_Q4K_WMMA=1      # Q4 WMMA gate/up/down path
```

## Controlled results

Every DS4 result below used two gfx1151 nodes, a balanced 128/128 expert split,
explicit OdinLink RDMA, identical binaries on both nodes, full-vocabulary logit
exchange, cache-free weights, and zero transport fallbacks.

### Official 100-case continuation suite

The tracked Flash fixture contains 100 official API continuations and top-20
odds. DS4 teacher-forced the same 2,289 reference tokens through every arm.

| Comparison | Control | Accelerated | Measured change |
|---|---:|---:|---:|
| Q2 average reference-token NLL | 0.562075980 | 0.563753691 | +0.001677711 (+0.298%) |
| Q2 API top-1 agreement | 83.399% | 83.661% | +0.262 percentage points |
| Q2 API pair-order agreement | 98.488% | 98.404% | -0.083 percentage points |
| Q2 first-token matches, 100 cases | 48 | 47 | -1 case |
| Q2 average greedy reference-prefix length | 5.27 | 5.05 | -0.22 token |
| Q4 average reference-token NLL | 0.528804242 | 0.528804242 | exactly unchanged |
| Q4 complete per-case TSV | SHA-256 `65349e43...f0f39ce` | same SHA-256 | bit-identical |

For Q2, the accelerated path won 49 cases and the control won 51. A
deterministic 20,000-resample paired case bootstrap placed the 95% interval for
the token-weighted NLL change at `[-0.006401, +0.009977]`; it includes zero.
This suite therefore does not establish a quality regression or improvement.
It does establish that the new arithmetic is not bit-identical.

Disabling Q4 WMMA in addition to IQ2 WMMA produced the exact same Q2 score
table as disabling IQ2 WMMA alone (SHA-256 `2049538d...5328be8`). The Q2 A/B
difference is therefore attributable to the fused IQ2 path in this workload.

The official prompts are only 12–28 tokens. That is sufficient for decode and
short-prefill scoring, but it does not engage the Q4 sorted-WMMA path's
32-token long-prefill threshold.

### Q4 2,048-token frontier

The fixed `ds4-bench-tp` prompt was therefore captured separately with all
129,280 logits at the 2,048-token frontier.

| Metric | Q4 WMMA disabled | Q4 WMMA enabled |
|---|---:|---:|
| Single-run diagnostic prefill | 75.37 t/s | 200.36 t/s |
| Argmax token | 65 | 65 |
| First generated-token fingerprint | `abd5cc8cd5077ba4` | `abd5cc8cd5077ba4` |
| Raw-logit mean absolute difference | — | 0.439990 |
| Raw-logit RMS difference | — | 0.567970 |
| Raw-logit maximum absolute difference | — | 3.306702 |
| Probability total-variation distance | — | 0.000001131 |
| KL, control to accelerated | — | 0.000001003 |
| Top-5 / top-10 overlap | — | 4/5 and 7/10 |

The 200.36/75.37 t/s figures prove the intended kernels were engaged and give
the direction and approximate scale of the speedup. They are one-token
diagnostic runs, not replacements for the three-run production medians in the
main README. The tiny probability distance alongside larger raw-logit movement
occurs because this frontier is highly peaked; the common-mode and low-mass
logit changes barely move the normalized distribution.

## Remote llama.cpp semantic cross-check

The provided NVIDIA endpoint was healthy and had
`deepseek-v4-flash-0731-ablit-q4k-ab-dspark` loaded. Three deterministic probes
were correct:

| Probe | Expected | Returned |
|---|---|---|
| Discount then tax | `$70.40` | `$70.40` |
| Python set-comprehension | `[1, 9]` with explanation | correct |
| Distractor retrieval | `7419, amber` | `7419, amber` |

This cross-check confirms coherent behavior from the same Q4 model family. It
does not measure DS4 acceleration impact: the remote server uses llama.cpp,
layer parallelism, and DSpark. It is intentionally secondary to the controlled
DS4 on/off tests.

## Reproduction

Build the TP-aware official scorer:

```sh
make -j8 strix-halo-quality-score
make test-tp-hello
```

Run the causal Q2 pair:

```sh
./run-tp-quality-score.sh q2-current-full /absolute/path/DeepSeek-V4-Flash-Q2_K-0731.gguf

./run-tp-quality-score.sh q2-iq2-disabled-full \
  /absolute/path/DeepSeek-V4-Flash-Q2_K-0731.gguf \
  DS4_ROCM_DISABLE_IQ2_I8_WMMA=1

python3 gguf-tools/quality-testing/compare_scores.py \
  "$DS4_RESEARCH_ROOT/accuracy-acceleration-2026-08-14/q2-iq2-disabled-full.tsv" \
  "$DS4_RESEARCH_ROOT/accuracy-acceleration-2026-08-14/q2-current-full.tsv"
```

Run the short-prompt Q4 pair by setting
`DS4_ROCM_TP_PREFILL_SKIP_UNOWNED=1` on both arms and adding
`DS4_ROCM_DISABLE_Q4K_WMMA=1` only to the control.

The full raw artifacts, including rank logs, score tables, frontier logits,
remote requests, and remote responses, live in the ignored local directory:

```text
$DS4_RESEARCH_ROOT/accuracy-acceleration-2026-08-14/
```

## Limits and next accuracy gate

- The official fixture is broad but short. It cannot establish long-context
  retrieval quality by itself.
- The 2,048-token test captures one frontier and one next-token decision. It is
  strong arithmetic evidence, not a broad semantic benchmark.
- The official model and the local model variant are not identical. Absolute
  NLL should not be treated as a model leaderboard; paired on/off differences
  remain valid because the model and reference are held constant.
- DSpark was excluded from the causal DS4 comparison. It requires a separate
  verifier/drafter acceptance study.

Future arithmetic candidates should pass, in order: the 100-case official
paired scorer, the 2,048-token full-logit frontier, and the fixed 2,048+300
`ds4-bench-tp` semantic/signature gate. A speed result alone is insufficient.
