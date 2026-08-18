# Strix Halo TP=2 optimization and validation record — August 2026

This note preserves the implementation details, rejected experiments, raw run
values, rollback switches, and maintainer validation procedure behind the
short user-facing results in the repository README. The public benchmark is
`ds4-bench-tp`: a fixed 2,048-token prefill followed by 300 generated tokens,
with mandatory RDMA and cache-free model weights.

## Benchmark provenance

| DeepSeek V4 0731 configuration | Accepted runs, prefill/decode t/s | Reported result |
|---|---|---|
| Original Q4_K TP=2 baseline | archived 34.11/9.96 | 34.11/9.96 |
| Current Q2_K over RoCE v2 | rebuilt final 179.24/19.11 | production validation 179.24/19.11 |
| Current Q4_K over OdinLink | rebuilt final 199.91/19.01 | production validation 199.91/19.01 |
| Current Q4_K over RoCE v2 | 219.56/19.37, 219.34/19.37, 219.62/19.35 | three-run median 219.56/19.37 |

The current Q4_K fingerprint is `5f8a983422299d76`; the current hybrid Q2_K
fingerprint is `f9cb3a8a17e95c71`. Each fingerprint is specific to the
documented model, prompt, balanced 128/128 expert split, and 300-token greedy
workload. It is a deterministic self-regression gate, not proof of numerical
parity with llama.cpp.

The RoCE runs used the same Q4_K model, binary, workload, and fingerprint as
the matched OdinLink controls. Matched 2,048-token-chunk OdinLink runs measured
198.55/14.88 and 192.63/14.84 t/s. Their midpoint was 195.59/14.86 versus
222.76/17.08 for RoCE v2: **+13.9% prefill and +14.9% decode**. RoCE replaces
only the TP communication slab with a 77.4 MiB mapped allocation; it adds no
expanded model-weight cache. The allocator and transport evidence is in
[`../roce-v2-2026-08-14/`](../roce-v2-2026-08-14/).

## Kernel and scheduling changes

The hybrid Q2_K path expands IQ2_XXS codebook groups directly into tile-local
INT8, reuses dynamically quantized Q8_1 activations, and uses native gfx1151
integer WMMA before the fused SwiGLU epilogue. It adds no expanded-weight
cache; the 2,048-token run needs about 9 MiB of reusable scratch per rank. TP
hello feature negotiation prevents independently launched ranks from entering
different arithmetic schedules. `DS4_ROCM_DISABLE_IQ2_I8_WMMA=1` is the
rollback switch.

The safe zero-weight-tile schedule skips work inside peer-only IQ2/Q2 and
Q4-subset tiles without changing expert counts or launch geometry. The more
aggressive hybrid-Q2 route omission was rejected because it changed full
logits and the 300-token fingerprint. The launcher reads GGUF metadata rather
than inferring the routed-expert layout from filenames and fails closed on
unknown layouts.

For Q4_K prefill, each rank omits peer-owned routed-expert work before building
expert tiles. Static mixed attention uses a single-pass online softmax on
gfx1151, reducing per-block LDS from about 32 KiB to 8 KiB and reading each KV
row once. These changes add no persistent model memory or wire traffic. The
online reduction changes the deterministic trajectory, so it was accepted
only after semantic smoke and repeat validation established the current
fingerprint.

Where documented HIP signal memory is unavailable, ROCm TP gates use ordered
HIP host callbacks before explicit RDMA exchange. The validated temporal
compressor batches the repeated F16 projection at its natural four-token
boundary. Together these exact, cache-free changes produced the current
19.37 t/s Q4_K RoCE median. The benchmark and deployment launchers enable them
for ordinary inference and negotiate the temporal feature between both ranks.

Detailed reports:

- [`../../docs/Q2-I8-WMMA.md`](../../docs/Q2-I8-WMMA.md)
- [`../../docs/Q4K-WMMA-PLAN.md`](../../docs/Q4K-WMMA-PLAN.md)
- [`../../docs/Q8-COMPACT-PREFILL.md`](../../docs/Q8-COMPACT-PREFILL.md)
- [`../../docs/ROCM-PREFILL-ATTENTION-REVIEW.md`](../../docs/ROCM-PREFILL-ATTENTION-REVIEW.md)
- [`../../docs/DECODE-RESEARCH-WRAP-UP-2026-08-06.md`](../../docs/DECODE-RESEARCH-WRAP-UP-2026-08-06.md)

## Cache policy and historical A/B

Q2_K and Q4_K production defaults do not expand quantized weights into a
persistent cache. The optional Q8-to-F16 cache consumed 9.85–9.91 GiB per rank
and drove reported VRAM use near 99%. In an earlier matched 300-token check,
disabling it changed decode from 13.80 to 13.79 t/s and prefill from 115.56 to
87.67 t/s. A later compact-Q8 token-tiled run reached 138.97 prefill and
13.32 decode t/s without the persistent cache.

The cache therefore remains warning-gated and must not be used for production
or accepted benchmark results:

```sh
# Diagnostic only: about 10 GiB of extra persistent memory per rank.
export DS4_ROCM_ENABLE_Q8_F16_CACHE=1
```

Experimental DSpark keeps a Q8 drafter on rank 0 and reserves another
10.15 GiB. It remains opt-in until revalidated on the same target-only
workload; DSpark uses an approximately 46/54 expert split while ordinary
decode remains balanced 128/128.

## Maintainer validation

Run a normal benchmark three times and report the median:

```sh
./run-tp-ds4-bench.sh q4-r1 /absolute/path/DeepSeek-V4-Flash-Q4_K.gguf
./run-tp-ds4-bench.sh q4-r2 /absolute/path/DeepSeek-V4-Flash-Q4_K.gguf
./run-tp-ds4-bench.sh q4-r3 /absolute/path/DeepSeek-V4-Flash-Q4_K.gguf
```

RoCE v2 uses the same harness:

```sh
DS4_BENCH_RDMA_PROFILE=roce-v2 \
  ./run-tp-ds4-bench.sh q4-roce-r1 \
  /absolute/path/DeepSeek-V4-Flash-Q4_K.gguf
```

Optimization candidates must name the expected fingerprint:

```sh
DS4_BENCH_CANDIDATE=1 \
DS4_BENCH_EXPECT_FNV64=5f8a983422299d76 \
  ./run-tp-ds4-bench.sh q4-candidate \
  /absolute/path/DeepSeek-V4-Flash-Q4_K.gguf

DS4_BENCH_CANDIDATE=1 \
DS4_BENCH_EXPECT_FNV64=f9cb3a8a17e95c71 \
  ./run-tp-ds4-bench.sh q2-candidate \
  /absolute/path/DeepSeek-V4-Flash-Q2_K.gguf
```

A short generation, fingerprint mismatch, transport error, RDMA fallback, or
enabled profiler/dump tooling rejects a candidate. Candidate mode also runs
isolated arithmetic and 1,532-token retrieval semantic cases. Before merging
a performance change, run the combined gate:

```sh
./scripts/pre-main-tp-smoke.sh /absolute/path/DeepSeek-V4-Flash-Q4_K.gguf \
  /absolute/path/DeepSeek-V4-Flash-Q2_K.gguf
```

An intentional numerical correction must not be forced to match an obsolete
fingerprint. Pass the corrected 84-step teacher control, compare full logits at
the first changed boundary, pass the semantic suite, and then record a new
three-run median and fingerprint.

## Diagnostic rollback switches

These are investigation tools, not user setup requirements:

```sh
export DS4_TP_BIG_DIRECT=0
export DS4_ROCM_DISABLE_Q4K_WMMA=1
export DS4_ROCM_DISABLE_IQ2_I8_WMMA=1
export DS4_ROCM_DISABLE_ATTN_OUT_LOW_PACK4=1
export DS4_ROCM_DISABLE_ATTN_OUT_EXPAND_PACK4=1
export DS4_ROCM_DISABLE_ATTN_Q_B_PACK4=1
export DS4_ROCM_ATTN_OUT_Q8_A_PREQ_TOKTILE=0
export DS4_ROCM_MOE_GATE_UP_EPILOGUE_COALESCED=0
export ODL_VERBS_WC_STREAM_COPY=0
```

The historical OdinLink WC-stream-copy experiment used an alternating
off/on/on/off A/B with byte-identical generated output and required nonzero
`stream_calls` with no unexplained fallback traffic. Its full provider
isolation record is in [`../../ODINLINK.md`](../../ODINLINK.md). Current users
should keep the optimized defaults rather than reproduce that experiment.

## Rejected whole-half shared-expert balancing — Step 26

The final decode experiment dynamically moved one canonical half of the Q8
shared expert from the heavy routed-expert rank to the light rank only for 5/1
and 6/0 route splits. A model-free oracle predicted 23.98 us/layer of net
saving. Synthetic gfx1151 tests proved exact assigned-half arithmetic,
fail-closed unassigned output, signed-zero behavior, and canonical rank-group
reconstruction. The prototype used no weight cache, but required a negotiated
feature, a variable 16/32 KiB FFN payload, GPU predicates, and reconstruction.

Both mandatory-RDMA RoCE v2 runs passed their exact fingerprints:

| Workload | Prefill | Decode | FNV64 |
|---|---:|---:|---|
| 2,048 + 100 smoke | 219.75 t/s | 19.53 t/s | `80a1a4084a25abca` |
| 2,048 + 300 production | 220.55 t/s | 19.40 t/s | `5f8a983422299d76` |

The production result improved decode by only 0.03 t/s (about 0.15%), inside
the accepted run spread and below the predeclared 19.55 t/s promotion gate.
It was therefore rejected without exposing Q2_K or OdinLink to unnecessary
protocol risk. Research branch `research/q4k-hipgraph-20260818` preserves the
experiment as commit `10f2463` followed by revert `6bd5db0`; model-free oracle
commit `43e9d1a` remains available for a future architecture that can recover
the balancing benefit without changing the wire protocol.
