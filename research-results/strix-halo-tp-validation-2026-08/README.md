# Strix Halo TP=2 optimization and validation record — August 2026

This note preserves the implementation details, rejected experiments, raw run
values, rollback switches, and maintainer validation procedure behind the
short user-facing results in the repository README. The public benchmark is
`ds4-bench-tp`: a fixed 2,048-token prefill followed by 300 generated tokens,
with mandatory RDMA and cache-free model weights.

## Benchmark provenance

The current Q4_K rows test
[antirez/deepseek-v4-gguf](https://huggingface.co/antirez/deepseek-v4-gguf).
The Q2_K row tests
[huihui-ai/Huihui-DeepSeek-V4-Flash-0731-abliterated-GGUF](https://huggingface.co/huihui-ai/Huihui-DeepSeek-V4-Flash-0731-abliterated-GGUF).

| DeepSeek V4 0731 configuration | Tested model | Accepted runs, prefill/decode t/s | Reported result |
|---|---|---|---|
| Original Q4_K TP=2 baseline | archived Q4_K artifact | archived 34.11/9.96 | 34.11/9.96 |
| Huihui Q2_K over RoCE v2 | Huihui `DeepSeek-V4-Flash-Q2_K-0731.gguf` | 202.86/19.49, 202.83/19.45, 201.68/19.51 | three-run median **202.83/19.49** |
| Antirez Q4_K over OdinLink | Antirez `DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix-0731.gguf` | 231.54/19.19, 231.46/19.13, 231.34/19.03 | three-run median **231.46/19.13** |
| Antirez Q4_K over RoCE v2 | same Antirez model | 273.70/19.54, 272.51/19.70, 271.42/19.44 | three-run median **272.51/19.54** |
| Current Q4_K + DSpark | target-model revalidation pending | — | opt-in; not a current performance claim |

The current Antirez Q4_K fingerprint is `b7694f9d11a3760e`; the current Huihui
hybrid Q2_K fingerprint is `f9cb3a8a17e95c71`. Each fingerprint is specific
to the named model, prompt, balanced 128/128 expert split, and 300-token greedy
workload. It is a deterministic self-regression gate, not proof of numerical
parity with llama.cpp or between the two model releases.

All six current Antirez Q4_K runs used the same 164,633,502,592-byte model,
`ds4` binary, 2,048+300 workload, balanced split, and cache-free settings. They
produced the same token fingerprint over both providers. Comparing the two
three-run medians, RoCE v2 was 17.7% faster for prefill and 2.1% faster for
decode than OdinLink. The exact artifact identities, binary hashes, and run
values are preserved in the
[`Antirez Q4_K RDMA validation`](../antirez-q4-rdma-2026-08-21/).

The previous public Q4_K results used Huihui
`DeepSeek-V4-Flash-Q4_K-0731.gguf` and fingerprint `5f8a983422299d76`.
Those values remain historical evidence rather than the current README claim:
OdinLink runs were 218.24/19.02, 217.72/18.88, and 220.39/18.83; RoCE v2 runs
were 258.64/19.14, 256.90/19.22, and 259.54/19.33.

An earlier August 14 transport A/B used the same Q4_K model, binary, workload,
and fingerprint on both providers. Matched 2,048-token-chunk OdinLink runs measured
198.55/14.88 and 192.63/14.84 t/s. Their midpoint was 195.59/14.86 versus
222.76/17.08 for RoCE v2: **+13.9% prefill and +14.9% decode**. RoCE replaces
only the TP communication slab with a 77.4 MiB mapped allocation; it adds no
expanded model-weight cache. The allocator and transport evidence is in
[`../roce-v2-2026-08-14/`](../roce-v2-2026-08-14/).

The preceding release had two clean rebuilt-binary checks at 219.22/19.22 and
220.00/19.36 t/s, both reproducing `5f8a983422299d76`. Its earlier controlled
three-run median was 219.56/19.37. The same preceding binary also passed Q2_K
over RoCE v2 at 180.19/19.17 with
`f9cb3a8a17e95c71`, and Q4_K over explicit OdinLink RDMA at 201.71/19.00 with
`5f8a983422299d76`. Neither compatibility check used provider fallback.

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
boundary. Together these exact, cache-free changes produced the then-current
19.22 t/s Huihui Q4_K RoCE median. The benchmark and deployment launchers enable them
for ordinary inference and negotiate the temporal feature between both ranks.

The current launcher adds two cache-free, non-DSpark prefill improvements. A
shape-gated M256/K128 Q8 projection applies on gfx1151. On mlx5, a four-wave
FFN consumer schedule overlaps each 512-row wave with the next attention
prefix; the same setting provider-gates itself off on OdinLink. A live two-node
32 MiB big-gate test was exact on both ranks and hid 52.5--59.2% of measured
wire time.

A 6,657-token A/B exercised three full 2,048-token chunks plus a 513-token
tail. Disabling both improvements measured 221.52/17.65; the launcher defaults
measured 231.65/17.81. Both produced `0d9eda8d7a1ff814`, and candidate mode
passed the semantic suite. This closes the full-chunk-to-tail transition that
the 2,048-token headline workload does not cover.

After launcher promotion, the combined pre-main gate passed again at
258.33/19.31 for Q4_K and 202.21/18.97 for Q2_K with their established exact
fingerprints. The ordinary deployment exposes symmetric rollback settings as
`PREFILL_FFN_WAVEFRONT=0` and `Q8_M256_K128=0`; the kernel-global defaults were
not changed. Deployment also pins the local `ds4-server` SHA-256 so a stale
coordinator cannot enter a different gate schedule.

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
./run-tp-ds4-bench.sh q4-r1 /absolute/path/DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix-0731.gguf
./run-tp-ds4-bench.sh q4-r2 /absolute/path/DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix-0731.gguf
./run-tp-ds4-bench.sh q4-r3 /absolute/path/DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix-0731.gguf
```

RoCE v2 uses the same harness:

```sh
DS4_BENCH_RDMA_PROFILE=roce-v2 \
  ./run-tp-ds4-bench.sh q4-roce-r1 \
  /absolute/path/DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix-0731.gguf
```

Optimization candidates must name the expected fingerprint:

```sh
DS4_BENCH_CANDIDATE=1 \
DS4_BENCH_EXPECT_FNV64=b7694f9d11a3760e \
  ./run-tp-ds4-bench.sh q4-candidate \
  /absolute/path/DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix-0731.gguf

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
DS4_PREMAIN_Q4_FNV64=b7694f9d11a3760e \
  ./scripts/pre-main-tp-smoke.sh \
  /absolute/path/DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix-0731.gguf \
  /absolute/path/DeepSeek-V4-Flash-Q2_K-0731.gguf
```

The pre-main script's built-in Q4_K default remains the historical Huihui
fingerprint `5f8a983422299d76`; the explicit override above selects the current
Antirez model without weakening either model-specific gate.

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

## Production incident — incremental prefill after temporal decode

The first 256K service deployment of the temporal schedule exposed a workload
missing from the fixed benchmark: a tool continuation decoded 11 tokens and
left two deferred compressor rows, then the next request appended an 89-token
suffix through resumed layer-major prefill. The batch path advanced recurrent
state to position 25,463 but left the decode-only pending marker at 25,372+2.
The next decode failed closed with:

```text
ROCm temporal compressor position discontinuity at layer 2
(25372 + 2 != 25463)
```

This was a session-state transition bug, not RDMA loss: neither node logged a
kernel panic, GPU reset, mlx5 error, or reboot. The worker intentionally exits
after a mirrored evaluation error, while the leader originally retained its
HTTP listener.

The fix materializes any deferred temporal rows before switching an extending
session from token decode to resumed layer-major prefill. Mirrored sync/eval
failures now also mark the leader TP transport failed immediately, and the
deployment status command requires coordinator, worker, and API health before
reporting readiness. The synthetic oracle includes the exact 11+89 boundary
for ratio-4 and ratio-128 layers; both recurrent states and emitted rows match
sequential updates bit for bit. The release gate additionally requires a real
multi-request API continuation that crosses the same boundary while both TP
ranks remain alive.

Validation on the fixed branch used matching binaries on both nodes:

| Gate | Prefill | Decode | Fingerprint/result |
|---|---:|---:|---|
| Q4_K RoCE v2, 2,048+300 | 219.02 t/s | 19.32 t/s | `5f8a983422299d76` |
| Q2_K RoCE v2, 2,048+300 | 179.05 t/s | 19.11 t/s | `f9cb3a8a17e95c71` |

The API reproduction decoded 11 tokens, appended a 245-token live suffix,
decoded eight, appended another 395-token live suffix, and decoded eight more.
The coordinator log retained the live ranges `343..588` and `596..991`; both
ranks stayed ready and the two post-prefill decode samples measured 19.83 and
19.60 t/s. Finally, terminating the test worker and issuing another request
returned HTTP 500 for that request and HTTP 503 from `/health`; the deployment
status reported `worker-stopped` and no longer reported the API ready.
