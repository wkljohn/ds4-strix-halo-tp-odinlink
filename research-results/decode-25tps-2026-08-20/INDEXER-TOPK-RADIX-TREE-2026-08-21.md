# Exact long-context indexer top-k radix tree

Date: 2026-08-21

## Root cause and boundary measurement

The fast indexer top-k path ended at 8,192 compressed rows.  With the model's
4:1 index compression ratio, decode immediately after a 32,768-token prefill
crossed into the generic chunked bitonic tree.  A matched Q4_K RoCE v2 test
using the same `promessi_sposi.txt` prefix, 300 generated tokens, 2,048-token
prefill chunks, exact cooperative HC stage, and cache-free defaults measured:

| Frontier | Prefill | Decode | Steady decode | First token | Fingerprint |
| --- | ---: | ---: | ---: | ---: | --- |
| 30,720 (below boundary) | 211.09 t/s | 16.83 t/s | 16.87 t/s | 83.496 ms | `784e3e95cd17a6d0` |
| 33,792 (above boundary) | 208.67 t/s | 16.49 t/s | 16.55 t/s | 115.807 ms | `59a7cf4d6737efbf` |

The boundary therefore cost about 1.15 ms per steady token and 32.3 ms on the
first token in this matched test.  It explains a useful part, but not most, of
the total long-context slowdown.

## Model-free candidate

`scripts/indexer_topk_long_bench.cu` compares the shipped two-stage bitonic
tree with an exact packed-key radix tree.  A single 12,288-item CUB block was
rejected at compile time because its 98,304-byte TempStorage exceeds gfx1151's
65,536-byte per-block limit.  The accepted layout radix-sorts independent
4,096-row chunks, keeps their exact top 512, then radix-sorts at most 2,048
candidates.

The lower 32 bits of each key encode the inverse original index, retaining the
established descending-score and ascending-index tie order.  Random finite
scores with deliberate ties matched the bitonic output exactly at every tested
shape.

| `n_comp` | Tokens | Existing | Radix tree | Speedup | Index differences |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8,193 | 1 | 74.320 us | 44.049 us | 1.687x | 0 |
| 8,448 | 1 | 74.564 us | 44.688 us | 1.669x | 0 |
| 12,288 | 1 | 74.700 us | 45.553 us | 1.640x | 0 |
| 8,448 | 16 | 123.627 us | 70.164 us | 1.762x | 0 |
| 8,448 | 256 | 1,325.054 us | 789.555 us | 1.678x | 0 |

At the real 8,448-row decode shape, the isolated saving is 29.876 us per
layer, or about 1.28 ms over 43 layers.

## Production validation

The exact one-token path is enabled by default after cross-model and
cross-provider validation. Disable it on both TP ranks with:

```sh
export DS4_ROCM_INDEXER_TOPK_RADIX_TREE=0
```

The path is restricted to gfx1151, one-token decode, `top_k=512`, and
`8192 < n_comp <= 16384`. Batched prefill retains the established path: an
alternating three-run Q4_K A/B found a positive decode median but a small
prefill regression when the radix tree was used for both. All other shapes and
architectures retain their previous dispatch. TP hello bit
`DS4_TP_FEATURE_INDEXER_TOPK_RADIX_TREE`
prevents independently launched ranks from selecting different index rows.
The candidate reuses the existing temporary arena and adds no persistent
memory.

The full 33,792+300 Q4_K RoCE v2 candidate measured:

| Path | Prefill | Decode | Steady decode | First token | Fingerprint |
| --- | ---: | ---: | ---: | ---: | --- |
| Existing bitonic tree | 208.67 t/s | 16.49 t/s | 16.55 t/s | 115.807 ms | `59a7cf4d6737efbf` |
| Exact radix tree | 207.54 t/s | 16.73 t/s | 16.74 t/s | 61.520 ms | `59a7cf4d6737efbf` |

The original all-token candidate gained 1.2% steady decode in its first run.
The completed alternating three-run medians were:

| Path | Median prefill | Median decode | Median steady decode |
| --- | ---: | ---: | ---: |
| Existing bitonic tree | 208.67 t/s | 16.49 t/s | 16.55 t/s |
| Radix tree for decode and prefill | 207.92 t/s | 16.73 t/s | 16.74 t/s |

That is +1.46% decode and +1.15% steady decode, but -0.36% prefill. The
production-disabled dispatch was therefore narrowed to one-token decode before
provider/model validation; batched prefill no longer selects it.

The first full decode-only rerun measured 207.60 prefill, 16.72 decode, and
16.74 steady decode t/s with the same `59a7cf4d6737efbf` fingerprint. Its
prefill result remained inside the observed 207.54--208.85 t/s run spread even
though the radix kernel was unreachable during prefill. Treat the earlier
0.36% prefill midpoint difference as run noise, not a candidate regression.

Short-context candidate gates proved the feature is inert below the boundary:

| Model/provider | Prefill | Decode | Fingerprint | Result |
| --- | ---: | ---: | --- | --- |
| Q4_K RoCE v2 | 257.70 t/s | 19.59 t/s | `5f8a983422299d76` | pass |
| Q2_K RoCE v2 | 202.83 t/s | 19.45 t/s | `f9cb3a8a17e95c71` | pass |
| Q4_K OdinLink | 217.01 t/s | 19.27 t/s | `5f8a983422299d76` | pass |

Each candidate passed the mandatory RDMA proof, exact 300-token fingerprint,
semantic smoke, retrieval case, and fail-closed feature negotiation.  The
ROCm model-free integration test is `make test-rocm-long-context`; TP hello is
covered by `make test-tp-hello`.

A matched 33,792+300 OdinLink check used the same 2,048-token chunks and long
prompt as the RoCE test. Control measured 179.60/15.94 prefill/decode t/s and
radix measured 180.47/15.94 t/s; both produced fingerprint
`59a7cf4d6737efbf`. The path is therefore exact and performance-neutral when
OdinLink communication masks its isolated saving. An earlier apparent
fingerprint mismatch was invalid evidence: that run accidentally used a
4,096-token prefill chunk, which changes the deterministic trajectory.
