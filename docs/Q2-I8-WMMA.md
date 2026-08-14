# Hybrid Q2 integer-WMMA prefill

Date: 2026-08-14

This path accelerates the IQ2_XXS gate/up projections in DeepSeek V4 Flash's
hybrid IQ2_XXS/Q2_K routed layers on gfx1151 TP=2. It does not create an
expanded-weight cache and does not change decode.

## Design

For prefill only, DS4 dynamically repacks the existing Q8_K activations into
the llama.cpp-compatible Q8_1 MMQ layout. Each workgroup expands only its
current IQ2_XXS weight tile into signed INT8 in LDS, executes native
`v_wmma_i32_16x16x16_iu8`, applies the original per-32-value IQ2 scales, and
writes the fused SwiGLU/router result. Gate and up share the activation tile.
Compact GGUF weights remain the only persistent weight representation.

The 2,048-token benchmark uses about 9 MiB of reusable temporary Q8_1 scratch
per rank. Scratch scales linearly with the prefill chunk; it is not a converted
model cache.

The implementation preserves the established safe route schedule. Exact-zero
peer routes are ignored inside the tile, hot experts use integer WMMA, and the
existing scalar path handles cold experts. The 16-token semantic case exercises
a mixed hot/cold schedule before the long benchmark.

## Dispatch safety

The path is default-on only when all of these are true:

- ROCm reports `gfx1151`;
- TP expert sharding is active;
- gate/up are IQ2_XXS with the validated 4096-by-2048 model shape;
- both independently launched ranks advertise
  `DS4_TP_FEATURE_IQ2_I8_WMMA` in the exact-matched TP hello;
- quality mode is off and temporary scratch allocation succeeds.

Any unsupported shape fails closed to the established FP16-dequant WMMA path.
An environment mismatch between ranks fails during TP hello rather than
running different arithmetic. `DS4_ROCM_DISABLE_IQ2_I8_WMMA=1` on both ranks
is the explicit rollback. The feature is compute-only: OdinLink, Mellanox, and
generic verbs use their existing transport paths.

## Correctness evidence

The standalone microharness in `scripts/iq2_wmma_microbench.cu` checks the
IQ2_XXS codebook/sign expansion and integer tile result against the established
DP4A arithmetic. It reported zero expansion or arithmetic mismatches and the
generated gfx1151 ISA contained native integer-WMMA instructions.

The pre-main gate requires both Q4_K and Q2_K to pass:

- mandatory RDMA with zero fallback traffic;
- exact 2,048-prefill + 300-decode token fingerprints;
- isolated arithmetic semantics;
- an isolated 1,532-token retrieval case;
- exact TP feature matching and observed IQ2 kernel engagement.

The accepted Q2 fingerprint is `f9cb3a8a17e95c71`. The arithmetic case
returned 4 and the retrieval case returned 731942 on every acceptance run.

## Performance

All results use balanced 128/128 experts, mandatory OdinLink RDMA, no DSpark,
and no Q8-to-FP16 weight cache.

| Path | Prefill | Decode | Fingerprint |
|---|---:|---:|---|
| Safe Q2 control before mixed-Q4 discovery | 127.07 t/s | 14.75 t/s | `c000c594c5ea0328` |
| Mixed Q4 discovery, before IQ2 integer WMMA | 149.85 t/s | 14.72 t/s | `fec62421edc8d73c` |
| Final IQ2 integer-WMMA median | **162.78 t/s** | **14.68 t/s** | `f9cb3a8a17e95c71` |

The final clean samples were 160.38/14.68, 162.78/13.77, and 164.07/14.73
prefill/decode t/s. The second run retained 14.03 steady decode after an
isolated 463 ms first-token stall; it remains in the median rather than being
discarded. Three earlier performance samples measured 165.25, 168.41, and
166.11 prefill t/s with the same fingerprint.

A final disable-switch control negotiated `iq2_i8=0` on both ranks, emitted no
integer-WMMA engagement log, reproduced the prior `fec62421edc8d73c`
fingerprint, and retained zero RDMA fallback calls. This proves the rollback
selects the established arithmetic path rather than only suppressing logging.

Candidate mode deliberately runs semantic cases before the timed workload and
therefore heats the UMA system. Its final pre-main result was 151.69/14.62 t/s;
that number is correctness evidence, not the clean performance measurement.

## Prior art and remaining work

The IQ2 expansion and Q8_1 activation representation follow llama.cpp's
IQ2_XXS MMQ structure; DS4's existing DP4A dot implementation is the local
correctness oracle. vLLM and AITER offer expert-sorting concepts but no
transferable gfx1151 GGUF IQ2_XXS kernel: their fused-MoE implementations target
CDNA MFMA, FP8, or conventional INT8 formats.

The next plausible Q2-only lever is an integer-WMMA Q2_K down projection using
the already validated Q4_K routed-down structure and Q8_1 activation stream.
It was deliberately left out of this promotion because the gate/up change
already exceeded the 160 t/s target and the down kernel requires a separate
min/scale correctness campaign.

Run the complete gate with:

```sh
./scripts/pre-main-tp-smoke.sh /absolute/path/model-Q4_K.gguf \
  /absolute/path/model-Q2_K.gguf
```
