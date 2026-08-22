# Exact shared-Q8 verifier weight reuse — 2026-08-22

## Result

The TP verifier previously evaluated the rank-local Q8 shared expert one row
at a time.  The accepted ROCm candidate reuses each compact Q8_0 weight block
across two to five verifier rows for both gate/up and the K-sliced down
projection.  It retains the target model's exact token trajectory and adds no
persistent allocation or expanded-weight cache.

| Final binary run | Provider / mode | Prefill | Decode | FNV64 |
|---|---|---:|---:|---|
| r1 | RoCE v2 + DSpark | 190.55 t/s | 16.19 t/s | `7174e214e05fd83e` |
| r2 | RoCE v2 + DSpark | 190.35 t/s | 16.01 t/s | `7174e214e05fd83e` |
| r3 | RoCE v2 + DSpark | 190.38 t/s | 15.94 t/s | `7174e214e05fd83e` |
| cross-provider | OdinLink + DSpark | 181.04 t/s | 16.35 t/s | `7174e214e05fd83e` |
| ordinary control | RoCE v2, no DSpark | 258.45 t/s | 19.65 t/s | `b7694f9d11a3760e` |

The final RoCE median is **190.38 prefill / 16.01 decode t/s**.  Decode is
6.4% above the preceding exact 15.04 t/s median.  The three DSpark runs retain
the same 143 cycles, 235 accepted drafts, 2.098 final tokens/cycle, and exact
acceptance histogram.  This is verifier execution improvement, not a changed
acceptance policy.

OdinLink recorded 37,908 WC streaming calls, 3.49 GB, and zero fallback calls
on both ranks.  The ordinary control did not negotiate or select the
DSpark-only feature.

## Arithmetic failure and ISA correction

The first live prototype fused gate/up/SwiGLU using the algebraically
equivalent direct-F32 kernel.  It reached 15.72 t/s but changed the token
fingerprint to `57f8dfd10bd62801`, so it was rejected.  A full-mantissa oracle
then showed that the shipped runtime dispatch is Q8 pair projection followed
by a separate SwiGLU launch; matching the formula was insufficient.

A second weight-outer gate/up attempt still differed in 1,659 W2 outputs.  ISA
inspection identified the exact cause.  The shipped gfx1151 kernel emits:

```text
scaled = activation * FP16_block_scale
accumulator = fma(scaled, int8_weight, accumulator)
```

The compiler reassociated the weight-outer source into
`int8_weight * scale` before the FMA.  An inline gfx1151 `v_mul_f32_e32`
forces the shipped first multiply while leaving the same wave reduction tree.
This made the weight-reusing gate/up kernel bit-exact at every supported
width.  Down projection uses the same per-token accumulator order and wave
reduction while sharing its Q8 load.

## Isolated production-shape gate

`tests/test_rocm_shared_verify_rows_exact.cu` uses non-power-of-two FP16 Q8
scales and full-mantissa signed F32 activations.  It compares the production
one-row dispatch with the candidate for gate/up/SwiGLU and both rank down
K-slices (`k_off=0` and `1024`).

| Width | Serial shared path | Weight-outer path | Speedup | Exact |
|---:|---:|---:|---:|:---:|
| 2 | 0.2100 ms | 0.1050 ms | 2.00x | yes |
| 3 | 0.3215 ms | 0.1142 ms | 2.81x | yes |
| 4 | 0.4202 ms | 0.1187 ms | 3.54x | yes |
| 5 | 0.5380 ms | 0.1280 ms | 4.20x | yes |

All widths clear the predeclared 1.25x isolated gate.  At W5 each kernel uses
10 KiB of tile-local LDS and discards it at launch completion.  No model-sized
or persistent buffer is introduced.

## Verifier accounting

The preceding exact implementation spent a three-run median 13,417.7 ms in
83 full verifiers, or 161.66 ms/invocation.  This candidate's corresponding
times were 12,052.8, 12,003.3, and 12,075.5 ms.  Their median is 145.21
ms/invocation, a 10.2% verifier reduction.

At the unchanged acceptance rate, matching ordinary 19.65 t/s by verifier
work alone requires approximately 103 ms/invocation.  The current path still
needs about another 29% verifier reduction to clear the lossless milestone.
Reaching 30 t/s with all other measured costs unchanged would require roughly
40 ms/invocation; it is not credible from another small shared-Q8 tweak.

## Dispatch and rollback

The candidate is restricted to the validated shared-expert geometry:

```text
gate/up: 4096 -> 1024
down:    rank K-slice 1024 of full 2048 -> 4096
rows:    2..5
```

TP hello feature bit 26 rejects asymmetric selection.  Enable it on both
ranks with:

```sh
export DS4_ROCM_DSPARK_SHARED_Q8_ROWS_EXACT=1
```

Leaving the variable unset is the rollback path.  The feature is advertised
only for an active ROCm DSpark session, so ordinary inference and other GPU
architectures retain their previous dispatch.

## HIP graph probe

A separate bounded probe found that ROCm 7.2 on gfx1151 rejects stream capture
on legacy stream 0, succeeds on a private stream, and succeeds on the default
stream when compiled with `-fgpu-default-stream=per-thread`.  The measured
verifier ledger nevertheless showed about 127 ms of kernel-active work before
this shared-Q8 change.  Even removing the complete host/launch gap could not
reach the ordinary milestone, so graph integration was rejected before adding
runtime complexity.  The temporary probe and extracted code objects were
removed after recording the result.

## Evidence

Raw CSV, manifests, and coordinator/worker logs are under `runs/` with tags:

```text
antirez-q4-dspark-shared-q8-weightouter-all-r1
antirez-q4-dspark-shared-q8-weightouter-all-r2
antirez-q4-dspark-shared-q8-weightouter-all-r3
antirez-q4-dspark-shared-q8-weightouter-all-odin-r1
antirez-q4-ordinary-shared-q8-weightouter-all-control-r1
```

The final binary hashes on both independent filesystems were:

```text
ds4          b3ddd0b1f6cd8b4e445bdc732fa12fe215d6d6069dbf2a75134213653dec18d0
ds4-bench-tp cba45a2b723479d360dcb8d63260cc5e401a6200ce559bf59bed8a11a3441e67
```
