# Q4_K decode launch census — 2026-08-21

## Current diagnostic

The current research-branch source at `7f2a6ed` was built with `PROFILE=1` and
run over RoCE v2 for 2,048 prefill + 50 decode tokens. The diagnostic completed
at 278.64 prefill / 19.53 decode t/s with short-run fingerprint
`7c63bb936d1cc076`. This is diagnostic evidence, not a 300-token candidate.
Rank 0's long startup was model-page warming, not a hang. Keeping
`DS4_TP_CONNECT_TIMEOUT_SEC=3600` allowed rank 1 to wait for the slower rank.

Layer-20 event medians after discarding the first two samples were:

| Stage | Rank 0 | Rank 1 |
| --- | ---: | ---: |
| Q path | 152 us | 157 us |
| KV path | 24 us | 27 us |
| attention output | 262 us | 222 us |
| FFN HC pre chain | 68 us | 72 us |
| router | 52 us | 56 us |
| shared gate/up | 108 us | 113 us |
| shared down | 64 us | 65 us |
| routed MoE folded segment | 247 us | 249 us |

`attn_hc_pre` is not a local-chain timer at layer 20: it includes the preceding
layer walk because it is the first marker in the selected layer. The per-kernel
census below is the usable HC bound.

After this run, the validated production binaries were restored on both nodes:

- `ds4`: `f513715844f0bde990dff0caef470d4288a3918b3cabb07d8b7c2e1213a613aa`
- `ds4-bench-tp`: `c3b724140b5c24d971c5865d4230818ce95b7a4c3820a16424133af5aa859a33`

## Existing 300-token rocprof census

The selected-region database is:

`research-results/q4k-step0-2026-08-17/profile/rocprof-step0-q4-roce-profile-r1/wkljohn-NucBox-EVO-X2/41934_results.db`

Its process metadata proves a 2,048 + 300 Q4_K RoCE v2 run. Exact layer
multiples show that the 442,210 dispatches are the decode window:

- 442,210 dispatches = 1,474.03 launches/token;
- 13.9941 seconds of GPU kernels = 46.647 ms/token;
- 19.1348-second dispatch span;
- 5.1407 seconds between dispatches = 17.136 ms/token under tracing.

The last number includes TP callbacks, real rank skew, and profiler overhead. It
is not a claim that 17.136 ms is locally fusible.

### Dominant exact kernel populations

| Kernel family | Calls/token | Kernel ms/token |
| --- | ---: | ---: |
| Q8 pack4 GEMV | 86 | 7.431 |
| paired F16 projection | 62 | 5.679 |
| Q8 non-pack4 GEMV | 44 | 4.649 |
| ordered-chunk F16 projection | 130 | 3.432 |
| shared gate/up SwiGLU | 43 | 3.883 |
| grouped Q8 attention output | 43 | 3.540 |
| mixed attention | 43 | 3.214 |
| Q4_K routed down | 43 | 3.171 |
| Q4_K routed gate | 43 | 2.998 |
| Q4_K routed up | 43 | 2.958 |
| paired Q8 projection | 43 | 1.559 |
| plain RMS | 87 | 1.240 |
| fused HC weighted-sum/norm tail | 86 | 0.870 |

The HC pre-chain appears exactly twice per layer. Removing the unrelated
one-per-token plain RMS and one-per-layer router projection gives this current
HC kernel budget:

```text
plain RMS                   about 1.23 ms/token
two F16 HC projections      about 2.27 ms/token
two fused HC tails          about 0.87 ms/token
------------------------------------------------
HC pre-chain kernel total   about 4.37 ms/token
```

This makes Entrpi/ds4's reported 2.0 ms/token cooperative-fusion gain an upper
bound worth testing, not a transferable result. An exact recompute-RMS fused
projection has a credible local target of roughly 1.2--1.5 ms/token: the
standalone RMS pool plus a small number of launch gaps, reduced by redundant
RMS work and any occupancy cost inside the fused projection.

### Gap attribution

Two transition pairs dominate the traced gap sum:

| Transition | Calls/token | Traced gap ms/token |
| --- | ---: | ---: |
| attention partial/add to HC expansion | 43 | 5.398 |
| FFN partial GEMV to HC expansion | 43 | 4.334 |

These are the two TP gate boundaries. Their traced values include profiler
inflation; they nevertheless confirm that the large residual pool is the TP
boundary/skew path, not ordinary local launch overhead.

All gaps up to 10 us sum to 5.479 ms/token under tracing. The established
untraced launch-only measurement is below 2 us/launch, so this cannot be used as
a production speedup estimate. It is only a census for finding fusion pairs.

## Resolution of the two Fable rounds

The rounds disagree on which oracle comes first, but the measurements make the
dependency clear:

1. HC fusion can plausibly contribute at least 1 ms/token and is independent of
   transport ordering. Build the exact model-free twin.
2. HC fusion alone cannot close the 4.14 ms/token gap from 19.32 to 21 t/s.
3. Gate-free FFN row balancing is the remaining large complementary pool. Its
   completion-word ordering must pass a sustained probe on RoCE v2 and
   OdinLink before integration.
4. GEMV epilogue and KV-post fusion remain small riders after the HC oracle.

Fable Round 1 misread `5/7 bitwise PASS` as five of seven cases passing. It is
the expert split ratio of the `non-half split` case. All five production-shape
oracle cases pass full-F32 bitwise comparison; there are no two unexplained
failures.

## HC cooperative oracles

`scripts/hc_cooperative_grid_probe.cu` first tested the intended 24-block by
256-thread geometry without model arithmetic. gfx1151 reports cooperative
launch support, 20 HIP multiprocessors, eight resident blocks per
multiprocessor, and a 160-block resident grid limit. Ten thousand launches with
three grid-wide synchronization points completed with zero ordering errors at
10.075 us/launch.

`scripts/hc_stage_exact_coop_bench.cu` then reproduced the current one-token
three-launch arithmetic at the exact 16,384 x 24 shape and compared two
single-launch cooperative twins. Both twins are bitwise identical in the final
4,096 F32 normalized values:

| Variant | Five-run range | Result |
| --- | ---: | --- |
| current RMS + ordered F16 projection + fused tail | 48.403--48.518 us | reference |
| cooperative, retaining current F32 flat scratch | 36.721--36.830 us | 23.9--24.3% faster, bitwise |
| cooperative, normalized value materialized per dot | 50.751--50.809 us | 4.6--4.9% slower, bitwise |

The scratch-preserving form is the surviving production candidate. It uses the
already allocated `flat_hc`, changes no persistent memory, preserves the
current 256-thread RMS tree, preserves every normalized F32 store, preserves
the ordered 32-chunk F16 dot, and executes the existing HC tail arithmetic
inside block zero. Its median saving is about 11.7 us/chain, or approximately
1.01 ms/token across two chains and 43 layers. The scratch-free form is closed:
repeating the normalization multiply inside all 24 dot rows costs more than the
saved scratch traffic.
