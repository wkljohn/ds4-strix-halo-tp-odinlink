# Single-node `ds4-on-spark` transfer audit

Date: 2026-08-21

Scope: ordinary greedy DeepSeek V4 Flash decode, without DSpark. The target is
cache-free gfx1151 TP=2 over either RoCE v2 or OdinLink. The public
[`ds4-on-spark`](https://github.com/Entrpi/ds4-on-spark) repository is an
installer and benchmark wrapper; release `v0.6.3` pins the engine in
[`Entrpi/ds4`](https://github.com/Entrpi/ds4).

## What the one-node result actually establishes

The wrapper reports about 20.0 t/s plain decode at 2K on one GB10. Its larger
decode headline includes DSpark and must not be used as the no-DSpark control.
The plain-path uplift is nevertheless real: it includes several exact or
quality-gated serial-decode fusions in the pinned engine, not merely CUDA
graphs, batching, or speculation.

The model is also different. Spark's reference GGUF is an approximately 81 GiB
asymmetric Q2 layout (IQ2_XXS routed gate/up, Q2_K routed down, Q8_0 dense
weights). This project must remeasure its own Q4_K and Q2_K baselines and may
transfer only operation structure, not Spark's throughput number.

## Concrete single-node decode changes

| Engine commit | Mechanism and reported local result | gfx1151 TP assessment |
|---|---|---|
| `135ab50` | One cooperative HC-stage kernel replaces RMS, F16 HC projection, combine, Sinkhorn/weighted sum, and output norm launch chains; reported about 1.9--2.0 ms/token saved | **Highest-priority oracle.** Our ROCm path fuses the HC tail but still launches RMS, projection, and tail separately twice per layer. A HIP cooperative twin must preserve the current operation order and global scratch values exactly; do not import the algebraically moved RMS scale if it changes the Q4/Q2 fingerprints. Transport-independent. |
| `69a5e83` | Fused Q head norm+RoPE and KV RoPE+FP8 quantize+raw store; 7 launches to 3, about 0.3--0.6 ms/token in the Spark A/B | **Partially present.** ROCm already fuses Q_B+Q norm+RoPE and KV normalization+RoPE. Its KV store still has a separate FP8 quantize and raw-store chain. Test only the missing KV tail in a standalone exact oracle. |
| `02f6914` | Cooperative F16 router projection+combine+top-6 select; roughly 20.7 to 14.7 us/layer in the CUDA prototype, but only about 0.2 ms/token end-to-end | **Low priority.** Current ROCm measurements put routing near 3 us/layer, so the available pool is much smaller than Spark's. Reprofile before porting. |
| `fd71740` | Cooperative paired compressor projections+combines+store; five launches to one, about 0.2 ms/token realized after graph capture | **Small but composable.** ROCm already pairs the two projections and temporally batches state work. A current post-temporal profile is required before an oracle. |
| `e221241` | Byte-neutral row-pair SoA Q2_K down repack replacing the raw representation; 0.6--0.9 ms/token | **Q2-only and conditional.** The load layout is useful prior art, but a second derived copy violates the cache-free rule. It is admissible only as a replacement representation or load-time in-place artifact, with Q4 unaffected and Q2 fingerprint exact. |
| `d8cf4f9`, `e50104f` | Per-layer/"decode island" CUDA graph capture | **Not the first step.** Prior gfx1151 probes bounded launch-only savings and TP callbacks cannot be captured. Revisit only after a fixed explicit HIP stream captures a large measured fraction while leaving both RDMA callbacks eager. |
| `17a7d76`, `e0ed742` | Head-group dense/indexed flash decode | **Long-context lane.** Large wins were reported at 240K--515K, not the fixed 2K headline. ROCm already has online head-group kernels; the next action is geometry/profile comparison, not a source transplant. |
| `0f62c62`, `dc51d64`, `1464c0f` | D2R/tensor-core and tiled Q4 expert work | **Prefill/verify-width prior art.** These do not establish an ordinary one-token Q4_K decode win on gfx1151. |

## Combination that can still challenge 21 t/s

The fresh Q4 RoCE control is 19.32 t/s, or 51.76 ms/token. Reaching 21 t/s
requires about 4.14 ms/token. No single Spark mechanism proves that saving on
gfx1151, but the following exact, transport-neutral stack is large enough to
test rather than dismiss:

1. Exact HC-chain cooperative fusion, isolated first.
2. Missing KV post/store fusion, isolated second.
3. Only after both pass independently, combine them and remeasure the residual
   FFN rank-skew pool.
4. If the combined local fusions leave at least about 1.5 ms/token, revisit the
   already-proven FFN row-shard arithmetic with a gate-free ordering primitive;
   never add a third host callback.

Each implementation begins as a model-free production-shape oracle. It must
match the current kernel chain bitwise before entering `ds4.c`. Integration is
then gated by the Q4 fingerprint `5f8a983422299d76`, Q2 fingerprint
`f9cb3a8a17e95c71`, three unprofiled Q4 RoCE v2 runs, and full 2048+300 runs
over both RDMA providers. A transport-neutral source change is not evidence of
transport neutrality until both providers pass.

## Do not transfer

- DSpark acceptance/yield-quench numbers into ordinary decode.
- Continuous batching or aggregate throughput into single-stream t/s.
- CUDA/Blackwell MMA instructions, persistent aligned-weight duplicates, or
  the resident weight server.
- Deep-context attention gains into the 2K headline without a matched run.
- Any fusion that preserves semantic quality but changes the exact arithmetic
  while claiming to be an identical-arithmetic optimization.
