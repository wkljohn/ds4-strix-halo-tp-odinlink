context: caller prompt only

I read the evidence packet, STATUS.md, FABLE-REVIEW-2026-08-21.md (round 1), SPARK-SINGLE-NODE-TRANSFER.md, and verified in the checked-out tree: `rocm/ds4_rocm_fp8_kv.cuh` really has `fp8_kv_quantize_kernel` and `store_raw_kv_batch_kernel` as two separate kernels (the missing half of `69a5e83`), the HC chain has a fused tail (`hc_split_weighted_sum_norm_fused_kernel`) but separate upstream launches, and SURVEY.md's HIP-graph line reads "Capture refused on DS4 legacy default stream; launch-only <2 µs" — a per-launch bound with capture never completed, **not** a measured total launch-gap pool.

**Verdict: proceed-with-changes**

## Risks that reorder the plan

1. **The aggregate inter-launch pool has never been measured, and three of the six requested directions live or die on it.** Epilogue fusion, HC-chain fusion, and persistent scheduling all monetize the same launches. AMD PR 67's arithmetic is ~1.4 µs/launch removed (0.23 ms for 160 launches); if DS4's decode graph is ~1,000–1,300 launches/token, the launch-gap pool is 1.5–3 ms — if it is ~400, the pool is <1 ms and only the memory-round-trip component of HC fusion survives. Failure: two kernels get written against a phantom pool. Fix: oracle 0 below is a zero-load read of the existing `41934_results.db`/event profiles; it must come first and its number caps every later claim.
2. **Double counting.** The −2.0 ms `135ab50` claim and a PR-67-style epilogue fold over the HC tail remove the *same* launches. Any stack toward 4.14 ms must be counted on disjoint launch sets, or round 1's 20.3–20.7 ceiling arithmetic quietly reappears with new names. Fix: attribute savings per launch-chain segment in oracle 0 and forbid summing overlapping segments.
3. **Fingerprint exposure is concentrated in exactly two places:** RMS recompute-per-block rounding inside the HC fusion (the local plan's "explicitly round scaled input to F32" is the right guard — keep it as a bitwise gate, not a hope), and any width-fusion that changes GEMV reduction order (oracle 4). Everything else here is order-preserving epilogue work and should be bitwise by construction.
4. **Persistent scheduling inherits the Step-1 defect.** gfx1151 mapped signal words showed rare lost arrivals; a device-side sequencer spinning on completion words hits that path thousands of times per run. It also cannot span the TP gates (RDMA callbacks stay host-side), so its scope is intra-segment only — which is why it is last, not first.
5. **Even the best case does not reach 21 alone.** 4.14 ms needs HC fusion (~1.5–2 ms if the Spark claim transfers at ~75%) *plus* two or three of the small folds *plus*, likely, the already-proven gate-free row-shard (~0.8–1 ms audited). Say this in the plan now; otherwise the campaign re-runs round 1's ceiling argument at 20.5.

## Ordered oracle sequence

- **Oracle 0 — read-only launch census (kills or funds everything).** From existing rocprof/event data: launches/token, inter-launch gap histogram, per-segment attribution (HC×2/layer, GEMV tails, KV post, compressor state). *Kill:* total gap in addressable chains <1.5 ms/token → drop oracles 2 and 5 entirely, keep only the memory-traffic case for oracle 1.
- **Oracle 1 — exact HC-stage cooperative fusion twin (`135ab50` analogue; RMS→F16 proj→existing fused tail, standalone, model-free).** *Kill:* not bitwise vs the shipped three-kernel chain (RMS rounding), or <15 µs/layer/chain saved (→ <~1.3 ms/token across both chains).
- **Oracle 2 — MMVQ epilogue folds (PR-67 analogue) on the top tail patterns from oracle 0, Q4_K first.** *Kill:* epilogue not bitwise vs separate kernel, or <3 µs saved per fused pair.
- **Oracle 3 — single-kernel KV RoPE+FP8-quantize+raw-store twin (verified missing; AITER 3320 boundary, not its arithmetic).** *Kill:* <5 µs/layer, or FP8 quantize order changes the fingerprint.
- **Oracle 4 — byte-neutral replacement-width fusion (PR-59/`e221241` analogue): load-time concatenation *replacing* the originals (no duplicate), one wider dispatch; measure waves/SIMD before/after.** *Kill:* rocprof shows occupancy at these N already ≥ the wide-dispatch level, or reduction-order change breaks the exact fingerprint (then it is numerical-correction-lane only), or it is flat as PR 59 was on large models. Q2 row-pair repack rides here as the Q2-only twin.
- **Oracle 5 — persistent intra-segment scheduler on completion words, only if 0 shows a residual gap pool and 1–2 leave ≥1 ms.** *Kill:* one lost arrival in the sustained 25,800-gate ordering probe, or cooperative-launch grid cap below the widest GEMV in the segment.

## The one ≥1 ms/token candidate

**Oracle 1, the HC-stage fusion.** It is the only direction with a primary-source claim ≥1 ms (−2.0 ms in `135ab50`), a locally verified un-fused six-launch/layer chain, transport independence, and savings that are part launch-gap, part eliminated F32 intermediate round-trips — so it survives even a small oracle-0 pool. Nothing else on this list plausibly clears 1 ms alone; do not let oracles 2–4 be sequenced ahead of it on convenience.

Gate every survivor exactly as SPARK-SINGLE-NODE-TRANSFER.md already specifies: bitwise oracle before `ds4.c`, then `5f8a983422299d76` / `f9cb3a8a17e95c71`, full 2048+300 on both RoCE v2 and OdinLink.
