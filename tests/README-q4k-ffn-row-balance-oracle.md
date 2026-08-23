# FFN row-balance and gate-free K-shard oracle

See `$DS4_RESEARCH_ROOT/decode-25tps-2026-08-20/CODEX-GATE-7.md`.

```
make test-rocm-q4k-ffn-row-balance-oracle
```

Before the numerical cases, the test runs a warm, device-resident,
kernel-only shape-cost gate. It alternates 33 `hipEvent` samples of the current
three-full-expert TP rank against both six-expert packed K-halves. Both arms
execute 40.5 MiB of unique Q4_K weight bytes through the shipped one-token
primitive. The test prints the worse-half ratio and a conservative whole-token
model bound to the frozen route-stage profile; host packing, uploads, pointer
selection, and readback are outside the timed interval.

The research gate is `ratio <= 1.10` and `modeled_save_ms >= 2.0`. A
1.8--2.0 ms result is conditional on an independent paired-lever model that
covers the full residual to 21 t/s. `ratio >= 1.25`, saving below 1.5 ms, or a
nonpositive saving stops K-shard integration. The timing result is a geometry
falsifier, not a full-model performance claim.

Drives shipped `ds4_gpu_routed_moe_one_tensor`. Candidate concatenates two
independently computed 1,024-row mid halves, then runs full-K (8 Q8_K block)
down on each 2,048-row output half and concatenates. `memcmp` vs the full
reference. This remains the bitwise Mechanism I control.

The same cases also run a separate Lane-B numerical precheck. The incumbent
uses expert-id ownership and full-K down on each rank. The candidate evaluates
all six routes on both ranks using packed 1,024-row gate/up halves and the
matching four-block down-K halves, then reuses the existing final rank add.
Distinct shared-expert partials are folded on the corresponding ranks in both
arms. Skew controls include 6/0 and 1/5 ownership. The predeclared isolated
envelope is `max_rel <= 2e-5` and `NMSE <= 1e-10`.

The final isolated case exercises the default-off registry-backed one-token
primitive. It declares and loads exact gate/up row halves plus the matching
down K half, proves the ordinary linear primitive refuses those source
tensors, and compares the new primitive bit-for-bit with an independently
packed direct-map reference. This still adds no production caller or TP
feature negotiation.

This is arithmetic only. It does not authorize the dispatcher, loader,
negotiated feature, a new golden FNV, or `main` promotion.
`DS4_ROCM_Q4K_DECODE_STAGE_MIDQ` stays off because that path assumes eight
Q8_K blocks per expert.

The exact output-row implementation from Step 23/24 remains rejected because
its extra `MOE_MID` ordering boundary consumed the balancing gain. The K-shard
precheck deliberately tests a different, gate-free design under Lane B. A
full-model experiment must map one K-half of every routed expert per rank,
disable the ordinary expert-id remap, negotiate the feature exactly, and pass
the full numerical and quality promotion gate before becoming a default.
