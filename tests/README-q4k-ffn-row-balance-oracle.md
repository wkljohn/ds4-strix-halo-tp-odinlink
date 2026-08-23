# FFN row-balance and gate-free K-shard oracle

See `$DS4_RESEARCH_ROOT/decode-25tps-2026-08-20/CODEX-GATE-7.md`.

```
make test-rocm-q4k-ffn-row-balance-oracle
```

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
