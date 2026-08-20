# FFN output-row balance oracle (Mechanism I)

See `research-results/decode-25tps-2026-08-20/CODEX-GATE-7.md`.

```
make test-rocm-q4k-ffn-row-balance-oracle
```

Drives shipped `ds4_gpu_routed_moe_one_tensor`. Candidate concatenates two
independently computed 1,024-row mid halves, then runs full-K (8 Q8_K block)
down on each 2,048-row output half and concatenates. `memcmp` vs the full
reference. Cases: 6/6, unowned/zero-weight, fused addend, 5/7 expert table.

This is arithmetic only. It does not authorize dispatcher, loader, extra
`MOE_MID`, or `main`. `DS4_ROCM_Q4K_DECODE_STAGE_MIDQ` stays off.

Predeclared landing if a later gate ever folds mid-exchange into the existing
FFN callback without a third `hipLaunchHostFunc` (Step 23/24): about
**20.1–20.4 t/s**, not 21. Realized skew recovery after removing the MOE_MID
tax is ~1.7–2.2 ms/token from 19.22; 21.0 still needs another ~2 ms pool.
Kill: bitwise failure, or K-sharded down (4-block mid).
