# Staged-MIDQ Q4_K one-token oracle

See also `$DS4_RESEARCH_ROOT/decode-25tps-2026-08-20/CODEX-GATE-5.md`.

```
make test-rocm-q4k-staged-midq-oracle
```

Two processes drive production-shape `ds4_gpu_routed_moe_one_tensor` with
`DS4_ROCM_Q4K_DECODE_STAGE_MIDQ=0` then `=1` (dispatcher caches the switch).
Dumps cover unowned-slot skip, a negative selected expert, and fused addend.
`cmp` requires bitwise equality. Timings are the full one-token MoE; a ≥10%
win versus the shipped kernel is the enable gate, not a test failure.
Default remains `STAGE_MIDQ=0`. No `main` promotion.
