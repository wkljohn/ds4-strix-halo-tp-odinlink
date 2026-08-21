# Isolated Q4_K one-token oracle

See also `research-results/decode-25tps-2026-08-20/ORACLE.md`.

```
make test-rocm-q4k-one-token-oracle
```

Drives shipped `ds4_gpu_routed_moe_one_tensor` (bitwise across cold model copies)
plus one-wave test-local Q4_K×Q8_K packed down (documented ULP vs CPU) and
MMVDQ F32-dequant control (allowed to lose). No production dispatcher.

Kill: ≥10% isolated vs shipped one-token MoE before a later Codex-gated
dispatcher commit. A 10% MoE win is not 25 t/s.
