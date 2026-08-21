# Non-DSpark release medians

Date: 2026-08-21

All current rows use `ds4-bench-tp`, a 2,048-token fixed frontier followed by
300 generated tokens, balanced 128/128 experts, cache-free weights, mandatory
RDMA, semantic smoke/retrieval checks, and the exact model-specific token
fingerprint. DSpark was deliberately not rebenchmarked.

| README row | Included runs (prefill/decode t/s) | Median | Fingerprint |
| --- | --- | --- | --- |
| Q2_K, RoCE v2 | 202.86/19.49; 202.83/19.45; 201.68/19.51 | **202.83/19.49** | `f9cb3a8a17e95c71` |
| Q4_K, OdinLink | 220.98/19.19; 217.01/19.27; 215.04/19.13 | **217.01/19.19** | `5f8a983422299d76` |
| Q4_K, RoCE v2 | 258.06/19.54; 257.28/19.60; 260.10/19.55 | **258.06/19.55** | `5f8a983422299d76` |

The final default-on pre-main run measured 256.30/19.52 for Q4_K and
201.68/19.51 for Q2_K over RoCE v2 and ended with
`PRE_MAIN_TP_SMOKE_PASSED`. Both ranks negotiated the exact cooperative HC and
long-context radix-tree feature bits. The final default-on OdinLink run was
215.04/19.13 and passed the same Q4_K fingerprint.

The Q4_K RoCE and Q2_K medians reuse earlier opt-in runs because the promoted
production default selects the byte-identical kernels; the final default-on
pre-main cases independently prove the default dispatch. The OdinLink median
likewise combines two opt-in identical-path runs with the final default-on
run. Raw logs and CSV files remain in this research directory's ignored
`runs/` subdirectory.
