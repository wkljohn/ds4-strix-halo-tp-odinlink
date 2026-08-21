# Decode 21 t/s campaign report — no-DSpark TP=2

Date: 2026-08-21
Branch: `research/q4k-decode-25tps-20260820`
Base production commit: `main` / `origin/main` at `2e7210a`
Headline mode: ordinary greedy TP=2, DSpark off, cache-free GGUF Q4_K/Q2_K,
mandatory RDMA, both OdinLink and RoCE v2 supported.

## Outcome

The implementation campaign is stopped at the validated production baseline:

| Model / transport | Workload | Prefill | Decode | Fingerprint |
|---|---:|---:|---:|---|
| Q4_K RoCE v2 | `ds4-bench-tp` 2048+300 | 258.64 t/s | 19.22 t/s | `5f8a983422299d76` |
| Q4_K OdinLink | 2048+300, 4K chunk | 218.24 t/s | 18.88 t/s | `5f8a983422299d76` |
| Q2_K RoCE v2 | 2048+300 | 203.00 t/s | 19.14 t/s | `f9cb3a8a17e95c71` |
| Q4_K RoCE v2 | 32K long-context | 209.70 t/s | 16.29 t/s | `20e75c02d148f4f9` |

The research branch contains test/oracle and default-off production probes, but
`main` remains unpromoted at `2e7210a`.

## Why 21 t/s is not currently approved

Moving Q4_K RoCE from 19.22 t/s to 21.00 t/s requires token time to fall from
52.03 ms to 47.62 ms, a 4.41 ms/token reduction.

The closed ledger says:

- The largest known pool is FFN gate-wait/expert-count skew. Recovering it
  requires the row-balance fold that `CODEX-GATE-8.md` denied: the existing FFN
  callback cannot both exchange `mid` before routed-down and exchange final
  halves after routed-down.
- The isolated one-token Q4_K oracle was already tried. `STATUS.md` records no
  `>=10%` win versus the shipped full one-token MoE.
- Mechanism E staged-MIDQ was bitwise but slower (`0.206 -> 0.213 ms`, -3.4%).
- `DS4_ROCM_TP_SLOT_BALANCE=1` live TP timed out because of stolen-expert decode
  copies; it remains default-off.
- No independent device-resident routing / `h_counts` idea currently has an
  audited `>=2 ms/token` ceiling. That is an unspecified K and remains denied by
  `CODEX-GATE-8.md`.

Therefore the next safe work is measurement/documentation, not another source
change.

## Branch inventory after `main`

| Commit | Type | Status |
|---|---|---|
| `3fd5aa6` | isolated gfx1151 Q4_K one-token decode oracle | test-only; no dispatcher |
| `d5583f7` | TP `collective_required` byte audit | test/audit |
| `b81bba3` | compressor output-row shard oracle | test-only |
| `c91735f` | staged-MIDQ Q4_K down + oracle | production code default-off; not enabled |
| `7787694` | Q4_K FFN output-row balance oracle | test-only arithmetic proof |
| `657fe1e` | TP slot-balance decode plus oracle | production code default-off; live timed out when enabled |

Before any future merge, the default-off production commits must be proven inert
on Q4_K and Q2_K, over both RoCE v2 and OdinLink.

## Required baseline matrix before further implementation

Run the current research branch with all new experimental flags default-off:

| Run | Acceptance |
|---|---|
| Q4_K RoCE v2 2048+300 | fingerprint `5f8a983422299d76`, decode near 19.22 t/s |
| Q4_K OdinLink 2048+300 | fingerprint `5f8a983422299d76`, no regression vs 18.88 t/s |
| Q2_K RoCE v2 2048+300 | fingerprint `f9cb3a8a17e95c71`, no regression vs 19.14 t/s |
| Q2_K OdinLink 2048+300 | fingerprint `f9cb3a8a17e95c71`, establish baseline |
| Optional Q4_K 32K RoCE/OdinLink | preserve long-context visibility |

A fingerprint mismatch, short generation, transport fallback, or throughput
regression should trigger a bisect of `3fd5aa6..657fe1e` before any new work.

## Reopen conditions

Code work can resume only if at least one of these becomes true:

1. A new callback/topology design proves row-balanced routed-MoE without adding
   a second latency gate and with exact Q4_K/Q2_K fingerprints.
2. A new byte-division mechanism is proved with a roofline above the target and
   no persistent expanded-weight cache.
3. A separate DSpark/speculative campaign is explicitly opened; its results must
   not be mixed with ordinary greedy decode numbers.
4. A framework-derived kernel idea is reduced to a standalone gfx1151 oracle that
   beats the current DS4 kernel at exact shapes before touching production paths.

## Prior-art scope closure

`SURVEY.md` now includes the original coverage plus an addendum for:

- Q2_K/IQ2 decode prior art and why it is a regression-gate first;
- Composable Kernel / CK;
- hipBLASLt;
- Qwen3.8-27B optimization work;
- ds4-on-spark / DGX Spark concrete differences.

Transfer ranking for no-DSpark Q4_K/Q2_K TP=2 decode:

1. llama.cpp RDNA3.5 Q4_K/Q5_K/Q6_K decode ideas: useful only as standalone
   oracles; current MMVDQ evidence is too small on large MoE.
2. DS4 row-balance byte division: largest possible pool, but blocked by callback
   topology until redesigned.
3. Q2_K/IQ2: preserve and remeasure; not a likely Q4_K decode breakthrough.
4. ds4-on-spark: useful for serving/benchmark/DSpark lane, not direct greedy
   GGUF kernel transfer.
5. CK/hipBLASLt/AITER/vLLM direct ports: low transfer without format conversion
   or a persistent cache.
