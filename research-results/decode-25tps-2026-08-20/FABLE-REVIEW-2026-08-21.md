# Fable review — 2026-08-21

The text below is retained verbatim. It is an advisory ceiling challenge, not
a stop authority for the active 21 t/s campaign.

> context: caller prompt only
>
> I read STATUS/SURVEY/FABLE-NOTE/GROK-NOTE/CODEX-GATE-2..8/ORACLE, BYTE-DIVISION, the full step0 README (Steps 0–25) and FABLE-REVIEW-6, and verified the load-bearing code claims: the FFN fold really does produce the final partial at `ds4.c:24857` (`tp_fold_ffn` → `ds4_gpu_routed_moe_one_tensor` → `ds4_gpu_tp_gate_encode` at :24897, so a mid exchange cannot precede down without a second ordering point), and `DS4_ROCM_TP_SLOT_BALANCE` is a one-shot direct-pointer override (`ds4_rocm_moe_launch.cuh:103–105, 1253–1261, 3879–3913`) that reads stolen-expert weights cold from the map during decode — consistent with the reported live timeout.
>
> **Verdict: proceed-with-changes** — uphold the STOP for ≥21 t/s under these constraints; the changes are to close one phantom pool the ledger still leaves open on paper, and to record the honest ceiling of the one remaining mechanism.
>
> **The arithmetic, which the STOP survives.** 19.22 → 21.0 needs 52.03 − 47.62 = **4.41 ms/token**. Every pool with a mechanism:
>
> 1. **FFN expert skew (~5.9 ms gross):** the only large pool. Step 23 measured the exact row-shard end-to-end at 19.29 (fingerprint-pass), and Step 24 attributed the loss almost entirely to the extra MOE_MID host callback (40–51 µs/layer), not the half-row arithmetic (~neutral) or the exchange (14–24 µs). Codex GATE-8's structural claim is correct as verified above: one callback cannot both order mid-before-down and finals-after-down. Audited net landing without the extra callback: **20.1–20.4 t/s**. This mechanism cannot reach 21 even if it works.
> 2. **Compressor/indexer split — the pool GATE-8's "no ≥2 ms remains" is most vulnerable on, and it still holds.** BYTE-DIVISION's 3.07 ms (half of 1,200.62 MiB) comes from the Step-0 audit taken at the **17.29 t/s pre-temporal era**. STATUS quotes it as current — the same era-mixing BYTE-DIVISION accuses the 8.39 GiB roofline of. Step 24's live measurements cap the amortized compressor boundary at ~21×(266–299 µs)/4 + 20×1.1 ms/128 ≈ **1.6–1.75 ms/token**; halving it saves ~0.8 ms gross, while the exchange it needs sits in the same layer segment (consumed before the ATTN gate, per GATE-4/STATUS) and a new per-layer gate costs the measured ~1.9 ms/token. **Net ≤ 0.** Fix: write this closure into STATUS/BYTE-DIVISION explicitly so the 33.5 t/s "live ceiling" stops being cited as a reachable pool.
> 3. **Attention static skew:** ≤ 33.7 µs × 43 ≈ 1.45 ms theoretical, never attributed (rank1 lateness cause is genuinely unmeasured — the one open fact in the ledger). Even fully harvested it doesn't bridge 20.4 → 21 plus margin.
> 4. **Kernel compute:** closed by measurement — oracle found no ≥10% vs shipped full MoE, MMVDQ lost, Mechanism E is −3.4% end-to-end on the harness.
>
> Max honest stack ≈ 20.3–20.7 < 21, before counting the CQ-pressure risk of any third per-layer gate on OdinLink (depth 63; Step 23's three-phase schedule was only ever full-run-validated on RoCE — the OdinLink 2048+300 three-gate run does not exist).
>
> **(a) Immediately benchmarkable, already implemented, default-off: effectively empty.** Do not spend two-node loads on: `STAGE_MIDQ=1` (measured −3.4%), `DS4_ROCM_TP_SLOT_BALANCE=1` (live timeout; stealing experts converts skew into extra cold bytes on a bandwidth-bound UMA — structurally wrong direction), `Q_B_QNORM_ROPE_FUSE` (fingerprint `45d1f85987f49e86`), the Step 23 atomic row-shard feature (measured 19.29, below control). None plausibly reaches 21; benchmarking them anyway would only burn the fixed-workload budget.
>
> **(b) The one smallest new implementation worth trying — with its ceiling stated up front:** a gate-free MOE_MID release for the already-exact Step 23/24 row-shard path — transport thread writes a completion word into device-visible memory, a minimal spin-wait before the down kernels, replacing the second `hipLaunchHostFunc`. It reuses proven-bitwise arithmetic and attacks exactly the measured 40–51 µs/layer tax. But: (i) its audited ceiling is **20.1–20.4, not 21** — it is only worth doing if the target relaxes; (ii) gfx1151 mapped signal words previously showed rare lost arrivals (Step 1), so it needs the 25,800-gate sustained ordering probe first; (iii) GATE-8 currently lists I-fold as STAY_CLOSED, so it requires a fresh Codex gate with the ≤20.4 ceiling written into the ask, one signature gate at a time, then full 2048+300 on **both** RoCE v2 and OdinLink with `5f8a983422299d76` and a Q2 run at `f9cb3a8a17e95c71` before any commit.
>
> **(c) Dead ends — keep closed, no re-derivation:** slot-balance/dynamic expert stealing; extra MOE_MID host callback; STAGE_MIDQ; compressor/indexer TP split at 2K (net-negative post-temporal, item 2); K-shard down (reassociates the quarter-wave tree); HIP graphs (<2 µs); temporal side-stream/all overlap (same-token consumers, UMA); Q_B prefetch/fusion; transport opcode/poller/MTU/QuickReduce (post ≤0.2 µs, wire ~1.5 ms bandwidth floor); more waves; MMVDQ; AITER/vLLM/Spark ports; grouped-head attention at 2K.
>
> **Decision:** the STOP at 19.22 for the ≥21 target is correct on the evidence in this tree; the campaign's own Step 24 numbers close the last paper pool (compressor split) that GATE-8 dismissed without citing them. The only defensible continuations are (1) a documentation commit closing that pool with the era correction, (2) the read-only attention-skew attribution and a post-temporal re-run of the byte audit — the two facts never measured — and (3) the gate-free MOE_MID primitive **only** if the target is renegotiated to ~20.3, under one-at-a-time gates and dual-transport full runs as above. `main` stays at `2e7210a`; nothing here licenses promotion.

Subsequent single-node Spark inspection found additional ordinary-decode fusion
commits that were not part of this prompt's evidence set. They are analyzed in
`SPARK-SINGLE-NODE-TRANSFER.md` and keep the 21 t/s campaign open.
