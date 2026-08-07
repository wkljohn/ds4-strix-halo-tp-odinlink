# DSpark >17 t/s: status quo, bottleneck, and paths

Research pass on top of `DSPARK-17TPS-HANDOVER-2026-08-07.md`.
Sources: Codex deep code investigation (session 019fdccb-105c-7031-bb96-b4cdf13ff719),
a Fable literature/outside-view pass, an independent roofline cross-check against
`ROOFLINE-AND-STRATEGY.md` (corrected block), and the raw logs in
`research-results/dspark-resident-2026-08-07/clean-width-sweep/`.

No code was changed while producing this document.

## 1. Status quo

Warm median **14.88 t/s** (14.81 / 15.12 / 14.88 over three identical `/reset` runs,
60 tokens, ctx 128, temp 0, seed 42). Non-DSpark target-only decode is 13.83 t/s.
So DSpark is currently buying only **+7.6%** over plain decode.

Per-cycle decomposition (12 cycles, 5.04 output tokens/cycle):

| stage | ms/cycle | share | tokens produced |
|---|---:|---:|---|
| target anchor eval (1 row) | 70.1 | 21% | 1 |
| DSpark proposal chain | 26.0 | 8% | 5 drafts |
| verifier batch (5 rows) | 233.4 | 71% | up to 5 |
| snapshot / readback | 0.4 | <1% | — |
| partial replay | 0.0 | 0% | — |
| **total** | **~330** | | **5.04** |

To exceed 17 t/s at the same tokens/cycle the budget is **294 ms/cycle** — about
**35 ms/cycle** must come out.

## 2. The bottleneck, quantified

### 2.1 The corrected byte model

From `ROOFLINE-AND-STRATEGY.md`'s **corrected** block (derived from our own Q4_K GGUF
header, components sum exactly to the 153.32 GiB total). Per decoded token:

| component | streamed per token | grows with batch? |
|---|---:|---|
| routed experts (6 of 256) | 3.40 GiB | **yes** — one load per token/expert pair |
| attention + indexer + compressor | 5.49 GiB | no — fixed per forward |
| other dense (shexp, norms, hc) | 1.20 GiB | no — fixed per forward |
| **active per token** | **10.09 GiB** | |

Effective per-node bandwidth: 195.6 GiB/s. Experts are sharded across the two nodes;
everything else is replicated on both.

### 2.2 Efficiency against roofline

| forward | bytes/node | roofline time | measured | efficiency |
|---|---:|---:|---:|---:|
| anchor, 1 row | 5.49 + 1.20 + 1.70 = **8.39 GiB** | 42.9 ms | 70.1 ms | **61%** |
| verifier, 5 rows | 5.49 + 1.20 + 8.50 = **15.19 GiB** | 77.7 ms | 233.4 ms | **33%** |

(The 5-row expert term is 1.70 x 5: the Q4_K path launches all 30 token/expert pairs
and does **not** deduplicate repeated experts across rows — Codex confirmed the sorted
/dedup tile path has a default minimum of 32 tokens, so `use_sorted_pairs` is false at
n=5. `ds4_rocm_moe_launch.cuh:48`, `:1230`, `:1864`.)

**This is the central finding, and it corrects the framing in the Codex report.**
Codex concluded the verifier is "primarily expert-weight-traffic bound". Expert traffic
*is* the largest single term at batch 5 (8.50 of 15.19 GiB/node, 56%) — but traffic
alone predicts a 1.81x cost ratio versus batch 1, i.e. **131 ms**. We measure 233 ms.

So there are two separate problems, and they need separate fixes:

- **(A) The expert path's efficiency collapses with batch width.** Holding the fixed
  weight stream at the batch-1 efficiency (55.9 ms) and attributing the remainder to
  experts:

  | | expert bytes/node | expert time | achieved | % of roofline |
  |---|---:|---:|---:|---:|
  | 1 row (6 pairs) | 1.70 GiB | 14.2 ms | 120 GiB/s | **61%** |
  | 5 rows (30 pairs) | 8.50 GiB | 177.5 ms | 48 GiB/s | **24%** |

  Same kernel, same bytes-per-pair, **2.5x worse efficiency**. This is the deepest
  anomaly in the system and the largest single pool of recoverable time.
- **(B) 5.49 GiB/node/forward of attention weights that should not be there at all.**
  See §2.3 — a structural fix, not a kernel fix.

Fable's independent literature pass computed ~33 GB/s/node for the 5-row verifier
against ~210 GB/s usable, converging on the same conclusion from different arithmetic:
**this rig is dequant/compute-bound, not HBM-bandwidth-bound.** That matters because it
inverts the standard remedy — see §3.3.

### 2.3 The attention head split is missing from the *verifier* path only

**Corrected 2026-08-07 after reading the code more carefully.** An earlier draft of
this section claimed the head split was absent from the DeepSeek ROCm path entirely.
That was too broad. The correct scope:

- **Single-token decode encoder** `metal_graph_encode_decode_layer_phase`
  (`ds4.c:21680`) **has it, unconditionally on for TP=2**:
  `tp_split_attn = g->tp_world == 2`, `tp_heads = DS4_N_HEAD / 2`,
  `tp_head0 = g->tp_rank * tp_heads` (`ds4.c:21727-21730`), used for the q_b rows,
  head RMS norm, RoPE tail and head-range attention. **The 70 ms anchor eval already
  benefits.**
- **Batch encoder** `metal_graph_encode_layer_batch` (`ds4.c:29518`) — which is what
  `metal_graph_verify_suffix_tops_impl` drives — **has no head split at all.** It
  instead offers `tp_row_split_attn` (`ds4.c:27500`), which splits the attention
  *score computation* by row range. That splits math, not weight streaming, which is
  consistent with the row-split experiment having produced bit-identical logits and
  no clear speed win.

So the redundancy is confined to the verifier — conveniently, the 71% of the cycle
that matters. Both ranks stream the full 5.49 GiB of attention/indexer/compressor
weights on every verify batch.

This makes the lever **more** tractable than first framed: the port target is a
sibling function in the same file with a working, unconditional reference ~8,000 lines
above it — not a port from `ds4_metal.m` or from the GLM graph.

Historical note, still true but less relevant: `ds4_rocm.cu:229` declares
`g_tp_attn_head_split` and `:1061` sets it, and nothing reads it. That particular
global is the Metal/GLM mechanism; the DeepSeek ROCm decode path implements its own
split directly from `g->tp_rank` instead.

### 2.3b Original survey of head-split implementations

`ROOFLINE-AND-STRATEGY.md` records that splitting attention heads across the two ranks
moves the ceiling from **23.3 t/s to 34.7 t/s** (bytes/node 8.39 -> 5.64). I verified
its implementation status in the current worktree:

- `ds4_rocm.cu:229` declares `g_tp_attn_head_split`; `ds4_rocm.cu:1061` sets it.
  **Nothing reads it.** The original doc's "written and never read" note still holds.
- The real implementation lives in `ds4_metal.m` (`ds4_gpu_tp_attn_head_range`,
  used at `:30922`, `:31057`, `:31998`, `:32135`) — Metal only.
- There *is* a working backend-agnostic ROCm implementation at `ds4.c:44530`,
  `:44654`, `:45343-45369` (head ownership + `tp_bounce_out` + combine over the
  existing big-gate exchange) — but it sits inside `glm_graph_forward_indexed_tokens`,
  which operates on `ds4_glm_gpu_graph` (**GLM-5.2**, `s->glm_graph`).
- DeepSeek-V4-Flash decode and the DSpark verifier use `s->graph` /
  `metal_graph_encode_layer_batch` (`ds4.c:29518`, reached from
  `metal_graph_verify_suffix_tops_impl` at `ds4.c:34541`). That encoder has **no head
  split at all**.
- Even in the GLM path it is gated by `n_tokens >= glm_tp_head_split_min()`, which
  defaults to **64** (`ds4.c:35163`, env `DS4_GLM_TP_HEAD_SPLIT_MIN`). A 5-row verifier
  and a 1-row anchor are both far below that floor.

Net effect: **both ranks redundantly stream the entire 5.49 GiB of attention / indexer /
compressor weights on every anchor eval and every verifier batch.** That is 65% of the
anchor's bytes and 36% of the verifier's.

Projected at the *currently measured* efficiencies (61% / 33%):

Halving the replicated attention stream cuts **22.9 ms off the fixed cost of every
forward** (55.9 -> 33.0 ms):

| | today | with head split |
|---|---:|---:|
| anchor | 70.1 ms | ~47 ms |
| verifier (5 rows) | 233.4 ms | ~210 ms |
| propose | 26.0 ms | 26.0 ms |
| **cycle** | **330 ms** | **~284 ms** |
| **throughput** | **15.1 t/s** | **~17.8 t/s** |

It clears the gate, but only just — and it is a port from an existing, working,
same-backend reference rather than new research.

### 2.4 OdinLink is not the bottleneck

Codex summed the 43 per-layer gate-callback entries in `coordinator-width5-profile.log`:
**37.4 ms total, 0.87 ms/layer**, out of a 233 ms verifier. ~196 ms is outside the TP
callback entirely, and the 37 ms itself includes peer arrival skew, not just wire time.
**No OdinLink change is justified by present evidence.** Revisit only if the
instrumentation in §4 shows a large removable provider/wire component.

## 3. Correcting the record on two rejected experiments

### 3.1 `DS4_DSPARK_CHAIN_CYCLES=1` was rejected against a stale baseline

Anchorless chaining is the textbook elimination of the 70 ms anchor, and Codex confirms
the state transition is already correct: `metal_graph_dspark_capture_prepare_chain`
(`ds4.c:26422`) converts the verifier's captured window into exactly what the next
proposal expects, the verifier already captures all three DSpark target-layer hidden
rows (`ds4.c:26324`, `:26342`), and `s->logits` is already refilled from the accepted
verifier row (`ds4.c:61882`, `:61943`). It measured 6.56 t/s and was rejected.

**That measurement is not valid against today's build.** File mtimes:

```
16:19  coordinator-width5-chain-window.log     <- the 6.56 t/s chain run
16:39  coordinator-width5-q8-dp4a.log          <- DP4A lands
16:50+ coordinator-width5-dp4a-prefixes.log    <- four-prefix zero-replay commit lands
```

The chain run predates both DP4A and the four-prefix commit. Its own stats confirm it:
`replay=133.974` (the old replay path was still active; today it is 0.000) and
`verify_upload=29.099` (today 1.7). Its contemporaneous baseline was 12.36 t/s with
297 ms/verifier-batch — not the 233 ms we have now.

What the chain run does show is a real, unexplained regression: 3975.4 ms of verifier
layer time over 8 batches = **497 ms/batch versus its own contemporary 297 ms**, a
1.67x inflation, while anchor time correctly collapsed to 134 ms. Codex found no code
mechanism for the handover's "lost target-weight warm-up" explanation and I agree it is
a hypothesis, not a finding.

Fable proposes a third and better-grounded hypothesis for the 6.56 t/s: **acceptance
collapse, not verifier inflation.** 6.56 t/s admits two decompositions — cycle time
inflates ~3x at constant tokens/cycle, or cycle time stays normal and tokens/cycle falls
to ~1.7-2.5. The latter is the canonical EAGLE-family failure mode: a drafter fed hidden
states it was not calibrated on. Here the mechanism is concrete — the verifier's capture
states come from the *batch* path (DP4A dynamic-quantized, ~0.38% RMS error) plus a crop,
while the DSpark Markov/chain drafter's statistics were built against *single-token* path
captures. This drafter is already known to be brittle to support-state content: support
top-k 4 -> 2 alone dropped acceptance to 81% and throughput to 7.36 t/s.

**Action: re-run `DS4_DSPARK_CHAIN_CYCLES=1` on the current build before anything else.**
One env var, no rebuild. The `DS4_DSPARK_STATS=1` line settles all three hypotheses at
once: read `accept_rate` / `miss_first` / `avg_accept` against `verify_ms` per batch.
- acceptance collapsed -> fix capture fidelity (e.g. recompute the seed row's state via
  the single-token kernel path — one row of dense work, no expert streaming);
- verify inflated at constant acceptance -> that inflation is the bug to debug;
- neither -> the stale-baseline explanation holds and chaining is simply viable now.

### 3.2 Draft-width sweeps are a dead end, analytically

Fitting the affine cost model to the measured data (fixed per-forward term F = 26.9 ms,
per-6-expert-load term E = 43.1 ms, per-token accept probability p = 0.93 which
reproduces the observed avg_accept 4.0 at width 5), and using the independent-routing
union `U(w) = 256(1-(1-6/256)^w)`:

| width | verify ms | propose ms | anchor ms | tokens/cycle | t/s |
|---:|---:|---:|---:|---:|---:|
| 3 | 193 | 18 | 70 | 3.61 | 12.8 |
| 4 | 213 | 22 | 70 | 4.35 | 14.2 |
| **5** | **233** | **26** | **70** | **5.04** | **15.3** |
| 6 | 271 | 30 | 70 | 5.69 | 15.4 |
| 7 | 308 | 34 | 70 | 6.30 | 15.3 |
| 8 | 345 | 38 | 70 | 6.88 | 15.2 |

The curve is **flat** — the expert-union cost of a wider batch exactly cancels the
acceptance gain. Width 5 is already optimal and no width sweep can reach 17 t/s.
Fable derived the same table independently (15.06 / 15.20 / 15.23 / 15.16 / 15.04 for
w=4..8) and Codex reached it a third way (~99% acceptance would be needed at the current
cycle time). Three independent derivations agree. **Do not spend runs on width.**

### 3.3 Two corrections that shrink the headline estimates

**Anchor elimination is worth ~29 ms/cycle, not 70.** Both Codex's rank-1 estimate
(19.2 t/s) and my own first pass were too optimistic. Removing the standalone anchor
does not remove its *work* — it removes only the duplicated **fixed** per-forward cost
(~23-29 ms). The anchor token's marginal expert-row cost migrates into the verify batch.
Verified numerically against the byte model:

| restructure | cycle | tokens | t/s |
|---|---:|---:|---:|
| today | 330 ms | 5.04 | 15.1 |
| chaining only | 259 ms | 4.36 | **16.8** |
| head split only | 284 ms | 5.04 | **17.8** |
| **both** | **236 ms** | **4.36** | **18.4** |

Fable reached ~16.7 t/s for the restructure alone by an independent affine fit
(`verify(w) = 29.25 + 40.75w`, versus my byte-model 271 ms at w=6 against its 274 ms —
close agreement). **Chaining alone does not clear the gate.** Plan on stacking levers.

**Tree/multi-branch drafting is *anti*-attractive here, inverting the usual intuition.**
The "extra tree nodes ride along on an already-paid expert union" argument holds only in
the HBM-bandwidth-bound regime the MoE-speculation papers assume. On a compute-bound rig
each extra row pays full per-row expert GEMM cost regardless of union overlap. The demand
side is missing too: at per-position acceptance α ≈ 0.955 with `miss_first=0`, a sibling
branch yields ~0.04-0.15 tokens for a full extra row — strongly negative. Tellingly the
MoE-specific literature *prunes* speculation trees rather than growing them (EcoSpec,
EVICT, AcceptMoE). Same reasoning kills expert-aware/routing-biased draft *selection*:
it needs candidate slack a chain drafter at α=0.955 does not have, and EcoSpec's own
DeepSeek result is only 1.15x on favourable hardware. **Skip trees and routing-biased
drafting.**

## 4. The instrumentation gap

`verify_layer` is 99.9% of `verify` with no internal breakdown, so §2.2's "102 ms of
non-traffic overhead" cannot currently be attributed. The previous attempt at a layer-
stage profiler produced a bogus 7,120 ms `hc_post` on layer 0
(`coordinator-width5-stage-profile.log:306`) because it ends command batches and
device-synchronizes at every boundary, which races the stream-written TP arrival flag
(`docs/DECODE-PROFILER-STALL.md:49`, `:101`).

Codex's proposed safe design, which I endorse:

1. Record HIP events on the **existing** compute stream at four boundaries (layer start,
   attention end, routed-MoE end, layer end) plus optional markers around labelled
   dense-Q8 calls. **Never** end command batches; **never** call `hipEventSynchronize`.
2. Harvest with `hipEventQuery` after the normal verifier readback, using a small ring
   of event sets and dropping a sample rather than blocking. The repo already has this
   non-blocking pattern at `rocm/ds4_rocm_runtime.cuh:6714`, `:6781`.
3. Keep TP timing in the service thread, which already measures detection / callback /
   release without GPU barriers (`ds4_rocm.cu:613`, `:649`, `:681`); extend its
   per-kind aggregation to `DS4_TP_BATCH`.
4. Run it on **both** ranks. Asymmetric synchronous profiling is exactly what turns
   normal peer skew into a fake `hc_post` stall.

Separately, the repo already contains an exact unique-expert-union counter
(`ds4.c:18310`, `:18321`, `:18390`) used for streaming prefill. Pointing it at the
verifier's router output would settle whether the 5-row union is 30 (independent) or
closer to 18-20 (correlated routing), which decides whether cross-row expert dedup is
worth anything.

## 5. Ranked paths to >17 t/s

| # | path | est. cycle effect | est. t/s | risk | notes |
|---:|---|---:|---:|---|---|
| 0 | **Diagnostic: chain-cycles stats run** | — | — | none | One env var, ~10 min, no rebuild. Settles the three competing explanations (§3.1) and gates #2. Do this first. |
| 1 | **TP attention head split in the DeepSeek ROCm batch encoder** | -23 ms/forward | **~17.8** | med | Biggest structural lever. Ceiling 23.3 -> 34.7 t/s. Working same-backend reference in the GLM path (`ds4.c:44530`+). Helps anchor *and* verifier. Also closes a latent correctness gap vs Metal. |
| 2 | **Anchorless chaining** (bonus token from the verifier's last accepted row) | -71 ms, -0.68 tok | **~16.8** | med | Universal in EAGLE-2/3, Medusa, DeepSeek MTP, vLLM v1 — no SOTA system runs a standalone anchor pass. Does **not** clear the gate alone. Gated on #0. |
| **1+2** | **both together** | | **~18.4** | | **The recommended target.** Real margin over the gate. |
| 3 | **Attack the expert-path efficiency collapse** (61% -> 24%, §2.2A) | -50 to -120 ms | 17.9-22+ | med/high | Largest recoverable pool in the system. Needs §4 instrumentation first. |
| 4 | Activation-weighted expert placement (EPLB-style rebalance of 118/138) | -5 to -15 ms | 15.6-16.0 | med | Each layer completes at max(rank0, rank1); 118/138 balances expert *count*, not activated work under Zipf routing. Measurement machinery already exists (`g_expert_profile`, `ds4.c:1226-1399`). |
| 5 | Overlap the TP batch gate with shared-expert work | -10 to -25 ms | 15.6-16.4 | med | Gate is issued before the shared expert runs (`ds4.c:29268`, `:29347`, `:29409`). Bundle filler. |
| 6 | Cross-row expert dedup / sorted-pair tiles at n<=8 | -15 to -40 ms | 15.9-17.2 | med/high | Value depends entirely on the measured union (§4). **Measure before building** — if compute-bound, worth ~0. |
| 7 | Shape-specialised DP4A dispatch | -3 to -10 ms | 15.3-15.6 | low | Bundle filler. |
| 8 | Reuse quantized activations across compatible projections | -1 to -5 ms | 15.2-15.4 | low | Cleanup; below threshold alone. `ds4.c:29097`, `:29107`. |
| — | Draft-width sweep | ~0 | ~15.2 | — | **Rejected analytically** by three independent derivations, §3.2. |
| — | Tree / multi-branch drafting | negative | — | — | **Rejected**, §3.3. Anti-attractive on a compute-bound rig at α=0.955. |
| — | Expert-aware / routing-biased draft selection | ~0 | — | — | **Rejected**, §3.3. Needs candidate slack a chain drafter lacks. |
| — | Literal 6-row anchor folding | — | — | — | **Rejected**: causally circular. Without the verifier's captured hidden state the later drafts don't exist yet; with it, you already have #2. |
| — | Draft/verify overlap scheduling, EP prefetch | ~0 | — | — | **Rejected**: 8% ceiling, genuinely serial dependency, no slower weight tier, no all-to-all. |
| — | OdinLink changes | -5 to -15 ms best case | 15.4-15.9 | — | **Not justified** (§2.4). |

Recommended order: **#0 (free diagnostic) -> §4 instrumentation -> #1 -> #2 -> #3**,
bundling #7/#8 opportunistically. Neither #1 nor #2 alone leaves comfortable margin;
plan on landing **both**.

## 6. Worktree hygiene (confirmed)

- `cuda_launch_q8_small_batch_dp4a()` is at **16 waves / 512 threads**
  (`rocm/ds4_rocm_matmul.cuh:167`, `:171`, `:174`) — the handover's explicitly rejected
  unstable setting. Must be reverted to `rows_per_block=8u` / 256 threads before any
  measurement.
- `DS4_DSPARK_CHAIN_CYCLES` is opt-in and not active by default (`ds4.c:61679`).
- The Q4_K sorted threshold is still the production default of 32
  (`ds4_rocm_moe_launch.cuh:48`), so the poor low-row sorted experiments are not
  silently live.
- Four-prefix commit is present and removes replay, but the handover's support-frontier
  equivalence concern (11 cycles `1:1,3:1,4:1,5:8` -> 12 cycles `1:2,3:1,4:2,5:7`)
  remains unvalidated. Output text is identical because verification is exact, but this
  needs closing before `dspark-good`.
- No other known-rejected experiment was found compiled on unconditionally.

## 7. Open questions

1. Does the 1.67x anchorless-verifier inflation reproduce on the current build?
2. What is the true 5-row expert union — 30, or closer to 20?
3. What is the composition of the ~102 ms/cycle non-traffic verifier overhead?
4. Is the GLM head-split implementation portable to `metal_graph_encode_layer_batch`,
   or does the DeepSeek MLA / indexer / compressor structure require a different
   ownership scheme?
