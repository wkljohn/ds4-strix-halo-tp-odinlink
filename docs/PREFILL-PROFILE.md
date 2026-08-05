# Where prefill time actually goes (measured, not estimated)

`DS4_ROCM_LAYER_STAGE_PROFILE=1` **on both ranks**, 3306-token prompt, TP=2 over
RDMA, row split disabled.

| part | stage | share |
|---|---|---|
| ffn | **routed_moe** | **46.6%** |
| ffn | **hc_post** | **35.2%** |
| attn | output_proj | 13.1% |
| attn | q_path | 1.7% |
| attn | **attention (SDPA)** | **1.2%** |
| ffn | hc_pre / shared_gate_up / shared_down | <1% each |
| attn | kv_path / norm / inv_rope / compressor | <1% each |

## The profiler must be enabled on BOTH ranks

First attempt enabled it on the coordinator only. Result: `ffn hc_post` at
**97.7%** and a total of **483 s of "profiled time" for a run that took about two
minutes** - i.e. the numbers were not additive GPU time at all.

Cause: the stage boundary calls `glm_graph_prefill_stage_sync_boundary()`, which
syncs the device. Syncing one rank and not the other breaks TP lockstep, so the
instrumented rank falls behind and waits at the FFN gate - and `hc_post` is
exactly where that gate completes. The profiler was measuring its own
interference.

**Rule: any stage profiler in a lockstep TP system must be symmetric, and the
total must be sanity-checked against wall clock.**

## Consequences for the optimisation plan

**The SDPA attention kernel is 1.2% of prefill.** Proposed work on it - hoisting
q into registers, templating `SCORE_CAP` to raise occupancy, widening the KV tile
- is bit-identical and well-reasoned, but doubling that kernel's speed buys
**+0.6% of prefill**. Not worth doing first, possibly not worth doing at all.

The real targets, in order:

1. **routed_moe, 46.6%.** Under TP each rank should evaluate only its owned
   experts. Patch 12 shards by RANGE for addressing but still EVALUATES all 6
   selected experts per token and discards ~3 by zero weight. At prefill that is
   2186 tokens x 6 experts of which ~half is thrown away. An expert skip was
   estimated at only +3-8% for DECODE (where MoE is a smaller share and L2 reuse
   hides the duplicate reads) - but prefill is compute-bound with a large batch,
   so the same change should be worth substantially more here.
2. **hc_post, 35.2%.** Hyper-connection post. Unexamined. Note this is also where
   the FFN gate completes, so part of this may still be peer-wait even with
   symmetric profiling - separate the two before optimising.
3. **output_proj, 13.1%.**

## Also measured and refuted

`--prefill-chunk` sweep on the 3306-token prompt: **28.96 (default 4096), 28.82
(2048), 27.41 (1024) t/s.** No win. This also refutes the hypothesis that chunk
4096 falls into a hipBLAS path needing ~5.4 GB of scratch - if it did, chunk 2048
would have been dramatically faster.


## Patch 20: the workaround was disabling two things, not one

`DS4_TP_PREFILL_SPLIT_MIN` gated BOTH `tp_row_split_attn` (ds4.c:26778) and
`tp_row_split_ffn` (ds4.c:28822). Only attention needs the missing range
kernels; the FFN split row-splits the replicated shared expert and every
`ds4_gpu_*` it touches is already implemented on ROCm. So the workaround for
attention was silently costing the FFN split too.

Patch 20 gives them independent knobs:
- `DS4_TP_PREFILL_SPLIT_MIN_ATTN` (must stay high until the range kernels exist)
- `DS4_TP_PREFILL_SPLIT_MIN_FFN` (can be 32)
- `DS4_TP_PREFILL_SPLIT_MIN` still works as the fallback for both.

Measured, 3306-token prompt, 3 samples each:

| config | prefill samples | mean |
|---|---|---|
| FFN split ON | 29.56, 29.56, 29.50 | **29.54 t/s** |
| both splits off | 28.96, 29.18, 28.82 | 28.99 t/s |

**+1.9%**, reproducible (the three ON samples span 0.06 t/s) and the
distributions do not overlap. Output stays coherent. Predicted was ~+7%; the
prediction was a FLOP-share estimate and the profile shows why it was optimistic
- the shared expert is `shared_gate_up` + `shared_down` = well under 1% of
prefill by measurement, not the ~13% of a layer the FLOP count suggested.

Useful side result: **prefill t/s on a long prompt is highly reproducible**
(spread <1%), unlike the 13-token prompt where it spanned 2.5x. Long-prompt
prefill is now a usable measurement instrument.

## Re-profile post-WMMA-fix (2026-08-05): output_proj is the new #1, and hc_post's boundary is unusable for measuring the TP gate

Attempted to re-profile with `DS4_ROCM_LAYER_STAGE_PROFILE=1` on both ranks
again, same 1702-token prompt, after this session's Q4_K down-projection WMMA
fix (`8b71a30`). Symmetric on both ranks as the rule above requires - but the
run still blew through a 300s timeout at layer 4/61.

**Root cause: the `ffn hc_post` stage boundary sits immediately after the
cross-node FFN all-reduce gate** (`ds4_gpu_tp_big_gate_encode`, `ds4.c:29129`,
followed by the canonical-order add at `ds4.c:29138`) - so profiling that
boundary forces a device sync right where this rank waits on its peer over
RDMA. Every layer showed one `hc_post` reading in the 20,000-60,000 ms range
(escalating: 20928, 22302, 59667, 25629 ms across 4 layers) while every other
stage in the same layers looked completely normal (tens to ~150 ms). This is
the exact caveat the original measurement above already flagged ("this may
still be peer-wait... separate the two before optimising") - just far more
severe post-fix, likely because forcing a sync exactly at this boundary
breaks whatever async/overlap the gate wait normally gets, similar in kind to
[[hip-stream-memory-ops-null-stream]]'s null-stream ordering trap. **A 1702 x
DS4_N_EMBD row exchange is tens of MB - it should take single-digit ms over
OdinLink RDMA, not tens of seconds, so this is a stall/back-off bug in the
gate-wait mechanism under profiling, not real transfer time.**

Confirmed this is profiler-induced, not a real regression: re-running WITHOUT
the profiler (plain `--rocm --tensor-parallel`, same env) reproduced
**71.74, 72.21 t/s** - consistent with the existing 74.80-75.65 t/s
golden number (small ~4% gap, within plausible run-to-run variance across
separate sessions, not confirmed as a new regression from the row-split
commit). One early sample (49-50 t/s) turned out to be the sanity script
itself missing `DS4_TP_BIG_DIRECT=1` - restoring it recovered the number.
**Lesson: always diff a new repro script against the last known-good env
line by line before trusting a surprising result.**

**The 4 clean (non-hc_post) layers are still useful signal.** Summed
per-layer, non-gate stage time ≈ 333.5 ms, split:

| stage | share of clean per-layer time |
|---|---|
| **output_proj** | **~42%** |
| **routed_moe** | **~27%** |
| attention (SDPA) | ~12% |
| q_path | ~6% |
| hc_pre (attn+ffn) | ~3.5% |
| indexer_setup | ~2.8% |
| shared_gate_up + shared_down | ~2.6% |
| compressor (prefill+commit+refresh) | ~2.5% |
| router, inv_rope, norm, kv_path, hc_post(attn) | <1% each |

**This flips the priority order.** Before the WMMA fix, routed_moe was 46.6%
and dominant. After it, **output_proj (the attention output projection,
`metal_graph_attention_output_dense_quant_tp` and friends) is now the single
largest measured prefill stage**, not routed_moe and not SDPA itself (still
tiny, consistent with the original 1.2% figure). Extrapolating the clean-stage
sum across 61 layers gives ≈20.3s of "real" compute for 1702 tokens (≈84 t/s
if the FFN gate exchange were fully hidden/overlapped) against the actual
measured 72-75 t/s - implying the **un-measurable-cleanly TP FFN gate costs
roughly 10-14% of prefill time** that isn't currently overlapped with
adjacent compute. Both are legitimate, Codex-independent next levers:

1. **output_proj** - now the top single-stage target. Worth checking whether
   it's already using WMMA/matrix cores for its dense GEMM the same way
   routed_moe now does, or still on a slower quantized path.
2. **Overlapping the FFN all-reduce gate with compute** - a real ~10-14%
   architectural opportunity, independent of (and safer than) the still-unsafe
   attention row split ([[ds4-attn-rowsplit-crash-2026-08-05]]).

Only 4 layers of clean data (profiler timeout cut the run short) - directionally
strong (the model's layers are architecturally homogeneous and all 4 samples
agreed closely) but should be corroborated with a full 61-layer run once the
hc_post/gate profiler-interaction bug is fixed or worked around (e.g. an
instrumentation point that profiles the gate call itself instead of the
boundary immediately after it).
