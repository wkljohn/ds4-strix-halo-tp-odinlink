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
