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
and dominant. After it, **output_proj (the attention output projection) is
now the single largest measured prefill stage**, not routed_moe and not SDPA
itself (still tiny, consistent with the original 1.2% figure). Extrapolating
the clean-stage sum across this GGUF's real 43 layers (`deepseek4.block_count`,
confirmed by parsing the GGUF header directly - DO NOT reuse "61 layers" from
earlier drafts of this doc, that number was never real) gives ≈14.3s of "real"
compute for 1702 tokens (≈119 t/s if the FFN gate exchange were fully hidden/
overlapped) against the actual measured 72-75 t/s - implying the
**un-measurable-cleanly TP FFN gate currently costs on the order of 35-40% of
prefill time**, much larger than a first pass with the wrong layer count
suggested. Treat this as a rough, indirect estimate (n=4 layers, extrapolated)
until corroborated with a full 43-layer run.

**Root cause of output_proj's cost, found by reading the code (not just
profiling): the mandatory `DS4_CUDA_NO_Q8_F16_CACHE=1` flag silently disables
output_proj's fast path too.** `cuda_q8_f16_cache_allowed()`
(`rocm/ds4_rocm_runtime.cuh:4972`) explicitly special-cases attn_output_a/b to
always be cache-eligible (line 4978-4983) - but checks
`getenv("DS4_CUDA_NO_Q8_F16_CACHE")` FIRST (line 4976) and returns 0
regardless if set. Every recipe this project has used requires that env var
to avoid the documented ~9.9 GiB OOM from the FULL cache (attn_output, q_lora,
ffn shared-expert weights, and more - see `cuda_q8_f16_cache_allowed`'s full
condition list). Side effect: `ds4_gpu_attention_output_q8_batch_f16_tensor`
(`rocm/ds4_rocm_attention_launch.cuh:940`, a cuBLAS f16 GEMM path, tried FIRST
for prefill's output_proj) always fails at its `cuda_q8_f16_ptr` call and
falls through to the slower `metal_graph_attention_output_dense_quant_batch`
fallback - on every single prefill call, this entire session. This was masked
before by routed_moe's even bigger pre-fix bottleneck; the WMMA fix exposed it.

## CONFIRMED WIN (2026-08-05): just dropping DS4_CUDA_NO_Q8_F16_CACHE=1 gets 94-99 t/s prefill

Tested the obvious experiment directly: what happens if the mandatory
NO_Q8_F16_CACHE flag is simply not set? This project already has a graceful
budget cap (`cuda_q8_f16_cache_has_budget`) that was either added since the
2026-08-03 OOM was documented or simply never exercised the same way before -
either way, the outcome today is clean, not a crash:

    ds4: ROCm q8 fp16 cache budget exhausted; using q8 kernels
         (request=64.00 MiB cached=9.85-9.91 GiB free=4.81-4.86 GiB
          reserve=4.80 GiB total=96.00 GiB)

The cache fills to ~9.85-9.91 GiB (matching the old "~9.9 GiB" figure
exactly), then gracefully stops and falls back to the original Q8 kernels
for anything that doesn't fit, always leaving its ~4.8 GiB reserve free.
**No OOM, no crash, on either rank, across 5 separate runs** (three
short -n 20/30 samples, one -n 500 long-decode stress test, one
--dump-logprobs correctness run).

**Prefill result: 94.97, 98.75, 94.59, 98.82 t/s (mean ~96.8, 4 samples)** -
up from the existing 74.80-75.65 t/s golden number, and now AT OR ABOVE the
top of the llama.cpp golden-evidence range (80-95 t/s,
[[llamacpp-prefill-golden-evidence-resolved]] /
`WHY-VLLM-PREFILL-IS-6X.md`) rather than approaching it from below. Decode
unaffected (10.36-11.07 t/s, consistent with the existing range).

**Correctness confirmed, not assumed:** `--dump-logprobs` diff of 30 greedy
steps against the established-good NO_Q8_F16_CACHE=1 baseline, same prompt,
`--temp 0`: **0/30 selected tokens differ, generated text byte-identical.**
Per-step logit values differ by up to ~1.6 (logprob up to ~0.21) - expected
floating-point variation between the F16 cuBLAS GEMM path and the on-the-fly
Q8_0 kernel path, not a correctness issue (the decision never changes).

**Root cause recap:** `cuda_q8_f16_cache_allowed()` special-cases
attn_output_a/b to always want caching (`ds4_rocm_runtime.cuh:4978-4983`),
but the env-var check earlier in the same function (line 4976) blocked
this unconditionally whenever `DS4_CUDA_NO_Q8_F16_CACHE=1` was set - which
every recipe in this project has always set, to avoid the documented OOM.
That silently killed the existing, already-correct fast path
(`ds4_gpu_attention_output_q8_batch_f16_tensor`,
`rocm/ds4_rocm_attention_launch.cuh:940`) for the ENTIRE session, on every
prefill call, with no error - just a silent fallback to a slower kernel.

**This changes the project's baseline recipe.** `DS4_CUDA_NO_Q8_F16_CACHE=1`
should no longer be treated as mandatory - see
[[ds4-tp-rocm-two-node]] and [[ds4-q8f16-cache-prefill-win-2026-08-05]] for
the corrected recipe. Not yet tested: contexts much longer than `-c 4096`
(this project's usual test size) or heavier concurrent memory pressure -
the ~4.8 GiB reserve margin should be re-checked before assuming this holds
at every deployment context length.

This closes out this iteration's remaining open levers - "overlapping the
FFN gate" (~35-40%, estimated above) is still open but unvalidated, and
output_proj's own remaining share should be re-measured now that its real
fast path is active.

## Clean, direct FFN gate-cost measurement (2026-08-05) - refines the estimate to ~39%

A Codex research pass (dispatched to design the gate-overlap redesign) found
an existing, previously-unused instrumentation point that avoids the
`hc_post` profiler-sync corruption entirely: `DS4_TP_BIGGATE_PROFILE=1`
(`ds4_tp.c:1227`), which times staging-copy vs wire+wait directly inside
`ds4_gpu_tp_big_gate_encode`'s own call path and prints cumulative stats
every 16 gates - no forced device sync at a stage boundary, so it doesn't
have the earlier problem.

Measured (same env as the "CONFIRMED WIN" run above, plus
`DS4_TP_BIGGATE_PROFILE=1`, ~1700-token prompt, `DS4_TP_BIG_DIRECT=1`):

    ds4-tp: big-gate 16 gates direct=1 | staging-copy 0.0 ms (0%) | wire+wait 2497.3 ms (100%) | ...
    ds4-tp: big-gate 32 gates direct=1 | staging-copy 0.0 ms (0%) | wire+wait 5195.6 ms (100%) | ...

Staging-copy is 0% as expected with direct mode. Per-gate rate over gates
17-32: (5195.6-2497.3)/16 ≈ 168.6 ms/gate. Extrapolating that rate for the
model's real 43 layers (11 more gates beyond 32, since the process ended
before a 48-gate checkpoint printed): total gate cost ≈ 5195.6 + 11*168.6 ≈
**7.05 s**. Measured end-to-end prefill was 94.96 t/s for ~1700 tokens ≈
**17.9 s** wall time. **Gate cost ≈ 7.05/17.9 ≈ 39% of prefill** - a solid,
directly-measured number, not an indirect stage-sum extrapolation, and it
lands close to the earlier (output_proj-fix-era) ~35-40% estimate rather
than the later, smaller-sample ~50%+ one. Treat ~39% as the current
best-evidence figure for the gate-overlap redesign's real potential.

This does NOT mean an overlap fix would recover the full 39% - some of that
wire+wait time is genuine RDMA transfer that can't be hidden, only the
portion where the GPU/CPU sit idle waiting instead of doing other useful
work can be reclaimed by pipelining. See `FFN-GATE-OVERLAP-RESEARCH.md` and
the Codex gate-overlap design writeup (row-chunk pipelining recommended,
next-layer overlap ruled out as unsafe due to the residual-stream
dependency) for the actual redesign plan.
