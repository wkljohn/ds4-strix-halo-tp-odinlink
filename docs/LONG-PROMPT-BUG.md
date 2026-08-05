# Prompts over ~32 tokens were completely broken (and the error blamed the wrong thing)

Found by finally testing a prompt that was not 13 tokens long. Everything in this
repo up to this point had been validated on ONE short prompt.

## Symptom

Any prompt >= 32 tokens:

    ds4: gpu layer 0 raw KV batch store failed
    ds4: gpu layer 0 attention batch encode failed
    ds4: gpu whole-prefill layer 0 encode failed
    ds4: prompt processing failed

62-token prompt: fails. 3306-token prompt: fails. 13-token prompt: works. That
threshold is `metal_graph_tp_prefill_split_min()` = **32** (ds4.c), the point at
which the TP row split engages.

## The message is a lie

`ds4.c:27152` prints "raw KV batch store failed" whenever `ok` is false at that
point - but the store is guarded by `if (ok && zero_prefix)`, so when `ok` was
already false the store NEVER RAN and the message blames it anyway. Instrumenting
the store proved this: neither its rejection path nor its launch path was reached.

The real failure is at **ds4.c:26999**:

    if (tp_row_split_attn &&
        (!tp_q || !tp_q_half || !tp_qr_norm || !tp_heads || !tp_attn_out)) {
        ok = false;         /* silent - no message at all */
    }

Under the row split, five row-range views of the batch tensors are built. If any
is NULL, `ok` goes false with no diagnostic, and the next `if (!ok)` further down
attributes it to whatever it happens to guard. Same misattribution class as
`cudaGetLastError()` reporting a stale error.

## Workaround (works today)

    DS4_TP_PREFILL_SPLIT_MIN=999999

Disables the row split, so prefill replicates attention across both ranks
instead of splitting rows. Measured with a 3306-token prompt:

    prefill: 29.18 t/s, generation: 10.49 t/s

and the model correctly comprehends the long context - it identifies all three
questions and reasons correctly about the third ("The answer should be No").
Decode throughput is unchanged (10.49 vs ~10.5), because the row split only
affects prefill.

**This should be the default until the view bug is fixed.** Costs some prefill
parallelism; buys the ability to use prompts longer than a sentence.

## Root cause found (patch 19)

Instrumented the guard. The failing view is `tp_q_half`, and its BASE is NULL:

    TP row-split view NULL: q=1 q_half=0 qr_norm=1 heads=1 attn_out=1
      [row0=0 rows=31 q_dim=32768 q_rank=1024 n_tokens=62 batch_q_half=(nil)]

`g->batch_q_half` is allocated only under `DS4_GPU_ATTN_COMP_CACHE_F16`
(ds4.c:17277), and that macro is:

    #if defined(__APPLE__)
    #define DS4_GPU_ATTN_COMP_CACHE_F16 1
    #else
    #define DS4_GPU_ATTN_COMP_CACHE_F16 0

So on ROCm the buffer never exists, its view is always NULL, and the guard always
fails. **The TP prefill row split was Apple-only by accident.** The non-TP path
already passes that same NULL through without complaint.

Patch 19 only demands the q_half view when the buffer it views exists.

## Next blocker after patch 19

With patch 19 the row split proceeds and immediately hits a genuinely missing
kernel - named instantly by the stub announcer:

    ds4: ROCm UNAVAILABLE STUB CALLED: ds4_gpu_attention_prefill_raw_heads_range_tensor

This is NOT composable from the existing `ds4_gpu_attention_prefill_raw_heads_tensor`.
The header (ds4_gpu.h:1770) says q is already a view of rows
`[q_row0, q_row0+n_q)` and raw_kv keeps all n_kv rows - but the ROCm kernel
`attention_static_mixed_heads8_online_kernel`
(rocm/ds4_rocm_attention.cuh:1191) takes the token index straight from
`blockIdx.x` and uses it for causal masking, so a q-range start needs a new
absolute-position parameter threaded through every branch of the dispatch, not
a wrapper.

Deliberately NOT attempted unsupervised: an off-by-one in causal masking is
silent wrongness that a smoke test passes, and there is no reference run on this
hardware to validate against.

## Therefore: DS4_TP_PREFILL_SPLIT_MIN=999999 is REQUIRED, not optional

Until that kernel exists, the row split cannot work on ROCm at all. Disabling it
is the only way to use prompts over 32 tokens. It costs prefill parallelism only
- decode is unaffected (10.49 vs ~10.5 t/s measured).

Note this also means **patches 10/11 (attention_output_q8_tp, kslice) have still
never executed on the prefill path** - the row split is where they would be
exercised, and it has never run successfully.

## Status update (2026-08-05): still unfixed, now more valuable, real gain estimated

Confirmed still current via both code inspection and Codex research: ROCm
now defaults `DS4_TP_PREFILL_SPLIT_MIN` to 1,000,000 automatically (no
longer requires the operator to set it - see commit `f359858`), but the
underlying missing kernels (`ds4_gpu_attention_prefill_raw_heads_range_tensor`
AND a second, separately-missing static-mixed-attention range API) are
still hard-fail stubs. Attention during prefill is still fully replicated
across both TP ranks. FFN's split threshold is independent and unaffected
(stays at 32).

**Re-scoped, honest estimate of the real gain** (Codex research pass,
2026-08-05): this is bigger than "half the measured 1.2% SDPA share." The
row-split boundary is broader than just SDPA - it also covers most of the
attention output projection (13.1% of profiled prefill time) and part of
q_path (1.7%). Conservative realized gain estimate: **4-7%** after
accounting for the TP exchange cost the split reintroduces; **7-9%**
plausible if that exchange is cheap; **~10%** possible post-WMMA-fix,
since accelerating routed MoE (this session's work) makes the unchanged
attention cost a LARGER fraction of the new, smaller total prefill time.

**This now lines up cleanly with a fresh, real hardware measurement**: at
a realistic ~1694-token prompt (not this project's usual ~50-token smoke
test), current post-WMMA-fix prefill measures **74.80-75.65 t/s** (3
samples, ~1.1% spread - see `WHY-VLLM-PREFILL-IS-6X.md`'s update). Against
the golden-evidence llama.cpp reference of 80-95 t/s, that's a remaining
gap of roughly 5-21% - right in the range this row-split fix is estimated
to close. The "40 vs 90" gap the operator originally asked about was
mostly a short-prompt measurement artifact (most experts never reaching
ds4's WMMA engagement threshold on such a short prompt) plus this
still-open replicated-attention issue - not a mysterious, unexplained
deficit.

**Implementation is real work, not a quick patch** (Codex research,
2026-08-05): the existing fast kernel conflates "local query row" and
"absolute causal position" (both are `blockIdx.x`) throughout
`ds4_gpu_attention_prefill_raw_heads_tensor()`'s every dispatch branch
(heads8-online path, hipBLAS path, scalar fallback) AND a SEPARATE
static-mixed-attention range API is also missing and must be implemented
too - fixing only the raw-range symbol would leave ordinary compressed
layers (the common case) still unable to split. Required validation is
non-trivial: coherent-looking text is NOT sufficient (a silently-wrong
causal window could still produce plausible output) - needs a tensor-level
diff between forced-split and forced-replicated output at multiple
boundary conditions (split threshold 31/32/33, window edges, compression
emission boundaries, the indexer/static-mixed transition), not just a
smoke test.

**Priority order going forward** (Codex recommendation, consistent with
the fresh measurement above): (1) implement + rigorously validate the
attention range split - the clear next lever; (2) re-profile current
post-WMMA long-prompt prefill on both ranks symmetrically to get accurate
current stage shares (the old profile predates this session's complete
WMMA path); (3) only then reconsider WMMA tile-width breadth, and if so,
prioritize wider tiles (J=32) over narrower ones (J=8 was considered and
is NOT actually what llama.cpp uses for RDNA3.5 Q4_K - its own config
table starts at J=16 - nor a natural fit for ds4's 16-column-based WMMA
accumulator design; the cold (<6 pairs/expert) population is likely small
at realistic prompt lengths anyway, consistent with the ~75 t/s long-prompt
result above already looking close to the llama.cpp reference without any
tile-width change).

## IMPLEMENTED but UNSAFE - real memory-corruption bug found, unresolved (2026-08-05)

Implementation attempted (`ds4-upstream@3b120e0`). Both range functions
are now real (not stubs), builds clean, `make test-tp-hello` passes, and
a full structural review confirmed every dispatch path (raw: fast/cuBLAS/
scalar; static-mixed: fast/cuBLAS/cuBLAS-tiled/scalar) threads `q_row0`/
`n_q` consistently.

**Process note**: Codex wrote the initial version but its account hit a
usage quota mid-task (`try again at Aug 11th, 2026`), before it could
build or test anything. The left-behind code did not compile: some
inconsistent variable renaming, one kernel signature not updated to match
its call site's new argument count, and two ENTIRELY OUT-OF-SCOPE kernels
(`attention_decode_mixed_kernel`, `attention_indexed_mixed_heads8_online_kernel`
- both already correctly designed for the range case via their own
existing `pos0` parameter, shared with decode, and neither needed any
change) were incorrectly touched and left broken. All fixed directly by
Sonnet: reverted the two out-of-scope kernels to their original form,
added the missing parameter to the one incomplete signature, and reverted
an unnecessary parameter addition to a KV-packing kernel that never
needed it (it packs the full, unsliced KV set, not query rows).

**Hardware validation (mandatory per this project's own established bar -
tensor/logprob comparison, not just coherent text) found a real crash,
not a subtle numerical bug.** Forcing `DS4_TP_PREFILL_SPLIT_MIN_ATTN=2`
against a forced-replicated baseline (`=999999`) segfaults on every
attempt, both a long (~1694-token) and short (~35-token) prompt. Two
separate gdb backtraces were captured on different runs:

1. `hipBLAS f16 matmul failed: status 2` in the ratio-4 COMPRESSOR tail
   projection (unrelated code, never touched by this change) → graceful
   `ok=false` cascade → then a SEPARATE thread segfaults inside
   `libhsa-runtime64.so.1` during what looks like async cleanup.
2. A direct SIGSEGV inside `moe_q4K_routed_wmma_kernel` - this session's
   completely separate down-projection MoE WMMA work
   (`routed_moe_launch`, `rocm/ds4_rocm_moe_launch.cuh:1292`) - at layer
   11, `n_tokens=42`, called from the ordinary FFN batch path.

**Two different crash sites, both in code this change never directly
touches, is the signature of memory corruption (a buffer overrun or a
race), not one deterministic logic bug in the new attention kernels
themselves.** One scoped hypothesis was tested and did NOT resolve it:
`cuda_tmp_alloc` (`rocm/ds4_rocm_runtime.cuh:567`) is a single GLOBAL
scratch buffer shared by every caller in this file and beyond (attention,
compressor, MoE, ...), reused only if a new request fits, otherwise freed
and reallocated with **no synchronization** against in-flight async GPU
work still reading the old buffer. The new range functions request
`n_q`-sized (smaller, rank-dependent) scratch instead of the square
path's stable `n_tokens`-sized scratch, which seemed likely to churn this
unsynchronized shared buffer more than before. Stabilizing the request
size to always match `n_tokens` (see the fix in the same commit) did NOT
stop the crash - so either this wasn't the (whole) mechanism, or there's
a second contributing factor.

**Status: DO NOT enable this feature.** The feature gate itself
(`metal_graph_tp_prefill_split_min_attn()`, `ds4.c` - NOT touched by this
commit) still defaults to 1,000,000 on ROCm, so none of this new code path
is reachable with default settings - this is dead code in normal
operation and does not affect any of this session's other validated
results (the down-projection WMMA fix, the 74.8 t/s long-prompt
measurement, Stage 0b/0c/0d decode diagnostics - none of these set
`DS4_TP_PREFILL_SPLIT_MIN_ATTN` below its safe default). Root-causing this
properly needs either deeper GPU-side debugging tooling than was
available this session, or Codex's help once its quota resets
(~2026-08-11) - do not attempt to re-enable or further stress-test this
path without that, and do not treat "it didn't crash this run" as
evidence of correctness given the intermittent, memory-corruption-shaped
failure pattern already observed.
