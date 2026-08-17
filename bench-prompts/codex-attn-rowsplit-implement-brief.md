# Implement the TP=2 prefill attention row split (raw-range + static-mixed-range)

Repo: a clean checkout of this fork. This is real implementation
work with a genuine correctness risk (silent wrong causal masking) - read
this brief in full before starting, and follow the validation plan at the
end exactly. Unrelated to the down-projection WMMA fix, decode profiling
instrumentation, and launch-config sweeps already shipped this session -
do not touch those files.

## Why

`ds4-strix-halo-tp/docs/LONG-PROMPT-BUG.md` (sibling repo, read if
reachable) documents that ds4's TP=2 prefill attention row-split has never
worked on ROCm: two required functions,
`ds4_gpu_attention_prefill_raw_heads_range_tensor` and
`ds4_gpu_attention_prefill_static_mixed_heads_range_tensor`, are hard-fail
stubs (`ds4_rocm_unavailable.cu:27-28`). Their call sites already exist and
are exercised on every TP=2 prefill chunk (`ds4.c:27252, 27265, 28193,
28210, 28241, 28254`) - the caller-side logic (which rank owns which rows,
`tp_row0`/`tp_rows` computation) is already correct and working
(`ds4.c:26840-26852`, the `tp_row_split_attn`/`tp_half_rows` block). Only
the ROCm kernel implementations are missing, so attention runs fully
REPLICATED on both ranks during prefill today - real, measured wasted
compute (Codex's own earlier research estimated 4-10% of prefill time).

## The exact interface contract (already specified, do not change)

`ds4_gpu.h:1802-1818` and `:1910-1931` document it precisely: `q` is a view
of the `n_q` local query rows at absolute chunk positions `[q_row0,
q_row0+n_q)`; `raw_kv` (and `comp_kv` for the mixed variant) keep ALL rows/
keys (every rank needs the full KV history); `heads` receives `n_q` local
output rows. The existing SQUARE entries
(`ds4_gpu_attention_prefill_raw_heads_tensor`,
`ds4_gpu_attention_prefill_static_mixed_heads_tensor`) are the `q_row0=0,
n_q=n_kv` (or `n_q=n_tokens`) special case and already work correctly -
use them as your reference implementation, do not redesign the algorithm.

## The precise, uniform bug pattern to fix (I traced this myself, verify it)

Every kernel dispatched by the two square functions conflates two distinct
concepts using a single `t = blockIdx.x`:

1. **Local output-row index** - which of the `n_q` local `q`/`heads` rows
   this block computes. This must range over `[0, n_q)` and index `q`/
   `heads` (which are LOCAL, `n_q`-sized views in the range case).
2. **Absolute causal position** - which token position in the full,
   unsplit chunk this row corresponds to, used for the window/causal
   count math (`raw_count`, `raw_start`, `comp_count`/`visible_comp`) and
   for indexing into `raw_kv`/`comp_kv` (which stay FULL-sized, unsplit,
   in both the square and range cases).

In the square case these are numerically identical (`q_row0=0`), which is
exactly why the bug has never surfaced. For the range case they must be
separated: keep `t_local = blockIdx.x` (bounds-checked against `n_q`, used
for `q`/`heads` indexing) and compute `t_abs = q_row0 + t_local` (used for
every causal-masking/window/count computation). Verify this analysis
yourself against the current code before implementing - do not take it on
faith - but this is what I found:

### Raw path (`ds4_gpu_attention_prefill_raw_heads_range_tensor`)

Reference square function: `ds4_gpu_attention_prefill_raw_heads_tensor`,
`rocm/ds4_rocm_attention_launch.cuh:151-249`. Three internal dispatch
paths, ALL need the fix:

1. **Fast path** (`n_tokens>1 && head_dim==512 && !quality && window<=768`):
   dispatches `attention_static_mixed_heads8_online_kernel`
   (`rocm/ds4_rocm_attention.cuh:1191-1313`). `t = blockIdx.x` at line 1203
   is used for the bounds check (`t >= n_tokens`) AND directly in
   `raw_count = window && t+1>window ? window : t+1` (line 1213) AND
   `comp_count = (t+1)/ratio` (line 1217) AND for indexing `q`/`heads`
   (lines 1242, 1307). This kernel has NO existing position parameter -
   you'll need to add one (e.g. `q_row0`), keep the bounds check and q/
   heads indexing on `t_local`, and use `t_abs = q_row0 + t_local`
   everywhere else. Do NOT modify this kernel in place if it's shared
   with a path you're not touching - check all callers first (it's called
   from BOTH `ds4_gpu_attention_prefill_raw_heads_tensor:165` and
   `attention_prefill_mixed_launch:689` with a fixed `q_row0=0`; adding a
   parameter with those existing calls passing 0 is safe and non-breaking,
   confirm this before proceeding).
2. **cuBLAS path** (`g_cublas_ready && n_tokens>1 && head_dim==512`,
   `:178-240`): the GEMM computes `q_row0`-agnostic partial scores already
   (it's just Q@K^T, no masking) - the masking happens in
   `attention_prefill_raw_softmax_kernel` (`rocm/ds4_rocm_attention.cuh:147-186`),
   where `t = blockIdx.x` is used both to index the scores row
   (`row = scores + h*n_tokens*n_keys + t*n_keys`, needs `t_local` since
   scores is sized for `n_q` rows in the range case) AND for the causal
   check `valid = k <= t && (window==0 || t-k<window)` (line 162, needs
   `t_abs`). The GEMM's `q` pointer arithmetic and `n_tokens`/output shapes
   need to become `n_q`-sized (rectangular, not square) throughout this
   path - read it carefully, this is the most involved of the three.
3. **Scalar fallback** (`:241-248`): dispatches
   `attention_prefill_raw_kernel` (`rocm/ds4_rocm_attention.cuh:11-65`).
   Same `t = blockIdx.x` pattern, used for the bounds check, `raw_count`/
   `raw_start` (lines 22-24), and q/heads indexing (lines 25, 57). Same
   fix: add `q_row0`, separate local vs absolute.

### Static-mixed path (`ds4_gpu_attention_prefill_static_mixed_heads_range_tensor`)

Reference square function:
`ds4_gpu_attention_prefill_static_mixed_heads_tensor`,
`rocm/ds4_rocm_attention_launch.cuh:839-858`, which delegates to
`attention_prefill_mixed_launch` (`:655-837`) - itself three internal
paths with the exact same `t`-conflation pattern:

1. **Fast path** (`:685-701`): reuses the SAME
   `attention_static_mixed_heads8_online_kernel` as above (with
   `q_row0=0` today) - the fix here is the same parameter addition, this
   call site just needs to pass the real `q_row0`.
2. **cuBLAS path** (`:702-814`): masking happens in
   `attention_prefill_mixed_softmax_kernel` (find its definition - not yet
   read, do so) - same local/absolute split needed as the raw softmax
   kernel, likely also `visible_comp`/`comp_count` math depending on
   absolute position. There is ALSO a `attention_prefill_mixed_cublas_tiled`
   fallback (`:713,729`, for huge tmp_bytes) you have not yet been shown -
   find and read it; it likely needs the identical treatment or must be
   explicitly rejected (return 0) for the range case if it's out of scope
   for this pass - your call, but state which you chose and why.
3. **Scalar fallback** (`:827-836`): dispatches
   `attention_prefill_mixed_kernel` (`rocm/ds4_rocm_attention.cuh:67-145`).
   Same `t = blockIdx.x` pattern (lines 81, 85-88, 111-116, 138-143).

## Requirements

- **Do not modify the existing square functions' call sites/behavior** -
  they must produce byte-identical results after your change (verify by
  literally diffing a forced-square run before/after your patch).
- Add the two new `extern "C"` functions matching the exact signatures in
  `ds4_gpu.h:1806-1818` and `:1915-1931`.
- Prefer parameterizing/extending the EXISTING kernels (adding a `q_row0`
  parameter, defaulting existing call sites to pass 0) over duplicating
  kernel bodies, where a kernel is shared between the square and range
  paths - less code, less drift risk. Where a kernel is NOT shared (e.g.
  if the scalar/cuBLAS paths turn out to need genuinely different grid
  shapes), use your judgement, but document why you chose one approach
  over the other.
- Match this codebase's existing bounds-checking and fail-closed style
  (see the square functions' argument validation blocks) - the range
  versions need equivalent checks against `n_q`/`q_row0+n_q<=n_kv` etc.
- Build cleanly (`make strix-halo`), run `make test-tp-hello`, confirm
  both pass.
- Do NOT attempt hardware validation yourself (no ROCm device/network in
  your sandbox, established in prior tasks on this project) - I have real
  hardware access and will run the mandatory validation below myself.

## MANDATORY validation plan (I will run this, but design your implementation to make it possible)

A coherent-looking generation is NOT sufficient evidence of correctness -
a silently-too-narrow causal window would still produce plausible text.
I need to be able to, at minimum:

1. Force `DS4_TP_PREFILL_SPLIT_MIN_ATTN=<small value>` to engage the row
   split on a real prompt, and compare FINAL LOGITS/greedy token choices
   (not just "looks fluent") against the same prompt/config with the row
   split forced off (`DS4_TP_PREFILL_SPLIT_MIN_ATTN=999999`), on both
   ranks. Exact-match is the bar (or the same tight tolerance this
   project already uses elsewhere for reduction-order-preserving changes -
   this change should NOT alter reduction order, so exact match is the
   right expectation, not "close").
2. Test at multiple prompt lengths straddling boundary conditions: right
   at the split threshold, odd token counts (so `tp_half_rows` splits
   unevenly), and long enough to exercise the compressed/indexed paths
   (`ratio!=0` layers) not just the raw path.
3. Ensure the existing debug tensor-dump mechanism
   (`metal_graph_debug_dump_tensor`, used throughout `ds4.c`) can be used
   or extended to compare per-rank `heads` output directly if a
   full-model logit comparison doesn't isolate the bug cleanly enough -
   note if you added or found a convenient dump point for this.

Report back: what you changed and why, which of the described paths you
implemented vs explicitly left unsupported (if any, with a clear reason
and a fail-closed return-0 fallback), confirm build/test pass, and
describe exactly how I should run the differential validation above
against your specific implementation (exact env vars, what output to
compare).
