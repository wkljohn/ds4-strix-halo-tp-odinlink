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
