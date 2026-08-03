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

## Still to fix

Which of the five views is NULL, and why. `ds4_gpu_tensor_view` IS defined on
ROCm (not a stub), so the likely cause is a NULL *base* tensor - `g->batch_q_half`
is the prime suspect since it is the one view built directly from a raw buffer
rather than through `metal_graph_tensor_row_range_view`.

Note this also means **patches 10/11 (attention_output_q8_tp, kslice) have still
never executed on the prefill path** - the row split is where they would be
exercised, and it has never run successfully.
