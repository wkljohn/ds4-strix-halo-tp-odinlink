# First end-to-end ds4 TP=2 generation on ROCm - runs, but output is CORRUPT

    TheergyParis半功... {st \-re − Responsedryd\  ）-“INgo
    ds4: prefill: 27.98 t/s, generation: 6.18 t/s

The plumbing works end to end. The numerics do not. "Paris" appears in the
stream, so the model is close to right and then corrupted - the exact
silent-wrongness mode flagged as the top risk in DS4-TP-EVALUATION.md
("shapes stay consistent, output is corrupt, nothing crashes").

**6.18 t/s is NOT a usable result.** It is the throughput of a wrong computation
and must not be compared to llama.cpp's 15 t/s until the output is correct.
Fixing correctness can only slow it down.

## What it took to get here (this session)

| patch | defect |
|---|---|
| 13 | `hipStreamWriteValue64` never lands on the NULL stream (gfx1151/ROCm 7.2). Gate moved to a dedicated stream. |
| 13 | `cuda_tmp_alloc` is one-slot; `routed_moe_launch` re-enters it while the patch-12 remap is live -> garbage `counts[]` -> unbounded loop. Dedicated buffer. |
| 14 | `ds4_gpu_matmul_quant_kslice_tensor` implemented - the TP shared-expert K-split (`tp_split_shared`, ds4.c:23882) routes Q8_0 through the GENERIC quant entry point, not the q8 one. |
| 15 | routed-MoE addend fold was not Metal-only at all; CUDA does it in 2 lines (ds4_cuda.cu:22255) and ROCm already had `ds4_gpu_add_tensor`. |

A general diagnostic that paid for itself immediately: making all 30
`ROCM_UNAVAILABLE_INT` stubs announce themselves once. They return 0 silently,
which is indistinguishable from a genuine validation failure; the announcer
named `matmul_quant_kslice` in a single run after three rounds of one-by-one
instrumentation had failed to find it.

## Leading suspect for the corruption: patch 15 double-counts the addend under TP

`ds4_gpu_routed_moe_one_tensor` now does `out += add_in` BEFORE the launch,
mirroring CUDA. Under TP=2 **both ranks execute this**, then the gate exchange
combines the two partial block outputs. If that combine is a SUM, the result is

    (add_in + sum(owned_0)) + (add_in + sum(owned_1)) = 2*add_in + sum(all)

i.e. the residual is counted twice, every layer. That would produce exactly this
signature: structurally plausible tokens ("The", "Paris") drifting into garbage
as the error compounds across 43 layers.

CUDA gets away with the same code because its TP is the single-process multi-GPU
tier path, which may fold the addend on one tier only - that needs checking, not
assuming.

## Verify before fixing

Do NOT just move the add. First establish the combine semantics:
1. Read what the FFN gate combine actually does with the two partials
   (sum vs. concat vs. overwrite) - `ds4_tp.c` gate slot handling and the Metal
   reference `ds4_metal.m`.
2. If it is a sum, the addend must be folded by exactly one rank (leader), or
   after the combine.
3. Only then re-measure. Correctness first, then the number.

Other candidates, not yet excluded:
- the unowned-pair convention (index 0, weight 0) interacting with
  `norm_topk_prob=true` if the router normalises AFTER selection
- attention head-split combine at group boundaries
- the shared-expert K-split halves (patch 14) not being summed correctly
