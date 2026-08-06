# ds4 TP=2 on ROCm: WORKING, correct output

    prompt : "What is the capital of France? Answer with one word."  (--temp 0, -n 200)
    output : "We need answer user's question. The user asked "What is the capital
              of France? Answer with one word. The answer is Paris. ..."
    ds4: prefill: 32.15 t/s, generation: 7.43 t/s

Coherent English, correct answer, no stubs called, no Metal-only paths, no errors
on either rank. DeepSeek V4 Flash Q4_K (153 GiB) across two
Strix Halo gfx1151 nodes, 80.76 GiB resident per rank, TCP transport.

The repetition after the answer is not a defect: this is a raw completion with no
chat template and no stop token, so a reasoning model restates and loops. It ran
to ~3390 tokens before being bounded with `-n 200`.

## The bug that was breaking it: patch 15, fixed by patch 16

Patch 15 folded the routed-MoE addend BEFORE `routed_moe_launch`, justified by a
comment claiming the launch "ACCUMULATES into out". **That premise was false.**
Every terminal write on ROCm is an assignment:

    rocm/ds4_rocm_moe.cuh:2091  out[row] = total;
    rocm/ds4_rocm_moe.cuh:2257  out[row] = total;   <- the Q4_K path decode takes
    rocm/ds4_rocm_moe_launch.cuh:1470  zero_kernel<<<...>>>(out->ptr, n)

and there is no `out[...] +=` anywhere in either file. So the pre-added addend
was overwritten and **the shared expert was silently dropped from every decode
layer**.

Upstream refused this case outright ("routed MoE addend fold is Metal-only")
because Metal folds the addend INSIDE the down kernel (ds4_metal.m:34779 ->
metal/moe.metal:6091-6094). Patch 15 replaced a fail-closed refusal with a silent
wrong answer, which is strictly worse than the refusal it removed.

Patch 16 moves the fold after the launch and uses `out_dim` rather than
`out->bytes / sizeof(float)` (the allocation size, a latent overrun).

## What this corrects in the earlier analysis

- **My "addend double-count" refutation was half right and half wrong.** The
  addend is not double-counted - the shared expert IS legitimately K-split per
  rank. But I concluded from that "patch 15 is correct" without ever asking
  whether the addend SURVIVES the launch. It did not.
- **The head-split refutation was vacuous.** `DS4_GLM_TP_HEAD_SPLIT_MIN` is read
  only inside the GLM graph; this model is DeepSeek-V4, so the run was
  byte-identical a priori and tested nothing.
- **Patches 10/11/14 were never on the prefill path.** For a ~20-token prompt,
  row-split needs n_tokens >= 32, so prefill replicates attention and the shared
  expert entirely. The 44 BIG gates observed corroborate this - attention
  row-split would have produced ~86.

## Numbers in context

| | bytes/node/token | measured | own ceiling | efficiency |
|---|---|---|---|---|
| llama.cpp no draft (Q8_K_XL, pipeline) | 16.9 GiB | 9.42 t/s | 13.2 t/s | 71% |
| **ds4 TP=2 (Q4_K, real TP)** | 8.91 GiB | **7.43 t/s** | 25.1 t/s | **30%** |

Still 30% of our own bandwidth ceiling, so the efficiency gap is the remaining
prize - not the attention split, and not the transport.
