# What actually executes on the TP path, traced (not assumed)

Twice in this project a scoping claim ("that function is not on our path") was
accepted without tracing the call chain, and both times it was wrong. This
document traces the chain end to end and records the evidence.

## The two TP mechanisms are NOT the same path

ds4 has two distinct tensor-parallel implementations, and they gate different code:

| | mechanism | gate variable | our topology? |
|---|---|---|---|
| **network TP** | 2 processes, 2 nodes, 1 GPU each, `ds4_tp.c` | `g->tp_world == 2`, `g->tp_rank` | **YES** |
| mgpu TP | 1 process, 2 GPUs, tiers, `ds4_gpu_mgpu.h` | `cuda_tp_attn`, `*_by_tier[]`, `copy_xdev` | no - ROCm is one GPU per process (`ds4_rocm_compat.cu:21`) |

Confusing the two is the whole trap: they call overlapping helpers under
different conditions.

## The chain that fires for us

`ds4.c:22609`:

    } else if (ok && g->tp_world == 2) {
        /* Group-sliced attention output: this rank computes its half of the
         * output groups and the matching k-window of the expand projection */
        const uint32_t tp_groups = n_groups / 2;
        ok = metal_graph_attention_output_dense_quant_tp(..., g->tp_rank * tp_groups, tp_groups, ...)

-> `metal_graph_attention_output_dense_quant_tp` (`ds4.c:24816`)
-> if `out_a->type == Q8_0 && out_b->type == Q8_0`: **`ds4_gpu_attention_output_q8_tp_tensor`** (patch 11)
-> which composes `attention_output_low_q8_tensor` + **`matmul_q8_0_kslice_rows_tensor`** (patch 10)

Both were "Metal-only" stubs upstream. Both are implemented now.

## The three silent stubs left in ds4_rocm_unavailable.cu

These are `return 0` with **no message** - the dangerous kind. Traced:

| stub | sole call site | gate | on our path? |
|---|---|---|---|
| `matmul_q8_0_kslice_hc_expand_add_tensor` | `ds4.c:22568` | `fuse_tp_attn_out_hc` => requires `cuda_tp_attn` | **no** - mgpu only |
| `shared_down_hc_expand_add_q8_0_tensor` | `ds4.c:23850` | `cuda_tp_moe_peer_tmp` (`tp_peer_tmp_by_tier`) | **no** - mgpu only |
| `matmul_quant_kslice_tensor` | `ds4.c:24738` | `dense_quant_tp` fallback when attn_output is **not Q8_0** | **no, for this model** - see below |

The third is the one that depends on the checkpoint, so it was checked against
the actual file rather than reasoned about.

## Checkpoint evidence (read from the GGUF header, no VRAM)

`scripts/gguf_tensor_types.py` on both local variants of
Huihui-DeepSeek-V4-Flash-0731-abliterated-GGUF:

    Q4_K   variant: F32 492, F16 359, Q8_0 345, Q4_K  129, I32 3
    mxfp4  variant: F32 492, F16 359, Q8_0 345, MXFP4 129, I32 3

    attn_output tensors: Q8_0, count=86   (= 43 layers x {a,b})

**The "Q4_K"/"mxfp4" in the filename describes the 129 expert tensors only.
Attention output stays Q8_0 in both.** So `dense_quant_tp` takes its Q8_0 branch
and `matmul_quant_kslice_tensor` is never reached. Implementing it is therefore
NOT a prerequisite - though it remains the one stub that would matter if the
checkpoint were ever re-quantised with non-Q8_0 attention output.

## Geometry, and why the fail-closed guards pass

From the same header:

    deepseek4.attention.head_count       64      -> n_groups
    deepseek4.embedding_length           4096    -> DS4_N_EMBD
    deepseek4.block_count                43
    blk.0.attn_output_a.weight  (4096, 8192)
    blk.0.attn_output_b.weight  (8192, 4096)     -> n_groups_total * rank = 8192 -> rank = 128

At `tp_world == 2`:

    tp_groups = 64 / 2 = 32
    group0    = tp_rank * 32     in {0, 32}
    k_off     = group0 * 128     in {0, 4096}    both % 32 == 0  OK
    k_cnt     = 32 * 128 = 4096              4096 % 32 == 0  OK

Two consequences worth recording:

1. **The 32-alignment guard passes.** Patch 10/11 fail closed on unaligned
   slices (a misaligned Q8_0 slice mixes neighbouring blocks' scales and is
   silently wrong). For this model the slices are aligned, so the guard permits
   rather than refuses.
2. **`n_groups` is even, so the split is exact.** Note that the network-TP site
   computes `n_groups / 2` with **no parity check**, unlike the mgpu site which
   requires `(n_groups % 2u) == 0u`. With an odd head count the two ranks would
   together cover `2 * (n/2) < n` groups and quietly drop one. 64 is even, so
   this does not bite here - but it is a live hazard for any other checkpoint
   and should be asserted before trusting a different model.
