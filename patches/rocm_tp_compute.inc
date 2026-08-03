
/* ------------------------------------------------------------------------
 * DS4-TP-gfx1151 (patch 11): the last two TP compute entry points.
 *
 * Both fire at layer 0 of the first token under TP=2 and both shipped as
 * "tensor parallelism is Metal-only" stubs returning 0.
 * ------------------------------------------------------------------------ */

/* Grouped attention output under a head split. This is pure composition - the
 * CUDA version (ds4_cuda.cu:15775-15831) is a wrapper too, not a kernel:
 * validate, view the owned head range, project low, then K-sliced matmul.
 *
 * THE ALIGNMENT GUARD IS LOAD-BEARING. k_off/k_cnt must be 32-aligned or the
 * slice mixes neighbouring Q8_0 blocks' scales. Upstream fails CLOSED here
 * (ds4_cuda.cu:15800) rather than corrupting silently - and an uneven group
 * split is the documented trap, since a split off group boundaries mixes heads
 * across groups with consistent shapes and wrong numbers. Keep the guard. */
extern "C" int ds4_gpu_attention_output_q8_tp_tensor(
        ds4_gpu_tensor *out, ds4_gpu_tensor *low, const void *model_map,
        uint64_t model_size, uint64_t out_a_offset, uint64_t out_b_offset,
        uint64_t group_dim, uint64_t rank, uint32_t n_groups_total,
        uint32_t group0, uint32_t group_cnt, uint64_t out_dim,
        const ds4_gpu_tensor *heads) {
    if (!out || !low || !heads || !model_map ||
        group_dim == 0 || rank == 0 || n_groups_total == 0 ||
        group_cnt == 0 || group0 > n_groups_total ||
        group_cnt > n_groups_total - group0 || out_dim == 0) {
        return 0;
    }
    const uint64_t blocks_a     = (group_dim + 31u) / 32u;
    const uint64_t row_a_bytes  = blocks_a * 34u;
    const uint64_t low_dim_total = (uint64_t)n_groups_total * rank;
    const uint64_t k_off = (uint64_t)group0 * rank;
    const uint64_t k_cnt = (uint64_t)group_cnt * rank;
    if ((k_off % 32u) != 0 || (k_cnt % 32u) != 0) return 0;
    if (heads->bytes < (uint64_t)(group0 + group_cnt) * group_dim * sizeof(float) ||
        low->bytes < k_cnt * sizeof(float) ||
        out->bytes < out_dim * sizeof(float)) {
        return 0;
    }

    ds4_gpu_tensor heads_slice = *heads;
    heads_slice.ptr = (char *)heads->ptr + (uint64_t)group0 * group_dim * sizeof(float);
    heads_slice.bytes = (uint64_t)group_cnt * group_dim * sizeof(float);
    heads_slice.owner = 0;

    const uint64_t a_off = out_a_offset + (uint64_t)group0 * rank * row_a_bytes;
    return ds4_gpu_attention_output_low_q8_tensor(low, model_map, model_size,
                                                  a_off, group_dim, rank,
                                                  group_cnt, &heads_slice) &&
           ds4_gpu_matmul_q8_0_kslice_rows_tensor(out, model_map, model_size,
                                                  out_b_offset, low_dim_total,
                                                  out_dim, k_off, k_cnt, low, 1u);
}

/* Hyper-connection expand + add. ROCm already has hc_expand_kernel
 * (rocm/ds4_rocm_hc.cuh:74); the existing non-add caller
 * (rocm/ds4_rocm_hc_output_launch.cuh:248) passes block_out twice with
 * has_add = 0, so the add variant is the same launch with block_add in slot 3
 * and has_add = 1.
 *
 * NOTE the ROCm kernel takes 12 parameters, not the 14 the CUDA build passes -
 * they are different variants, so this is NOT a copy of the CUDA call. Using
 * cuda_hc_hc_token_count for the token count rather than CUDA's raw division,
 * matching the surrounding ROCm code. */
extern "C" int ds4_gpu_hc_expand_add_tensor(
        ds4_gpu_tensor *out_hc, const ds4_gpu_tensor *block_out,
        const ds4_gpu_tensor *block_add, const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *post, const ds4_gpu_tensor *comb,
        uint32_t n_embd, uint32_t n_hc) {
    uint64_t n_tokens64 = 0;
    if (!out_hc || !block_out || !block_add || !residual_hc || !post || !comb ||
        n_embd == 0 || n_hc == 0 ||
        !cuda_hc_hc_token_count(out_hc, n_embd, n_hc, &n_tokens64) ||
        n_tokens64 == 0 || n_tokens64 > UINT32_MAX) {
        return 0;
    }
    const uint32_t n_tokens = (uint32_t)n_tokens64;
    const uint64_t n_elem = (uint64_t)n_tokens * n_hc * n_embd;
    hc_expand_kernel<<<(unsigned)((n_elem + 255) / 256), 256>>>(
            (float *)out_hc->ptr,
            (const float *)block_out->ptr,
            (const float *)block_add->ptr,
            (const float *)residual_hc->ptr,
            (const float *)post->ptr,
            (const float *)comb->ptr,
            n_embd, n_hc, n_tokens,
            n_hc, (uint32_t)(n_hc * n_hc), 1);
    return cuda_ok(cudaGetLastError(), "hc_expand_add launch");
}
