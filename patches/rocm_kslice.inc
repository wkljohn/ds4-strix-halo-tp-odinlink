
/* ------------------------------------------------------------------------
 * DS4-TP-gfx1151 (patch 10): K-sliced Q8_0 matmul.
 *
 * Needed by ds4_gpu_attention_output_q8_tp_tensor, which fires at layer 0 of
 * the first token under TP=2. Upstream ships it as a silent `return 0` stub in
 * ds4_rocm_unavailable.cu, safe only while TP is gated off.
 *
 * NO NEW KERNEL IS REQUIRED. matmul_q8_0_f32_sharedx_warp_rows_w32_kernel
 * (rocm/ds4_rocm_q8.cuh:465) addresses weights as
 *     wr  = w + row * row_bytes
 *     blk = wr + b * 34
 * and loops b over n_blocks. So passing
 *     w         = base + block_start * 34    (per-row slice start)
 *     n_blocks  = slice_blocks               (only the slice)
 *     row_bytes = FULL row bytes             (stride unchanged)
 * makes it read exactly the K-slice of every row. The caller supplies an x
 * already offset to the slice, matching ds4_cuda.cu:27498-27505.
 *
 * DECODE ONLY. This uses the n_tok == 1 shared-x path; n_tok > 1 fails closed
 * with a message rather than silently computing the wrong thing. Prefill is
 * blocked by separate unavailable-stubs anyway.
 * ------------------------------------------------------------------------ */
extern "C" int ds4_gpu_matmul_q8_0_kslice_rows_tensor(
        ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
        uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim,
        uint64_t in_start, uint64_t in_count,
        const ds4_gpu_tensor *x, uint64_t n_tok) {
    if (!out || !x || !model_map || in_dim == 0u || out_dim == 0u ||
        in_count == 0u || n_tok == 0u) return 0;
    /* Q8_0 blocks are 32 elements; a slice that does not land on a block
     * boundary would silently mix neighbouring blocks' scales. Fail closed -
     * ds4_cuda.cu:15800 applies the identical guard. */
    if ((in_start % 32u) != 0u || (in_count % 32u) != 0u ||
        in_start > in_dim || in_count > in_dim - in_start) return 0;
    if (n_tok != 1u) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX
                "q8_0 kslice: only n_tok==1 (decode) implemented, got %llu\n",
                (unsigned long long)n_tok);
        return 0;
    }
    const uint64_t full_blocks  = (in_dim + 31u) / 32u;
    const uint64_t block_start  = in_start / 32u;
    const uint64_t slice_blocks = in_count / 32u;
    const uint64_t row_bytes    = full_blocks * 34u;
    uint64_t weight_bytes = 0;
    if (weight_offset > model_size ||
        !cuda_u64_mul_checked(out_dim, row_bytes, &weight_bytes) ||
        weight_bytes > model_size - weight_offset) return 0;
    if (x->bytes < in_count * sizeof(float) ||
        out->bytes < out_dim * sizeof(float)) return 0;
    /* The shared-x kernel stages the whole slice in LDS. */
    if (in_count > 8192u) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX
                "q8_0 kslice: slice %llu exceeds the shared-x LDS budget\n",
                (unsigned long long)in_count);
        return 0;
    }
    const char *wptr = cuda_model_range_ptr(model_map, weight_offset, weight_bytes,
                                            "q8_0_kslice");
    if (!wptr) return 0;

    const unsigned rows_per_block = 32u;
    const unsigned threads = rows_per_block * 32u;
    matmul_q8_0_f32_sharedx_warp_rows_w32_kernel<<<
            (unsigned)((out_dim + rows_per_block - 1u) / rows_per_block),
            threads,
            (size_t)in_count * sizeof(float)>>>(
            (float *)out->ptr,
            reinterpret_cast<const unsigned char *>(wptr) + block_start * 34u,
            (const float *)x->ptr,
            (uint32_t)slice_blocks,
            out_dim,
            row_bytes);
    return cuda_ok(cudaGetLastError(), "matmul_q8_0 kslice rows launch");
}

/* Thin wrapper: slices x by x_elem_off, then defers. Mirrors
 * ds4_cuda.cu:27483-27505. */
extern "C" int ds4_gpu_matmul_q8_0_kslice_tensor(
        ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
        uint64_t weight_offset, uint64_t full_in_dim, uint64_t k_off,
        uint64_t k_cnt, uint64_t out_dim, const ds4_gpu_tensor *x,
        uint64_t x_elem_off) {
    if (!x || x_elem_off > x->bytes / sizeof(float) ||
        k_cnt > x->bytes / sizeof(float) - x_elem_off) return 0;
    ds4_gpu_tensor x_slice = *x;
    x_slice.ptr = (char *)x->ptr + x_elem_off * sizeof(float);
    x_slice.bytes = k_cnt * sizeof(float);
    x_slice.owner = 0;
    return ds4_gpu_matmul_q8_0_kslice_rows_tensor(
            out, model_map, model_size, weight_offset,
            full_in_dim, out_dim, k_off, k_cnt, &x_slice, 1u);
}
