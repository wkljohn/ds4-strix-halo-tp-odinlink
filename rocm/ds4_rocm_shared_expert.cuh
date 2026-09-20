extern "C" int ds4_gpu_swiglu_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *gate, const ds4_gpu_tensor *up, uint32_t n, float clamp, float weight) {
    if (!cuda_tensor_has_f32(out, n) || !cuda_tensor_has_f32(gate, n) || !cuda_tensor_has_f32(up, n)) return 0;
    if (n == 0u) return 1;
    swiglu_kernel<<<(n + 255) / 256, 256>>>((float *)out->ptr, (const float *)gate->ptr, (const float *)up->ptr, n, clamp, weight);
    return cuda_ok(cudaGetLastError(), "swiglu launch");
}

static int shared_gate_up_swiglu_sharedx_enabled(void) {
    static int enabled = -1;
    if (enabled < 0) {
        const char *env = getenv("DS4_ROCM_SHARED_GU_SWIGLU_FUSE");
        enabled = env && env[0] == '1' && env[1] == '\0';
    }
    return enabled;
}

/* Experimental M>1 paired WMMA path.  It is opt-in because its f16 staged
 * activation/weight arithmetic belongs to Lane B and must not silently alter
 * the established scalar/F32 production path. */
static int shared_gate_up_wmma_batch_enabled(void) {
    static int enabled = -1;
    if (enabled < 0) {
        const char *env = getenv("DS4_ROCM_SHARED_GU_WMMA_BATCH");
        enabled = env && strcmp(env, "1") == 0;
    }
    return enabled;
}

static uint32_t shared_gate_up_wmma_batch_tile(void) {
    const char *env = getenv("DS4_ROCM_SHARED_GU_WMMA_BATCH_TILE");
    if (env && strcmp(env, "256") == 0) return 256u;
    return 128u;
}

#if defined(DS4_ENABLE_TEST_HOOKS) && DS4_ENABLE_TEST_HOOKS
/* Component-only dense admission. Neither production dispatch nor the
 * existing kernel bodies change. The fixture independently checks GGUF types. */
extern "C" int ds4_gpu_test_glm5_dense_prefill(
        ds4_gpu_tensor *gate, ds4_gpu_tensor *up, ds4_gpu_tensor *mid,
        const void *model_map, uint64_t model_size, uint64_t gate_offset,
        uint64_t up_offset, const ds4_gpu_tensor *x, uint32_t rows,
        unsigned mode, unsigned store_gate_up) {
    constexpr uint64_t k = 4096u, n = 12288u, row_bytes = k / 32u * 34u;
    constexpr uint64_t weight_bytes = n * row_bytes;
    if (!model_map || (rows != 256u && rows != 1024u) || mode > 2u ||
        store_gate_up > 1u || g_quality_mode || cuda_runtime_config()->graph_dump ||
        !cuda_model_range_fits(model_size, gate_offset, weight_bytes) ||
        !cuda_model_range_fits(model_size, up_offset, weight_bytes) ||
        (gate_offset < up_offset + weight_bytes && up_offset < gate_offset + weight_bytes)) return 0;
    const ds4_gpu_tensor *buffers[] = {x, gate, up, mid};
    const uint64_t bytes[] = {rows * k * sizeof(float), rows * n * sizeof(float),
                             rows * n * sizeof(float), rows * n * sizeof(float)};
    for (unsigned i = 0; i < 4; ++i) {
        if (!cuda_tensor_has_bytes(buffers[i], bytes[i]) || !buffers[i]->ptr ||
            (uintptr_t)buffers[i]->ptr > UINTPTR_MAX - bytes[i]) return 0;
        for (unsigned j = 0; j < i; ++j) {
            const uintptr_t a = (uintptr_t)buffers[i]->ptr, b = (uintptr_t)buffers[j]->ptr;
            if (a < b + bytes[j] && b < a + bytes[i]) return 0;
        }
    }
    const char *wg = cuda_model_range_ptr(model_map, gate_offset, weight_bytes, "test_dense_gate_q8");
    const char *wu = cuda_model_range_ptr(model_map, up_offset, weight_bytes, "test_dense_up_q8");
    if (!wg || !wu) return 0;
    if (mode == 0u) {
        shared_gate_up_swiglu_q8_0_batch_sharedx_w32_kernel<16u,16u>
            <<<dim3(n / 32u, rows / 16u),1024u,16u * 16u * 32u * sizeof(float)>>>(
                (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                (const unsigned char *)wg, (const unsigned char *)wu, (const float *)x->ptr,
                k / 32u, n, rows, row_bytes, (int)store_gate_up, 10.0f);
    } else if (mode == 1u) {
        shared_gate_up_swiglu_q8_0_batch_wmma_kernel<128u,128u>
            <<<dim3(n / 128u, rows / 64u),256u>>>(
                (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                (const unsigned char *)wg, (const unsigned char *)wu, (const float *)x->ptr,
                rows, k, n, row_bytes, (int)store_gate_up, 10.0f);
    } else {
        // Explicit geometry for the separate-GEMM control: no cached selector
        // can silently substitute another kernel in this component experiment.
        matmul_q8_0_f32_batch_wmma_kernel<128u,128u>
            <<<dim3(n / 128u, rows / 64u),256u>>>(
                (float *)gate->ptr, (const unsigned char *)wg,
                (const float *)x->ptr, rows, k, n, row_bytes);
        if (!cuda_ok(cudaGetLastError(), "test dense gate launch")) return 0;
        matmul_q8_0_f32_batch_wmma_kernel<128u,128u>
            <<<dim3(n / 128u, rows / 64u),256u>>>(
                (float *)up->ptr, (const unsigned char *)wu,
                (const float *)x->ptr, rows, k, n, row_bytes);
        if (!cuda_ok(cudaGetLastError(), "test dense up launch")) return 0;
        return ds4_gpu_swiglu_tensor(mid, gate, up, rows * n, 10.0f, 1.0f);
    }
    return cuda_ok(cudaGetLastError(), "test dense prefill launch");
}
#endif

extern "C" int ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        float                   clamp) {
    if (!gate || !up || !mid || !model_map || !x ||
        in_dim == 0u || out_dim == 0u || in_dim > UINT32_MAX || out_dim > UINT32_MAX) {
        return 0;
    }
    const uint64_t blocks = (in_dim + 31u) / 32u;
    uint64_t row_bytes = 0;
    uint64_t weight_bytes = 0;
    uint64_t x_bytes = 0;
    uint64_t out_bytes = 0;
    if (!cuda_u64_mul_checked(blocks, 34u, &row_bytes) ||
        !cuda_u64_mul_checked(out_dim, row_bytes, &weight_bytes) ||
        !cuda_u64_mul3_checked(in_dim, 1u, sizeof(float), &x_bytes) ||
        !cuda_u64_mul3_checked(out_dim, 1u, sizeof(float), &out_bytes) ||
        !cuda_tensor_has_bytes(x, x_bytes) || !cuda_tensor_has_bytes(gate, out_bytes) ||
        !cuda_tensor_has_bytes(up, out_bytes) || !cuda_tensor_has_bytes(mid, out_bytes)) {
        return 0;
    }
    if (shared_gate_up_swiglu_sharedx_enabled() &&
        in_dim == 4096u && out_dim == 1024u && blocks == 128u &&
        cuda_model_range_fits(model_size, gate_offset, weight_bytes) &&
        cuda_model_range_fits(model_size, up_offset, weight_bytes)) {
        const char *wg = cuda_model_range_ptr(
            model_map, gate_offset, weight_bytes, "shared_gate_q8_fused_sharedx");
        const char *wu = cuda_model_range_ptr(
            model_map, up_offset, weight_bytes, "shared_up_q8_fused_sharedx");
        if (!wg || !wu) return 0;
        const int store_gate_up =
            (g_quality_mode || cuda_runtime_config()->graph_dump) ? 1 : 0;
        const unsigned rows_per_block = 32u;
        shared_gate_up_swiglu_q8_0_sharedx_rows_w32_kernel<<<
                (unsigned)((out_dim + rows_per_block - 1u) / rows_per_block),
                rows_per_block * 32u,
                (size_t)in_dim * sizeof(float)>>>(
                (float *)gate->ptr,
                (float *)up->ptr,
                (float *)mid->ptr,
                reinterpret_cast<const unsigned char *>(wg),
                reinterpret_cast<const unsigned char *>(wu),
                (const float *)x->ptr,
                (uint32_t)blocks,
                out_dim,
                row_bytes,
                store_gate_up,
                clamp);
        return cuda_ok(cudaGetLastError(),
                       "shared gate/up exact q8 fused swiglu launch");
    }
    if (in_dim == 4096u && (in_dim & 31u) == 0u &&
        cuda_model_range_fits(model_size, gate_offset, weight_bytes) &&
        cuda_model_range_fits(model_size, up_offset, weight_bytes) &&
        !cuda_runtime_config()->disable_shared_gate_up_fused_w32) {
        const char *wg = cuda_model_range_ptr(model_map, gate_offset, weight_bytes, "shared_gate_q8");
        const char *wu = cuda_model_range_ptr(model_map, up_offset, weight_bytes, "shared_up_q8");
        if (!wg || !wu) return 0;
        const int store_gate_up = (g_quality_mode || cuda_runtime_config()->graph_dump) ? 1 : 0;
        const unsigned rows_per_block = 32u;
        shared_gate_up_swiglu_q8_0_rows_w32_kernel<<<
                (unsigned)((out_dim + rows_per_block - 1u) / rows_per_block),
                rows_per_block * 32u>>>(
                (float *)gate->ptr,
                (float *)up->ptr,
                (float *)mid->ptr,
                reinterpret_cast<const unsigned char *>(wg),
                reinterpret_cast<const unsigned char *>(wu),
                (const float *)x->ptr,
                (uint32_t)blocks,
                out_dim,
                row_bytes,
                store_gate_up,
                clamp);
        return cuda_ok(cudaGetLastError(), "shared gate/up fused q8 launch");
    }
    return ds4_gpu_matmul_q8_0_pair_tensor(gate, up,
                                             model_map, model_size,
                                             gate_offset, up_offset,
                                             in_dim, out_dim, out_dim,
                                             x, 1) &&
           ds4_gpu_swiglu_tensor(mid, gate, up, (uint32_t)out_dim, clamp, 1.0f);
}

#if defined(DS4_ENABLE_TEST_HOOKS) && DS4_ENABLE_TEST_HOOKS
/* Exact-kernel test hook for measuring multi-stream overlap.  Production
 * builds do not contain this symbol.  The caller owns all stream dependencies
 * and must use disjoint output tensors. */
extern "C" int ds4_gpu_test_shared_gate_up_swiglu_q8_0_stream_tensor(
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        float                   clamp,
        void                   *stream_handle) {
    if (!gate || !up || !mid || !model_map || !x || !stream_handle ||
        in_dim != 4096u || out_dim != 1024u) {
        return 0;
    }
    const uint64_t blocks = in_dim / 32u;
    const uint64_t row_bytes = blocks * 34u;
    const uint64_t weight_bytes = out_dim * row_bytes;
    const uint64_t out_bytes = out_dim * sizeof(float);
    if (!cuda_model_range_fits(model_size, gate_offset, weight_bytes) ||
        !cuda_model_range_fits(model_size, up_offset, weight_bytes) ||
        !cuda_tensor_has_f32(x, in_dim) ||
        !cuda_tensor_has_bytes(gate, out_bytes) ||
        !cuda_tensor_has_bytes(up, out_bytes) ||
        !cuda_tensor_has_bytes(mid, out_bytes)) {
        return 0;
    }
    const char *wg = cuda_model_range_ptr(
        model_map, gate_offset, weight_bytes, "test_shared_gate_q8_stream");
    const char *wu = cuda_model_range_ptr(
        model_map, up_offset, weight_bytes, "test_shared_up_q8_stream");
    if (!wg || !wu) return 0;
    cudaStream_t stream = (cudaStream_t)stream_handle;
    const unsigned rows_per_block = 32u;
    const unsigned threads = rows_per_block * 32u;
    if (shared_gate_up_swiglu_sharedx_enabled()) {
        const int store_gate_up =
            (g_quality_mode || cuda_runtime_config()->graph_dump) ? 1 : 0;
        shared_gate_up_swiglu_q8_0_sharedx_rows_w32_kernel<<<
                (unsigned)((out_dim + rows_per_block - 1u) / rows_per_block),
                threads,
                (size_t)in_dim * sizeof(float),
                stream>>>(
                (float *)gate->ptr,
                (float *)up->ptr,
                (float *)mid->ptr,
                reinterpret_cast<const unsigned char *>(wg),
                reinterpret_cast<const unsigned char *>(wu),
                (const float *)x->ptr,
                (uint32_t)blocks,
                out_dim,
                row_bytes,
                store_gate_up,
                clamp);
        return cuda_ok(cudaGetLastError(),
                       "test shared fused gate/up/SwiGLU stream launch");
    }
    if (q8_decode_pair_dp4a_enabled()) return 0;
    matmul_q8_0_pair_f32_sharedx_warp_rows_w32_kernel<<<
            (unsigned)((out_dim + rows_per_block - 1u) / rows_per_block),
            threads,
            (size_t)in_dim * sizeof(float),
            stream>>>(
            (float *)gate->ptr,
            (float *)up->ptr,
            reinterpret_cast<const unsigned char *>(wg),
            reinterpret_cast<const unsigned char *>(wu),
            (const float *)x->ptr,
            (uint32_t)blocks,
            out_dim,
            out_dim,
            row_bytes);
    if (!cuda_ok(cudaGetLastError(),
                 "test shared gate/up exact q8 pair stream launch")) {
        return 0;
    }
    swiglu_kernel<<<
            (unsigned)((out_dim + 255u) / 256u),
            256u,
            0,
            stream>>>(
            (float *)mid->ptr,
            (const float *)gate->ptr,
            (const float *)up->ptr,
            (uint32_t)out_dim,
            clamp,
            1.0f);
    return cuda_ok(cudaGetLastError(),
                   "test shared SwiGLU exact stream launch");
}

extern "C" int ds4_gpu_test_shared_down_q8_0_stream_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        void                   *stream_handle) {
    if (!out || !model_map || !x || !stream_handle ||
        in_dim != 1024u || out_dim != 4096u) {
        return 0;
    }
    const uint64_t blocks = in_dim / 32u;
    const uint64_t row_bytes = blocks * 34u;
    const uint64_t weight_bytes = out_dim * row_bytes;
    if (!cuda_model_range_fits(model_size, weight_offset, weight_bytes) ||
        !cuda_tensor_has_f32(x, in_dim) ||
        !cuda_tensor_has_f32(out, out_dim)) {
        return 0;
    }
    const char *weights = cuda_model_range_ptr(
        model_map, weight_offset, weight_bytes, "test_shared_down_q8_stream");
    if (!weights) return 0;
    cudaStream_t stream = (cudaStream_t)stream_handle;
    const unsigned rows_per_block = 32u;
    const unsigned threads = rows_per_block * 32u;
    matmul_q8_0_f32_sharedx_warp_rows_w32_kernel<<<
            (unsigned)((out_dim + rows_per_block - 1u) / rows_per_block),
            threads,
            (size_t)in_dim * sizeof(float),
            stream>>>(
            (float *)out->ptr,
            reinterpret_cast<const unsigned char *>(weights),
            (const float *)x->ptr,
            (uint32_t)blocks,
            out_dim,
            row_bytes);
    return cuda_ok(cudaGetLastError(),
                   "test shared down exact q8 stream launch");
}
#endif

extern "C" int ds4_gpu_shared_gate_up_swiglu_q8_0_rows_scalar_tensor(
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok,
        float                   clamp) {
    (void)gate; (void)up; (void)mid; (void)model_map; (void)model_size;
    (void)gate_offset; (void)up_offset; (void)in_dim; (void)out_dim;
    (void)x; (void)n_tok; (void)clamp;
    return 0;
}

extern "C" int ds4_gpu_shared_gate_up_swiglu_q8_0_batch_tensor(
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok);

static int ds4_gpu_shared_gate_up_swiglu_q8_0_batch_clamp_tensor(
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok,
        float                   clamp);

extern "C" int ds4_gpu_shared_mid_swiglu_q8_0_tensor(
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        float                   clamp) {
    if (!mid || out_dim == 0u || out_dim > UINT32_MAX) return 0;
    uint64_t tmp_bytes = 0;
    if (!cuda_u64_mul3_checked(2u, out_dim, sizeof(float), &tmp_bytes)) return 0;
    void *tmp = cuda_tmp_alloc(tmp_bytes, "shared gate/up mid wrapper");
    if (!tmp) return 0;
    ds4_gpu_tensor gate_tmp = { tmp, out_dim * sizeof(float), 0 };
    ds4_gpu_tensor up_tmp = { (char *)tmp + out_dim * sizeof(float),
                              out_dim * sizeof(float),
                              0 };
    return ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(&gate_tmp,
                                                     &up_tmp,
                                                     mid,
                                                     model_map,
                                                     model_size,
                                                     gate_offset,
                                                     up_offset,
                                                     in_dim,
                                                     out_dim,
                                                     x,
                                                     clamp);
}

extern "C" int ds4_gpu_shared_gate_up_swiglu_q8_0_model_view_tensor(
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        float                   clamp) {
    return ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(gate,
                                                     up,
                                                     mid,
                                                     model_map,
                                                     model_size,
                                                     gate_offset,
                                                     up_offset,
                                                     in_dim,
                                                     out_dim,
                                                     x,
                                                     clamp);
}

extern "C" int ds4_gpu_shared_gate_up_swiglu_q8_0_rows_tensor(
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok,
        float                   clamp) {
    if (n_tok == 1u) {
        return ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(gate,
                                                         up,
                                                         mid,
                                                         model_map,
                                                         model_size,
                                                         gate_offset,
                                                         up_offset,
                                                         in_dim,
                                                         out_dim,
                                                         x,
                                                         clamp);
    }
    return ds4_gpu_shared_gate_up_swiglu_q8_0_batch_clamp_tensor(
        gate, up, mid, model_map, model_size, gate_offset, up_offset,
        in_dim, out_dim, x, n_tok, clamp);
}

static cudaStream_t g_shared_gate_up_stream = NULL;
static cudaEvent_t g_shared_gate_up_ready_event = NULL;
static void *g_shared_gate_up_tmp = NULL;
static uint64_t g_shared_gate_up_tmp_bytes = 0;
static int g_shared_gate_up_pending = 0;

static int cuda_shared_gate_up_async_wait_internal(void) {
    if (!g_shared_gate_up_pending) return 1;
    cudaError_t err = cudaStreamSynchronize(g_shared_gate_up_stream);
    g_shared_gate_up_pending = 0;
    if (err != cudaSuccess) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "shared gate/up async wait failed: %s\n", cudaGetErrorString(err));
        (void)cudaGetLastError();
        return 0;
    }
    return 1;
}

static void *cuda_shared_gate_up_async_tmp_alloc(uint64_t bytes) {
    if (bytes == 0) return NULL;
    if (g_shared_gate_up_tmp_bytes >= bytes) return g_shared_gate_up_tmp;
    if (g_shared_gate_up_tmp) {
        (void)cuda_shared_gate_up_async_wait_internal();
        (void)cudaFree(g_shared_gate_up_tmp);
        g_shared_gate_up_tmp = NULL;
        g_shared_gate_up_tmp_bytes = 0;
    }
    void *ptr = NULL;
    cudaError_t err = cudaMalloc(&ptr, (size_t)bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "shared gate/up async temp alloc failed (%.2f MiB): %s\n",
                (double)bytes / 1048576.0, cudaGetErrorString(err));
        (void)cudaGetLastError();
        return NULL;
    }
    g_shared_gate_up_tmp = ptr;
    g_shared_gate_up_tmp_bytes = bytes;
    return g_shared_gate_up_tmp;
}

static void cuda_shared_gate_up_async_cleanup(void) {
    if (g_shared_gate_up_stream) {
        (void)cuda_shared_gate_up_async_wait_internal();
    }
    if (g_shared_gate_up_tmp) {
        (void)cudaFree(g_shared_gate_up_tmp);
        g_shared_gate_up_tmp = NULL;
        g_shared_gate_up_tmp_bytes = 0;
    }
    if (g_shared_gate_up_ready_event) {
        (void)cudaEventDestroy(g_shared_gate_up_ready_event);
        g_shared_gate_up_ready_event = NULL;
    }
    if (g_shared_gate_up_stream) {
        (void)cudaStreamDestroy(g_shared_gate_up_stream);
        g_shared_gate_up_stream = NULL;
    }
}

extern "C" int ds4_gpu_shared_gate_up_swiglu_q8_0_async_tensor(
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        float                   clamp) {
    if (g_quality_mode || cuda_runtime_config()->graph_dump) return 0;
    if (g_shared_gate_up_pending && !cuda_shared_gate_up_async_wait_internal()) return 0;
    if (!gate || !up || !mid || !model_map || !x ||
        in_dim == 0u || out_dim == 0u || in_dim > UINT32_MAX || out_dim > UINT32_MAX) {
        return 0;
    }
    const uint64_t blocks = (in_dim + 31u) / 32u;
    uint64_t row_bytes = 0;
    uint64_t weight_bytes = 0;
    if (!cuda_u64_mul_checked(blocks, 34u, &row_bytes) ||
        !cuda_u64_mul_checked(out_dim, row_bytes, &weight_bytes)) {
        return 0;
    }
    if (g_quality_mode ||
        !gate || !up || !mid || !model_map || !x ||
        in_dim == 0u || out_dim == 0u || in_dim > UINT32_MAX || out_dim > UINT32_MAX ||
        gate_offset > model_size || up_offset > model_size ||
        weight_bytes > model_size - gate_offset ||
        weight_bytes > model_size - up_offset ||
        x->bytes < in_dim * sizeof(float) ||
        gate->bytes < out_dim * sizeof(float) ||
        up->bytes < out_dim * sizeof(float) ||
        mid->bytes < out_dim * sizeof(float)) {
        return 0;
    }
    const char *wg = cuda_model_range_ptr(model_map, gate_offset, weight_bytes, "shared_gate_q8_pair_async");
    const char *wu = cuda_model_range_ptr(model_map, up_offset, weight_bytes, "shared_up_q8_pair_async");
    if (!wg || !wu) return 0;
    if (!g_shared_gate_up_stream) {
        int least_priority = 0;
        int greatest_priority = 0;
#ifdef __HIP_PLATFORM_AMD__
        hipError_t err = hipDeviceGetStreamPriorityRange(&least_priority, &greatest_priority);
        if (err == hipSuccess) {
            err = hipStreamCreateWithPriority(&g_shared_gate_up_stream, cudaStreamNonBlocking, least_priority);
        } else {
            (void)cudaGetLastError();
            err = hipStreamCreateWithFlags(&g_shared_gate_up_stream, cudaStreamNonBlocking);
        }
        if (err != hipSuccess) return 0;
#else
        cudaError_t err = cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority);
        if (err == cudaSuccess) {
            err = cudaStreamCreateWithPriority(&g_shared_gate_up_stream, cudaStreamNonBlocking, least_priority);
        } else {
            (void)cudaGetLastError();
            err = cudaStreamCreateWithFlags(&g_shared_gate_up_stream, cudaStreamNonBlocking);
        }
        if (err != cudaSuccess) return 0;
#endif
    }
    if (!g_shared_gate_up_ready_event) {
        cudaError_t err = cudaEventCreateWithFlags(&g_shared_gate_up_ready_event, cudaEventDisableTiming);
        if (err != cudaSuccess) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX "shared gate/up async event create failed: %s\n", cudaGetErrorString(err));
            (void)cudaGetLastError();
            return 0;
        }
    }
    /*
     * This stream is intentionally non-blocking so it can overlap routed MoE.
     * Non-blocking streams do not inherit default-stream ordering, so explicitly
     * wait until the default-stream producer of x (ffn_norm) has completed before
     * quantizing it here.
     */
    cudaError_t dep_err = cudaEventRecord(g_shared_gate_up_ready_event, 0);
    if (dep_err != cudaSuccess) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "shared gate/up async dependency record failed: %s\n", cudaGetErrorString(dep_err));
        (void)cudaGetLastError();
        return 0;
    }
#ifdef __HIP_PLATFORM_AMD__
    dep_err = hipStreamWaitEvent(g_shared_gate_up_stream, g_shared_gate_up_ready_event, 0);
#else
    dep_err = cudaStreamWaitEvent(g_shared_gate_up_stream, g_shared_gate_up_ready_event, 0);
#endif
    if (dep_err != cudaSuccess) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "shared gate/up async dependency wait failed: %s\n", cudaGetErrorString(dep_err));
        (void)cudaGetLastError();
        return 0;
    }
    const uint64_t xq_bytes = blocks * 32u;
    const uint64_t scale_offset = (xq_bytes + 15u) & ~15ull;
    const uint64_t tmp_bytes = scale_offset + blocks * sizeof(float);
    void *tmp = cuda_shared_gate_up_async_tmp_alloc(tmp_bytes);
    if (!tmp) return 0;
    int8_t *xq = (int8_t *)tmp;
    float *xscale = (float *)((char *)tmp + scale_offset);
    const int use_dp4a = 1;
    dim3 qgrid((unsigned)blocks, 1, 1);
    quantize_q8_0_f32_kernel<<<qgrid, 32, 0, g_shared_gate_up_stream>>>(xq, xscale, (const float *)x->ptr, in_dim, blocks);
    if (!cuda_ok(cudaGetLastError(), "shared gate/up async quantize launch")) return 0;
    matmul_q8_0_pair_preq_warp8_kernel<<<((unsigned)out_dim + 7u) / 8u, 256, 0, g_shared_gate_up_stream>>>(
            (float *)gate->ptr,
            (float *)up->ptr,
            reinterpret_cast<const unsigned char *>(wg),
            reinterpret_cast<const unsigned char *>(wu),
            xq,
            xscale,
            in_dim,
            out_dim,
            out_dim,
            blocks,
            use_dp4a);
    if (!cuda_ok(cudaGetLastError(), "shared gate/up async pair launch")) return 0;
    swiglu_kernel<<<((unsigned)out_dim + 255u) / 256u, 256, 0, g_shared_gate_up_stream>>>(
            (float *)mid->ptr,
            (const float *)gate->ptr,
            (const float *)up->ptr,
            (uint32_t)out_dim,
            clamp,
            1.0f);
    if (!cuda_ok(cudaGetLastError(), "shared gate/up async swiglu launch")) return 0;
    g_shared_gate_up_pending = 1;
    return 1;
}

extern "C" int ds4_gpu_shared_gate_up_async_wait(void) {
    return cuda_shared_gate_up_async_wait_internal();
}

extern "C" int ds4_gpu_shared_gate_up_swiglu_q8_0_batch_tensor(
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok) {
    return ds4_gpu_shared_gate_up_swiglu_q8_0_batch_clamp_tensor(
        gate, up, mid, model_map, model_size, gate_offset, up_offset,
        in_dim, out_dim, x, n_tok, 0.0f);
}

static int ds4_gpu_shared_gate_up_swiglu_q8_0_batch_clamp_tensor(
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok,
        float                   clamp) {
    uint64_t x_bytes = 0, out_bytes = 0;
    if (!gate || !up || !mid || !model_map || !x || n_tok == 0 ||
        (in_dim & 31u) != 0u || in_dim == 0u || out_dim == 0u ||
        in_dim > UINT32_MAX || out_dim > UINT32_MAX || n_tok > UINT32_MAX ||
        !cuda_u64_mul3_checked(n_tok, in_dim, sizeof(float), &x_bytes) ||
        !cuda_u64_mul3_checked(n_tok, out_dim, sizeof(float), &out_bytes) ||
        x->bytes < x_bytes || gate->bytes < out_bytes || up->bytes < out_bytes || mid->bytes < out_bytes) {
        return 0;
    }
    const uint64_t blocks = (in_dim + 31u) / 32u;
    uint64_t row_bytes = 0, weight_bytes = 0;
    if (!cuda_u64_mul_checked(blocks, 34u, &row_bytes) ||
        !cuda_u64_mul_checked(out_dim, row_bytes, &weight_bytes)) return 0;
    if (gate_offset > model_size || up_offset > model_size ||
        weight_bytes > model_size - gate_offset || weight_bytes > model_size - up_offset) {
        return 0;
    }
    const char *wg = cuda_model_range_ptr(model_map, gate_offset, weight_bytes, "shared_gate_q8_batch");
    const char *wu = cuda_model_range_ptr(model_map, up_offset, weight_bytes, "shared_up_q8_batch");
    if (!wg || !wu) return 0;

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
    if (!g_quality_mode && !cuda_runtime_config()->graph_dump &&
        shared_gate_up_wmma_batch_enabled() && n_tok >= 64u &&
        in_dim == 4096u && out_dim == 1024u) {
        const uint32_t m_tile = shared_gate_up_wmma_batch_tile();
        const dim3 grid((uint32_t)((out_dim + m_tile - 1u) / m_tile),
                        (uint32_t)((n_tok + 63u) / 64u), 1u);
        const int store_gate_up = 0;
        if (m_tile == 256u) {
            shared_gate_up_swiglu_q8_0_batch_wmma_kernel<256u, 128u>
                <<<grid, 512u>>>(
                    (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                    reinterpret_cast<const unsigned char *>(wg),
                    reinterpret_cast<const unsigned char *>(wu),
                    (const float *)x->ptr, (uint32_t)n_tok, (uint32_t)in_dim,
                    (uint32_t)out_dim, row_bytes, store_gate_up, clamp);
        } else {
            shared_gate_up_swiglu_q8_0_batch_wmma_kernel<128u, 128u>
                <<<grid, 256u>>>(
                    (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                    reinterpret_cast<const unsigned char *>(wg),
                    reinterpret_cast<const unsigned char *>(wu),
                    (const float *)x->ptr, (uint32_t)n_tok, (uint32_t)in_dim,
                    (uint32_t)out_dim, row_bytes, store_gate_up, clamp);
        }
        return cuda_ok(cudaGetLastError(),
                       "shared gate/up fused q8 WMMA batch launch");
    }
#endif

    const uint32_t rows_per_block = 32u;
    const uint32_t tile = 16u;
    const uint32_t block_tile = 16u;
    const dim3 grid((uint32_t)((out_dim + rows_per_block - 1u) / rows_per_block),
                    (uint32_t)((n_tok + tile - 1u) / tile),
                    1u);
    const size_t shmem = (size_t)tile * block_tile * 32u * sizeof(float);
    const int store_gate_up = (g_quality_mode || cuda_runtime_config()->graph_dump) ? 1 : 0;
#define DS4_LAUNCH_SHARED_GU_BATCH(TT, BT) \
    shared_gate_up_swiglu_q8_0_batch_sharedx_w32_kernel<TT, BT><<<grid, rows_per_block * 32u, shmem>>>( \
            (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr, \
            reinterpret_cast<const unsigned char *>(wg), reinterpret_cast<const unsigned char *>(wu), \
            (const float *)x->ptr, (uint32_t)blocks, (uint32_t)out_dim, (uint32_t)n_tok, row_bytes, store_gate_up, clamp)
    DS4_LAUNCH_SHARED_GU_BATCH(16u, 16u);
#undef DS4_LAUNCH_SHARED_GU_BATCH
    return cuda_ok(cudaGetLastError(), "shared gate/up fused q8 batch launch");
}
