template <uint32_t BT>
static void cuda_launch_q8_batch_sharedx_bt(
        float *out,
        const unsigned char *w,
        const float *x,
        uint32_t n_blocks,
        uint32_t out_dim,
        uint32_t n_tok,
        uint64_t row_bytes,
        dim3 grid,
        uint32_t rows_per_block,
        uint32_t tile) {
    const size_t shmem = (size_t)tile * BT * 32u * sizeof(float);
    if (tile == 2u) {
        matmul_q8_0_f32_batch_sharedx_warp_rows_w32_toktile_kernel<2u, BT><<<grid, rows_per_block * 32u, shmem>>>(out, w, x, n_blocks, out_dim, n_tok, row_bytes);
    } else if (tile == 4u) {
        matmul_q8_0_f32_batch_sharedx_warp_rows_w32_toktile_kernel<4u, BT><<<grid, rows_per_block * 32u, shmem>>>(out, w, x, n_blocks, out_dim, n_tok, row_bytes);
    } else if (tile == 8u) {
        matmul_q8_0_f32_batch_sharedx_warp_rows_w32_toktile_kernel<8u, BT><<<grid, rows_per_block * 32u, shmem>>>(out, w, x, n_blocks, out_dim, n_tok, row_bytes);
    } else if (tile == 16u) {
        matmul_q8_0_f32_batch_sharedx_warp_rows_w32_toktile_kernel<16u, BT><<<grid, rows_per_block * 32u, shmem>>>(out, w, x, n_blocks, out_dim, n_tok, row_bytes);
    } else {
        matmul_q8_0_f32_batch_sharedx_warp_rows_w32_toktile_kernel<32u, BT><<<grid, rows_per_block * 32u, shmem>>>(out, w, x, n_blocks, out_dim, n_tok, row_bytes);
    }
}

/* Cache-free gfx1151 prefill tile. M=128 lets eight waves reuse the same
 * 64-token activation tile across twice as many output rows. */
static int q8_batch_wmma_m128_enabled(void) {
    static int enabled = -1;
    static int gfx1151;
    if (enabled < 0) {
        const char *value = getenv("DS4_ROCM_Q8_BATCH_WMMA_M128");
        const char *disable = getenv("DS4_ROCM_DISABLE_Q8_BATCH_WMMA_M128");
        enabled = (value == NULL || strcmp(value, "0") != 0) &&
                  !(disable != NULL && strcmp(disable, "1") == 0);
        int device = 0;
        cudaDeviceProp prop;
        memset(&prop, 0, sizeof(prop));
        if (cudaGetDevice(&device) == cudaSuccess &&
            cudaGetDeviceProperties(&prop, device) == cudaSuccess) {
            gfx1151 = strncmp(prop.gcnArchName, "gfx1151", 7) == 0;
        } else {
            (void)cudaGetLastError();
        }
    }
    return enabled && gfx1151;
}

/* Research candidate: group four Q8 blocks under one activation staging
 * barrier.  The 136-half LDS stride is essential on gfx1151; the natural
 * 128-half stride aliases token rows onto one bank phase. */
static int q8_batch_wmma_k128_padded_enabled(void) {
    static int enabled = -1;
    if (enabled < 0) {
        const char *value = getenv("DS4_ROCM_Q8_BATCH_WMMA_K128_PADDED");
        const char *disable = getenv(
                "DS4_ROCM_DISABLE_Q8_BATCH_WMMA_K128_PADDED");
        enabled = (value == NULL || strcmp(value, "0") != 0) &&
                  !(disable != NULL && strcmp(disable, "1") == 0);
    }
    return enabled;
}

/* Opt-in research tile: sixteen waves reuse the padded K=128 activation
 * stage across 256 output rows. Keep it separate from the production M=128
 * default until full-model Q4/Q2 and both-provider gates pass. */
static int q8_batch_wmma_m256_k128_enabled(void) {
    static int enabled = -1;
    if (enabled < 0) {
        const char *value = getenv("DS4_ROCM_Q8_BATCH_WMMA_M256_K128");
        enabled = value != NULL && strcmp(value, "1") == 0;
    }
    return enabled;
}

static unsigned attention_output_expand_threads(void) {
    static unsigned threads;
    if (threads == 0u) {
        const char *value = getenv("DS4_ROCM_ATTN_OUT_EXPAND_THREADS");
        threads = 1024u;
        if (value && strcmp(value, "1024") != 0) {
            if (strcmp(value, "512") == 0) threads = 512u;
            else if (strcmp(value, "256") == 0) threads = 256u;
            else fprintf(stderr, DS4_GPU_LOG_PREFIX
                         "unsupported DS4_ROCM_ATTN_OUT_EXPAND_THREADS=%s; using 1024\n",
                         value);
        }
    }
    return threads;
}

static int attention_output_expand_pack4_enabled(void) {
    static int enabled = -1;
    static int gfx1151;
    if (enabled < 0) {
        const char *value = getenv("DS4_ROCM_ATTN_OUT_EXPAND_PACK4");
        const char *disable = getenv("DS4_ROCM_DISABLE_ATTN_OUT_EXPAND_PACK4");
        enabled = (value == NULL || strcmp(value, "0") != 0) &&
                  !(disable != NULL && strcmp(disable, "1") == 0);
        int device = 0;
        cudaDeviceProp prop;
        memset(&prop, 0, sizeof(prop));
        if (cudaGetDevice(&device) == cudaSuccess &&
            cudaGetDeviceProperties(&prop, device) == cudaSuccess) {
            gfx1151 = strncmp(prop.gcnArchName, "gfx1151", 7) == 0;
        } else {
            (void)cudaGetLastError();
        }
    }
    return enabled && gfx1151;
}

static int attention_q_b_pack4_enabled(void) {
    static int enabled = -1;
    static int gfx1151;
    if (enabled < 0) {
        const char *value = getenv("DS4_ROCM_ATTN_Q_B_PACK4");
        const char *disable = getenv("DS4_ROCM_DISABLE_ATTN_Q_B_PACK4");
        enabled = (value == NULL || strcmp(value, "0") != 0) &&
                  !(disable != NULL && strcmp(disable, "1") == 0);
        int device = 0;
        cudaDeviceProp prop;
        memset(&prop, 0, sizeof(prop));
        if (cudaGetDevice(&device) == cudaSuccess &&
            cudaGetDeviceProperties(&prop, device) == cudaSuccess) {
            gfx1151 = strncmp(prop.gcnArchName, "gfx1151", 7) == 0;
        } else {
            (void)cudaGetLastError();
        }
    }
    return enabled && gfx1151;
}

static int attention_q_b_qnorm_rope_enabled(void) {
    static int enabled = -1;
    if (enabled < 0) {
        const char *value = getenv("DS4_ROCM_ATTENTION_Q_B_QNORM_ROPE_FUSE");
        enabled = value && strcmp(value, "1") == 0;
    }
    return enabled;
}

static int q8_pair_f32_pack4_enabled(void) {
    static int enabled = -1;
    static int gfx1151;
    if (enabled < 0) {
        const char *value = getenv("DS4_ROCM_Q8_PAIR_F32_PACK4");
        /* Explicit 0 is a testable fallback without latching the one-time
         * dispatch cache, so the focused test can compare both kernels. */
        if (value && strcmp(value, "0") == 0) return 0;
        /* This is the exact compact-weight gfx1151 decode path. It does not
         * allocate a repacked or expanded weight cache. */
        enabled = !value || strcmp(value, "1") == 0;
        int device = 0;
        cudaDeviceProp prop;
        memset(&prop, 0, sizeof(prop));
        if (cudaGetDevice(&device) == cudaSuccess &&
            cudaGetDeviceProperties(&prop, device) == cudaSuccess) {
            gfx1151 = strncmp(prop.gcnArchName, "gfx1151", 7) == 0;
        } else {
            (void)cudaGetLastError();
        }
    }
    return enabled && gfx1151;
}

/* Diagnostic-only sweep for the decode compressor_proj kernel
 * (matmul_f16_pair_f32_sharedx_warp_rows_w32_kernel), which measured as
 * the second-largest decode attention sub-stage (~131us). Pure launch-
 * configuration change - the kernel already derives rows_per_block from
 * blockDim.x (ds4_rocm_common.cuh:195), so this has zero effect on math.
 * The equivalent sweep for the larger attn_output kernels measured no
 * win (DECODE-ACCELERATION-PLAN.md), so treat this purely as a cheap
 * experiment, not an expected win. */
static unsigned compressor_proj_rows_per_block(void) {
    static unsigned rows;
    if (rows == 0u) {
        const char *value = getenv("DS4_ROCM_COMPRESSOR_PROJ_ROWS");
        rows = 32u;
        if (value && strcmp(value, "32") != 0) {
            if (strcmp(value, "16") == 0) rows = 16u;
            else if (strcmp(value, "8") == 0) rows = 8u;
            else fprintf(stderr, DS4_GPU_LOG_PREFIX
                         "unsupported DS4_ROCM_COMPRESSOR_PROJ_ROWS=%s; using 32\n",
                         value);
        }
    }
    return rows;
}

static int f16_pair_temporal_supported(void) {
    static int initialized;
    static int gfx1151;
    if (!initialized) {
        initialized = 1;
        int device = 0;
        cudaDeviceProp prop;
        memset(&prop, 0, sizeof(prop));
        if (cudaGetDevice(&device) == cudaSuccess &&
            cudaGetDeviceProperties(&prop, device) == cudaSuccess) {
            gfx1151 = strncmp(prop.gcnArchName, "gfx1151", 7) == 0;
        } else {
            (void)cudaGetLastError();
        }
    }
    return gfx1151 && !g_quality_mode && !cuda_runtime_config()->graph_dump;
}

extern "C" int ds4_gpu_f16_pair_temporal_supported(void) {
    return f16_pair_temporal_supported();
}

static int f16_pair_five_row_enabled(void) {
    static int enabled = -1;
    static int gfx1151;
    if (enabled < 0) {
        const char *disable = getenv("DS4_ROCM_DISABLE_F16_PAIR_FIVE_ROW");
        enabled = !(disable && strcmp(disable, "1") == 0);
        int device = 0;
        cudaDeviceProp prop;
        memset(&prop, 0, sizeof(prop));
        if (cudaGetDevice(&device) == cudaSuccess &&
            cudaGetDeviceProperties(&prop, device) == cudaSuccess) {
            gfx1151 = strncmp(prop.gcnArchName, "gfx1151", 7) == 0;
        } else {
            (void)cudaGetLastError();
        }
    }
    return enabled && gfx1151;
}

static int f16_hc_five_row_enabled(void) {
    static int enabled = -1;
    static int gfx1151;
    if (enabled < 0) {
        const char *disable = getenv("DS4_ROCM_DISABLE_F16_HC_FIVE_ROW");
        enabled = !(disable && strcmp(disable, "1") == 0);
        int device = 0;
        cudaDeviceProp prop;
        memset(&prop, 0, sizeof(prop));
        if (cudaGetDevice(&device) == cudaSuccess &&
            cudaGetDeviceProperties(&prop, device) == cudaSuccess) {
            gfx1151 = strncmp(prop.gcnArchName, "gfx1151", 7) == 0;
        } else {
            (void)cudaGetLastError();
        }
    }
    return enabled && gfx1151;
}

static int f16_hc_five_row_mode(void) {
    if (!f16_hc_five_row_enabled()) return 0;
    static int mode = -1;
    if (mode < 0) {
        const char *value = getenv("DS4_ROCM_F16_HC_FIVE_ROW_MODE");
        /* Independent token reductions were both slightly faster in the
         * cold-weight oracle and materially more stable for DSpark acceptance
         * than cross-token accumulation. Keep reuse as an experiment only. */
        mode = 2;
        if (value && strcmp(value, "reuse") == 0) {
            mode = 1;
        } else if (value && strcmp(value, "token") != 0) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX
                    "unsupported DS4_ROCM_F16_HC_FIVE_ROW_MODE=%s; using token\n",
                    value);
        }
    }
    return mode;
}

static int f16_indexer_q_wvsplit_cus(void) {
#if defined(__HIP_PLATFORM_AMD__)
    static int cus = -1;
    if (cus < 0) {
        cus = 0;
        const char *enable =
            getenv("DS4_ROCM_ENABLE_F16_INDEXER_Q_WVSPLIT");
        if (!enable || strcmp(enable, "1") != 0) return cus;
        const char *disable =
            getenv("DS4_ROCM_DISABLE_F16_INDEXER_Q_WVSPLIT");
        if (disable && strcmp(disable, "1") == 0) return cus;
        int device = 0;
        cudaDeviceProp prop;
        memset(&prop, 0, sizeof(prop));
        if (cudaGetDevice(&device) == cudaSuccess &&
            cudaGetDeviceProperties(&prop, device) == cudaSuccess &&
            strncmp(prop.gcnArchName, "gfx1151", 7) == 0) {
            cus = prop.multiProcessorCount;
        } else {
            (void)cudaGetLastError();
        }
    }
    return cus;
#else
    return 0;
#endif
}

static void cuda_launch_q8_batch_sharedx(
        float *out,
        const unsigned char *w,
        const float *x,
        uint32_t n_blocks,
        uint32_t out_dim,
        uint32_t n_tok,
        uint64_t row_bytes,
        uint32_t rows_per_block,
        uint32_t tile,
        uint32_t block_tile) {
    const dim3 grid((out_dim + rows_per_block - 1u) / rows_per_block,
                    (n_tok + tile - 1u) / tile,
                    1u);
    if (block_tile == 8u) {
        cuda_launch_q8_batch_sharedx_bt<8u>(out, w, x, n_blocks, out_dim, n_tok, row_bytes, grid, rows_per_block, tile);
    } else if (block_tile == 32u) {
        cuda_launch_q8_batch_sharedx_bt<32u>(out, w, x, n_blocks, out_dim, n_tok, row_bytes, grid, rows_per_block, tile);
    } else {
        cuda_launch_q8_batch_sharedx_bt<16u>(out, w, x, n_blocks, out_dim, n_tok, row_bytes, grid, rows_per_block, tile);
    }
}

/* The token-tiled Q8 batch kernel has compiled shapes for 2/4/8/16/32 rows.
 * Normal prefill benefits from tile 32, but a DSpark verifier has only 2--5
 * rows: reserving and filling a 32-row LDS tile wastes most of the 64 KiB and
 * suppresses occupancy. Keep the established dispatch as the default while
 * the exact small-row shapes are measured independently. */
static uint32_t q8_batch_token_tile(uint64_t n_tok) {
    const char *env = getenv("DS4_ROCM_Q8_SMALL_BATCH_TILE");
    const bool enabled = env && env[0] == '1' && env[1] == '\0';
    if (!enabled || n_tok > 8u) return 32u;
    if (n_tok <= 2u) return 2u;
    if (n_tok <= 4u) return 4u;
    return 8u;
}

static uint32_t q8_small_batch_block_tile(uint64_t n_tok) {
    if (n_tok > 8u) return 16u;
    const char *env = getenv("DS4_ROCM_Q8_SMALL_BATCH_BLOCK_TILE");
    if (!env || !env[0] || strcmp(env, "16") == 0) return 16u;
    if (strcmp(env, "8") == 0) return 8u;
    if (strcmp(env, "32") == 0) return 32u;
    static int warned = 0;
    if (!warned) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX
                "unsupported DS4_ROCM_Q8_SMALL_BATCH_BLOCK_TILE=%s; using 16\n",
                env);
        warned = 1;
    }
    return 16u;
}

static int q8_small_batch_dp4a_enabled(void) {
    const char *env = getenv("DS4_ROCM_Q8_SMALL_BATCH_DP4A");
    return env && env[0] == '1' && env[1] == '\0';
}

static int q8_reuse_quant_enabled(void) {
    static int enabled = -1;
    if (enabled < 0) {
        const char *env = getenv("DS4_ROCM_Q8_REUSE_QUANT");
        enabled = env && env[0] == '1' && env[1] == '\0';
    }
    return enabled;
}

/* Speculative verification must agree numerically with the committed target
 * path.  The one-token paired-DP4A projection is intentionally approximate;
 * using it only for committed decode reduced measured DSpark acceptance from
 * 51.43% to 29.83%.  The ROCm graph marks speculative processes before the
 * first launch.  A research-only override keeps controlled A/B possible. */
static int g_q8_decode_pair_dp4a_speculative = 0;

static int q8_decode_pair_dp4a_enabled(void) {
    if (g_q8_decode_pair_dp4a_speculative) {
        const char *allow =
            getenv("DS4_ROCM_Q8_DECODE_PAIR_DP4A_SPECULATIVE");
        if (!allow || allow[0] != '1' || allow[1] != '\0') return 0;
    }
    static int enabled = -1;
    if (enabled < 0) {
        const char *env = getenv("DS4_ROCM_Q8_DECODE_PAIR_DP4A");
        enabled = env && env[0] == '1' && env[1] == '\0';
    }
    return enabled;
}

static int q8_dp4a_shape_dispatch_enabled(void) {
    static int enabled = -1;
    if (enabled < 0) {
        const char *env = getenv("DS4_ROCM_Q8_DP4A_SHAPE_DISPATCH");
        enabled = env && env[0] == '1' && env[1] == '\0';
    }
    return enabled;
}

static uint32_t q8_small_batch_dp4a_rows_per_block(
        uint32_t n_blocks, uint32_t out_dim) {
    if (!q8_dp4a_shape_dispatch_enabled()) return 8u;

    const uint32_t in_dim = n_blocks * 32u;
    /* Standalone gfx1151 sweeps selected 16 waves for these exact DFlash
     * projection shapes.  Q-B (1024 -> 32768, or 16384 after TP slicing)
     * was flat/slightly better at the established 8-wave launch, as were
     * shapes not covered by the sweep. */
    if ((in_dim == 4096u && out_dim == 8192u) ||
        (in_dim == 8192u && out_dim == 4096u) ||
        (in_dim == 12288u && out_dim == 4096u)) {
        return 16u;
    }
    return 8u;
}

static void cuda_launch_q8_small_batch_dp4a(
        float *out, const unsigned char *w, const int8_t *xq,
        const float *xscale, uint32_t n_blocks, uint32_t out_dim,
        uint32_t n_tok, uint64_t row_bytes) {
    /* 8 waves / 256 threads is the proven default.  A global 16-wave launch
     * was unstable end-to-end (11.86-15.03 t/s), so the opt-in dispatcher
     * limits 16 waves to shapes where the standalone sweep selected it.  The
     * kernel derives rows_per_block from blockDim, so grid and threads change
     * together. */
    const uint32_t rows_per_block =
        q8_small_batch_dp4a_rows_per_block(n_blocks, out_dim);
    const uint32_t threads = rows_per_block * 32u;
    const dim3 grid((out_dim + rows_per_block - 1u) / rows_per_block);
    if (n_tok <= 2u) {
        matmul_q8_0_preq_toktile_dp4a_kernel<2u><<<grid, threads>>>(
                out, w, xq, xscale, n_blocks, out_dim, n_tok, row_bytes);
    } else if (n_tok == 3u) {
        matmul_q8_0_preq_toktile_dp4a_kernel<3u><<<grid, threads>>>(
                out, w, xq, xscale, n_blocks, out_dim, n_tok, row_bytes);
    } else if (n_tok <= 4u) {
        matmul_q8_0_preq_toktile_dp4a_kernel<4u><<<grid, threads>>>(
                out, w, xq, xscale, n_blocks, out_dim, n_tok, row_bytes);
    } else if (n_tok == 5u) {
        matmul_q8_0_preq_toktile_dp4a_kernel<5u><<<grid, threads>>>(
                out, w, xq, xscale, n_blocks, out_dim, n_tok, row_bytes);
    } else {
        matmul_q8_0_preq_toktile_dp4a_kernel<8u><<<grid, threads>>>(
                out, w, xq, xscale, n_blocks, out_dim, n_tok, row_bytes);
    }
}

template <uint32_t BT>
static void cuda_launch_grouped_q8_a_sharedx_bt(
        float *low,
        const unsigned char *w,
        const float *heads,
        uint32_t n_tokens,
        uint32_t n_groups,
        uint32_t n_blocks,
        uint32_t rank,
        uint64_t row_bytes,
        dim3 grid,
        uint32_t rows_per_block,
        uint32_t tile) {
    const size_t shmem = (size_t)tile * BT * 32u * sizeof(float);
    if (tile == 2u) {
        grouped_q8_0_a_f32_batch_sharedx_chunked_w32_kernel<2u, BT><<<grid, rows_per_block * 32u, shmem>>>(low, w, heads, n_tokens, n_groups, n_blocks, rank, row_bytes);
    } else if (tile == 4u) {
        grouped_q8_0_a_f32_batch_sharedx_chunked_w32_kernel<4u, BT><<<grid, rows_per_block * 32u, shmem>>>(low, w, heads, n_tokens, n_groups, n_blocks, rank, row_bytes);
    } else if (tile == 8u) {
        grouped_q8_0_a_f32_batch_sharedx_chunked_w32_kernel<8u, BT><<<grid, rows_per_block * 32u, shmem>>>(low, w, heads, n_tokens, n_groups, n_blocks, rank, row_bytes);
    } else if (tile == 16u) {
        grouped_q8_0_a_f32_batch_sharedx_chunked_w32_kernel<16u, BT><<<grid, rows_per_block * 32u, shmem>>>(low, w, heads, n_tokens, n_groups, n_blocks, rank, row_bytes);
    } else {
        grouped_q8_0_a_f32_batch_sharedx_chunked_w32_kernel<32u, BT><<<grid, rows_per_block * 32u, shmem>>>(low, w, heads, n_tokens, n_groups, n_blocks, rank, row_bytes);
    }
}

static void cuda_launch_grouped_q8_a_sharedx(
        float *low,
        const unsigned char *w,
        const float *heads,
        uint32_t n_tokens,
        uint32_t n_groups,
        uint32_t n_blocks,
        uint32_t rank,
        uint64_t row_bytes,
        uint32_t rows_per_block,
        uint32_t tile,
        uint32_t block_tile) {
    const uint32_t row_blocks = (rank + rows_per_block - 1u) / rows_per_block;
    const dim3 grid(n_groups * row_blocks,
                    (n_tokens + tile - 1u) / tile,
                    1u);
    if (block_tile == 8u) {
        cuda_launch_grouped_q8_a_sharedx_bt<8u>(low, w, heads, n_tokens, n_groups, n_blocks, rank, row_bytes, grid, rows_per_block, tile);
    } else if (block_tile == 32u) {
        cuda_launch_grouped_q8_a_sharedx_bt<32u>(low, w, heads, n_tokens, n_groups, n_blocks, rank, row_bytes, grid, rows_per_block, tile);
    } else {
        cuda_launch_grouped_q8_a_sharedx_bt<16u>(low, w, heads, n_tokens, n_groups, n_blocks, rank, row_bytes, grid, rows_per_block, tile);
    }
}

template <uint32_t BT>
static void cuda_launch_grouped_q8_a_sharedx_strided_bt(
        float *low,
        const unsigned char *w,
        const float *heads,
        uint32_t n_tokens,
        uint32_t n_groups,
        uint32_t n_blocks,
        uint32_t rank,
        uint32_t x_token_stride,
        uint32_t x_group_stride,
        uint64_t row_bytes,
        dim3 grid,
        uint32_t rows_per_block,
        uint32_t tile) {
    const size_t shmem = (size_t)tile * BT * 32u * sizeof(float);
    if (tile == 2u) {
        grouped_q8_0_a_f32_batch_sharedx_chunked_strided_w32_kernel<2u, BT>
            <<<grid, rows_per_block * 32u, shmem>>>(
                low, w, heads, n_tokens, n_groups, n_blocks, rank,
                x_token_stride, x_group_stride, row_bytes);
    } else if (tile == 4u) {
        grouped_q8_0_a_f32_batch_sharedx_chunked_strided_w32_kernel<4u, BT>
            <<<grid, rows_per_block * 32u, shmem>>>(
                low, w, heads, n_tokens, n_groups, n_blocks, rank,
                x_token_stride, x_group_stride, row_bytes);
    } else if (tile == 8u) {
        grouped_q8_0_a_f32_batch_sharedx_chunked_strided_w32_kernel<8u, BT>
            <<<grid, rows_per_block * 32u, shmem>>>(
                low, w, heads, n_tokens, n_groups, n_blocks, rank,
                x_token_stride, x_group_stride, row_bytes);
    } else if (tile == 16u) {
        grouped_q8_0_a_f32_batch_sharedx_chunked_strided_w32_kernel<16u, BT>
            <<<grid, rows_per_block * 32u, shmem>>>(
                low, w, heads, n_tokens, n_groups, n_blocks, rank,
                x_token_stride, x_group_stride, row_bytes);
    } else {
        grouped_q8_0_a_f32_batch_sharedx_chunked_strided_w32_kernel<32u, BT>
            <<<grid, rows_per_block * 32u, shmem>>>(
                low, w, heads, n_tokens, n_groups, n_blocks, rank,
                x_token_stride, x_group_stride, row_bytes);
    }
}

static void cuda_launch_grouped_q8_a_sharedx_strided(
        float *low,
        const unsigned char *w,
        const float *heads,
        uint32_t n_tokens,
        uint32_t n_groups,
        uint32_t n_blocks,
        uint32_t rank,
        uint32_t x_token_stride,
        uint32_t x_group_stride,
        uint64_t row_bytes,
        uint32_t rows_per_block,
        uint32_t tile,
        uint32_t block_tile) {
    const uint32_t row_blocks =
        (rank + rows_per_block - 1u) / rows_per_block;
    const dim3 grid(n_groups * row_blocks,
                    (n_tokens + tile - 1u) / tile,
                    1u);
    if (block_tile == 8u) {
        cuda_launch_grouped_q8_a_sharedx_strided_bt<8u>(
            low, w, heads, n_tokens, n_groups, n_blocks, rank,
            x_token_stride, x_group_stride, row_bytes, grid,
            rows_per_block, tile);
    } else if (block_tile == 32u) {
        cuda_launch_grouped_q8_a_sharedx_strided_bt<32u>(
            low, w, heads, n_tokens, n_groups, n_blocks, rank,
            x_token_stride, x_group_stride, row_bytes, grid,
            rows_per_block, tile);
    } else {
        cuda_launch_grouped_q8_a_sharedx_strided_bt<16u>(
            low, w, heads, n_tokens, n_groups, n_blocks, rank,
            x_token_stride, x_group_stride, row_bytes, grid,
            rows_per_block, tile);
    }
}

static int cuda_matmul_q8_0_tensor_f16_gemm(
        ds4_gpu_tensor *out,
        const void *model_map,
        uint64_t model_size,
        uint64_t weight_offset,
        uint64_t in_dim,
        uint64_t out_dim,
        const ds4_gpu_tensor *x,
        uint64_t n_tok,
        const char *label) {
    if (!g_cublas_ready || !out || !x || !model_map ||
        in_dim == 0u || out_dim == 0u || n_tok == 0u ||
        in_dim > UINT32_MAX || out_dim > UINT32_MAX || n_tok > UINT32_MAX) return 0;
    const uint64_t blocks = (in_dim + 31u) / 32u;
    uint64_t row_bytes = 0, weight_bytes = 0, x_bytes = 0, out_bytes = 0;
    if (weight_offset > model_size ||
        !cuda_u64_mul_checked(blocks, 34u, &row_bytes) ||
        !cuda_u64_mul_checked(out_dim, row_bytes, &weight_bytes) ||
        weight_bytes > model_size - weight_offset ||
        !cuda_u64_mul3_checked(n_tok, in_dim, sizeof(float), &x_bytes) ||
        !cuda_u64_mul3_checked(n_tok, out_dim, sizeof(float), &out_bytes) ||
        x->bytes < x_bytes || out->bytes < out_bytes) return 0;
    const __half *w_f16 = cuda_q8_f16_ptr(model_map, weight_offset, weight_bytes, in_dim, out_dim, label);
    if (!w_f16) return 0;
    const uint64_t xh_count = n_tok * in_dim;
    __half *xh = (__half *)cuda_tmp_alloc(xh_count * sizeof(__half), "q8 f16 gemm activations");
    if (!xh) return 0;
    f32_to_f16_kernel<<<(xh_count + 255u) / 256u, 256>>>(xh, (const float *)x->ptr, xh_count);
    if (!cuda_ok(cudaGetLastError(), "q8 f16 activation convert launch")) return 0;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasStatus_t st = cublasGemmEx(g_cublas,
                                     CUBLAS_OP_T,
                                     CUBLAS_OP_N,
                                     (int)out_dim,
                                     (int)n_tok,
                                     (int)in_dim,
                                     &alpha,
                                     w_f16,
                                     CUDA_R_16F,
                                     (int)in_dim,
                                     xh,
                                     CUDA_R_16F,
                                     (int)in_dim,
                                     &beta,
                                     out->ptr,
                                     CUDA_R_32F,
                                     (int)out_dim,
                                     CUBLAS_COMPUTE_32F,
                                     CUBLAS_GEMM_DEFAULT);
    if (st == CUBLAS_STATUS_SUCCESS) return 1;
    fprintf(stderr, "ds4: " DS4_GPU_BLAS_NAME " q8 f16 matmul failed: status %d\n", (int)st);
    cuda_q8_f16_cache_disable_after_failure(DS4_GPU_BLAS_NAME " f16 matmul failure",
                                            in_dim * out_dim * sizeof(__half));
    return 0;
}

static int cuda_matmul_q8_0_tensor_f16_gemm_out_half(
        ds4_gpu_tensor *out_h,
        const void *model_map,
        uint64_t model_size,
        uint64_t weight_offset,
        uint64_t in_dim,
        uint64_t out_dim,
        const ds4_gpu_tensor *x,
        uint64_t n_tok,
        const char *label) {
    if (!g_cublas_ready || !out_h || !x || !model_map ||
        in_dim == 0u || out_dim == 0u || n_tok == 0u ||
        in_dim > UINT32_MAX || out_dim > UINT32_MAX || n_tok > UINT32_MAX) return 0;
    const uint64_t blocks = (in_dim + 31u) / 32u;
    uint64_t row_bytes = 0, weight_bytes = 0, x_bytes = 0, out_bytes = 0;
    if (weight_offset > model_size ||
        !cuda_u64_mul_checked(blocks, 34u, &row_bytes) ||
        !cuda_u64_mul_checked(out_dim, row_bytes, &weight_bytes) ||
        weight_bytes > model_size - weight_offset ||
        !cuda_u64_mul3_checked(n_tok, in_dim, sizeof(float), &x_bytes) ||
        !cuda_u64_mul3_checked(n_tok, out_dim, sizeof(__half), &out_bytes) ||
        x->bytes < x_bytes || out_h->bytes < out_bytes) return 0;
    const __half *w_f16 = cuda_q8_f16_ptr(model_map, weight_offset, weight_bytes, in_dim, out_dim, label);
    if (!w_f16) return 0;
    const uint64_t xh_count = n_tok * in_dim;
    __half *xh = (__half *)cuda_tmp_alloc(xh_count * sizeof(__half), "q8 f16-out gemm activations");
    if (!xh) return 0;
    f32_to_f16_kernel<<<(xh_count + 255u) / 256u, 256>>>(xh, (const float *)x->ptr, xh_count);
    if (!cuda_ok(cudaGetLastError(), "q8 f16-out activation convert launch")) return 0;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasStatus_t st = cublasGemmEx(g_cublas,
                                     CUBLAS_OP_T,
                                     CUBLAS_OP_N,
                                     (int)out_dim,
                                     (int)n_tok,
                                     (int)in_dim,
                                     &alpha,
                                     w_f16,
                                     CUDA_R_16F,
                                     (int)in_dim,
                                     xh,
                                     CUDA_R_16F,
                                     (int)in_dim,
                                     &beta,
                                     out_h->ptr,
                                     CUDA_R_16F,
                                     (int)out_dim,
                                     CUBLAS_COMPUTE_32F,
                                     CUBLAS_GEMM_DEFAULT);
    if (st == CUBLAS_STATUS_SUCCESS) return 1;
    fprintf(stderr, "ds4: " DS4_GPU_BLAS_NAME " q8 f16-out matmul failed: status %d\n", (int)st);
    cuda_q8_f16_cache_disable_after_failure(DS4_GPU_BLAS_NAME " f16-out matmul failure",
                                            in_dim * out_dim * sizeof(__half));
    return 0;
}

extern "C" int ds4_gpu_matmul_q8_0_f16_out_tensor(
        ds4_gpu_tensor       *out_h,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok) {
    return cuda_matmul_q8_0_tensor_f16_gemm_out_half(out_h, model_map, model_size,
                                                     weight_offset, in_dim, out_dim,
                                                     x, n_tok, "q8_f16_out");
}

static int cuda_matmul_q8_0_tensor_labeled(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim, const ds4_gpu_tensor *x, uint64_t n_tok, const char *label) {
    if (!out || !x || !model_map ||
        in_dim == 0u || out_dim == 0u || n_tok == 0u ||
        in_dim > UINT32_MAX || out_dim > UINT32_MAX || n_tok > UINT32_MAX) return 0;
    uint64_t blocks = (in_dim + 31u) / 32u;
    uint64_t row_bytes = 0, weight_bytes = 0, x_bytes = 0, out_bytes = 0;
    if (weight_offset > model_size ||
        !cuda_u64_mul_checked(blocks, 34u, &row_bytes) ||
        !cuda_u64_mul_checked(out_dim, row_bytes, &weight_bytes) ||
        weight_bytes > model_size - weight_offset ||
        !cuda_u64_mul3_checked(n_tok, in_dim, sizeof(float), &x_bytes) ||
        !cuda_u64_mul3_checked(n_tok, out_dim, sizeof(float), &out_bytes) ||
        x->bytes < x_bytes || out->bytes < out_bytes) return 0;
    if (n_tok > 1 && !g_quality_mode &&
        cuda_runtime_config()->shared_down_cublas && in_dim == 2048u && out_dim == 4096u &&
        cuda_matmul_q8_0_tensor_f16_gemm(out, model_map, model_size, weight_offset,
                                         in_dim, out_dim, x, n_tok, label ? label : "shared_expert")) {
        return 1;
    }
    const char *wptr = cuda_model_range_ptr(model_map, weight_offset, weight_bytes, "q8_0");
    if (!wptr) return 0;
    if (n_tok == 1) {
        /* DeepSeek V4 TP=2 decode Q-B projection: each rank owns 32 heads,
         * hence 16,384 Q8 rows over the 1,024-wide LoRA activation.  Reuse
         * the hardened four-block-per-wave kernel from the attention output
         * expansion.  Keep this exact-shape and gfx1151-only so other Q8
         * users and GPU generations retain their established dispatch. */
        if (attention_q_b_pack4_enabled() &&
            in_dim == 1024u && out_dim == 16384u && blocks == 32u) {
            const unsigned threads = 1024u;
            const unsigned rows_per_block = threads / 32u;
            matmul_q8_0_f32_sharedx_warp_rows_w32_pack4_kernel<<<
                    (unsigned)((out_dim + rows_per_block - 1u) / rows_per_block),
                    threads,
                    (size_t)in_dim * sizeof(float)>>>(
                    (float *)out->ptr,
                    reinterpret_cast<const unsigned char *>(wptr),
                    (const float *)x->ptr,
                    (uint32_t)blocks,
                    out_dim,
                    blocks * 34u);
            return cuda_ok(cudaGetLastError(), "matmul_q8_0 attention q_b pack4 launch");
        }
        const bool extended_sharedx =
            in_dim > 8192u &&
            in_dim <= 16384u &&
            cuda_runtime_config()->q8_decode_sharedx_64k;
        if ((in_dim & 31u) == 0u &&
            (in_dim <= 8192u || extended_sharedx)) {
            const unsigned rows_per_block = 32u;
            const unsigned threads = rows_per_block * 32u;
            matmul_q8_0_f32_sharedx_warp_rows_w32_kernel<<<
                    (unsigned)((out_dim + rows_per_block - 1u) / rows_per_block),
                    threads,
                    (size_t)in_dim * sizeof(float)>>>(
                    (float *)out->ptr,
                    reinterpret_cast<const unsigned char *>(wptr),
                    (const float *)x->ptr,
                    (uint32_t)blocks,
                    out_dim,
                    blocks * 34u);
            const cudaError_t launch_err = cudaGetLastError();
            if (launch_err == cudaSuccess) {
                if (extended_sharedx) {
                    static int notice_printed = 0;
                    if (!notice_printed) {
                        fprintf(stderr,
                                DS4_GPU_LOG_PREFIX
                                "Q8 one-token shared-input kernel enabled "
                                "through 64 KiB LDS (in_dim=%llu)\n",
                                (unsigned long long)in_dim);
                        notice_printed = 1;
                    }
                }
                return 1;
            }
            if (!extended_sharedx) {
                return cuda_ok(launch_err,
                               "matmul_q8_0 f32 sharedx launch");
            }
            static int fallback_notice_printed = 0;
            if (!fallback_notice_printed) {
                fprintf(stderr,
                        DS4_GPU_LOG_PREFIX
                        "Q8 64 KiB shared-input launch unavailable "
                        "(%s); falling back to the warp-row kernel\n",
                        cudaGetErrorString(launch_err));
                fallback_notice_printed = 1;
            }
        }
        matmul_q8_0_f32_warp8_kernel<<<((unsigned)out_dim + 7u) / 8u, 256>>>(
                (float *)out->ptr,
                reinterpret_cast<const unsigned char *>(wptr),
                (const float *)x->ptr,
                in_dim,
                out_dim,
                blocks);
        return cuda_ok(cudaGetLastError(), "matmul_q8_0 f32 warp launch");
    }
    if (n_tok > 1) {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
        if (q8_small_batch_dp4a_enabled() && n_tok <= 8u &&
            (in_dim & 31u) == 0u) {
            const uint64_t xq_bytes = n_tok * blocks * 32u;
            const uint64_t scale_offset = (xq_bytes + 15u) & ~15ull;
            const uint64_t tmp_bytes =
                scale_offset + n_tok * blocks * sizeof(float);
            void *tmp = cuda_tmp_alloc(tmp_bytes, "q8 small-batch dp4a");
            if (!tmp) return 0;
            int8_t *xq = (int8_t *)tmp;
            float *xscale = (float *)((char *)tmp + scale_offset);
            const dim3 qgrid((uint32_t)blocks, (uint32_t)n_tok, 1u);
            quantize_q8_0_f32_kernel<<<qgrid, 32>>>(
                    xq, xscale, (const float *)x->ptr, in_dim, blocks);
            if (!cuda_ok(cudaGetLastError(),
                         "q8 small-batch dp4a quantize launch")) return 0;
            cuda_launch_q8_small_batch_dp4a(
                    (float *)out->ptr,
                    reinterpret_cast<const unsigned char *>(wptr),
                    xq, xscale, (uint32_t)blocks, (uint32_t)out_dim,
                    (uint32_t)n_tok, blocks * 34u);
            return cuda_ok(cudaGetLastError(),
                           "q8 small-batch dp4a matmul launch");
        }
#endif
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
        if (!g_quality_mode && (in_dim % 32u) == 0u &&
            out_dim >= 1024u &&
            n_tok >= 256u &&
            in_dim <= UINT32_MAX && out_dim <= UINT32_MAX && n_tok <= UINT32_MAX) {
            if (q8_batch_wmma_m128_enabled()) {
                if (out_dim >= 1024u && q8_batch_wmma_k128_padded_enabled()) {
                    if (out_dim >= 8192u &&
                        q8_batch_wmma_m256_k128_enabled()) {
                        const dim3 grid(
                                (uint32_t)((out_dim + 255u) / 256u),
                                (uint32_t)((n_tok + 63u) / 64u), 1u);
                        matmul_q8_0_f32_batch_wmma_kernel<256u, 128u>
                                <<<grid, 512u>>>(
                                (float *)out->ptr,
                                reinterpret_cast<const unsigned char *>(wptr),
                                (const float *)x->ptr,
                                (uint32_t)n_tok,
                                (uint32_t)in_dim,
                                (uint32_t)out_dim,
                                blocks * 34u);
                    } else {
                        const dim3 grid(
                                (uint32_t)((out_dim + 127u) / 128u),
                                (uint32_t)((n_tok + 63u) / 64u), 1u);
                        matmul_q8_0_f32_batch_wmma_kernel<128u, 128u>
                                <<<grid, 256u>>>(
                                (float *)out->ptr,
                                reinterpret_cast<const unsigned char *>(wptr),
                                (const float *)x->ptr,
                                (uint32_t)n_tok,
                                (uint32_t)in_dim,
                                (uint32_t)out_dim,
                                blocks * 34u);
                    }
                } else {
                    const dim3 grid(
                            (uint32_t)((out_dim + 127u) / 128u),
                            (uint32_t)((n_tok + 63u) / 64u), 1u);
                    matmul_q8_0_f32_batch_wmma_kernel<128u><<<grid, 256u>>>(
                            (float *)out->ptr,
                            reinterpret_cast<const unsigned char *>(wptr),
                            (const float *)x->ptr,
                            (uint32_t)n_tok,
                            (uint32_t)in_dim,
                            (uint32_t)out_dim,
                            blocks * 34u);
                }
            } else {
                const dim3 grid((uint32_t)((out_dim + 63u) / 64u),
                                (uint32_t)((n_tok + 63u) / 64u),
                                1u);
                matmul_q8_0_f32_batch_wmma_kernel<64u><<<grid, 128u>>>(
                        (float *)out->ptr,
                        reinterpret_cast<const unsigned char *>(wptr),
                        (const float *)x->ptr,
                        (uint32_t)n_tok,
                        (uint32_t)in_dim,
                        (uint32_t)out_dim,
                        blocks * 34u);
            }
            return cuda_ok(cudaGetLastError(), "matmul_q8_0 f32 batch wmma launch");
        }
#endif
        if ((in_dim & 31u) == 0u && out_dim <= UINT32_MAX && n_tok <= UINT32_MAX) {
            const uint32_t rows_per_block = 32u;
            const uint32_t tile = q8_batch_token_tile(n_tok);
            const uint32_t block_tile = q8_small_batch_block_tile(n_tok);
            cuda_launch_q8_batch_sharedx((float *)out->ptr,
                                         reinterpret_cast<const unsigned char *>(wptr),
                                         (const float *)x->ptr,
                                         (uint32_t)blocks,
                                         (uint32_t)out_dim,
                                         (uint32_t)n_tok,
                                         blocks * 34u,
                                         rows_per_block,
                                         tile,
                                         block_tile);
            return cuda_ok(cudaGetLastError(), "matmul_q8_0 f32 batch sharedx launch");
        }
        dim3 bgrid(((unsigned)out_dim + 7u) / 8u, (unsigned)n_tok, 1);
        matmul_q8_0_f32_batch_warp8_kernel<<<bgrid, 256>>>(
                (float *)out->ptr,
                reinterpret_cast<const unsigned char *>(wptr),
                (const float *)x->ptr,
                in_dim,
                out_dim,
                n_tok,
                blocks);
        return cuda_ok(cudaGetLastError(), "matmul_q8_0 f32 batch warp launch");
    }
    if (g_cublas_ready && n_tok > 1) {
        const __half *w_f16 = cuda_q8_f16_ptr(model_map, weight_offset, weight_bytes, in_dim, out_dim, label);
        if (w_f16) {
            const uint64_t xh_count = n_tok * in_dim;
            __half *xh = (__half *)cuda_tmp_alloc(xh_count * sizeof(__half), "q8 f16 gemm activations");
            if (!xh) return 0;
            f32_to_f16_kernel<<<(xh_count + 255) / 256, 256>>>(xh, (const float *)x->ptr, xh_count);
            if (!cuda_ok(cudaGetLastError(), "q8 f16 activation convert launch")) return 0;
            const float alpha = 1.0f;
            const float beta = 0.0f;
            cublasStatus_t st = cublasGemmEx(g_cublas,
                                             CUBLAS_OP_T,
                                             CUBLAS_OP_N,
                                             (int)out_dim,
                                             (int)n_tok,
                                             (int)in_dim,
                                             &alpha,
                                             w_f16,
                                             CUDA_R_16F,
                                             (int)in_dim,
                                             xh,
                                             CUDA_R_16F,
                                             (int)in_dim,
                                             &beta,
                                             out->ptr,
                                             CUDA_R_32F,
                                             (int)out_dim,
                                             CUBLAS_COMPUTE_32F,
                                             CUBLAS_GEMM_DEFAULT);
            if (st == CUBLAS_STATUS_SUCCESS) return 1;
            fprintf(stderr, "ds4: " DS4_GPU_BLAS_NAME " q8 f16 matmul failed: status %d\n", (int)st);
            cuda_q8_f16_cache_disable_after_failure(DS4_GPU_BLAS_NAME " f16 matmul failure",
                                                    in_dim * out_dim * sizeof(__half));
            /* The F16 expansion cache is only an optimization.  If cuBLAS
             * rejects the cached path under memory pressure, retry the same
             * operation through the native Q8 kernels below. */
        }
    }
    const uint64_t xq_bytes = n_tok * blocks * 32u;
    const uint64_t scale_offset = (xq_bytes + 15u) & ~15ull;
    const uint64_t tmp_bytes = scale_offset + n_tok * blocks * sizeof(float);
    void *tmp = cuda_tmp_alloc(tmp_bytes, "q8_0 prequant");
    if (!tmp) return 0;
    int8_t *xq = (int8_t *)tmp;
    float *xscale = (float *)((char *)tmp + scale_offset);
    const int use_dp4a = 1;
    dim3 qgrid((unsigned)blocks, (unsigned)n_tok, 1);
    quantize_q8_0_f32_kernel<<<qgrid, 32>>>(xq, xscale, (const float *)x->ptr, in_dim, blocks);
    if (!cuda_ok(cudaGetLastError(), "matmul_q8_0 quantize launch")) return 0;
    if (blocks <= 32u) {
        dim3 bgrid(((unsigned)out_dim + 7u) / 8u, (unsigned)n_tok, 1);
        matmul_q8_0_preq_batch_warp8_kernel<<<bgrid, 256>>>(
                (float *)out->ptr,
                reinterpret_cast<const unsigned char *>(wptr),
                xq,
                xscale,
                in_dim,
                out_dim,
                n_tok,
                blocks,
                use_dp4a);
        return cuda_ok(cudaGetLastError(), "matmul_q8_0 batch warp launch");
    }
    dim3 grid((unsigned)out_dim, (unsigned)n_tok, 1);
    matmul_q8_0_preq_kernel<<<grid, 256>>>((float *)out->ptr,
                                           reinterpret_cast<const unsigned char *>(wptr),
                                           xq,
                                           xscale,
                                           in_dim, out_dim, n_tok, blocks,
                                           use_dp4a);
    return cuda_ok(cudaGetLastError(), "matmul_q8_0 launch");
}

extern "C" int ds4_gpu_matmul_q8_0_tensor(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim, const ds4_gpu_tensor *x, uint64_t n_tok) {
    return cuda_matmul_q8_0_tensor_labeled(out, model_map, model_size, weight_offset,
                                           in_dim, out_dim, x, n_tok, "q8_0");
}

extern "C" int ds4_gpu_attention_q_b_qnorm_rope_q8_0_tensor(
        ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
        uint64_t weight_offset, uint64_t in_dim, uint32_t n_head,
        uint32_t head_dim, const ds4_gpu_tensor *x, uint32_t n_rot,
        uint32_t pos0, uint32_t n_ctx_orig, float freq_base,
        float freq_scale, float ext_factor, float attn_factor,
        float beta_fast, float beta_slow, float eps) {
    if (!attention_q_b_qnorm_rope_enabled()) return 0;
    if (!out || !x || !model_map || in_dim != 1024u || n_head != 32u ||
        head_dim != 512u || n_rot == 0u || n_rot > head_dim ||
        (n_rot & 1u) || x->bytes < 1024u * sizeof(float) ||
        out->bytes < (uint64_t)n_head * head_dim * sizeof(float)) return 0;
    const uint64_t blocks = 32u;
    const uint64_t row_bytes = blocks * 34u;
    const uint64_t weight_bytes =
        (uint64_t)n_head * head_dim * row_bytes;
    if (weight_offset > model_size || weight_bytes > model_size - weight_offset)
        return 0;
    const char *wptr = cuda_model_range_ptr(
            model_map, weight_offset, weight_bytes, "attention_q_b_fused");
    if (!wptr) return 0;
    attention_q_b_qnorm_rope_q8_0_head_kernel<<<n_head, 1024u, 7168u>>>(
            (float *)out->ptr,
            reinterpret_cast<const unsigned char *>(wptr),
            (const float *)x->ptr,
            n_head, row_bytes, n_rot, pos0, n_ctx_orig, freq_base,
            freq_scale, ext_factor, attn_factor, beta_fast, beta_slow, eps);
    return cuda_ok(cudaGetLastError(),
                   "attention q_b qnorm rope fused launch");
}

extern "C" int ds4_gpu_matmul_q8_0_decode_mpp_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok) {
    return cuda_matmul_q8_0_tensor_labeled(out,
                                           model_map,
                                           model_size,
                                           weight_offset,
                                           in_dim,
                                           out_dim,
                                           x,
                                           n_tok,
                                           "q8_0_decode");
}

extern "C" int ds4_gpu_matmul_q8_0_decode_mpp_model_view_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok) {
    return cuda_matmul_q8_0_tensor_labeled(out,
                                           model_map,
                                           model_size,
                                           weight_offset,
                                           in_dim,
                                           out_dim,
                                           x,
                                           n_tok,
                                           "q8_0_decode_model_view");
}

extern "C" int ds4_gpu_matmul_q8_0_rows_scalar_tensor(
        ds4_gpu_tensor *out,
        const void *model_map,
        uint64_t model_size,
        uint64_t weight_offset,
        uint64_t in_dim,
        uint64_t out_dim,
        const ds4_gpu_tensor *x,
        uint64_t n_tok) {
    (void)out; (void)model_map; (void)model_size; (void)weight_offset;
    (void)in_dim; (void)out_dim; (void)x; (void)n_tok;
    return 0;
}

extern "C" int ds4_gpu_matmul_q8_0_pair_tensor(
        ds4_gpu_tensor *out0,
        ds4_gpu_tensor *out1,
        const void *model_map,
        uint64_t model_size,
        uint64_t weight0_offset,
        uint64_t weight1_offset,
        uint64_t in_dim,
        uint64_t out0_dim,
        uint64_t out1_dim,
        const ds4_gpu_tensor *x,
        uint64_t n_tok) {
    if (!out0 || !out1 || !x || !model_map ||
        in_dim == 0 || out0_dim == 0 || out1_dim == 0 || n_tok == 0 ||
        in_dim > UINT32_MAX || out0_dim > UINT32_MAX || out1_dim > UINT32_MAX || n_tok > UINT32_MAX) {
        return 0;
    }
    if (n_tok != 1 &&
        !(q8_reuse_quant_enabled() && q8_small_batch_dp4a_enabled() &&
          n_tok <= 8u && (in_dim & 31u) == 0u)) {
        return cuda_matmul_q8_0_tensor_labeled(out0, model_map, model_size, weight0_offset,
                                               in_dim, out0_dim, x, n_tok, "q8_0_pair0") &&
               cuda_matmul_q8_0_tensor_labeled(out1, model_map, model_size, weight1_offset,
                                               in_dim, out1_dim, x, n_tok, "q8_0_pair1");
    }
    const uint64_t blocks = (in_dim + 31u) / 32u;
    uint64_t row_bytes = 0, weight0_bytes = 0, weight1_bytes = 0;
    if (weight0_offset > model_size || weight1_offset > model_size ||
        !cuda_u64_mul_checked(blocks, 34u, &row_bytes) ||
        !cuda_u64_mul_checked(out0_dim, row_bytes, &weight0_bytes) ||
        !cuda_u64_mul_checked(out1_dim, row_bytes, &weight1_bytes)) {
        return 0;
    }
    if (weight0_bytes > model_size - weight0_offset ||
        weight1_bytes > model_size - weight1_offset ||
        x->bytes < n_tok * in_dim * sizeof(float) ||
        out0->bytes < n_tok * out0_dim * sizeof(float) ||
        out1->bytes < n_tok * out1_dim * sizeof(float)) {
        return 0;
    }
    const char *w0 = cuda_model_range_ptr(model_map, weight0_offset, weight0_bytes, "q8_0_pair0");
    const char *w1 = cuda_model_range_ptr(model_map, weight1_offset, weight1_bytes, "q8_0_pair1");
    if (!w0 || !w1) return 0;
    const uint64_t max_out = out0_dim > out1_dim ? out0_dim : out1_dim;
    if (n_tok == 1u && q8_decode_pair_dp4a_enabled() &&
        (in_dim & 31u) == 0u) {
        const uint64_t xq_bytes = blocks * 32u;
        const uint64_t scale_offset = (xq_bytes + 15u) & ~15ull;
        const uint64_t tmp_bytes = scale_offset + blocks * sizeof(float);
        void *tmp = cuda_tmp_alloc(tmp_bytes, "q8_0 decode pair shared prequant");
        if (!tmp) return 0;
        int8_t *xq = (int8_t *)tmp;
        float *xscale = (float *)((char *)tmp + scale_offset);
        quantize_q8_0_f32_kernel<<<dim3((uint32_t)blocks, 1u, 1u), 32>>>(
            xq, xscale, (const float *)x->ptr, in_dim, blocks);
        if (!cuda_ok(cudaGetLastError(),
                     "q8_0 decode pair quantize launch")) return 0;
        cuda_launch_q8_small_batch_dp4a(
            (float *)out0->ptr,
            reinterpret_cast<const unsigned char *>(w0),
            xq, xscale, (uint32_t)blocks, (uint32_t)out0_dim,
            1u, row_bytes);
        if (!cuda_ok(cudaGetLastError(),
                     "q8_0 decode pair first DP4A launch")) return 0;
        cuda_launch_q8_small_batch_dp4a(
            (float *)out1->ptr,
            reinterpret_cast<const unsigned char *>(w1),
            xq, xscale, (uint32_t)blocks, (uint32_t)out1_dim,
            1u, row_bytes);
        return cuda_ok(cudaGetLastError(),
                       "q8_0 decode pair second DP4A launch");
    }
    if (n_tok != 1) {
        const uint64_t xq_bytes = n_tok * blocks * 32u;
        const uint64_t scale_offset = (xq_bytes + 15u) & ~15ull;
        const uint64_t tmp_bytes =
            scale_offset + n_tok * blocks * sizeof(float);
        void *tmp = cuda_tmp_alloc(tmp_bytes, "q8_0 pair shared prequant");
        if (!tmp) return 0;
        int8_t *xq = (int8_t *)tmp;
        float *xscale = (float *)((char *)tmp + scale_offset);
        const dim3 qgrid((uint32_t)blocks, (uint32_t)n_tok, 1u);
        quantize_q8_0_f32_kernel<<<qgrid, 32>>>(
                xq, xscale, (const float *)x->ptr, in_dim, blocks);
        if (!cuda_ok(cudaGetLastError(),
                     "q8_0 pair shared quantize launch")) return 0;
        cuda_launch_q8_small_batch_dp4a(
                (float *)out0->ptr,
                reinterpret_cast<const unsigned char *>(w0),
                xq, xscale, (uint32_t)blocks, (uint32_t)out0_dim,
                (uint32_t)n_tok, row_bytes);
        if (!cuda_ok(cudaGetLastError(),
                     "q8_0 pair shared first matmul launch")) return 0;
        cuda_launch_q8_small_batch_dp4a(
                (float *)out1->ptr,
                reinterpret_cast<const unsigned char *>(w1),
                xq, xscale, (uint32_t)blocks, (uint32_t)out1_dim,
                (uint32_t)n_tok, row_bytes);
        return cuda_ok(cudaGetLastError(),
                       "q8_0 pair shared second matmul launch");
    }
    if ((in_dim & 31u) == 0u && in_dim <= 8192u) {
        const unsigned rows_per_block = 32u;
        const unsigned threads = rows_per_block * 32u;
        if (q8_pair_f32_pack4_enabled() && in_dim == 4096u &&
            out0_dim == 1024u && out1_dim == 512u) {
            matmul_q8_0_pair_f32_sharedx_warp_rows_w32_pack4_kernel<<<
                    (unsigned)((max_out + rows_per_block - 1u) /
                               rows_per_block),
                    threads,
                    (size_t)in_dim * sizeof(float)>>>(
                (float *)out0->ptr,
                (float *)out1->ptr,
                reinterpret_cast<const unsigned char *>(w0),
                reinterpret_cast<const unsigned char *>(w1),
                (const float *)x->ptr,
                (uint32_t)blocks,
                out0_dim,
                out1_dim,
                row_bytes);
            return cuda_ok(cudaGetLastError(),
                           "matmul_q8_0 pair f32 pack4 launch");
        }
        matmul_q8_0_pair_f32_sharedx_warp_rows_w32_kernel<<<
                (unsigned)((max_out + rows_per_block - 1u) / rows_per_block),
                threads,
                (size_t)in_dim * sizeof(float)>>>(
                (float *)out0->ptr,
                (float *)out1->ptr,
                reinterpret_cast<const unsigned char *>(w0),
                reinterpret_cast<const unsigned char *>(w1),
                (const float *)x->ptr,
                (uint32_t)blocks,
                out0_dim,
                out1_dim,
                blocks * 34u);
        return cuda_ok(cudaGetLastError(), "matmul_q8_0 pair f32 sharedx launch");
    }
    matmul_q8_0_pair_f32_warp8_kernel<<<((unsigned)max_out + 7u) / 8u, 256>>>(
            (float *)out0->ptr,
            (float *)out1->ptr,
            reinterpret_cast<const unsigned char *>(w0),
            reinterpret_cast<const unsigned char *>(w1),
            (const float *)x->ptr,
            in_dim,
            out0_dim,
            out1_dim,
            blocks);
    return cuda_ok(cudaGetLastError(), "matmul_q8_0 pair f32 warp launch");
}

static int cuda_matmul_q8_0_hc_expand_tensor_labeled(
        ds4_gpu_tensor       *out_hc,
        ds4_gpu_tensor       *block_out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        const ds4_gpu_tensor *block_add,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc,
        const char             *label) {
    if (!out_hc || !block_out || !x || !residual_hc || !split || !model_map ||
        in_dim == 0 || out_dim == 0 || n_embd == 0 || n_hc == 0 ||
        out_dim != (uint64_t)n_embd) {
        return 0;
    }
    const uint64_t blocks = (in_dim + 31) / 32;
    if (weight_offset > model_size || out_dim > UINT64_MAX / (blocks * 34)) return 0;
    const uint64_t weight_bytes = out_dim * blocks * 34;
    const uint64_t hc_bytes = (uint64_t)n_hc * n_embd * sizeof(float);
    const uint64_t split_bytes = (uint64_t)(2u * n_hc + n_hc * n_hc) * sizeof(float);
    if (weight_bytes > model_size - weight_offset ||
        x->bytes < in_dim * sizeof(float) ||
        block_out->bytes < out_dim * sizeof(float) ||
        residual_hc->bytes < hc_bytes ||
        split->bytes < split_bytes ||
        out_hc->bytes < hc_bytes ||
        (block_add && block_add->bytes < out_dim * sizeof(float))) {
        return 0;
    }
    const char *wptr = cuda_model_range_ptr(model_map, weight_offset, weight_bytes, label ? label : "q8_0_hc_expand");
    if (!wptr) return 0;
    if ((in_dim & 31u) == 0u && in_dim <= 8192u) {
        const unsigned rows_per_block = 32u;
        const unsigned threads = rows_per_block * 32u;
        matmul_q8_0_hc_expand_f32_sharedx_warp_rows_w32_kernel<<<
                (unsigned)((out_dim + rows_per_block - 1u) / rows_per_block),
                threads,
                (size_t)in_dim * sizeof(float)>>>(
                (float *)out_hc->ptr,
                (float *)block_out->ptr,
                block_add ? (const float *)block_add->ptr : (const float *)block_out->ptr,
                (const float *)residual_hc->ptr,
                (const float *)split->ptr,
                reinterpret_cast<const unsigned char *>(wptr),
                (const float *)x->ptr,
                (uint32_t)blocks,
                out_dim,
                blocks * 34u,
                n_embd,
                n_hc,
                block_add ? 1 : 0);
        return cuda_ok(cudaGetLastError(), "matmul_q8_0_hc_expand f32 sharedx launch");
    }
    matmul_q8_0_hc_expand_f32_warp8_kernel<<<((unsigned)out_dim + 7u) / 8u, 256>>>(
            (float *)out_hc->ptr,
            (float *)block_out->ptr,
            block_add ? (const float *)block_add->ptr : (const float *)block_out->ptr,
            (const float *)residual_hc->ptr,
            (const float *)split->ptr,
            reinterpret_cast<const unsigned char *>(wptr),
            (const float *)x->ptr,
            in_dim,
            out_dim,
            n_embd,
            n_hc,
            blocks,
            block_add ? 1 : 0);
    return cuda_ok(cudaGetLastError(), "matmul_q8_0_hc_expand f32 launch");
}

extern "C" int ds4_gpu_matmul_f16_tensor(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim, const ds4_gpu_tensor *x, uint64_t n_tok) {
    if (!out || !x || !model_map ||
        in_dim == 0u || out_dim == 0u || n_tok == 0u ||
        in_dim > UINT32_MAX || out_dim > UINT32_MAX || n_tok > UINT32_MAX) return 0;
    uint64_t weight_bytes = 0, x_bytes = 0, out_bytes = 0;
    if (weight_offset > model_size ||
        !cuda_u64_mul3_checked(out_dim, in_dim, sizeof(uint16_t), &weight_bytes) ||
        weight_bytes > model_size - weight_offset ||
        !cuda_u64_mul3_checked(n_tok, in_dim, sizeof(float), &x_bytes) ||
        !cuda_u64_mul3_checked(n_tok, out_dim, sizeof(float), &out_bytes) ||
        x->bytes < x_bytes || out->bytes < out_bytes) return 0;
    const char *wptr = cuda_model_range_ptr(model_map, weight_offset, weight_bytes, "f16");
    if (!wptr) return 0;
    const __half *w = (const __half *)wptr;
    const int ordered_decode = n_tok == 1u;
    const int hc5_shape = n_tok == 5u && in_dim == 16384u && out_dim == 24u;
    const int hc5_mode = hc5_shape ? f16_hc_five_row_mode() : 0;
    const int hc5_selected = hc5_mode != 0 &&
        !g_quality_mode && !cuda_runtime_config()->graph_dump;
    if (hc5_selected) {
        const uint64_t xh_count = n_tok * in_dim;
        __half *xh = (__half *)cuda_tmp_alloc(
                xh_count * sizeof(__half), "f16 HC five-row activations");
        if (!xh) return 0;
        f32_to_f16_kernel<<<(xh_count + 255u) / 256u, 256>>>(
                xh, (const float *)x->ptr, xh_count);
        if (!cuda_ok(cudaGetLastError(), "f16 HC five-row conversion launch")) return 0;
        if (hc5_mode == 2) {
            matmul_f16_hc_token_block_kernel<<<dim3(24u, 5u), 256u>>>(
                    (float *)out->ptr, w, xh,
                    (uint32_t)in_dim, (uint32_t)out_dim);
        } else {
            matmul_f16_hc_five_row_kernel<5u><<<24u, 256u>>>(
                    (float *)out->ptr, w, xh,
                    (uint32_t)in_dim, (uint32_t)out_dim);
        }
        return cuda_ok(cudaGetLastError(), "f16 HC five-row launch");
    }
    const int indexer_q_cus =
        (in_dim == 1024u || in_dim == 1536u) &&
        out_dim == 8192u && n_tok <= 5u &&
        !g_quality_mode && !cuda_runtime_config()->graph_dump
            ? f16_indexer_q_wvsplit_cus() : 0;
    if (indexer_q_cus > 0) {
#if defined(__HIP_PLATFORM_AMD__)
        const uint64_t xh_count = n_tok * in_dim;
        __half *xh = (__half *)cuda_tmp_alloc(
                xh_count * sizeof(__half), "f16 indexer Q activations");
        if (!xh) return 0;
        f32_to_f16_kernel<<<(xh_count + 255u) / 256u, 256u>>>(
                xh, (const float *)x->ptr, xh_count);
        if (!cuda_ok(cudaGetLastError(),
                     "f16 indexer Q activation conversion launch")) return 0;
        const dim3 block(32u, 16u);
        switch (n_tok) {
            case 1u: matmul_f16_indexer_q_wvsplit_kernel<1u><<<indexer_q_cus, block>>>(
                         (float *)out->ptr, w, xh, (uint32_t)in_dim, 8192u); break;
            case 2u: matmul_f16_indexer_q_wvsplit_kernel<2u><<<indexer_q_cus, block>>>(
                         (float *)out->ptr, w, xh, (uint32_t)in_dim, 8192u); break;
            case 3u: matmul_f16_indexer_q_wvsplit_kernel<3u><<<indexer_q_cus, block>>>(
                         (float *)out->ptr, w, xh, (uint32_t)in_dim, 8192u); break;
            case 4u: matmul_f16_indexer_q_wvsplit_kernel<4u><<<indexer_q_cus, block>>>(
                         (float *)out->ptr, w, xh, (uint32_t)in_dim, 8192u); break;
            case 5u: matmul_f16_indexer_q_wvsplit_kernel<5u><<<indexer_q_cus, block>>>(
                         (float *)out->ptr, w, xh, (uint32_t)in_dim, 8192u); break;
            default: return 0;
        }
        return cuda_ok(cudaGetLastError(), "f16 indexer Q wvSplitK launch");
#endif
    }
    if (g_cublas_ready && n_tok > 1) {
        const uint64_t xh_count = n_tok * in_dim;
        __half *xh = (__half *)cuda_tmp_alloc(xh_count * sizeof(__half), "f16 gemm activations");
        if (!xh) return 0;
        f32_to_f16_kernel<<<(xh_count + 255) / 256, 256>>>(xh, (const float *)x->ptr, xh_count);
        if (!cuda_ok(cudaGetLastError(), "f16 activation convert launch")) return 0;
        const float alpha = 1.0f;
        const float beta = 0.0f;
        cublasStatus_t st = cublasGemmEx(g_cublas,
                                         CUBLAS_OP_T,
                                         CUBLAS_OP_N,
                                         (int)out_dim,
                                         (int)n_tok,
                                         (int)in_dim,
                                         &alpha,
                                         w,
                                         CUDA_R_16F,
                                         (int)in_dim,
                                         xh,
                                         CUDA_R_16F,
                                         (int)in_dim,
                                         &beta,
                                         out->ptr,
                                         CUDA_R_32F,
                                         (int)out_dim,
                                         CUBLAS_COMPUTE_32F,
                                         CUBLAS_GEMM_DEFAULT);
        return cublas_ok(st, "f16 matmul");
    }
    /* The 4096x256 F16 router projection is latency-bound and the ordered
     * 32-thread row kernel is at least as fast on gfx1151; keep shared-X for
     * compressor/indexer F16 decode where reusing x across rows is the win. */
    const bool f16_decode_router_shape = (in_dim == 4096u && out_dim == 256u);
    if (n_tok == 1u && !g_quality_mode && !cuda_runtime_config()->graph_dump &&
        !f16_decode_router_shape) {
        if (in_dim <= 8192u && in_dim * sizeof(float) <= 65536u) {
            const uint32_t rows_per_block = 32u;
            matmul_f16_f32_sharedx_warp_rows_w32_kernel<<<
                    ((unsigned)out_dim + rows_per_block - 1u) / rows_per_block,
                    rows_per_block * 32u,
                    (size_t)in_dim * sizeof(float)>>>(
                    (float *)out->ptr, w, (const float *)x->ptr, (uint32_t)in_dim, out_dim);
            return cuda_ok(cudaGetLastError(), "matmul_f16 sharedx launch");
        }
    }
    dim3 grid((unsigned)out_dim, (unsigned)n_tok, 1);
    if (ordered_decode) {
        matmul_f16_ordered_chunks_kernel<<<grid, 32>>>((float *)out->ptr, w, (const float *)x->ptr, in_dim, out_dim, n_tok);
        return cuda_ok(cudaGetLastError(), "matmul_f16_ordered_chunks launch");
    }
    matmul_f16_kernel<<<grid, 256>>>((float *)out->ptr, w, (const float *)x->ptr, in_dim, out_dim, n_tok);
    return cuda_ok(cudaGetLastError(), "matmul_f16 launch");
}

static int matmul_bf16_f32_toktile_w32_launch(
        float *out, const uint16_t *weight, const float *x,
        uint32_t in_dim, uint32_t out_dim, uint32_t n_tok) {
    uint32_t first = 0u;
    const uint32_t chunks32 = n_tok / 32u;
    if (chunks32 > 0u) {
        matmul_bf16_f32_toktile_w32_kernel<32u><<<
                dim3(out_dim, chunks32), kDs4Bf16ToktileThreads>>>(
            out, weight, x, in_dim, out_dim);
        if (!cuda_ok(cudaGetLastError(),
                     "matmul_bf16 token-tile32 launch")) return 0;
        first = chunks32 * 32u;
    }
    static const int tail25_disabled =
        getenv("DS4_ROCM_DISABLE_BF16_TAIL25_FUSION") != NULL;
    if (!tail25_disabled && n_tok - first == 25u) {
        matmul_bf16_f32_toktile_w32_kernel<25u><<<
                out_dim, kDs4Bf16ToktileThreads>>>(
            out + (uint64_t)first * out_dim, weight,
            x + (uint64_t)first * in_dim, in_dim, out_dim);
        if (!cuda_ok(cudaGetLastError(),
                     "matmul_bf16 token-tail25 launch")) return 0;
        static int tail25_reported = 0;
        if (!tail25_reported &&
            getenv("DS4_ROCM_BF16_BATCH_TOKTILE_VERBOSE") != NULL) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX
                    "BF16 F32 tail25 fusion engaged: tokens=%u "
                    "in=%u out=%u\n", n_tok, in_dim, out_dim);
            tail25_reported = 1;
        }
        return true;
    }
#define DS4_BF16_LAUNCH_TAIL(T) do {                                      \
        if (n_tok - first >= (T)) {                                       \
            matmul_bf16_f32_toktile_w32_kernel<(T)><<<                  \
                out_dim, kDs4Bf16ToktileThreads>>>(                     \
                out + (uint64_t)first * out_dim, weight,                  \
                x + (uint64_t)first * in_dim, in_dim, out_dim);           \
            if (!cuda_ok(cudaGetLastError(),                              \
                         "matmul_bf16 token-tail launch")) return 0;     \
            first += (T);                                                 \
        }                                                                 \
    } while (0)
    DS4_BF16_LAUNCH_TAIL(16u);
    DS4_BF16_LAUNCH_TAIL(8u);
    DS4_BF16_LAUNCH_TAIL(4u);
    DS4_BF16_LAUNCH_TAIL(2u);
    DS4_BF16_LAUNCH_TAIL(1u);
#undef DS4_BF16_LAUNCH_TAIL
    return first == n_tok;
}

static int matmul_bf16_f32_wmma_hilo_m256_launch(
        float *out, const uint16_t *weight, const float *x,
        uint32_t in_dim, uint32_t out_dim, uint32_t n_tok) {
    matmul_bf16_f32_wmma_hilo_m256_kernel<2u><<<
            dim3((out_dim + 31u) / 32u, (n_tok + 255u) / 256u),
            16u * 32u>>>(out, weight, x, in_dim, out_dim, n_tok);
    return cuda_ok(cudaGetLastError(),
                   "matmul_bf16 WMMA hi/lo M256 launch");
}

/* GLM-KDA-only optional dispatch.  Return -1 when the guarded candidate is
 * not applicable, 0 on a hard validation/launch failure, and 1 on launch. */
extern "C" int ds4_gpu_matmul_bf16_wmma_hilo_tensor(
        ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
        uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim,
        const ds4_gpu_tensor *x, uint64_t n_tok) {
    static const int batch_toktile_disabled =
        getenv("DS4_ROCM_DISABLE_BF16_BATCH_TOKTILE") != NULL;
    if (in_dim > UINT32_MAX || out_dim > UINT32_MAX || n_tok > UINT32_MAX ||
        !ds4_bf16_wmma_hilo_dispatch_allowed(
            true, batch_toktile_disabled, (uint32_t)in_dim,
            (uint32_t)out_dim, (uint32_t)n_tok, g_quality_mode,
            cuda_runtime_config()->graph_dump)) return -1;
    if (!out || !x || !out->ptr || !x->ptr || !model_map ||
        out_dim > UINT64_MAX / in_dim) return 0;
    const uint64_t weight_elems = out_dim * in_dim;
    if (weight_elems > UINT64_MAX / sizeof(uint16_t) ||
        n_tok > UINT64_MAX / in_dim || n_tok > UINT64_MAX / out_dim)
        return 0;
    const uint64_t weight_bytes = weight_elems * sizeof(uint16_t);
    const uint64_t x_elems = n_tok * in_dim;
    const uint64_t out_elems = n_tok * out_dim;
    if (x_elems > UINT64_MAX / sizeof(float) ||
        out_elems > UINT64_MAX / sizeof(float) ||
        weight_offset > model_size ||
        weight_bytes > model_size - weight_offset ||
        x->bytes < x_elems * sizeof(float) ||
        out->bytes < out_elems * sizeof(float) ||
        cuda_u64_ranges_overlap(
            (uint64_t)(uintptr_t)out->ptr, out_elems * sizeof(float),
            (uint64_t)(uintptr_t)x->ptr, x_elems * sizeof(float))) return 0;
    const char *wptr = cuda_model_range_ptr(
        model_map, weight_offset, weight_bytes, "glm5_bf16_wmma_hilo");
    if (!wptr) return 0;
    return matmul_bf16_f32_wmma_hilo_m256_launch(
        (float *)out->ptr, (const uint16_t *)wptr, (const float *)x->ptr,
        (uint32_t)in_dim, (uint32_t)out_dim, (uint32_t)n_tok);
}

static int matmul_bf16_f32_rowtile2x16_w32_launch(
        float *out, const uint16_t *weight, const float *x,
        uint32_t in_dim, uint32_t out_dim, uint32_t n_tok) {
    if (n_tok == 0u || (n_tok % 16u) != 0u) return 0;
    matmul_bf16_f32_rowtile_w32_kernel<2u, 16u><<<
            dim3((out_dim + 1u) / 2u, n_tok / 16u),
            kDs4Bf16ToktileThreads>>>(
        out, weight, x, in_dim, out_dim);
    return cuda_ok(cudaGetLastError(),
                   "matmul_bf16 adjacent-row tile2x16 launch");
}

static int matmul_bf16_f32_skinny_exact_toktile_launch(
        float *out, const uint16_t *weight, const float *x,
        uint32_t in_dim, uint32_t out_dim, uint32_t n_tok) {
    uint32_t first = 0u;
    const uint32_t chunks32 = n_tok / 32u;
    if (chunks32 > 0u) {
        matmul_bf16_f32_skinny_exact_toktile_kernel<32u><<<
                dim3(out_dim, chunks32), kDs4Bf16ToktileThreads>>>(
            out, weight, x, in_dim, out_dim);
        if (!cuda_ok(cudaGetLastError(),
                     "matmul_bf16 skinny exact tile32 launch")) return 0;
        first = chunks32 * 32u;
    }
#define DS4_BF16_SKINNY_EXACT_LAUNCH_TAIL(T) do {                       \
        if (n_tok - first >= (T)) {                                     \
            matmul_bf16_f32_skinny_exact_toktile_kernel<(T)><<<         \
                out_dim, kDs4Bf16ToktileThreads>>>(                     \
                out + (uint64_t)first * out_dim, weight,                \
                x + (uint64_t)first * in_dim, in_dim, out_dim);         \
            if (!cuda_ok(cudaGetLastError(),                            \
                         "matmul_bf16 skinny exact tail launch"))      \
                return 0;                                               \
            first += (T);                                               \
        }                                                               \
    } while (0)
    DS4_BF16_SKINNY_EXACT_LAUNCH_TAIL(16u);
    DS4_BF16_SKINNY_EXACT_LAUNCH_TAIL(8u);
    DS4_BF16_SKINNY_EXACT_LAUNCH_TAIL(4u);
    DS4_BF16_SKINNY_EXACT_LAUNCH_TAIL(2u);
    DS4_BF16_SKINNY_EXACT_LAUNCH_TAIL(1u);
#undef DS4_BF16_SKINNY_EXACT_LAUNCH_TAIL
    return first == n_tok;
}

static int matmul_bf16_f32_kslice_toktile_w32_launch(
        float *out, const uint16_t *weight, const float *x,
        uint32_t full_in_dim, uint32_t k_off, uint32_t k_cnt,
        uint32_t out_dim, uint32_t n_tok) {
    uint32_t first = 0u;
    const uint32_t chunks32 = n_tok / 32u;
    if (chunks32 > 0u) {
        matmul_bf16_f32_kslice_toktile_w32_kernel<32u><<<
                dim3(out_dim, chunks32), kDs4Bf16ToktileThreads>>>(
            out, weight, x, full_in_dim, k_off, k_cnt, out_dim);
        if (!cuda_ok(cudaGetLastError(),
                     "matmul_bf16 kslice token-tile32 launch")) return 0;
        first = chunks32 * 32u;
    }
#define DS4_BF16_KSLICE_LAUNCH_TAIL(T) do {                              \
        if (n_tok - first >= (T)) {                                      \
            matmul_bf16_f32_kslice_toktile_w32_kernel<(T)><<<            \
                out_dim, kDs4Bf16ToktileThreads>>>(                       \
                out + (uint64_t)first * out_dim, weight,                 \
                x + (uint64_t)first * k_cnt, full_in_dim, k_off, k_cnt,  \
                out_dim);                                                \
            if (!cuda_ok(cudaGetLastError(),                              \
                         "matmul_bf16 kslice token-tail launch"))        \
                return 0;                                                \
            first += (T);                                                \
        }                                                                \
    } while (0)
    DS4_BF16_KSLICE_LAUNCH_TAIL(16u);
    DS4_BF16_KSLICE_LAUNCH_TAIL(8u);
    DS4_BF16_KSLICE_LAUNCH_TAIL(4u);
    DS4_BF16_KSLICE_LAUNCH_TAIL(2u);
    DS4_BF16_KSLICE_LAUNCH_TAIL(1u);
#undef DS4_BF16_KSLICE_LAUNCH_TAIL
    return first == n_tok;
}

static int matmul_bf16_f32_lowrank128_toktile_w32_launch(
        float *out, const uint16_t *weight, const float *x,
        uint32_t out_dim, uint32_t n_tok) {
    uint32_t first = 0u;
    const uint32_t chunks32 = n_tok / 32u;
    if (chunks32 > 0u) {
        matmul_bf16_f32_lowrank128_toktile_w32_kernel<32u><<<
                dim3(out_dim, chunks32), 32u>>>(
            out, weight, x, out_dim);
        if (!cuda_ok(cudaGetLastError(),
                     "matmul_bf16 low-rank token-tile32 launch")) return 0;
        first = chunks32 * 32u;
    }
#define DS4_BF16_LOWRANK_LAUNCH_TAIL(T) do {                            \
        if (n_tok - first >= (T)) {                                     \
            matmul_bf16_f32_lowrank128_toktile_w32_kernel<(T)><<<       \
                out_dim, 32u>>>(                                        \
                out + (uint64_t)first * out_dim, weight,                \
                x + (uint64_t)first * 128u, out_dim);                   \
            if (!cuda_ok(cudaGetLastError(),                            \
                         "matmul_bf16 low-rank token-tail launch"))     \
                return 0;                                               \
            first += (T);                                               \
        }                                                               \
    } while (0)
    DS4_BF16_LOWRANK_LAUNCH_TAIL(16u);
    DS4_BF16_LOWRANK_LAUNCH_TAIL(8u);
    DS4_BF16_LOWRANK_LAUNCH_TAIL(4u);
    DS4_BF16_LOWRANK_LAUNCH_TAIL(2u);
    DS4_BF16_LOWRANK_LAUNCH_TAIL(1u);
#undef DS4_BF16_LOWRANK_LAUNCH_TAIL
    return first == n_tok;
}

extern "C" int ds4_gpu_matmul_bf16_tensor(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim, const ds4_gpu_tensor *x, uint64_t n_tok) {
    if (!out || !x || !model_map || in_dim == 0 || out_dim == 0 || n_tok == 0) return 0;
    if (weight_offset > model_size || out_dim > UINT64_MAX / in_dim) return 0;
    const uint64_t weight_elems = out_dim * in_dim;
    if (weight_elems > UINT64_MAX / sizeof(uint16_t)) return 0;
    const uint64_t weight_bytes = weight_elems * sizeof(uint16_t);
    if (weight_bytes > model_size - weight_offset ||
        n_tok > UINT64_MAX / in_dim ||
        n_tok * in_dim > UINT64_MAX / sizeof(float) ||
        x->bytes < n_tok * in_dim * sizeof(float) ||
        n_tok > UINT64_MAX / out_dim ||
        n_tok * out_dim > UINT64_MAX / sizeof(float) ||
        out->bytes < n_tok * out_dim * sizeof(float)) return 0;
    const char *wptr = cuda_model_range_ptr(model_map, weight_offset, weight_bytes, "bf16");
    if (!wptr) return 0;
    if (n_tok == 1u && in_dim <= 8192u &&
        in_dim * sizeof(float) <= 65536u &&
        getenv("DS4_ROCM_DISABLE_BF16_SHAREDX") == NULL) {
        static const char *decode_qkv_prefetch_value =
            getenv("DS4_ROCM_GLM5_BF16_QKV_PREFETCH");
        static const char *decode_qkv_rows_value =
            getenv("DS4_ROCM_GLM5_BF16_QKV_ROWS_PER_BLOCK");
        static const char *decode_qkv_nontemporal_value =
            getenv("DS4_ROCM_GLM5_BF16_QKV_NONTEMPORAL");
        static const char *decode_output_prefetch_value =
            getenv("DS4_ROCM_GLM5_BF16_OUTPUT_PREFETCH");
        static const char *decode_output_rows_value =
            getenv("DS4_ROCM_GLM5_BF16_OUTPUT_ROWS_PER_BLOCK");
        static const char *decode_output_nontemporal_value =
            getenv("DS4_ROCM_GLM5_BF16_OUTPUT_NONTEMPORAL");
        static const uint32_t decode_qkv_prefetch =
            decode_qkv_prefetch_value == NULL ? 0u :
            strcmp(decode_qkv_prefetch_value, "0") == 0 ? 0u :
            strcmp(decode_qkv_prefetch_value, "4") == 0 ? 4u :
            strcmp(decode_qkv_prefetch_value, "8") == 0 ? 8u :
            strcmp(decode_qkv_prefetch_value, "16") == 0 ? 16u :
            strcmp(decode_qkv_prefetch_value, "32") == 0 ? 32u :
            strcmp(decode_qkv_prefetch_value, "64") == 0 ? 64u : UINT32_MAX;
        static const int decode_qkv_prefetch_valid =
            decode_qkv_prefetch != UINT32_MAX;
        static const int decode_qkv_nontemporal =
            decode_qkv_nontemporal_value != NULL &&
            strcmp(decode_qkv_nontemporal_value, "1") == 0;
        static const int decode_qkv_nontemporal_valid =
            decode_qkv_nontemporal_value == NULL ||
            strcmp(decode_qkv_nontemporal_value, "0") == 0 ||
            decode_qkv_nontemporal;
        static const uint32_t decode_qkv_rows_per_block =
            decode_qkv_rows_value == NULL ? 32u :
            strcmp(decode_qkv_rows_value, "8") == 0 ? 8u :
            strcmp(decode_qkv_rows_value, "16") == 0 ? 16u :
            strcmp(decode_qkv_rows_value, "32") == 0 ? 32u : 0u;
        static const uint32_t decode_output_prefetch =
            decode_output_prefetch_value == NULL ? 0u :
            strcmp(decode_output_prefetch_value, "0") == 0 ? 0u :
            strcmp(decode_output_prefetch_value, "4") == 0 ? 4u :
            strcmp(decode_output_prefetch_value, "8") == 0 ? 8u :
            strcmp(decode_output_prefetch_value, "16") == 0 ? 16u :
            strcmp(decode_output_prefetch_value, "32") == 0 ? 32u :
            strcmp(decode_output_prefetch_value, "64") == 0 ? 64u :
            UINT32_MAX;
        static const int decode_output_prefetch_valid =
            decode_output_prefetch != UINT32_MAX;
        static const int decode_output_nontemporal =
            decode_output_nontemporal_value != NULL &&
            strcmp(decode_output_nontemporal_value, "1") == 0;
        static const int decode_output_nontemporal_valid =
            decode_output_nontemporal_value == NULL ||
            strcmp(decode_output_nontemporal_value, "0") == 0 ||
            decode_output_nontemporal;
        static const uint32_t decode_output_rows_per_block =
            decode_output_rows_value == NULL ? 32u :
            strcmp(decode_output_rows_value, "8") == 0 ? 8u :
            strcmp(decode_output_rows_value, "16") == 0 ? 16u :
            strcmp(decode_output_rows_value, "32") == 0 ? 32u : 0u;
        if (!decode_qkv_prefetch_valid || !decode_qkv_nontemporal_valid ||
            (decode_qkv_nontemporal && decode_qkv_prefetch == 0u) ||
            decode_qkv_rows_per_block == 0u ||
            !decode_output_prefetch_valid ||
            !decode_output_nontemporal_valid ||
            (decode_output_nontemporal && decode_output_prefetch == 0u) ||
            decode_output_rows_per_block == 0u) {
            static int invalid_reported;
            if (!invalid_reported) {
                fprintf(stderr, DS4_GPU_LOG_PREFIX
                        "invalid GLM5 BF16 exact-order research selector\n");
                invalid_reported = 1;
            }
            return 0;
        }
        const int decode_qkv_shape = in_dim == 4096u && out_dim == 4096u;
        const int decode_output_shape = in_dim == 8192u && out_dim == 2048u;
        const uint32_t decode_exact_prefetch = decode_qkv_shape
            ? decode_qkv_prefetch
            : decode_output_shape ? decode_output_prefetch : 0u;
        const int decode_exact_nontemporal = decode_qkv_shape
            ? decode_qkv_nontemporal
            : decode_output_shape ? decode_output_nontemporal : 0;
        const uint32_t rows_per_block = decode_qkv_shape
            ? decode_qkv_rows_per_block
            : decode_output_shape ? decode_output_rows_per_block : 32u;
        static const int decode_mlp64_disabled =
            getenv("DS4_ROCM_DISABLE_BF16_DECODE_MLP64") != NULL;
        const dim3 grid(
            ((unsigned)out_dim + rows_per_block - 1u) / rows_per_block);
        const dim3 block(rows_per_block * 32u);
        const size_t shared_bytes = (size_t)in_dim * sizeof(float);
        if ((decode_qkv_shape || decode_output_shape) &&
            decode_exact_prefetch != 0u) {
#define DS4_LAUNCH_BF16_QKV_PREFETCH(P, NT) \
            matmul_bf16_f32_sharedx_exact_prefetch_warp_rows_w32_kernel< \
                    P, NT><<<grid, block, shared_bytes>>>( \
                    (float *)out->ptr, (const uint16_t *)wptr, \
                    (const float *)x->ptr, (uint32_t)in_dim, out_dim)
            if (decode_exact_prefetch == 4u) {
                if (decode_exact_nontemporal)
                    DS4_LAUNCH_BF16_QKV_PREFETCH(4u, true);
                else DS4_LAUNCH_BF16_QKV_PREFETCH(4u, false);
            } else if (decode_exact_prefetch == 8u) {
                if (decode_exact_nontemporal)
                    DS4_LAUNCH_BF16_QKV_PREFETCH(8u, true);
                else DS4_LAUNCH_BF16_QKV_PREFETCH(8u, false);
            } else if (decode_exact_prefetch == 16u) {
                if (decode_exact_nontemporal)
                    DS4_LAUNCH_BF16_QKV_PREFETCH(16u, true);
                else DS4_LAUNCH_BF16_QKV_PREFETCH(16u, false);
            } else if (decode_exact_prefetch == 32u) {
                if (decode_exact_nontemporal)
                    DS4_LAUNCH_BF16_QKV_PREFETCH(32u, true);
                else DS4_LAUNCH_BF16_QKV_PREFETCH(32u, false);
            } else {
                if (decode_exact_nontemporal)
                    DS4_LAUNCH_BF16_QKV_PREFETCH(64u, true);
                else DS4_LAUNCH_BF16_QKV_PREFETCH(64u, false);
            }
#undef DS4_LAUNCH_BF16_QKV_PREFETCH
            static int qkv_prefetch_reported;
            static int output_prefetch_reported;
            int *prefetch_reported = decode_qkv_shape
                ? &qkv_prefetch_reported : &output_prefetch_reported;
            if (!*prefetch_reported) {
                fprintf(stderr, DS4_GPU_LOG_PREFIX
                        "GLM5 BF16 %s exact prefetch engaged "
                        "window=%u rows_per_block=%u nontemporal=%d\n",
                        decode_qkv_shape ? "QKV" : "output",
                        decode_exact_prefetch, rows_per_block,
                        decode_exact_nontemporal);
                *prefetch_reported = 1;
            }
        } else if (!decode_mlp64_disabled && in_dim >= 4096u) {
            static const char *split_order_value =
                getenv("DS4_ROCM_BF16_FULL_SPLIT_ORDER");
            static const char *legacy_split_order_value =
                getenv("DS4_ROCM_BF16_FULL_LEGACY_SPLIT_ORDER");
            static const int split_order = split_order_value != NULL &&
                strcmp(split_order_value, "1") == 0;
            static const int legacy_split_order =
                legacy_split_order_value != NULL &&
                strcmp(legacy_split_order_value, "1") == 0;
            static const int split_order_env_valid =
                (split_order_value == NULL ||
                 strcmp(split_order_value, "0") == 0 || split_order) &&
                (legacy_split_order_value == NULL ||
                 strcmp(legacy_split_order_value, "0") == 0 ||
                 legacy_split_order) &&
                !(split_order && legacy_split_order);
            if (!split_order_env_valid) {
                static int invalid_reported;
                if (!invalid_reported) {
                    fprintf(stderr, DS4_GPU_LOG_PREFIX
                            "invalid or conflicting BF16 full split-order "
                            "diagnostic selectors\n");
                    invalid_reported = 1;
                }
                return 0;
            }
            if (split_order && in_dim == 8192u) {
                matmul_bf16_f32_sharedx_mlp64_split4096_warp_rows_w32_kernel<<<
                        grid, block, shared_bytes>>>(
                        (float *)out->ptr,
                        (const uint16_t *)wptr,
                        (const float *)x->ptr,
                        out_dim);
            } else if (legacy_split_order && in_dim == 8192u) {
                /* Independent lawful-reorder null.  Keep the historical
                 * two-accumulator source path available under a distinct
                 * diagnostic selector: the repaired split-order control above
                 * intentionally shares its exact half-reduction body with the
                 * TP K-slice and therefore cannot calibrate recurrent
                 * sensitivity independently. */
                matmul_bf16_f32_sharedx_mlp64_warp_rows_w32_kernel<<<
                        grid, block, shared_bytes>>>(
                        (float *)out->ptr,
                        (const uint16_t *)wptr,
                        (const float *)x->ptr,
                        (uint32_t)in_dim,
                        out_dim,
                        1);
            } else {
                matmul_bf16_f32_sharedx_mlp64_warp_rows_w32_kernel<<<
                        grid, block, shared_bytes>>>(
                        (float *)out->ptr,
                        (const uint16_t *)wptr,
                        (const float *)x->ptr,
                        (uint32_t)in_dim,
                        out_dim,
                        0);
            }
        } else {
            matmul_bf16_f32_sharedx_warp_rows_w32_kernel<<<
                    grid, block, shared_bytes>>>(
                    (float *)out->ptr,
                    (const uint16_t *)wptr,
                    (const float *)x->ptr,
                    (uint32_t)in_dim,
                    out_dim);
        }
        return cuda_ok(cudaGetLastError(), "matmul_bf16 sharedx launch");
    }
    /* Launcher/test contract: rollback variables are set before process start,
     * so cache the environment lookup on the first BF16 projection. */
    static const int batch_toktile_disabled =
        getenv("DS4_ROCM_DISABLE_BF16_BATCH_TOKTILE") != NULL;
    static const int lowrank128_toktile_disabled =
        getenv("DS4_ROCM_DISABLE_BF16_LOWRANK128_TOKTILE") != NULL;
    static const char *rowtile2x16_value =
        getenv("DS4_ROCM_GLM5_BF16_ROWTILE2X16");
    static const int rowtile2x16_enabled = rowtile2x16_value != NULL &&
        strcmp(rowtile2x16_value, "1") == 0;
    static const int rowtile2x16_valid = rowtile2x16_value == NULL ||
        strcmp(rowtile2x16_value, "0") == 0 || rowtile2x16_enabled;
    if (!rowtile2x16_valid) {
        static int invalid_reported;
        if (!invalid_reported) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX
                    "invalid GLM5 BF16 adjacent-row tile selector\n");
            invalid_reported = 1;
        }
        return 0;
    }
    const bool rowtile2x16_shape =
        in_dim <= UINT32_MAX && out_dim <= UINT32_MAX &&
        n_tok <= UINT32_MAX &&
        ds4_bf16_rowtile2x16_dispatch_allowed(
            rowtile2x16_enabled, batch_toktile_disabled,
            (uint32_t)in_dim, (uint32_t)out_dim, (uint32_t)n_tok,
            g_quality_mode, cuda_runtime_config()->graph_dump);
    if (rowtile2x16_shape) {
        static int rowtile2x16_reported;
        if (!rowtile2x16_reported) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX
                    "GLM5 BF16 adjacent-row tile2x16 engaged: "
                    "tokens=%llu in=%llu out=%llu\n",
                    (unsigned long long)n_tok,
                    (unsigned long long)in_dim,
                    (unsigned long long)out_dim);
            rowtile2x16_reported = 1;
        }
        return matmul_bf16_f32_rowtile2x16_w32_launch(
            (float *)out->ptr, (const uint16_t *)wptr,
            (const float *)x->ptr, (uint32_t)in_dim,
            (uint32_t)out_dim, (uint32_t)n_tok);
    }
    static const char *skinny_exact_value =
        getenv("DS4_ROCM_GLM5_BF16_SKINNY_EXACT_TOKTILE");
    static const int skinny_exact_enabled =
        skinny_exact_value != NULL && strcmp(skinny_exact_value, "1") == 0;
    static const int skinny_exact_valid =
        skinny_exact_value == NULL || strcmp(skinny_exact_value, "0") == 0 ||
        skinny_exact_enabled;
    if (!skinny_exact_valid) {
        static int invalid_reported;
        if (!invalid_reported) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX
                    "invalid GLM5 BF16 skinny exact token-tile selector\n");
            invalid_reported = 1;
        }
        return 0;
    }
    const bool skinny_exact_shape = skinny_exact_enabled &&
        ((in_dim == 4096u && (out_dim == 32u || out_dim == 128u)) ||
         (in_dim == 16384u && out_dim == 24u));
    if (skinny_exact_shape && n_tok >= 16u && in_dim <= UINT32_MAX &&
        out_dim <= UINT32_MAX && n_tok <= UINT32_MAX && !g_quality_mode &&
        !cuda_runtime_config()->graph_dump) {
        static int skinny_exact_reported;
        if (!skinny_exact_reported) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX
                    "GLM5 BF16 skinny exact token-tile engaged: "
                    "tokens=%llu in=%llu out=%llu\n",
                    (unsigned long long)n_tok,
                    (unsigned long long)in_dim,
                    (unsigned long long)out_dim);
            skinny_exact_reported = 1;
        }
        return matmul_bf16_f32_skinny_exact_toktile_launch(
            (float *)out->ptr, (const uint16_t *)wptr,
            (const float *)x->ptr, (uint32_t)in_dim,
            (uint32_t)out_dim, (uint32_t)n_tok);
    }
    /* Full KDA expands 128 -> 8192.  True TP computes one contiguous head
     * half at a time, so the same row-major kernel sees 128 -> 4096 after
     * advancing wptr to rank 1's first row. */
    const bool low_rank_128 = !lowrank128_toktile_disabled &&
        in_dim == 128u && (out_dim == 8192u || out_dim == 4096u);
    if (!batch_toktile_disabled && n_tok >= 16u &&
        ((in_dim >= 1024u && out_dim >= 1024u) || low_rank_128) &&
        in_dim <= UINT32_MAX && out_dim <= UINT32_MAX &&
        n_tok <= UINT32_MAX && !g_quality_mode &&
        !cuda_runtime_config()->graph_dump) {
        static int verbose_reported = 0;
        if (!verbose_reported &&
            getenv("DS4_ROCM_BF16_BATCH_TOKTILE_VERBOSE") != NULL) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX
                    "BF16 F32 token-tile path engaged: tokens=%llu "
                    "in=%llu out=%llu\n",
                    (unsigned long long)n_tok,
                    (unsigned long long)in_dim,
                    (unsigned long long)out_dim);
            verbose_reported = 1;
        }
        static int low_rank_reported = 0;
        if (!low_rank_reported && low_rank_128 &&
            getenv("DS4_ROCM_BF16_BATCH_TOKTILE_VERBOSE") != NULL) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX
                    "BF16 F32 low-rank token-tile path engaged: tokens=%llu "
                    "in=%llu out=%llu\n",
                    (unsigned long long)n_tok,
                    (unsigned long long)in_dim,
                    (unsigned long long)out_dim);
            low_rank_reported = 1;
        }
        if (low_rank_128) {
            return matmul_bf16_f32_lowrank128_toktile_w32_launch(
                (float *)out->ptr, (const uint16_t *)wptr,
                (const float *)x->ptr, (uint32_t)out_dim,
                (uint32_t)n_tok);
        }
        return matmul_bf16_f32_toktile_w32_launch(
            (float *)out->ptr, (const uint16_t *)wptr,
            (const float *)x->ptr, (uint32_t)in_dim,
            (uint32_t)out_dim, (uint32_t)n_tok);
    }
    const dim3 grid((unsigned)out_dim, (unsigned)n_tok, 1);
    matmul_bf16_kernel<<<grid, 256>>>((float *)out->ptr,
                                      (const uint16_t *)wptr,
                                      (const float *)x->ptr,
                                      in_dim, out_dim, n_tok);
    return cuda_ok(cudaGetLastError(), "matmul_bf16 launch");
}

extern "C" int ds4_gpu_matmul_bf16_kslice_rows_tensor(
        ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
        uint64_t weight_offset, uint64_t full_in_dim, uint64_t out_dim,
        uint64_t k_off, uint64_t k_cnt, const ds4_gpu_tensor *x,
        uint64_t n_rows) {
    if (!out || !x || !model_map || full_in_dim == 0u || out_dim == 0u ||
        k_cnt == 0u || n_rows == 0u || full_in_dim > UINT32_MAX ||
        out_dim > UINT32_MAX || k_off > UINT32_MAX || k_cnt > UINT32_MAX ||
        n_rows > UINT32_MAX || (k_off & 31u) != 0u ||
        (k_cnt & 31u) != 0u || k_off > full_in_dim ||
        k_cnt > full_in_dim - k_off) return 0;
    uint64_t weight_elems = 0u, x_elems = 0u, out_elems = 0u;
    if (weight_offset > model_size ||
        !cuda_u64_mul_checked(out_dim, full_in_dim, &weight_elems) ||
        weight_elems > UINT64_MAX / sizeof(uint16_t) ||
        weight_elems * sizeof(uint16_t) > model_size - weight_offset ||
        !cuda_u64_mul_checked(n_rows, k_cnt, &x_elems) ||
        x_elems > UINT64_MAX / sizeof(float) ||
        x->bytes < x_elems * sizeof(float) ||
        !cuda_u64_mul_checked(n_rows, out_dim, &out_elems) ||
        out_elems > UINT64_MAX / sizeof(float) ||
        out->bytes < out_elems * sizeof(float) ||
        cuda_u64_ranges_overlap(
            (uint64_t)(uintptr_t)out->ptr, out_elems * sizeof(float),
            (uint64_t)(uintptr_t)x->ptr, x_elems * sizeof(float))) return 0;
    const uint64_t weight_bytes = weight_elems * sizeof(uint16_t);
    const char *wptr = cuda_model_range_ptr(
        model_map, weight_offset, weight_bytes, "bf16_kslice");
    if (!wptr) return 0;
    const uint32_t full = (uint32_t)full_in_dim;
    const uint32_t off = (uint32_t)k_off;
    const uint32_t count = (uint32_t)k_cnt;
    const uint32_t rows = (uint32_t)out_dim;
    const uint32_t tokens = (uint32_t)n_rows;
    /* Diagnostic/promotion lane: use one reduction tree for decode and
     * batched rows so route comparisons isolate KDA recurrence rather than
     * comparing the shared-x decode tree with the token-tile tree. */
    static const char *route_toktile_value =
        getenv("DS4_ROCM_KSLICE_ROUTE_TOKTILE");
    static const int route_toktile = route_toktile_value != NULL &&
        strcmp(route_toktile_value, "1") == 0;
    if (tokens == 1u && count <= 8192u &&
        count * sizeof(float) <= 65536u && !route_toktile) {
        const uint32_t rows_per_block = 32u;
        static const int compensated_reduce =
            getenv("DS4_ROCM_KSLICE_COMPENSATED_REDUCE") != NULL;
        static const int pair_accum =
            getenv("DS4_ROCM_KSLICE_PAIR_ACCUM") != NULL;
        static const int kahan_accum =
            getenv("DS4_ROCM_KSLICE_KAHAN_ACCUM") != NULL;
        static const int exact_half_reduce =
            getenv("DS4_ROCM_DISABLE_KSLICE_EXACT_HALF_REDUCE") == NULL;
        if (exact_half_reduce && count == 4096u && !compensated_reduce &&
            !pair_accum && !kahan_accum) {
            matmul_bf16_f32_sharedx_mlp64_kslice_exact4096_warp_rows_w32_kernel<<<
                    (rows + rows_per_block - 1u) / rows_per_block,
                    rows_per_block * 32u,
                    (size_t)count * sizeof(float)>>>(
                (float *)out->ptr, (const uint16_t *)wptr,
                (const float *)x->ptr, full, off, rows);
            return cuda_ok(
                cudaGetLastError(),
                "matmul_bf16 kslice exact4096 sharedx launch");
        }
        matmul_bf16_f32_sharedx_mlp64_kslice_warp_rows_w32_kernel<<<
                (rows + rows_per_block - 1u) / rows_per_block,
                rows_per_block * 32u,
                (size_t)count * sizeof(float)>>>(
            (float *)out->ptr, (const uint16_t *)wptr,
            (const float *)x->ptr, full, off, count, rows,
            compensated_reduce, pair_accum, kahan_accum);
        return cuda_ok(cudaGetLastError(),
                       "matmul_bf16 kslice sharedx launch");
    }
    if ((tokens >= 16u || route_toktile) &&
        count >= 1024u && rows >= 1024u &&
        !g_quality_mode && !cuda_runtime_config()->graph_dump) {
        return matmul_bf16_f32_kslice_toktile_w32_launch(
            (float *)out->ptr, (const uint16_t *)wptr,
            (const float *)x->ptr, full, off, count, rows, tokens);
    }
    matmul_bf16_kslice_kernel<<<dim3(rows, tokens), 256u>>>(
        (float *)out->ptr, (const uint16_t *)wptr,
        (const float *)x->ptr, full, off, count, rows, tokens);
    return cuda_ok(cudaGetLastError(), "matmul_bf16 kslice launch");
}

extern "C" int ds4_gpu_matmul_bf16_kslice_tensor(
        ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
        uint64_t weight_offset, uint64_t full_in_dim, uint64_t k_off,
        uint64_t k_cnt, uint64_t out_dim, const ds4_gpu_tensor *x,
        uint64_t x_elem_off) {
    if (!x || x_elem_off > x->bytes / sizeof(float) ||
        k_cnt > x->bytes / sizeof(float) - x_elem_off) return 0;
    ds4_gpu_tensor compact = *x;
    compact.ptr = (char *)x->ptr + x_elem_off * sizeof(float);
    compact.bytes = k_cnt * sizeof(float);
    compact.owner = 0;
    return ds4_gpu_matmul_bf16_kslice_rows_tensor(
        out, model_map, model_size, weight_offset, full_in_dim, out_dim,
        k_off, k_cnt, &compact, 1u);
}

extern "C" int ds4_gpu_matmul_f16_pair_temporal_tensor(
        ds4_gpu_tensor *out0,
        ds4_gpu_tensor *out1,
        const void *model_map,
        uint64_t model_size,
        uint64_t weight0_offset,
        uint64_t weight1_offset,
        uint64_t in_dim,
        uint64_t out_dim,
        const ds4_gpu_tensor *x,
        uint64_t n_tok) {
    if (!out0 || !out1 || !x || !model_map ||
        in_dim > UINT32_MAX || out_dim > UINT32_MAX || n_tok > UINT32_MAX) {
        return 0;
    }
    const int temporal_shape = n_tok >= 2u && n_tok <= 4u && in_dim == 4096u &&
        (out_dim == 256u || out_dim == 512u || out_dim == 1024u);
    if (!temporal_shape || !f16_pair_temporal_supported()) return 0;
    uint64_t weight_bytes = 0, x_bytes = 0, out_bytes = 0;
    if (weight0_offset > model_size || weight1_offset > model_size ||
        !cuda_u64_mul3_checked(out_dim, in_dim, sizeof(uint16_t), &weight_bytes) ||
        !cuda_u64_mul3_checked(n_tok, in_dim, sizeof(float), &x_bytes) ||
        !cuda_u64_mul3_checked(n_tok, out_dim, sizeof(float), &out_bytes) ||
        weight_bytes > model_size - weight0_offset ||
        weight_bytes > model_size - weight1_offset ||
        x->bytes < x_bytes || out0->bytes < out_bytes || out1->bytes < out_bytes) {
        return 0;
    }
    const __half *w0 = (const __half *)cuda_model_range_ptr(
            model_map, weight0_offset, weight_bytes, "f16_pair_temporal_0");
    const __half *w1 = (const __half *)cuda_model_range_ptr(
            model_map, weight1_offset, weight_bytes, "f16_pair_temporal_1");
    if (!w0 || !w1) return 0;
    const uint32_t rows_per_block = compressor_proj_rows_per_block();
    const dim3 grid(((uint32_t)out_dim + rows_per_block - 1u) / rows_per_block);
    const dim3 block(rows_per_block * 32u);
    const size_t shared_bytes = (size_t)n_tok * (size_t)in_dim * sizeof(float);
    switch ((uint32_t)n_tok) {
        case 2u:
            matmul_f16_pair_f32_temporal_rows_w32_kernel<2u><<<grid, block, shared_bytes>>>(
                    (float *)out0->ptr, (float *)out1->ptr, w0, w1,
                    (const float *)x->ptr, (uint32_t)in_dim, (uint32_t)out_dim);
            break;
        case 3u:
            matmul_f16_pair_f32_temporal_rows_w32_kernel<3u><<<grid, block, shared_bytes>>>(
                    (float *)out0->ptr, (float *)out1->ptr, w0, w1,
                    (const float *)x->ptr, (uint32_t)in_dim, (uint32_t)out_dim);
            break;
        case 4u:
            matmul_f16_pair_f32_temporal_rows_w32_kernel<4u><<<grid, block, shared_bytes>>>(
                    (float *)out0->ptr, (float *)out1->ptr, w0, w1,
                    (const float *)x->ptr, (uint32_t)in_dim, (uint32_t)out_dim);
            break;
        default:
            return 0;
    }
    return cuda_ok(cudaGetLastError(), "matmul_f16_pair temporal launch");
}

extern "C" int ds4_gpu_matmul_f16_pair_tensor(
        ds4_gpu_tensor *out0,
        ds4_gpu_tensor *out1,
        const void *model_map,
        uint64_t model_size,
        uint64_t weight0_offset,
        uint64_t weight1_offset,
        uint64_t in_dim,
        uint64_t out_dim,
        const ds4_gpu_tensor *x,
        uint64_t n_tok) {
    if (!out0 || !out1 || !x || !model_map || in_dim == 0 || out_dim == 0 || n_tok == 0 ||
        in_dim > UINT32_MAX || out_dim > UINT32_MAX || n_tok > UINT32_MAX) {
        return 0;
    }
    const int pair5_shape = n_tok == 5u && in_dim == 4096u &&
        (out_dim == 256u || out_dim == 512u || out_dim == 1024u);
    const int pair5_selected = pair5_shape && f16_pair_five_row_enabled() &&
        !g_quality_mode && !cuda_runtime_config()->graph_dump;
    if (pair5_selected) {
        uint64_t weight_bytes = 0, x_bytes = 0, out_bytes = 0;
        if (weight0_offset > model_size || weight1_offset > model_size ||
            !cuda_u64_mul3_checked(out_dim, in_dim, sizeof(uint16_t), &weight_bytes) ||
            !cuda_u64_mul3_checked(n_tok, in_dim, sizeof(float), &x_bytes) ||
            !cuda_u64_mul3_checked(n_tok, out_dim, sizeof(float), &out_bytes) ||
            weight_bytes > model_size - weight0_offset ||
            weight_bytes > model_size - weight1_offset ||
            x->bytes < x_bytes || out0->bytes < out_bytes || out1->bytes < out_bytes) {
            return 0;
        }
        const __half *w0 = (const __half *)cuda_model_range_ptr(
                model_map, weight0_offset, weight_bytes, "f16_pair5_0");
        const __half *w1 = (const __half *)cuda_model_range_ptr(
                model_map, weight1_offset, weight_bytes, "f16_pair5_1");
        if (!w0 || !w1) return 0;
        const uint64_t xh_count = n_tok * in_dim;
        __half *xh = (__half *)cuda_tmp_alloc(
                xh_count * sizeof(__half), "f16 pair five-row activations");
        if (!xh) return 0;
        f32_to_f16_kernel<<<(xh_count + 255u) / 256u, 256>>>(
                xh, (const float *)x->ptr, xh_count);
        if (!cuda_ok(cudaGetLastError(), "f16 pair five-row conversion launch")) return 0;
        const uint32_t rows_per_block = 8u;
        matmul_f16_pair_five_row_kernel<5u><<<
                ((uint32_t)out_dim + rows_per_block - 1u) / rows_per_block,
                rows_per_block * 32u>>>(
                (float *)out0->ptr, (float *)out1->ptr, w0, w1, xh,
                (uint32_t)in_dim, (uint32_t)out_dim);
        return cuda_ok(cudaGetLastError(), "f16 pair five-row launch");
    }
    if (n_tok != 1) {
        return ds4_gpu_matmul_f16_tensor(out0, model_map, model_size, weight0_offset,
                                           in_dim, out_dim, x, n_tok) &&
               ds4_gpu_matmul_f16_tensor(out1, model_map, model_size, weight1_offset,
                                           in_dim, out_dim, x, n_tok);
    }
    uint64_t weight_bytes = 0;
    if (weight0_offset > model_size || weight1_offset > model_size ||
        !cuda_u64_mul3_checked(out_dim, in_dim, sizeof(uint16_t), &weight_bytes)) {
        return 0;
    }
    if (weight_bytes > model_size - weight0_offset ||
        weight_bytes > model_size - weight1_offset ||
        x->bytes < in_dim * sizeof(float) ||
        out0->bytes < out_dim * sizeof(float) ||
        out1->bytes < out_dim * sizeof(float)) {
        return 0;
    }
    const __half *w0 = (const __half *)cuda_model_range_ptr(model_map, weight0_offset, weight_bytes, "f16_pair0");
    const __half *w1 = (const __half *)cuda_model_range_ptr(model_map, weight1_offset, weight_bytes, "f16_pair1");
    if (!w0 || !w1) return 0;
    if (!g_quality_mode && !cuda_runtime_config()->graph_dump) {
        if (in_dim <= 8192u && in_dim * sizeof(float) <= 65536u) {
            const uint32_t rows_per_block = compressor_proj_rows_per_block();
            matmul_f16_pair_f32_sharedx_warp_rows_w32_kernel<<<
                    ((unsigned)out_dim + rows_per_block - 1u) / rows_per_block,
                    rows_per_block * 32u,
                    (size_t)in_dim * sizeof(float)>>>(
                    (float *)out0->ptr, (float *)out1->ptr, w0, w1,
                    (const float *)x->ptr, (uint32_t)in_dim, out_dim);
            return cuda_ok(cudaGetLastError(), "matmul_f16_pair sharedx launch");
        }
    }
    matmul_f16_pair_ordered_chunks_kernel<<<(unsigned)out_dim, 32>>>(
        (float *)out0->ptr,
        (float *)out1->ptr,
        w0,
        w1,
        (const float *)x->ptr,
        in_dim,
        out_dim,
        out_dim);
    return cuda_ok(cudaGetLastError(), "matmul_f16_pair_ordered_chunks launch");
}

extern "C" int ds4_gpu_matmul_f16_pair_compressor_store_tensor(
        ds4_gpu_tensor *out_kv,
        ds4_gpu_tensor *out_score,
        ds4_gpu_tensor *state_kv,
        ds4_gpu_tensor *state_score,
        const void *model_map,
        uint64_t model_size,
        uint64_t weight_kv_offset,
        uint64_t weight_score_offset,
        uint64_t ape_offset,
        uint32_t ape_type,
        uint64_t in_dim,
        uint32_t width,
        const ds4_gpu_tensor *x,
        uint32_t ratio,
        uint32_t pos) {
    (void)out_kv;
    (void)out_score;
    (void)state_kv;
    (void)state_score;
    (void)model_map;
    (void)model_size;
    (void)weight_kv_offset;
    (void)weight_score_offset;
    (void)ape_offset;
    (void)ape_type;
    (void)in_dim;
    (void)width;
    (void)x;
    (void)ratio;
    (void)pos;
    return 0;
}

extern "C" int ds4_gpu_matmul_f32_tensor(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim, const ds4_gpu_tensor *x, uint64_t n_tok) {
    if (!out || !x || !model_map || in_dim == 0 || out_dim == 0 || n_tok == 0 ||
        in_dim > UINT32_MAX || out_dim > UINT32_MAX || n_tok > UINT32_MAX) return 0;
    uint64_t weight_bytes = 0, x_bytes = 0, out_bytes = 0;
    if (weight_offset > model_size ||
        !cuda_u64_mul3_checked(out_dim, in_dim, sizeof(float), &weight_bytes) ||
        weight_bytes > model_size - weight_offset ||
        !cuda_u64_mul3_checked(n_tok, in_dim, sizeof(float), &x_bytes) ||
        !cuda_u64_mul3_checked(n_tok, out_dim, sizeof(float), &out_bytes) ||
        x->bytes < x_bytes || out->bytes < out_bytes) return 0;
    const char *wptr = cuda_model_range_ptr(model_map, weight_offset, weight_bytes, "f32");
    if (!wptr) return 0;
    const float *w = (const float *)wptr;
    if (g_cublas_ready && n_tok > 1) {
        const float alpha = 1.0f;
        const float beta = 0.0f;
        cublasStatus_t st = cublasSgemm(g_cublas,
                                        CUBLAS_OP_T,
                                        CUBLAS_OP_N,
                                        (int)out_dim,
                                        (int)n_tok,
                                        (int)in_dim,
                                        &alpha,
                                        w,
                                        (int)in_dim,
                                        (const float *)x->ptr,
                                        (int)in_dim,
                                        &beta,
                                        (float *)out->ptr,
                                        (int)out_dim);
        return cublas_ok(st, "f32 matmul");
    }
    dim3 grid((unsigned)out_dim, (unsigned)n_tok, 1);
    matmul_f32_kernel<<<grid, 256>>>((float *)out->ptr, w, (const float *)x->ptr, in_dim, out_dim, n_tok);
    return cuda_ok(cudaGetLastError(), "matmul_f32 launch");
}

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
 * Decode retains the one-row shared-x path. Multi-row callers quantize only
 * the compact activation slice and reuse the existing DP4A token-tile kernel
 * in bounded eight-row chunks. Passing the full source row stride while
 * shifting the weight base to block_start preserves the packed Q8_0 layout
 * without copying or expanding weights.
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
    if (n_tok > UINT32_MAX || out_dim > UINT32_MAX ||
        in_count > UINT32_MAX) return 0;
    const uint64_t full_blocks  = (in_dim + 31u) / 32u;
    const uint64_t block_start  = in_start / 32u;
    const uint64_t slice_blocks = in_count / 32u;
    const uint64_t row_bytes    = full_blocks * 34u;
    uint64_t weight_bytes = 0;
    if (weight_offset > model_size ||
        !cuda_u64_mul_checked(out_dim, row_bytes, &weight_bytes) ||
        weight_bytes > model_size - weight_offset) return 0;
    uint64_t x_elements = 0u, out_elements = 0u;
    if (!cuda_u64_mul_checked(n_tok, in_count, &x_elements) ||
        !cuda_u64_mul_checked(n_tok, out_dim, &out_elements) ||
        x_elements > UINT64_MAX / sizeof(float) ||
        out_elements > UINT64_MAX / sizeof(float) ||
        x->bytes < x_elements * sizeof(float) ||
        out->bytes < out_elements * sizeof(float)) return 0;
    /* Only the one-row shared-x kernel stages the whole slice in LDS. */
    if (n_tok == 1u && in_count > 8192u) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX
                "q8_0 kslice: slice %llu exceeds the shared-x LDS budget\n",
                (unsigned long long)in_count);
        return 0;
    }
    const char *wptr = cuda_model_range_ptr(model_map, weight_offset, weight_bytes,
                                            "q8_0_kslice");
    if (!wptr) return 0;

    if (n_tok > 1u) {
        uint64_t quantized_count = 0u, scale_count = 0u;
        if (!cuda_u64_mul3_checked(n_tok, slice_blocks, 32u,
                                   &quantized_count) ||
            !cuda_u64_mul_checked(n_tok, slice_blocks, &scale_count)) {
            return 0;
        }
        const uint64_t scale_offset =
            (quantized_count + 15u) & ~UINT64_C(15);
        if (scale_count > (UINT64_MAX - scale_offset) / sizeof(float)) {
            return 0;
        }
        const uint64_t tmp_bytes =
            scale_offset + scale_count * sizeof(float);
        void *tmp = cuda_tmp_alloc(tmp_bytes, "q8_0 kslice rows prequant");
        if (!tmp) return 0;
        int8_t *xq = (int8_t *)tmp;
        float *xscale = (float *)((char *)tmp + scale_offset);
        const dim3 qgrid((uint32_t)slice_blocks, (uint32_t)n_tok, 1u);
        quantize_q8_0_f32_kernel<<<qgrid, 32>>>(
            xq, xscale, (const float *)x->ptr, in_count, slice_blocks);
        if (!cuda_ok(cudaGetLastError(),
                     "q8_0 kslice rows quantize launch")) return 0;
        const unsigned char *slice_w =
            reinterpret_cast<const unsigned char *>(wptr) +
            block_start * 34u;
        uint64_t token_base = 0u;
        while (token_base < n_tok) {
            const uint32_t rows = (uint32_t)(
                n_tok - token_base > 8u ? 8u : n_tok - token_base);
            cuda_launch_q8_small_batch_dp4a(
                (float *)out->ptr + token_base * out_dim,
                slice_w,
                xq + token_base * slice_blocks * 32u,
                xscale + token_base * slice_blocks,
                (uint32_t)slice_blocks, (uint32_t)out_dim,
                rows, row_bytes);
            if (!cuda_ok(cudaGetLastError(),
                         "q8_0 kslice rows DP4A launch")) return 0;
            token_base += rows;
        }
        return 1;
    }

    const unsigned threads = attention_output_expand_threads();
    const unsigned rows_per_block = threads / 32u;
    if (attention_output_expand_pack4_enabled() &&
        in_dim == 8192u && in_start <= 4096u &&
        (in_start % 4096u) == 0u && in_count == 4096u &&
        out_dim == 4096u && full_blocks == 256u && slice_blocks == 128u) {
        matmul_q8_0_f32_sharedx_warp_rows_w32_pack4_kernel<<<
                (unsigned)((out_dim + rows_per_block - 1u) / rows_per_block),
                threads,
                (size_t)in_count * sizeof(float)>>>(
                (float *)out->ptr,
                reinterpret_cast<const unsigned char *>(wptr) + block_start * 34u,
                (const float *)x->ptr,
                (uint32_t)slice_blocks,
                out_dim,
                row_bytes);
        return cuda_ok(cudaGetLastError(), "matmul_q8_0 kslice rows pack4 launch");
    }
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

/* GLM-5.3 MLA produces full-width head rows but each TP rank owns one
 * contiguous half. Reuse the existing strided F32xQ8 token-tile kernel so
 * prompt rows share weight loads without introducing Q8 activation rounding. */
extern "C" int ds4_rocm_q8_kslice_f32_rows_strided(
        ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
        uint64_t weight_offset, uint64_t full_in_dim, uint64_t out_dim,
        uint64_t in_start, uint64_t in_count, const ds4_gpu_tensor *x,
        uint64_t x_elem_start, uint64_t n_tokens,
        uint64_t x_token_stride) {
    if (getenv("DS4_ROCM_GLM5_DISABLE_STRIDED_F32_ROWS") != nullptr)
        return -1;
    if (!out || !x || !model_map ||
        full_in_dim == 0u || out_dim == 0u || in_count == 0u ||
        (in_start & 31u) != 0u || (in_count & 31u) != 0u ||
        in_start > full_in_dim || in_count > full_in_dim - in_start ||
        x_token_stride < x_elem_start + in_count || n_tokens > UINT32_MAX ||
        out_dim > UINT32_MAX || in_count > UINT32_MAX ||
        x_token_stride > UINT32_MAX) return 0;
    /* This entry point is a multi-row accelerator. Let callers preserve the
     * established one-row backend path instead of treating it as a failure. */
    if (n_tokens <= 1u) return -1;
    const uint64_t full_blocks = (full_in_dim + 31u) / 32u;
    const uint64_t block_start = in_start / 32u;
    const uint64_t slice_blocks = in_count / 32u;
    const uint64_t row_bytes = full_blocks * 34u;
    uint64_t weight_bytes = 0u, out_elements = 0u, last_x = 0u;
    if (weight_offset > model_size ||
        !cuda_u64_mul_checked(out_dim, row_bytes, &weight_bytes) ||
        weight_bytes > model_size - weight_offset ||
        !cuda_u64_mul_checked(n_tokens, out_dim, &out_elements) ||
        out_elements > UINT64_MAX / sizeof(float) ||
        out->bytes < out_elements * sizeof(float) ||
        !cuda_u64_mul_checked(n_tokens - 1u, x_token_stride, &last_x) ||
        last_x > UINT64_MAX - x_elem_start - in_count ||
        x->bytes < (last_x + x_elem_start + in_count) * sizeof(float)) return 0;
    const char *wptr = cuda_model_range_ptr(
        model_map, weight_offset, weight_bytes, "q8_0_kslice_strided");
    if (!wptr) return 0;
    const unsigned char *slice_w =
        reinterpret_cast<const unsigned char *>(wptr) + block_start * 34u;
    const float *slice_x = (const float *)x->ptr + x_elem_start;
    /* BT=16 and tile_m=32 consume exactly 64 KiB of dynamic LDS. Keep these
     * launch constants paired; increasing either requires a new LDS audit. */
    cuda_launch_grouped_q8_a_sharedx_strided(
        (float *)out->ptr, slice_w, slice_x, (uint32_t)n_tokens, 1u,
        (uint32_t)slice_blocks, (uint32_t)out_dim,
        (uint32_t)x_token_stride, 0u, row_bytes,
        32u, 32u, 16u);
    return cuda_ok(cudaGetLastError(),
                   "q8_0 kslice strided F32 rows launch");
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

/* ------------------------------------------------------------------------
 * DS4-TP-gfx1151 (patch 14): generic dense-quant K-slice.
 *
 * Reached from metal_graph_matmul_dense_quant_kslice (ds4.c:24738), which under
 * TP K-splits the SHARED EXPERT down projection across the two ranks
 * (ds4.c:23882, the `tp_split_shared` branch: k_off = tp_rank*(shared_dim/2),
 * k_cnt = shared_dim/2).
 *
 * Note that helper passes w->type straight through with NO Q8_0 special case,
 * so a Q8_0 tensor lands HERE rather than at the q8-specific entry point. That
 * is exactly why prefill passed and decode failed: this is the shared-expert
 * path, not the attention path. An earlier analysis concluded this stub was
 * unreachable "because attn_output is Q8_0" - true of the attention call site,
 * and irrelevant to this one.
 *
 * ffn_down_shexp is Q8_0 in both variants of this checkpoint (verified from the
 * GGUF header: shexp gate/up/down are Q8_0 x43 each), so Q8_0 delegates to the
 * patch-10 slice. Q4_K/Q4_0 are accepted by tensor_type_is_dense_quant
 * (ds4.c:4275) but are NOT implemented here - they fail closed and say so
 * rather than returning a silent 0.
 * ------------------------------------------------------------------------ */
extern "C" int ds4_gpu_matmul_quant_kslice_tensor(
        ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
        uint64_t weight_offset, uint32_t weight_type, uint64_t full_in_dim,
        uint64_t k_off, uint64_t k_cnt, uint64_t out_dim,
        const ds4_gpu_tensor *x, uint64_t x_elem_off) {
    if (weight_type == 8u) {   /* DS4_TENSOR_Q8_0 (ds4.c:2044) */
        return ds4_gpu_matmul_q8_0_kslice_tensor(out, model_map, model_size,
                                                 weight_offset, full_in_dim,
                                                 k_off, k_cnt, out_dim,
                                                 x, x_elem_off);
    }
    fprintf(stderr, DS4_GPU_LOG_PREFIX
            "quant kslice: weight type %u not implemented (only Q8_0=8). "
            "The TP shared-expert K-split needs this type.\n",
            weight_type);
    return 0;
}
