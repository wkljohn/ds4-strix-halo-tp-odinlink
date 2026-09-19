/* Research-only Metal-inspired Q8_0 scheduling, adapted from c285966.
 * The repair keeps shared activation staging and production arithmetic while
 * using a 16-wave CTA for single-pointer rows,
 * while restoring production's per-lane block order and exact ordered scale
 * multiply. Packed GGUF bytes and full source row strides are retained. */
__global__ static void matmul_q8_0_pair_f32_sharedx_warp_rows_w32_pack4_kernel(
        float *, float *, const unsigned char *, const unsigned char *,
        const float *, uint32_t, uint64_t, uint64_t, uint64_t);
template <bool PAIR, unsigned ROWS_PER_BLOCK = 8u>
__global__ static void glm5_q8_decode_tile_kernel(
        float *out0, float *out1, const unsigned char *w0,
        const unsigned char *w1, const float *x, uint32_t n_blocks,
        uint32_t n_rows, uint64_t row_bytes) {
    extern __shared__ float sx[];
    for (unsigned i = threadIdx.x; i < n_blocks*32u; i += blockDim.x)
        sx[i] = x[i];
    __syncthreads();
    const unsigned lane = threadIdx.x & 31u;
    const unsigned row = blockIdx.x*ROWS_PER_BLOCK + (threadIdx.x >> 5u);
    if (row >= n_rows) return;
    const unsigned pair = blockIdx.y;
    const unsigned char *w = pair ? w1 : w0;
    float *out = pair ? out1 : out0;
    const unsigned char *wr = w + (uint64_t)row*row_bytes;
    float acc = 0.0f;
    if constexpr (PAIR) {
        for (unsigned b = 0u; b < n_blocks; ++b) {
            const unsigned char *blk = wr + (uint64_t)b*34u;
            const float d = q8_0_scale_broadcast_w32(blk);
            const int8_t q = ((const int8_t *)(blk + 2u))[lane];
            acc = fmaf(d * (float)q, sx[(b << 5u) + lane], acc);
        }
        acc = warp_sum_f32(acc);
        if (lane == 0u) out[row] = acc;
        return;
    }
    unsigned b = 0u;
    for (; b + 8u <= n_blocks; b += 8u) {
        float scales[8];
        int8_t weights[8];
        float inputs[8];
#pragma unroll
        for (unsigned u = 0u; u < 8u; ++u) {
            const unsigned char *blk = wr + (uint64_t)(b + u)*34u;
            uint16_t bits = 0u;
            if (lane == 0u)
                bits = (uint16_t)blk[0] | ((uint16_t)blk[1] << 8u);
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
            bits = __shfl(bits, 0, 32);
#else
            bits = __shfl_sync(FULL_WARP_MASK, bits, 0, 32);
#endif
            scales[u] = __half2float(__ushort_as_half(bits));
            weights[u] = ((const int8_t *)(blk + 2u))[lane];
            inputs[u] = sx[((b + u) << 5u) + lane];
        }
#pragma unroll
        for (unsigned u = 0u; u < 8u; ++u) {
            const float scaled = q8_exact_ordered_mul(scales[u],
                                                       (float)weights[u]);
            acc += scaled * inputs[u];
        }
    }
    for (; b < n_blocks; ++b) {
        const unsigned char *blk = wr + (uint64_t)b*34u;
        const float d = q8_0_scale_broadcast_w32(blk);
        const int8_t q = ((const int8_t *)(blk + 2u))[lane];
        acc += d * (float)q * sx[(b << 5u) + lane];
    }
    acc = warp_sum_f32(acc);
    if (lane == 0u) out[row] = acc;
}

/* Speed-only attribution lane retained for the old 10.8 t/s measurement.
 * Four lanes cooperate on one Q8 block and each warp emits two adjacent rows.
 * This is intentionally opt-in (mode 3): its reduction order is not the
 * production order and it must never enter a promotion candidate. */
__global__ static void glm5_q8_decode_tile_fast_kernel(
        float *out0, float *out1, const unsigned char *w0,
        const unsigned char *w1, const float *x, uint32_t n_blocks,
        uint32_t n_rows, uint64_t row_bytes) {
    extern __shared__ float sx[];
    for (unsigned i = threadIdx.x; i < n_blocks * 32u; i += blockDim.x)
        sx[i] = x[i];
    __syncthreads();
    const unsigned lane = threadIdx.x & 31u;
    const unsigned row = (blockIdx.x * 8u + (threadIdx.x >> 5u)) * 2u;
    if (row >= n_rows) return;
    const unsigned char *w = blockIdx.y ? w1 : w0;
    float *out = blockIdx.y ? out1 : out0;
    float acc0 = 0.0f, acc1 = 0.0f;
    for (unsigned base = 0u; base < n_blocks; base += 4u) {
        const unsigned b = base + (lane >> 3u);
        if (b >= n_blocks) continue;
        const unsigned char *a = w + (uint64_t)row * row_bytes + b * 34u;
        const unsigned char *c = a + row_bytes;
        const float d0 = q8_0_scale_scalar(a), d1 = q8_0_scale_scalar(c);
#pragma unroll
        for (unsigned pair = 0u; pair < 2u; ++pair) {
            const unsigned j = (lane & 7u) * 4u + pair * 2u;
            const float x0 = sx[b * 32u + j], x1 = sx[b * 32u + j + 1u];
            const uint16_t p0 = *reinterpret_cast<const uint16_t *>(a + 2u + j);
            const uint16_t p1 = *reinterpret_cast<const uint16_t *>(c + 2u + j);
            acc0 = fmaf(__fmul_rn(d0, (float)(int8_t)(p0 & 255u)), x0, acc0);
            acc0 = fmaf(__fmul_rn(d0, (float)(int8_t)(p0 >> 8u)), x1, acc0);
            acc1 = fmaf(__fmul_rn(d1, (float)(int8_t)(p1 & 255u)), x0, acc1);
            acc1 = fmaf(__fmul_rn(d1, (float)(int8_t)(p1 >> 8u)), x1, acc1);
        }
    }
    acc0 = warp_sum_f32(acc0);
    acc1 = warp_sum_f32(acc1);
    if (lane == 0u) { out[row] = acc0; out[row + 1u] = acc1; }
}

/* Paired variant keeps both accumulators in the same warp, matching the
 * established pair kernel's expression order while using the repaired
 * eight-wave CTA and shared activation tile. */
__global__ static void glm5_q8_decode_tile_pair_kernel(
        float *out0, float *out1, const unsigned char *w0,
        const unsigned char *w1, const float *x, uint32_t n_blocks,
        uint32_t n_rows, uint64_t row_bytes) {
    extern __shared__ float sx[];
    for (unsigned i = threadIdx.x; i < n_blocks*32u; i += blockDim.x)
        sx[i] = x[i];
    __syncthreads();
    const unsigned lane = threadIdx.x & 31u;
    const unsigned wave = threadIdx.x >> 5u;
    const unsigned row = blockIdx.x*8u + wave;
    if (row >= n_rows) return;
    const unsigned char *wr0 = w0 + (uint64_t)row*row_bytes;
    const unsigned char *wr1 = w1 + (uint64_t)row*row_bytes;
    float acc0 = 0.0f, acc1 = 0.0f;
    for (unsigned b = 0u; b < n_blocks; ++b) {
        const float xv = sx[(b << 5u) + lane];
        const unsigned char *blk0 = wr0 + (uint64_t)b*34u;
        const unsigned char *blk1 = wr1 + (uint64_t)b*34u;
        const float d0 = q8_0_scale_broadcast_w32(blk0);
        const float d1 = q8_0_scale_broadcast_w32(blk1);
        const int8_t q0 = ((const int8_t *)(blk0 + 2u))[lane];
        const int8_t q1 = ((const int8_t *)(blk1 + 2u))[lane];
        acc0 += d0 * (float)q0 * xv;
        acc1 += d1 * (float)q1 * xv;
    }
    acc0 = warp_sum_f32(acc0);
    acc1 = warp_sum_f32(acc1);
    if (lane == 0u) { out0[row] = acc0; out1[row] = acc1; }
}

/* MLX qmv-style output-row reuse, translated to native Q8_0. No affine
 * codec or persistent weight copy. Each row retains mode 1's block/lane
 * order, explicit scale rounding and wave reduction. Preserve mode 1's
 * eight-block load window as well: row reuse alone serializes scale/weight
 * memory latency. Two rows use eight waves, four rows use four waves. */
template <unsigned ROWS_PER_WAVE, unsigned ROWS_PER_CTA = 16u,
          bool BROADCAST_SCALE_BITS = false>
__global__ static void glm5_q8_decode_multirow_kernel(
        float *out, const unsigned char *w, const float *x,
        uint32_t n_blocks, uint32_t n_rows, uint64_t row_bytes) {
    extern __shared__ float sx[];
    for (unsigned i = threadIdx.x; i < n_blocks * 32u; i += blockDim.x)
        sx[i] = x[i];
    __syncthreads();
    const unsigned lane = threadIdx.x & 31u;
    static_assert(ROWS_PER_CTA % ROWS_PER_WAVE == 0u, "complete row groups");
    const unsigned row0 = blockIdx.x * ROWS_PER_CTA + (threadIdx.x >> 5u) * ROWS_PER_WAVE;
    // The leaf launcher requires complete CTA tiles. All rows in this tile
    // exist; check once after the barrier.
    if (row0 >= n_rows) return;
    float acc[ROWS_PER_WAVE] = {};
    unsigned b = 0;
    for (; b + 8u <= n_blocks; b += 8u) {
        float inputs[8];
        float scales[ROWS_PER_WAVE][8];
        int8_t weights[ROWS_PER_WAVE][8];
#pragma unroll
        for (unsigned u = 0; u < 8u; ++u) inputs[u] = sx[(b + u) * 32u + lane];
        if constexpr (BROADCAST_SCALE_BITS) {
#pragma unroll
            for (unsigned u = 0; u < 8u; ++u) {
                uint16_t bits[ROWS_PER_WAVE] = {};
#pragma unroll
                for (unsigned r = 0; r < ROWS_PER_WAVE; ++r) {
                    const unsigned char *blk = w + (uint64_t)(row0 + r) * row_bytes + (b + u) * 34u;
                    if (lane == 0u)
                        bits[r] = (uint16_t)blk[0] | ((uint16_t)blk[1] << 8u);
                    weights[r][u] = ((const int8_t *)(blk + 2u))[lane];
                }
                // Issue each row's scale/weight pair before waiting on the
                // broadcasts, retaining the full eight-block load window.
#pragma unroll
                for (unsigned r = 0; r < ROWS_PER_WAVE; ++r) {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
                    bits[r] = __shfl(bits[r], 0, 32);
#else
                    bits[r] = __shfl_sync(FULL_WARP_MASK, bits[r], 0, 32);
#endif
                    scales[r][u] = __half2float(__ushort_as_half(bits[r]));
                }
            }
        } else {
#pragma unroll
            for (unsigned r = 0; r < ROWS_PER_WAVE; ++r) {
#pragma unroll
                for (unsigned u = 0; u < 8u; ++u) {
                    const unsigned char *blk = w + (uint64_t)(row0 + r) * row_bytes + (b + u) * 34u;
                    scales[r][u] = q8_0_scale_broadcast_w32(blk);
                    weights[r][u] = ((const int8_t *)(blk + 2u))[lane];
                }
            }
        }
#pragma unroll
        for (unsigned r = 0; r < ROWS_PER_WAVE; ++r) {
#pragma unroll
            for (unsigned u = 0; u < 8u; ++u)
                acc[r] += q8_exact_ordered_mul(scales[r][u], (float)weights[r][u]) * inputs[u];
        }
    }
    for (; b < n_blocks; ++b) {
        const float xv = sx[b * 32u + lane];
#pragma unroll
        for (unsigned r = 0; r < ROWS_PER_WAVE; ++r) {
            const unsigned char *blk = w + (uint64_t)(row0 + r) * row_bytes + b * 34u;
            const float d = q8_0_scale_broadcast_w32(blk);
            const int8_t q = ((const int8_t *)(blk + 2u))[lane];
            acc[r] += q8_exact_ordered_mul(d, (float)q) * xv;
        }
    }
#pragma unroll
    for (unsigned r = 0; r < ROWS_PER_WAVE; ++r) {
        acc[r] = warp_sum_f32(acc[r]);
        if (lane == 0u) out[row0 + r] = acc[r];
    }
}

static int glm5_q8_decode_tile_mode(void) {
#if defined(DS4_GFX1151_WAVE32)
    const char *value = getenv("DS4_ROCM_GLM5_Q8_DECODE_TILE");
    const char *glm = getenv("DS4_GLM5_NEXT_ENABLE_ORDINARY");
    if (!glm || strcmp(glm, "1") || !value) return 0;
    const int mode = !strcmp(value, "1") ? 1 : !strcmp(value, "2") ? 2 :
                     !strcmp(value, "3") ? 3 : !strcmp(value, "4") ? 4 :
                     !strcmp(value, "5") ? 5 : !strcmp(value, "6") ? 6 : 0;
    if (!mode) return 0;
    static const bool supported = []() {
        int device = 0;
        cudaDeviceProp prop{};
        return cudaGetDevice(&device) == cudaSuccess &&
            cudaGetDeviceProperties(&prop, device) == cudaSuccess &&
            strncmp(prop.gcnArchName, "gfx1151", 7) == 0 && prop.warpSize == 32;
    }();
    return supported ? mode : 0;
#else
    return 0;
#endif
}

/* Exact original-GLM shapes, including distinct full-row strides for K slices.
 * 0 means leave dispatch untouched; launch errors are returned, never retried
 * through a different arithmetic path. Mode 2 measures the existing pack4
 * schedule on the same shapes for attribution. */
static int glm5_q8_decode_tile_shape(uint32_t blocks, uint64_t rows,
                                     uint64_t stride) {
    if (blocks == 48u && (rows == 16384u || rows == 8192u) &&
        stride == 48u*34u) return 1;
    if (blocks == 128u && rows == 1024u && stride == 128u*34u) return 2;
    if (blocks == 256u && rows == 4096u && stride == 512u*34u) return 3;
    if (blocks == 32u && rows == 4096u && stride == 64u*34u) return 4;
    return 0;
}

static cudaError_t cuda_launch_glm5_q8_decode_tile(
        int mode, float *out0, float *out1, const unsigned char *w0,
        const unsigned char *w1, const float *x, uint32_t blocks,
        uint32_t rows, uint64_t stride, cudaStream_t stream = 0) {
    const unsigned pairs = w1 ? 2u : 1u;
    // The mode2 attribution kernel reads four complete blocks per iteration.
    // Every production whitelist shape satisfies this; guard the leaf too.
    if (rows % 16u || blocks == 0u || blocks > 256u || blocks % 4u ||
        stride < (uint64_t)blocks*34u || stride % 2u ||
        ((uintptr_t)w0 & 1u) || (w1 && ((uintptr_t)w1 & 1u)))
        return cudaErrorInvalidValue;
    if (blocks == 48u && rows == 8192u && stride == 48u * 34u) {
        static bool owned_reported;
        if (!owned_reported) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX
                    "GLM5 owned-head q_b tile K=1536 N=8192 mode=%d\n", mode);
            owned_reported = true;
        }
    }
    // Mode6 changes only the MLA output K slice. Other shapes and paired
    // projections retain mode1. It does not change the meanings of modes4/5.
    if (mode == 6 && !w1 && blocks == 256u && stride == 512u * 34u) {
        constexpr unsigned rows_per_wave = 2u;
        constexpr unsigned waves_per_cta = 16u;
        constexpr unsigned rows_per_cta = rows_per_wave * waves_per_cta;
        if (rows != 4096u || rows % rows_per_cta) return cudaErrorInvalidValue;
        static bool reported;
        if (!reported) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX "GLM5 Q8 multirow mode=6 "
                    "K=8192 N=4096 rows_per_cta=32 waves=16\n");
            reported = true;
        }
        glm5_q8_decode_multirow_kernel<rows_per_wave, rows_per_cta, true><<<
            rows / rows_per_cta, waves_per_cta * 32u,
            blocks * 32u * sizeof(float), stream>>>(
                out0, w0, x, blocks, rows, stride);
    } else if ((mode == 4 || mode == 5) && !w1) {
        if (mode == 4)
            glm5_q8_decode_multirow_kernel<2u><<<
                (rows + 15u) / 16u, 256u, blocks * 32u * sizeof(float), stream>>>(
                    out0, w0, x, blocks, rows, stride);
        else
            glm5_q8_decode_multirow_kernel<4u><<<
                (rows + 15u) / 16u, 128u, blocks * 32u * sizeof(float), stream>>>(
                    out0, w0, x, blocks, rows, stride);
    } else if (mode == 3) {
        glm5_q8_decode_tile_fast_kernel<<<dim3((rows + 15u) / 16u, pairs),
            256u, blocks * 32u * sizeof(float), stream>>>(
                out0, out1, w0, w1, x, blocks, rows, stride);
    } else if (mode == 1 || mode == 4 || mode == 5 || mode == 6) {
        if (w1)
            glm5_q8_decode_tile_pair_kernel<<<
                dim3((rows + 7u)/8u), 256u, blocks*32u*sizeof(float), stream>>>(
                    out0, out1, w0, w1, x, blocks, rows, stride);
        else
            glm5_q8_decode_tile_kernel<false, 16u><<<
                dim3((rows + 15u)/16u, pairs), 512u,
                blocks*32u*sizeof(float), stream>>>(
                    out0, out1, w0, w1, x, blocks, rows, stride);
    } else {
        for (unsigned p = 0; p < pairs; ++p) {
            matmul_q8_0_f32_sharedx_warp_rows_w32_pack4_kernel<<<
                rows/8u, 256u, blocks*32u*sizeof(float), stream>>>(
                    p ? out1 : out0, p ? w1 : w0, x, blocks, rows, stride);
            const cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) return err;
        }
    }
    return cudaGetLastError();
}

/* Small verification batches: share original Q8_0 blocks across independent
 * M1 accumulators. Token panels bound LDS at 32 KiB, including K=12288 down.
 * Paired dense M1 and scalar down have distinct scale-expression contracts. */
template <unsigned Tokens, bool Pair>
__global__ static void glm5_dense_q8_small_m_kernel(
        float *out0, float *out1, const unsigned char *w0,
        const unsigned char *w1, const float *x, unsigned in_dim,
        unsigned out_dim) {
    static_assert(Tokens == 2u || Tokens == 4u || Tokens == 6u || Tokens == 8u, "small M");
    constexpr unsigned Panel = 1024u, Rows = 8u;
    __shared__ float sx[Tokens][Panel];
    const unsigned lane = threadIdx.x & 31u;
    const unsigned row = blockIdx.x * Rows + (threadIdx.x >> 5u);
    const uint64_t stride = (in_dim / 32u) * 34u;
    float sum0[Tokens] = {}, sum1[Tokens] = {};
    for (unsigned first = 0u; first < in_dim; first += Panel) {
        for (unsigned i = threadIdx.x; i < Tokens * Panel; i += Rows * 32u)
            sx[i / Panel][i % Panel] = x[(uint64_t)(i / Panel) * in_dim + first + i % Panel];
        __syncthreads();
        for (unsigned b = 0u; b < Panel / 32u; ++b) {
            const uint64_t offset = (uint64_t)row * stride + (first / 32u + b) * 34u;
            const unsigned char *a = w0 + offset;
            const float d0 = q8_0_scale_broadcast_w32(a);
            const int8_t q0 = ((const int8_t *)(a + 2u))[lane];
            if constexpr (Pair) {
                const unsigned char *c = w1 + offset;
                const float d1 = q8_0_scale_broadcast_w32(c);
                const int8_t q1 = ((const int8_t *)(c + 2u))[lane];
#pragma unroll
                for (unsigned t = 0u; t < Tokens; ++t) {
                    const float xv = sx[t][b * 32u + lane];
                    // The production pair's fast-math ISA rounds d*x before
                    // FMA with q. Sharing d*q across tokens changes rounding.
                    sum0[t] = fmaf(q8_exact_ordered_mul(d0, xv), (float)q0, sum0[t]);
                    sum1[t] = fmaf(q8_exact_ordered_mul(d1, xv), (float)q1, sum1[t]);
                }
            } else {
                const float scaled = q8_exact_ordered_mul(d0, (float)q0);
#pragma unroll
                for (unsigned t = 0u; t < Tokens; ++t)
                    sum0[t] += scaled * sx[t][b * 32u + lane];
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (unsigned t = 0u; t < Tokens; ++t) {
        const float value0 = warp_sum_f32(sum0[t]);
        if (lane == 0u) out0[(uint64_t)t * out_dim + row] = value0;
        if constexpr (Pair) {
            const float value1 = warp_sum_f32(sum1[t]);
            if (lane == 0u) out1[(uint64_t)t * out_dim + row] = value1;
        }
    }
}

extern "C" int ds4_rocm_glm5_dense_q8_small_m(
        ds4_gpu_tensor *out0, ds4_gpu_tensor *out1,
        const void *model_map, uint64_t model_size,
        uint64_t offset0, uint64_t offset1,
        uint32_t in_dim, uint32_t out_dim,
        const ds4_gpu_tensor *x, uint32_t tokens) {
    const bool pair = out1 != nullptr;
    if (!out0 || !x || !model_map ||
        (tokens != 2u && tokens != 4u && tokens != 6u && tokens != 8u) ||
        (pair ? (in_dim != 4096u || out_dim != 12288u) :
                (in_dim != 12288u || out_dim != 4096u))) return 0;
    const char *prefetch = getenv("DS4_ROCM_GLM5_Q8_SHAREDX_PREFETCH");
    const char *nt = getenv("DS4_ROCM_GLM5_Q8_SHAREDX_NONTEMPORAL");
    const char *rows = getenv("DS4_ROCM_GLM5_Q8_SHAREDX_ROWS_PER_BLOCK");
    if (glm5_q8_decode_tile_mode() != 1 || g_quality_mode ||
        !cuda_runtime_config()->q8_decode_sharedx_64k ||
        !prefetch || strcmp(prefetch, "8") != 0 ||
        (nt && strcmp(nt, "0") != 0 && strcmp(nt, "1") != 0) ||
        (rows && strcmp(rows, "8") != 0 && strcmp(rows, "16") != 0 && strcmp(rows, "32") != 0))
        return 0;
    const uint64_t x_bytes = (uint64_t)tokens * in_dim * sizeof(float);
    const uint64_t out_bytes = (uint64_t)tokens * out_dim * sizeof(float);
    const uint64_t weight_bytes = (uint64_t)out_dim * (in_dim / 32u) * 34u;
    if (!cuda_tensor_has_bytes(x, x_bytes) || !cuda_tensor_has_bytes(out0, out_bytes) ||
        (pair && !cuda_tensor_has_bytes(out1, out_bytes)) ||
        !cuda_model_range_fits(model_size, offset0, weight_bytes) || offset0 % 2u ||
        (pair && (!cuda_model_range_fits(model_size, offset1, weight_bytes) || offset1 % 2u))) return 0;
    auto disjoint = [](const ds4_gpu_tensor *a, uint64_t na,
                       const ds4_gpu_tensor *b, uint64_t nb) {
        const uintptr_t ap = (uintptr_t)a->ptr, bp = (uintptr_t)b->ptr;
        return ap && bp && ap % 4u == 0u && bp % 4u == 0u &&
            (ap <= bp ? na <= bp - ap : nb <= ap - bp);
    };
    if (!disjoint(out0, out_bytes, x, x_bytes) ||
        (pair && (!disjoint(out1, out_bytes, x, x_bytes) ||
                  !disjoint(out0, out_bytes, out1, out_bytes)))) return 0;
    const auto *w0 = (const unsigned char *)cuda_model_range_ptr(model_map, offset0, weight_bytes, "dense_q8_small_m");
    const auto *w1 = pair ? (const unsigned char *)cuda_model_range_ptr(model_map, offset1, weight_bytes, "dense_q8_small_m_pair") : nullptr;
    if (!w0 || (pair && !w1)) return 0;
#define DS4_DENSE_Q8_LAUNCH(M, P) \
    glm5_dense_q8_small_m_kernel<M, P><<<out_dim / 8u, 256u>>>( \
        (float *)out0->ptr, pair ? (float *)out1->ptr : nullptr, w0, w1, \
        (const float *)x->ptr, in_dim, out_dim)
    if (pair) {
        if (tokens == 2u) DS4_DENSE_Q8_LAUNCH(2u, true);
        else if (tokens == 4u) DS4_DENSE_Q8_LAUNCH(4u, true);
        else if (tokens == 6u) DS4_DENSE_Q8_LAUNCH(6u, true);
        else DS4_DENSE_Q8_LAUNCH(8u, true);
    } else {
        if (tokens == 2u) DS4_DENSE_Q8_LAUNCH(2u, false);
        else if (tokens == 4u) DS4_DENSE_Q8_LAUNCH(4u, false);
        else if (tokens == 6u) DS4_DENSE_Q8_LAUNCH(6u, false);
        else DS4_DENSE_Q8_LAUNCH(8u, false);
    }
#undef DS4_DENSE_Q8_LAUNCH
    return cuda_ok(cudaGetLastError(), "GLM5 dense Q8 small-M exact launch");
}

/* Shared experts use two scalar M1 projections, unlike the dense paired
 * oracle above. Round scale*code once, then reuse that original Q8 block for
 * each independent token accumulator. Retain the full GGUF down-row stride.
 * This leaf is not selected by production dispatch. */
template <unsigned Tokens, bool Pair>
__global__ static void glm5_shared_q8_small_m_kernel(
        float *out0, float *out1, const unsigned char *w0,
        const unsigned char *w1, const float *x, unsigned in_dim,
        unsigned out_dim, uint64_t row_bytes) {
    static_assert(Tokens == 2u || Tokens == 4u || Tokens == 6u || Tokens == 8u, "small M");
    constexpr unsigned Panel = 1024u, Rows = 8u;
    __shared__ float sx[Tokens][Panel];
    const unsigned lane = threadIdx.x & 31u;
    const unsigned row = blockIdx.x * Rows + (threadIdx.x >> 5u);
    float sum0[Tokens] = {}, sum1[Tokens] = {};
    for (unsigned first = 0u; first < in_dim; first += Panel) {
        for (unsigned i = threadIdx.x; i < Tokens * Panel; i += Rows * 32u)
            sx[i / Panel][i % Panel] = x[(uint64_t)(i / Panel) * in_dim + first + i % Panel];
        __syncthreads();
        for (unsigned b = 0u; b < Panel / 32u; ++b) {
            const uint64_t offset = (uint64_t)row * row_bytes + (first / 32u + b) * 34u;
            const unsigned char *a = w0 + offset;
            const float d0 = q8_0_scale_broadcast_w32(a);
            const float scaled0 = q8_exact_ordered_mul(d0, (float)((const int8_t *)(a + 2u))[lane]);
            float scaled1 = 0.0f;
            if constexpr (Pair) {
                const unsigned char *c = w1 + offset;
                const float d1 = q8_0_scale_broadcast_w32(c);
                scaled1 = q8_exact_ordered_mul(d1, (float)((const int8_t *)(c + 2u))[lane]);
            }
#pragma unroll
            for (unsigned t = 0u; t < Tokens; ++t) {
                const float xv = sx[t][b * 32u + lane];
                sum0[t] += scaled0 * xv;
                if constexpr (Pair) sum1[t] += scaled1 * xv;
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (unsigned t = 0u; t < Tokens; ++t) {
        const float value0 = warp_sum_f32(sum0[t]);
        if (lane == 0u) out0[(uint64_t)t * out_dim + row] = value0;
        if constexpr (Pair) {
            const float value1 = warp_sum_f32(sum1[t]);
            if (lane == 0u) out1[(uint64_t)t * out_dim + row] = value1;
        }
    }
}

extern "C" int ds4_rocm_glm5_shared_q8_small_m(
        ds4_gpu_tensor *out0, ds4_gpu_tensor *out1,
        const void *model_map, uint64_t model_size,
        uint64_t offset0, uint64_t offset1,
        uint32_t in_dim, uint32_t out_dim, uint64_t row_bytes,
        uint32_t k_first, const ds4_gpu_tensor *x, uint32_t tokens) {
    const bool pair = out1 != nullptr;
    if (!out0 || !x || !model_map ||
        (tokens != 2u && tokens != 4u && tokens != 6u && tokens != 8u) ||
        (pair ? (in_dim != 4096u || out_dim != 1024u || row_bytes != 4352u || k_first != 0u) :
                (in_dim != 1024u || out_dim != 4096u || row_bytes != 2176u ||
                 (k_first != 0u && k_first != 1024u) || offset1 != 0u))) return 0;
    const char *prefetch = getenv("DS4_ROCM_GLM5_Q8_SHAREDX_PREFETCH");
    const char *nt = getenv("DS4_ROCM_GLM5_Q8_SHAREDX_NONTEMPORAL");
    const char *rows = getenv("DS4_ROCM_GLM5_Q8_SHAREDX_ROWS_PER_BLOCK");
    if (glm5_q8_decode_tile_mode() != 1 || g_quality_mode ||
        !cuda_runtime_config()->q8_decode_sharedx_64k ||
        !prefetch || strcmp(prefetch, "8") != 0 ||
        (nt && strcmp(nt, "0") != 0 && strcmp(nt, "1") != 0) ||
        (rows && strcmp(rows, "8") != 0 && strcmp(rows, "16") != 0 && strcmp(rows, "32") != 0))
        return 0;
    const uint64_t x_bytes = (uint64_t)tokens * in_dim * sizeof(float);
    const uint64_t out_bytes = (uint64_t)tokens * out_dim * sizeof(float);
    const uint64_t weight_bytes = (uint64_t)out_dim * row_bytes;
    if (!cuda_tensor_has_bytes(x, x_bytes) || !cuda_tensor_has_bytes(out0, out_bytes) ||
        (pair && !cuda_tensor_has_bytes(out1, out_bytes)) ||
        !cuda_model_range_fits(model_size, offset0, weight_bytes) || offset0 % 2u ||
        (pair && (!cuda_model_range_fits(model_size, offset1, weight_bytes) || offset1 % 2u))) return 0;
    auto disjoint = [](const ds4_gpu_tensor *a, uint64_t na,
                       const ds4_gpu_tensor *b, uint64_t nb) {
        const uintptr_t ap = (uintptr_t)a->ptr, bp = (uintptr_t)b->ptr;
        return ap && bp && ap % 4u == 0u && bp % 4u == 0u &&
            (ap <= bp ? na <= bp - ap : nb <= ap - bp);
    };
    if (!disjoint(out0, out_bytes, x, x_bytes) ||
        (pair && (!disjoint(out1, out_bytes, x, x_bytes) ||
                  !disjoint(out0, out_bytes, out1, out_bytes)))) return 0;
    const auto *w0 = (const unsigned char *)cuda_model_range_ptr(model_map, offset0, weight_bytes, "shared_q8_small_m");
    const auto *w1 = pair ? (const unsigned char *)cuda_model_range_ptr(model_map, offset1, weight_bytes, "shared_q8_small_m_pair") : nullptr;
    if (!w0 || (pair && !w1)) return 0;
    w0 += (uint64_t)(k_first / 32u) * 34u;
#define DS4_SHARED_Q8_LAUNCH(M, P) \
    glm5_shared_q8_small_m_kernel<M, P><<<out_dim / 8u, 256u>>>( \
        (float *)out0->ptr, pair ? (float *)out1->ptr : nullptr, w0, w1, \
        (const float *)x->ptr, in_dim, out_dim, row_bytes)
    if (pair) {
        if (tokens == 2u) DS4_SHARED_Q8_LAUNCH(2u, true);
        else if (tokens == 4u) DS4_SHARED_Q8_LAUNCH(4u, true);
        else if (tokens == 6u) DS4_SHARED_Q8_LAUNCH(6u, true);
        else DS4_SHARED_Q8_LAUNCH(8u, true);
    } else {
        if (tokens == 2u) DS4_SHARED_Q8_LAUNCH(2u, false);
        else if (tokens == 4u) DS4_SHARED_Q8_LAUNCH(4u, false);
        else if (tokens == 6u) DS4_SHARED_Q8_LAUNCH(6u, false);
        else DS4_SHARED_Q8_LAUNCH(8u, false);
    }
#undef DS4_SHARED_Q8_LAUNCH
    return cuda_ok(cudaGetLastError(), "GLM5 shared Q8 small-M exact launch");
}

/* Research MLA input projections. A 512-element panel exactly divides both
 * K4096 and K1536; the latter must not use the shared FFN's full Panel1024
 * loads. Each lane keeps the scalar block order and scale*code rounding. */
template <unsigned Tokens>
__global__ static void glm5_mla_prelude_q8_small_m_kernel(
        float *out, const unsigned char *weight, const float *x,
        unsigned in_dim, unsigned out_dim, uint64_t row_bytes) {
    static_assert(Tokens == 2u || Tokens == 4u || Tokens == 6u, "native widths");
    constexpr unsigned Panel = 512u, Rows = 8u;
    __shared__ float sx[Tokens][Panel];
    const unsigned lane = threadIdx.x & 31u;
    const unsigned row = blockIdx.x * Rows + (threadIdx.x >> 5u);
    float sums[Tokens] = {};
    for (unsigned first = 0u; first < in_dim; first += Panel) {
        for (unsigned i = threadIdx.x; i < Tokens * Panel; i += Rows * 32u)
            sx[i / Panel][i % Panel] =
                x[(uint64_t)(i / Panel) * in_dim + first + i % Panel];
        __syncthreads();
        for (unsigned b = 0u; b < Panel / 32u; ++b) {
            const unsigned char *block = weight + (uint64_t)row * row_bytes +
                (first / 32u + b) * 34u;
            const float scale = q8_0_scale_broadcast_w32(block);
            const float scaled = q8_exact_ordered_mul(
                scale, (float)((const int8_t *)(block + 2u))[lane]);
#pragma unroll
            for (unsigned t = 0u; t < Tokens; ++t)
                sums[t] += scaled * sx[t][b * 32u + lane];
        }
        __syncthreads();
    }
#pragma unroll
    for (unsigned t = 0u; t < Tokens; ++t) {
        const float result = warp_sum_f32(sums[t]);
        if (lane == 0u) out[(uint64_t)t * out_dim + row] = result;
    }
}

static int glm5_mla_prelude_q8_small_m_impl(
        ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
        uint64_t offset, uint32_t weight_type, uint32_t in_dim,
        uint32_t full_out_dim, uint32_t row_first, uint32_t out_dim,
        uint64_t row_bytes, const ds4_gpu_tensor *x, uint32_t tokens,
        bool launch) {
    const bool q_a_or_kv = in_dim == 4096u && row_first == 0u &&
        full_out_dim == out_dim && (out_dim == 1536u || out_dim == 512u);
    const bool q_b = in_dim == 1536u && full_out_dim == 16384u &&
        out_dim == 8192u && (row_first == 0u || row_first == 8192u);
    if (!out || !x || !model_map || ((uintptr_t)model_map & 1u) ||
        weight_type != 8u || (!q_a_or_kv && !q_b) ||
        (tokens != 2u && tokens != 4u && tokens != 6u) ||
        row_bytes != (uint64_t)in_dim / 32u * 34u) return 0;
    const char *prefetch = getenv("DS4_ROCM_GLM5_Q8_SHAREDX_PREFETCH");
    const char *nt = getenv("DS4_ROCM_GLM5_Q8_SHAREDX_NONTEMPORAL");
    const char *rows = getenv("DS4_ROCM_GLM5_Q8_SHAREDX_ROWS_PER_BLOCK");
    if (glm5_q8_decode_tile_mode() != 1 || g_quality_mode ||
        !cuda_runtime_config()->q8_decode_sharedx_64k ||
        !prefetch || strcmp(prefetch, "8") ||
        (nt && strcmp(nt, "0") && strcmp(nt, "1")) ||
        (rows && strcmp(rows, "8") && strcmp(rows, "16") && strcmp(rows, "32")))
        return 0;
    const uint64_t x_bytes = (uint64_t)tokens * in_dim * sizeof(float);
    const uint64_t out_bytes = (uint64_t)tokens * out_dim * sizeof(float);
    if (!cuda_tensor_has_bytes(x, x_bytes) || !cuda_tensor_has_bytes(out, out_bytes) ||
        offset % 2u || !cuda_model_range_fits(
            model_size, offset, (uint64_t)full_out_dim * row_bytes)) return 0;
    const uintptr_t op = (uintptr_t)out->ptr, xp = (uintptr_t)x->ptr;
    if (!op || !xp || op % 4u || xp % 4u ||
        (op <= xp ? out_bytes > xp - op : x_bytes > op - xp)) return 0;
    const uint64_t local_offset = offset + (uint64_t)row_first * row_bytes;
    const uint64_t local_bytes = (uint64_t)out_dim * row_bytes;
    if (!cuda_model_range_is_cached(model_map, local_offset, local_bytes)) return 0;
    const auto *weight = (const unsigned char *)cuda_model_range_ptr(
        model_map, local_offset, local_bytes, "mla_prelude_q8_small_m");
    if (!weight) return 0;
    const uintptr_t wp = (uintptr_t)weight;
    if (wp % 2u || (op <= wp ? out_bytes > wp - op : local_bytes > op - wp)) return 0;
    if (!launch) return 1;
#define DS4_MLA_PRELUDE_Q8_LAUNCH(M) \
    glm5_mla_prelude_q8_small_m_kernel<M><<<out_dim / 8u, 256u>>>( \
        (float *)out->ptr, weight, (const float *)x->ptr, in_dim, out_dim, row_bytes)
    if (tokens == 2u) DS4_MLA_PRELUDE_Q8_LAUNCH(2u);
    else if (tokens == 4u) DS4_MLA_PRELUDE_Q8_LAUNCH(4u);
    else DS4_MLA_PRELUDE_Q8_LAUNCH(6u);
#undef DS4_MLA_PRELUDE_Q8_LAUNCH
    return cuda_ok(cudaGetLastError(), "GLM5 MLA prelude Q8 small-M exact launch");
}

extern "C" int ds4_rocm_glm5_mla_prelude_q8_small_m_supported(
        ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
        uint64_t offset, uint32_t weight_type, uint32_t in_dim,
        uint32_t full_out_dim, uint32_t row_first, uint32_t out_dim,
        uint64_t row_bytes, const ds4_gpu_tensor *x, uint32_t tokens) {
    return glm5_mla_prelude_q8_small_m_impl(out, model_map, model_size, offset,
        weight_type, in_dim, full_out_dim, row_first, out_dim, row_bytes, x, tokens, false);
}

extern "C" int ds4_rocm_glm5_mla_prelude_q8_small_m(
        ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
        uint64_t offset, uint32_t weight_type, uint32_t in_dim,
        uint32_t full_out_dim, uint32_t row_first, uint32_t out_dim,
        uint64_t row_bytes, const ds4_gpu_tensor *x, uint32_t tokens) {
    return glm5_mla_prelude_q8_small_m_impl(out, model_map, model_size, offset,
        weight_type, in_dim, full_out_dim, row_first, out_dim, row_bytes, x, tokens, true);
}

/* Dedicated original-Q8 MLA output leaf. Keep the shared-expert admission
 * unchanged: this tensor has a full K16384 source row and a packed K8192
 * activation row. The caller establishes Q8_0 type and rank/slice ownership.
 * Reuse the exact same kernel instantiation and scalar accumulation order. */
static int glm5_mla_output_q8_small_m_impl(
        ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
        uint64_t offset, uint32_t full_in_dim, uint32_t k_first,
        uint32_t in_dim, uint32_t out_dim, uint64_t row_bytes,
        const ds4_gpu_tensor *x, uint32_t tokens, bool launch) {
    if (!out || !x || !model_map ||
        (tokens != 2u && tokens != 4u && tokens != 6u) ||
        full_in_dim != 16384u || in_dim != 8192u || out_dim != 4096u ||
        row_bytes != 17408u || (k_first != 0u && k_first != 8192u)) return 0;
    const char *prefetch = getenv("DS4_ROCM_GLM5_Q8_SHAREDX_PREFETCH");
    const char *nt = getenv("DS4_ROCM_GLM5_Q8_SHAREDX_NONTEMPORAL");
    const char *rows = getenv("DS4_ROCM_GLM5_Q8_SHAREDX_ROWS_PER_BLOCK");
    if (glm5_q8_decode_tile_mode() != 1 || g_quality_mode ||
        !cuda_runtime_config()->q8_decode_sharedx_64k ||
        !prefetch || strcmp(prefetch, "8") ||
        (nt && strcmp(nt, "0") && strcmp(nt, "1")) ||
        (rows && strcmp(rows, "8") && strcmp(rows, "16") && strcmp(rows, "32")))
        return 0;
    const uint64_t x_bytes = (uint64_t)tokens * in_dim * sizeof(float);
    const uint64_t out_bytes = (uint64_t)tokens * out_dim * sizeof(float);
    const uint64_t weight_bytes = (uint64_t)out_dim * row_bytes;
    if (!cuda_tensor_has_bytes(x, x_bytes) || !cuda_tensor_has_bytes(out, out_bytes) ||
        offset % 2u || !cuda_model_range_fits(model_size, offset, weight_bytes)) return 0;
    const uintptr_t op = (uintptr_t)out->ptr, xp = (uintptr_t)x->ptr;
    if (!op || !xp || op % 4u || xp % 4u ||
        (op <= xp ? out_bytes > xp - op : x_bytes > op - xp)) return 0;
    /* Refuse lazy registration/copies: integration must bind a resident
     * original tensor before private verification or timed work begins. */
    if (!cuda_model_range_is_cached(model_map, offset, weight_bytes)) return 0;
    const auto *w = (const unsigned char *)cuda_model_range_ptr(
        model_map, offset, weight_bytes, "mla_output_q8_small_m");
    if (!w) return 0;
    if (!launch) return 1;
    w += (uint64_t)(k_first / 32u) * 34u;
#define DS4_MLA_OUTPUT_Q8_LAUNCH(M) \
    glm5_shared_q8_small_m_kernel<M, false><<<out_dim / 8u, 256u>>>( \
        (float *)out->ptr, nullptr, w, nullptr, (const float *)x->ptr, \
        in_dim, out_dim, row_bytes)
    if (tokens == 2u) DS4_MLA_OUTPUT_Q8_LAUNCH(2u);
    else if (tokens == 4u) DS4_MLA_OUTPUT_Q8_LAUNCH(4u);
    else DS4_MLA_OUTPUT_Q8_LAUNCH(6u);
#undef DS4_MLA_OUTPUT_Q8_LAUNCH
    const int ok = cuda_ok(cudaGetLastError(), "GLM5 MLA output Q8 small-M exact launch");
    if (ok) ++g_glm5_mla_output_calls[tokens / 2u - 1u];
    return ok;
}

extern "C" int ds4_rocm_glm5_mla_output_q8_small_m_supported(
        ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
        uint64_t offset, uint32_t full_in_dim, uint32_t k_first,
        uint32_t in_dim, uint32_t out_dim, uint64_t row_bytes,
        const ds4_gpu_tensor *x, uint32_t tokens) {
    return glm5_mla_output_q8_small_m_impl(out, model_map, model_size, offset,
        full_in_dim, k_first, in_dim, out_dim, row_bytes, x, tokens, false);
}

extern "C" int ds4_rocm_glm5_mla_output_q8_small_m(
        ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
        uint64_t offset, uint32_t full_in_dim, uint32_t k_first,
        uint32_t in_dim, uint32_t out_dim, uint64_t row_bytes,
        const ds4_gpu_tensor *x, uint32_t tokens) {
    return glm5_mla_output_q8_small_m_impl(out, model_map, model_size, offset,
        full_in_dim, k_first, in_dim, out_dim, row_bytes, x, tokens, true);
}
