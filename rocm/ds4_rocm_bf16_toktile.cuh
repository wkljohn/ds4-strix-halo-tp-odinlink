// Cache-free multi-row BF16-weight projection shared by production and its
// standalone real-shape/tail diagnostic. The including translation unit must
// provide HIP builtins and fixed-width integer types.

static constexpr uint32_t kDs4Bf16ToktileThreads = 256u;
static constexpr uint32_t kDs4Bf16ToktileWaves =
    kDs4Bf16ToktileThreads / 32u;
static_assert(kDs4Bf16ToktileThreads % 32u == 0u,
              "BF16 token tile requires complete wave32 groups");

static __device__ __forceinline__ float ds4_bf16_ordered_mul(float a,
                                                              float b) {
    float out;
    asm("v_mul_f32 %0, %1, %2" : "=v"(out) : "v"(a), "v"(b));
    return out;
}

static __device__ __forceinline__ float ds4_bf16_ordered_add(float a,
                                                              float b) {
    float out;
    asm("v_add_f32 %0, %1, %2" : "=v"(out) : "v"(a), "v"(b));
    return out;
}

template <uint32_t TokenTile>
__global__ static void matmul_bf16_f32_toktile_w32_kernel(
        float *out,
        const uint16_t *w,
        const float *x,
        uint32_t in_dim,
        uint32_t out_dim) {
    const uint32_t row = blockIdx.x;
    const uint32_t token_base = blockIdx.y * TokenTile;
    if (row >= out_dim) return;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t wave = threadIdx.x >> 5u;
    const uint16_t *wr = w + (uint64_t)row * in_dim;
    float sums[TokenTile] = {};
    for (uint32_t i = threadIdx.x; i < in_dim; i += blockDim.x) {
        const float weight =
            __uint_as_float((uint32_t)wr[i] << 16u);
#pragma unroll
        for (uint32_t token = 0u; token < TokenTile; ++token) {
            sums[token] += weight *
                x[(uint64_t)(token_base + token) * in_dim + i];
        }
    }
    __shared__ float wave_sums[TokenTile][kDs4Bf16ToktileWaves];
#pragma unroll
    for (uint32_t token = 0u; token < TokenTile; ++token) {
        float value = sums[token];
#pragma unroll
        for (uint32_t offset = 16u; offset > 0u; offset >>= 1u)
            value += __shfl_down(value, offset, 32u);
        if (lane == 0u) wave_sums[token][wave] = value;
    }
    __syncthreads();
    if (threadIdx.x < TokenTile) {
        float value = 0.0f;
#pragma unroll
        for (uint32_t wv = 0u; wv < kDs4Bf16ToktileWaves; ++wv)
            value += wave_sums[threadIdx.x][wv];
        out[(uint64_t)(token_base + threadIdx.x) * out_dim + row] = value;
    }
}

/* Exact-order token tiling for narrow-output BF16 projections.  The generic
 * production kernel owns one [row, token] per block and reduces its 256
 * thread-local K chains through a 256-wide LDS butterfly.  Preserve those
 * chains and that butterfly independently for each token, but reuse every
 * weight load across a 32-token prompt tile.  This is deliberately distinct
 * from the lower-LDS wave reduction above, whose association differs from the
 * generic kernel. */
template <uint32_t TokenTile>
__global__ static void matmul_bf16_f32_skinny_exact_toktile_kernel(
        float *out,
        const uint16_t *w,
        const float *x,
        uint32_t in_dim,
        uint32_t out_dim) {
    static_assert(TokenTile >= 1u && TokenTile <= 32u,
                  "skinny exact BF16 token tile must be 1..32");
    const uint32_t row = blockIdx.x;
    const uint32_t token_base = blockIdx.y * TokenTile;
    const uint32_t tid = threadIdx.x;
    if (row >= out_dim || tid >= kDs4Bf16ToktileThreads) return;
    const uint16_t *wr = w + (uint64_t)row * in_dim;
    float sums[TokenTile] = {};
    for (uint32_t i = tid; i < in_dim; i += kDs4Bf16ToktileThreads) {
        const float weight = __uint_as_float((uint32_t)wr[i] << 16u);
#pragma unroll
        for (uint32_t token = 0u; token < TokenTile; ++token)
            sums[token] += weight *
                x[(uint64_t)(token_base + token) * in_dim + i];
    }
    __shared__ float partial[TokenTile][kDs4Bf16ToktileThreads];
#pragma unroll
    for (uint32_t token = 0u; token < TokenTile; ++token)
        partial[token][tid] = sums[token];
    __syncthreads();
    for (uint32_t stride = kDs4Bf16ToktileThreads >> 1u;
         stride > 0u; stride >>= 1u) {
        if (tid < stride) {
#pragma unroll
            for (uint32_t token = 0u; token < TokenTile; ++token)
                partial[token][tid] += partial[token][tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0u) {
#pragma unroll
        for (uint32_t token = 0u; token < TokenTile; ++token)
            out[(uint64_t)(token_base + token) * out_dim + row] =
                partial[token][0];
    }
}

template <uint32_t TokenTile>
__global__ static void matmul_bf16_f32_kslice_toktile_w32_kernel(
        float *out,
        const uint16_t *w,
        const float *x,
        uint32_t full_in_dim,
        uint32_t k_off,
        uint32_t k_cnt,
        uint32_t out_dim) {
    const uint32_t row = blockIdx.x;
    const uint32_t token_base = blockIdx.y * TokenTile;
    if (row >= out_dim) return;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t wave = threadIdx.x >> 5u;
    const uint16_t *wr = w + (uint64_t)row * full_in_dim + k_off;
    float sums[TokenTile] = {};
    for (uint32_t i = threadIdx.x; i < k_cnt; i += blockDim.x) {
        const float weight = __uint_as_float((uint32_t)wr[i] << 16u);
#pragma unroll
        for (uint32_t token = 0u; token < TokenTile; ++token) {
            sums[token] += weight *
                x[(uint64_t)(token_base + token) * k_cnt + i];
        }
    }
    __shared__ float wave_sums[TokenTile][kDs4Bf16ToktileWaves];
#pragma unroll
    for (uint32_t token = 0u; token < TokenTile; ++token) {
        float value = sums[token];
#pragma unroll
        for (uint32_t offset = 16u; offset > 0u; offset >>= 1u)
            value += __shfl_down(value, offset, 32u);
        if (lane == 0u) wave_sums[token][wave] = value;
    }
    __syncthreads();
    if (threadIdx.x < TokenTile) {
        float value = 0.0f;
#pragma unroll
        for (uint32_t wv = 0u; wv < kDs4Bf16ToktileWaves; ++wv)
            value += wave_sums[threadIdx.x][wv];
        out[(uint64_t)(token_base + threadIdx.x) * out_dim + row] = value;
    }
}

/* Exact-order GLM-5 KDA low-rank expansion. The scalar 256-thread kernel has
 * only 128 active lanes for this shape. Its first effective reduction steps
 * are (x[lane] + x[lane+64]) + (x[lane+32] + x[lane+96]), followed by the
 * ordinary 32-lane tree. Preserve that order while sharing four BF16 weights
 * across a tile of prompt rows. */
template <uint32_t TokenTile>
__global__ static void matmul_bf16_f32_lowrank128_toktile_w32_kernel(
        float *out,
        const uint16_t *w,
        const float *x,
        uint32_t out_dim) {
    static_assert(TokenTile >= 1u && TokenTile <= 32u,
                  "low-rank BF16 token tile must fit one wave");
    const uint32_t row = blockIdx.x;
    const uint32_t token_base = blockIdx.y * TokenTile;
    const uint32_t lane = threadIdx.x;
    if (row >= out_dim || lane >= 32u) return;
    const uint16_t *wr = w + (uint64_t)row * 128u;
    const float w0 = __uint_as_float((uint32_t)wr[lane] << 16u);
    const float w1 = __uint_as_float((uint32_t)wr[lane + 32u] << 16u);
    const float w2 = __uint_as_float((uint32_t)wr[lane + 64u] << 16u);
    const float w3 = __uint_as_float((uint32_t)wr[lane + 96u] << 16u);
    float sums[TokenTile];
#pragma unroll
    for (uint32_t token = 0u; token < TokenTile; ++token) {
        const float *xr = x + (uint64_t)(token_base + token) * 128u;
        const float p0 = ds4_bf16_ordered_mul(w0, xr[lane]);
        const float p1 = ds4_bf16_ordered_mul(w1, xr[lane + 32u]);
        const float p2 = ds4_bf16_ordered_mul(w2, xr[lane + 64u]);
        const float p3 = ds4_bf16_ordered_mul(w3, xr[lane + 96u]);
        const float pair02 = ds4_bf16_ordered_add(p0, p2);
        const float pair13 = ds4_bf16_ordered_add(p1, p3);
        sums[token] = ds4_bf16_ordered_add(pair02, pair13);
    }
#pragma unroll
    for (uint32_t offset = 16u; offset > 0u; offset >>= 1u) {
#pragma unroll
        for (uint32_t token = 0u; token < TokenTile; ++token)
            sums[token] = ds4_bf16_ordered_add(
                sums[token], __shfl_down(sums[token], offset, 32u));
    }
    if (lane == 0u) {
#pragma unroll
        for (uint32_t token = 0u; token < TokenTile; ++token)
            out[(uint64_t)(token_base + token) * out_dim + row] =
                sums[token];
    }
}
