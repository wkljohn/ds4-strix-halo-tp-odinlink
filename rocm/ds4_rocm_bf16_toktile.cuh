// Cache-free multi-row BF16-weight projection shared by production and its
// standalone real-shape/tail diagnostic. The including translation unit must
// provide HIP builtins and fixed-width integer types.

static constexpr uint32_t kDs4Bf16ToktileThreads = 256u;
static constexpr uint32_t kDs4Bf16ToktileWaves =
    kDs4Bf16ToktileThreads / 32u;
static_assert(kDs4Bf16ToktileThreads % 32u == 0u,
              "BF16 token tile requires complete wave32 groups");

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
