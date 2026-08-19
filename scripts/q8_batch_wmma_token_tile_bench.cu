// Small-step harness for increasing token reuse in the generic Q8_0 prefill
// WMMA kernel. It compares the production 64x64 output/token tile with both
// a 64x128 token-reuse candidate and a 128x64 output-row-reuse candidate
// before any production dispatch is changed.
//
// Build:
//   /opt/rocm/bin/hipcc -O3 -ffast-math -fno-finite-math-only \
//     --offload-arch=gfx1151 scripts/q8_batch_wmma_token_tile_bench.cu \
//     -o /tmp/q8_batch_wmma_token_tile_bench

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

static void check(hipError_t rc, const char *where) {
    if (rc != hipSuccess) {
        std::fprintf(stderr, "%s: %s\n", where, hipGetErrorString(rc));
        std::exit(1);
    }
}

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;

template <uint32_t M_TILE, uint32_t N_TILE, uint32_t K_STAGE = 32u>
__launch_bounds__(2u * M_TILE, 1)
__global__ static void q8_wmma_kernel(float *out, const unsigned char *w,
        const float *x, uint32_t n_tokens, uint32_t in_dim,
        uint32_t out_dim, uint64_t row_bytes) {
    static_assert(M_TILE == 64u || M_TILE == 128u, "tested output-row tiles");
    static_assert(N_TILE == 64u || N_TILE == 128u, "tested token tiles");
    static_assert(K_STAGE == 32u || K_STAGE == 128u, "tested K stages");
    constexpr uint32_t K_TILE = 32u, WARPS = M_TILE / 16u;
    constexpr uint32_t LDS_STRIDE = K_STAGE == 128u ? 136u : K_STAGE;
    constexpr uint32_t M_PER_WARP = M_TILE / WARPS;
    constexpr uint32_t NT = N_TILE / 16u;
    const uint32_t block_m = blockIdx.x * M_TILE;
    const uint32_t block_n = blockIdx.y * N_TILE;
    if (block_m >= out_dim || block_n >= n_tokens) return;
    const uint32_t tid = threadIdx.x, wave = tid >> 5u;
    const uint32_t lane = tid & 31u, lane16 = lane & 15u;
    const uint32_t warp_m = block_m + wave * M_PER_WARP;
    const uint32_t my_row = warp_m + lane16;
    const uint32_t safe_row = my_row < out_dim ? my_row : out_dim - 1u;
    const unsigned char *row_base = w + (uint64_t)safe_row * row_bytes;
    const uint32_t n_blocks = in_dim >> 5u;
    float8_t acc[NT];
#pragma unroll
    for (uint32_t i = 0; i < NT; ++i)
        acc[i] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    extern __shared__ _Float16 lds_x[];
    constexpr uint32_t BLOCKS_PER_STAGE = K_STAGE / K_TILE;
    for (uint32_t stage_bi = 0; stage_bi < n_blocks;
         stage_bi += BLOCKS_PER_STAGE) {
        for (uint32_t j = tid; j < N_TILE * K_STAGE; j += blockDim.x) {
            const uint32_t nt = j / K_STAGE, kk = j % K_STAGE;
            const uint32_t tok = block_n + nt;
            lds_x[nt * LDS_STRIDE + kk] =
                tok < n_tokens && stage_bi * K_TILE + kk < in_dim
                ? (_Float16)x[(uint64_t)tok * in_dim + stage_bi * K_TILE + kk]
                : (_Float16)0.0f;
        }
        __syncthreads();
        for (uint32_t sb = 0; sb < BLOCKS_PER_STAGE && stage_bi + sb < n_blocks; sb++) {
            const unsigned char *bp = row_base + (uint64_t)(stage_bi + sb) * 34u;
            _Float16 sc;
            uint16_t bits;
            __builtin_memcpy(&bits, bp, 2);
            __builtin_memcpy(&sc, &bits, 2);
            const int8_t *w0 = reinterpret_cast<const int8_t *>(bp + 2u);
            const int8_t *w1 = reinterpret_cast<const int8_t *>(bp + 18u);
            half16_t a0, a1;
#pragma unroll
            for (uint32_t i = 0; i < 16u; ++i) {
                a0[i] = sc * (_Float16)(float)(int)w0[i];
                a1[i] = sc * (_Float16)(float)(int)w1[i];
            }
#pragma unroll
            for (uint32_t nt = 0; nt < NT; ++nt) {
                const _Float16 *xb = lds_x +
                    (nt * 16u + lane16) * LDS_STRIDE + sb * K_TILE;
                const half16_t b0 = *reinterpret_cast<const half16_t *>(xb);
                const half16_t b1 = *reinterpret_cast<const half16_t *>(xb + 16u);
                acc[nt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a0, b0, acc[nt]);
                acc[nt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a1, b1, acc[nt]);
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (uint32_t nt = 0; nt < NT; ++nt) {
        const uint32_t tok = block_n + nt * 16u + lane16;
        if (tok >= n_tokens) continue;
#pragma unroll
        for (uint32_t j = 0; j < 8u; ++j) {
            const uint32_t row = warp_m + 2u * j + (lane >> 4u);
            if (row < out_dim) out[(uint64_t)tok * out_dim + row] = acc[nt][j];
        }
    }
}

struct Buffers {
    uint32_t tokens, in_dim, out_dim;
    uint64_t row_bytes;
    std::vector<unsigned char> w;
    std::vector<float> x;
    unsigned char *dw = nullptr;
    float *dx = nullptr, *d64 = nullptr, *dn128 = nullptr, *dm128 = nullptr;
    float *dk128 = nullptr;
    Buffers(uint32_t t, uint32_t k, uint32_t m)
        : tokens(t), in_dim(k), out_dim(m), row_bytes((uint64_t)(k / 32u) * 34u),
          w((uint64_t)m * row_bytes), x((uint64_t)t * k) {
        std::mt19937 rng(19);
        std::uniform_real_distribution<float> rf(-1.0f, 1.0f);
        std::uniform_int_distribution<int> ri(-127, 127);
        for (uint32_t row = 0; row < m; ++row) {
            for (uint32_t b = 0; b < k / 32u; ++b) {
                unsigned char *p = w.data() + (uint64_t)row * row_bytes + (uint64_t)b * 34u;
                const _Float16 scale = (_Float16)(0.002f + 0.02f * std::fabs(rf(rng)));
                __builtin_memcpy(p, &scale, 2);
                for (uint32_t i = 0; i < 32u; ++i)
                    reinterpret_cast<int8_t *>(p + 2u)[i] = (int8_t)ri(rng);
            }
        }
        for (float &v : x) v = rf(rng);
        const size_t out_bytes = (size_t)t * m * sizeof(float);
        check(hipMalloc(&dw, w.size()), "weights allocation");
        check(hipMalloc(&dx, x.size() * sizeof(float)), "activation allocation");
        check(hipMalloc(&d64, out_bytes), "64 output allocation");
        check(hipMalloc(&dn128, out_bytes), "N128 output allocation");
        check(hipMalloc(&dm128, out_bytes), "M128 output allocation");
        check(hipMalloc(&dk128, out_bytes), "K128 output allocation");
        check(hipMemcpy(dw, w.data(), w.size(), hipMemcpyHostToDevice), "weights copy");
        check(hipMemcpy(dx, x.data(), x.size() * sizeof(float), hipMemcpyHostToDevice), "activation copy");
    }
    ~Buffers() {
        (void)hipFree(dw); (void)hipFree(dx); (void)hipFree(d64);
        (void)hipFree(dn128); (void)hipFree(dm128); (void)hipFree(dk128);
    }
};

template <uint32_t M_TILE, uint32_t N_TILE>
static void launch(Buffers &b, float *out) {
    const dim3 grid((b.out_dim + M_TILE - 1u) / M_TILE,
                    (b.tokens + N_TILE - 1u) / N_TILE);
    q8_wmma_kernel<M_TILE, N_TILE>
        <<<grid, 2u * M_TILE, N_TILE * 32u * sizeof(_Float16)>>>(
        out, b.dw, b.dx, b.tokens, b.in_dim, b.out_dim, b.row_bytes);
}

template <uint32_t M_TILE, uint32_t N_TILE>
static void launch_k128(Buffers &b, float *out) {
    const dim3 grid((b.out_dim + M_TILE - 1u) / M_TILE,
                    (b.tokens + N_TILE - 1u) / N_TILE);
    q8_wmma_kernel<M_TILE, N_TILE, 128u>
        <<<grid, 2u * M_TILE, N_TILE * 136u * sizeof(_Float16)>>>(
        out, b.dw, b.dx, b.tokens, b.in_dim, b.out_dim, b.row_bytes);
}

template <uint32_t M_TILE, uint32_t N_TILE>
static float elapsed(Buffers &b, float *out, int iterations) {
    hipEvent_t start, stop;
    check(hipEventCreate(&start), "start event");
    check(hipEventCreate(&stop), "stop event");
    for (int i = 0; i < 2; ++i) launch<M_TILE, N_TILE>(b, out);
    check(hipDeviceSynchronize(), "warmup");
    check(hipEventRecord(start), "record start");
    for (int i = 0; i < iterations; ++i) launch<M_TILE, N_TILE>(b, out);
    check(hipEventRecord(stop), "record stop");
    check(hipEventSynchronize(stop), "sync stop");
    float ms = 0.0f;
    check(hipEventElapsedTime(&ms, start, stop), "elapsed");
    check(hipEventDestroy(start), "destroy start");
    check(hipEventDestroy(stop), "destroy stop");
    return ms / iterations;
}

template <uint32_t M_TILE, uint32_t N_TILE>
static float elapsed_k128(Buffers &b, float *out, int iterations) {
    hipEvent_t start, stop;
    check(hipEventCreate(&start), "K128 start event");
    check(hipEventCreate(&stop), "K128 stop event");
    for (int i = 0; i < 2; ++i) launch_k128<M_TILE, N_TILE>(b, out);
    check(hipDeviceSynchronize(), "K128 warmup");
    check(hipEventRecord(start), "K128 record start");
    for (int i = 0; i < iterations; ++i) launch_k128<M_TILE, N_TILE>(b, out);
    check(hipEventRecord(stop), "K128 record stop");
    check(hipEventSynchronize(stop), "K128 sync stop");
    float ms = 0.0f;
    check(hipEventElapsedTime(&ms, start, stop), "K128 elapsed");
    check(hipEventDestroy(start), "K128 destroy start");
    check(hipEventDestroy(stop), "K128 destroy stop");
    return ms / iterations;
}

int main() {
    {
        Buffers b(193u, 96u, 80u);
        launch<64, 64>(b, b.d64);
        launch<64, 128>(b, b.dn128);
        launch<128, 64>(b, b.dm128);
        launch_k128<128, 64>(b, b.dk128);
        check(hipDeviceSynchronize(), "correctness kernels");
        const size_t n = (size_t)b.tokens * b.out_dim;
        std::vector<float> a(n), c(n), d(n), e(n);
        check(hipMemcpy(a.data(), b.d64, n * sizeof(float), hipMemcpyDeviceToHost), "copy 64");
        check(hipMemcpy(c.data(), b.dn128, n * sizeof(float), hipMemcpyDeviceToHost), "copy N128");
        check(hipMemcpy(d.data(), b.dm128, n * sizeof(float), hipMemcpyDeviceToHost), "copy M128");
        check(hipMemcpy(e.data(), b.dk128, n * sizeof(float), hipMemcpyDeviceToHost), "copy K128");
        size_t n128_mismatch = 0, m128_mismatch = 0, k128_mismatch = 0;
        double n128_max_abs = 0.0, m128_max_abs = 0.0, k128_max_abs = 0.0;
        for (size_t i = 0; i < n; ++i) {
            if (std::memcmp(&a[i], &c[i], sizeof(float))) ++n128_mismatch;
            if (std::memcmp(&a[i], &d[i], sizeof(float))) ++m128_mismatch;
            if (std::memcmp(&d[i], &e[i], sizeof(float))) ++k128_mismatch;
            n128_max_abs = std::max(n128_max_abs, std::fabs((double)a[i] - c[i]));
            m128_max_abs = std::max(m128_max_abs, std::fabs((double)a[i] - d[i]));
            k128_max_abs = std::max(k128_max_abs, std::fabs((double)d[i] - e[i]));
        }
        std::printf("correctness n128_bit_mismatches=%zu/%zu n128_max_abs=%.9g "
                    "m128_bit_mismatches=%zu/%zu m128_max_abs=%.9g "
                    "k128_bit_mismatches=%zu/%zu k128_max_abs=%.9g\n",
                    n128_mismatch, n, n128_max_abs,
                    m128_mismatch, n, m128_max_abs,
                    k128_mismatch, n, k128_max_abs);
        if (n128_mismatch || m128_mismatch || k128_mismatch) return 2;
    }
    struct Shape { const char *name; uint32_t in_dim, out_dim; };
    const Shape shapes[] = {
        {"q_b", 1536u, 32768u},
        {"indexer_q_b", 1536u, 8192u},
        {"q_a", 4096u, 1536u},
        {"compressor", 4096u, 1024u},
        {"kv", 4096u, 512u},
    };
    for (const Shape &s : shapes) {
        Buffers b(2048u, s.in_dim, s.out_dim);
        const int iterations = s.out_dim >= 8192u ? 3 : 6;
        const float ms64 = elapsed<64, 64>(b, b.d64, iterations);
        const float msn128 = elapsed<64, 128>(b, b.dn128, iterations);
        const float msm128 = elapsed<128, 64>(b, b.dm128, iterations);
        const float msk128 = elapsed_k128<128, 64>(b, b.dk128, iterations);
        std::printf("shape=%s tokens=2048 in=%u out=%u tile64_ms=%.4f "
                    "n128_ms=%.4f n128_change=%+.1f%% m128_ms=%.4f "
                    "m128_change=%+.1f%% k128_ms=%.4f k128_vs_m128=%+.1f%%\n",
                    s.name, s.in_dim, s.out_dim, ms64,
                    msn128, 100.0f * (msn128 / ms64 - 1.0f),
                    msm128, 100.0f * (msm128 / ms64 - 1.0f),
                    msk128, 100.0f * (msk128 / msm128 - 1.0f));
    }
    return 0;
}
