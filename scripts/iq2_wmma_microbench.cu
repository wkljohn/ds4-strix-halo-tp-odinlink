// IQ2_XXS x Q8_1 gfx1151 primitive validation.
//
// This deliberately stops below the routed-MoE integration boundary.  It
// validates, in order, the two operations needed by the proposed Q2 prefill
// kernel: IQ2 codebook/sign expansion into signed int8 tiles and native
// 16x16x16 integer WMMA against Q8_1 activations.  The DP4A arm is an
// independent in-process oracle and a small performance control.
//
// Build from the repository root:
//   /opt/rocm-7.2.0/bin/hipcc -O3 -ffast-math -g \
//     -fno-finite-math-only -D__HIP_PLATFORM_AMD__ \
//     --offload-arch=gfx1151 \
//     scripts/iq2_wmma_microbench.cu -o scripts/iq2_wmma_microbench

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

enum { QK = 256, M = 64, N = 16, GROUPS = 8 };

struct block_iq2_xxs {
    uint16_t d;
    uint16_t qs[QK / 8];
};

struct q8_1_mmq_block {
    half2 ds[4];
    int8_t qs[128];
};

static_assert(sizeof(block_iq2_xxs) == 66, "IQ2_XXS layout drift");
static_assert(sizeof(q8_1_mmq_block) == 144, "Q8_1 MMQ layout drift");

#include "../ds4_iq2_tables_cuda.inc"

#define HIP_CHECK(expr) do {                                                   \
    const hipError_t hip_check_e = (expr);                                     \
    if (hip_check_e != hipSuccess) {                                           \
        std::fprintf(stderr, "%s:%d: %s: %s\n", __FILE__, __LINE__, #expr,    \
                     hipGetErrorString(hip_check_e));                          \
        std::exit(2);                                                          \
    }                                                                          \
} while (0)

static __device__ __forceinline__ int32_t dp4a_i8(
        int32_t a, int32_t b, int32_t c) {
    union bits_i8x4 { int32_t i; char4 v; } av, bv;
    av.i = a;
    bv.i = b;
    return amd_mixed_dot(av.v, bv.v, c, false);
}

static __device__ __forceinline__ int32_t vcmpne4(uint32_t a, uint32_t b) {
    uint32_t diff = a ^ b;
    diff |= diff >> 1;
    diff |= diff >> 2;
    diff |= diff >> 4;
    diff = (diff & 0x01010101u) * 0xffu;
    return (int32_t)diff;
}

static __device__ __forceinline__ int32_t vsub4(int32_t a, int32_t b) {
    const uint32_t ua = (uint32_t)a;
    const uint32_t ub = (uint32_t)b;
    return (int32_t)(((ua | 0x80808080u) - (ub & 0x7f7f7f7fu)) ^
                     ((ua ^ ~ub) & 0x80808080u));
}

static __host__ __device__ __forceinline__ uint32_t unpack_iq2_signs(
        uint32_t v) {
#if defined(__HIP_DEVICE_COMPILE__)
    const uint32_t p = __popc(v) & 1u;
#else
    const uint32_t p = (uint32_t)__builtin_popcount(v) & 1u;
#endif
    return (v ^ (p << 7u)) * 0x01010101u;
}

static __device__ __forceinline__ void expand_i8x8_device(
        uint8_t grid_idx, uint32_t sign_idx, int32_t *lo, int32_t *hi) {
    const uint32_t signs = unpack_iq2_signs(cuda_ksigns_iq2xs[sign_idx]);
    const int32_t sm0 = vcmpne4(signs & 0x08040201u, 0);
    const int32_t sm1 = vcmpne4(signs & 0x80402010u, 0);
    const uint64_t grid = cuda_iq2xxs_grid[grid_idx];
    *lo = vsub4((int32_t)(uint32_t)grid ^ sm0, sm0);
    *hi = vsub4((int32_t)(uint32_t)(grid >> 32) ^ sm1, sm1);
}

static __device__ __forceinline__ void expand_group_device(
        const block_iq2_xxs &w, int group, int32_t out[8], float *scale) {
    const uint16_t *q = w.qs + group * 4;
    const uint32_t grids = (uint32_t)q[0] | ((uint32_t)q[1] << 16);
    const uint32_t aux = (uint32_t)q[2] | ((uint32_t)q[3] << 16);
#pragma unroll
    for (int g = 0; g < 4; ++g) {
        expand_i8x8_device((uint8_t)(grids >> (8 * g)),
                           (aux >> (7 * g)) & 127u,
                           &out[2 * g], &out[2 * g + 1]);
    }
    *scale = __half2float(*reinterpret_cast<const half *>(&w.d)) *
             (float)(2u * (aux >> 28) + 1u) * 0.125f;
}

using i32x4 = int32_t __attribute__((ext_vector_type(4)));
using i32x8 = int32_t __attribute__((ext_vector_type(8)));

struct wmma_ab_frag { int32_t x[8]; };

static __device__ __forceinline__ wmma_ab_frag load_mirrored_16x8(
        const int32_t *p, int stride) {
    wmma_ab_frag r;
    const int row = (int)(threadIdx.x & 15u);
    *reinterpret_cast<i32x4 *>(&r.x[0]) =
        *reinterpret_cast<const i32x4 *>(p + row * stride);
    *reinterpret_cast<i32x4 *>(&r.x[4]) =
        *reinterpret_cast<const i32x4 *>(p + row * stride + 4);
    return r;
}

static __device__ __forceinline__ i32x8 wmma_i8(
        const wmma_ab_frag &a, const wmma_ab_frag &b, i32x8 c) {
    c = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
        true, *reinterpret_cast<const i32x4 *>(&a.x[0]),
        true, *reinterpret_cast<const i32x4 *>(&b.x[0]), c, true);
    c = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
        true, *reinterpret_cast<const i32x4 *>(&a.x[4]),
        true, *reinterpret_cast<const i32x4 *>(&b.x[4]), c, true);
    return c;
}

// Dump every expanded byte and scale before testing matrix arithmetic.
__global__ static void expand_kernel(
        const block_iq2_xxs *weights, int8_t *expanded, float *scales) {
    const int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= M * GROUPS) return;
    const int row = idx / GROUPS;
    const int group = idx % GROUPS;
    int32_t v[8];
    float scale;
    expand_group_device(weights[row], group, v, &scale);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        *reinterpret_cast<int32_t *>(expanded + row * QK + group * 32 + i * 4) = v[i];
    }
    scales[row * GROUPS + group] = scale;
}

// Full 64x16x256 tile.  XS matches llama.cpp's Q8_0 SRAM layout for
// IQ2_XXS; YS is DS4's validated Q8_1 tile layout.
__global__ __launch_bounds__(256) static void iq2_wmma_tile_kernel(
        const block_iq2_xxs *weights,
        const q8_1_mmq_block *acts,
        float *out) {
    constexpr int XS = 84;
    constexpr int YS = 36;
    __shared__ int32_t sx[M * XS];
    __shared__ int32_t sy[N * YS];
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int wave = tid >> 5;

    for (int idx = tid; idx < M * GROUPS; idx += blockDim.x) {
        const int row = idx / GROUPS;
        const int group = idx % GROUPS;
        int32_t v[8];
        float scale;
        expand_group_device(weights[row], group, v, &scale);
#pragma unroll
        for (int i = 0; i < 8; ++i) sx[row * XS + group * 8 + i] = v[i];
        reinterpret_cast<float *>(sx + row * XS + 64)[group] = scale;
    }
    __syncthreads();

    float acc[8] = {};
#pragma unroll
    for (int half = 0; half < 2; ++half) {
        for (int idx = tid; idx < N * YS; idx += blockDim.x) {
            const int token = idx / YS;
            const int e = idx - token * YS;
            sy[idx] = reinterpret_cast<const int32_t *>(acts + half * N + token)[e];
        }
        __syncthreads();

#pragma unroll
        for (int group4 = 0; group4 < 4; ++group4) {
            const int group = half * 4 + group4;
            if (wave < 4) {
                const wmma_ab_frag a = load_mirrored_16x8(
                    sx + wave * 16 * XS + group * 8, XS);
                const wmma_ab_frag b = load_mirrored_16x8(
                    sy + 4 + group4 * 8, YS);
                i32x8 c = {};
                c = wmma_i8(a, b, c);
#pragma unroll
                for (int l = 0; l < 8; ++l) {
                    const int row = wave * 16 + 2 * l + lane / 16;
                    const int token = lane & 15;
                    const float ws = reinterpret_cast<const float *>(
                        sx + row * XS + 64)[group];
                    const float as = __half2float(
                        reinterpret_cast<const half2 *>(sy + token * YS)[group4].x);
                    acc[l] += ws * as * (float)c[l];
                }
            }
        }
        __syncthreads();
    }

    if (wave < 4) {
#pragma unroll
        for (int l = 0; l < 8; ++l) {
            const int row = wave * 16 + 2 * l + lane / 16;
            const int token = lane & 15;
            out[token * M + row] = acc[l];
        }
    }
}

__global__ static void iq2_dp4a_tile_kernel(
        const block_iq2_xxs *weights,
        const q8_1_mmq_block *acts,
        float *out) {
    const int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= M * N) return;
    const int row = idx / N;
    const int token = idx % N;
    float acc = 0.0f;
#pragma unroll
    for (int group = 0; group < GROUPS; ++group) {
        int32_t v[8];
        float ws;
        expand_group_device(weights[row], group, v, &ws);
        const q8_1_mmq_block &a = acts[(group / 4) * N + token];
        const int8_t *q = a.qs + (group % 4) * 32;
        int32_t dot = 0;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            dot = dp4a_i8(v[i], *reinterpret_cast<const int32_t *>(q + 4 * i), dot);
        }
        acc += ws * __half2float(a.ds[group % 4].x) * (float)dot;
    }
    out[token * M + row] = acc;
}

static uint16_t half_bits(float x) {
    const half h = __float2half_rn(x);
    uint16_t bits;
    std::memcpy(&bits, &h, sizeof(bits));
    return bits;
}

static void expand_group_host(
        const block_iq2_xxs &w, int group,
        const uint64_t *grid_table, const uint8_t *sign_table,
        int8_t out[32], float *scale) {
    const uint16_t *q = w.qs + group * 4;
    const uint32_t grids = (uint32_t)q[0] | ((uint32_t)q[1] << 16);
    const uint32_t aux = (uint32_t)q[2] | ((uint32_t)q[3] << 16);
    for (int g = 0; g < 4; ++g) {
        const uint64_t grid = grid_table[(grids >> (8 * g)) & 255u];
        const uint32_t signs = unpack_iq2_signs(
            sign_table[(aux >> (7 * g)) & 127u]);
        for (int i = 0; i < 8; ++i) {
            int value = (int)((grid >> (8 * i)) & 255u);
            if (signs & (1u << i)) value = -value;
            out[g * 8 + i] = (int8_t)value;
        }
    }
    half hd;
    std::memcpy(&hd, &w.d, sizeof(hd));
    *scale = __half2float(hd) * (float)(2u * (aux >> 28) + 1u) * 0.125f;
}

static float median_us(hipEvent_t start, hipEvent_t stop,
                       void (*launch)(const block_iq2_xxs *,
                                      const q8_1_mmq_block *, float *),
                       const block_iq2_xxs *w,
                       const q8_1_mmq_block *a, float *out) {
    constexpr int warm = 100, reps = 2000;
    for (int i = 0; i < warm; ++i) launch(w, a, out);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<float> samples;
    samples.reserve(20);
    for (int s = 0; s < 20; ++s) {
        HIP_CHECK(hipEventRecord(start));
        for (int i = 0; i < reps; ++i) launch(w, a, out);
        HIP_CHECK(hipEventRecord(stop));
        HIP_CHECK(hipEventSynchronize(stop));
        float ms = 0.0f;
        HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
        samples.push_back(ms * 1000.0f / (float)reps);
    }
    std::sort(samples.begin(), samples.end());
    return 0.5f * (samples[9] + samples[10]);
}

static void launch_wmma(const block_iq2_xxs *w,
                        const q8_1_mmq_block *a, float *out) {
    iq2_wmma_tile_kernel<<<1, 256>>>(w, a, out);
}

static void launch_dp4a(const block_iq2_xxs *w,
                        const q8_1_mmq_block *a, float *out) {
    iq2_dp4a_tile_kernel<<<(M * N + 255) / 256, 256>>>(w, a, out);
}

int main() {
    hipDeviceProp_t prop{};
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    std::printf("device=%s arch=%s\n", prop.name, prop.gcnArchName);

    uint64_t grid_table[256];
    uint8_t sign_table[128];
    HIP_CHECK(hipMemcpyFromSymbol(grid_table, HIP_SYMBOL(cuda_iq2xxs_grid),
                                  sizeof(grid_table)));
    HIP_CHECK(hipMemcpyFromSymbol(sign_table, HIP_SYMBOL(cuda_ksigns_iq2xs),
                                  sizeof(sign_table)));

    std::mt19937 rng(0x495132u);
    std::uniform_int_distribution<int> u16(0, 65535);
    std::uniform_int_distribution<int> qi8(-127, 127);
    std::vector<block_iq2_xxs> hw(M);
    std::vector<q8_1_mmq_block> ha(2 * N);
    for (auto &w : hw) {
        w.d = half_bits(0.002f + (float)(rng() % 100) * 0.00001f);
        for (auto &q : w.qs) q = (uint16_t)u16(rng);
    }
    for (auto &a : ha) {
        for (auto &q : a.qs) q = (int8_t)qi8(rng);
        for (int g = 0; g < 4; ++g) {
            int sum = 0;
            for (int i = 0; i < 32; ++i) sum += a.qs[g * 32 + i];
            const float d = 0.003f + (float)(rng() % 100) * 0.00001f;
            a.ds[g] = __floats2half2_rn(d, d * (float)sum);
        }
    }

    block_iq2_xxs *dw = nullptr;
    q8_1_mmq_block *da = nullptr;
    int8_t *dexpanded = nullptr;
    float *dscales = nullptr, *dout = nullptr;
    HIP_CHECK(hipMalloc(&dw, hw.size() * sizeof(*dw)));
    HIP_CHECK(hipMalloc(&da, ha.size() * sizeof(*da)));
    HIP_CHECK(hipMalloc(&dexpanded, M * QK));
    HIP_CHECK(hipMalloc(&dscales, M * GROUPS * sizeof(float)));
    HIP_CHECK(hipMalloc(&dout, M * N * sizeof(float)));
    HIP_CHECK(hipMemcpy(dw, hw.data(), hw.size() * sizeof(*dw), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(da, ha.data(), ha.size() * sizeof(*da), hipMemcpyHostToDevice));

    expand_kernel<<<(M * GROUPS + 255) / 256, 256>>>(
        dw, dexpanded, dscales);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<int8_t> expanded(M * QK);
    std::vector<float> scales(M * GROUPS);
    HIP_CHECK(hipMemcpy(expanded.data(), dexpanded, expanded.size(), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(scales.data(), dscales, scales.size() * sizeof(float), hipMemcpyDeviceToHost));
    int expansion_bad = 0;
    for (int row = 0; row < M; ++row) {
        for (int group = 0; group < GROUPS; ++group) {
            int8_t ref[32];
            float scale;
            expand_group_host(hw[row], group, grid_table, sign_table, ref, &scale);
            if (std::memcmp(ref, expanded.data() + row * QK + group * 32, 32) != 0 ||
                scales[row * GROUPS + group] != scale) expansion_bad++;
        }
    }
    std::printf("expansion cases=%d bad=%d\n", M * GROUPS, expansion_bad);

    launch_dp4a(dw, da, dout);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<float> ref(M * N), got(M * N);
    HIP_CHECK(hipMemcpy(ref.data(), dout, ref.size() * sizeof(float), hipMemcpyDeviceToHost));
    launch_wmma(dw, da, dout);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(got.data(), dout, got.size() * sizeof(float), hipMemcpyDeviceToHost));
    float max_abs = 0.0f, max_rel = 0.0f;
    int arithmetic_bad = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        const float ae = std::fabs(got[i] - ref[i]);
        const float re = ae / std::max(1.0e-7f, std::fabs(ref[i]));
        max_abs = std::max(max_abs, ae);
        max_rel = std::max(max_rel, re);
        if (ae > 2.0e-4f + 2.0e-5f * std::fabs(ref[i])) arithmetic_bad++;
    }
    std::printf("arithmetic outputs=%zu bad=%d max_abs=%.9g max_rel=%.9g\n",
                got.size(), arithmetic_bad, max_abs, max_rel);

    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    const float wmma_us = median_us(start, stop, launch_wmma, dw, da, dout);
    const float dp4a_us = median_us(start, stop, launch_dp4a, dw, da, dout);
    std::printf("timing tile=64x16x256 wmma_us=%.4f dp4a_us=%.4f speedup=%.3fx\n",
                wmma_us, dp4a_us, dp4a_us / wmma_us);

    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    HIP_CHECK(hipFree(dout));
    HIP_CHECK(hipFree(dscales));
    HIP_CHECK(hipFree(dexpanded));
    HIP_CHECK(hipFree(da));
    HIP_CHECK(hipFree(dw));
    return expansion_bad || arithmetic_bad ? 1 : 0;
}
