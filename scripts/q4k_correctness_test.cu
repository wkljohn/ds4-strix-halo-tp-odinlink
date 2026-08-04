// Q4_K x Q8_K dot-product correctness oracle - exhaustive edge cases +
// an independently-derived double-precision reference, kept deliberately
// small (one file, no framework).
//
// WHY THIS EXISTS (see docs/Q4K-WMMA-PLAN.md, Integration fact 6):
// llama.cpp, vLLM, and ds4's own dev_dot_q4_K_q8_K_block* are all the SAME
// ggml lineage - three "independent" GPU implementations agreeing with each
// other proves transcription fidelity, not correctness against the actual
// Q4_K spec. This file is the missing fourth reference: a from-scratch
// double-precision CPU derivation, checked against ds4's real GPU kernel
// (dev_dot_q4_K_q8_K_block, ds4_rocm_moe.cuh:274) at deterministic edge
// cases plus randomized cases, using a combined absolute+relative
// tolerance (a bare relative-error gate is unstable near zero, per the
// plan doc's correction 6).
//
// Also serves as the WMMA kernel's oracle once it exists - see the
// PLUG_IN_WMMA_HERE marker below for the extension point. Running this
// against DP4A now (before WMMA exists) is not wasted effort: it is an
// independent correctness check of ds4's SHIPPING kernel that has never
// been run before, using a reference that owes nothing to ds4/llama.cpp/
// vLLM's shared derivation.
//
// Does NOT touch ds4-upstream/ - new file only. Build:
//
//   /opt/rocm-7.2.0/bin/hipcc -O3 -ffast-math -g -fno-finite-math-only \
//     -pthread -D__HIP_PLATFORM_AMD__ -Wno-unused-command-line-argument \
//     --offload-arch=gfx1151 -Wall -Wextra \
//     q4k_correctness_test.cu -o q4k_correctness_test -lm -pthread
//
// Exit code 0 = all cases passed. Non-zero = at least one failure, printed.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>
#include <random>
#include <string>
#include <functional>

#define CUDA_QK_K 256

static __device__ __forceinline__ int32_t __dp4a(int32_t a, int32_t b, int32_t c) {
    union ds4_i8x4_bits { int32_t i; char4 v; } av, bv;
    av.i = a; bv.i = b;
    return amd_mixed_dot(av.v, bv.v, c, false);
}

// ---------------------------------------------------------------------
// Verbatim layouts, ds4-upstream/ds4_rocm.cu:65-83
// ---------------------------------------------------------------------
typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[12];
    uint8_t qs[CUDA_QK_K / 2];
} cuda_block_q4_K;

typedef struct {
    float d;
    int8_t qs[CUDA_QK_K];
    int16_t bsums[CUDA_QK_K / 16];
} cuda_block_q8_K;

static_assert(sizeof(cuda_block_q4_K) == 144, "layout drift");
static_assert(sizeof(cuda_block_q8_K) == 292, "layout drift");

// ---------------------------------------------------------------------
// Verbatim from ds4-upstream/rocm/ds4_rocm_moe.cuh:10,250,264,274 - the
// SHIPPING GPU kernel under test. This is the ONLY thing copied from
// ds4/ggml lineage in this file - everything below it (the CPU reference)
// is derived fresh from the Q4_K format spec, not transcribed from here.
// ---------------------------------------------------------------------
__device__ static float dev_f16_to_f32(uint16_t v) {
    return __half2float(*reinterpret_cast<const __half *>(&v));
}
__device__ static void dev_q4_K_get_scale_min(uint32_t j, const uint8_t *scales, uint8_t *d_out, uint8_t *m_out) {
    if (j < 4u) {
        *d_out = scales[j] & 63u;
        *m_out = scales[j + 4u] & 63u;
    } else {
        *d_out = (scales[j + 4u] & 0x0fu) | ((scales[j - 4u] >> 6u) << 4u);
        *m_out = (scales[j + 4u] >> 4u) | ((scales[j] >> 6u) << 4u);
    }
}
__device__ __forceinline__ static int32_t dev_dot_q4_32(const uint8_t *qs, const int8_t *q8, int shift) {
    int32_t sum = 0;
    #pragma unroll
    for (uint32_t i = 0; i < 32u; i += 4u) {
        const int32_t v = (*(const int32_t *)(qs + i) >> shift) & 0x0f0f0f0f;
        sum = __dp4a(v, *(const int32_t *)(q8 + i), sum);
    }
    return sum;
}
__device__ static float dev_dot_q4_K_q8_K_block(const cuda_block_q4_K *x, const cuda_block_q8_K *y) {
    const float xd = dev_f16_to_f32(x->d);
    const float xmin = dev_f16_to_f32(x->dmin);
    int isum = 0, summs = 0;
    #pragma unroll
    for (uint32_t j = 0; j < 8u; j++) {
        uint8_t sc, m;
        dev_q4_K_get_scale_min(j, x->scales, &sc, &m);
        summs += (int)m * (int)(y->bsums[2u * j] + y->bsums[2u * j + 1u]);
        const uint32_t byte_off = (j >> 1u) * 32u;
        const int shift = (j & 1u) ? 4 : 0;
        isum += (int)sc * dev_dot_q4_32(x->qs + byte_off, y->qs + j * 32u, shift);
    }
    return y->d * xd * (float)isum - y->d * xmin * (float)summs;
}

__global__ static void dot_block_kernel(const cuda_block_q4_K *x, const cuda_block_q8_K *y, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = dev_dot_q4_K_q8_K_block(&x[i], &y[i]);
}

// ---------------------------------------------------------------------
// Independent CPU reference, derived fresh from the Q4_K spec, double
// precision throughout. Scale/min bit-unpacking (a fixed, unambiguous
// wire format, not itself a correctness question worth re-deriving)
// reuses the standard scheme; everything from dequant onward is a naive
// per-element accumulation, NOT the isum/summs block-shortcut the GPU
// kernel uses - this independently checks that shortcut is valid.
// ---------------------------------------------------------------------
static void host_q4_K_get_scale_min(uint32_t j, const uint8_t *scales, uint8_t *d_out, uint8_t *m_out) {
    if (j < 4u) {
        *d_out = scales[j] & 63u;
        *m_out = scales[j + 4u] & 63u;
    } else {
        *d_out = (scales[j + 4u] & 0x0fu) | ((scales[j - 4u] >> 6u) << 4u);
        *m_out = (scales[j + 4u] >> 4u) | ((scales[j] >> 6u) << 4u);
    }
}

static double ref_dot_q4K_q8K_double(const cuda_block_q4_K *x, const cuda_block_q8_K *y) {
    // f16->f32: use the same IEEE-754 half decode any correct implementation
    // must use - this is a fixed numeric format, not a design choice.
    auto half_to_double = [](uint16_t h) -> double {
        uint32_t sign = (uint32_t)(h >> 15) & 1u;
        uint32_t exp = (uint32_t)(h >> 10) & 0x1fu;
        uint32_t mant = (uint32_t)h & 0x3ffu;
        double val;
        if (exp == 0) {
            val = ldexp((double)mant, -24); // subnormal
        } else if (exp == 31) {
            val = mant ? NAN : INFINITY;
        } else {
            val = ldexp((double)(mant | 0x400), (int)exp - 25);
        }
        return sign ? -val : val;
    };
    const double d_x = half_to_double(x->d);
    const double dmin_x = half_to_double(x->dmin);
    const double d_y = (double)y->d;

    double acc = 0.0;
    for (uint32_t j = 0; j < 8u; j++) {
        uint8_t sc, m;
        host_q4_K_get_scale_min(j, x->scales, &sc, &m);
        const uint32_t byte_off = (j >> 1u) * 32u;
        const int shift = (j & 1u) ? 4 : 0;
        for (uint32_t i = 0; i < 32u; i++) {
            uint8_t packed = x->qs[byte_off + i];
            uint8_t nibble = (uint8_t)((packed >> shift) & 0x0fu);
            double w = (double)sc * (double)nibble; // scale term
            double q8v = (double)y->qs[j * 32u + i];
            acc += d_x * w * d_y * q8v;
            acc -= dmin_x * (double)m * d_y * q8v;
        }
    }
    return acc;
}

// ---------------------------------------------------------------------
// Deterministic edge-case block generators
// ---------------------------------------------------------------------
static cuda_block_q4_K make_q4K(uint8_t qs_fill, uint8_t scale_fill, uint16_t d, uint16_t dmin,
                                 const uint8_t *qs_override = nullptr, const uint8_t *scales_override = nullptr) {
    cuda_block_q4_K b{};
    b.d = d; b.dmin = dmin;
    for (int i = 0; i < 12; i++) b.scales[i] = scales_override ? scales_override[i] : scale_fill;
    for (int i = 0; i < CUDA_QK_K / 2; i++) b.qs[i] = qs_override ? qs_override[i] : qs_fill;
    return b;
}
static cuda_block_q8_K make_q8K(int8_t qs_fill, float d, const int8_t *qs_override = nullptr) {
    cuda_block_q8_K b{};
    b.d = d;
    for (int i = 0; i < CUDA_QK_K; i++) b.qs[i] = qs_override ? qs_override[i] : qs_fill;
    for (int j = 0; j < CUDA_QK_K / 16; j++) {
        int32_t s = 0;
        for (int k = 0; k < 16; k++) s += b.qs[j * 16 + k];
        b.bsums[j] = (int16_t)s;
    }
    return b;
}
static uint8_t alternating_nibble_byte() { return 0xA5; } // 1010 0101 -> nibbles {5,A} alternating

// __float2half_rn returns a `half` VALUE; cuda_block_q4_K.d/.dmin are raw
// uint16_t bit-storage fields (matching ds4's production layout). Assigning
// a half directly to a uint16_t performs a VALUE conversion (0.03f -> 0),
// not a bit-reinterpret - every call site below must go through this.
static uint16_t f16_bits(float f) {
    half h = __float2half_rn(f);
    uint16_t bits;
    __builtin_memcpy(&bits, &h, sizeof(bits));
    return bits;
}

struct TestCase { std::string name; cuda_block_q4_K x; cuda_block_q8_K y; };

static std::vector<TestCase> build_deterministic_cases() {
    std::vector<TestCase> cases;
    uint16_t d1 = f16_bits(0.03f);
    uint16_t dm1 = f16_bits(0.015f);

    cases.push_back({"all_zero_nibbles_zero_scale", make_q4K(0x00, 0, d1, dm1), make_q8K(1, 0.01f)});
    cases.push_back({"all_15_nibbles_max_scale63", make_q4K(0xFF, 63, d1, dm1), make_q8K(1, 0.01f)});
    cases.push_back({"scale0_min0", make_q4K(0x77, 0, 0, 0), make_q8K(5, 0.01f)});
    cases.push_back({"scale63_min63", {}, make_q8K(5, 0.01f)}); // filled below
    {
        uint8_t sc63[12]; for (int i = 0; i < 12; i++) sc63[i] = 63 | (63 << 6); // sc and m both 63 pattern approx
        cases.back().x = make_q4K(0x77, 0, d1, dm1, nullptr, sc63);
    }
    cases.push_back({"zero_d", make_q4K(0x88, 32, 0, dm1), make_q8K(-3, 0.02f)});
    cases.push_back({"zero_dmin", make_q4K(0x88, 32, d1, 0), make_q8K(-3, 0.02f)});
    {
        uint8_t alt[CUDA_QK_K / 2];
        for (int i = 0; i < CUDA_QK_K / 2; i++) alt[i] = alternating_nibble_byte();
        cases.push_back({"alternating_nibbles", make_q4K(0, 32, d1, dm1, alt), make_q8K(1, 0.01f)});
    }
    cases.push_back({"activation_all_p127", make_q4K(0x55, 40, d1, dm1), make_q8K(127, 0.01f)});
    cases.push_back({"activation_all_n127", make_q4K(0x55, 40, d1, dm1), make_q8K(-127, 0.01f)});
    {
        int8_t alt[CUDA_QK_K];
        for (int i = 0; i < CUDA_QK_K; i++) alt[i] = (i & 1) ? 127 : -127;
        cases.push_back({"activation_alternating_p127_n127", make_q4K(0x55, 40, d1, dm1), make_q8K(0, 0.01f, alt)});
    }
    return cases;
}

static bool close_enough(double a, double b, double atol, double rtol) {
    return std::fabs(a - b) <= atol + rtol * std::fabs(b);
}

int main() {
    int dev_count = 0;
    hipError_t herr = hipGetDeviceCount(&dev_count);
    if (herr != hipSuccess || dev_count < 1) { fprintf(stderr, "no ROCm device found\n"); return 2; }
    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, 0);
    printf("Device 0: %s (gcnArch %s)\n\n", prop.name, prop.gcnArchName);

    std::vector<TestCase> cases = build_deterministic_cases();

    std::mt19937 rng(424242);
    std::uniform_int_distribution<int> byte_dist(0, 255);
    std::uniform_int_distribution<int> q8_dist(-127, 127);
    for (int r = 0; r < 200; r++) {
        cuda_block_q4_K x{};
        x.d = f16_bits(0.005f + 0.05f * (byte_dist(rng) / 255.0f));
        x.dmin = f16_bits(0.002f + 0.02f * (byte_dist(rng) / 255.0f));
        for (auto &s : x.scales) s = (uint8_t)byte_dist(rng);
        for (auto &q : x.qs) q = (uint8_t)byte_dist(rng);
        cuda_block_q8_K y{};
        y.d = 0.001f + 0.02f * (byte_dist(rng) / 255.0f);
        for (int i = 0; i < CUDA_QK_K; i++) y.qs[i] = (int8_t)q8_dist(rng);
        for (int j = 0; j < CUDA_QK_K / 16; j++) {
            int32_t s = 0;
            for (int k = 0; k < 16; k++) s += y.qs[j * 16 + k];
            y.bsums[j] = (int16_t)s;
        }
        cases.push_back({"random_" + std::to_string(r), x, y});
    }

    const int N = (int)cases.size();
    std::vector<cuda_block_q4_K> h_x(N);
    std::vector<cuda_block_q8_K> h_y(N);
    std::vector<double> ref(N);
    for (int i = 0; i < N; i++) {
        h_x[i] = cases[i].x;
        h_y[i] = cases[i].y;
        ref[i] = ref_dot_q4K_q8K_double(&h_x[i], &h_y[i]);
    }

    cuda_block_q4_K *d_x; cuda_block_q8_K *d_y; float *d_out;
    hipMalloc(&d_x, N * sizeof(cuda_block_q4_K));
    hipMalloc(&d_y, N * sizeof(cuda_block_q8_K));
    hipMalloc(&d_out, N * sizeof(float));
    hipMemcpy(d_x, h_x.data(), N * sizeof(cuda_block_q4_K), hipMemcpyHostToDevice);
    hipMemcpy(d_y, h_y.data(), N * sizeof(cuda_block_q8_K), hipMemcpyHostToDevice);
    dot_block_kernel<<<(N + 63) / 64, 64>>>(d_x, d_y, d_out, N);
    hipError_t launch_err = hipGetLastError();
    if (launch_err != hipSuccess) { fprintf(stderr, "launch failed: %s\n", hipGetErrorString(launch_err)); return 2; }
    hipDeviceSynchronize();
    std::vector<float> gpu_out(N);
    hipMemcpy(gpu_out.data(), d_out, N * sizeof(float), hipMemcpyDeviceToHost);

    // PLUG_IN_WMMA_HERE: once the WMMA kernel exists, add a third
    // comparison column here (gpu WMMA result vs ref, and vs DP4A),
    // ideally in scale-equalized mode for a tight tolerance per
    // Q4K-WMMA-PLAN.md Integration fact 6.

    int failures = 0;
    // Q4_K weight magnitudes here are O(0.01-0.05), Q8 values O(100),
    // 256 terms summed -> results are typically O(10-1000). Use a modest
    // absolute floor plus a relative bound; ds4's own kernel accumulates
    // in float32 internally so this checks GPU-float-vs-CPU-double
    // agreement, not bit-exact reproduction.
    const double ATOL = 1e-2, RTOL = 1e-3;
    for (int i = 0; i < N; i++) {
        double g = (double)gpu_out[i];
        bool ok = close_enough(g, ref[i], ATOL, RTOL);
        if (!ok) {
            failures++;
            printf("FAIL  %-40s ref=%.6f gpu=%.6f absdiff=%.6g\n",
                   cases[i].name.c_str(), ref[i], g, std::fabs(g - ref[i]));
        }
    }
    printf("\n%d/%d cases passed (independent double-precision CPU reference vs\n"
           "ds4's shipping dev_dot_q4_K_q8_K_block on real gfx1151 hardware)\n",
           N - failures, N);

    hipFree(d_x); hipFree(d_y); hipFree(d_out);
    return failures ? 1 : 0;
}
