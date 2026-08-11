// Shape oracle for the five-row F16 projections in the DSpark verifier.
//
// The wave-split-K layout follows vLLM's ROCm skinny GEMM design, including
// its N=5 speculative-verification specialization (vLLM PR #40687).  This
// standalone test deliberately precedes any DS4 dispatch change.
//
// Build:
//   /opt/rocm/bin/hipcc -O3 -ffast-math -fno-finite-math-only \
//     --offload-arch=gfx1151 scripts/dspark_f16_wvsplitk_bench.cu \
//     -lhipblas -o /tmp/dspark_f16_wvsplitk_bench

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

static void check(hipError_t rc, const char *where) {
    if (rc != hipSuccess) {
        std::fprintf(stderr, "%s: %s\n", where, hipGetErrorString(rc));
        std::exit(1);
    }
}

static void check_blas(hipblasStatus_t rc, const char *where) {
    if (rc != HIPBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "%s: hipBLAS status %d\n", where, (int)rc);
        std::exit(1);
    }
}

__global__ static void f32_to_f16(_Float16 *dst, const float *src,
                                  uint32_t n) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = (_Float16)src[i];
}

union alignas(16) pack8h {
    _Float16 h[8];
    float4 raw;
    float packed2[4];
};

static __device__ __forceinline__ void dot2_acc(float &acc, float a, float b) {
    asm("v_dot2_f32_f16 %0, %1, %2, %0"
        : "+v"(acc) : "v"(a), "v"(b));
}

// One persistent workgroup per CU. Each wave streams one or more weight rows,
// while all waves reuse the five activation rows staged in LDS. The five rows
// remain independent accumulators; only weight and activation loads are shared.
template <uint32_t TOKENS, uint32_t YTILE, uint32_t UNROLL>
__launch_bounds__(512, 1)
__global__ static void f16_wvsplitk_n5(
        float *out, const _Float16 *weights, const _Float16 *x,
        uint32_t kdim, uint32_t mdim) {
    __shared__ __align__(16) _Float16 sx[32768];
    const uint32_t lane = threadIdx.x;
    const uint32_t wave = threadIdx.y;
    const uint32_t linear = wave * 32u + lane;

    const uint32_t x_elems = TOKENS * kdim;
    for (uint32_t i = linear * 8u; i < x_elems; i += 512u * 8u) {
        reinterpret_cast<float4 *>(sx)[i / 8u] =
            reinterpret_cast<const float4 *>(x)[i / 8u];
    }
    __syncthreads();

    uint32_t m0 = (blockIdx.x * 16u + wave) * YTILE;
    const uint32_t m_stride = gridDim.x * 16u * YTILE;
    while (m0 < mdim) {
        float sum[TOKENS][YTILE] = {};
        for (uint32_t k0 = 0; k0 < kdim; k0 += 32u * 8u * UNROLL) {
            pack8h av[TOKENS][UNROLL];
            pack8h wv[YTILE][UNROLL];
#pragma unroll
            for (uint32_t u = 0; u < UNROLL; ++u) {
                const uint32_t k = k0 + u * 32u * 8u + lane * 8u;
                if (k < kdim) {
#pragma unroll
                    for (uint32_t t = 0; t < TOKENS; ++t) {
                        av[t][u].raw = reinterpret_cast<const float4 *>(
                            sx + (uint64_t)t * kdim + k)[0];
                    }
#pragma unroll
                    for (uint32_t y = 0; y < YTILE; ++y) {
                        const uint32_t m = min(m0 + y, mdim - 1u);
                        wv[y][u].raw = reinterpret_cast<const float4 *>(
                            weights + (uint64_t)m * kdim + k)[0];
                    }
                } else {
#pragma unroll
                    for (uint32_t t = 0; t < TOKENS; ++t) av[t][u].raw = {};
#pragma unroll
                    for (uint32_t y = 0; y < YTILE; ++y) wv[y][u].raw = {};
                }
            }
#pragma unroll
            for (uint32_t u = 0; u < UNROLL; ++u) {
#pragma unroll
                for (uint32_t t = 0; t < TOKENS; ++t) {
#pragma unroll
                    for (uint32_t y = 0; y < YTILE; ++y) {
#pragma unroll
                        for (uint32_t p = 0; p < 4u; ++p) {
                            dot2_acc(sum[t][y], av[t][u].packed2[p],
                                     wv[y][u].packed2[p]);
                        }
                    }
                }
            }
        }
#pragma unroll
        for (uint32_t mask = 16u; mask != 0u; mask >>= 1u) {
#pragma unroll
            for (uint32_t t = 0; t < TOKENS; ++t) {
#pragma unroll
                for (uint32_t y = 0; y < YTILE; ++y) {
                    sum[t][y] += __shfl_xor(sum[t][y], mask, 32);
                }
            }
        }
        if (lane == 0u) {
#pragma unroll
            for (uint32_t t = 0; t < TOKENS; ++t) {
#pragma unroll
                for (uint32_t y = 0; y < YTILE; ++y) {
                    const uint32_t m = m0 + y;
                    if (m < mdim) out[(uint64_t)t * mdim + m] = sum[t][y];
                }
            }
        }
        m0 += m_stride;
    }
}

static float elapsed(hipEvent_t a, hipEvent_t b) {
    float ms = 0.0f;
    check(hipEventElapsedTime(&ms, a, b), "hipEventElapsedTime");
    return ms;
}

template <uint32_t TOKENS>
static void run_shape(uint32_t kdim, uint32_t mdim, uint32_t iterations,
                      uint32_t weight_sets) {
    const uint64_t w_count = (uint64_t)kdim * mdim;
    const uint64_t w_total = w_count * weight_sets;
    const uint64_t x_count = (uint64_t)TOKENS * kdim;
    const uint64_t o_count = (uint64_t)TOKENS * mdim;
    std::mt19937 rng(12345u + kdim + mdim);
    std::uniform_real_distribution<float> dist(-0.25f, 0.25f);
    std::vector<_Float16> hw(w_total);
    std::vector<float> hx(x_count);
    for (auto &v : hw) v = (_Float16)dist(rng);
    for (auto &v : hx) v = dist(rng);

    _Float16 *dw = nullptr, *dxh = nullptr;
    float *dxf = nullptr, *dref = nullptr, *dcand = nullptr;
    check(hipMalloc(&dw, w_total * sizeof(*dw)), "hipMalloc weights");
    check(hipMalloc(&dxh, x_count * sizeof(*dxh)), "hipMalloc xh");
    check(hipMalloc(&dxf, x_count * sizeof(*dxf)), "hipMalloc xf");
    check(hipMalloc(&dref, o_count * sizeof(*dref)), "hipMalloc ref");
    check(hipMalloc(&dcand, o_count * sizeof(*dcand)), "hipMalloc cand");
    check(hipMemcpy(dw, hw.data(), w_total * sizeof(*dw), hipMemcpyHostToDevice),
          "copy weights");
    check(hipMemcpy(dxf, hx.data(), x_count * sizeof(*dxf), hipMemcpyHostToDevice),
          "copy x");
    f32_to_f16<<<(x_count + 255u) / 256u, 256u>>>(dxh, dxf, x_count);
    check(hipGetLastError(), "convert launch");

    hipblasHandle_t blas;
    check_blas(hipblasCreate(&blas), "hipblasCreate");
    const float alpha = 1.0f, beta = 0.0f;
    auto launch_ref = [&](uint32_t set) {
        const _Float16 *wset = dw + (uint64_t)(set % weight_sets) * w_count;
        check_blas(hipblasGemmEx(blas, HIPBLAS_OP_T, HIPBLAS_OP_N,
                                 (int)mdim, (int)TOKENS, (int)kdim,
                                 &alpha, wset, HIP_R_16F, (int)kdim,
                                 dxh, HIP_R_16F, (int)kdim,
                                 &beta, dref, HIP_R_32F, (int)mdim,
                                 HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT),
                   "hipblasGemmEx");
    };
    int device = 0, cu_count = 0;
    check(hipGetDevice(&device), "hipGetDevice");
    check(hipDeviceGetAttribute(&cu_count, hipDeviceAttributeMultiprocessorCount,
                                device), "CU count");
    auto launch_cand = [&](uint32_t set) {
        const _Float16 *wset = dw + (uint64_t)(set % weight_sets) * w_count;
        const dim3 block(32u, 16u);
        if (mdim >= 512u) {
            f16_wvsplitk_n5<TOKENS, 4u, 2u><<<cu_count, block>>>(
                dcand, wset, dxh, kdim, mdim);
        } else {
            f16_wvsplitk_n5<TOKENS, 1u, 4u><<<cu_count, block>>>(
                dcand, wset, dxh, kdim, mdim);
        }
        check(hipGetLastError(), "wvSplitK launch");
    };

    launch_ref(0u);
    launch_cand(0u);
    check(hipDeviceSynchronize(), "warmup synchronize");
    std::vector<float> href(o_count), hcand(o_count);
    check(hipMemcpy(href.data(), dref, o_count * sizeof(float), hipMemcpyDeviceToHost),
          "copy reference");
    check(hipMemcpy(hcand.data(), dcand, o_count * sizeof(float), hipMemcpyDeviceToHost),
          "copy candidate");
    double sq = 0.0, ref_sq = 0.0;
    float max_abs = 0.0f;
    for (uint64_t i = 0; i < o_count; ++i) {
        const float d = std::fabs(href[i] - hcand[i]);
        max_abs = std::max(max_abs, d);
        sq += (double)d * d;
        ref_sq += (double)href[i] * href[i];
    }

    hipEvent_t a, b;
    check(hipEventCreate(&a), "event a");
    check(hipEventCreate(&b), "event b");
    check(hipEventRecord(a), "record ref a");
    for (uint32_t i = 0; i < iterations; ++i) launch_ref(i);
    check(hipEventRecord(b), "record ref b");
    check(hipEventSynchronize(b), "sync ref");
    const float ref_ms = elapsed(a, b) / iterations;
    check(hipEventRecord(a), "record candidate a");
    for (uint32_t i = 0; i < iterations; ++i) launch_cand(i);
    check(hipEventRecord(b), "record candidate b");
    check(hipEventSynchronize(b), "sync candidate");
    const float cand_ms = elapsed(a, b) / iterations;

    std::printf("shape N=%u K=%u M=%u sets=%u hipblas_ms=%.6f wvsplitk_ms=%.6f "
                "speedup=%.3fx max_abs=%.9g rel_rms=%.9g\n",
                TOKENS, kdim, mdim, weight_sets, ref_ms, cand_ms,
                ref_ms / cand_ms, max_abs,
                std::sqrt(sq / std::max(ref_sq, 1e-30)));

    hipEventDestroy(b);
    hipEventDestroy(a);
    hipblasDestroy(blas);
    hipFree(dcand);
    hipFree(dref);
    hipFree(dxf);
    hipFree(dxh);
    hipFree(dw);
}

int main() {
    // V4 Flash Q4_K stores 21 alternating-layer F16 indexer-Q matrices at
    // K=1024, M=8192. Rotate a 256 MiB working set so the oracle does not
    // mistake a repeatedly cache-hot single matrix for production behavior.
    run_shape<1u>(1024u, 8192u, 128u, 16u);
    run_shape<2u>(1024u, 8192u, 128u, 16u);
    run_shape<3u>(1024u, 8192u, 128u, 16u);
    run_shape<4u>(1024u, 8192u, 128u, 16u);
    run_shape<5u>(1024u, 8192u, 128u, 16u);
    run_shape<5u>(4096u, 64u, 500u, 1u); // per-head weight projection
    return 0;
}
