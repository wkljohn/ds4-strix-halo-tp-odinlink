// Standalone gfx1151 oracle for fusing the ratio-4 compressor maintenance
// performed by a five-row DSpark verifier.  It compares the current ordered
// launch sequence (store each row; pool/RMS/RoPE/shift on emit) with one block
// that preserves the same row order and reduction tree.
//
// This intentionally stops before FP8/indexer-QAT conversion: conversion does
// not feed compressor state and can remain a separate batched post-pass.
//
// Build:
//   /opt/rocm/bin/hipcc -O3 -ffast-math -fno-finite-math-only \
//     --offload-arch=gfx1151 scripts/dspark_compressor_five_row_bench.cu \
//     -o /tmp/dspark_compressor_five_row_bench
#include "../ds4_rocm.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define HIP_OK(expr) do {                                                   \
    hipError_t e_ = (expr);                                                 \
    if (e_ != hipSuccess) {                                                 \
        std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,          \
                     hipGetErrorString(e_));                                \
        std::exit(1);                                                       \
    }                                                                       \
} while (0)

__device__ __forceinline__ float rope_ramp(
        float low, float high, uint32_t i) {
    const float y = ((float)(i / 2u) - low) /
                    fmaxf(0.001f, high - low);
    return 1.0f - fminf(1.0f, fmaxf(0.0f, y));
}

__device__ __forceinline__ void rope_pair(
        float *row, uint32_t head_dim, uint32_t n_rot, uint32_t pair,
        uint32_t pos, uint32_t n_ctx_orig, float freq_base,
        float freq_scale, float ext_factor, float attn_factor,
        float beta_fast, float beta_slow) {
    const uint32_t i = pair * 2u;
    const uint32_t n_nope = head_dim - n_rot;
    float corr0 = 0.0f, corr1 = 0.0f;
    if (ext_factor != 0.0f) {
        const float denom = 2.0f * logf(freq_base);
        corr0 = floorf((float)n_rot *
                       logf((float)n_ctx_orig /
                            (beta_fast * 2.0f * (float)M_PI)) / denom);
        corr1 = ceilf((float)n_rot *
                      logf((float)n_ctx_orig /
                           (beta_slow * 2.0f * (float)M_PI)) / denom);
        corr0 = fmaxf(0.0f, corr0);
        corr1 = fminf((float)(n_rot - 1u), corr1);
    }
    const float theta_scale = powf(freq_base, -2.0f / (float)n_rot);
    const float theta_extrap =
        (float)pos * powf(theta_scale, (float)pair);
    const float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    float mscale = attn_factor;
    if (ext_factor != 0.0f) {
        const float mix = rope_ramp(corr0, corr1, i) * ext_factor;
        theta = theta_interp * (1.0f - mix) + theta_extrap * mix;
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    const float c = cosf(theta) * mscale;
    const float s = sinf(theta) * mscale;
    const float x0 = row[n_nope + i];
    const float x1 = row[n_nope + i + 1u];
    row[n_nope + i] = x0 * c - x1 * s;
    row[n_nope + i + 1u] = x0 * s + x1 * c;
}

__global__ void reference_store(
        float *state_kv, float *state_sc, const float *kv, const float *sc,
        const float *ape, uint32_t head_dim, uint32_t pos, uint32_t token) {
    const uint32_t width = 2u * head_dim;
    const uint32_t phase = pos & 3u;
    const uint32_t dst = 4u + phase;
    for (uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;
         j < width; j += gridDim.x * blockDim.x) {
        state_kv[(uint64_t)dst * width + j] =
            kv[(uint64_t)token * width + j];
        state_sc[(uint64_t)dst * width + j] =
            sc[(uint64_t)token * width + j] +
            ape[(uint64_t)phase * width + j];
    }
}

__global__ void reference_pool(
        float *out, const float *state_kv, const float *state_sc,
        uint32_t head_dim) {
    const uint32_t d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= head_dim) return;
    const uint32_t width = 2u * head_dim;
    float v[8], s[8], max_s = -INFINITY;
#pragma unroll
    for (uint32_t r = 0; r < 4u; ++r) {
        v[r] = state_kv[(uint64_t)r * width + d];
        s[r] = state_sc[(uint64_t)r * width + d];
        max_s = fmaxf(max_s, s[r]);
        v[4u + r] = state_kv[(uint64_t)(4u + r) * width + head_dim + d];
        s[4u + r] = state_sc[(uint64_t)(4u + r) * width + head_dim + d];
        max_s = fmaxf(max_s, s[4u + r]);
    }
    float den = 0.0f, acc = 0.0f;
#pragma unroll
    for (uint32_t i = 0; i < 8u; ++i) {
        const float w = expf(s[i] - max_s);
        den += w;
        acc += v[i] * w;
    }
    out[d] = den != 0.0f ? acc / den : 0.0f;
}

__global__ void reference_rms(
        float *rows, const float *weight, uint32_t n,
        uint32_t n_rows, float eps) {
    const uint32_t row_id = blockIdx.x;
    if (row_id >= n_rows) return;
    float *row = rows + (uint64_t)row_id * n;
    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = row[i];
        sum += v * v;
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = 128u; stride != 0u; stride >>= 1u) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }
    const float scale = rsqrtf(partial[0] / (float)n + eps);
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        row[i] = row[i] * scale * weight[i];
    }
}

__global__ void reference_rope(
        float *rows, uint32_t n_rows, uint32_t head_dim, uint32_t n_rot,
        uint32_t pos0, uint32_t pos_stride,
        uint32_t n_ctx_orig, float freq_base, float freq_scale,
        float ext_factor, float attn_factor, float beta_fast,
        float beta_slow) {
    const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t pairs_per_row = n_rot / 2u;
    if (gid < n_rows * pairs_per_row) {
        const uint32_t row_id = gid / pairs_per_row;
        const uint32_t pair = gid - row_id * pairs_per_row;
        rope_pair(rows + (uint64_t)row_id * head_dim,
                  head_dim, n_rot, pair, pos0 + row_id * pos_stride, n_ctx_orig,
                  freq_base, freq_scale, ext_factor, attn_factor,
                  beta_fast, beta_slow);
    }
}

__global__ void reference_shift(
        float *state_kv, float *state_sc, uint32_t width) {
    const uint64_t half = 4ull * width;
    for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < half; i += (uint64_t)gridDim.x * blockDim.x) {
        const float v = state_kv[half + i];
        const float s = state_sc[half + i];
        state_kv[i] = v;
        state_sc[i] = s;
        state_kv[half + i] = v;
        state_sc[half + i] = s;
    }
}

__global__ void fused_five_rows(
        float *out, float *state_kv, float *state_sc,
        const float *kv, const float *sc, const float *ape,
        uint32_t head_dim, uint32_t pos0) {
    const uint32_t tid = threadIdx.x;
    const uint32_t width = 2u * head_dim;
    uint32_t emitted = 0;
    for (uint32_t t = 0; t < 5u; ++t) {
        const uint32_t pos = pos0 + t;
        const uint32_t phase = pos & 3u;
        const uint32_t dst = 4u + phase;
        for (uint32_t j = tid; j < width; j += blockDim.x) {
            state_kv[(uint64_t)dst * width + j] =
                kv[(uint64_t)t * width + j];
            state_sc[(uint64_t)dst * width + j] =
                sc[(uint64_t)t * width + j] +
                ape[(uint64_t)phase * width + j];
        }
        __syncthreads();
        if (((pos + 1u) & 3u) == 0u) {
            float *row = out + (uint64_t)emitted * head_dim;
            for (uint32_t d = tid; d < head_dim; d += blockDim.x) {
                float v[8], s[8], max_s = -INFINITY;
#pragma unroll
                for (uint32_t r = 0; r < 4u; ++r) {
                    v[r] = state_kv[(uint64_t)r * width + d];
                    s[r] = state_sc[(uint64_t)r * width + d];
                    max_s = fmaxf(max_s, s[r]);
                    v[4u + r] =
                        state_kv[(uint64_t)(4u + r) * width + head_dim + d];
                    s[4u + r] =
                        state_sc[(uint64_t)(4u + r) * width + head_dim + d];
                    max_s = fmaxf(max_s, s[4u + r]);
                }
                float den = 0.0f, acc = 0.0f;
#pragma unroll
                for (uint32_t i = 0; i < 8u; ++i) {
                    const float w = expf(s[i] - max_s);
                    den += w;
                    acc += v[i] * w;
                }
                row[d] = den != 0.0f ? acc / den : 0.0f;
            }
            __syncthreads();
            const uint64_t half = 4ull * width;
            for (uint64_t i = tid; i < half; i += blockDim.x) {
                const float v = state_kv[half + i];
                const float s = state_sc[half + i];
                state_kv[i] = v;
                state_sc[i] = s;
                state_kv[half + i] = v;
                state_sc[half + i] = s;
            }
            ++emitted;
            __syncthreads();
        }
    }
}

template <typename Launch>
double time_gpu(Launch launch, int warmup = 20, int iters = 500) {
    for (int i = 0; i < warmup; ++i) launch();
    HIP_OK(hipDeviceSynchronize());
    hipEvent_t a, b;
    HIP_OK(hipEventCreate(&a));
    HIP_OK(hipEventCreate(&b));
    HIP_OK(hipEventRecord(a));
    for (int i = 0; i < iters; ++i) launch();
    HIP_OK(hipEventRecord(b));
    HIP_OK(hipEventSynchronize(b));
    float ms = 0.0f;
    HIP_OK(hipEventElapsedTime(&ms, a, b));
    HIP_OK(hipEventDestroy(a));
    HIP_OK(hipEventDestroy(b));
    return (double)ms / iters;
}

struct DeviceBuffers {
    float *state0 = nullptr, *score0 = nullptr;
    float *state_ref = nullptr, *score_ref = nullptr;
    float *state_fused = nullptr, *score_fused = nullptr;
    float *kv = nullptr, *sc = nullptr, *ape = nullptr, *weight = nullptr;
    float *out_ref = nullptr, *out_fused = nullptr;
};

static void alloc_buffers(DeviceBuffers &d, uint32_t head_dim) {
    const uint64_t width = 2ull * head_dim;
    const uint64_t state_bytes = 8ull * width * sizeof(float);
    const uint64_t input_bytes = 5ull * width * sizeof(float);
    const uint64_t ape_bytes = 4ull * width * sizeof(float);
    const uint64_t out_bytes = 2ull * head_dim * sizeof(float);
    HIP_OK(hipMalloc(&d.state0, state_bytes));
    HIP_OK(hipMalloc(&d.score0, state_bytes));
    HIP_OK(hipMalloc(&d.state_ref, state_bytes));
    HIP_OK(hipMalloc(&d.score_ref, state_bytes));
    HIP_OK(hipMalloc(&d.state_fused, state_bytes));
    HIP_OK(hipMalloc(&d.score_fused, state_bytes));
    HIP_OK(hipMalloc(&d.kv, input_bytes));
    HIP_OK(hipMalloc(&d.sc, input_bytes));
    HIP_OK(hipMalloc(&d.ape, ape_bytes));
    HIP_OK(hipMalloc(&d.weight, head_dim * sizeof(float)));
    HIP_OK(hipMalloc(&d.out_ref, out_bytes));
    HIP_OK(hipMalloc(&d.out_fused, out_bytes));
}

static void free_buffers(DeviceBuffers &d) {
    HIP_OK(hipFree(d.out_fused)); HIP_OK(hipFree(d.out_ref));
    HIP_OK(hipFree(d.weight)); HIP_OK(hipFree(d.ape));
    HIP_OK(hipFree(d.sc)); HIP_OK(hipFree(d.kv));
    HIP_OK(hipFree(d.score_fused)); HIP_OK(hipFree(d.state_fused));
    HIP_OK(hipFree(d.score_ref)); HIP_OK(hipFree(d.state_ref));
    HIP_OK(hipFree(d.score0)); HIP_OK(hipFree(d.state0));
}

static uint32_t emitted_rows(uint32_t pos0) {
    uint32_t n = 0;
    for (uint32_t t = 0; t < 5u; ++t) {
        n += (((pos0 + t + 1u) & 3u) == 0u);
    }
    return n;
}

static void launch_reference(DeviceBuffers &d, uint32_t head_dim,
                             uint32_t pos0) {
    const uint32_t width = 2u * head_dim;
    const size_t state_bytes = 8ull * width * sizeof(float);
    HIP_OK(hipMemcpyAsync(d.state_ref, d.state0, state_bytes,
                          hipMemcpyDeviceToDevice));
    HIP_OK(hipMemcpyAsync(d.score_ref, d.score0, state_bytes,
                          hipMemcpyDeviceToDevice));
    uint32_t emitted = 0;
    for (uint32_t t = 0; t < 5u; ++t) {
        const uint32_t pos = pos0 + t;
        reference_store<<<(width + 255u) / 256u, 256>>>(
            d.state_ref, d.score_ref, d.kv, d.sc, d.ape,
            head_dim, pos, t);
        if (((pos + 1u) & 3u) == 0u) {
            float *row = d.out_ref + (uint64_t)emitted * head_dim;
            reference_pool<<<(head_dim + 255u) / 256u, 256>>>(
                row, d.state_ref, d.score_ref, head_dim);
            reference_rms<<<1, 256>>>(row, d.weight, head_dim, 1u, 1e-6f);
            reference_rope<<<1, 256>>>(row, 1u, head_dim, 64u,
                pos + 1u - 4u, 4u, 4096u, 10000.0f, 0.5f, 1.0f,
                1.0f, 32.0f, 1.0f);
            reference_shift<<<((uint64_t)4u * width + 255u) / 256u, 256>>>(
                d.state_ref, d.score_ref, width);
            ++emitted;
        }
    }
}

static void launch_fused(DeviceBuffers &d, uint32_t head_dim,
                         uint32_t pos0) {
    const uint32_t width = 2u * head_dim;
    const size_t state_bytes = 8ull * width * sizeof(float);
    HIP_OK(hipMemcpyAsync(d.state_fused, d.state0, state_bytes,
                          hipMemcpyDeviceToDevice));
    HIP_OK(hipMemcpyAsync(d.score_fused, d.score0, state_bytes,
                          hipMemcpyDeviceToDevice));
    fused_five_rows<<<1, 256>>>(
        d.out_fused, d.state_fused, d.score_fused, d.kv, d.sc,
        d.ape, head_dim, pos0);
    const uint32_t emits = emitted_rows(pos0);
    reference_rms<<<emits, 256>>>(
        d.out_fused, d.weight, head_dim, emits, 1e-6f);
    const uint32_t rope_pairs = emits * 32u;
    reference_rope<<<(rope_pairs + 255u) / 256u, 256>>>(
        d.out_fused, emits, head_dim, 64u, pos0 & ~3u, 4u,
        4096u, 10000.0f, 0.5f, 1.0f, 1.0f, 32.0f, 1.0f);
}

static void compare(const char *label, const std::vector<float> &a,
                    const std::vector<float> &b) {
    uint64_t bit_diff = 0;
    double sq = 0.0, ref_sq = 0.0;
    float max_abs = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        uint32_t ua = 0, ub = 0;
        std::memcpy(&ua, &a[i], sizeof(ua));
        std::memcpy(&ub, &b[i], sizeof(ub));
        bit_diff += ua != ub;
        const float e = std::fabs(a[i] - b[i]);
        max_abs = std::max(max_abs, e);
        sq += (double)e * e;
        ref_sq += (double)a[i] * a[i];
    }
    std::printf(" %s[bit_diff=%llu/%zu max_abs=%.9g rel_rms=%.9g]",
                label, (unsigned long long)bit_diff, a.size(), max_abs,
                ref_sq != 0.0 ? std::sqrt(sq / ref_sq) : 0.0);
}

static void run_shape(uint32_t head_dim) {
    const uint32_t width = 2u * head_dim;
    const size_t state_n = 8ull * width;
    const size_t input_n = 5ull * width;
    const size_t ape_n = 4ull * width;
    std::mt19937 rng(0x4d535034u + head_dim);
    std::uniform_real_distribution<float> val(-1.0f, 1.0f);
    std::vector<float> state(state_n), score(state_n), kv(input_n),
                       sc(input_n), ape(ape_n), weight(head_dim);
    for (float &x : state) x = val(rng);
    for (float &x : score) x = 0.25f * val(rng);
    for (float &x : kv) x = val(rng);
    for (float &x : sc) x = 0.25f * val(rng);
    for (float &x : ape) x = 0.1f * val(rng);
    for (float &x : weight) x = 0.8f + 0.4f * (val(rng) + 1.0f) * 0.5f;

    DeviceBuffers d;
    alloc_buffers(d, head_dim);
    HIP_OK(hipMemcpy(d.state0, state.data(), state.size() * sizeof(float),
                     hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(d.score0, score.data(), score.size() * sizeof(float),
                     hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(d.kv, kv.data(), kv.size() * sizeof(float),
                     hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(d.sc, sc.data(), sc.size() * sizeof(float),
                     hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(d.ape, ape.data(), ape.size() * sizeof(float),
                     hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(d.weight, weight.data(), weight.size() * sizeof(float),
                     hipMemcpyHostToDevice));

    for (uint32_t phase = 0; phase < 4u; ++phase) {
        const uint32_t pos0 = 2048u + phase;
        launch_reference(d, head_dim, pos0);
        launch_fused(d, head_dim, pos0);
        HIP_OK(hipDeviceSynchronize());
        const uint32_t emits = emitted_rows(pos0);
        std::vector<float> out_a((size_t)emits * head_dim), out_b(out_a.size());
        std::vector<float> state_a(state_n), state_b(state_n);
        std::vector<float> score_a(state_n), score_b(state_n);
        HIP_OK(hipMemcpy(out_a.data(), d.out_ref,
                         out_a.size() * sizeof(float), hipMemcpyDeviceToHost));
        HIP_OK(hipMemcpy(out_b.data(), d.out_fused,
                         out_b.size() * sizeof(float), hipMemcpyDeviceToHost));
        HIP_OK(hipMemcpy(state_a.data(), d.state_ref,
                         state_a.size() * sizeof(float), hipMemcpyDeviceToHost));
        HIP_OK(hipMemcpy(state_b.data(), d.state_fused,
                         state_b.size() * sizeof(float), hipMemcpyDeviceToHost));
        HIP_OK(hipMemcpy(score_a.data(), d.score_ref,
                         score_a.size() * sizeof(float), hipMemcpyDeviceToHost));
        HIP_OK(hipMemcpy(score_b.data(), d.score_fused,
                         score_b.size() * sizeof(float), hipMemcpyDeviceToHost));
        std::printf("shape=H%u phase=%u emits=%u", head_dim, phase, emits);
        compare("out", out_a, out_b);
        compare("state", state_a, state_b);
        compare("score", score_a, score_b);
        std::printf("\n");
    }

    const uint32_t bench_pos0 = 2048u;
    const double ref_ms = time_gpu([&] {
        launch_reference(d, head_dim, bench_pos0);
    });
    const double fused_ms = time_gpu([&] {
        launch_fused(d, head_dim, bench_pos0);
    });
    std::printf("shape=H%u reference_ms=%.6f fused_ms=%.6f speedup=%.3fx "
                "launches=9->3(including two D2D resets)\n",
                head_dim, ref_ms, fused_ms, ref_ms / fused_ms);
    free_buffers(d);
}

int main() {
    hipDeviceProp_t prop{};
    HIP_OK(hipGetDeviceProperties(&prop, 0));
    std::printf("device=%s arch=%s CUs=%d ntok=5 ratio=4\n",
                prop.name, prop.gcnArchName, prop.multiProcessorCount);
    run_shape(512u);
    run_shape(128u);
    return 0;
}
