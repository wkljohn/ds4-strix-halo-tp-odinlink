#pragma once

/*
 * GLM-5 KDA recurrence reference kernel.
 *
 * This is deliberately a small, layout-explicit kernel rather than a graph
 * entry point.  It mirrors the gated-delta-net recurrence used by llama.cpp:
 * state[h][i][j] = exp(g[t,h,i]) * state[h][i][j]
 *                  + k[t,h,i] * beta[t,h] * (v[t,h,j] - prediction[j])
 * prediction[j] = sum_i state[h][i][j] * k[t,h,i]
 * output[j] = sum_i state[h][i][j] * q[t,h,i]
 *
 * The tensors are contiguous row-major: [token][head][channel], and state is
 * [head][key_channel][value_channel].  The production path will replace the
 * intentionally simple column-per-thread implementation after this contract
 * is integrated with the BF16 projections and TP state ownership.
 */

__global__ static void ds4_glm5_kda_ref_kernel(
        const float *q, const float *k, const float *v,
        const float *g, const float *beta,
        float *state, float *out,
        uint32_t n_tokens, uint32_t n_heads, uint32_t channels) {
    const uint32_t h = blockIdx.x;
    const uint32_t j = threadIdx.x;
    if (h >= n_heads || j >= channels) return;

    float s[128];
    if (channels > 128) return;
    for (uint32_t i = 0; i < channels; ++i) {
        s[i] = state[((uint64_t) h * channels + i) * channels + j];
    }

    for (uint32_t t = 0; t < n_tokens; ++t) {
        const uint64_t base = ((uint64_t) t * n_heads + h) * channels;
        float prediction = 0.0f;
        for (uint32_t i = 0; i < channels; ++i) {
            prediction += expf(g[base + i]) * s[i] * k[base + i];
        }
        const float b = beta[(uint64_t) t * n_heads + h];
        for (uint32_t i = 0; i < channels; ++i) {
            s[i] = expf(g[base + i]) * s[i] +
                   k[base + i] * b * (v[base + j] - prediction);
        }
        float y = 0.0f;
        for (uint32_t i = 0; i < channels; ++i) {
            y += s[i] * q[base + i];
        }
        out[base + j] = y;
    }
    for (uint32_t i = 0; i < channels; ++i) {
        state[((uint64_t) h * channels + i) * channels + j] = s[i];
    }
}

/* Warp-tiled variant for the fixed GLM-5 shape (64 heads x 128 channels).
 * Four 32-lane gfx1151 wavefronts cover four value columns; grid.z covers all
 * 128 columns. It keeps four state elements per lane and uses FP32 wave
 * reductions for the key dot product and emitted output. */
__device__ static inline float ds4_glm5_wave_sum32(float x) {
    for (int d = 16; d > 0; d >>= 1) x += __shfl_down(x, d, 32);
    return x;
}

__global__ static void ds4_glm5_kda_warp128_kernel(
        const float *q, const float *k, const float *v,
        const float *g, const float *beta,
        float *state, float *out,
        uint32_t n_tokens, uint32_t n_heads) {
    constexpr uint32_t C = 128;
    const uint32_t h = blockIdx.y;
    const uint32_t lane = threadIdx.x;
    const uint32_t warp = threadIdx.y;
    const uint32_t j = blockIdx.z * 4u + warp;
    if (h >= n_heads || j >= C || lane >= 32) return;
    const uint64_t sbase = ((uint64_t)h * C) * C + j;
    float s0 = state[sbase + (uint64_t)lane * C];
    float s1 = state[sbase + (uint64_t)(lane + 32u) * C];
    float s2 = state[sbase + (uint64_t)(lane + 64u) * C];
    float s3 = state[sbase + (uint64_t)(lane + 96u) * C];
    for (uint32_t t = 0; t < n_tokens; ++t) {
        const uint64_t base = ((uint64_t)t * n_heads + h) * C;
        float dot = expf(g[base + lane]) * s0 * k[base + lane] +
                    expf(g[base + lane + 32u]) * s1 * k[base + lane + 32u] +
                    expf(g[base + lane + 64u]) * s2 * k[base + lane + 64u] +
                    expf(g[base + lane + 96u]) * s3 * k[base + lane + 96u];
        dot = ds4_glm5_wave_sum32(dot);
        dot = __shfl(dot, 0, 32);
        const float d = beta[(uint64_t)t * n_heads + h] * (v[base + j] - dot);
        s0 = expf(g[base + lane]) * s0 + k[base + lane] * d;
        s1 = expf(g[base + lane + 32u]) * s1 + k[base + lane + 32u] * d;
        s2 = expf(g[base + lane + 64u]) * s2 + k[base + lane + 64u] * d;
        s3 = expf(g[base + lane + 96u]) * s3 + k[base + lane + 96u] * d;
        float y = s0 * q[base + lane] + s1 * q[base + lane + 32u] +
                  s2 * q[base + lane + 64u] + s3 * q[base + lane + 96u];
        y = ds4_glm5_wave_sum32(y);
        if (lane == 0) out[base + j] = y;
    }
    state[sbase + (uint64_t)lane * C] = s0;
    state[sbase + (uint64_t)(lane + 32u) * C] = s1;
    state[sbase + (uint64_t)(lane + 64u) * C] = s2;
    state[sbase + (uint64_t)(lane + 96u) * C] = s3;
}
