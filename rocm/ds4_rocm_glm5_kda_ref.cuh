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
            prediction += s[i] * k[base + i];
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
