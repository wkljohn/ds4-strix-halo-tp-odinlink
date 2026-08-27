#pragma once

/* Fixed-shape GLM-5 depthwise causal convolution. Input/output are
 * [token, channel], weights are [channel, tap], and history is
 * [channel, tap-1] oldest-first. Each channel is owned by one thread so a
 * continuation call observes exactly the state committed by the prior call. */
__global__ static void ds4_glm5_causal_conv4_kernel(
        float *output,
        float *history,
        const float *input,
        const float *weight,
        uint32_t n_tokens,
        uint32_t channels) {
    const uint32_t channel = blockIdx.x * blockDim.x + threadIdx.x;
    if (channel >= channels) return;
    float h0 = history[(uint64_t)channel * 3u];
    float h1 = history[(uint64_t)channel * 3u + 1u];
    float h2 = history[(uint64_t)channel * 3u + 2u];
    const uint64_t w = (uint64_t)channel * 4u;
    for (uint32_t token = 0; token < n_tokens; ++token) {
        const uint64_t index = (uint64_t)token * channels + channel;
        const float current = input[index];
        const float raw = h0 * weight[w] + h1 * weight[w + 1u] +
                          h2 * weight[w + 2u] + current * weight[w + 3u];
        output[index] = raw / (1.0f + expf(-raw));
        h0 = h1;
        h1 = h2;
        h2 = current;
    }
    history[(uint64_t)channel * 3u] = h0;
    history[(uint64_t)channel * 3u + 1u] = h1;
    history[(uint64_t)channel * 3u + 2u] = h2;
}

__device__ static inline float ds4_glm5_wave_sum32(float value) {
    for (int delta = 16; delta > 0; delta >>= 1)
        value += __shfl_down(value, delta, 32);
    return value;
}

#if defined(DS4_GFX1151_WAVE32) && DS4_GFX1151_WAVE32
/* Fixed GLM-5 KDA recurrence. Four gfx1151 wave32 wavefronts cover four
 * value columns; grid.z covers all 128 columns. Each lane retains four key
 * rows in registers across the token loop. */
__global__ static void ds4_glm5_kda_wave32_kernel(
        float *output,
        float *state,
        const float *q,
        const float *k,
        const float *v,
        const float *gate,
        const float *beta,
        uint32_t n_tokens) {
    constexpr uint32_t heads = 64u;
    constexpr uint32_t channels = 128u;
    const uint32_t head = blockIdx.y;
    const uint32_t lane = threadIdx.x;
    const uint32_t wave = threadIdx.y;
    const uint32_t value_channel = blockIdx.z * 4u + wave;
    if (head >= heads || lane >= 32u || value_channel >= channels) return;

    const uint64_t state_base =
        ((uint64_t)head * channels) * channels + value_channel;
    float s0 = state[state_base + (uint64_t)lane * channels];
    float s1 = state[state_base + (uint64_t)(lane + 32u) * channels];
    float s2 = state[state_base + (uint64_t)(lane + 64u) * channels];
    float s3 = state[state_base + (uint64_t)(lane + 96u) * channels];
    for (uint32_t token = 0; token < n_tokens; ++token) {
        const uint64_t base =
            ((uint64_t)token * heads + head) * channels;
        const float d0 = expf(gate[base + lane]);
        const float d1 = expf(gate[base + lane + 32u]);
        const float d2 = expf(gate[base + lane + 64u]);
        const float d3 = expf(gate[base + lane + 96u]);
        s0 *= d0;
        s1 *= d1;
        s2 *= d2;
        s3 *= d3;
        float prediction = s0 * k[base + lane] +
                           s1 * k[base + lane + 32u] +
                           s2 * k[base + lane + 64u] +
                           s3 * k[base + lane + 96u];
        prediction = ds4_glm5_wave_sum32(prediction);
        prediction = __shfl(prediction, 0, 32);
        const float correction = beta[(uint64_t)token * heads + head] *
            (v[base + value_channel] - prediction);
        s0 += k[base + lane] * correction;
        s1 += k[base + lane + 32u] * correction;
        s2 += k[base + lane + 64u] * correction;
        s3 += k[base + lane + 96u] * correction;
        float emitted = s0 * q[base + lane] +
                        s1 * q[base + lane + 32u] +
                        s2 * q[base + lane + 64u] +
                        s3 * q[base + lane + 96u];
        emitted = ds4_glm5_wave_sum32(emitted);
        if (lane == 0u) output[base + value_channel] = emitted;
    }
    state[state_base + (uint64_t)lane * channels] = s0;
    state[state_base + (uint64_t)(lane + 32u) * channels] = s1;
    state[state_base + (uint64_t)(lane + 64u) * channels] = s2;
    state[state_base + (uint64_t)(lane + 96u) * channels] = s3;
}
#endif
