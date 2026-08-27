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
