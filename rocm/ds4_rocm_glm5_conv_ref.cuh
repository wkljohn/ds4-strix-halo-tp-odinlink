#pragma once

/* GLM-5's depthwise causal short convolution reference kernel.  Input and
 * output are [token, channel], weight is [channel, tap], and history is
 * [channel, tap-1] with the oldest sample at index zero. */
__global__ static void ds4_glm5_causal_conv4_ref_kernel(
        const float *input, const float *weight, float *history,
        float *output, uint32_t n_tokens, uint32_t channels) {
    const uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    float h0 = history[c * 3 + 0];
    float h1 = history[c * 3 + 1];
    float h2 = history[c * 3 + 2];
    for (uint32_t t = 0; t < n_tokens; ++t) {
        const float x = input[(uint64_t)t * channels + c];
        const float y = h0 * weight[c * 4 + 0] + h1 * weight[c * 4 + 1] +
                        h2 * weight[c * 4 + 2] + x * weight[c * 4 + 3];
        output[(uint64_t)t * channels + c] = y / (1.0f + expf(-y));
        h0 = h1; h1 = h2; h2 = x;
    }
    history[c * 3 + 0] = h0;
    history[c * 3 + 1] = h1;
    history[c * 3 + 2] = h2;
}
