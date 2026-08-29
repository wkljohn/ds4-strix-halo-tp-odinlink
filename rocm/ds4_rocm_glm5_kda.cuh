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
        uint32_t n_tokens,
        uint32_t heads) {
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

__device__ static inline float ds4_glm5_sigmoid(float value) {
    if (value >= 0.0f) {
        const float e = expf(-value);
        return 1.0f / (1.0f + e);
    }
    const float e = expf(value);
    return e / (1.0f + e);
}

/* One block owns one [head, token] pair so both norms have a fixed reduction
 * order. q additionally carries GLM-5's 1/sqrt(128) attention scale. */
__global__ static void ds4_glm5_kda_qk_norm_kernel(
        float *q,
        float *k,
        uint32_t n_tokens,
        uint32_t heads) {
    constexpr uint32_t channels = 128u;
    const uint32_t row = blockIdx.x;
    const uint32_t lane = threadIdx.x;
    if (row >= n_tokens * heads || lane >= channels) return;
    __shared__ float qsum[channels];
    __shared__ float ksum[channels];
    const uint64_t index = (uint64_t)row * channels + lane;
    const float qv = q[index];
    const float kv = k[index];
    qsum[lane] = qv * qv;
    ksum[lane] = kv * kv;
    __syncthreads();
    for (uint32_t stride = channels / 2u; stride != 0u; stride >>= 1u) {
        if (lane < stride) {
            qsum[lane] += qsum[lane + stride];
            ksum[lane] += ksum[lane + stride];
        }
        __syncthreads();
    }
    q[index] = qv * rsqrtf(qsum[0] + 1.0e-6f) *
               0.08838834764831845f;
    k[index] = kv * rsqrtf(ksum[0] + 1.0e-6f);
}

__global__ static void ds4_glm5_kda_forget_kernel(
        float *forget,
        const float *dt_bias,
        const float *a_log,
        uint64_t values,
        uint32_t heads) {
    constexpr uint32_t channels = 128u;
    const uint64_t index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= values) return;
    const uint32_t channel = (uint32_t)(index % channels);
    const uint32_t head = (uint32_t)((index / channels) % heads);
    const float scaled = expf(a_log[head]) *
                         (forget[index] + dt_bias[(uint64_t)head * channels +
                                                  channel]);
    forget[index] = -5.0f * ds4_glm5_sigmoid(scaled);
}

__global__ static void ds4_glm5_kda_beta_kernel(float *beta,
                                                  uint64_t values) {
    const uint64_t index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index < values) beta[index] = ds4_glm5_sigmoid(beta[index]);
}

__global__ static void ds4_glm5_kda_gated_norm_kernel(
        float *output,
        const float *input,
        const float *gate,
        const float *weight,
        uint32_t n_tokens,
        uint32_t heads,
        float norm_eps) {
    constexpr uint32_t channels = 128u;
    const uint32_t row = blockIdx.x;
    const uint32_t lane = threadIdx.x;
    if (row >= n_tokens * heads || lane >= channels) return;
    __shared__ float squares[channels];
    const uint64_t index = (uint64_t)row * channels + lane;
    const float value = input[index];
    squares[lane] = value * value;
    __syncthreads();
    for (uint32_t stride = channels / 2u; stride != 0u; stride >>= 1u) {
        if (lane < stride) squares[lane] += squares[lane + stride];
        __syncthreads();
    }
    const float scale = rsqrtf(squares[0] / (float)channels + norm_eps);
    output[index] = value * scale * weight[lane] *
                    ds4_glm5_sigmoid(gate[index]);
}

/* Compose the two canonical 32-head rank halves into token-major 64-head
 * order without changing any floating-point value. */
__global__ static void ds4_glm5_kda_compose_head_halves_kernel(
        float *full,
        const float *rank0,
        const float *rank1,
        uint64_t values) {
    constexpr uint32_t half = 4096u;
    constexpr uint32_t full_width = 8192u;
    const uint64_t index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= values) return;
    const uint32_t column = (uint32_t)(index % full_width);
    const uint64_t token = index / full_width;
    full[index] = column < half
        ? rank0[token * half + column]
        : rank1[token * half + column - half];
}
