/*
 * Production-geometry GLM-5.3 NoPE decode attention microbenchmark.
 *
 * Build:
 *   $DS4_ROCM_HOME/bin/hipcc -O3 -ffast-math -g \
 *     -fno-finite-math-only --offload-arch=gfx1151 \
 *     -mno-wavefrontsize64 -DDS4_GFX1151_WAVE32=1 \
 *     -o /tmp/glm5-nope-attn-bench \
 *     scripts/glm5_nope_attention_exact_bench.cu
 */

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

namespace {

constexpr uint32_t kHeads = 64u;
constexpr uint32_t kLatent = 512u;
constexpr uint32_t kSelected = 2051u;
constexpr uint32_t kCacheRows = 2051u;
constexpr uint32_t kLayers = 11u;
constexpr uint32_t kThreads = 256u;
constexpr uint32_t kHeadGroup = 8u;
constexpr uint32_t kRowTile = 32u;
constexpr uint32_t kColumnTile = 32u;
constexpr uint32_t kPvRowTile = 16u;
constexpr uint32_t kPrefillTokens = 256u;
constexpr uint32_t kPrefillSelected = 2048u;
constexpr uint32_t kPrefillRowTile = 16u;
constexpr uint32_t kPrefillScoreHeadGroup = 64u;
constexpr uint32_t kPrefillScoreThreads =
    kPrefillScoreHeadGroup * kPrefillRowTile;
constexpr uint32_t kPrefillPvHeadGroup = 16u;
constexpr uint32_t kPrefillPvThreads =
    kPrefillPvHeadGroup * kColumnTile;

#define HIP_CHECK(expr) do {                                                \
    const hipError_t status_ = (expr);                                      \
    if (status_ != hipSuccess) {                                            \
        std::fprintf(stderr, "HIP failure %s:%d: %s\n",                    \
                     __FILE__, __LINE__, hipGetErrorString(status_));        \
        std::exit(1);                                                       \
    }                                                                       \
} while (0)

__global__ void incumbent_kernel(
        float *out,
        const float *low,
        const float *cache,
        const int32_t *selected,
        uint32_t n_selected,
        uint32_t cache_rows) {
    const uint32_t head = blockIdx.x;
    extern __shared__ float sh[];
    float *max_red = sh;
    float *sum_red = sh + 256u;
    float *scores = sh + 512u;
    const float *head_low = low + (uint64_t)head * kLatent;

    float local_max = -INFINITY;
    for (uint32_t s = threadIdx.x; s < n_selected; s += kThreads) {
        const int32_t row_i = selected[s];
        const bool valid = row_i >= 0 && (uint32_t)row_i < cache_rows;
        float score = -INFINITY;
        if (valid) {
            const float *row = cache + (uint64_t)(uint32_t)row_i * kLatent;
            float dot = 0.0f;
            for (uint32_t j = 0u; j < kLatent; ++j) dot += head_low[j] * row[j];
            score = dot * 0.0625f;
        }
        scores[s] = score;
        local_max = fmaxf(local_max, score);
    }
    max_red[threadIdx.x] = local_max;
    __syncthreads();
    for (uint32_t stride = 128u; stride > 0u; stride >>= 1u) {
        if (threadIdx.x < stride) {
            max_red[threadIdx.x] =
                fmaxf(max_red[threadIdx.x], max_red[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    const float maximum = max_red[0];
    const bool valid_maximum = isfinite(maximum);
    float local_sum = 0.0f;
    for (uint32_t s = threadIdx.x; s < n_selected; s += kThreads) {
        const float weight =
            valid_maximum ? expf(scores[s] - maximum) : 0.0f;
        scores[s] = weight;
        local_sum += weight;
    }
    sum_red[threadIdx.x] = local_sum;
    __syncthreads();
    for (uint32_t stride = 128u; stride > 0u; stride >>= 1u) {
        if (threadIdx.x < stride) {
            sum_red[threadIdx.x] += sum_red[threadIdx.x + stride];
        }
        __syncthreads();
    }
    const float denom = fmaxf(sum_red[0], 1.0e-20f);
    for (uint32_t j = threadIdx.x; j < kLatent; j += kThreads) {
        float acc = 0.0f;
        for (uint32_t s = 0u; s < n_selected; ++s) {
            const int32_t row_i = selected[s];
            if (row_i >= 0 && (uint32_t)row_i < cache_rows) {
                acc += scores[s] *
                    cache[(uint64_t)(uint32_t)row_i * kLatent + j];
            }
        }
        out[(uint64_t)head * kLatent + j] =
            valid_maximum ? acc / denom : 0.0f;
    }
}

__global__ void incumbent_qk_weights_kernel(
        float *weights,
        float *denoms,
        const float *low,
        const float *cache,
        const int32_t *selected,
        uint32_t n_selected,
        uint32_t cache_rows) {
    const uint32_t head = blockIdx.x;
    extern __shared__ float sh[];
    float *max_red = sh;
    float *sum_red = sh + 256u;
    float *scores = sh + 512u;
    const float *head_low = low + (uint64_t)head * kLatent;
    float local_max = -INFINITY;
    for (uint32_t s = threadIdx.x; s < n_selected; s += kThreads) {
        const int32_t row_i = selected[s];
        const bool valid = row_i >= 0 && (uint32_t)row_i < cache_rows;
        float score = -INFINITY;
        if (valid) {
            const float *row = cache + (uint64_t)(uint32_t)row_i * kLatent;
            float dot = 0.0f;
            for (uint32_t j = 0u; j < kLatent; ++j) dot += head_low[j] * row[j];
            score = dot * 0.0625f;
        }
        scores[s] = score;
        local_max = fmaxf(local_max, score);
    }
    max_red[threadIdx.x] = local_max;
    __syncthreads();
    for (uint32_t stride = 128u; stride > 0u; stride >>= 1u) {
        if (threadIdx.x < stride) {
            max_red[threadIdx.x] =
                fmaxf(max_red[threadIdx.x], max_red[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    const float maximum = max_red[0];
    float local_sum = 0.0f;
    for (uint32_t s = threadIdx.x; s < n_selected; s += kThreads) {
        const float weight = isfinite(maximum) ? expf(scores[s] - maximum) : 0.0f;
        scores[s] = weight;
        weights[(uint64_t)head * n_selected + s] = weight;
        local_sum += weight;
    }
    sum_red[threadIdx.x] = local_sum;
    __syncthreads();
    for (uint32_t stride = 128u; stride > 0u; stride >>= 1u) {
        if (threadIdx.x < stride) {
            sum_red[threadIdx.x] += sum_red[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0u) denoms[head] = fmaxf(sum_red[0], 1.0e-20f);
}

/* Eight heads share each selected-row load.  One thread still owns one
 * (head,row) scalar chain and visits j=0..511 in incumbent order. */
__global__ void shared_qk_kernel(
        float *raw_scores,
        const float *low,
        const float *cache,
        const int32_t *selected,
        uint32_t n_selected,
        uint32_t cache_rows) {
    const uint32_t head_lane = threadIdx.x / kRowTile;
    const uint32_t row_lane = threadIdx.x % kRowTile;
    const uint32_t head = blockIdx.x * kHeadGroup + head_lane;
    const uint32_t s = blockIdx.y * kRowTile + row_lane;
    const bool in_range = s < n_selected;
    const int32_t row_i = in_range ? selected[s] : -1;
    const bool valid = row_i >= 0 && (uint32_t)row_i < cache_rows;
    const uint32_t load_row = valid ? (uint32_t)row_i : 0u;
    const float *head_low = low + (uint64_t)head * kLatent;
    __shared__ float tile[kRowTile][257u];
    float dot = 0.0f;

    for (uint32_t half = 0u; half < 2u; ++half) {
        for (uint32_t linear = threadIdx.x;
             linear < kRowTile * 256u;
             linear += kThreads) {
            const uint32_t tile_row = linear / 256u;
            const uint32_t column = linear % 256u;
            const uint32_t tile_s = blockIdx.y * kRowTile + tile_row;
            const int32_t tile_row_i =
                tile_s < n_selected ? selected[tile_s] : -1;
            const uint32_t source_row =
                tile_row_i >= 0 && (uint32_t)tile_row_i < cache_rows
                    ? (uint32_t)tile_row_i : 0u;
            tile[tile_row][column] =
                cache[(uint64_t)source_row * kLatent + half * 256u + column];
        }
        __syncthreads();
        for (uint32_t column = 0u; column < 256u; ++column) {
            dot += head_low[half * 256u + column] * tile[row_lane][column];
        }
        __syncthreads();
    }
    if (in_range) {
        raw_scores[(uint64_t)head * n_selected + s] =
            valid ? dot * 0.0625f : -INFINITY;
    }
    (void)load_row;
}

__global__ void exact_softmax_kernel(
        float *weights,
        float *denoms,
        uint32_t n_selected) {
    const uint32_t head = blockIdx.x;
    extern __shared__ float sh[];
    float *max_red = sh;
    float *sum_red = sh + 256u;
    float local_max = -INFINITY;
    for (uint32_t s = threadIdx.x; s < n_selected; s += kThreads) {
        local_max = fmaxf(local_max, weights[(uint64_t)head * n_selected + s]);
    }
    max_red[threadIdx.x] = local_max;
    __syncthreads();
    for (uint32_t stride = 128u; stride > 0u; stride >>= 1u) {
        if (threadIdx.x < stride) {
            max_red[threadIdx.x] =
                fmaxf(max_red[threadIdx.x], max_red[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    const float maximum = max_red[0];
    const bool valid_maximum = isfinite(maximum);
    float local_sum = 0.0f;
    for (uint32_t s = threadIdx.x; s < n_selected; s += kThreads) {
        const uint64_t index = (uint64_t)head * n_selected + s;
        const float weight =
            valid_maximum ? expf(weights[index] - maximum) : 0.0f;
        weights[index] = weight;
        local_sum += weight;
    }
    sum_red[threadIdx.x] = local_sum;
    __syncthreads();
    for (uint32_t stride = 128u; stride > 0u; stride >>= 1u) {
        if (threadIdx.x < stride) {
            sum_red[threadIdx.x] += sum_red[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0u) {
        denoms[head] = fmaxf(sum_red[0], 1.0e-20f);
    }
}

/* Eight heads share each cache value.  Every thread owns one
 * (head,channel) accumulator and visits selected rows in incumbent order. */
__global__ void shared_pv_kernel(
        float *out,
        const float *weights,
        const float *denoms,
        const float *cache,
        const int32_t *selected,
        uint32_t n_selected,
        uint32_t cache_rows) {
    const uint32_t head_lane = threadIdx.x / kColumnTile;
    const uint32_t channel_lane = threadIdx.x % kColumnTile;
    const uint32_t head = blockIdx.x * kHeadGroup + head_lane;
    const uint32_t channel = blockIdx.y * kColumnTile + channel_lane;
    __shared__ float cache_tile[kPvRowTile][kColumnTile];
    __shared__ float weight_tile[kHeadGroup][kPvRowTile];
    __shared__ uint32_t valid_tile[kPvRowTile];
    float acc = 0.0f;

    for (uint32_t base = 0u; base < n_selected; base += kPvRowTile) {
        for (uint32_t linear = threadIdx.x;
             linear < kPvRowTile * kColumnTile;
             linear += kThreads) {
            const uint32_t row_lane = linear / kColumnTile;
            const uint32_t column_lane = linear % kColumnTile;
            const uint32_t s = base + row_lane;
            const int32_t row_i = s < n_selected ? selected[s] : -1;
            const bool valid = row_i >= 0 && (uint32_t)row_i < cache_rows;
            cache_tile[row_lane][column_lane] = cache[
                (uint64_t)(valid ? (uint32_t)row_i : 0u) * kLatent +
                blockIdx.y * kColumnTile + column_lane];
        }
        if (threadIdx.x < kHeadGroup * kPvRowTile) {
            const uint32_t weight_head = threadIdx.x / kPvRowTile;
            const uint32_t row_lane = threadIdx.x % kPvRowTile;
            const uint32_t s = base + row_lane;
            weight_tile[weight_head][row_lane] =
                s < n_selected
                    ? weights[(uint64_t)(blockIdx.x * kHeadGroup + weight_head) *
                              n_selected + s]
                    : 0.0f;
        }
        if (threadIdx.x < kPvRowTile) {
            const uint32_t s = base + threadIdx.x;
            const int32_t row_i = s < n_selected ? selected[s] : -1;
            valid_tile[threadIdx.x] =
                row_i >= 0 && (uint32_t)row_i < cache_rows;
        }
        __syncthreads();
#pragma unroll 1
        for (uint32_t row_lane = 0u; row_lane < kPvRowTile; ++row_lane) {
            if (valid_tile[row_lane]) {
                acc += weight_tile[head_lane][row_lane] *
                    cache_tile[row_lane][channel_lane];
            }
        }
        __syncthreads();
    }
    out[(uint64_t)head * kLatent + channel] = acc / denoms[head];
}

/* Production-order prefill references.  They deliberately launch once per
 * head, matching the current live path after its gather step. */
__global__ void prefill_score_scalar_kernel(
        float *scores,
        const float *low,
        const float *cache,
        uint32_t head,
        uint32_t pos0) {
    const uint32_t token = blockIdx.x;
    const uint32_t visible = min(kPrefillSelected, pos0 + token + 1u);
    const float *head_low =
        low + ((uint64_t)token * kHeads + head) * kLatent;
    float *score_row = scores +
        ((uint64_t)head * kPrefillTokens + token) * kPrefillSelected;
    for (uint32_t s = threadIdx.x; s < kPrefillSelected; s += blockDim.x) {
        float acc = 0.0f;
        if (s < visible) {
            const float *key = cache + (uint64_t)s * kLatent;
#pragma unroll 1
            for (uint32_t j = 0u; j < kLatent; ++j) {
                acc += head_low[j] * key[j];
            }
        }
        score_row[s] = acc;
    }
}

/* One launch covers all heads and tokens.  Eight heads share each cache-row
 * load, while one thread retains each (head,row) ascending-j dot chain. */
__global__ void prefill_score_shared_kernel(
        float *scores,
        const float *low,
        const float *cache,
        uint32_t pos0) {
    const uint32_t token = blockIdx.z;
    const uint32_t head_lane = threadIdx.x / kPrefillRowTile;
    const uint32_t row_lane = threadIdx.x % kPrefillRowTile;
    const uint32_t head =
        blockIdx.x * kPrefillScoreHeadGroup + head_lane;
    const uint32_t s = blockIdx.y * kPrefillRowTile + row_lane;
    const uint32_t visible = min(kPrefillSelected, pos0 + token + 1u);
    const bool in_range = s < kPrefillSelected;
    const bool visible_row = s < visible;
    const float *head_low =
        low + ((uint64_t)token * kHeads + head) * kLatent;
    __shared__ float tile[kPrefillRowTile][257u];
    float dot = 0.0f;

    if (blockIdx.y * kPrefillRowTile >= visible) {
        if (in_range) {
            scores[((uint64_t)head * kPrefillTokens + token) *
                   kPrefillSelected + s] = 0.0f;
        }
        return;
    }
    for (uint32_t half = 0u; half < 2u; ++half) {
        for (uint32_t linear = threadIdx.x;
             linear < kPrefillRowTile * 256u;
             linear += kPrefillScoreThreads) {
            const uint32_t tile_row = linear / 256u;
            const uint32_t column = linear % 256u;
            const uint32_t tile_s =
                blockIdx.y * kPrefillRowTile + tile_row;
            tile[tile_row][column] = tile_s < visible
                ? cache[(uint64_t)tile_s * kLatent +
                        half * 256u + column]
                : 0.0f;
        }
        __syncthreads();
        if (visible_row) {
#pragma unroll 1
            for (uint32_t column = 0u; column < 256u; ++column) {
                dot += head_low[half * 256u + column] *
                    tile[row_lane][column];
            }
        }
        __syncthreads();
    }
    if (in_range) {
        scores[((uint64_t)head * kPrefillTokens + token) *
               kPrefillSelected + s] = visible_row ? dot : 0.0f;
    }
}

__global__ void prefill_softmax_scalar_kernel(
        float *normalizers,
        float *scores,
        uint32_t head,
        uint32_t pos0,
        float scale) {
    const uint32_t token = blockIdx.x;
    float *row = scores +
        ((uint64_t)head * kPrefillTokens + token) * kPrefillSelected;
    __shared__ float reduce[kThreads];
    const uint32_t visible = min(kPrefillSelected, pos0 + token + 1u);
    float vmax = -INFINITY;
    for (uint32_t s = threadIdx.x; s < kPrefillSelected; s += blockDim.x) {
        const float v = s < visible ? row[s] * scale : -INFINITY;
        row[s] = v;
        vmax = fmaxf(vmax, v);
    }
    reduce[threadIdx.x] = vmax;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1u; stride != 0u; stride >>= 1u) {
        if (threadIdx.x < stride) {
            reduce[threadIdx.x] =
                fmaxf(reduce[threadIdx.x], reduce[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    const float row_max = reduce[0];
    float sum = 0.0f;
    for (uint32_t s = threadIdx.x; s < kPrefillSelected; s += blockDim.x) {
        const float v = expf(row[s] - row_max);
        row[s] = v;
        sum += v;
    }
    reduce[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1u; stride != 0u; stride >>= 1u) {
        if (threadIdx.x < stride) {
            reduce[threadIdx.x] += reduce[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0u) {
        normalizers[(uint64_t)head * kPrefillTokens + token] =
            fmaxf(reduce[0], 1.0e-20f);
    }
}

__global__ void prefill_softmax_heads_kernel(
        float *normalizers,
        float *scores,
        uint32_t pos0,
        float scale) {
    const uint32_t head = blockIdx.x / kPrefillTokens;
    const uint32_t token = blockIdx.x - head * kPrefillTokens;
    float *row = scores +
        ((uint64_t)head * kPrefillTokens + token) * kPrefillSelected;
    __shared__ float reduce[kThreads];
    const uint32_t visible = min(kPrefillSelected, pos0 + token + 1u);
    float vmax = -INFINITY;
    for (uint32_t s = threadIdx.x; s < kPrefillSelected; s += blockDim.x) {
        const float v = s < visible ? row[s] * scale : -INFINITY;
        row[s] = v;
        vmax = fmaxf(vmax, v);
    }
    reduce[threadIdx.x] = vmax;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1u; stride != 0u; stride >>= 1u) {
        if (threadIdx.x < stride) {
            reduce[threadIdx.x] =
                fmaxf(reduce[threadIdx.x], reduce[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    const float row_max = reduce[0];
    float sum = 0.0f;
    for (uint32_t s = threadIdx.x; s < kPrefillSelected; s += blockDim.x) {
        const float v = expf(row[s] - row_max);
        row[s] = v;
        sum += v;
    }
    reduce[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1u; stride != 0u; stride >>= 1u) {
        if (threadIdx.x < stride) {
            reduce[threadIdx.x] += reduce[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0u) {
        normalizers[(uint64_t)head * kPrefillTokens + token] =
            fmaxf(reduce[0], 1.0e-20f);
    }
}

__global__ void prefill_pv_scalar_kernel(
        float *out,
        const float *weights,
        const float *normalizers,
        const float *cache,
        uint32_t head,
        uint32_t pos0) {
    const uint32_t token = blockIdx.x;
    const uint32_t visible = min(kPrefillSelected, pos0 + token + 1u);
    const float *weight_row = weights +
        ((uint64_t)head * kPrefillTokens + token) * kPrefillSelected;
    for (uint32_t channel = threadIdx.x;
         channel < kLatent;
         channel += blockDim.x) {
        float acc = 0.0f;
#pragma unroll 1
        for (uint32_t s = 0u; s < visible; ++s) {
            acc += weight_row[s] *
                cache[(uint64_t)s * kLatent + channel];
        }
        out[((uint64_t)token * kHeads + head) * kLatent + channel] =
            acc / normalizers[(uint64_t)head * kPrefillTokens + token];
    }
}

/* Every thread owns one (token,head,channel) accumulator and retains the
 * scalar path's ascending-s order.  A head group shares each cache tile. */
__global__ void prefill_pv_shared_kernel(
        float *out,
        const float *weights,
        const float *normalizers,
        const float *cache,
        uint32_t pos0) {
    const uint32_t token = blockIdx.z;
    const uint32_t head_lane = threadIdx.x / kColumnTile;
    const uint32_t channel_lane = threadIdx.x % kColumnTile;
    const uint32_t head =
        blockIdx.x * kPrefillPvHeadGroup + head_lane;
    const uint32_t channel = blockIdx.y * kColumnTile + channel_lane;
    const uint32_t visible = min(kPrefillSelected, pos0 + token + 1u);
    __shared__ float cache_tile[kPvRowTile][kColumnTile];
    __shared__ float weight_tile[kPrefillPvHeadGroup][kPvRowTile];
    float acc = 0.0f;

    for (uint32_t base = 0u; base < visible; base += kPvRowTile) {
        for (uint32_t linear = threadIdx.x;
             linear < kPvRowTile * kColumnTile;
             linear += kPrefillPvThreads) {
            const uint32_t row_lane = linear / kColumnTile;
            const uint32_t column_lane = linear % kColumnTile;
            const uint32_t s = base + row_lane;
            cache_tile[row_lane][column_lane] = s < visible
                ? cache[(uint64_t)s * kLatent +
                        blockIdx.y * kColumnTile + column_lane]
                : 0.0f;
        }
        if (threadIdx.x < kPrefillPvHeadGroup * kPvRowTile) {
            const uint32_t weight_head = threadIdx.x / kPvRowTile;
            const uint32_t row_lane = threadIdx.x % kPvRowTile;
            const uint32_t s = base + row_lane;
            weight_tile[weight_head][row_lane] = s < visible
                ? weights[((uint64_t)(blockIdx.x * kPrefillPvHeadGroup +
                                      weight_head) *
                           kPrefillTokens + token) * kPrefillSelected + s]
                : 0.0f;
        }
        __syncthreads();
#pragma unroll 1
        for (uint32_t row_lane = 0u; row_lane < kPvRowTile; ++row_lane) {
            if (base + row_lane < visible) {
                acc += weight_tile[head_lane][row_lane] *
                    cache_tile[row_lane][channel_lane];
            }
        }
        __syncthreads();
    }
    out[((uint64_t)token * kHeads + head) * kLatent + channel] =
        acc / normalizers[(uint64_t)head * kPrefillTokens + token];
}

uint64_t fnv1a64(const void *data, size_t bytes) {
    const uint8_t *p = static_cast<const uint8_t *>(data);
    uint64_t hash = UINT64_C(14695981039346656037);
    for (size_t i = 0; i < bytes; ++i) {
        hash ^= p[i];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

enum class Mode { incumbent, qk_only, shared, hybrid };

void launch_mode(
        Mode mode,
        float *out,
        float *scores,
        float *denoms,
        const float *low,
        const float *cache,
        const int32_t *selected) {
    const size_t incumbent_shmem = (512u + kSelected) * sizeof(float);
    if (mode == Mode::incumbent) {
        incumbent_kernel<<<kHeads, kThreads, incumbent_shmem>>>(
            out, low, cache, selected, kSelected, kCacheRows);
    } else if (mode == Mode::qk_only || mode == Mode::hybrid) {
        incumbent_qk_weights_kernel<<<kHeads, kThreads, incumbent_shmem>>>(
            scores, denoms, low, cache, selected, kSelected, kCacheRows);
        if (mode == Mode::hybrid) {
            const dim3 pv_grid(kHeads / kHeadGroup,
                               kLatent / kColumnTile, 1u);
            shared_pv_kernel<<<pv_grid, kThreads>>>(
                out, scores, denoms, cache, selected, kSelected, kCacheRows);
        }
    } else {
        const dim3 qk_grid(kHeads / kHeadGroup,
                           (kSelected + kRowTile - 1u) / kRowTile, 1u);
        shared_qk_kernel<<<qk_grid, kThreads>>>(
            scores, low, cache, selected, kSelected, kCacheRows);
        exact_softmax_kernel<<<kHeads, kThreads, 512u * sizeof(float)>>>(
            scores, denoms, kSelected);
        const dim3 pv_grid(kHeads / kHeadGroup,
                           kLatent / kColumnTile, 1u);
        shared_pv_kernel<<<pv_grid, kThreads>>>(
            out, scores, denoms, cache, selected, kSelected, kCacheRows);
    }
}

float benchmark(
        Mode mode,
        const std::vector<float *> &outputs,
        float *scores,
        float *denoms,
        const std::vector<float *> &lows,
        const std::vector<float *> &caches,
        const int32_t *selected) {
    constexpr uint32_t warmups = 3u;
    constexpr uint32_t repeats = 12u;
    for (uint32_t repeat = 0u; repeat < warmups; ++repeat) {
        for (uint32_t layer = 0u; layer < kLayers; ++layer) {
            launch_mode(mode, outputs[layer], scores, denoms, lows[layer],
                        caches[layer], selected);
        }
    }
    HIP_CHECK(hipDeviceSynchronize());
    hipEvent_t begin = nullptr;
    hipEvent_t end = nullptr;
    HIP_CHECK(hipEventCreate(&begin));
    HIP_CHECK(hipEventCreate(&end));
    HIP_CHECK(hipEventRecord(begin));
    for (uint32_t repeat = 0u; repeat < repeats; ++repeat) {
        for (uint32_t layer = 0u; layer < kLayers; ++layer) {
            launch_mode(mode, outputs[layer], scores, denoms, lows[layer],
                        caches[layer], selected);
        }
    }
    HIP_CHECK(hipEventRecord(end));
    HIP_CHECK(hipEventSynchronize(end));
    float elapsed_ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&elapsed_ms, begin, end));
    HIP_CHECK(hipEventDestroy(end));
    HIP_CHECK(hipEventDestroy(begin));
    return elapsed_ms / (float)(repeats * kLayers);
}

void launch_prefill_score(bool shared, float *scores, const float *low,
                          const float *cache, uint32_t pos0) {
    if (shared) {
        const dim3 grid(kHeads / kPrefillScoreHeadGroup,
                        kPrefillSelected / kPrefillRowTile,
                        kPrefillTokens);
        prefill_score_shared_kernel<<<grid, kPrefillScoreThreads>>>(
            scores, low, cache, pos0);
    } else {
        for (uint32_t head = 0u; head < kHeads; ++head) {
            prefill_score_scalar_kernel<<<kPrefillTokens, kThreads>>>(
                scores, low, cache, head, pos0);
        }
    }
}

void launch_prefill_softmax(bool shared, float *normalizers, float *scores,
                            uint32_t pos0) {
    const float scale = 1.0f / sqrtf(256.0f);
    if (shared) {
        prefill_softmax_heads_kernel<<<kHeads * kPrefillTokens, kThreads>>>(
            normalizers, scores, pos0, scale);
    } else {
        for (uint32_t head = 0u; head < kHeads; ++head) {
            prefill_softmax_scalar_kernel<<<kPrefillTokens, kThreads>>>(
                normalizers, scores, head, pos0, scale);
        }
    }
}

void launch_prefill_pv(bool shared, float *out, const float *weights,
                       const float *normalizers, const float *cache,
                       uint32_t pos0) {
    if (shared) {
        const dim3 grid(kHeads / kPrefillPvHeadGroup,
                        kLatent / kColumnTile,
                        kPrefillTokens);
        prefill_pv_shared_kernel<<<grid, kPrefillPvThreads>>>(
            out, weights, normalizers, cache, pos0);
    } else {
        for (uint32_t head = 0u; head < kHeads; ++head) {
            prefill_pv_scalar_kernel<<<kPrefillTokens, kThreads>>>(
                out, weights, normalizers, cache, head, pos0);
        }
    }
}

float benchmark_prefill_score(bool shared, float *scores, const float *low,
                              const float *cache, uint32_t pos0) {
    constexpr uint32_t warmups = 2u;
    constexpr uint32_t repeats = 5u;
    for (uint32_t i = 0u; i < warmups; ++i) {
        launch_prefill_score(shared, scores, low, cache, pos0);
    }
    HIP_CHECK(hipDeviceSynchronize());
    hipEvent_t begin = nullptr, end = nullptr;
    HIP_CHECK(hipEventCreate(&begin));
    HIP_CHECK(hipEventCreate(&end));
    HIP_CHECK(hipEventRecord(begin));
    for (uint32_t i = 0u; i < repeats; ++i) {
        launch_prefill_score(shared, scores, low, cache, pos0);
    }
    HIP_CHECK(hipEventRecord(end));
    HIP_CHECK(hipEventSynchronize(end));
    float elapsed_ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&elapsed_ms, begin, end));
    HIP_CHECK(hipEventDestroy(end));
    HIP_CHECK(hipEventDestroy(begin));
    return elapsed_ms / repeats;
}

float benchmark_prefill_pv(bool shared, float *out, const float *weights,
                           const float *normalizers, const float *cache,
                           uint32_t pos0) {
    constexpr uint32_t warmups = 2u;
    constexpr uint32_t repeats = 5u;
    for (uint32_t i = 0u; i < warmups; ++i) {
        launch_prefill_pv(shared, out, weights, normalizers, cache, pos0);
    }
    HIP_CHECK(hipDeviceSynchronize());
    hipEvent_t begin = nullptr, end = nullptr;
    HIP_CHECK(hipEventCreate(&begin));
    HIP_CHECK(hipEventCreate(&end));
    HIP_CHECK(hipEventRecord(begin));
    for (uint32_t i = 0u; i < repeats; ++i) {
        launch_prefill_pv(shared, out, weights, normalizers, cache, pos0);
    }
    HIP_CHECK(hipEventRecord(end));
    HIP_CHECK(hipEventSynchronize(end));
    float elapsed_ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&elapsed_ms, begin, end));
    HIP_CHECK(hipEventDestroy(end));
    HIP_CHECK(hipEventDestroy(begin));
    return elapsed_ms / repeats;
}

}  // namespace

int main() {
    int device = 0;
    hipDeviceProp_t properties = {};
    HIP_CHECK(hipGetDeviceProperties(&properties, device));
    std::fprintf(stderr, "device=%s layers=%u rows=%u selected=%u\n",
                 properties.name, kLayers, kCacheRows, kSelected);

    std::vector<int32_t> host_selected(kSelected);
    for (uint32_t i = 0u; i < 2048u; ++i) {
        const uint32_t pool = i >> 2u;
        host_selected[i] =
            (int32_t)((((pool * 157u + 17u) & 511u) << 2u) + (i & 3u));
    }
    host_selected[2048u] = 2048;
    host_selected[2049u] = 2049;
    host_selected[2050u] = 2050;
    host_selected[127u] = -1;
    host_selected[1023u] = -1;
    host_selected[2047u] = -1;
    int32_t *device_selected = nullptr;
    HIP_CHECK(hipMalloc(&device_selected, host_selected.size() * sizeof(int32_t)));
    HIP_CHECK(hipMemcpy(device_selected, host_selected.data(),
                        host_selected.size() * sizeof(int32_t),
                        hipMemcpyHostToDevice));

    std::vector<float *> device_lows(kLayers, nullptr);
    std::vector<float *> device_caches(kLayers, nullptr);
    std::vector<float *> incumbent_outputs(kLayers, nullptr);
    std::vector<float *> qk_outputs(kLayers, nullptr);
    std::vector<float *> shared_outputs(kLayers, nullptr);
    std::vector<float *> hybrid_outputs(kLayers, nullptr);
    std::vector<float> host_low((uint64_t)kHeads * kLatent);
    std::vector<float> host_cache((uint64_t)kCacheRows * kLatent);
    const size_t output_bytes = (uint64_t)kHeads * kLatent * sizeof(float);
    for (uint32_t layer = 0u; layer < kLayers; ++layer) {
        for (size_t i = 0; i < host_low.size(); ++i) {
            host_low[i] =
                (float)((int)((i * 23u + layer * 7u) % 71u) - 35) * 0.001819f;
        }
        for (size_t i = 0; i < host_cache.size(); ++i) {
            host_cache[i] =
                (float)((int)((i * 29u + layer * 13u) % 79u) - 39) * 0.001117f +
                (float)((int)((i + layer) % 11u) - 5) * 0.000007f;
        }
        HIP_CHECK(hipMalloc(&device_lows[layer], host_low.size() * sizeof(float)));
        HIP_CHECK(hipMalloc(&device_caches[layer], host_cache.size() * sizeof(float)));
        HIP_CHECK(hipMalloc(&incumbent_outputs[layer], output_bytes));
        HIP_CHECK(hipMalloc(&qk_outputs[layer], output_bytes));
        HIP_CHECK(hipMalloc(&shared_outputs[layer], output_bytes));
        HIP_CHECK(hipMalloc(&hybrid_outputs[layer], output_bytes));
        HIP_CHECK(hipMemcpy(device_lows[layer], host_low.data(),
                            host_low.size() * sizeof(float), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(device_caches[layer], host_cache.data(),
                            host_cache.size() * sizeof(float), hipMemcpyHostToDevice));
    }
    float *scores = nullptr;
    float *denoms = nullptr;
    HIP_CHECK(hipMalloc(&scores,
                        (uint64_t)kHeads * kSelected * sizeof(float)));
    HIP_CHECK(hipMalloc(&denoms, kHeads * sizeof(float)));

    for (uint32_t layer = 0u; layer < kLayers; ++layer) {
        launch_mode(Mode::incumbent, incumbent_outputs[layer], scores, denoms,
                    device_lows[layer], device_caches[layer], device_selected);
        launch_mode(Mode::shared, shared_outputs[layer], scores, denoms,
                    device_lows[layer], device_caches[layer], device_selected);
        launch_mode(Mode::hybrid, hybrid_outputs[layer], scores, denoms,
                    device_lows[layer], device_caches[layer], device_selected);
    }
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<float> incumbent((uint64_t)kHeads * kLatent);
    std::vector<float> shared((uint64_t)kHeads * kLatent);
    std::vector<float> hybrid((uint64_t)kHeads * kLatent);
    for (uint32_t layer = 0u; layer < kLayers; ++layer) {
        HIP_CHECK(hipMemcpy(incumbent.data(), incumbent_outputs[layer],
                            output_bytes, hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(shared.data(), shared_outputs[layer],
                            output_bytes, hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(hybrid.data(), hybrid_outputs[layer],
                            output_bytes, hipMemcpyDeviceToHost));
        if (std::memcmp(incumbent.data(), shared.data(), output_bytes) != 0) {
            size_t first = 0u;
            while (first < incumbent.size() &&
                   std::memcmp(&incumbent[first], &shared[first], sizeof(float)) == 0) {
                ++first;
            }
            double max_abs = 0.0;
            for (size_t i = 0; i < incumbent.size(); ++i) {
                max_abs = std::max(max_abs,
                    std::fabs((double)incumbent[i] - shared[i]));
            }
            std::fprintf(stderr,
                         "FAIL layer=%u first=%zu incumbent=%.9g shared=%.9g "
                         "max_abs=%.9g\n",
                         layer, first, incumbent[first], shared[first], max_abs);
            return 2;
        }
        if (std::memcmp(incumbent.data(), hybrid.data(), output_bytes) != 0) {
            size_t first = 0u;
            while (first < incumbent.size() &&
                   std::memcmp(&incumbent[first], &hybrid[first], sizeof(float)) == 0) {
                ++first;
            }
            double max_abs = 0.0;
            for (size_t i = 0; i < incumbent.size(); ++i) {
                max_abs = std::max(max_abs,
                    std::fabs((double)incumbent[i] - hybrid[i]));
            }
            std::fprintf(stderr,
                         "FAIL hybrid layer=%u first=%zu incumbent=%.9g "
                         "hybrid=%.9g max_abs=%.9g\n",
                         layer, first, incumbent[first], hybrid[first], max_abs);
            return 3;
        }
    }
    const uint64_t output_fnv = fnv1a64(shared.data(), output_bytes);
    std::fprintf(stderr, "bit_identical=1 last_layer_fnv64=%016llx\n",
                 (unsigned long long)output_fnv);

    const float incumbent_ms = benchmark(
        Mode::incumbent, incumbent_outputs, scores, denoms,
        device_lows, device_caches, device_selected);
    const float qk_ms = benchmark(
        Mode::qk_only, qk_outputs, scores, denoms,
        device_lows, device_caches, device_selected);
    const float shared_ms = benchmark(
        Mode::shared, shared_outputs, scores, denoms,
        device_lows, device_caches, device_selected);
    const float hybrid_ms = benchmark(
        Mode::hybrid, hybrid_outputs, scores, denoms,
        device_lows, device_caches, device_selected);
    std::printf("path,ms_per_layer,speedup\n");
    std::printf("incumbent,%.6f,1.000000\n", incumbent_ms);
    std::printf("qk_softmax_only,%.6f,%.6f\n",
                qk_ms, incumbent_ms / qk_ms);
    std::printf("pv_residual_estimate,%.6f,%.6f\n",
                incumbent_ms - qk_ms,
                incumbent_ms / std::max(incumbent_ms - qk_ms, 1.0e-9f));
    std::printf("shared_split,%.6f,%.6f\n",
                shared_ms, incumbent_ms / shared_ms);
    std::printf("incumbent_qk_shared_pv,%.6f,%.6f\n",
                hybrid_ms, incumbent_ms / hybrid_ms);

    const uint64_t prefill_low_count =
        (uint64_t)kPrefillTokens * kHeads * kLatent;
    const uint64_t prefill_score_count =
        (uint64_t)kHeads * kPrefillTokens * kPrefillSelected;
    const uint64_t prefill_output_count = prefill_low_count;
    const uint64_t prefill_normalizer_count =
        (uint64_t)kHeads * kPrefillTokens;
    std::vector<float> prefill_low(prefill_low_count);
    for (uint64_t i = 0u; i < prefill_low_count; ++i) {
        prefill_low[i] =
            (float)((int)((i * 37u + 19u) % 97u) - 48) * 0.000913f;
    }

    float *d_prefill_low = nullptr;
    float *d_normalizer_reference = nullptr;
    float *d_normalizer_shared = nullptr;
    float *d_score_reference = nullptr;
    float *d_score_shared = nullptr;
    float *d_pv_reference = nullptr;
    float *d_pv_shared = nullptr;
    HIP_CHECK(hipMalloc(&d_prefill_low,
                        prefill_low_count * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_normalizer_reference,
                        prefill_normalizer_count * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_normalizer_shared,
                        prefill_normalizer_count * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_score_reference,
                        prefill_score_count * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_score_shared,
                        prefill_score_count * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_pv_reference,
                        prefill_output_count * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_pv_shared,
                        prefill_output_count * sizeof(float)));
    HIP_CHECK(hipMemcpy(d_prefill_low, prefill_low.data(),
                        prefill_low_count * sizeof(float),
                        hipMemcpyHostToDevice));

    std::vector<float> score_reference(prefill_score_count);
    std::vector<float> score_shared(prefill_score_count);
    std::vector<float> normalizer_reference(prefill_normalizer_count);
    std::vector<float> normalizer_shared(prefill_normalizer_count);
    std::vector<float> pv_reference(prefill_output_count);
    std::vector<float> pv_shared(prefill_output_count);
    std::printf("prefill_pos0,score_scalar_ms,score_shared_ms,score_speedup,"
                "pv_scalar_ms,pv_shared_ms,pv_speedup,combined_speedup\n");
    for (const uint32_t pos0 : {0u, 1792u}) {
        launch_prefill_score(false, d_score_reference, d_prefill_low,
                             device_caches[0], pos0);
        launch_prefill_score(true, d_score_shared, d_prefill_low,
                             device_caches[0], pos0);
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(score_reference.data(), d_score_reference,
                            prefill_score_count * sizeof(float),
                            hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(score_shared.data(), d_score_shared,
                            prefill_score_count * sizeof(float),
                            hipMemcpyDeviceToHost));
        if (std::memcmp(score_reference.data(), score_shared.data(),
                        prefill_score_count * sizeof(float)) != 0) {
            size_t first = 0u;
            while (first < score_reference.size() &&
                   std::memcmp(&score_reference[first], &score_shared[first],
                               sizeof(float)) == 0) ++first;
            std::fprintf(stderr,
                         "FAIL prefill score pos0=%u first=%zu "
                         "scalar=%.9g shared=%.9g\n",
                         pos0, first, score_reference[first],
                         score_shared[first]);
            return 4;
        }
        launch_prefill_softmax(false, d_normalizer_reference,
                               d_score_reference, pos0);
        launch_prefill_softmax(true, d_normalizer_shared,
                               d_score_shared, pos0);
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(score_reference.data(), d_score_reference,
                            prefill_score_count * sizeof(float),
                            hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(score_shared.data(), d_score_shared,
                            prefill_score_count * sizeof(float),
                            hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(normalizer_reference.data(),
                            d_normalizer_reference,
                            prefill_normalizer_count * sizeof(float),
                            hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(normalizer_shared.data(), d_normalizer_shared,
                            prefill_normalizer_count * sizeof(float),
                            hipMemcpyDeviceToHost));
        if (std::memcmp(score_reference.data(), score_shared.data(),
                        prefill_score_count * sizeof(float)) != 0 ||
            std::memcmp(normalizer_reference.data(), normalizer_shared.data(),
                        prefill_normalizer_count * sizeof(float)) != 0) {
            std::fprintf(stderr,
                         "FAIL prefill softmax pos0=%u scale=%.9g\n",
                         pos0, 1.0f / sqrtf(256.0f));
            return 5;
        }
        launch_prefill_pv(false, d_pv_reference, d_score_reference,
                          d_normalizer_reference, device_caches[0], pos0);
        launch_prefill_pv(true, d_pv_shared, d_score_shared,
                          d_normalizer_shared, device_caches[0], pos0);
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(pv_reference.data(), d_pv_reference,
                            prefill_output_count * sizeof(float),
                            hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(pv_shared.data(), d_pv_shared,
                            prefill_output_count * sizeof(float),
                            hipMemcpyDeviceToHost));
        if (std::memcmp(pv_reference.data(), pv_shared.data(),
                        prefill_output_count * sizeof(float)) != 0) {
            size_t first = 0u;
            while (first < pv_reference.size() &&
                   std::memcmp(&pv_reference[first], &pv_shared[first],
                               sizeof(float)) == 0) ++first;
            std::fprintf(stderr,
                         "FAIL prefill PV pos0=%u first=%zu "
                         "scalar=%.9g shared=%.9g\n",
                         pos0, first, pv_reference[first], pv_shared[first]);
            return 6;
        }
        const float score_scalar_ms = benchmark_prefill_score(
            false, d_score_reference, d_prefill_low, device_caches[0], pos0);
        const float score_shared_ms = benchmark_prefill_score(
            true, d_score_shared, d_prefill_low, device_caches[0], pos0);
        launch_prefill_softmax(false, d_normalizer_reference,
                               d_score_reference, pos0);
        launch_prefill_softmax(true, d_normalizer_shared,
                               d_score_shared, pos0);
        const float pv_scalar_ms = benchmark_prefill_pv(
            false, d_pv_reference, d_score_reference,
            d_normalizer_reference, device_caches[0], pos0);
        const float pv_shared_ms = benchmark_prefill_pv(
            true, d_pv_shared, d_score_shared,
            d_normalizer_shared, device_caches[0], pos0);
        std::printf("%u,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                    pos0, score_scalar_ms, score_shared_ms,
                    score_scalar_ms / score_shared_ms,
                    pv_scalar_ms, pv_shared_ms,
                    pv_scalar_ms / pv_shared_ms,
                    (score_scalar_ms + pv_scalar_ms) /
                        (score_shared_ms + pv_shared_ms));
    }
    std::fprintf(stderr, "prefill_bit_identical=1 tokens=%u heads=%u "
                         "selected=%u\n",
                 kPrefillTokens, kHeads, kPrefillSelected);

    HIP_CHECK(hipFree(d_pv_shared));
    HIP_CHECK(hipFree(d_pv_reference));
    HIP_CHECK(hipFree(d_score_shared));
    HIP_CHECK(hipFree(d_score_reference));
    HIP_CHECK(hipFree(d_normalizer_shared));
    HIP_CHECK(hipFree(d_normalizer_reference));
    HIP_CHECK(hipFree(d_prefill_low));

    HIP_CHECK(hipFree(denoms));
    HIP_CHECK(hipFree(scores));
    for (uint32_t layer = 0u; layer < kLayers; ++layer) {
        HIP_CHECK(hipFree(hybrid_outputs[layer]));
        HIP_CHECK(hipFree(shared_outputs[layer]));
        HIP_CHECK(hipFree(qk_outputs[layer]));
        HIP_CHECK(hipFree(incumbent_outputs[layer]));
        HIP_CHECK(hipFree(device_caches[layer]));
        HIP_CHECK(hipFree(device_lows[layer]));
    }
    HIP_CHECK(hipFree(device_selected));
    return 0;
}
