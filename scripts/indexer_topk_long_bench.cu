// Exact gfx1151 oracle for the long-context indexer top-k boundary.
//
// DS4's current n_comp > 8192 path bitonic-sorts 4096-row chunks and then
// bitonic-sorts their candidates.  Compare that exact arithmetic with two
// radix-sort layouts before changing production:
//   1. 4096-item radix chunks plus a 2048-item radix merge.
//
// A one-block 12288-item CUB sort was also compiled first and rejected: its
// 98304-byte TempStorage exceeds gfx1151's 65536-byte per-block limit.
//
// Build:
//   /opt/rocm/bin/hipcc -O3 --offload-arch=gfx1151 -I. \
//     scripts/indexer_topk_long_bench.cu -o /tmp/indexer_topk_long_bench

#include "ds4_rocm.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

static constexpr uint32_t kThreads = 512u;
static constexpr uint32_t kTopK = 512u;
static constexpr uint32_t kChunk = 4096u;

static void check(hipError_t rc, const char *where) {
    if (rc != hipSuccess) {
        std::fprintf(stderr, "%s: %s\n", where, hipGetErrorString(rc));
        std::exit(1);
    }
}

__device__ __forceinline__ static bool better(
        float av, uint32_t ai, float bv, uint32_t bi) {
    return av > bv || (av == bv && ai < bi);
}

__device__ __forceinline__ static uint32_t ordered_key(float v) {
    const uint32_t u = __float_as_uint(v);
    return (u & 0x80000000u) ? ~u : (u ^ 0x80000000u);
}

__device__ __forceinline__ static uint64_t packed_key(float v, uint32_t idx) {
    return ((uint64_t)ordered_key(v) << 32u) |
           (uint64_t)(UINT32_MAX - idx);
}

template <uint32_t SORT_N>
__global__ static void bitonic_chunk(
        uint32_t *candidates, const float *scores, uint32_t n_comp,
        uint32_t n_tokens, uint32_t stride) {
    const uint32_t t = blockIdx.x;
    const uint32_t chunk = blockIdx.y;
    const uint32_t tid = threadIdx.x;
    const uint32_t start = chunk * SORT_N;
    if (t >= n_tokens || start >= n_comp) return;
    __shared__ float vals[SORT_N];
    __shared__ uint32_t idxs[SORT_N];
    const float *row = scores + (uint64_t)t * n_comp;
    for (uint32_t i = tid; i < SORT_N; i += blockDim.x) {
        const uint32_t idx = start + i;
        vals[i] = idx < n_comp ? row[idx] : -INFINITY;
        idxs[i] = idx < n_comp ? idx : UINT32_MAX;
    }
    __syncthreads();
    for (uint32_t k = 2u; k <= SORT_N; k <<= 1u) {
        for (uint32_t j = k >> 1u; j; j >>= 1u) {
            for (uint32_t i = tid; i < SORT_N; i += blockDim.x) {
                const uint32_t other = i ^ j;
                if (other > i) {
                    const float av = vals[i], bv = vals[other];
                    const uint32_t ai = idxs[i], bi = idxs[other];
                    const bool desc = (i & k) == 0u;
                    const bool swap = desc ? better(bv, bi, av, ai)
                                           : better(av, ai, bv, bi);
                    if (swap) {
                        vals[i] = bv; idxs[i] = bi;
                        vals[other] = av; idxs[other] = ai;
                    }
                }
            }
            __syncthreads();
        }
    }
    uint32_t *out = candidates + (uint64_t)t * stride + chunk * kTopK;
    for (uint32_t i = tid; i < kTopK; i += blockDim.x) out[i] = idxs[i];
}

template <uint32_t SORT_N>
__global__ static void bitonic_merge(
        uint32_t *selected, const uint32_t *candidates, const float *scores,
        uint32_t n_comp, uint32_t n_tokens, uint32_t count, uint32_t stride) {
    const uint32_t t = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    if (t >= n_tokens) return;
    __shared__ float vals[SORT_N];
    __shared__ uint32_t idxs[SORT_N];
    const float *row = scores + (uint64_t)t * n_comp;
    const uint32_t *cand = candidates + (uint64_t)t * stride;
    for (uint32_t i = tid; i < SORT_N; i += blockDim.x) {
        const uint32_t idx = i < count ? cand[i] : UINT32_MAX;
        vals[i] = idx < n_comp ? row[idx] : -INFINITY;
        idxs[i] = idx;
    }
    __syncthreads();
    for (uint32_t k = 2u; k <= SORT_N; k <<= 1u) {
        for (uint32_t j = k >> 1u; j; j >>= 1u) {
            for (uint32_t i = tid; i < SORT_N; i += blockDim.x) {
                const uint32_t other = i ^ j;
                if (other > i) {
                    const float av = vals[i], bv = vals[other];
                    const uint32_t ai = idxs[i], bi = idxs[other];
                    const bool desc = (i & k) == 0u;
                    const bool swap = desc ? better(bv, bi, av, ai)
                                           : better(av, ai, bv, bi);
                    if (swap) {
                        vals[i] = bv; idxs[i] = bi;
                        vals[other] = av; idxs[other] = ai;
                    }
                }
            }
            __syncthreads();
        }
    }
    uint32_t *out = selected + (uint64_t)t * kTopK;
    for (uint32_t i = tid; i < kTopK; i += blockDim.x) out[i] = idxs[i];
}

template <uint32_t ITEMS>
__global__ static void cub_chunks(
        uint32_t *candidates, const float *scores, uint32_t n_comp,
        uint32_t n_tokens, uint32_t stride) {
    using Sort = cub::BlockRadixSort<uint64_t, kThreads, ITEMS>;
    __shared__ typename Sort::TempStorage storage;
    const uint32_t t = blockIdx.x;
    const uint32_t chunk = blockIdx.y;
    const uint32_t tid = threadIdx.x;
    const uint32_t start = chunk * (kThreads * ITEMS);
    if (t >= n_tokens || start >= n_comp) return;
    const float *row = scores + (uint64_t)t * n_comp;
    uint64_t keys[ITEMS];
#pragma unroll
    for (uint32_t item = 0; item < ITEMS; item++) {
        const uint32_t idx = start + tid * ITEMS + item;
        keys[item] = idx < n_comp ? packed_key(row[idx], idx)
                                  : packed_key(-INFINITY, UINT32_MAX);
    }
    Sort(storage).SortDescending(keys);
    uint32_t *out = candidates + (uint64_t)t * stride + chunk * kTopK;
#pragma unroll
    for (uint32_t item = 0; item < ITEMS; item++) {
        const uint32_t rank = tid * ITEMS + item;
        if (rank < kTopK) out[rank] = UINT32_MAX - (uint32_t)keys[item];
    }
}

template <uint32_t ITEMS>
__global__ static void cub_merge(
        uint32_t *selected, const uint32_t *candidates, const float *scores,
        uint32_t n_comp, uint32_t n_tokens, uint32_t count, uint32_t stride) {
    using Sort = cub::BlockRadixSort<uint64_t, kThreads, ITEMS>;
    __shared__ typename Sort::TempStorage storage;
    const uint32_t t = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    if (t >= n_tokens) return;
    const float *row = scores + (uint64_t)t * n_comp;
    const uint32_t *cand = candidates + (uint64_t)t * stride;
    uint64_t keys[ITEMS];
#pragma unroll
    for (uint32_t item = 0; item < ITEMS; item++) {
        const uint32_t slot = tid * ITEMS + item;
        const uint32_t idx = slot < count ? cand[slot] : UINT32_MAX;
        keys[item] = idx < n_comp ? packed_key(row[idx], idx)
                                  : packed_key(-INFINITY, UINT32_MAX);
    }
    Sort(storage).SortDescending(keys);
#pragma unroll
    for (uint32_t item = 0; item < ITEMS; item++) {
        const uint32_t rank = tid * ITEMS + item;
        if (rank < kTopK) {
            selected[(uint64_t)t * kTopK + rank] =
                UINT32_MAX - (uint32_t)keys[item];
        }
    }
}

template <class F>
static float time_us(F launch, uint32_t reps) {
    hipEvent_t begin = nullptr, end = nullptr;
    check(hipEventCreate(&begin), "event begin");
    check(hipEventCreate(&end), "event end");
    for (uint32_t i = 0; i < 20; i++) launch();
    check(hipDeviceSynchronize(), "warmup");
    check(hipEventRecord(begin), "record begin");
    for (uint32_t i = 0; i < reps; i++) launch();
    check(hipEventRecord(end), "record end");
    check(hipEventSynchronize(end), "wait end");
    float ms = 0.0f;
    check(hipEventElapsedTime(&ms, begin, end), "elapsed");
    (void)hipEventDestroy(begin);
    (void)hipEventDestroy(end);
    return ms * 1000.0f / (float)reps;
}

static bool run(uint32_t n_comp, uint32_t n_tokens) {
    const uint32_t n_chunks = (n_comp + kChunk - 1u) / kChunk;
    const uint32_t stride = n_chunks * kTopK;
    const uint64_t score_n = (uint64_t)n_comp * n_tokens;
    const uint64_t selected_n = (uint64_t)kTopK * n_tokens;
    const uint64_t candidate_n = (uint64_t)stride * n_tokens;
    std::mt19937 rng(1151u + n_comp + 17u * n_tokens);
    std::uniform_real_distribution<float> dist(-8.0f, 8.0f);
    std::vector<float> scores(score_n);
    for (uint64_t i = 0; i < score_n; i++) {
        // Deliberate ties exercise the stable lower-index rule.
        scores[i] = i % 29u == 0u ? 1.25f : dist(rng);
    }
    float *d_scores = nullptr;
    uint32_t *d_candidates = nullptr, *d_ref = nullptr, *d_chunked = nullptr;
    check(hipMalloc(&d_scores, score_n * sizeof(float)), "scores alloc");
    check(hipMalloc(&d_candidates, candidate_n * sizeof(uint32_t)), "candidate alloc");
    check(hipMalloc(&d_ref, selected_n * sizeof(uint32_t)), "ref alloc");
    check(hipMalloc(&d_chunked, selected_n * sizeof(uint32_t)), "chunked alloc");
    check(hipMemcpy(d_scores, scores.data(), score_n * sizeof(float),
                    hipMemcpyHostToDevice), "scores upload");
    const dim3 chunk_grid(n_tokens, n_chunks, 1u);
    auto current = [&] {
        bitonic_chunk<4096><<<chunk_grid, 1024>>>(
            d_candidates, d_scores, n_comp, n_tokens, stride);
        bitonic_merge<4096><<<n_tokens, 1024>>>(
            d_ref, d_candidates, d_scores, n_comp, n_tokens, stride, stride);
    };
    auto chunked = [&] {
        cub_chunks<8><<<chunk_grid, kThreads>>>(
            d_candidates, d_scores, n_comp, n_tokens, stride);
        cub_merge<4><<<n_tokens, kThreads>>>(
            d_chunked, d_candidates, d_scores, n_comp, n_tokens,
            stride, stride);
    };
    current(); chunked();
    check(hipDeviceSynchronize(), "correctness run");
    std::vector<uint32_t> ref(selected_n), got_chunked(selected_n);
    check(hipMemcpy(ref.data(), d_ref, selected_n * sizeof(uint32_t),
                    hipMemcpyDeviceToHost), "ref read");
    check(hipMemcpy(got_chunked.data(), d_chunked, selected_n * sizeof(uint32_t),
                    hipMemcpyDeviceToHost), "chunked read");
    uint64_t chunked_diff = 0;
    for (uint64_t i = 0; i < selected_n; i++) {
        chunked_diff += ref[i] != got_chunked[i];
    }
    const uint32_t reps = n_tokens <= 16u ? 1000u : 100u;
    const float current_us = time_us(current, reps);
    const float chunked_us = time_us(chunked, reps);
    std::printf("n_comp=%u tokens=%u current_us=%.3f radix_tree_us=%.3f "
                "radix_tree_speedup=%.3fx chunked_diff=%llu\n",
                n_comp, n_tokens, current_us, chunked_us,
                current_us / chunked_us,
                (unsigned long long)chunked_diff);
    (void)hipFree(d_chunked);
    (void)hipFree(d_ref);
    (void)hipFree(d_candidates);
    (void)hipFree(d_scores);
    return chunked_diff == 0u;
}

int main() {
    bool ok = true;
    for (uint32_t n_comp : {8193u, 8448u, 12288u}) {
        for (uint32_t n_tokens : {1u, 16u, 256u}) {
            ok = run(n_comp, n_tokens) && ok;
        }
    }
    return ok ? 0 : 1;
}
