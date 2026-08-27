// Harness-only comparison of DS4's production static flash-attention kernel
// against an arithmetic-identical variant that reads KV rows directly from
// global memory instead of staging each four-row chunk through LDS.

#include "../ds4_gpu.h"

typedef int (*ds4_tp_devcopy_fn)(void *, const void *, uint64_t);
extern "C" void ds4_tp_set_devcopy(ds4_tp_devcopy_fn) {}

#include "../ds4_rocm.cu"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

__global__ static void cache_pollute_kernel(uint32_t *words, size_t n,
                                            uint32_t salt) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += (size_t)gridDim.x * blockDim.x) {
        words[i] = words[i] * 1664525u + salt + (uint32_t)i;
    }
}

/* Two adjacent query rows share each raw/comp KV load while retaining an
 * independent online-softmax stream and the original per-row visit order. */
#ifndef DS4_ATTN_T2_HEAD_WARPS
#define DS4_ATTN_T2_HEAD_WARPS 4
#endif

__launch_bounds__(32 * DS4_ATTN_T2_HEAD_WARPS)
__global__ static void bench_attention_static_mixed_heads4_flash_direct_t2_kernel(
        float *heads,
        const float *sinks,
        const float *q,
        const float *raw_kv,
        const float *comp_kv,
        uint32_t n_q,
        uint32_t q_row0,
        uint32_t n_comp,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_head,
        uint32_t head_dim) {
    const uint32_t t0_local = blockIdx.x * 2u;
    const uint32_t t1_local = t0_local + 1u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t head = blockIdx.y * DS4_ATTN_T2_HEAD_WARPS + warp;
    if (t0_local >= n_q || head_dim != 512u) return;
    const bool valid_head = head < n_head;
    if (!valid_head) return;
    const bool valid0 = valid_head;
    const bool valid1 = valid_head && t1_local < n_q;
    const uint32_t t0_abs = q_row0 + t0_local;
    const uint32_t t1_abs = q_row0 + t1_local;

    const uint32_t raw0_count =
        window != 0u && t0_abs + 1u > window ? window : t0_abs + 1u;
    const uint32_t raw1_count = valid1
        ? (window != 0u && t1_abs + 1u > window ? window : t1_abs + 1u)
        : 0u;
    const uint32_t raw0_start = t0_abs + 1u - raw0_count;
    const uint32_t raw1_start = valid1 ? t1_abs + 1u - raw1_count : raw0_start;
    const uint32_t raw_union_start = raw0_start < raw1_start ? raw0_start : raw1_start;
    const uint32_t raw0_end = raw0_start + raw0_count;
    const uint32_t raw1_end = raw1_start + raw1_count;
    const uint32_t raw_union_end = raw0_end > raw1_end ? raw0_end : raw1_end;
    uint32_t comp0_count = 0u, comp1_count = 0u;
    if (n_comp != 0u && ratio != 0u) {
        comp0_count = (t0_abs + 1u) / ratio;
        if (comp0_count > n_comp) comp0_count = n_comp;
        if (valid1) {
            comp1_count = (t1_abs + 1u) / ratio;
            if (comp1_count > n_comp) comp1_count = n_comp;
        }
    }
    const uint32_t comp_union_count =
        comp0_count > comp1_count ? comp0_count : comp1_count;
    const float scale = rsqrtf((float)head_dim);

    float4 q00 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 q01 = q00, q02 = q00, q03 = q00;
    float4 q10 = q00, q11 = q00, q12 = q00, q13 = q00;
    if (valid0) {
        const float4 *q4 = (const float4 *)(q +
            ((uint64_t)t0_local * n_head + head) * head_dim);
        q00 = q4[lane +  0u]; q01 = q4[lane + 32u];
        q02 = q4[lane + 64u]; q03 = q4[lane + 96u];
    }
    if (valid1) {
        const float4 *q4 = (const float4 *)(q +
            ((uint64_t)t1_local * n_head + head) * head_dim);
        q10 = q4[lane +  0u]; q11 = q4[lane + 32u];
        q12 = q4[lane + 64u]; q13 = q4[lane + 96u];
    }

    float max0 = valid0 ? sinks[head] : -INFINITY;
    float sum0 = valid0 ? 1.0f : 0.0f;
    float max1 = valid1 ? sinks[head] : -INFINITY;
    float sum1 = valid1 ? 1.0f : 0.0f;
    float4 o00 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 o01 = o00, o02 = o00, o03 = o00;
    float4 o10 = o00, o11 = o00, o12 = o00, o13 = o00;

#define DS4_FLASH_T2_UPDATE(score, max_s, sum_s, o0, o1, o2, o3) \
    do { \
        const float new_m = fmaxf((max_s), score); \
        const float old_scale = expf((max_s) - new_m); \
        const float row_scale = expf(score - new_m); \
        (sum_s) = (sum_s) * old_scale + row_scale; \
        (o0).x = (o0).x * old_scale + k0.x * row_scale; \
        (o0).y = (o0).y * old_scale + k0.y * row_scale; \
        (o0).z = (o0).z * old_scale + k0.z * row_scale; \
        (o0).w = (o0).w * old_scale + k0.w * row_scale; \
        (o1).x = (o1).x * old_scale + k1.x * row_scale; \
        (o1).y = (o1).y * old_scale + k1.y * row_scale; \
        (o1).z = (o1).z * old_scale + k1.z * row_scale; \
        (o1).w = (o1).w * old_scale + k1.w * row_scale; \
        (o2).x = (o2).x * old_scale + k2.x * row_scale; \
        (o2).y = (o2).y * old_scale + k2.y * row_scale; \
        (o2).z = (o2).z * old_scale + k2.z * row_scale; \
        (o2).w = (o2).w * old_scale + k2.w * row_scale; \
        (o3).x = (o3).x * old_scale + k3.x * row_scale; \
        (o3).y = (o3).y * old_scale + k3.y * row_scale; \
        (o3).z = (o3).z * old_scale + k3.z * row_scale; \
        (o3).w = (o3).w * old_scale + k3.w * row_scale; \
        (max_s) = new_m; \
    } while (0)

/* Interleave the two independent warp reductions.  Each score retains the
 * exact warp_sum_f32 addition order, but the scheduler can hide one shuffle
 * dependency chain behind the other. */
#define DS4_FLASH_T2_PAIR_STEP(use0, use1) \
    do { \
        float score0 = dot4_f32(q00, k0) + dot4_f32(q01, k1) + \
                       dot4_f32(q02, k2) + dot4_f32(q03, k3); \
        float score1 = dot4_f32(q10, k0) + dot4_f32(q11, k1) + \
                       dot4_f32(q12, k2) + dot4_f32(q13, k3); \
        for (int offset = 16; offset > 0; offset >>= 1) { \
            score0 += __shfl_down(score0, offset, 32); \
            score1 += __shfl_down(score1, offset, 32); \
        } \
        score0 *= scale; \
        score1 *= scale; \
        score0 = __shfl_sync(FULL_WARP_MASK, score0, 0); \
        score1 = __shfl_sync(FULL_WARP_MASK, score1, 0); \
        if (use0) \
            DS4_FLASH_T2_UPDATE(score0, max0, sum0, o00, o01, o02, o03); \
        if (use1) \
            DS4_FLASH_T2_UPDATE(score1, max1, sum1, o10, o11, o12, o13); \
    } while (0)

#define DS4_FLASH_T2_LOAD_STEP(kv_base, row, use0, use1) \
    do { \
        const float4 *kv4 = (const float4 *)((kv_base) + \
            (uint64_t)(row) * head_dim); \
        const float4 k0 = kv4[lane +  0u]; \
        const float4 k1 = kv4[lane + 32u]; \
        const float4 k2 = kv4[lane + 64u]; \
        const float4 k3 = kv4[lane + 96u]; \
        DS4_FLASH_T2_PAIR_STEP(use0, use1); \
    } while (0)

    const uint32_t raw_both_start =
        raw0_start > raw1_start ? raw0_start : raw1_start;
    const uint32_t raw_both_end =
        raw0_end < raw1_end ? raw0_end : raw1_end;
#pragma clang loop unroll(disable)
    for (uint32_t row = raw_union_start; row < raw_both_start; row++) {
        if (raw0_start < raw1_start)
            DS4_FLASH_T2_LOAD_STEP(raw_kv, row, true, false);
        else
            DS4_FLASH_T2_LOAD_STEP(raw_kv, row, false, true);
    }
#pragma clang loop unroll(disable)
    for (uint32_t row = raw_both_start; row < raw_both_end; row++) {
        DS4_FLASH_T2_LOAD_STEP(raw_kv, row, true, true);
    }
#pragma clang loop unroll(disable)
    for (uint32_t row = raw_both_end; row < raw_union_end; row++) {
        if (raw0_end > raw1_end)
            DS4_FLASH_T2_LOAD_STEP(raw_kv, row, true, false);
        else
            DS4_FLASH_T2_LOAD_STEP(raw_kv, row, false, true);
    }

    const uint32_t comp_both_end =
        comp0_count < comp1_count ? comp0_count : comp1_count;
#pragma clang loop unroll(disable)
    for (uint32_t row = 0; row < comp_both_end; row++) {
        DS4_FLASH_T2_LOAD_STEP(comp_kv, row, true, true);
    }
#pragma clang loop unroll(disable)
    for (uint32_t row = comp_both_end; row < comp_union_count; row++) {
        if (comp0_count > comp1_count)
            DS4_FLASH_T2_LOAD_STEP(comp_kv, row, true, false);
        else
            DS4_FLASH_T2_LOAD_STEP(comp_kv, row, false, true);
    }
#undef DS4_FLASH_T2_LOAD_STEP
#undef DS4_FLASH_T2_PAIR_STEP
#undef DS4_FLASH_T2_UPDATE

#define DS4_FLASH_T2_STORE(t, sum_s, o0, o1, o2, o3) \
    do { \
        const float inv_s = (sum_s) == 0.0f ? 0.0f : 1.0f / (sum_s); \
        (o0).x *= inv_s; (o0).y *= inv_s; (o0).z *= inv_s; (o0).w *= inv_s; \
        (o1).x *= inv_s; (o1).y *= inv_s; (o1).z *= inv_s; (o1).w *= inv_s; \
        (o2).x *= inv_s; (o2).y *= inv_s; (o2).z *= inv_s; (o2).w *= inv_s; \
        (o3).x *= inv_s; (o3).y *= inv_s; (o3).z *= inv_s; (o3).w *= inv_s; \
        float4 *out4 = (float4 *)(heads + \
            ((uint64_t)(t) * n_head + head) * head_dim); \
        out4[lane +  0u] = (o0); out4[lane + 32u] = (o1); \
        out4[lane + 64u] = (o2); out4[lane + 96u] = (o3); \
    } while (0)
    if (valid0) DS4_FLASH_T2_STORE(t0_local, sum0, o00, o01, o02, o03);
    if (valid1) DS4_FLASH_T2_STORE(t1_local, sum1, o10, o11, o12, o13);
#undef DS4_FLASH_T2_STORE
}

static void check_hip(hipError_t rc, const char *what) {
    if (rc != hipSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, hipGetErrorString(rc));
        std::exit(1);
    }
}

struct Bench {
    uint32_t n_q, q_row0, n_comp, window, ratio;
    uint32_t n_head, head_dim = 512u;
    std::vector<float> sinks, q, raw, comp;
    float *d_sinks = nullptr, *d_q = nullptr, *d_raw = nullptr,
          *d_comp = nullptr, *d_ref = nullptr, *d_direct = nullptr,
          *d_t2 = nullptr;

    Bench(uint32_t q_count, uint32_t row0, uint32_t comp_count,
          uint32_t window_rows, uint32_t compression_ratio,
          uint32_t heads, unsigned seed)
        : n_q(q_count), q_row0(row0), n_comp(comp_count),
          window(window_rows), ratio(compression_ratio), n_head(heads),
          sinks(heads), q((size_t)q_count * heads * head_dim),
          raw((size_t)(row0 + q_count) * head_dim),
          comp((size_t)std::max(comp_count, 1u) * head_dim) {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> value(-0.125f, 0.125f);
        for (float &v : sinks) v = value(rng);
        for (float &v : q) v = value(rng);
        for (float &v : raw) v = value(rng);
        for (float &v : comp) v = value(rng);
        const size_t out_bytes = q.size() * sizeof(float);
        check_hip(hipMalloc(&d_sinks, sinks.size() * sizeof(float)), "sinks alloc");
        check_hip(hipMalloc(&d_q, q.size() * sizeof(float)), "q alloc");
        check_hip(hipMalloc(&d_raw, raw.size() * sizeof(float)), "raw alloc");
        check_hip(hipMalloc(&d_comp, comp.size() * sizeof(float)), "comp alloc");
        check_hip(hipMalloc(&d_ref, out_bytes), "ref alloc");
        check_hip(hipMalloc(&d_direct, out_bytes), "direct alloc");
        check_hip(hipMalloc(&d_t2, out_bytes), "t2 alloc");
        check_hip(hipMemcpy(d_sinks, sinks.data(), sinks.size()*sizeof(float),
                            hipMemcpyHostToDevice), "sinks copy");
        check_hip(hipMemcpy(d_q, q.data(), q.size()*sizeof(float),
                            hipMemcpyHostToDevice), "q copy");
        check_hip(hipMemcpy(d_raw, raw.data(), raw.size()*sizeof(float),
                            hipMemcpyHostToDevice), "raw copy");
        check_hip(hipMemcpy(d_comp, comp.data(), comp.size()*sizeof(float),
                            hipMemcpyHostToDevice), "comp copy");
    }
    ~Bench() {
        (void)hipFree(d_sinks); (void)hipFree(d_q); (void)hipFree(d_raw);
        (void)hipFree(d_comp); (void)hipFree(d_ref); (void)hipFree(d_direct);
        (void)hipFree(d_t2);
    }
};

struct CachePolluter {
    static constexpr size_t kBytes = 64u * 1024u * 1024u;
    uint32_t *d_words = nullptr;
    CachePolluter() {
        check_hip(hipMalloc(&d_words, kBytes), "polluter alloc");
        check_hip(hipMemset(d_words, 0x5a, kBytes), "polluter initialize");
    }
    ~CachePolluter() { (void)hipFree(d_words); }
    void run(uint32_t salt) {
        cache_pollute_kernel<<<1024, 256>>>(d_words,
                                            kBytes / sizeof(uint32_t), salt);
        check_hip(hipGetLastError(), "polluter launch");
        check_hip(hipDeviceSynchronize(), "polluter sync");
    }
};

static void launch_ref(Bench &b) {
    dim3 grid(b.n_q, (b.n_head + 7u) / 8u, 1u);
    attention_static_mixed_heads8_flash_kernel<<<grid, 256>>>(
        b.d_ref, b.d_sinks, b.d_q, b.d_raw,
        b.n_comp ? b.d_comp : b.d_raw, b.n_q, b.q_row0,
        b.n_comp, b.window, b.ratio, b.n_head, b.head_dim);
    check_hip(hipGetLastError(), "reference launch");
}

static void launch_direct(Bench &b) {
    dim3 grid(b.n_q, (b.n_head + 7u) / 8u, 1u);
    attention_static_mixed_heads8_flash_direct_kernel<<<grid, 256>>>(
        b.d_direct, b.d_sinks, b.d_q, b.d_raw,
        b.n_comp ? b.d_comp : b.d_raw, b.n_q, b.q_row0,
        b.n_comp, b.window, b.ratio, b.n_head, b.head_dim);
    check_hip(hipGetLastError(), "direct launch");
}

static void launch_t2(Bench &b) {
    dim3 grid((b.n_q + 1u) / 2u,
              (b.n_head + DS4_ATTN_T2_HEAD_WARPS - 1u) /
                  DS4_ATTN_T2_HEAD_WARPS,
              1u);
    bench_attention_static_mixed_heads4_flash_direct_t2_kernel<<<
        grid, 32 * DS4_ATTN_T2_HEAD_WARPS>>>(
        b.d_t2, b.d_sinks, b.d_q, b.d_raw,
        b.n_comp ? b.d_comp : b.d_raw, b.n_q, b.q_row0,
        b.n_comp, b.window, b.ratio, b.n_head, b.head_dim);
    check_hip(hipGetLastError(), "t2 launch");
}

static float time_kernel_cold(Bench &b, bool direct, CachePolluter &polluter,
                              uint32_t salt) {
    hipEvent_t begin, end;
    check_hip(hipEventCreate(&begin), "event begin");
    check_hip(hipEventCreate(&end), "event end");
    polluter.run(salt);
    check_hip(hipEventRecord(begin), "event record begin");
    direct ? launch_direct(b) : launch_ref(b);
    check_hip(hipEventRecord(end), "event record end");
    check_hip(hipEventSynchronize(end), "event sync");
    float elapsed = 0.0f;
    check_hip(hipEventElapsedTime(&elapsed, begin, end), "elapsed");
    (void)hipEventDestroy(begin); (void)hipEventDestroy(end);
    return elapsed;
}

static float time_t2_cold(Bench &b, CachePolluter &polluter, uint32_t salt) {
    hipEvent_t begin, end;
    check_hip(hipEventCreate(&begin), "t2 event begin");
    check_hip(hipEventCreate(&end), "t2 event end");
    polluter.run(salt);
    check_hip(hipEventRecord(begin), "t2 event record begin");
    launch_t2(b);
    check_hip(hipEventRecord(end), "t2 event record end");
    check_hip(hipEventSynchronize(end), "t2 event sync");
    float elapsed = 0.0f;
    check_hip(hipEventElapsedTime(&elapsed, begin, end), "t2 elapsed");
    (void)hipEventDestroy(begin); (void)hipEventDestroy(end);
    return elapsed;
}

static float median6(float values[6]) {
    std::sort(values, values + 6);
    return 0.5f * (values[2] + values[3]);
}

static void timing_shape(uint32_t n_q, uint32_t n_comp,
                         CachePolluter &polluter) {
    const uint32_t q_row0 = 2048u - n_q;
    Bench b(n_q, q_row0, n_comp, 128u, 4u, 64u,
            9u + n_q + n_comp);
    launch_ref(b); launch_direct(b); launch_t2(b);
    check_hip(hipDeviceSynchronize(), "timing warmup");
    float refs[6], directs[6];
    for (int pair = 0; pair < 6; ++pair) {
        if ((pair & 1) == 0) {
            refs[pair] = time_kernel_cold(b, false, polluter, 4u * pair + 1u);
            directs[pair] = time_kernel_cold(b, true, polluter, 4u * pair + 2u);
        } else {
            directs[pair] = time_kernel_cold(b, true, polluter, 4u * pair + 1u);
            refs[pair] = time_kernel_cold(b, false, polluter, 4u * pair + 2u);
        }
        std::printf("attn_direct_cold pair=%d n_q=%u n_comp=%u q_row0=%u "
                    "ref_ms=%.4f direct_ms=%.4f change=%+.1f%%\n",
                    pair + 1, n_q, n_comp, q_row0, refs[pair], directs[pair],
                    100.0f * (directs[pair] / refs[pair] - 1.0f));
    }
    const float ref_median = median6(refs);
    const float direct_median = median6(directs);
    std::printf("attn_direct_cold_median n_q=%u n_comp=%u q_row0=%u "
                "ref_ms=%.4f direct_ms=%.4f change=%+.1f%%\n",
                n_q, n_comp, q_row0,
                ref_median, direct_median,
                100.0f * (direct_median / ref_median - 1.0f));
    float direct_t2[6], t2s[6];
    for (int pair = 0; pair < 6; ++pair) {
        if ((pair & 1) == 0) {
            direct_t2[pair] = time_kernel_cold(
                b, true, polluter, 100u + 4u * pair + 1u);
            t2s[pair] = time_t2_cold(
                b, polluter, 100u + 4u * pair + 2u);
        } else {
            t2s[pair] = time_t2_cold(
                b, polluter, 100u + 4u * pair + 1u);
            direct_t2[pair] = time_kernel_cold(
                b, true, polluter, 100u + 4u * pair + 2u);
        }
        std::printf("attn_t2_cold pair=%d n_q=%u n_comp=%u direct_ms=%.4f "
                    "t2_ms=%.4f change=%+.1f%%\n", pair + 1, n_q, n_comp,
                    direct_t2[pair], t2s[pair],
                    100.0f * (t2s[pair] / direct_t2[pair] - 1.0f));
    }
    const float direct_t2_median = median6(direct_t2);
    const float t2_median = median6(t2s);
    std::printf("attn_t2_cold_median n_q=%u n_comp=%u direct_ms=%.4f "
                "t2_ms=%.4f change=%+.1f%%\n", n_q, n_comp,
                direct_t2_median, t2_median,
                100.0f * (t2_median / direct_t2_median - 1.0f));
}

static int exact_shape(uint32_t n_q, uint32_t q_row0, uint32_t n_comp,
                       uint32_t window, uint32_t ratio, uint32_t n_head) {
    Bench b(n_q, q_row0, n_comp, window, ratio, n_head,
            100u + n_q + q_row0 + n_comp + window + n_head);
    launch_ref(b); launch_direct(b); launch_t2(b);
    check_hip(hipDeviceSynchronize(), "exact sync");
    std::vector<float> ref(b.q.size()), direct(b.q.size()), t2(b.q.size());
    check_hip(hipMemcpy(ref.data(), b.d_ref, ref.size()*sizeof(float),
                        hipMemcpyDeviceToHost), "ref copy");
    check_hip(hipMemcpy(direct.data(), b.d_direct,
                        direct.size()*sizeof(float), hipMemcpyDeviceToHost),
              "direct copy");
    check_hip(hipMemcpy(t2.data(), b.d_t2, t2.size()*sizeof(float),
                        hipMemcpyDeviceToHost), "t2 copy");
    size_t bitdiff = 0u;
    size_t t2_bitdiff = 0u;
    for (size_t i = 0; i < ref.size(); ++i)
        bitdiff += std::memcmp(&ref[i], &direct[i], sizeof(float)) != 0;
    for (size_t i = 0; i < ref.size(); ++i)
        t2_bitdiff += std::memcmp(&ref[i], &t2[i], sizeof(float)) != 0;
    std::printf("attn_direct_correctness n_q=%u q_row0=%u n_comp=%u "
                "window=%u ratio=%u n_head=%u bitdiff=%zu/%zu\n",
                n_q, q_row0, n_comp, window, ratio, n_head,
                bitdiff, ref.size());
    std::printf("attn_t2_correctness n_q=%u q_row0=%u bitdiff=%zu/%zu\n",
                n_q, q_row0, t2_bitdiff, ref.size());
    return bitdiff != 0u || t2_bitdiff != 0u;
}

int main(int argc, char **argv) {
    uint32_t timing_q = 512u;
    uint32_t timing_comp = 640u;
    if (argc >= 2) timing_q = (uint32_t)std::strtoul(argv[1], nullptr, 10);
    if (argc >= 3) timing_comp = (uint32_t)std::strtoul(argv[2], nullptr, 10);
    if (argc > 3 || timing_q == 0u || timing_q > 2048u ||
        (2048u % timing_q) != 0u || timing_comp > 640u) {
        std::fprintf(stderr,
                     "usage: %s [timing_q: divisor of 2048] [n_comp: 0..640]\n",
                     argv[0]);
        return 2;
    }
    int failed = 0;
    for (uint32_t n_q : {1u, 5u, 17u, 33u, 127u, 381u})
        failed |= exact_shape(n_q, 0u, 640u, 128u, 4u, 16u);
    /* Production rank geometries and dispatch boundaries omitted by the
     * original smoke set.  The q_row0=2048 case reaches the 768-row cap. */
    failed |= exact_shape(512u, 1536u, 640u, 128u, 4u, 64u);
    failed |= exact_shape(512u, 2048u, 640u, 128u, 4u, 64u);
    failed |= exact_shape(2048u, 0u, 16u, 128u, 128u, 64u);
    failed |= exact_shape(2048u, 0u, 512u, 128u, 4u, 64u);
    failed |= exact_shape(17u, 1536u, 512u, 256u, 4u, 128u);
    failed |= exact_shape(17u, 128u, 0u, 128u, 4u, 13u);
    if (failed) return 1;

    CachePolluter polluter;
    for (uint32_t n_q : {8u, 32u, 64u})
        timing_shape(n_q, timing_comp, polluter);
    if (timing_q != 8u && timing_q != 32u && timing_q != 64u)
        timing_shape(timing_q, timing_comp, polluter);
    return 0;
}
