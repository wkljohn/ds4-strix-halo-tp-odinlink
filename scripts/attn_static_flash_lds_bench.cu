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
          *d_comp = nullptr, *d_ref = nullptr, *d_direct = nullptr;

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

static float median6(float values[6]) {
    std::sort(values, values + 6);
    return 0.5f * (values[2] + values[3]);
}

static void timing_shape(uint32_t n_q, CachePolluter &polluter) {
    const uint32_t q_row0 = 2048u - n_q;
    Bench b(n_q, q_row0, 640u, 128u, 4u, 64u, 9u + n_q);
    launch_ref(b); launch_direct(b);
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
        std::printf("attn_direct_cold pair=%d n_q=%u q_row0=%u "
                    "ref_ms=%.4f direct_ms=%.4f change=%+.1f%%\n",
                    pair + 1, n_q, q_row0, refs[pair], directs[pair],
                    100.0f * (directs[pair] / refs[pair] - 1.0f));
    }
    const float ref_median = median6(refs);
    const float direct_median = median6(directs);
    std::printf("attn_direct_cold_median n_q=%u q_row0=%u ref_ms=%.4f "
                "direct_ms=%.4f change=%+.1f%%\n", n_q, q_row0,
                ref_median, direct_median,
                100.0f * (direct_median / ref_median - 1.0f));
}

static int exact_shape(uint32_t n_q, uint32_t q_row0, uint32_t n_comp,
                       uint32_t window, uint32_t ratio, uint32_t n_head) {
    Bench b(n_q, q_row0, n_comp, window, ratio, n_head,
            100u + n_q + q_row0 + n_comp + window + n_head);
    launch_ref(b); launch_direct(b);
    check_hip(hipDeviceSynchronize(), "exact sync");
    std::vector<float> ref(b.q.size()), direct(b.q.size());
    check_hip(hipMemcpy(ref.data(), b.d_ref, ref.size()*sizeof(float),
                        hipMemcpyDeviceToHost), "ref copy");
    check_hip(hipMemcpy(direct.data(), b.d_direct,
                        direct.size()*sizeof(float), hipMemcpyDeviceToHost),
              "direct copy");
    size_t bitdiff = 0u;
    for (size_t i = 0; i < ref.size(); ++i)
        bitdiff += std::memcmp(&ref[i], &direct[i], sizeof(float)) != 0;
    std::printf("attn_direct_correctness n_q=%u q_row0=%u n_comp=%u "
                "window=%u ratio=%u n_head=%u bitdiff=%zu/%zu\n",
                n_q, q_row0, n_comp, window, ratio, n_head,
                bitdiff, ref.size());
    return bitdiff != 0u;
}

int main(int argc, char **argv) {
    uint32_t timing_q = 512u;
    if (argc == 2) timing_q = (uint32_t)std::strtoul(argv[1], nullptr, 10);
    if (timing_q == 0u || timing_q > 2048u || (2048u % timing_q) != 0u) {
        std::fprintf(stderr, "usage: %s [timing_q: divisor of 2048]\n", argv[0]);
        return 2;
    }
    int failed = 0;
    for (uint32_t n_q : {1u, 5u, 17u, 33u, 127u, 381u})
        failed |= exact_shape(n_q, 0u, 640u, 128u, 4u, 16u);
    /* Production rank geometries and dispatch boundaries omitted by the
     * original smoke set.  The q_row0=2048 case reaches the 768-row cap. */
    failed |= exact_shape(512u, 1536u, 640u, 128u, 4u, 64u);
    failed |= exact_shape(512u, 2048u, 640u, 128u, 4u, 64u);
    failed |= exact_shape(17u, 1536u, 512u, 256u, 4u, 128u);
    failed |= exact_shape(17u, 128u, 0u, 128u, 4u, 13u);
    if (failed) return 1;

    CachePolluter polluter;
    for (uint32_t n_q : {8u, 32u, 64u}) timing_shape(n_q, polluter);
    if (timing_q != 8u && timing_q != 32u && timing_q != 64u)
        timing_shape(timing_q, polluter);
    return 0;
}
