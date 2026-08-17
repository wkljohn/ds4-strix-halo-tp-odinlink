// State-machine oracle for deferring compressor stores until an emit or
// session-save flush. It compares the production sequential transition with
// batched stores across misaligned ratio-4 and ratio-128 windows.
//
// Build/run:
//   hipcc -O3 -ffast-math -fno-finite-math-only --offload-arch=gfx1151 \
//     scripts/compressor_temporal_state_bench.cu \
//     -o /tmp/compressor_temporal_state_bench
//   /tmp/compressor_temporal_state_bench

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

static void check(hipError_t rc, const char *what) {
    if (rc != hipSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, hipGetErrorString(rc));
        std::exit(1);
    }
}

__global__ static void store_rows(
        const float *kv, const float *sc, float *state_kv,
        float *state_sc, const __half *ape, uint32_t width, uint32_t ratio,
        uint32_t pos0, uint32_t n_tokens) {
    const uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t n = (uint64_t)n_tokens * width;
    if (gid >= n) return;
    const uint32_t t = (uint32_t)(gid / width);
    const uint32_t j = (uint32_t)(gid - (uint64_t)t * width);
    const uint32_t phase = (pos0 + t) % ratio;
    const uint32_t dst = ratio == 4u ? ratio + phase : phase;
    state_kv[(uint64_t)dst * width + j] = kv[(uint64_t)t * width + j];
    state_sc[(uint64_t)dst * width + j] =
        sc[(uint64_t)t * width + j] +
        __half2float(ape[(uint64_t)phase * width + j]);
}

__global__ static void pool_row(
        float *row, const float *state_kv, const float *state_sc,
        uint32_t head_dim, uint32_t ratio) {
    const uint32_t d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= head_dim) return;
    const uint32_t width = (ratio == 4u ? 2u : 1u) * head_dim;
    float vals[128];
    float scores[128];
    float max_s = -INFINITY;
    uint32_t n = 0;
    if (ratio == 4u) {
        for (uint32_t r = 0; r < 4u; r++) {
            vals[n] = state_kv[(uint64_t)r * width + d];
            scores[n] = state_sc[(uint64_t)r * width + d];
            max_s = fmaxf(max_s, scores[n++]);
        }
        for (uint32_t r = 0; r < 4u; r++) {
            vals[n] = state_kv[(uint64_t)(ratio + r) * width + head_dim + d];
            scores[n] = state_sc[(uint64_t)(ratio + r) * width + head_dim + d];
            max_s = fmaxf(max_s, scores[n++]);
        }
    } else {
        for (uint32_t r = 0; r < ratio; r++) {
            vals[n] = state_kv[(uint64_t)r * width + d];
            scores[n] = state_sc[(uint64_t)r * width + d];
            max_s = fmaxf(max_s, scores[n++]);
        }
    }
    float den = 0.0f, acc = 0.0f;
    for (uint32_t i = 0; i < n; i++) {
        const float w = expf(scores[i] - max_s);
        den += w;
        acc += vals[i] * w;
    }
    row[d] = den != 0.0f ? acc / den : 0.0f;
}

__global__ static void shift_ratio4(float *state_kv, float *state_sc,
                                    uint32_t width) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t half = 4ull * width;
    if (i >= half) return;
    const float v = state_kv[half + i];
    const float s = state_sc[half + i];
    state_kv[i] = v;
    state_sc[i] = s;
    state_kv[half + i] = v;
    state_sc[half + i] = s;
}

struct result {
    std::vector<float> state_kv;
    std::vector<float> state_sc;
    std::vector<float> emitted;
};

static result run_path(
        bool deferred, uint32_t ratio, uint32_t width, uint32_t start_pos,
        uint32_t tokens, const std::vector<uint32_t> &save_flush_after,
        const std::vector<float> &kv, const std::vector<float> &sc,
        const std::vector<__half> &ape, const std::vector<float> &initial_kv,
        const std::vector<float> &initial_sc) {
    const uint32_t head_dim = ratio == 4u ? width / 2u : width;
    const uint32_t state_rows = ratio == 4u ? 8u : ratio;
    const uint64_t state_n = (uint64_t)state_rows * width;
    const uint32_t max_emits = (tokens + ratio - 1u) / ratio + 2u;
    float *dkv{}, *dsc{}, *dstate_kv{}, *dstate_sc{}, *demitted{};
    __half *dape{};
    check(hipMalloc(&dkv, kv.size() * sizeof(float)), "malloc kv");
    check(hipMalloc(&dsc, sc.size() * sizeof(float)), "malloc sc");
    check(hipMalloc(&dape, ape.size() * sizeof(__half)), "malloc ape");
    check(hipMalloc(&dstate_kv, state_n * sizeof(float)), "malloc state kv");
    check(hipMalloc(&dstate_sc, state_n * sizeof(float)), "malloc state sc");
    check(hipMalloc(&demitted, (uint64_t)max_emits * head_dim * sizeof(float)),
          "malloc emitted");
    check(hipMemcpy(dkv, kv.data(), kv.size() * sizeof(float), hipMemcpyHostToDevice), "copy kv");
    check(hipMemcpy(dsc, sc.data(), sc.size() * sizeof(float), hipMemcpyHostToDevice), "copy sc");
    check(hipMemcpy(dape, ape.data(), ape.size() * sizeof(__half), hipMemcpyHostToDevice), "copy ape");
    check(hipMemcpy(dstate_kv, initial_kv.data(), state_n * sizeof(float), hipMemcpyHostToDevice), "copy state kv");
    check(hipMemcpy(dstate_sc, initial_sc.data(), state_n * sizeof(float), hipMemcpyHostToDevice), "copy state sc");

    uint32_t pending_start = 0u, pending_count = 0u, emits = 0u;
    for (uint32_t t = 0; t < tokens; t++) {
        const uint32_t pos = start_pos + t;
        if (!deferred) {
            store_rows<<<(width + 255u) / 256u, 256u>>>(
                dkv + (uint64_t)t * width, dsc + (uint64_t)t * width,
                dstate_kv, dstate_sc, dape, width, ratio, pos, 1u);
        } else {
            if (pending_count == 0u) pending_start = t;
            pending_count++;
        }
        const bool emit = ((pos + 1u) % ratio) == 0u;
        const bool save_flush =
            std::find(save_flush_after.begin(), save_flush_after.end(), t + 1u) !=
            save_flush_after.end();
        if (deferred && (emit || save_flush)) {
            const uint64_t n = (uint64_t)pending_count * width;
            store_rows<<<(n + 255u) / 256u, 256u>>>(
                dkv + (uint64_t)pending_start * width,
                dsc + (uint64_t)pending_start * width,
                dstate_kv, dstate_sc, dape, width, ratio,
                start_pos + pending_start, pending_count);
            pending_count = 0u;
        }
        if (emit) {
            pool_row<<<(head_dim + 255u) / 256u, 256u>>>(
                demitted + (uint64_t)emits * head_dim,
                dstate_kv, dstate_sc, head_dim, ratio);
            emits++;
            if (ratio == 4u) {
                const uint64_t n = 4ull * width;
                shift_ratio4<<<(n + 255u) / 256u, 256u>>>(
                    dstate_kv, dstate_sc, width);
            }
        }
    }
    if (deferred && pending_count != 0u) {
        const uint64_t n = (uint64_t)pending_count * width;
        store_rows<<<(n + 255u) / 256u, 256u>>>(
            dkv + (uint64_t)pending_start * width,
            dsc + (uint64_t)pending_start * width,
            dstate_kv, dstate_sc, dape, width, ratio,
            start_pos + pending_start, pending_count);
    }
    check(hipDeviceSynchronize(), "path sync");
    result out;
    out.state_kv.resize(state_n);
    out.state_sc.resize(state_n);
    out.emitted.resize((uint64_t)emits * head_dim);
    check(hipMemcpy(out.state_kv.data(), dstate_kv, state_n * sizeof(float), hipMemcpyDeviceToHost), "read state kv");
    check(hipMemcpy(out.state_sc.data(), dstate_sc, state_n * sizeof(float), hipMemcpyDeviceToHost), "read state sc");
    check(hipMemcpy(out.emitted.data(), demitted, out.emitted.size() * sizeof(float), hipMemcpyDeviceToHost), "read emitted");
    (void)hipFree(dkv); (void)hipFree(dsc); (void)hipFree(dape);
    (void)hipFree(dstate_kv); (void)hipFree(dstate_sc); (void)hipFree(demitted);
    return out;
}

static bool case_run(uint32_t ratio, uint32_t width, uint32_t start_pos,
                     uint32_t tokens, std::vector<uint32_t> save_flush_after) {
    const uint32_t state_rows = ratio == 4u ? 8u : ratio;
    std::mt19937 gen(101u + ratio + width + start_pos);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    std::vector<float> kv((uint64_t)tokens * width);
    std::vector<float> sc((uint64_t)tokens * width);
    std::vector<__half> ape((uint64_t)ratio * width);
    std::vector<float> initial_kv((uint64_t)state_rows * width);
    std::vector<float> initial_sc((uint64_t)state_rows * width);
    for (float &v : kv) v = dist(gen);
    for (float &v : sc) v = dist(gen);
    for (__half &v : ape) v = __float2half(dist(gen));
    for (float &v : initial_kv) v = dist(gen);
    for (float &v : initial_sc) v = dist(gen);
    const result sequential = run_path(false, ratio, width, start_pos, tokens,
                                       save_flush_after, kv, sc, ape,
                                       initial_kv, initial_sc);
    const result deferred = run_path(true, ratio, width, start_pos, tokens,
                                     save_flush_after, kv, sc, ape,
                                     initial_kv, initial_sc);
    const bool kv_equal = sequential.state_kv == deferred.state_kv;
    const bool sc_equal = sequential.state_sc == deferred.state_sc;
    const bool emitted_equal = sequential.emitted == deferred.emitted;
    std::printf("ratio=%u width=%u start_pos=%u tokens=%u save_flushes=%zu "
                "state_kv=%d state_sc=%d emitted=%d emitted_rows=%zu\n",
                ratio, width, start_pos, tokens, save_flush_after.size(),
                kv_equal, sc_equal, emitted_equal,
                sequential.emitted.size() / (ratio == 4u ? width / 2u : width));
    return kv_equal && sc_equal && emitted_equal;
}

int main() {
    bool ok = true;
    ok &= case_run(4u, 1024u, 5u, 19u, {7u, 17u});
    ok &= case_run(4u, 256u, 6u, 21u, {5u, 14u});
    ok &= case_run(128u, 512u, 37u, 300u, {73u, 211u});
    return ok ? 0 : 2;
}
