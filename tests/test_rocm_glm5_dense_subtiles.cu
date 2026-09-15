#include "ds4_gpu.h"
#include "ds4_gpu_mgpu.h"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

#define CHECK(x) do { if (!(x)) { \
    std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); return false; \
} } while (0)

using Tensor = std::unique_ptr<ds4_gpu_tensor, decltype(&ds4_gpu_tensor_free)>;
static Tensor alloc(uint64_t bytes) {
    return Tensor(ds4_gpu_tensor_alloc(bytes), ds4_gpu_tensor_free);
}
static Tensor view(const ds4_gpu_tensor *base, uint64_t offset, uint64_t bytes) {
    return Tensor(ds4_gpu_tensor_view(base, offset, bytes), ds4_gpu_tensor_free);
}
static constexpr uint32_t heads = 64, width = 512, cap = 2048;
static constexpr uint64_t row = uint64_t(heads) * width;

static int attention(ds4_gpu_tensor *out, const ds4_gpu_tensor *q,
                     const ds4_gpu_tensor *low, const ds4_gpu_tensor *cache,
                     uint32_t count, uint32_t pos, uint32_t selected) {
    return ds4_gpu_glm_attention_indexed_batch_lora_causal_tensor(
        out, q, low, cache, nullptr, count, pos, selected, cap, false,
        heads, width, 256, 0, cap, 10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
}

static bool run_case(uint32_t count, uint32_t pos) {
    const uint64_t elements = uint64_t(count) * row;
    std::vector<float> low(elements), cache(uint64_t(cap) * width);
    uint32_t rng = 731u + pos + count;
    auto next = [&rng]() {
        rng = rng * 1664525u + 1013904223u;
        return float(int32_t(rng >> 8) - 8388608) * 0.00000011920928955078125f;
    };
    for (float &x : low) x = next() * 1.371f;
    for (float &x : cache) x = next() * 0.4137f;
    auto dlow = alloc(elements * sizeof(float));
    auto dq = alloc(uint64_t(count) * heads * 256 * sizeof(float));
    auto dcache = alloc(cache.size() * sizeof(float));
    auto dout = alloc(elements * sizeof(float));
    auto dref = alloc(elements * sizeof(float));
    CHECK(dlow && dq && dcache && dout && dref);
    CHECK(ds4_gpu_tensor_write(dlow.get(), 0, low.data(), elements * sizeof(float)));
    CHECK(ds4_gpu_tensor_fill_f32(dq.get(), 0.0f, uint64_t(count) * heads * 256));
    CHECK(ds4_gpu_tensor_write(dcache.get(), 0, cache.data(), cache.size() * sizeof(float)));
    CHECK(ds4_gpu_tensor_fill_f32(dout.get(), 123.25f, elements));
    CHECK(attention(dout.get(), dq.get(), dlow.get(), dcache.get(), count, pos, pos + count));
    for (uint32_t base = 0; base < count; base += 256) {
        auto qv = view(dq.get(), uint64_t(base) * heads * 256 * sizeof(float),
                       uint64_t(256) * heads * 256 * sizeof(float));
        auto lv = view(dlow.get(), uint64_t(base) * row * sizeof(float), 256 * row * sizeof(float));
        auto ov = view(dref.get(), uint64_t(base) * row * sizeof(float), 256 * row * sizeof(float));
        CHECK(qv && lv && ov);
        CHECK(attention(ov.get(), qv.get(), lv.get(), dcache.get(), 256,
                        pos + base, pos + base + 256));
    }
    CHECK(ds4_gpu_synchronize());
    std::vector<float> actual(elements), reference(elements);
    CHECK(ds4_gpu_tensor_read(dout.get(), 0, actual.data(), elements * sizeof(float)));
    CHECK(ds4_gpu_tensor_read(dref.get(), 0, reference.data(), elements * sizeof(float)));
    for (float x : actual) CHECK(std::isfinite(x));
    CHECK(std::memcmp(actual.data(), reference.data(), elements * sizeof(float)) == 0);

    // Independent double-precision oracle at both sides of the subtile seam.
    double max_error = 0;
    for (uint32_t token : {0u, 255u, 256u, count - 1}) {
        for (uint32_t h : {0u, 63u}) {
            const uint32_t visible = pos + token + 1;
            std::vector<double> scores(visible);
            for (uint32_t s = 0; s < visible; ++s) {
                double sum = 0;
                for (uint32_t j = 0; j < width; ++j)
                    sum += double(low[uint64_t(token) * row + h * width + j]) *
                           double(cache[uint64_t(s) * width + j]);
                scores[s] = sum / 16.0;
            }
            const double maximum = *std::max_element(scores.begin(), scores.end());
            double denom = 0;
            for (double &s : scores) { s = std::exp(s - maximum); denom += s; }
            for (uint32_t j : {0u, 511u}) {
                double value = 0;
                for (uint32_t s = 0; s < visible; ++s)
                    value += scores[s] * cache[uint64_t(s) * width + j];
                const double error = std::abs(value / denom -
                    actual[uint64_t(token) * row + h * width + j]);
                max_error = std::max(max_error, error);
                CHECK(error < 2e-5);
            }
        }
    }
    // Mutating future rows, including the second tile, cannot affect tile one.
    const uint64_t future = uint64_t(pos + 256) * width;
    for (uint64_t i = future; i < cache.size(); ++i) cache[i] = next() * 13.37f;
    CHECK(ds4_gpu_tensor_write(dcache.get(), 0, cache.data(), cache.size() * sizeof(float)));
    CHECK(attention(dout.get(), dq.get(), dlow.get(), dcache.get(), count, pos, pos + count));
    CHECK(ds4_gpu_synchronize());
    CHECK(ds4_gpu_tensor_read(dout.get(), 0, reference.data(), 256 * row * sizeof(float)));
    CHECK(std::memcmp(actual.data(), reference.data(), 256 * row * sizeof(float)) == 0);

    auto short_out = view(dout.get(), 0, elements * sizeof(float) - sizeof(float));
    CHECK(short_out);
    CHECK(!attention(short_out.get(), dq.get(), dlow.get(), dcache.get(), count, pos, pos + count));
    CHECK(!attention(dout.get(), dq.get(), dlow.get(), dcache.get(), count, cap - 1, cap));
    CHECK(!attention(dout.get(), dq.get(), dlow.get(), dcache.get(), count, pos, pos + count - 1));
    std::fprintf(stderr, "dense-subtiles count=%u pos=%u exact=1 future_mask=1 oracle_max_abs=%.9g\n",
                 count, pos, max_error);
    return true;
}

int main() {
    hipDeviceProp_t properties{};
    if (hipGetDeviceProperties(&properties, 0) != hipSuccess ||
        std::strncmp(properties.gcnArchName, "gfx1151", 7) != 0 || properties.warpSize != 32)
        return 2;
    setenv("DS4_ROCM_GLM_CAUSAL_ATTN_HEAD_SHARED", "1", 1);
    ds4_gpu_config config{};
    config.n_gpus = 1;
    if (!ds4_gpu_init_multi(&config)) return 2;
    const bool ok = run_case(512, 0) && run_case(512, 13) && run_case(512, 1536) &&
                    run_case(1024, 0) && run_case(1024, 1024);
    ds4_gpu_cleanup();
    return ok ? 0 : 1;
}
