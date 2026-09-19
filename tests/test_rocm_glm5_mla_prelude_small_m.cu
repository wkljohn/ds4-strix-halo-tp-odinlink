// Original GGUF Q8_0 MLA input projections versus independent production M1.
#include "ds4_glm5_next_exec.h"
#include "ds4_gpu_mgpu.h"
#include "glm5_gguf_test.hpp"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define REQUIRE(x) do { if (!(x)) { std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); std::exit(1); } } while (0)

struct Guarded {
    ds4_gpu_tensor *storage, *view;
    uint64_t count, tail;
    explicit Guarded(uint64_t n, uint64_t extra_tail = 0u) :
        storage(ds4_gpu_tensor_alloc((n + 32u + extra_tail) * 4u)),
        view(storage ? ds4_gpu_tensor_view(storage, 64u, n * 4u) : nullptr),
        count(n), tail(extra_tail) {
        REQUIRE(view);
    }
    ~Guarded() { ds4_gpu_tensor_free(view); ds4_gpu_tensor_free(storage); }
    void poison() { REQUIRE(ds4_gpu_tensor_fill_f32(storage, 12345.0f, count + 32u + tail)); }
    std::vector<float> read() {
        std::vector<float> values(count + 32u + tail);
        REQUIRE(ds4_gpu_tensor_read(storage, 0u, values.data(), values.size() * 4u));
        for (unsigned i = 0u; i < 16u; ++i)
            REQUIRE(values[i] == 12345.0f && values[count + 16u + i] == 12345.0f);
        for (uint64_t i = count + 32u; i < values.size(); ++i) REQUIRE(values[i] == 12345.0f);
        for (uint64_t i = 16u; i < count + 16u; ++i) REQUIRE(std::isfinite(values[i]));
        return {values.begin() + 16, values.begin() + 16 + count};
    }
};

struct Projection { unsigned layer, k, full_n, n; uint64_t offset; const char *role; };

int main() {
    const char *path = std::getenv("DS4_GLM5_MODEL"); REQUIRE(path);
    const char *tile = std::getenv("DS4_ROCM_GLM5_Q8_DECODE_TILE");
    const char *prefetch = std::getenv("DS4_ROCM_GLM5_Q8_SHAREDX_PREFETCH");
    REQUIRE(tile && std::strcmp(tile, "1") == 0);
    REQUIRE(prefetch && std::strcmp(prefetch, "8") == 0);
    const char *nt = std::getenv("DS4_ROCM_GLM5_Q8_SHAREDX_NONTEMPORAL");
    const char *rows = std::getenv("DS4_ROCM_GLM5_Q8_SHAREDX_ROWS_PER_BLOCK");
    const bool had_nt = nt != nullptr, had_rows = rows != nullptr;
    const std::string original_nt = nt ? nt : "", original_rows = rows ? rows : "";
    REQUIRE(setenv("DS4_GLM5_NEXT_ENABLE_ORDINARY", "1", 1) == 0);
    Glm5TestGGUF gguf; REQUIRE(gguf.open_file(path));
    std::vector<Projection> projections;
    std::vector<uint64_t> offsets, sizes;
    for (unsigned layer = 3u; layer < 45u; layer += 4u) {
        for (const char *role : {"q_a", "q_b", "kv_a_mqa"}) {
            const bool qb = std::strcmp(role, "q_b") == 0;
            const unsigned k = qb ? 1536u : 4096u;
            const unsigned full_n = qb ? 16384u : !std::strcmp(role, "q_a") ? 1536u : 512u;
            uint64_t offset = 0u; char name[80];
            std::snprintf(name, sizeof(name), "blk.%u.attn_%s.weight", layer, role);
            REQUIRE(gguf.tensor(name, {k, full_n}, 8u, offset));
            projections.push_back({layer, k, full_n, qb ? 8192u : full_n, offset, role});
            offsets.push_back(offset); sizes.push_back((uint64_t)full_n * (k / 32u) * 34u);
        }
    }
    REQUIRE(projections.size() == 33u);
    REQUIRE(ds4_gpu_init()); ds4_gpu_set_glm_model(true); ds4_gpu_set_q8_cache_suppressed(1);
    REQUIRE(ds4_gpu_set_model_fd_for_map(gguf.fd, gguf.map));
    {
        Guarded x(6u * 4096u), y(6u * 1536u);
        x.poison(); y.poison();
        REQUIRE(!ds4_rocm_glm5_mla_prelude_q8_small_m_supported(y.view, gguf.map,
            gguf.size, offsets[0], 8u, 4096u, 1536u, 0u, 1536u, 4352u, x.view, 6u));
        REQUIRE(!ds4_rocm_glm5_mla_prelude_q8_small_m(y.view, gguf.map,
            gguf.size, offsets[0], 8u, 4096u, 1536u, 0u, 1536u, 4352u, x.view, 6u));
        for (float value : y.read()) REQUIRE(value == 12345.0f);
    }
    REQUIRE(ds4_gpu_set_model_map_spans(gguf.map, gguf.size, offsets.data(),
        sizes.data(), offsets.size(), *std::max_element(sizes.begin(), sizes.end())));
    uint64_t compared = 0u;
    for (const auto &p : projections) for (unsigned rank = 0u; rank < 2u; ++rank)
        for (unsigned m : {2u, 4u, 6u}) {
        const unsigned first = p.full_n == 16384u ? rank * p.n : 0u;
        const uint64_t stride = (uint64_t)p.k / 32u * 34u;
        // Keep packed token rows; pad only the allocation's end so a final
        // K1536->K2048 overread encounters explicit poison, never unmapped data.
        Guarded x((uint64_t)m * p.k, p.k == 1536u ? 512u : 0u), y((uint64_t)m * p.n);
        std::vector<ds4_gpu_tensor *> xs(m), ys(m);
        for (unsigned t = 0u; t < m; ++t) {
            xs[t] = ds4_gpu_tensor_view(x.view, (uint64_t)t * p.k * 4u, p.k * 4u);
            ys[t] = ds4_gpu_tensor_view(y.view, (uint64_t)t * p.n * 4u, p.n * 4u);
            REQUIRE(xs[t] && ys[t]);
        }
        auto run = [&](bool batch) {
            if (batch) return ds4_rocm_glm5_mla_prelude_q8_small_m(y.view, gguf.map,
                gguf.size, p.offset, 8u, p.k, p.full_n, first, p.n, stride, x.view, m);
            for (unsigned t = 0u; t < m; ++t)
                if (!ds4_gpu_matmul_q8_0_tensor(ys[t], gguf.map, gguf.size,
                        p.offset + (uint64_t)first * stride, p.k, p.n, xs[t], 1u)) return 0;
            return 1;
        };
        REQUIRE(ds4_rocm_glm5_mla_prelude_q8_small_m_supported(y.view, gguf.map,
            gguf.size, p.offset, 8u, p.k, p.full_n, first, p.n, stride, x.view, m));
        for (unsigned seed = 0u; seed < 3u; ++seed) {
            std::vector<float> input(x.count);
            for (uint64_t i = 0u; i < input.size(); ++i)
                input[i] = seed == 0u ? 0.0f : seed == 1u ?
                    ((i & 1u) ? -1.0f : 1.0f) * (i % 3u ? 1e-4f : 10.0f) :
                    ((int)((i * 193u + (i / p.k) * 761u + p.layer * 31u) % 997u) - 498) /
                        (1001.3f + (i % 7u));
            x.poison(); REQUIRE(ds4_gpu_tensor_write(x.view, 0u, input.data(), x.count * 4u));
            y.poison(); REQUIRE(run(false) && ds4_gpu_synchronize());
            const auto reference = y.read();
            y.poison(); REQUIRE(run(true) && ds4_gpu_synchronize());
            const auto candidate = y.read();
            for (uint64_t i = 0u; i < y.count; ++i) if (std::memcmp(&reference[i], &candidate[i], 4u)) {
                std::fprintf(stderr, "DIFF layer=%u role=%s rank=%u m=%u seed=%u index=%llu scalar=%.9g batch=%.9g\n",
                    p.layer, p.role, rank, m, seed, (unsigned long long)i, reference[i], candidate[i]);
                return 1;
            }
            REQUIRE(x.read() == input); compared += y.count;
        }
        if (p.layer == 3u) {
            auto refused = [&](ds4_gpu_tensor *out, uint64_t size, uint64_t offset,
                    unsigned type, unsigned k, unsigned full_n, unsigned row_first,
                    unsigned n, uint64_t row_bytes, ds4_gpu_tensor *in, unsigned tokens) {
                REQUIRE(!ds4_rocm_glm5_mla_prelude_q8_small_m_supported(out, gguf.map,
                    size, offset, type, k, full_n, row_first, n, row_bytes, in, tokens));
                REQUIRE(!ds4_rocm_glm5_mla_prelude_q8_small_m(out, gguf.map,
                    size, offset, type, k, full_n, row_first, n, row_bytes, in, tokens));
            };
            const auto before = y.read();
#define REJECT(OUT, SIZE, OFF, TYPE, K, FULL, FIRST, N, STRIDE, IN, M) \
            refused(OUT, SIZE, OFF, TYPE, K, FULL, FIRST, N, STRIDE, IN, M)
            for (unsigned bad : {0u, 1u, 3u, 5u, 7u, 8u})
                REJECT(y.view, gguf.size, p.offset, 8u, p.k, p.full_n, first, p.n, stride, x.view, bad);
            REJECT(y.view, gguf.size, p.offset, 30u, p.k, p.full_n, first, p.n, stride, x.view, m);
            REJECT(y.view, gguf.size, p.offset, 8u, p.k - 32u, p.full_n, first, p.n, stride, x.view, m);
            REJECT(y.view, gguf.size, p.offset, 8u, p.k, p.full_n, first + 1u, p.n, stride, x.view, m);
            REJECT(y.view, gguf.size, p.offset, 8u, p.k, p.full_n + 1u, first, p.n, stride, x.view, m);
            REJECT(y.view, gguf.size, p.offset, 8u, p.k, p.full_n, p.full_n, p.n, stride, x.view, m);
            REJECT(y.view, gguf.size, p.offset, 8u, p.k, p.full_n, first, p.n - 1u, stride, x.view, m);
            REJECT(y.view, gguf.size, p.offset, 8u, p.k, p.full_n, first, p.n, stride + 34u, x.view, m);
            REJECT(y.view, gguf.size, p.offset + 1u, 8u, p.k, p.full_n, first, p.n, stride, x.view, m);
            REJECT(y.view, UINT64_MAX, UINT64_MAX - 1u, 8u, p.k, p.full_n, first, p.n, stride, x.view, m);
            REJECT(y.view, p.offset + (uint64_t)p.full_n * stride - 1u, p.offset, 8u, p.k, p.full_n, first, p.n, stride, x.view, m);
            REJECT(x.view, gguf.size, p.offset, 8u, p.k, p.full_n, first, p.n, stride, x.view, m);
            auto *short_x = ds4_gpu_tensor_view(x.view, 0u, x.count * 4u - 4u);
            auto *short_y = ds4_gpu_tensor_view(y.view, 0u, y.count * 4u - 4u);
            auto *unaligned = ds4_gpu_tensor_view(y.storage, 65u, y.count * 4u);
            auto *unaligned_x = ds4_gpu_tensor_view(x.storage, 65u, x.count * 4u);
            REQUIRE(short_x && short_y && unaligned && unaligned_x);
            REJECT(y.view, gguf.size, p.offset, 8u, p.k, p.full_n, first, p.n, stride, short_x, m);
            REJECT(short_y, gguf.size, p.offset, 8u, p.k, p.full_n, first, p.n, stride, x.view, m);
            REJECT(unaligned, gguf.size, p.offset, 8u, p.k, p.full_n, first, p.n, stride, x.view, m);
            REJECT(y.view, gguf.size, p.offset, 8u, p.k, p.full_n, first, p.n, stride, unaligned_x, m);
            Guarded touching(x.count + y.count);
            touching.poison();
            auto *overlap = ds4_gpu_tensor_view(touching.view, 4u, x.count * 4u);
            REQUIRE(overlap);
            REJECT(touching.view, gguf.size, p.offset, 8u, p.k, p.full_n, first, p.n, stride, overlap, m);
            for (float value : touching.read()) REQUIRE(value == 12345.0f);
            ds4_gpu_tensor_free(overlap);
            REQUIRE(!ds4_rocm_glm5_mla_prelude_q8_small_m_supported(y.view, gguf.map + 1u,
                gguf.size - 1u, p.offset, 8u, p.k, p.full_n, first, p.n, stride, x.view, m));
            REQUIRE(!ds4_rocm_glm5_mla_prelude_q8_small_m(y.view, gguf.map + 1u,
                gguf.size - 1u, p.offset, 8u, p.k, p.full_n, first, p.n, stride, x.view, m));
            for (const char *mode : {"0", "2", "3", "4", "5", "6", "invalid"}) {
                REQUIRE(setenv("DS4_ROCM_GLM5_Q8_DECODE_TILE", mode, 1) == 0);
                REJECT(y.view, gguf.size, p.offset, 8u, p.k, p.full_n, first, p.n, stride, x.view, m);
            }
            REQUIRE(setenv("DS4_ROCM_GLM5_Q8_DECODE_TILE", "1", 1) == 0);
            REQUIRE(unsetenv("DS4_GLM5_NEXT_ENABLE_ORDINARY") == 0);
            REJECT(y.view, gguf.size, p.offset, 8u, p.k, p.full_n, first, p.n, stride, x.view, m);
            REQUIRE(setenv("DS4_GLM5_NEXT_ENABLE_ORDINARY", "1", 1) == 0);
            REQUIRE(setenv("DS4_ROCM_GLM5_Q8_SHAREDX_PREFETCH", "0", 1) == 0);
            REJECT(y.view, gguf.size, p.offset, 8u, p.k, p.full_n, first, p.n, stride, x.view, m);
            REQUIRE(setenv("DS4_ROCM_GLM5_Q8_SHAREDX_PREFETCH", "8", 1) == 0);
            ds4_gpu_set_quality(true);
            REJECT(y.view, gguf.size, p.offset, 8u, p.k, p.full_n, first, p.n, stride, x.view, m);
            ds4_gpu_set_quality(false);
            REQUIRE(setenv("DS4_ROCM_GLM5_Q8_SHAREDX_NONTEMPORAL", "invalid", 1) == 0);
            REJECT(y.view, gguf.size, p.offset, 8u, p.k, p.full_n, first, p.n, stride, x.view, m);
            REQUIRE((had_nt ? setenv("DS4_ROCM_GLM5_Q8_SHAREDX_NONTEMPORAL", original_nt.c_str(), 1) :
                unsetenv("DS4_ROCM_GLM5_Q8_SHAREDX_NONTEMPORAL")) == 0);
            REQUIRE(setenv("DS4_ROCM_GLM5_Q8_SHAREDX_ROWS_PER_BLOCK", "7", 1) == 0);
            REJECT(y.view, gguf.size, p.offset, 8u, p.k, p.full_n, first, p.n, stride, x.view, m);
            REQUIRE((had_rows ? setenv("DS4_ROCM_GLM5_Q8_SHAREDX_ROWS_PER_BLOCK", original_rows.c_str(), 1) :
                unsetenv("DS4_ROCM_GLM5_Q8_SHAREDX_ROWS_PER_BLOCK")) == 0);
#undef REJECT
            ds4_gpu_tensor_free(short_x); ds4_gpu_tensor_free(short_y); ds4_gpu_tensor_free(unaligned);
            ds4_gpu_tensor_free(unaligned_x);
            REQUIRE(y.read() == before);
            const auto input = x.read();
            auto *adjacent_out = ds4_gpu_tensor_view(touching.view, 0u, y.count * 4u);
            auto *adjacent_x = ds4_gpu_tensor_view(touching.view, y.count * 4u, x.count * 4u);
            REQUIRE(adjacent_out && adjacent_x);
            REQUIRE(ds4_gpu_tensor_write(adjacent_x, 0u, input.data(), x.count * 4u));
            REQUIRE(ds4_rocm_glm5_mla_prelude_q8_small_m(adjacent_out, gguf.map,
                gguf.size, p.offset, 8u, p.k, p.full_n, first, p.n, stride, adjacent_x, m));
            REQUIRE(ds4_gpu_synchronize());
            const auto adjacent = touching.read();
            REQUIRE(!std::memcmp(adjacent.data(), before.data(), y.count * 4u));
            REQUIRE(!std::memcmp(adjacent.data() + y.count, input.data(), x.count * 4u));
            compared += y.count;
            ds4_gpu_tensor_free(adjacent_out); ds4_gpu_tensor_free(adjacent_x);
            hipEvent_t start, end;
            REQUIRE(hipEventCreate(&start) == hipSuccess && hipEventCreate(&end) == hipSuccess);
            for (unsigned sample = 0u; sample < 12u; ++sample) for (unsigned turn = 0u; turn < 2u; ++turn) {
                const unsigned batch = turn ^ (sample & 1u);
                REQUIRE(ds4_gpu_synchronize());
                const auto began = std::chrono::steady_clock::now();
                REQUIRE(hipEventRecord(start) == hipSuccess && run(batch));
                REQUIRE(hipEventRecord(end) == hipSuccess && hipEventSynchronize(end) == hipSuccess);
                const double wall = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - began).count();
                float device = 0.0f; REQUIRE(hipEventElapsedTime(&device, start, end) == hipSuccess);
                if (sample >= 3u)
                    std::printf("MLA_PRELUDE_SAMPLE role=%s rank=%u m=%u sample=%u batch=%u device_ms=%.6f wall_ms=%.6f\n",
                        p.role, rank, m, sample - 3u, batch, device, wall);
            }
            REQUIRE(hipEventDestroy(start) == hipSuccess && hipEventDestroy(end) == hipSuccess);
        }
        std::printf("MLA_PRELUDE_EXACT layer=%u role=%s rank=%u m=%u PASS\n", p.layer, p.role, rank, m);
        std::fflush(stdout);
        for (unsigned t = 0u; t < m; ++t) { ds4_gpu_tensor_free(xs[t]); ds4_gpu_tensor_free(ys[t]); }
    }
    std::printf("PASS MLA prelude exact finite_values=%llu M=2/4/6 weights=original\n", (unsigned long long)compared);
    ds4_gpu_cleanup();
}
