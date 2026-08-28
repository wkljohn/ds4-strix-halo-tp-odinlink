/* Byte-exact bounded-loader gate for one GLM-5.3 routed Q4_K expert.
 *
 * The production model cannot be resident in this boot's Linux-backed GTT.
 * This test therefore declares the real layer-3 tables but loads only one
 * rank-local 1024-column expert window at a time into three reusable slots.
 * It must never materialize a complete packed expert table. */

#include "ds4_gpu.h"
#include "glm5_gguf_test.hpp"

#include <hip/hip_runtime.h>

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr uint32_t kExperts = 288;
constexpr uint32_t kFfn = 2048;
constexpr uint32_t kEmbed = 4096;
constexpr uint32_t kHalfFfn = 1024;
constexpr uint64_t kQ4KBlockBytes = 144;
constexpr uint64_t kGateRowBytes = (kEmbed / 256u) * kQ4KBlockBytes;
constexpr uint64_t kDownRowBytes = (kFfn / 256u) * kQ4KBlockBytes;
constexpr uint64_t kHalfGateBytes =
    (uint64_t)kHalfFfn * kGateRowBytes;
constexpr uint64_t kHalfDownRowBytes = kDownRowBytes / 2u;
constexpr uint64_t kHalfDownBytes =
    (uint64_t)kEmbed * kHalfDownRowBytes;
static_assert(kHalfGateBytes == 2359296u, "GLM5 gate half geometry");
static_assert(kHalfDownBytes == 2359296u, "GLM5 down half geometry");

#define CHECK(cond, msg) do {                                                \
    if (!(cond)) {                                                           \
        std::fprintf(stderr, "FAIL: %s (line %d)\n", (msg), __LINE__);      \
        return 1;                                                            \
    }                                                                        \
} while (0)

uint64_t fnv1a64(const void *data, uint64_t bytes) {
    const auto *p = static_cast<const unsigned char *>(data);
    uint64_t hash = UINT64_C(1469598103934665603);
    for (uint64_t i = 0; i < bytes; ++i) {
        hash ^= p[i];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

bool tensor_range(const Glm5TestGGUF &gguf, uint64_t offset,
                  uint64_t bytes) {
    return offset <= gguf.size && bytes <= gguf.size - offset;
}

void expected_row_half(std::vector<unsigned char> &out,
                       const Glm5TestGGUF &gguf, uint64_t tensor_offset,
                       uint32_t expert, uint32_t half) {
    const uint64_t full_expert_bytes =
        (uint64_t)kFfn * kGateRowBytes;
    const uint64_t source = tensor_offset +
        (uint64_t)expert * full_expert_bytes +
        (uint64_t)half * kHalfGateBytes;
    std::memcpy(out.data(), gguf.map + source, (size_t)kHalfGateBytes);
}

void expected_down_half(std::vector<unsigned char> &out,
                        const Glm5TestGGUF &gguf, uint64_t tensor_offset,
                        uint32_t expert, uint32_t half) {
    const uint64_t full_expert_bytes =
        (uint64_t)kEmbed * kDownRowBytes;
    const unsigned char *source = gguf.map + tensor_offset +
        (uint64_t)expert * full_expert_bytes;
    for (uint32_t row = 0; row < kEmbed; ++row) {
        std::memcpy(out.data() + (uint64_t)row * kHalfDownRowBytes,
                    source + (uint64_t)row * kDownRowBytes +
                        (uint64_t)half * kHalfDownRowBytes,
                    (size_t)kHalfDownRowBytes);
    }
}

bool declare_half(const Glm5TestGGUF &gguf, uint64_t gate_offset,
                  uint64_t up_offset, uint64_t down_offset, uint32_t half) {
    const uint32_t row_base = half * kHalfFfn;
    const uint64_t column_base = (uint64_t)half * kHalfDownRowBytes;
    return ds4_gpu_q4k_packed_slice_declare(
               gguf.map, gguf.size, gate_offset, kExperts, kFfn,
               kGateRowBytes, row_base, kHalfFfn, 0u, kGateRowBytes,
               DS4_GPU_Q4K_PACKED_ROW_RANGE) &&
           ds4_gpu_q4k_packed_slice_declare(
               gguf.map, gguf.size, up_offset, kExperts, kFfn,
               kGateRowBytes, row_base, kHalfFfn, 0u, kGateRowBytes,
               DS4_GPU_Q4K_PACKED_ROW_RANGE) &&
           ds4_gpu_q4k_packed_slice_declare(
               gguf.map, gguf.size, down_offset, kExperts, kEmbed,
               kDownRowBytes, 0u, kEmbed, column_base, kHalfDownRowBytes,
               DS4_GPU_Q4K_PACKED_K_RANGE);
}

}  // namespace

int main() {
    const char *model = std::getenv("DS4_GLM5_MODEL");
    CHECK(model && model[0], "DS4_GLM5_MODEL is required");

    Glm5TestGGUF gguf;
    CHECK(gguf.open_file(model), "open GLM-5.3 GGUF");
    uint64_t gate_offset = 0, up_offset = 0, down_offset = 0;
    CHECK(gguf.tensor("blk.3.ffn_gate_exps.weight",
                      {kEmbed, kFfn, kExperts}, 12u, gate_offset),
          "bind layer-3 gate experts");
    CHECK(gguf.tensor("blk.3.ffn_up_exps.weight",
                      {kEmbed, kFfn, kExperts}, 12u, up_offset),
          "bind layer-3 up experts");
    CHECK(gguf.tensor("blk.3.ffn_down_exps.weight",
                      {kFfn, kEmbed, kExperts}, 12u, down_offset),
          "bind layer-3 down experts");
    CHECK(tensor_range(gguf, gate_offset,
                       (uint64_t)kExperts * kFfn * kGateRowBytes) &&
          tensor_range(gguf, up_offset,
                       (uint64_t)kExperts * kFfn * kGateRowBytes) &&
          tensor_range(gguf, down_offset,
                       (uint64_t)kExperts * kEmbed * kDownRowBytes),
          "expert tensor ranges");

    int devices = 0;
    CHECK(hipGetDeviceCount(&devices) == hipSuccess && devices > 0,
          "gfx1151 device available");
    CHECK(ds4_gpu_init(), "initialize ROCm backend");
    CHECK(ds4_gpu_set_model_map(gguf.map, gguf.size), "set model map");
    CHECK(ds4_gpu_set_model_fd_for_map(gguf.fd, gguf.map),
          "set model fd for map");
    CHECK(declare_half(gguf, gate_offset, up_offset, down_offset, 0u) &&
          declare_half(gguf, gate_offset, up_offset, down_offset, 1u),
          "declare both rank-local halves");
    CHECK(ds4_gpu_q4k_packed_slice_bytes() == 0u,
          "declaration creates no full-table residency");

    ds4_gpu_tensor *gate = ds4_gpu_tensor_alloc(kHalfGateBytes);
    ds4_gpu_tensor *up = ds4_gpu_tensor_alloc(kHalfGateBytes);
    ds4_gpu_tensor *down = ds4_gpu_tensor_alloc(kHalfDownBytes);
    CHECK(gate && up && down, "allocate three reusable expert slots");
    const uint64_t reusable_device_bytes =
        ds4_gpu_tensor_bytes(gate) + ds4_gpu_tensor_bytes(up) +
        ds4_gpu_tensor_bytes(down);
    CHECK(reusable_device_bytes ==
              2u * kHalfGateBytes + kHalfDownBytes,
          "exact reusable slot allocation");

    std::vector<unsigned char> expected(kHalfGateBytes);
    std::vector<unsigned char> got(kHalfGateBytes);
    const uint32_t expert_ids[] = {0u, 6u, 67u, 211u, 287u};
    /* Warm the exact same source-page and host-allocation pattern before the
     * device-memory stability measurement.  On gfx1151 UMA, first-touching
     * GGUF mmap pages is reflected by hipMemGetInfo even though it is not a
     * device allocation. */
    for (uint32_t half = 0; half < 2u; ++half) {
        const uint32_t row_base = half * kHalfFfn;
        const uint64_t column_base = (uint64_t)half * kHalfDownRowBytes;
        for (uint32_t expert : expert_ids) {
            CHECK(ds4_gpu_q4k_packed_slice_load_expert(
                      gguf.map, gate_offset, row_base, kHalfFfn,
                      0u, kGateRowBytes, expert, gate) &&
                  ds4_gpu_q4k_packed_slice_load_expert(
                      gguf.map, up_offset, row_base, kHalfFfn,
                      0u, kGateRowBytes, expert, up) &&
                  ds4_gpu_q4k_packed_slice_load_expert(
                      gguf.map, down_offset, 0u, kEmbed, column_base,
                      kHalfDownRowBytes, expert, down),
                  "warm selected expert loader");
        }
    }
    size_t free_before = 0, total_before = 0;
    CHECK(hipMemGetInfo(&free_before, &total_before) == hipSuccess,
          "measure device memory before reuse loop");
    uint64_t aggregate = UINT64_C(1469598103934665603);
    for (uint32_t half = 0; half < 2u; ++half) {
        const uint32_t row_base = half * kHalfFfn;
        const uint64_t column_base = (uint64_t)half * kHalfDownRowBytes;
        for (uint32_t expert : expert_ids) {
            CHECK(ds4_gpu_q4k_packed_slice_load_expert(
                      gguf.map, gate_offset, row_base, kHalfFfn,
                      0u, kGateRowBytes, expert, gate),
                  "load selected gate expert half");
            expected_row_half(expected, gguf, gate_offset, expert, half);
            CHECK(ds4_gpu_tensor_read(gate, 0u, got.data(), kHalfGateBytes) &&
                      std::memcmp(expected.data(), got.data(),
                                  (size_t)kHalfGateBytes) == 0,
                  "gate expert half is byte-exact");
            aggregate ^= fnv1a64(got.data(), kHalfGateBytes);
            aggregate *= UINT64_C(1099511628211);

            CHECK(ds4_gpu_q4k_packed_slice_load_expert(
                      gguf.map, up_offset, row_base, kHalfFfn,
                      0u, kGateRowBytes, expert, up),
                  "load selected up expert half");
            expected_row_half(expected, gguf, up_offset, expert, half);
            CHECK(ds4_gpu_tensor_read(up, 0u, got.data(), kHalfGateBytes) &&
                      std::memcmp(expected.data(), got.data(),
                                  (size_t)kHalfGateBytes) == 0,
                  "up expert half is byte-exact");
            aggregate ^= fnv1a64(got.data(), kHalfGateBytes);
            aggregate *= UINT64_C(1099511628211);

            CHECK(ds4_gpu_q4k_packed_slice_load_expert(
                      gguf.map, down_offset, 0u, kEmbed, column_base,
                      kHalfDownRowBytes, expert, down),
                  "load selected down expert half");
            expected_down_half(expected, gguf, down_offset, expert, half);
            CHECK(ds4_gpu_tensor_read(down, 0u, got.data(), kHalfDownBytes) &&
                      std::memcmp(expected.data(), got.data(),
                                  (size_t)kHalfDownBytes) == 0,
                  "down expert half is byte-exact");
            aggregate ^= fnv1a64(got.data(), kHalfDownBytes);
            aggregate *= UINT64_C(1099511628211);

            CHECK(ds4_gpu_q4k_packed_slice_bytes() == 0u,
                  "selected loads create no full-table residency");
        }
    }

    size_t free_after = 0, total_after = 0;
    CHECK(hipMemGetInfo(&free_after, &total_after) == hipSuccess &&
              total_after == total_before,
          "measure stable device memory after reuse loop");
    const uint64_t device_growth = free_before > free_after
        ? (uint64_t)(free_before - free_after) : 0u;
    std::fprintf(stderr,
                 "glm53_expert_window memory free_before=%zu free_after=%zu "
                 "growth=%" PRIu64 "\n",
                 free_before, free_after, device_growth);
    /* hipMemGetInfo includes some Linux-backed mmap/page-cache movement on
     * this UMA platform.  Cap unattributed growth at 16 MiB: over 40x below
     * a one-tensor packed rank half and over 100x below gate+up+down. */
    CHECK(device_growth <= 16u * 1048576u,
          "repeated selected loads do not accumulate table residency");

    CHECK(!ds4_gpu_q4k_packed_slice_load_expert(
              gguf.map, gate_offset, 0u, kHalfFfn, 0u, kGateRowBytes,
              kExperts, gate),
          "out-of-range expert rejected");
    ds4_gpu_tensor *small = ds4_gpu_tensor_alloc(kHalfGateBytes - 1u);
    CHECK(small, "allocate undersized negative-control slot");
    CHECK(!ds4_gpu_q4k_packed_slice_load_expert(
              gguf.map, gate_offset, 0u, kHalfFfn, 0u, kGateRowBytes,
              0u, small),
          "undersized destination rejected");
    CHECK(ds4_gpu_q4k_packed_slice_bytes() == 0u,
          "negative controls preserve zero table residency");
    const uint64_t packed_table_bytes =
        ds4_gpu_q4k_packed_slice_bytes();

    ds4_gpu_tensor_free(small);
    ds4_gpu_tensor_free(down);
    ds4_gpu_tensor_free(up);
    ds4_gpu_tensor_free(gate);
    ds4_gpu_cleanup();

    std::printf("glm53_expert_window fnv64=%016" PRIx64
                " experts=5 halves=2 reusable_device_bytes=%" PRIu64
                " measured_device_growth=%" PRIu64
                " full_table_device_bytes=%" PRIu64 " byte_exact=1\n",
                aggregate, reusable_device_bytes, device_growth,
                packed_table_bytes);
    return 0;
}
