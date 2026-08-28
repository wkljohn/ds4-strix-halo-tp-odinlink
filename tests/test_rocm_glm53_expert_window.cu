/* Byte-exact bounded-loader gate for one GLM-5.3 routed Q4_K expert.
 *
 * The production model cannot be resident in this boot's Linux-backed GTT.
 * This test therefore declares the real layer-3 tables but loads only one
 * rank-local 1024-column expert window at a time into three reusable slots.
 * It must never materialize a complete packed expert table. */

#include "ds4_gpu.h"
#include "ds4_gpu_mgpu.h"
#include "glm5_gguf_test.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

extern "C" int ds4_gpu_routed_moe_batch_q4k_direct_control(
        ds4_gpu_tensor *, ds4_gpu_tensor *, ds4_gpu_tensor *,
        ds4_gpu_tensor *, ds4_gpu_tensor *, const void *, const void *,
        const void *, uint64_t, uint64_t, uint64_t, uint64_t,
        const ds4_gpu_tensor *, const ds4_gpu_tensor *, uint32_t, uint32_t,
        float, const ds4_gpu_tensor *, uint32_t, uint32_t, uint32_t);

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

ds4_gpu_q4k_window_cache_config cache_config(
        const Glm5TestGGUF &gguf, uint64_t gate_offset,
        uint64_t up_offset, uint64_t down_offset, uint32_t half,
        uint32_t slots = 8u) {
    ds4_gpu_q4k_window_cache_config config = {};
    config.model_map = gguf.map;
    config.gate_offset = gate_offset;
    config.up_offset = up_offset;
    config.down_offset = down_offset;
    config.n_expert = kExperts;
    config.gate_row_base = half * kHalfFfn;
    config.gate_row_count = kHalfFfn;
    config.gate_column_byte_count = kGateRowBytes;
    config.down_row_count = kEmbed;
    config.down_column_byte_base = (uint64_t)half * kHalfDownRowBytes;
    config.down_column_byte_count = kHalfDownRowBytes;
    config.slots = slots;
    return config;
}

bool verify_cache_routes(
        ds4_gpu_q4k_window_cache *cache,
        const Glm5TestGGUF &gguf,
        uint64_t gate_offset, uint64_t up_offset, uint64_t down_offset,
        uint32_t half, const int32_t *experts, const int32_t *slots,
        uint32_t count) {
    std::vector<unsigned char> expected(kHalfGateBytes);
    std::vector<unsigned char> gate(kHalfGateBytes);
    std::vector<unsigned char> up(kHalfGateBytes);
    std::vector<unsigned char> down(kHalfDownBytes);
    for (uint32_t i = 0; i < count; ++i) {
        if (slots[i] < 0 || slots[i] >= 8 ||
            !ds4_gpu_q4k_window_cache_read_slot(
                cache, (uint32_t)slots[i],
                gate.data(), gate.size(), up.data(), up.size(),
                down.data(), down.size())) return false;
        expected_row_half(expected, gguf, gate_offset,
                          (uint32_t)experts[i], half);
        if (std::memcmp(expected.data(), gate.data(),
                        (size_t)kHalfGateBytes) != 0) return false;
        expected_row_half(expected, gguf, up_offset,
                          (uint32_t)experts[i], half);
        if (std::memcmp(expected.data(), up.data(),
                        (size_t)kHalfGateBytes) != 0) return false;
        expected_down_half(expected, gguf, down_offset,
                           (uint32_t)experts[i], half);
        if (std::memcmp(expected.data(), down.data(),
                        (size_t)kHalfDownBytes) != 0) return false;
    }
    return true;
}

}  // namespace

int main() {
    const char *model = std::getenv("DS4_GLM5_MODEL");
    CHECK(model && model[0], "DS4_GLM5_MODEL is required");

    Glm5TestGGUF gguf;
    CHECK(gguf.open_file(model), "open GLM-5.3 GGUF");
    uint64_t gate_offset = 0, up_offset = 0, down_offset = 0;
    uint64_t extra_gate_offset = 0, extra_up_offset = 0;
    uint64_t extra_down_offset = 0;
    CHECK(gguf.tensor("blk.3.ffn_gate_exps.weight",
                      {kEmbed, kFfn, kExperts}, 12u, gate_offset),
          "bind layer-3 gate experts");
    CHECK(gguf.tensor("blk.3.ffn_up_exps.weight",
                      {kEmbed, kFfn, kExperts}, 12u, up_offset),
          "bind layer-3 up experts");
    CHECK(gguf.tensor("blk.3.ffn_down_exps.weight",
                      {kFfn, kEmbed, kExperts}, 12u, down_offset),
          "bind layer-3 down experts");
    CHECK(gguf.tensor("blk.4.ffn_gate_exps.weight",
                      {kEmbed, kFfn, kExperts}, 12u, extra_gate_offset),
          "bind layer-4 gate experts for descriptor-growth control");
    CHECK(gguf.tensor("blk.4.ffn_up_exps.weight",
                      {kEmbed, kFfn, kExperts}, 12u, extra_up_offset) &&
          gguf.tensor("blk.4.ffn_down_exps.weight",
                      {kFfn, kEmbed, kExperts}, 12u, extra_down_offset),
          "bind remaining layer-4 experts for descriptor-growth control");
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

    /* Force complete recycling and then a mixed hit/eviction epoch. */
    ds4_gpu_q4k_window_cache_config bad_slots =
        cache_config(gguf, gate_offset, up_offset, down_offset, 0u, 7u);
    CHECK(ds4_gpu_q4k_window_cache_create(&bad_slots) == nullptr,
          "cache refuses fewer slots than top-8 routing requires");
    ds4_gpu_q4k_window_cache_config bad_coupling =
        cache_config(gguf, gate_offset, up_offset, down_offset, 0u);
    bad_coupling.down_column_byte_base = kHalfDownRowBytes;
    CHECK(ds4_gpu_q4k_window_cache_create(&bad_coupling) == nullptr,
          "cache refuses mismatched gate/down rank halves");
    ds4_gpu_q4k_window_cache_config excessive_slots =
        cache_config(gguf, gate_offset, up_offset, down_offset, 0u, 33u);
    CHECK(ds4_gpu_q4k_window_cache_create(&excessive_slots) == nullptr,
          "cache enforces the fixed 32-slot policy ceiling");

    const uint64_t expected_cache_bytes =
        8u * (2u * kHalfGateBytes + kHalfDownBytes);
    size_t cache_free_before = 0, cache_total_before = 0;
    CHECK(hipMemGetInfo(&cache_free_before, &cache_total_before) == hipSuccess,
          "measure memory before compact cache allocation");
    ds4_gpu_q4k_window_cache_config config0 =
        cache_config(gguf, gate_offset, up_offset, down_offset, 0u);
    ds4_gpu_q4k_window_cache_config config1 =
        cache_config(gguf, gate_offset, up_offset, down_offset, 1u);
    ds4_gpu_q4k_window_cache *cache0 =
        ds4_gpu_q4k_window_cache_create(&config0);
    CHECK(cache0, "create rank-0 compact cache");
    size_t cache_free_after = 0, cache_total_after = 0;
    CHECK(hipMemGetInfo(&cache_free_after, &cache_total_after) == hipSuccess &&
              cache_total_after == cache_total_before &&
              cache_free_before >= cache_free_after,
          "measure compact cache allocation");
    const uint64_t measured_cache_bytes =
        (uint64_t)(cache_free_before - cache_free_after);
    CHECK(measured_cache_bytes >= expected_cache_bytes &&
              measured_cache_bytes <= expected_cache_bytes + 8u * 1048576u,
          "measured rank cache stays within fixed allocation envelope");
    ds4_gpu_q4k_window_cache *cache1 =
        ds4_gpu_q4k_window_cache_create(&config1);
    CHECK(cache1, "create peer rank-half cache");
    /* The six layer-3 descriptors leave libstdc++'s vector capacity at eight.
     * Appending both layer-4 halves grows the registry to twelve and forces
     * the 8->16 reallocation that invalidated the old raw-pointer cache. */
    CHECK(declare_half(gguf, extra_gate_offset, extra_up_offset,
                       extra_down_offset, 0u) &&
          declare_half(gguf, extra_gate_offset, extra_up_offset,
                       extra_down_offset, 1u),
          "descriptor reallocation remains valid with live caches");
    const int32_t route_a[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    const int32_t route_b[8] = {8, 9, 10, 11, 12, 13, 14, 15};
    const int32_t route_c[8] = {12, 13, 14, 15, 100, 101, 102, 103};
    int32_t slots_a[8] = {}, slots_a_hit[8] = {};
    int32_t slots_b[8] = {}, slots_c0[8] = {}, slots_c1[8] = {};
    CHECK(ds4_gpu_q4k_window_cache_prepare(
              cache0, route_a, 8u, slots_a) &&
          verify_cache_routes(cache0, gguf, gate_offset, up_offset,
                              down_offset, 0u, route_a, slots_a, 8u),
          "first cache epoch byte-exact");
    CHECK(ds4_gpu_q4k_window_cache_prepare(
              cache0, route_a, 8u, slots_a_hit) &&
          std::memcmp(slots_a, slots_a_hit, sizeof(slots_a)) == 0,
          "cache hit preserves compact slot mapping");
    CHECK(ds4_gpu_q4k_window_cache_prepare(
              cache0, route_b, 8u, slots_b) &&
          verify_cache_routes(cache0, gguf, gate_offset, up_offset,
                              down_offset, 0u, route_b, slots_b, 8u),
          "complete slot recycle is byte-exact");
    CHECK(ds4_gpu_q4k_window_cache_prepare(
              cache0, route_c, 8u, slots_c0) &&
          verify_cache_routes(cache0, gguf, gate_offset, up_offset,
                              down_offset, 0u, route_c, slots_c0, 8u),
          "mixed hit and LRU eviction epoch is byte-exact");
    CHECK(ds4_gpu_q4k_window_cache_prepare(
              cache1, route_c, 8u, slots_c1) &&
          verify_cache_routes(cache1, gguf, gate_offset, up_offset,
                              down_offset, 1u, route_c, slots_c1, 8u),
          "peer rank-half cache is independently byte-exact");

    std::vector<unsigned char> rank0_gate(kHalfGateBytes);
    std::vector<unsigned char> rank0_up(kHalfGateBytes);
    std::vector<unsigned char> rank0_down(kHalfDownBytes);
    std::vector<unsigned char> rank1_gate(kHalfGateBytes);
    std::vector<unsigned char> rank1_up(kHalfGateBytes);
    std::vector<unsigned char> rank1_down(kHalfDownBytes);
    CHECK(ds4_gpu_q4k_window_cache_read_slot(
              cache0, (uint32_t)slots_c0[0], rank0_gate.data(),
              rank0_gate.size(), rank0_up.data(), rank0_up.size(),
              rank0_down.data(), rank0_down.size()) &&
          ds4_gpu_q4k_window_cache_read_slot(
              cache1, (uint32_t)slots_c1[0], rank1_gate.data(),
              rank1_gate.size(), rank1_up.data(), rank1_up.size(),
              rank1_down.data(), rank1_down.size()) &&
          std::memcmp(rank0_gate.data(), rank1_gate.data(),
                      (size_t)kHalfGateBytes) != 0 &&
          std::memcmp(rank0_up.data(), rank1_up.data(),
                      (size_t)kHalfGateBytes) != 0 &&
          std::memcmp(rank0_down.data(), rank1_down.data(),
                      (size_t)kHalfDownBytes) != 0,
          "rank-half identity cannot collide");
    CHECK(!ds4_gpu_q4k_window_cache_read_slot(
              cache0, (uint32_t)slots_c0[0], rank0_gate.data(),
              rank0_gate.size() - 1u, rank0_up.data(), rank0_up.size(),
              rank0_down.data(), rank0_down.size()),
          "slot readback refuses undersized destinations");

    ds4_gpu_q4k_window_cache_stats stats0 = {}, stats1 = {};
    CHECK(ds4_gpu_q4k_window_cache_get_stats(cache0, &stats0) &&
          ds4_gpu_q4k_window_cache_get_stats(cache1, &stats1),
          "read cache statistics");
    CHECK(stats0.prepares == 4u && stats0.hits == 12u &&
          stats0.misses == 20u && stats0.fills == 20u &&
          stats0.evictions == 12u && stats0.resident_count == 8u &&
          stats0.slot_count == 8u &&
          stats0.capacity_bytes == expected_cache_bytes,
          "rank-0 cache hit/miss/eviction accounting");
    CHECK(stats1.prepares == 1u && stats1.hits == 0u &&
          stats1.misses == 8u && stats1.fills == 8u &&
          stats1.evictions == 0u && stats1.resident_count == 8u &&
          stats1.slot_count == 8u &&
          stats1.capacity_bytes == expected_cache_bytes,
          "rank-1 cache accounting");
    const void *view_gate = nullptr, *view_up = nullptr, *view_down = nullptr;
    uint64_t view_gate_bytes = 0, view_down_bytes = 0;
    CHECK(ds4_gpu_q4k_window_cache_device_view(
              cache0, &view_gate, &view_up, &view_down,
              &view_gate_bytes, &view_down_bytes) &&
          view_gate && view_up && view_down &&
          view_gate_bytes == kHalfGateBytes &&
          view_down_bytes == kHalfDownBytes,
          "compact cache exposes direct MoE slot slabs");
    std::vector<unsigned char> view_gate_copy(kHalfGateBytes);
    std::vector<unsigned char> view_up_copy(kHalfGateBytes);
    std::vector<unsigned char> view_down_copy(kHalfDownBytes);
    CHECK(hipMemcpy(
              view_gate_copy.data(),
              static_cast<const char *>(view_gate) +
                  (uint64_t)slots_c0[0] * view_gate_bytes,
              (size_t)view_gate_bytes, hipMemcpyDeviceToHost) == hipSuccess &&
          hipMemcpy(
              view_up_copy.data(),
              static_cast<const char *>(view_up) +
                  (uint64_t)slots_c0[0] * view_gate_bytes,
              (size_t)view_gate_bytes, hipMemcpyDeviceToHost) == hipSuccess &&
          hipMemcpy(
              view_down_copy.data(),
              static_cast<const char *>(view_down) +
                  (uint64_t)slots_c0[0] * view_down_bytes,
              (size_t)view_down_bytes, hipMemcpyDeviceToHost) == hipSuccess &&
          std::memcmp(view_gate_copy.data(), rank0_gate.data(),
                      (size_t)kHalfGateBytes) == 0 &&
          std::memcmp(view_up_copy.data(), rank0_up.data(),
                      (size_t)kHalfGateBytes) == 0 &&
          std::memcmp(view_down_copy.data(), rank0_down.data(),
                      (size_t)kHalfDownBytes) == 0,
          "published slab bases and ordering are byte-exact");
    const int32_t invalid_low[1] = {-2};
    const int32_t invalid_high[1] = {(int32_t)kExperts};
    const int32_t too_many[9] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    int32_t negative_slots[9] = {};
    CHECK(!ds4_gpu_q4k_window_cache_prepare(
              cache0, route_a, 0u, negative_slots) &&
          !ds4_gpu_q4k_window_cache_prepare(
              cache0, invalid_low, 1u, negative_slots) &&
          !ds4_gpu_q4k_window_cache_prepare(
              cache0, invalid_high, 1u, negative_slots) &&
          !ds4_gpu_q4k_window_cache_prepare(
              cache0, too_many, 9u, negative_slots),
          "invalid and over-capacity cache routes fail closed");
    const int32_t host_padding[1] = {-1};
    CHECK(ds4_gpu_q4k_window_cache_prepare(
              cache0, host_padding, 1u, negative_slots) &&
          negative_slots[0] == -1,
          "host router padding preserves the kernel skip sentinel");
    const int32_t duplicates[8] =
        {12, 12, 13, 13, 100, 100, 103, 103};
    int32_t duplicate_slots[8] = {};
    CHECK(ds4_gpu_q4k_window_cache_prepare(
              cache0, duplicates, 8u, duplicate_slots) &&
          duplicate_slots[0] == duplicate_slots[1] &&
          duplicate_slots[2] == duplicate_slots[3] &&
          duplicate_slots[4] == duplicate_slots[5] &&
          duplicate_slots[6] == duplicate_slots[7],
          "duplicate global IDs share their compact slots");
    std::vector<int32_t> batch_ids(512u);
    std::vector<float> batch_weights(512u, 0.125f);
    for (uint32_t i = 0; i < batch_ids.size(); ++i) {
        batch_ids[i] = route_c[i & 7u];
        if (i % 17u == 0u) {
            batch_ids[i] = -1;
            batch_weights[i] = 0.0f;
        }
    }
    ds4_gpu_tensor *batch_ids_device =
        ds4_gpu_tensor_alloc(batch_ids.size() * sizeof(int32_t));
    ds4_gpu_tensor *batch_weights_device =
        ds4_gpu_tensor_alloc(batch_weights.size() * sizeof(float));
    ds4_gpu_tensor batch_slots_device = {};
    std::vector<int32_t> batch_slots(batch_ids.size());
    CHECK(batch_ids_device && batch_weights_device &&
          ds4_gpu_tensor_write(batch_ids_device, 0u, batch_ids.data(),
                               batch_ids.size() * sizeof(int32_t)) &&
          ds4_gpu_tensor_write(batch_weights_device, 0u,
                               batch_weights.data(),
                               batch_weights.size() * sizeof(float)) &&
          ds4_gpu_q4k_window_cache_prepare_device(
              cache0, batch_ids_device, batch_weights_device,
              (uint32_t)batch_ids.size(), &batch_slots_device) &&
          ds4_gpu_tensor_read(&batch_slots_device, 0u, batch_slots.data(),
                              batch_slots.size() * sizeof(int32_t)),
          "pair count is independent from the compact expert count");
    int32_t batch_expert_slot[kExperts];
    std::fill_n(batch_expert_slot, kExperts, -2);
    bool batch_slot_seen[8] = {};
    for (uint32_t i = 0; i < batch_slots.size(); ++i) {
        CHECK(batch_ids[i] == -1 ? batch_slots[i] == -1 :
              (batch_slots[i] >= 0 && batch_slots[i] < 8),
              "batched compact slot and padding range");
        if (batch_ids[i] >= 0) {
            int32_t &known = batch_expert_slot[(uint32_t)batch_ids[i]];
            CHECK(known == -2 || known == batch_slots[i],
                  "equal batched expert IDs share one compact slot");
            known = batch_slots[i];
            batch_slot_seen[(uint32_t)batch_slots[i]] = true;
        }
    }
    CHECK(std::all_of(batch_slot_seen, batch_slot_seen + 8,
                      [](bool seen) { return seen; }),
          "eight batched experts occupy eight distinct compact slots");
    ds4_gpu_tensor_free(batch_weights_device);
    ds4_gpu_tensor_free(batch_ids_device);

    /* Drive the real routed-MoE consumer through the same eviction epochs.
     * Its independent control table is packed directly from the GGUF in route
     * order; it does not read cache storage or reuse the cache slot mapping. */
    CHECK(setenv("DS4_ROCM_Q4K_KSHARD_RESEARCH", "1", 1) == 0,
          "enable isolated Q4_K consumer control");
    constexpr uint64_t kPairs = 8u;
    constexpr uint64_t kMidBytes = kPairs * kHalfFfn * sizeof(float);
    constexpr uint64_t kDownBytes = kPairs * kEmbed * sizeof(float);
    constexpr uint64_t kOutBytes = kEmbed * sizeof(float);
    ds4_gpu_tensor *moe_selected = ds4_gpu_tensor_alloc(
        kPairs * sizeof(int32_t));
    ds4_gpu_tensor *moe_weights = ds4_gpu_tensor_alloc(
        kPairs * sizeof(float));
    ds4_gpu_tensor *moe_input = ds4_gpu_tensor_alloc(
        kEmbed * sizeof(float));
    ds4_gpu_tensor *moe_gate = ds4_gpu_tensor_alloc(kMidBytes);
    ds4_gpu_tensor *moe_up = ds4_gpu_tensor_alloc(kMidBytes);
    ds4_gpu_tensor *moe_mid = ds4_gpu_tensor_alloc(kMidBytes);
    ds4_gpu_tensor *moe_down = ds4_gpu_tensor_alloc(kDownBytes);
    ds4_gpu_tensor *moe_cached = ds4_gpu_tensor_alloc(kOutBytes);
    ds4_gpu_tensor *moe_direct = ds4_gpu_tensor_alloc(kOutBytes);
    CHECK(moe_selected && moe_weights && moe_input && moe_gate && moe_up &&
          moe_mid && moe_down && moe_cached && moe_direct,
          "allocate compact-window MoE component tensors");
    std::vector<float> moe_input_host(kEmbed);
    for (uint32_t i = 0; i < kEmbed; ++i) {
        moe_input_host[i] =
            (float)((int)((i * 29u + 17u) % 251u) - 125) / 512.0f;
    }
    const float moe_weight_host[8] = {
        8.0f / 36.0f, 7.0f / 36.0f, 6.0f / 36.0f, 5.0f / 36.0f,
        4.0f / 36.0f, 3.0f / 36.0f, 2.0f / 36.0f, 1.0f / 36.0f};
    CHECK(ds4_gpu_tensor_write(moe_input, 0u, moe_input_host.data(),
                               kEmbed * sizeof(float)) &&
          ds4_gpu_tensor_write(moe_weights, 0u, moe_weight_host,
                               sizeof(moe_weight_host)),
          "upload compact-window MoE input and route weights");
    void *direct_gate = nullptr, *direct_up = nullptr, *direct_down = nullptr;
    const uint64_t direct_table_bytes = 8u * kHalfGateBytes;
    CHECK(hipMalloc(&direct_gate, direct_table_bytes) == hipSuccess &&
          hipMalloc(&direct_up, direct_table_bytes) == hipSuccess &&
          hipMalloc(&direct_down, direct_table_bytes) == hipSuccess,
          "allocate independent direct-control tables");
    std::vector<unsigned char> direct_gate_host(direct_table_bytes);
    std::vector<unsigned char> direct_up_host(direct_table_bytes);
    std::vector<unsigned char> direct_down_host(direct_table_bytes);
    std::vector<unsigned char> direct_one(kHalfGateBytes);
    std::vector<float> cached_output(kEmbed), direct_output(kEmbed);
    const int32_t *moe_epochs[4] = {route_a, route_a, route_b, route_c};
    const int32_t compact_ids[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    uint64_t moe_output_fnv = UINT64_C(1469598103934665603);
    uint64_t first_moe_hash = 0u, moe_nonzero = 0u;
    bool moe_outputs_changed = false;
    for (uint32_t half = 0; half < 2u; ++half) {
        ds4_gpu_q4k_window_cache *cache = half == 0u ? cache0 : cache1;
        for (uint32_t epoch = 0; epoch < 4u; ++epoch) {
            const int32_t *route = moe_epochs[epoch];
            for (uint32_t slot = 0; slot < 8u; ++slot) {
                expected_row_half(direct_one, gguf, gate_offset,
                                  (uint32_t)route[slot], half);
                std::memcpy(direct_gate_host.data() +
                                (uint64_t)slot * kHalfGateBytes,
                            direct_one.data(), (size_t)kHalfGateBytes);
                expected_row_half(direct_one, gguf, up_offset,
                                  (uint32_t)route[slot], half);
                std::memcpy(direct_up_host.data() +
                                (uint64_t)slot * kHalfGateBytes,
                            direct_one.data(), (size_t)kHalfGateBytes);
                expected_down_half(direct_one, gguf, down_offset,
                                   (uint32_t)route[slot], half);
                std::memcpy(direct_down_host.data() +
                                (uint64_t)slot * kHalfDownBytes,
                            direct_one.data(), (size_t)kHalfDownBytes);
            }
            CHECK(hipMemcpy(direct_gate, direct_gate_host.data(),
                            direct_table_bytes,
                            hipMemcpyHostToDevice) == hipSuccess &&
                  hipMemcpy(direct_up, direct_up_host.data(),
                            direct_table_bytes,
                            hipMemcpyHostToDevice) == hipSuccess &&
                  hipMemcpy(direct_down, direct_down_host.data(),
                            direct_table_bytes,
                            hipMemcpyHostToDevice) == hipSuccess &&
                  ds4_gpu_tensor_write(moe_selected, 0u, route,
                                       8u * sizeof(int32_t)),
                  "upload independent epoch tables and global IDs");
            CHECK(ds4_gpu_routed_moe_one_packed_q4k_window_tensor(
                      moe_cached, moe_gate, moe_up, moe_mid, moe_down,
                      cache, moe_selected, moe_weights, 8u, 10.0f,
                      moe_input, nullptr, 3u) &&
                  ds4_gpu_synchronize() &&
                  ds4_gpu_tensor_read(moe_cached, 0u,
                                      cached_output.data(), kOutBytes),
                  "execute bounded compact-window routed MoE");
            CHECK(ds4_gpu_tensor_write(moe_selected, 0u, compact_ids,
                                       sizeof(compact_ids)) &&
                  ds4_gpu_routed_moe_batch_q4k_direct_control(
                      moe_direct, moe_gate, moe_up, moe_mid, moe_down,
                      direct_gate, direct_up, direct_down,
                      kHalfGateBytes, kGateRowBytes,
                      kHalfDownBytes, kHalfDownRowBytes,
                      moe_selected, moe_weights, 8u, 8u, 10.0f,
                      moe_input, 3u, 1u, kHalfFfn) &&
                  ds4_gpu_synchronize() &&
                  ds4_gpu_tensor_read(moe_direct, 0u,
                                      direct_output.data(), kOutBytes) &&
                  std::memcmp(cached_output.data(), direct_output.data(),
                              (size_t)kOutBytes) == 0,
                  "compact-window output matches direct packed oracle");
            const uint64_t output_hash =
                fnv1a64(cached_output.data(), kOutBytes);
            if (first_moe_hash == 0u) first_moe_hash = output_hash;
            else moe_outputs_changed |= output_hash != first_moe_hash;
            for (float value : cached_output) {
                CHECK(std::isfinite(value),
                      "compact-window MoE output remains finite");
                moe_nonzero += value != 0.0f;
            }
            moe_output_fnv ^= output_hash;
            moe_output_fnv *= UINT64_C(1099511628211);
        }
    }
    CHECK(moe_nonzero != 0u && moe_outputs_changed &&
          moe_output_fnv == UINT64_C(0x9b64bad7c433cb8f),
          "MoE epochs are nontrivial, distinct, and deterministic");
    int32_t padded_route[8];
    float padded_weights[8];
    std::memcpy(padded_route, route_c, sizeof(padded_route));
    std::memcpy(padded_weights, moe_weight_host, sizeof(padded_weights));
    padded_route[7] = -1;
    padded_weights[7] = 0.0f;
    for (uint32_t slot = 0; slot < 8u; ++slot) {
        expected_row_half(direct_one, gguf, gate_offset,
                          (uint32_t)route_c[slot], 0u);
        std::memcpy(direct_gate_host.data() +
                        (uint64_t)slot * kHalfGateBytes,
                    direct_one.data(), (size_t)kHalfGateBytes);
        expected_row_half(direct_one, gguf, up_offset,
                          (uint32_t)route_c[slot], 0u);
        std::memcpy(direct_up_host.data() +
                        (uint64_t)slot * kHalfGateBytes,
                    direct_one.data(), (size_t)kHalfGateBytes);
        expected_down_half(direct_one, gguf, down_offset,
                           (uint32_t)route_c[slot], 0u);
        std::memcpy(direct_down_host.data() +
                        (uint64_t)slot * kHalfDownBytes,
                    direct_one.data(), (size_t)kHalfDownBytes);
    }
    CHECK(hipMemcpy(direct_gate, direct_gate_host.data(), direct_table_bytes,
                    hipMemcpyHostToDevice) == hipSuccess &&
          hipMemcpy(direct_up, direct_up_host.data(), direct_table_bytes,
                    hipMemcpyHostToDevice) == hipSuccess &&
          hipMemcpy(direct_down, direct_down_host.data(), direct_table_bytes,
                    hipMemcpyHostToDevice) == hipSuccess,
          "upload half-0 padded-route oracle tables");
    CHECK(ds4_gpu_tensor_write(moe_selected, 0u, padded_route,
                               sizeof(padded_route)) &&
          ds4_gpu_tensor_write(moe_weights, 0u, padded_weights,
                               sizeof(padded_weights)) &&
          ds4_gpu_routed_moe_one_packed_q4k_window_tensor(
              moe_cached, moe_gate, moe_up, moe_mid, moe_down,
              cache0, moe_selected, moe_weights, 8u, 10.0f,
              moe_input, nullptr, 3u) &&
          ds4_gpu_synchronize() &&
          ds4_gpu_tensor_read(moe_cached, 0u, cached_output.data(),
                              kOutBytes),
          "zero-weight router padding preserves the kernel skip sentinel");
    int32_t padded_compact_ids[8];
    std::memcpy(padded_compact_ids, compact_ids, sizeof(padded_compact_ids));
    padded_compact_ids[7] = -1;
    CHECK(ds4_gpu_tensor_write(moe_selected, 0u, padded_compact_ids,
                               sizeof(padded_compact_ids)) &&
          ds4_gpu_routed_moe_batch_q4k_direct_control(
              moe_direct, moe_gate, moe_up, moe_mid, moe_down,
              direct_gate, direct_up, direct_down,
              kHalfGateBytes, kGateRowBytes,
              kHalfDownBytes, kHalfDownRowBytes,
              moe_selected, moe_weights, 8u, 8u, 10.0f,
              moe_input, 3u, 1u, kHalfFfn) &&
          ds4_gpu_synchronize() &&
          ds4_gpu_tensor_read(moe_direct, 0u, direct_output.data(),
                              kOutBytes) &&
          std::memcmp(cached_output.data(), direct_output.data(),
                      (size_t)kOutBytes) == 0,
          "zero-weight padding output matches the direct skip oracle");
    padded_weights[7] = 0.125f;
    CHECK(ds4_gpu_tensor_write(moe_selected, 0u, padded_route,
                               sizeof(padded_route)) &&
          ds4_gpu_tensor_write(moe_weights, 0u, padded_weights,
                               sizeof(padded_weights)) &&
          !ds4_gpu_routed_moe_one_packed_q4k_window_tensor(
              moe_cached, moe_gate, moe_up, moe_mid, moe_down,
              cache0, moe_selected, moe_weights, 8u, 10.0f,
              moe_input, nullptr, 3u),
          "nonzero-weight router padding fails closed");
    std::fill_n(padded_route, 8u, -1);
    std::fill_n(padded_weights, 8u, 0.0f);
    CHECK(ds4_gpu_tensor_write(moe_selected, 0u, padded_route,
                               sizeof(padded_route)) &&
          ds4_gpu_tensor_write(moe_weights, 0u, padded_weights,
                               sizeof(padded_weights)) &&
          ds4_gpu_routed_moe_one_packed_q4k_window_tensor(
              moe_cached, moe_gate, moe_up, moe_mid, moe_down,
              cache0, moe_selected, moe_weights, 8u, 10.0f,
              moe_input, nullptr, 3u) &&
          ds4_gpu_synchronize() &&
          ds4_gpu_tensor_read(moe_cached, 0u, cached_output.data(),
                              kOutBytes) &&
          std::all_of(cached_output.begin(), cached_output.end(),
                      [](float value) { return value == 0.0f; }),
          "all-padding route produces an exact zero contribution");
    std::memcpy(padded_route, route_a, sizeof(padded_route));
    std::memcpy(padded_weights, moe_weight_host, sizeof(padded_weights));
    padded_weights[0] = INFINITY;
    CHECK(ds4_gpu_tensor_write(moe_selected, 0u, padded_route,
                               sizeof(padded_route)) &&
          ds4_gpu_tensor_write(moe_weights, 0u, padded_weights,
                               sizeof(padded_weights)) &&
          !ds4_gpu_routed_moe_one_packed_q4k_window_tensor(
              moe_cached, moe_gate, moe_up, moe_mid, moe_down,
              cache0, moe_selected, moe_weights, 8u, 10.0f,
              moe_input, nullptr, 3u),
          "non-finite route weight fails closed independently of fast-math");
    CHECK(hipFree(direct_down) == hipSuccess &&
          hipFree(direct_up) == hipSuccess &&
          hipFree(direct_gate) == hipSuccess,
          "free independent direct-control tables");
    ds4_gpu_tensor_free(moe_direct);
    ds4_gpu_tensor_free(moe_cached);
    ds4_gpu_tensor_free(moe_down);
    ds4_gpu_tensor_free(moe_mid);
    ds4_gpu_tensor_free(moe_up);
    ds4_gpu_tensor_free(moe_gate);
    ds4_gpu_tensor_free(moe_input);
    ds4_gpu_tensor_free(moe_weights);
    ds4_gpu_tensor_free(moe_selected);
    CHECK(ds4_gpu_q4k_packed_slice_bytes() == 0u,
          "window caches retain no full packed table");
    ds4_gpu_q4k_window_cache_destroy(cache1);
    CHECK(!ds4_gpu_q4k_window_cache_get_stats(cache1, &stats1),
          "destroyed cache handles fail closed");
    ds4_gpu_q4k_window_cache_destroy(cache0);

    ds4_gpu_tensor_free(small);
    ds4_gpu_tensor_free(down);
    ds4_gpu_tensor_free(up);
    ds4_gpu_tensor_free(gate);
    ds4_gpu_cleanup();

    std::printf("glm53_expert_window fnv64=%016" PRIx64
                " experts=5 halves=2 reusable_device_bytes=%" PRIu64
                " measured_device_growth=%" PRIu64
                " full_table_device_bytes=%" PRIu64
                " cache_bytes=%" PRIu64
                " measured_cache_bytes=%" PRIu64
                " cache_fills=%" PRIu64 " cache_evictions=%" PRIu64
                " moe_output_fnv=%016" PRIx64
                " rank_half_distinct=1 byte_exact=1 moe_exact=1\n",
                aggregate, reusable_device_bytes, device_growth,
                packed_table_bytes, expected_cache_bytes,
                measured_cache_bytes,
                stats0.fills, stats0.evictions, moe_output_fnv);
    return 0;
}
