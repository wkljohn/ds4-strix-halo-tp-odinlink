/* GPU-side descriptor oracle for exact routed-MoE producer waves.
 *
 * This deliberately stops before launching arithmetic kernels. It proves the
 * device planner preserves the full stable buckets and full-count thresholds,
 * while tile_ends prevent a partial final tile from entering the next wave.
 */

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK(cond, msg) do {                                                \
    if (!(cond)) {                                                           \
        std::fprintf(stderr, "FAIL: %s (line %d)\n", msg, __LINE__);        \
        return 1;                                                            \
    }                                                                        \
} while (0)

static void hip_check(hipError_t rc, const char *where) {
    if (rc != hipSuccess) {
        std::fprintf(stderr, "FAIL: %s: %s\n", where, hipGetErrorString(rc));
        std::exit(1);
    }
}

__global__ static void build_wave_ranges(
        uint32_t *wave_starts, uint32_t *wave_counts,
        const uint32_t *counts, const uint32_t *offsets,
        const uint32_t *sorted_pairs, uint32_t total_experts,
        uint32_t used_experts, uint32_t row_first, uint32_t row_end) {
    const uint32_t expert = blockIdx.x * blockDim.x + threadIdx.x;
    if (expert >= total_experts) return;
    const uint32_t count = counts[expert];
    const uint32_t base = offsets[expert];
    uint32_t first = 0;
    while (first < count &&
           sorted_pairs[base + first] / used_experts < row_first) first++;
    uint32_t end = first;
    while (end < count &&
           sorted_pairs[base + end] / used_experts < row_end) end++;
    wave_starts[expert] = first;
    wave_counts[expert] = end - first;
}

__global__ static void build_tile_offsets(
        uint32_t *tile_offsets, uint32_t *tile_total,
        const uint32_t *wave_counts, uint32_t total_experts) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    uint32_t total = 0;
    for (uint32_t expert = 0; expert < total_experts; expert++) {
        tile_offsets[expert] = total;
        total += (wave_counts[expert] + 15u) / 16u;
    }
    tile_offsets[total_experts] = total;
    *tile_total = total;
}

__global__ static void build_wave_tiles(
        uint32_t *tile_experts, uint32_t *tile_starts, uint32_t *tile_ends,
        const uint32_t *tile_offsets, const uint32_t *wave_starts,
        const uint32_t *wave_counts, uint32_t total_experts) {
    const uint32_t expert = blockIdx.x * blockDim.x + threadIdx.x;
    if (expert >= total_experts) return;
    const uint32_t first = wave_starts[expert];
    const uint32_t count = wave_counts[expert];
    const uint32_t end = first + count;
    const uint32_t tile0 = tile_offsets[expert];
    for (uint32_t tile = 0; tile < (count + 15u) / 16u; tile++) {
        tile_experts[tile0 + tile] = expert;
        tile_starts[tile0 + tile] = first + tile * 16u;
        tile_ends[tile0 + tile] = end;
    }
}

__global__ static void repack_rows(
        uint32_t *dst, const uint32_t *src, uint32_t full_rows,
        uint32_t blocks, uint32_t row_first, uint32_t row_count) {
    const uint32_t row_local = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t block = blockIdx.y;
    if (row_local >= row_count || block >= blocks) return;
    const uint32_t row = row_first + row_local;
    dst[(uint64_t)block * full_rows + row] =
        src[(uint64_t)row * blocks + block];
}

static int run_case(uint32_t nwaves) {
    constexpr uint32_t tokens = 73, used = 6, total_experts = 32;
    const uint32_t pair_count = tokens * used;
    std::vector<int32_t> selected(pair_count);
    uint32_t state = 0xD54u;
    for (uint32_t pair = 0; pair < pair_count; pair++) {
        state = state * 1664525u + 1013904223u;
        selected[pair] = pair % 19u == 0u ? -1 : (int32_t)(state % 28u);
    }
    std::vector<uint32_t> live;
    for (uint32_t pair = 0; pair < pair_count; pair++)
        if (selected[pair] >= 0) live.push_back(pair);
    uint32_t cursor = 0;
    const uint32_t forced_counts[4] = {1, 15, 16, 17};
    for (uint32_t i = 0; i < 4; i++) {
        for (uint32_t j = 0; j < forced_counts[i]; j++)
            selected[live[cursor++]] = (int32_t)(28u + i);
    }

    std::vector<uint32_t> counts(total_experts, 0), offsets(total_experts + 1, 0);
    for (int32_t expert : selected)
        if (expert >= 0) counts[(uint32_t)expert]++;
    for (uint32_t expert = 0; expert < total_experts; expert++)
        offsets[expert + 1] = offsets[expert] + counts[expert];
    std::vector<uint32_t> sorted_pairs(offsets.back()), cursors = offsets;
    for (uint32_t pair = 0; pair < pair_count; pair++) {
        const int32_t expert = selected[pair];
        if (expert >= 0) sorted_pairs[cursors[(uint32_t)expert]++] = pair;
    }
    CHECK(counts[28] == 1 && counts[29] == 15 &&
          counts[30] == 16 && counts[31] == 17,
          "full-count threshold neighborhoods");

    const uint32_t tile_capacity =
        (uint32_t)((sorted_pairs.size() + 15u) / 16u + total_experts);
    uint32_t *d_counts, *d_offsets, *d_pairs, *d_wave_starts, *d_wave_counts;
    uint32_t *d_tile_offsets, *d_tile_total, *d_te, *d_ts, *d_tend;
    hip_check(hipMalloc(&d_counts, counts.size() * sizeof(uint32_t)), "counts");
    hip_check(hipMalloc(&d_offsets, offsets.size() * sizeof(uint32_t)), "offsets");
    hip_check(hipMalloc(&d_pairs, sorted_pairs.size() * sizeof(uint32_t)), "pairs");
    hip_check(hipMalloc(&d_wave_starts, counts.size() * sizeof(uint32_t)), "wave starts");
    hip_check(hipMalloc(&d_wave_counts, counts.size() * sizeof(uint32_t)), "wave counts");
    hip_check(hipMalloc(&d_tile_offsets, offsets.size() * sizeof(uint32_t)), "tile offsets");
    hip_check(hipMalloc(&d_tile_total, sizeof(uint32_t)), "tile total");
    hip_check(hipMalloc(&d_te, tile_capacity * sizeof(uint32_t)), "tile experts");
    hip_check(hipMalloc(&d_ts, tile_capacity * sizeof(uint32_t)), "tile starts");
    hip_check(hipMalloc(&d_tend, tile_capacity * sizeof(uint32_t)), "tile ends");
    hip_check(hipMemcpy(d_counts, counts.data(), counts.size() * sizeof(uint32_t), hipMemcpyHostToDevice), "copy counts");
    hip_check(hipMemcpy(d_offsets, offsets.data(), offsets.size() * sizeof(uint32_t), hipMemcpyHostToDevice), "copy offsets");
    hip_check(hipMemcpy(d_pairs, sorted_pairs.data(), sorted_pairs.size() * sizeof(uint32_t), hipMemcpyHostToDevice), "copy pairs");

    std::vector<uint8_t> seen(pair_count, 0);
    for (uint32_t wave = 0; wave < nwaves; wave++) {
        const uint32_t row_first = tokens * wave / nwaves;
        const uint32_t row_end = tokens * (wave + 1u) / nwaves;
        build_wave_ranges<<<1, 64>>>(d_wave_starts, d_wave_counts,
            d_counts, d_offsets, d_pairs, total_experts, used,
            row_first, row_end);
        build_tile_offsets<<<1, 1>>>(d_tile_offsets, d_tile_total,
            d_wave_counts, total_experts);
        build_wave_tiles<<<1, 64>>>(d_te, d_ts, d_tend, d_tile_offsets,
            d_wave_starts, d_wave_counts, total_experts);
        hip_check(hipDeviceSynchronize(), "wave planner");
        uint32_t tile_total = 0;
        hip_check(hipMemcpy(&tile_total, d_tile_total, sizeof(tile_total),
                            hipMemcpyDeviceToHost), "copy tile total");
        CHECK(tile_total <= tile_capacity, "tile capacity");
        std::vector<uint32_t> te(tile_total), ts(tile_total), tend(tile_total);
        hip_check(hipMemcpy(te.data(), d_te, tile_total * sizeof(uint32_t), hipMemcpyDeviceToHost), "copy tile experts");
        hip_check(hipMemcpy(ts.data(), d_ts, tile_total * sizeof(uint32_t), hipMemcpyDeviceToHost), "copy tile starts");
        hip_check(hipMemcpy(tend.data(), d_tend, tile_total * sizeof(uint32_t), hipMemcpyDeviceToHost), "copy tile ends");
        for (uint32_t tile = 0; tile < tile_total; tile++) {
            CHECK(ts[tile] < tend[tile], "nonempty tile");
            CHECK(tend[tile] <= counts[te[tile]], "tile end within full count");
            for (uint32_t local = ts[tile];
                 local < std::min(ts[tile] + 16u, tend[tile]); local++) {
                const uint32_t pair = sorted_pairs[offsets[te[tile]] + local];
                CHECK(pair / used >= row_first && pair / used < row_end,
                      "tile pair belongs to wave");
                CHECK(seen[pair]++ == 0, "pair covered once");
            }
        }
    }
    for (uint32_t pair = 0; pair < pair_count; pair++)
        CHECK(seen[pair] == (selected[pair] >= 0 ? 1 : 0),
              "complete live-pair coverage");

    /* Repack base/stride proof: wave-local launches must reproduce a single
     * full transposition exactly. */
    constexpr uint32_t blocks = 16;
    std::vector<uint32_t> src((uint64_t)pair_count * blocks);
    for (uint32_t row = 0; row < pair_count; row++)
        for (uint32_t block = 0; block < blocks; block++)
            src[(uint64_t)row * blocks + block] = row * 131u + block;
    uint32_t *d_src, *d_full, *d_wave;
    const size_t repack_bytes = src.size() * sizeof(uint32_t);
    hip_check(hipMalloc(&d_src, repack_bytes), "repack source");
    hip_check(hipMalloc(&d_full, repack_bytes), "full repack");
    hip_check(hipMalloc(&d_wave, repack_bytes), "wave repack");
    hip_check(hipMemcpy(d_src, src.data(), repack_bytes, hipMemcpyHostToDevice), "copy repack source");
    hip_check(hipMemset(d_full, 0, repack_bytes), "clear full repack");
    hip_check(hipMemset(d_wave, 0, repack_bytes), "clear wave repack");
    repack_rows<<<dim3((pair_count + 127u) / 128u, blocks), 128>>>(
        d_full, d_src, pair_count, blocks, 0, pair_count);
    for (uint32_t wave = 0; wave < nwaves; wave++) {
        const uint32_t row_first = (tokens * wave / nwaves) * used;
        const uint32_t row_end = (tokens * (wave + 1u) / nwaves) * used;
        repack_rows<<<dim3((row_end - row_first + 127u) / 128u, blocks), 128>>>(
            d_wave, d_src, pair_count, blocks, row_first, row_end - row_first);
    }
    hip_check(hipDeviceSynchronize(), "repack kernels");
    std::vector<uint32_t> full(src.size()), waved(src.size());
    hip_check(hipMemcpy(full.data(), d_full, repack_bytes, hipMemcpyDeviceToHost), "copy full repack");
    hip_check(hipMemcpy(waved.data(), d_wave, repack_bytes, hipMemcpyDeviceToHost), "copy wave repack");
    CHECK(full == waved, "wave repack matches full repack");

    (void)hipFree(d_wave); (void)hipFree(d_full); (void)hipFree(d_src);
    (void)hipFree(d_tend); (void)hipFree(d_ts); (void)hipFree(d_te);
    (void)hipFree(d_tile_total); (void)hipFree(d_tile_offsets);
    (void)hipFree(d_wave_counts); (void)hipFree(d_wave_starts);
    (void)hipFree(d_pairs); (void)hipFree(d_offsets); (void)hipFree(d_counts);
    std::printf("test_rocm_moe_wave_plan: PASS waves=%u live_pairs=%zu\n",
                nwaves, sorted_pairs.size());
    return 0;
}

int main() {
    if (run_case(2) != 0) return 1;
    if (run_case(4) != 0) return 1;
    return 0;
}
