/* Fail-closed byte oracle for the Q4_K routed-expert packed-slice registry.
 * It covers both contiguous output-row halves and a block-aligned K half
 * selected from every down-projection row through the production fd loader. */

#include "ds4_gpu.h"

#include <hip/hip_runtime.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

typedef int (*ds4_tp_devcopy_fn)(void *, const void *, uint64_t);
extern "C" void ds4_tp_set_devcopy(ds4_tp_devcopy_fn) {}

enum {
    Q4_K_BLOCK_BYTES = 144,
    N_EXPERT = 8,
    SOURCE_ROWS = 512,
    ROW_HALF = 256,
    GATE_BLOCKS = 4,
    DOWN_BLOCKS = 8,
};

#define CHECK(cond, msg) do {                                                \
    if (!(cond)) {                                                           \
        fprintf(stderr, "FAIL: %s (line %d)\n", (msg), __LINE__);          \
        return 1;                                                            \
    }                                                                        \
} while (0)

static uint64_t align_up(uint64_t v, uint64_t a) {
    return (v + a - 1u) / a * a;
}

static int pwrite_full(int fd, const void *src, uint64_t bytes) {
    uint64_t done = 0;
    while (done < bytes) {
        ssize_t n = pwrite(fd, (const char *)src + done,
                           (size_t)(bytes - done), (off_t)done);
        if (n <= 0) return 0;
        done += (uint64_t)n;
    }
    return 1;
}

static uint64_t fnv1a64(const void *data, uint64_t bytes) {
    const unsigned char *p = (const unsigned char *)data;
    uint64_t h = UINT64_C(1469598103934665603);
    for (uint64_t i = 0; i < bytes; i++) {
        h ^= p[i];
        h *= UINT64_C(1099511628211);
    }
    return h;
}

static void pack_expected(
        unsigned char *dst, const unsigned char *model,
        uint64_t tensor_offset, uint64_t source_row_bytes,
        uint32_t row_base, uint32_t row_count,
        uint64_t column_byte_base, uint64_t column_byte_count) {
    const uint64_t source_expert_bytes =
        (uint64_t)SOURCE_ROWS * source_row_bytes;
    const uint64_t packed_expert_bytes =
        (uint64_t)row_count * column_byte_count;
    for (uint32_t expert = 0; expert < N_EXPERT; expert++) {
        const unsigned char *src = model + tensor_offset +
            (uint64_t)expert * source_expert_bytes +
            (uint64_t)row_base * source_row_bytes + column_byte_base;
        unsigned char *out = dst +
            (uint64_t)expert * packed_expert_bytes;
        for (uint32_t row = 0; row < row_count; row++) {
            memcpy(out + (uint64_t)row * column_byte_count,
                   src + (uint64_t)row * source_row_bytes,
                   (size_t)column_byte_count);
        }
    }
}

int main(void) {
    const uint64_t gate_row_bytes =
        (uint64_t)GATE_BLOCKS * Q4_K_BLOCK_BYTES;
    const uint64_t down_row_bytes =
        (uint64_t)DOWN_BLOCKS * Q4_K_BLOCK_BYTES;
    const uint64_t gate_bytes =
        (uint64_t)N_EXPERT * SOURCE_ROWS * gate_row_bytes;
    const uint64_t down_bytes =
        (uint64_t)N_EXPERT * SOURCE_ROWS * down_row_bytes;
    const uint64_t gate_offset = 4096u;
    const uint64_t down_offset = align_up(gate_offset + gate_bytes, 4096u);
    const uint64_t up_offset = align_up(down_offset + down_bytes, 4096u);
    const uint64_t model_bytes = align_up(up_offset + gate_bytes, 4096u);
    unsigned char *model = (unsigned char *)malloc((size_t)model_bytes);
    CHECK(model != NULL, "model allocation");
    for (uint64_t i = 0; i < model_bytes; i++) {
        model[i] = (unsigned char)((i * 131u + (i >> 7u) * 17u + 29u) & 255u);
    }

    char path[] = "/tmp/ds4-q4k-packed-slice-XXXXXX";
    const int fd = mkstemp(path);
    CHECK(fd >= 0, "mkstemp");
    CHECK(pwrite_full(fd, model, model_bytes), "model file write");
    CHECK(ds4_gpu_init(), "GPU init");
    CHECK(ds4_gpu_set_model_map(model, model_bytes), "set model map");
    CHECK(ds4_gpu_set_model_fd_for_map(fd, model), "set model fd");

    const uint32_t gate_row_base = ROW_HALF;
    const uint64_t down_column_base = down_row_bytes / 2u;
    unsigned char alternate_map[64] = {};
    CHECK(ds4_gpu_cache_model_range(model, model_bytes,
                                     gate_offset, gate_bytes,
                                     "packed-predeclared-linear"),
          "predeclare linear residency");
    CHECK(!ds4_gpu_q4k_packed_slice_declare(
              model, model_bytes, gate_offset, N_EXPERT, SOURCE_ROWS,
              gate_row_bytes, gate_row_base, ROW_HALF, 0u, gate_row_bytes,
              DS4_GPU_Q4K_PACKED_ROW_RANGE),
          "declare after linear residency rejected");
    CHECK(ds4_gpu_set_model_map(alternate_map, sizeof(alternate_map)),
          "release predeclared linear residency");
    CHECK(ds4_gpu_set_model_map(model, model_bytes),
          "restore model after linear residency test");
    CHECK(ds4_gpu_set_model_fd_for_map(fd, model),
          "restore model fd after residency test");

    CHECK(ds4_gpu_q4k_packed_slice_declare(
              model, model_bytes, gate_offset, N_EXPERT, SOURCE_ROWS,
              gate_row_bytes, gate_row_base, ROW_HALF, 0u, gate_row_bytes,
              DS4_GPU_Q4K_PACKED_ROW_RANGE),
          "declare row slice");
    CHECK(ds4_gpu_q4k_packed_slice_declare(
              model, model_bytes, down_offset, N_EXPERT, SOURCE_ROWS,
              down_row_bytes, 0u, SOURCE_ROWS, down_column_base,
              down_row_bytes / 2u, DS4_GPU_Q4K_PACKED_K_RANGE),
          "declare K slice");
    CHECK(ds4_gpu_q4k_packed_slice_declare(
              model, model_bytes, up_offset, N_EXPERT, SOURCE_ROWS,
              gate_row_bytes, gate_row_base, ROW_HALF, 0u, gate_row_bytes,
              DS4_GPU_Q4K_PACKED_ROW_RANGE),
          "declare unloaded peer row slice");
    CHECK(ds4_gpu_q4k_packed_slice_declare(
              model, model_bytes, down_offset, N_EXPERT, SOURCE_ROWS,
              down_row_bytes, 0u, SOURCE_ROWS, down_column_base,
              down_row_bytes / 2u, DS4_GPU_Q4K_PACKED_K_RANGE),
          "idempotent declaration");
    CHECK(!ds4_gpu_q4k_packed_slice_declare(
              model, model_bytes, down_offset, N_EXPERT - 1u, SOURCE_ROWS,
              down_row_bytes, 0u, SOURCE_ROWS, down_column_base,
              down_row_bytes / 2u, DS4_GPU_Q4K_PACKED_K_RANGE),
          "conflicting declaration rejected");
    CHECK(!ds4_gpu_q4k_packed_slice_declare(
              model, model_bytes, down_offset, N_EXPERT, SOURCE_ROWS,
              down_row_bytes, 0u, SOURCE_ROWS,
              down_column_base + 1u, down_row_bytes / 2u,
              DS4_GPU_Q4K_PACKED_K_RANGE),
          "unaligned K slice rejection");
    CHECK(!ds4_gpu_q4k_packed_slice_declare(
              model, model_bytes, gate_offset, N_EXPERT, SOURCE_ROWS,
              gate_row_bytes, gate_row_base, ROW_HALF / 2u,
              0u, gate_row_bytes, DS4_GPU_Q4K_PACKED_ROW_RANGE),
          "partial row range rejected");
    CHECK(!ds4_gpu_q4k_packed_slice_declare(
              model, model_bytes, gate_offset, N_EXPERT, SOURCE_ROWS,
              gate_row_bytes, gate_row_base, ROW_HALF,
              0u, gate_row_bytes / 2u,
              DS4_GPU_Q4K_PACKED_ROW_RANGE),
          "partial row columns rejected");
    CHECK(!ds4_gpu_q4k_packed_slice_declare(
              model, model_bytes, down_offset, N_EXPERT, SOURCE_ROWS,
              down_row_bytes, 0u, SOURCE_ROWS - ROW_HALF,
              down_column_base, down_row_bytes / 2u,
              DS4_GPU_Q4K_PACKED_K_RANGE),
          "partial K rows rejected");

    const void *resolved = (const void *)(uintptr_t)1u;
    uint64_t resolved_bytes = 1u;
    uint64_t resolved_expert_bytes = 1u;
    uint64_t resolved_row_bytes = 1u;
    CHECK(!ds4_gpu_q4k_packed_slice_resolve(
              model, down_offset, N_EXPERT, SOURCE_ROWS, down_row_bytes,
              0u, SOURCE_ROWS, down_column_base, down_row_bytes / 2u,
              DS4_GPU_Q4K_PACKED_K_RANGE, &resolved, &resolved_bytes,
              &resolved_expert_bytes, &resolved_row_bytes),
          "unloaded descriptor does not resolve");
    CHECK(resolved == NULL && resolved_bytes == 0u &&
              resolved_expert_bytes == 0u && resolved_row_bytes == 0u,
          "failed resolver clears outputs");

    CHECK(ds4_gpu_q4k_packed_slice_load(
              model, gate_offset, gate_row_base, ROW_HALF,
              0u, gate_row_bytes),
          "load row slice");
    CHECK(ds4_gpu_q4k_packed_slice_load(
              model, down_offset, 0u, SOURCE_ROWS,
              down_column_base, down_row_bytes / 2u),
          "load K slice");

    /* Model the executor's fail-closed partial-residency state without
     * perturbing a live full-trunk allocation: gate and down are loaded while
     * up remains declared-only.  The streaming cache must refuse to couple
     * this mixed-ownership triple. */
    const void *gate_resolved = NULL;
    const void *up_resolved = NULL;
    const void *down_resolved = NULL;
    uint64_t ignored_bytes = 0u, ignored_expert = 0u, ignored_row = 0u;
    CHECK(ds4_gpu_q4k_packed_slice_resolve(
              model, gate_offset, N_EXPERT, SOURCE_ROWS, gate_row_bytes,
              gate_row_base, ROW_HALF, 0u, gate_row_bytes,
              DS4_GPU_Q4K_PACKED_ROW_RANGE, &gate_resolved, &ignored_bytes,
              &ignored_expert, &ignored_row) && gate_resolved,
          "partial triple resolves loaded gate");
    CHECK(!ds4_gpu_q4k_packed_slice_resolve(
              model, up_offset, N_EXPERT, SOURCE_ROWS, gate_row_bytes,
              gate_row_base, ROW_HALF, 0u, gate_row_bytes,
              DS4_GPU_Q4K_PACKED_ROW_RANGE, &up_resolved, &ignored_bytes,
              &ignored_expert, &ignored_row) && !up_resolved,
          "partial triple refuses unloaded up");
    CHECK(ds4_gpu_q4k_packed_slice_resolve(
              model, down_offset, N_EXPERT, SOURCE_ROWS, down_row_bytes,
              0u, SOURCE_ROWS, down_column_base, down_row_bytes / 2u,
              DS4_GPU_Q4K_PACKED_K_RANGE, &down_resolved, &ignored_bytes,
              &ignored_expert, &ignored_row) && down_resolved,
          "partial triple resolves loaded down");
    ds4_gpu_q4k_window_cache_config partial = {};
    partial.model_map = model;
    partial.gate_offset = gate_offset;
    partial.up_offset = up_offset;
    partial.down_offset = down_offset;
    partial.n_expert = N_EXPERT;
    partial.gate_row_base = gate_row_base;
    partial.gate_row_count = ROW_HALF;
    partial.gate_column_byte_count = gate_row_bytes;
    partial.down_row_count = SOURCE_ROWS;
    partial.down_column_byte_base = down_column_base;
    partial.down_column_byte_count = down_row_bytes / 2u;
    partial.slots = N_EXPERT;
    CHECK(ds4_gpu_q4k_window_cache_create(&partial) == NULL,
          "partial loaded triple refuses streaming-cache ownership");

    const uint64_t gate_packed_bytes =
        (uint64_t)N_EXPERT * ROW_HALF * gate_row_bytes;
    const uint64_t down_packed_bytes =
        (uint64_t)N_EXPERT * SOURCE_ROWS * (down_row_bytes / 2u);
    unsigned char *expected = (unsigned char *)malloc((size_t)down_packed_bytes);
    unsigned char *got = (unsigned char *)malloc((size_t)down_packed_bytes);
    CHECK(expected && got, "readback buffers");

    CHECK(ds4_gpu_q4k_packed_slice_resolve(
              model, down_offset, N_EXPERT, SOURCE_ROWS, down_row_bytes,
              0u, SOURCE_ROWS, down_column_base, down_row_bytes / 2u,
              DS4_GPU_Q4K_PACKED_K_RANGE, &resolved, &resolved_bytes,
              &resolved_expert_bytes, &resolved_row_bytes),
          "resolve exact loaded K descriptor");
    CHECK(resolved != NULL && resolved_bytes == down_packed_bytes &&
              resolved_expert_bytes ==
                  (uint64_t)SOURCE_ROWS * (down_row_bytes / 2u) &&
              resolved_row_bytes == down_row_bytes / 2u,
          "resolved K descriptor geometry");
    CHECK(hipMemcpy(got, resolved, (size_t)down_packed_bytes,
                    hipMemcpyDeviceToHost) == hipSuccess,
          "resolved pointer copy");
    pack_expected(expected, model, down_offset, down_row_bytes,
                  0u, SOURCE_ROWS, down_column_base, down_row_bytes / 2u);
    CHECK(memcmp(expected, got, (size_t)down_packed_bytes) == 0,
          "resolved pointer bytes");
    CHECK(!ds4_gpu_q4k_packed_slice_resolve(
              model, down_offset, N_EXPERT, SOURCE_ROWS - 1u,
              down_row_bytes, 0u, SOURCE_ROWS, down_column_base,
              down_row_bytes / 2u, DS4_GPU_Q4K_PACKED_K_RANGE,
              &resolved, &resolved_bytes, &resolved_expert_bytes,
              &resolved_row_bytes),
          "mismatched descriptor geometry rejected");
    CHECK(resolved == NULL && resolved_bytes == 0u,
          "mismatched resolver clears outputs");

    pack_expected(expected, model, gate_offset, gate_row_bytes,
                  gate_row_base, ROW_HALF, 0u, gate_row_bytes);
    CHECK(ds4_gpu_q4k_packed_slice_readback(
              model, gate_offset, gate_row_base, ROW_HALF,
              0u, gate_row_bytes, got, gate_packed_bytes),
          "row readback");
    CHECK(memcmp(expected, got, (size_t)gate_packed_bytes) == 0,
          "row slice bytes");
    const uint64_t gate_hash = fnv1a64(got, gate_packed_bytes);
    CHECK(gate_hash == UINT64_C(0xeee496fd886b6b83),
          "row slice frozen FNV");

    pack_expected(expected, model, down_offset, down_row_bytes,
                  0u, SOURCE_ROWS, down_column_base, down_row_bytes / 2u);
    CHECK(ds4_gpu_q4k_packed_slice_readback(
              model, down_offset, 0u, SOURCE_ROWS,
              down_column_base, down_row_bytes / 2u,
              got, down_packed_bytes),
          "K readback");
    CHECK(memcmp(expected, got, (size_t)down_packed_bytes) == 0,
          "K slice bytes");
    const uint64_t down_hash = fnv1a64(got, down_packed_bytes);
    CHECK(down_hash == UINT64_C(0x662be17fcaa0e383),
          "K slice frozen FNV");
    CHECK(ds4_gpu_q4k_packed_slice_bytes() ==
              gate_packed_bytes + down_packed_bytes,
          "packed byte accounting");

    CHECK(ds4_gpu_set_model_map(alternate_map, sizeof(alternate_map)),
          "alternate map transition");
    CHECK(ds4_gpu_q4k_packed_slice_readback(
              model, down_offset, 0u, SOURCE_ROWS,
              down_column_base, down_row_bytes / 2u,
              got, down_packed_bytes),
          "packed lifetime after map transition");
    CHECK(memcmp(expected, got, (size_t)down_packed_bytes) == 0,
          "packed lifetime bytes");
    CHECK(ds4_gpu_set_model_map(model, model_bytes), "restore model map");

    CHECK(ds4_gpu_cache_model_range(model, model_bytes,
                                     0u, 1024u,
                                     "packed-disjoint-linear"),
          "disjoint linear range remains available");
    CHECK(!ds4_gpu_cache_model_range(model, model_bytes,
                                     gate_offset, gate_bytes,
                                     "packed-negative-linear"),
          "linear range rejected");
    const uint64_t span_offset = down_offset;
    const uint64_t span_bytes = down_bytes;
    CHECK(!ds4_gpu_set_model_map_spans(model, model_bytes,
                                       &span_offset, &span_bytes, 1u,
                                       span_bytes),
          "intersecting span rejected");
    CHECK(!ds4_gpu_set_model_map_range(model, model_bytes,
                                       down_offset, down_bytes, down_bytes),
          "intersecting range rejected");

    ds4_gpu_cleanup();

    resolved = (const void *)(uintptr_t)1u;
    resolved_bytes = 1u;
    resolved_expert_bytes = 1u;
    resolved_row_bytes = 1u;
    CHECK(!ds4_gpu_q4k_packed_slice_resolve(
              model, down_offset, N_EXPERT, SOURCE_ROWS, down_row_bytes,
              0u, SOURCE_ROWS, down_column_base, down_row_bytes / 2u,
              DS4_GPU_Q4K_PACKED_K_RANGE, &resolved, &resolved_bytes,
              &resolved_expert_bytes, &resolved_row_bytes),
          "cleanup invalidates descriptor resolver");
    CHECK(resolved == NULL && resolved_bytes == 0u &&
              resolved_expert_bytes == 0u && resolved_row_bytes == 0u,
          "cleanup resolver clears outputs");

    CHECK(ds4_gpu_init(), "GPU reinit for no-fd path");
    CHECK(ds4_gpu_set_model_map(model, model_bytes),
          "set no-fd model map");
    CHECK(ds4_gpu_set_model_fd_for_map(-1, model),
          "disable model fd");
    CHECK(ds4_gpu_q4k_packed_slice_declare(
              model, model_bytes, down_offset, N_EXPERT, SOURCE_ROWS,
              down_row_bytes, 0u, SOURCE_ROWS, down_column_base,
              down_row_bytes / 2u, DS4_GPU_Q4K_PACKED_K_RANGE),
          "declare no-fd K slice");
    CHECK(ds4_gpu_q4k_packed_slice_load(
              model, down_offset, 0u, SOURCE_ROWS,
              down_column_base, down_row_bytes / 2u),
          "load no-fd K slice");
    CHECK(ds4_gpu_q4k_packed_slice_readback(
              model, down_offset, 0u, SOURCE_ROWS,
              down_column_base, down_row_bytes / 2u,
              got, down_packed_bytes),
          "no-fd K readback");
    CHECK(memcmp(expected, got, (size_t)down_packed_bytes) == 0,
          "no-fd K slice bytes");
    CHECK(fnv1a64(got, down_packed_bytes) == down_hash,
          "no-fd K slice FNV");
    ds4_gpu_cleanup();

    printf("packed_q4k_slice_registry row_fnv64=%016llx "
           "k_fnv64=%016llx bytes=%llu fd_staging=1 no_fd=1 "
           "ring_wrap=1 declare_after_cache=blocked "
           "linear_fail_closed=1 disjoint_linear=1 lifetime=1 "
           "exact_resolver=1\n",
           (unsigned long long)gate_hash,
           (unsigned long long)down_hash,
           (unsigned long long)(gate_packed_bytes + down_packed_bytes));

    free(got);
    free(expected);
    close(fd);
    unlink(path);
    free(model);
    return 0;
}
