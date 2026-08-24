/* Atomic residency oracle for the default-off Q4_K K-shard installer. */
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

#define CHECK(cond, msg) do {                                                \
    if (!(cond)) {                                                           \
        fprintf(stderr, "FAIL: %s (line %d)\n", (msg), __LINE__);           \
        return 1;                                                            \
    }                                                                        \
} while (0)

enum {
    Q4_BLOCK = 144,
    N_EXPERT = 8,
    GATE_ROWS = 2048,
    DOWN_ROWS = 4096,
    GATE_ROW_BYTES = 16 * Q4_BLOCK,
    DOWN_ROW_BYTES = 8 * Q4_BLOCK,
};

static uint64_t align_up(uint64_t v, uint64_t a) {
    return (v + a - 1u) / a * a;
}

static int pwrite_full(int fd, const void *src, uint64_t bytes) {
    uint64_t done = 0;
    while (done < bytes) {
        const ssize_t n = pwrite(fd, (const char *)src + done,
                                 (size_t)(bytes - done), (off_t)done);
        if (n <= 0) return 0;
        done += (uint64_t)n;
    }
    return 1;
}

static int exchange_stub(void *, uint32_t, uint32_t, uint64_t) {
    return 1;
}

static int resolve_one(const void *model, uint64_t offset,
                       uint32_t source_rows, uint64_t source_row_bytes,
                       uint32_t row_base, uint32_t row_count,
                       uint64_t column_base, uint64_t column_count,
                       ds4_gpu_q4k_packed_slice_kind kind,
                       uint64_t want_expert_bytes) {
    const void *ptr = NULL;
    uint64_t bytes = 0, expert_bytes = 0, row_bytes = 0;
    if (!ds4_gpu_q4k_packed_slice_resolve(
            model, offset, N_EXPERT, source_rows, source_row_bytes,
            row_base, row_count, column_base, column_count, kind,
            &ptr, &bytes, &expert_bytes, &row_bytes)) return 0;
    return ptr != NULL && expert_bytes == want_expert_bytes &&
           bytes == (uint64_t)N_EXPERT * want_expert_bytes &&
           row_bytes == column_count &&
           (const char *)ptr + 7u * expert_bytes < (const char *)ptr + bytes;
}

int main(void) {
    const uint64_t gate_expert_bytes =
        (uint64_t)GATE_ROWS * GATE_ROW_BYTES;
    const uint64_t down_expert_bytes =
        (uint64_t)DOWN_ROWS * DOWN_ROW_BYTES;
    const uint64_t tensor_gate_bytes = N_EXPERT * gate_expert_bytes;
    const uint64_t tensor_down_bytes = N_EXPERT * down_expert_bytes;
    const uint64_t gate_offset = 128u * 1024u;
    const uint64_t up_offset =
        align_up(gate_offset + tensor_gate_bytes, 4096u);
    const uint64_t down_offset =
        align_up(up_offset + tensor_gate_bytes, 4096u);
    const uint64_t model_bytes =
        align_up(down_offset + tensor_down_bytes, 4096u);
    unsigned char *model = (unsigned char *)malloc((size_t)model_bytes);
    CHECK(model, "model allocation");
    for (uint64_t i = 0; i < model_bytes; ++i) {
        model[i] = (unsigned char)((i * 73u + (i >> 11u) * 19u + 7u) & 255u);
    }

    char path[] = "/tmp/ds4-q4k-kshard-install-XXXXXX";
    const int fd = mkstemp(path);
    CHECK(fd >= 0 && pwrite_full(fd, model, model_bytes), "model file");
    char bad_path[] = "/tmp/ds4-q4k-kshard-bad-XXXXXX";
    const int bad_fd = mkstemp(bad_path);
    CHECK(bad_fd >= 0 && ftruncate(bad_fd, 4096) == 0, "short model file");

    CHECK(ds4_gpu_init(), "GPU init");
    ds4_gpu_tensor *slab = ds4_gpu_tensor_alloc_rdma_host(4096u);
    CHECK(slab, "TP slab");
    CHECK(ds4_gpu_tp_init(0u, slab, 0u, exchange_stub, NULL), "TP init");
    CHECK(ds4_gpu_tp_expert_shard_active() == 1, "ordinary remap active");

    const ds4_gpu_q4k_kshard_layer layer = {
        gate_offset, up_offset, down_offset, N_EXPERT,
        4096u, 2048u, 4096u,
        GATE_ROW_BYTES, GATE_ROW_BYTES, DOWN_ROW_BYTES
    };
    const uint64_t dense_offset = 0u, dense_size = 2048u;
    ds4_gpu_q4k_kshard_windows windows;

    CHECK(unsetenv("DS4_ROCM_Q4K_KSHARD_RESEARCH") == 0, "research unset");
    CHECK(!ds4_gpu_q4k_kshard_install(
              model, model_bytes, fd, 0u, &dense_offset, &dense_size, 1u,
              dense_size, &layer, 1u), "default-off refusal");
    CHECK(ds4_gpu_tp_expert_shard_active() == 1 &&
              ds4_gpu_q4k_packed_slice_bytes() == 0u &&
              !ds4_gpu_q4k_kshard_windows_get(&windows),
          "default-off state unchanged");

    CHECK(setenv("DS4_ROCM_Q4K_KSHARD_RESEARCH", "1", 1) == 0,
          "research enabled");
    CHECK(!ds4_gpu_q4k_kshard_install(
              model, model_bytes, fd, 0u, NULL, NULL, 0u, 0u,
              &layer, 1u), "empty dense mapping refused");
    CHECK(!ds4_gpu_q4k_kshard_install(
              model, model_bytes, fd, 0u, &gate_offset,
              &tensor_gate_bytes, 1u, tensor_gate_bytes, &layer, 1u),
          "routed dense overlap refused");
    CHECK(ds4_gpu_tp_expert_shard_active() == 1 &&
              ds4_gpu_q4k_packed_slice_bytes() == 0u,
          "overlap failure atomic");

    CHECK(!ds4_gpu_q4k_kshard_install(
              model, model_bytes, bad_fd, 0u, &dense_offset, &dense_size, 1u,
              dense_size, &layer, 1u), "short fd load failure");
    CHECK(ds4_gpu_tp_expert_shard_active() == 1 &&
              ds4_gpu_q4k_packed_slice_bytes() == 0u &&
              !ds4_gpu_q4k_kshard_windows_get(&windows),
          "load failure atomic");
    CHECK(!ds4_gpu_cache_model_range(model, model_bytes, gate_offset,
                                     tensor_gate_bytes,
                                     "kshard-failed-linear"),
          "failed install leaves routed tensors fail closed");

    CHECK(ds4_gpu_q4k_kshard_install(
              model, model_bytes, fd, 0u, &dense_offset, &dense_size, 1u,
              dense_size, &layer, 1u), "rank0 install");
    CHECK(ds4_gpu_q4k_kshard_windows_get(&windows), "rank0 windows");
    CHECK(windows.rank == 0u && windows.row_base == 0u &&
              windows.row_count == 1024u &&
              windows.down_column_byte_base == 0u &&
              windows.down_column_byte_count == 4u * Q4_BLOCK &&
              windows.expert_mid_dim == 1024u && windows.n_expert == N_EXPERT,
          "rank0 window geometry");
    const uint64_t packed_gate_expert =
        (uint64_t)1024u * GATE_ROW_BYTES;
    const uint64_t packed_down_expert =
        (uint64_t)DOWN_ROWS * 4u * Q4_BLOCK;
    const uint64_t packed_total =
        (uint64_t)N_EXPERT *
        (2u * packed_gate_expert + packed_down_expert);
    CHECK(ds4_gpu_q4k_packed_slice_bytes() == packed_total,
          "half-K byte accounting");
    CHECK(ds4_gpu_tp_expert_shard_active() == 0, "remap suspended");
    CHECK(resolve_one(model, gate_offset, GATE_ROWS, GATE_ROW_BYTES,
                      0u, 1024u, 0u, GATE_ROW_BYTES,
                      DS4_GPU_Q4K_PACKED_ROW_RANGE, packed_gate_expert) &&
              resolve_one(model, up_offset, GATE_ROWS, GATE_ROW_BYTES,
                          0u, 1024u, 0u, GATE_ROW_BYTES,
                          DS4_GPU_Q4K_PACKED_ROW_RANGE, packed_gate_expert) &&
              resolve_one(model, down_offset, DOWN_ROWS, DOWN_ROW_BYTES,
                          0u, DOWN_ROWS, 0u, 4u * Q4_BLOCK,
                          DS4_GPU_Q4K_PACKED_K_RANGE, packed_down_expert),
          "rank0 all-expert addressability");
    CHECK(!ds4_gpu_cache_model_range(model, model_bytes, gate_offset,
                                     tensor_gate_bytes, "kshard-linear"),
          "linear full tensor refused");
    CHECK(ds4_gpu_q4k_kshard_install(
              model, model_bytes, fd, 0u, &dense_offset, &dense_size, 1u,
              dense_size, &layer, 1u) &&
              ds4_gpu_q4k_packed_slice_bytes() == packed_total,
          "idempotent install");
    CHECK(!ds4_gpu_q4k_kshard_install(
              model, model_bytes, fd, 1u, &dense_offset, &dense_size, 1u,
              dense_size, &layer, 1u), "conflicting install refused");
    CHECK(ds4_gpu_q4k_kshard_windows_get(&windows) && windows.rank == 0u,
          "conflict preserves install");

    ds4_gpu_q4k_kshard_release();
    CHECK(ds4_gpu_tp_expert_shard_active() == 1 &&
              ds4_gpu_q4k_packed_slice_bytes() == 0u &&
              !ds4_gpu_q4k_kshard_windows_get(&windows),
          "release restores remap");
    CHECK(!ds4_gpu_cache_model_range(model, model_bytes, gate_offset,
                                     tensor_gate_bytes,
                                     "kshard-released-linear"),
          "release leaves routed tensors fail closed");
    CHECK(ds4_gpu_q4k_kshard_install(
              model, model_bytes, fd, 1u, &dense_offset, &dense_size, 1u,
              dense_size, &layer, 1u), "rank1 install");
    CHECK(ds4_gpu_q4k_kshard_windows_get(&windows) &&
              windows.rank == 1u && windows.row_base == 1024u &&
              windows.down_column_byte_base == 4u * Q4_BLOCK,
          "rank1 window geometry");
    CHECK(resolve_one(model, gate_offset, GATE_ROWS, GATE_ROW_BYTES,
                      1024u, 1024u, 0u, GATE_ROW_BYTES,
                      DS4_GPU_Q4K_PACKED_ROW_RANGE, packed_gate_expert) &&
              resolve_one(model, up_offset, GATE_ROWS, GATE_ROW_BYTES,
                          1024u, 1024u, 0u, GATE_ROW_BYTES,
                          DS4_GPU_Q4K_PACKED_ROW_RANGE,
                          packed_gate_expert) &&
              resolve_one(model, down_offset, DOWN_ROWS, DOWN_ROW_BYTES,
                          0u, DOWN_ROWS, 4u * Q4_BLOCK, 4u * Q4_BLOCK,
                          DS4_GPU_Q4K_PACKED_K_RANGE, packed_down_expert),
          "rank1 all-expert addressability");

    ds4_gpu_cleanup();
    CHECK(!ds4_gpu_q4k_kshard_windows_get(&windows) &&
              ds4_gpu_q4k_packed_slice_bytes() == 0u &&
              ds4_gpu_tp_expert_shard_active() == 1,
          "cleanup restores remap and clears install");
    ds4_gpu_tp_shutdown();
    ds4_gpu_tensor_free(slab);

    /* Production starts with rank-owned routed images.  The K-shard
     * transition evacuates those allocations and replaces them with fresh
     * packed slices without increasing persistent routed-weight residency.
     * Release is deliberately fail-closed until engine cleanup: the old
     * allocation addresses must never be re-published. */
    CHECK(ds4_gpu_init(), "borrowed GPU re-init");
    slab = ds4_gpu_tensor_alloc_rdma_host(4096u);
    CHECK(slab, "borrowed TP slab");
    CHECK(ds4_gpu_tp_init(0u, slab, 0u, exchange_stub, NULL),
          "borrowed TP init");
    const uint64_t half_gate = tensor_gate_bytes / 2u;
    const uint64_t half_down = tensor_down_bytes / 2u;
    const uint64_t resident_offsets[] = {
        dense_offset, gate_offset, up_offset, down_offset,
    };
    const uint64_t resident_sizes[] = {
        dense_size, half_gate, half_gate, half_down,
    };
    CHECK(ds4_gpu_set_model_fd_for_map(fd, model), "borrowed model fd");
    CHECK(ds4_gpu_set_model_map_spans(
              model, model_bytes, resident_offsets, resident_sizes, 4u,
              half_gate), "rank-owned resident images");
    size_t free_before = 0, total_before = 0;
    size_t free_after = 0, total_after = 0;
    CHECK(hipMemGetInfo(&free_before, &total_before) == hipSuccess,
          "borrowed memory before");
    CHECK(ds4_gpu_q4k_kshard_install(
              model, model_bytes, fd, 0u, &dense_offset, &dense_size, 1u,
              dense_size, &layer, 1u), "borrowed rank0 install");
    CHECK(hipMemGetInfo(&free_after, &total_after) == hipSuccess,
          "borrowed memory after");
    CHECK(free_after + 1048576u >= free_before,
          "borrowed transition adds no routed allocation");
    CHECK(resolve_one(model, gate_offset, GATE_ROWS, GATE_ROW_BYTES,
                      0u, 1024u, 0u, GATE_ROW_BYTES,
                      DS4_GPU_Q4K_PACKED_ROW_RANGE, packed_gate_expert),
          "borrowed packed addressability");
    ds4_gpu_q4k_kshard_release();
    CHECK(!ds4_gpu_cache_model_range(model, model_bytes, gate_offset,
                                     half_gate, "evacuated-release-linear"),
          "evacuated release remains fail closed");
    ds4_gpu_cleanup();
    ds4_gpu_tp_shutdown();
    ds4_gpu_tensor_free(slab);

    printf("q4k_kshard_install experts=%u packed_bytes=%llu "
           "rank0=pass rank1=pass remap_restore=pass failure_atomic=pass "
           "release_atomic=pass linear_fail_closed=pass idempotent=pass "
           "zero_net_evacuation=pass cleanup_required=pass\n",
           N_EXPERT, (unsigned long long)packed_total);
    close(bad_fd);
    close(fd);
    unlink(bad_path);
    unlink(path);
    free(model);
    return 0;
}
