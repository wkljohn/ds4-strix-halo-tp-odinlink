/* H7.5: rank-1 K-shard arithmetic through installer-owned packed windows. */
#include "ds4_gpu.h"
#include "ds4_gpu_mgpu.h"

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
    Q4_K_TYPE = 12,
    QK_K = 256,
    Q4_BLOCK = 144,
    N_EXPERT = 8,
    N_USED = 6,
    IN_DIM = 4096,
    FULL_MID = 2048,
    HALF_MID = 1024,
    OUT_DIM = 4096,
    GATE_ROW_BYTES = 16 * Q4_BLOCK,
    DOWN_ROW_BYTES = 8 * Q4_BLOCK,
    HALF_DOWN_ROW_BYTES = 4 * Q4_BLOCK,
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

static void pack_q4k_block(unsigned char *dst, uint32_t seed) {
    dst[0] = 0x00;
    dst[1] = 0x28;
    dst[2] = 0x00;
    dst[3] = 0x00;
    for (uint32_t i = 0; i < 4; ++i) dst[4 + i] = 1;
    for (uint32_t i = 4; i < 8; ++i) dst[4 + i] = 0;
    for (uint32_t i = 8; i < 12; ++i) dst[4 + i] = 1;
    for (uint32_t i = 0; i < 128; ++i) {
        const uint8_t lo = (uint8_t)((seed + 3u * i + 1u) & 15u);
        const uint8_t hi = (uint8_t)((seed + 5u * i + 7u) & 15u);
        dst[16 + i] = (uint8_t)(lo | (hi << 4));
    }
}

static void pack_q4k_table(unsigned char *dst, uint32_t rows,
                           uint32_t blocks_per_row, uint32_t salt) {
    for (uint32_t e = 0; e < N_EXPERT; ++e) {
        for (uint32_t row = 0; row < rows; ++row) {
            for (uint32_t block = 0; block < blocks_per_row; ++block) {
                const uint64_t index =
                    ((uint64_t)e * rows + row) * blocks_per_row + block;
                pack_q4k_block(dst + index * Q4_BLOCK,
                               salt + 17u * e + 13u * row + 7u * block);
            }
        }
    }
}

static void pack_row_half(unsigned char *dst, const unsigned char *src,
                          uint32_t rank, int interleaved) {
    for (uint32_t e = 0; e < N_EXPERT; ++e) {
        for (uint32_t row = 0; row < HALF_MID; ++row) {
            const uint32_t source_row = interleaved ?
                (2u * (row / QK_K) + rank) * QK_K + row % QK_K :
                rank * HALF_MID + row;
            memcpy(dst + ((uint64_t)e * HALF_MID + row) * GATE_ROW_BYTES,
                   src + ((uint64_t)e * FULL_MID + source_row) *
                             GATE_ROW_BYTES,
                   GATE_ROW_BYTES);
        }
    }
}

static void pack_down_half(unsigned char *dst, const unsigned char *src,
                           uint32_t rank, int interleaved) {
    for (uint32_t e = 0; e < N_EXPERT; ++e) {
        for (uint32_t row = 0; row < OUT_DIM; ++row) {
            for (uint32_t block = 0; block < 4u; ++block) {
                const uint32_t source_block = interleaved ?
                    2u * block + rank : 4u * rank + block;
                memcpy(dst + ((uint64_t)e * OUT_DIM + row) *
                                 HALF_DOWN_ROW_BYTES +
                                 (uint64_t)block * Q4_BLOCK,
                       src + ((uint64_t)e * OUT_DIM + row) * DOWN_ROW_BYTES +
                                 (uint64_t)source_block * Q4_BLOCK,
                       Q4_BLOCK);
            }
        }
    }
}

static int exchange_stub(void *, uint32_t, uint32_t, uint64_t) {
    return 1;
}

static int alloc_tensor(ds4_gpu_tensor *tensor, uint64_t bytes) {
    memset(tensor, 0, sizeof(*tensor));
    return ds4_gpu_tensor_alloc_on(tensor, 0, bytes) == 0;
}

static int upload(ds4_gpu_tensor *tensor, const void *src, uint64_t bytes) {
    return ds4_gpu_tensor_write(tensor, 0, src, bytes) != 0;
}

static uint64_t fnv1a64(const void *data, uint64_t bytes) {
    const unsigned char *p = (const unsigned char *)data;
    uint64_t h = UINT64_C(1469598103934665603);
    for (uint64_t i = 0; i < bytes; ++i) {
        h ^= p[i];
        h *= UINT64_C(1099511628211);
    }
    return h;
}

static int run_direct_reference(
        ds4_gpu_tensor *out, ds4_gpu_tensor *gate, ds4_gpu_tensor *up,
        ds4_gpu_tensor *mid, ds4_gpu_tensor *down,
        const void *compact, uint64_t compact_bytes,
        uint64_t gate_bytes, uint64_t down_bytes,
        const ds4_gpu_tensor *selected, const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *x, const ds4_gpu_tensor *shared_addend) {
    return ds4_gpu_set_model_map(compact, compact_bytes) &&
           ds4_gpu_routed_moe_one_tensor(
               out, gate, up, mid, down, compact, compact_bytes,
               0u, gate_bytes, gate_bytes * 2u,
               Q4_K_TYPE, Q4_K_TYPE,
               gate_bytes / N_EXPERT, GATE_ROW_BYTES,
               down_bytes / N_EXPERT, HALF_DOWN_ROW_BYTES,
               IN_DIM, HALF_MID, OUT_DIM,
               selected, weights, N_EXPERT, N_USED, 0.0f,
               x, shared_addend, 0u, true);
}

int main(void) {
    int devices = 0;
    CHECK(hipGetDeviceCount(&devices) == hipSuccess && devices > 0,
          "ROCm device required");
    ds4_gpu_config config = {};
    config.n_gpus = 1;
    config.device_indices[0] = 0;
    CHECK(ds4_gpu_init_multi(&config), "initialize ROCm");
    CHECK(setenv("DS4_ROCM_Q4K_KSHARD_RESEARCH", "1", 1) == 0,
          "research enabled");
    CHECK(unsetenv("DS4_ROCM_Q4K_DECODE_STAGE_MIDQ") == 0, "MIDQ off");
    const char *interleave_env =
        getenv("DS4_ROCM_Q4K_KSHARD_INTERLEAVED");
    const int interleaved = interleave_env &&
        interleave_env[0] == '1' && interleave_env[1] == '\0';

    const uint64_t gate_expert = (uint64_t)FULL_MID * GATE_ROW_BYTES;
    const uint64_t down_expert = (uint64_t)OUT_DIM * DOWN_ROW_BYTES;
    const uint64_t gate_bytes = (uint64_t)N_EXPERT * gate_expert;
    const uint64_t down_bytes = (uint64_t)N_EXPERT * down_expert;
    const uint64_t dense_offset = 0u;
    const uint64_t dense_bytes = 4096u;
    const uint64_t gate_offset = dense_bytes;
    const uint64_t up_offset = align_up(gate_offset + gate_bytes, 4096u);
    const uint64_t down_offset = align_up(up_offset + gate_bytes, 4096u);
    const uint64_t model_bytes =
        align_up(down_offset + down_bytes, 4096u);
    unsigned char *model = (unsigned char *)malloc((size_t)model_bytes);
    CHECK(model, "full model allocation");
    memset(model, 0x5a, (size_t)dense_bytes);
    pack_q4k_table(model + gate_offset, FULL_MID, IN_DIM / QK_K, 11u);
    pack_q4k_table(model + up_offset, FULL_MID, IN_DIM / QK_K, 37u);
    pack_q4k_table(model + down_offset, OUT_DIM, FULL_MID / QK_K, 73u);

    const uint64_t half_gate_expert =
        (uint64_t)HALF_MID * GATE_ROW_BYTES;
    const uint64_t half_down_expert =
        (uint64_t)OUT_DIM * HALF_DOWN_ROW_BYTES;
    const uint64_t compact_gate_bytes =
        (uint64_t)N_EXPERT * half_gate_expert;
    const uint64_t compact_down_bytes =
        (uint64_t)N_EXPERT * half_down_expert;
    const uint64_t compact_bytes =
        compact_gate_bytes * 2u + compact_down_bytes;
    unsigned char *compact =
        (unsigned char *)malloc((size_t)compact_bytes);
    CHECK(compact, "compact reference allocation");
    pack_row_half(compact, model + gate_offset, 1u, interleaved);
    pack_row_half(compact + compact_gate_bytes,
                  model + up_offset, 1u, interleaved);
    pack_down_half(compact + compact_gate_bytes * 2u,
                   model + down_offset, 1u, interleaved);

    char path[] = "/tmp/ds4-q4k-kshard-compose-XXXXXX";
    const int fd = mkstemp(path);
    CHECK(fd >= 0 && pwrite_full(fd, model, model_bytes), "model file");

    ds4_gpu_tensor out = {}, gate = {}, up = {}, mid = {}, down = {};
    ds4_gpu_tensor selected = {}, weights = {}, x = {}, addend = {};
    CHECK(alloc_tensor(&out, OUT_DIM * sizeof(float)), "out");
    CHECK(alloc_tensor(&gate, (uint64_t)N_USED * HALF_MID * sizeof(float)),
          "gate");
    CHECK(alloc_tensor(&up, (uint64_t)N_USED * HALF_MID * sizeof(float)),
          "up");
    CHECK(alloc_tensor(&mid, (uint64_t)N_USED * HALF_MID * sizeof(float)),
          "mid");
    CHECK(alloc_tensor(&down, (uint64_t)N_USED * OUT_DIM * sizeof(float)),
          "down");
    CHECK(alloc_tensor(&selected, N_USED * sizeof(int32_t)), "selected");
    CHECK(alloc_tensor(&weights, N_USED * sizeof(float)), "weights");
    CHECK(alloc_tensor(&x, IN_DIM * sizeof(float)), "x");
    CHECK(alloc_tensor(&addend, OUT_DIM * sizeof(float)), "shared addend");

    const int32_t selected_host[N_USED] = {0, 1, 2, 3, 4, 5};
    const float weights_host[N_USED] =
        {0.31f, 0.23f, 0.17f, 0.13f, 0.09f, 0.07f};
    float x_host[IN_DIM], addend_host[OUT_DIM];
    for (uint32_t i = 0; i < IN_DIM; ++i)
        x_host[i] = (float)((int)(i % 31u) - 15) * 0.03125f;
    for (uint32_t i = 0; i < OUT_DIM; ++i)
        addend_host[i] = (float)((int)(i % 23u) - 11) * 0.0029296875f;
    CHECK(upload(&selected, selected_host, sizeof(selected_host)),
          "upload selected");
    CHECK(upload(&weights, weights_host, sizeof(weights_host)),
          "upload weights");
    CHECK(upload(&x, x_host, sizeof(x_host)), "upload x");
    CHECK(upload(&addend, addend_host, sizeof(addend_host)),
          "upload shared addend");

    ds4_gpu_tensor *slab = ds4_gpu_tensor_alloc_rdma_host(4096u);
    CHECK(slab, "TP slab");
    CHECK(ds4_gpu_tp_init(1u, slab, 0u, exchange_stub, NULL), "TP rank1 init");
    const ds4_gpu_q4k_kshard_layer layer = {
        gate_offset, up_offset, down_offset, N_EXPERT,
        IN_DIM, FULL_MID, OUT_DIM,
        GATE_ROW_BYTES, GATE_ROW_BYTES, DOWN_ROW_BYTES
    };
    const uint64_t resident_offsets[] = {
        dense_offset,
        gate_offset + gate_bytes / 2u,
        up_offset + gate_bytes / 2u,
        down_offset + down_bytes / 2u,
    };
    const uint64_t resident_sizes[] = {
        dense_bytes, gate_bytes / 2u, gate_bytes / 2u, down_bytes / 2u,
    };
    CHECK(ds4_gpu_set_model_fd_for_map(fd, model), "resident model fd");
    CHECK(ds4_gpu_set_model_map_spans(
              model, model_bytes, resident_offsets, resident_sizes, 4u,
              gate_bytes / 2u), "rank1 pre-transition residency");
    CHECK(ds4_gpu_q4k_kshard_install(
              model, model_bytes, fd, 1u,
              &dense_offset, &dense_bytes, 1u, dense_bytes,
              &layer, 1u), "rank1 atomic install");
    ds4_gpu_q4k_kshard_windows windows = {};
    CHECK(ds4_gpu_q4k_kshard_windows_get(&windows) &&
              windows.rank == 1u && windows.row_base == HALF_MID &&
              windows.down_column_byte_base == HALF_DOWN_ROW_BYTES,
          "rank1 installed windows");
    const void *gate_p = NULL, *up_p = NULL, *down_p = NULL;
    uint64_t rb = 0, eb = 0, pb = 0;
    CHECK(ds4_gpu_q4k_packed_slice_resolve(
              model, gate_offset, N_EXPERT, FULL_MID, GATE_ROW_BYTES,
              HALF_MID, HALF_MID, 0u, GATE_ROW_BYTES,
              DS4_GPU_Q4K_PACKED_ROW_RANGE, &gate_p, &pb, &eb, &rb),
          "evacuated gate resolve");
    CHECK(ds4_gpu_q4k_packed_slice_resolve(
              model, up_offset, N_EXPERT, FULL_MID, GATE_ROW_BYTES,
              HALF_MID, HALF_MID, 0u, GATE_ROW_BYTES,
              DS4_GPU_Q4K_PACKED_ROW_RANGE, &up_p, &pb, &eb, &rb),
          "evacuated up resolve");
    CHECK(ds4_gpu_q4k_packed_slice_resolve(
              model, down_offset, N_EXPERT, OUT_DIM, DOWN_ROW_BYTES,
              0u, OUT_DIM, HALF_DOWN_ROW_BYTES, HALF_DOWN_ROW_BYTES,
              DS4_GPU_Q4K_PACKED_K_RANGE, &down_p, &pb, &eb, &rb),
          "evacuated down resolve");
    CHECK(ds4_gpu_routed_moe_one_packed_q4k_tensor(
              &out, &gate, &up, &mid, &down,
              model, model_bytes, gate_offset, up_offset, down_offset,
              N_EXPERT, GATE_ROW_BYTES, DOWN_ROW_BYTES,
              windows.row_base, windows.row_count,
              windows.down_column_byte_base,
              windows.down_column_byte_count,
              &selected, &weights, N_USED, 0.0f, &x, &addend, 0u, NULL),
          "rank1 installed packed compose");
    CHECK(hipDeviceSynchronize() == hipSuccess, "candidate sync");
    float candidate[OUT_DIM], reference[OUT_DIM];
    CHECK(ds4_gpu_tensor_read(&out, 0, candidate, sizeof(candidate)),
          "candidate read");

    ds4_gpu_q4k_kshard_release();
    CHECK(ds4_gpu_tp_expert_shard_active() == 1,
          "rank1 remap restored");
    ds4_gpu_tp_shutdown();
    ds4_gpu_tensor_free(slab);
    /* The direct compact reference represents all selected experts rather
     * than rank-1 ownership. Disable ownership only inside this isolated
     * oracle after TP is shut down; no runtime state or tensor allocation is
     * destroyed between the candidate and reference reads. */
    ds4_gpu_tp_suspend_expert_sharding(1);
    CHECK(run_direct_reference(
              &out, &gate, &up, &mid, &down,
              compact, compact_bytes, compact_gate_bytes,
              compact_down_bytes, &selected, &weights, &x, &addend),
          "independent compact reference");
    CHECK(hipDeviceSynchronize() == hipSuccess, "reference sync");
    CHECK(ds4_gpu_tensor_read(&out, 0, reference, sizeof(reference)),
          "reference read");
    CHECK(memcmp(candidate, reference, sizeof(candidate)) == 0,
          "rank1 installed compose exact reference");
    ds4_gpu_tp_suspend_expert_sharding(0);

    printf("q4k_kshard_compose rank=1 layout=%s addend=nonzero exact=pass "
           "fnv64=%016llx packed_bytes=%llu\n",
           interleaved ? "even-odd" : "contiguous",
           (unsigned long long)fnv1a64(candidate, sizeof(candidate)),
           (unsigned long long)compact_bytes);

    ds4_gpu_tensor_free_in_place(&addend);
    ds4_gpu_tensor_free_in_place(&x);
    ds4_gpu_tensor_free_in_place(&weights);
    ds4_gpu_tensor_free_in_place(&selected);
    ds4_gpu_tensor_free_in_place(&down);
    ds4_gpu_tensor_free_in_place(&mid);
    ds4_gpu_tensor_free_in_place(&up);
    ds4_gpu_tensor_free_in_place(&gate);
    ds4_gpu_tensor_free_in_place(&out);
    ds4_gpu_cleanup();
    close(fd);
    unlink(path);
    free(compact);
    free(model);
    return 0;
}
