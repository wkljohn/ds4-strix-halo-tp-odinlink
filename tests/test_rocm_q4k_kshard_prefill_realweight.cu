/* Production-scale real-weight numeric gate for H8 half-K Q4_K WMMA.
 *
 * The test loads one real 256-expert Q4_K layer directly from the documented
 * GGUF offsets, packs both 1024-row/K halves on-device, and compares the GPU
 * sum of both half-K outputs with the full-K production WMMA output at 2048
 * tokens using a frozen production route tensor. No expanded-weight cache is
 * created on disk or retained after the process exits.
 */
#include "ds4_gpu.h"
#include "ds4_gpu_mgpu.h"

#include <hip/hip_runtime.h>
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define CHECK(cond, msg) do {                                                \
    if (!(cond)) {                                                           \
        fprintf(stderr, "FAIL: %s (line %d)\n", (msg), __LINE__);           \
        return 1;                                                            \
    }                                                                        \
} while (0)

typedef int (*ds4_tp_devcopy_fn)(void *, const void *, uint64_t);
extern "C" void ds4_tp_set_devcopy(ds4_tp_devcopy_fn) {}
extern "C" int ds4_gpu_routed_moe_batch_q4k_direct_control(
        ds4_gpu_tensor *, ds4_gpu_tensor *, ds4_gpu_tensor *,
        ds4_gpu_tensor *, ds4_gpu_tensor *, const void *, const void *,
        const void *, uint64_t, uint64_t, uint64_t, uint64_t,
        const ds4_gpu_tensor *, const ds4_gpu_tensor *, uint32_t, uint32_t,
        float, const ds4_gpu_tensor *, uint32_t, uint32_t, uint32_t);

enum {
    IN_DIM = 4096,
    MID_FULL = 2048,
    MID_HALF = 1024,
    OUT_DIM = 4096,
    N_TOTAL = 256,
    N_USED = 6,
    TOKENS = 2048,
    QK_K = 256,
    Q4_K_BYTES = 144,
};

static constexpr float MAX_REL_LIMIT = 5.0e-4f;
static constexpr double NMSE_LIMIT = 5.0e-8;
static constexpr double REF_SQ_FLOOR = 1.0e-12;

static int alloc_tensor(ds4_gpu_tensor *tensor, uint64_t bytes) {
    memset(tensor, 0, sizeof(*tensor));
    return ds4_gpu_tensor_alloc_on(tensor, 0, bytes) == 0;
}

static int alloc_upload(ds4_gpu_tensor *tensor, const void *src,
                        uint64_t bytes) {
    return alloc_tensor(tensor, bytes) &&
           ds4_gpu_tensor_write(tensor, 0, src, bytes) != 0;
}

static int read_exact(const char *path, void *dst, size_t bytes) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return 0;
    const size_t got = fread(dst, 1, bytes, fp);
    const int extra = fgetc(fp);
    const int close_rc = fclose(fp);
    return got == bytes && extra == EOF && close_rc == 0;
}

static int read_prefix(const char *path, void *dst, size_t bytes) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return 0;
    const size_t got = fread(dst, 1, bytes, fp);
    const int close_rc = fclose(fp);
    return got == bytes && close_rc == 0;
}

static int parse_u64_env(const char *name, uint64_t *out) {
    const char *value = getenv(name);
    if (!value || !value[0]) return 0;
    errno = 0;
    char *end = NULL;
    const unsigned long long parsed = strtoull(value, &end, 0);
    if (errno || end == value || *end != '\0') return 0;
    *out = (uint64_t)parsed;
    return 1;
}

static int upload_file_range(void *device, int fd, uint64_t offset,
                             uint64_t bytes) {
    const size_t chunk = 64u * 1024u * 1024u;
    void *host = NULL;
    if (posix_memalign(&host, 4096u, chunk) != 0 || !host) return 0;
    uint64_t done = 0;
    int ok = 1;
    while (done < bytes) {
        const size_t want = (size_t)((bytes - done) < chunk ?
                                     (bytes - done) : chunk);
        size_t got = 0;
        while (got < want) {
            const ssize_t rc = pread(fd, (char *)host + got, want - got,
                                     (off_t)(offset + done + got));
            if (rc < 0 && errno == EINTR) continue;
            if (rc <= 0) { ok = 0; break; }
            got += (size_t)rc;
        }
        if (!ok || hipMemcpy((char *)device + done, host, want,
                             hipMemcpyHostToDevice) != hipSuccess) {
            ok = 0;
            break;
        }
        done += want;
    }
    free(host);
    return ok;
}

static int pack_halves(void *dst_gate[2], void *dst_up[2], void *dst_down[2],
                       const void *full_gate, const void *full_up,
                       const void *full_down, uint64_t gate_expert_bytes,
                       uint64_t half_gate_expert_bytes,
                       uint64_t down_expert_bytes,
                       uint64_t down_row_bytes,
                       uint64_t half_down_expert_bytes,
                       uint64_t half_down_row_bytes,
                       int interleaved) {
    for (uint32_t e = 0; e < N_TOTAL; ++e) {
        for (uint32_t half = 0; half < 2u; ++half) {
            const uint64_t gate_dst = (uint64_t)e * half_gate_expert_bytes;
            const uint64_t down_dst = (uint64_t)e * half_down_expert_bytes;
            if (!interleaved) {
                const uint64_t gate_src = (uint64_t)e * gate_expert_bytes +
                    (uint64_t)half * half_gate_expert_bytes;
                if (hipMemcpy((char *)dst_gate[half] + gate_dst,
                              (const char *)full_gate + gate_src,
                              half_gate_expert_bytes,
                              hipMemcpyDeviceToDevice) != hipSuccess ||
                    hipMemcpy((char *)dst_up[half] + gate_dst,
                              (const char *)full_up + gate_src,
                              half_gate_expert_bytes,
                              hipMemcpyDeviceToDevice) != hipSuccess) {
                    return 0;
                }
                const uint64_t down_src = (uint64_t)e * down_expert_bytes +
                    (uint64_t)half * half_down_row_bytes;
                if (hipMemcpy2D((char *)dst_down[half] + down_dst,
                                half_down_row_bytes,
                                (const char *)full_down + down_src,
                                down_row_bytes,
                                half_down_row_bytes, OUT_DIM,
                                hipMemcpyDeviceToDevice) != hipSuccess) {
                    return 0;
                }
                continue;
            }

            /* Preserve the production eight-lane down-dot reduction tree
             * across two four-lane TP ranks.  Rank 0 owns Q4_K mid blocks
             * {0,2,4,6}; rank 1 owns {1,3,5,7}.  With the existing four-lane
             * reduction, rank 0 forms (a0+a4)+(a2+a6), rank 1 forms
             * (a1+a5)+(a3+a7), and the canonical rank0+rank1 combine exactly
             * matches the full eight-lane tree.  Gate/up rows must use the
             * same block permutation as the down columns. */
            const uint64_t gate_chunk_bytes =
                (uint64_t)QK_K * (gate_expert_bytes / MID_FULL);
            for (uint32_t chunk = 0; chunk < 4u; ++chunk) {
                const uint32_t source_chunk = 2u * chunk + half;
                const uint64_t gate_src = (uint64_t)e * gate_expert_bytes +
                    (uint64_t)source_chunk * gate_chunk_bytes;
                const uint64_t packed_dst = gate_dst +
                    (uint64_t)chunk * gate_chunk_bytes;
                if (hipMemcpy((char *)dst_gate[half] + packed_dst,
                              (const char *)full_gate + gate_src,
                              gate_chunk_bytes,
                              hipMemcpyDeviceToDevice) != hipSuccess ||
                    hipMemcpy((char *)dst_up[half] + packed_dst,
                              (const char *)full_up + gate_src,
                              gate_chunk_bytes,
                              hipMemcpyDeviceToDevice) != hipSuccess) {
                    return 0;
                }
                const uint64_t down_src = (uint64_t)e * down_expert_bytes +
                    (uint64_t)source_chunk * Q4_K_BYTES;
                if (hipMemcpy2D((char *)dst_down[half] + down_dst +
                                    (uint64_t)chunk * Q4_K_BYTES,
                                half_down_row_bytes,
                                (const char *)full_down + down_src,
                                down_row_bytes,
                                Q4_K_BYTES, OUT_DIM,
                                hipMemcpyDeviceToDevice) != hipSuccess) {
                    return 0;
                }
            }
        }
    }
    return hipDeviceSynchronize() == hipSuccess;
}

__global__ static void fill_input(float *x, uint64_t count) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        const int v = (int)((i * 17u + i / IN_DIM * 13u) % 101u) - 50;
        x[i] = (float)v * 0.00390625f;
    }
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

static void print_decode_compose_metrics(const char *name,
                                         const float *reference,
                                         const float *candidate) {
    double diff_sq = 0.0, ref_sq = 0.0;
    float max_abs = 0.0f;
    uint64_t changed = 0;
    for (uint32_t row = 0; row < OUT_DIM; ++row) {
        const float ref = reference[row];
        const float got = candidate[row];
        const float err = fabsf(got - ref);
        max_abs = fmaxf(max_abs, err);
        diff_sq += (double)err * err;
        ref_sq += (double)ref * ref;
        changed += memcmp(&ref, &got, sizeof(float)) != 0;
    }
    printf("decode_compose=%s changed=%llu max_abs=%.9e nmse=%.9e "
           "fnv=%016llx\n", name, (unsigned long long)changed, max_abs,
           diff_sq / fmax(1.0e-30, ref_sq),
           (unsigned long long)fnv1a64(candidate,
                                        OUT_DIM * sizeof(float)));
}

static int launch(ds4_gpu_tensor *out,
                  ds4_gpu_tensor *gate, ds4_gpu_tensor *up,
                  ds4_gpu_tensor *mid, ds4_gpu_tensor *down,
                  const void *gate_w, const void *up_w, const void *down_w,
                  uint64_t gate_expert_bytes, uint64_t gate_row_bytes,
                  uint64_t down_expert_bytes, uint64_t down_row_bytes,
                  ds4_gpu_tensor *selected, ds4_gpu_tensor *weights,
                  ds4_gpu_tensor *x, uint32_t mid_dim,
                  uint32_t n_tokens, uint32_t n_used) {
    return ds4_gpu_routed_moe_batch_q4k_direct_control(
        out, gate, up, mid, down, gate_w, up_w, down_w,
        gate_expert_bytes, gate_row_bytes,
        down_expert_bytes, down_row_bytes,
        selected, weights, N_TOTAL, n_used, 10.0f,
        x, 0u, n_tokens, mid_dim);
}

int main(void) {
    int devices = 0;
    CHECK(hipGetDeviceCount(&devices) == hipSuccess && devices > 0,
          "ROCm device required");
    const char *model_path = getenv("DS4_TEST_Q4K_MODEL");
    const char *route_prefix = getenv("DS4_TEST_Q4K_ROUTE_CAPTURE_PREFIX");
    const char *layout_env = getenv("DS4_TEST_Q4K_INTERLEAVED");
    const int interleaved = layout_env && layout_env[0] == '1' &&
                            layout_env[1] == '\0';
    const char *decode_env = getenv("DS4_TEST_Q4K_DECODE_ONE");
    const uint32_t tokens = decode_env && decode_env[0] == '1' &&
                            decode_env[1] == '\0' ? 1u : TOKENS;
    CHECK(model_path && model_path[0] && route_prefix && route_prefix[0],
          "model and route prefix required");
    uint64_t gate_offset = 0, up_offset = 0, down_offset = 0;
    CHECK(parse_u64_env("DS4_TEST_Q4K_GATE_OFFSET", &gate_offset) &&
          parse_u64_env("DS4_TEST_Q4K_UP_OFFSET", &up_offset) &&
          parse_u64_env("DS4_TEST_Q4K_DOWN_OFFSET", &down_offset),
          "real GGUF tensor offsets required");
    CHECK(setenv("DS4_ROCM_Q4K_KSHARD_RESEARCH", "1", 1) == 0 &&
          setenv("DS4_ROCM_Q4K_WMMA_PAIR_GATE_UP", "1", 1) == 0 &&
          setenv("DS4_ROCM_Q4K_WMMA_FUSE_MID", "1", 1) == 0 &&
          setenv("DS4_ROCM_TP_PREFILL_SKIP_UNOWNED", "1", 1) == 0,
          "production half-K controls");
    CHECK(unsetenv("DS4_ROCM_Q4K_WMMA_LAYER_LOG") == 0,
          "disable intrusive logging");

    ds4_gpu_config config = {};
    config.n_gpus = 1;
    config.device_indices[0] = 0;
    CHECK(ds4_gpu_init_multi(&config), "initialize ROCm");

    const uint64_t gate_row_bytes = 16u * Q4_K_BYTES;
    const uint64_t gate_expert_bytes = MID_FULL * gate_row_bytes;
    const uint64_t half_gate_expert_bytes = MID_HALF * gate_row_bytes;
    const uint64_t down_row_bytes = 8u * Q4_K_BYTES;
    const uint64_t half_down_row_bytes = 4u * Q4_K_BYTES;
    const uint64_t down_expert_bytes = OUT_DIM * down_row_bytes;
    const uint64_t half_down_expert_bytes = OUT_DIM * half_down_row_bytes;
    const uint64_t table_bytes = N_TOTAL * gate_expert_bytes;
    const uint64_t half_table_bytes = N_TOTAL * half_gate_expert_bytes;
    CHECK(down_expert_bytes == gate_expert_bytes &&
          half_down_expert_bytes == half_gate_expert_bytes,
          "DeepSeek Q4_K expert table geometry");

    int fd = open(model_path, O_RDONLY | O_CLOEXEC);
    CHECK(fd >= 0, "open real GGUF");
    void *full_gate = NULL, *full_up = NULL, *full_down = NULL;
    void *half_gate[2] = {}, *half_up[2] = {}, *half_down[2] = {};
    CHECK(hipMalloc(&full_gate, table_bytes) == hipSuccess &&
          hipMalloc(&full_up, table_bytes) == hipSuccess &&
          hipMalloc(&full_down, table_bytes) == hipSuccess,
          "allocate full real tables");
    CHECK(upload_file_range(full_gate, fd, gate_offset, table_bytes) &&
          upload_file_range(full_up, fd, up_offset, table_bytes) &&
          upload_file_range(full_down, fd, down_offset, table_bytes),
          "upload real Q4_K tables");
    CHECK(close(fd) == 0, "close real GGUF");
    for (uint32_t h = 0; h < 2u; ++h) {
        CHECK(hipMalloc(&half_gate[h], half_table_bytes) == hipSuccess &&
              hipMalloc(&half_up[h], half_table_bytes) == hipSuccess &&
              hipMalloc(&half_down[h], half_table_bytes) == hipSuccess,
              "allocate packed half tables");
    }
    CHECK(pack_halves(half_gate, half_up, half_down,
                      full_gate, full_up, full_down,
                      gate_expert_bytes, half_gate_expert_bytes,
                      down_expert_bytes, down_row_bytes,
                      half_down_expert_bytes, half_down_row_bytes,
                      interleaved),
          "pack real Q4_K halves");

    const uint64_t pairs = (uint64_t)tokens * N_USED;
    const uint64_t out_bytes = (uint64_t)tokens * OUT_DIM * sizeof(float);
    const uint64_t mid_bytes = pairs * MID_FULL * sizeof(float);
    const uint64_t down_bytes = pairs * OUT_DIM * sizeof(float);
    const uint64_t x_bytes = (uint64_t)tokens * IN_DIM * sizeof(float);
    ds4_gpu_tensor selected = {}, weights = {}, x = {};
    ds4_gpu_tensor gate = {}, up = {}, mid = {}, down = {};
    ds4_gpu_tensor out_full = {}, out_half0 = {}, out_half1 = {}, out_sum = {};
    int32_t *host_selected = (int32_t *)malloc((size_t)pairs * sizeof(int32_t));
    float *host_weights = (float *)malloc((size_t)pairs * sizeof(float));
    char route_path[1024], weights_path[1024];
    snprintf(route_path, sizeof(route_path), "%s_layer0_pos0_topk.i32",
             route_prefix);
    snprintf(weights_path, sizeof(weights_path), "%s_layer0_pos0_weights.f32",
             route_prefix);
    CHECK(host_selected && host_weights &&
          (tokens == TOKENS ?
              read_exact(route_path, host_selected,
                         (size_t)pairs * sizeof(int32_t)) :
              read_prefix(route_path, host_selected,
                          (size_t)pairs * sizeof(int32_t))) &&
          (tokens == TOKENS ?
              read_exact(weights_path, host_weights,
                         (size_t)pairs * sizeof(float)) :
              read_prefix(weights_path, host_weights,
                          (size_t)pairs * sizeof(float))),
          "read frozen layer-0 routes");
    for (uint64_t p = 0; p < pairs; ++p) {
        CHECK(host_selected[p] >= 0 && host_selected[p] < N_TOTAL &&
              isfinite(host_weights[p]), "validate frozen layer-0 route");
    }
    unsigned char expert_seen[N_TOTAL] = {};
    uint32_t expert_coverage = 0;
    for (uint64_t p = 0; p < pairs; ++p) {
        const uint32_t expert = (uint32_t)host_selected[p];
        if (!expert_seen[expert]) {
            expert_seen[expert] = 1u;
            expert_coverage++;
        }
    }
    const uint64_t route_fnv = fnv1a64(host_selected,
                                        pairs * sizeof(int32_t));
    const uint64_t weight_fnv = fnv1a64(host_weights,
                                         pairs * sizeof(float));
    CHECK(alloc_upload(&selected, host_selected, pairs * sizeof(int32_t)) &&
          alloc_upload(&weights, host_weights, pairs * sizeof(float)) &&
          alloc_tensor(&x, x_bytes) &&
          alloc_tensor(&gate, mid_bytes) &&
          alloc_tensor(&up, mid_bytes) &&
          alloc_tensor(&mid, mid_bytes) &&
          alloc_tensor(&down, down_bytes) &&
          alloc_tensor(&out_full, out_bytes) &&
          alloc_tensor(&out_half0, out_bytes) &&
          alloc_tensor(&out_half1, out_bytes) &&
          alloc_tensor(&out_sum, out_bytes), "allocate numeric tensors");
    const uint64_t x_count = (uint64_t)tokens * IN_DIM;
    fill_input<<<(x_count + 255u) / 256u, 256>>>((float *)x.ptr, x_count);
    CHECK(hipGetLastError() == hipSuccess && hipDeviceSynchronize() == hipSuccess,
          "fill deterministic activation");

    CHECK(launch(&out_full, &gate, &up, &mid, &down,
                 full_gate, full_up, full_down,
                 gate_expert_bytes, gate_row_bytes,
                 down_expert_bytes, down_row_bytes,
                 &selected, &weights, &x, MID_FULL, tokens, N_USED),
          "full real-weight production path");
    CHECK(launch(&out_half0, &gate, &up, &mid, &down,
                 half_gate[0], half_up[0], half_down[0],
                 half_gate_expert_bytes, gate_row_bytes,
                 half_down_expert_bytes, half_down_row_bytes,
                 &selected, &weights, &x, MID_HALF, tokens, N_USED),
          "half0 real-weight production path");
    CHECK(launch(&out_half1, &gate, &up, &mid, &down,
                 half_gate[1], half_up[1], half_down[1],
                 half_gate_expert_bytes, gate_row_bytes,
                 half_down_expert_bytes, half_down_row_bytes,
                 &selected, &weights, &x, MID_HALF, tokens, N_USED),
          "half1 real-weight production path");
    CHECK(ds4_gpu_add_tensor(&out_sum, &out_half0, &out_half1,
                             tokens * OUT_DIM) != 0 &&
          hipDeviceSynchronize() == hipSuccess, "compose half-K outputs");

    if (tokens == 1u) {
        const uint64_t routed_slot_bytes =
            (uint64_t)N_USED * OUT_DIM * sizeof(float);
        const uint64_t exchange_slot_bytes =
            (uint64_t)(N_USED + 1u) * OUT_DIM * sizeof(float);
        void *slot_full_dev = NULL, *slot_half0_dev = NULL,
             *slot_half1_dev = NULL;
        CHECK(hipMalloc(&slot_full_dev, exchange_slot_bytes) == hipSuccess &&
              hipMalloc(&slot_half0_dev, exchange_slot_bytes) == hipSuccess &&
              hipMalloc(&slot_half1_dev, exchange_slot_bytes) == hipSuccess &&
              hipMemset(slot_full_dev, 0, exchange_slot_bytes) == hipSuccess &&
              hipMemset(slot_half0_dev, 0, exchange_slot_bytes) == hipSuccess &&
              hipMemset(slot_half1_dev, 0, exchange_slot_bytes) == hipSuccess,
              "allocate per-slot decode outputs");
        for (uint32_t slot = 0; slot < N_USED; ++slot) {
            ds4_gpu_tensor sel_one = selected;
            ds4_gpu_tensor weight_one = weights;
            ds4_gpu_tensor full_one = out_full;
            ds4_gpu_tensor half0_one = out_half0;
            ds4_gpu_tensor half1_one = out_half1;
            sel_one.ptr = (char *)selected.ptr +
                          (uint64_t)slot * sizeof(int32_t);
            sel_one.bytes = sizeof(int32_t);
            sel_one.owner = 0;
            weight_one.ptr = (char *)weights.ptr +
                             (uint64_t)slot * sizeof(float);
            weight_one.bytes = sizeof(float);
            weight_one.owner = 0;
            full_one.ptr = (char *)slot_full_dev +
                           (uint64_t)slot * OUT_DIM * sizeof(float);
            full_one.owner = 0;
            half0_one.ptr = (char *)slot_half0_dev +
                            (uint64_t)slot * OUT_DIM * sizeof(float);
            half0_one.owner = 0;
            half1_one.ptr = (char *)slot_half1_dev +
                            (uint64_t)slot * OUT_DIM * sizeof(float);
            half1_one.owner = 0;
            CHECK(launch(&full_one, &gate, &up, &mid, &down,
                         full_gate, full_up, full_down,
                         gate_expert_bytes, gate_row_bytes,
                         down_expert_bytes, down_row_bytes,
                         &sel_one, &weight_one, &x, MID_FULL, 1u, 1u) &&
                  launch(&half0_one, &gate, &up, &mid, &down,
                         half_gate[0], half_up[0], half_down[0],
                         half_gate_expert_bytes, gate_row_bytes,
                         half_down_expert_bytes, half_down_row_bytes,
                         &sel_one, &weight_one, &x, MID_HALF, 1u, 1u) &&
                  launch(&half1_one, &gate, &up, &mid, &down,
                         half_gate[1], half_up[1], half_down[1],
                         half_gate_expert_bytes, gate_row_bytes,
                         half_down_expert_bytes, half_down_row_bytes,
                         &sel_one, &weight_one, &x, MID_HALF, 1u, 1u),
                  "launch per-slot decode controls");
        }
        CHECK(hipDeviceSynchronize() == hipSuccess,
              "per-slot decode controls sync");
        float *slot_full = (float *)malloc((size_t)routed_slot_bytes);
        float *slot_half0 = (float *)malloc((size_t)routed_slot_bytes);
        float *slot_half1 = (float *)malloc((size_t)routed_slot_bytes);
        float *owner_ref = (float *)malloc(OUT_DIM * sizeof(float));
        float *rank_group = (float *)malloc(OUT_DIM * sizeof(float));
        float *owner_group = (float *)malloc(OUT_DIM * sizeof(float));
        float *slot_group = (float *)malloc(OUT_DIM * sizeof(float));
        float *slot_owner_group = (float *)malloc(OUT_DIM * sizeof(float));
        CHECK(slot_full && slot_half0 && slot_half1 && owner_ref &&
              rank_group && owner_group && slot_group && slot_owner_group &&
              hipMemcpy(slot_full, slot_full_dev, routed_slot_bytes,
                        hipMemcpyDeviceToHost) == hipSuccess &&
              hipMemcpy(slot_half0, slot_half0_dev, routed_slot_bytes,
                        hipMemcpyDeviceToHost) == hipSuccess &&
              hipMemcpy(slot_half1, slot_half1_dev, routed_slot_bytes,
                        hipMemcpyDeviceToHost) == hipSuccess,
              "read per-slot decode controls");
        for (uint32_t row = 0; row < OUT_DIM; ++row) {
            volatile float ref0 = 0.0f, ref1 = 0.0f;
            volatile float half0_all = 0.0f, half1_all = 0.0f;
            volatile float half0_owner0 = 0.0f, half1_owner0 = 0.0f;
            volatile float half0_owner1 = 0.0f, half1_owner1 = 0.0f;
            volatile float per_slot = 0.0f;
            volatile float per_slot_owner0 = 0.0f;
            volatile float per_slot_owner1 = 0.0f;
            for (uint32_t slot = 0; slot < N_USED; ++slot) {
                const uint64_t i = (uint64_t)slot * OUT_DIM + row;
                const volatile float expert =
                    slot_half0[i] + slot_half1[i];
                if (host_selected[slot] < (N_TOTAL / 2)) {
                    ref0 = ref0 + slot_full[i];
                    half0_owner0 = half0_owner0 + slot_half0[i];
                    half1_owner0 = half1_owner0 + slot_half1[i];
                    per_slot_owner0 = per_slot_owner0 + expert;
                } else {
                    ref1 = ref1 + slot_full[i];
                    half0_owner1 = half0_owner1 + slot_half0[i];
                    half1_owner1 = half1_owner1 + slot_half1[i];
                    per_slot_owner1 = per_slot_owner1 + expert;
                }
                half0_all = half0_all + slot_half0[i];
                half1_all = half1_all + slot_half1[i];
                per_slot = per_slot + expert;
            }
            owner_ref[row] = ref0 + ref1;
            rank_group[row] = half0_all + half1_all;
            const volatile float owner0 = half0_owner0 + half1_owner0;
            const volatile float owner1 = half0_owner1 + half1_owner1;
            owner_group[row] = owner0 + owner1;
            slot_group[row] = per_slot;
            slot_owner_group[row] = per_slot_owner0 + per_slot_owner1;
        }
        print_decode_compose_metrics("rank", owner_ref, rank_group);
        print_decode_compose_metrics("owner", owner_ref, owner_group);
        print_decode_compose_metrics("slot", owner_ref, slot_group);
        print_decode_compose_metrics("slot-owner", owner_ref,
                                     slot_owner_group);
        ds4_gpu_tensor local_slots = {};
        ds4_gpu_tensor peer_slots = {};
        local_slots.ptr = slot_half0_dev;
        local_slots.bytes = exchange_slot_bytes;
        peer_slots.ptr = slot_half1_dev;
        peer_slots.bytes = exchange_slot_bytes;
        CHECK(ds4_gpu_q4k_kshard_slot_owner_combine_tensor(
                  &out_sum, &local_slots, &peer_slots, &selected,
                  0u, N_USED, OUT_DIM, N_TOTAL / 2u) != 0 &&
              hipDeviceSynchronize() == hipSuccess,
              "production slot-owner combine");
        float *slot_owner_gpu = (float *)malloc(OUT_DIM * sizeof(float));
        CHECK(slot_owner_gpu &&
              ds4_gpu_tensor_read(&out_sum, 0, slot_owner_gpu,
                                  OUT_DIM * sizeof(float)) != 0,
              "read production slot-owner combine");
        print_decode_compose_metrics("slot-owner-gpu", owner_ref,
                                     slot_owner_gpu);
        free(slot_owner_gpu);
        free(slot_owner_group);
        free(slot_group);
        free(owner_group);
        free(rank_group);
        free(owner_ref);
        free(slot_half1);
        free(slot_half0);
        free(slot_full);
        CHECK(hipFree(slot_half1_dev) == hipSuccess &&
              hipFree(slot_half0_dev) == hipSuccess &&
              hipFree(slot_full_dev) == hipSuccess,
              "free per-slot decode outputs");
    }

    float *reference = (float *)malloc((size_t)out_bytes);
    float *candidate = (float *)malloc((size_t)out_bytes);
    CHECK(reference && candidate &&
          ds4_gpu_tensor_read(&out_full, 0, reference, out_bytes) != 0 &&
          ds4_gpu_tensor_read(&out_sum, 0, candidate, out_bytes) != 0,
          "read real-weight outputs");
    double diff_sq = 0.0, ref_sq = 0.0;
    float max_rel = 0.0f, max_abs = 0.0f, max_ref = 0.0f;
    uint64_t nonfinite = 0, changed = 0;
    const uint64_t elems = (uint64_t)tokens * OUT_DIM;
    for (uint64_t i = 0; i < elems; ++i) {
        const float ref = reference[i];
        const float got = candidate[i];
        if (!isfinite(ref) || !isfinite(got)) nonfinite++;
        const float abs_err = fabsf(got - ref);
        const float rel = abs_err / fmaxf(1.0f, fabsf(ref));
        max_abs = fmaxf(max_abs, abs_err);
        max_rel = fmaxf(max_rel, rel);
        max_ref = fmaxf(max_ref, fabsf(ref));
        diff_sq += (double)abs_err * abs_err;
        ref_sq += (double)ref * ref;
        changed += memcmp(&ref, &got, sizeof(float)) != 0;
    }
    const double nmse = diff_sq / fmax(1.0e-30, ref_sq);
    const uint64_t ref_fnv = fnv1a64(reference, out_bytes);
    const uint64_t candidate_fnv = fnv1a64(candidate, out_bytes);
    printf("test_rocm_q4k_kshard_prefill_realweight: tokens=%u layer=0 "
           "layout=%s "
           "elems=%llu changed=%llu nonfinite=%llu max_abs=%.9e "
           "max_rel=%.9e max_ref=%.9e ref_sq=%.9e nmse=%.9e "
           "max_rel_limit=%.9e nmse_limit=%.9e ref_sq_floor=%.9e "
           "route_fnv=%016llx weight_fnv=%016llx expert_coverage=%u/%u "
           "reference_fnv=%016llx candidate_fnv=%016llx decision=%s\n",
           tokens, interleaved ? "even-odd" : "contiguous",
           (unsigned long long)elems, (unsigned long long)changed,
           (unsigned long long)nonfinite, max_abs, max_rel, max_ref, ref_sq,
           nmse, MAX_REL_LIMIT, NMSE_LIMIT, REF_SQ_FLOOR,
           (unsigned long long)route_fnv,
           (unsigned long long)weight_fnv,
           expert_coverage, N_TOTAL,
           (unsigned long long)ref_fnv, (unsigned long long)candidate_fnv,
           nonfinite == 0 && changed > 0 && ref_sq > REF_SQ_FLOOR &&
                   max_rel <= MAX_REL_LIMIT && nmse <= NMSE_LIMIT ?
               "PASS" : "FAIL");

    const bool pass = nonfinite == 0 && changed > 0 &&
                      ref_sq > REF_SQ_FLOOR &&
                      max_rel <= MAX_REL_LIMIT && nmse <= NMSE_LIMIT;
    free(candidate);
    free(reference);
    free(host_weights);
    free(host_selected);
    ds4_gpu_tensor_free_in_place(&out_sum);
    ds4_gpu_tensor_free_in_place(&out_half1);
    ds4_gpu_tensor_free_in_place(&out_half0);
    ds4_gpu_tensor_free_in_place(&out_full);
    ds4_gpu_tensor_free_in_place(&down);
    ds4_gpu_tensor_free_in_place(&mid);
    ds4_gpu_tensor_free_in_place(&up);
    ds4_gpu_tensor_free_in_place(&gate);
    ds4_gpu_tensor_free_in_place(&x);
    ds4_gpu_tensor_free_in_place(&weights);
    ds4_gpu_tensor_free_in_place(&selected);
    for (uint32_t h = 0; h < 2u; ++h) {
        CHECK(hipFree(half_down[h]) == hipSuccess &&
              hipFree(half_up[h]) == hipSuccess &&
              hipFree(half_gate[h]) == hipSuccess,
              "free packed half tables");
    }
    CHECK(hipFree(full_down) == hipSuccess &&
          hipFree(full_up) == hipSuccess &&
          hipFree(full_gate) == hipSuccess, "free full real tables");
    ds4_gpu_cleanup();
    return pass ? 0 : 1;
}
