/* Synthetic bit-exact control for dynamically assigned TP shared halves. */

#include "ds4_gpu.h"
#include "ds4_gpu_mgpu.h"
#include "ds4_tp_shared_balance.h"

#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef int (*ds4_tp_devcopy_fn)(void *, const void *, uint64_t);
extern "C" void ds4_tp_set_devcopy(ds4_tp_devcopy_fn) {}

#define CHECK(cond, msg) do {                                                \
    if (!(cond)) {                                                           \
        fprintf(stderr, "FAIL: %s (line %d)\n", (msg), __LINE__);           \
        return 1;                                                            \
    }                                                                        \
} while (0)

enum {
    IN_DIM = 4096,
    SHARED_DIM = 2048,
    HALF = SHARED_DIM / 2,
    OUT_DIM = 4096,
};

static void pack_q8(unsigned char *weights, uint64_t in_dim,
                    uint64_t out_dim, uint32_t salt) {
    const uint64_t blocks = in_dim / 32u;
    for (uint64_t row = 0; row < out_dim; row++) {
        for (uint64_t block = 0; block < blocks; block++) {
            unsigned char *dst = weights +
                (row * blocks + block) * 34u;
            dst[0] = 0x00u;
            dst[1] = 0x2cu; /* fp16 0.0625 */
            for (uint64_t lane = 0; lane < 32u; lane++) {
                dst[2u + lane] = (unsigned char)(int8_t)(
                    (int)((row * 11u + block * 7u + lane * 3u + salt) % 29u) -
                    14);
            }
        }
    }
}

static int alloc_tensor(ds4_gpu_tensor *t, uint64_t bytes) {
    memset(t, 0, sizeof(*t));
    return ds4_gpu_tensor_alloc_on(t, 0, bytes) == 0;
}

static void make_selection(int32_t selected[6], uint32_t rank0_count) {
    for (uint32_t i = 0; i < rank0_count; i++) selected[i] = (int32_t)(9u + i);
    for (uint32_t i = rank0_count; i < 6u; i++)
        selected[i] = (int32_t)(137u + i);
}

int main(void) {
    int devices = 0;
    CHECK(hipGetDeviceCount(&devices) == hipSuccess && devices > 0,
          "ROCm device required");
    ds4_gpu_config config = {};
    config.n_gpus = 1;
    config.device_indices[0] = 0;
    CHECK(ds4_gpu_init_multi(&config), "initialize ROCm backend");

    const uint64_t gu_row = (IN_DIM / 32u) * 34u;
    const uint64_t down_row = (SHARED_DIM / 32u) * 34u;
    const uint64_t gate_bytes = (uint64_t)SHARED_DIM * gu_row;
    const uint64_t down_bytes = (uint64_t)OUT_DIM * down_row;
    const uint64_t up_offset = gate_bytes;
    const uint64_t down_offset = 2u * gate_bytes;
    const uint64_t model_bytes = down_offset + down_bytes;
    unsigned char *model = (unsigned char *)malloc((size_t)model_bytes);
    float *host_x = (float *)malloc((size_t)IN_DIM * sizeof(float));
    float *ref_mid_h = (float *)malloc((size_t)HALF * sizeof(float));
    float *got_mid_h = (float *)malloc((size_t)HALF * sizeof(float));
    float *ref_out_h = (float *)malloc((size_t)OUT_DIM * sizeof(float));
    float *got_out_h = (float *)malloc((size_t)OUT_DIM * sizeof(float));
    float *sentinel_mid = (float *)malloc((size_t)HALF * sizeof(float));
    float *sentinel_out = (float *)malloc((size_t)OUT_DIM * sizeof(float));
    CHECK(model && host_x && ref_mid_h && got_mid_h && ref_out_h && got_out_h &&
          sentinel_mid && sentinel_out, "allocate host controls");
    pack_q8(model, IN_DIM, SHARED_DIM, 3u);
    pack_q8(model + up_offset, IN_DIM, SHARED_DIM, 17u);
    pack_q8(model + down_offset, SHARED_DIM, OUT_DIM, 23u);
    for (uint32_t i = 0; i < IN_DIM; i++)
        host_x[i] = (float)((int)(i % 61u) - 30) / 256.0f;
    for (uint32_t i = 0; i < HALF; i++) {
        uint32_t u = UINT32_C(0x4a000000) + i;
        memcpy(&sentinel_mid[i], &u, sizeof(u));
    }
    for (uint32_t i = 0; i < OUT_DIM; i++) {
        uint32_t u = UINT32_C(0x4b000000) + i;
        memcpy(&sentinel_out[i], &u, sizeof(u));
    }
    CHECK(ds4_gpu_set_model_map(model, model_bytes), "install synthetic map");

    ds4_gpu_tensor x = {}, selected = {}, gate = {}, up = {};
    ds4_gpu_tensor ref_mid = {}, got_mid = {}, ref_out = {}, got_out = {};
    CHECK(alloc_tensor(&x, (uint64_t)IN_DIM * sizeof(float)) &&
          alloc_tensor(&selected, 6u * sizeof(int32_t)) &&
          alloc_tensor(&gate, (uint64_t)HALF * sizeof(float)) &&
          alloc_tensor(&up, (uint64_t)HALF * sizeof(float)) &&
          alloc_tensor(&ref_mid, (uint64_t)HALF * sizeof(float)) &&
          alloc_tensor(&got_mid, (uint64_t)HALF * sizeof(float)) &&
          alloc_tensor(&ref_out, (uint64_t)OUT_DIM * sizeof(float)) &&
          alloc_tensor(&got_out, (uint64_t)OUT_DIM * sizeof(float)),
          "allocate GPU controls");
    CHECK(ds4_gpu_tensor_write(&x, 0, host_x,
                               (uint64_t)IN_DIM * sizeof(float)), "upload x");

    for (uint32_t count0 = 0; count0 <= 6u; count0++) {
        int32_t ids[6];
        make_selection(ids, count0);
        const ds4_tp_shared_balance d =
            ds4_tp_shared_balance_decide(ids, 6u, 128u, 256u);
        CHECK(d.valid && ds4_gpu_tensor_write(&selected, 0, ids, sizeof(ids)),
              "upload selected ids");
        for (uint32_t rank = 0; rank < 2u; rank++) {
            for (uint32_t half = 0; half < 2u; half++) {
                const uint64_t half_off = (uint64_t)half * HALF * gu_row;
                CHECK(ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(
                          &gate, &up, &ref_mid, model, model_bytes,
                          half_off, up_offset + half_off,
                          IN_DIM, HALF, &x, 7.0f) &&
                      ds4_gpu_matmul_q8_0_kslice_rows_tensor(
                          &ref_out, model, model_bytes, down_offset,
                          SHARED_DIM, OUT_DIM, (uint64_t)half * HALF, HALF,
                          &ref_mid, 1u),
                      "run canonical half reference");
                CHECK(ds4_gpu_tensor_write(&got_mid, 0, sentinel_mid,
                                            (uint64_t)HALF * sizeof(float)) &&
                      ds4_gpu_tensor_write(&got_out, 0, sentinel_out,
                                            (uint64_t)OUT_DIM * sizeof(float)),
                      "reset candidate sentinels");
                CHECK(ds4_gpu_tp_shared_balance_half_q8_0_tensor(
                          &got_out, &got_mid, model, model_bytes,
                          0u, up_offset, down_offset,
                          IN_DIM, SHARED_DIM, OUT_DIM, half,
                          &x, &selected, 128u, 256u, rank, 7.0f),
                      "run balanced half candidate");
                CHECK(ds4_gpu_tensor_read(&ref_mid, 0, ref_mid_h,
                                           (uint64_t)HALF * sizeof(float)) &&
                      ds4_gpu_tensor_read(&got_mid, 0, got_mid_h,
                                           (uint64_t)HALF * sizeof(float)) &&
                      ds4_gpu_tensor_read(&ref_out, 0, ref_out_h,
                                           (uint64_t)OUT_DIM * sizeof(float)) &&
                      ds4_gpu_tensor_read(&got_out, 0, got_out_h,
                                           (uint64_t)OUT_DIM * sizeof(float)),
                      "read half controls");
                const bool assigned = d.shared_owner[half] == rank;
                CHECK(memcmp(got_mid_h, assigned ? ref_mid_h : sentinel_mid,
                             (size_t)HALF * sizeof(float)) == 0,
                      "balanced mid assignment is bit exact");
                CHECK(memcmp(got_out_h, assigned ? ref_out_h : sentinel_out,
                             (size_t)OUT_DIM * sizeof(float)) == 0,
                      "balanced down assignment is bit exact");
            }
        }
    }

    enum { GROUP_N = 64 };
    float routed_h[2][GROUP_N], shared_h[2][GROUP_N];
    float extra_local_h[GROUP_N], extra_peer_h[GROUP_N];
    float ref_group_h[2][GROUP_N], got_group_h[2][GROUP_N];
    for (uint32_t r = 0; r < 2u; r++) {
        for (uint32_t i = 0; i < GROUP_N; i++) {
            routed_h[r][i] = (float)((int)(i * 7u + r * 3u) - 97) / 64.0f;
            shared_h[r][i] = (float)((int)(i * 5u + r * 11u) - 83) / 128.0f;
        }
    }
    routed_h[0][0] = 0.0f;
    shared_h[0][0] = -0.0f;
    routed_h[1][0] = -0.0f;
    shared_h[1][0] = -0.0f;
    ds4_gpu_tensor routed[2] = {}, shared[2] = {}, main_group[2] = {};
    ds4_gpu_tensor ref_group[2] = {}, rebuilt[2] = {};
    ds4_gpu_tensor extra_local = {}, extra_peer = {};
    for (uint32_t r = 0; r < 2u; r++) {
        CHECK(alloc_tensor(&routed[r], sizeof(routed_h[r])) &&
              alloc_tensor(&shared[r], sizeof(shared_h[r])) &&
              alloc_tensor(&main_group[r], sizeof(routed_h[r])) &&
              alloc_tensor(&ref_group[r], sizeof(routed_h[r])) &&
              alloc_tensor(&rebuilt[r], sizeof(routed_h[r])) &&
              ds4_gpu_tensor_write(&routed[r], 0, routed_h[r],
                                   sizeof(routed_h[r])) &&
              ds4_gpu_tensor_write(&shared[r], 0, shared_h[r],
                                   sizeof(shared_h[r])) &&
              ds4_gpu_add_tensor(&ref_group[r], &routed[r], &shared[r],
                                 GROUP_N),
              "allocate canonical group controls");
    }
    CHECK(alloc_tensor(&extra_local, sizeof(extra_local_h)) &&
          alloc_tensor(&extra_peer, sizeof(extra_peer_h)),
          "allocate extra-half controls");
    for (uint32_t count0 = 0; count0 <= 6u; count0++) {
        int32_t ids[6];
        make_selection(ids, count0);
        const ds4_tp_shared_balance d =
            ds4_tp_shared_balance_decide(ids, 6u, 128u, 256u);
        CHECK(d.valid && ds4_gpu_tensor_write(&selected, 0, ids, sizeof(ids)),
              "upload grouping ids");
        for (uint32_t r = 0; r < 2u; r++) {
            CHECK(ds4_gpu_tp_shared_balance_main_tensor(
                      &main_group[r], &routed[r], &shared[r], &selected,
                      128u, 256u, r, GROUP_N),
                  "assemble rank main group");
        }
        for (uint32_t observer = 0; observer < 2u; observer++) {
            const uint32_t heavy = d.heavy_rank;
            const uint32_t light = d.light_rank;
            const float *local_src =
                d.move_heavy_half && observer == light ?
                    shared_h[heavy] : sentinel_out;
            const float *peer_src =
                d.move_heavy_half && observer == heavy ?
                    shared_h[heavy] : sentinel_out;
            memcpy(extra_local_h, local_src, sizeof(extra_local_h));
            memcpy(extra_peer_h, peer_src, sizeof(extra_peer_h));
            CHECK(ds4_gpu_tensor_write(&extra_local, 0, extra_local_h,
                                        sizeof(extra_local_h)) &&
                  ds4_gpu_tensor_write(&extra_peer, 0, extra_peer_h,
                                        sizeof(extra_peer_h)) &&
                  ds4_gpu_tp_shared_balance_reconstruct_tensor(
                      &rebuilt[0], &rebuilt[1],
                      &main_group[0], &main_group[1],
                      &extra_local, &extra_peer, &selected,
                      128u, 256u, observer, GROUP_N),
                  "reconstruct canonical groups");
            for (uint32_t r = 0; r < 2u; r++) {
                CHECK(ds4_gpu_tensor_read(&ref_group[r], 0, ref_group_h[r],
                                           sizeof(ref_group_h[r])) &&
                      ds4_gpu_tensor_read(&rebuilt[r], 0, got_group_h[r],
                                           sizeof(got_group_h[r])) &&
                      memcmp(ref_group_h[r], got_group_h[r],
                             sizeof(ref_group_h[r])) == 0,
                      "GPU canonical group reconstruction is bit exact");
            }
        }
    }
    fprintf(stderr,
            "test_rocm_tp_shared_balance: PASS "
            "(7 splits x 2 ranks x 2 halves; canonical groups)\n");

    ds4_gpu_tensor_free_in_place(&x);
    ds4_gpu_tensor_free_in_place(&selected);
    ds4_gpu_tensor_free_in_place(&gate);
    ds4_gpu_tensor_free_in_place(&up);
    ds4_gpu_tensor_free_in_place(&ref_mid);
    ds4_gpu_tensor_free_in_place(&got_mid);
    ds4_gpu_tensor_free_in_place(&ref_out);
    ds4_gpu_tensor_free_in_place(&got_out);
    for (uint32_t r = 0; r < 2u; r++) {
        ds4_gpu_tensor_free_in_place(&routed[r]);
        ds4_gpu_tensor_free_in_place(&shared[r]);
        ds4_gpu_tensor_free_in_place(&main_group[r]);
        ds4_gpu_tensor_free_in_place(&ref_group[r]);
        ds4_gpu_tensor_free_in_place(&rebuilt[r]);
    }
    ds4_gpu_tensor_free_in_place(&extra_local);
    ds4_gpu_tensor_free_in_place(&extra_peer);
    ds4_gpu_cleanup();
    free(model);
    free(host_x);
    free(ref_mid_h);
    free(got_mid_h);
    free(ref_out_h);
    free(got_out_h);
    free(sentinel_mid);
    free(sentinel_out);
    return 0;
}
