/* Exact current-kernel overlap-efficiency gate for gfx1151 decode.
 *
 * This test runs the shipped Q8_0 shared gate/up/SwiGLU kernel and shipped
 * cache-free Q4_K routed-MoE kernel at production one-token dimensions.  The
 * only test hook is a stream-selectable launch of the same shared kernel;
 * production builds do not contain that hook.  Outputs and backend scratch
 * are disjoint.  Serial and concurrent orders are alternated across samples.
 */

#include "ds4_gpu.h"
#include "ds4_gpu_mgpu.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CHECK(cond, msg) do {                                                \
    if (!(cond)) {                                                           \
        std::fprintf(stderr, "FAIL: %s (line %d)\n", (msg), __LINE__);      \
        return 1;                                                            \
    }                                                                        \
} while (0)

#define MUST(cond, msg) do {                                                 \
    if (!(cond)) {                                                           \
        std::fprintf(stderr, "FAIL: %s (line %d)\n", (msg), __LINE__);      \
        std::exit(1);                                                        \
    }                                                                        \
} while (0)

typedef int (*ds4_tp_devcopy_fn)(void *, const void *, uint64_t);
extern "C" void ds4_tp_set_devcopy(ds4_tp_devcopy_fn) {}

extern "C" int ds4_gpu_test_shared_gate_up_swiglu_q8_0_stream_tensor(
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        float                   clamp,
        void                   *stream_handle);
extern "C" int ds4_gpu_test_shared_down_q8_0_stream_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        void                   *stream_handle);

enum {
    Q4_K_TYPE = 12,
    QK_K = 256,
    Q4_K_BLOCK_BYTES = 144,
    Q8_0_BLOCK_BYTES = 34,
    N_TOTAL_EXPERT = 8,
    N_USED = 6,
    IN_DIM = 4096,
    ROUTED_MID_DIM = 2048,
    SHARED_MID_DIM = 1024,
    OUT_DIM = 4096,
    WARMUP = 8,
    SAMPLES = 31,
    ITERS = 16,
};

struct fixture {
    unsigned char *model = nullptr;
    uint64_t model_bytes = 0;
    uint64_t routed_gate_off = 0;
    uint64_t routed_up_off = 0;
    uint64_t routed_down_off = 0;
    uint64_t routed_gate_expert_bytes = 0;
    uint64_t routed_gate_row_bytes = 0;
    uint64_t routed_down_expert_bytes = 0;
    uint64_t routed_down_row_bytes = 0;
    uint64_t shared_gate_off = 0;
    uint64_t shared_up_off = 0;
    uint64_t shared_down_off = 0;

    ds4_gpu_tensor out{}, gate{}, up{}, mid{}, down{};
    ds4_gpu_tensor selected{}, weights{}, x{}, add_in{};
    ds4_gpu_tensor shared_gate{}, shared_up{}, shared_mid{}, shared_out{};
};

static void pack_q4k_block(unsigned char *dst, uint32_t seed) {
    dst[0] = 0x00;
    dst[1] = 0x28; /* fp16 1/32 */
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

static void pack_q4k_table(unsigned char *dst, uint32_t experts,
                           uint32_t rows, uint32_t blocks_per_row,
                           uint32_t salt) {
    for (uint32_t e = 0; e < experts; ++e) {
        for (uint32_t row = 0; row < rows; ++row) {
            for (uint32_t b = 0; b < blocks_per_row; ++b) {
                const uint64_t i = ((uint64_t)e * rows + row) *
                                   blocks_per_row + b;
                pack_q4k_block(dst + i * Q4_K_BLOCK_BYTES,
                               salt + 17u * e + 13u * row + 7u * b);
            }
        }
    }
}

static void pack_q8_table(unsigned char *dst, uint32_t rows,
                          uint32_t blocks_per_row, uint32_t salt) {
    for (uint32_t row = 0; row < rows; ++row) {
        for (uint32_t b = 0; b < blocks_per_row; ++b) {
            unsigned char *block = dst +
                ((uint64_t)row * blocks_per_row + b) * Q8_0_BLOCK_BYTES;
            block[0] = 0x00;
            block[1] = 0x28; /* fp16 1/32 */
            for (uint32_t i = 0; i < 32; ++i) {
                const int value = (int)((salt + 11u * row + 7u * b +
                                         3u * i) & 255u) - 128;
                block[2 + i] = (unsigned char)(int8_t)value;
            }
        }
    }
}

static int alloc_tensor(ds4_gpu_tensor *tensor, uint64_t bytes) {
    std::memset(tensor, 0, sizeof(*tensor));
    return ds4_gpu_tensor_alloc_on(tensor, 0, bytes) == 0;
}

static int upload(ds4_gpu_tensor *tensor, const void *src, uint64_t bytes) {
    return ds4_gpu_tensor_write(tensor, 0, src, bytes) != 0;
}

static uint64_t fnv1a64(const void *data, size_t bytes) {
    const unsigned char *p = (const unsigned char *)data;
    uint64_t hash = UINT64_C(1469598103934665603);
    for (size_t i = 0; i < bytes; ++i) {
        hash ^= p[i];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static int run_routed(fixture *f) {
    return ds4_gpu_routed_moe_one_tensor(
        &f->out, &f->gate, &f->up, &f->mid, &f->down,
        f->model, f->model_bytes,
        f->routed_gate_off, f->routed_up_off, f->routed_down_off,
        Q4_K_TYPE, Q4_K_TYPE,
        f->routed_gate_expert_bytes, f->routed_gate_row_bytes,
        f->routed_down_expert_bytes, f->routed_down_row_bytes,
        IN_DIM, ROUTED_MID_DIM, OUT_DIM,
        &f->selected, &f->weights, N_TOTAL_EXPERT, N_USED,
        0.0f, &f->x, &f->add_in, 0, true);
}

static int run_shared_default(fixture *f) {
    return ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(
        &f->shared_gate, &f->shared_up, &f->shared_mid,
        f->model, f->model_bytes, f->shared_gate_off, f->shared_up_off,
        IN_DIM, SHARED_MID_DIM, &f->x, 0.0f) &&
           ds4_gpu_matmul_q8_0_decode_mpp_tensor(
        &f->shared_out, f->model, f->model_bytes, f->shared_down_off,
        SHARED_MID_DIM, OUT_DIM, &f->shared_mid, 1u);
}

static int run_shared_stream(fixture *f, hipStream_t stream) {
    return ds4_gpu_test_shared_gate_up_swiglu_q8_0_stream_tensor(
        &f->shared_gate, &f->shared_up, &f->shared_mid,
        f->model, f->model_bytes, f->shared_gate_off, f->shared_up_off,
        IN_DIM, SHARED_MID_DIM, &f->x, 0.0f, (void *)stream) &&
           ds4_gpu_test_shared_down_q8_0_stream_tensor(
        &f->shared_out, f->model, f->model_bytes, f->shared_down_off,
        SHARED_MID_DIM, OUT_DIM, &f->shared_mid, (void *)stream);
}

enum measure_mode {
    MEASURE_SHARED,
    MEASURE_ROUTED,
    MEASURE_SERIAL,
    MEASURE_CONCURRENT,
};

static double measure(fixture *f, measure_mode mode, hipStream_t side,
                      hipEvent_t input_ready, hipEvent_t producer_ready) {
    MUST(hipDeviceSynchronize() == hipSuccess, "pre-measure synchronize");
    const auto begin = std::chrono::steady_clock::now();
    for (uint32_t i = 0; i < ITERS; ++i) {
        if (mode == MEASURE_SHARED) {
            MUST(run_shared_default(f), "measure shared");
        } else if (mode == MEASURE_ROUTED) {
            MUST(run_routed(f), "measure routed");
        } else if (mode == MEASURE_SERIAL) {
            MUST(run_shared_default(f), "measure serial shared");
            MUST(run_routed(f), "measure serial routed");
        } else {
            MUST(hipEventRecord(input_ready, nullptr) == hipSuccess,
                 "record concurrent input");
            MUST(hipStreamWaitEvent(side, input_ready, 0) == hipSuccess,
                 "wait concurrent input");
            MUST(run_shared_stream(f, side), "measure concurrent shared");
            MUST(run_routed(f), "measure concurrent routed");
            MUST(hipStreamSynchronize(side) == hipSuccess,
                 "join concurrent shared");
        }
        MUST(hipEventRecord(producer_ready, nullptr) == hipSuccess,
             "record producer boundary");
        MUST(hipEventSynchronize(producer_ready) == hipSuccess,
             "wait producer boundary");
    }
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - begin).count() / (double)ITERS;
}

static double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2u];
}

static int set_owned_routes(fixture *f, uint32_t owned) {
    int32_t selected[N_USED] = {};
    float weights[N_USED] = {};
    for (uint32_t i = 0; i < N_USED; ++i) {
        selected[i] = i < owned ? (int32_t)i : 0;
        weights[i] = i < owned ? 1.0f / (float)N_USED : 0.0f;
    }
    return upload(&f->selected, selected, sizeof(selected)) &&
           upload(&f->weights, weights, sizeof(weights));
}

static int verify_concurrent_identity(fixture *f, hipStream_t side,
                                      hipEvent_t input_ready) {
    float serial_shared[OUT_DIM], concurrent_shared[OUT_DIM];
    float serial_out[OUT_DIM], concurrent_out[OUT_DIM];
    MUST(run_shared_default(f), "identity serial shared");
    MUST(run_routed(f), "identity serial routed");
    MUST(hipDeviceSynchronize() == hipSuccess, "identity serial sync");
    MUST(ds4_gpu_tensor_read(&f->shared_out, 0, serial_shared,
                             sizeof(serial_shared)), "read serial shared");
    MUST(ds4_gpu_tensor_read(&f->out, 0, serial_out, sizeof(serial_out)),
         "read serial routed");

    MUST(hipEventRecord(input_ready, nullptr) == hipSuccess,
         "identity input record");
    MUST(hipStreamWaitEvent(side, input_ready, 0) == hipSuccess,
         "identity input wait");
    MUST(run_shared_stream(f, side), "identity concurrent shared");
    MUST(run_routed(f), "identity concurrent routed");
    MUST(hipStreamSynchronize(side) == hipSuccess,
         "identity shared join");
    MUST(hipDeviceSynchronize() == hipSuccess, "identity concurrent sync");
    MUST(ds4_gpu_tensor_read(&f->shared_out, 0, concurrent_shared,
                             sizeof(concurrent_shared)),
         "read concurrent shared");
    MUST(ds4_gpu_tensor_read(&f->out, 0, concurrent_out,
                             sizeof(concurrent_out)),
         "read concurrent routed");

    const uint64_t shared_serial =
        fnv1a64(serial_shared, sizeof(serial_shared));
    const uint64_t shared_concurrent =
        fnv1a64(concurrent_shared, sizeof(concurrent_shared));
    const uint64_t routed_serial = fnv1a64(serial_out, sizeof(serial_out));
    const uint64_t routed_concurrent =
        fnv1a64(concurrent_out, sizeof(concurrent_out));
    std::printf("OVERLAP_IDENTITY shared_serial=%016llx "
                "shared_concurrent=%016llx routed_serial=%016llx "
                "routed_concurrent=%016llx bitwise=%s\n",
                (unsigned long long)shared_serial,
                (unsigned long long)shared_concurrent,
                (unsigned long long)routed_serial,
                (unsigned long long)routed_concurrent,
                std::memcmp(serial_shared, concurrent_shared,
                            sizeof(serial_shared)) == 0 &&
                        std::memcmp(serial_out, concurrent_out,
                                    sizeof(serial_out)) == 0
                    ? "true" : "false");
    return std::memcmp(serial_shared, concurrent_shared,
                       sizeof(serial_shared)) == 0 &&
           std::memcmp(serial_out, concurrent_out, sizeof(serial_out)) == 0;
}

int main() {
    CHECK(unsetenv("DS4_ROCM_Q8_DECODE_PAIR_DP4A") == 0,
          "select exact Q8 pair path");
    CHECK(setenv("DS4_ROCM_SHARED_GU_SWIGLU_FUSE", "1", 1) == 0,
          "select benchmark-default shared gate/up path");
    int devices = 0;
    CHECK(hipGetDeviceCount(&devices) == hipSuccess && devices > 0,
          "ROCm device required");
    ds4_gpu_config config{};
    config.n_gpus = 1;
    config.device_indices[0] = 0;
    CHECK(ds4_gpu_init_multi(&config), "initialize ROCm backend");
    CHECK(setenv("DS4_ROCM_TP_SKIP_UNOWNED", "1", 1) == 0,
          "enable exact unowned skip");

    fixture f;
    const uint64_t in_blocks = IN_DIM / QK_K;
    const uint64_t routed_mid_blocks = ROUTED_MID_DIM / QK_K;
    f.routed_gate_row_bytes = in_blocks * Q4_K_BLOCK_BYTES;
    f.routed_down_row_bytes = routed_mid_blocks * Q4_K_BLOCK_BYTES;
    f.routed_gate_expert_bytes =
        ROUTED_MID_DIM * f.routed_gate_row_bytes;
    f.routed_down_expert_bytes = OUT_DIM * f.routed_down_row_bytes;
    f.routed_gate_off = 0;
    f.routed_up_off = N_TOTAL_EXPERT * f.routed_gate_expert_bytes;
    f.routed_down_off = f.routed_up_off +
                        N_TOTAL_EXPERT * f.routed_gate_expert_bytes;
    const uint64_t routed_end = f.routed_down_off +
                                N_TOTAL_EXPERT * f.routed_down_expert_bytes;
    const uint64_t shared_row_bytes = (IN_DIM / 32u) * Q8_0_BLOCK_BYTES;
    const uint64_t shared_weight_bytes = SHARED_MID_DIM * shared_row_bytes;
    const uint64_t shared_down_row_bytes =
        (SHARED_MID_DIM / 32u) * Q8_0_BLOCK_BYTES;
    const uint64_t shared_down_weight_bytes =
        OUT_DIM * shared_down_row_bytes;
    f.shared_gate_off = routed_end;
    f.shared_up_off = f.shared_gate_off + shared_weight_bytes;
    f.shared_down_off = f.shared_up_off + shared_weight_bytes;
    f.model_bytes = f.shared_down_off + shared_down_weight_bytes;
    f.model = (unsigned char *)std::malloc((size_t)f.model_bytes);
    CHECK(f.model, "allocate synthetic model");
    pack_q4k_table(f.model + f.routed_gate_off, N_TOTAL_EXPERT,
                   ROUTED_MID_DIM, (uint32_t)in_blocks, 11u);
    pack_q4k_table(f.model + f.routed_up_off, N_TOTAL_EXPERT,
                   ROUTED_MID_DIM, (uint32_t)in_blocks, 37u);
    pack_q4k_table(f.model + f.routed_down_off, N_TOTAL_EXPERT,
                   OUT_DIM, (uint32_t)routed_mid_blocks, 73u);
    pack_q8_table(f.model + f.shared_gate_off, SHARED_MID_DIM,
                  IN_DIM / 32u, 19u);
    pack_q8_table(f.model + f.shared_up_off, SHARED_MID_DIM,
                  IN_DIM / 32u, 53u);
    pack_q8_table(f.model + f.shared_down_off, OUT_DIM,
                  SHARED_MID_DIM / 32u, 89u);
    CHECK(ds4_gpu_set_model_map(f.model, f.model_bytes),
          "install synthetic model");

    const uint64_t routed_values = (uint64_t)N_USED * ROUTED_MID_DIM;
    CHECK(alloc_tensor(&f.out, OUT_DIM * sizeof(float)), "allocate routed out");
    CHECK(alloc_tensor(&f.gate, routed_values * sizeof(float)),
          "allocate routed gate");
    CHECK(alloc_tensor(&f.up, routed_values * sizeof(float)),
          "allocate routed up");
    CHECK(alloc_tensor(&f.mid, routed_values * sizeof(float)),
          "allocate routed mid");
    CHECK(alloc_tensor(&f.down, (uint64_t)N_USED * OUT_DIM * sizeof(float)),
          "allocate routed down");
    CHECK(alloc_tensor(&f.selected, N_USED * sizeof(int32_t)),
          "allocate selected");
    CHECK(alloc_tensor(&f.weights, N_USED * sizeof(float)),
          "allocate weights");
    CHECK(alloc_tensor(&f.x, IN_DIM * sizeof(float)), "allocate input");
    CHECK(alloc_tensor(&f.add_in, OUT_DIM * sizeof(float)), "allocate addend");
    CHECK(alloc_tensor(&f.shared_gate, SHARED_MID_DIM * sizeof(float)),
          "allocate shared gate");
    CHECK(alloc_tensor(&f.shared_up, SHARED_MID_DIM * sizeof(float)),
          "allocate shared up");
    CHECK(alloc_tensor(&f.shared_mid, SHARED_MID_DIM * sizeof(float)),
          "allocate shared mid");
    CHECK(alloc_tensor(&f.shared_out, OUT_DIM * sizeof(float)),
          "allocate shared out");

    float hx[IN_DIM], hadd[OUT_DIM];
    for (uint32_t i = 0; i < IN_DIM; ++i)
        hx[i] = (float)((int)(i % 31u) - 15) * 0.03125f;
    for (uint32_t i = 0; i < OUT_DIM; ++i)
        hadd[i] = (float)((int)(i % 19u) - 9) * 0.00390625f;
    CHECK(upload(&f.x, hx, sizeof(hx)), "upload input");
    CHECK(upload(&f.add_in, hadd, sizeof(hadd)), "upload addend");

    hipStream_t side = nullptr;
    hipEvent_t input_ready = nullptr, producer_ready = nullptr;
    CHECK(hipStreamCreateWithFlags(&side, hipStreamNonBlocking) == hipSuccess,
          "create side stream");
    CHECK(hipEventCreateWithFlags(&input_ready, hipEventDisableTiming) ==
              hipSuccess,
          "create input event");
    CHECK(hipEventCreateWithFlags(&producer_ready, hipEventDisableTiming) ==
              hipSuccess,
          "create producer event");

    bool all_identity = true;
    bool representative_pass = false;
    const uint32_t occupancies[] = {0u, 3u, 6u};
    for (uint32_t owned : occupancies) {
        CHECK(set_owned_routes(&f, owned), "set owned routes");
        for (uint32_t i = 0; i < WARMUP; ++i) {
            CHECK(run_shared_default(&f), "warm shared");
            CHECK(run_routed(&f), "warm routed");
        }
        CHECK(hipDeviceSynchronize() == hipSuccess, "warmup synchronize");
        const bool identity = verify_concurrent_identity(&f, side, input_ready);
        all_identity = all_identity && identity;

        std::vector<double> shared, routed, serial, concurrent;
        shared.reserve(SAMPLES);
        routed.reserve(SAMPLES);
        serial.reserve(SAMPLES);
        concurrent.reserve(SAMPLES);
        for (uint32_t sample = 0; sample < SAMPLES; ++sample) {
            double s, r, q, c;
            if ((sample & 1u) == 0u) {
                s = measure(&f, MEASURE_SHARED, side,
                            input_ready, producer_ready);
                r = measure(&f, MEASURE_ROUTED, side,
                            input_ready, producer_ready);
                q = measure(&f, MEASURE_SERIAL, side,
                            input_ready, producer_ready);
                c = measure(&f, MEASURE_CONCURRENT, side,
                            input_ready, producer_ready);
            } else {
                c = measure(&f, MEASURE_CONCURRENT, side,
                            input_ready, producer_ready);
                q = measure(&f, MEASURE_SERIAL, side,
                            input_ready, producer_ready);
                r = measure(&f, MEASURE_ROUTED, side,
                            input_ready, producer_ready);
                s = measure(&f, MEASURE_SHARED, side,
                            input_ready, producer_ready);
            }
            shared.push_back(s);
            routed.push_back(r);
            serial.push_back(q);
            concurrent.push_back(c);
            std::printf("OVERLAP_SAMPLE owned=%u sample=%u shared_ms=%.6f "
                        "routed_ms=%.6f serial_ms=%.6f concurrent_ms=%.6f\n",
                        owned, sample + 1u, s, r, q, c);
        }
        const double shared_median = median(shared);
        const double routed_median = median(routed);
        const double serial_median = median(serial);
        const double concurrent_median = median(concurrent);
        const double smaller = std::min(shared_median, routed_median);
        const double hidden = shared_median + routed_median - concurrent_median;
        const double efficiency = smaller > 0.0 ? hidden / smaller : 0.0;
        const bool pass = identity && efficiency >= 0.80;
        if (owned == 3u) representative_pass = pass;
        std::printf("OVERLAP_RESULT owned=%u samples=%u iters=%u "
                    "shared_ms=%.6f routed_ms=%.6f serial_ms=%.6f "
                    "concurrent_ms=%.6f hidden_ms=%.6f efficiency=%.4f "
                    "identity=%s verdict=%s\n",
                    owned, SAMPLES, ITERS, shared_median, routed_median,
                    serial_median, concurrent_median, hidden, efficiency,
                    identity ? "true" : "false",
                    pass ? "PASS" : "NO_GO");
    }

    (void)hipEventDestroy(producer_ready);
    (void)hipEventDestroy(input_ready);
    (void)hipStreamDestroy(side);
    ds4_gpu_tensor_free_in_place(&f.shared_out);
    ds4_gpu_tensor_free_in_place(&f.shared_mid);
    ds4_gpu_tensor_free_in_place(&f.shared_up);
    ds4_gpu_tensor_free_in_place(&f.shared_gate);
    ds4_gpu_tensor_free_in_place(&f.add_in);
    ds4_gpu_tensor_free_in_place(&f.x);
    ds4_gpu_tensor_free_in_place(&f.weights);
    ds4_gpu_tensor_free_in_place(&f.selected);
    ds4_gpu_tensor_free_in_place(&f.down);
    ds4_gpu_tensor_free_in_place(&f.mid);
    ds4_gpu_tensor_free_in_place(&f.up);
    ds4_gpu_tensor_free_in_place(&f.gate);
    ds4_gpu_tensor_free_in_place(&f.out);
    std::free(f.model);
    ds4_gpu_cleanup();
    std::printf("OVERLAP_GATE representative_owned=3 threshold=0.8000 "
                "identity_all=%s verdict=%s\n",
                all_identity ? "true" : "false",
                all_identity && representative_pass ? "PASS" : "NO_GO");
    return all_identity ? 0 : 1;
}
