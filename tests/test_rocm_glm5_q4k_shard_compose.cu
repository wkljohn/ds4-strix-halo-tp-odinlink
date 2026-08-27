#include "ds4_gpu.h"
#include "ds4_gpu_mgpu.h"
#include "tests/glm5_gguf_test.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CHECK(expr, message) do {                                           \
    if (!(expr)) {                                                          \
        std::fprintf(stderr, "FAIL %s (line %d)\n", message, __LINE__);    \
        return false;                                                       \
    }                                                                       \
} while (0)

typedef int (*ds4_tp_devcopy_fn)(void *, const void *, uint64_t);
extern "C" void ds4_tp_set_devcopy(ds4_tp_devcopy_fn) {}
extern "C" int ds4_gpu_routed_moe_batch_q4k_direct_control(
        ds4_gpu_tensor *, ds4_gpu_tensor *, ds4_gpu_tensor *,
        ds4_gpu_tensor *, ds4_gpu_tensor *, const void *, const void *,
        const void *, uint64_t, uint64_t, uint64_t, uint64_t,
        const ds4_gpu_tensor *, const ds4_gpu_tensor *, uint32_t, uint32_t,
        float, const ds4_gpu_tensor *, uint32_t, uint32_t, uint32_t);

namespace {

constexpr uint32_t kInput = 4096;
constexpr uint32_t kFullMid = 2048;
constexpr uint32_t kHalfMid = 1024;
constexpr uint32_t kOutput = 4096;
constexpr uint32_t kTotalExperts = 288;
constexpr uint32_t kUsed = 8;
// Shipping Q4_K prefill selects the sorted WMMA path at 32 tokens.
constexpr uint32_t kTokens = 32;
constexpr uint32_t kQ4Block = 144;
constexpr uint32_t kQk = 256;
constexpr uint32_t kExpertIds[kUsed] = {0, 1, 17, 63, 127, 191, 255, 287};
static_assert((uint64_t)kTokens * kOutput <= UINT32_MAX,
              "ds4_gpu_add_tensor element count must fit uint32_t");

struct RuntimeGuard {
    bool active = false;
    ~RuntimeGuard() {
        if (active) ds4_gpu_cleanup();
    }
};

struct DeviceWeights {
    void *gate_full = nullptr;
    void *up_full = nullptr;
    void *down_full = nullptr;
    void *gate_half[2] = {};
    void *up_half[2] = {};
    void *down_half[2] = {};

    ~DeviceWeights() {
        auto release = [](void *ptr, const char *name) {
            if (!ptr) return;
            const hipError_t rc = hipFree(ptr);
            if (rc != hipSuccess)
                std::fprintf(stderr, "WARN hipFree %s: %s\n", name,
                             hipGetErrorString(rc));
        };
        for (uint32_t half = 0; half < 2; ++half) {
            release(down_half[half], "down_half");
            release(up_half[half], "up_half");
            release(gate_half[half], "gate_half");
        }
        release(down_full, "down_full");
        release(up_full, "up_full");
        release(gate_full, "gate_full");
    }
};

struct ComponentTensors {
    ds4_gpu_tensor selected = {}, weights = {}, input = {};
    ds4_gpu_tensor gate = {}, up = {}, mid = {}, down = {};
    ds4_gpu_tensor out_full = {}, out_half0 = {}, out_half1 = {}, out_sum = {};

    ~ComponentTensors() {
        ds4_gpu_tensor_free_in_place(&out_sum);
        ds4_gpu_tensor_free_in_place(&out_half1);
        ds4_gpu_tensor_free_in_place(&out_half0);
        ds4_gpu_tensor_free_in_place(&out_full);
        ds4_gpu_tensor_free_in_place(&down);
        ds4_gpu_tensor_free_in_place(&mid);
        ds4_gpu_tensor_free_in_place(&up);
        ds4_gpu_tensor_free_in_place(&gate);
        ds4_gpu_tensor_free_in_place(&input);
        ds4_gpu_tensor_free_in_place(&weights);
        ds4_gpu_tensor_free_in_place(&selected);
    }
};

bool alloc_tensor(ds4_gpu_tensor &tensor, uint64_t bytes) {
    std::memset(&tensor, 0, sizeof(tensor));
    return ds4_gpu_tensor_alloc_on(&tensor, 0, bytes) == 0;
}

bool upload(ds4_gpu_tensor &tensor, const void *src, uint64_t bytes) {
    return alloc_tensor(tensor, bytes) &&
           ds4_gpu_tensor_write(&tensor, 0, src, bytes) != 0;
}

uint64_t fnv1a64(const void *data, uint64_t bytes) {
    const auto *p = static_cast<const uint8_t *>(data);
    uint64_t hash = UINT64_C(1469598103934665603);
    for (uint64_t i = 0; i < bytes; ++i) {
        hash ^= p[i];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

bool tensor_range(const Glm5TestGGUF &gguf, uint64_t offset, uint64_t bytes) {
    return offset <= gguf.size && bytes <= gguf.size - offset;
}

bool gather_experts(const Glm5TestGGUF &gguf, uint64_t offset,
                    uint64_t expert_bytes, std::vector<uint8_t> &compact) {
    compact.resize((size_t)kUsed * expert_bytes);
    for (uint32_t slot = 0; slot < kUsed; ++slot) {
        const uint64_t source = offset + (uint64_t)kExpertIds[slot] * expert_bytes;
        CHECK(tensor_range(gguf, source, expert_bytes),
              "real GGUF expert range");
        std::memcpy(compact.data() + (uint64_t)slot * expert_bytes,
                    gguf.map + source, (size_t)expert_bytes);
    }
    return true;
}

void pack_gate_half(const std::vector<uint8_t> &full,
                    uint64_t full_expert_bytes, uint64_t half_expert_bytes,
                    uint32_t half, std::vector<uint8_t> &packed) {
    packed.resize((size_t)kUsed * half_expert_bytes);
    for (uint32_t expert = 0; expert < kUsed; ++expert) {
        std::memcpy(packed.data() + (uint64_t)expert * half_expert_bytes,
                    full.data() + (uint64_t)expert * full_expert_bytes +
                        (uint64_t)half * half_expert_bytes,
                    (size_t)half_expert_bytes);
    }
}

void pack_down_half(const std::vector<uint8_t> &full,
                    uint64_t full_expert_bytes, uint64_t full_row_bytes,
                    uint64_t half_expert_bytes, uint64_t half_row_bytes,
                    uint32_t half, std::vector<uint8_t> &packed) {
    packed.resize((size_t)kUsed * half_expert_bytes);
    for (uint32_t expert = 0; expert < kUsed; ++expert) {
        for (uint32_t row = 0; row < kOutput; ++row) {
            const uint64_t source = (uint64_t)expert * full_expert_bytes +
                                    (uint64_t)row * full_row_bytes +
                                    (uint64_t)half * half_row_bytes;
            const uint64_t destination =
                (uint64_t)expert * half_expert_bytes +
                (uint64_t)row * half_row_bytes;
            std::memcpy(packed.data() + destination, full.data() + source,
                        (size_t)half_row_bytes);
        }
    }
}

bool launch(ds4_gpu_tensor &out, ds4_gpu_tensor &gate,
            ds4_gpu_tensor &up, ds4_gpu_tensor &mid,
            ds4_gpu_tensor &down, const void *gate_weight,
            const void *up_weight, const void *down_weight,
            uint64_t gate_expert_bytes, uint64_t gate_row_bytes,
            uint64_t down_expert_bytes, uint64_t down_row_bytes,
            const ds4_gpu_tensor &selected, const ds4_gpu_tensor &weights,
            const ds4_gpu_tensor &input, uint32_t mid_dim) {
    const int launched = ds4_gpu_routed_moe_batch_q4k_direct_control(
        &out, &gate, &up, &mid, &down,
        gate_weight, up_weight, down_weight,
        gate_expert_bytes, gate_row_bytes,
        down_expert_bytes, down_row_bytes,
        &selected, &weights, kUsed, kUsed, 10.0f, &input,
        3u, kTokens, mid_dim);
    if (!launched || !ds4_gpu_synchronize()) return false;
    return hipGetLastError() == hipSuccess;
}

bool run_test() {
    const char *model = std::getenv("DS4_GLM5_MODEL");
    CHECK(model && model[0], "DS4_GLM5_MODEL");
    Glm5TestGGUF gguf;
    CHECK(gguf.open_file(model), "open GLM5 GGUF file");

    uint64_t gate_offset = 0, up_offset = 0, down_offset = 0;
    CHECK(gguf.tensor("blk.3.ffn_gate_exps.weight",
                      {kInput, kFullMid, kTotalExperts}, 12u, gate_offset) &&
          gguf.tensor("blk.3.ffn_up_exps.weight",
                      {kInput, kFullMid, kTotalExperts}, 12u, up_offset) &&
          gguf.tensor("blk.3.ffn_down_exps.weight",
                      {kFullMid, kOutput, kTotalExperts}, 12u, down_offset),
          "bind same-GGUF layer-3 routed Q4_K tensors");

    const uint64_t gate_row_bytes = (kInput / kQk) * kQ4Block;
    const uint64_t full_gate_expert = (uint64_t)kFullMid * gate_row_bytes;
    const uint64_t half_gate_expert = (uint64_t)kHalfMid * gate_row_bytes;
    const uint64_t full_down_row = (kFullMid / kQk) * kQ4Block;
    const uint64_t half_down_row = (kHalfMid / kQk) * kQ4Block;
    const uint64_t full_down_expert = (uint64_t)kOutput * full_down_row;
    const uint64_t half_down_expert = (uint64_t)kOutput * half_down_row;
    CHECK(full_gate_expert == full_down_expert &&
          half_gate_expert == half_down_expert,
          "GLM5 2048/1024 expert byte symmetry");

    // The split is exactly four Q4_K/Q8_K blocks. Production dynamically
    // quantizes each 256-value intermediate block independently, so both
    // halves preserve the full path's quantization scales. A future change to
    // row-wide activation scaling invalidates this split-consistency premise.

    std::vector<uint8_t> gate_full, up_full, down_full;
    std::vector<uint8_t> gate_half[2], up_half[2], down_half[2];
    CHECK(gather_experts(gguf, gate_offset, full_gate_expert, gate_full) &&
          gather_experts(gguf, up_offset, full_gate_expert, up_full) &&
          gather_experts(gguf, down_offset, full_down_expert, down_full),
          "gather eight real GLM5 experts");
    for (uint32_t half = 0; half < 2; ++half) {
        pack_gate_half(gate_full, full_gate_expert, half_gate_expert,
                       half, gate_half[half]);
        pack_gate_half(up_full, full_gate_expert, half_gate_expert,
                       half, up_half[half]);
        pack_down_half(down_full, full_down_expert, full_down_row,
                       half_down_expert, half_down_row,
                       half, down_half[half]);
    }

    CHECK(setenv("DS4_ROCM_Q4K_KSHARD_RESEARCH", "1", 1) == 0 &&
          setenv("DS4_ROCM_Q4K_WMMA_PAIR_GATE_UP", "1", 1) == 0 &&
          setenv("DS4_ROCM_Q4K_WMMA_FUSE_MID", "1", 1) == 0 &&
          unsetenv("DS4_ROCM_Q4K_WMMA_LAYER_LOG") == 0 &&
          unsetenv("DS4_ROCM_TP_PREFILL_SKIP_UNOWNED") == 0 &&
          unsetenv("DS4_ROCM_TP_SKIP_UNOWNED") == 0,
          "enable isolated packed arithmetic control");
    int devices = 0;
    CHECK(hipGetDeviceCount(&devices) == hipSuccess && devices > 0,
          "gfx1151 device available");
    ds4_gpu_config config = {};
    config.n_gpus = 1;
    config.device_indices[0] = 0;
    RuntimeGuard runtime;
    CHECK(ds4_gpu_init_multi(&config), "initialize gfx1151");
    runtime.active = true;

    DeviceWeights device;
    const uint64_t full_table_bytes = (uint64_t)kUsed * full_gate_expert;
    const uint64_t half_table_bytes = (uint64_t)kUsed * half_gate_expert;
    CHECK(hipMalloc(&device.gate_full, full_table_bytes) == hipSuccess &&
          hipMalloc(&device.up_full, full_table_bytes) == hipSuccess &&
          hipMalloc(&device.down_full, full_table_bytes) == hipSuccess,
          "allocate compact full expert tables");
    CHECK(hipMemcpy(device.gate_full, gate_full.data(), full_table_bytes,
                    hipMemcpyHostToDevice) == hipSuccess &&
          hipMemcpy(device.up_full, up_full.data(), full_table_bytes,
                    hipMemcpyHostToDevice) == hipSuccess &&
          hipMemcpy(device.down_full, down_full.data(), full_table_bytes,
                    hipMemcpyHostToDevice) == hipSuccess,
          "upload compact full expert tables");
    for (uint32_t half = 0; half < 2; ++half) {
        CHECK(hipMalloc(&device.gate_half[half], half_table_bytes) == hipSuccess &&
              hipMalloc(&device.up_half[half], half_table_bytes) == hipSuccess &&
              hipMalloc(&device.down_half[half], half_table_bytes) == hipSuccess,
              "allocate compact half expert tables");
        CHECK(hipMemcpy(device.gate_half[half], gate_half[half].data(),
                        half_table_bytes, hipMemcpyHostToDevice) == hipSuccess &&
              hipMemcpy(device.up_half[half], up_half[half].data(),
                        half_table_bytes, hipMemcpyHostToDevice) == hipSuccess &&
              hipMemcpy(device.down_half[half], down_half[half].data(),
                        half_table_bytes, hipMemcpyHostToDevice) == hipSuccess,
              "upload compact half expert tables");
    }

    std::vector<int32_t> selected((size_t)kTokens * kUsed);
    std::vector<float> route_weights((size_t)kTokens * kUsed);
    std::vector<float> input((size_t)kTokens * kInput);
    for (uint32_t token = 0; token < kTokens; ++token) {
        float sum = 0.0f;
        for (uint32_t slot = 0; slot < kUsed; ++slot) {
            // A fixed coprime permutation makes sorted-pair preparation
            // non-trivial while every token still exercises all eight slots.
            const uint32_t compact_slot = (slot * 5u + token * 3u) & 7u;
            selected[(size_t)token * kUsed + slot] = (int32_t)compact_slot;
            const float value = (float)(kUsed - slot + (token & 1u)) * 0.03125f;
            route_weights[(size_t)token * kUsed + slot] = value;
            sum += value;
        }
        for (uint32_t slot = 0; slot < kUsed; ++slot)
            route_weights[(size_t)token * kUsed + slot] /= sum;
        for (uint32_t i = 0; i < kInput; ++i) {
            const int value = (int)((i * 17u + token * 13u) % 127u) - 63;
            input[(size_t)token * kInput + i] = (float)value / 256.0f;
        }
    }

    ComponentTensors tensors;
    ds4_gpu_tensor &selected_gpu = tensors.selected;
    ds4_gpu_tensor &weights_gpu = tensors.weights;
    ds4_gpu_tensor &input_gpu = tensors.input;
    ds4_gpu_tensor &gate = tensors.gate;
    ds4_gpu_tensor &up = tensors.up;
    ds4_gpu_tensor &mid = tensors.mid;
    ds4_gpu_tensor &down = tensors.down;
    ds4_gpu_tensor &out_full = tensors.out_full;
    ds4_gpu_tensor &out_half0 = tensors.out_half0;
    ds4_gpu_tensor &out_half1 = tensors.out_half1;
    ds4_gpu_tensor &out_sum = tensors.out_sum;
    const uint64_t pair_count = (uint64_t)kTokens * kUsed;
    const uint64_t mid_bytes = pair_count * kFullMid * sizeof(float);
    const uint64_t down_bytes = pair_count * kOutput * sizeof(float);
    const uint64_t out_bytes = (uint64_t)kTokens * kOutput * sizeof(float);
    CHECK(upload(selected_gpu, selected.data(),
                 pair_count * sizeof(int32_t)) &&
          upload(weights_gpu, route_weights.data(),
                 pair_count * sizeof(float)) &&
          upload(input_gpu, input.data(),
                 (uint64_t)input.size() * sizeof(float)) &&
          alloc_tensor(gate, mid_bytes) && alloc_tensor(up, mid_bytes) &&
          alloc_tensor(mid, mid_bytes) && alloc_tensor(down, down_bytes) &&
          alloc_tensor(out_full, out_bytes) &&
          alloc_tensor(out_half0, out_bytes) &&
          alloc_tensor(out_half1, out_bytes) &&
          alloc_tensor(out_sum, out_bytes),
          "allocate GLM5 MoE component tensors");

    CHECK(launch(out_full, gate, up, mid, down,
                 device.gate_full, device.up_full, device.down_full,
                 full_gate_expert, gate_row_bytes,
                 full_down_expert, full_down_row,
                 selected_gpu, weights_gpu, input_gpu, kFullMid),
          "execute full 2048 top-8 Q4_K control");
    CHECK(launch(out_half0, gate, up, mid, down,
                 device.gate_half[0], device.up_half[0], device.down_half[0],
                 half_gate_expert, gate_row_bytes,
                 half_down_expert, half_down_row,
                 selected_gpu, weights_gpu, input_gpu, kHalfMid) &&
          launch(out_half1, gate, up, mid, down,
                 device.gate_half[1], device.up_half[1], device.down_half[1],
                 half_gate_expert, gate_row_bytes,
                 half_down_expert, half_down_row,
                 selected_gpu, weights_gpu, input_gpu, kHalfMid),
          "execute both 1024 top-8 Q4_K shards");
    CHECK(ds4_gpu_add_tensor(&out_sum, &out_half0, &out_half1,
                             kTokens * kOutput) &&
          ds4_gpu_synchronize(), "compose both rank outputs");

    std::vector<float> reference((size_t)kTokens * kOutput);
    std::vector<float> candidate((size_t)kTokens * kOutput);
    std::vector<float> half0((size_t)kTokens * kOutput);
    std::vector<float> half1((size_t)kTokens * kOutput);
    CHECK(ds4_gpu_tensor_read(&out_full, 0, reference.data(), out_bytes) &&
          ds4_gpu_tensor_read(&out_sum, 0, candidate.data(), out_bytes) &&
          ds4_gpu_tensor_read(&out_half0, 0, half0.data(), out_bytes) &&
          ds4_gpu_tensor_read(&out_half1, 0, half1.data(), out_bytes),
          "read full, individual-half, and composed outputs");
    long double error_sq = 0.0L, reference_sq = 0.0L;
    long double half0_sq = 0.0L, half1_sq = 0.0L;
    long double half_delta_sq = 0.0L;
    long double half0_full_delta_sq = 0.0L, half1_full_delta_sq = 0.0L;
    float max_abs = 0.0f, max_rel = 0.0f;
    uint64_t changed = 0, nonfinite = 0;
    for (size_t i = 0; i < reference.size(); ++i) {
        const float ref = reference[i], got = candidate[i];
        const float h0 = half0[i], h1 = half1[i];
        if (!std::isfinite(ref) || !std::isfinite(got) ||
            !std::isfinite(h0) || !std::isfinite(h1)) ++nonfinite;
        const float error = std::fabs(got - ref);
        max_abs = std::max(max_abs, error);
        max_rel = std::max(max_rel, error / std::max(1.0f, std::fabs(ref)));
        error_sq += (long double)error * error;
        reference_sq += (long double)ref * ref;
        half0_sq += (long double)h0 * h0;
        half1_sq += (long double)h1 * h1;
        const long double half_delta = (long double)h0 - h1;
        const long double half0_full_delta = (long double)h0 - ref;
        const long double half1_full_delta = (long double)h1 - ref;
        half_delta_sq += half_delta * half_delta;
        half0_full_delta_sq += half0_full_delta * half0_full_delta;
        half1_full_delta_sq += half1_full_delta * half1_full_delta;
        changed += std::memcmp(&ref, &got, sizeof(float)) != 0;
    }
    const double nmse = reference_sq == 0.0L
        ? (double)error_sq : (double)(error_sq / reference_sq);
    const bool independent_halves =
        half0_sq > 1.0e-12L && half1_sq > 1.0e-12L &&
        half_delta_sq > 1.0e-12L &&
        half0_full_delta_sq > 1.0e-12L &&
        half1_full_delta_sq > 1.0e-12L;
    const bool pass = nonfinite == 0 && reference_sq > 1.0e-12L &&
                      independent_halves &&
                      max_abs <= 3.0e-7f && max_rel <= 3.0e-7f &&
                      nmse <= 1.0e-11;
    const uint64_t gate_weight_fnv = fnv1a64(gate_full.data(),
                                              gate_full.size());
    const uint64_t up_weight_fnv = fnv1a64(up_full.data(), up_full.size());
    const uint64_t down_weight_fnv = fnv1a64(down_full.data(),
                                              down_full.size());
    std::fprintf(stderr,
        "GLM5 Q4_K shard split-consistency layer=3 descriptor_experts=288 "
        "compact_table=8 gathered_ids=0,1,17,63,127,191,255,287 "
        "tokens=%u changed_telemetry=%llu nonfinite=%llu "
        "independent_halves=%d max_abs=%.9g max_rel=%.9g nmse=%.9g "
        "gate_weight_fnv=%016llx up_weight_fnv=%016llx "
        "down_weight_fnv=%016llx full_fnv=%016llx shard_fnv=%016llx\n",
        kTokens, (unsigned long long)changed, (unsigned long long)nonfinite,
        independent_halves, max_abs, max_rel, nmse,
        (unsigned long long)gate_weight_fnv,
        (unsigned long long)up_weight_fnv,
        (unsigned long long)down_weight_fnv,
        (unsigned long long)fnv1a64(reference.data(), out_bytes),
        (unsigned long long)fnv1a64(candidate.data(), out_bytes));
    CHECK(pass, "same-GGUF 1024/1024 top-8 numerical envelope");
    std::fprintf(stderr,
                 "PASS same-GGUF GLM5 Q4_K top-8 shard split-consistency\n");
    return true;
}

}  // namespace

int main() { return run_test() ? 0 : 1; }
