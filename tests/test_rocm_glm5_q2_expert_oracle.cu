/* Same-GGUF component oracle for GLM-5.3 mixed IQ2_XXS/Q2_K routed MoE.
 *
 * The independent reference dequantizes one real expert with llama.cpp's
 * ggml-base implementation and evaluates every dot product on the host. The
 * production backend is exercised at decode, scalar-batch, and top-8 hot-WMMA
 * shapes. This is a diagnostic ranking gate, not a promotion-quality oracle.
 */
#include "ds4_gpu.h"
#include "tests/glm5_gguf_test.hpp"

#include <hip/hip_runtime.h>
#include <dlfcn.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

/* ABI-only declarations for the pinned llama.cpp dequantizer entry points.
 * Keeping these tiny layouts local avoids a build dependency on an external
 * llama.cpp source checkout; static assertions bind the expected ABI. */
constexpr uint32_t kQuantBlock = 256;
struct block_iq2_xxs {
    uint16_t d;
    uint16_t qs[kQuantBlock / 8];
};
struct block_q2_K {
    uint8_t scales[kQuantBlock / 16];
    uint8_t qs[kQuantBlock / 4];
    uint16_t d;
    uint16_t dmin;
};
struct block_q8_K {
    float d;
    int8_t qs[kQuantBlock];
    int16_t bsums[kQuantBlock / 16];
};
static_assert(sizeof(block_iq2_xxs) == 66, "llama.cpp IQ2_XXS ABI changed");
static_assert(sizeof(block_q2_K) == 84, "llama.cpp Q2_K ABI changed");
static_assert(sizeof(block_q8_K) == 292, "Q8_K intermediate ABI changed");

using deq_iq2_fn = void (*)(const block_iq2_xxs *, float *, int64_t);
using deq_q2_fn = void (*)(const block_q2_K *, float *, int64_t);
using ds4_tp_devcopy_fn = int (*)(void *, const void *, uint64_t);
extern "C" void ds4_tp_set_devcopy(ds4_tp_devcopy_fn) {}

namespace {
constexpr uint32_t kExpert = 17;
constexpr uint32_t kIn = 4096;
constexpr uint32_t kMid = 2048;
constexpr uint32_t kOut = 4096;
constexpr uint64_t kIq2RowBytes = 16u * 66u;
constexpr uint64_t kQ2RowBytes = 8u * 84u;

struct Metrics {
    double nrmse = 0.0;
    double cosine = 0.0;
    double max_abs = 0.0;
    uint32_t nonfinite = 0;
};

struct ReferenceWeights {
    std::vector<float> gate;
    std::vector<float> up;
    std::vector<float> down;
};

struct ReferenceRow {
    std::vector<float> mid;
    std::vector<float> out;
};

static Metrics compare(const float *got, const float *ref, size_t n) {
    long double ref2 = 0.0L, got2 = 0.0L, err2 = 0.0L, dot = 0.0L;
    Metrics m;
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(got[i]) || !std::isfinite(ref[i])) {
            ++m.nonfinite;
            continue;
        }
        const long double g = got[i], r = ref[i], e = g - r;
        ref2 += r * r;
        got2 += g * g;
        err2 += e * e;
        dot += g * r;
        m.max_abs = std::max(m.max_abs, std::fabs((double)e));
    }
    m.nrmse = std::sqrt((double)(err2 / std::max(ref2, 1.0e-30L)));
    m.cosine = (double)(dot / std::sqrt(std::max(ref2 * got2, 1.0e-30L)));
    return m;
}

static float dot_row(const float *a, const float *b, uint32_t n) {
    double sum = 0.0;
    for (uint32_t i = 0; i < n; ++i) sum += (double)a[i] * b[i];
    return (float)sum;
}

static bool read_first_row(const char *path, std::vector<float> *row) {
    if (!path || !path[0] || !row) return false;
    FILE *fp = std::fopen(path, "rb");
    if (!fp) {
        std::fprintf(stderr, "FAIL open captured input %s\n", path);
        return false;
    }
    row->resize(kIn);
    const size_t got = std::fread(row->data(), sizeof(float), kIn, fp);
    std::fclose(fp);
    if (got != kIn) {
        std::fprintf(stderr, "FAIL captured input has %zu floats; need %u\n",
                     got, kIn);
        return false;
    }
    for (uint32_t i = 0; i < kIn; ++i) {
        if (!std::isfinite((*row)[i])) {
            std::fprintf(stderr, "FAIL captured input nonfinite at %u\n", i);
            return false;
        }
    }
    return true;
}

static bool read_exact(const char *path, void *dst, size_t bytes) {
    if (!path || !path[0] || !dst) return false;
    FILE *fp = std::fopen(path, "rb");
    if (!fp) {
        std::fprintf(stderr, "FAIL open fixture %s\n", path);
        return false;
    }
    const bool ok = std::fread(dst, 1, bytes, fp) == bytes;
    const int extra = std::fgetc(fp);
    std::fclose(fp);
    if (!ok || extra != EOF) {
        std::fprintf(stderr, "FAIL fixture %s is not exactly %zu bytes\n",
                     path, bytes);
        return false;
    }
    return true;
}

static ReferenceRow make_reference(const ReferenceWeights &w,
                                   const std::vector<float> &x) {
    ReferenceRow r;
    r.mid.resize(kMid);
    r.out.resize(kOut);
    for (uint32_t i = 0; i < kMid; ++i) {
        float gate = dot_row(w.gate.data() + (uint64_t)i * kIn, x.data(), kIn);
        float up = dot_row(w.up.data() + (uint64_t)i * kIn, x.data(), kIn);
        gate = std::min(gate, 10.0f);
        up = std::max(-10.0f, std::min(up, 10.0f));
        r.mid[i] = gate / (1.0f + std::exp(-gate)) * up;
    }
    for (uint32_t i = 0; i < kOut; ++i) {
        r.out[i] = dot_row(w.down.data() + (uint64_t)i * kMid,
                           r.mid.data(), kMid);
    }
    return r;
}

static ds4_gpu_tensor *alloc_tensor(uint64_t bytes, const char *name) {
    ds4_gpu_tensor *t = ds4_gpu_tensor_alloc(bytes);
    if (!t) std::fprintf(stderr, "FAIL allocate %s (%llu bytes)\n", name,
                         (unsigned long long)bytes);
    return t;
}

static bool write_tensor(ds4_gpu_tensor *t, const void *data,
                         uint64_t bytes, const char *name) {
    if (t && ds4_gpu_tensor_write(t, 0, data, bytes)) return true;
    std::fprintf(stderr, "FAIL write %s\n", name);
    return false;
}

static void print_metrics(const char *case_name, const char *stage,
                          const Metrics &m) {
    std::printf("case=%s stage=%s nrmse=%.9g cosine=%.12g max_abs=%.9g nonfinite=%u\n",
                case_name, stage, m.nrmse, m.cosine, m.max_abs, m.nonfinite);
}

static bool metrics_sane(const Metrics &m) {
    return m.nonfinite == 0 && std::isfinite(m.nrmse) &&
           std::isfinite(m.cosine) && m.cosine >= 0.99;
}

static bool run_decode_case(const char *case_name, const Glm5TestGGUF &g,
                            uint64_t gate_offset, uint64_t up_offset,
                            uint64_t down_offset, const std::vector<float> &x,
                            const ReferenceRow &ref) {
    ds4_gpu_tensor *tx = alloc_tensor(kIn * sizeof(float), "decode x");
    ds4_gpu_tensor *out = alloc_tensor(kOut * sizeof(float), "decode out");
    ds4_gpu_tensor *gate = alloc_tensor(kMid * sizeof(float), "decode gate");
    ds4_gpu_tensor *up = alloc_tensor(kMid * sizeof(float), "decode up");
    ds4_gpu_tensor *mid = alloc_tensor(kMid * sizeof(float), "decode mid");
    ds4_gpu_tensor *down = alloc_tensor(kOut * sizeof(float), "decode down");
    ds4_gpu_tensor *selected = alloc_tensor(sizeof(int32_t), "decode route");
    ds4_gpu_tensor *weights = alloc_tensor(sizeof(float), "decode weight");
    const int32_t route = 0;
    const float weight = 1.0f;
    bool ok = tx && out && gate && up && mid && down && selected && weights &&
        write_tensor(tx, x.data(), kIn * sizeof(float), "decode x") &&
        write_tensor(selected, &route, sizeof(route), "decode route") &&
        write_tensor(weights, &weight, sizeof(weight), "decode weight") &&
        ds4_gpu_routed_moe_one_tensor(
            out, gate, up, mid, down, g.map, g.size,
            gate_offset, up_offset, down_offset, 16u, 10u,
            kMid * kIq2RowBytes, kIq2RowBytes,
            kOut * kQ2RowBytes, kQ2RowBytes,
            kIn, kMid, kOut, selected, weights, 1u, 1u, 10.0f, tx,
            nullptr, 3u, false) && ds4_gpu_synchronize();
    std::vector<float> gpu_mid(kMid), gpu_out(kOut);
    ok = ok && ds4_gpu_tensor_read(mid, 0, gpu_mid.data(), kMid * sizeof(float)) &&
         ds4_gpu_tensor_read(out, 0, gpu_out.data(), kOut * sizeof(float));
    if (ok) {
        const Metrics mm = compare(gpu_mid.data(), ref.mid.data(), kMid);
        const Metrics mo = compare(gpu_out.data(), ref.out.data(), kOut);
        print_metrics(case_name, "mid_vs_canonical", mm);
        print_metrics(case_name, "out_vs_canonical", mo);
        ok = metrics_sane(mm) && metrics_sane(mo) && mo.nrmse < 0.08;
    }
    ds4_gpu_tensor_free(weights); ds4_gpu_tensor_free(selected);
    ds4_gpu_tensor_free(down); ds4_gpu_tensor_free(mid);
    ds4_gpu_tensor_free(up); ds4_gpu_tensor_free(gate);
    ds4_gpu_tensor_free(out); ds4_gpu_tensor_free(tx);
    return ok;
}

struct BatchResult {
    Metrics mid;
    Metrics mid_row_identity;
    Metrics down_only;
    Metrics full;
    Metrics row_identity;
    Metrics mid_quant;
    Metrics down_row_identity;
    uint64_t down_diff_count = 0;
    uint32_t down_max_half_ulp = 0;
    uint64_t out_fnv64 = 0;
    ds4_gpu_test_glm5_q2_dispatch dispatch = {};
};

static uint64_t fnv1a64(const void *data, size_t bytes) {
    const unsigned char *p = (const unsigned char *)data;
    uint64_t h = UINT64_C(1469598103934665603);
    for (size_t i = 0; i < bytes; ++i) {
        h ^= p[i];
        h *= UINT64_C(1099511628211);
    }
    return h;
}

static float half_bits_to_float(uint16_t bits) {
    _Float16 h;
    std::memcpy(&h, &bits, sizeof(bits));
    return (float)h;
}

static bool run_batch_case(const char *case_name, const Glm5TestGGUF &g,
                           uint64_t gate_offset, uint64_t up_offset,
                           uint64_t down_offset, const ReferenceWeights &rw,
                           const std::vector<float> &x,
                           const ReferenceRow &ref, uint32_t n_tokens,
                           const std::vector<float> &slot_weights,
                           bool q8_mid, bool force_scalar,
                           BatchResult *result) {
    const uint32_t n_expert = (uint32_t)slot_weights.size();
    const uint64_t pairs = (uint64_t)n_tokens * n_expert;
    std::vector<float> hx((uint64_t)n_tokens * kIn);
    std::vector<int32_t> routes(pairs, 0);
    std::vector<float> weights(pairs);
    for (uint32_t t = 0; t < n_tokens; ++t) {
        std::memcpy(hx.data() + (uint64_t)t * kIn, x.data(), kIn * sizeof(float));
        for (uint32_t s = 0; s < n_expert; ++s)
            weights[(uint64_t)t * n_expert + s] = slot_weights[s];
    }

    ds4_gpu_tensor *tx = alloc_tensor(hx.size() * sizeof(float), "batch x");
    ds4_gpu_tensor *out = alloc_tensor((uint64_t)n_tokens * kOut * sizeof(float), "batch out");
    ds4_gpu_tensor *gate = alloc_tensor(pairs * kMid * sizeof(float), "batch gate");
    ds4_gpu_tensor *up = alloc_tensor(pairs * kMid * sizeof(float), "batch up");
    ds4_gpu_tensor *mid = alloc_tensor(pairs * kMid * sizeof(float), "batch mid");
    ds4_gpu_tensor *down = alloc_tensor(pairs * kOut * sizeof(float), "batch down");
    ds4_gpu_tensor *selected = alloc_tensor(pairs * sizeof(int32_t), "batch routes");
    ds4_gpu_tensor *gpu_weights = alloc_tensor(pairs * sizeof(float), "batch weights");
    bool ok = tx && out && gate && up && mid && down && selected && gpu_weights &&
        write_tensor(tx, hx.data(), hx.size() * sizeof(float), "batch x") &&
        write_tensor(selected, routes.data(), pairs * sizeof(int32_t), "batch routes") &&
        write_tensor(gpu_weights, weights.data(), pairs * sizeof(float), "batch weights");

    if (q8_mid) setenv("DS4_ROCM_GLM5_BATCH_Q8_MID_DOWN", "1", 1);
    else unsetenv("DS4_ROCM_GLM5_BATCH_Q8_MID_DOWN");
    if (force_scalar) setenv("DS4_ROCM_Q2_DOWN_FORCE_SCALAR", "1", 1);
    else unsetenv("DS4_ROCM_Q2_DOWN_FORCE_SCALAR");
    ds4_gpu_test_reset_glm5_q2_dispatch();
    bool mid_is_f16 = true;
    ok = ok && ds4_gpu_routed_moe_batch_tensor(
        out, gate, up, mid, down, g.map, g.size,
        gate_offset, up_offset, down_offset, 16u, 10u,
        kMid * kIq2RowBytes, kIq2RowBytes,
        kOut * kQ2RowBytes, kQ2RowBytes,
        kIn, kMid, kOut, selected, gpu_weights, 1u, n_expert, 10.0f,
        tx, 3u, n_tokens, &mid_is_f16, false) && ds4_gpu_synchronize();
    ok = ok && !mid_is_f16 &&
         ds4_gpu_test_get_glm5_q2_dispatch(&result->dispatch);

    std::vector<float> gpu_mid(pairs * kMid);
    std::vector<float> gpu_out((uint64_t)n_tokens * kOut);
    ok = ok && ds4_gpu_tensor_read(mid, 0, gpu_mid.data(), gpu_mid.size() * sizeof(float)) &&
         ds4_gpu_tensor_read(out, 0, gpu_out.data(), gpu_out.size() * sizeof(float));
    if (ok) {
        std::vector<float> canonical_mid((uint64_t)n_expert * kMid);
        std::vector<float> summed_gpu_mid(kMid, 0.0f);
        float weight_sum = 0.0f;
        for (uint32_t s = 0; s < n_expert; ++s) {
            weight_sum += slot_weights[s];
            for (uint32_t i = 0; i < kMid; ++i) {
                canonical_mid[(uint64_t)s * kMid + i] = ref.mid[i] * slot_weights[s];
                summed_gpu_mid[i] += gpu_mid[(uint64_t)s * kMid + i];
            }
        }
        std::vector<float> canonical_out(kOut), from_gpu_mid(kOut);
        for (uint32_t i = 0; i < kOut; ++i) {
            canonical_out[i] = ref.out[i] * weight_sum;
            from_gpu_mid[i] = dot_row(rw.down.data() + (uint64_t)i * kMid,
                                      summed_gpu_mid.data(), kMid);
        }
        result->mid = compare(gpu_mid.data(), canonical_mid.data(), canonical_mid.size());
        result->mid_row_identity = compare(
            gpu_mid.data() + (uint64_t)n_expert * kMid,
            gpu_mid.data(), (uint64_t)n_expert * kMid);
        result->down_only = compare(gpu_out.data(), from_gpu_mid.data(), kOut);
        result->full = compare(gpu_out.data(), canonical_out.data(), kOut);
        result->row_identity = compare(gpu_out.data() + kOut, gpu_out.data(), kOut);

        if (q8_mid) {
            const uint64_t qblocks = pairs * (kMid / kQuantBlock);
            std::vector<block_q8_K> quantized(qblocks);
            std::vector<float> dequant_mid((uint64_t)n_expert * kMid);
            ok = ds4_gpu_tensor_read(gate, 0, quantized.data(),
                                     qblocks * sizeof(block_q8_K));
            if (ok) {
                for (uint32_t s = 0; s < n_expert; ++s) {
                    for (uint32_t b = 0; b < kMid / kQuantBlock; ++b) {
                        const block_q8_K &qb = quantized[(uint64_t)s * (kMid / kQuantBlock) + b];
                        for (uint32_t j = 0; j < kQuantBlock; ++j)
                            dequant_mid[(uint64_t)s * kMid + b * kQuantBlock + j] = qb.d * qb.qs[j];
                    }
                }
                result->mid_quant = compare(dequant_mid.data(), gpu_mid.data(), dequant_mid.size());
            }
        }

        std::printf("dispatch case=%s tokens=%u used=%u sorted=%u requested_q8=%u "
                    "float_down=%u f16_down=%u wmma_hot=%u hot_count=%u\n",
                    case_name, result->dispatch.n_tokens, result->dispatch.n_expert,
                    result->dispatch.use_sorted_pairs, result->dispatch.requested_q8_mid,
                    result->dispatch.use_float_down, result->dispatch.use_f16_down,
                    result->dispatch.use_wmma_hot, result->dispatch.hot_count);
        print_metrics(case_name, "mid_vs_canonical", result->mid);
        print_metrics(case_name, "mid_identical_row", result->mid_row_identity);
        if (q8_mid) print_metrics(case_name, "q8_dequant_vs_gpu_mid", result->mid_quant);
        print_metrics(case_name, "down_vs_gpu_mid_oracle", result->down_only);
        print_metrics(case_name, "out_vs_canonical", result->full);
        print_metrics(case_name, "identical_row", result->row_identity);

        const bool expected_hot = !q8_mid && !force_scalar && pairs >= 8u;
        ok = result->dispatch.n_tokens == n_tokens &&
             result->dispatch.n_expert == n_expert &&
             result->dispatch.use_sorted_pairs == 1u &&
             result->dispatch.requested_q8_mid == (q8_mid ? 1u : 0u) &&
             result->dispatch.use_float_down == (q8_mid ? 0u : 1u) &&
             (!q8_mid ? result->dispatch.use_f16_down == 1u : true) &&
             result->dispatch.use_wmma_hot == (expected_hot ? 1u : 0u) &&
             metrics_sane(result->mid) && metrics_sane(result->down_only) &&
             metrics_sane(result->full) && result->row_identity.nonfinite == 0 &&
             result->mid_row_identity.nrmse <= 1.0e-7 &&
             result->row_identity.nrmse <= 1.0e-7;
    }

    unsetenv("DS4_ROCM_GLM5_BATCH_Q8_MID_DOWN");
    unsetenv("DS4_ROCM_Q2_DOWN_FORCE_SCALAR");
    ds4_gpu_tensor_free(gpu_weights); ds4_gpu_tensor_free(selected);
    ds4_gpu_tensor_free(down); ds4_gpu_tensor_free(mid);
    ds4_gpu_tensor_free(up); ds4_gpu_tensor_free(gate);
    ds4_gpu_tensor_free(out); ds4_gpu_tensor_free(tx);
    if (!ok) std::fprintf(stderr, "FAIL case %s\n", case_name);
    return ok;
}

static bool run_dataset(const char *name, const Glm5TestGGUF &g,
                        uint64_t gate_offset, uint64_t up_offset,
                        uint64_t down_offset, const ReferenceWeights &rw,
                        const std::vector<float> &x) {
    const auto mm = std::minmax_element(x.begin(), x.end());
    std::fprintf(stderr, "dataset=%s input_minmax=[%g,%g]\n",
                 name, *mm.first, *mm.second);
    const ReferenceRow ref = make_reference(rw, x);
    const std::vector<float> one = {1.0f};
    const std::vector<float> realistic = {0.2f};
    const std::vector<float> top8 = {
        0.23f, 0.18f, 0.14f, 0.12f, 0.10f, 0.09f, 0.075f, 0.065f};
    BatchResult b2_float, b2_q8, p8_scalar, p8_hot, p8_q8;
    bool ok = run_decode_case((std::string(name) + "-D0").c_str(), g,
                              gate_offset, up_offset, down_offset, x, ref) &&
        run_batch_case((std::string(name) + "-B2-float-w1").c_str(), g,
                       gate_offset, up_offset, down_offset, rw, x, ref,
                       2u, one, false, false, &b2_float) &&
        run_batch_case((std::string(name) + "-B2-q8-w02").c_str(), g,
                       gate_offset, up_offset, down_offset, rw, x, ref,
                       2u, realistic, true, false, &b2_q8) &&
        run_batch_case((std::string(name) + "-P16-top8-float-scalar").c_str(), g,
                       gate_offset, up_offset, down_offset, rw, x, ref,
                       16u, top8, false, true, &p8_scalar) &&
        run_batch_case((std::string(name) + "-P16-top8-float-wmma").c_str(), g,
                       gate_offset, up_offset, down_offset, rw, x, ref,
                       16u, top8, false, false, &p8_hot) &&
        run_batch_case((std::string(name) + "-P16-top8-q8").c_str(), g,
                       gate_offset, up_offset, down_offset, rw, x, ref,
                       16u, top8, true, false, &p8_q8);
    if (ok) {
        const double rel_gap = std::fabs(p8_hot.full.nrmse - p8_q8.full.nrmse) /
            std::max(std::min(p8_hot.full.nrmse, p8_q8.full.nrmse), 1.0e-30);
        const char *winner = rel_gap < 0.10 ? "tie" :
            (p8_hot.full.nrmse < p8_q8.full.nrmse ? "float-wmma" : "q8-mid");
        std::printf("ranking dataset=%s primary=out_nrmse winner=%s relative_gap=%.6g "
                    "float_scalar=%.9g float_wmma=%.9g q8_mid=%.9g\n",
                    name, winner, rel_gap, p8_scalar.full.nrmse,
                    p8_hot.full.nrmse, p8_q8.full.nrmse);
    }
    return ok;
}

struct ActualTop8Fixture {
    std::vector<unsigned char> model;
    std::vector<float> canonical_mid;
    std::vector<float> canonical_out;
    std::vector<float> down;
    std::vector<float> weights;
    uint64_t gate_offset = 0;
    uint64_t up_offset = 0;
    uint64_t down_offset = 0;
};

static bool build_actual_top8_fixture(
        const Glm5TestGGUF &g, uint64_t gate_table, uint64_t up_table,
        uint64_t down_table, deq_iq2_fn deq_iq2, deq_q2_fn deq_q2,
        const std::vector<float> &x, const int32_t global_routes[8],
        const float route_weights[8], ActualTop8Fixture *f) {
    constexpr uint32_t used = 8;
    const uint64_t gate_expert_bytes = (uint64_t)kMid * kIq2RowBytes;
    const uint64_t down_expert_bytes = (uint64_t)kOut * kQ2RowBytes;
    const uint64_t gate_table_bytes = used * gate_expert_bytes;
    const uint64_t down_table_bytes = used * down_expert_bytes;
    f->gate_offset = 0;
    f->up_offset = gate_table_bytes;
    f->down_offset = gate_table_bytes * 2u;
    f->model.resize(f->down_offset + down_table_bytes);
    f->canonical_mid.resize((uint64_t)used * kMid);
    f->canonical_out.assign(kOut, 0.0f);
    f->down.resize((uint64_t)used * kOut * kMid);
    f->weights.assign(route_weights, route_weights + used);
    std::vector<float> gate_row(kIn), up_row(kIn);
    for (uint32_t s = 0; s < used; ++s) {
        const int32_t expert = global_routes[s];
        if (expert < 0 || expert >= 288) {
            std::fprintf(stderr, "FAIL invalid captured expert %d\n", expert);
            return false;
        }
        std::memcpy(f->model.data() + f->gate_offset + (uint64_t)s * gate_expert_bytes,
                    g.map + gate_table + (uint64_t)expert * gate_expert_bytes,
                    gate_expert_bytes);
        std::memcpy(f->model.data() + f->up_offset + (uint64_t)s * gate_expert_bytes,
                    g.map + up_table + (uint64_t)expert * gate_expert_bytes,
                    gate_expert_bytes);
        std::memcpy(f->model.data() + f->down_offset + (uint64_t)s * down_expert_bytes,
                    g.map + down_table + (uint64_t)expert * down_expert_bytes,
                    down_expert_bytes);
        float *mid = f->canonical_mid.data() + (uint64_t)s * kMid;
        for (uint32_t row = 0; row < kMid; ++row) {
            deq_iq2((const block_iq2_xxs *)(f->model.data() + f->gate_offset +
                        (uint64_t)s * gate_expert_bytes + (uint64_t)row * kIq2RowBytes),
                    gate_row.data(), kIn);
            deq_iq2((const block_iq2_xxs *)(f->model.data() + f->up_offset +
                        (uint64_t)s * gate_expert_bytes + (uint64_t)row * kIq2RowBytes),
                    up_row.data(), kIn);
            float gate = std::min(dot_row(gate_row.data(), x.data(), kIn), 10.0f);
            float up = std::max(-10.0f,
                std::min(dot_row(up_row.data(), x.data(), kIn), 10.0f));
            mid[row] = gate / (1.0f + std::exp(-gate)) * up * route_weights[s];
        }
        float *down = f->down.data() + (uint64_t)s * kOut * kMid;
        for (uint32_t row = 0; row < kOut; ++row) {
            deq_q2((const block_q2_K *)(f->model.data() + f->down_offset +
                        (uint64_t)s * down_expert_bytes + (uint64_t)row * kQ2RowBytes),
                   down + (uint64_t)row * kMid, kMid);
            f->canonical_out[row] +=
                dot_row(down + (uint64_t)row * kMid, mid, kMid);
        }
    }
    return true;
}

static bool run_actual_top8_case(const char *case_name,
                                 const ActualTop8Fixture &f,
                                 const std::vector<float> &x,
                                 bool q8_mid, bool force_scalar,
                                 bool force_f32_down,
                                 BatchResult *result) {
    constexpr uint32_t used = 8, tokens = 16;
    constexpr uint64_t pairs = (uint64_t)used * tokens;
    std::vector<float> hx((uint64_t)tokens * kIn);
    std::vector<int32_t> routes(pairs);
    std::vector<float> weights(pairs);
    for (uint32_t t = 0; t < tokens; ++t) {
        std::memcpy(hx.data() + (uint64_t)t * kIn, x.data(), kIn * sizeof(float));
        for (uint32_t s = 0; s < used; ++s) {
            routes[(uint64_t)t * used + s] = (int32_t)s;
            weights[(uint64_t)t * used + s] = f.weights[s];
        }
    }
    ds4_gpu_tensor *tx = alloc_tensor(hx.size() * sizeof(float), "actual x");
    ds4_gpu_tensor *out = alloc_tensor((uint64_t)tokens * kOut * sizeof(float), "actual out");
    ds4_gpu_tensor *gate = alloc_tensor(pairs * kMid * sizeof(float), "actual gate");
    ds4_gpu_tensor *up = alloc_tensor(pairs * kMid * sizeof(float), "actual up");
    ds4_gpu_tensor *mid = alloc_tensor(pairs * kMid * sizeof(float), "actual mid");
    ds4_gpu_tensor *down = alloc_tensor(pairs * kOut * sizeof(float), "actual down");
    ds4_gpu_tensor *selected = alloc_tensor(pairs * sizeof(int32_t), "actual routes");
    ds4_gpu_tensor *gpu_weights = alloc_tensor(pairs * sizeof(float), "actual weights");
    bool ok = tx && out && gate && up && mid && down && selected && gpu_weights &&
        write_tensor(tx, hx.data(), hx.size() * sizeof(float), "actual x") &&
        write_tensor(selected, routes.data(), pairs * sizeof(int32_t), "actual routes") &&
        write_tensor(gpu_weights, weights.data(), pairs * sizeof(float), "actual weights");
    if (q8_mid) setenv("DS4_ROCM_GLM5_BATCH_Q8_MID_DOWN", "1", 1);
    else unsetenv("DS4_ROCM_GLM5_BATCH_Q8_MID_DOWN");
    if (force_scalar) setenv("DS4_ROCM_Q2_DOWN_FORCE_SCALAR", "1", 1);
    else unsetenv("DS4_ROCM_Q2_DOWN_FORCE_SCALAR");
    if (force_f32_down) setenv("DS4_ROCM_TEST_Q2_DOWN_F32", "1", 1);
    else unsetenv("DS4_ROCM_TEST_Q2_DOWN_F32");
    ds4_gpu_test_reset_glm5_q2_dispatch();
    bool mid_is_f16 = true;
    ok = ok && ds4_gpu_routed_moe_batch_tensor(
        out, gate, up, mid, down, f.model.data(), f.model.size(),
        f.gate_offset, f.up_offset, f.down_offset, 16u, 10u,
        kMid * kIq2RowBytes, kIq2RowBytes, kOut * kQ2RowBytes, kQ2RowBytes,
        kIn, kMid, kOut, selected, gpu_weights, used, used, 10.0f,
        tx, 3u, tokens, &mid_is_f16, false) && ds4_gpu_synchronize() &&
        !mid_is_f16 && ds4_gpu_test_get_glm5_q2_dispatch(&result->dispatch);
    std::vector<float> gpu_mid(pairs * kMid);
    std::vector<float> gpu_out((uint64_t)tokens * kOut);
    ok = ok && ds4_gpu_tensor_read(mid, 0, gpu_mid.data(), gpu_mid.size() * sizeof(float)) &&
         ds4_gpu_tensor_read(out, 0, gpu_out.data(), gpu_out.size() * sizeof(float));
    if (ok) {
        std::vector<float> from_gpu_mid(kOut, 0.0f);
        for (uint32_t s = 0; s < used; ++s) {
            const float *m = gpu_mid.data() + (uint64_t)s * kMid;
            const float *dw = f.down.data() + (uint64_t)s * kOut * kMid;
            for (uint32_t row = 0; row < kOut; ++row)
                from_gpu_mid[row] += dot_row(dw + (uint64_t)row * kMid, m, kMid);
        }
        result->mid = compare(gpu_mid.data(), f.canonical_mid.data(),
                              f.canonical_mid.size());
        result->mid_row_identity = compare(
            gpu_mid.data() + (uint64_t)used * kMid,
            gpu_mid.data(), (uint64_t)used * kMid);
        result->down_only = compare(gpu_out.data(), from_gpu_mid.data(), kOut);
        result->full = compare(gpu_out.data(), f.canonical_out.data(), kOut);
        result->row_identity = compare(gpu_out.data() + kOut, gpu_out.data(), kOut);
        result->out_fnv64 = fnv1a64(gpu_out.data(), gpu_out.size() * sizeof(float));
        std::vector<float> down0((uint64_t)used * kOut);
        std::vector<float> down1((uint64_t)used * kOut);
        if (result->dispatch.use_f16_down) {
            std::vector<uint16_t> raw(pairs * kOut);
            ok = ds4_gpu_tensor_read(down, 0, raw.data(), raw.size() * sizeof(uint16_t));
            if (ok) {
                for (uint32_t s = 0; s < used; ++s) {
                    for (uint32_t row = 0; row < kOut; ++row) {
                        const uint16_t a = raw[(uint64_t)s * kOut + row];
                        const uint16_t b = raw[((uint64_t)used + s) * kOut + row];
                        down0[(uint64_t)s * kOut + row] = half_bits_to_float(a);
                        down1[(uint64_t)s * kOut + row] = half_bits_to_float(b);
                        if (a != b) {
                            ++result->down_diff_count;
                            const uint32_t d = a > b ? a - b : b - a;
                            result->down_max_half_ulp =
                                std::max(result->down_max_half_ulp, d);
                        }
                    }
                }
            }
        } else {
            std::vector<float> raw(pairs * kOut);
            ok = ds4_gpu_tensor_read(down, 0, raw.data(), raw.size() * sizeof(float));
            if (ok) {
                for (uint32_t s = 0; s < used; ++s) {
                    std::memcpy(down0.data() + (uint64_t)s * kOut,
                                raw.data() + (uint64_t)s * kOut,
                                kOut * sizeof(float));
                    std::memcpy(down1.data() + (uint64_t)s * kOut,
                                raw.data() + ((uint64_t)used + s) * kOut,
                                kOut * sizeof(float));
                }
            }
        }
        if (ok) result->down_row_identity =
            compare(down1.data(), down0.data(), down0.size());
        if (q8_mid) {
            const uint64_t qblocks = pairs * (kMid / kQuantBlock);
            std::vector<block_q8_K> quantized(qblocks);
            std::vector<float> dequant_mid((uint64_t)used * kMid);
            ok = ds4_gpu_tensor_read(gate, 0, quantized.data(), qblocks * sizeof(block_q8_K));
            if (ok) {
                for (uint32_t s = 0; s < used; ++s) {
                    for (uint32_t b = 0; b < kMid / kQuantBlock; ++b) {
                        const block_q8_K &qb = quantized[(uint64_t)s * (kMid / kQuantBlock) + b];
                        for (uint32_t j = 0; j < kQuantBlock; ++j)
                            dequant_mid[(uint64_t)s * kMid + b * kQuantBlock + j] = qb.d * qb.qs[j];
                    }
                }
                result->mid_quant = compare(dequant_mid.data(), gpu_mid.data(), dequant_mid.size());
            }
        }
        std::printf("dispatch case=%s tokens=%u used=%u sorted=%u requested_q8=%u "
                    "float_down=%u f16_down=%u wmma_hot=%u hot_count=%u\n",
                    case_name, result->dispatch.n_tokens, result->dispatch.n_expert,
                    result->dispatch.use_sorted_pairs, result->dispatch.requested_q8_mid,
                    result->dispatch.use_float_down, result->dispatch.use_f16_down,
                    result->dispatch.use_wmma_hot, result->dispatch.hot_count);
        print_metrics(case_name, "mid_vs_canonical", result->mid);
        print_metrics(case_name, "mid_identical_row", result->mid_row_identity);
        if (q8_mid) print_metrics(case_name, "q8_dequant_vs_gpu_mid", result->mid_quant);
        print_metrics(case_name, "down_vs_gpu_mid_oracle", result->down_only);
        print_metrics(case_name, "pre_sum_down_identical_row", result->down_row_identity);
        print_metrics(case_name, "out_vs_canonical", result->full);
        print_metrics(case_name, "identical_row", result->row_identity);
        std::printf("identity_detail case=%s pre_sum_diff_count=%llu "
                    "pre_sum_max_half_ulp=%u out_fnv64=%016llx\n",
                    case_name, (unsigned long long)result->down_diff_count,
                    result->down_max_half_ulp,
                    (unsigned long long)result->out_fnv64);
        const bool expected_hot = !q8_mid && !force_scalar;
        bool row_identity_ok = false;
        if (expected_hot && !force_f32_down) {
            /* rocWMMA fragment rows differ deterministically at the last F32
             * bits; the F16 store can snap those values by one half ULP. This
             * is a component diagnostic bound, not a promotion tolerance. */
            row_identity_ok = result->row_identity.nrmse <= 1.0e-4 &&
                result->row_identity.max_abs <= std::ldexp(1.0, -12) &&
                result->down_max_half_ulp <= 1u;
        } else if (expected_hot) {
            row_identity_ok = result->row_identity.nrmse <= 1.0e-5 &&
                result->row_identity.max_abs <= 1.0e-5;
        } else {
            row_identity_ok = result->row_identity.nrmse <= 1.0e-7;
        }
        ok = result->dispatch.n_tokens == tokens && result->dispatch.n_expert == used &&
             result->dispatch.use_sorted_pairs == 1u &&
             result->dispatch.requested_q8_mid == (q8_mid ? 1u : 0u) &&
             result->dispatch.use_float_down == (q8_mid ? 0u : 1u) &&
             result->dispatch.use_f16_down ==
                 ((!q8_mid && !force_f32_down) ? 1u : 0u) &&
             result->dispatch.use_wmma_hot == (expected_hot ? 1u : 0u) &&
             (!expected_hot || result->dispatch.hot_count == used) &&
             metrics_sane(result->mid) && metrics_sane(result->down_only) &&
             metrics_sane(result->full) &&
             result->mid_row_identity.nrmse <= 1.0e-7 &&
             result->down_row_identity.nonfinite == 0 &&
             result->row_identity.nonfinite == 0 && row_identity_ok;
    }
    unsetenv("DS4_ROCM_GLM5_BATCH_Q8_MID_DOWN");
    unsetenv("DS4_ROCM_Q2_DOWN_FORCE_SCALAR");
    unsetenv("DS4_ROCM_TEST_Q2_DOWN_F32");
    ds4_gpu_tensor_free(gpu_weights); ds4_gpu_tensor_free(selected);
    ds4_gpu_tensor_free(down); ds4_gpu_tensor_free(mid); ds4_gpu_tensor_free(up);
    ds4_gpu_tensor_free(gate); ds4_gpu_tensor_free(out); ds4_gpu_tensor_free(tx);
    if (!ok) std::fprintf(stderr, "FAIL case %s\n", case_name);
    return ok;
}
}  // namespace

int main() {
    const char *model_path = std::getenv("DS4_GLM5_MODEL");
    const char *llama_path = std::getenv("DS4_LLAMA_GGML_BASE");
    if (!model_path || !llama_path) {
        std::fprintf(stderr, "FAIL DS4_GLM5_MODEL and DS4_LLAMA_GGML_BASE are required\n");
        return 2;
    }

    Glm5TestGGUF g;
    if (!g.open_file(model_path)) return 1;
    uint64_t gate_table = 0, up_table = 0, down_table = 0;
    if (!g.tensor("blk.3.ffn_gate_exps.weight", {kIn, kMid, 288}, 16, gate_table) ||
        !g.tensor("blk.3.ffn_up_exps.weight", {kIn, kMid, 288}, 16, up_table) ||
        !g.tensor("blk.3.ffn_down_exps.weight", {kMid, kOut, 288}, 10, down_table)) {
        std::fprintf(stderr, "FAIL expected mixed-Q2 tensors\n");
        return 1;
    }
    const uint64_t gate_offset = gate_table + (uint64_t)kExpert * kMid * kIq2RowBytes;
    const uint64_t up_offset = up_table + (uint64_t)kExpert * kMid * kIq2RowBytes;
    const uint64_t down_offset = down_table + (uint64_t)kExpert * kOut * kQ2RowBytes;

    void *handle = dlopen(llama_path, RTLD_NOW);
    if (!handle) {
        std::fprintf(stderr, "FAIL dlopen %s: %s\n", llama_path, dlerror());
        return 1;
    }
    const auto deq_iq2 = (deq_iq2_fn)dlsym(handle, "dequantize_row_iq2_xxs");
    const auto deq_q2 = (deq_q2_fn)dlsym(handle, "dequantize_row_q2_K");
    if (!deq_iq2 || !deq_q2) {
        std::fprintf(stderr, "FAIL llama dequant symbols\n");
        return 1;
    }

    ReferenceWeights rw;
    rw.gate.resize((uint64_t)kMid * kIn);
    rw.up.resize((uint64_t)kMid * kIn);
    rw.down.resize((uint64_t)kOut * kMid);
    for (uint32_t row = 0; row < kMid; ++row) {
        deq_iq2((const block_iq2_xxs *)(g.map + gate_offset + (uint64_t)row * kIq2RowBytes),
                rw.gate.data() + (uint64_t)row * kIn, kIn);
        deq_iq2((const block_iq2_xxs *)(g.map + up_offset + (uint64_t)row * kIq2RowBytes),
                rw.up.data() + (uint64_t)row * kIn, kIn);
    }
    for (uint32_t row = 0; row < kOut; ++row) {
        deq_q2((const block_q2_K *)(g.map + down_offset + (uint64_t)row * kQ2RowBytes),
               rw.down.data() + (uint64_t)row * kMid, kMid);
    }

    if (!ds4_gpu_init() || ds4_gpu_tp_expert_shard_active() ||
        !ds4_gpu_set_model_map(g.map, g.size) ||
        !ds4_gpu_set_model_fd_for_map(g.fd, g.map) ||
        !ds4_gpu_cache_model_range(g.map, g.size, gate_offset,
                                   kMid * kIq2RowBytes, "oracle IQ2 gate") ||
        !ds4_gpu_cache_model_range(g.map, g.size, up_offset,
                                   kMid * kIq2RowBytes, "oracle IQ2 up") ||
        !ds4_gpu_cache_model_range(g.map, g.size, down_offset,
                                   kOut * kQ2RowBytes, "oracle Q2 down")) {
        std::fprintf(stderr, "FAIL initialize/cache backend or TP unexpectedly active\n");
        return 1;
    }

    std::vector<float> synthetic(kIn);
    for (uint32_t i = 0; i < kIn; ++i)
        synthetic[i] = std::sin(i * 0.017f) + 0.25f * std::cos(i * 0.0031f);
    bool ok = run_dataset("synthetic", g, gate_offset, up_offset, down_offset,
                          rw, synthetic);

    const char *captured_path = std::getenv("DS4_GLM5_CAPTURED_X");
    std::vector<float> captured;
    if (captured_path && captured_path[0]) {
        ok = read_first_row(captured_path, &captured) &&
             run_dataset("captured", g, gate_offset, up_offset, down_offset,
                         rw, captured) && ok;
    } else {
        std::fprintf(stderr, "note: DS4_GLM5_CAPTURED_X unset; synthetic cases only\n");
    }

    const char *routes_path = std::getenv("DS4_GLM5_CAPTURED_ROUTES");
    const char *weights_path = std::getenv("DS4_GLM5_CAPTURED_WEIGHTS");
    if (captured_path && routes_path && weights_path) {
        int32_t routes[8];
        float weights[8];
        ActualTop8Fixture fixture;
        ok = read_exact(routes_path, routes, sizeof(routes)) &&
             read_exact(weights_path, weights, sizeof(weights)) &&
             build_actual_top8_fixture(g, gate_table, up_table, down_table,
                                       deq_iq2, deq_q2, captured,
                                       routes, weights, &fixture) && ok;
        if (ok) {
            ds4_gpu_cleanup();
            ok = ds4_gpu_init() &&
                 !ds4_gpu_tp_expert_shard_active() &&
                 ds4_gpu_set_model_map(fixture.model.data(), fixture.model.size()) &&
                 ds4_gpu_cache_model_range(fixture.model.data(), fixture.model.size(),
                    0u, fixture.model.size(), "oracle actual top8 packed model");
            BatchResult scalar, hot, hot_f32, q8;
            bool cases_ok = ok;
            if (ok) {
                const bool scalar_ok = run_actual_top8_case(
                    "actual-P16-top8-float-scalar", fixture, captured,
                    false, true, false, &scalar);
                const bool hot_ok = run_actual_top8_case(
                    "actual-P16-top8-float-wmma", fixture, captured,
                    false, false, false, &hot);
                const bool hot_f32_ok = run_actual_top8_case(
                    "actual-P16-top8-float-wmma-f32", fixture, captured,
                    false, false, true, &hot_f32);
                const bool q8_ok = run_actual_top8_case(
                    "actual-P16-top8-q8", fixture, captured,
                    true, false, false, &q8);
                const bool wmma_close_to_scalar =
                    std::fabs(hot.full.nrmse - scalar.full.nrmse) <= 5.0e-5 &&
                    hot.down_only.nrmse < 1.0e-3;
                cases_ok = scalar_ok && hot_ok && hot_f32_ok && q8_ok &&
                    wmma_close_to_scalar;
            }
            if (ok) {
                const double rel_gap = std::fabs(hot.full.nrmse - q8.full.nrmse) /
                    std::max(std::min(hot.full.nrmse, q8.full.nrmse), 1.0e-30);
                const char *winner = rel_gap < 0.10 ? "tie" :
                    (hot.full.nrmse < q8.full.nrmse ? "float-wmma" : "q8-mid");
                std::printf("ranking dataset=actual-top8 primary=out_nrmse winner=%s "
                            "relative_gap=%.6g float_scalar=%.9g float_wmma=%.9g "
                            "float_wmma_f32=%.9g q8_mid=%.9g\n", winner, rel_gap,
                            scalar.full.nrmse, hot.full.nrmse,
                            hot_f32.full.nrmse, q8.full.nrmse);
            }
            ok = ok && cases_ok;
        }
    } else {
        std::fprintf(stderr, "note: actual top-8 fixture paths unset\n");
    }
    ds4_gpu_cleanup();
    dlclose(handle);
    std::printf("test_rocm_glm5_q2_expert_oracle: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
