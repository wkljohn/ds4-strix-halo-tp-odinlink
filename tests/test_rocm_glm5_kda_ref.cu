#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include <vector>

#include "ds4_gpu.h"

typedef int (*ds4_tp_devcopy_fn)(void *, const void *, uint64_t);
extern "C" void ds4_tp_set_devcopy(ds4_tp_devcopy_fn) {}

enum { H = 64, C = 128 };

#define CHECK(expr, message) do { \
    if (!(expr)) { \
        std::fprintf(stderr, "FAIL %s\n", message); \
        return false; \
    } \
} while (0)

struct Inputs {
    uint32_t tokens;
    std::vector<float> q, k, v, gate, beta, state;
};

static Inputs make_inputs(uint32_t tokens, int gate_mode,
                          bool nonzero_state) {
    Inputs x;
    x.tokens = tokens;
    const size_t token_values = (size_t)tokens * H * C;
    x.q.resize(token_values);
    x.k.resize(token_values);
    x.v.resize(token_values);
    x.gate.resize(token_values);
    x.beta.resize((size_t)tokens * H);
    x.state.resize((size_t)H * C * C);
    for (size_t i = 0; i < token_values; ++i) {
        x.q[i] = 0.002f * float(int(i % 17u) - 8);
        x.k[i] = 0.003f * float(int(i % 13u) - 6);
        x.v[i] = 0.007f * float(int(i % 11u) - 5);
        if (gate_mode == 0) x.gate[i] = -1.0e-7f;
        else if (gate_mode == 1) x.gate[i] = -5.0f;
        else {
            static const float mixed[] = {
                -1.0e-7f, -5.0f, -0.2f, -3.0f,
                -0.9f, -4.0f, -0.01f, -2.0f,
            };
            x.gate[i] = mixed[i % 8u];
        }
    }
    for (size_t i = 0; i < x.beta.size(); ++i) {
        if (i % 3u == 0u) x.beta[i] = 1.0e-7f;
        else if (i % 3u == 1u) x.beta[i] = 0.5f;
        else x.beta[i] = 1.0f - 1.0e-7f;
    }
    if (nonzero_state) {
        for (size_t i = 0; i < x.state.size(); ++i)
            x.state[i] = 0.0001f * float(int(i % 23u) - 11);
    }
    return x;
}

static void host_recurrence(const Inputs &x,
                            std::vector<float> &state,
                            std::vector<float> &output) {
    output.assign((size_t)x.tokens * H * C, 0.0f);
    for (uint32_t token = 0; token < x.tokens; ++token) {
        for (uint32_t head = 0; head < H; ++head) {
            const uint64_t base = ((uint64_t)token * H + head) * C;
            const uint64_t state_base = (uint64_t)head * C * C;
            float prediction[C] = {};
            for (uint32_t i = 0; i < C; ++i) {
                const float decay = std::exp(x.gate[base + i]);
                for (uint32_t j = 0; j < C; ++j)
                    state[state_base + (uint64_t)i * C + j] *= decay;
            }
            for (uint32_t i = 0; i < C; ++i) {
                const float key = x.k[base + i];
                for (uint32_t j = 0; j < C; ++j)
                    prediction[j] += key * state[state_base + (uint64_t)i * C + j];
            }
            const float beta = x.beta[(uint64_t)token * H + head];
            for (uint32_t i = 0; i < C; ++i) {
                const float key_beta = x.k[base + i] * beta;
                for (uint32_t j = 0; j < C; ++j) {
                    state[state_base + (uint64_t)i * C + j] +=
                        key_beta * (x.v[base + j] - prediction[j]);
                }
            }
            for (uint32_t i = 0; i < C; ++i) {
                const float query = x.q[base + i];
                for (uint32_t j = 0; j < C; ++j)
                    output[base + j] +=
                        query * state[state_base + (uint64_t)i * C + j];
            }
        }
    }
}

static float max_error(const std::vector<float> &a,
                       const std::vector<float> &b) {
    if (a.size() != b.size()) return INFINITY;
    float error = 0.0f;
    for (size_t i = 0; i < a.size(); ++i)
        error = std::max(error, std::fabs(a[i] - b[i]));
    return error;
}

static double l2_norm(const std::vector<float> &values) {
    long double sum = 0.0;
    for (float value : values) sum += (long double)value * value;
    return std::sqrt((double)sum);
}

static bool gpu_recurrence(const Inputs &x,
                           std::initializer_list<uint32_t> chunks,
                           std::vector<float> &state,
                           std::vector<float> &output) {
    uint32_t total = 0;
    for (uint32_t count : chunks) total += count;
    CHECK(total == x.tokens, "chunk sizes cover input");
    ds4_gpu_tensor *q = ds4_gpu_tensor_alloc(x.q.size() * sizeof(float));
    ds4_gpu_tensor *k = ds4_gpu_tensor_alloc(x.k.size() * sizeof(float));
    ds4_gpu_tensor *v = ds4_gpu_tensor_alloc(x.v.size() * sizeof(float));
    ds4_gpu_tensor *gate = ds4_gpu_tensor_alloc(x.gate.size() * sizeof(float));
    ds4_gpu_tensor *beta = ds4_gpu_tensor_alloc(x.beta.size() * sizeof(float));
    ds4_gpu_tensor *state_gpu = ds4_gpu_tensor_alloc(x.state.size() * sizeof(float));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(x.q.size() * sizeof(float));
    CHECK(q && k && v && gate && beta && state_gpu && out,
          "allocate recurrence tensors");
    CHECK(ds4_gpu_tensor_write(q, 0, x.q.data(), x.q.size() * sizeof(float)) &&
          ds4_gpu_tensor_write(k, 0, x.k.data(), x.k.size() * sizeof(float)) &&
          ds4_gpu_tensor_write(v, 0, x.v.data(), x.v.size() * sizeof(float)) &&
          ds4_gpu_tensor_write(gate, 0, x.gate.data(), x.gate.size() * sizeof(float)) &&
          ds4_gpu_tensor_write(beta, 0, x.beta.data(), x.beta.size() * sizeof(float)) &&
          ds4_gpu_tensor_write(state_gpu, 0, x.state.data(),
                               x.state.size() * sizeof(float)),
          "upload recurrence tensors");
    uint32_t pos = 0;
    for (uint32_t count : chunks) {
        const uint64_t vector_offset = (uint64_t)pos * H * C * sizeof(float);
        const uint64_t vector_bytes = (uint64_t)count * H * C * sizeof(float);
        const uint64_t beta_offset = (uint64_t)pos * H * sizeof(float);
        const uint64_t beta_bytes = (uint64_t)count * H * sizeof(float);
        ds4_gpu_tensor *qv = ds4_gpu_tensor_view(q, vector_offset, vector_bytes);
        ds4_gpu_tensor *kv = ds4_gpu_tensor_view(k, vector_offset, vector_bytes);
        ds4_gpu_tensor *vv = ds4_gpu_tensor_view(v, vector_offset, vector_bytes);
        ds4_gpu_tensor *gv = ds4_gpu_tensor_view(gate, vector_offset, vector_bytes);
        ds4_gpu_tensor *bv = ds4_gpu_tensor_view(beta, beta_offset, beta_bytes);
        ds4_gpu_tensor *ov = ds4_gpu_tensor_view(out, vector_offset, vector_bytes);
        CHECK(qv && kv && vv && gv && bv && ov, "create recurrence chunk views");
        const int encoded = ds4_gpu_glm5_kda_recurrent_tensor(
            ov, state_gpu, qv, kv, vv, gv, bv, count, H, C);
        ds4_gpu_tensor_free(ov);
        ds4_gpu_tensor_free(bv);
        ds4_gpu_tensor_free(gv);
        ds4_gpu_tensor_free(vv);
        ds4_gpu_tensor_free(kv);
        ds4_gpu_tensor_free(qv);
        CHECK(encoded, "encode recurrence chunk");
        pos += count;
    }
    CHECK(ds4_gpu_synchronize(), "synchronize recurrence");
    state.resize(x.state.size());
    output.resize(x.q.size());
    CHECK(ds4_gpu_tensor_read(state_gpu, 0, state.data(),
                              state.size() * sizeof(float)) &&
          ds4_gpu_tensor_read(out, 0, output.data(), output.size() * sizeof(float)),
          "read recurrence results");
    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(state_gpu);
    ds4_gpu_tensor_free(beta);
    ds4_gpu_tensor_free(gate);
    ds4_gpu_tensor_free(v);
    ds4_gpu_tensor_free(k);
    ds4_gpu_tensor_free(q);
    return true;
}

static bool run_oracle_case(uint32_t tokens, int gate_mode,
                            bool nonzero_state) {
    const Inputs x = make_inputs(tokens, gate_mode, nonzero_state);
    std::vector<float> reference_state = x.state;
    std::vector<float> reference_output;
    host_recurrence(x, reference_state, reference_output);
    std::vector<float> got_state, got_output;
    CHECK(gpu_recurrence(x, {tokens}, got_state, got_output),
          "run GPU recurrence");
    const float output_error = max_error(reference_output, got_output);
    const float state_error = max_error(reference_state, got_state);
    const double reference_norm = l2_norm(reference_state);
    const double norm_drift = std::fabs(l2_norm(got_state) - reference_norm) /
                              std::max(reference_norm, 1.0e-30);
    CHECK(output_error <= 3.0e-6f && state_error <= 3.0e-6f &&
          norm_drift <= 1.0e-6, "recurrence matches host oracle");
    std::fprintf(stderr,
                 "PASS GLM5 KDA tokens=%u gate=%d nonzero=%d output_err=%.9g state_err=%.9g norm_drift=%.9g\n",
                 tokens, gate_mode, nonzero_state ? 1 : 0,
                 output_error, state_error, norm_drift);
    return true;
}

static bool run_chunk_equivalence(void) {
    const Inputs x = make_inputs(129, 2, true);
    std::vector<float> one_state, one_output;
    CHECK(gpu_recurrence(x, {129}, one_state, one_output),
          "run one-call recurrence");
    const std::initializer_list<uint32_t> chunkings[] = {
        {1, 128}, {2, 127}, {3, 126}, {127, 1, 1},
    };
    for (const auto &chunks : chunkings) {
        std::vector<float> state, output;
        CHECK(gpu_recurrence(x, chunks, state, output),
              "run chunked recurrence");
        const float output_error = max_error(one_output, output);
        const float state_error = max_error(one_state, state);
        const double norm = l2_norm(one_state);
        const double drift = std::fabs(l2_norm(state) - norm) /
                             std::max(norm, 1.0e-30);
        CHECK(output_error <= 3.0e-6f && state_error <= 3.0e-6f &&
              drift <= 1.0e-6, "chunked recurrence equivalent");
    }
    std::fprintf(stderr, "PASS GLM5 KDA chunk equivalence\n");
    return true;
}

static bool rejected_calls_preserve_state(void) {
    const Inputs x = make_inputs(1, 2, true);
    const uint64_t vector_bytes = (uint64_t)H * C * sizeof(float);
    ds4_gpu_tensor *q = ds4_gpu_tensor_alloc(vector_bytes);
    ds4_gpu_tensor *k = ds4_gpu_tensor_alloc(vector_bytes);
    ds4_gpu_tensor *v = ds4_gpu_tensor_alloc(vector_bytes);
    ds4_gpu_tensor *gate = ds4_gpu_tensor_alloc(vector_bytes);
    ds4_gpu_tensor *beta = ds4_gpu_tensor_alloc((uint64_t)H * sizeof(float));
    ds4_gpu_tensor *state = ds4_gpu_tensor_alloc(x.state.size() * sizeof(float));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(vector_bytes);
    ds4_gpu_tensor *short_out = ds4_gpu_tensor_alloc(vector_bytes - sizeof(float));
    CHECK(q && k && v && gate && beta && state && out && short_out,
          "allocate rejection tensors");
    CHECK(ds4_gpu_tensor_write(state, 0, x.state.data(),
                               x.state.size() * sizeof(float)),
          "upload rejection state");
    CHECK(!ds4_gpu_glm5_kda_recurrent_tensor(
              NULL, state, q, k, v, gate, beta, 1, H, C),
          "reject null output");
    CHECK(!ds4_gpu_glm5_kda_recurrent_tensor(
              out, state, q, k, v, gate, beta, 0, H, C),
          "reject zero tokens");
    CHECK(!ds4_gpu_glm5_kda_recurrent_tensor(
              out, state, q, k, v, gate, beta, 1, H - 1, C),
          "reject head count");
    CHECK(!ds4_gpu_glm5_kda_recurrent_tensor(
              out, state, q, k, v, gate, beta, 1, H, C - 1),
          "reject head dimension");
    CHECK(!ds4_gpu_glm5_kda_recurrent_tensor(
              short_out, state, q, k, v, gate, beta, 1, H, C),
          "reject short output");
    CHECK(!ds4_gpu_glm5_kda_recurrent_tensor(
              state, state, q, k, v, gate, beta, 1, H, C),
          "reject output/state alias");
    CHECK(ds4_gpu_synchronize(), "synchronize rejected recurrence");
    std::vector<float> after(x.state.size());
    CHECK(ds4_gpu_tensor_read(state, 0, after.data(),
                              after.size() * sizeof(float)),
          "read rejected state");
    const bool preserved = max_error(x.state, after) == 0.0f;
    ds4_gpu_tensor_free(short_out);
    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(state);
    ds4_gpu_tensor_free(beta);
    ds4_gpu_tensor_free(gate);
    ds4_gpu_tensor_free(v);
    ds4_gpu_tensor_free(k);
    ds4_gpu_tensor_free(q);
    CHECK(preserved, "rejected recurrence preserves state");
    std::fprintf(stderr, "PASS GLM5 KDA rejection preserves state\n");
    return true;
}

int main(void) {
    if (!ds4_gpu_init()) {
        std::fprintf(stderr, "FAIL initialize ROCm backend\n");
        return 1;
    }
    bool ok = true;
    for (uint32_t length : {1u, 2u, 3u, 127u, 128u, 129u})
        ok &= run_oracle_case(length, 2, length != 1u);
    ok &= run_oracle_case(3, 0, true);
    ok &= run_oracle_case(3, 1, true);
    ok &= run_chunk_equivalence();
    ok &= rejected_calls_preserve_state();
    ds4_gpu_cleanup();
    return ok ? 0 : 1;
}
