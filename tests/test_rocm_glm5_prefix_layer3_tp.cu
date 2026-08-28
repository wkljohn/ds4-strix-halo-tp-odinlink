#include "ds4_glm5_next_exec.h"
#include "ds4_gpu.h"
#include "ds4_gpu_mgpu.h"
extern "C" {
#include "ds4.h"
#include "ds4_tp.h"
}
#include "tests/glm5_gguf_test.hpp"
#include "tests/glm5_next_real_offsets.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#define CHECK(expr, message) do {                                        \
    if (!(expr)) {                                                       \
        std::fprintf(stderr, "FAIL %s (line %d)\n", message, __LINE__); \
        return false;                                                    \
    }                                                                    \
} while (0)

namespace {

constexpr uint32_t kWidth = 4096u;
constexpr uint32_t kHcWidth = 4u * kWidth;
constexpr uint64_t kPrefixFNV = UINT64_C(0x6b704c8b12a398ef);

uint64_t fnv64(const void *data, uint64_t bytes) {
    const auto *p = static_cast<const uint8_t *>(data);
    uint64_t hash = UINT64_C(1469598103934665603);
    for (uint64_t i = 0u; i < bytes; ++i) {
        hash ^= p[i];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

struct VectorError {
    double nrmse = 0.0;
    double cosine = 0.0;
    double max_abs = 0.0;
};

VectorError vector_error_data(const float *reference, const float *candidate,
                              size_t count) {
    VectorError result;
    if (!reference || !candidate || count == 0u) {
        result.nrmse = INFINITY;
        result.cosine = -1.0;
        result.max_abs = INFINITY;
        return result;
    }
    long double diff2 = 0.0L, ref2 = 0.0L;
    long double dot = 0.0L, candidate2 = 0.0L;
    for (size_t i = 0u; i < count; ++i) {
        const long double a = reference[i];
        const long double b = candidate[i];
        const long double d = b - a;
        diff2 += d * d;
        ref2 += a * a;
        dot += a * b;
        candidate2 += b * b;
        result.max_abs = std::max(result.max_abs, std::fabs((double)d));
    }
    result.nrmse = ref2 > 0.0L ? std::sqrt((double)(diff2 / ref2)) :
        (diff2 == 0.0L ? 0.0 : INFINITY);
    result.cosine = ref2 > 0.0L && candidate2 > 0.0L ?
        (double)(dot / std::sqrt(ref2 * candidate2)) : 0.0;
    return result;
}

VectorError vector_error(const std::vector<float> &reference,
                         const std::vector<float> &candidate) {
    if (reference.size() != candidate.size()) {
        VectorError invalid;
        invalid.nrmse = INFINITY;
        invalid.cosine = -1.0;
        invalid.max_abs = INFINITY;
        return invalid;
    }
    return vector_error_data(
        reference.data(), candidate.data(), reference.size());
}

struct TensorGuard {
    ds4_gpu_tensor *value = nullptr;
    ~TensorGuard() { ds4_gpu_tensor_free(value); }
};

struct StateGuard {
    ds4_glm5_next_state value = {};
    ~StateGuard() { ds4_glm5_next_state_free(&value); }
};

struct WorkspaceGuard {
    ds4_glm5_next_workspace *value = nullptr;
    ~WorkspaceGuard() { ds4_glm5_next_workspace_destroy(value); }
};

struct CodecGuard {
    ds4_glm5_next_text_codec *value = nullptr;
    ~CodecGuard() { ds4_glm5_next_text_codec_close(value); }
};

struct TokensGuard {
    ds4_tokens value = {};
    ~TokensGuard() { ds4_tokens_free(&value); }
};

struct TpGuard {
    ds4_tp *tp = nullptr;
    ds4_gpu_tensor *slab = nullptr;
    ds4_gpu_tensor *big_out = nullptr;
    ds4_gpu_tensor *big_in = nullptr;
    ~TpGuard() {
        ds4_gpu_tensor_free(big_in);
        ds4_gpu_tensor_free(big_out);
        if (tp) ds4_tp_free(tp);
        ds4_gpu_tensor_free(slab);
    }
};

bool create_tp(const Glm5TestGGUF &gguf, bool leader,
               const char *host, const char *device, int port,
               uint32_t context_capacity,
               TpGuard &guard, ds4_glm5_next_exec_ctx &exec,
               uint64_t &sequence) {
    char direct_rows[32] = {};
    CHECK(std::snprintf(direct_rows, sizeof(direct_rows), "%u",
                        context_capacity) > 0,
          "format exact TP batch row capacity");
    CHECK(setenv("DS4_TP_BIG_DIRECT", "1", 1) == 0 &&
          setenv("DS4_TP_BIG_DIRECT_MAX_ROWS", direct_rows, 1) == 0,
          "select mandatory direct RDMA");
    ds4_tp_options options = {};
    options.role = leader ? DS4_TP_LEADER : DS4_TP_WORKER;
    options.requested = true;
    options.listen_host = leader ? host : nullptr;
    options.listen_port = leader ? port : 0;
    options.leader_host = leader ? nullptr : host;
    options.leader_port = leader ? 0 : port;
    options.transport = DS4_TP_TRANSPORT_RDMA;
    options.rdma_device = device;
    const char *gid_text = getenv("DS4_GLM5_TP_RDMA_GID_INDEX");
    if (gid_text && gid_text[0]) {
        char *end = nullptr;
        errno = 0;
        const long gid = strtol(gid_text, &end, 10);
        CHECK(errno == 0 && end && *end == '\0' && gid >= 0 && gid <= 255,
              "valid explicit RDMA GID index");
        options.rdma_gid_index = (int)gid;
        options.rdma_gid_index_set = true;
    }

    ds4_tp_identity identity = {};
    identity.gguf_bytes = gguf.size;
    identity.model_id = 3u;
    identity.n_layer = 46u;
    identity.n_embd = kWidth;
    identity.n_vocab = 154880u;
    identity.quant_bits = 4u;
    identity.ctx_size = context_capacity;
    identity.runtime_features =
        DS4_TP_FEATURE_Q4K_WMMA | DS4_TP_FEATURE_Q4K_KSHARD;
    identity.gate_slot_start = 3u * DS4_TP_GATES_PER_LAYER;
    identity.gate_slot_step = 1u;
    CHECK(ds4_glm5_next_build_tp_gate_mask(identity.gate_slot_mask,
                                            &identity.gates_per_token),
          "GLM5.3 hybrid TP gate schedule");

    char error[256] = {};
    CHECK(ds4_tp_create(&guard.tp, &options, &identity,
                        error, sizeof(error)), error);
    CHECK(ds4_tp_is_rdma(guard.tp) &&
          ds4_tp_requires_host_slab(guard.tp),
          "GLM5 layer3 test selected RoCE mapped-host RDMA");
    const uint64_t slab_bytes = ds4_tp_alloc_slab_bytes(guard.tp);
    guard.slab = ds4_gpu_tensor_alloc_rdma_host(slab_bytes);
    CHECK(guard.slab &&
          ds4_tp_attach_slab(guard.tp, ds4_gpu_tensor_contents(guard.slab),
                             error, sizeof(error)), error);
    const uint64_t row_bytes = (uint64_t)kWidth * sizeof(float);
    const uint64_t direct_bytes = (uint64_t)context_capacity * row_bytes;
    guard.big_out = ds4_gpu_tensor_view(
        guard.slab, ds4_tp_slab_big_out_offset(guard.tp), direct_bytes);
    guard.big_in = ds4_gpu_tensor_view(
        guard.slab, ds4_tp_slab_big_in_offset(guard.tp), direct_bytes);
    CHECK(guard.big_out && guard.big_in &&
          ds4_tp_big_gate_is_direct(
              guard.tp, ds4_gpu_tensor_contents(guard.big_out),
              ds4_gpu_tensor_contents(guard.big_in), direct_bytes),
          "GLM5 layer3 uses direct registered big-gate rows");

    exec.tp = guard.tp;
    exec.tp_rank = leader ? 0u : 1u;
    exec.tp_big_out = guard.big_out;
    exec.tp_big_in = guard.big_in;
    exec.tp_big_out_host = ds4_gpu_tensor_contents(guard.big_out);
    exec.tp_big_in_host = ds4_gpu_tensor_contents(guard.big_in);
    exec.tp_sequence = &sequence;
    return true;
}

struct LayerTiming {
    double kda_dense_ms = 0.0;
    double kda_routed_ms = 0.0;
    double mla_routed_ms = 0.0;
    uint64_t kda_dense_calls = 0u;
    uint64_t kda_routed_calls = 0u;
    uint64_t mla_routed_calls = 0u;
};

bool execute_full_token(ds4_glm5_next_exec_ctx &exec,
                        ds4_glm5_next_state &state,
                        ds4_glm5_next_workspace *workspace,
                        uint32_t token,
                        ds4_gpu_tensor *&current,
                        ds4_gpu_tensor *&output,
                        LayerTiming *timing = nullptr) {
    CHECK(ds4_glm5_next_embed_token(&exec, token, current),
          "embed text token");
    for (uint32_t il = 0u; il < DS4_GLM5_NEXT_TRUNK_COUNT; ++il) {
        const auto begin = std::chrono::steady_clock::now();
        CHECK(ds4_glm5_next_layer_forward(
                  &exec, il, &state, workspace, current, output),
              "execute text token through complete trunk");
        if (timing) {
            const double ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - begin).count();
            const ds4_glm5_next_layer_offsets &layer = exec.model->layer[il];
            if (layer.attention == DS4_GLM5_NEXT_ATTN_KDA &&
                layer.ffn == DS4_GLM5_NEXT_FFN_DENSE) {
                timing->kda_dense_ms += ms;
                timing->kda_dense_calls++;
            } else if (layer.attention == DS4_GLM5_NEXT_ATTN_KDA &&
                       layer.ffn == DS4_GLM5_NEXT_FFN_ROUTED) {
                timing->kda_routed_ms += ms;
                timing->kda_routed_calls++;
            } else if (layer.attention == DS4_GLM5_NEXT_ATTN_MLA &&
                       layer.ffn == DS4_GLM5_NEXT_FFN_ROUTED) {
                timing->mla_routed_ms += ms;
                timing->mla_routed_calls++;
            }
        }
        std::swap(current, output);
    }
    return true;
}

bool read_argmax(ds4_glm5_next_exec_ctx &exec,
                 ds4_glm5_next_workspace *workspace,
                 ds4_gpu_tensor *current,
                 ds4_gpu_tensor *logits,
                 ds4_gpu_tensor *gpu_argmax,
                 bool validate_full_logits,
                 std::vector<float> &host_logits,
                 uint32_t &argmax,
                 uint64_t &logits_hash) {
    CHECK(ds4_glm5_next_output_logits(&exec, workspace, current, logits),
          "execute text vocabulary projection");
    if (!validate_full_logits) {
        int32_t gpu_token = -1;
        CHECK(gpu_argmax &&
                  ds4_gpu_argmax_tensor(gpu_argmax, logits, 154880u) &&
                  ds4_gpu_synchronize() &&
                  ds4_gpu_tensor_read(gpu_argmax, 0u, &gpu_token,
                                      sizeof(gpu_token)) &&
                  gpu_token >= 0 && gpu_token < 154880,
              "select text vocabulary argmax on GPU");
        argmax = (uint32_t)gpu_token;
        logits_hash = fnv64(&gpu_token, sizeof(gpu_token));
        return true;
    }
    CHECK(ds4_gpu_synchronize(), "synchronize text vocabulary projection");
    CHECK(ds4_gpu_tensor_read(logits, 0u, host_logits.data(),
                              host_logits.size() * sizeof(float)),
          "read text vocabulary logits");
    logits_hash = fnv64(host_logits.data(),
                        host_logits.size() * sizeof(float));
    argmax = 0u;
    CHECK(std::isfinite(host_logits[0]), "text logits start finite");
    for (uint32_t token = 1u; token < host_logits.size(); ++token) {
        CHECK(std::isfinite(host_logits[token]), "all text logits are finite");
        if (host_logits[token] > host_logits[argmax]) argmax = token;
    }
    return true;
}

bool run() {
    const char *model = std::getenv("DS4_GLM5_MODEL");
    const char *role = std::getenv("DS4_GLM5_TP_ROLE");
    const char *host = std::getenv("DS4_GLM5_TP_HOST");
    const char *device = std::getenv("DS4_GLM5_TP_RDMA_DEVICE");
    const char *port_text = std::getenv("DS4_GLM5_TP_PORT");
    CHECK(model && model[0] && role && host && host[0] && device && device[0] &&
          port_text && port_text[0], "required model and TP environment");
    const bool leader = std::strcmp(role, "leader") == 0;
    CHECK(leader || std::strcmp(role, "worker") == 0,
          "TP role is exactly leader or worker");
    char *end = nullptr;
    const long port_long = std::strtol(port_text, &end, 10);
    CHECK(end && *end == '\0' && port_long >= 1024 && port_long <= 65535,
          "TP port is valid");

    Glm5TestGGUF gguf;
    CHECK(gguf.open_file(model), "open GLM5 GGUF");
    ds4_glm5_next_model_offsets offsets = {};
    CHECK(glm5_next_bind_real_offsets(gguf, offsets),
          "bind complete real GLM5 offsets");
    const char *full_trunk_env = std::getenv("DS4_GLM5_FULL_TRUNK");
    const bool full_trunk = full_trunk_env &&
                            std::strcmp(full_trunk_env, "1") == 0;
    const char *kda_batch_env =
        std::getenv("DS4_GLM5_KDA_ROUTED_BATCH_TEST");
    const bool kda_batch_test = kda_batch_env &&
        std::strcmp(kda_batch_env, "1") == 0;
    CHECK(!kda_batch_env || kda_batch_test ||
              std::strcmp(kda_batch_env, "0") == 0,
          "KDA routed batch test selector is exactly 0 or 1");
    const char *kda_batch_rows_env =
        std::getenv("DS4_GLM5_KDA_ROUTED_BATCH_ROWS");
    char *kda_batch_rows_end = nullptr;
    const unsigned long kda_batch_rows_long = kda_batch_rows_env ?
        std::strtoul(kda_batch_rows_env, &kda_batch_rows_end, 10) : 3ul;
    CHECK((!kda_batch_rows_env ||
              (kda_batch_rows_end && *kda_batch_rows_end == '\0')) &&
              (kda_batch_rows_long == 3ul || kda_batch_rows_long == 33ul),
          "KDA routed batch rows are the bounded 3 or 33 fixture");
    const uint32_t kda_batch_rows = (uint32_t)kda_batch_rows_long;
    const char *kda_profile_repeats_env =
        std::getenv("DS4_GLM5_KDA_ROUTED_PROFILE_REPEATS");
    char *kda_profile_repeats_end = nullptr;
    const unsigned long kda_profile_repeats_long =
        kda_profile_repeats_env ?
        std::strtoul(kda_profile_repeats_env,
                     &kda_profile_repeats_end, 10) : 0ul;
    CHECK((!kda_profile_repeats_env ||
              (kda_profile_repeats_end &&
               *kda_profile_repeats_end == '\0')) &&
              kda_profile_repeats_long <= 16ul &&
              (kda_profile_repeats_long == 0ul ||
               (kda_batch_test && kda_batch_rows == 33u)),
          "KDA routed profile repeats are bounded and use 33 rows");
    const uint32_t kda_profile_repeats =
        (uint32_t)kda_profile_repeats_long;
    const char *full_tokens_env = std::getenv("DS4_GLM5_FULL_TOKENS");
    const uint32_t full_tokens = !full_tokens_env ? 1u :
        std::strcmp(full_tokens_env, "1") == 0 ? 1u :
        std::strcmp(full_tokens_env, "2") == 0 ? 2u : 0u;
    CHECK(full_tokens != 0u &&
              (full_trunk || full_tokens == 1u),
          "full-token count is bounded and requires full-trunk mode");
    const char *text_prompt = std::getenv("DS4_GLM5_TEXT_PROMPT");
    const bool text_mode = text_prompt != nullptr;
    const char *perf_mode_env = std::getenv("DS4_GLM5_PERF_MODE");
    const bool perf_mode = perf_mode_env &&
                           std::strcmp(perf_mode_env, "1") == 0;
    CHECK(!perf_mode_env || perf_mode || std::strcmp(perf_mode_env, "0") == 0,
          "performance mode is exactly 0 or 1");
    const char *layer_timing_env = std::getenv("DS4_GLM5_LAYER_TIMING");
    const bool layer_timing = layer_timing_env &&
                              std::strcmp(layer_timing_env, "1") == 0;
    CHECK(!layer_timing_env || layer_timing ||
              std::strcmp(layer_timing_env, "0") == 0,
          "layer timing is exactly 0 or 1");
    const char *text_generate_env = std::getenv("DS4_GLM5_TEXT_GENERATE");
    char *text_generate_end = nullptr;
    const unsigned long text_generate_long = text_generate_env ?
        std::strtoul(text_generate_env, &text_generate_end, 10) : 4ul;
    const bool text_generate_valid = !text_generate_env ||
        (text_generate_end && *text_generate_end == '\0');
    CHECK(!text_mode ||
              (full_trunk && text_prompt[0] && text_generate_long >= 1ul &&
               text_generate_long <= 128ul && text_generate_valid),
          "real-text mode requires full trunk and 1..128 generated tokens");
    const uint32_t text_generate = (uint32_t)text_generate_long;
    const char *teacher_ids_env =
        std::getenv("DS4_GLM5_TEXT_TEACHER_IDS");
    std::vector<uint32_t> teacher_ids;
    if (teacher_ids_env && teacher_ids_env[0]) {
        const char *cursor = teacher_ids_env;
        while (*cursor) {
            char *teacher_end = nullptr;
            const unsigned long value =
                std::strtoul(cursor, &teacher_end, 10);
            CHECK(teacher_end != cursor && value < 154880ul &&
                      (*teacher_end == ',' || *teacher_end == '\0'),
                  "teacher token list is comma-separated vocabulary IDs");
            teacher_ids.push_back((uint32_t)value);
            cursor = *teacher_end == ',' ? teacher_end + 1 : teacher_end;
            CHECK(*cursor || *teacher_end == '\0',
                  "teacher token list has no empty trailing item");
        }
        CHECK(text_mode && teacher_ids.size() == text_generate,
              "teacher token count must equal generated-token bound");
    }
    CHECK(!perf_mode || (text_mode && teacher_ids.empty()),
          "performance mode requires greedy text generation");
    CHECK(!kda_batch_test || (full_trunk && !text_mode),
          "KDA routed batch test requires full trunk and no text mode");
    CodecGuard codec;
    TokensGuard prompt_tokens;
    if (text_mode) {
        CHECK(ds4_glm5_next_text_codec_open(&codec.value, model) == 0 &&
                  codec.value,
              "open exact same-GGUF GLM5 text codec");
        ds4_glm5_next_text_codec_encode_chat(
            codec.value, nullptr, text_prompt, DS4_THINK_MAX,
            &prompt_tokens.value);
        CHECK(prompt_tokens.value.len > 0 && prompt_tokens.value.len <= 64,
              "real chat prompt token count is bounded to 1..64");
    }
    const uint32_t context_capacity = text_mode ?
        (uint32_t)prompt_tokens.value.len + text_generate :
        kda_batch_test ? kda_batch_rows + 1u : 2u;
    Glm5NextKShardPlan kshard;
    CHECK(!full_trunk || glm5_next_build_kshard_plan(gguf, offsets, kshard),
          "build exact full-trunk compact residency plan");
    ds4_gpu_config config = {};
    config.n_gpus = 1u;
    config.device_indices[0] = 0u;
    CHECK(ds4_gpu_init_multi(&config) &&
          ds4_gpu_set_model_fd_for_map(gguf.fd, gguf.map) &&
          (full_trunk || ds4_gpu_set_model_map(gguf.map, gguf.size)),
          "initialize gfx1151 without broad full-trunk mmap registration");

    StateGuard state;
    WorkspaceGuard workspace;
    TensorGuard current, output, logits, gpu_argmax;
    CHECK(ds4_glm5_next_state_init(&state.value, &offsets,
                                   context_capacity, nullptr) &&
          (workspace.value = ds4_glm5_next_workspace_create()) != nullptr &&
          (current.value = ds4_gpu_tensor_alloc(
              (uint64_t)kHcWidth * sizeof(float))) != nullptr &&
          (output.value = ds4_gpu_tensor_alloc(
              (uint64_t)kHcWidth * sizeof(float))) != nullptr &&
          (!full_trunk ||
           (logits.value = ds4_gpu_tensor_alloc(
               UINT64_C(154880) * sizeof(float))) != nullptr) &&
          (!perf_mode ||
           (gpu_argmax.value = ds4_gpu_tensor_alloc(sizeof(int32_t))) != nullptr),
          "allocate resident state and production workspace");

    ds4_glm5_next_exec_ctx exec = {};
    exec.model_map = gguf.map;
    exec.model_size = gguf.size;
    exec.model = &offsets;
    exec.trace_prefix = std::getenv("DS4_GLM5_NEXT_TRACE_PREFIX");
    if (exec.trace_prefix) {
        const char *layer_text = std::getenv("DS4_GLM5_NEXT_TRACE_LAYER");
        const char *token_text = std::getenv("DS4_GLM5_NEXT_TRACE_TOKEN");
        char *layer_end = nullptr, *token_end = nullptr;
        const unsigned long layer = layer_text ?
            std::strtoul(layer_text, &layer_end, 10) : 3u;
        const bool trace_all = token_text && std::strcmp(token_text, "all") == 0;
        const unsigned long token = !token_text || trace_all ? 0u :
            std::strtoul(token_text, &token_end, 10);
        CHECK((!layer_text || (layer_end && *layer_end == '\0')) &&
              (!token_text || trace_all ||
               (token_end && *token_end == '\0')) &&
              layer < DS4_GLM5_NEXT_TRUNK_COUNT && token <= UINT32_MAX,
              "valid staged trace selector");
        exec.trace_layer = (uint32_t)layer;
        exec.trace_token = trace_all ? UINT32_MAX : (uint32_t)token;
    }
    uint64_t sequence = 0u;
    TpGuard tp;
    CHECK(create_tp(gguf, leader, host, device, (int)port_long,
                    context_capacity,
                    tp, exec, sequence),
          "create persistent GLM5 layer3 RoCE transport");
    char rank_error[256] = {};
    CHECK(ds4_tp_hash_check(
              tp.tp, UINT64_C(0x474c4d3552414e4b),
              UINT64_C(0x52414e4b00000000) ^ exec.tp_rank,
              rank_error, sizeof(rank_error)) == -1,
          "TP roles must prove distinct rank identities");
    char mode_error[256] = {};
    const uint64_t prompt_hash = text_mode ?
        fnv64(prompt_tokens.value.v,
              (uint64_t)prompt_tokens.value.len * sizeof(int)) : 0u;
    const uint64_t teacher_hash = teacher_ids.empty() ? 0u :
        fnv64(teacher_ids.data(),
              teacher_ids.size() * sizeof(teacher_ids[0]));
    const uint64_t mode_hash = UINT64_C(0x474c4d354d4f4400) ^
        ((uint64_t)full_trunk << 8u) ^ full_tokens ^
        ((uint64_t)text_mode << 16u) ^ ((uint64_t)text_generate << 24u) ^
        ((uint64_t)perf_mode << 56u) ^
        ((uint64_t)layer_timing << 57u) ^
        ((uint64_t)kda_batch_test << 58u) ^
        ((uint64_t)kda_batch_rows << 32u) ^
        prompt_hash ^ teacher_hash;
    if (text_mode) {
        std::fprintf(stderr,
            "GLM5 text prompt role=%s tokens=%d token_fnv=%016llx ids=",
            role, prompt_tokens.value.len,
            (unsigned long long)prompt_hash);
        for (int i = 0; i < prompt_tokens.value.len; ++i) {
            std::fprintf(stderr, "%s%d", i ? "," : "",
                         prompt_tokens.value.v[i]);
        }
        std::fprintf(stderr, "\n");
    }
    CHECK(ds4_tp_hash_check(tp.tp, UINT64_C(0x474c4d354d4f4445),
                            mode_hash, mode_error, sizeof(mode_error)) == 1,
          "TP roles must negotiate the same full-trunk/text mode");
    if (full_trunk) {
        constexpr uint64_t install_headroom = UINT64_C(2) << 30u;
        size_t free_before_install = 0u, total_before_install = 0u;
        CHECK(hipMemGetInfo(&free_before_install, &total_before_install) ==
                  hipSuccess,
              "query bounded residency budget before install");
        const uint64_t packed_required = kshard.packed_total_bytes;
        const uint64_t residency_required =
            kshard.dense_total_bytes + packed_required;
        std::fprintf(stderr,
            "GLM5 full-trunk plan rank=%u dense_spans=%zu "
            "dense_bytes=%llu packed_bytes=%llu required_bytes=%llu "
            "headroom_bytes=%llu hip_free=%llu hip_total=%llu\n",
            exec.tp_rank, kshard.dense_offsets.size(),
            (unsigned long long)kshard.dense_total_bytes,
            (unsigned long long)packed_required,
            (unsigned long long)residency_required,
            (unsigned long long)install_headroom,
            (unsigned long long)free_before_install,
            (unsigned long long)total_before_install);
        CHECK(residency_required <= free_before_install &&
                  install_headroom <= free_before_install - residency_required,
              "full-trunk residency exceeds HIP-free GTT; reduce BIOS UMA "
              "carveout and retain at least 2 GiB install headroom");
        CHECK(ds4_gpu_q4k_kshard_install(
                  gguf.map, gguf.size, gguf.fd, exec.tp_rank,
                  kshard.dense_offsets.data(), kshard.dense_sizes.data(),
                  (uint32_t)kshard.dense_offsets.size(),
                  kshard.dense_max_tensor_bytes, kshard.layers.data(),
                  (uint32_t)kshard.layers.size()),
              "install bounded all-layer Q4 K-shard before model access");
        ds4_gpu_q4k_kshard_windows windows = {};
        CHECK(ds4_gpu_q4k_kshard_windows_get(&windows) &&
                  windows.rank == exec.tp_rank && windows.n_layers == 42u &&
                  windows.n_expert == 288u &&
                  windows.expert_in_dim == 4096u &&
                  windows.expert_mid_dim == 1024u &&
                  windows.out_dim == 4096u &&
                  windows.row_base == exec.tp_rank * 1024u &&
                  windows.row_count == 1024u &&
                  windows.down_column_byte_base ==
                      (uint64_t)exec.tp_rank * 4u * 144u &&
                  windows.down_column_byte_count == 4u * 144u,
              "installed Q4 shard geometry matches the negotiated rank");
        const uint64_t packed_from_windows =
            (uint64_t)windows.n_layers * windows.n_expert *
            (2u * windows.packed_gate_expert_bytes +
             windows.packed_down_expert_bytes);
        CHECK(packed_from_windows == packed_required &&
                  ds4_gpu_q4k_packed_slice_bytes() == packed_from_windows,
              "installed Q4 bytes derive exactly from published windows");
        size_t free_after_install = 0u, total_after_install = 0u;
        CHECK(hipMemGetInfo(&free_after_install, &total_after_install) ==
                  hipSuccess && total_after_install == total_before_install &&
                  free_after_install >= install_headroom,
              "full-trunk install must retain measured 2 GiB HIP headroom");
        std::fprintf(stderr,
            "GLM5 full-trunk residency rank=%u dense_spans=%zu "
            "dense_bytes=%llu packed_q4_bytes=%llu\n",
            exec.tp_rank, kshard.dense_offsets.size(),
            (unsigned long long)kshard.dense_total_bytes,
            (unsigned long long)ds4_gpu_q4k_packed_slice_bytes());
        std::fprintf(stderr,
            "GLM5 full-trunk post-install rank=%u hip_free=%llu "
            "hip_total=%llu installed_delta=%llu\n",
            exec.tp_rank, (unsigned long long)free_after_install,
            (unsigned long long)total_after_install,
            (unsigned long long)(free_before_install - free_after_install));
    }

    if (text_mode || kda_batch_test) {
        char ready_error[256] = {};
        CHECK(ds4_gpu_synchronize() &&
                  ds4_tp_hash_check(
                      tp.tp, UINT64_C(0x474c4d3552454144),
                      UINT64_C(0x46554c4c5452554e), ready_error,
                      sizeof(ready_error)) == 1,
              ready_error[0] ? ready_error :
                  "both ranks ready after full-trunk residency");
    }

    if (kda_batch_test) {
        using Clock = std::chrono::steady_clock;
        const uint32_t rows = kda_batch_rows;
        std::vector<uint32_t> ids;
        if (rows == 3u) {
            ids = {42u, 154822u, 154824u};
        } else {
            ids = {
                154822u, 154824u, 154826u, 25062u, 287u, 29905u, 371u,
                25u, 7487u, 154827u, 675u, 279u, 11478u, 7735u, 369u,
                6623u, 323u, 279u, 3150u, 315u, 41907u, 323u, 4968u,
                18110u, 558u, 13u, 21754u, 304u, 825u, 11646u, 13u,
                154828u, 154841u,
            };
        }
        CHECK(ids.size() == rows, "select exact KDA routed token fixture");
        constexpr uint32_t continuation_id = 17u;
        StateGuard sequential_state, batch_state;
        WorkspaceGuard batch_workspace;
        TensorGuard batch_ids, batch_input, batch_output, sequential_output;
        CHECK(ds4_glm5_next_state_init(
                  &sequential_state.value, &offsets, rows + 1u, nullptr) &&
              ds4_glm5_next_state_init(
                  &batch_state.value, &offsets, rows + 1u, nullptr) &&
              (batch_workspace.value =
                   ds4_glm5_next_workspace_create_capacity(rows)) != nullptr &&
              (batch_ids.value = ds4_gpu_tensor_alloc(
                   (uint64_t)rows * sizeof(uint32_t))) != nullptr &&
              (batch_input.value = ds4_gpu_tensor_alloc(
                   (uint64_t)rows * kHcWidth * sizeof(float))) != nullptr &&
              (batch_output.value = ds4_gpu_tensor_alloc(
                   (uint64_t)rows * kHcWidth * sizeof(float))) != nullptr &&
              (sequential_output.value = ds4_gpu_tensor_alloc(
                   (uint64_t)rows * kHcWidth * sizeof(float))) != nullptr &&
              ds4_gpu_tensor_write(batch_ids.value, 0u, ids.data(),
                                    (uint64_t)rows * sizeof(uint32_t)) &&
              ds4_glm5_next_embed_tokens(
                  &exec, batch_ids.value, rows, batch_input.value),
              "allocate exact KDA+routed row comparison");

        const auto sequential_begin = Clock::now();
        for (uint32_t row = 0u; row < rows; ++row) {
            ds4_gpu_tensor *row_input = ds4_gpu_tensor_view(
                batch_input.value,
                (uint64_t)row * kHcWidth * sizeof(float),
                (uint64_t)kHcWidth * sizeof(float));
            ds4_gpu_tensor *row_output = ds4_gpu_tensor_view(
                sequential_output.value,
                (uint64_t)row * kHcWidth * sizeof(float),
                (uint64_t)kHcWidth * sizeof(float));
            const bool row_ok = row_input && row_output &&
                ds4_glm5_next_layer_forward(
                    &exec, 4u, &sequential_state.value,
                    workspace.value, row_input, row_output);
            ds4_gpu_tensor_free(row_output);
            ds4_gpu_tensor_free(row_input);
            CHECK(row_ok,
                  "execute sequential KDA+routed comparison row");
        }
        const auto sequential_end = Clock::now();

        const auto batch_begin = Clock::now();
        CHECK(ds4_glm5_next_layer_forward_batch(
                  &exec, 4u, &batch_state.value, batch_workspace.value,
                  batch_input.value, batch_output.value, rows),
              "execute production KDA+routed batch");
        const auto batch_end = Clock::now();
        const double sequential_ms =
            std::chrono::duration<double, std::milli>(
                sequential_end - sequential_begin).count();
        const double batch_ms =
            std::chrono::duration<double, std::milli>(
                batch_end - batch_begin).count();
        std::vector<float> sequential((uint64_t)rows * kHcWidth);
        std::vector<float> batch((uint64_t)rows * kHcWidth);
        CHECK(ds4_gpu_tensor_read(
                  sequential_output.value, 0u, sequential.data(),
                  sequential.size() * sizeof(float)) &&
              ds4_gpu_tensor_read(batch_output.value, 0u, batch.data(),
                                  batch.size() * sizeof(float)),
              "read sequential and production KDA+routed rows");
        const VectorError batch_error = vector_error(sequential, batch);
        VectorError worst_row;
        worst_row.cosine = 1.0;
        uint32_t worst_nrmse_row = 0u, worst_cosine_row = 0u;
        uint32_t worst_max_abs_row = 0u;
        for (uint32_t row = 0u; row < rows; ++row) {
            const VectorError row_error = vector_error_data(
                sequential.data() + (uint64_t)row * kHcWidth,
                batch.data() + (uint64_t)row * kHcWidth, kHcWidth);
            if (row == 0u || row_error.nrmse > worst_row.nrmse) {
                worst_row.nrmse = row_error.nrmse;
                worst_nrmse_row = row;
            }
            if (row == 0u || row_error.cosine < worst_row.cosine) {
                worst_row.cosine = row_error.cosine;
                worst_cosine_row = row;
            }
            if (row == 0u || row_error.max_abs > worst_row.max_abs) {
                worst_row.max_abs = row_error.max_abs;
                worst_max_abs_row = row;
            }
        }
        std::fprintf(stderr,
            "GLM5 KDA+routed batch measurement role=%s rows=%u "
            "nrmse=%.9g cosine=%.12g max_abs=%.9g "
            "worst_row_nrmse=%.9g worst_nrmse_row=%u "
            "worst_row_cosine=%.12g worst_cosine_row=%u "
            "worst_row_max_abs=%.9g worst_max_abs_row=%u "
            "sequential_ms=%.3f batch_ms=%.3f speedup=%.3fx\n",
            role, rows, batch_error.nrmse, batch_error.cosine,
            batch_error.max_abs,
            worst_row.nrmse, worst_nrmse_row,
            worst_row.cosine, worst_cosine_row,
            worst_row.max_abs, worst_max_abs_row,
            sequential_ms, batch_ms,
            sequential_ms / batch_ms);
        /* These bounds are pinned from the independently recorded 33-row
         * measurement with approximately 3x headroom. The batch Q4_K WMMA
         * lane intentionally has a different reduction path; later full-logit
         * and teacher gates decide whether that arithmetic is promotable. */
        const double nrmse_bound = rows == 3u ? 4.1e-3 : 7.0e-3;
        const double cosine_bound = rows == 3u ? 0.999997 : 0.999993;
        const double max_abs_bound = rows == 3u ? 3.6e-4 : 2.5e-3;
        CHECK(batch_error.nrmse <= nrmse_bound &&
                  batch_error.cosine >= cosine_bound &&
                  batch_error.max_abs <= max_abs_bound,
              "KDA+routed batch matches sequential same-GGUF execution");
        const uint64_t batch_hash =
            fnv64(batch.data(), batch.size() * sizeof(float));
        char batch_hash_error[256] = {};
        CHECK(ds4_tp_hash_check(
                  tp.tp, UINT64_C(0x474c4d354b424154), batch_hash,
                  batch_hash_error, sizeof(batch_hash_error)) == 1,
              batch_hash_error);

        ds4_glm5_kda_digest sequential_digest = {};
        ds4_glm5_kda_digest batch_digest = {};
        CHECK(ds4_glm5_kda_layer_digest(
                  &sequential_state.value.kda.layer[4], batch_output.value,
                  1u, &sequential_digest) &&
              ds4_glm5_kda_layer_digest(
                  &batch_state.value.kda.layer[4], batch_output.value,
                  1u, &batch_digest),
              "read sequential and batch KDA recurrent digests");
        const bool recurrent_digest_equal =
            sequential_digest.q_history_fnv64 ==
                batch_digest.q_history_fnv64 &&
            sequential_digest.k_history_fnv64 ==
                batch_digest.k_history_fnv64 &&
            sequential_digest.v_history_fnv64 ==
                batch_digest.v_history_fnv64 &&
            sequential_digest.recurrent_fnv64 ==
                batch_digest.recurrent_fnv64 &&
            sequential_digest.token_count == batch_digest.token_count;
        const uint64_t batch_digest_hash =
            fnv64(&batch_digest, sizeof(batch_digest));
        char batch_digest_error[256] = {};
        CHECK(ds4_tp_hash_check(
                  tp.tp, UINT64_C(0x474c4d354b444947), batch_digest_hash,
                  batch_digest_error, sizeof(batch_digest_error)) == 1,
              batch_digest_error);
        std::fprintf(stderr,
            "GLM5 KDA+routed state role=%s recurrent_digest_equal=%d "
            "sequential_recurrent=%016llx batch_recurrent=%016llx "
            "batch_digest=%016llx\n",
            role, recurrent_digest_equal ? 1 : 0,
            (unsigned long long)sequential_digest.recurrent_fnv64,
            (unsigned long long)batch_digest.recurrent_fnv64,
            (unsigned long long)batch_digest_hash);

        std::vector<float> sequential_continue(kHcWidth);
        std::vector<float> batch_continue(kHcWidth);
        CHECK(ds4_glm5_next_embed_token(
                  &exec, continuation_id, current.value) &&
              ds4_glm5_next_layer_forward(
                  &exec, 4u, &sequential_state.value, workspace.value,
                  current.value, output.value) &&
              ds4_gpu_tensor_read(output.value, 0u,
                                   sequential_continue.data(),
                                   (uint64_t)kHcWidth * sizeof(float)) &&
              ds4_glm5_next_layer_forward(
                  &exec, 4u, &batch_state.value, workspace.value,
                  current.value, output.value) &&
              ds4_gpu_tensor_read(output.value, 0u, batch_continue.data(),
                                   (uint64_t)kHcWidth * sizeof(float)),
              "execute one-row continuation after KDA+routed batch");
        const VectorError continuation_error =
            vector_error(sequential_continue, batch_continue);
        std::fprintf(stderr,
            "GLM5 KDA+routed continuation measurement role=%s "
            "nrmse=%.9g cosine=%.12g max_abs=%.9g\n",
            role, continuation_error.nrmse, continuation_error.cosine,
            continuation_error.max_abs);
        CHECK(continuation_error.nrmse <= 5.0e-4 &&
                  continuation_error.cosine >= 0.9999999 &&
                  continuation_error.max_abs <= 2.5e-5,
              "ordinary decode continues from KDA+routed batch state");
        const uint64_t continuation_hash = fnv64(
            batch_continue.data(), batch_continue.size() * sizeof(float));
        char continuation_hash_error[256] = {};
        CHECK(ds4_tp_hash_check(
                  tp.tp, UINT64_C(0x474c4d354b434f4e), continuation_hash,
                  continuation_hash_error,
                  sizeof(continuation_hash_error)) == 1,
              continuation_hash_error);
        CHECK(sequential_state.value.kda.layer[4].token_count == rows + 1u &&
                  batch_state.value.kda.layer[4].token_count == rows + 1u &&
                  sequential_state.value.valid && batch_state.value.valid,
              "both KDA+routed paths commit identical recurrent length");
        std::fprintf(stderr,
            "PASS GLM5 KDA+routed batch role=%s rows=%u layer=4 "
            "nrmse=%.9g cosine=%.12g max_abs=%.9g output=%016llx "
            "continue_nrmse=%.9g continue_cosine=%.12g "
            "continue_max_abs=%.9g continuation=%016llx "
            "sequential_ms=%.3f batch_ms=%.3f speedup=%.3fx "
            "tp_seq=%llu packed_q4_bytes=%llu rdma=1\n",
            role, rows, batch_error.nrmse, batch_error.cosine,
            batch_error.max_abs, (unsigned long long)batch_hash,
            continuation_error.nrmse, continuation_error.cosine,
            continuation_error.max_abs,
            (unsigned long long)continuation_hash,
            sequential_ms, batch_ms, sequential_ms / batch_ms,
            (unsigned long long)sequence,
            (unsigned long long)ds4_gpu_q4k_packed_slice_bytes());
        if (kda_profile_repeats > 0u) {
            std::vector<double> profile_ms;
            profile_ms.reserve(kda_profile_repeats > 1u ?
                               kda_profile_repeats - 1u : 1u);
            for (uint32_t repeat = 0u; repeat < kda_profile_repeats;
                 ++repeat) {
                StateGuard profile_state;
                CHECK(ds4_glm5_next_state_init(
                          &profile_state.value, &offsets, rows + 1u,
                          nullptr),
                      "allocate isolated KDA+routed profile state");
                const auto profile_begin = Clock::now();
                CHECK(ds4_glm5_next_layer_forward_batch(
                          &exec, 4u, &profile_state.value,
                          batch_workspace.value, batch_input.value,
                          batch_output.value, rows),
                      "execute isolated KDA+routed profile repeat");
                const double elapsed_ms =
                    std::chrono::duration<double, std::milli>(
                        Clock::now() - profile_begin).count();
                std::fprintf(stderr,
                    "GLM5 KDA+routed profile role=%s rows=%u "
                    "repeat=%u warmup=%d elapsed_ms=%.3f tp_seq=%llu\n",
                    role, rows, repeat, repeat == 0u ? 1 : 0,
                    elapsed_ms, (unsigned long long)sequence);
                if (repeat > 0u || kda_profile_repeats == 1u)
                    profile_ms.push_back(elapsed_ms);
            }
            std::sort(profile_ms.begin(), profile_ms.end());
            const size_t mid = profile_ms.size() / 2u;
            const double median_ms = profile_ms.size() % 2u ?
                profile_ms[mid] :
                0.5 * (profile_ms[mid - 1u] + profile_ms[mid]);
            std::fprintf(stderr,
                "GLM5 KDA+routed profile summary role=%s rows=%u "
                "measured=%zu median_ms=%.3f min_ms=%.3f max_ms=%.3f\n",
                role, rows, profile_ms.size(), median_ms,
                profile_ms.front(), profile_ms.back());
        }
        return true;
    }

    if (text_mode) {
        using Clock = std::chrono::steady_clock;
        const auto prompt_begin = Clock::now();
        uint32_t executed = 0u;
        LayerTiming prompt_layer_timing;
        for (int i = 0; i < prompt_tokens.value.len; ++i) {
            CHECK(execute_full_token(
                      exec, state.value, workspace.value,
                      (uint32_t)prompt_tokens.value.v[i],
                      current.value, output.value,
                      layer_timing ? &prompt_layer_timing : nullptr),
                  "execute exact chat prompt token");
            executed++;
        }
        CHECK(state.value.valid &&
                  sequence == (uint64_t)executed * 53u,
              "chat prompt commits one complete TP schedule per token");
        const auto prompt_end = Clock::now();
        if (layer_timing) {
            const double classified = prompt_layer_timing.kda_dense_ms +
                prompt_layer_timing.kda_routed_ms +
                prompt_layer_timing.mla_routed_ms;
            std::fprintf(stderr,
                "GLM5 prompt layer timing role=%s tokens=%d "
                "kda_dense_ms=%.3f kda_dense_calls=%llu "
                "kda_routed_ms=%.3f kda_routed_calls=%llu "
                "mla_routed_ms=%.3f mla_routed_calls=%llu "
                "classified_ms=%.3f\n",
                role, prompt_tokens.value.len,
                prompt_layer_timing.kda_dense_ms,
                (unsigned long long)prompt_layer_timing.kda_dense_calls,
                prompt_layer_timing.kda_routed_ms,
                (unsigned long long)prompt_layer_timing.kda_routed_calls,
                prompt_layer_timing.mla_routed_ms,
                (unsigned long long)prompt_layer_timing.mla_routed_calls,
                classified);
        }

        std::vector<float> host_logits(perf_mode ? 0u : 154880u);
        std::vector<uint32_t> generated;
        std::string text;
        double teacher_nll_sum = 0.0;
        double projection_ms = 0.0;
        double recurrent_ms = 0.0;
        const auto decode_begin = Clock::now();
        for (uint32_t step = 0u; step < text_generate; ++step) {
            uint32_t top1 = 0u;
            uint64_t logits_hash = 0u;
            const auto projection_begin = Clock::now();
            CHECK(read_argmax(exec, workspace.value, current.value,
                              logits.value, gpu_argmax.value, !perf_mode,
                              host_logits, top1, logits_hash),
                  "select real-text greedy token");
            const auto projection_end = Clock::now();
            projection_ms += std::chrono::duration<double, std::milli>(
                projection_end - projection_begin).count();
            char logits_error[256] = {};
            CHECK(ds4_tp_hash_check(
                      tp.tp,
                      UINT64_C(0x474c4d3554584c00) ^ step,
                      logits_hash, logits_error,
                      sizeof(logits_error)) == 1,
                  logits_error);
            const uint32_t token = teacher_ids.empty() ?
                top1 : teacher_ids[step];
            double teacher_nll = 0.0;
            if (!teacher_ids.empty()) {
                const float max_logit = *std::max_element(
                    host_logits.begin(), host_logits.end());
                double exp_sum = 0.0;
                for (float value : host_logits)
                    exp_sum += std::exp((double)value - max_logit);
                teacher_nll = (double)max_logit + std::log(exp_sum) -
                              host_logits[token];
                CHECK(std::isfinite(teacher_nll) && teacher_nll >= 0.0,
                      "teacher token has finite nonnegative NLL");
                teacher_nll_sum += teacher_nll;
            }
            generated.push_back(token);
            const bool stop = ds4_glm5_next_text_codec_token_is_stop(
                codec.value, (int)token);
            if (!stop) {
                size_t piece_len = 0u;
                char *piece = ds4_glm5_next_text_codec_token_text(
                    codec.value, (int)token, &piece_len);
                CHECK(piece != nullptr, "detokenize generated GLM5 token");
                text.append(piece, piece_len);
                free(piece);
            }
            if (teacher_ids.empty()) {
                std::fprintf(stderr,
                    "GLM5 text step role=%s step=%u token=%u top1=%u "
                    "logits=%016llx\n",
                    role, step, token, top1,
                    (unsigned long long)logits_hash);
            } else {
                std::fprintf(stderr,
                    "GLM5 text step role=%s step=%u token=%u top1=%u "
                    "teacher_nll=%.9g logits=%016llx\n",
                    role, step, token, top1, teacher_nll,
                    (unsigned long long)logits_hash);
            }
            if (stop) {
                CHECK(teacher_ids.empty() || step + 1u == teacher_ids.size(),
                      "teacher stop token must be the final prescribed token");
                break;
            }
            if (step + 1u < text_generate) {
                const auto recurrent_begin = Clock::now();
                CHECK(execute_full_token(exec, state.value, workspace.value,
                                         token, current.value, output.value),
                      "feed generated token into recurrent GLM5 state");
                const auto recurrent_end = Clock::now();
                recurrent_ms += std::chrono::duration<double, std::milli>(
                    recurrent_end - recurrent_begin).count();
                executed++;
                CHECK(sequence == (uint64_t)executed * 53u,
                      "generated token commits one complete TP schedule");
            }
        }
        const auto decode_end = Clock::now();
        CHECK(!generated.empty(), "real-text generation produced a token");
        CHECK(teacher_ids.empty() || generated.size() == teacher_ids.size(),
              "teacher-forced run consumed every prescribed token");
        const uint64_t generated_hash = fnv64(
            generated.data(), generated.size() * sizeof(generated[0]));
        char generated_error[256] = {};
        CHECK(ds4_tp_hash_check(tp.tp, UINT64_C(0x474c4d355458544f),
                                generated_hash, generated_error,
                                sizeof(generated_error)) == 1,
              generated_error);
        const double prompt_ms =
            std::chrono::duration<double, std::milli>(
                prompt_end - prompt_begin).count();
        const double decode_ms =
            std::chrono::duration<double, std::milli>(
                decode_end - decode_begin).count();
        std::fprintf(stderr,
            "GLM5 staged timing role=%s prompt_tokens=%d "
            "prompt_ms=%.3f prefill_tps=%.6f generated=%zu "
            "decode_ms=%.3f decode_tps=%.6f projection_ms=%.3f "
            "recurrent_ms=%.3f full_logit_validation=%d\n",
            role, prompt_tokens.value.len, prompt_ms,
            prompt_tokens.value.len * 1000.0 / prompt_ms,
            generated.size(), decode_ms,
            generated.size() * 1000.0 / decode_ms,
            projection_ms, recurrent_ms, perf_mode ? 0 : 1);
        if (!teacher_ids.empty()) {
            const uint64_t teacher_nll_hash =
                fnv64(&teacher_nll_sum, sizeof(teacher_nll_sum));
            char teacher_error[256] = {};
            CHECK(ds4_tp_hash_check(
                      tp.tp, UINT64_C(0x474c4d35544e4c4c),
                      teacher_nll_hash, teacher_error,
                      sizeof(teacher_error)) == 1,
                  teacher_error);
            std::fprintf(stderr,
                "PASS GLM5 teacher-forced role=%s tokens=%zu "
                "token_fnv=%016llx nll_sum=%.9g nll_mean=%.9g "
                "nll_fnv=%016llx\n",
                role, teacher_ids.size(),
                (unsigned long long)teacher_hash, teacher_nll_sum,
                teacher_nll_sum / teacher_ids.size(),
                (unsigned long long)teacher_nll_hash);
        }
        if (leader) {
            std::fprintf(stdout, "%s\n", text.c_str());
            std::fflush(stdout);
        }
        std::fprintf(stderr,
            "PASS GLM5 real-text role=%s prompt_tokens=%d generated=%zu "
            "generated_fnv=%016llx tp_seq=%llu packed_q4_bytes=%llu "
            "rdma=1\n",
            role, prompt_tokens.value.len, generated.size(),
            (unsigned long long)generated_hash,
            (unsigned long long)sequence,
            (unsigned long long)ds4_gpu_q4k_packed_slice_bytes());
        return true;
    }

    CHECK(ds4_glm5_next_embed_token(&exec, 42u, current.value),
          "embed token 42");
    for (uint32_t il = 0u; il < 3u; ++il) {
        CHECK(ds4_glm5_next_layer_forward(
                  &exec, il, &state.value, workspace.value,
                  current.value, output.value),
              "execute production dense prefix");
        std::swap(current.value, output.value);
    }
    std::vector<float> prefix(kHcWidth);
    CHECK(ds4_gpu_tensor_read(current.value, 0u, prefix.data(),
                              prefix.size() * sizeof(float)),
          "read production dense prefix");
    const uint64_t prefix_hash =
        fnv64(prefix.data(), prefix.size() * sizeof(float));
    if (prefix_hash != kPrefixFNV) {
        std::fprintf(stderr,
                     "dense-prefix fingerprint expected=%016llx observed=%016llx\n",
                     (unsigned long long)kPrefixFNV,
                     (unsigned long long)prefix_hash);
    }
    CHECK(prefix_hash == kPrefixFNV,
          "production prefix matches frozen real-GGUF fixture");

    if (full_trunk) {
        for (uint32_t il = 3u; il < DS4_GLM5_NEXT_TRUNK_COUNT; ++il) {
            CHECK(ds4_glm5_next_layer_forward(
                      &exec, il, &state.value, workspace.value,
                      current.value, output.value),
                  "execute complete GLM5.3 trunk layer over RoCE");
            std::swap(current.value, output.value);
        }
        CHECK(state.value.valid && sequence == 53u,
              "complete trunk emits the exact hybrid 53-gate schedule");
        for (uint32_t il = 0u; il < DS4_GLM5_NEXT_TRUNK_COUNT; ++il) {
            if (ds4_glm5_next_layer_is_mla(il)) {
                CHECK(state.value.mla[il].token_count == 1u,
                      "full trunk commits one MLA cache row per MLA layer");
            } else {
                CHECK(state.value.kda.layer[il].token_count == 1u,
                      "full trunk commits one recurrent KDA step per KDA layer");
            }
        }
        std::vector<float> trunk(kHcWidth);
        CHECK(ds4_gpu_tensor_read(current.value, 0u, trunk.data(),
                                  trunk.size() * sizeof(float)),
              "read complete trunk output");
        const uint64_t trunk_hash = fnv64(
            trunk.data(), trunk.size() * sizeof(float));
        char trunk_error[256] = {};
        CHECK(ds4_tp_hash_check(tp.tp, UINT64_C(0x474c4d3546554c4c),
                                trunk_hash, trunk_error,
                                sizeof(trunk_error)) == 1,
              trunk_error);
        CHECK(ds4_glm5_next_output_logits(
                  &exec, workspace.value, current.value, logits.value) &&
              ds4_gpu_synchronize(),
              "execute replicated GLM5.3 output head");
        std::vector<float> full_logits(154880u);
        CHECK(ds4_gpu_tensor_read(
                  logits.value, 0u, full_logits.data(),
                  full_logits.size() * sizeof(float)),
              "read complete GLM5.3 vocabulary logits");
        const uint64_t logits_hash = fnv64(
            full_logits.data(), full_logits.size() * sizeof(float));
        char logits_error[256] = {};
        CHECK(ds4_tp_hash_check(tp.tp, UINT64_C(0x474c4d354c4f4749),
                                logits_hash, logits_error,
                                sizeof(logits_error)) == 1,
              logits_error);
        uint32_t argmax = 0u;
        float logit_min = full_logits[0], logit_max = full_logits[0];
        CHECK(std::isfinite(full_logits[0]), "all logits are finite");
        for (uint32_t token = 1u; token < full_logits.size(); ++token) {
            CHECK(std::isfinite(full_logits[token]), "all logits are finite");
            logit_min = std::min(logit_min, full_logits[token]);
            logit_max = std::max(logit_max, full_logits[token]);
            if (full_logits[token] > full_logits[argmax]) argmax = token;
        }
        CHECK(logit_max > logit_min,
              "complete vocabulary logits are nonconstant");
        const uint64_t packed = ds4_gpu_q4k_packed_slice_bytes();
        const uint64_t expected_packed = kshard.packed_total_bytes;
        CHECK(packed == expected_packed,
              "full trunk owns one compact Q4 shard per routed layer");
        std::fprintf(stderr,
            "PASS GLM5 prefix->layer3 full-trunk role=%s "
            "trunk_output=%016llx logits=%016llx argmax=%u "
            "tp_seq=%llu packed_q4_bytes=%llu rdma=1\n",
            role, (unsigned long long)trunk_hash,
            (unsigned long long)logits_hash, argmax,
            (unsigned long long)sequence, (unsigned long long)packed);
        if (full_tokens == 2u) {
            CHECK(argmax == 154822u,
                  "first greedy token matches the frozen two-token fixture");
            CHECK(ds4_glm5_next_embed_token(&exec, argmax, current.value),
                  "embed first greedy argmax for second full token");
            for (uint32_t il = 0u; il < DS4_GLM5_NEXT_TRUNK_COUNT; ++il) {
                CHECK(ds4_glm5_next_layer_forward(
                          &exec, il, &state.value, workspace.value,
                          current.value, output.value),
                      "execute second greedy token through complete trunk");
                std::swap(current.value, output.value);
            }
            CHECK(state.value.valid && sequence == 106u,
                  "two full tokens emit two exact 53-gate schedules");
            for (uint32_t il = 0u; il < DS4_GLM5_NEXT_TRUNK_COUNT; ++il) {
                if (ds4_glm5_next_layer_is_mla(il)) {
                    CHECK(state.value.mla[il].token_count == 2u,
                          "second token commits every MLA layer state");
                } else {
                    CHECK(state.value.kda.layer[il].token_count == 2u,
                          "second token commits every KDA layer state");
                }
            }
            std::vector<float> trunk2(kHcWidth);
            CHECK(ds4_gpu_tensor_read(current.value, 0u, trunk2.data(),
                                      trunk2.size() * sizeof(float)),
                  "read second greedy full-trunk output");
            const uint64_t trunk2_hash = fnv64(
                trunk2.data(), trunk2.size() * sizeof(float));
            char trunk2_error[256] = {};
            CHECK(ds4_tp_hash_check(tp.tp, UINT64_C(0x474c4d3547325452),
                                    trunk2_hash, trunk2_error,
                                    sizeof(trunk2_error)) == 1,
                  trunk2_error);
            CHECK(ds4_glm5_next_output_logits(
                      &exec, workspace.value, current.value, logits.value) &&
                  ds4_gpu_synchronize(),
                  "execute second greedy vocabulary projection");
            CHECK(ds4_gpu_tensor_read(
                      logits.value, 0u, full_logits.data(),
                      full_logits.size() * sizeof(float)),
                  "read second greedy vocabulary logits");
            const uint64_t logits2_hash = fnv64(
                full_logits.data(), full_logits.size() * sizeof(float));
            char logits2_error[256] = {};
            CHECK(ds4_tp_hash_check(tp.tp, UINT64_C(0x474c4d3547324c47),
                                    logits2_hash, logits2_error,
                                    sizeof(logits2_error)) == 1,
                  logits2_error);
            uint32_t argmax2 = 0u;
            float logit2_min = full_logits[0], logit2_max = full_logits[0];
            CHECK(std::isfinite(full_logits[0]),
                  "second-token logits start finite");
            for (uint32_t token = 1u; token < full_logits.size(); ++token) {
                CHECK(std::isfinite(full_logits[token]),
                      "all second-token logits are finite");
                logit2_min = std::min(logit2_min, full_logits[token]);
                logit2_max = std::max(logit2_max, full_logits[token]);
                if (full_logits[token] > full_logits[argmax2]) argmax2 = token;
            }
            CHECK(logit2_max > logit2_min &&
                      trunk2_hash != trunk_hash &&
                      trunk2_hash == UINT64_C(0x53f8b3cef301d00c) &&
                      logits2_hash == UINT64_C(0xb3fa54df32db51ec) &&
                      argmax2 == 20u &&
                      ds4_gpu_q4k_packed_slice_bytes() == expected_packed,
                  "second-token frozen fixture is stateful and cache-free");
            std::fprintf(stderr,
                "PASS GLM5 prefix->layer3 greedy2 role=%s "
                "greedy1=%u greedy2_trunk=%016llx "
                "greedy2_logits=%016llx greedy2=%u tp_seq=%llu "
                "packed_q4_bytes=%llu rdma=1\n",
                role, argmax, (unsigned long long)trunk2_hash,
                (unsigned long long)logits2_hash, argmax2,
                (unsigned long long)sequence,
                (unsigned long long)ds4_gpu_q4k_packed_slice_bytes());
        }
        ds4_glm5_next_state_free(&state.value);
        std::memset(&state.value, 0, sizeof(state.value));
        sequence = 0u;
        CHECK(ds4_glm5_next_state_init(&state.value, &offsets, 2u, nullptr) &&
                  ds4_glm5_next_embed_token(&exec, 42u, current.value),
              "reset state and re-embed token 42 for multi-token coverage");
        for (uint32_t il = 0u; il < 3u; ++il) {
            CHECK(ds4_glm5_next_layer_forward(
                      &exec, il, &state.value, workspace.value,
                      current.value, output.value),
                  "rebuild token-0 dense-prefix KDA state after full trunk");
            std::swap(current.value, output.value);
        }
        std::vector<float> replay_prefix(kHcWidth);
        CHECK(ds4_gpu_tensor_read(current.value, 0u, replay_prefix.data(),
                                  replay_prefix.size() * sizeof(float)) &&
                  fnv64(replay_prefix.data(),
                        replay_prefix.size() * sizeof(float)) == kPrefixFNV,
              "rebuilt token-0 prefix and KDA state match frozen fixture");
    }

    CHECK(ds4_glm5_next_layer_forward(
              &exec, 3u, &state.value, workspace.value,
              current.value, output.value),
          "execute production MLA+routed layer3 over RoCE");
    CHECK(state.value.valid && state.value.mla[3].valid &&
          state.value.mla[3].token_count == 1u && sequence == 2u,
          "layer3 commits exactly one token after both exchanges");

    std::vector<float> result(kHcWidth), kv(512u), index_key(128u),
        pool_gate(128u);
    CHECK(ds4_gpu_tensor_read(output.value, 0u, result.data(),
                              result.size() * sizeof(float)) &&
          ds4_gpu_tensor_read(state.value.mla[3].compact_kv, 0u,
                              kv.data(), kv.size() * sizeof(float)) &&
          ds4_gpu_tensor_read(state.value.mla[3].index_tail, 0u,
                              index_key.data(), index_key.size() * sizeof(float)) &&
          ds4_gpu_tensor_read(state.value.mla[3].pool_gate_tail, 0u,
                              pool_gate.data(), pool_gate.size() * sizeof(float)),
          "read committed layer3 output and resident token0 rows");
    const uint64_t output_hash = fnv64(
        result.data(), result.size() * sizeof(float));
    const uint64_t kv_hash = fnv64(kv.data(), kv.size() * sizeof(float));
    const uint64_t index_hash = fnv64(
        index_key.data(), index_key.size() * sizeof(float));
    const uint64_t gate_hash = fnv64(
        pool_gate.data(), pool_gate.size() * sizeof(float));
    const uint64_t layer3_packed_q4_bytes =
        ds4_gpu_q4k_packed_slice_bytes();
    char error[256] = {};
    CHECK(ds4_tp_hash_check(tp.tp, UINT64_C(0x474c4d35334f5554),
                            output_hash, error, sizeof(error)) == 1, error);
    CHECK(layer3_packed_q4_bytes != 0u,
          "layer3 owns compact Q4 rank residency");

    CHECK(ds4_glm5_next_layer_forward(
              &exec, 4u, &state.value, workspace.value,
              output.value, current.value),
          "compose production KDA+routed layer4 over RoCE");
    ds4_glm5_kda_digest kda4_token0 = {};
    CHECK(state.value.kda.layer[4].token_count == 1u && sequence == 3u &&
          ds4_glm5_kda_layer_digest(&state.value.kda.layer[4], current.value,
                                    kHcWidth, &kda4_token0),
          "layer4 commits one recurrent KDA step and one FFN exchange");
    const uint64_t packed_q4_bytes = ds4_gpu_q4k_packed_slice_bytes();
    CHECK((full_trunk ? packed_q4_bytes == layer3_packed_q4_bytes
                      : packed_q4_bytes > layer3_packed_q4_bytes) &&
          ds4_tp_hash_check(
              tp.tp, UINT64_C(0x474c4d35344b4430),
              fnv64(&kda4_token0, sizeof(kda4_token0)),
              error, sizeof(error)) == 1,
          "layer4 state/output are rank-identical and add one Q4 shard");

    CHECK(ds4_glm5_next_embed_token(&exec, 43u, current.value),
          "embed token 43");
    for (uint32_t il = 0u; il < 3u; ++il) {
        CHECK(ds4_glm5_next_layer_forward(
                  &exec, il, &state.value, workspace.value,
                  current.value, output.value),
              "execute second-token dense prefix");
        std::swap(current.value, output.value);
    }
    CHECK(ds4_glm5_next_layer_forward(
              &exec, 3u, &state.value, workspace.value,
              current.value, output.value),
          "execute second-token MLA cache-read layer3 over RoCE");
    CHECK(state.value.valid && state.value.mla[3].token_count == 2u &&
          sequence == 5u &&
          ds4_gpu_q4k_packed_slice_bytes() == packed_q4_bytes,
          "second token commits after two exchanges without Q4 duplication");

    std::vector<float> result1(kHcWidth), kv2(1024u), index2(256u),
        pool2(256u);
    CHECK(ds4_gpu_tensor_read(output.value, 0u, result1.data(),
                              result1.size() * sizeof(float)) &&
          ds4_gpu_tensor_read(state.value.mla[3].compact_kv, 0u,
                              kv2.data(), kv2.size() * sizeof(float)) &&
          ds4_gpu_tensor_read(state.value.mla[3].index_tail, 0u,
                              index2.data(), index2.size() * sizeof(float)) &&
          ds4_gpu_tensor_read(state.value.mla[3].pool_gate_tail, 0u,
                              pool2.data(), pool2.size() * sizeof(float)),
          "read second-token output and two resident cache rows");
    CHECK(std::memcmp(kv2.data(), kv.data(), kv.size() * sizeof(float)) == 0 &&
          std::memcmp(index2.data(), index_key.data(),
                      index_key.size() * sizeof(float)) == 0 &&
          std::memcmp(pool2.data(), pool_gate.data(),
                      pool_gate.size() * sizeof(float)) == 0,
          "second token preserves the committed token0 cache rows exactly");
    const uint64_t output1_hash = fnv64(
        result1.data(), result1.size() * sizeof(float));
    const uint64_t kv2_hash = fnv64(kv2.data(), kv2.size() * sizeof(float));
    const uint64_t index2_hash = fnv64(
        index2.data(), index2.size() * sizeof(float));
    const uint64_t gate2_hash = fnv64(
        pool2.data(), pool2.size() * sizeof(float));
    uint64_t state_hash = fnv64(&output1_hash, sizeof(output1_hash));
    state_hash ^= kv2_hash;
    state_hash ^= index2_hash;
    state_hash ^= gate2_hash;
    CHECK(output1_hash != output_hash &&
          std::memcmp(kv2.data(), kv2.data() + kv.size(),
                      kv.size() * sizeof(float)) != 0 &&
          ds4_tp_hash_check(tp.tp, UINT64_C(0x474c4d3533544f4b),
                            state_hash, error, sizeof(error)) == 1,
          "second-token output/state are nontrivial and rank-identical");

    CHECK(ds4_glm5_next_layer_forward(
              &exec, 4u, &state.value, workspace.value,
              output.value, current.value),
          "execute second-token KDA+routed layer4 over RoCE");
    ds4_glm5_kda_digest kda4_token1 = {};
    CHECK(state.value.valid && state.value.kda.layer[4].token_count == 2u &&
          sequence == 6u &&
          ds4_gpu_q4k_packed_slice_bytes() == packed_q4_bytes &&
          ds4_glm5_kda_layer_digest(&state.value.kda.layer[4], current.value,
                                    kHcWidth, &kda4_token1) &&
          kda4_token1.token_count == 2u &&
          kda4_token1.output_fnv64 != kda4_token0.output_fnv64 &&
          ds4_tp_hash_check(
              tp.tp, UINT64_C(0x474c4d35344b4431),
              fnv64(&kda4_token1, sizeof(kda4_token1)),
              error, sizeof(error)) == 1,
          "layer4 token1 advances recurrent state without Q4 duplication");
    std::fprintf(stderr,
        "PASS GLM5 prefix->layer3 token0 role=%s output=%016llx kv=%016llx "
        "index=%016llx pool_gate=%016llx token1_output=%016llx "
        "kv2=%016llx index2=%016llx pool2=%016llx "
        "layer4_token0=%016llx layer4_token1=%016llx tp_seq=%llu "
        "layer3_packed_q4_bytes=%llu packed_q4_bytes=%llu "
        "window_cache_bytes=0 rdma=1\n",
        role, (unsigned long long)output_hash,
        (unsigned long long)kv_hash, (unsigned long long)index_hash,
        (unsigned long long)gate_hash, (unsigned long long)output1_hash,
        (unsigned long long)kv2_hash, (unsigned long long)index2_hash,
        (unsigned long long)gate2_hash,
        (unsigned long long)kda4_token0.output_fnv64,
        (unsigned long long)kda4_token1.output_fnv64,
        (unsigned long long)sequence,
        (unsigned long long)layer3_packed_q4_bytes,
        (unsigned long long)packed_q4_bytes);
    return true;
}

}  // namespace

int main() {
    const bool ok = run();
    ds4_gpu_cleanup();
    return ok ? 0 : 1;
}
