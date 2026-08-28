#include "ds4_glm5_kda.h"
#include "ds4_gpu.h"
#include "ds4_gpu_mgpu.h"
extern "C" {
#include "ds4_tp.h"
}

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <string>
#include <unordered_map>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define CHECK(expr, message) do { \
    if (!(expr)) { \
        std::fprintf(stderr, "FAIL %s (line %d)\n", message, __LINE__); \
        return false; \
    } \
} while (0)

struct Cursor {
    const uint8_t *base;
    uint64_t size;
    uint64_t pos;

    bool take(void *out, uint64_t bytes) {
        if (pos > size || bytes > size - pos) return false;
        std::memcpy(out, base + pos, (size_t)bytes);
        pos += bytes;
        return true;
    }
    bool u32(uint32_t &value) { return take(&value, sizeof(value)); }
    bool u64(uint64_t &value) { return take(&value, sizeof(value)); }
    bool string(std::string &value) {
        uint64_t length = 0;
        if (!u64(length) || length > size - pos || length > SIZE_MAX) return false;
        value.assign((const char *)base + pos, (size_t)length);
        pos += length;
        return true;
    }
    bool skip(uint64_t bytes) {
        if (bytes > size - pos) return false;
        pos += bytes;
        return true;
    }
};

struct TensorInfo {
    std::vector<uint64_t> dims;
    uint32_t type;
    uint64_t relative_offset;
};

static bool skip_metadata(Cursor &cursor, uint32_t type) {
    static const uint8_t scalar_bytes[] = {1, 1, 2, 2, 4, 4, 4, 1};
    if (type < sizeof(scalar_bytes)) return cursor.skip(scalar_bytes[type]);
    if (type == 8u) {
        std::string ignored;
        return cursor.string(ignored);
    }
    if (type == 9u) {
        uint32_t element_type = 0;
        uint64_t count = 0;
        if (!cursor.u32(element_type) || !cursor.u64(count)) return false;
        for (uint64_t i = 0; i < count; ++i) {
            if (!skip_metadata(cursor, element_type)) return false;
        }
        return true;
    }
    if (type >= 10u && type <= 12u) return cursor.skip(8u);
    return false;
}

struct MappedGGUF {
    int fd = -1;
    uint8_t *map = nullptr;
    uint64_t size = 0;
    uint64_t data_start = 0;
    float rms_norm_eps = 0.0f;
    std::unordered_map<std::string, TensorInfo> tensors;

    void close_all() {
        if (map && size) munmap(map, (size_t)size);
        if (fd >= 0) close(fd);
        fd = -1;
        map = nullptr;
        size = 0;
    }
};

static bool open_gguf(const char *path, MappedGGUF &gguf) {
    gguf.fd = open(path, O_RDONLY | O_CLOEXEC);
    CHECK(gguf.fd >= 0, "open GLM5 GGUF");
    struct stat st = {};
    CHECK(fstat(gguf.fd, &st) == 0 && st.st_size > 0,
          "stat GLM5 GGUF");
    gguf.size = (uint64_t)st.st_size;
    gguf.map = (uint8_t *)mmap(nullptr, (size_t)gguf.size, PROT_READ,
                               MAP_PRIVATE, gguf.fd, 0);
    CHECK(gguf.map != MAP_FAILED, "mmap GLM5 GGUF");
    Cursor cursor = {gguf.map, gguf.size, 0};
    uint32_t magic = 0, version = 0;
    uint64_t tensor_count = 0, metadata_count = 0;
    CHECK(cursor.u32(magic) && cursor.u32(version) &&
          cursor.u64(tensor_count) && cursor.u64(metadata_count) &&
          magic == UINT32_C(0x46554747) && version == 3u,
          "parse GGUF v3 header");
    uint32_t alignment = 32u;
    for (uint64_t i = 0; i < metadata_count; ++i) {
        std::string key;
        uint32_t type = 0;
        CHECK(cursor.string(key) && cursor.u32(type), "parse GGUF metadata key");
        if (key == "general.alignment" && type == 4u) {
            CHECK(cursor.u32(alignment) && alignment != 0u,
                  "parse GGUF alignment");
        } else if (key ==
                       "glm5-next.attention.layer_norm_rms_epsilon" &&
                   type == 6u) {
            CHECK(cursor.take(&gguf.rms_norm_eps,
                              sizeof(gguf.rms_norm_eps)) &&
                      gguf.rms_norm_eps == 1.0e-5f,
                  "parse expected GLM5-next RMS epsilon");
        } else {
            CHECK(skip_metadata(cursor, type), "skip GGUF metadata value");
        }
    }
    for (uint64_t i = 0; i < tensor_count; ++i) {
        std::string name;
        uint32_t dimensions = 0;
        CHECK(cursor.string(name) && cursor.u32(dimensions) &&
              dimensions > 0u && dimensions <= 4u,
              "parse GGUF tensor name/dimensions");
        TensorInfo info;
        info.dims.resize(dimensions);
        for (uint32_t d = 0; d < dimensions; ++d)
            CHECK(cursor.u64(info.dims[d]), "parse GGUF tensor dimension");
        CHECK(cursor.u32(info.type) && cursor.u64(info.relative_offset),
              "parse GGUF tensor type/offset");
        gguf.tensors.emplace(std::move(name), std::move(info));
    }
    CHECK(cursor.pos <= UINT64_MAX - (alignment - 1u),
          "GGUF data alignment overflow");
    gguf.data_start = (cursor.pos + alignment - 1u) / alignment * alignment;
    CHECK(gguf.data_start <= gguf.size, "GGUF data section in file");
    CHECK(gguf.rms_norm_eps == 1.0e-5f,
          "GLM5-next RMS epsilon metadata is required");
    return true;
}

static bool tensor_offset(const MappedGGUF &gguf, const char *name,
                          std::initializer_list<uint64_t> dims,
                          uint32_t type, uint64_t &offset) {
    const auto found = gguf.tensors.find(name);
    CHECK(found != gguf.tensors.end(), "required layer-0 tensor present");
    const TensorInfo &info = found->second;
    CHECK(info.type == type && info.dims.size() == dims.size() &&
          std::equal(info.dims.begin(), info.dims.end(), dims.begin()),
          "layer-0 tensor type and shape");
    CHECK(info.relative_offset <= UINT64_MAX - gguf.data_start,
          "layer-0 tensor offset overflow");
    offset = gguf.data_start + info.relative_offset;
    CHECK(offset < gguf.size, "layer-0 tensor offset in model");
    return true;
}

static bool bind_layer0(const MappedGGUF &g,
                        ds4_glm5_kda_weight_offsets &w) {
    return tensor_offset(g, "blk.0.attn_norm.weight", {4096}, 0, w.attn_norm) &&
           tensor_offset(g, "blk.0.kda_q.weight", {4096, 8192}, 30, w.q) &&
           tensor_offset(g, "blk.0.kda_k.weight", {4096, 8192}, 30, w.k) &&
           tensor_offset(g, "blk.0.kda_v.weight", {4096, 8192}, 30, w.v) &&
           tensor_offset(g, "blk.0.kda_output.weight", {8192, 4096}, 30, w.output) &&
           tensor_offset(g, "blk.0.kda_q_conv.weight", {4, 1, 8192}, 0, w.q_conv) &&
           tensor_offset(g, "blk.0.kda_k_conv.weight", {4, 1, 8192}, 0, w.k_conv) &&
           tensor_offset(g, "blk.0.kda_v_conv.weight", {4, 1, 8192}, 0, w.v_conv) &&
           tensor_offset(g, "blk.0.kda_f_a.weight", {4096, 128}, 30, w.f_a) &&
           tensor_offset(g, "blk.0.kda_f_b.weight", {128, 8192}, 30, w.f_b) &&
           tensor_offset(g, "blk.0.kda_g_a.weight", {4096, 128}, 30, w.g_a) &&
           tensor_offset(g, "blk.0.kda_g_b.weight", {128, 8192}, 30, w.g_b) &&
           tensor_offset(g, "blk.0.kda_beta.weight", {4096, 64}, 30, w.beta) &&
           tensor_offset(g, "blk.0.kda_o_norm.weight", {128}, 0, w.o_norm) &&
           tensor_offset(g, "blk.0.kda_dt_bias.weight", {8192}, 0, w.dt_bias) &&
           tensor_offset(g, "blk.0.kda_a_log.weight", {64}, 0, w.a_log);
}

static bool read_f32_file(const std::string &path, uint64_t count,
                          std::vector<float> &values) {
    FILE *fp = std::fopen(path.c_str(), "rb");
    CHECK(fp != nullptr, "open oracle FP32 dump");
    values.resize((size_t)count);
    const size_t got = std::fread(values.data(), sizeof(float),
                                  (size_t)count, fp);
    const int extra = std::fgetc(fp);
    const int close_rc = std::fclose(fp);
    CHECK(got == count && extra == EOF && close_rc == 0,
          "read exact oracle FP32 dump");
    return true;
}

struct ErrorStats {
    double max_abs = 0.0;
    double max_rel = 0.0;
    long double error_sq = 0.0;
    long double reference_sq = 0.0;
    bool finite = true;
};

static ErrorStats compare(const std::vector<float> &reference,
                          const std::vector<float> &candidate) {
    ErrorStats stats;
    if (reference.size() != candidate.size()) {
        stats.finite = false;
        return stats;
    }
    for (size_t i = 0; i < reference.size(); ++i) {
        const double r = reference[i];
        const double c = candidate[i];
        if (!std::isfinite(r) || !std::isfinite(c)) stats.finite = false;
        const double error = std::fabs(c - r);
        stats.max_abs = std::max(stats.max_abs, error);
        stats.max_rel = std::max(stats.max_rel,
            error / std::max(std::fabs(r), 1.0e-6));
        stats.error_sq += (long double)error * error;
        stats.reference_sq += (long double)r * r;
    }
    return stats;
}

static double nmse(const ErrorStats &stats) {
    return (double)(stats.error_sq /
        std::max(stats.reference_sq, (long double)1.0e-30));
}

static bool read_state(const ds4_glm5_kda_layer_state &state,
                       std::vector<float> &values) {
    const uint64_t history =
        (uint64_t)DS4_GLM5_KDA_CHANNELS * DS4_GLM5_KDA_HISTORY;
    const uint64_t recurrent =
        (uint64_t)DS4_GLM5_KDA_HEADS * DS4_GLM5_KDA_HEAD_DIM *
        DS4_GLM5_KDA_HEAD_DIM;
    values.resize((size_t)(3u * history + recurrent));
    return ds4_gpu_tensor_read(state.q_history, 0, values.data(),
                               history * sizeof(float)) &&
           ds4_gpu_tensor_read(state.k_history, 0, values.data() + history,
                               history * sizeof(float)) &&
           ds4_gpu_tensor_read(state.v_history, 0,
                               values.data() + 2u * history,
                               history * sizeof(float)) &&
           ds4_gpu_tensor_read(state.recurrent, 0,
                               values.data() + 3u * history,
                               recurrent * sizeof(float));
}

static bool handoff_case(uint32_t prefill,
                         const ds4_glm5_kda_weight_offsets &weights,
                         const MappedGGUF &gguf) {
    const uint32_t total = prefill + 1u;
    std::vector<float> host_input((size_t)total * 4096u);
    for (uint64_t i = 0; i < host_input.size(); ++i) {
        const int32_t centered = (int32_t)((i * 17u + 23u) % 257u) - 128;
        host_input[i] = (float)centered * (1.0f / 1024.0f);
    }
    ds4_gpu_tensor *input = ds4_gpu_tensor_alloc(
        (uint64_t)host_input.size() * sizeof(float));
    ds4_gpu_tensor *chunk_output = ds4_gpu_tensor_alloc(
        (uint64_t)prefill * 4096u * sizeof(float));
    ds4_gpu_tensor *chunk_last = ds4_gpu_tensor_alloc(4096u * sizeof(float));
    ds4_gpu_tensor *token_output = ds4_gpu_tensor_alloc(
        (uint64_t)total * 4096u * sizeof(float));
    CHECK(input && chunk_output && chunk_last && token_output,
          "allocate handoff tensors");
    CHECK(ds4_gpu_tensor_write(input, 0, host_input.data(),
                               (uint64_t)host_input.size() * sizeof(float)),
          "upload handoff input");

    ds4_glm5_layer_kind schedule = {.layer = 0u, .is_kda = true};
    ds4_glm5_kda_slot chunk = {}, tokenwise = {};
    ds4_glm5_kda_workspace chunk_workspace = {}, token_workspace = {};
    CHECK(ds4_glm5_kda_slot_init(&chunk, &schedule, 1u, 1u, nullptr) &&
          ds4_glm5_kda_slot_init(&tokenwise, &schedule, 1u, 1u, nullptr) &&
          ds4_glm5_kda_workspace_init(&chunk_workspace, prefill) &&
          ds4_glm5_kda_workspace_init(&token_workspace, 1u),
          "allocate handoff states/workspaces");

    ds4_gpu_tensor *prefill_input = ds4_gpu_tensor_view(
        input, 0, (uint64_t)prefill * 4096u * sizeof(float));
    ds4_gpu_tensor *decode_input = ds4_gpu_tensor_view(
        input, (uint64_t)prefill * 4096u * sizeof(float),
        4096u * sizeof(float));
    CHECK(prefill_input && decode_input, "create handoff input views");
    CHECK(ds4_glm5_kda_layer_forward(
              &chunk.layer[0], &chunk_workspace, &weights,
              gguf.map, gguf.size, prefill_input, chunk_output, prefill,
              gguf.rms_norm_eps) &&
          ds4_glm5_kda_layer_forward(
              &chunk.layer[0], &token_workspace, &weights,
              gguf.map, gguf.size, decode_input, chunk_last, 1u,
              gguf.rms_norm_eps),
          "execute prefill plus decode handoff");

    for (uint32_t token = 0; token < total; ++token) {
        ds4_gpu_tensor *one = ds4_gpu_tensor_view(
            input, (uint64_t)token * 4096u * sizeof(float),
            4096u * sizeof(float));
        ds4_gpu_tensor *one_output = ds4_gpu_tensor_view(
            token_output, (uint64_t)token * 4096u * sizeof(float),
            4096u * sizeof(float));
        CHECK(one != nullptr && one_output != nullptr,
              "create tokenwise input/output views");
        const bool ok = ds4_glm5_kda_layer_forward(
            &tokenwise.layer[0], &token_workspace, &weights,
            gguf.map, gguf.size, one, one_output, 1u,
            gguf.rms_norm_eps);
        ds4_gpu_tensor_free(one_output);
        ds4_gpu_tensor_free(one);
        CHECK(ok, "execute tokenwise comparison");
    }
    CHECK(ds4_gpu_synchronize(), "synchronize handoff case");
    std::vector<float> chunk_out((size_t)total * 4096u);
    std::vector<float> token_out((size_t)total * 4096u);
    std::vector<float> chunk_state, token_state;
    CHECK(ds4_gpu_tensor_read(chunk_output, 0, chunk_out.data(),
                              (uint64_t)prefill * 4096u * sizeof(float)) &&
          ds4_gpu_tensor_read(chunk_last, 0,
                              chunk_out.data() + (size_t)prefill * 4096u,
                              4096u * sizeof(float)) &&
          ds4_gpu_tensor_read(token_output, 0, token_out.data(),
                              (uint64_t)total * 4096u * sizeof(float)) &&
          read_state(chunk.layer[0], chunk_state) &&
          read_state(tokenwise.layer[0], token_state),
          "read handoff results");
    const ErrorStats output_error = compare(token_out, chunk_out);
    const ErrorStats state_error = compare(token_state, chunk_state);
    CHECK(chunk.layer[0].token_count == total &&
          tokenwise.layer[0].token_count == total,
          "handoff token counts agree");
    CHECK(output_error.finite && state_error.finite &&
          output_error.max_abs <= 5.0e-4 && nmse(output_error) <= 1.0e-7 &&
          state_error.max_abs <= 5.0e-5 && nmse(state_error) <= 1.0e-7,
          "handoff numerical envelope");
    std::fprintf(stderr,
        "PASS GLM5 KDA handoff prefill=%u output_abs=%.9g output_nmse=%.9g "
        "state_abs=%.9g state_nmse=%.9g\n",
        prefill, output_error.max_abs, nmse(output_error),
        state_error.max_abs, nmse(state_error));

    ds4_gpu_tensor_free(decode_input);
    ds4_gpu_tensor_free(prefill_input);
    ds4_glm5_kda_workspace_free(&token_workspace);
    ds4_glm5_kda_workspace_free(&chunk_workspace);
    ds4_glm5_kda_slot_free(&tokenwise);
    ds4_glm5_kda_slot_free(&chunk);
    ds4_gpu_tensor_free(token_output);
    ds4_gpu_tensor_free(chunk_last);
    ds4_gpu_tensor_free(chunk_output);
    ds4_gpu_tensor_free(input);
    return true;
}

static uint64_t fnv64(const std::vector<float> &values) {
    const uint8_t *bytes = (const uint8_t *)values.data();
    const uint64_t count = (uint64_t)values.size() * sizeof(float);
    uint64_t hash = UINT64_C(1469598103934665603);
    for (uint64_t i = 0; i < count; ++i) {
        hash ^= bytes[i];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static bool failure_invalidation(
        ds4_glm5_kda_slot &slot,
        ds4_glm5_kda_workspace &workspace,
        const ds4_glm5_kda_weight_offsets &weights,
        const MappedGGUF &gguf,
        ds4_gpu_tensor *input,
        ds4_gpu_tensor *output) {
    CHECK(ds4_glm5_kda_slot_reset(&slot),
          "reset before invalid norm-epsilon case");
    CHECK(!ds4_glm5_kda_layer_forward(
              &slot.layer[0], &workspace, &weights,
              gguf.map, gguf.size, input, output, 2u, 0.0f),
          "non-positive model norm epsilon fails closed");
    CHECK(slot.valid && slot.layer[0].valid &&
              slot.layer[0].token_count == 0u,
          "invalid model norm epsilon does not mutate resident state");
    for (uint32_t stage = DS4_GLM5_KDA_FAIL_INPUT_NORM;
         stage <= DS4_GLM5_KDA_FAIL_OUTPUT_PROJECTION; ++stage) {
        CHECK(ds4_glm5_kda_slot_reset(&slot), "reset before injected failure");
        ds4_glm5_kda_test_fail_after(stage);
        CHECK(!ds4_glm5_kda_layer_forward(
                  &slot.layer[0], &workspace, &weights,
                  gguf.map, gguf.size, input, output, 2u,
                  gguf.rms_norm_eps),
              "injected stage fails forward");
        CHECK(!slot.valid && !slot.layer[0].valid &&
              slot.layer[0].token_count == 0u,
              "failed forward invalidates slot without advancing tokens");
        ds4_glm5_kda_test_fail_after(DS4_GLM5_KDA_FAIL_NONE);
        CHECK(!ds4_glm5_kda_layer_forward(
                  &slot.layer[0], &workspace, &weights,
                  gguf.map, gguf.size, input, output, 2u,
                  gguf.rms_norm_eps),
              "invalid state refuses continuation");
    }
    ds4_glm5_kda_test_fail_after(DS4_GLM5_KDA_FAIL_NONE);
    CHECK(ds4_glm5_kda_slot_reset(&slot), "reset after injected failures");
    return true;
}

static bool full_model_residency(const MappedGGUF &gguf) {
    constexpr uint32_t layers = 46u;
    ds4_glm5_layer_kind schedule[layers] = {};
    uint32_t kda_count = 0;
    for (uint32_t layer = 0; layer < layers; ++layer) {
        char kda_name[64], mla_name[64];
        std::snprintf(kda_name, sizeof(kda_name),
                      "blk.%u.kda_q.weight", layer);
        std::snprintf(mla_name, sizeof(mla_name),
                      "blk.%u.attn_q_a.weight", layer);
        const bool has_kda = gguf.tensors.count(kda_name) == 1u;
        const bool has_mla = gguf.tensors.count(mla_name) == 1u;
        CHECK(has_kda != has_mla,
              "same-GGUF layer has exactly one attention family");
        schedule[layer] = {.layer = layer, .is_kda = has_kda};
        if (has_kda) ++kda_count;
    }
    CHECK(kda_count == 34u, "same-GGUF schedule has 34 KDA layers");

    ds4_glm5_kda_slot slot = {};
    ds4_glm5_kda_workspace workspace = {};
    CHECK(ds4_glm5_kda_slot_init(&slot, schedule, layers, 1u, stderr) &&
          ds4_glm5_kda_workspace_init(&workspace, 2048u),
          "allocate full same-GGUF KDA state and 2048-token workspace");
    CHECK(slot.bytes == UINT64_C(152633344) &&
          workspace.bytes == UINT64_C(370671616),
          "full KDA residency byte accounting is exact");
    std::fprintf(stderr,
        "PASS GLM5 KDA full residency state_bytes=%llu workspace_bytes=%llu "
        "total_bytes=%llu\n",
        (unsigned long long)slot.bytes,
        (unsigned long long)workspace.bytes,
        (unsigned long long)(slot.bytes + workspace.bytes));
    ds4_glm5_kda_workspace_free(&workspace);
    ds4_glm5_kda_slot_free(&slot);
    return true;
}

static bool run_test(void) {
    ds4_tp_test_reset_exchange_calls();
    const char *model = std::getenv("DS4_GLM5_MODEL");
    const char *prefix = std::getenv("DS4_GLM5_KDA_ORACLE_PREFIX");
    CHECK(model && model[0] && prefix && prefix[0],
          "model and oracle prefix environment");
    MappedGGUF gguf;
    CHECK(open_gguf(model, gguf), "load GLM5 GGUF directory");
    ds4_glm5_kda_weight_offsets weights = {};
    CHECK(bind_layer0(gguf, weights), "bind real layer-0 offsets");

    uint32_t tokens = 2u;
    if (const char *tokens_env = std::getenv(
            "DS4_GLM5_KDA_ORACLE_TOKENS")) {
        char *tokens_end = nullptr;
        const unsigned long parsed = std::strtoul(
            tokens_env, &tokens_end, 10);
        CHECK(tokens_end && *tokens_end == '\0' && parsed >= 2u &&
                  parsed <= 64u,
              "oracle token count is 2..64");
        tokens = (uint32_t)parsed;
    }
    std::vector<float> input_ref, output_ref, history_ref, state_ref;
    CHECK(read_f32_file(std::string(prefix) + ".input.f32",
                        (uint64_t)tokens * 4096u, input_ref) &&
          read_f32_file(std::string(prefix) + ".output.f32",
                        (uint64_t)tokens * 4096u, output_ref) &&
          read_f32_file(std::string(prefix) + ".history.f32",
                        (uint64_t)3u * DS4_GLM5_KDA_CHANNELS *
                            DS4_GLM5_KDA_HISTORY, history_ref) &&
          read_f32_file(std::string(prefix) + ".state.f32",
                        (uint64_t)DS4_GLM5_KDA_HEADS *
                            DS4_GLM5_KDA_HEAD_DIM * DS4_GLM5_KDA_HEAD_DIM,
                        state_ref),
          "load same-GGUF oracle dumps");

    ds4_gpu_config config = {};
    config.n_gpus = 1;
    config.device_indices[0] = 0;
    CHECK(ds4_gpu_init_multi(&config), "initialize ROCm");
    CHECK(ds4_gpu_set_model_fd_for_map(gguf.fd, gguf.map) &&
          ds4_gpu_set_model_map(gguf.map, gguf.size),
          "register GLM5 GGUF map");
    CHECK(full_model_residency(gguf),
          "full same-GGUF KDA residency allocation");

    ds4_glm5_layer_kind schedule = {.layer = 0u, .is_kda = true};
    ds4_glm5_kda_slot slot = {};
    ds4_glm5_kda_workspace workspace = {};
    CHECK(ds4_glm5_kda_slot_init(&slot, &schedule, 1u, 1u, stderr),
          "allocate resident layer state");
    CHECK(ds4_glm5_kda_workspace_init(&workspace, tokens),
          "allocate two-token workspace");
    ds4_gpu_tensor *input = ds4_gpu_tensor_alloc(
        (uint64_t)input_ref.size() * sizeof(float));
    ds4_gpu_tensor *output = ds4_gpu_tensor_alloc(
        (uint64_t)output_ref.size() * sizeof(float));
    CHECK(input && output &&
          ds4_gpu_tensor_write(input, 0, input_ref.data(),
                               (uint64_t)input_ref.size() * sizeof(float)),
          "upload deterministic layer input");
    CHECK(ds4_glm5_kda_layer_forward(
              &slot.layer[0], &workspace, &weights,
              gguf.map, gguf.size, input, output, tokens,
              gguf.rms_norm_eps),
          "execute complete layer-0 KDA adapter");
    CHECK(ds4_gpu_synchronize(), "synchronize complete KDA layer");
    CHECK(slot.layer[0].valid && slot.layer[0].token_count == tokens,
          "successful layer advances token count");

    std::vector<float> output_got(output_ref.size());
    std::vector<float> history_got(history_ref.size());
    std::vector<float> state_got(state_ref.size());
    const uint64_t one_history =
        (uint64_t)DS4_GLM5_KDA_CHANNELS * DS4_GLM5_KDA_HISTORY;
    CHECK(ds4_gpu_tensor_read(output, 0, output_got.data(),
                              output_got.size() * sizeof(float)) &&
          ds4_gpu_tensor_read(slot.layer[0].q_history, 0,
                              history_got.data(), one_history * sizeof(float)) &&
          ds4_gpu_tensor_read(slot.layer[0].k_history, 0,
                              history_got.data() + one_history,
                              one_history * sizeof(float)) &&
          ds4_gpu_tensor_read(slot.layer[0].v_history, 0,
                              history_got.data() + 2u * one_history,
                              one_history * sizeof(float)) &&
          ds4_gpu_tensor_read(slot.layer[0].recurrent, 0,
                              state_got.data(), state_got.size() * sizeof(float)),
          "read complete layer outputs and state");
    const ErrorStats out_error = compare(output_ref, output_got);
    const ErrorStats history_error = compare(history_ref, history_got);
    const ErrorStats state_error = compare(state_ref, state_got);
    std::fprintf(stderr,
        "GLM5 KDA layer0 output_abs=%.9g output_rel=%.9g output_nmse=%.9g "
        "history_abs=%.9g history_rel=%.9g state_abs=%.9g state_rel=%.9g "
        "output_fnv=%016llx history_fnv=%016llx state_fnv=%016llx\n",
        out_error.max_abs, out_error.max_rel, nmse(out_error),
        history_error.max_abs, history_error.max_rel,
        state_error.max_abs, state_error.max_rel,
        (unsigned long long)fnv64(output_got),
        (unsigned long long)fnv64(history_got),
        (unsigned long long)fnv64(state_got));
    CHECK(out_error.finite && history_error.finite && state_error.finite,
          "complete layer values are finite");
    CHECK(out_error.max_abs <= 5.0e-4 && nmse(out_error) <= 1.0e-7,
          "complete layer output numerical envelope");
    CHECK(history_error.max_abs <= 5.0e-4 &&
          state_error.max_abs <= 5.0e-5,
          "complete layer resident-state numerical envelope");

    /* Deliberately replay the historical wrong constant.  This is a
     * discrimination control: the metadata-derived oracle must reject it by
     * a wide margin, otherwise the test could copy the implementation bug. */
    ds4_glm5_kda_slot wrong_eps = {};
    ds4_glm5_kda_workspace wrong_eps_workspace = {};
    ds4_gpu_tensor *wrong_eps_output = ds4_gpu_tensor_alloc(
        (uint64_t)output_ref.size() * sizeof(float));
    CHECK(wrong_eps_output &&
          ds4_glm5_kda_slot_init(&wrong_eps, &schedule, 1u, 1u, stderr) &&
          ds4_glm5_kda_workspace_init(&wrong_eps_workspace, tokens) &&
          ds4_glm5_kda_layer_forward(
              &wrong_eps.layer[0], &wrong_eps_workspace, &weights,
              gguf.map, gguf.size, input, wrong_eps_output, tokens, 1.0e-6f) &&
          ds4_gpu_synchronize(),
          "execute historical wrong-epsilon negative control");
    std::vector<float> wrong_eps_got(output_ref.size());
    CHECK(ds4_gpu_tensor_read(
              wrong_eps_output, 0, wrong_eps_got.data(),
              (uint64_t)wrong_eps_got.size() * sizeof(float)),
          "read wrong-epsilon negative control");
    const ErrorStats wrong_eps_error = compare(output_ref, wrong_eps_got);
    std::fprintf(stderr,
        "GLM5 KDA wrong-eps rejection output_abs=%.9g output_nmse=%.9g\n",
        wrong_eps_error.max_abs, nmse(wrong_eps_error));
    CHECK(wrong_eps_error.finite && wrong_eps_error.max_abs > 1.0e-3 &&
              nmse(wrong_eps_error) > 1.0e-2,
          "metadata-derived oracle rejects historical 1e-6 epsilon");
    ds4_gpu_tensor_free(wrong_eps_output);
    ds4_glm5_kda_workspace_free(&wrong_eps_workspace);
    ds4_glm5_kda_slot_free(&wrong_eps);

    ds4_glm5_kda_slot peer = {};
    ds4_glm5_kda_workspace peer_workspace = {};
    ds4_gpu_tensor *peer_output = ds4_gpu_tensor_alloc(
        (uint64_t)output_ref.size() * sizeof(float));
    CHECK(peer_output &&
          ds4_glm5_kda_slot_init(&peer, &schedule, 1u, 1u, stderr) &&
          ds4_glm5_kda_workspace_init(&peer_workspace, tokens) &&
          ds4_glm5_kda_layer_forward(
              &peer.layer[0], &peer_workspace, &weights,
              gguf.map, gguf.size, input, peer_output, tokens,
              gguf.rms_norm_eps) &&
          ds4_gpu_synchronize(),
          "execute independent replicated rank");
    ds4_glm5_kda_digest local_digest = {}, peer_digest = {};
    CHECK(ds4_glm5_kda_layer_digest(
              &slot.layer[0], output, output_ref.size(), &local_digest) &&
          ds4_glm5_kda_layer_digest(
              &peer.layer[0], peer_output, output_ref.size(), &peer_digest) &&
          ds4_glm5_kda_digest_equal(&local_digest, &peer_digest),
          "independent ranks produce identical state/output digests");
    ds4_glm5_kda_workspace_free(&peer_workspace);
    ds4_glm5_kda_slot_free(&peer);
    ds4_gpu_tensor_free(peer_output);

    for (uint32_t length : {1u, 2u, 3u, 127u, 128u, 129u, 2048u})
        CHECK(handoff_case(length, weights, gguf),
              "prefill/decode handoff case");
    CHECK(ds4_tp_test_get_exchange_calls() == 0u,
          "KDA execution invokes no TP exchange API");
    CHECK(failure_invalidation(slot, workspace, weights, gguf, input, output),
          "failure invalidation contract");

    ds4_gpu_tensor_free(output);
    ds4_gpu_tensor_free(input);
    ds4_glm5_kda_workspace_free(&workspace);
    ds4_glm5_kda_slot_free(&slot);
    ds4_gpu_cleanup();
    gguf.close_all();
    std::fprintf(stderr, "PASS complete real-GGUF GLM5 KDA layer\n");
    return true;
}

int main(void) {
    return run_test() ? 0 : 1;
}
