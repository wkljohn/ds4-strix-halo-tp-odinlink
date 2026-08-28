#include "ds4_glm5_next_exec.h"
#include "ds4_gpu.h"
#include "ds4_gpu_mgpu.h"
extern "C" {
#include "ds4_tp.h"
}
#include "tests/glm5_gguf_test.hpp"
#include "tests/glm5_next_real_offsets.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
constexpr uint64_t kPrefixFNV = UINT64_C(0xb6c2590232ac924b);

uint64_t fnv64(const void *data, uint64_t bytes) {
    const auto *p = static_cast<const uint8_t *>(data);
    uint64_t hash = UINT64_C(1469598103934665603);
    for (uint64_t i = 0u; i < bytes; ++i) {
        hash ^= p[i];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
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
               TpGuard &guard, ds4_glm5_next_exec_ctx &exec,
               uint64_t &sequence) {
    CHECK(unsetenv("DS4_TP_VERBS_LIB") == 0 &&
          setenv("DS4_TP_BIG_DIRECT", "1", 1) == 0 &&
          setenv("DS4_TP_BIG_DIRECT_MAX_ROWS", "1", 1) == 0,
          "select mandatory system-verbs direct RoCE");
    ds4_tp_options options = {};
    options.role = leader ? DS4_TP_LEADER : DS4_TP_WORKER;
    options.requested = true;
    options.listen_host = leader ? host : nullptr;
    options.listen_port = leader ? port : 0;
    options.leader_host = leader ? nullptr : host;
    options.leader_port = leader ? 0 : port;
    options.transport = DS4_TP_TRANSPORT_RDMA;
    options.rdma_device = device;
    options.rdma_gid_index = 3;
    options.rdma_gid_index_set = true;

    ds4_tp_identity identity = {};
    identity.gguf_bytes = gguf.size;
    identity.model_id = 3u;
    identity.n_layer = 46u;
    identity.n_embd = kWidth;
    identity.n_vocab = 154880u;
    identity.quant_bits = 4u;
    identity.ctx_size = 2u;
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
    guard.big_out = ds4_gpu_tensor_view(
        guard.slab, ds4_tp_slab_big_out_offset(guard.tp), row_bytes);
    guard.big_in = ds4_gpu_tensor_view(
        guard.slab, ds4_tp_slab_big_in_offset(guard.tp), row_bytes);
    CHECK(guard.big_out && guard.big_in &&
          ds4_tp_big_gate_is_direct(
              guard.tp, ds4_gpu_tensor_contents(guard.big_out),
              ds4_gpu_tensor_contents(guard.big_in), row_bytes),
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
    const char *full_tokens_env = std::getenv("DS4_GLM5_FULL_TOKENS");
    const uint32_t full_tokens = !full_tokens_env ? 1u :
        std::strcmp(full_tokens_env, "1") == 0 ? 1u :
        std::strcmp(full_tokens_env, "2") == 0 ? 2u : 0u;
    CHECK(full_tokens != 0u &&
              (full_trunk || full_tokens == 1u),
          "full-token count is bounded and requires full-trunk mode");
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
    TensorGuard current, output, logits;
    CHECK(ds4_glm5_next_state_init(&state.value, &offsets, 2u, nullptr) &&
          (workspace.value = ds4_glm5_next_workspace_create()) != nullptr &&
          (current.value = ds4_gpu_tensor_alloc(
              (uint64_t)kHcWidth * sizeof(float))) != nullptr &&
          (output.value = ds4_gpu_tensor_alloc(
              (uint64_t)kHcWidth * sizeof(float))) != nullptr &&
          (!full_trunk ||
           (logits.value = ds4_gpu_tensor_alloc(
               UINT64_C(154880) * sizeof(float))) != nullptr),
          "allocate resident state and production workspace");

    ds4_glm5_next_exec_ctx exec = {};
    exec.model_map = gguf.map;
    exec.model_size = gguf.size;
    exec.model = &offsets;
    uint64_t sequence = 0u;
    TpGuard tp;
    CHECK(create_tp(gguf, leader, host, device, (int)port_long,
                    tp, exec, sequence),
          "create persistent GLM5 layer3 RoCE transport");
    char rank_error[256] = {};
    CHECK(ds4_tp_hash_check(
              tp.tp, UINT64_C(0x474c4d3552414e4b),
              UINT64_C(0x52414e4b00000000) ^ exec.tp_rank,
              rank_error, sizeof(rank_error)) == -1,
          "TP roles must prove distinct rank identities");
    char mode_error[256] = {};
    const uint64_t mode_hash = UINT64_C(0x474c4d354d4f4400) ^
        ((uint64_t)full_trunk << 8u) ^ full_tokens;
    CHECK(ds4_tp_hash_check(tp.tp, UINT64_C(0x474c4d354d4f4445),
                            mode_hash, mode_error, sizeof(mode_error)) == 1,
          "TP roles must negotiate the same full-trunk token count");
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
                              prefix.size() * sizeof(float)) &&
          fnv64(prefix.data(), prefix.size() * sizeof(float)) == kPrefixFNV,
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
