#include "ds4_tp.h"
#include "ds4_glm5_next_runtime.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int check(const char *name, uint32_t local, uint32_t peer, int want,
                 const char *want_error) {
    char err[160] = "";
    const int got = ds4_tp_test_hello_validate_runtime_features(
            local, peer, err, sizeof(err));
    if (got != want) {
        fprintf(stderr,
                "FAIL %s: local=0x%08x peer=0x%08x got=%d want=%d\n",
                name, local, peer, got, want);
        return 0;
    }
    if (want_error && strcmp(err, want_error) != 0) {
        fprintf(stderr, "FAIL %s: error='%s' want='%s'\n",
                name, err, want_error);
        return 0;
    }
    fprintf(stderr, "PASS %s\n", name);
    return 1;
}

static int check_transport(const char *name,
                           ds4_tp_transport requested,
                           int local_rdma_ok,
                           int peer_rdma_ok,
                           int want_ok,
                           int want_active,
                           const char *want_error) {
    char err[160] = "";
    int active = -1;
    const int got = ds4_tp_test_select_transport(
            requested, local_rdma_ok, peer_rdma_ok, &active,
            err, sizeof(err));
    if (got != want_ok || active != want_active) {
        fprintf(stderr,
                "FAIL %s: got ok=%d active=%d want ok=%d active=%d\n",
                name, got, active, want_ok, want_active);
        return 0;
    }
    if (want_error && strcmp(err, want_error) != 0) {
        fprintf(stderr, "FAIL %s: error='%s' want='%s'\n",
                name, err, want_error);
        return 0;
    }
    if (!want_error && err[0] != '\0') {
        fprintf(stderr, "FAIL %s: unexpected error='%s'\n", name, err);
        return 0;
    }
    fprintf(stderr, "PASS %s\n", name);
    return 1;
}

static int check_connect_timeout(const char *name, const char *value,
                                 uint64_t want) {
    if (value) {
        if (setenv("DS4_TP_CONNECT_TIMEOUT_SEC", value, 1) != 0) {
            perror("setenv DS4_TP_CONNECT_TIMEOUT_SEC");
            return 0;
        }
    } else if (unsetenv("DS4_TP_CONNECT_TIMEOUT_SEC") != 0) {
        perror("unsetenv DS4_TP_CONNECT_TIMEOUT_SEC");
        return 0;
    }
    const uint64_t got = ds4_tp_test_connect_timeout_sec();
    if (got != want) {
        fprintf(stderr, "FAIL %s: got=%llu want=%llu\n", name,
                (unsigned long long)got, (unsigned long long)want);
        return 0;
    }
    fprintf(stderr, "PASS %s\n", name);
    return 1;
}

static void build_glm53_mask(uint64_t mask[DS4_TP_GATE_MASK_WORDS],
                             uint32_t *count,
                             uint32_t features) {
    if (!ds4_glm5_next_build_tp_gate_mask(mask, count, features)) *count = 0;
}

static int check_glm53_gate_schedule(void) {
    uint64_t mask[DS4_TP_GATE_MASK_WORDS];
    uint32_t count = 0;
    build_glm53_mask(mask, &count, 0u);
    char err[160] = "";
    if (count != 53u ||
        !ds4_tp_test_gate_schedule_validate(mask, count, 92u,
                                            err, sizeof(err))) {
        fprintf(stderr, "FAIL GLM53 gate mask: count=%u error='%s'\n",
                count, err);
        return 0;
    }
    static const uint32_t want[] = {6u, 7u, 9u, 11u, 13u, 14u, 15u};
    for (uint32_t i = 0; i < sizeof(want) / sizeof(want[0]); ++i) {
        const uint32_t got = ds4_tp_test_gate_slot(
            mask, 6u, 1u, count, 92u, (uint64_t)i + 1u);
        if (got != want[i]) {
            fprintf(stderr,
                    "FAIL GLM53 gate order seq=%u got=%u want=%u\n",
                    i + 1u, got, want[i]);
            return 0;
        }
    }
    const uint32_t wrap = ds4_tp_test_gate_slot(
        mask, 6u, 1u, count, 92u, (uint64_t)count + 1u);
    if (wrap != 6u) {
        fprintf(stderr, "FAIL GLM53 gate wrap got=%u want=6\n", wrap);
        return 0;
    }
    build_glm53_mask(mask, &count, DS4_TP_FEATURE_GLM5_KDA_TP);
    err[0] = '\0';
    if (count != 87u ||
        !ds4_tp_test_gate_schedule_validate(mask, count, 92u,
                                            err, sizeof(err))) {
        fprintf(stderr, "FAIL GLM53 KDA-TP gate mask: count=%u error='%s'\n",
                count, err);
        return 0;
    }
    static const uint32_t kda_want[] = {0u, 2u, 4u, 6u, 7u, 8u, 9u};
    for (uint32_t i = 0; i < sizeof(kda_want) / sizeof(kda_want[0]); ++i) {
        const uint32_t got = ds4_tp_test_gate_slot(
            mask, 0u, 1u, count, 92u, (uint64_t)i + 1u);
        if (got != kda_want[i]) {
            fprintf(stderr,
                    "FAIL GLM53 KDA-TP order seq=%u got=%u want=%u\n",
                    i + 1u, got, kda_want[i]);
            return 0;
        }
    }
    const uint32_t kda_wrap = ds4_tp_test_gate_slot(
        mask, 0u, 1u, count, 92u, (uint64_t)count + 1u);
    if (kda_wrap != 0u) {
        fprintf(stderr, "FAIL GLM53 KDA-TP wrap got=%u want=0\n", kda_wrap);
        return 0;
    }
    fprintf(stderr, "PASS GLM53 53-gate hybrid schedule\n");
    fprintf(stderr, "PASS GLM53 87-gate KDA-TP schedule\n");
    return 1;
}

static int check_invalid_gate_schedules(void) {
    uint64_t mask[DS4_TP_GATE_MASK_WORDS];
    uint32_t count = 0;
    build_glm53_mask(mask, &count, 0u);
    char err[160] = "";
    int ok = !ds4_tp_test_gate_schedule_validate(mask, count - 1u, 92u,
                                                  err, sizeof(err));
    if (!ok || strstr(err, "53 bits, 52 gates, 92 slots") == NULL) {
        fprintf(stderr, "FAIL gate mask count refusal: '%s'\n", err);
        return 0;
    }
    if (ds4_tp_test_gate_slot(mask, 6u, 1u, 0u, 92u, 1u) != 92u) {
        fprintf(stderr, "FAIL zero-count masked gate did not fail closed\n");
        return 0;
    }
    mask[1] |= UINT64_C(1) << 40u; /* slot 104, beyond the 92-slot slab */
    err[0] = '\0';
    ok = !ds4_tp_test_gate_schedule_validate(mask, count + 1u, 92u,
                                              err, sizeof(err));
    if (!ok || strstr(err, "54 bits, 54 gates, 92 slots") == NULL) {
        fprintf(stderr, "FAIL out-of-range gate mask refusal: '%s'\n", err);
        return 0;
    }
    fprintf(stderr, "PASS malformed gate schedules fail closed\n");
    return 1;
}

int main(void) {
    int ok = 1;
    ok &= check_glm53_gate_schedule();
    ok &= check_invalid_gate_schedules();
    ok &= check("hello equal-enabled",
                DS4_TP_FEATURE_Q4K_WMMA, DS4_TP_FEATURE_Q4K_WMMA, 1, NULL);
    ok &= check("hello equal-disabled", 0, 0, 1, NULL);
    ok &= check("hello mismatched-feature-masks",
                DS4_TP_FEATURE_Q4K_WMMA, 0, 0,
                "tp hello: runtime feature mismatch (local=0x00000001 peer=0x00000000)");
    ok &= check("hello equal-iq2-i8-wmma",
                DS4_TP_FEATURE_IQ2_I8_WMMA,
                DS4_TP_FEATURE_IQ2_I8_WMMA, 1, NULL);
    ok &= check("hello mismatched-iq2-i8-wmma",
                DS4_TP_FEATURE_IQ2_I8_WMMA, 0, 0,
                "tp hello: runtime feature mismatch (local=0x00000040 peer=0x00000000)");
    ok &= check("hello equal-temporal-compressor",
                DS4_TP_FEATURE_TEMPORAL_COMPRESSOR,
                DS4_TP_FEATURE_TEMPORAL_COMPRESSOR, 1, NULL);
    ok &= check("hello mismatched-temporal-compressor",
                DS4_TP_FEATURE_TEMPORAL_COMPRESSOR, 0, 0,
                "tp hello: runtime feature mismatch (local=0x00000080 peer=0x00000000)");
    ok &= check("hello equal-q4k-fused-mid",
                DS4_TP_FEATURE_Q4K_FUSED_MID,
                DS4_TP_FEATURE_Q4K_FUSED_MID, 1, NULL);
    ok &= check("hello mismatched-q4k-fused-mid",
                DS4_TP_FEATURE_Q4K_FUSED_MID, 0, 0,
                "tp hello: runtime feature mismatch (local=0x00010000 peer=0x00000000)");
    ok &= check("hello equal-hc-stage-exact-coop",
                DS4_TP_FEATURE_HC_STAGE_EXACT_COOP,
                DS4_TP_FEATURE_HC_STAGE_EXACT_COOP, 1, NULL);
    ok &= check("hello mismatched-hc-stage-exact-coop",
                DS4_TP_FEATURE_HC_STAGE_EXACT_COOP, 0, 0,
                "tp hello: runtime feature mismatch (local=0x00020000 peer=0x00000000)");
    ok &= check("hello equal-indexer-topk-radix-tree",
                DS4_TP_FEATURE_INDEXER_TOPK_RADIX_TREE,
                DS4_TP_FEATURE_INDEXER_TOPK_RADIX_TREE, 1, NULL);
    ok &= check("hello mismatched-indexer-topk-radix-tree",
                DS4_TP_FEATURE_INDEXER_TOPK_RADIX_TREE, 0, 0,
                "tp hello: runtime feature mismatch (local=0x00040000 peer=0x00000000)");
    ok &= check("hello equal-q4k-kshard",
                DS4_TP_FEATURE_Q4K_KSHARD,
                DS4_TP_FEATURE_Q4K_KSHARD, 1, NULL);
    ok &= check("hello mismatched-q4k-kshard",
                DS4_TP_FEATURE_Q4K_KSHARD, 0, 0,
                "tp hello: runtime feature mismatch (local=0x00080000 peer=0x00000000)");
    ok &= check("hello equal-glm5-resident-kda",
                DS4_TP_FEATURE_GLM5_RESIDENT_KDA,
                DS4_TP_FEATURE_GLM5_RESIDENT_KDA, 1, NULL);
    ok &= check("hello mismatched-glm5-resident-kda",
                DS4_TP_FEATURE_GLM5_RESIDENT_KDA, 0, 0,
                "tp hello: runtime feature mismatch (local=0x00100000 peer=0x00000000)");
    ok &= check("hello equal-glm5-small-gate",
                DS4_TP_FEATURE_GLM5_SMALL_GATE,
                DS4_TP_FEATURE_GLM5_SMALL_GATE, 1, NULL);
    ok &= check("hello mismatched-glm5-small-gate",
                DS4_TP_FEATURE_GLM5_SMALL_GATE, 0, 0,
                "tp hello: runtime feature mismatch (local=0x00200000 peer=0x00000000)");
    ok &= check("hello equal-glm5-kda-tp",
                DS4_TP_FEATURE_GLM5_KDA_TP,
                DS4_TP_FEATURE_GLM5_KDA_TP, 1, NULL);
    ok &= check("hello mismatched-glm5-kda-tp",
                DS4_TP_FEATURE_GLM5_KDA_TP, 0, 0,
                "tp hello: runtime feature mismatch (local=0x00400000 peer=0x00000000)");
    ok &= check("hello equal-glm5-kda-output-kslice",
                DS4_TP_FEATURE_GLM5_KDA_OUTPUT_KSLICE,
                DS4_TP_FEATURE_GLM5_KDA_OUTPUT_KSLICE, 1, NULL);
    ok &= check("hello mismatched-glm5-kda-output-kslice",
                DS4_TP_FEATURE_GLM5_KDA_OUTPUT_KSLICE, 0, 0,
                "tp hello: runtime feature mismatch (local=0x00800000 peer=0x00000000)");
    ok &= check("hello q4k-wmma-kshard mismatch",
                DS4_TP_FEATURE_Q4K_WMMA | DS4_TP_FEATURE_Q4K_KSHARD,
                DS4_TP_FEATURE_Q4K_WMMA, 0,
                "tp hello: runtime feature mismatch (local=0x00080001 peer=0x00000001)");
    ok &= check("hello equal-batch-attn-head-split",
                DS4_TP_FEATURE_BATCH_ATTN_HEAD_SPLIT,
                DS4_TP_FEATURE_BATCH_ATTN_HEAD_SPLIT, 1, NULL);
    ok &= check("hello mismatched-batch-attn-head-split",
                DS4_TP_FEATURE_BATCH_ATTN_HEAD_SPLIT, 0, 0,
                "tp hello: runtime feature mismatch (local=0x00000002 peer=0x00000000)");
    ok &= check("hello equal-odinlink-batch-async",
                DS4_TP_FEATURE_ODINLINK_BATCH_ASYNC,
                DS4_TP_FEATURE_ODINLINK_BATCH_ASYNC, 1, NULL);
    ok &= check("hello mismatched-odinlink-batch-async",
                DS4_TP_FEATURE_ODINLINK_BATCH_ASYNC, 0, 0,
                "tp hello: runtime feature mismatch (local=0x00000004 peer=0x00000000)");
    const uint32_t split_118 = ds4_tp_feature_expert_split(118u);
    const uint32_t split_128 = ds4_tp_feature_expert_split(128u);
    ok &= check("hello equal-asymmetric-expert-split",
                split_118 | DS4_TP_FEATURE_Q4K_WMMA,
                split_118 | DS4_TP_FEATURE_Q4K_WMMA, 1, NULL);
    ok &= check("hello mismatched-expert-split",
                split_118 | DS4_TP_FEATURE_Q4K_WMMA,
                split_128 | DS4_TP_FEATURE_Q4K_WMMA, 0,
                "tp hello: runtime feature mismatch (local=0x00007601 peer=0x00008001)");
    if (ds4_tp_feature_expert_split_value(split_118) != 118u) {
        fprintf(stderr, "FAIL expert split feature round trip\n");
        ok = 0;
    } else {
        fprintf(stderr, "PASS expert split feature round trip\n");
    }
    const uint32_t prior_features =
        DS4_TP_FEATURE_Q4K_WMMA |
        DS4_TP_FEATURE_BATCH_ATTN_HEAD_SPLIT |
        DS4_TP_FEATURE_ODINLINK_BATCH_ASYNC |
        DS4_TP_FEATURE_RDMA_LOGITS |
        DS4_TP_FEATURE_RANK0_FULL_LOGITS |
        DS4_TP_FEATURE_GREEDY_TOP2 |
        DS4_TP_FEATURE_IQ2_I8_WMMA |
        DS4_TP_FEATURE_TEMPORAL_COMPRESSOR |
        DS4_TP_FEATURE_EXPERT_SPLIT_MASK |
        DS4_TP_FEATURE_Q4K_FUSED_MID |
        DS4_TP_FEATURE_HC_STAGE_EXACT_COOP |
        DS4_TP_FEATURE_INDEXER_TOPK_RADIX_TREE |
        DS4_TP_FEATURE_Q4K_KSHARD |
        DS4_TP_FEATURE_GLM5_RESIDENT_KDA |
        DS4_TP_FEATURE_GLM5_SMALL_GATE |
        DS4_TP_FEATURE_GLM5_KDA_TP |
        DS4_TP_FEATURE_GLM5_GPU_ROW_GATE;
    if ((DS4_TP_FEATURE_GLM5_KDA_OUTPUT_KSLICE & prior_features) != 0u) {
        fprintf(stderr,
                "FAIL GLM5 KDA output K-slice feature overlaps prior bits\n");
        ok = 0;
    } else {
        fprintf(stderr,
                "PASS GLM5 KDA output K-slice feature is disjoint\n");
    }
    if (ds4_tp_glm5_resident_kda_feature(1, 1, 1, 1) !=
            DS4_TP_FEATURE_GLM5_RESIDENT_KDA ||
        ds4_tp_glm5_resident_kda_feature(0, 1, 1, 1) != 0u ||
        ds4_tp_glm5_resident_kda_feature(1, 0, 1, 1) != 0u ||
        ds4_tp_glm5_resident_kda_feature(1, 1, 0, 1) != 0u ||
        ds4_tp_glm5_resident_kda_feature(1, 1, 1, 0) != 0u) {
        fprintf(stderr, "FAIL GLM5 resident KDA advertisement predicate\n");
        ok = 0;
    } else {
        fprintf(stderr, "PASS GLM5 resident KDA advertisement predicate\n");
    }
    if (ds4_tp_glm5_kda_tp_feature("1", 1, 1, 0, 1) !=
            DS4_TP_FEATURE_GLM5_KDA_TP ||
        ds4_tp_glm5_kda_tp_feature(NULL, 1, 1, 0, 1) != 0u ||
        ds4_tp_glm5_kda_tp_feature("0", 1, 1, 0, 1) != 0u ||
        ds4_tp_glm5_kda_tp_feature("true", 1, 1, 0, 1) != 0u ||
        ds4_tp_glm5_kda_tp_feature("1", 0, 1, 0, 1) != 0u ||
        ds4_tp_glm5_kda_tp_feature("1", 1, 0, 0, 1) != 0u ||
        ds4_tp_glm5_kda_tp_feature("1", 1, 1, 1, 1) != 0u ||
        ds4_tp_glm5_kda_tp_feature("1", 1, 1, 0, 0) != 0u) {
        fprintf(stderr, "FAIL GLM5 KDA-TP advertisement predicate\n");
        ok = 0;
    } else {
        fprintf(stderr, "PASS GLM5 KDA-TP advertisement predicate\n");
    }
    if (ds4_tp_glm5_small_gate_feature("1", 1, 1, 0) !=
            DS4_TP_FEATURE_GLM5_SMALL_GATE ||
        ds4_tp_glm5_small_gate_feature(NULL, 1, 1, 0) != 0u ||
        ds4_tp_glm5_small_gate_feature("0", 1, 1, 0) != 0u ||
        ds4_tp_glm5_small_gate_feature("true", 1, 1, 0) != 0u ||
        ds4_tp_glm5_small_gate_feature("1", 0, 1, 0) != 0u ||
        ds4_tp_glm5_small_gate_feature("1", 1, 0, 0) != 0u ||
        ds4_tp_glm5_small_gate_feature("1", 1, 1, 1) != 0u) {
        fprintf(stderr, "FAIL GLM5 small-gate advertisement predicate\n");
        ok = 0;
    } else {
        fprintf(stderr, "PASS GLM5 small-gate advertisement predicate\n");
    }
    if (ds4_tp_glm5_gpu_row_gate_feature(
            "1", DS4_TP_FEATURE_GLM5_SMALL_GATE, NULL) !=
            DS4_TP_FEATURE_GLM5_GPU_ROW_GATE ||
        ds4_tp_glm5_gpu_row_gate_feature(NULL,
                                         DS4_TP_FEATURE_GLM5_SMALL_GATE, NULL) != 0u ||
        ds4_tp_glm5_gpu_row_gate_feature("0",
                                         DS4_TP_FEATURE_GLM5_SMALL_GATE, NULL) != 0u ||
        ds4_tp_glm5_gpu_row_gate_feature("1", 0u, NULL) != 0u ||
        ds4_tp_glm5_gpu_row_gate_feature(
            "1", DS4_TP_FEATURE_GLM5_SMALL_GATE, "1") != 0u) {
        fprintf(stderr, "FAIL GLM5 GPU row-gate advertisement predicate\n");
        ok = 0;
    } else {
        fprintf(stderr, "PASS GLM5 GPU row-gate advertisement predicate\n");
    }
    if (ds4_tp_glm5_kda_output_kslice_feature(
            "1", DS4_TP_FEATURE_GLM5_KDA_TP, 1) !=
            DS4_TP_FEATURE_GLM5_KDA_OUTPUT_KSLICE ||
        ds4_tp_glm5_kda_output_kslice_feature(
            NULL, DS4_TP_FEATURE_GLM5_KDA_TP, 1) != 0u ||
        ds4_tp_glm5_kda_output_kslice_feature(
            "0", DS4_TP_FEATURE_GLM5_KDA_TP, 1) != 0u ||
        ds4_tp_glm5_kda_output_kslice_feature(
            "true", DS4_TP_FEATURE_GLM5_KDA_TP, 1) != 0u ||
        ds4_tp_glm5_kda_output_kslice_feature("1", 0u, 1) != 0u ||
        ds4_tp_glm5_kda_output_kslice_feature(
            "1", DS4_TP_FEATURE_GLM5_KDA_TP, 0) != 0u) {
        fprintf(stderr,
                "FAIL GLM5 KDA output K-slice advertisement predicate\n");
        ok = 0;
    } else {
        fprintf(stderr,
                "PASS GLM5 KDA output K-slice advertisement predicate\n");
    }
    const char *invalid_kshard_env[] = {
        NULL, "", "0", "true", "10", "1 ", "1\n"
    };
    for (size_t i = 0;
         i < sizeof(invalid_kshard_env) / sizeof(invalid_kshard_env[0]);
         ++i) {
        if (ds4_tp_q4k_kshard_feature(invalid_kshard_env[i], 1, 0, 1, 1) != 0u) {
            fprintf(stderr, "FAIL q4k-kshard malformed env case %zu\n", i);
            ok = 0;
        }
    }
    if (ds4_tp_q4k_kshard_feature("1", 1, 0, 1, 1) !=
            DS4_TP_FEATURE_Q4K_KSHARD ||
        ds4_tp_q4k_kshard_feature("1", 0, 0, 1, 1) != 0u ||
        ds4_tp_q4k_kshard_feature("1", 1, 1, 1, 1) != 0u ||
        ds4_tp_q4k_kshard_feature("1", 1, 0, 0, 1) != 0u ||
        ds4_tp_q4k_kshard_feature("1", 1, 0, 1, 0) != 0u) {
        fprintf(stderr, "FAIL q4k-kshard advertisement predicate\n");
        ok = 0;
    } else {
        fprintf(stderr, "PASS q4k-kshard advertisement predicate\n");
    }
    if (ds4_tp_test_rdma_provider_decode_max_msg("odl_tb5_0") != 131072) {
        fprintf(stderr, "FAIL OdinLink decode message policy\n");
        ok = 0;
    } else {
        fprintf(stderr, "PASS OdinLink decode message policy\n");
    }
    if (ds4_tp_test_rdma_provider_decode_max_msg("mlx5_0") != 16384 ||
        ds4_tp_test_rdma_provider_decode_max_msg(NULL) != 16384) {
        fprintf(stderr, "FAIL generic/Mellanox decode message policy\n");
        ok = 0;
    } else {
        fprintf(stderr, "PASS generic/Mellanox decode message policy\n");
    }
    if (ds4_tp_test_rdma_negotiate_decode_max_msg(131072, 16384) != 16384 ||
        ds4_tp_test_rdma_negotiate_decode_max_msg(16384, 131072) != 16384 ||
        ds4_tp_test_rdma_negotiate_decode_max_msg(131072, 131072) != 131072) {
        fprintf(stderr, "FAIL decode message negotiation\n");
        ok = 0;
    } else {
        fprintf(stderr, "PASS decode message negotiation\n");
    }
    ok &= check_transport("transport auto falls back without RDMA",
                          DS4_TP_TRANSPORT_AUTO, 0, 0, 1, 0, NULL);
    ok &= check_transport("transport auto uses mutual RDMA",
                          DS4_TP_TRANSPORT_AUTO, 1, 1, 1, 1, NULL);
    ok &= check_transport("transport tcp stays TCP",
                          DS4_TP_TRANSPORT_TCP, 1, 1, 1, 0, NULL);
    ok &= check_transport("explicit RDMA succeeds mutually",
                          DS4_TP_TRANSPORT_RDMA, 1, 1, 1, 1, NULL);
    ok &= check_transport("explicit RDMA rejects missing local device",
                          DS4_TP_TRANSPORT_RDMA, 0, 1, 0, 0,
                          "tp: --transport rdma but this side has no active device");
    ok &= check_transport("explicit RDMA rejects missing peer device",
                          DS4_TP_TRANSPORT_RDMA, 1, 0, 0, 0,
                          "tp: --transport rdma but the peer side has no active device");
    ok &= check_connect_timeout("connect timeout default", NULL, 1800u);
    ok &= check_connect_timeout("connect timeout override", "2400", 2400u);
    ok &= check_connect_timeout("connect timeout rejects zero", "0", 1800u);
    ok &= check_connect_timeout("connect timeout rejects junk", "30s", 1800u);
    ok &= check_connect_timeout("connect timeout maximum", "86400", 86400u);
    unsetenv("DS4_TP_CONNECT_TIMEOUT_SEC");
    return ok ? 0 : 1;
}
