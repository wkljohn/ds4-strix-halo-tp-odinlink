#include "ds4_tp.h"

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

static int check_gate_slot(const char *name, uint32_t features,
                           uint32_t calls_per_token, uint64_t seq,
                           uint32_t want) {
    const uint32_t got = ds4_tp_test_gate_slot(
            2u, features, 0u, 1u, calls_per_token, seq);
    if (got != want) {
        fprintf(stderr, "FAIL %s: seq=%llu got=%u want=%u\n", name,
                (unsigned long long)seq, got, want);
        return 0;
    }
    fprintf(stderr, "PASS %s\n", name);
    return 1;
}

int main(void) {
    int ok = 1;
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
    ok &= check("hello q4k-wmma-kshard mismatch",
                DS4_TP_FEATURE_Q4K_WMMA | DS4_TP_FEATURE_Q4K_KSHARD,
                DS4_TP_FEATURE_Q4K_WMMA, 0,
                "tp hello: runtime feature mismatch (local=0x00080001 peer=0x00000001)");
    ok &= check("hello equal-q4k-kshard-interleaved",
                DS4_TP_FEATURE_Q4K_KSHARD |
                    DS4_TP_FEATURE_Q4K_KSHARD_INTERLEAVED,
                DS4_TP_FEATURE_Q4K_KSHARD |
                    DS4_TP_FEATURE_Q4K_KSHARD_INTERLEAVED,
                1, NULL);
    ok &= check("hello mismatched-q4k-kshard-interleaved",
                DS4_TP_FEATURE_Q4K_KSHARD |
                    DS4_TP_FEATURE_Q4K_KSHARD_INTERLEAVED,
                DS4_TP_FEATURE_Q4K_KSHARD, 0,
                "tp hello: runtime feature mismatch (local=0x00180000 peer=0x00080000)");
    ok &= check("hello equal-q4k-kshard-slot-reconstruct",
                DS4_TP_FEATURE_Q4K_KSHARD_INTERLEAVED |
                    DS4_TP_FEATURE_Q4K_KSHARD_SLOT_RECONSTRUCT,
                DS4_TP_FEATURE_Q4K_KSHARD_INTERLEAVED |
                    DS4_TP_FEATURE_Q4K_KSHARD_SLOT_RECONSTRUCT,
                1, NULL);
    ok &= check("hello mismatched-q4k-kshard-slot-reconstruct",
                DS4_TP_FEATURE_Q4K_KSHARD_INTERLEAVED |
                    DS4_TP_FEATURE_Q4K_KSHARD_SLOT_RECONSTRUCT,
                DS4_TP_FEATURE_Q4K_KSHARD_INTERLEAVED, 0,
                "tp hello: runtime feature mismatch (local=0x00300000 peer=0x00100000)");
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
    ok &= check("hello equal-q4k-row-shard",
                DS4_TP_FEATURE_Q4K_ROW_SHARD,
                DS4_TP_FEATURE_Q4K_ROW_SHARD, 1, NULL);
    ok &= check("hello mismatched-q4k-row-shard",
                DS4_TP_FEATURE_Q4K_ROW_SHARD, 0, 0,
                "tp hello: runtime feature mismatch (local=0x00400000 peer=0x00000000)");
    ok &= check("hello equal-q4k-row-shard-overlap",
                DS4_TP_FEATURE_Q4K_ROW_SHARD |
                    DS4_TP_FEATURE_Q4K_ROW_SHARD_OVERLAP,
                DS4_TP_FEATURE_Q4K_ROW_SHARD |
                    DS4_TP_FEATURE_Q4K_ROW_SHARD_OVERLAP, 1, NULL);
    ok &= check("hello mismatched-q4k-row-shard-overlap",
                DS4_TP_FEATURE_Q4K_ROW_SHARD |
                    DS4_TP_FEATURE_Q4K_ROW_SHARD_OVERLAP,
                DS4_TP_FEATURE_Q4K_ROW_SHARD, 0,
                "tp hello: runtime feature mismatch (local=0x00c00000 peer=0x00400000)");
    /* Two layers: ordinary slots are [attn0,ffn0,attn1,ffn1] =
     * [0,1,3,4].  Row sharding inserts mid slots [2,5] before each FFN. */
    const uint32_t ordinary_slots[] = {0u, 1u, 3u, 4u, 0u};
    const uint32_t row_shard_slots[] = {0u, 2u, 1u, 3u, 5u, 4u, 0u};
    for (uint32_t i = 0; i < sizeof(ordinary_slots) / sizeof(ordinary_slots[0]); i++) {
        char name[64];
        snprintf(name, sizeof(name), "ordinary gate slot %u", i + 1u);
        ok &= check_gate_slot(name, 0u, 4u, i + 1u, ordinary_slots[i]);
    }
    for (uint32_t i = 0; i < sizeof(row_shard_slots) / sizeof(row_shard_slots[0]); i++) {
        char name[64];
        snprintf(name, sizeof(name), "row-shard gate slot %u", i + 1u);
        ok &= check_gate_slot(name, DS4_TP_FEATURE_Q4K_ROW_SHARD,
                              6u, i + 1u, row_shard_slots[i]);
    }
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
