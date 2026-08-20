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
