#include "ds4_tp.h"

#include <stdio.h>
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

int main(void) {
    int ok = 1;
    ok &= check("hello equal-enabled",
                DS4_TP_FEATURE_Q4K_WMMA, DS4_TP_FEATURE_Q4K_WMMA, 1, NULL);
    ok &= check("hello equal-disabled", 0, 0, 1, NULL);
    ok &= check("hello mismatched-feature-masks",
                DS4_TP_FEATURE_Q4K_WMMA, 0, 0,
                "tp hello: runtime feature mismatch (local=0x00000001 peer=0x00000000)");
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
    return ok ? 0 : 1;
}
