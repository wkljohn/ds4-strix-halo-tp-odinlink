#include "ds4.h"
#include "ds4_tp.h"
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>

#define REQUIRE(x) do { if (!(x)) { \
    fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); return 1; \
} } while (0)

static int full_argmax(const float *x, int n, int excluded) {
    int best = -1;
    for (int i = 0; i < n; ++i)
        if (i != excluded && (best < 0 || x[i] > x[best])) best = i;
    return best;
}

int main(void) {
    /* Ties are frequent in these fixtures. Exercise every possible excluded
     * token, both ownership halves and negative finite extremes. */
    unsigned rng = 17u;
    for (int trial = 0; trial < 1000; ++trial) {
        float values[32];
        for (unsigned i = 0; i < 32u; ++i) {
            rng = rng * 1664525u + 1013904223u;
            values[i] = trial == 0 ? -FLT_MAX : (float)(rng % 7u) - 3.0f;
        }
        ds4_tp_logits_top2 halves[2];
        for (unsigned rank = 0; rank < 2u; ++rank) {
            REQUIRE(ds4_tp_logits_top2_make(values + rank * 16u, rank * 16u,
                                             16u, &halves[rank]));
            REQUIRE(ds4_tp_logits_top2_valid(&halves[rank], rank * 16u, 16u));
        }
        for (int excluded = -1; excluded < 32; ++excluded) {
            int best = -1;
            for (int rank = 0; rank < 2; ++rank)
                for (int j = 0; j < 2; ++j) {
                    const int id = halves[rank].id[j];
                    if (id != excluded && (best < 0 || values[id] > values[best] ||
                        (values[id] == values[best] && id < best))) best = id;
                }
            REQUIRE(best == full_argmax(values, 32, excluded));
        }
    }
    fprintf(stderr, "PASS 33000 compact/full greedy selections, ties and exclusions\n");
    float values[4] = {-0.0f, 0.0f, -FLT_MAX, FLT_MAX};
    ds4_tp_logits_top2 top;
    REQUIRE(ds4_tp_logits_top2_make(values, 4u, 4u, &top));
    REQUIRE(top.id[0] == 7 && top.id[1] == 4);
    const float bad[] = {NAN, INFINITY, -INFINITY};
    for (unsigned i = 0; i < 3u; ++i) {
        values[2] = bad[i];
        REQUIRE(!ds4_tp_logits_top2_make(values, 4u, 4u, &top));
        REQUIRE(top.id[0] == -1 && top.id[1] == -1);
        top = (ds4_tp_logits_top2){{4, 5}, {2.0f, bad[i]}};
        REQUIRE(!ds4_tp_logits_top2_valid(&top, 4u, 4u));
    }
    top = (ds4_tp_logits_top2){{4, 5}, {2.0f, 1.0f}};
    REQUIRE(ds4_tp_logits_top2_valid(&top, 4u, 4u));
    REQUIRE(!ds4_tp_logits_top2_valid(&top, 0u, 4u));
    top.id[1] = 4;
    REQUIRE(!ds4_tp_logits_top2_valid(&top, 4u, 4u));
    top = (ds4_tp_logits_top2){{5, 4}, {1.0f, 1.0f}};
    REQUIRE(!ds4_tp_logits_top2_valid(&top, 4u, 4u));
    top.value[1] = 2.0f;
    REQUIRE(!ds4_tp_logits_top2_valid(&top, 4u, 4u));
    top.id[1] = -1;
    REQUIRE(!ds4_tp_logits_top2_valid(&top, 4u, 4u));
    REQUIRE(!ds4_tp_logits_top2_make(values, INT32_MAX, 4u, &top));
    REQUIRE(!ds4_tp_logits_top2_make(values, 0u, 1u, &top));
    REQUIRE(!ds4_tp_logits_top2_make(NULL, 0u, 4u, &top));
    fprintf(stderr, "PASS invalid/nonfinite/unsorted/out-of-range records refused\n");
    char err[200];
    const uint64_t flag = DS4_TP_PREFILL_CONFIG_GLM5_OUTPUT_TOP2_RDMA;
    REQUIRE(ds4_tp_test_hello_validate_prefill_config(flag, flag, err, sizeof(err)));
    REQUIRE(!ds4_tp_test_hello_validate_prefill_config(flag, 0u, err, sizeof(err)));
    REQUIRE(!ds4_tp_test_hello_validate_prefill_config(0u, flag, err, sizeof(err)));
    REQUIRE(!ds4_tp_test_hello_validate_prefill_config(flag, UINT64_C(1) << 37,
                                                       err, sizeof(err)));
    fprintf(stderr, "PASS compact hello mismatch and retired full-half bit refused\n");
    for (int transport = DS4_TP_TRANSPORT_AUTO;
         transport <= DS4_TP_TRANSPORT_TCP; ++transport)
        for (int active = 0; active < 2; ++active)
            for (int negotiated = 0; negotiated < 2; ++negotiated)
                REQUIRE(ds4_tp_test_top2_rdma_refusal(transport, active, negotiated));
    fprintf(stderr, "PASS missing RDMA slab clears stale record; no socket payload or fallback\n");
    const int desync = ds4_tp_test_top2_rdma_desync();
    REQUIRE(desync != 0);
    fprintf(stderr, "%s stale exchange sequence\n", desync > 0 ? "PASS" : "SKIP (no verbs)");
    REQUIRE(ds4_test_glm5_compact_consumers());
    fprintf(stderr, "PASS both-rank session APIs: greedy/exclusion/stale/full-logit refusal\n");
    return 0;
}
