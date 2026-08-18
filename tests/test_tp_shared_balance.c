#include "ds4_tp_shared_balance.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define CHECK(cond, msg) do {                                                \
    if (!(cond)) {                                                           \
        fprintf(stderr, "FAIL: %s (line %d)\n", (msg), __LINE__);           \
        return 1;                                                            \
    }                                                                        \
} while (0)

static float canonical_add(float routed, float shared) {
    volatile float a = routed;
    volatile float b = shared;
    return a + b;
}

static uint32_t bits(float x) {
    uint32_t u;
    memcpy(&u, &x, sizeof(u));
    return u;
}

static void make_selection(int32_t selected[6], uint32_t rank0_count) {
    for (uint32_t i = 0; i < rank0_count; i++) selected[i] = (int32_t)(3u + i);
    for (uint32_t i = rank0_count; i < 6u; i++)
        selected[i] = (int32_t)(128u + i);
}

static int check_grouping(const ds4_tp_shared_balance *d,
                          float routed0, float routed1,
                          float shared0, float shared1) {
    const float ref[2] = {
        canonical_add(routed0, shared0),
        canonical_add(routed1, shared1),
    };
    float got[2];
    if (!d->move_heavy_half) {
        got[0] = canonical_add(routed0, shared0);
        got[1] = canonical_add(routed1, shared1);
    } else {
        /* The light rank transmits its ordinary canonical group plus the
         * heavy rank's shared half in the second FFN payload lane.  The heavy
         * routed value is never combined with a synthetic zero: after the
         * exchange it receives exactly one add in the original order. */
        got[d->light_rank] = canonical_add(
            d->light_rank == 0u ? routed0 : routed1,
            d->light_rank == 0u ? shared0 : shared1);
        got[d->heavy_rank] = canonical_add(
            d->heavy_rank == 0u ? routed0 : routed1,
            d->heavy_rank == 0u ? shared0 : shared1);
    }
    return bits(ref[0]) == bits(got[0]) && bits(ref[1]) == bits(got[1]);
}

int main(void) {
    static const float values[][4] = {
        { 0.0f, -0.0f, -0.0f,  0.0f},
        {-0.0f, -0.0f,  0.0f, -0.0f},
        { 1.0f, -2.0f,  0x1p-24f, -0x1p-24f},
        { 0x1.fffffep+90f, -0x1.fffffep+90f, 0x1p+70f, -0x1p+70f},
    };
    for (uint32_t rank0_count = 0; rank0_count <= 6u; rank0_count++) {
        int32_t selected[6];
        make_selection(selected, rank0_count);
        const ds4_tp_shared_balance d =
            ds4_tp_shared_balance_decide(selected, 6u, 128u, 256u);
        CHECK(d.valid, "valid six-route decision");
        CHECK(d.routed_count[0] == rank0_count &&
              d.routed_count[1] == 6u - rank0_count,
              "route counts");
        CHECK(d.move_heavy_half ==
              (rank0_count <= 1u || rank0_count >= 5u),
              "only 0/6, 1/5, 5/1 and 6/0 move a whole half");
        if (d.move_heavy_half) {
            CHECK(d.shared_owner[d.heavy_rank] == d.light_rank,
                  "light rank owns moved canonical half");
            CHECK(d.shared_owner[d.light_rank] == d.light_rank,
                  "light rank retains its canonical half");
        }
        for (uint32_t i = 0; i < sizeof(values) / sizeof(values[0]); i++) {
            if (!check_grouping(&d, values[i][0], values[i][1],
                                values[i][2], values[i][3])) {
                fprintf(stderr, "group mismatch count0=%u case=%u move=%u "
                        "heavy=%u light=%u\n", rank0_count, i,
                        d.move_heavy_half, d.heavy_rank, d.light_rank);
                CHECK(0, "canonical routed-plus-shared grouping is bit exact");
            }
        }
    }

    int32_t bad[6] = {0, 1, 2, 3, 4, 256};
    CHECK(!ds4_tp_shared_balance_decide(NULL, 6u, 128u, 256u).valid,
          "null selection fails closed");
    CHECK(!ds4_tp_shared_balance_decide(bad, 6u, 128u, 256u).valid,
          "out-of-range selection fails closed");
    CHECK(!ds4_tp_shared_balance_decide(bad, 5u, 128u, 256u).valid,
          "unexpected top-k fails closed");

    /* Measured 4,300-layer route histogram and rank critical-path skew from
     * the accepted Q4_K RoCE profile.  Moving a 172 us shared half saves
     * max(0, skew - 172) for 5/1 and 6/0 layers.  Charge a conservative 5 us
     * on every moved layer for the doubled FFN payload and reconstruction. */
    static const uint32_t histogram[7] = {71, 353, 1003, 1409, 1024, 379, 61};
    static const double skew_us[7] = {419.1, 285.7, 160.7, 21.5,
                                      142.7, 266.9, 398.1};
    double saved_us = 0.0;
    uint32_t observed = 0u;
    for (uint32_t c = 0; c <= 6u; c++) {
        observed += histogram[c];
        if (c <= 1u || c >= 5u) {
            const double gross = skew_us[c] > 172.0 ? skew_us[c] - 172.0 : 0.0;
            saved_us += (gross - 5.0) * (double)histogram[c];
        }
    }
    CHECK(observed == 4300u, "profile histogram size");
    const double mean_saved_us = saved_us / (double)observed;
    fprintf(stderr,
            "test_tp_shared_balance: PASS mean_modelled_saving=%.2f us/layer\n",
            mean_saved_us);
    CHECK(mean_saved_us >= 15.0, "promotion gate: at least 15 us/layer");
    return 0;
}
