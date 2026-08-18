#ifndef DS4_TP_SHARED_BALANCE_H
#define DS4_TP_SHARED_BALANCE_H

#include <stdbool.h>
#include <stdint.h>

/* Decode-only scheduling decision for the two canonical shared-expert halves.
 * Both TP ranks hold the same six selected expert ids, so this function must
 * remain a pure function of those ids and the hello-negotiated split. */
typedef struct {
    uint32_t routed_count[2];
    uint32_t shared_owner[2];
    uint32_t heavy_rank;
    uint32_t light_rank;
    bool valid;
    bool move_heavy_half;
} ds4_tp_shared_balance;

static inline ds4_tp_shared_balance ds4_tp_shared_balance_decide(
        const int32_t *selected,
        uint32_t n_selected,
        uint32_t first_rank1,
        uint32_t n_expert) {
    ds4_tp_shared_balance d = {
        .routed_count = {0u, 0u},
        .shared_owner = {0u, 1u},
        .heavy_rank = 0u,
        .light_rank = 1u,
        .valid = false,
        .move_heavy_half = false,
    };
    if (!selected || n_selected != 6u || first_rank1 == 0u ||
        first_rank1 >= n_expert) {
        return d;
    }
    for (uint32_t i = 0; i < n_selected; i++) {
        const int32_t expert = selected[i];
        if (expert < 0 || (uint32_t)expert >= n_expert) return d;
        d.routed_count[(uint32_t)expert >= first_rank1]++;
    }
    d.valid = true;
    if (d.routed_count[1] > d.routed_count[0]) {
        d.heavy_rank = 1u;
        d.light_rank = 0u;
    }
    const uint32_t delta = d.routed_count[d.heavy_rank] -
                           d.routed_count[d.light_rank];
    /* A whole shared half costs about the same as two routed experts.  Move
     * it only for the 5/1 and 6/0 cases; 4/2 would over-correct the critical
     * path and is reserved for a future finer-grained schedule. */
    d.move_heavy_half = delta >= 4u;
    if (d.move_heavy_half) d.shared_owner[d.heavy_rank] = d.light_rank;
    return d;
}

#endif
