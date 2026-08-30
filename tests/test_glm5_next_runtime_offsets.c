#include "ds4_glm5_next_runtime.h"
#include "ds4_tp.h"

#include <stdio.h>
#include <string.h>

#define CHECK(expr, message) do { \
    if (!(expr)) { fprintf(stderr, "FAIL %s\n", message); return 0; } \
} while (0)

static void fill_words(void *value, size_t bytes, uint64_t *next) {
    uint64_t *word = value;
    for (size_t i = 0u; i < bytes / sizeof(uint64_t); ++i)
        word[i] = (*next)++;
}

static void make_valid(ds4_glm5_next_model_offsets *model) {
    memset(model, 0, sizeof(*model));
    uint64_t next = 1u;
    model->token_embd = next++;
    model->output_norm = next++;
    model->output = next++;
    model->nextn_eh_proj = next++;
    model->layer_count = DS4_GLM5_NEXT_LAYER_COUNT;
    model->trunk_count = DS4_GLM5_NEXT_TRUNK_COUNT;
    model->nextn_count = 1u;
    model->rms_norm_eps = 1.0e-5f;
    model->hc_eps = 1.0e-6f;
    for (uint32_t il = 0u; il < model->layer_count; ++il) {
        ds4_glm5_next_layer_offsets *layer = &model->layer[il];
        layer->layer = il;
        layer->is_trunk = il < model->trunk_count;
        layer->attn_norm = next++;
        layer->ffn_norm = next++;
        const bool mla = il == 45u || (il & 3u) == 3u;
        layer->attention = mla ? DS4_GLM5_NEXT_ATTN_MLA :
                                 DS4_GLM5_NEXT_ATTN_KDA;
        if (mla) fill_words(&layer->mla, sizeof(layer->mla), &next);
        else {
            fill_words(&layer->kda, sizeof(layer->kda), &next);
            layer->kda.q_type = 0u;
            layer->kda.k_type = 0u;
            layer->kda.v_type = 0u;
            layer->kda.output_type = 0u;
            layer->kda.f_a_type = 0u;
            layer->kda.f_b_type = 0u;
            layer->kda.g_a_type = 0u;
            layer->kda.g_b_type = 0u;
            layer->kda.beta_type = 0u;
        }
        if (il < DS4_GLM5_NEXT_LEADING_DENSE) {
            layer->ffn = DS4_GLM5_NEXT_FFN_DENSE;
            layer->ffn_weight.gate = next++;
            layer->ffn_weight.up = next++;
            layer->ffn_weight.down = next++;
        } else {
            layer->ffn = DS4_GLM5_NEXT_FFN_ROUTED;
            fill_words(&layer->ffn_weight.gate_exps,
                       sizeof(layer->ffn_weight) - 3u * sizeof(uint64_t),
                       &next);
        }
        if (layer->is_trunk)
            fill_words(&layer->hc, sizeof(layer->hc), &next);
    }
}

static int test_contract(void) {
    uint32_t visible = 0u;
    CHECK(!ds4_glm5_next_mla_dense_selection_visible(0u, 1u, NULL),
          "dense-selection policy rejects a missing result");
    CHECK(ds4_glm5_next_mla_dense_selection_visible(0u, 1u, &visible) &&
              visible == 1u,
          "dense-selection policy accepts the first visible row");
    CHECK(ds4_glm5_next_mla_dense_selection_visible(7u, 8u, &visible) &&
              visible == 8u,
          "dense-selection policy crosses the former four-token limit");
    CHECK(ds4_glm5_next_mla_dense_selection_visible(2047u, 4096u, &visible) &&
              visible == DS4_GLM5_NEXT_INDEX_TOP_K,
          "dense-selection policy covers the official top-k boundary");
    CHECK(!ds4_glm5_next_mla_dense_selection_visible(2048u, 4096u, &visible),
          "token 2049 fails closed until pooled selection is implemented");
    CHECK(!ds4_glm5_next_mla_dense_selection_visible(8u, 8u, &visible),
          "dense-selection policy enforces state capacity");
    uint32_t pools = 0u, selected_pools = 0u, selected_tokens = 0u;
    CHECK(!ds4_glm5_next_mla_sparse_selection_plan(
              2047u, 4096u, 2048u, 4u, &visible, &pools,
              &selected_pools, &selected_tokens),
          "sparse-selection policy rejects the dense side of crossover");
    CHECK(ds4_glm5_next_mla_sparse_selection_plan(
              2048u, 4096u, 2048u, 4u, &visible, &pools,
              &selected_pools, &selected_tokens) &&
              visible == 2049u && pools == 512u &&
              selected_pools == 512u && selected_tokens == 2049u,
          "sparse-selection policy appends the first current tail row");
    CHECK(ds4_glm5_next_mla_sparse_selection_plan(
              2049u, 4096u, 2048u, 4u, &visible, &pools,
              &selected_pools, &selected_tokens) &&
              visible == 2050u && pools == 512u &&
              selected_pools == 512u && selected_tokens == 2050u,
          "sparse-selection policy appends the second current tail row");
    CHECK(ds4_glm5_next_mla_sparse_selection_plan(
              2051u, 4096u, 2048u, 4u, &visible, &pools,
              &selected_pools, &selected_tokens) &&
              visible == 2052u && pools == 513u &&
              selected_pools == 512u && selected_tokens == 2048u,
          "sparse-selection policy selects after publishing a new pool");
    CHECK(ds4_glm5_next_mla_sparse_selection_plan(
              2052u, 4096u, 2048u, 4u, &visible, &pools,
              &selected_pools, &selected_tokens) &&
              visible == 2053u && pools == 513u &&
              selected_pools == 512u && selected_tokens == 2049u,
          "sparse-selection policy appends tail after the new pool");
    CHECK(!ds4_glm5_next_mla_sparse_selection_plan(
              4096u, 4096u, 2048u, 4u, &visible, &pools,
              &selected_pools, &selected_tokens),
          "sparse-selection policy enforces state capacity");
    CHECK(ds4_glm5_next_mla_dense_selection_visible_for_topk(
              7u, 16u, 8u, &visible) && visible == 8u &&
          !ds4_glm5_next_mla_dense_selection_visible_for_topk(
              8u, 16u, 8u, &visible),
          "scaled policy crosses from dense token 8 to pooled token 9");
    CHECK(!ds4_glm5_next_mla_dense_selection_visible_for_topk(
              0u, 16u, 0u, &visible),
          "scaled policy rejects a zero top-k");
    CHECK(ds4_glm5_next_prefill_chunk(0u, 4096u, 1024u) == 1024u &&
          ds4_glm5_next_prefill_chunk(1024u, 3072u, 1024u) == 1024u,
          "prefill planner retains dense 1024-row tiles");
    CHECK(ds4_glm5_next_prefill_chunk(1536u, 1024u, 1024u) == 512u,
          "prefill planner stops a straddling tile at sparse crossover");
    CHECK(ds4_glm5_next_prefill_chunk(2048u, 512u, 1024u) == 1u &&
          ds4_glm5_next_prefill_chunk(2347u, 1u, 1024u) == 1u,
          "prefill planner uses scalar sparse execution after crossover");
    CHECK(ds4_glm5_next_prefill_chunk(0u, 8u, 1u) == 1u &&
          ds4_glm5_next_prefill_chunk(0u, 0u, 1024u) == 0u,
          "prefill planner preserves scalar and empty contracts");
    uint64_t gate_mask[DS4_GLM5_NEXT_TP_GATE_MASK_WORDS] = {0};
    uint32_t gate_count = 0;
    CHECK(ds4_glm5_next_build_tp_gate_mask(gate_mask, &gate_count, 0u) &&
          gate_count == 53u,
          "shared GLM5.3 53-gate TP schedule");
    CHECK((gate_mask[0] & (UINT64_C(1) << 6u)) != 0u &&
          (gate_mask[0] & (UINT64_C(1) << 7u)) != 0u &&
          (gate_mask[0] & (UINT64_C(1) << 8u)) == 0u &&
          (gate_mask[1] & (UINT64_C(1) << (89u - 64u))) != 0u,
          "GLM5.3 TP mask covers MLA+FFN and final trunk FFN");
    uint64_t kda_gate_mask[DS4_GLM5_NEXT_TP_GATE_MASK_WORDS] = {0};
    uint32_t kda_gate_count = 0;
    CHECK(ds4_glm5_next_build_tp_gate_mask(
              kda_gate_mask, &kda_gate_count,
              DS4_TP_FEATURE_GLM5_KDA_TP) &&
          kda_gate_count == 87u,
          "shared GLM5.3 87-gate KDA-TP schedule");
    for (uint32_t word = 0u; word < DS4_GLM5_NEXT_TP_GATE_MASK_WORDS; ++word)
        CHECK((kda_gate_mask[word] & gate_mask[word]) == gate_mask[word],
              "KDA-TP schedule is a strict superset of baseline schedule");
    for (uint32_t il = 0u; il < DS4_GLM5_NEXT_TRUNK_COUNT; ++il) {
        const uint32_t slot = il * 2u;
        const bool added =
            (kda_gate_mask[slot / 64u] &
             (UINT64_C(1) << (slot % 64u))) != 0u &&
            (gate_mask[slot / 64u] &
             (UINT64_C(1) << (slot % 64u))) == 0u;
        CHECK(added == !ds4_glm5_next_layer_is_mla(il),
              "KDA-TP adds exactly the KDA attention slots");
    }
    ds4_glm5_next_model_offsets model;
    make_valid(&model);
    CHECK(ds4_glm5_next_model_offsets_validate(&model),
          "valid 45-layer trunk plus one nextn block");
    model.rms_norm_eps = 0.0f;
    CHECK(!ds4_glm5_next_model_offsets_validate(&model),
          "missing model RMS epsilon rejected");
    make_valid(&model);
    model.hc_eps = 0.0f;
    CHECK(!ds4_glm5_next_model_offsets_validate(&model),
          "missing model hyper-connection epsilon rejected");
    make_valid(&model);
    model.trunk_count = 46u;
    CHECK(!ds4_glm5_next_model_offsets_validate(&model),
          "46-layer first-token trunk rejected");
    make_valid(&model);
    model.nextn_eh_proj = 0u;
    CHECK(!ds4_glm5_next_model_offsets_validate(&model),
          "missing nextn projection rejected");
    make_valid(&model);
    model.layer[2].ffn = DS4_GLM5_NEXT_FFN_ROUTED;
    CHECK(!ds4_glm5_next_model_offsets_validate(&model),
          "routed FFN in leading dense layer rejected");
    make_valid(&model);
    model.layer[3].ffn = DS4_GLM5_NEXT_FFN_DENSE;
    CHECK(!ds4_glm5_next_model_offsets_validate(&model),
          "dense FFN in routed layer rejected");
    make_valid(&model);
    model.layer[45].hc.attn_fn = 1u;
    CHECK(!ds4_glm5_next_model_offsets_validate(&model),
          "mHC weights on nextn-only block rejected");
    make_valid(&model);
    model.layer[7].mla.output = 0u;
    CHECK(!ds4_glm5_next_model_offsets_validate(&model),
          "incomplete MLA offsets rejected");
    make_valid(&model);
    model.layer[0].kda.q = 0u;
    CHECK(!ds4_glm5_next_model_offsets_validate(&model),
          "incomplete KDA offsets rejected");
    make_valid(&model);
    model.layer[0].mla.q_a = 1u;
    CHECK(!ds4_glm5_next_model_offsets_validate(&model),
          "mixed KDA and MLA offsets rejected");
    make_valid(&model);
    model.layer[0].hc.attn_fn = 0u;
    CHECK(!ds4_glm5_next_model_offsets_validate(&model),
          "incomplete trunk mHC offsets rejected");
    make_valid(&model);
    model.layer[0].attention = DS4_GLM5_NEXT_ATTN_INVALID;
    CHECK(!ds4_glm5_next_model_offsets_validate(&model),
          "invalid attention kind rejected");
    make_valid(&model);
    model.layer[0].is_trunk = false;
    CHECK(!ds4_glm5_next_model_offsets_validate(&model),
          "trunk marker mismatch rejected");
    make_valid(&model);
    model.layer[0].ffn_weight.gate_exps = 1u;
    CHECK(!ds4_glm5_next_model_offsets_validate(&model),
          "mixed dense and routed FFN offsets rejected");
    make_valid(&model);
    model.nextn_count = 2u;
    CHECK(!ds4_glm5_next_model_offsets_validate(&model),
          "multiple nextn blocks rejected");
    return 1;
}

int main(void) {
    const int ok = test_contract();
    if (ok) fprintf(stderr, "PASS GLM5-next runtime offset contract\n");
    return ok ? 0 : 1;
}
