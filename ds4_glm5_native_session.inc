/* Native session orchestration. Included after ds4_session is defined; the
 * socket/arithmetic fixture includes these same functions with controlled
 * backend operations to test publication and failure ordering. */
#include <time.h>
static double glm5_native_wall_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int ds4_session_glm5_native_teach(ds4_session *s,
        ds4_glm5_next_workspace *target_ws, const ds4_gpu_tensor *hc,
        const int *tokens, uint32_t rows) {
    if (!s->glm5_native_rows) return 1;
    const uint32_t prefix = (uint32_t)s->checkpoint.len;
    const uint64_t row = (uint64_t)DS4_N_EMBD * sizeof(float);
    if (!rows || !tokens || prefix > (uint32_t)s->ctx_size ||
        rows > (uint32_t)s->ctx_size - prefix ||
        s->glm5_native_previous_valid != (prefix != 0u) ||
        s->glm5_native_state.mla[45].token_count != (prefix ? prefix - 1u : 0u)) return 0;
    int ok = 1;
    /* Keep target prefill's configured batch; only teacher scratch is tiled.
     * No checkpoint history is published until every warm tile succeeds. */
    for (uint32_t offset = 0; ok && offset < rows; offset += 256u) {
        const uint32_t n = rows - offset < 256u ? rows - offset : 256u;
        const bool has_previous = prefix + offset != 0u;
        const uint32_t skip = has_previous ? 0u : 1u, count = n - skip;
        ds4_gpu_tensor *target = ds4_gpu_tensor_view(hc, offset * row * 4u, n * row * 4u);
        ds4_gpu_tensor *normalized = ds4_gpu_tensor_view(s->glm5_native_normalized, 0, n * row);
        ds4_gpu_tensor *shifted = count ? ds4_gpu_tensor_view(s->glm5_native_shifted, 0, count * row) : NULL;
        ok = target && normalized && (!count || shifted) &&
            ds4_glm5_next_draft_target_hidden_rows(&s->glm5_next_exec, target_ws, target, normalized, n);
        uint32_t ids[256];
        for (uint32_t t = 0; ok && t < count; ++t) {
            ok = tokens[offset + t + skip] >= 0 && (uint32_t)tokens[offset + t + skip] < DS4_N_VOCAB;
            ids[t] = (uint32_t)tokens[offset + t + skip];
        }
        if (ok && count) {
            if (has_previous) ok = ds4_gpu_tensor_copy(shifted, 0, s->glm5_native_previous, 0, row);
            const uint32_t remaining = count - (has_previous ? 1u : 0u);
            if (ok && remaining) ok = ds4_gpu_tensor_copy(shifted, has_previous ? row : 0u,
                normalized, 0, remaining * row);
            ds4_glm5_next_workspace *warm = s->glm5_native_ws;
            if (count > 1u) {
                if (ds4_glm5_next_workspace_capacity(s->glm5_native_warm_ws) != count) {
                    ds4_glm5_next_workspace_destroy(s->glm5_native_warm_ws);
                    s->glm5_native_warm_ws = ds4_glm5_next_draft_workspace_create_rows(count, s->ctx_size);
                }
                warm = s->glm5_native_warm_ws;
            }
            if (ok) ok = warm && ds4_glm5_next_draft_warm_rows(&s->glm5_next_exec,
                &s->glm5_native_state, warm, shifted, ids, count);
        }
        if (ok) ok = ds4_gpu_tensor_copy(s->glm5_native_previous, 0, normalized,
            (n - 1u) * row, row) && ds4_gpu_synchronize();
        ds4_gpu_tensor_free(shifted); ds4_gpu_tensor_free(normalized); ds4_gpu_tensor_free(target);
    }
    s->glm5_native_previous_valid = ok != 0;
    if (!ok) {
        ds4_glm5_next_state_invalidate(&s->glm5_native_state);
        ds4_glm5_next_state_invalidate(&s->glm5_next_state);
    }
    return ok;
}

static int glm5_native_argmax(const float *logits) {
    if (!logits || !isfinite(logits[0])) return -1;
    int best = 0;
    for (uint32_t i = 1u; i < DS4_N_VOCAB; ++i) {
        if (!isfinite(logits[i])) return -1;
        if (logits[i] > logits[best]) best = (int)i;
    }
    return best;
}

static int ds4_session_glm5_native_cycle_run(ds4_session *s,
        const ds4_tp_native_cycle *c, int *accepted_out, char *err, size_t errlen) {
    if (!s || !s->engine || !s->engine->tp.active || !c) return -1;
    ds4_engine *e = s->engine;
    ds4_glm5_next_exec_ctx *x = &s->glm5_next_exec;
    const uint32_t rows = c->rows;
    const char *profile_env = getenv("DS4_GLM5_NATIVE_PHASE_PROFILE");
    const bool profile = profile_env && strcmp(profile_env, "1") == 0;
    const char *draft_profile_env = getenv("DS4_GLM5_NATIVE_DRAFT_PROFILE");
    const bool draft_profile = draft_profile_env && strcmp(draft_profile_env, "1") == 0;
    const double cycle_start = profile ? glm5_native_wall_sec() : 0.0;
    double draft_sec = 0.0, agree_sec = 0.0, verify_sec = 0.0;
    double select_sec = 0.0, commit_sec = 0.0, phase_start = 0.0;
    uint32_t tokens[8] = {0}, predictions[8] = {0};
    const uint64_t row = (uint64_t)DS4_N_EMBD * sizeof(float);
    const uint64_t vocab_row = (uint64_t)DS4_N_VOCAB * sizeof(float);
    bool ok = s->glm5_next_ready && s->glm5_native_rows && s->checkpoint_valid &&
        s->glm5_native_previous_valid && c->session_id == s->tp_session_id &&
        c->cycle && c->cycle == s->glm5_native_cycle + 1u &&
        c->prefix == (uint32_t)s->checkpoint.len && c->prefix &&
        ds4_tp_glm5_native_width_valid(rows) && rows <= s->glm5_native_rows &&
        c->prefix <= (uint32_t)s->ctx_size && rows <= (uint32_t)s->ctx_size - c->prefix &&
        c->root >= 0 && (uint32_t)c->root < DS4_N_VOCAB && c->eos >= -1 &&
        (c->eos < 0 || (uint32_t)c->eos < DS4_N_VOCAB) && c->root != c->eos &&
        s->glm5_native_state.mla[45].token_count + 1u == c->prefix &&
        s->glm5_next_state.valid && s->glm5_native_state.valid && !ds4_tp_failed(e->tp.ctx);
    const uint32_t wi = rows >= 6u ? 2u : rows == 4u ? 1u : 0u;
    ds4_glm5_next_workspace *batch = ok ? s->glm5_native_verify_ws[wi] : NULL;
    ok = ok && ds4_tp_glm5_native_workspace_rows(s->glm5_native_rows, wi) == rows &&
        ds4_glm5_next_workspace_capacity(batch) == rows;
    ds4_gpu_tensor *hc[2] = {NULL, NULL}, *logits = NULL, *normalized = NULL;
    if (ok) {
        hc[0] = ds4_gpu_tensor_view(s->glm5_native_hc[0], 0, rows * row * 4u);
        hc[1] = ds4_gpu_tensor_view(s->glm5_native_hc[1], 0, rows * row * 4u);
        logits = ds4_gpu_tensor_view(s->glm5_native_logits, 0, rows * vocab_row);
        normalized = ds4_gpu_tensor_view(s->glm5_native_normalized, 0, rows * row);
        ok = batch && hc[0] && hc[1] && logits && normalized;
    }
    if (ok && s->checkpoint.cap - s->checkpoint.len < (int)rows) {
        const int capacity = s->checkpoint.len + (int)rows;
        int *history = realloc(s->checkpoint.v, (size_t)capacity * sizeof(*history));
        ok = history != NULL;
        if (ok) { s->checkpoint.v = history; s->checkpoint.cap = capacity; }
    }
    tokens[0] = (uint32_t)c->root;
    ds4_glm5_next_mla_state *view = NULL;
    if (ok) ok = ds4_glm5_next_mla_verify_begin(&s->glm5_native_state.mla[45], rows - 1u, &view) &&
        ds4_gpu_tensor_copy(s->glm5_native_chain, 0, s->glm5_native_previous, 0, row) &&
        ds4_gpu_synchronize();
    if (profile) phase_start = glm5_native_wall_sec();
    if (!ds4_tp_native_agree(e->tp.ctx, c, 0u, 0u, tokens, ok, err, errlen)) goto failed;
    if (profile) agree_sec += glm5_native_wall_sec() - phase_start;
    s->glm5_native_cycle = c->cycle;
    ds4_glm5_next_workspace_begin_decode(s->glm5_next_ws);
    for (uint32_t t = 0; t < rows - 1u; ++t) {
        if (profile) phase_start = glm5_native_wall_sec();
        const double step_begin = draft_profile ? glm5_native_wall_sec() : 0.0;
        if (ok) ok = ds4_glm5_next_draft_step(x, view, s->glm5_native_ws,
            s->glm5_native_chain, tokens[t], s->glm5_native_hidden, s->glm5_next_logits);
        const double step_end = draft_profile ? glm5_native_wall_sec() : 0.0;
        if (ok) ok = ds4_gpu_tensor_read(s->glm5_next_logits, 0, s->sample_probs, vocab_row);
        const double read_end = draft_profile ? glm5_native_wall_sec() : 0.0;
        const int proposed = ok ? glm5_native_argmax(s->sample_probs) : -1;
        const double argmax_end = draft_profile ? glm5_native_wall_sec() : 0.0;
        ok = ok && proposed >= 0;
        tokens[t + 1u] = proposed >= 0 ? (uint32_t)proposed : UINT32_MAX;
        if (ok && t + 1u < rows - 1u) ok = ds4_gpu_tensor_copy(s->glm5_native_chain,
            0, s->glm5_native_hidden, 0, row);
        const double chain_end = draft_profile ? glm5_native_wall_sec() : 0.0;
        if (profile) {
            const double now = glm5_native_wall_sec();
            draft_sec += now - phase_start; phase_start = now;
        }
        if (!ds4_tp_native_agree(e->tp.ctx, c, t + 1u, 0u, tokens, ok, err, errlen)) goto failed;
        if (profile) agree_sec += glm5_native_wall_sec() - phase_start;
        if (draft_profile) fprintf(stderr,
            "NATIVE_DRAFT_HOST rank=%u cycle=%llu prefix=%u step=%u token=%u proposal=%u step_ms=%.6f readback_ms=%.6f argmax_ms=%.6f chain_submit_ms=%.6f agreement_ms=%.6f\n",
            x->tp_rank, (unsigned long long)c->cycle, c->prefix, t, tokens[t], tokens[t + 1u],
            (step_end - step_begin) * 1000.0, (read_end - step_end) * 1000.0,
            (argmax_end - read_end) * 1000.0, (chain_end - argmax_end) * 1000.0,
            (glm5_native_wall_sec() - chain_end) * 1000.0);
    }
    if (profile) phase_start = glm5_native_wall_sec();
    ok = ds4_glm5_next_target_verify(x, &s->glm5_next_state, batch, s->glm5_next_ws,
        tokens, rows, hc[0], hc[1], logits);
    if (profile) {
        const double now = glm5_native_wall_sec();
        verify_sec = now - phase_start; phase_start = now;
    }
    ok = ok && ds4_gpu_tensor_read(logits, 0, s->glm5_native_host_logits, rows * vocab_row);
    for (uint32_t t = 0; ok && t < rows; ++t) {
        const int next = glm5_native_argmax(s->glm5_native_host_logits + (uint64_t)t * DS4_N_VOCAB);
        ok = next >= 0; predictions[t] = (uint32_t)next;
    }
    uint32_t accepted = ok ? 1u : 0u;
    /* EOS remains unconsumed for the caller: interactive generation stops,
     * whereas the fixed-length benchmark selects its next non-EOS root. */
    while (ok && accepted < rows && tokens[accepted] != (uint32_t)c->eos &&
        tokens[accepted] == predictions[accepted - 1u]) ++accepted;
    if (profile) {
        const double now = glm5_native_wall_sec();
        select_sec = now - phase_start; phase_start = now;
    }
    if (!ds4_tp_native_agree(e->tp.ctx, c, 8u, accepted, predictions, ok, err, errlen)) goto failed;
    if (profile) {
        const double now = glm5_native_wall_sec();
        agree_sec += now - phase_start; phase_start = now;
    }
    ok = ds4_glm5_next_draft_target_hidden(x, batch, hc[1], normalized) &&
        ds4_glm5_next_draft_refresh(x, &s->glm5_native_state, s->glm5_native_ws,
            normalized, tokens, c->prefix, rows, accepted, s->glm5_native_previous) &&
        ds4_glm5_next_target_verify_finish(x, &s->glm5_next_state, accepted) &&
        ds4_gpu_tensor_copy(s->glm5_next_cur, 0, hc[1], (accepted - 1u) * row * 4u, row * 4u) &&
        ds4_gpu_tensor_copy(s->glm5_next_logits, 0, logits, (accepted - 1u) * vocab_row, vocab_row) &&
        ds4_gpu_synchronize();
    if (profile) {
        const double now = glm5_native_wall_sec();
        commit_sec = now - phase_start; phase_start = now;
    }
    if (!ds4_tp_native_agree(e->tp.ctx, c, 9u, accepted, predictions, ok, err, errlen)) goto failed;
    if (profile) agree_sec += glm5_native_wall_sec() - phase_start;
    /* Publication has no allocation or device work and follows both ranks'
     * successful target commit AND native teacher refresh. */
    memcpy(s->logits, s->glm5_native_host_logits + (uint64_t)(accepted - 1u) * DS4_N_VOCAB, vocab_row);
    for (uint32_t t = 0; t < accepted; ++t) {
        token_vec_push(&s->checkpoint, (int)tokens[t]);
        if (accepted_out) accepted_out[t] = (int)tokens[t];
    }
    s->checkpoint_valid = true;
    ds4_gpu_tensor_free(normalized); ds4_gpu_tensor_free(logits);
    ds4_gpu_tensor_free(hc[0]); ds4_gpu_tensor_free(hc[1]);
    if (profile) fprintf(stderr, "NATIVE_PHASE rank=%u cycle=%llu prefix=%u rows=%u accepted=%u draft_ms=%.6f verify_ms=%.6f select_ms=%.6f commit_ms=%.6f agree_ms=%.6f total_ms=%.6f\n",
        x->tp_rank, (unsigned long long)c->cycle, c->prefix, rows, accepted,
        draft_sec * 1000.0, verify_sec * 1000.0, select_sec * 1000.0,
        commit_sec * 1000.0, agree_sec * 1000.0,
        (glm5_native_wall_sec() - cycle_start) * 1000.0);
    return (int)accepted;
failed:
    ds4_gpu_synchronize();
    ds4_gpu_tensor_free(normalized); ds4_gpu_tensor_free(logits);
    ds4_gpu_tensor_free(hc[0]); ds4_gpu_tensor_free(hc[1]);
    ds4_tp_mark_failed(e->tp.ctx);
    ds4_session_invalidate(s);
    return -1;
}
