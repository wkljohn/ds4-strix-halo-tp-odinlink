# GLM5.3 ordinary-session integration boundary

This branch contains a validated staged GLM5.3 KDA/MLA executor, but the
ordinary engine remains fail-closed. The legacy `ds4_session_is_glm()` path
uses `ds4_glm_gpu_graph` and `ds4_weights`; those types cannot represent
GLM5.3 layer-0 KDA tensors and must not be reused.

## Required session-owned state

Add to `ds4_session` only for `DS4_VARIANT_GLM53`:

- `ds4_glm5_next_state` resident recurrent/KDA state;
- a capacity-matched `ds4_glm5_next_workspace`;
- current/output hidden tensors sized `tokens * DS4_N_HC * DS4_N_EMBD`;
- one vocabulary-logits tensor and an execution context pointing at
  `e->glm5_next->offsets`, the model map, and the existing TP transport/slab.

Allocate and bind these after `ds4_engine_tp_bind()` has completed, so the
negotiated feature mask, rank, RoCE/OdinLink exchange callbacks, and big-gate
views are final. Free them before the engine's TP slab and model mappings.

## Execution order

1. `ds4_session_create`: initialize state for the requested context and reject
   unsupported non-TP/partial slices explicitly.
2. `ds4_session_sync`: for a fresh prompt, call `embed_tokens`, then
   `layer_forward_batch` for every trunk layer, retaining only the final hidden
   row for `output_logits`; commit the token timeline only after all layers and
   output projection succeed.
3. `ds4_session_eval`: use the one-token embed/layer/output sequence, advance
   KDA/MLA state atomically, and exchange only the negotiated GLM5 gate slots.
4. Prefix reuse must reset/rebuild the staged state on divergence; never reuse
   the legacy GLM dense/indexed KV cache.
5. On any transport or kernel error, invalidate the whole staged state and
   refuse continuation until a fresh sync.

The first integration gate should run ordinary `ds4-server` on the same
33-token prompt used by `full-trunk-perf-20260829`, compare leader/worker token
FNV and full-logit hashes, then repeat with 121-token/32-token decode. Only
after that gate passes should `DS4_GLM5_NEXT_ENABLE_ORDINARY` cease being a
research opt-in. No DeepSeek code path should be changed by this work.
