# GLM-5 Resident KDA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the first correct, cache-free, single-slot resident KDA runtime component for text-only GLM-5.3-Flash Q4_K TP=2 on gfx1151 without enabling the incomplete full GLM graph.

**Architecture:** Keep strict GGUF binding in `ds4.c`, place KDA state ownership and layer orchestration in a focused `ds4_glm5_kda` module, and expose only the two approved tensor-resident ROCm operations through `ds4_gpu.h`. Each TP rank replicates KDA state and arithmetic; exact TP hello negotiation and per-layer digests prove lockstep without adding KDA network traffic.

**Tech Stack:** C11, HIP/ROCm 7.14 targeting gfx1151 wave32, DS4 tensor API, GGUF mmap weights, Python FP64 reference oracles, Make, mandatory RDMA TP controls.

**Spec:** `docs/superpowers/specs/2026-08-27-glm5-resident-kda-design.md`

## Global Constraints

- Work only on `research/glm5-next-tp2` in `/home/wkljohn/Desktop/cc/ds4-glm5-next-tp2`; preserve unrelated dirty state.
- Set `DS4_RESEARCH_ROOT=/home/wkljohn/Desktop/cc/research-results`; never create a worktree-local `research-results` path.
- Use the validated ROCm 7.14 toolchain and compile gfx1151 with wave32 assumptions explicitly tested.
- Derive the KDA layer schedule from validated GGUF tensor presence; do not hard-code a modulo schedule or a count of 34 in runtime allocation.
- Support one active sequence slot initially; allocation must print exact bytes before allocating.
- Keep q/k/v convolution histories and recurrent state FP32 and resident; add no persistent converted-weight cache.
- Replicate KDA state and computation on both TP ranks; add no KDA collective or wire traffic.
- Missing tensors, unsupported shapes, allocation failure, rank feature mismatch, operation failure, or digest mismatch must fail closed.
- A failed KDA forward invalidates the slot; continuation requires an explicit reset.
- Decode and prefill use the same sequential recurrence and state contract.
- Do not enable the full `glm5-next` graph, MTP, vision, speculative mutation, provider fallback, or a chunk-parallel recurrence in this plan.
- Existing DeepSeek V4 Q4_K and Q2_K paths must remain byte-for-byte unreachable from GLM-specific dispatch and pass their established controls.
- No performance claim or promotion is permitted until the stage's correctness gates pass; candidate builds may not contain profiling or dump instrumentation.

---

## File Structure

- Create `ds4_glm5_kda.h`: fixed GLM-5 dimensions, schedule descriptors, resident slot/workspace types, and the narrow component API used by `ds4.c`.
- Create `ds4_glm5_kda.c`: overflow-safe size calculation, schedule validation, resident allocation/reset/invalidation/free, and ordered layer-adapter calls.
- Create `rocm/ds4_rocm_glm5_kda.cuh`: production conv4, preparation, recurrent wave32, gated normalization, and digest kernels plus checked launch helpers.
- Modify `ds4_gpu.h`: declare only the two approved public GLM-5 tensor operations.
- Modify `ds4_rocm_compat.cu`: validate opaque tensor bounds/device/contiguity and dispatch the production GLM-5 kernels.
- Modify `ds4.c`: translate bound GGUF tensors into KDA descriptors, allocate one slot, expose test hooks, and keep normal execution fail-closed after the isolated component gate.
- Modify `ds4_tp.h`, `ds4_tp.c`, and `tests/test_tp_hello.c`: negotiate the resident-KDA capability exactly.
- Extend `tests/test_glm5_next_oracles.py`: independent FP64 sequence, continuation, adversarial-gate, and chunk-equivalence references.
- Create `tests/test_glm5_kda_state.c`: CPU-testable schedule, overflow, lifecycle, failure-invalidation, and memory-accounting gates with a fake GPU allocator.
- Replace the reference-only coverage in `tests/test_rocm_glm5_conv_ref.cu` and `tests/test_rocm_glm5_kda_ref.cu` with production-API component tests.
- Create `tests/test_rocm_glm5_kda_layer.cu`: one complete real-GGUF KDA-layer, continuation, digest, and prefill-to-decode handoff gate.
- Modify `Makefile`: add focused targets and link `ds4_glm5_kda.o` only where required.
- Create `docs/GLM5-KDA.md`: supported scope, memory, reset/failure contract, commands, and explicit exclusions.

---

### Task 1: Strengthen the Independent Sequential Oracles

**Files:**
- Modify: `tests/test_glm5_next_oracles.py`
- Modify: `scripts/probe-glm5-next-kda-payload.py`
- Test: `tests/test_glm5_next_oracles.py`

**Interfaces:**
- Consumes: existing `kda_reference(state, q, k, v, beta, gate)` and the real GGUF parser used by `probe-glm5-next-kda-payload.py`.
- Produces: `conv4_sequence(history, values, weights)`, `kda_sequence(state, q, k, v, beta, gate)`, `max_abs_rel(reference, candidate)`, and a deterministic JSON payload containing outputs, final state norms, and hashes.

- [ ] **Step 1: Write failing chunk/continuation and adversarial-gate tests**

Add tests that split the same deterministic 129-token input as `[129]`, `[1, 128]`, `[2, 127]`, `[3, 126]`, and `[127, 1, 1]`; assert identical history and bounded output/state error. Add recurrence cases with `gate` values `-1e-7`, `-5.0`, mixed values, and a nonzero state. The core assertions are:

```python
for chunks in ([129], [1, 128], [2, 127], [3, 126], [127, 1, 1]):
    got_out, got_hist = run_conv_chunks(x, w, initial_hist, chunks)
    abs_err, rel_err = max_abs_rel(one_call_out, got_out)
    assert abs_err <= 1e-12 and rel_err <= 1e-12
    assert got_hist == one_call_hist

for gate in ([-1e-7] * 8, [-5.0] * 8,
             [-1e-7, -5.0, -0.2, -3.0, -0.9, -4.0, -0.01, -2.0]):
    out, final = kda_sequence(nonzero_state(), q, k, v, beta, gate)
    assert all(math.isfinite(x) for row in final for x in row)
    assert all(math.isfinite(x) for row in out for x in row)
```

- [ ] **Step 2: Run the oracle gate and confirm the new symbols fail**

Run:

```bash
python3 tests/test_glm5_next_oracles.py
```

Expected: FAIL with `NameError` for `run_conv_chunks`, `max_abs_rel`, or `kda_sequence`.

- [ ] **Step 3: Implement the independent FP64 helpers**

Use Python `float` arithmetic and explicit loops. Apply decay before prediction, update state only after the complete prediction vector exists, and return deep-copied final state. `max_abs_rel` must use `max(abs(reference), 1e-30)` as the relative denominator.

- [ ] **Step 4: Emit a content-addressed real-weight oracle payload**

Extend the probe output with these exact keys:

```json
{
  "model_sha256": "the 64-character lowercase SHA-256 of the complete GGUF",
  "layer": 0,
  "tokens": 2,
  "output_f32_sha256": "SHA-256 of the 4096*2 little-endian FP32 output bytes",
  "history_f32_sha256": "SHA-256 of concatenated q/k/v final-history bytes",
  "state_f32_sha256": "SHA-256 of the final recurrent-state bytes",
  "output_l2": 0.0,
  "state_l2": 0.0
}
```

Write the payload only to the path supplied by `--output`; do not default into the source tree. The caller will pass a path below `$DS4_RESEARCH_ROOT/glm5-next-tp2/`.

- [ ] **Step 5: Run the CPU and real-GGUF gates**

Run:

```bash
export DS4_RESEARCH_ROOT=/home/wkljohn/Desktop/cc/research-results
python3 tests/test_glm5_next_oracles.py
python3 scripts/probe-glm5-next-kda-payload.py \
  --output "$DS4_RESEARCH_ROOT/glm5-next-tp2/kda-layer0-oracle.json" \
  /home/wkljohn/Desktop/cc/models/antirez-glm-5.3-flash-gguf/GLM-5.3-Flash-Q4_K.gguf
```

Expected: all CPU oracle tests print `PASS`; the probe prints its existing payload pass plus the three hashes and two finite norms.

- [ ] **Step 6: Commit the oracle contract**

```bash
git add tests/test_glm5_next_oracles.py scripts/probe-glm5-next-kda-payload.py
git commit -m "tests: strengthen GLM5 KDA sequence oracles"
```

### Task 2: Add Overflow-Safe Resident State Ownership

**Files:**
- Create: `ds4_glm5_kda.h`
- Create: `ds4_glm5_kda.c`
- Create: `tests/test_glm5_kda_state.c`
- Modify: `Makefile`

**Interfaces:**
- Consumes: `ds4_gpu_tensor_alloc()`, `ds4_gpu_tensor_fill_f32()`, and `ds4_gpu_tensor_free()` from `ds4_gpu.h`.
- Produces:

```c
typedef struct {
    uint32_t layer;
    bool is_kda;
} ds4_glm5_layer_kind;

typedef struct {
    ds4_gpu_tensor *q_history;
    ds4_gpu_tensor *k_history;
    ds4_gpu_tensor *v_history;
    ds4_gpu_tensor *recurrent;
    uint64_t token_count;
    bool valid;
} ds4_glm5_kda_layer_state;

typedef struct {
    ds4_glm5_kda_layer_state *layer;
    uint32_t layer_count;
    uint32_t kda_count;
    uint64_t bytes;
    bool valid;
} ds4_glm5_kda_slot;

int ds4_glm5_kda_slot_init(ds4_glm5_kda_slot *slot,
                           const ds4_glm5_layer_kind *schedule,
                           uint32_t layer_count,
                           uint32_t slot_count,
                           FILE *accounting);
int ds4_glm5_kda_slot_reset(ds4_glm5_kda_slot *slot);
void ds4_glm5_kda_slot_invalidate(ds4_glm5_kda_slot *slot);
void ds4_glm5_kda_slot_free(ds4_glm5_kda_slot *slot);
```

- [ ] **Step 1: Write a fake-allocator lifecycle test**

Test schedules with zero, one, and 34 KDA layers; assert one layer consumes exactly `4,489,216` bytes, a 34-layer slot consumes `152,633,344` bytes, the accounting line is printed before the first allocation, reset zeroes all four tensors, and an injected allocation failure frees earlier tensors and leaves the slot invalid.

- [ ] **Step 2: Add explicit overflow and slot-count failures**

Call `ds4_glm5_kda_slot_init` with `slot_count == 0`, `slot_count == 2`, and a synthetic layer count whose multiplication would overflow `uint64_t`. Expected result is false with no allocation. The one-slot baseline must reject two slots rather than silently multiplying memory.

- [ ] **Step 3: Run the focused test and confirm it fails to link**

Run:

```bash
make tests/test_glm5_kda_state
```

Expected: FAIL because `ds4_glm5_kda_slot_init` and lifecycle functions are undefined.

- [ ] **Step 4: Implement the minimal state owner**

Define constants in the header:

```c
enum {
    DS4_GLM5_KDA_CHANNELS = 8192,
    DS4_GLM5_KDA_HISTORY = 3,
    DS4_GLM5_KDA_HEADS = 64,
    DS4_GLM5_KDA_HEAD_DIM = 128,
    DS4_GLM5_KDA_MAX_SLOTS = 1,
};
```

Compute every product with a helper that rejects `a != 0 && b > UINT64_MAX / a`. Print:

```text
ds4: GLM5 KDA resident state: slots=1 layers=%u bytes=%llu MiB=%.2f
```

before allocating. Allocate q/k/v histories and recurrence separately per bound KDA layer so cleanup after partial failure is deterministic. `reset` fills every tensor with `0.0f`, then sets every KDA layer token count to zero and valid=true; `invalidate` sets both layer and slot validity false without mutating memory.

- [ ] **Step 5: Run state tests and sanitizers**

Run:

```bash
make tests/test_glm5_kda_state
./tests/test_glm5_kda_state
CC=clang CFLAGS='-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer' \
  make -B tests/test_glm5_kda_state
ASAN_OPTIONS=detect_leaks=1 ./tests/test_glm5_kda_state
```

Expected: both runs PASS with no sanitizer finding.

- [ ] **Step 6: Commit resident ownership**

```bash
git add ds4_glm5_kda.h ds4_glm5_kda.c tests/test_glm5_kda_state.c Makefile
git commit -m "glm5-next: add resident KDA state ownership"
```

### Task 3: Promote Causal Conv4 Behind the Approved GPU API

**Files:**
- Modify: `ds4_gpu.h`
- Create: `rocm/ds4_rocm_glm5_kda.cuh`
- Modify: `ds4_rocm_compat.cu`
- Modify: `tests/test_rocm_glm5_conv_ref.cu`
- Modify: `Makefile`

**Interfaces:**
- Consumes: contiguous F32 tensors and the fixed `[token][8192]`, `[8192][4]`, `[8192][3]` layouts.
- Produces:

```c
int ds4_gpu_glm5_causal_conv4_tensor(
    ds4_gpu_tensor *out,
    ds4_gpu_tensor *history,
    const ds4_gpu_tensor *input,
    const ds4_gpu_tensor *weight,
    uint32_t n_tokens,
    uint32_t channels);
```

- [ ] **Step 1: Rewrite the test to call the public API**

Use full production width `channels=8192`. For lengths `1, 2, 3, 127, 128, 129`, compare every output and all `8192*3` history values to an independent host implementation. Run both from zero history and from a deterministic nonzero continuation history.

- [ ] **Step 2: Add fail-closed shape/bounds cases**

Assert failure for `channels != 8192`, `n_tokens == 0`, undersized output/input/history/weight, null tensors, and aliased `out == history`. Ensure each rejection occurs before any launch and leaves history unchanged.

- [ ] **Step 3: Build and confirm the public symbol is missing**

Run:

```bash
make -B tests/test_rocm_glm5_conv_ref
```

Expected: compile or link failure naming `ds4_gpu_glm5_causal_conv4_tensor`.

- [ ] **Step 4: Implement production validation and launch**

Move the kernel into `rocm/ds4_rocm_glm5_kda.cuh`; use 256 threads and `(channels + 255) / 256` blocks. In `ds4_rocm_compat.cu`, reject multiplication overflow and require exact minimum bytes:

```c
out:     n_tokens * channels * sizeof(float)
input:   n_tokens * channels * sizeof(float)
weight:  channels * 4 * sizeof(float)
history: channels * 3 * sizeof(float)
```

The kernel keeps the three history values in registers, performs four FP32 multiply-adds, applies SiLU, and writes the final history only after all tokens for that channel finish.

- [ ] **Step 5: Run component and legacy smoke gates**

Run:

```bash
make -B test-rocm-glm5-conv-ref test-tp-hello
make DS4_GLM5_MODEL=/home/wkljohn/Desktop/cc/models/antirez-glm-5.3-flash-gguf/GLM-5.3-Flash-Q4_K.gguf test-glm5-next-contract
```

Expected: production conv gate passes every length; existing TP hello and GLM contract remain green.

- [ ] **Step 6: Commit production conv4**

```bash
git add ds4_gpu.h ds4_rocm_compat.cu rocm/ds4_rocm_glm5_kda.cuh \
  tests/test_rocm_glm5_conv_ref.cu Makefile
git commit -m "rocm: add checked GLM5 causal conv4 operation"
```

### Task 4: Promote the Wave32 Sequential KDA Recurrence

**Files:**
- Modify: `ds4_gpu.h`
- Modify: `rocm/ds4_rocm_glm5_kda.cuh`
- Modify: `ds4_rocm_compat.cu`
- Modify: `tests/test_rocm_glm5_kda_ref.cu`

**Interfaces:**
- Consumes: F32 `q/k/v/gate` `[token][64][128]`, F32 `beta` `[token][64]`, and F32 state `[64][128][128]`.
- Produces:

```c
int ds4_gpu_glm5_kda_recurrent_tensor(
    ds4_gpu_tensor *out,
    ds4_gpu_tensor *state,
    const ds4_gpu_tensor *q,
    const ds4_gpu_tensor *k,
    const ds4_gpu_tensor *v,
    const ds4_gpu_tensor *gate,
    const ds4_gpu_tensor *beta,
    uint32_t n_tokens,
    uint32_t n_heads,
    uint32_t head_dim);
```

- [ ] **Step 1: Convert the test to the public API and adversarial matrix**

Test tokens `1, 2, 3, 127, 128, 129`, both zero and nonzero state, beta near 0 and 1, and channel-wise gates near 0, at -5, and mixed. Compare output and final state to the host sequential oracle; report maximum absolute error, maximum relative error, reference/candidate state L2 norms, and drift.

- [ ] **Step 2: Add chunk-equivalence calls on the same device state**

Run the same 129 tokens as one call and the five chunkings from Task 1. Require:

```c
max_abs_output <= 3.0e-6f
max_abs_state  <= 3.0e-6f
relative_l2_state_drift <= 1.0e-6
```

These bounds admit expected wave-reduction order while rejecting recurrence-order changes.

- [ ] **Step 3: Add shape and alias rejection tests**

Reject `n_heads != 64`, `head_dim != 128`, zero tokens, undersized tensors, `out == state`, and any input aliasing mutable state. Verify rejected calls do not change state.

- [ ] **Step 4: Run and confirm the API is absent**

Run:

```bash
make -B tests/test_rocm_glm5_kda_ref
```

Expected: compile or link failure for `ds4_gpu_glm5_kda_recurrent_tensor`.

- [ ] **Step 5: Implement the checked wave32 launcher**

Use four 32-lane waves per block. Each lane owns key rows `lane`, `lane+32`, `lane+64`, `lane+96`; each wave owns one value column and `grid.z=32` covers all 128 value columns. Apply `expf(gate) * state` before the key prediction, broadcast the wave sum from lane zero, apply the delta, then reduce `q^T state`. Add `static_assert(warpSize == 32)` under gfx1151 compilation and a runtime device-property rejection otherwise.

- [ ] **Step 6: Run production recurrence and cross-component gates**

Run:

```bash
make -B test-rocm-glm5-kda-ref test-rocm-glm5-conv-ref
make DS4_GLM5_MODEL=/home/wkljohn/Desktop/cc/models/antirez-glm-5.3-flash-gguf/GLM-5.3-Flash-Q4_K.gguf \
  test-glm5-next-contract test-glm5-next-kda-projection
```

Expected: all commands PASS and include bounded output/state drift for every chunking.

- [ ] **Step 7: Commit the recurrence API**

```bash
git add ds4_gpu.h ds4_rocm_compat.cu rocm/ds4_rocm_glm5_kda.cuh \
  tests/test_rocm_glm5_kda_ref.cu
git commit -m "rocm: add checked GLM5 wave32 KDA recurrence"
```

### Task 5: Bind the GGUF-Derived Schedule and Negotiate TP Capability

**Files:**
- Modify: `ds4.c`
- Modify: `ds4_tp.h`
- Modify: `tests/test_tp_hello.c`
- Create: `tests/test_glm5_kda_binding.c`
- Modify: `Makefile`

**Interfaces:**
- Consumes: `ds4_glm5_next_weights`, `model_find_tensor()`, and the state API from Task 2.
- Produces:

```c
#define DS4_TP_FEATURE_GLM5_RESIDENT_KDA (UINT32_C(1) << 20)

int ds4_glm5_kda_build_schedule(ds4_glm5_layer_kind *out,
                                uint32_t capacity,
                                const bool *has_kda_q,
                                const bool *has_mla_q,
                                uint32_t layer_count,
                                uint32_t *kda_count);
```

- [ ] **Step 1: Add schedule tests independent of the model filename**

Test the real 46-entry tensor-presence pattern, an all-KDA synthetic pattern, a layer with both KDA and MLA tensors, and a layer with neither. The latter two must fail. Change the real schedule's layer numbers while preserving presence to prove no modulo assumption is used.

- [ ] **Step 2: Add exact TP feature tests**

Add equal-enabled and mismatched cases for bit 20. The mismatch must return:

```text
tp hello: runtime feature mismatch (local=0x00100000 peer=0x00000000)
```

Also assert the bit does not overlap `DS4_TP_FEATURE_EXPERT_SPLIT_MASK` or any prior feature.

- [ ] **Step 3: Run and verify failures**

Run:

```bash
make -B tests/test_glm5_kda_binding tests/test_tp_hello
```

Expected: binding test fails to link and TP feature tests fail to compile before implementation.

- [ ] **Step 4: Build the schedule from bound tensor presence**

After `glm5_next_weights_bind`, set each layer's booleans from `kda_q != NULL` and `mla_q_a != NULL`; require exactly one attention family per layer. Store the derived schedule and count. Do not compare the result to 34 in runtime code; the model-contract test may continue recording 34 for this specific GGUF.

- [ ] **Step 5: Advertise resident KDA only for the validated ROCm path**

Set bit 20 only when architecture is `glm5-next`, ROCm is active, the schedule is valid, and every KDA tensor layout has passed strict validation. Both ranks use the existing exact feature-word equality check; no provider-specific condition may affect the bit.

- [ ] **Step 6: Run binding and negotiation tests**

Run:

```bash
make -B test-tp-hello tests/test_glm5_kda_binding
./tests/test_glm5_kda_binding
```

Expected: schedule and feature tests PASS, including malformed schedules and mismatch refusal.

- [ ] **Step 7: Commit binding and negotiation**

```bash
git add ds4.c ds4_tp.h tests/test_tp_hello.c tests/test_glm5_kda_binding.c Makefile
git commit -m "glm5-next: negotiate GGUF-derived resident KDA"
```

### Task 6: Execute One Complete Real-GGUF KDA Layer

**Files:**
- Modify: `ds4_glm5_kda.h`
- Modify: `ds4_glm5_kda.c`
- Modify: `rocm/ds4_rocm_glm5_kda.cuh`
- Modify: `ds4_rocm_compat.cu`
- Modify: `ds4.c`
- Create: `tests/test_rocm_glm5_kda_layer.cu`
- Modify: `Makefile`

**Interfaces:**
- Consumes: BF16 matrix multiplication, weighted RMSNorm, the two public APIs, bound weight offsets, model mmap, one resident layer state, and F32 scratch tensors.
- Produces:

```c
typedef struct {
    uint64_t attn_norm, q, k, v, output;
    uint64_t q_conv, k_conv, v_conv;
    uint64_t f_a, f_b, g_a, g_b, beta, o_norm, dt_bias, a_log;
} ds4_glm5_kda_weight_offsets;

typedef struct {
    ds4_gpu_tensor *norm, *q, *k, *v, *f_low, *forget;
    ds4_gpu_tensor *g_low, *out_gate, *beta, *recurrent_out, *flat;
    uint32_t capacity_tokens;
} ds4_glm5_kda_workspace;

typedef struct {
    const ds4_glm5_kda_weight_offsets *weights;
    const void *model_map;
    uint64_t model_size;
    ds4_glm5_kda_layer_state *state;
    ds4_glm5_kda_workspace *workspace;
    const ds4_gpu_tensor *input;
    ds4_gpu_tensor *output;
    uint32_t n_tokens;
} ds4_glm5_kda_device_args;

int ds4_glm5_kda_layer_forward(ds4_glm5_kda_layer_state *state,
                               ds4_glm5_kda_workspace *workspace,
                               const ds4_glm5_kda_weight_offsets *weights,
                               const void *model_map,
                               uint64_t model_size,
                               const ds4_gpu_tensor *input,
                               ds4_gpu_tensor *output,
                               uint32_t n_tokens);

/* Internal ROCm adapter declared in ds4_glm5_kda.h, not ds4_gpu.h. */
int ds4_rocm_glm5_kda_layer_execute(
    const ds4_glm5_kda_device_args *args);
```

- [ ] **Step 1: Write the end-to-end layer test against the oracle payload**

Load layer 0 from the documented Q4_K GGUF, allocate a deterministic two-token 4096-wide input, execute the complete adapter, and compare all 8192 output floats, the three histories, recurrent state, and token count to the Task 1 payload. Record FP32 hashes plus maximum absolute/relative errors; require finite outputs and matching token count.

- [ ] **Step 2: Add failure-invalidation cases**

Inject failure after each stage—input norm, q/k/v projection, each conv, gate preparation, recurrence, gated output norm, and output projection—through `DS4_GLM5_KDA_TEST_HOOKS`. Each case must return failure and set `state->valid=false`; a subsequent forward must fail until `ds4_glm5_kda_slot_reset` succeeds.

- [ ] **Step 3: Run the test and confirm missing adapter symbols**

Run:

```bash
make -B tests/test_rocm_glm5_kda_layer
```

Expected: link failure for `ds4_glm5_kda_layer_forward`.

- [ ] **Step 4: Implement fixed-shape preparation kernels**

In the ROCm-internal header, add launch helpers that do only arithmetic not represented by existing DS4 primitives:

```text
q: per-head FP32 L2 normalize, then multiply by 1/sqrt(128)
k: per-head FP32 L2 normalize
forget: -5 * sigmoid(exp(a_log[head]) * (f_b + dt_bias)[channel])
beta: sigmoid(beta_projection[head])
output: RMSNorm(recurrent_out, eps=1e-6, weight=o_norm) * sigmoid(out_gate)
```

Keep these helpers internal to `ds4_rocm_compat.cu`; the only public GPU operations remain the two approved functions. `ds4_rocm_glm5_kda_layer_execute` resolves the F32 convolution ranges through the existing model-range resolver and wraps those already-resident addresses in non-owning tensor descriptors before calling the public conv operation; it must neither duplicate nor convert the convolution weights. Use one kernel per logically testable transform initially; fusion is a later performance lane.

- [ ] **Step 5: Implement ordered layer orchestration**

The C lifecycle wrapper calls `ds4_rocm_glm5_kda_layer_execute`; the ROCm adapter must use this exact sequence and stop at the first failure:

```c
rms_norm(input) -> BF16 q/k/v projections -> q/k/v conv4
-> q/k normalization -> BF16 f_a/f_b and beta projections
-> forget/beta transform -> KDA recurrence
-> BF16 g_a/g_b -> gated weighted RMSNorm
-> BF16 output projection
```

Update `token_count` only after output projection succeeds. Do not restore partially modified GPU state on error; mark invalid and require reset.

- [ ] **Step 6: Run real-layer and component gates**

Run:

```bash
export DS4_GLM5_MODEL=/home/wkljohn/Desktop/cc/models/antirez-glm-5.3-flash-gguf/GLM-5.3-Flash-Q4_K.gguf
make -B test-rocm-glm5-conv-ref test-rocm-glm5-kda-ref \
  test-glm5-next-contract test-glm5-next-kda-projection \
  test-rocm-glm5-kda-layer
```

Expected: all gates PASS; the real-layer report includes output/history/state hashes and bounded errors.

- [ ] **Step 7: Commit the complete layer adapter**

```bash
git add ds4_glm5_kda.h ds4_glm5_kda.c rocm/ds4_rocm_glm5_kda.cuh \
  ds4_rocm_compat.cu ds4.c tests/test_rocm_glm5_kda_layer.cu Makefile
git commit -m "glm5-next: execute one resident KDA layer"
```

### Task 7: Validate Prefill/Decode Handoff and TP Replication

**Files:**
- Modify: `ds4_glm5_kda.h`
- Modify: `ds4_glm5_kda.c`
- Modify: `tests/test_rocm_glm5_kda_layer.cu`
- Create: `tests/test_glm5_kda_tp_digest.c`
- Modify: `Makefile`

**Interfaces:**
- Consumes: the complete layer adapter and resident state from Tasks 2 and 6.
- Produces:

```c
typedef struct {
    uint64_t output_fnv64;
    uint64_t q_history_fnv64;
    uint64_t k_history_fnv64;
    uint64_t v_history_fnv64;
    uint64_t recurrent_fnv64;
    uint64_t token_count;
} ds4_glm5_kda_digest;

int ds4_glm5_kda_layer_digest(const ds4_glm5_kda_layer_state *state,
                              const ds4_gpu_tensor *output,
                              uint64_t output_floats,
                              ds4_glm5_kda_digest *digest);
int ds4_glm5_kda_digest_equal(const ds4_glm5_kda_digest *rank0,
                              const ds4_glm5_kda_digest *rank1);
```

- [ ] **Step 1: Add handoff equivalence tests**

For `N=1, 2, 3, 127, 128, 129, 2048`, compare `prefill(N) + decode(1)` with tokenwise `N+1`. Record output error, state error, history error, and state-norm drift. Use the same numerical bounds as Task 4.

- [ ] **Step 2: Add deterministic rank-digest tests**

Run two independently allocated states from identical inputs and weights; compare all six fields after each layer call. Flip one byte in each state component in turn and require `ds4_glm5_kda_digest_equal` to reject it. Token-count mismatch must also reject.

- [ ] **Step 3: Add a no-transport assertion**

Under `DS4_TP_TEST_HOOKS`, count calls to TP gate/exchange functions during KDA layer execution. Require the count to remain zero while rank digests match. This proves replication rather than an accidental collective.

- [ ] **Step 4: Run and confirm digest symbols are absent**

Run:

```bash
make -B tests/test_glm5_kda_tp_digest test-rocm-glm5-kda-layer
```

Expected: compile/link failure for the digest functions.

- [ ] **Step 5: Implement digest and commit-on-success semantics**

Compute FNV-1a over raw FP32 bytes read only after the normal command completion point. Include token count as a separately compared scalar. In production, digest calculation is enabled only by the correctness harness; it must not add readback to ordinary inference.

- [ ] **Step 6: Run handoff, digest, and no-transport gates**

Run:

```bash
make -B test-rocm-glm5-kda-layer tests/test_glm5_kda_tp_digest
./tests/test_glm5_kda_tp_digest
```

Expected: all handoff lengths and corruption cases PASS; KDA transport calls report zero.

- [ ] **Step 7: Commit handoff and TP determinism**

```bash
git add ds4_glm5_kda.h ds4_glm5_kda.c tests/test_rocm_glm5_kda_layer.cu \
  tests/test_glm5_kda_tp_digest.c Makefile
git commit -m "tests: gate GLM5 KDA handoff and TP determinism"
```

### Task 8: Integrate the Isolated Runtime Gate Without Opening the Full Graph

**Files:**
- Modify: `ds4.c`
- Modify: `Makefile`
- Create: `docs/GLM5-KDA.md`
- Modify: `scripts/check-glm5-next-gguf.py`

**Interfaces:**
- Consumes: validated binding, resident allocator, complete layer adapter, TP feature, and all component tests.
- Produces: `make test-glm5-resident-kda`, which validates the component and then proves normal GLM execution remains fail-closed at the next unimplemented graph boundary.

- [ ] **Step 1: Add the aggregate test target first**

Define:

```make
test-glm5-resident-kda: test-glm5-next-contract \
        test-glm5-next-kda-projection \
        test-rocm-glm5-conv-ref \
        test-rocm-glm5-kda-ref \
        test-rocm-glm5-kda-layer \
        test-tp-hello
	./tests/test_glm5_kda_state
	./tests/test_glm5_kda_binding
	./tests/test_glm5_kda_tp_digest
```

Run it before adding the remaining integration and record the expected failure at the newly missing fail-closed assertion.

- [ ] **Step 2: Allocate one resident slot only after strict model validation**

During the isolated GLM component setup, bind and validate all layouts, derive the schedule, print memory accounting, allocate one slot, reset it, run the documented one-layer gate, and free it on every exit path. Do not attach this temporary component harness to server sessions yet.

- [ ] **Step 3: Preserve an explicit incomplete-graph refusal**

After the isolated KDA component succeeds, terminate with this message:

```text
ds4: glm5-next resident KDA component validated; sparse MLA/indexer, mHC, FFN, output, and full TP graph remain unimplemented; refusing inference
```

Add a subprocess test that requires nonzero exit and this exact message. This prevents component progress from accidentally advertising usable GLM inference.

- [ ] **Step 4: Document user-visible scope and memory**

In `docs/GLM5-KDA.md`, state that the branch validates one complete real-GGUF KDA layer but does not serve GLM-5.3 yet. Include the 145.56 MiB single-slot state figure for the currently validated 34-layer GGUF, explain that runtime derives the count, document the zero persistent weight-cache rule, list the aggregate test command, and list the remaining graph stages.

- [ ] **Step 5: Run the complete component gate**

Run:

```bash
export DS4_RESEARCH_ROOT=/home/wkljohn/Desktop/cc/research-results
export DS4_GLM5_MODEL=/home/wkljohn/Desktop/cc/models/antirez-glm-5.3-flash-gguf/GLM-5.3-Flash-Q4_K.gguf
make -B test-glm5-resident-kda
```

Expected: every component gate passes and the runtime-refusal subprocess confirms the full graph remains unavailable.

- [ ] **Step 6: Run unchanged DeepSeek regressions with the same binary**

Use the established source-versioned controls and write evidence only beneath `$DS4_RESEARCH_ROOT`:

```bash
make -B strix-halo test-quality-gates test-tp-hello
./scripts/pre-main-tp-smoke.sh \
  /absolute/path/to/validated/DeepSeek-V4-Flash-Q4_K.gguf \
  /absolute/path/to/validated/DeepSeek-V4-Flash-Q2_K.gguf
```

Expected: Q4_K and Q2_K gates pass their current promoted arithmetic contracts on mandatory RDMA; logs contain no GLM5 KDA feature or kernel dispatch.

- [ ] **Step 7: Record hardware proof when both nodes are reachable**

On each independent filesystem, verify the same commit/binary/model hashes, approximately 124 GiB visible gfx1151 pool, and mandatory RDMA. Run the component gate once with OdinLink and once with RoCE v2 solely to prove identical capability negotiation and no KDA transport. If nodes remain unreachable, record that as unproven and do not claim hardware completion.

- [ ] **Step 8: Commit the isolated stage**

```bash
git add ds4.c Makefile docs/GLM5-KDA.md scripts/check-glm5-next-gguf.py
git commit -m "glm5-next: gate resident KDA component"
```

### Task 9: Final Review, Evidence, and Branch Handoff

**Files:**
- Modify only if verification finds a defect: files already listed in Tasks 1-8.
- Research evidence: `/home/wkljohn/Desktop/cc/research-results/glm5-next-tp2/` only.

**Interfaces:**
- Consumes: all prior task commits and workspace gate policy.
- Produces: a clean reviewed research branch plus content-addressed local evidence; no main merge and no full-model performance claim.

- [ ] **Step 1: Classify the work under the promotion policy**

Read `/home/wkljohn/Desktop/cc/research-results/policies/GATE-PROMOTION.md`, record the lane in the dossier, and bind evidence to branch, commit, binary SHA-256, model SHA-256, ROCm version, gfx1151 target, and test commands.

- [ ] **Step 2: Run a fresh complete verification from a clean build**

```bash
make clean
export DS4_RESEARCH_ROOT=/home/wkljohn/Desktop/cc/research-results
export DS4_GLM5_MODEL=/home/wkljohn/Desktop/cc/models/antirez-glm-5.3-flash-gguf/GLM-5.3-Flash-Q4_K.gguf
make -j"$(nproc)" strix-halo
make test-glm5-resident-kda test-quality-gates
git status --short
```

Expected: build and gates PASS; only intentional tracked changes, if any, appear in status.

- [ ] **Step 3: Perform correctness and code review**

Review every diff against the spec, emphasizing decay-before-prediction, wave32 ownership, overflow checks, state invalidation, derived scheduling, exact feature matching, no KDA transport, and no expanded cache. Obtain an Opus 5 review of the content-addressed commit and store it in the candidate dossier. Fable review is required only if the gate policy classifies this component stage as a performance candidate; if unavailable, record the missing review and do not promote.

- [ ] **Step 4: Verify repository and research-store hygiene**

```bash
test ! -e research-results
git status --short --branch
find /home/wkljohn/Desktop/cc/research-results/glm5-next-tp2 -maxdepth 2 -type f -print | sort
```

Expected: no worktree-local research store; evidence is present only in the canonical path.

- [ ] **Step 5: Commit review corrections separately**

If review found defects, fix them test-first, rerun Step 2, then stage only this feature's reviewed files and commit the corrections:

```bash
git add ds4_glm5_kda.h ds4_glm5_kda.c ds4_gpu.h ds4_rocm_compat.cu \
  rocm/ds4_rocm_glm5_kda.cuh ds4.c ds4_tp.h tests/test_glm5_next_oracles.py \
  tests/test_glm5_kda_state.c tests/test_glm5_kda_binding.c \
  tests/test_rocm_glm5_conv_ref.cu tests/test_rocm_glm5_kda_ref.cu \
  tests/test_rocm_glm5_kda_layer.cu tests/test_glm5_kda_tp_digest.c Makefile \
  docs/GLM5-KDA.md scripts/check-glm5-next-gguf.py \
  scripts/probe-glm5-next-kda-payload.py
git commit -m "fix: address resident KDA review findings"
```

If no defects were found, do not create an empty commit.

- [ ] **Step 6: Hand off the research branch**

Report the exact branch and HEAD, component gates, measured state bytes, oracle error bounds, whether OdinLink/RoCE hardware proof was available, and the explicit remaining blocker: full GLM-5.3 inference still needs sparse MLA/indexer, mHC, FFN, output, and complete TP graph stages.
