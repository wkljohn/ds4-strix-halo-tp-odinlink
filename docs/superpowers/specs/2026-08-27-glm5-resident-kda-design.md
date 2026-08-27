# GLM-5 Resident KDA Design

## Scope

This design adds the first resident runtime component for cache-free,
text-only GLM-5.3-Flash Q4_K TP=2 inference on gfx1151: the Kimi Delta
Attention (KDA) layers. It does not enable the complete `glm5-next` graph.
Sparse MLA/indexer, mHC, dense/shared/routed FFN, output, complete TP, and
production promotion remain later independently gated stages.

The model remains fail-closed until a complete real-GGUF KDA layer passes the
validation sequence below. MTP, vision, expanded-weight caches, provider
fallbacks, and speculative state mutation are out of scope.

## Arithmetic contract

For each KDA layer, DS4 executes this order:

1. RMS-normalize the 4096-wide collapsed layer input.
2. Project BF16 `q`, `k`, and `v` from 4096 to 8192.
3. Run independent four-tap depthwise causal convolutions and SiLU on q/k/v.
4. Reshape each result to 64 heads by 128 channels.
5. L2-normalize q and k per head in FP32 with epsilon `1e-6`; scale q by
   `1/sqrt(128)` exactly once.
6. Project the channel-wise forget gate through BF16 `f_a` and `f_b`, add the
   FP32 `dt_bias`, and apply
   `-5 * sigmoid(exp(A_log[head]) * gate[channel])`.
7. Project the per-head input gate through BF16 `kda_beta` and apply sigmoid.
8. Apply decay to the FP32 recurrent state before computing the key-memory
   prediction, then apply the delta update and emit `q^T state`.
9. Project the output gate through BF16 `g_a` and `g_b`, apply gated FP32
   RMSNorm with `kda_o_norm`, flatten to 8192, and apply BF16 `kda_output` to
   return 4096 values.

The sequential decode recurrence is the initial prefill implementation too.
This avoids introducing a second arithmetic path before the baseline is
correct. A later chunked algorithm requires its own numerical-correction lane.

## Resident state

The layer schedule is derived from validated GGUF tensor presence. No runtime
code may assume that a particular count or modulo schedule is sufficient.

Each active sequence slot owns, for every KDA layer:

- Three convolution histories, one each for q/k/v, stored as
  `[8192][3]` FP32 with oldest sample first. Although the Transformers cache
  stores four samples, a four-tap convolution needs only the previous three
  plus the current projected value; the component oracle proves this state
  contract across call boundaries.
- One recurrent state stored as `[64][128][128]` FP32 in
  `[head][key_channel][value_channel]` order.
- A validity/token counter. Reset zeros both histories and recurrence state.
  A failed forward invalidates the slot; it must not be reused without reset.

Memory per sequence slot per rank is:

```text
recurrence: 64 * 128 * 128 * 4             = 4,194,304 bytes/layer
histories:  3 * 8192 * 3 * 4               =   294,912 bytes/layer
total:                                         4,489,216 bytes/layer
34 currently validated KDA layers:           152,633,344 bytes (145.56 MiB)
```

The allocator computes the count from the bound schedule and checks all
multiplications for overflow. Slot count is explicit; memory accounting must
be printed before allocation. No persistent converted-weight buffer is
allowed.

## TP ownership

The baseline replicates KDA projections, state, and output on both TP ranks.
This matches DS4's existing contract: non-expert weights are replicated while
routed-expert Q4_K weights are divided. Both ranks begin with identical input,
execute deterministic KDA arithmetic, and retain identical KDA state.

This intentionally avoids a per-layer q/k/v all-gather or KDA output
all-reduce. The TP hello feature contract must include the GLM-5 resident KDA
capability so independently launched ranks cannot select different paths.
Rank divergence is checked at layer boundaries using deterministic test
digests; a mismatch fails closed.

Head-sharded KDA is not part of the baseline. It would save approximately
72.8 MiB per slot per rank at TP=2 but require column-parallel projection and
row-parallel output composition, adding communication and a second numerical
path before correctness is established.

## GPU API boundary

The ROCm backend exposes two GLM-5-specific operations without changing the
generic tensor ABI:

```c
int ds4_gpu_glm5_causal_conv4_tensor(
    ds4_gpu_tensor *out,
    ds4_gpu_tensor *history,
    const ds4_gpu_tensor *input,
    const ds4_gpu_tensor *weight,
    uint32_t n_tokens,
    uint32_t channels);

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

Both functions validate fixed GLM-5 shapes, F32 element storage, device
identity, contiguity, and output sizes. Unsupported shapes return failure;
there is no CPU or generic-kernel fallback in a ROCm GLM-5 run.

The existing BF16 matmul, RMSNorm, elementwise, allocation, and stream
facilities are reused where their documented layout matches. A new operation
is added only when no existing DS4 primitive represents the arithmetic.

## Decode and prefill data flow

Decode and prefill call the same layer adapter and mutate the same resident
state contract. Decode supplies one token. Initial prefill may supply any
positive token count but the recurrence processes tokens sequentially in
increasing position order.

For continuation prefill, convolution history and recurrent state begin at
the prior committed token count. The adapter updates the token count only
after every KDA operation in the layer succeeds. If any operation fails, the
sequence slot is marked invalid and inference aborts rather than using partial
state.

The first implementation supports a single active slot. Multi-slot server
support is admitted only after the allocator accounts for the additional
145.56 MiB per slot and isolated reset/continuation tests pass.

## Validation gates

Each gate blocks the next:

1. **Convolution component:** Compare ROCm with an independent FP64 host
   implementation for sequence start, continuation, and chunks of lengths
   1, 2, 3, 127, 128, and 129. Confirm history after every call.
2. **Recurrence component:** Compare every output and final state with an
   independent sequential host implementation for gates near zero, gates near
   one, mixed gates, and nonzero initial state.
3. **Chunk equivalence:** For the sequential baseline, processing a token
   vector in the listed chunks must match a one-call execution within an
   explicitly recorded maximum absolute and relative error. The gate records
   state-norm drift as well as output error.
4. **Real-GGUF layer:** Execute one complete KDA layer from the documented
   Q4_K GGUF and compare all 4096 output values and resident state against an
   independent same-GGUF oracle. Record hashes and bounded errors.
5. **Prefill-to-decode handoff:** Prefill N tokens and decode one, then compare
   with tokenwise execution of N+1 for short and long N.
6. **TP determinism:** Run the same input on both ranks, compare layer output,
   history, recurrence-state, and token-count digests, and prove no KDA
   transport occurred.
7. **Regression:** Run existing DeepSeek V4 Q4_K and Q2_K control gates using
   the same binary. GLM-specific APIs must remain unreachable for those model
   families.

No performance result is a candidate until all correctness gates for its
stage pass. Profiling or dump instrumentation is excluded from candidate
builds.

## Failure behavior

- Missing or mismatched GLM-5 tensors fail during model validation.
- A rank capability mismatch fails TP negotiation.
- Allocation failure reports the requested bytes and slot count, then aborts
  before model execution.
- A GPU operation failure invalidates the sequence slot and terminates the
  request; partial state is never silently reused.
- Any unexpected fallback, provider mismatch, rank digest mismatch, or
  numerical-gate failure rejects the candidate.

## Review conclusions

Opus 5 recommended explicit per-slot memory accounting, chunk-boundary tests,
state invalidation after failed work, and a real-layer oracle before dispatch.
Those recommendations are included. Its suggestion to head-shard the initial
KDA state was not adopted because DS4 already replicates the corresponding
non-expert weights; replication avoids adding a new collective to every KDA
layer and fits the measured 124 GiB target for the initial single-slot
baseline.
