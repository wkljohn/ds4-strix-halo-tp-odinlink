# Antirez Q4 TP=2 DSpark decode research — 2026-08-21

This is the durable lab record for the `research/dspark-antirez-q4-32tps-20260821`
branch. The target is 32 decode tok/s under `ds4-bench-tp`, with mandatory RDMA,
without regressing ordinary non-DSpark inference. Results in this directory are
research evidence, not README performance claims.

## Fixed test identity

- Target: `DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix-0731.gguf`
- DSpark support: `DeepSeek-V4-Flash-DSpark-support-0731.gguf`
- Provider: ConnectX-4 Lx, RC/RoCE v2
- Workload: 2,048-token frontier plus 300 generated tokens, unless a run is
  explicitly marked as a short diagnostic
- Ordinary split: 128/128 experts
- DSpark split: 118/138 experts
- `ds4` SHA-256: `cbb8b5e20f582c3bed372238734a5f1a05c651a545fec4b8605310f856aa25ad`
- `ds4-bench-tp` SHA-256: `83c6fa658187b3530aa84a577d8609839c2b96f7fbdaeea5eb7ca074af13c695`
- Target sampled SHA-256: `73ee63957de60fcd65c0eadb6bf826de95ac45ca2a26fd42a08917aa75ff3cc1`
- Support sampled SHA-256: `87b3fe8ecf4625989b0b310968237a0a214fe4e031053bdfdbb9d6e8c07bf469`

## Clean retest after concurrent weight-conversion activity

Earlier runs were treated as provisional because another GPU/UMA-heavy weight
conversion was active. Both nodes were subsequently verified idle (zero KFD
processes and zero reported GPU use), and the controls were repeated serially.

| Test | Prefill | Decode | Fingerprint | Interpretation |
| --- | ---: | ---: | --- | --- |
| Ordinary production control | 273.57 t/s | 19.61 t/s | `b7694f9d11a3760e` | Clean match to the established ordinary trajectory and performance envelope |
| Full-logit ordinary control | 273.64 t/s | 19.46 t/s | `b7694f9d11a3760e` | Greedy top-2 does not change this 300-token greedy trajectory |
| DSpark-flag-matched serial oracle | 200.79 t/s | 15.66 t/s | `2aa153138c195efc` | DSpark-compatible flags/split already cost 20.1% decode before drafting |
| DSpark width 1 | 199.32 t/s | 5.26 t/s | `813b426a7aee8597` | 66.56% one-token acceptance, but verifier arithmetic diverges and is far too expensive |

The contaminated flag-matched serial run was 201.01/15.63. Its clean repeat was
200.79/15.66, so the concurrent conversion did not materially alter that
particular diagnostic. Only the clean runs are retained as decision evidence.

The width-1 authoritative stats are 299 one-token proposals, 199 accepted draft
tokens, 100 first misses, and 66.56% acceptance. The generic CSV `gen_cycles`
field is not the DSpark cycle count and must not be used to interpret this run.

## Width-1 fidelity bisect

A three-token diagnostic enabled `DS4_DSPARK_VALIDATE_VERIFIER_CAPTURE=1`.
It is excluded from performance evidence. Batch and serial replay were exact at
`input_hc`, `attention_pre`, and `attention_norm`. Rank-0 attention heads first
differed by about 1.5e-6 max absolute error. The Q8 attention-output projection
amplified that to 0.0451 max / 0.0117 RMS at `attention_out` in layer 0, and the
error grew through later layers.

Two existing TP verifier experiments were checked independently:

| Experiment | Layer-0 `attention_out` max error | Result |
| --- | ---: | --- |
| Default replicated verifier | 0.04510 | First material divergence |
| `DS4_TP_BATCH_ATTN_LEGACY_DECODE=1` | 3.54786 | Rejected; stale rank-0-only topology is much worse |
| `DS4_TP_BATCH_ATTN_HEAD_SPLIT=1` | 0.04510 | Correct ownership topology, but does not remove the upstream tiny numerical difference |

The legacy switch predates commit `64d450f` (`Fix TP attention ownership`) and
models an older rank-0-only reduction. It must not be promoted.

The deeper bisect then separated the attention stages:

| Incremental change | First remaining layer-0 difference | Result |
| --- | --- | --- |
| Reuse the serial Q8 activation quantization | `q_b` projection | `q_a_raw` and `q_a_norm` became bit-exact |
| Add correct 32-head TP ownership | attention core, about 1–2e-6 max | Q projection, head norm, and Q RoPE became bit-exact |
| Run each verifier row through the serial attention scan | attention-output projection | heads before and after inverse RoPE became bit-exact |
| Run each verifier row through the serial owned-group Q8 output projection | later FFN/MoE work | all captured layer-0 attention and FFN-input stages became bit-exact |

This locates two real architectural mismatches rather than one generic
"batching error": the multi-row attention scan changes the reduction order by
about 1e-6, then the multi-row Q8 output projection chooses different
activation-scale boundaries. Reusing the established one-row TP decomposition
removes both. The layer-end replay still begins with small error after the
captured `ffn_norm` boundary and amplifies across layers, so the next bisect is
the shared/routed MoE verifier, not attention.

These are short capture runs and not throughput results. Their approximately
3.2–3.3 t/s diagnostic rate includes synchronous tensor readback and must not
be compared with production decode.

## `ds4-on-spark` source comparison

The comparison used `Entrpi/ds4` tag `v0.5.0`, commit
`d9c8587f3e080ce40fd961a9dd09c66c294a6b10`.

Relevant techniques in that CUDA/GB10 fork are:

- aligned Q8_0 verifier kernels specialized for widths 1–8;
- head-group flash decode for dense and indexed attention;
- an indexer scorer that stages weights once and loops verifier tokens;
- first-owner expert dedup for overlapping gate/up expert selections;
- a terminal yield/quench controller that stops losing speculation;
- producer-side quantized activation reuse and fused attention-output stages.

The source-level correspondence is now clearer:

| `ds4-on-spark` design | TP=2 relevance | Current verdict |
| --- | --- | --- |
| Serialize KV-bound attention/state changes | Directly transferable | Confirmed: it makes verifier attention bit-exact |
| Batch weight-bound dense projections | Transferable only if the same quantization tier is retained | Current batch Q8 output tier is not arithmetic-equivalent; one-row owned-group projection fixes it |
| Width-1–8 aligned Q8 verifier tiers | Promising after fidelity | Port only after the serial oracle matches through every layer |
| First-owner gate/up expert dedup | Promising for width 2–5 when routes overlap | Requires a TP-owned expert implementation and exact per-row accumulation order |
| Head-group flash decode | Architecture-specific kernel idea | NVIDIA CUDA code is not portable, but KV staging once per head group is worth a gfx1151 design |
| Indexer token-loop | Likely useful at wider verify widths/deeper context | Must preserve per-token cache/index visibility |
| Terminal yield quench | Operationally useful, not a kernel speedup | Add only after measured TP break-even; it cannot rescue an intrinsically slow verifier |

The fork is single-GPU, so it has no answer for our batch-versus-TP-reduction
fidelity boundary. Its parity report gives the more useful architectural rule:
batch weight-bound projections, but retain serial arithmetic/order for the
cheap KV-bound attention scan and state mutation. Its release also reports that
an always-batched exact two-row verifier was neutral or slower when routed rows
selected disjoint experts, so batching alone is not assumed to be a win.

## Incremental next steps

1. Extend the layer-0 capture through shared expert, routed gate/up, routed
   down, rank partial, and TP reduction boundaries. Find the first FFN/MoE
   difference after the now-exact `ffn_norm` input.
2. Make width 1 match the flag-matched serial oracle through all 43 layers
   before testing speed.
3. Re-enable the ordinary proven kernel set as one bounded DSpark-only A/B and
   require unchanged DSpark fingerprint/acceptance.
4. Only then test widths 2, 3, and 5 and compute break-even from measured
   proposal, target-anchor, verifier, and accepted-token costs.
5. Port small-width ideas from `ds4-on-spark` one kernel class at a time, with
   a serial oracle and rollback switch for every arithmetic substitution.
6. After every shared-code change, rerun the ordinary 128/128 fingerprint and
   performance control. A DSpark gain is rejected if ordinary decode regresses.

The current blocker is not drafter acceptance alone. Width 5 previously cost
about 5.20 ordinary token steps per cycle while producing only 2.94 output
tokens per cycle, and width 1 shows that verifier arithmetic is not yet on the
serial trajectory. Correctness and verifier cost must be repaired before
higher acceptance can become useful throughput.
