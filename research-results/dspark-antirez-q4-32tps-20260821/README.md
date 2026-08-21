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

## Exact verifier checkpoint and production-width cost

The FFN bisect identified the remaining fidelity error as the association of
the shared expert with each rank's routed partial. Ordinary serial TP computes
and reduces:

```text
(routed rank 0 + shared rank 0) + (routed rank 1 + shared rank 1)
```

The original batch verifier instead reduced routed rank partials first and
then added a replicated full shared expert. Although algebraically similar,
the float association differs and the error compounds through 43 layers.
`DS4_DSPARK_VERIFY_TP_SHARED_SPLIT=1` restores the serial TP association.

With Q8 activation reuse, 32-head ownership, per-row serial attention, the
owned-group Q8 output projection, fused serial head norm/RoPE, and shared
expert split enabled, a width-1 capture was bit-exact at every recorded stage
and at the end of all 43 layers across consecutive cycles.

The corresponding instrumentation-free runs establish the current cost:

| Verifier | Prefill | Decode | Acceptance / yield | Verifier cost | Fingerprint |
| --- | ---: | ---: | ---: | ---: | --- |
| Exact width 1 | 200.11 t/s | 4.25 t/s | 68.90%; 206/299 drafts | 59,989 ms / 299 cycles | `f16349df33cf1d37` |
| Exact width 5, serial routed MoE | 200.91 t/s | 6.55 t/s | 33.67%; 235/698 drafts; 1.643 accepted/cycle | 39,363 ms / 143 cycles | `7174e214e05fd83e` |

The exact width-5 verifier is therefore correctness-green but economically
far from the 32 t/s target: about 275 ms per cycle is spent in verification.
The next optimization must keep the exact topology while sharing weight reads
across rows; merely replaying the one-row route five times cannot win.

### TP transport attribution (2026-08-21)

A separate `PROFILE=1` build measured the exact width-5 verifier over RoCE v2
with `DS4_TP_RDMA_GATE_PROFILE=1`,
`DS4_TP_SERVICE_INTERVAL_PROFILE=1`, and
`DS4_TP_BIGGATE_PROFILE=1`. These switches remain compiled out of production
builds. The profiling run retained the Q8 drafter, 118/138 expert split, and
the exact serial-routed-MoE verifier. Its token rate is diagnostic only; the
transport counters are the evidence of interest.

The first 48 bulk gates included the fixed 2,048-token prefill. Subtracting
that prefix from the final 544-gate cumulative counters isolates 496 verifier
exchanges across eight speculative cycles:

| Rank | Verifier wire/wait | Staging copy | Combined | Per cycle | Share of 2,892.5 ms verifier |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 932.5 ms | 87.1 ms | 1,019.6 ms | 127.5 ms | 35.3% |
| 1 | 999.5 ms | 89.7 ms | 1,089.2 ms | 136.2 ms | 37.7% |

The critical-path estimate is therefore about 136 ms of TP transport and
staging per speculative cycle. Each verifier exchange carries approximately
75 KiB, but the current bulk protocol costs about 1.9--2.0 ms per exchange.
This is a latency/barrier problem rather than saturation of the 25 Gbit/s
link. For comparison, ordinary 16 KiB row gates in the same run cost roughly
57--116 us depending on rank and gate.

This changes the next-step ordering. Do not continue broad drafter or kernel
tuning until one bounded transport experiment tests a persistent verifier
receive window on the decode-latency QP (16 KiB chunks, pre-posted receives,
deferred send completions), or an RC `RDMA_WRITE_WITH_IMM` equivalent into
fixed registered verifier slots. The protocol must retain exact sequence/slot
validation and fail closed. If it cannot reduce an isolated 75--80 KiB
exchange below 250 us and the full verifier below two ordinary target steps,
stop the DSpark performance effort.

Relevant prior art supports this shape of experiment rather than a new
drafter quantization: RCCL recommends aggregating/batching small collectives,
DeepEP exposes persistent registered low-latency receive buffers and async
completion, and libibverbs defines `RDMA_WRITE_WITH_IMM` specifically to place
data at a fixed remote address while producing a remote completion. These are
design references only; none is assumed to work correctly on gfx1151 without
the isolated two-rank harness and kernel-panic-safe failure tests.

### Rejected batched-routed-MoE transplant

One short experiment re-enabled the existing batched routed-MoE kernel while
retaining rank-owned shared partials. It generated the expected short-run token
fingerprint but was not verifier-equivalent: layer 0 already differed by
0.00932 max / 0.00133 RMS, growing to 4.28 / 0.358 by layer 42. At selected
layer 1 the first visible inputs were already different and the error grew at
attention output. The experiment is rejected and its code was reverted. Token
agreement over three generated tokens is not sufficient evidence for a kernel
substitution.

## Detailed `ds4-on-spark` implementation audit

The source audit was expanded from the v0.5.0 snapshot to its full history.
The most relevant upstream commits and their actual mechanisms are:

| Commit | Implementation | Reported single-GPU effect | TP=2 disposition |
| --- | --- | --- | --- |
| [`7ffe821`](https://github.com/Entrpi/ds4/commit/7ffe82141df6da30b9674cf198df1670f3b0c1e6) | Aligned Q8_0 weight-outer kernel for verifier widths 2–8; reads each weight stream once and accumulates all activation columns | 240K step 59.4→55.2 ms | Highest-priority dense Q8 idea; implement against rank-owned shapes and require exact row outputs |
| [`6e27e1e`](https://github.com/Entrpi/ds4/commit/6e27e1e7ebcb9cb42dafd31a17a1a478690beee9) | First-owner expert dedup: the first slot for an expert reads its weights once and computes every matching verifier column | gate/up family 22.0→18.9 ms/step; 1.53× at 18 distinct of 30 slots | Strongest MoE idea, but their kernel is IQ2_XXS on CUDA; ours needs Q4_K, per-rank IDs, and serial-equivalent folds |
| [`18b40d7`](https://github.com/Entrpi/ds4/commit/18b40d77626125aa4cf17bbfd5f35f90a28ae34e) | Indexer token loop: one CTA stages a K tile once, then evaluates consecutive verifier tokens | 240K 55.2→50.6 ms; width-5 isolated 897→566 µs | Useful mainly at deep context; preserve each token's visibility and TP ownership |
| [`17a7d76`](https://github.com/Entrpi/ds4/commit/17a7d76) | Head-group flash decode: eight heads share each staged KV row tile with online softmax and fixed-order split combine | 240K 76.3→66.2 ms; 515K 95.0→73.5 ms | CUDA kernel is not portable, but row staging across owned heads is architecturally transferable |
| [`e0ed742`](https://github.com/Entrpi/ds4/commit/e0ed742) | Indexed-attention gather version of the same head-group design | smaller incremental gain after head-group decode | Lower priority until our verifier ledger shows indexed attention dominant |
| [`4ad0c9b`](https://github.com/Entrpi/ds4/commit/4ad0c9b) | WMMA indexer scoring for multi-sequence decode | 3.97× scorer, about 13% deep-decode reduction | gfx1151 requires a separate WMMA mapping and isolated ISA/correctness harness |

Two implementation details matter more than the headline numbers:

1. The Spark verifier is explicitly hybrid: weight-bound dense/MoE work is
   multi-column, while KV/state-sensitive operations preserve deterministic
   ordering. It is not one generic batched forward.
2. Its first-owner expert optimization is content-driven inside a fixed launch
   grid, so it remains graph-capture-safe. The performance benefit comes from
   duplicate expert selections across verifier rows, not from changing routing.

The fork reports a width-5 verifier cost around 2.03–2.08 ordinary decode steps
on GB10, giving a break-even near 51% draft acceptance. Our exact width-5 path
currently costs much more because routed Q4 is replayed row-by-row and TP adds
communication/reduction. Its terminal quench policy can protect poor prompts,
but it cannot create the required verifier speed and will be considered only
after our measured TP break-even is competitive.

### Transfer order after the audit

1. Build an isolated Q4_K first-owner/deduplicated gate+up test using captured
   live per-rank routes. Require bit-exact outputs versus the exact serial
   verifier for overlap counts from zero to the measured maximum.
2. Build an exact width-2–5 Q8 weight-outer projection harness for gfx1151.
   Reuse Q8_1 activations, inspect emitted ISA, and compare each column with
   repeated one-row projections before integrating it.
3. Profile an exact width-5 cycle to quantify routed MoE, dense Q8, attention,
   indexer, RDMA, proposal, and target-anchor time. Port indexer/head-group
   designs only if those families are in the top ledger.
4. Keep every new dispatch behind DSpark-only shape gates and rerun ordinary
   128/128 Q4 control after shared backend changes.

This is a design transfer, not a source transplant: GB10 uses CUDA warp-32 and
different quantized model tensors, while gfx1151 uses wavefront-32 ROCm plus a
two-rank expert and attention decomposition. The reusable idea is to amortize
immutable weight/KV reads across verifier columns without altering target
arithmetic, ownership, or state order.

### Current Entrpi main ROCm DSpark audit

The audit also fetched current `Entrpi/ds4` main at
[`84cc882`](https://github.com/Entrpi/ds4/commit/84cc882352757baf628a1776badf7cc54d584e28)
(2026-08-09), which newly enables DSpark on ROCm. That commit adds two missing
backend primitives rather than a fast TP verifier:

- non-causal raw batch attention for the three-layer DSpark draft model;
- an on-device Q8_0 Markov correction plus vocabulary argmax, avoiding a host
  logits round trip.

Our fork already contains the non-causal ROCm draft-attention kernel. It still
routes `ds4_gpu_dspark_markov_argmax_tensor` to the unavailable stub, so the
on-device Markov primitive is a clean, directly portable proposal-side change.
It is not the current main bottleneck: our exact width-5 measurement spends
about 14.6 ms/cycle proposing but 275.3 ms/cycle verifying. Port it only after
an isolated output/argmax test, and expect a modest gain rather than a verifier
breakthrough.

Entrpi's own release gate records single-node Strix Halo IQ2 results of 16.70
t/s ordinary and 11.40 t/s with DSpark (64-token C fixture). It explicitly
states that ROCm DSpark is not yet expected to beat ordinary decode. This
supports our diagnosis: merely enabling the drafter or committing batched
verifier state directly is insufficient; the weight-bound verifier families
must be made width-efficient. Our `replay=0` exact runs already use direct
state commit, so their 6.55 t/s is not caused by replay fallback.

### Ordinary-path isolation gate after fidelity work

After reverting the rejected batched-MoE experiment, both nodes were rebuilt
from identical source and the full ordinary 2,048+300-token control was rerun:

```text
prefill 275.65 t/s
decode  19.54 t/s (19.58 steady)
FNV64   b7694f9d11a3760e
binary  da669e37ea50e473c8a8dfe5d81d888b09f1d42894519075a0e3bf9b2a5c3b10
```

The fingerprint exactly matches the clean 273.57/19.61 production control and
the throughput remains in its established noise envelope. The DSpark-only
fidelity work therefore does not regress ordinary Q4 TP=2 inference.

### Slab-direct verifier latency profile

The current verifier and its two per-layer RoCE exchanges were profiled before
the next optimization. The exact diagnostic fingerprint remained
`7174e214e05fd83e`. A 43-layer verifier invocation costs about 246.0 ms wall
time: approximately 102 ms attention, 94--101 ms routed MoE, and 24--32 ms in
the FFN combine/residual tail. The Q8 proposal costs only 14.6 ms per
speculative cycle.

The registered five-row payload itself completes in about 62--69 us. Rank 0's
full attention/FFN callbacks average 204/408 us because it reaches both gates
before rank 1; the extra time is peer compute skew, not verbs setup or link
bandwidth. See [VERIFIER-LATENCY-PROFILE-20260821.md](VERIFIER-LATENCY-PROFILE-20260821.md)
for the per-rank stage and transport tables.
# Exact Q8 attention-output weight-outer stage

The first Q8 multi-row attempt was correctly rejected: it produced 14.07 t/s
but changed FNV64 from `7174e214e05fd83e` to `7ea6813f88bdea4d`, and a
non-binary activation oracle exposed the changed rounding DAG. ISA inspection
then showed that the shipped gfx1151 kernels form a four-value `q*x` dot before
applying the FP16 block scale, with a distinct `q3,q1,q2,q0` schedule for the
second low accumulator and the expansion projection.

The final implementation transcribes those emitted DAGs with reassociation
disabled, pairs adjacent low rows, and reuses each raw Q8_0 weight block across
two to five verifier tokens. It creates no expanded-weight copy or persistent
cache. The exact same-binary oracle uses non-power-of-two Q8 scales and
non-binary activations, covers both TP-owned group halves, and reports the low
and expansion results separately.

| Verifier width | Serial projection | Weight-outer | Speedup | Exact |
| ---: | ---: | ---: | ---: | :---: |
| 2 | 0.250 ms | 0.150 ms | 1.661x | yes |
| 3 | 0.367 ms | 0.287 ms | 1.280x | yes |
| 4 | 0.476 ms | 0.349 ms | 1.365x | yes |
| 5 | 0.592 ms | 0.415 ms | 1.428x | yes |

All widths also remain bit-exact with both projection thread counts forced to
256, exercising a different block/grid ownership. The launch gate requires
that the low-row and expansion dimensions divide their rows-per-block, so a
future shape cannot return some waves before an in-loop barrier. TP hello
negotiates the feature explicitly and fails closed on a mismatch. ROCm-only
calls and feature advertisement are compile-time guarded; a CPU build links
successfully, and ordinary inference never selects the DSpark-only path.

### Full `ds4-bench-tp` result

All measurements use the fixed 2,048+300 Antirez workload, 118/138 DSpark
expert split, Q8 drafter resident only on rank 0, no expanded-weight cache,
and mandatory explicit RDMA.

| Run | Provider | Prefill | Decode | FNV64 |
| --- | --- | ---: | ---: | --- |
| Final hardened r1 | RoCE v2 | 189.76 t/s | 13.71 t/s | `7174e214e05fd83e` |
| Final hardened r2 | RoCE v2 | 189.71 t/s | 13.73 t/s | `7174e214e05fd83e` |
| Final hardened r3 | RoCE v2 | 191.12 t/s | 13.71 t/s | `7174e214e05fd83e` |
| Pre-review cross-provider check | OdinLink | 181.30 t/s | 13.86 t/s | `7174e214e05fd83e` |
| Ordinary regression control | RoCE v2 | 256.59 t/s | 19.75 t/s | `b7694f9d11a3760e` |

The final hardened RoCE v2 median is **189.76 prefill / 13.71 decode t/s**, a
5.2% decode gain over the exact 13.03 t/s first-owner baseline. It does **not**
clear the preset 13.80 t/s promotion gate, so the kernel remains opt-in and the
Q8 lane stops here. An earlier pre-review binary measured 13.78/13.80/13.80
t/s over RoCE v2 and 13.86 t/s over OdinLink with the same fingerprint; those
numbers are retained as noise/cross-provider evidence, not substituted for
the final median.

Verifier time nevertheless fell from about 200.6 ms/cycle to a 107.6 ms/cycle
final median. Acceptance did not move: 235 of 698 draft proposals were
accepted across 143 cycles, producing about 2.10 final tokens/cycle. The
milestone blocker is now acceptance/token yield rather than Q8 verifier
latency; 32 t/s is not credible without roughly 4.2--5 final tokens/cycle.

The stats logger previously counted the already-emitted top-level token again
for every chained verifier call, reporting `first_tokens=143`. The corrected
definition counts only calls entering with exactly one top-level token: this
workload has 65 such tokens plus 235 accepted drafts, or exactly 300 final
tokens / 143 verifier cycles = 2.098 tokens/cycle. The logger now prints that
value directly and reports cumulative accepted/eligible counts for each draft
position; these counters are diagnostics only and do not enter scheduling.

Enable the opt-in candidate symmetrically on both ranks with:

```sh
export DS4_ROCM_Q8_ATTN_OUT_WEIGHT_OUTER=1
```

The launcher command is the Q4 first-owner reproduction in
[`Q4-FIRST-OWNER-20260821.md`](Q4-FIRST-OWNER-20260821.md) plus that switch.
The next implementation lane is exact QKV-to-core/head-group/indexer
multi-row reuse; its isolated stop gate remains at least 1.25x with the known
fingerprint unchanged.

## QKV-to-core kernel trace and indexer stop gate

A coordinator-only `rocprofv3 --runtime-trace --selected-regions` run used the
same 2,048+300 Antirez workload, RoCE v2, exact fingerprint
`7174e214e05fd83e`, and the opt-in exact Q8 attention-output candidate. Both
ranks used binary SHA-256
`78f5eba10931047c826ac4744134bd26fd18b6efff58c325498d68b1451d14df`.
The profiler perturbs timing, so its 13.23 t/s result is diagnostic only.

| Serial QKV/core family | Calls | GPU time over 83 full verifiers | Per full verifier |
|---|---:|---:|---:|
| Indexed + non-indexed attention scans | 20,210 | 1,472.09 ms | 17.74 ms |
| Indexer score + top-k + mask | 25,389 | 458.18 ms | 5.52 ms |
| Raw-KV store | 20,837 | 27.66 ms | 0.33 ms |

The attention scan is 3.2x the complete score/top-k/mask indexer term. The raw
trace is under
`runs/rocprof-antirez-q4-dspark-qkv-kerneltrace-r1/`; it is intentionally
git-excluded because ROCm emitted a 1.4 GB SQLite database.

The bounded exact indexer experiment stages one 128-float compressed K row per
CTA and evaluates verifier tokens sequentially with the shipped
`float4`/`warp_sum`/`partial[4]`/`h0` arithmetic. It receives the authoritative
per-token host `index_counts`; it never reconstructs visibility from position.
Full-mantissa inputs, near-ReLU-boundary rows, mixed visibility, cap-flat
counts, the top-k boundary, W1, scores, selected IDs, and masks all passed
bitwise.

| Width | Serial score | K-reuse token loop | Speedup | Exact |
|---:|---:|---:|---:|---:|
| 1 | 0.01326 ms | 0.01332 ms | 0.995x | yes |
| 2 | 0.02644 ms | 0.02422 ms | 1.092x | yes |
| 3 | 0.03976 ms | 0.03500 ms | 1.136x | yes |
| 4 | 0.05314 ms | 0.04732 ms | 1.123x | yes |
| 5 | 0.06682 ms | 0.05898 ms | 1.133x | yes |

Every W2--W5 result misses the preset 1.25x stop gate. A mistakenly concurrent
three-process timing produced larger W3--W5 figures; those measurements are
invalid and explicitly discarded. Fable's failure review also rejected a
grid-Y retry: one token already exposes 576 CTAs, so extra token concurrency is
unlikely to recover W2's missing 15 percentage points, while per-token top-k
and mask launch costs would remain. The experiment is not integrated. The
next target is exact attention head-group/multi-row reuse.

The DSpark stats anchor counter now increments at the public sampling-cycle
entry, outside the chained verifier and before the final one-token tail return.
This fixes the diagnostic total from 299 to the benchmark's 300 generated
tokens without changing scheduling, acceptance, or model arithmetic.

## Exact two-head indexed-attention batching

The next bounded experiment targeted the indexed-attention scan rather than
the smaller indexer term.  It pairs two independent 256-thread head groups in
one 512-thread CTA for verifier widths 2--5.  Each head retains the shipped
serial `float4` dot order, eight-wave reduction, unsorted top-k order, sink
placement, and raw-then-compressed accumulation.  The production gate is
ROCm + TP head split + DSpark + ratio-4 indexed attention + compact F32 cache;
TP hello feature bit 23 rejects asymmetric enablement.  No weight expansion,
model cache, or persistent allocation is added.

| Width | Repeated isolated speedup | Bitwise exact |
|---:|---:|:---:|
| 2 | 1.94--2.04x | yes |
| 3 | 1.41--1.52x | yes |
| 4 | 1.55--1.59x | yes |
| 5 | 1.59--1.74x | yes |

The first full candidate changed the token fingerprint even though the
isolated oracle passed.  A temporary serial diagnostic preserved the same
top-k/raw-row schedule and restored `7174e214e05fd83e`, isolating the failure
to the paired kernel.  A live paired-versus-serial comparator then found
intermittent whole-head losses: exactly 512 or 1,024 mismatched values.  The
two reductions had reused the same eight shared-memory slots; a leading wave
could begin the sum reduction before a trailing wave consumed the max.  The
fix assigns distinct max and sum storage.  Twenty repeated isolated W2--W5
tests passed bitwise afterward, followed by a complete 2,048+300 live
comparison with zero mismatch events and the exact fingerprint.

The clean final binary produced:

| Run | Provider | Prefill | Decode | FNV64 |
|---|---|---:|---:|---|
| r1 | RoCE v2 | 188.58 t/s | 14.23 t/s | `7174e214e05fd83e` |
| r2 | RoCE v2 | 190.18 t/s | 14.21 t/s | `7174e214e05fd83e` |
| r3 | RoCE v2 | 190.05 t/s | 14.24 t/s | `7174e214e05fd83e` |
| Cross-provider | OdinLink | 181.36 t/s | 14.27 t/s | `7174e214e05fd83e` |
| Ordinary control | RoCE v2 | 258.60 t/s | 19.32 t/s | `b7694f9d11a3760e` |

The RoCE v2 median is **190.05 prefill / 14.23 decode t/s**.  That is 9.2%
above the 13.03 t/s exact first-owner baseline and 3.8% above the preceding
13.71 t/s Q8 weight-outer median.  The acceptance histogram remains exactly
`21,5,11,5,3,5,1,3,2,2,2,1,0,0,0,0,4`; the speedup is verifier execution only.
OdinLink reported 37,908 streaming copies, 3.49 GB, and zero fallback calls.
The ordinary control did not advertise or select the DSpark-only feature and
retained its known fingerprint.

Enable the accepted candidate symmetrically with:

```sh
export DS4_ROCM_DSPARK_EXACT_ATTN_HEAD2=1
```

The remaining economic gap is still large: at 2.098 final tokens per cycle,
30 t/s permits only 69.9 ms for the complete cycle.  With approximately
39.8 ms outside the measured 107.6 ms verifier, the verifier would need to
approach 30 ms.  The next measurement therefore separates verifier wall time,
summed GPU work, and launch count by compute island before another kernel is
attempted; acceptance changes remain deferred until verifier hits are cheaper.

## Verifier launch ledger and exact batched vocabulary argmax

A compiler-gated ROCTx range around the complete target verifier made it
possible to join verifier wall intervals to the coordinator's rocprofv3 kernel
and HIP API records.  The range and dynamic ROCTx dependency exist only in
`PROFILE=1` builds; the production binary compiles them out.  The diagnostic
2,048+300 run retained fingerprint `7174e214e05fd83e` and measured 83 complete
target-verifier invocations:

| Per full verifier invocation | Mean |
|---|---:|
| Wall interval | 181.92 ms |
| GPU kernel-active union | 142.02 ms |
| Wall minus kernel-active | 39.90 ms |
| Kernel dispatches | 4,375.5 |
| Explicit `hipLaunchKernel` calls | 3,539.5 |
| CPU time inside launch APIs | 44.88 ms |

The wall/kernel gap clears the predeclared 25 ms launch-ledger gate, but the
family breakdown exposed a more bounded defect first.  The final vocabulary
top-1 for a multi-row verifier called the generic single-thread
`indexer_topk_kernel`, scanning 129,280 logits serially.  It appeared once per
full verifier and cost 12.34 ms.  This operation is independent across rows and
does not require a general top-k implementation.

The replacement launches one 1,024-thread workgroup per row and performs the
same strict-greater reduction with the same lower-index tie break as the
existing parallel one-row argmax.  A production-shape oracle covers ties,
all-negative-infinity input, and widths 1--4:

| Rows | Generic top-1 | Parallel row argmax | Isolated speedup | Exact |
|---:|---:|---:|---:|:---:|
| 1 | 12.116 ms | 0.0155 ms | 782.0x | yes |
| 2 | 12.122 ms | 0.0158 ms | 769.3x | yes |
| 3 | 12.135 ms | 0.0157 ms | 774.5x | yes |
| 4 | 12.143 ms | 0.0159 ms | 764.0x | yes |

The rows=1 timing is an oracle coverage point, not a production saving: the
existing verifier already used its parallel one-row argmax when only one top
row was present.  Production savings accrue on the multi-row calls.  The ROCm
flags explicitly place `-fno-finite-math-only` after `-ffast-math`, so the
negative-infinity sentinel remains in contract.  A follow-up oracle also
covers the legacy generic path's unusual NaN rule: NaN at column zero remains
selected, while a NaN in any later column loses the strict-greater comparison.

TP hello feature bit 24 rejects asymmetric enablement.  The candidate adds no
persistent allocation or expanded-weight cache and is selected only for an
active ROCm DSpark session with:

```sh
export DS4_ROCM_DSPARK_BATCH_ARGMAX=1
```

Three instrumentation-free RoCE v2 candidate runs and both required controls
produced:

| Run | Provider / mode | Prefill | Decode | FNV64 |
|---|---|---:|---:|---|
| r1 | RoCE v2 + DSpark | 190.75 t/s | 14.88 t/s | `7174e214e05fd83e` |
| r2 | RoCE v2 + DSpark | 190.73 t/s | 14.71 t/s | `7174e214e05fd83e` |
| r3 | RoCE v2 + DSpark | 188.00 t/s | 14.85 t/s | `7174e214e05fd83e` |
| Cross-provider | OdinLink + DSpark | 181.34 t/s | 15.05 t/s | `7174e214e05fd83e` |
| Ordinary control | RoCE v2, no DSpark | 258.84 t/s | 19.52 t/s | `b7694f9d11a3760e` |

The production RoCE median is **190.73 prefill / 14.85 decode t/s**, 4.4%
above the exact two-head median and 14.0% above the 13.03 t/s exact baseline.
All DSpark runs retained the exact acceptance histogram
`21,5,11,5,3,5,1,3,2,2,2,1,0,0,0,0,4`.  OdinLink reported 37,908 streaming
copies, 3.49 GB, and zero fallback calls on both ranks.  The ordinary path did
not negotiate the DSpark feature and remained within 0.3% of its 19.57 t/s
reference while preserving its exact fingerprint.

At unchanged 2.098-token/cycle yield, 30 t/s allows 69.9 ms per cycle.  The
measured non-verifier term consumes about 39.8 ms, leaving 30.1 ms of
cycle-normalized verifier budget, or about 51.9 ms per full verifier invocation
at the observed 83/143 invocation rate.  Current verifier wall is about
181.9 ms before this 12.3 ms removal, so reaching 30 t/s still requires roughly
a further 3.3x verifier reduction, higher accepted yield, or both.
