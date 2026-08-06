# TP=2 ROCm decode research and implementation plan

Date: 2026-08-06

This is the required research gate before the next decode implementation.
It follows three independent passes over DS4, current vLLM, current
llama.cpp, ROCm AITER, upstream issues and pull requests, and official HIP
documentation. NVIDIA DGX Spark / DSpark results were excluded: they do not
describe the gfx1151 memory system, ROCm kernels, or this TP transport.

No decode candidate described here is enabled by default. The repository was
returned to a clean production state before this plan was written.

## Scope and pinned sources

The target is batch-one decode of DeepSeek V4 Flash Q4_K on two Ryzen AI MAX+
395 (gfx1151) nodes with TP=2. Machine tuning and speculative decoding are out
of scope. Kernel/runtime work must remain transport-neutral; any OdinLink-only
change must be gated so that the Mellanox/RDMA path is unchanged.

Source snapshots inspected:

- DS4: this repository through `53f2d80`
- llama.cpp: `6a32c29a746a2e44de463de647f9f6661eb5086b`
- vLLM: `81be2e09aebfd1c45b3ed9f73d2850da8a72984c`
- ROCm AITER: `00b271b95f8a3405fa211e509c63441040557305`

Primary upstream references:

- [vLLM DeepSeek V4 implementation and optimization write-up](https://github.com/vllm-project/vllm-project.github.io/blob/main/_posts/2026-04-24-deepseek-v4.md)
- [vLLM ROCm DeepSeek V4 performance tracker](https://github.com/vllm-project/vllm/issues/41820)
- [vLLM ROCm graph/fallback work](https://github.com/vllm-project/vllm/pull/41601)
- [vLLM closed ROCm multistream experiment](https://github.com/vllm-project/vllm/pull/43365)
- [llama.cpp gfx1151 dequant-to-float matvec](https://github.com/ggml-org/llama.cpp/pull/26301)
- [llama.cpp architecture-specific MMVQ tuning](https://github.com/ggml-org/llama.cpp/pull/19478)
- [ROCm AITER](https://github.com/ROCm/aiter)
- [official HIP graph documentation](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/hipgraph.html)

## Round 1: execution-path comparison

### DS4

The existing service profiler establishes the budget before looking for a
solution:

- About 88% of the decode gate interval is GPU compute, 11% is RDMA work, and
  1% is gate detection.
- The attention half is about 63% of GPU time; FFN is about 37%.
- The real ATTN-gate callback is about 89-119 us, versus about 1177-1196 us of
  compute before the gate publishes.
- Within the safely measurable attention stages, `attn_output` is about 582 us
  and `compressor_proj` is about 131 us. `attn_output` is the only observed
  sub-stage with a comfortable standalone 5% end-to-end ceiling.

The active one-token attention-output path already keeps activations in F32
and dequantizes Q8_0 weights during the dot product. It does **not** use the
available activation-prequantized DP4A path. The low projection uses a
two-rows-per-wave shared-activation kernel; the expand projection uses one
row per wave. Prior 1024/512/256-thread launch sweeps did not improve either
projection.

### vLLM

Current vLLM decomposes the pre-attention work into indexer, KV compressor,
and sliding-window/KV insertion branches. On CUDA it fuses small kernels and
overlaps independent branches on auxiliary streams. Its published low-batch
result attributes a 5-6% end-to-end latency reduction to multistream overlap.

The current ROCm implementation is intentionally serial because earlier
auxiliary-stream attempts hung. Two later ROCm multistream PRs were closed
without hardware correctness, stability, and performance validation. This is
useful design evidence, but not a patch DS4 can safely copy.

vLLM's ROCm path also uses AITER sparse attention and block-scaled FP8 GEMMs on
MI-class targets. Those tensor formats and kernels do not match DS4's GGUF
Q8_0 attention projections on gfx1151.

### llama.cpp

llama.cpp has separate single-token MMVQ and flash-attention vector/tile
paths, with architecture- and quant-specific warp tables. The important
lesson is not a single universal AMD configuration: gfx1151 behaves
differently from discrete RDNA3 and RDNA4.

## Round 2: ROCm kernels, fusion, scheduling, and transport

The AITER implementations with the strongest published numbers primarily
target MI355X/gfx950 or newer accelerator layouts (FP8/FP4, block scales, and
large-server execution). A direct AITER port would first require format
conversion or new kernels and has no evidence of a gain on gfx1151 Q8_0.

vLLM's fused norm+router PR measured about 2.1% at concurrency one on B300
TP=8. Its ROCm sparse-MLA work reports a roughly 100 us kernel on MI355X.
Both are valid later cleanup ideas, but neither maps to DS4's measured dominant
projection window or has an evidence-backed 5% ceiling here.

vLLM's QuickReduce evidence also warns against forcing quantized collectives
for small messages: its low-concurrency result regressed when the quantized
path was forced, while a size threshold recovered only a small average gain.
DS4's transport is already only about 11% of the measured gate interval, so a
transport-only change needs to remove nearly half of all transport time to
reach 5% overall. OdinLink work remains eligible, but it is not the first
decode target and must stay behind provider/capability dispatch.

HIP graphs can capture HIP operations associated with streams. Ordinary host
work is not captured. DS4's TP protocol includes host service work and
cross-stream memory wait/write operations between compute regions. Therefore
a full-token graph is not a drop-in optimization: the only plausible design
is graph segments between TP gates, after separately proving that the wait and
write operations used by DS4 are capture/replay safe on the installed ROCm.

## Round 3: upstream changes and gfx1151 applicability

The strongest directly applicable evidence came from current llama.cpp work:

1. PR 19478 found that increasing MMVQ warps helped RDNA4, but regressed
   Strix Halo. Its gfx1151 measurements and follow-up testing found one wave
   already optimal; extra waves add reduction overhead after shared LPDDR5X
   bandwidth is saturated. Other llama.cpp RDNA changes reached the same
   conclusion for complex K quants.
2. PR 26301 adds a gfx1151-default decode path that keeps activations in F32,
   dequantizes K-quant weights to float, and uses packed `float2`/`float4`
   activation loads. Across 38 models it reports mean +3.09%, median +2.78%,
   with gains shrinking for larger models. It also avoids the accuracy loss of
   quantizing activations to Q8_1.
3. That direction agrees with DS4's existing one-token Q8_0 arithmetic. It
   rejects activation prequantization as the leading candidate, but its packed
   load geometry and controlled row reuse are worth adapting to DS4's Q8_0
   projection rather than copying its Q4_K kernel literally.

The current vLLM ROCm tracker independently prioritizes graphs, dtype-removal,
fusion, sparse attention, and CSA multistream. The graph PR is still open and
mixes correctness fallbacks, datatype fixes, sparse attention, FP4/FP8 work,
and graph enablement, so its large MI355X result does not isolate graph benefit.
The closed multistream PR explicitly left ROCm correctness, long-decode
stability, and performance unchecked. These are leads, not transferable
benchmarks.

## Evidence-ranked implementation plan

### 1. Split the dominant timing window without changing execution

Add reusable, opt-in HIP-event timing around the low projection and the expand
projection. Both execute on the same compute stream, so this split avoids the
invalid cross-stream timing assumption documented in the older investigation.
Measure both ranks over a long decode and report distributions, not only means.

Exit condition: identify which projection owns enough of the token budget that
a realistic kernel improvement can produce at least 5% end-to-end. If neither
does, stop projection work and recalculate the ranking.

### 2. Redesign only the dominant F32-activation Q8_0 projection kernel

Build an isolated kernel harness for its exact production dimensions. Keep the
current kernel as the reference and try, in order:

1. Packed Q8_0 weight loads paired with `float2`/`float4` activation loads,
   using sub-wave groups to process more than one 32-value Q8 block per
   iteration.
2. Controlled 1/2/4-row activation reuse, with compile-time variants so VGPR
   pressure and occupancy are measured rather than guessed.
3. A gfx1151-only dispatch selected by architecture and exact supported shape;
   all other GPUs and all transports retain the existing kernel.

Do not add more waves per output row, do not quantize the activation, and do
not change the Q8_0 on-disk format. Packed/sub-wave accumulation can change
floating-point association, so it requires greedy-output validation rather
than being labeled bit-exact in advance.

Promotion threshold: at least 10% faster for the isolated projection and at
least 5% median end-to-end decode improvement. A smaller improvement may be
kept only as an opt-in experiment, not made the Strix TP=2 default.

### 3. Try one guarded pre-attention overlap only after the kernel result

Use the Round 1 timeline to choose exactly one independent branch to move to a
nonblocking auxiliary stream. Follow vLLM's event fan-out/join and tensor
lifetime pattern, but explicitly replace DS4's reliance on legacy default-
stream synchronization at that boundary. Enable it only for one-token decode
and only behind an environment switch during validation.

Before implementation, compute the overlap ceiling from the new trace. Do not
proceed unless the shorter independent branch accounts for at least 5% of
token time. Require a 1000-token two-node stability run because upstream ROCm
attempts were closed without resolving the relevant hang risk.

### 4. Graph feasibility probe, not a speculative rewrite

Create a minimal HIP program using the installed ROCm version to test capture
and replay of the exact event and stream wait/write primitives used by DS4. If
that works, measure launch overhead for one gate-bounded compute segment and
estimate the full-token ceiling. Implement segmented graphs only if the
measured ceiling is at least 5% and pointers/shapes remain stable across token
steps. Otherwise close this direction with the probe evidence.

### 5. Revisit OdinLink only with a measured provider-side target

Repeat alternating provider A/B only after compute candidates, with callback
time and end-to-end decode recorded together. Any provider optimization must
be negotiated or capability-gated; Mellanox and non-OdinLink providers must
retain the old behavior. The current 11% transport share makes this a lower
probability standalone 5% candidate.

## Explicitly deprioritized or rejected

- More waves/workgroup-only tuning: upstream gfx1151 evidence and DS4's live
  sweeps are negative.
- Two-row expand as a simple geometry change: a preliminary pre-plan probe was
  slower (11.31 versus 10.57 t/s for its first completed comparison); the run
  was stopped before a full A/B and the code was removed.
- Activation Q8 prequantization/DP4A: contradicts current gfx1151 evidence,
  adds quantization work, and reduces arithmetic fidelity.
- Quality-only split-K: inactive in the production configuration.
- Fusing only the low/expand intermediate: saves about 16 KiB against tens of
  MiB of weight traffic per layer and has no plausible 5% ceiling.
- Wholesale AITER/vLLM kernel ports: wrong GPU generation and tensor format.
- Norm/router and small elementwise fusion: worthwhile only after the measured
  dominant work; published isolated effects are below the first-pass threshold.
- Speculative decoding and machine configuration: outside this task.

## Validation and promotion protocol

Every candidate follows the same sequence:

1. Build once, copy the binary separately to both nodes, and verify SHA256 on
   both nodes. The filesystems are not shared.
2. Launch coordinator and worker concurrently; do not serialize startup around
   waiting for the coordinator port.
3. Run alternating control/candidate A-B-B-A decode legs with identical prompt,
   seed, token count, provider, and TP settings. Repeat if the measured effect
   is near the noise floor.
4. Compare both per-rank timings and end-to-end generation t/s. A faster rank
   that leaves the slower rank unchanged is not a TP win.
5. Validate deterministic greedy tokens over several prompts. For kernels that
   intentionally reassociate F32 sums, also compare tensor error and a longer
   generation; do not claim bit-exactness.
6. Rerun long-prompt prefill to reject decode wins that regress the published
   prefill path.
7. Keep a kill switch until the two-node result and stability run pass. Kernel
   switches must be transport-neutral; provider switches must leave Mellanox
   dispatch untouched.

## First implementation result: packed Q8 low projection

The same-stream event split measured the TP attention-output projections on
both ranks at about 0.237 ms for low and 0.223-0.226 ms for expand. This made
the low projection large enough to clear the 5% end-to-end threshold with a
substantial kernel improvement.

The implemented gfx1151 path keeps two output rows per wave but divides each
wave into four eight-lane groups. Each group consumes one 32-value Q8_0 block,
so the 4,096-wide dot product executes 32 block iterations instead of 128.
The final hardened form uses aligned `float4` activation reads, explicit byte
weight reads, and per-lane scale loads. It is selected only for the exact
TP=2 shape (`group_dim=4096`, `rank=1024`, four local groups) on gfx1151.
Other shapes and architectures retain the original kernel.

Measured results:

- low projection: 0.236-0.237 ms control versus 0.096 ms hardened candidate,
  about 59% faster;
- controlled production decode: 11.15 to 11.78 t/s, +5.7%;
- hardened 1,000-token stability observation: 11.93 t/s, no stalls, and zero
  OdinLink provider fallbacks;
- long 9,881-byte prefill regression run: 139.61 t/s;
- first-layer low tensor relative RMS error versus the reference accumulation:
  2.78e-7, with 7.01e-7 maximum absolute error.

An earlier version used width-8 shuffle scale broadcasts and unaligned
`char4` weight loads. It passed math and code repeat tests but diverged on a
multilingual greedy repeat, while the kill-switch control was stable. That
version was rejected. Removing those operations made all three prompt pairs
byte-identical. The production kernel intentionally reassociates FP32 sums and
is therefore numerically validated, not described as bit-exact.

The kernel defaults on only for the guarded gfx1151 shape and can be disabled
with `DS4_ROCM_DISABLE_ATTN_OUT_LOW_PACK4=1`. It contains no transport or
provider branching, so Mellanox RDMA and generic verbs behavior are unchanged.

The next measured target is attention-output expand, now about 0.225 ms per
layer invocation. Any expand redesign must repeat the same opt-in, numerical,
long-run, prefill-regression, and transport-neutral promotion protocol.
