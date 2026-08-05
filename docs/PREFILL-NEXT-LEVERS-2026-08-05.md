# Next prefill levers - 3-round Codex research (2026-08-05)

Status: research only, nothing implemented. Dispatched Codex to do a
structured 3-round sweep (broad inventory -> filter/deepen -> rank) across
local vLLM upstream (`/home/wkljohn/Desktop/cc/vllm-upstream`, real git
checkout, one caveat: partial/promisor clone with some scheduler/RDNA
kernel blobs unfetchable), local llama.cpp upstream + project forks
(`llama.cpp-upstream-latest`, `llama.cpp-strix-halo-RCCL-RDMA`,
`llama.cpp-tq3-hip-new`), and the web (network access confirmed available
to Codex in this environment). This was AFTER two fixes already landed this
session (WMMA down-proj, Q8_F16 cache) and after the not-yet-implemented
FFN-gate-overlap design (`FFN-GATE-OVERLAP-RESEARCH.md`) - explicitly
instructed not to re-propose either as new.

Current bottleneck shape at dispatch time: routed_moe ~50% of measurable
per-layer compute, output_proj ~14%, q_path ~11%, attention/SDPA ~8%; the
TP FFN all-reduce gate separately measured at ~39% of total prefill wall
time, currently fully blocking (zero compute/communication overlap).

## Ranked candidates (highest confidence/impact first)

### 1. Dynamic gfx1151 Q4_K WMMA geometry, not a hardcoded J=16 tile

ds4's Q4_K WMMA path forces `routing_tile_m=16` unconditionally
(`ds4_rocm_moe_launch.cuh:876`, confirmed again this round). llama.cpp's
own RDNA3.5-specific Q4_K table spans J=16 through 128 with per-shape
thread count, LDS, and occupancy tuning
(`llama.cpp-upstream-latest/ggml/src/ggml-cuda/mmq-config-rdna3-5.cuh:116`),
selected at runtime by picking the smallest tile count that fits
(`mmq.cuh:1469`), with per-expert bounds to skip empty/exhausted tiles
(`mmq.cuh:1006`). ds4 already has precedent for this style of dispatch in
its NON-Q4K path (an existing LDS/occupancy-driven 4-vs-8 tile selector,
`ds4_rocm_moe_launch.cuh:860`) - the Q4_K WMMA path just never got the same
treatment. A real-world gfx1151 Q4_K MoE tuning report
([llama.cpp issue #21284](https://github.com/ggml-org/llama.cpp/issues/21284))
shows ~20% aggregate prefill improvement, though that bundles several
changes (reduced VGPR pressure, different geometry, faster intrinsics), not
an isolated geometry estimate.

**Confidence:** medium-high some gain exists (routed_moe is still the
single largest measured stage); medium on magnitude. **Complexity:**
medium-high, kernel-level (multiple tile instantiations + a real
dispatch selector, following the existing non-Q4K precedent).
**Validation bar:** should be arithmetic-neutral in principle, but new
tiling can expose routing/tail/accumulation bugs - exact logit/selected-
token comparison required across uneven expert-bucket distributions and
prompt lengths not divisible by any candidate tile width.

### 2. OdinLink completion-signaling cadence and event-driven worker wakeup

Attacks the 39% FFN-gate cost from a different angle than the
not-yet-implemented row-chunk overlap design: reducing the actual
wire+wait time itself, not overlapping it with compute. ds4 already has
windowed bulk RDMA (128 KiB messages, 32 slots) - NOT a new finding, don't
re-propose plain chunking. What's new: every bulk send is currently fully
signaled (`ds4_tp.c:1298`) - selective completion signaling could reduce
CQ/provider overhead while keeping a safe slot-reuse watermark; the
128 KiB/32-slot policy could be re-derived from the measured ~9.2 Gb/s
real OdinLink bandwidth (not the advertised 80 Gb/s) rather than an assumed
value; and a sibling project fork
(`llama.cpp-strix-halo-RCCL-RDMA/odinlink/FINDINGS.md:858`) documented a
busy-poll worker-wakeup CPU-stealing problem and a failed naive fix -
worth reading before attempting an eventfd/condvar-based wakeup here.

**Confidence:** medium - real evidence that provider behavior and
busy-spin matter, but no measurement yet of how much of the current
~168.6 ms/layer gate cost is signaling overhead specifically vs. genuine
transfer time. **Complexity:** medium, transport/host-scheduling.
**Validation bar:** high - this project's RDMA provider has a real history
of reordering/dropping completions silently; needs byte-level full-duplex
stress plus exact output comparison, not a throughput-only test.

### 3. BF16/F16 wire payload for the FFN gate exchange (real numerical risk)

Halve the exchanged bytes for the `n_tokens x DS4_N_EMBD` F32 partial sum
(`ds4.c:29127`) by rounding through a lower-precision wire format before
transfer - pattern seen in a sibling fork's collective
(`llama.cpp-tq3-hip-new/ggml/src/ggml-cuda/allreduce.cu:79`) and AMD's
QuickReduce compressed-collective approach. Directly targets the
low-bandwidth (~9.2 Gb/s measured) Thunderbolt link.

**Confidence:** medium for wire-time reduction; low-medium for an
acceptable accuracy/perf tradeoff overall - pack/unpack kernel overhead and
the non-wire portion of gate cost determine the real payoff.
**Complexity:** medium-high, and unlike every other fix this session, this
one is NOT expected to be bit-exact - it changes the actual computation
(partials get rounded before the canonical add). **Validation bar:**
highest numerical risk of this whole list - needs both rank assignments
tested, long-context drift checks, logit deltas (not just top-1 token
match), and should stay strictly opt-in unless an explicit accuracy
tolerance is agreed on first. Do not treat this the same way as the
row-split fix's "0.000000 diff" bar - that bar is the wrong success
criterion here by design.

### 4. Producer-ready transfer from routed-MoE down-projection (highest risk, not recommended yet)

vLLM fuses GEMM output directly into reduce-scatter as producer tiles
become ready
(`vllm-upstream/vllm/compilation/passes/fusion/collective_fusion.py:341`).
Applied to ds4, this would start the gate exchange DURING routed-MoE
compute rather than after the full layer output exists - a full stage
earlier than the row-chunk design in `FFN-GATE-OVERLAP-RESEARCH.md`.

**Confidence:** medium-low. ds4's expert-sorted top-6-per-token
accumulation makes "is this row's final value actually ready" a hard
question to answer safely - a row isn't done just because one expert's
tile finished; it needs ALL of that token's owned expert contributions
merged first. **Complexity:** very high - combined kernel + scheduler +
TP protocol redesign. **Explicitly flagged by Codex as higher-risk than
anything implemented this session**: publishing a row early would produce
PLAUSIBLE BUT WRONG output (a silent-correctness bug, this project's worst
failure mode), not a crash. Needs per-row readiness assertions and stress
on skewed expert distributions before even attempting. Not recommended as
a next step without a much stronger justification than the levers above.

### 5. Fuse canonical-add with HC-expansion inside released chunks

Small, safe, low-value on its own: AITER-style fused collective epilogues
(`vllm-upstream/vllm/compilation/passes/fusion/allreduce_rms_fusion.py:1192`)
suggest fusing the post-gate add directly with HC expansion per chunk. Real
and implementable, but HC compute is already measured under 1% of prefill
time - this only matters as a small addition on top of the row-chunk
overlap design once THAT is built, not as a standalone win.

## Overall take (Codex's own synthesis)

\#1 (WMMA geometry) is the strongest next lever - closest hardware/workload
match to a real published result, attacks the still-dominant routed_moe
share. #2 (transport tuning) is the best incremental attack on the 39%
gate cost specifically. #3 (wire compression) has real upside but breaks
this session's "bit-exact or don't ship it" pattern - treat differently.
#4 is explicitly not recommended yet given the silent-correctness risk
class. #5 is only worth doing alongside the existing gate-overlap design,
not alone.

## Sources consulted

Local: `vllm-upstream` (git checkout, 2026-07-29, partial/promisor clone),
`llama.cpp-upstream-latest` (git checkout, 2026-08-02),
`llama.cpp-strix-halo-RCCL-RDMA`, `llama.cpp-tq3-hip-new`.

Web: [llama.cpp issue #21284](https://github.com/ggml-org/llama.cpp/issues/21284) (gfx1151 Q4_K MoE tuning, ~20% aggregate),
[AITER releases](https://github.com/ROCm/aiter/releases),
[AMD Kimi K2.5 fused-MoE tuning](https://rocm.blogs.amd.com/artificial-intelligence/kimi-k2.5-optimize/README.html),
[ROCm/ATOM](https://github.com/ROCm/ATOM) (two-stage compute/comm overlap),
[QuickReduce](https://rocm.blogs.amd.com/artificial-intelligence/quick-reduce-3/README.html) (compressed collectives),
[MoEShard](https://arxiv.org/abs/2503.08467),
[Speculative MoE](https://arxiv.org/abs/2503.04398),
[AMD Strix Halo optimization guide](https://rocm.docs.amd.com/en/docs-7.2.0/how-to/system-optimization/strixhalo.html).
