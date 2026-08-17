# Next prefill levers - 3-round Codex research (2026-08-05)

Status: research only, nothing implemented. Dispatched Codex to do a
structured 3-round sweep (broad inventory -> filter/deepen -> rank) across
local vLLM upstream checkout (a real git
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

**CONCLUDED (2026-08-05, same day): NOT worth pursuing as an automatic
production feature.** Full plan → Phase A harness proof → Codex review →
corrective fix → skewed/Zipf stress test → final Codex assessment, all
hardware-validated. See `ds4-upstream`'s down-projection harness work
(commits `ba310ca`, `45a6cb8`, `1db4c9a` in `ds4-strix-halo-tp`) and
[[ds4-wmma-geometry-phase-a-down]].

Summary of the full arc: J=32 down-projection is numerically correct
(bit-identical where the code path is shared, within-tolerance where it
isn't) and, at UNIFORM 6-expert synthetic distributions, gives a clean
+9.1% speedup when selected by the proposed heuristic
(`pair_count > 16*n_total_expert`). But once realistic, non-uniform
(Zipf-distributed, 256-expert) timing cases were added, the selector-filtered
result dropped to only +3.2% with a real regression: a Zipf(s=1.2) case
whose AVERAGE pairs/expert clears the threshold (18, just 12.5% over 16)
is actually ~11.9% SLOWER with J=32, because most of its 256 experts
individually hold far fewer than 16 pairs each - a few "hot" experts drag
the average up while production (which picks ONE tile width for the whole
batch) wastes tile capacity applying J=32 to all the many small ones.
Codex's assessment: no better selection statistic is derivable from the
same cheap, sync-free information the plan required (`pair_count` and
`n_total_expert` alone can't distinguish a uniform distribution from a
skewed one with the same average) - a real fix would need per-expert
histogram data, which requires the device-to-host synchronization the
plan specifically wanted to avoid.

**Final verdict: keep J=16 as the unconditional production default.**
J=32 remains available as a validated, correct, opt-in experimental path
for explicitly pre-confirmed workload shapes, but automatic selection is
not safe to ship. This closes out the WMMA-geometry lever as the top
priority - #2 below (OdinLink transport tuning) or the not-yet-implemented
FFN-gate-overlap design are the next candidates if pursuing further
prefill work.

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

## Post-optimization closeout (2026-08-06)

This ranking was revisited after compact-Q8 token reuse and the coalesced
Q4_K MoE epilogue raised the cache-free reference to 167.73 t/s. The new
epilogue reduced its own 43-layer trace contribution from 18.42% to 0.32%,
so the remaining Amdahl ceilings are materially smaller than those used in
the original ranking.

The generic attention projection-B WMMA kernel remained a plausible target
at 9.75% of the follow-up trace. A standalone bit-exact harness now lives at
`scripts/q8_batch_wmma_token_tile_bench.cu`. It compares the production
64-token tile with a 128-token tile over the major model shapes before any
production dispatch change:

| 2,048-token Q8 projection | Tile 64 | Tile 128 | Change |
|---|---:|---:|---:|
| q_b, 1,536 -> 32,768 | 22.290 ms | 19.599 ms | -12.1% |
| indexer q_b, 1,536 -> 8,192 | 5.024 ms | 4.854 ms | -3.4% |
| q_a, 4,096 -> 1,536 | 2.778 ms | 3.336 ms | +20.1% |
| compressor, 4,096 -> 1,024 | 2.213 ms | 1.916 ms | -13.4% |
| KV, 4,096 -> 512 | 1.116 ms | 1.590 ms | +42.4% |

The two tiles produced zero bit mismatches in 15,440 deliberately
non-aligned outputs. Selective use on q_b/compressor has only about a 1%
end-to-end ceiling because the 12--13% gains apply to a kernel family that
is itself under 10% of the trace. A first repeat recorded a 360.97 ms q_b
page/system outlier; the clean repeat above and the initial 19.38 ms result
agree. Raw output preserves both rather than deleting the outlier.

The other candidates do not clear the required credible 5% bar:

- indexed plus static attention totals about 8.4%, requiring an implausible
  >59% combined reduction merely to reach 5% end to end;
- the already-validated J=32 Q4_K result remains unsafe for automatic use:
  +9.1% on uniform routing became +3.2% on realistic skew and included an
  11.9% regression;
- chunking the exact FFN exchange after routed-MoE completes can overlap only
  the sub-1% HC expansion; overlapping the transfer with down projection
  would require new per-token producer-readiness and creates a silent partial-
  sum correctness hazard;
- a provider-only RDMA_WRITE wrapper does not bypass OdinLink's queued copy
  or receive placement. A worthwhile version requires new driver-level direct
  placement, not an inference-side verb substitution.

Conclusion: no remaining exact, low-risk prefill change has a credible >=5%
ceiling in the current profile. Prefill work should resume only with new
profile evidence or a real OdinLink direct-placement primitive. The active
optimization focus moves to decode.

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
