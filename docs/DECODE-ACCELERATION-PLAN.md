# Decode acceleration plan (TP=2, no DSpark)

Scope: decode throughput only, on the working two-node RDMA TP setup
(DeepSeek-V4-Flash Q4_K, gfx1151 x2). Explicitly **excludes DSpark/MTP
speculative decoding** — that is real, tractable work (a bounded CUDA->HIP
port, see below) but it is a separate task with its own acceptance-rate
risk, tracked independently. This plan is "make the existing greedy decode
path faster," nothing else.

Produced from an independent Codex research pass (2026-08-04), code-read
only — no hardware profiling has been run yet under this plan. Distinguish
throughout: **measured** (recorded in prior docs) vs **code-read-only**
(conclusions from the pinned tree, no new run).

## The roofline this project has been using is stale

`README.md:32` and `ROOFLINE-PROVENANCE.md:23-29` state a 25.1 t/s decode
ceiling from "experts-only" TP sharding (8.91 GiB/node/token), and list
attention head-splitting as a **future, untested** direction that would
raise the ceiling to 36.3 t/s (6.17 GiB/node/token).

That is wrong today. Code read confirms the DeepSeek decode path already
splits attention heads and output groups per rank, unconditionally,
whenever `tp_world == 2`:

- `tp_heads` = half the model's heads, each rank starting at a different
  head offset (`ds4.c:21552-21555`)
- the Q projection reads only this rank's head rows (`ds4.c:21743-21762`)
- head norm / RoPE run over `tp_heads`, not all heads (`ds4.c:21768-21795`)
- the attention kernel gets only `tp_heads`, sink weights offset to the
  rank's range (`ds4.c:22393-22435`)
- the output projection computes only this rank's half, partials exchanged
  and combined at the attention gate (`ds4.c:22624-22693`)
- the ROCm group-slice implementation is real, not a stub — it views the
  owned heads, does the matching low projection, fails closed on
  misalignment (`patches/rocm_tp_compute.inc:9-39`)

So **the applicable ceiling is 36.3 t/s, not 25.1 t/s**. Current decode
(10.46-10.62 t/s measured, `PATCH17-SPIN.md:45-53`) is running at
**~29% of ceiling, not 42%**. The gap to close is bigger than this project
believed, not smaller. There is no "implement the head split" action item
— it's done. What's missing is figuring out where the other ~71% of
per-token time actually goes, and validating that the split doesn't have
a correctness or performance bug hiding in it (it was apparently never
isolated with an on/off switch, so it's never been A/B'd against a
replicated-attention baseline on this exact hardware).

## What's structurally serialized per token

86 lockstep gate exchanges per token (`ds4_tp.c:124-133`), each one a full
stop-and-wait: GPU publishes a sequence number and parks on a host release
flag (`ds4_rocm.cu:522-524`); the service thread notices and calls the
transport synchronously (`ds4_rocm.cu:277-320`); the RDMA callback posts
the send, spins until the peer's completion, posts the next receive, and
only then returns (`ds4_tp.c:1011-1079`). The receive window hides
receive-*posting* setup but does not pipeline a layer's compute past its
required peer sum (`ds4_tp.c:1031-1037,1061-1077`). Patch 17 (already
shipped) cut service-thread detection latency by ~23% but did not remove
this per-gate lockstep dependency itself.

Transport bandwidth is very likely *not* the bottleneck here (unlike
prefill, where the CPU-staging-copy fix just landed a real ~25% win) -
the 86-exchange structure means the cost is plausibly bubbles and rank
skew, not bytes on the wire. This has **not been measured** - it's the
leading hypothesis from code structure, not a result.

## Stage 0 — measure before touching anything (diagnostic only, not a speedup)

Two existing profiling facilities, neither yet used together correctly for
decode:

- `DS4_ROCM_DECODE_STAGE_PROFILE`, filterable to one layer via
  `DS4_ROCM_DECODE_STAGE_PROFILE_LAYER` (`ds4.c:26647-26655`) - the
  decode-specific stage profiler (do not confuse with the broader
  `DS4_ROCM_LAYER_STAGE_PROFILE`, `ds4.c:26636-26644`).
- `DS4_ROCM_MOE_DECODE_PROFILE` - GPU-event timing around the one-token
  routed-MoE gate/up, quantization, and down work
  (`rocm/ds4_rocm_moe_launch.cuh:13-46,80-90`).

**Hard requirement carried over from the prefill work**: enable on BOTH
ranks. This project already burned real time on exactly this mistake once
- coordinator-only profiling produced a nonsensical 97.7%-in-one-stage /
483s-of-profiled-time-for-a-2-minute-run result, because the stage
boundary syncs the device and makes the unprofiled rank's peer wait at the
gate (`PREFILL-PROFILE.md:16-29`). Symmetric-only, no exceptions.

Also be aware the profiler is invasive, not free: enabling it flips
`decode_stage_profile` true, which **disables fused decode paths**
(`ds4.c:21558-21564`), and several overlap/deferred-reduction paths
explicitly require profiling off to run at all (`ds4.c:22946-22960,
23231-23255,23286-23305`). Symmetric stage totals can point at a broad
area (attention vs FFN vs router vs MoE); they cannot be read as an
uninstrumented timing decomposition, and a stage's measured time can
still include peer-wait bleed from the *previous* stage's gate (the same
caveat prefill's `hc_post` already carries, `PREFILL-PROFILE.md:47-49`).

**Command** (one layer first, not all 43):

    DS4_ROCM_DECODE_STAGE_PROFILE=1 \
    DS4_ROCM_DECODE_STAGE_PROFILE_LAYER=20 \
    DS4_ROCM_MOE_DECODE_PROFILE=1

on both coordinator and worker. >=500 generated tokens (100-token runs are
documented too noisy below ~15% changes). Record: wall-clock token time;
per-rank stage totals; the slower rank per stage; `attn_output`,
`attn_hc_post`, `router`, `routed_moe`, `shared_gate_up`, `shared_down`,
`ffn_hc_post` (stage boundaries at `ds4.c:21635-21649, 22695-22725,
22772-22823, 23366-23455, 23513-23653, 23687-24001`).

**Stage 0b (less invasive, do second):** instrument the service thread's
three intervals separately - GPU-arrival detection latency, transport
callback duration, release-to-next-GPU-arrival - to split "service thread
woke up late" from "RDMA/peer actually took this long" from "GPU had real
compute between gates." No code-read-only analysis can honestly assign the
missing ~71% among these without this measurement.

**Go/no-go:** this stage doesn't have a go/no-go, it's the input to
picking Stage 1's target. Do not skip to a kernel or overlap change before
this exists - the two prior kernel-level bets in this project both looked
right on paper and lost on measurement; decode's gap being "obviously" the
86-gate lockstep is the same kind of paper-plausible guess until measured.

## Stage 1 — attack the measured dominant cost, one thing at a time

Pick exactly ONE target from Stage 0's result. Candidates, ranked by prior
probability of a real win (all estimates, not measurements - Stage 0
determines which of these is even worth attempting):

| Priority | Direction | P(real >=5% win) | Plausible gain | Effort |
|---|---|---|---|---|
| 1 | Reduce rank-skew / gate-idle time; overlap independent compute before a gate | 65% | 5-20% | medium/high |
| 2 | Tune the dominant one-token GEMV/DP4A stage Stage 0 identifies | 60% | 5-15% | medium |
| 3 | Skip unowned routed experts instead of compute-and-mask both ranks evaluating all 6 selected experts | 40% | 3-8% | medium |
| 4 | Recover selected/shared-expert overlap under TP (currently gated off: `tp_world < 2` required, `ds4.c:23286-23305`) | 35% | 3-10% | high correctness risk |
| 5 | Per-token allocation cleanup | <15% | <3% | low/medium - only pursue if Stage 0 shows allocator time, none found by code read so far |

Explicitly **not on this list**: WMMA for decode's MoE kernel. Decode's
one-token MoE path is already GEMV-shaped DP4A work sized for batch=1
(`rocm/ds4_rocm_moe_launch.cuh:2077-2116,2185-2247`) - there is no evidence
a matrix-core kernel helps a single-row GEMV, and the separate prefill WMMA
investigation's own Stage -1 finding (llama.cpp's WMMA-MMQ has a
*documented regression* on RDNA3 for at least one quant type) is a reason
for skepticism, not optimism, if anyone proposes it later.

Each Stage 1 attempt: implement behind a default-off env var kill switch,
validate correctness first (greedy output vs known-good baseline, exact
match required - this is the same codebase that had a silent-wrong-output
bug from an accumulation-semantics assumption, so "looks fluent" is not a
correctness check), then measure >=500 tokens, symmetric profiling to
confirm the targeted stage actually shrank and nothing else grew to
compensate.

## Stage 2 — re-verify the head split is actually helping, not just present

Not urgent, but on the list because "the split is real code" was already
one documentation error - "the split is measurably beneficial and bug-free
under TP" has never been checked either. Smallest test: force replicated
(un-split) decode attention behind a temporary diagnostic switch, compare
output correctness and throughput against the current split-on path,
>=500 tokens. Implementation effort for the split itself is ~zero (it
exists); this stage is validation, not new work. Low priority relative to
Stages 0-1 since there's no specific reason to suspect it's broken, only
that it was never isolated and measured on its own.

**Quick check done 2026-08-05 (NOT the >=500-token validation above - a short,
single-rep run, treat as a weak signal only):** `DS4_GLM_TP_HEAD_SPLIT_MIN`
default (64, engaged - this model has exactly 64 heads) vs forced to
99999999 (full replication), same short prompt, `--temp 0`:

    split (default):  prefill 27.45 t/s, generation 10.60 t/s
    replicated:        prefill 16.43 t/s, generation 10.78 t/s

**Decode is within noise between the two (10.60 vs 10.78, ~2%) - no clear win
or regression from the split on this quick pass.** Prefill numbers are not a
clean comparison here (confounded by `DS4_TP_PREFILL_SPLIT_MIN=999999`
forcing sequential prefill on an already-short/noisy prompt - prefill has
ranged 16-38 t/s across unrelated runs this session on similar short prompts).
Does not block or resolve the >=500-token validation above; still open.

## Explicitly deferred - DSpark/MTP (separate task, not in this plan)

Tracked separately per direct instruction. One-line status for context:
ROCm is missing `ds4_gpu_attention_noncausal_raw_batch_heads_tensor`
(routed to a hard-fail stub, `ds4_rocm_unavailable.cu:9-23`), but CUDA
already has a complete reference kernel with a CPU-verification mode
(`ds4_cuda.cu:13500-13621`) - this is a bounded HIP port of existing logic,
not new kernel research. Realistic expectation from this project's own
llama.cpp precedent on the same checkpoint/hardware: **~11% uplift**
(15 -> 16.6 t/s), not a multiplicative jump. Acceptance rate must be measured
against the exact trunk before assuming any of that 11% carries over here.
When this task starts, do it after
Stage 0-1 above, since a faster greedy baseline makes any DSpark
accept/reject economics easier to read.

## Stage 0b RESULT: the missing time is GPU compute between gates, not RDMA (2026-08-05)

`DECODE-PROFILER-STALL.md`'s stall blocked the full stage profiler
indefinitely, so Stage 0b (ds4-upstream commit `ca99a31`) added separate,
lighter instrumentation directly in the TP service thread
(`ds4_rocm.cu:328-357`, `ds4_tp_pump`/`ds4_tp_service_thread`) - opt-in via
`DS4_TP_SERVICE_INTERVAL_PROFILE=1`, `clock_gettime(CLOCK_MONOTONIC)` only,
zero device synchronization, so it does not reproduce the stall. Implemented
by Codex as a guarded background agent, reviewed and one correctness bug
fixed (a stale-timestamp reuse on back-to-back gate hits) by Claude Sonnet 5
before hardware validation.

**Validated on both live nodes**: a 600-token run with profiling on
completes cleanly (the old profiler died at token 2-3) with throughput
matching the profiling-off baseline (10.87 vs 10.88 t/s generation) -
confirms the instrumentation itself isn't perturbing the measurement.

**Result, symmetric on both ranks** (channel 0 = row gates, the decode-
relevant channel; 86 gates/token):

| interval | rank 0 | rank 1 | share |
|---|---|---|---|
| detect_upper_bound (poll latency) | 12.1us | 5.8us | ~1% |
| callback (RDMA transport round-trip) | 111.8us | 132.0us | ~11% |
| release_to_arrival (this rank's own GPU compute between gates) | 958.4us | 937.6us | **~88%** |

The three intervals sum to ~1069us/gate; times 86 gates/token = ~91.9ms,
matching the measured ~92ms/token (10.88 t/s) almost exactly - a
self-consistent measurement, not just plausible-looking numbers.

**This overturns the plan's leading hypothesis.** The "86 lockstep gate
exchanges... bubbles and rank skew" framing (top of this document) assumed
transport/gate-wait time was the likely dominant cost. It is not - RDMA
transport is only ~11% of per-token time on both ranks. ~88% is each rank's
own GPU compute between successive gates. That directly promotes **Stage 1
priority #2** ("tune the dominant one-token GEMV/DP4A stage Stage 0
identifies") over priority #1 (rank-skew/gate-idle overlap) - there is
little evidence of idle bubble time to reclaim; the GPU appears busy
computing for the large majority of the token budget, not waiting.

**Not yet done** (at the time the above was written): this measured
aggregate per-gate compute time, not WHICH kernel/stage within that ~958us
dominates. Resolved below without needing the stalling full profiler.

## Stage 0b extended RESULT: attention dominates, not routed MoE (2026-08-05)

The stage-profiler-stall root-cause chase (`DECODE-PROFILER-STALL.md`) did
not produce a reliable fix. Instead of continuing that path or building new
GPU-event instrumentation, Stage 0b's own machinery was extended cheaply
(`ds4-upstream` commit `8efef88`): row gates alternate ATTN (gate=0) / FFN
(gate=1) within a layer, so `release_to_arrival` was bucketed by which gate
is arriving. The interval ending at an FFN-gate arrival is
router+routed_moe+shared_ffn compute; the interval ending at an ATTN-gate
arrival is the next layer's attention compute. Zero new synchronization,
zero new stall risk, same opt-in `DS4_TP_SERVICE_INTERVAL_PROFILE=1`.

**Result, symmetric on both ranks, self-consistent** (average of the two
matches the previously-measured aggregate ~958-961us almost exactly):

| stage | rank 0 | rank 1 | share |
|---|---|---|---|
| attention (attn gate arrival) | 1202.7us | 1185.3us | **~63%** |
| FFN/MoE (ffn gate arrival) | 719.4us | 696.2us | ~37% |

**This reframes Stage 1's target.** All of this project's prior kernel
work (the Q4_K integer-WMMA port, both gate/up and the down-projection fix)
targeted the routed-MoE path, and only helped prefill - decode's routed-MoE
work is GEMV-shaped (batch=1) and structurally cannot use WMMA (settled
earlier, not revisited). But decode's actual dominant per-layer cost is
**attention** (DeepSeek's MLA compressor/indexer chain - compressor_proj,
compressor_update, compressor_quantize, indexer_compressor_proj/update/qat,
q_path, kv_path, attn_output, per `ds4.c`'s decode stage names), not MoE.
Stage 1 priority #2 ("tune the dominant one-token GEMV/DP4A stage") should
now specifically target attention's decode path, not routed-MoE decode
kernels - the latter's share (~37%) makes it a secondary target at best.

**Not yet done**: this still only measures the two gate-delimited halves of
each layer, not which SPECIFIC sub-stage within the attention half (MLA
compression vs indexer vs RoPE vs the Q/K/V/output projections themselves)
dominates that ~1.2ms. That finer breakdown would need per-substage timing
within the attention half specifically - the same gate-bucketing trick
does not go finer than the two gates the code already provides, so this
would require either the stalling stage profiler (unresolved) or new
non-synchronizing GPU-event instrumentation scoped just to the attention
sub-stages.

## Stage 0c: attention sub-stage breakdown - one concentrated hotspot, not spread thin (2026-08-05)

Built the non-synchronizing GPU-event instrumentation the previous section
called for (`ds4-upstream` commit `0938a0d`): `hipEvent_t` recorded at each
existing decode stage boundary inside the attention half, a 16-slot reusable
event pool, deferred `hipEventQuery`-gated readback - never
`hipDeviceSynchronize`, so it cannot reproduce the stage-profiler stall.
Implemented by Codex (two earlier dispatch attempts were blocked by a
read-only sandbox from a direct CLI invocation; routing through the proper
guarded-implementer channel fixed it). Reviewed and one correctness bug
fixed before validation: `hipEventQuery` returning `hipErrorNotReady` (a
normal, expected value) was never cleared via `hipGetLastError()` - this
project already shipped that exact bug class once (the
`hipMallocSignalMemory` latched-error issue), and left uncleared here it
could have spuriously failed any of the hundreds of unrelated
`cuda_ok(cudaGetLastError(), ...)` call sites throughout the exact
compressor/indexer/attention kernels this profiler instruments.

**Validated on both live nodes**: no spurious failures (despite frequent,
by-design, harmless sample drops - roughly half of attempts get dropped
when the 16-slot pool wraps before the oldest slot's events complete;
n=11000-12692 samples over a 600-token run is still a large, stable
sample), coherent output preserved, throughput unaffected (10.68 vs 10.80
t/s decode).

**Result** (rank 0, means over ~12700 samples):

| sub-stage | mean | share of measured total |
|---|---|---|
| kv_path | 18us | 2% |
| compressor_proj | 131us | 15% |
| compressor_update | 17us | 2% |
| compressor_indexer | 35us | 4% |
| attn_inv_rope | 60us | 7% |
| **attn_output** | **582us** | **68%** |
| attn_hc_post | 14us | 2% |

**One concentrated hotspot, not many small kernels.** `attn_output` is
~4.4x the next-largest stage (`compressor_proj`). This is genuinely useful
- it means Stage 1 doesn't need to fuse a long chain of small ops; it needs
to understand and speed up one specific window.

**Caveat, important**: the measured sum (~856us) is meaningfully less than
the coarser gate-bucketed attention-half measurement from the previous
section (~1.2ms). The gap is very likely q_path/RMSNorm time before the
`start` event (not yet instrumented) plus, more importantly: **the
`attn_output` window as coded (`ds4.c:22452-22695`) spans BOTH the
TP-sliced attention-output projection matmul AND the ATTN TP-gate exchange
itself** (`ds4_gpu_tp_gate_encode` is called partway through this window,
`ds4.c:22681`, before the window's closing event). The 582us figure could
be dominated by either the actual matmul compute, or by this rank waiting
on its peer's gate send, or some mix - this instrumentation cannot yet
tell those apart, since the gate wait happens on a different HIP stream
(`g_tp_stream`) than the events recorded here (stream 0, matching where
the compute kernels run).

**Next step, not yet done**: split `attn_output` into (a) time up to the
gate encode call and (b) time from the gate encode to the window's end, by
adding one more GPU event immediately after `ds4_gpu_tp_gate_encode`
returns. This determines whether Stage 1 should target the matmul kernel
itself or the gate-exchange/overlap structure - a materially different fix
in each case. Do this before attempting any actual kernel change here;
picking the wrong one wastes effort and risks a correctness bug in dense
TP+quantization code, exactly the kind of thing that already went wrong
once this project (the down-projection WMMA tile-width bug).

## Stage 0d: initial method was unsound - CORRECTED below (2026-08-05)

**This section originally claimed a clean pre-gate/post-gate split using a
GPU event recorded right after `ds4_gpu_tp_gate_encode()` (commit
`a1f4b6a`: pre-gate ~566us, post-gate ~6us, concluding "real compute, not
gate-wait"). That specific split was methodologically unsound and its
exact numbers should not be trusted** - `g_tp_stream` is created via plain
`hipStreamCreate` (a *blocking* stream, `ds4_rocm.cu:622`), and this
project's TP design intentionally relies on legacy default-stream
synchronization for correctness (`ds4_rocm.cu:239`'s own prior comment).
Under that semantics, a stream-0 event recorded right after the gate
encode call absorbs the ENTIRE wait for `g_tp_stream`'s pending
`WaitValue` to clear - the "pre-gate" number was contaminated with hidden
gate-wait time, and "post-gate" only looked clean because by the time that
first stream-0 event fires, the wait is already satisfied and nothing
downstream needs to wait again. Caught via direct user question ("did you
measure the bottleneck properly") - a good reminder to interrogate stream-
ordering assumptions before trusting a GPU-event split across any boundary
that involves a second stream.

**Corrected measurement** (`ds4-upstream` commit `5561c3d`): extended the
ALREADY-SOUND Stage 0b service-thread mechanism (pure `clock_gettime()` on
the CPU side polling thread, no GPU stream semantics involved at all -
this is what actually makes Stage 0b's original findings trustworthy) with
gate-bucketing on the `callback` stat (the real wall-clock cost of a
gate's cross-rank exchange), matching the bucketing already applied to
`release_to_arrival`.

**Validated on both live nodes, symmetric - this time on solid ground**:

| | rank 0 | rank 1 |
|---|---|---|
| `callback_gate=0_attn` (real ATTN-gate exchange cost) | 88.6us | 118.8us |
| `release_to_arrival_gate=0_attn` (real compute before the gate publishes - NOT affected by the stream-ordering issue, since "arrival" = compute finishing, independent of any wait) | 1195.7us | 1176.6us |

**Corrected conclusion: the real gate-exchange cost (~89-119us) is a clear
minority against the ~1177-1196us of real compute preceding it.** The
QUALITATIVE conclusion from the original (unsound) Stage 0d - that
attention's cost is dominated by real compute, not gate-wait - turns out
to still be directionally correct, but on the ORIGINAL evidence it was not
reliably established. The downstream decision to pursue kernel-tuning
(launch-config sweeps) over gate-exchange/overlap restructuring was the
right call, just not for the reason originally given. `release_to_arrival`
itself remains sound throughout (it was never measured via the
cross-stream GPU-event mechanism) - only the FINER `attn_output` pre/post
split from this specific section was affected. Stage 0c's other sub-stage
boundaries (`kv_path` through `attn_inv_rope`) are unaffected too, since no
`g_tp_stream` operation is interposed among them.

## Stage 1 kernel analysis (2026-08-05): TP=2 silently bypasses the existing split-K tuning

Codex research pass (read-only, no changes) traced the exact production
path for this model's shape. Confirmed: `attn_output_a`/`attn_output_b`
are Q8_0 (verified directly from the GGUF header, not assumed), so decode
dispatches to `ds4_gpu_attention_output_q8_tp_tensor`
(`rocm/ds4_rocm_hc_output_launch.cuh:461`), composing:

1. `ds4_gpu_attention_output_low_q8_tensor` - the down-projection to the
   compressed rank dimension.
2. `ds4_gpu_matmul_q8_0_kslice_rows_tensor` - the K-sliced expand back up.

**Finding: the existing specialized split-K fast path in the low
projection (`grouped_q8_0_a_partial16_w32_kernel`, explicitly commented as
matching "the production HIP decode path") is hardcoded to
`group_dim==4096 && rank==1024 && n_groups==8` - the FULL (non-TP-sliced)
shape.** Under TP=2, each rank only sees its local `n_groups=4` (8 total
groups split across 2 ranks), so this condition is false and the kernel
silently falls through to a more generic path
(`grouped_q8_0_a_f32_sharedx_rows_w32_2row_kernel`) with only 64
workgroups on Strix Halo's 40 CUs (~1.6 workgroups/CU). This specialization
predates the TP wrapper and was never updated to also match the TP-local
shape - a real, concrete gap, not a vague "maybe slow" guess.

**Two candidate fixes identified, different risk profiles:**

1. **Workgroup-size sweep (near-zero correctness risk)** - both the
   currently-firing low kernel and the expand kernel
   (`matmul_q8_0_f32_sharedx_warp_rows_w32_kernel`) use maximum 1024-thread
   workgroups (64 and 128 workgroups respectively on 40 CUs). This is pure
   launch-configuration tuning - same math, same accumulation order, same
   output bit-for-bit - just parallelism structure. Try this first.
2. **Generalize split-K to `n_groups==4`** (medium confidence, real
   correctness risk) - the kernel accepts arbitrary group counts in
   principle, but split-K changes the accumulation order, which the
   existing code comments explicitly warn can "cross FP8 KV midpoints in
   downstream layers." Requires exact greedy-output equivalence
   validation, not "numerically close."

**Explicitly rejected, with real math**: fusing the low and expand kernels
to avoid materializing the intermediate to global memory. The intermediate
is only 16KiB against >35MB of unavoidable Q8_0 weight reads per layer -
even ideal fusion saves single-digit microseconds against a measured 566us
window. Also no block-local fusion is possible without replicating the
full low-projection per output-row block. Not worth pursuing.

**Also checked and ruled out**: no material redundant compute across TP
ranks - each rank computes distinct groups and consumes the matching
K-slice, already minimal.

**If neither of the two candidates helps**: the next evidence-backed
target is `compressor_proj` (~131us, the second-largest measured
sub-stage), not further attention-output work.

Next step: implement and validate the workgroup-size sweep first (the safe
one), on live hardware, with exact-output-equivalence + real timing
comparison, before considering the split-K generalization.

## Workgroup-size sweep RESULT: no measured win (2026-08-05)

Implemented (`ds4-upstream` commit `6c15c68`):
`DS4_ROCM_ATTN_OUT_LOW_THREADS` / `DS4_ROCM_ATTN_OUT_EXPAND_THREADS`
(1024 default / 512 / 256), pure launch-configuration change, zero effect
on math - both kernels already derive rows-per-block from `blockDim.x`.
Default reproduces current behavior exactly.

**Measured on both live nodes: no measurable decode speedup.** All four
configs tested (low512, low256, expand512, expand256) landed at
10.42-10.70 t/s generation, indistinguishable from the (corrected)
baseline of 10.89/10.93 t/s across two reruns. (One initial baseline
reading of 6.72 t/s was investigated and confirmed to be a one-off fluke -
rerunning the identical unset-env-var config twice gave 10.89 and 10.93
t/s, matching this project's established historical baseline; always
rerun an anomalous baseline before trusting it as the comparison point.)

Honest result: the occupancy theory (64 workgroups is too few for 40 CUs)
did not translate into a measured win here. Either the kernel isn't
occupancy-bound at this shape, or the actual bottleneck is elsewhere
(memory bandwidth already saturated, or launch/sync overhead dominates
regardless of block size). Not pursuing further variations of this
specific lever.

**Remaining candidate, then DISQUALIFIED (2026-08-05): the split-K
generalization to the TP-local `n_groups==4` shape.** Before dispatching
an implementation, traced the kernel body directly and found it's already
generic with respect to `n_groups` (the hardcoded `4096`/`16`/`8` values
in `grouped_q8_0_a_partial16_w32_kernel` are tied to `group_dim`, which
stays 4096 under TP=2, not to `n_groups`) - so the fix would have been a
low-risk guard relaxation, not new kernel math. However, checking WHY the
split-K path is gated (`rocm/ds4_rocm_attention_launch.cuh:1305`,
`!cuda_runtime_config()->disable_splitk_attn_out_low`) found:
`disable_splitk_attn_out_low = !g_quality_mode`
(`rocm/ds4_rocm_runtime.cuh:4769`), and `g_quality_mode` is only ever true
when the CLI's `--quality` flag is explicitly passed
(`ds4_cli.c:1877`) - which this project has NEVER done in any benchmark
run this entire session. **The split-K path has been completely inactive
in every measurement so far, for both the full-model `n_groups==8` case
and TP=2's `n_groups==4` case.** It is a quality-mode-only path (likely
trading speed for numerical fidelity to "the old-HIP backend", per its own
comments), not a speed path our default decode configuration has ever
touched. Generalizing its guard condition to also accept `n_groups==4`
would have had ZERO effect on our actual measured decode performance
unless `--quality` is also passed - a different operating mode entirely,
not comparable to anything benchmarked in this document. **Not pursuing
this lever** - it doesn't apply to the configuration this project actually
runs.

**Both attn_output candidates are now exhausted for the default (non-
quality) decode configuration.** The workgroup sweep measured no win: the
launch config genuinely doesn't move the needle. Split-K doesn't even
apply to our configuration. The generic `sharedx_rows_w32_2row` kernel
currently in use appears to already be near-optimal for quick, low-risk
levers at this shape. Falling back to the plan's own stated next target:
`compressor_proj` (~131us, second-largest measured sub-stage) - though
note its smaller absolute size (131us vs attn_output's 566us) means even a
substantial relative improvement there yields a smaller absolute win
(~2-3ms/token at best for a 50% improvement, vs attn_output's much larger
theoretical ceiling that these two candidates failed to reach). A genuinely
new kernel design for attn_output (not just launch-config or split-K
reuse) remains a possible larger-effort avenue, out of scope for quick
iteration.

## compressor_proj analysis and sweep RESULT: also no win (2026-08-05)

Codex research pass on `compressor_proj` (read-only, applying explicit
skepticism given the attn_output precedent) found one real correction to
this document's earlier assumption: `ds4_gpu_matmul_f16_pair_compressor_store_tensor`
(the function this project believed was the active "fused"
projection+state-store path) is actually a stub on ROCm that discards its
arguments and returns 0 - decode has always fallen back to
`ds4_gpu_matmul_f16_pair_tensor` (projection only) plus a separate
`compressor_store_kernel` launch. Metal has the real fusion; ROCm doesn't.
Porting it is a genuine, concrete opportunity, but its savings fall under
`compressor_update` (~16-17us measured), not the 131us `compressor_proj`
boundary - a small ceiling either way.

For `compressor_proj` itself: launch is narrow (32 or 16 blocks depending
on compression ratio, on 40 CUs), circumstantially similar to
attn_output's under-occupancy story, but Codex explicitly declined to
treat this as evidence-backed, citing the attn_output sweep's failure as
direct caution. Implemented and tested anyway as a cheap experiment
(`ds4-upstream` commit `d2fe1c5`, `DS4_ROCM_COMPRESSOR_PROJ_ROWS`
32default/16/8, same zero-math-effect launch-config pattern).

**Measured on both live nodes: no measurable win**, exactly as predicted.
rows16 (10.79 t/s) and rows8 (10.73 t/s) indistinguishable from baseline
(10.47 t/s) - all within this project's established noise band.

**Status: both of decode's two largest measured attention sub-stages
(attn_output 566us, compressor_proj 131us) have had their quick, low-risk
launch-config levers tested and neither shows a measurable win.** Decode
throughput remains at baseline (~10.4-10.9 t/s) throughout this entire
investigation thread. This is a genuine, hardware-validated result, not a
gap in effort - three separate launch-config experiments (attn_output low,
attn_output expand, compressor_proj) were implemented and measured on live
TP=2 hardware, all negative. The remaining paths are qualitatively
different in kind, not degree: a real kernel redesign for attn_output, or
porting the missing Metal fusion for compressor_update's small ceiling -
both larger-effort undertakings than the quick-iteration levers this
thread has exhausted.
