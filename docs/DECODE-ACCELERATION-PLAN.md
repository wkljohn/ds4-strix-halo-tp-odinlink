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
(15 -> 16.6 t/s), not a multiplicative jump - and a base-model drafter is
already documented as not transferring well to this abliterated trunk
(`PLAN.md:202-205`), so acceptance rate must be measured before assuming
any of that 11% carries over here. When this task starts, do it after
Stage 0-1 above, since a faster greedy baseline makes any DSpark
accept/reject economics easier to read.
