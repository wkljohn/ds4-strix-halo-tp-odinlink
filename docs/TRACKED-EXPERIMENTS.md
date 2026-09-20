# Tracked DS4 experiments

This workflow covers ordinary inference, transport, kernel, and harness
research. DSpark-specific checks apply only when a DSpark candidate is in
scope. A local checkpoint records work; it is not permission to merge or push.

The original GLM5.3 Flash track requires Antirez's unchanged GGUF (2026-09-13).
The user subsequently selected `GLM-5.3-Flash-Uncensored-Q4_K-ds4.gguf` for a
distinct support and performance scope (2026-09-15). Preserve each artifact's
bytes and quantization. Establish a same-model baseline and quality reference;
the original artifact's promotion does not approve the selected artifact.
Do not substitute a compact/requantized model or requantize weights at runtime.
Compact-model tracks remain paused and unmerged. Mandatory RoCE v2, zero
payload fallback and no persistent expanded-weight cache still apply.

## Before a change

1. Record the named branch, parent commit, exact hypothesis, baseline run tags,
   intended variables, candidate lane, and smallest falsifiable test in a
   dossier under `$DS4_RESEARCH_ROOT`.
2. Preserve dirty work. Attribute changes with scoped commits or dated source
   snapshots; do not infer causality from a large HEAD-to-dirty diff.
3. Check both nodes for existing GPU jobs before reserving a test. Read the
   coordinator's first failure before diagnosing a worker connection refusal.

## Research-track lifecycle

Named research branches are durable tracks. Mark a branch `paused` or
`given-up` when its direction is stopped, but retain its source, evidence, and
status notes. Such a branch stays out of `main` and is not a release baseline.

Periodic audits may identify a useful idea in an older track. Transfer it by
starting a fresh branch from current `main` and recording an adaptation ledger
entry with the donor branch/commit, patch or source hash, decision, and new
commit. Re-test the adapted code under the current gates; never promote by
directly merging a paused or given-up branch.

## Source and build checkpoints

- Commit each scoped source or harness change before publishable timing. For an
  early diagnostic, preserve the complete source snapshot, tracked diff,
  untracked source files, parent, and SHA-256 inventory; label it diagnostic.
- Build into a unique artifact directory. Record the exact command,
  compiler/ROCm identity, defines for all compilation units, build log,
  dependency/helper inventory, and hashes of `ds4` and `ds4-bench-tp`.
- Freeze each artifact after building. Record both nodes' hashes, model and
  drafter fingerprints where applicable, provider, and feature negotiation.
  Rebuilding in place invalidates the old artifact identity.

## Matched tests

1. Run the smallest diagnostic that exercises the change. Pass engine controls
   as trailing `NAME=VALUE` launcher arguments and inspect both rank
   environments in the manifest.
2. Freeze a comparison recipe before examining performance: keep workload,
   prompt bytes, model, toolchain, provider, residency, split, and diagnostic
   state fixed. List the one intended change or an explicitly paired switch.
3. Use one matched pair for smoke. Three alternating headline pairs plus the
   4,096-token cross-disciplinary screen are internal qualification only.
   Promotion starts at the frozen 5/7/9 formal looks and adds the final matched
   context screen at 8,192 tokens or longer plus both DeepSeek regressions.
   Long-context runs use the ordinary no-DSpark control, numerical checks, and
   mandatory zero-fallback RDMA proof.
   Call `scripts/candidate-gate.py begin-pair` before either arm of every
   headline pair. Set `DS4_BENCH_CANDIDATE_ID` for both headline launches; the
   launcher starts the remote supervisor, records its generated run ID and
   declared order before coordinator inference, and refuses a duplicate arm,
   mislabeled order, or arm that contradicts the frozen AB/BA order. Afterward,
   `record-result` seals the CSV state, manifest, both logs, both statuses, and
   completion/cleanup attestations before another arm may start. The producer
   flushes its completion attestation before exposing the timing row. Before
   baseline or candidate launch, its executable must report the SHA-256 of the
   live `ds4_bench.c`; genesis and promotion independently resolve the
   committed bytes. A stale producer is ineligible and consumes no formal arm.
   Finish both arms before beginning the next pair. Invalidate only the latest
   pair when its first journaled arm failed before producing a complete result;
   once the second arm is journaled the pair cannot be replaced. Name the exact
   run ID and retain its manifest, both rank logs, coordinator and worker
   statuses, the adjacent result CSV's absence or incomplete bytes, and the
   producer/launcher completion attestations. Both statuses must be nonsignal
   and at least one must be nonzero. The sole signal exception is an attested
   launcher cleanup `TERM` on the worker behind a nonsignal nonzero coordinator
   status. A completed 300-token timing row or completion attestation cannot be
   invalidated. A valid
   slow result remains in the sequence. A pair index has at most two attempts and a
   candidate has at most two invalidations total; promotion discloses the
   count plus every earlier same-lane formal candidate, with exact-switch and
   available binary-match flags. Closing a candidate does not reset its formal sequence: a source
   commit can initialize only one formal candidate. A repaired or distinct
   hypothesis needs a new commit and candidate ID.
   A genuine infrastructure failure after the second arm is journaled closes
   that candidate; recovery starts at a new source commit and repeats the formal
   sequence. This cost is intentional because a selective second-arm rerun would
   condition the replacement on an already observed first-arm result.
4. Compare manifests with `scripts/compare-bench-manifests.py` and apply the
   candidate-gate, raw-log, CSV, and both-status validators. A manifest check
   does not prove build provenance, numerical correctness, or promotion.
5. Preserve failed runs beside successful runs and bind every reported result
   to its source and frozen artifact.

Repeated-Student promotion additionally requires the active baseline's
machine-recomputed bank of nine alternating control/control pairs under the
same source, model, binary, workload, effective environment, and RoCE v2 route.
This bank is reusable across candidates but cannot reuse candidate artifacts.
If its variance, drift, order/label bias, skew, or residual checks fail, use the
predeclared fixed-nine exact-sign path.

## Completion

Run targeted CPU/protocol checks and
`scripts/check-research-root-contract.sh` before committing. Keep candidate
kernels default-off until their applicable gates pass. Keep advisor reviews in
the promotion dossier; a repair or experiment commit is not a promotion.

The quality launcher's `deepseek-ordinary` teacher arm records complete
independent-case logits over RoCE v2 with both clean rank terminal proofs,
matching DeepSeek feature negotiation and hashes of the selected fixture
contents. `compare-teacher-logits.py --score-arm-mode deepseek-sdk` permits
different frozen executable identities with otherwise matching experiment
settings, reopens terminal/dump evidence, and remains threshold-free diagnostic
output. It cannot grant a Lane B admission. When an unchanged inference binary
is reused under a newer harness, record its separate source and executable
pins; retain the stale-build check and original build dossier. See
`$DS4_RESEARCH_ROOT/candidates/glm53-mtp-20260919/deepseek-teacher-admission-plan.md`.

## Active uncensored-model verifier track

The production-unused dense-prefill fixture compares the existing F32
K4096/N12288 token tile, two explicit original-Q8 WMMA128 GEMMs plus SwiGLU,
and paired WMMA128 with clamp10 at M256/M1024. Test-only admission, NaN poison,
one-hot addressing, precise host dots and original frozen production-control
hashes distinguish addressing and build effects from Lane B arithmetic.
It changes no production selector, weights, window size or transport. Full-FFN
component timing is not model throughput or quality admission. The uniquely
frozen build helper and active dossier's dense-prefill-component-plan.md bind
the hook and independent control separately. Integration and actual4K/8K model
timing remain required before any production performance claim.

`DS4_GLM5_MLA_OUTPUT_CAPTURE_PREFIX` with explicit capture position3072/7168
records production MLA heads and local output at layers3/23/43, M1024,
full-head stride16384. Exclusive files bind rank/layer/position/weight offset
and run ID; reads synchronize without enabling graph/quality mode or changing
arithmetic. Enabled runs are diagnostic only. The MLA fixture's
`--replay PREFIX 3072|7168` requires exact reproduction of captured incumbent
output before reporting original-Q8 and FP16-rounded dot errors. Explicit
MLA_OUTPUT_WMMA0/1 also reports successful candidate/incumbent launch counts
at process exit; these counts require separate clean completion proof.
See `mla-output-production-plan.md` in the active ROCm10 dossier. This
capture/replay does not provide Lane B quality admission or change defaults.

The existing original-Q8 MLA-output WMMA fixture accepts explicit
`--rows 256|1024` (default256), with output guards, both rank slices and all11
MLA layers. It retains strided/gathered exactness, sampled original-Q8 dots,
rollback and tail/refusal checks. Its synthetic inputs and analytic rounding
envelope diagnose addressing/numerics, not model quality. The fixture build
links frozen6eba8bd production objects while recording its newer harness
source separately; no inference selector/default changes. See
`mla-output-m1024-plan.md` in the active ROCm10 dossier.

The component fixture's `--geometry` comparison tests native N32/N64 at
M256/K16, existing SkinnyOnly and fused-shared-A native using a separate
component API; the older native mode0/1 contract stays unchanged. N64 is a
template-only probe with the original low LDS panel, exact F32 skinny gates,
original weights and no production dispatch switch. Both real rank slices at
M1024 must match native N32 QKV and independently compiled parent skinny
outputs, with poison/canaries and argument refusals. Standalone skinny time
is not an additive decomposition of the complete kernel. See
`bf16-six-native-geometry-plan.md` in the active ROCm10 dossier.

The default-off `NativeQkv` specialization of the live six-pointer BF16
kernel drops the residual QKV product, retaining original weights and exact
F32 skinny gates. `DS4_ROCM_GLM5_BF16_KDA_SIX_LIVE_NATIVE_QKV=1` selects it
within the existing six-prefill path; malformed values, alternate geometries,
quality/graph mode and missing prefill admission refuse. Complete M256 tiles
use the candidate; incomplete physical tails retain the incumbent and report
separate call/row counters. Decode retains its original arithmetic. The
real-weight `test_rocm_glm5_six_native.cu` compares against an independently
compiled parent, rounded-input controls and sampled FP64 dots before reporting
complete-six timings. Its `--stream` walks all34 KDA layers on both slices;
these synthetic-activation diagnostics do not establish model quality or
throughput. See `bf16-six-live-native-plan.md` in the active ROCm10 dossier.
The production-object fixture's `--live-native-qkv` arm requires exact skinny
and rounded-input QKV outputs, exercises selector/conflict/buffer refusal, and
records complete-six timing. It adds no new numerical tolerance. Model timing
and engaged-state quality remain required before promotion.

`compare-teacher-logits.py --score-arm-mode q4k-global` and
`q4k-global-repeat` compare explicit GLOBAL0/1 or same-mode captures using
the existing kda-tp scorer arm. These modes are diagnostic-only: legacy GLM
captures do not attest prompt/continuation contents at capture time. They
require `--score-fixture`/`--score-root`/`--score-rdma-gid`, one case per process, full-M batched
prefill and both-rank engagement, original same-build identities, complete
dump inventories, clean terminal proof, RoCE v2 and zero fallback/cache.
Quality/threshold bypasses refuse. Fixture content is hashed at comparison,
not misrepresented as capture-time proof. Existing modes and tolerances stay
unchanged. See the active dossier's `global-long-quality-plan.md`.
Legacy GLM scorer dumps can report `quant_bits=0` because the engine's public
query inspects the DeepSeek weight table. Only these global diagnostics may
accept that sentinel, after independently reading the original model's GGUF
architecture and Q4_K routed tensor family and matching its recorded size.
The dump remains unchanged; DeepSeek capture validation still rejects zero.

`DS4_ROCM_GLM5_Q4K_PREFILL_GLOBAL=1` is a default-off Lane B experiment
within the grouped/partition256 recipe. Complete M512/768/1024 use one expert
domain, original I64/J16 kernels and packed weights, threshold6 gate/up and
threshold1 down, absolute route IDs and ordered output sums. It changes
hot/cold arithmetic selection, not model quantization or transport. M256 and
irregular tails retain the incumbent. Malformed flags, missing prerequisites,
and schedule/wide conflicts refuse. Metadata shrinks; no weight cache grows.
The fixture's `--prefill-global-numerical` retains exact M256/grouped controls
and reports candidate errors/timing without certifying quality. See the active
ROCm10 dossier's `q4k-global-domain-plan.md`; formal promotion still needs the
matching baseline and engaged numerical/quality evidence.

The Q4_K partition fixture accepts `--prefill-wide128-numerical` for explicitly
unqualified error/timing diagnostics. Independent M256 and same-build controls
remain exact; nonfinite values and failed admission checks still fail. The
candidate reports final/up/mid errors before complete-MoE timing, without an
exactness or quality pass. This preserves earlier failed Lane A evidence and
does not qualify an arithmetic change for promotion. New arithmetic designs
use prospective Lane B numerical and quality evidence. See the active
ROCm10 dossier's `q4k-numerical-diagnostic-plan.md` for artifact binding.

`DS4_ROCM_GLM5_Q4K_PREFILL_WIDE128=1` is a separate default-off grouped
gate/up tile experiment. Eight waves share a J16 activation panel across128
output columns, staging oneK128half at a time in20,736bytes LDS. Original
packed Q4_K, M256 expert domains, thresholds and down projection are unchanged.
It requires the grouped/partition recipe and schedule0; invalid or conflicting
selectors refuse. Smaller/tail groups retain incumbent dispatch. The original
kernel stays intact for same-build and independentM256exactness controls.
See `q4k-wide-activation-plan.md` in the active ROCm10 dossier.

`DS4_ROCM_GLM5_Q4K_PREFILL_SCHEDULE=0/1/2` is a default-off Lane A
grouped-prefill scheduling experiment: incumbent capacity grid, static
grid-stride or atomic job queue. `DS4_ROCM_GLM5_Q4K_PREFILL_WORKERS=120/240`
bounds workgroups (default120). It requires the existing packed Q4_K grouped
and M256 partition recipe; only full outer M512/768/1024 groups change.
Smaller/tail groups use the incumbent. Arithmetic, hot/cold thresholds, ordered
route sums, original weights and transport are unchanged. Three atomic counters
reuse otherwise-unused IQ2 metadata in the existing Q4_K scratch allocation.
Malformed settings and incompatible admission refuse. No default changes or
performance claims precede correctness and model gates. The active dossier's
`q4k-prefill-scheduling-plan.md` records the CIRU donor and bounded test plan.

`scripts/build-glm53-research.sh` accepts `DS4_RESEARCH_PROFILE=1` for a
separate `glm53-profile-REV` frozen artifact. It explicitly enables the
existing CPU and HIP profiling hooks; unset/0 keeps the ordinary full build.
Invalid values refuse and existing artifacts are never rebuilt in place.
Profiled rates are diagnostic and cannot replace clean timing. The current
transport attribution plan lives in the active ROCm10 dossier as
`prefill-transport-attribution-plan.md`; no transport selector or completion
boundary changes merely because the hooks are compiled.

The active track was fast-forwarded to main `eda6285` for the unchanged
uncensored GGUF on pinned ROCm10. Its new dossier is
`$DS4_RESEARCH_ROOT/candidates/glm53-uncensored-rocm10-20260919/`.
`ds4_rocm_glm5_mla_prelude_q8_small_m` is a production-unused Lane A leaf:
M2/4/6 original Q8_0 Q_a/KV_a and rank-local Q_b, complete K512 panels,
resident-only admission and scalar accumulation order. The real-weight
fixture covers all eleven MLA layers and both rank slices. No inference
selector or transport change is enabled by this leaf. Results and unresolved
checks belong in that dossier's `METRICS.md`; no new promotion exception is
inherited from the SDK migration.

`DS4_GLM5_NATIVE_DRAFT_PROFILE=1` separately instruments the native draft
layer's completed input, MLA, routed FFN and norm/head/publication spans,
plus session readback, host selection, chain submission and agreement.
It adds diagnostic fences only when exactly1; unset/0 keeps the previous
schedule. Per-step proposal IDs bind both ranks' rejected work as well as
accepted output. Prefill, arithmetic and weights are unchanged. Profiled
throughput is not clean timing evidence; see `draft-subphase-plan.md` in
the active ROCm10 dossier.

`DS4_ROCM_GLM5_BF16_SMALL_M_PREFETCH=4/8` is a default-off Lane A
weight-load scheduling experiment inside exact M2/M4/M6 BF16 projections.
It reuses original pointers, existing activation LDS and scalar arithmetic
order; it allocates no weights. K128 keeps the previous kernel, M8 with the
option refuses, and large prefill is unchanged. See
`bf16-small-m-prefetch-plan.md` in the active ROCm10 dossier for tests and
results; a local fixture gain does not establish whole-model throughput.
The separate `DS4_ROCM_GLM5_BF16_SMALL_M_PANEL=512` diagnostic halves
activation LDS while preserving scalar K traversal. Default is1024;512
requires prefetch0and refusesM8. It changes no persistent storage and
must pass its own real-GGUF and model checks before selection.

`research/glm53-uncensored-six-kda-20260917` extends current main with
default-off exact BF16 reuse and accepted-prefix target verification for the
unchanged uncensored Q4 GGUF. Its dossier is
`$DS4_RESEARCH_ROOT/candidates/glm53-uncensored-20260915/`. Local echo-peer
checks do not establish real TP throughput or promotion. Session publication
requires coordinator agreement on both ranks' verification and commit outcomes.

The explicit native block45 draft step uses its own compact MLA state and a
bounded eight-expert window of original Q4_K bytes. Its workspace remains bound
to one state, immutable model and TP link. Native residual/norm semantics adapt
donors `fa09e19` and `519cf21` from the retained September 14 successor; they do
not merge that track. The adaptation, primary-source references and test scope
are recorded in the same dossier under `research/native-draft-step-plan-20260918.md`.
This step alone does not enable session drafting or establish acceptance/speed.

The default-off `DS4_GLM5_NATIVE_DRAFT=2/4/6/8` session experiment connects native
drafting, exact target verification and accepted-prefix refresh. Both ranks
negotiate the width and agree before history publication; failures invalidate
both states. Socket and controlled-arithmetic tests exercise orchestration,
including 8K frontiers, but do not establish actual RoCE, 8K model quality,
representative acceptance or whole-model speed. See the same dossier's
`research/native-session-plan-91508e1.md` and associated result report. Do not
enable this experiment for deployment before those checks and advisor review.

Width 6 means six target rows (root plus five proposals), with a distinct
canonical hello bit41 and exact M6 BF16/Q8 dispatch. It retains the existing
eight-expert native window and allocates only M2/M4/M6 verifier workspaces.
Short generation tails select 4/2/1 explicitly. Mixed width encodings and
workspace/configuration mismatch refuse before drafting; no rounding to M8.
See `research/native-width6-plan-3fae014.md` in the same dossier for validation.

`DS4_TP_BULK_RECV_READY=1` is a separate default-off transport experiment.
It negotiates hello bit39 and confirms both mlx5 receive queues before bulk
sends, retaining the registered slab and RDMA payload path. The small-payload
RoCE fixture is `tests/test_tp_bulk_small`; model evidence and protocol review
are still required. See `research/bulk-receive-ready-plan-3bfefc5.md` in the
same candidate dossier.

`DS4_GLM5_NATIVE_WINDOW_REUSE=1` separately retains matching original Q4_K
expert entries in the native workspace's existing eight-slot window. Its
54MiB capacity is unchanged; generic trunk window rebinding is unaffected.
This remains default-off, with native owner/model/rank binding and terminal
consumer synchronization required. The rationale and validation scope are in
`research/native-window-reuse-plan-48bb294.md` in the same dossier.

`DS4_ROCM_GLM5_VERIFY_SHARED_Q8=1` is a default-off exact shared-expert
batch experiment for resident KDA routed trunk layers in the target verifier.
It retains scalar mHC prefixes and original Q8_0 bytes, then shares weights
across M2/4/8 using existing activation scratch. Router agreement, routed Q4_K
evaluation and TP exchanges remain per token. Unsupported shared-pair modes
or nonresident layouts refuse. MLA and ordinary decode are unchanged by the
selector. See `research/shared-q8-integration-plan-d02a636.md` for the bounded
hypothesis and required real-weight, network and promotion checks.

`DS4_ROCM_GLM5_VERIFY_FFN_HANDOFF=1` separately batches route readbacks,
agreements and FFN payload exchanges across M2/4/8 in resident KDA routed
verification. It retains scalar router/expert arithmetic and existing scratch,
requires shared-Q8 batching and negotiated bit40, and stays default-off.
Layer agreements carry failure status with bounded I/O; negotiated bulk header
waits also have deadlines and terminate both channels on failure. The optional
`DS4_GLM5_NATIVE_PHASE_PROFILE=1` reports completed native-cycle boundaries
without extra GPU fences. See `research/handoff-plan-927ae0e.md` in the same
dossier. Echo-peer component fixtures and protocol tests do not establish
whole-model speed, real RoCE correctness or promotion.

`DS4_ROCM_GLM5_VERIFY_MLA_FFN_HANDOFF=1` extends the native shared-Q8/FFN
schedule to resident MLA trunk layers. It requires the existing handoff and
shared-Q8 settings, a valid native width, and negotiated bit42. It preserves
serial causal attention and saves each residual in existing batch scratch
before the shared projection and one FFN handoff. Prefill, native drafting,
model bytes and expert-window capacity are unchanged. Selector/configuration
mismatch refuses. This is default-off Lane A research; the plan and measured
results belong in `research/mla-ffn-handoff-plan-710f97e.md` in the dossier.

`DS4_ROCM_GLM5_VERIFY_FFN_QUEUE=1` queues the existing resident M1 expert
kernels and activation copies across verifier rows on stream0, with one
completion fence before phase1 status agreement. It requires native drafting,
FFN handoff, and negotiated bit43; malformed or mismatched settings refuse.
The default-off control completes each row separately. No prefill, model,
weight allocation or native draft-window change is involved. Fixture-only
HIP event wrappers measure packed expert device intervals separately from
complete loop wall time; they are not linked into model executables. The plan
is `research/ffn-queue-plan-d65372d.md` in the candidate dossier.

`DS4_GLM5_VERIFY_ROUTE_PROFILE=1` records host-only route overlap from the
existing validated verifier readback, including rejected proposals. It adds
no GPU transfer, allocation or weight cache. Existing `DS4_GLM5_VERIFY_PROFILE`
also separates completed FFN handoff sections and shared preparation. These
are instrumented diagnostic timings, including host/peer waits; existing
layer-profile fences remain and neither these spans nor nominal byte rates
are hardware ceilings. Use real two-rank runs for actual route distributions;
echo-peer fixture activations are not a model-route oracle. See
`research/verifier-route-profile-plan-11cbed2.md` in the candidate dossier.

`DS4_GLM5_VERIFY_MLA_PROFILE=1` adds completed subphase timings only inside
target verification's scalar MLA rows. Ordinary decode, drafting and prefill
pass no profile. It preserves operation order and adds diagnostic fences;
the index/state bucket deliberately includes interleaved index projections.
Exchange time includes peer waiting, not pure network latency. It uses no
persistent storage and does not alter widths or the communication schedule.
Compare same-build profile-off/on runs and keep instrumented timings separate.
The plan is `research/mla-subphase-plan-ea6d855.md` in the candidate dossier.

`DS4_ROCM_GLM5_VERIFY_MLA_ATTN_HANDOFF=1` defers the native verifier's
attention-output reduction until all causal MLA rows have produced heads.
It preserves M1 output projections and scalar residuals, retains each row's
heads and mHC split in existing batch scratch, and exchanges one row block.
Negotiated bit44 requires native2/4/6, owned32heads and MLA/shared/FFN handoff;
M8 refuses. A completed phase2 agreement precedes payload, including failure.
The local `DS4_GLM5_VERIFY_MLA_ROW_SYNC=1` diagnostic retains per-row completion
fences and requires this handoff. Both selectors default off; prefill and
drafting retain their original schedule. No model or weight-cache change.
See `research/mla-attn-handoff-plan-cc04c40.md` in the candidate dossier.

The dedicated `ds4_rocm_glm5_mla_output_q8_small_m` leaf tests exact output
weight reuse for M2/M4/M6, source K16384, local K8192 and N4096. It reuses
the existing shared-Q8 kernel with the original full packed row stride and
float activation rows and adds no weight allocation. Real-weight tests cover all trunk MLA output tensors,
both rank slices, panel/token isolation, canaries and refused shapes/modes.
See `research/mla-output-leaf-plan-f9146c7.md`; a component result does not
establish full-model performance or permission to enable this leaf.

`DS4_ROCM_GLM5_VERIFY_MLA_OUTPUT_BATCH=1` selects this leaf only within native
MLA attention handoff. It defaults off, requires the existing negotiated
handoff, preflights resident weights/modes/buffers before private replay and
keeps mandatory completion agreement on launch failure. There is no scalar
retry after failed batch launch. Both effective rank settings must match;
the wire schedule is unchanged. Session startup checks all eleven resident
output tensors for each exact workspace; per-layer checks remain and mark
transport failed if a later admission check fails. Cleanup reports M2/M4/M6
submissions; clean run completion is checked separately. Fixtures assert
batch-versus-scalar engagement per arm and exercise startup refusal.
Prefill, drafting and ordinary scalar tails keep their existing dispatch.
See `research/mla-output-integration-plan-3fd2018.md` for real-rank fixtures
and prospective same-binary4K/8K model comparisons before any promotion.

`DS4_ROCM_GLM5_VERIFY_EXPERT_PAIRS=0/1/2` is a default-off native6 routed
expert experiment: incumbent, scalar six-row schedule, or paired six-row
schedule. Hello bits45/46 distinguish the modes and require native6 and both
KDA/MLA FFN handoffs. Smaller2/4/1tails keep the incumbent path. It groups at
most two routes per original packed expert; no weights or window capacity
change. Admission rejects incompatible selectors, missing residency, invalid
routes, insufficient or aliased scratch. Completed route readbacks end the
earlier MLA scratch lifetime before reuse. Down/add still run row by row with
completion fences when queue0; failure drains before agreement, without retry.
The same-schedule scalar control isolates weight reuse from batching. Reports
and the four-arm plan are in `research/expert-gateup-integration-plan-e01a353.md`
and its implementation addendum in the uncensored dossier. This is Lane A
research, not a promoted model speed claim.
