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
