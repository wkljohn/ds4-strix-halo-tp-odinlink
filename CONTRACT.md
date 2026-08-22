# DS4 repository contract

## Research evidence

The repository never owns a root-level `research-results` path. All raw and
curated research evidence is written to the branch-independent directory named
by `DS4_RESEARCH_ROOT`, whose default is the sibling directory
`../research-results`.

- Do not create a worktree-local directory, symlink, gitlink, or tracked path
  named `research-results`.
- Do not force-add ignored research artifacts.
- Benchmark and deployment tools must resolve the canonical root through
  `scripts/ds4-research-root.sh`.
- The two TP nodes have independent filesystems. Use
  `DS4_PEER_RESEARCH_ROOT` when the peer canonical path differs.
- Compact research reports and archive manifests live only on the orphan Git
  branch `research/results-archive`. User-facing maintained material belongs in
  `README.md` or `docs/`.
- Never overwrite divergent evidence or delete a source before checksum
  verification.

Run `scripts/check-research-root-contract.sh` before committing.

Classify performance candidates and validate their immutable evidence dossier
with `scripts/candidate-gate.py`. The authoritative lane A/B/C policy is
`$DS4_RESEARCH_ROOT/policies/GATE-PROMOTION.md`.

## Performance candidates

Performance work must remain on a named research branch until its applicable
correctness, mandatory-RDMA, long-context, ordinary-regression, and repeated
timing gates pass. Raw evidence goes to `DS4_RESEARCH_ROOT`; only source and
maintained documentation are merged to `main`.

Three repetitions of the 2,048+300 production workload establish the reported
median. They do not establish workload diversity. Every ordinary inference
candidate must also pass `scripts/run-tp-diverse-bench.sh`: one 4,096+300 run
over the frozen cross-disciplinary v1 prompt. Its manifest and CSV are checked
against the versioned three-run baseline by `scripts/diverse-bench-gate.py`.
The diverse run is a regression screen, not another headline benchmark.

DSpark candidates use the same frozen prompt and workload with
`DS4_BENCH_DSPARK=1`. Their diversity summary must be created in `dspark` mode
from three frozen DSpark baselines, the candidate run, and three ordinary runs
made by the same binary. The gate requires the 118/138 expert split, five-token
draft width, a matching support-model fingerprint, no main confidence knob,
and no regression against either the DSpark baseline or the same-stack
ordinary control. This is a final candidate-promotion gate, not a per-kernel
microbenchmark.
