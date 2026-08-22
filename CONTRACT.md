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

## Performance candidates

Performance work must remain on a named research branch until its applicable
correctness, mandatory-RDMA, long-context, ordinary-regression, and repeated
timing gates pass. Raw evidence goes to `DS4_RESEARCH_ROOT`; only source and
maintained documentation are merged to `main`.
