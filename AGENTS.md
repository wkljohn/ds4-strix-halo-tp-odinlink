# Agent instructions

Read `PROJECT.md` and `CONTRACT.md` before modifying or benchmarking this
repository.

Never create `./research-results`, including as a symlink. Resolve the canonical
archive with `scripts/ds4-research-root.sh`, write evidence beneath
`$DS4_RESEARCH_ROOT`, and run `scripts/check-research-root-contract.sh` before
committing. Preserve dirty worktree state and do not stage unrelated changes.
