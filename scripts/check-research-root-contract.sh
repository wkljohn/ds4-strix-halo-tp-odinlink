#!/usr/bin/env bash
set -euo pipefail

repo=$(git rev-parse --show-toplevel)
bad=0
if [[ -e $repo/research-results || -L $repo/research-results ]]; then
  echo "error: forbidden worktree-local path: $repo/research-results" >&2
  bad=1
fi
if git ls-files --error-unmatch research-results >/dev/null 2>&1; then
  echo "error: Git still tracks research-results paths" >&2
  git ls-files research-results >&2
  bad=1
fi
if git ls-files -s | awk '$1 == "160000" && $4 == "research-results" {found=1} END {exit !found}'; then
  echo "error: forbidden research-results gitlink" >&2
  bad=1
fi
exit "$bad"
