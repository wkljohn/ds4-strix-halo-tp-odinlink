#!/usr/bin/env bash
set -euo pipefail

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
# shellcheck disable=SC1091
source "$repo/scripts/ds4-research-root.sh"

fixture=$(mktemp -d)
trap 'rm -r -- "$fixture"' EXIT
mkdir -p "$fixture/worktree"

unset DS4_RESEARCH_ROOT DS4_PEER_RESEARCH_ROOT
ds4_resolve_research_roots "$fixture/worktree"
[[ $DS4_RESEARCH_ROOT == "$fixture/research-results" ]]
[[ $DS4_PEER_RESEARCH_ROOT == "$fixture/research-results" ]]

DS4_RESEARCH_ROOT=$fixture/local-archive
DS4_PEER_RESEARCH_ROOT=$fixture/peer-archive
ds4_resolve_research_roots "$fixture/worktree"
[[ $DS4_RESEARCH_ROOT == "$fixture/local-archive" ]]
[[ $DS4_PEER_RESEARCH_ROOT == "$fixture/peer-archive" ]]

DS4_RESEARCH_ROOT=relative/archive
if ds4_resolve_research_roots "$fixture/worktree" 2>/dev/null; then
  echo "error: relative canonical root was accepted" >&2
  exit 1
fi

DS4_RESEARCH_ROOT=$fixture/worktree/inside
if ds4_resolve_research_roots "$fixture/worktree" 2>/dev/null; then
  echo "error: worktree-contained canonical root was accepted" >&2
  exit 1
fi

DS4_RESEARCH_ROOT=$fixture/local-archive
mkdir -p "$fixture/worktree/research-results"
if ds4_resolve_research_roots "$fixture/worktree" 2>/dev/null; then
  echo "error: worktree-local research-results was accepted" >&2
  exit 1
fi

echo "PASS research-root-contract"
