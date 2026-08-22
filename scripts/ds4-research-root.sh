#!/usr/bin/env bash

# Source this file, then call ds4_resolve_research_roots REPO_ROOT.
ds4_resolve_research_roots() {
  local repo_root=${1:?repository root required}
  local workspace_root
  workspace_root=$(cd -- "$repo_root/.." && pwd)

  if [[ -e $repo_root/research-results || -L $repo_root/research-results ]]; then
    echo "error: worktree-local research-results violates CONTRACT.md: $repo_root/research-results" >&2
    return 2
  fi

  DS4_RESEARCH_ROOT=${DS4_RESEARCH_ROOT:-$workspace_root/research-results}
  DS4_PEER_RESEARCH_ROOT=${DS4_PEER_RESEARCH_ROOT:-$DS4_RESEARCH_ROOT}
  case $DS4_RESEARCH_ROOT in
    /*) ;;
    *) echo "error: DS4_RESEARCH_ROOT must be absolute" >&2; return 2 ;;
  esac
  case $DS4_PEER_RESEARCH_ROOT in
    /*) ;;
    *) echo "error: DS4_PEER_RESEARCH_ROOT must be absolute" >&2; return 2 ;;
  esac
  case $DS4_RESEARCH_ROOT/ in
    "$repo_root"/*)
      echo "error: DS4_RESEARCH_ROOT must be outside the Git worktree" >&2
      return 2
      ;;
  esac
  export DS4_RESEARCH_ROOT DS4_PEER_RESEARCH_ROOT
}
