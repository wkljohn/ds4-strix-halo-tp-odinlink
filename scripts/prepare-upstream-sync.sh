#!/usr/bin/env bash
set -euo pipefail

canonical_url=https://github.com/antirez/ds4.git
target_ref=${1:-upstream/main}
repo_root=$(git rev-parse --show-toplevel)
cd "$repo_root"

if [[ -n $(git status --porcelain) ]]; then
    echo "error: upstream sync requires a clean worktree" >&2
    exit 1
fi

if [[ $(git branch --show-current) != main ]]; then
    echo "error: run this helper from the fork's main branch" >&2
    exit 1
fi

if ! git remote get-url upstream >/dev/null 2>&1; then
    git remote add upstream "$canonical_url"
fi

echo "Fetching canonical DS4 from $(git remote get-url upstream)"
git fetch upstream --prune

if ! git rev-parse --verify --quiet "$target_ref^{commit}" >/dev/null; then
    echo "error: target '$target_ref' is not a commit" >&2
    exit 1
fi

if ! git merge-base HEAD "$target_ref" >/dev/null; then
    echo "error: no merge base with '$target_ref'; do not use --allow-unrelated-histories" >&2
    exit 1
fi

if git merge-base --is-ancestor "$target_ref" HEAD; then
    echo "$target_ref is already integrated"
    exit 0
fi

target_short=$(git rev-parse --short=12 "$target_ref")
sync_branch="sync/upstream-$target_short"
if git show-ref --verify --quiet "refs/heads/$sync_branch"; then
    echo "error: branch '$sync_branch' already exists" >&2
    exit 1
fi

git switch -c "$sync_branch"
if ! git merge --no-ff --no-commit "$target_ref"; then
    echo
    echo "Upstream conflicts are ready for review on $sync_branch."
    echo "Resolve them and run the validation in docs/UPSTREAM-SYNC.md,"
    echo "or return to the original state with: git merge --abort"
    exit 1
fi

echo
echo "Upstream changes are staged but not committed on $sync_branch."
echo "Review and validate them using docs/UPSTREAM-SYNC.md before committing."

