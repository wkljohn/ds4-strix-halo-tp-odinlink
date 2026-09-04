#!/usr/bin/env bash
# Q4 release curve: five 2K prefill increments, each followed by 300 decode tokens.
set -euo pipefail

REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
TAG=${1:?usage: run-tp-context-sweep.sh TAG Q4_MODEL [EXTRA_ENV=VALUE ...]}
MODEL=${2:?usage: run-tp-context-sweep.sh TAG Q4_MODEL [EXTRA_ENV=VALUE ...]}
shift 2

# shellcheck disable=SC1091
source "$REPO/scripts/ds4-research-root.sh"
ds4_resolve_research_roots "$REPO"
PROMPT_DIR=$DS4_RESEARCH_ROOT/bench-prompts
PROMPT=$PROMPT_DIR/cross-discipline-long10k-v1.md
mkdir -p "$PROMPT_DIR"
tmp=$(mktemp "$PROMPT_DIR/.cross-discipline-long10k-v1.XXXXXX")
trap 'rm -f -- "$tmp"' EXIT
"$REPO/scripts/generate-diverse-bench-prompt.py" --cycles 48 > "$tmp"
prompt_prefix=$(head -c 2048 "$tmp")
for discipline in software-debugging quantitative-science \
                  policy-document-retrieval structured-data-analysis; do
  grep -Fq "[$discipline" <<<"$prompt_prefix" || {
    echo "error: long-context prompt prefix is missing $discipline" >&2
    exit 1
  }
done
unset prompt_prefix discipline
mv -f -- "$tmp" "$PROMPT"
trap - EXIT

echo "context_sweep=v1 frontiers=2048,4096,6144,8192,10240 generated_tokens=300"
DS4_BENCH_PROMPT_FILE=$PROMPT \
DS4_BENCH_FRONTIER=2048 \
DS4_BENCH_FRONTIER_MAX=10240 \
DS4_BENCH_STEP_INCR=2048 \
DS4_BENCH_TOKENS=300 \
DS4_BENCH_CONTEXT=10752 \
DS4_BENCH_PREFILL_CHUNK=2048 \
  "$REPO/run-tp-ds4-bench.sh" "$TAG" "$MODEL" "$@"

OUT=${DS4_BENCH_OUT:-$DS4_RESEARCH_ROOT/bench-runs}
"$REPO/scripts/render-tp-bench-table.py" "$OUT/$TAG.csv"
