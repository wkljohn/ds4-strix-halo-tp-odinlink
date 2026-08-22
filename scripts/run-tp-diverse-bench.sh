#!/usr/bin/env bash
# One-load ordinary-inference diversity screen: 4096 prompt + 300 decode tokens.
set -euo pipefail

REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
TAG=${1:?usage: run-tp-diverse-bench.sh TAG MODEL [EXTRA_ENV=VALUE ...]}
MODEL=${2:?usage: run-tp-diverse-bench.sh TAG MODEL [EXTRA_ENV=VALUE ...]}
shift 2

# shellcheck disable=SC1091
source "$REPO/scripts/ds4-research-root.sh"
ds4_resolve_research_roots "$REPO"
PROMPT_DIR=$DS4_RESEARCH_ROOT/bench-prompts
PROMPT=$PROMPT_DIR/cross-discipline-v1.md
mkdir -p "$PROMPT_DIR"
tmp=$(mktemp "$PROMPT_DIR/.cross-discipline-v1.XXXXXX")
trap 'rm -f -- "$tmp"' EXIT
"$REPO/scripts/generate-diverse-bench-prompt.py" > "$tmp"
prompt_prefix=$(head -c 2048 "$tmp")
for discipline in software-debugging quantitative-science \
                  policy-document-retrieval structured-data-analysis; do
  grep -Fq "[$discipline" <<<"$prompt_prefix" || {
    echo "error: diverse prompt prefix is missing $discipline" >&2
    exit 1
  }
done
unset prompt_prefix discipline
mv -f -- "$tmp" "$PROMPT"
trap - EXIT

echo "diverse_matrix=v1 disciplines=software-debugging,quantitative-science,policy-document-retrieval,structured-data-analysis"
echo "diverse_matrix_shape=frontier:4096 generated_tokens:300 model_loads:1"

DS4_BENCH_PROMPT_FILE=$PROMPT \
DS4_BENCH_FRONTIER=4096 \
DS4_BENCH_TOKENS=300 \
DS4_BENCH_CONTEXT=4608 \
DS4_BENCH_PREFILL_CHUNK=2048 \
  "$REPO/run-tp-ds4-bench.sh" "$TAG" "$MODEL" "$@"

OUT=${DS4_BENCH_OUT:-$DS4_RESEARCH_ROOT/bench-runs}
"$REPO/scripts/render-tp-bench-table.py" "$OUT/$TAG.csv"
