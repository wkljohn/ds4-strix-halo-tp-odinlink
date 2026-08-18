#!/usr/bin/env bash
# OpenAI-compatible serving benchmark for the loaded TP=2 server.
#
# This complements run-tp-ds4-bench.sh: it measures client-visible API
# behavior with llama-benchy, while ds4-bench-tp remains the correctness and
# kernel-candidate gate.
set -euo pipefail

LLAMA_BENCHY_COMMIT=e9be344578cec17745066b220798b80a0d2686d3
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd -- "$SCRIPT_DIR/.." && pwd)

PORT=${PORT:-8090}
BASE_URL=${BASE_URL:-http://127.0.0.1:$PORT/v1}
MODEL=${MODEL:-deepseek-v4-flash}
TOKENIZER=${TOKENIZER:-${TOKENIZER_DIR:-deepseek-ai/DeepSeek-V4-Flash}}
API_KEY=${API_KEY:-}
PP=(2048)
TG=(32 128 300)
DEPTH=(0 4096 16384)
RUNS=3
WARMUP_RUNS=1
OUT=${OUT:-$REPO/research-results/api-bench/$(date -u +%Y%m%dT%H%M%SZ).csv}
CONFIRMED=0
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: scripts/run-api-bench.sh --confirm-dedicated-server [options]

Benchmarks a running OpenAI-compatible ds4-server with a pinned llama-benchy.
The default sweep is pp=2048, tg=32/128/300, depth=0/4096/16384, runs=3,
concurrency=1, and generation-latency measurement.

Options:
  --base-url URL      API root including /v1 (default: http://127.0.0.1:8090/v1)
  --api-key KEY       Bearer token, if the endpoint requires one
  --model NAME        Served model name (default: deepseek-v4-flash)
  --tokenizer ID      Hugging Face ID or local tokenizer directory
  --pp N...           Prompt-token sizes
  --tg N...           Generated-token sizes
  --depth N...        Existing-context depths
  --runs N            Measured runs per shape (default: 3)
  --warmup-runs N     Discarded warmups per shape (default: 1)
  --out FILE.csv      Result path under research-results/ by default
  --confirm-dedicated-server
                      Required: this benchmark replaces the server's live session
  --dry-run           Validate and print the pinned command without contacting API
  -h, --help          Show this help
EOF
}

while (($#)); do
  case $1 in
    --base-url) BASE_URL=${2:?missing URL}; shift 2 ;;
    --api-key) API_KEY=${2:?missing key}; shift 2 ;;
    --model) MODEL=${2:?missing model}; shift 2 ;;
    --tokenizer) TOKENIZER=${2:?missing tokenizer}; shift 2 ;;
    --pp|--tg|--depth)
      option=$1
      shift
      values=()
      while (($#)) && [[ $1 != --* ]]; do values+=("$1"); shift; done
      ((${#values[@]} > 0)) || { echo "error: $option requires at least one value" >&2; exit 2; }
      for value in "${values[@]}"; do
        [[ $value =~ ^[0-9]+$ ]] || { echo "error: invalid value for $option: $value" >&2; exit 2; }
      done
      case $option in
        --pp) PP=("${values[@]}") ;;
        --tg) TG=("${values[@]}") ;;
        --depth) DEPTH=("${values[@]}") ;;
      esac
      ;;
    --runs) RUNS=${2:?missing run count}; shift 2 ;;
    --warmup-runs) WARMUP_RUNS=${2:?missing warmup count}; shift 2 ;;
    --out) OUT=${2:?missing output path}; shift 2 ;;
    --confirm-dedicated-server) CONFIRMED=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "error: unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ $RUNS =~ ^[1-9][0-9]*$ ]] || { echo "error: --runs must be positive" >&2; exit 2; }
[[ $WARMUP_RUNS =~ ^[0-9]+$ ]] || { echo "error: --warmup-runs must be non-negative" >&2; exit 2; }
[[ $OUT == *.csv ]] || { echo "error: --out must end in .csv" >&2; exit 2; }
BASE_URL=${BASE_URL%/}

BENCHY_SOURCE="git+https://github.com/eugr/llama-benchy@$LLAMA_BENCHY_COMMIT"
cmd=(uvx --from "$BENCHY_SOURCE" llama-benchy
  --base-url "$BASE_URL"
  --model "$MODEL"
  --served-model-name "$MODEL"
  --tokenizer "$TOKENIZER"
  --pp "${PP[@]}"
  --tg "${TG[@]}"
  --depth "${DEPTH[@]}"
  --concurrency 1
  --runs "$RUNS"
  --warmup-runs "$WARMUP_RUNS"
  --latency-mode generation
  --no-cache
  --exit-on-first-fail
  --format csv
  --save-result "$OUT")
[[ -z $API_KEY ]] || cmd+=(--api-key "$API_KEY")

if ((DRY_RUN)); then
  printable=("${cmd[@]}")
  if [[ -n $API_KEY ]]; then
    for i in "${!printable[@]}"; do
      [[ ${printable[$i]} != "$API_KEY" ]] || printable[$i]='<redacted>'
    done
  fi
  printf '%q ' "${printable[@]}"
  printf '\n'
  exit 0
fi

((CONFIRMED == 1)) || {
  echo "error: use --confirm-dedicated-server; this sweep replaces the live server session" >&2
  exit 2
}
command -v uvx >/dev/null 2>&1 || {
  echo "error: uvx is required: https://docs.astral.sh/uv/" >&2
  exit 1
}

curl_args=(--fail --silent --show-error --max-time 5)
[[ -z $API_KEY ]] || curl_args+=(-H "Authorization: Bearer $API_KEY")
models_json=$(curl "${curl_args[@]}" "$BASE_URL/models") || {
  echo "error: cannot reach $BASE_URL/models" >&2
  exit 1
}
grep -Fq '"id"' <<<"$models_json" || {
  echo "error: endpoint returned no served model" >&2
  exit 1
}

mkdir -p "$(dirname -- "$OUT")"
meta=${OUT%.csv}.meta.txt
[[ ! -e $OUT && ! -e $meta ]] || {
  echo "error: refusing to overwrite existing result or metadata: $OUT" >&2
  exit 1
}
{
  printf 'date_utc=%s\n' "$(date -u +%FT%TZ)"
  printf 'ds4_commit=%s\n' "$(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo unknown)"
  printf 'llama_benchy_commit=%s\n' "$LLAMA_BENCHY_COMMIT"
  printf 'base_url=%s\n' "$BASE_URL"
  printf 'model=%s\n' "$MODEL"
  printf 'tokenizer=%s\n' "$TOKENIZER"
  printf 'pp=%s\n' "${PP[*]}"
  printf 'tg=%s\n' "${TG[*]}"
  printf 'depth=%s\n' "${DEPTH[*]}"
  printf 'runs=%s\n' "$RUNS"
  printf 'warmup_runs=%s\n' "$WARMUP_RUNS"
  printf 'concurrency=1\n'
} >"$meta"

printf 'results=%s\nmetadata=%s\n' "$OUT" "$meta"
exec "${cmd[@]}"
