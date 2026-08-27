#!/usr/bin/env bash
set -euo pipefail

model=${DS4_GLM5_MODEL:-}
if [[ -z "$model" || ! -f "$model" ]]; then
    echo "error: DS4_GLM5_MODEL must name the validated GLM-5.3 GGUF" >&2
    exit 2
fi

expected='ds4: glm5-next resident KDA component validated; sparse MLA/indexer, mHC, FFN, output, and full TP graph remain unimplemented; refusing inference'
log=$(mktemp)
lock=$(mktemp -u /tmp/ds4-glm5-refusal.XXXXXX.lock)
trap 'rm -f "$log" "$lock"' EXIT

set +e
DS4_LOCK_FILE="$lock" timeout 45 ./ds4 \
    -m "$model" --rocm -n 1 -p hi >"$log" 2>&1
rc=$?
set -e

if [[ $rc -eq 0 || $rc -eq 124 ]]; then
    echo "FAIL glm5-next runtime did not fail closed (exit=$rc)" >&2
    tail -40 "$log" >&2
    exit 1
fi
if ! grep -Fqx "$expected" "$log"; then
    echo "FAIL glm5-next runtime refusal changed or was not reached" >&2
    tail -40 "$log" >&2
    exit 1
fi
echo "PASS glm5-next runtime remains fail-closed after resident KDA validation"
