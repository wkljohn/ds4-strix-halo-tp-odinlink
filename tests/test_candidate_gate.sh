#!/usr/bin/env bash
set -euo pipefail

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
fixture=$(mktemp -d)
trap 'rm -r -- "$fixture"' EXIT
export DS4_RESEARCH_ROOT=$fixture/archive
mkdir -p "$DS4_RESEARCH_ROOT"

prompt_hash=$("$repo/scripts/generate-diverse-bench-prompt.py" | sha256sum | awk '{print $1}')
[[ $prompt_hash == 24d19432acab4d4cd2971d938b3c013fcfad1010ed701218bc7bdc1b630ecfef ]] || {
  echo "error: cross-disciplinary prompt changed without a version bump" >&2
  exit 1
}

"$repo/scripts/candidate-gate.py" init test-lane-a A >/dev/null
dossier=$DS4_RESEARCH_ROOT/candidates/test-lane-a
printf 'verified evidence\n' > "$dossier/evidence.txt"
hash=$(sha256sum "$dossier/evidence.txt" | awk '{print $1}')

make_run() {
  local name=$1 prefill=$2 decode=$3 candidate=$4
  cat > "$dossier/$name.csv" <<EOF
ctx_tokens,prefill_tokens,prefill_tps,gen_tokens,gen_tps,gen_first_ms,gen_steady_tokens,gen_steady_tps,kvcache_bytes,gen_cycles,gen_token_fnv64
4096,4096,$prefill,300,$decode,70.0,299,$decode,0,300,1234567890abcdef
EOF
  cat > "$dossier/$name.manifest" <<EOF
bench_config_sha256=fixture
model_size=1
model_sample_sha256=fixture-model
prompt_sha256=24d19432acab4d4cd2971d938b3c013fcfad1010ed701218bc7bdc1b630ecfef
frontier=4096
generated_tokens=300
context=4608
prefill_chunk=2048
rdma_profile=odinlink
coordinator_rdma_device=odl_tb5_0
worker_rdma_device=odl_tb5_0
rdma_gid_index=n/a
candidate=$candidate
dspark=0
EOF
}
make_run baseline-1 200.0 15.00 1
make_run baseline-2 201.0 15.10 1
make_run baseline-3 199.0 14.90 1
make_run diverse-candidate 202.0 15.20 1
"$repo/scripts/diverse-bench-gate.py" create --lane A \
  --baseline "$dossier/baseline-1.csv" \
  --baseline "$dossier/baseline-2.csv" \
  --baseline "$dossier/baseline-3.csv" \
  --candidate "$dossier/diverse-candidate.csv" \
  --output "$dossier/diverse-summary.json" >/dev/null
diverse_hash=$(sha256sum "$dossier/diverse-summary.json" | awk '{print $1}')

make_run diverse-regressed 202.0 13.50 1
if "$repo/scripts/diverse-bench-gate.py" create --lane A \
  --baseline "$dossier/baseline-1.csv" \
  --baseline "$dossier/baseline-2.csv" \
  --baseline "$dossier/baseline-3.csv" \
  --candidate "$dossier/diverse-regressed.csv" \
  --output "$dossier/diverse-regressed-summary.json" >/dev/null 2>&1; then
  echo "error: diverse benchmark gate accepted a decode regression" >&2
  exit 1
fi

python3 - "$dossier/candidate.json" "$hash" "$diverse_hash" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
digest = sys.argv[2]
diverse_digest = sys.argv[3]
value = json.loads(path.read_text())
kinds = [
    "ordinary-benchmark",
    "candidate-benchmark",
    "candidate-benchmark",
    "candidate-benchmark",
    "long-context",
    "transport-proof",
    "ordinary-regression",
    "cross-discipline-long",
    "fable-review",
    "grok-review",
    "exact-fingerprint",
    "rollback-proof",
]
value["evidence"] = []
for kind in kinds:
    if kind == "cross-discipline-long":
        value["evidence"].append({"kind": kind, "path": "diverse-summary.json", "sha256": diverse_digest})
    else:
        value["evidence"].append({"kind": kind, "path": "evidence.txt", "sha256": digest})
for key in value["claims"]:
    value["claims"][key] = True
path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
PY

"$repo/scripts/candidate-gate.py" check test-lane-a >/dev/null
"$repo/scripts/candidate-gate.py" promote test-lane-a >/dev/null
test -s "$dossier/PROMOTED.json"

printf 'tampered\n' >> "$dossier/evidence.txt"
if "$repo/scripts/candidate-gate.py" check test-lane-a >/dev/null 2>&1; then
  echo "error: candidate gate accepted tampered evidence" >&2
  exit 1
fi

echo "PASS candidate-gate"
