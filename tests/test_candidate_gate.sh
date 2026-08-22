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
  local mode=${5:-ordinary} fingerprint=${6:-1234567890abcdef}
  cat > "$dossier/$name.csv" <<EOF
ctx_tokens,prefill_tokens,prefill_tps,gen_tokens,gen_tps,gen_first_ms,gen_steady_tokens,gen_steady_tps,kvcache_bytes,gen_cycles,gen_token_fnv64
4096,4096,$prefill,300,$decode,70.0,299,$decode,0,300,$fingerprint
EOF
  local dspark=0 strict=0 mtp_size= mtp_sha= worker_env= coordinator_env=
  if [[ $mode == dspark ]]; then
    dspark=1
    mtp_size=5989114272
    mtp_sha=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
    worker_env='DS4_TP_EXPERT_SPLIT=118 DS4_DSPARK_MAX_DRAFT_TOKENS=5 DS4_DSPARK_SCHEDULER=0'
    coordinator_env="$worker_env DS4_DSPARK_RESIDENT_Q8=1"
  fi
  cat > "$dossier/$name.manifest" <<EOF
bench_config_sha256=fixture
ds4_sha256=fixture-ds4
ds4_bench_tp_sha256=fixture-bench
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
dspark=$dspark
dspark_strict=$strict
mtp_size=$mtp_size
mtp_sample_sha256=$mtp_sha
worker_env=$worker_env
coordinator_env=$coordinator_env
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

make_run dspark-baseline-1 198.0 15.00 1 dspark fedcba0987654321
make_run dspark-baseline-2 199.0 15.10 1 dspark fedcba0987654321
make_run dspark-baseline-3 197.0 14.90 1 dspark fedcba0987654321
make_run dspark-candidate 200.0 15.20 1 dspark fedcba0987654321
"$repo/scripts/diverse-bench-gate.py" create --lane A --mode dspark \
  --baseline "$dossier/dspark-baseline-1.csv" \
  --baseline "$dossier/dspark-baseline-2.csv" \
  --baseline "$dossier/dspark-baseline-3.csv" \
  --ordinary "$dossier/baseline-1.csv" \
  --ordinary "$dossier/baseline-2.csv" \
  --ordinary "$dossier/baseline-3.csv" \
  --candidate "$dossier/dspark-candidate.csv" \
  --output "$dossier/dspark-diverse-summary.json" >/dev/null
DS4_RESEARCH_ROOT=$DS4_RESEARCH_ROOT \
  "$repo/scripts/diverse-bench-gate.py" verify \
  "$dossier/dspark-diverse-summary.json" >/dev/null
dspark_diverse_hash=$(sha256sum "$dossier/dspark-diverse-summary.json" | awk '{print $1}')

"$repo/scripts/candidate-gate.py" init test-dspark-lane-a A >/dev/null
dspark_dossier=$DS4_RESEARCH_ROOT/candidates/test-dspark-lane-a
cp "$dossier/evidence.txt" "$dspark_dossier/evidence.txt"
cp "$dossier/dspark-diverse-summary.json" "$dspark_dossier/dspark-diverse-summary.json"
python3 - "$dspark_dossier/candidate.json" "$hash" "$dspark_diverse_hash" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
digest = sys.argv[2]
diverse_digest = sys.argv[3]
value = json.loads(path.read_text())
value["dspark"] = True
kinds = [
    "ordinary-benchmark", "candidate-benchmark", "candidate-benchmark",
    "candidate-benchmark", "long-context", "transport-proof",
    "ordinary-regression", "cross-discipline-long", "fable-review",
    "grok-review", "exact-fingerprint", "rollback-proof",
    "same-stack-ordinary", "verifier-logits", "dspark-acceptance",
]
value["evidence"] = []
for kind in kinds:
    if kind == "cross-discipline-long":
        value["evidence"].append({"kind": kind, "path": "dspark-diverse-summary.json", "sha256": diverse_digest})
    else:
        value["evidence"].append({"kind": kind, "path": "evidence.txt", "sha256": digest})
for key in value["claims"]:
    value["claims"][key] = True
for key in ("accepted_target_quality_preserved", "acceptance_profile_recorded", "same_stack_target_used"):
    value["claims"][key] = True
path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
PY
"$repo/scripts/candidate-gate.py" check test-dspark-lane-a >/dev/null

# A DSpark dossier must not satisfy its long-context requirement with an
# ordinary-mode summary, even if that summary is otherwise valid and hashed.
cp "$dossier/diverse-summary.json" "$dspark_dossier/ordinary-summary.json"
ordinary_summary_hash=$(sha256sum "$dspark_dossier/ordinary-summary.json" | awk '{print $1}')
python3 - "$dspark_dossier/candidate.json" "$ordinary_summary_hash" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
value = json.loads(path.read_text())
for item in value["evidence"]:
    if item["kind"] == "cross-discipline-long":
        item["path"] = "ordinary-summary.json"
        item["sha256"] = sys.argv[2]
path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
PY
if "$repo/scripts/candidate-gate.py" check test-dspark-lane-a >/dev/null 2>&1; then
  echo "error: DSpark candidate gate accepted an ordinary diverse summary" >&2
  exit 1
fi

make_run dspark-slower-than-ordinary 200.0 14.00 1 dspark fedcba0987654321
if "$repo/scripts/diverse-bench-gate.py" create --lane A --mode dspark \
  --baseline "$dossier/dspark-baseline-1.csv" \
  --baseline "$dossier/dspark-baseline-2.csv" \
  --baseline "$dossier/dspark-baseline-3.csv" \
  --ordinary "$dossier/baseline-1.csv" \
  --ordinary "$dossier/baseline-2.csv" \
  --ordinary "$dossier/baseline-3.csv" \
  --candidate "$dossier/dspark-slower-than-ordinary.csv" \
  --output "$dossier/dspark-slow-summary.json" >/dev/null 2>&1; then
  echo "error: DSpark diverse gate accepted a candidate slower than ordinary" >&2
  exit 1
fi

make_run dspark-scheduled 200.0 15.20 1 dspark fedcba0987654321
sed -i 's/DS4_DSPARK_SCHEDULER=0/DS4_DSPARK_SCHEDULER=1/' \
  "$dossier/dspark-scheduled.manifest"
if "$repo/scripts/diverse-bench-gate.py" create --lane A --mode dspark \
  --baseline "$dossier/dspark-baseline-1.csv" \
  --baseline "$dossier/dspark-baseline-2.csv" \
  --baseline "$dossier/dspark-baseline-3.csv" \
  --ordinary "$dossier/baseline-1.csv" \
  --ordinary "$dossier/baseline-2.csv" \
  --ordinary "$dossier/baseline-3.csv" \
  --candidate "$dossier/dspark-scheduled.csv" \
  --output "$dossier/dspark-scheduled-summary.json" >/dev/null 2>&1; then
  echo "error: DSpark diverse gate accepted the main scheduler" >&2
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
