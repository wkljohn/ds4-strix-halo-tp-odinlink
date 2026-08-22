#!/usr/bin/env bash
set -euo pipefail

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
fixture=$(mktemp -d)
trap 'rm -r -- "$fixture"' EXIT
export DS4_RESEARCH_ROOT=$fixture/archive
mkdir -p "$DS4_RESEARCH_ROOT"

"$repo/scripts/candidate-gate.py" init test-lane-a A >/dev/null
dossier=$DS4_RESEARCH_ROOT/candidates/test-lane-a
printf 'verified evidence\n' > "$dossier/evidence.txt"
hash=$(sha256sum "$dossier/evidence.txt" | awk '{print $1}')

python3 - "$dossier/candidate.json" "$hash" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
digest = sys.argv[2]
value = json.loads(path.read_text())
kinds = [
    "ordinary-benchmark",
    "candidate-benchmark",
    "candidate-benchmark",
    "candidate-benchmark",
    "long-context",
    "transport-proof",
    "ordinary-regression",
    "fable-review",
    "grok-review",
    "exact-fingerprint",
    "rollback-proof",
]
value["evidence"] = [
    {"kind": kind, "path": "evidence.txt", "sha256": digest} for kind in kinds
]
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
