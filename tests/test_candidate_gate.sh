#!/usr/bin/env bash
set -euo pipefail

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
if [[ ${DS4_GATE_CLEAN_TEST:-0} != 1 ]] &&
   [[ -n $(git -C "$repo" status --porcelain=v1 -uall) ]]; then
  clean_fixture=$(mktemp -d)
  trap 'rm -r -- "$clean_fixture"' EXIT
  git clone -q --no-local "$repo" "$clean_fixture/repo"
  git -C "$repo" diff --binary HEAD | git -C "$clean_fixture/repo" apply
  while IFS= read -r -d '' path; do
    mkdir -p "$clean_fixture/repo/$(dirname -- "$path")"
    cp -a -- "$repo/$path" "$clean_fixture/repo/$path"
  done < <(git -C "$repo" ls-files -o --exclude-standard -z)
  git -C "$clean_fixture/repo" add -A
  git -C "$clean_fixture/repo" -c user.name=DS4-Gate-Test \
    -c user.email=gate-test.invalid commit -qm 'test fixture snapshot'
  (cd "$clean_fixture/repo" && DS4_GATE_CLEAN_TEST=1 ./tests/test_candidate_gate.sh)
  exit
fi
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
source_commit=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["source"]["commit"])' "$dossier/candidate.json")
printf 'verified evidence\n' > "$dossier/evidence.txt"
hash=$(sha256sum "$dossier/evidence.txt" | awk '{print $1}')

make_run() {
  local name=$1 prefill=$2 decode=$3 candidate=$4
  local fingerprint=${5:-1234567890abcdef}
  local lane=${6:-A}
  local baseline=${7:-}
  local frontier=${8:-4096}
  local context=${9:-4608}
  local prompt_sha=${10:-24d19432acab4d4cd2971d938b3c013fcfad1010ed701218bc7bdc1b630ecfef}
  local expected=
  local model_sample=${test_model_sample:-fixture-model}
  local model_size=${test_model_size:-1}
  local model_path=${test_model_path:-/model.gguf}
  [[ $lane != A ]] || expected=$fingerprint
  cat > "$dossier/$name.csv" <<EOF
ctx_tokens,prefill_tokens,prefill_tps,gen_tokens,gen_tps,gen_first_ms,gen_steady_tokens,gen_steady_tps,kvcache_bytes,gen_cycles,gen_token_fnv64
$frontier,$frontier,$prefill,300,$decode,70.0,299,$decode,0,300,$fingerprint
EOF
  cat > "$dossier/$name.manifest" <<EOF
tag=$name
run_id=$name-run-id
bench_config_sha256=fixture
model=$model_path
model_size=$model_size
model_sample_sha256=$model_sample
prompt_sha256=$prompt_sha
frontier=$frontier
generated_tokens=300
context=$context
prefill_chunk=2048
rdma_profile=odinlink
coordinator_rdma_device=odl_tb5_0
worker_rdma_device=odl_tb5_0
rdma_gid_index=n/a
candidate=$candidate
candidate_lane=$lane
baseline_id=$baseline
expected_fnv64=$expected
source_commit=$source_commit
source_dirty=0
dspark=0
EOF
}
make_run baseline-1 200.0 15.00 1
make_run baseline-2 201.0 15.10 1
make_run diverse-candidate 202.0 15.20 1
"$repo/scripts/diverse-bench-gate.py" create --lane A \
  --baseline "$dossier/baseline-1.csv" \
  --baseline "$dossier/baseline-2.csv" \
  --candidate "$dossier/diverse-candidate.csv" \
  --output "$dossier/diverse-summary.json" >/dev/null
diverse_hash=$(sha256sum "$dossier/diverse-summary.json" | awk '{print $1}')

make_run diverse-regressed 202.0 13.50 1
if "$repo/scripts/diverse-bench-gate.py" create --lane A \
  --baseline "$dossier/baseline-1.csv" \
  --baseline "$dossier/baseline-2.csv" \
  --candidate "$dossier/diverse-regressed.csv" \
  --output "$dossier/diverse-regressed-summary.json" >/dev/null 2>&1; then
  echo "error: diverse benchmark gate accepted a decode regression" >&2
  exit 1
fi

python3 - "$dossier/candidate.json" "$hash" "$diverse_hash" "$dossier" <<'PY'
import json
import hashlib
import sys
from pathlib import Path

path = Path(sys.argv[1])
digest = sys.argv[2]
diverse_digest = sys.argv[3]
dossier = Path(sys.argv[4])
value = json.loads(path.read_text())
value["source"]["dirty"] = False
kinds = [
    "ordinary-benchmark",
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
candidate_runs = iter(("baseline-1.csv", "baseline-2.csv"))
for kind in kinds:
    if kind == "cross-discipline-long":
        value["evidence"].append({"kind": kind, "path": "diverse-summary.json", "sha256": diverse_digest})
    elif kind == "candidate-benchmark":
        run = next(candidate_runs)
        run_digest = hashlib.sha256((dossier / run).read_bytes()).hexdigest()
        manifest_digest = hashlib.sha256((dossier / run.replace(".csv", ".manifest")).read_bytes()).hexdigest()
        value["evidence"].append({"kind": kind, "path": run, "sha256": run_digest,
                                  "manifest_sha256": manifest_digest})
    else:
        value["evidence"].append({"kind": kind, "path": "evidence.txt", "sha256": digest})
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

# Lane B is allowed to establish a new deterministic fingerprint, but only
# from independently recomputed logits and paired quality tables.
printf 'x' > "$DS4_RESEARCH_ROOT/fixture-model.gguf"
read -r baseline_id test_model_sample < <(python3 - "$DS4_RESEARCH_ROOT" "$DS4_RESEARCH_ROOT/fixture-model.gguf" <<'PY'
import hashlib
import csv
import io
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
model_path = Path(sys.argv[2])
model_size = model_path.stat().st_size
sample = hashlib.sha256(f"{model_size}\n".encode())
with model_path.open("rb") as stream:
    for offset in (0, max(0, model_size // 2 - 4 * 1024 * 1024),
                   max(0, model_size - 8 * 1024 * 1024)):
        stream.seek(offset)
        sample.update(stream.read(8 * 1024 * 1024))
model_sample = sample.hexdigest()
model_sha = hashlib.sha256(model_path.read_bytes()).hexdigest()
logit_value = {
    "source": "ds4-bench-frozen-teacher", "model": str(model_path),
    "backend": "rocm", "quality": False, "dspark": False,
    "dspark_strict": False, "quant_bits": 4, "prefix_tokens": 2048,
    "decode_step": 0, "position": 2048, "vocab": 4,
    "teacher_token": 3, "teacher_logit": 3.0, "argmax_id": 3,
    "argmax_logit": 3.0, "runner_up_id": 2, "runner_up_logit": 2.0,
    "top1_margin": 1.0, "teacher_gap": 0.0,
    "logits": [0.0, 1.0, 2.0, 3.0],
}
logit_files = []
for index in range(300):
    item = dict(logit_value)
    item["decode_step"] = index
    item["position"] = 2048 + index
    logit_files.append({
        "name": f"decode_{index:06d}.logits.json",
        "sha256": hashlib.sha256(json.dumps(item).encode()).hexdigest(),
    })
numerical_manifest = (
    f"model={model_path}\nmodel_size={model_size}\nmodel_sample_sha256={model_sample}\n"
    f"source_commit={'0' * 40}\nsource_dirty=0\ntoolchain_id=synthetic-old\n"
    f"prefix_tokens=2048\nfile_count=300\nfrozen_token_sha256={'c' * 64}\ndspark=0\n"
)
numerical_manifest_sha = hashlib.sha256(numerical_manifest.encode()).hexdigest()
fields = ["id", "target_tokens", "nll", "avg_nll", "api_top1_count",
          "api_top1_match", "api_pair_total", "api_pair_agree"]
quality_stream = io.StringIO(newline="")
writer = csv.DictWriter(quality_stream, fieldnames=fields, delimiter="\t")
writer.writeheader()
for index in range(100):
    average = 0.5 + 0.01 * (index % 5)
    writer.writerow({"id": f"case_{index:03d}", "target_tokens": 24,
                     "nll": average * 24, "avg_nll": average,
                     "api_top1_count": 24, "api_top1_match": 22,
                     "api_pair_total": 24, "api_pair_agree": 22})
quality_sha = hashlib.sha256(quality_stream.getvalue().encode()).hexdigest()
quality_manifest = (
    f"model={model_path}\nmodel_size={model_size}\n"
    f"model_sample_sha256={model_sample}\nsource_commit={'0' * 40}\n"
    "source_dirty=0\ndspark=0\n"
)
quality_manifest_sha = hashlib.sha256(quality_manifest.encode()).hexdigest()
record = {
    "schema_version": 1,
    "kind": "ds4-numerical-baseline",
    "key": {
        "model_sample_sha256": model_sample,
        "model_sha256": model_sha,
        "model_size": model_size,
        "quantization": "Q4_K",
        "source_commit": "0" * 40,
        "toolchain_id": "synthetic-old",
        "architecture": "gfx1151",
        "tp_degree": 2,
        "expert_split": "128/128",
        "decode_mode": "ordinary-greedy",
        "workload_id": "ds4-bench-tp-2048x300",
        "workload": {
            "prompt_sha256": "6dff0f4bc6000881259d96b2126b9c4f86f377efbaaa349e0a49d6da0435d34b",
            "frontier": "2048", "generated_tokens": "300",
            "context": "2560", "prefill_chunk": "2048", "dspark": "0",
            "frozen_token_sha256": "c" * 64,
        },
        "rdma_providers": ["odinlink", "roce-v2"],
    },
    "reference": {
        "fnv64": "1234567890abcdef",
        "numerical": {"files": logit_files, "manifest_sha256": numerical_manifest_sha},
        "quality": {"sha256": quality_sha, "manifest_sha256": quality_manifest_sha},
    },
    "thresholds": {
        "numerical": {
            "e_bound": 0.1, "max_abs": 0.1, "p99_abs": 0.1,
            "nmse": 0.0001, "tvd": 0.0001, "kl": 0.0001,
            "min_top5_overlap": 4, "min_top20_overlap": 4,
            "min_teacher_steps": 300, "allow_quality_difference": False,
        },
        "quality": {
            "min_cases": 100, "min_target_tokens": 2289,
            "max_mean_nll_delta": 0.0,
            "max_ci95_high_nll_delta": 0.02,
            "min_api_top1_rate_delta": 0.0,
            "min_api_pair_rate_delta": 0.0,
        },
    },
    "provenance": {"candidate_id": "synthetic", "lane_origin": "bootstrap"},
}
digest = hashlib.sha256(json.dumps(record, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
path = root / "baselines" / "sha256" / f"{digest}.json"
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
print(f"sha256:{digest} {model_sample}")
PY
)

"$repo/scripts/candidate-gate.py" init test-lane-b B >/dev/null
dossier=$DS4_RESEARCH_ROOT/candidates/test-lane-b
source_commit=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["source"]["commit"])' "$dossier/candidate.json")
test_model_size=1
test_model_path=$DS4_RESEARCH_ROOT/fixture-model.gguf
printf 'reviewed evidence\n' > "$dossier/evidence.txt"
hash=$(sha256sum "$dossier/evidence.txt" | awk '{print $1}')
make_run candidate-1 202.0 15.20 1 fedcba0987654321 B "$baseline_id" 2048 2560 6dff0f4bc6000881259d96b2126b9c4f86f377efbaaa349e0a49d6da0435d34b
make_run candidate-2 201.0 15.10 1 fedcba0987654321 B "$baseline_id" 2048 2560 6dff0f4bc6000881259d96b2126b9c4f86f377efbaaa349e0a49d6da0435d34b
make_run diverse-baseline-1 200.0 15.00 1 fedcba0987654321 B "$baseline_id"
make_run diverse-baseline-2 201.0 15.10 1 fedcba0987654321 B "$baseline_id"
make_run diverse-candidate 202.0 15.20 1 fedcba0987654321 B "$baseline_id"
"$repo/scripts/diverse-bench-gate.py" create --lane B \
  --baseline "$dossier/diverse-baseline-1.csv" \
  --baseline "$dossier/diverse-baseline-2.csv" \
  --candidate "$dossier/diverse-candidate.csv" \
  --output "$dossier/diverse-summary.json" >/dev/null

mkdir -p "$dossier/logits-reference" "$dossier/logits-candidate"
python3 - "$dossier" "$baseline_id" "$DS4_RESEARCH_ROOT/fixture-model.gguf" \
  "$source_commit" "$test_model_sample" <<'PY'
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
baseline_id = sys.argv[2]
model_path = sys.argv[3]
source_commit = sys.argv[4]
model_sample = sys.argv[5]
value = {
    "source": "ds4-bench-frozen-teacher", "model": model_path,
    "backend": "rocm", "quality": False, "dspark": False,
    "dspark_strict": False, "quant_bits": 4, "prefix_tokens": 2048,
    "decode_step": 0, "position": 2048, "vocab": 4,
    "teacher_token": 3, "teacher_logit": 3.0, "argmax_id": 3,
    "argmax_logit": 3.0, "runner_up_id": 2, "runner_up_logit": 2.0,
    "top1_margin": 1.0, "teacher_gap": 0.0,
    "logits": [0.0, 1.0, 2.0, 3.0],
}
for directory in ("logits-reference", "logits-candidate"):
    for index in range(300):
        item = dict(value)
        item["decode_step"] = index
        item["position"] = 2048 + index
        (root / directory / f"decode_{index:06d}.logits.json").write_text(json.dumps(item))
(root / "logits-reference" / "manifest").write_text(
    f"model={model_path}\nmodel_size=1\nmodel_sample_sha256={model_sample}\n"
    f"source_commit={'0' * 40}\nsource_dirty=0\ntoolchain_id=synthetic-old\n"
    f"prefix_tokens=2048\nfile_count=300\nfrozen_token_sha256={'c' * 64}\ndspark=0\n")
(root / "logits-candidate" / "manifest").write_text(
    f"model={model_path}\nmodel_size=1\nmodel_sample_sha256={model_sample}\n"
    f"source_commit={source_commit}\nsource_dirty=0\ntoolchain_id=synthetic-new\n"
    f"prefix_tokens=2048\nfile_count=300\nfrozen_token_sha256={'c' * 64}\ndspark=0\n")
numerical = {
    "baseline_id": baseline_id, "e_bound": 0.1, "max_abs": 0.1,
    "p99_abs": 0.1, "nmse": 0.0001, "tvd": 0.0001, "kl": 0.0001,
    "min_top5_overlap": 4, "min_top20_overlap": 4,
    "min_teacher_steps": 300,
}
(root / "numerical-thresholds.json").write_text(json.dumps(numerical, indent=2, sort_keys=True) + "\n")
quality = {
    "baseline_id": baseline_id, "min_cases": 100, "min_target_tokens": 2289,
    "max_mean_nll_delta": 0.0, "max_ci95_high_nll_delta": 0.02,
    "min_api_top1_rate_delta": 0.0, "min_api_pair_rate_delta": 0.0,
}
(root / "quality-thresholds.json").write_text(json.dumps(quality, indent=2, sort_keys=True) + "\n")
fields = ["id", "target_tokens", "nll", "avg_nll", "api_top1_count",
          "api_top1_match", "api_pair_total", "api_pair_agree"]
reference_averages = [0.5 + 0.01 * (index % 5) for index in range(100)]
deltas = [-0.01, 0.005, -0.005]
candidate_averages = [value + deltas[index % len(deltas)]
                      for index, value in enumerate(reference_averages)]
for name, averages in (("quality-reference.tsv", reference_averages),
                       ("quality-candidate.tsv", candidate_averages)):
    with (root / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for index, average in enumerate(averages):
            writer.writerow({"id": f"case_{index:03d}", "target_tokens": 24,
                             "nll": average * 24, "avg_nll": average,
                             "api_top1_count": 24, "api_top1_match": 22,
                             "api_pair_total": 24, "api_pair_agree": 22})
    manifest_source = "0" * 40 if name.startswith("quality-reference") else source_commit
    (root / name.replace(".tsv", ".manifest")).write_text(
        f"model={model_path}\nmodel_size=1\nmodel_sample_sha256={model_sample}\n"
        f"source_commit={manifest_source}\nsource_dirty=0\ndspark=0\n")
PY
"$repo/scripts/compare-teacher-logits.py" \
  "$dossier/logits-reference" "$dossier/logits-candidate" \
  --thresholds "$dossier/numerical-thresholds.json" \
  --output "$dossier/numerical-summary.json" >/dev/null
"$repo/scripts/compare-quality-scores.py" \
  "$dossier/quality-reference.tsv" "$dossier/quality-candidate.tsv" \
  --thresholds "$dossier/quality-thresholds.json" \
  --output "$dossier/quality-summary.json" >/dev/null

python3 - "$dossier/candidate.json" "$dossier" "$hash" "$baseline_id" \
  "$test_model_sample" "$DS4_RESEARCH_ROOT/fixture-model.gguf" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
dossier = Path(sys.argv[2])
generic_hash = sys.argv[3]
baseline_id = sys.argv[4]
model_sample = sys.argv[5]
model_path = sys.argv[6]
value = json.loads(path.read_text())
value["source"]["dirty"] = False
value["baseline_id"] = baseline_id
value["model"] = {"path": model_path, "sample_sha256": model_sample,
                  "sha256": hashlib.sha256(Path(model_path).read_bytes()).hexdigest(),
                  "size": 1, "quantization": "Q4_K"}
value["toolchain"] = {"id": "synthetic-new"}
value["transport"] = {"providers": ["odinlink"]}
value["evidence"] = []
generic = ["ordinary-benchmark", "long-context", "transport-proof",
           "ordinary-regression", "fable-review", "grok-review",
           "full-logits", "teacher-forced", "semantic-retrieval"]
for kind in generic:
    value["evidence"].append({"kind": kind, "path": "evidence.txt", "sha256": generic_hash})
for name in ("candidate-1.csv", "candidate-2.csv"):
    digest = hashlib.sha256((dossier / name).read_bytes()).hexdigest()
    manifest_digest = hashlib.sha256((dossier / name.replace(".csv", ".manifest")).read_bytes()).hexdigest()
    value["evidence"].append({"kind": "candidate-benchmark", "path": name,
                              "sha256": digest, "manifest_sha256": manifest_digest})
for kind, name in (("cross-discipline-long", "diverse-summary.json"),
                   ("numerical-envelope", "numerical-summary.json"),
                   ("reference-score", "quality-summary.json")):
    digest = hashlib.sha256((dossier / name).read_bytes()).hexdigest()
    value["evidence"].append({"kind": kind, "path": name, "sha256": digest})
path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
PY

# A previously passing summary cannot self-certify after its raw logits change.
python3 - "$dossier/logits-candidate/decode_000000.logits.json" <<'PY'
import json
import sys
path = sys.argv[1]
value = json.load(open(path))
value["logits"] = [0.0, 1.0, 4.0, 3.0]
open(path, "w").write(json.dumps(value))
PY
if "$repo/scripts/candidate-gate.py" check test-lane-b >/dev/null 2>&1; then
  echo "error: lane B accepted a stale passing summary over changed raw logits" >&2
  exit 1
fi
python3 - "$dossier/logits-candidate/decode_000000.logits.json" <<'PY'
import json
import sys
path = sys.argv[1]
value = json.load(open(path))
value["logits"] = [0.0, 1.0, 2.0, 3.0]
open(path, "w").write(json.dumps(value))
PY

"$repo/scripts/candidate-gate.py" check test-lane-b >/dev/null
"$repo/scripts/candidate-gate.py" promote test-lane-b >/dev/null
new_baseline=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["new_baseline_id"])' "$dossier/PROMOTED.json")
[[ $new_baseline =~ ^sha256:[0-9a-f]{64}$ ]]
test -s "$DS4_RESEARCH_ROOT/baselines/sha256/${new_baseline#sha256:}.json"
if "$repo/scripts/candidate-gate.py" promote test-lane-b >/dev/null 2>&1; then
  echo "error: append-only gate promoted the same candidate twice" >&2
  exit 1
fi

echo "PASS candidate-gate"
