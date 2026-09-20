#!/usr/bin/env python3
"""Compare frozen-token decode logits without autoregressive cascade.

The tool is diagnostic unless an explicit versioned threshold JSON is supplied.
It never turns its built-in defaults into a lane-B acceptance policy.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import math
import re
import runpy
import shlex
import sys
from pathlib import Path

import numpy as np

from ds4_gate_stats import StatsError, bca_interval, wilson_one_sided


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path, *, reference: bool = False) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    logits = value.get("logits")
    vocab = value.get("vocab")
    if not isinstance(vocab, int) or not isinstance(logits, list) or len(logits) != vocab:
        raise ValueError(f"{path}: malformed logits/vocab")
    if any(not isinstance(item, (int, float)) or not math.isfinite(item)
           for item in logits):
        raise ValueError(f"{path}: logits contain null, NaN, Inf, or non-numbers")
    allowed_sources = {
        "ds4-bench-frozen-teacher",
        "ds4-score-official-frozen-teacher",
    }
    if reference:
        allowed_sources.add("ds4-canonical-oracle")
    if value.get("source") not in allowed_sources:
        role = "reference" if reference else "candidate"
        raise ValueError(f"{path}: invalid {role} logit producer")
    for field in ("prefix_tokens", "decode_step", "position", "teacher_token",
                  "quant_bits", "argmax_id", "runner_up_id"):
        if not isinstance(value.get(field), int):
            raise ValueError(f"{path}: missing or invalid {field}")
    if value.get("source") == "ds4-score-official-frozen-teacher":
        if not isinstance(value.get("case_id"), str) or not value["case_id"]:
            raise ValueError(f"{path}: missing or invalid case_id")
        if not isinstance(value.get("case_step"), int):
            raise ValueError(f"{path}: missing or invalid case_step")
    for field in ("quality", "dspark", "dspark_strict"):
        if type(value.get(field)) is not bool:
            raise ValueError(f"{path}: missing or invalid {field}")
    return value


def top_ids(values: list[float], count: int) -> list[int]:
    selected = heapq.nlargest(
        count, enumerate(values), key=lambda item: (item[1], -item[0]))
    return [index for index, _ in selected]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def score_fixture(path: Path, root: Path, start: int, count: int) -> tuple[list, str]:
    rows = [line.split('\t') for line in path.read_text().splitlines()
            if line and not line.startswith('#')][start:start + count]
    require(len(rows) == count, 'incomplete selected fixture')
    seen, snapshot = set(), []
    for row in rows:
        require(3 <= len(row) <= 4 and row[0] not in seen, 'invalid or duplicate fixture case')
        seen.add(row[0])
        hashes = []
        for index, name in enumerate(row[1:]):
            require(bool(name) or index == 2, 'empty prompt or continuation path')
            hashes.append(sha256(root / name) if name else None)
        snapshot.append([row[0], *hashes])
    digest = hashlib.sha256(json.dumps(snapshot, separators=(',', ':')).encode()).hexdigest()
    return rows, digest


def validate_score_capture(directory: Path, scores_path: Path, model: str,
                           case_ids: list[str] | None = None) -> None:
    with scores_path.open() as stream:
        scores = list(csv.DictReader(stream, delimiter='\t'))
    ids = [row['id'] for row in scores]
    require(bool(ids) and len(set(ids)) == len(ids), 'empty or duplicate score cases')
    require(case_ids is None or ids == case_ids, 'score/fixture case mismatch')
    expected = []
    for row in scores:
        prefix, count = int(row['prompt_tokens']), int(row['target_tokens'])
        require(prefix > 0 and count > 0, 'invalid case length')
        expected.extend((row['id'], step, prefix) for step in range(count))
    files = sorted(directory.glob('decode_*.logits.json'))
    require([p.name for p in files] == [f'decode_{i:06d}.logits.json' for i in range(len(expected))],
            'incomplete or noncontiguous teacher dumps')
    signature = None
    for index, (path, (case, step, prefix)) in enumerate(zip(files, expected)):
        value = load(path)
        require(value['source'] == 'ds4-score-official-frozen-teacher' and
                value.get('backend') == 'rocm' and value.get('model') == model,
                'wrong logit producer/backend/model')
        require(all(value[key] is False for key in ('quality', 'dspark', 'dspark_strict')),
                'teacher capture must use ordinary production arithmetic')
        for key, wanted in (('case_id', case), ('case_step', step), ('prefix_tokens', prefix),
                            ('position', prefix + step), ('decode_step', index)):
            require(value.get(key) == wanted and type(value.get(key)) is type(wanted),
                    f'{path.name}: invalid {key}')
        logits, vocab = value['logits'], value['vocab']
        require(type(vocab) is int and vocab > 1 and
                all(type(x) in (int, float) for x in logits), 'invalid vocabulary/logits')
        require(type(value['quant_bits']) is int and value['quant_bits'] in (2, 4), 'invalid quantization')
        current = (vocab, value['quant_bits'])
        require(signature is None or signature == current, 'inconsistent vocabulary/quantization')
        signature = current
        for key in ('teacher_token', 'argmax_id', 'runner_up_id'):
            require(type(value[key]) is int and 0 <= value[key] < vocab, f'invalid {key}')
        top = top_ids(logits, 2)
        require(top == [value['argmax_id'], value['runner_up_id']], 'invalid top-two metadata')
        for key, computed in (('teacher_logit', logits[value['teacher_token']]),
                ('argmax_logit', logits[top[0]]), ('runner_up_logit', logits[top[1]]),
                ('top1_margin', logits[top[0]] - logits[top[1]]),
                ('teacher_gap', logits[top[0]] - logits[value['teacher_token']])):
            actual = value.get(key)
            # Round-tripped %.9g floats and FP32-computed differences have
            # small serialization error; this is not an SDK drift tolerance.
            require(type(actual) in (int, float) and math.isfinite(actual) and
                    math.isclose(actual, computed, rel_tol=2e-6, abs_tol=2e-5),
                    f'inconsistent {key}')


def capture_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description='Validate a teacher capture without quality thresholds')
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--fixture', required=True, type=Path)
    parser.add_argument('--start-case', required=True, type=int)
    parser.add_argument('--cases', required=True, type=int)
    parser.add_argument('--dumps', type=Path)
    parser.add_argument('--scores', type=Path)
    parser.add_argument('--model')
    args = parser.parse_args(argv)
    try:
        require(args.start_case >= 0 and args.cases > 0, 'invalid case range')
        require(all((args.dumps, args.scores, args.model)) or not any((args.dumps, args.scores, args.model)),
                'capture validation requires dumps, scores and model')
        rows, digest = score_fixture(args.fixture, args.root, args.start_case, args.cases)
        if args.dumps:
            validate_score_capture(args.dumps, args.scores, args.model, [row[0] for row in rows])
        print(digest)
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f'quality-logits: {error}', file=sys.stderr)
        return 1


def verify_score_capture_artifacts(directory: Path, meta: dict, *, fixture_bound: bool = True) -> Path:
    """Reopen shared score, terminal and complete teacher-dump evidence."""
    terminal = runpy.run_path(str(Path(__file__).with_name('compare-quality-scores.py')))['terminal_proof']
    identity_keys = ('ds4_sha256', 'scorer_sha256', 'files_sha256')
    if fixture_bound:
        identity_keys += ('fixture_content_sha256',)
    require(all(re.fullmatch(r'[0-9a-f]{64}', meta.get(key, '')) for key in identity_keys),
            'invalid teacher capture identity')
    scores = Path(meta['scores_path'])
    terminal(scores, meta, required=True)
    require(sha256(directory / 'files.sha256') == meta['files_sha256'], 'changed dump inventory')
    inventory = []
    for line in (directory / 'files.sha256').read_text().splitlines():
        digest, name = line.split('  ', 1)
        require(re.fullmatch(r'decode_[0-9]{6}.logits.json', name) is not None,
                'invalid dump inventory filename')
        require(sha256(directory / name) == digest, 'changed teacher dump')
        inventory.append(name)
    require(inventory == [p.name for p in sorted(directory.glob('decode_*.logits.json'))],
            'dump inventory coverage differs')
    require(len(inventory) == int(meta['teacher_positions']), 'wrong teacher position count')
    validate_score_capture(directory, scores, meta['model'])
    with scores.open() as stream:
        require(len(list(csv.DictReader(stream, delimiter='\t'))) == int(meta['cases']),
                'wrong case count')
    return scores


def verify_deepseek_captures(directories: tuple[Path, Path], manifests: tuple[dict, dict]) -> None:
    """An attested SDK diagnostic only; never an admission or threshold bypass."""
    require(manifests[0].get('run_id') != manifests[1].get('run_id'), 'teacher comparison reuses a process run')
    prior_env = prior_features = None
    for directory, meta in zip(directories, manifests):
        require(meta.get('teacher_arm') == 'deepseek-ordinary' and meta.get('model_arch') == 'deepseek4',
                'DeepSeek comparison requires two deepseek-ordinary captures')
        require(meta['source_dirty'] == '0' and meta['rdma_profile'] == 'roce-v2',
                'DeepSeek capture requires clean source and RoCE v2')
        verify_score_capture_artifacts(directory, meta)
        for rank in ('coordinator', 'worker'):
            items = [item.split('=', 1) for item in shlex.split(meta[rank + '_env'])]
            env = dict(items)
            require(len(env) == len(items), 'duplicate effective setting')
            require(env.pop('DS4_BENCH_RUN_ID', None) == meta['run_id'], 'mixed effective run identity')
            require(env.get('DS4_TP_RDMA_LOGITS') == '1' and env.get('DS4_TP_GREEDY_TOP2', '0') == '0' and
                    env.get('DS4_GLM5_NATIVE_DRAFT', '0') == '0' and not any(k.startswith('DS4_DSPARK_') for k in env),
                    'DeepSeek teacher capture requires ordinary full RDMA logits')
            require(prior_env is None or env == prior_env, 'DeepSeek effective settings differ')
            prior_env = env
            feature = meta[rank + '_features'].replace('startup rank=1 ', 'startup rank=0 ')
            require(prior_features is None or feature == prior_features, 'DeepSeek negotiated features differ')
            prior_features = feature


def verify_global_captures(directories: tuple[Path, Path], manifests: tuple[dict, dict],
                           repeat: bool, fixture: Path, source_root: Path) -> str:
    """Diagnostic only: legacy GLM captures lack capture-time fixture digests."""
    key = 'DS4_ROCM_GLM5_Q4K_PREFILL_GLOBAL'
    require(manifests[0].get('run_id') != manifests[1].get('run_id'),
            'teacher comparison reuses a process run')
    prior_env = None
    modes = []
    for directory, meta in zip(directories, manifests):
        require(meta.get('teacher_arm') == 'kda-tp' and
                meta.get('model_arch') == 'glm5-next' and meta['source_dirty'] == '0' and
                meta['rdma_profile'] == 'roce-v2', 'global capture requires clean GLM over RoCE v2')
        require(meta['cases'] == '1', 'global diagnostic requires one case per process for engagement proof')
        require(sha256(fixture) == meta['quality_input_sha256'], 'global fixture manifest differs')
        rows, content_digest = score_fixture(fixture, source_root, int(meta['start_case']), 1)
        scores = verify_score_capture_artifacts(directory, meta, fixture_bound=False)
        validate_score_capture(directory, scores, meta['model'], [rows[0][0]])
        rank_modes = []
        for rank, index in (('coordinator', 0), ('worker', 1)):
            items = [item.split('=', 1) for item in shlex.split(meta[rank + '_env'])]
            env = dict(items)
            require(len(env) == len(items), 'duplicate effective setting')
            require(env.pop('DS4_BENCH_RUN_ID', None) == meta['run_id'], 'mixed effective run identity')
            mode = env.pop(key, None)
            require(mode in ('0', '1'), 'global capture requires explicit GLOBAL=0 or GLOBAL=1')
            rank_modes.append(mode)
            required = dict(DS4_ROCM_GLM5_Q4K_PREFILL_GROUPED='1',
                            DS4_ROCM_GLM5_Q4K_PREFILL_PARTITION='256',
                            DS4_GLM5_KDA_TP='1', DS4_GLM5_KDA_OUTPUT_KSLICE='0',
                            DS4_GLM5_NATIVE_DRAFT='0', DS4_TP_RDMA_LOGITS='1',
                            DS4_GLM5_SPARSE_BATCH_BRIDGE='1', DS4_GLM5_PREFILL_PROOF='1')
            require(all(env.get(k) == v for k, v in required.items()) and
                    env.get('DS4_GLM5_NEXT_PREFILL_BATCH') in ('512', '768', '1024') and
                    env.get('DS4_TP_GREEDY_TOP2', '0') == '0' and
                    not any(k.startswith('DS4_DSPARK_') for k in env),
                    'invalid global capture prerequisites')
            require(prior_env is None or env == prior_env, 'global effective settings differ')
            prior_env = env
            log = Path(meta[rank + '_log_path']).read_text(errors='replace')
            proof = 'ds4-tp: transport proof requested=rdma active=rdma payload_fallback_calls=0 failed=0'
            require([line for line in log.splitlines() if line.startswith('ds4-tp: transport proof ')] == [proof]
                    and re.search(r'rdma GID index \d+ \(RoCE v2\)', log) is not None and
                    re.search(r'expanded_weight_cache_bytes=0(?:\s|$)', log) is not None and
                    re.search(r'expanded_weight_cache_bytes=[1-9]', log) is None,
                    'global capture lacks RoCE/zero-fallback/cache proof')
            engaged = re.findall(r'global expert domain engaged rank=(\d+) rows=(\d+) groups=1', log)
            batch = env['DS4_GLM5_NEXT_PREFILL_BATCH']
            require((bool(engaged) and (str(index), batch) in engaged and
                     all(r == str(index) and m in ('512', '768', '1024') for r, m in engaged))
                    if mode == '1' else not engaged,
                    'global capture engagement disagrees with rank/setting')
            if mode == '0':
                require(f'grouped prefill engaged rows={batch} groups={int(batch) // 256} ' in log,
                        'global control lacks grouped engagement')
            with scores.open() as stream:
                prefix = int(next(csv.DictReader(stream, delimiter='\t'))['prompt_tokens'])
                require(prefix >= int(batch),
                        'global capture has no full-batch prefix')
            if rank == 'coordinator':
                proofs = re.findall(r'GLM5 prefill execution rank=0 start=0 prompt_tokens=(\d+) '
                                    r'requested_batch=(\d+) batched_tiles=(\d+) batched_rows=(\d+) '
                                    r'scalar_rows=(\d+) min_tile=(\d+) max_tile=(\d+)', log)
                require(len(proofs) == 1, 'missing or ambiguous global prefill proof')
                tokens, requested, tiles, batched, scalar, smallest, largest = map(int, proofs[0])
                require(tokens == prefix and requested == largest == int(batch) and
                        tiles > 0 and smallest > 0 and batched + scalar == prefix and
                        batched >= int(batch) and 0 <= scalar < 256,
                        'global capture did not execute the declared batched prefix')
        require(rank_modes[0] == rank_modes[1], 'global rank settings differ')
        modes.append(rank_modes[0])
    require((modes[0] == modes[1]) if repeat else modes == ['0', '1'],
            'global comparison requires same-mode repeat' if repeat else
            'global comparison requires explicit GLOBAL=0 versus GLOBAL=1')
    return content_digest


def probability_metrics(reference: np.ndarray, candidate: np.ndarray) -> tuple[float, float, float, float]:
    ref_max = float(reference.max())
    cand_max = float(candidate.max())
    ref_exp = np.exp(reference - ref_max)
    cand_exp = np.exp(candidate - cand_max)
    ref_sum = float(ref_exp.sum(dtype=np.float64))
    cand_sum = float(cand_exp.sum(dtype=np.float64))
    ref_log_z = ref_max + math.log(ref_sum)
    cand_log_z = cand_max + math.log(cand_sum)
    p = ref_exp / ref_sum
    q = cand_exp / cand_sum
    tvd = 0.5 * float(np.abs(p - q).sum(dtype=np.float64))
    kl = float(np.sum(p * ((reference - ref_log_z) -
                           (candidate - cand_log_z)), dtype=np.float64))
    return tvd, max(0.0, kl), ref_log_z, cand_log_z


def distribution(values: list[float]) -> dict[str, float]:
    """Return a stable, explicit distribution summary for a non-empty sample."""
    sample = np.asarray(values, dtype=np.float64)
    if sample.size == 0 or not np.all(np.isfinite(sample)):
        raise ValueError("distribution input must be non-empty and finite")
    return {
        "min": float(sample.min()),
        "mean": float(sample.mean(dtype=np.float64)),
        "median": float(np.quantile(sample, 0.50, method="linear")),
        "p90": float(np.quantile(sample, 0.90, method="linear")),
        "p95": float(np.quantile(sample, 0.95, method="linear")),
        "p99": float(np.quantile(sample, 0.99, method="linear")),
        "max": float(sample.max()),
    }


def load_thresholds(path: Path | None) -> dict | None:
    if path is None:
        return None
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if value.get("schema_version") == 2:
        required = {
            "schema_version", "baseline_id", "min_teacher_steps",
            "allow_quality_difference", "decision", "distribution", "safety",
        }
        if set(value) != required:
            raise ValueError(f"{path}: invalid v2 threshold fields")
        if not isinstance(value["baseline_id"], str) or not value["baseline_id"]:
            raise ValueError(f"{path}: baseline_id must be a non-empty string")
        if (not isinstance(value["min_teacher_steps"], int) or
                value["min_teacher_steps"] < 300):
            raise ValueError(f"{path}: min_teacher_steps must be at least 300")
        if type(value["allow_quality_difference"]) is not bool:
            raise ValueError(f"{path}: allow_quality_difference must be boolean")
        decision = value["decision"]
        distribution_limits = value["distribution"]
        safety = value["safety"]
        if not all(isinstance(item, dict)
                   for item in (decision, distribution_limits, safety)):
            raise ValueError(f"{path}: v2 threshold sections must be objects")
        if set(decision) != {
                "e_bound", "confidence_level",
                "max_near_tie_cluster_rate_upper"}:
            raise ValueError(f"{path}: invalid decision threshold fields")
        if set(distribution_limits) != {
                "bootstrap_method", "bootstrap_resamples", "bootstrap_seed",
                "cluster_mode", "block_size", "min_clusters",
                "max_mean_kl_upper", "max_mean_tvd_upper",
                "max_mean_teacher_nll_delta_upper",
                "min_same_top1_cluster_rate_lower", "soft_limits",
                "max_soft_exceedance_cluster_rate_upper"}:
            raise ValueError(f"{path}: invalid distribution threshold fields")
        if distribution_limits.get("bootstrap_method") != "bca":
            raise ValueError(f"{path}: bootstrap_method must be bca")
        if distribution_limits.get("cluster_mode") != "case-or-contiguous-block":
            raise ValueError(f"{path}: cluster_mode must be case-or-contiguous-block")
        if (not isinstance(distribution_limits["bootstrap_resamples"], int) or
                distribution_limits["bootstrap_resamples"] < 1000 or
                not isinstance(distribution_limits["bootstrap_seed"], int) or
                isinstance(distribution_limits["bootstrap_seed"], bool) or
                distribution_limits["bootstrap_seed"] < 0):
            raise ValueError(f"{path}: invalid deterministic bootstrap settings")
        if (not isinstance(distribution_limits["block_size"], int) or
                isinstance(distribution_limits["block_size"], bool) or
                distribution_limits["block_size"] < 4 or
                not isinstance(distribution_limits["min_clusters"], int) or
                isinstance(distribution_limits["min_clusters"], bool) or
                distribution_limits["min_clusters"] < 2):
            raise ValueError(f"{path}: invalid cluster coverage settings")
        soft = distribution_limits["soft_limits"]
        if not isinstance(soft, dict) or set(soft) != {
                "centered_p99_abs", "centered_nrms", "kl", "tvd"}:
            raise ValueError(f"{path}: invalid soft-limit fields")
        if set(safety) != {
                "max_centered_abs", "max_centered_nrms", "max_kl", "max_tvd",
                "max_abs_teacher_nll_delta"}:
            raise ValueError(f"{path}: invalid hard-safety fields")
        scalar_limits = {
            "e_bound": decision["e_bound"],
            "max_near_tie_cluster_rate_upper":
                decision["max_near_tie_cluster_rate_upper"],
            "confidence_level": decision["confidence_level"],
            "max_mean_kl_upper": distribution_limits["max_mean_kl_upper"],
            "max_mean_tvd_upper": distribution_limits["max_mean_tvd_upper"],
            "max_mean_teacher_nll_delta_upper":
                distribution_limits["max_mean_teacher_nll_delta_upper"],
            "min_same_top1_cluster_rate_lower":
                distribution_limits["min_same_top1_cluster_rate_lower"],
            "max_soft_exceedance_cluster_rate_upper":
                distribution_limits["max_soft_exceedance_cluster_rate_upper"],
            **{f"soft.{key}": item for key, item in soft.items()},
            **{f"safety.{key}": item for key, item in safety.items()},
        }
        for key, item in scalar_limits.items():
            if not isinstance(item, (int, float)) or not math.isfinite(item):
                raise ValueError(f"{path}: {key} must be finite")
        for key in (
                "e_bound", "max_near_tie_cluster_rate_upper",
                "max_mean_kl_upper", "max_mean_tvd_upper",
                "max_soft_exceedance_cluster_rate_upper",
                *[f"soft.{key}" for key in soft],
                *[f"safety.{key}" for key in safety]):
            if scalar_limits[key] < 0:
                raise ValueError(f"{path}: {key} must be nonnegative")
        for key in ("confidence_level", "min_same_top1_cluster_rate_lower",
                    "max_near_tie_cluster_rate_upper",
                    "max_soft_exceedance_cluster_rate_upper"):
            item = scalar_limits[key]
            if not 0 <= item <= 1:
                raise ValueError(f"{path}: {key} must be between zero and one")
        if not 0.5 < decision["confidence_level"] < 1.0:
            raise ValueError(f"{path}: confidence_level must be between 0.5 and 1")
        return value

    required = {
        "baseline_id", "e_bound", "max_abs", "p99_abs", "nmse", "tvd", "kl",
        "min_top5_overlap", "min_top20_overlap",
    }
    missing = required - value.keys()
    if missing:
        raise ValueError(f"{path}: missing thresholds: {', '.join(sorted(missing))}")
    if not isinstance(value["baseline_id"], str) or not value["baseline_id"]:
        raise ValueError(f"{path}: baseline_id must be a non-empty string")
    for key in ("e_bound", "max_abs", "p99_abs", "nmse", "tvd", "kl"):
        item = value[key]
        if not isinstance(item, (int, float)) or not math.isfinite(item) or item < 0:
            raise ValueError(f"{path}: {key} must be finite and nonnegative")
    for key, limit in (("min_top5_overlap", 5), ("min_top20_overlap", 20)):
        if not isinstance(value[key], int) or not 0 <= value[key] <= limit:
            raise ValueError(f"{path}: {key} must be between 0 and {limit}")
    return value


def compare_pair(reference: dict, candidate: dict, thresholds: dict | None,
                 allow_quality_difference: bool = False,
                 require_production_quality: bool = False) -> dict:
    if require_production_quality and (reference["quality"] or candidate["quality"]):
        raise ValueError("Q8 decode tile comparison requires quality=false in both arms")
    for field in ("vocab", "prefix_tokens", "decode_step", "position",
                  "teacher_token", "quant_bits", "quality", "dspark",
                  "dspark_strict"):
        if field == "quality" and allow_quality_difference:
            continue
        if reference.get(field) != candidate.get(field):
            raise ValueError(f"metadata {field}: {reference.get(field)!r} != {candidate.get(field)!r}")
    if (reference.get("source") == "ds4-score-official-frozen-teacher" or
            candidate.get("source") == "ds4-score-official-frozen-teacher"):
        for field in ("case_id", "case_step"):
            if reference.get(field) != candidate.get(field):
                raise ValueError(
                    f"metadata {field}: {reference.get(field)!r} != {candidate.get(field)!r}")

    ref_values = reference["logits"]
    cand_values = candidate["logits"]
    ref = np.asarray(ref_values, dtype=np.float64)
    cand = np.asarray(cand_values, dtype=np.float64)
    differences = np.abs(ref - cand)
    signed_difference = cand - ref
    common_shift = float(signed_difference.mean(dtype=np.float64))
    centered_difference = signed_difference - common_shift
    centered_absolute = np.abs(centered_difference)
    centered_reference = ref - float(ref.mean(dtype=np.float64))
    centered_reference_rms = math.sqrt(float(np.mean(
        centered_reference * centered_reference, dtype=np.float64)))
    centered_error_rms = math.sqrt(float(np.mean(
        centered_difference * centered_difference, dtype=np.float64)))
    centered_nrms = (centered_error_rms / max(centered_reference_rms, 1.0e-30)
                     if centered_error_rms else 0.0)
    sum_diff_sq = float(np.dot(differences, differences))
    sum_ref_sq = float(np.dot(ref, ref))
    ref_top20 = top_ids(ref_values, min(20, len(ref_values)))
    cand_top20 = top_ids(cand_values, min(20, len(cand_values)))
    ref_margin = ref[ref_top20[0]] - ref[ref_top20[1]] if len(ref_top20) > 1 else math.inf
    cand_margin = cand[cand_top20[0]] - cand[cand_top20[1]] if len(cand_top20) > 1 else math.inf
    teacher_token = reference["teacher_token"]
    if not 0 <= teacher_token < len(ref):
        raise ValueError(f"teacher_token {teacher_token} is outside vocab {len(ref)}")
    tvd, kl, ref_log_z, cand_log_z = probability_metrics(ref, cand)
    reference_teacher_nll = ref_log_z - float(ref[teacher_token])
    candidate_teacher_nll = cand_log_z - float(cand[teacher_token])
    result = {
        "decode_step": reference["decode_step"],
        "position": reference["position"],
        "teacher_token": reference["teacher_token"],
        "reference_argmax": ref_top20[0],
        "candidate_argmax": cand_top20[0],
        "argmax_equal": ref_top20[0] == cand_top20[0],
        "reference_margin": ref_margin,
        "candidate_margin": cand_margin,
        "max_abs": float(differences.max()),
        "p99_abs": float(np.quantile(differences, 0.99, method="higher")),
        "nmse": sum_diff_sq / sum_ref_sq if sum_ref_sq else (0.0 if not sum_diff_sq else math.inf),
        "tvd": tvd,
        "kl": kl,
        "top5_overlap": len(set(ref_top20[:5]) & set(cand_top20[:5])),
        "top20_overlap": len(set(ref_top20) & set(cand_top20)),
        "mean_signed_error": float(np.mean(cand - ref)),
        "positive_error_fraction": float(np.mean(cand > ref)),
        "reference_teacher_nll": reference_teacher_nll,
        "candidate_teacher_nll": candidate_teacher_nll,
        "teacher_nll_delta": candidate_teacher_nll - reference_teacher_nll,
        "common_logit_shift": common_shift,
        "centered_max_abs": float(centered_absolute.max()),
        "centered_p99_abs": float(np.quantile(
            centered_absolute, 0.99, method="higher")),
        "centered_nrms": centered_nrms,
    }
    if reference.get("source") == "ds4-score-official-frozen-teacher":
        result["case_id"] = reference["case_id"]
        result["case_step"] = reference["case_step"]
    if thresholds is None:
        result["envelope_pass"] = None
        result["far_margin_inversion"] = None
        result["hard_safety_pass"] = None
        result["soft_exceedance"] = None
    elif thresholds.get("schema_version") == 2:
        decision = thresholds["decision"]
        soft = thresholds["distribution"]["soft_limits"]
        safety = thresholds["safety"]
        result["far_margin_inversion"] = bool(
            not result["argmax_equal"] and
            ref_margin > 2.0 * decision["e_bound"])
        result["hard_safety_pass"] = bool(
            result["centered_max_abs"] <= safety["max_centered_abs"] and
            result["centered_nrms"] <= safety["max_centered_nrms"] and
            result["kl"] <= safety["max_kl"] and
            result["tvd"] <= safety["max_tvd"] and
            abs(result["teacher_nll_delta"]) <=
                safety["max_abs_teacher_nll_delta"])
        result["soft_exceedance"] = bool(
            result["centered_p99_abs"] > soft["centered_p99_abs"] or
            result["centered_nrms"] > soft["centered_nrms"] or
            result["kl"] > soft["kl"] or result["tvd"] > soft["tvd"])
        result["envelope_pass"] = bool(
            not result["far_margin_inversion"] and
            result["hard_safety_pass"])
    else:
        result["far_margin_inversion"] = bool(
            not result["argmax_equal"] and ref_margin > 2.0 * thresholds["e_bound"])
        result["envelope_pass"] = bool(
            not result["far_margin_inversion"] and
            result["max_abs"] <= thresholds["max_abs"] and
            result["p99_abs"] <= thresholds["p99_abs"] and
            result["nmse"] <= thresholds["nmse"] and
            result["tvd"] <= thresholds["tvd"] and
            result["kl"] <= thresholds["kl"] and
            result["top5_overlap"] >= thresholds["min_top5_overlap"] and
            result["top20_overlap"] >= thresholds["min_top20_overlap"])
        result["hard_safety_pass"] = result["envelope_pass"]
        result["soft_exceedance"] = None
    return result


def cluster_step_indices(steps: list[dict], block_size: int) -> tuple[list[np.ndarray], str]:
    """Group dependent teacher positions before statistical inference.

    score_official dumps provide a real case boundary. A single teacher stream
    does not, so it is split into deterministic contiguous blocks. Individual
    positions remain diagnostics, never independent trials.
    """
    have_case = ["case_id" in item for item in steps]
    if any(have_case) and not all(have_case):
        raise ValueError("teacher dumps mix case-clustered and unclustered positions")
    groups: list[np.ndarray] = []
    if all(have_case):
        case_ids = list(dict.fromkeys(str(item["case_id"]) for item in steps))
        for case_id in case_ids:
            groups.append(np.asarray([
                index for index, item in enumerate(steps)
                if str(item["case_id"]) == case_id
            ], dtype=np.int64))
        unit = "case"
    else:
        groups = [
            np.arange(start, min(start + block_size, len(steps)), dtype=np.int64)
            for start in range(0, len(steps), block_size)
        ]
        unit = "contiguous-block"
    if len(groups) < 2 or any(group.size == 0 for group in groups):
        raise ValueError("teacher comparison has fewer than two dependency clusters")
    return groups, unit


def clustered_mean_interval(values: list[float], groups: list[np.ndarray], *,
                            confidence_level: float, n_resamples: int,
                            seed: int) -> dict[str, float]:
    sample = np.asarray(values, dtype=np.float64)

    def statistic(cluster_indices: np.ndarray) -> float:
        positions = np.concatenate([groups[int(index)] for index in cluster_indices])
        return float(sample[positions].mean(dtype=np.float64))

    return bca_interval(
        len(groups), statistic, confidence_level=confidence_level,
        n_resamples=n_resamples, seed=seed)


def main() -> int:
    if sys.argv[1:2] == ['--validate-capture']:
        return capture_main(sys.argv[2:])
    parser = argparse.ArgumentParser()
    parser.add_argument("reference_dir", type=Path)
    parser.add_argument("candidate_dir", type=Path)
    parser.add_argument("--thresholds", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--steps-output", type=Path,
                        help="write every per-position metric as JSON Lines")
    parser.add_argument('--score-fixture', type=Path,
                        help='fixture TSV for the diagnostic global grouping modes')
    parser.add_argument('--score-root', type=Path,
                        help='fixture path root for the diagnostic global grouping modes')
    parser.add_argument("--allow-quality-difference", action="store_true",
                        help="explicitly compare an unfused quality oracle with an optimized path")
    parser.add_argument(
        "--score-arm-mode", choices=(
            "kda-tp", "kda-kslice", "repeat", "q8-decode-tile", "full-split-order-null",
            "null-vs-kslice", "fallback-vs-kslice",
            "attn-scalar-vs-f32-gemm",
            "attn-scalar-vs-f32-gemm-sync",
            "attn-scalar-vs-f32-gemm-postdiv",
            "attn-scalar-vs-f32-gemm-default-math",
            "attn-scalar-vs-f32-gemm-postdiv-default-math",
            "attn-scalar-vs-f32-gemm-postdiv-pv-scalar",
            "attn-scalar-vs-f32-gemm-postdiv-pv-scalar-default-math",
            "attn-scalar-vs-f32-gemm-postdiv-score-pv-scalar",
            "attn-scalar-vs-exact-split",
            "attn-repeat", "deepseek-sdk", "q4k-global", "q4k-global-repeat"),
        help="required score_official arm relationship; deepseek-sdk is threshold-free only")
    args = parser.parse_args()
    deepseek_sdk = args.score_arm_mode == 'deepseek-sdk'
    global_capture = args.score_arm_mode in ('q4k-global', 'q4k-global-repeat')
    global_fixture_digest = None
    try:
        if deepseek_sdk and (args.thresholds or args.allow_quality_difference):
            raise ValueError('DeepSeek SDK comparison is diagnostic-only, without thresholds or quality-mode changes')
        if args.score_arm_mode == "q8-decode-tile" and args.allow_quality_difference:
            raise ValueError("Q8 decode tile comparison forbids --allow-quality-difference")
        if global_capture and args.allow_quality_difference:
            raise ValueError('global capture forbids --allow-quality-difference')
        if global_capture:
            require(not args.thresholds, 'global comparison is diagnostic-only pending capture-time fixture binding')
            require(args.score_fixture is not None and args.score_root is not None,
                    'global comparison requires --score-fixture and --score-root')
        else:
            require(args.score_fixture is None and args.score_root is None,
                    'score fixture arguments require a global comparison mode')
        thresholds = load_thresholds(args.thresholds)
        reference_files = sorted(args.reference_dir.glob("decode_*.logits.json"))
        candidate_files = sorted(args.candidate_dir.glob("decode_*.logits.json"))
        if not reference_files or [p.name for p in reference_files] != [p.name for p in candidate_files]:
            raise ValueError("reference and candidate decode-logit file sets must be non-empty and identical")
        first_pair = (load(reference_files[0], reference=True),
                      load(candidate_files[0]))
        if global_capture:
            require(all(value['quant_bits'] == 4 for value in first_pair),
                    'global grouping diagnostic requires Q4 captures')
        score_official_source = any(
            value.get("source") == "ds4-score-official-frozen-teacher"
            for value in first_pair)
        if score_official_source:
            if args.score_arm_mode is None:
                raise ValueError("--score-arm-mode is required for score_official dumps")
            def read_manifest(path: Path) -> dict[str, str]:
                values: dict[str, str] = {}
                for line in path.read_text(encoding="utf-8").splitlines():
                    if "=" not in line:
                        raise ValueError(f"{path}: malformed manifest line")
                    key, value = line.split("=", 1)
                    if not key or key in values:
                        raise ValueError(f"{path}: duplicate or empty manifest key {key!r}")
                    values[key] = value
                return values

            reference_manifest = read_manifest(args.reference_dir / "manifest")
            candidate_manifest = read_manifest(args.candidate_dir / "manifest")
            identity_fields = (
                "producer", "source_commit", "source_dirty", "model",
                "model_size", "model_sample_sha256", "ds4_sha256",
                "scorer_sha256", "quality_input_sha256", "start_case",
                "cases", "teacher_positions", "rdma_profile",
            )
            if deepseek_sdk:
                identity_fields = tuple(key for key in identity_fields if key not in
                                        ('ds4_sha256', 'scorer_sha256')) + (
                    'inference_source_commit', 'model_arch', 'context', 'fixture_content_sha256',
                    'quality_launcher_sha256', 'capture_validator_sha256')
            if global_capture:
                identity_fields += ('inference_source_commit', 'model_arch', 'context',
                                    'quality_launcher_sha256')
            missing = [field for field in identity_fields
                       if field not in reference_manifest or
                       field not in candidate_manifest]
            if missing:
                raise ValueError("teacher-logit manifests lack identity fields: " +
                                 ", ".join(missing))
            for field in identity_fields:
                if reference_manifest[field] != candidate_manifest[field]:
                    raise ValueError(
                        f"manifest {field}: {reference_manifest[field]!r} != "
                        f"{candidate_manifest[field]!r}")
            if deepseek_sdk:
                verify_deepseek_captures((args.reference_dir, args.candidate_dir),
                                        (reference_manifest, candidate_manifest))
            if global_capture:
                global_fixture_digest = verify_global_captures(
                    (args.reference_dir, args.candidate_dir),
                    (reference_manifest, candidate_manifest),
                    args.score_arm_mode == 'q4k-global-repeat', args.score_fixture, args.score_root)
            if not deepseek_sdk:
                expected_arms = {
                    "kda-tp": ("kda-off", "kda-tp"),
                    "kda-kslice": ("kda-tp", "kda-kslice"),
                    "repeat": ("kda-kslice", "kda-kslice"),
                    "q8-decode-tile": ("kda-tp", "kda-tp"),
                    "q4k-global": ("kda-tp", "kda-tp"),
                    "q4k-global-repeat": ("kda-tp", "kda-tp"),
                    "full-split-order-null": ("kda-tp", "kda-tp"),
                    "null-vs-kslice": ("kda-tp", "kda-kslice"),
                    "fallback-vs-kslice": ("kda-tp", "kda-kslice"),
                    "attn-scalar-vs-f32-gemm": ("attn-scalar", "attn-gemm-f32"),
                    "attn-scalar-vs-f32-gemm-sync": ("attn-scalar", "attn-gemm-f32"),
                    "attn-scalar-vs-f32-gemm-postdiv": ("attn-scalar", "attn-gemm-f32"),
                    "attn-scalar-vs-f32-gemm-default-math":
                        ("attn-scalar", "attn-gemm-f32"),
                    "attn-scalar-vs-f32-gemm-postdiv-default-math":
                        ("attn-scalar", "attn-gemm-f32"),
                    "attn-scalar-vs-f32-gemm-postdiv-pv-scalar":
                        ("attn-scalar", "attn-gemm-f32"),
                    "attn-scalar-vs-f32-gemm-postdiv-pv-scalar-default-math":
                        ("attn-scalar", "attn-gemm-f32"),
                    "attn-scalar-vs-f32-gemm-postdiv-score-pv-scalar":
                        ("attn-scalar", "attn-gemm-f32"),
                    "attn-scalar-vs-exact-split":
                        ("attn-scalar", "attn-exact-split"),
                    "attn-repeat": ("attn-scalar", "attn-scalar"),
                }[args.score_arm_mode]
                actual_arms = (reference_manifest.get("teacher_arm"),
                               candidate_manifest.get("teacher_arm"))
                if actual_arms != expected_arms:
                    raise ValueError(
                        f"score arm relationship {actual_arms!r} != {expected_arms!r}")

                def parse_env(encoded: str) -> dict[str, str]:
                    values: dict[str, str] = {}
                    for item in shlex.split(encoded):
                        if "=" not in item:
                            raise ValueError(f"malformed extra_env item {item!r}")
                        key, value = item.split("=", 1)
                        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) or key in values:
                            raise ValueError(f"duplicate or invalid extra_env key {key!r}")
                        values[key] = value
                    return values

                ref_env = parse_env(reference_manifest.get("extra_env", ""))
                cand_env = parse_env(candidate_manifest.get("extra_env", ""))
                selectors = {"DS4_GLM5_KDA_TP", "DS4_GLM5_KDA_OUTPUT_KSLICE"}
                if global_capture:
                    selectors.add('DS4_ROCM_GLM5_Q4K_PREFILL_GLOBAL')
                    # The effective maps are authoritative; declared switches
                    # must also agree with each corresponding rank map.
                    for meta, declared in ((reference_manifest, ref_env), (candidate_manifest, cand_env)):
                        effective = dict(item.split('=', 1) for item in shlex.split(meta['coordinator_env']))
                        require(all(effective.get(k) == v for k, v in declared.items()),
                                'declared global settings differ from effective settings')
                if args.score_arm_mode == "q8-decode-tile":
                    tile_key = "DS4_ROCM_GLM5_Q8_DECODE_TILE"
                    if (ref_env.get(tile_key), cand_env.get(tile_key)) != ("0", "1"):
                        raise ValueError("Q8 decode tile comparison requires explicit TILE=0 versus TILE=1")
                    selectors.add(tile_key)
                elif args.score_arm_mode == "full-split-order-null":
                    null_key = "DS4_ROCM_BF16_FULL_SPLIT_ORDER"
                    if null_key in ref_env or cand_env.get(null_key) != "1":
                        raise ValueError(
                            "full-split-order null requires the legal reorder only "
                            "in the candidate arm")
                    selectors.add(null_key)
                elif args.score_arm_mode == "null-vs-kslice":
                    null_key = "DS4_ROCM_BF16_FULL_SPLIT_ORDER"
                    if ref_env.get(null_key) != "1" or null_key in cand_env:
                        raise ValueError(
                            "null-vs-kslice requires the legal reorder only in "
                            "the reference arm")
                    selectors.add(null_key)
                elif args.score_arm_mode == "fallback-vs-kslice":
                    fallback_key = "DS4_ROCM_DISABLE_BF16_DECODE_MLP64"
                    if ref_env.get(fallback_key) != "1" or fallback_key in cand_env:
                        raise ValueError(
                            "fallback-vs-kslice requires the independent BF16 "
                            "fallback only in the reference arm")
                    selectors.add(fallback_key)
                elif args.score_arm_mode == "attn-scalar-vs-exact-split":
                    exact_key = "DS4_ROCM_GLM_CAUSAL_ATTN_EXACT_SPLIT"
                    research_keys = {
                        "DS4_ROCM_GLM_CAUSAL_ATTN_GEMM_NOPE",
                        "DS4_ROCM_GLM_CAUSAL_ATTN_GEMM_NOPE_F32",
                        "DS4_ROCM_GLM_CAUSAL_ATTN_GEMM_NOPE_POSTDIV",
                        "DS4_ROCM_GLM_CAUSAL_ATTN_GEMM_NOPE_PV_SCALAR",
                        "DS4_ROCM_GLM_CAUSAL_ATTN_GEMM_NOPE_SCORE_SCALAR",
                        "DS4_ROCM_GLM_CAUSAL_ATTN_GEMM_NOPE_DEFAULT_MATH",
                        "DS4_ROCM_GLM_CAUSAL_ATTN_GEMM_NOPE_SYNC",
                    }
                    if ref_env.get(exact_key) != "0" or exact_key in cand_env or \
                            any(key in ref_env or key in cand_env
                                for key in research_keys):
                        raise ValueError(
                            "exact-split comparison requires rollback only in the "
                            "scalar arm and no research selectors")
                    selectors.add(exact_key)
                elif args.score_arm_mode.startswith("attn-scalar-vs-f32-gemm"):
                    nope_key = "DS4_ROCM_GLM_CAUSAL_ATTN_GEMM_NOPE"
                    f32_key = "DS4_ROCM_GLM_CAUSAL_ATTN_GEMM_NOPE_F32"
                    if (nope_key in ref_env or f32_key in ref_env or
                            cand_env.get(nope_key) != "1" or
                            cand_env.get(f32_key) != "1"):
                        raise ValueError(
                            "attention comparison requires FP32 NoPE GEMM only "
                            "in the candidate arm")
                    selectors.update((nope_key, f32_key))
                    diagnostic = {
                        "attn-scalar-vs-f32-gemm": None,
                        "attn-scalar-vs-f32-gemm-sync":
                            "DS4_ROCM_GLM_CAUSAL_ATTN_GEMM_NOPE_SYNC",
                        "attn-scalar-vs-f32-gemm-postdiv":
                            "DS4_ROCM_GLM_CAUSAL_ATTN_GEMM_NOPE_POSTDIV",
                        "attn-scalar-vs-f32-gemm-default-math":
                            "DS4_ROCM_GLM_CAUSAL_ATTN_GEMM_NOPE_DEFAULT_MATH",
                        "attn-scalar-vs-f32-gemm-postdiv-default-math": None,
                        "attn-scalar-vs-f32-gemm-postdiv-pv-scalar": None,
                        "attn-scalar-vs-f32-gemm-postdiv-pv-scalar-default-math":
                            None,
                        "attn-scalar-vs-f32-gemm-postdiv-score-pv-scalar": None,
                    }[args.score_arm_mode]
                    diagnostic_keys = {
                        "DS4_ROCM_GLM_CAUSAL_ATTN_GEMM_NOPE_SYNC",
                        "DS4_ROCM_GLM_CAUSAL_ATTN_GEMM_NOPE_POSTDIV",
                        "DS4_ROCM_GLM_CAUSAL_ATTN_GEMM_NOPE_DEFAULT_MATH",
                        "DS4_ROCM_GLM_CAUSAL_ATTN_GEMM_NOPE_PV_SCALAR",
                        "DS4_ROCM_GLM_CAUSAL_ATTN_GEMM_NOPE_SCORE_SCALAR",
                    }
                    if args.score_arm_mode == \
                            "attn-scalar-vs-f32-gemm-postdiv-default-math":
                        required = {
                            "DS4_ROCM_GLM_CAUSAL_ATTN_GEMM_NOPE_POSTDIV",
                            "DS4_ROCM_GLM_CAUSAL_ATTN_GEMM_NOPE_DEFAULT_MATH",
                        }
                        if any(cand_env.get(key) != "1" for key in required) or \
                                any(key not in required and key in cand_env
                                    for key in diagnostic_keys):
                            raise ValueError(
                                "attention comparison requires postdiv and "
                                "default-math selectors")
                        selectors.update(required)
                    elif args.score_arm_mode == \
                            "attn-scalar-vs-f32-gemm-postdiv-score-pv-scalar":
                        required = {
                            "DS4_ROCM_GLM_CAUSAL_ATTN_GEMM_NOPE_POSTDIV",
                            "DS4_ROCM_GLM_CAUSAL_ATTN_GEMM_NOPE_PV_SCALAR",
                            "DS4_ROCM_GLM_CAUSAL_ATTN_GEMM_NOPE_SCORE_SCALAR",
                        }
                        if any(cand_env.get(key) != "1" for key in required) or \
                                any(key not in required and key in cand_env
                                    for key in diagnostic_keys):
                            raise ValueError(
                                "attention comparison requires postdiv, scalar "
                                "score, and scalar PV selectors")
                        selectors.update(required)
                    elif args.score_arm_mode == \
                            "attn-scalar-vs-f32-gemm-postdiv-pv-scalar":
                        required = {
                            "DS4_ROCM_GLM_CAUSAL_ATTN_GEMM_NOPE_POSTDIV",
                            "DS4_ROCM_GLM_CAUSAL_ATTN_GEMM_NOPE_PV_SCALAR",
                        }
                        if any(cand_env.get(key) != "1" for key in required) or \
                                any(key not in required and key in cand_env
                                    for key in diagnostic_keys):
                            raise ValueError(
                                "attention comparison requires postdiv and "
                                "PV-scalar selectors")
                        selectors.update(required)
                    elif args.score_arm_mode == \
                            "attn-scalar-vs-f32-gemm-postdiv-pv-scalar-default-math":
                        required = {
                            "DS4_ROCM_GLM_CAUSAL_ATTN_GEMM_NOPE_POSTDIV",
                            "DS4_ROCM_GLM_CAUSAL_ATTN_GEMM_NOPE_PV_SCALAR",
                            "DS4_ROCM_GLM_CAUSAL_ATTN_GEMM_NOPE_DEFAULT_MATH",
                        }
                        if any(cand_env.get(key) != "1" for key in required) or \
                                any(key not in required and key in cand_env
                                    for key in diagnostic_keys):
                            raise ValueError(
                                "attention comparison requires postdiv, PV-scalar, "
                                "and default-math selectors")
                        selectors.update(required)
                    elif diagnostic is None:
                        if any(key in cand_env for key in diagnostic_keys):
                            raise ValueError(
                                "base attention comparison forbids repair selectors")
                    else:
                        if cand_env.get(diagnostic) != "1" or any(
                                key != diagnostic and key in cand_env
                                for key in diagnostic_keys):
                            raise ValueError(
                                f"attention comparison requires only {diagnostic}")
                        selectors.add(diagnostic)
                if ({k: v for k, v in ref_env.items() if k not in selectors} !=
                        {k: v for k, v in cand_env.items() if k not in selectors}):
                    raise ValueError("score arms differ outside the declared selectors")
                selector_expectations = {
                    "kda-off": ("0", "0"),
                    "kda-tp": ("1", "0"),
                    "kda-kslice": ("1", "1"),
                    "attn-scalar": ("1", "0"),
                    "attn-gemm-f32": ("1", "0"),
                    "attn-exact-split": ("1", "0"),
                }
                for manifest, env, arm in (
                        (reference_manifest, ref_env, actual_arms[0]),
                        (candidate_manifest, cand_env, actual_arms[1])):
                    expected_tp, expected_slice = selector_expectations[arm]
                    if (env.get("DS4_GLM5_KDA_TP"),
                            env.get("DS4_GLM5_KDA_OUTPUT_KSLICE")) != (
                                expected_tp, expected_slice):
                        raise ValueError(f"manifest arm {arm} has the wrong KDA selectors")
                    expected_features = re.compile(
                        rf"GLM5 TP features: kda_tp={expected_tp} "
                        rf"kda_output_kslice={expected_slice}$")
                    for field in ("coordinator_features", "worker_features"):
                        if not expected_features.search(manifest.get(field, "")):
                            raise ValueError(f"manifest arm {arm} has invalid {field}")
        elif args.score_arm_mode is not None:
            raise ValueError("--score-arm-mode applies only to score_official dumps")

        steps = [compare_pair(first_pair[0], first_pair[1], thresholds,
                              args.allow_quality_difference,
                              global_capture or args.score_arm_mode == "q8-decode-tile")]
        for ref_path, cand_path in zip(reference_files[1:], candidate_files[1:]):
            steps.append(compare_pair(load(ref_path, reference=True),
                                      load(cand_path), thresholds,
                                      args.allow_quality_difference,
                                      global_capture or args.score_arm_mode == "q8-decode-tile"))
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"teacher-logits: FAIL {error}", file=sys.stderr)
        return 1

    v2 = thresholds is not None and thresholds.get("schema_version") == 2
    mismatches = [item for item in steps if not item["argmax_equal"]]
    breaches = ([item for item in steps if not item["envelope_pass"]]
                if thresholds is not None else [])
    far_margin = ([item for item in steps if item["far_margin_inversion"]]
                  if thresholds is not None else [])
    mismatch_reference_margins = [
        float(item["reference_margin"]) for item in mismatches]
    soft_exceedances = ([item for item in steps if item["soft_exceedance"]]
                        if v2 else [])
    hard_safety_breaches = ([item for item in steps
                             if item["hard_safety_pass"] is False]
                            if v2 else [])
    distributions = {
        key: distribution([float(item[key]) for item in steps])
        for key in (
            "max_abs", "p99_abs", "nmse", "tvd", "kl",
            "reference_margin", "candidate_margin", "top5_overlap",
            "top20_overlap", "mean_signed_error", "positive_error_fraction",
            "reference_teacher_nll", "candidate_teacher_nll",
            "teacher_nll_delta", "common_logit_shift", "centered_max_abs",
            "centered_p99_abs", "centered_nrms",
        )
    }
    aggregate_gate = None
    aggregate_passed = True
    if v2:
        decision = thresholds["decision"]
        limits = thresholds["distribution"]
        confidence = float(decision["confidence_level"])
        try:
            clusters, cluster_unit = cluster_step_indices(
                steps, limits["block_size"])
        except ValueError as error:
            print(f"teacher-logits: FAIL invalid dependency clustering: {error}",
                  file=sys.stderr)
            return 1
        bootstrap = {
            "confidence_level": confidence,
            "method": limits["bootstrap_method"],
            "resamples": limits["bootstrap_resamples"],
            "seed": limits["bootstrap_seed"],
            "cluster_mode": limits["cluster_mode"],
            "cluster_unit": cluster_unit,
            "block_size": limits["block_size"],
            "clusters": len(clusters),
        }
        try:
            mean_kl = clustered_mean_interval(
                [item["kl"] for item in steps],
                clusters,
                confidence_level=confidence,
                n_resamples=limits["bootstrap_resamples"],
                seed=limits["bootstrap_seed"],
            )
            mean_tvd = clustered_mean_interval(
                [item["tvd"] for item in steps],
                clusters,
                confidence_level=confidence,
                n_resamples=limits["bootstrap_resamples"],
                seed=limits["bootstrap_seed"] + 1,
            )
            mean_teacher_nll_delta = clustered_mean_interval(
                [item["teacher_nll_delta"] for item in steps],
                clusters,
                confidence_level=confidence,
                n_resamples=limits["bootstrap_resamples"],
                seed=limits["bootstrap_seed"] + 2,
            )
            same_top_clusters = sum(
                all(steps[int(index)]["argmax_equal"] for index in group)
                for group in clusters)
            near_tie_clusters = sum(
                any(not steps[int(index)]["argmax_equal"] and
                    not steps[int(index)]["far_margin_inversion"]
                    for index in group)
                for group in clusters)
            soft_clusters = sum(
                any(steps[int(index)]["soft_exceedance"] for index in group)
                for group in clusters)
            same_top = wilson_one_sided(
                same_top_clusters, len(clusters), confidence)
            near_tie = wilson_one_sided(
                near_tie_clusters, len(clusters), confidence)
            soft_rate = wilson_one_sided(
                soft_clusters, len(clusters), confidence)
        except StatsError as error:
            print(f"teacher-logits: FAIL invalid statistical gate: {error}",
                  file=sys.stderr)
            return 1
        checks = {
            "teacher_coverage": len(steps) >= thresholds["min_teacher_steps"],
            "cluster_coverage": len(clusters) >= limits["min_clusters"],
            "mean_kl": mean_kl["one_sided_upper"] <=
                limits["max_mean_kl_upper"],
            "mean_tvd": mean_tvd["one_sided_upper"] <=
                limits["max_mean_tvd_upper"],
            "mean_teacher_nll_delta":
                mean_teacher_nll_delta["one_sided_upper"] <=
                limits["max_mean_teacher_nll_delta_upper"],
            "same_top1_cluster_rate": same_top["one_sided_lower"] >=
                limits["min_same_top1_cluster_rate_lower"],
            "near_tie_cluster_rate": near_tie["one_sided_upper"] <=
                decision["max_near_tie_cluster_rate_upper"],
            "soft_exceedance_cluster_rate": soft_rate["one_sided_upper"] <=
                limits["max_soft_exceedance_cluster_rate_upper"],
        }
        aggregate_passed = all(checks.values())
        aggregate_gate = {
            "bootstrap": bootstrap,
            "mean_kl": mean_kl,
            "mean_tvd": mean_tvd,
            "mean_teacher_nll_delta": mean_teacher_nll_delta,
            "same_top1_cluster_rate": same_top,
            "same_top1_clusters": same_top_clusters,
            "near_tie_cluster_rate": near_tie,
            "near_tie_clusters": near_tie_clusters,
            "soft_exceedance_cluster_rate": soft_rate,
            "soft_exceedance_clusters": soft_clusters,
            "soft_exceedances": len(soft_exceedances),
            "checks": checks,
            "passed": aggregate_passed,
        }
    case_summaries = []
    if all("case_id" in item for item in steps):
        case_ids = list(dict.fromkeys(item["case_id"] for item in steps))
        for case_id in case_ids:
            case_steps = [item for item in steps if item["case_id"] == case_id]
            case_summaries.append({
                "case_id": case_id,
                "steps": len(case_steps),
                "argmax_mismatches": sum(
                    not item["argmax_equal"] for item in case_steps),
                "teacher_nll_reference_mean": distribution([
                    item["reference_teacher_nll"] for item in case_steps
                ])["mean"],
                "teacher_nll_candidate_mean": distribution([
                    item["candidate_teacher_nll"] for item in case_steps
                ])["mean"],
                "teacher_nll_delta_mean": distribution([
                    item["teacher_nll_delta"] for item in case_steps
                ])["mean"],
                "kl": distribution([item["kl"] for item in case_steps]),
                "tvd": distribution([item["tvd"] for item in case_steps]),
                "max_abs": distribution([
                    item["max_abs"] for item in case_steps]),
            })

    summary = {
        "baseline_id": thresholds["baseline_id"] if thresholds else None,
        "allow_quality_difference": args.allow_quality_difference,
        "mode": "gate" if thresholds else "diagnostic",
        "score_arm_mode": args.score_arm_mode,
        "fixture_content_sha256_at_comparison": global_fixture_digest,
        "capture_time_fixture_bound": False if global_capture else None,
        "steps": len(steps),
        "argmax_mismatches": len(mismatches),
        "far_margin_inversions": len(far_margin) if thresholds else None,
        "first_argmax_mismatch": mismatches[0] if mismatches else None,
        "argmax_mismatch_reference_margin": (
            distribution(mismatch_reference_margins)
            if mismatch_reference_margins else None),
        "max_argmax_mismatch_reference_margin": (
            max(mismatch_reference_margins)
            if mismatch_reference_margins else None),
        "envelope_breaches": len(breaches) if thresholds else None,
        "hard_safety_breaches": len(hard_safety_breaches) if v2 else None,
        "first_envelope_breach": breaches[0] if breaches else None,
        "aggregate_gate": aggregate_gate,
        "max_abs": max(item["max_abs"] for item in steps),
        "max_p99_abs": max(item["p99_abs"] for item in steps),
        "max_nmse": max(item["nmse"] for item in steps),
        "max_tvd": max(item["tvd"] for item in steps),
        "max_kl": max(item["kl"] for item in steps),
        "min_top5_overlap": min(item["top5_overlap"] for item in steps),
        "min_top20_overlap": min(item["top20_overlap"] for item in steps),
        "max_abs_mean_signed_error": max(
            abs(item["mean_signed_error"]) for item in steps),
        "max_abs_common_logit_shift": max(
            abs(item["common_logit_shift"]) for item in steps),
        "max_centered_abs": max(item["centered_max_abs"] for item in steps),
        "max_centered_p99_abs": max(
            item["centered_p99_abs"] for item in steps),
        "max_centered_nrms": max(item["centered_nrms"] for item in steps),
        "positive_error_fraction_range": [
            min(item["positive_error_fraction"] for item in steps),
            max(item["positive_error_fraction"] for item in steps),
        ],
        "distributions": distributions,
        "case_summaries": case_summaries,
        "sources": {
            "reference_dir": str(args.reference_dir.resolve()),
            "candidate_dir": str(args.candidate_dir.resolve()),
            "reference_manifest": str((args.reference_dir / "manifest").resolve()),
            "reference_manifest_sha256": sha256(args.reference_dir / "manifest"),
            "candidate_manifest": str((args.candidate_dir / "manifest").resolve()),
            "candidate_manifest_sha256": sha256(args.candidate_dir / "manifest"),
            "pairs": [
                {
                    "name": reference.name,
                    "reference_sha256": sha256(reference),
                    "candidate_sha256": sha256(candidate),
                }
                for reference, candidate in zip(reference_files, candidate_files)
            ],
        },
        "thresholds_sha256": sha256(args.thresholds) if args.thresholds else None,
        "worst_steps": {
            key: max(steps, key=lambda item: item[key])
            for key in ("max_abs", "p99_abs", "nmse", "tvd", "kl",
                        "centered_max_abs", "centered_p99_abs", "centered_nrms")
        },
        "passed": (not breaches and not far_margin and aggregate_passed
                   if thresholds is not None else None),
    }
    encoded = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    if args.steps_output:
        with args.steps_output.open("w", encoding="utf-8") as handle:
            for item in steps:
                handle.write(json.dumps(item, sort_keys=True) + "\n")
    sys.stdout.write(encoded)
    if thresholds is not None and not summary["passed"]:
        print("teacher-logits: FAIL versioned lane-B envelope", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
