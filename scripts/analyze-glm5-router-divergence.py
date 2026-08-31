#!/usr/bin/env python3
"""Locate routed-expert selection amplification between two GLM traces."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
from pathlib import Path

import numpy as np


def load_gguf_directory(repo: Path, model: Path):
    source = repo / "scripts" / "probe-glm5-next-kda-payload.py"
    spec = importlib.util.spec_from_file_location("glm5_kda_payload", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_directory(model)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        while chunk := fp.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_env(path: Path) -> dict[str, list[str]]:
    values: dict[str, list[str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values.setdefault(key, []).append(value)
    return values


def one(env: dict[str, list[str]], key: str) -> str:
    values = env.get(key, [])
    if len(values) != 1:
        raise ValueError(f"expected one {key} in run.env, got {values!r}")
    return values[0]


def metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    if reference.shape != candidate.shape or reference.size == 0:
        raise ValueError("metric arrays have incompatible shapes")
    delta = candidate.astype(np.float64) - reference.astype(np.float64)
    ref = reference.astype(np.float64)
    denominator = max(float(np.dot(ref, ref)), 1.0e-30)
    return {
        "max_abs": float(np.max(np.abs(delta))),
        "mae": float(np.mean(np.abs(delta))),
        "nmse": float(np.dot(delta, delta) / denominator),
        "mean_signed_error": float(np.mean(delta)),
        "positive_error_fraction": float(np.mean(delta > 0.0)),
    }


def stable_router(weight: np.ndarray, bias: np.ndarray, hidden: np.ndarray):
    logits = np.asarray(weight @ hidden, dtype=np.float32)
    probability = np.asarray(
        1.0 / (1.0 + np.exp(-logits)), dtype=np.float32)
    adjusted = np.asarray(probability + bias, dtype=np.float32)
    order = np.argsort(-adjusted, kind="stable")
    return order, adjusted


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parent.parent
    for path in (args.model, args.reference, args.candidate):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    ref_env = read_env(args.reference / "run.env")
    cand_env = read_env(args.candidate / "run.env")
    identity_keys = (
        "source_head", "test_source_sha256", "launcher_sha256",
        "binary_sha256", "model_size", "model_sample_sha256",
        "source_diff_sha256", "rdma_profile", "text_prompt_sha256",
        "text_teacher_ids_sha256", "trace_token",
    )
    identity = {}
    for key in identity_keys:
        ref_value = one(ref_env, key)
        cand_value = one(cand_env, key)
        if ref_value != cand_value:
            raise ValueError(
                f"trace identity mismatch for {key}: "
                f"{ref_value!r} != {cand_value!r}")
        identity[key] = ref_value
    if one(ref_env, "kda_output_kslice") != "0" or \
       one(cand_env, "kda_output_kslice") != "1":
        raise ValueError("expected full reference and K-slice candidate")

    data_start, tensors = load_gguf_directory(repo, args.model)
    layer_re = re.compile(r"trace\.l([0-9]+)\.router_ids\.i32$")
    ref_layers = {
        int(match.group(1))
        for path in args.reference.glob("trace.l*.router_ids.i32")
        if (match := layer_re.match(path.name))
    }
    cand_layers = {
        int(match.group(1))
        for path in args.candidate.glob("trace.l*.router_ids.i32")
        if (match := layer_re.match(path.name))
    }
    if not ref_layers or ref_layers != cand_layers:
        raise ValueError("reference/candidate router layer sets differ")

    results = []
    used_hashes: dict[str, str] = {}
    first_order_difference = None
    first_set_difference = None
    for layer in sorted(ref_layers):
        tensor_name = f"blk.{layer}.ffn_gate_inp.weight"
        bias_name = f"blk.{layer}.exp_probs_b.bias"
        weight_desc = tensors.get(tensor_name)
        bias_desc = tensors.get(bias_name)
        if weight_desc is None or weight_desc[:2] != ((4096, 288), 0):
            raise ValueError(f"unexpected router tensor {tensor_name}: {weight_desc}")
        if bias_desc is None or bias_desc[:2] != ((288,), 0):
            raise ValueError(f"unexpected router bias {bias_name}: {bias_desc}")
        weight = np.memmap(
            args.model, dtype="<f4", mode="r",
            offset=data_start + weight_desc[2], shape=(288, 4096))
        bias = np.memmap(
            args.model, dtype="<f4", mode="r",
            offset=data_start + bias_desc[2], shape=(288,))

        arms = {}
        for arm, directory in (("reference", args.reference),
                               ("candidate", args.candidate)):
            ids_path = directory / f"trace.l{layer}.router_ids.i32"
            weights_path = directory / f"trace.l{layer}.router_weights.f32"
            hidden_path = directory / f"trace.l{layer}.ffn_hidden.f32"
            output_path = directory / f"trace.l{layer}.output_hc.f32"
            ids = np.fromfile(ids_path, dtype="<i4")
            selected_weights = np.fromfile(weights_path, dtype="<f4")
            hidden = np.fromfile(hidden_path, dtype="<f4")
            output = np.fromfile(output_path, dtype="<f4")
            if ids.shape != (8,) or selected_weights.shape != (8,) or \
               hidden.shape != (4096,) or output.shape != (16384,):
                raise ValueError(f"unexpected trace shape in {directory} layer {layer}")
            order, adjusted = stable_router(weight, bias, hidden)
            if not np.array_equal(ids, order[:8].astype(np.int32)):
                raise ValueError(
                    f"CPU router did not reproduce {arm} layer {layer}: "
                    f"trace={ids.tolist()} cpu={order[:8].tolist()}")
            adjacent = adjusted[order[:7]] - adjusted[order[1:8]]
            arms[arm] = {
                "ids": ids.tolist(),
                "boundary_margin": float(adjusted[order[7]] - adjusted[order[8]]),
                "min_selected_adjacent_margin": float(np.min(adjacent)),
                "min_selected_adjacent_position": int(np.argmin(adjacent)),
                "selected_weight_min": float(np.min(selected_weights)),
                "selected_weight_max": float(np.max(selected_weights)),
                "hidden": hidden,
                "output": output,
            }
            for path in (ids_path, weights_path, hidden_path, output_path):
                key = f"{arm}/{path.name}"
                used_hashes[key] = sha256(path)

        reference_ids = arms["reference"]["ids"]
        candidate_ids = arms["candidate"]["ids"]
        same_order = reference_ids == candidate_ids
        same_set = set(reference_ids) == set(candidate_ids)
        if not same_order and first_order_difference is None:
            first_order_difference = layer
        if not same_set and first_set_difference is None:
            first_set_difference = layer
        ref_set = set(reference_ids)
        cand_set = set(candidate_ids)
        results.append({
            "layer": layer,
            "same_order": same_order,
            "same_set": same_set,
            "reference_only_ids": sorted(ref_set - cand_set),
            "candidate_only_ids": sorted(cand_set - ref_set),
            "reference": {key: value for key, value in arms["reference"].items()
                          if key not in ("hidden", "output")},
            "candidate": {key: value for key, value in arms["candidate"].items()
                          if key not in ("hidden", "output")},
            "hidden_difference": metrics(
                arms["reference"]["hidden"], arms["candidate"]["hidden"]),
            "output_hc_difference": metrics(
                arms["reference"]["output"], arms["candidate"]["output"]),
        })

    report = {
        "schema": "ds4-glm5-router-divergence-v1",
        "model": str(args.model.resolve()),
        "model_size": args.model.stat().st_size,
        "reference": str(args.reference.resolve()),
        "candidate": str(args.candidate.resolve()),
        "identity": identity,
        "layers": len(results),
        "first_order_difference": first_order_difference,
        "first_set_difference": first_set_difference,
        "order_difference_count": sum(not row["same_order"] for row in results),
        "set_difference_count": sum(not row["same_set"] for row in results),
        "used_files_sha256": used_hashes,
        "per_layer": results,
    }
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(
        f"first_order_difference={first_order_difference} "
        f"first_set_difference={first_set_difference} "
        f"order_difference_count={report['order_difference_count']} "
        f"set_difference_count={report['set_difference_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
