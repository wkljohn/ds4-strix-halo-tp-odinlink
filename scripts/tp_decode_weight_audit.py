#!/usr/bin/env python3
"""Report the model bytes touched by one TP decode token.

This is an accounting tool, not a performance model.  It follows the current
inter-process TP=2 graph in ds4.c: attention heads/output groups and the shared
expert intermediate are divided between ranks, while the compressor/indexer
state builders remain replicated.  Routed-expert bytes are reported for a
balanced three-of-six selection; the actual value is token-dependent.
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import sys
from collections import defaultdict
from pathlib import Path


LAYER_RE = re.compile(r"^blk\.(\d+)\.(.+)$")


def load_gguf_parser(repo: Path):
    parser_path = repo / "gguf-tools/mixed/splice_mixed_expert_layers_gguf.py"
    spec = importlib.util.spec_from_file_location("ds4_gguf_parser", parser_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load GGUF parser from {parser_path}")
    module = importlib.util.module_from_spec(spec)
    # Python 3.14 dataclasses expect the module to be registered while its
    # class annotations are evaluated.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def classify(suffix: str) -> tuple[str, float, str]:
    """Return (category, bytes-read fraction per rank, reason)."""
    if suffix == "attn_q_b.weight":
        return "already_tp_divided", 0.5, "Q-b output rows follow owned heads"
    if suffix in {"attn_output_a.weight", "attn_output_b.weight"}:
        return "already_tp_divided", 0.5, "owned output groups and matching K slice"
    if suffix in {
        "ffn_gate_shexp.weight",
        "ffn_up_shexp.weight",
        "ffn_down_shexp.weight",
    }:
        return "already_tp_divided", 0.5, "shared-expert intermediate lane slice"
    if suffix in {
        "ffn_gate_exps.weight",
        "ffn_up_exps.weight",
        "ffn_down_exps.weight",
    }:
        # The fraction is filled in from the balanced selected-expert count.
        return "routed_selected", -1.0, "balanced three-of-six routed selection"
    if suffix in {
        "attn_compressor_kv.weight",
        "attn_compressor_gate.weight",
        "indexer_compressor_kv.weight",
        "indexer_compressor_gate.weight",
        "indexer.attn_q_b.weight",
        "indexer.proj.weight",
    }:
        return "collective_required", 1.0, "replicated state/top-k builder"
    if suffix in {"attn_q_a.weight", "attn_kv.weight"}:
        return "collective_required", 1.0, "paired shared Q-lora/KV projection"
    if suffix.startswith("ffn_gate_inp"):
        return "replicated", 1.0, "router input"
    if suffix.startswith("hc_"):
        return "replicated", 1.0, "shared HC control path"
    if suffix in {"attn_norm.weight", "ffn_norm.weight"}:
        return "replicated", 1.0, "normalization"
    if any(part in suffix for part in ("compressor_ape", "compressor_norm")):
        return "replicated_small", 1.0, "compressor update metadata"
    if suffix in {
        "attn_sinks.weight",
        "attn_q_a_norm.weight",
        "attn_kv_a_norm.weight",
    }:
        return "replicated_small", 1.0, "attention scale/norm"
    if suffix == "ffn_gate_tid2eid.weight":
        return "token_lookup", 0.0, "one token-indexed row, not the full tensor"
    return "other", 1.0, "not assigned to a decode hot-path class"


def mib(n_bytes: float) -> str:
    return f"{n_bytes / (1 << 20):,.2f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--selected", type=int, default=6,
                        help="routed experts selected by the model (default: 6)")
    parser.add_argument("--selected-per-rank", type=float, default=3.0,
                        help="balanced routed experts executed per rank (default: 3)")
    args = parser.parse_args()
    if args.selected <= 0 or not 0 <= args.selected_per_rank <= args.selected:
        parser.error("selected counts must satisfy 0 <= per-rank <= selected")

    gguf = load_gguf_parser(args.repo.resolve())
    info = gguf.parse_gguf(args.model.resolve())
    categories: dict[str, float] = defaultdict(float)
    full_categories: dict[str, float] = defaultdict(float)
    reasons: dict[str, set[str]] = defaultdict(set)
    layer_ids: set[int] = set()
    unmatched: list[str] = []

    for tensor in info.tensors:
        match = LAYER_RE.match(tensor.name)
        if not match:
            continue
        layer_ids.add(int(match.group(1)))
        suffix = match.group(2)
        category, fraction, reason = classify(suffix)
        if category == "routed_selected":
            # Tensor dim[2] is the number of experts.  Reading one selected
            # expert touches exactly one contiguous expert slab.
            if len(tensor.dims) != 3 or tensor.dims[2] == 0:
                raise ValueError(f"unexpected routed tensor shape: {tensor.name} {tensor.dims}")
            fraction = args.selected_per_rank / tensor.dims[2]
        categories[category] += tensor.n_bytes * fraction
        full_categories[category] += tensor.n_bytes
        reasons[category].add(reason)
        if category == "other":
            unmatched.append(tensor.name)

    if not layer_ids:
        raise ValueError("no blk.N tensors found")
    expected = set(range(max(layer_ids) + 1))
    if layer_ids != expected:
        raise ValueError("layer IDs are not contiguous from zero")

    ordered = [
        "already_tp_divided",
        "routed_selected",
        "collective_required",
        "replicated",
        "replicated_small",
        "token_lookup",
        "other",
    ]
    print(f"model: {info.path}")
    print(f"layers: {len(layer_ids)}")
    print("scope: one decode token, one rank, TP=2; model bytes read (not resident VRAM)")
    print()
    print(f"{'class':24s} {'rank read MiB':>14s} {'full tensor MiB':>16s}  reason")
    print(f"{'-' * 24} {'-' * 14} {'-' * 16}  {'-' * 36}")
    for category in ordered:
        if category not in full_categories:
            continue
        reason = "; ".join(sorted(reasons[category]))
        print(f"{category:24s} {mib(categories[category]):>14s} "
              f"{mib(full_categories[category]):>16s}  {reason}")
    total = sum(categories.values())
    hot = total - categories.get("other", 0.0) - categories.get("token_lookup", 0.0)
    print()
    print(f"accounted hot-path model reads/rank/token: {mib(hot)} MiB")
    print(f"all classified layer tensor reads/rank/token: {mib(total)} MiB")
    print("routed-expert estimate assumes a balanced 3/3 split of six selected experts")
    print("collective_required is not free TP work: splitting it needs a new exchange/top-k design")
    if unmatched:
        print(f"other tensors ({len(unmatched)}):")
        for name in unmatched:
            print(f"  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
