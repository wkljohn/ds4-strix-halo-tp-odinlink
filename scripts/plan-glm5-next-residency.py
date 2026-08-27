#!/usr/bin/env python3
"""Compute cache-free per-rank GLM-5.3 TP=2 residency from GGUF descriptors."""
from __future__ import annotations

import importlib.util
import os
import re
import sys
from pathlib import Path


def parser_for(repo: Path):
    p = repo / "gguf-tools/mixed/splice_mixed_expert_layers_gguf.py"
    spec = importlib.util.spec_from_file_location("ds4_gguf_parser", p)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load parser")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


EXPERT = re.compile(r"^blk\.\d+\.ffn_(?:gate|up|down)_exps\.weight$")


def main(argv):
    if len(argv) != 3:
        print(f"usage: {argv[0]} REPO MODEL.gguf", file=sys.stderr)
        return 2
    info = parser_for(Path(argv[1])).parse_gguf(Path(argv[2]))
    total = sum(t.n_bytes for t in info.tensors)
    expert = sum(t.n_bytes for t in info.tensors if EXPERT.match(t.name))
    replicated = total - expert
    # Experts are split on the intermediate axis; every rank receives one
    # 1024-row/input half, not a second full copy.
    per_rank = replicated + expert // 2
    scratch = 512 * 1024 * 1024
    # The bring-up contract deliberately budgets against 112 GiB even when a
    # host exposes a larger aperture.  This leaves host/driver headroom and
    # prevents a 124 GiB lab boot from silently becoming a runtime dependency.
    gtt_target_gib = int(os.environ.get("DS4_GLM5_GTT_TARGET_GIB", "112"))
    if gtt_target_gib < 1:
        raise ValueError("DS4_GLM5_GTT_TARGET_GIB must be positive")
    gtt_target = gtt_target_gib * 1024**3
    if expert % 2:
        raise ValueError("expert bytes are not divisible by TP=2")
    print("PASS GLM5-next residency plan")
    print(f"gguf_weight_bytes={total} ({total/1024**3:.3f} GiB)")
    print(f"replicated_bytes={replicated} ({replicated/1024**3:.3f} GiB)")
    print(f"expert_bytes={expert} ({expert/1024**3:.3f} GiB)")
    print(f"tp2_weight_bytes_per_rank={per_rank} ({per_rank/1024**3:.3f} GiB)")
    print(f"scratch_budget_bytes={scratch} ({scratch/1024**3:.3f} GiB)")
    print(f"required_resident_bytes={per_rank + scratch} "
          f"({(per_rank + scratch)/1024**3:.3f} GiB)")
    print(f"target_gtt_bytes={gtt_target} ({gtt_target/1024**3:.3f} GiB)")
    headroom = gtt_target - per_rank - scratch
    print(f"headroom_after_scratch_bytes={headroom} ({headroom/1024**3:.3f} GiB)")
    if headroom <= 0:
        raise ValueError("target GTT cannot fit planned cache-free resident layout")
    print("persistent_weight_cache=disabled")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
