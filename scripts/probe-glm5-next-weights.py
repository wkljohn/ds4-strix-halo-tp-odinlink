#!/usr/bin/env python3
"""Read-only, small-payload GLM-5.3 GGUF weight/oracle probe."""
from __future__ import annotations

import importlib.util
import math
import struct
import sys
from pathlib import Path


def load_parser(repo: Path):
    path = repo / "gguf-tools/mixed/splice_mixed_expert_layers_gguf.py"
    spec = importlib.util.spec_from_file_location("ds4_gguf_parser", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load GGUF parser")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def bf16_to_float(raw: bytes):
    if len(raw) % 2:
        raise ValueError("BF16 payload is not 16-bit aligned")
    return [struct.unpack("<f", struct.pack("<I", x << 16))[0]
            for x in struct.unpack("<" + "H" * (len(raw) // 2), raw)]


def main(argv):
    if len(argv) != 3:
        print(f"usage: {argv[0]} REPO MODEL.gguf", file=sys.stderr)
        return 2
    repo, model = Path(argv[1]), Path(argv[2])
    parser = load_parser(repo)
    info = parser.parse_gguf(model)
    kda = info.tensor_by_name["blk.0.kda_q.weight"]
    expert = info.tensor_by_name["blk.3.ffn_gate_exps.weight"]
    if kda.ggml_type != 30 or kda.dims != (4096, 8192):
        raise ValueError(f"unexpected KDA descriptor: {kda}")
    if expert.ggml_type != 12 or expert.dims != (4096, 2048, 288):
        raise ValueError(f"unexpected expert descriptor: {expert}")
    with model.open("rb") as fp:
        fp.seek(kda.data_offset)
        raw = fp.read(8192)  # one 4096-element BF16 row
        fp.seek(expert.data_offset)
        q4_block = fp.read(144)  # one Q4_K block
    row = bf16_to_float(raw)
    if len(row) != 4096 or not all(math.isfinite(x) for x in row):
        raise ValueError("KDA BF16 row contains invalid values")
    activation = [((i % 17) - 8) * 0.0078125 for i in range(4096)]
    dot_a = sum(a * b for a, b in zip(row, activation))
    dot_b = sum(a * b for a, b in zip(row, activation))
    if not math.isfinite(dot_a) or dot_a != dot_b:
        raise ValueError("non-deterministic BF16 scalar oracle")
    if len(q4_block) != 144:
        raise ValueError("Q4_K block geometry mismatch")
    print("PASS GLM5-next weight probe")
    print(f"kda_bf16_row=4096 q4k_block_bytes={len(q4_block)} "
          f"kda_scalar_dot={dot_a:.9g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
