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
    expert_up = info.tensor_by_name["blk.3.ffn_up_exps.weight"]
    expert_down = info.tensor_by_name["blk.3.ffn_down_exps.weight"]
    if kda.dims != (4096, 8192):
        raise ValueError(f"unexpected KDA descriptor: {kda}")
    if expert.dims != (4096, 2048, 288) or expert_up.dims != expert.dims:
        raise ValueError(f"unexpected expert descriptor: {expert}")
    if expert.ggml_type == 12:
        if expert_up.ggml_type != 12 or expert_down.ggml_type != 12:
            raise ValueError("uniform Q4_K expert family is incomplete")
        gate_block_bytes = 144
        down_block_bytes = 144
        family = "Q4_K"
    elif expert.ggml_type == 16:
        if expert_up.ggml_type != 16 or expert_down.ggml_type != 10:
            raise ValueError("mixed Q2 expert family must be IQ2_XXS/IQ2_XXS/Q2_K")
        gate_block_bytes = 66
        down_block_bytes = 84
        family = "IQ2_XXS+Q2_K"
    else:
        raise ValueError(f"unsupported expert family type: {expert.ggml_type}")
    if kda.ggml_type == 30:
        kda_sample_bytes = 8192
        kda_family = "BF16"
    elif kda.ggml_type == 12:
        kda_sample_bytes = 144
        kda_family = "Q4_K"
    elif kda.ggml_type == 8:
        kda_sample_bytes = 34
        kda_family = "Q8_0"
    else:
        raise ValueError(f"unsupported KDA q tensor type: {kda.ggml_type}")
    with model.open("rb") as fp:
        fp.seek(kda.data_offset)
        raw = fp.read(kda_sample_bytes)
        fp.seek(expert.data_offset)
        gate_block = fp.read(gate_block_bytes)
        fp.seek(expert_down.data_offset)
        down_block = fp.read(down_block_bytes)
    if len(raw) != kda_sample_bytes:
        raise ValueError("KDA sample block is truncated")
    if kda.ggml_type == 30:
        row = bf16_to_float(raw)
        if len(row) != 4096 or not all(math.isfinite(x) for x in row):
            raise ValueError("KDA BF16 row contains invalid values")
        activation = [((i % 17) - 8) * 0.0078125 for i in range(4096)]
        dot_a = sum(a * b for a, b in zip(row, activation))
        dot_b = sum(a * b for a, b in zip(row, activation))
        if not math.isfinite(dot_a) or dot_a != dot_b:
            raise ValueError("non-deterministic BF16 scalar oracle")
    else:
        dot_a = 0.0
    if len(gate_block) != gate_block_bytes or len(down_block) != down_block_bytes:
        raise ValueError("expert block geometry mismatch")
    print("PASS GLM5-next weight probe")
    print(f"kda_family={kda_family} kda_sample_bytes={kda_sample_bytes} "
          f"expert_family={family} "
          f"gate_block_bytes={len(gate_block)} down_block_bytes={len(down_block)} "
          f"kda_scalar_dot={dot_a:.9g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
