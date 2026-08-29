#!/usr/bin/env python3
"""Same-GGUF routed-expert layout oracle for the GLM-5.3 staged executor.

This is deliberately a payload-free gate.  It proves the byte geometry needed
for a 1024/1024 TP row split, including the mixed IQ2_XXS gate/up and Q2_K
down layout, before any GPU kernel is allowed to reinterpret a Q4 descriptor.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys


def load_checker():
    path = pathlib.Path(__file__).parents[1] / "scripts/check-glm5-next-gguf.py"
    spec = importlib.util.spec_from_file_location("glm5_checker", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load GGUF checker")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fail(message: str) -> None:
    raise AssertionError(message)


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} MODEL.gguf", file=sys.stderr)
        return 2
    checker = load_checker()
    meta, tensors = checker.read(pathlib.Path(sys.argv[1]))
    if meta.get("general.architecture") != "glm5-next":
        fail("not a glm5-next GGUF")

    gate = tensors["blk.3.ffn_gate_exps.weight"]
    up = tensors["blk.3.ffn_up_exps.weight"]
    down = tensors["blk.3.ffn_down_exps.weight"]
    if gate[0] != (4096, 2048, 288) or up[0] != gate[0]:
        fail("gate/up dimensions are not 4096x2048x288")
    if down[0] != (2048, 4096, 288):
        fail("down dimensions are not 2048x4096x288")
    if gate[1] == 16 and up[1] == 16 and down[1] == 10:
        gate_block, down_block, family = 66, 84, "IQ2_XXS+Q2_K"
    elif gate[1] == 12 and up[1] == 12 and down[1] == 12:
        gate_block, down_block, family = 144, 144, "Q4_K"
    else:
        fail(f"unsupported routed family types={gate[1]}/{up[1]}/{down[1]}")

    gate_row = (4096 // 256) * gate_block
    down_row = (2048 // 256) * down_block
    gate_expert = 2048 * gate_row
    down_expert = 4096 * down_row
    if gate_expert <= 0 or down_expert <= 0:
        fail("invalid expert byte geometry")

    # Both ranks consume exactly one intermediate half.  Gate/up are row
    # contiguous; down is column-strided inside each output row.
    half_rows = 1024
    half_gate = half_rows * gate_row
    half_down_columns = (half_rows // 256) * down_block
    if half_gate != gate_expert // 2:
        fail("gate/up half is not exactly 1024 rows")
    if half_down_columns * 2 != down_row:
        fail("down row does not split into two 1024-column byte spans")
    for rank in (0, 1):
        row_base = rank * half_rows
        gate_slice = gate[2] + row_base * gate_row
        up_slice = up[2] + row_base * gate_row
        down_column_base = rank * half_down_columns
        if gate_slice < gate[2] or up_slice < up[2]:
            fail("gate/up slice underflows tensor")
        if down_column_base not in (0, half_down_columns):
            fail("invalid down column split")
    print(
        "PASS GLM5 routed layout family=%s gate_row_bytes=%d down_row_bytes=%d "
        "gate_expert_bytes=%d down_expert_bytes=%d rank_half_gate=%d "
        "rank_half_down_columns=%d" %
        (family, gate_row, down_row, gate_expert, down_expert, half_gate,
         half_down_columns)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
