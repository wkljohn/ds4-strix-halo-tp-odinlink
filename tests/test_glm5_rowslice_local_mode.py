#!/usr/bin/env python3
"""Regression guard for the local-only rowslice diagnostic."""

from pathlib import Path


source = (Path(__file__).resolve().parents[1] / "ds4_glm5_next_exec.c").read_text(
    encoding="utf-8"
)
if "DS4_GLM5_KDA_OUTPUT_ROWSLICE_LOCAL" not in source:
    raise SystemExit("FAIL rowslice local diagnostic switch is missing")
if "tp_exchange_aux_bytes(ctx, il, row_bytes)" not in source:
    raise SystemExit("FAIL rowslice transport exchange anchor is missing")
if "w->down, ctx->model_map, ctx->model_size" not in source:
    raise SystemExit("FAIL local rowslice must use device-resident scratch")
print("PASS rowslice local diagnostic switch and transport anchor exist")
