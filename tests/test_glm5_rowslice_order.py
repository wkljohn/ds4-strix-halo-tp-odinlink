#!/usr/bin/env python3
"""Regression guard: rowslice must compose gated heads before TP exchange."""

from pathlib import Path


SOURCE = Path(__file__).resolve().parents[1] / "ds4_glm5_next_exec.c"
text = SOURCE.read_text(encoding="utf-8")
start = text.index("const int output_rowslice")
exchange = text.index("tp_exchange_rows(ctx", start)
compose = text.index("ds4_glm5_kda_compose_head_halves", start)

if exchange > compose:
    raise SystemExit(
        "FAIL GLM5 rowslice composes gated-head halves before peer exchange"
    )
print("PASS GLM5 rowslice composes gated-head halves after peer exchange")
