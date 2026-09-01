#!/usr/bin/env python3
"""Static guard for opt-in GLM5 tensor trace configuration."""

from pathlib import Path

root = Path(__file__).resolve().parents[1]
source = (root / "ds4.c").read_text(
    encoding="utf-8"
)
source += (root / "ds4_glm5_next_exec.c").read_text(encoding="utf-8")
for name in ("DS4_GLM5_TRACE_PREFIX", "DS4_GLM5_TRACE_LAYER", "DS4_GLM5_TRACE_TOKEN", "DS4_GLM5_TRACE_RANKED"):
    if name not in source:
        raise SystemExit(f"FAIL missing {name}")
if ".trace_prefix" not in source or ".trace_layer" not in source or ".trace_token" not in source:
    raise SystemExit("FAIL trace fields are not initialized")
print("PASS GLM5 trace configuration anchors exist")
