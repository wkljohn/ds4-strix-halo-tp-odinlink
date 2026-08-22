#!/usr/bin/env python3
"""Render a ds4-bench CSV as a compact Markdown performance table."""

from __future__ import annotations

import csv
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: render-tp-bench-table.py RESULT.csv", file=sys.stderr)
        return 2
    path = Path(sys.argv[1])
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        print(f"error: no benchmark rows in {path}", file=sys.stderr)
        return 1
    print("| Workload | Context | Prefill | Decode | Steady decode | Fingerprint |")
    print("|---|---:|---:|---:|---:|---|")
    for row in rows:
        ctx = int(row["ctx_tokens"])
        print(
            f"| Cross-discipline v1 | {ctx:,} | {float(row['prefill_tps']):.2f} t/s | "
            f"{float(row['gen_tps']):.2f} t/s | {float(row['gen_steady_tps']):.2f} t/s | "
            f"`{row['gen_token_fnv64']}` |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
