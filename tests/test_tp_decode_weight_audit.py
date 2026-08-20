#!/usr/bin/env python3
"""Fail-closed TP=2 decode byte accounting (Codex option B, 2026-08-20)."""
from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
AUDIT = REPO / "scripts" / "tp_decode_weight_audit.py"
MODEL = Path(
    "/home/wkljohn/Desktop/cc/models/"
    "Huihui-DeepSeek-V4-Flash-0731-abliterated-GGUF/"
    "DeepSeek-V4-Flash-Q4_K-0731.gguf"
)
EXPECTED_COLLECTIVE_MIB = 1200.62


class TpDecodeWeightAudit(unittest.TestCase):
    def test_collective_required_reconciles(self) -> None:
        self.assertTrue(AUDIT.is_file(), "audit script missing")
        self.assertTrue(MODEL.is_file(), f"Q4_K GGUF missing: {MODEL}")
        proc = subprocess.run(
            [
                sys.executable,
                str(AUDIT),
                str(MODEL),
                "--repo",
                str(REPO),
                "--fail-closed",
                "--expect-collective-mib",
                f"{EXPECTED_COLLECTIVE_MIB:.2f}",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        sys.stdout.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        self.assertIn("collective_required families:", proc.stdout)
        self.assertIn("paired shared Q-lora/KV projection", proc.stdout)
        self.assertIn("replicated state/top-k builder", proc.stdout)
        self.assertIn("fail_closed: no unknown blk classifications", proc.stdout)
        self.assertIn(f"collective_required_mib={EXPECTED_COLLECTIVE_MIB:.2f}", proc.stdout)


if __name__ == "__main__":
    unittest.main()
