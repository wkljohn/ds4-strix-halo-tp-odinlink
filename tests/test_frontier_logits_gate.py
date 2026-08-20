#!/usr/bin/env python3
import json
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / "scripts" / "compare-frontier-logits.py"


class FrontierLogitsGateTest(unittest.TestCase):
    def run_gate(self, reference, candidate):
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for name, logits in (("reference", reference), ("candidate", candidate)):
                path = Path(tmp) / f"{name}.json"
                path.write_text(json.dumps({
                    "vocab": len(logits),
                    "frontier_tokens": 2048,
                    "prompt_tokens": 2048,
                    "quant_bits": 4,
                    "logits": logits,
                }), encoding="utf-8")
                paths.append(path)
            return subprocess.run(
                [str(GATE), str(paths[0]), str(paths[1])],
                text=True, capture_output=True, check=False)

    def test_exact_and_constant_shift_pass(self):
        reference = [float(i) / 8.0 for i in range(32)]
        exact = self.run_gate(reference, reference)
        self.assertEqual(exact.returncode, 0, exact.stderr)
        shifted = self.run_gate(reference, [value + 3.306702 for value in reference])
        self.assertEqual(shifted.returncode, 0, shifted.stderr)

    def test_top_five_reorder_fails(self):
        reference = [float(i) for i in range(32)]
        candidate = reference.copy()
        candidate[-2], candidate[-3] = candidate[-3], candidate[-2]
        result = self.run_gate(reference, candidate)
        self.assertNotEqual(result.returncode, 0)

    def test_distribution_drift_fails(self):
        reference = [0.0] * 32
        candidate = reference.copy()
        candidate[0] = 1.0
        result = self.run_gate(reference, candidate)
        self.assertNotEqual(result.returncode, 0)

    def test_nonfinite_dump_fails(self):
        result = self.run_gate([0.0] * 32, [None] + [0.0] * 31)
        self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
