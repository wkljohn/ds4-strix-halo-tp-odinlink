#!/usr/bin/env python3
import json
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / "scripts" / "compare-frontier-logits.py"


class FrontierLogitsGateTest(unittest.TestCase):
    def run_gate(self, reference, candidate, *extra):
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
                [str(GATE), str(paths[0]), str(paths[1]), *extra],
                text=True, capture_output=True, check=False)

    def test_exact_and_constant_shift_pass(self):
        reference = [float(i) / 8.0 for i in range(32)]
        exact = self.run_gate(reference, reference)
        self.assertEqual(exact.returncode, 0, exact.stderr)
        shifted = self.run_gate(reference, [value + 3.306702 for value in reference])
        self.assertEqual(shifted.returncode, 0, shifted.stderr)

    def test_nearby_top_five_reorder_is_recorded_but_passes(self):
        reference = [0.0] * 32
        reference[-5:] = [2.0, 3.0, 3.00002, 4.0, 5.0]
        candidate = reference.copy()
        candidate[-3], candidate[-4] = candidate[-4], candidate[-3]
        result = self.run_gate(reference, candidate)
        self.assertEqual(result.returncode, 0, result.stderr)
        value = json.loads(result.stdout)
        self.assertFalse(value["top5_rank_equal"])
        self.assertEqual(value["top5_overlap"], 5)

    def test_near_tie_argmax_change_can_pass_a_versioned_bound(self):
        reference = [0.0] * 32
        reference[-2:] = [1.0, 1.05]
        candidate = reference.copy()
        candidate[-2:] = [1.06, 1.04]
        result = self.run_gate(reference, candidate, "--e-bound", "0.1",
                               "--max-tvd", "0.1", "--max-kl", "0.1")
        self.assertEqual(result.returncode, 0, result.stderr)
        value = json.loads(result.stdout)
        self.assertFalse(value["argmax_equal"])
        self.assertFalse(value["far_margin_inversion"])

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
