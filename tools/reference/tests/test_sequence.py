import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "compare_sequence.py"


class SequenceGateTest(unittest.TestCase):
    def write(self, prefix, position, values):
        suffix = f".pos{position}" if position else ""
        Path(f"{prefix}{suffix}.logits").write_bytes(struct.pack(f"<{len(values)}f", *values))

    def run_gate(self, actual, reference, tolerance="0.001"):
        result = subprocess.run([sys.executable, str(SCRIPT), str(actual), str(reference), "--positions", "2",
                                 "--expected-values", "3", "--max-relative-l2", tolerance], capture_output=True, text=True)
        return result.returncode, json.loads(result.stdout)

    def test_later_position_failure_is_not_hidden_by_first(self):
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory)/"actual", Path(directory)/"reference"
            for position in range(2):
                self.write(a, position, [1., 2., 3.]);self.write(b, position, [1., 2., 3.])
            self.assertEqual(self.run_gate(a,b)[0], 0)
            self.write(a, 1, [1., 2., 4.])
            code, result = self.run_gate(a,b)
            self.assertEqual(code, 1)
            self.assertTrue(result["positions"][0]["passed"])
            self.assertFalse(result["positions"][1]["passed"])

    def test_missing_nonfinite_or_wrong_vocabulary_refuses(self):
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory)/"actual", Path(directory)/"reference"
            for position in range(2):
                self.write(a,position,[1.,2.,3.]);self.write(b,position,[1.,2.,3.])
            for values in ([float("nan"),2.,3.], [1.,2.]):
                self.write(a,1,values)
                self.assertEqual(self.run_gate(a,b)[0], 2)
            Path(f"{a}.pos1.logits").unlink()
            self.assertEqual(self.run_gate(a,b)[0], 2)

    def test_invalid_tolerance_refuses(self):
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory)/"actual", Path(directory)/"reference"
            self.write(a,0,[1.,2.,3.]);self.write(b,0,[1.,2.,3.])
            for tolerance in ("-1", "nan", "inf"):
                self.assertEqual(self.run_gate(a,b,tolerance)[0], 2)
