"""Consumer entry-point checks; behavioral coverage lives in the pinned shared package."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
SHIM = ROOT / "tools/perf_extensions/workload.py"
SHARED = ROOT / "tools/perf/bench_check.py"


class CompatibilityTest(unittest.TestCase):
    def run_cli(self, command):
        return subprocess.run([sys.executable, *map(str, command)], capture_output=True, text=True)

    def test_old_entry_point_exposes_shared_workload_actions(self):
        result = self.run_cli([SHIM, "--help"])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("measure", result.stdout)
        self.assertIn("check", result.stdout)

    def test_invalid_arguments_are_structured(self):
        result = self.run_cli([SHIM, "check"])
        self.assertEqual(result.returncode, 2)
        self.assertEqual(json.loads(result.stdout)["status"], "refused")
        self.assertNotIn("Traceback", result.stderr)

    def test_shared_and_old_entry_point_return_same_outcome(self):
        with tempfile.TemporaryDirectory() as directory:
            args = ["check", "--results", str(Path(directory) / "missing.json"), "--max-rss-mib", "256"]
            shared = self.run_cli([SHARED, "workload", *args])
            old = self.run_cli([SHIM, *args])
            self.assertEqual(old.returncode, shared.returncode)
            self.assertEqual(json.loads(old.stdout), json.loads(shared.stdout))
