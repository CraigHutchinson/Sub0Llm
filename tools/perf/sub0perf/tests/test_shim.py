"""The command line that consumers and installed hooks call by path: tools/perf/bench_check.py."""

import subprocess
import sys
import unittest
from pathlib import Path

SHIM = Path(__file__).resolve().parents[2] / "bench_check.py"


class ShimTest(unittest.TestCase):
    def test_help_lists_every_command(self):
        result = subprocess.run([sys.executable, str(SHIM), "--help"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        for command in ("run", "check", "baseline", "auto", "accept", "gate", "install-hooks", "note-checkout"):
            self.assertIn(command, result.stdout)

    def test_missing_arguments_are_a_usage_error(self):
        result = subprocess.run([sys.executable, str(SHIM), "accept"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("accept needs", result.stderr)

    def test_installed_hooks_point_at_the_shim(self):
        from ..common import tooling_script
        self.assertEqual(tooling_script(), SHIM)


if __name__ == "__main__":
    unittest.main()
