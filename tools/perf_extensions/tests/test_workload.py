import copy
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from tools.perf_extensions import workload as w
from sub0perf.power import PowerState


class WorkloadTest(unittest.TestCase):
    def record(self):
        return {"schema": "sub0llm-workload-1", "correctness": {"exit_code": 0},
                "samples": [{"elapsed_s": 1., "peak_rss_mib": 100.}] * 3,
                **{key: "same" for key in ("scope", "identity", "inputs", "cache", "warmups", "arguments", "correctness_arguments", "correctness_binary_sha256", "adapter_sha256", "timer_poll_s", "environment", "affinity", "host", "system", "power")}}

    def test_memory_budget_independent_of_elapsed(self):
        record = self.record()
        self.assertEqual(w.check(record, None, 99)["memory_verdict"], "over-budget")
        record["samples"] = [{"elapsed_s": 1000., "peak_rss_mib": 1.}] * 3
        result = w.check(record, None, 99)
        self.assertEqual(result["memory_verdict"], "within-budget")
        self.assertEqual(result["elapsed_verdict"], "unjudged")

    def test_identity_and_method_mismatches_refused(self):
        record = self.record()
        for key in ("identity", "inputs", "cache", "arguments", "correctness_arguments", "correctness_binary_sha256", "adapter_sha256", "environment", "affinity", "host", "power"):
            other = copy.deepcopy(record)
            other[key] = "different"
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                w.check(record, other, 100)

    def test_failed_correctness_short_records_and_nonfinite_metrics_refused(self):
        record = self.record()
        for change in (lambda r: r["correctness"].update(exit_code=1),
                       lambda r: r.update(samples=[]),
                       lambda r: r["samples"][0].update(peak_rss_mib=float("nan"))):
            bad = copy.deepcopy(record)
            change(bad)
            with self.assertRaises(ValueError):
                w.check(bad, None, 100)
        with self.assertRaises(ValueError):
            w.check(record, None, float("nan"))

    @unittest.skipUnless(sys.platform == "linux", "Linux RSS adapter")
    def test_actual_child_memory_exit_status_and_timeout(self):
        sample = w.execute([sys.executable, "-c", "x=bytearray(32*1024*1024); print(len(x))"], ".", None, 5)
        self.assertGreater(sample["peak_rss_mib"], 32)
        with self.assertRaises(ValueError):
            w.execute([sys.executable, "-c", "raise SystemExit(7)"], ".", None, 5)
        with self.assertRaises(TimeoutError):
            w.execute([sys.executable, "-c", "import time; time.sleep(10)"], ".", None, .05)

    def measure_case(self, directory, correctness="pass"):
        root = Path(directory)
        subprocess.run(["git", "init", "-q", str(root)], check=True)
        subprocess.run(["git", "-C", str(root), "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "--allow-empty", "-qm", "fixture"], check=True)
        (root / "input").write_text("fixture")
        spec = {"command": [sys.executable, "-c", "pass"],
                "correctness": [sys.executable, "-c", correctness],
                "identity": {"prompt": "fixture", "workers": 1},
                "inputs": {"fixture": "input"}, "cache": "uncontrolled"}
        path = root / "spec.json"
        path.write_text(json.dumps(spec))
        return path

    @patch.object(w, "power_state", return_value=PowerState(state="ac"))
    @patch.object(w, "wait_for_quiet", return_value=(0., 0., True))
    def test_failed_oracle_never_runs_measured_command(self, quiet, power):
        with tempfile.TemporaryDirectory() as directory:
            spec = self.measure_case(directory)
            with patch.object(w, "execute", side_effect=ValueError("oracle failed")) as run:
                with self.assertRaisesRegex(ValueError, "oracle failed"):
                    w.measure(spec, directory)
                self.assertEqual(run.call_count, 1)

    @patch.object(w, "power_state", return_value=PowerState(state="ac"))
    @patch.object(w, "wait_for_quiet", return_value=(.5, 0., False))
    def test_contention_prevents_measurement(self, quiet, power):
        with tempfile.TemporaryDirectory() as directory:
            spec = self.measure_case(directory)
            with patch.object(w, "execute", return_value={}) as run:
                with self.assertRaisesRegex(ValueError, "busy"):
                    w.measure(spec, directory)
                self.assertEqual(run.call_count, 1)

    @unittest.skipUnless(sys.platform == "linux", "Linux RSS adapter")
    @patch.object(w, "power_state", return_value=PowerState(state="ac"))
    @patch.object(w, "wait_for_quiet", return_value=(0., 0., True))
    def test_real_measurement_has_binary_input_identity_and_three_samples(self, quiet, power):
        with tempfile.TemporaryDirectory() as directory:
            record = w.measure(self.measure_case(directory), directory)
            self.assertEqual(len(record["samples"]), 3)
            self.assertEqual(record["correctness"]["exit_code"], 0)
            self.assertEqual(len(record["binaries"]["measured"]), 64)
            self.assertEqual(record["inputs"]["fixture"], w.digest(Path(directory)/"input"))
            self.assertEqual(w.check(record, record, 1000)["compatibility"], "matched")

    @patch.object(w, "power_state", return_value=PowerState(state="ac"))
    @patch.object(w, "wait_for_quiet", return_value=(0., 0., True))
    def test_input_change_during_run_invalidates_record(self, quiet, power):
        with tempfile.TemporaryDirectory() as directory:
            spec = self.measure_case(directory)
            def mutate(*args):
                (Path(directory) / "input").write_text("changed")
                return {"elapsed_s": 1., "peak_rss_mib": 1.}
            with patch.object(w, "execute", side_effect=mutate):
                with self.assertRaisesRegex(ValueError, "changed"):
                    w.measure(spec, directory)

    @patch.object(w, "power_state", side_effect=[PowerState(state="ac"), PowerState(state="battery")])
    @patch.object(w, "wait_for_quiet", return_value=(0., 0., True))
    def test_power_change_prevents_measured_command(self, quiet, power):
        with tempfile.TemporaryDirectory() as directory:
            spec = self.measure_case(directory)
            with patch.object(w, "execute", return_value={}) as run:
                with self.assertRaisesRegex(ValueError, "power state changed"):
                    w.measure(spec, directory)
                self.assertEqual(run.call_count, 1)
