import copy
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from .. import workload as w
from ..power import PowerState


class WorkloadTest(unittest.TestCase):
    def record(self):
        return {"schema": "sub0perf-workload-1", "correctness": {"exit_code": 0},
                "scope": "whole-process invocation", "elapsed_verdict": "unjudged",
                "identity": {"fixture": "fixed"}, "inputs": {"fixture": "a" * 64},
                "cache": "uncontrolled", "warmups": 0, "arguments": [], "correctness_arguments": [],
                "correctness_binary_sha256": "b" * 64, "adapter_sha256": "c" * 64,
                "timer_poll_s": .01, "environment": {}, "affinity": [0], "host": "host", "system": "system",
                "power": {"state": "ac", "saver": False},
                "samples": [{"elapsed_s": 1., "peak_rss_mib": 100.}] * 3}

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
            if key in ("identity", "inputs"):
                other[key] = {"different": "d" * 64}
            elif key in ("correctness_binary_sha256", "adapter_sha256"):
                other[key] = "d" * 64
            elif key in ("arguments", "correctness_arguments"):
                other[key] = ["different"]
            elif key == "environment":
                other[key] = {"OMP_NUM_THREADS": "2"}
            elif key == "affinity":
                other[key] = [1]
            elif key == "power":
                other[key] = {"state": "unknown", "saver": False}
            elif key == "cache":
                other[key], other["warmups"] = "warm", 1
            else:
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
        subprocess.run(["git", "-C", str(root), "-c", "core.hooksPath=/dev/null", "-c", "commit.gpgsign=false", "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "--allow-empty", "-qm", "fixture"], check=True)
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

    def test_incomplete_and_wrongly_typed_evidence_is_refused(self):
        for mutate in (lambda r: r.pop("inputs"),
                       lambda r: r["correctness"].update(exit_code=False),
                       lambda r: r.update(inputs={"fixture": "not-a-hash"}),
                       lambda r: r.update(affinity=[True]),
                       lambda r: r.update(environment=[]),
                       lambda r: r.update(power={"state": "battery"}),
                       lambda r: r.update(identity={"seed": float("nan")})):
            bad = self.record()
            mutate(bad)
            with self.assertRaises(ValueError):
                w.check(bad, None, 100)

    def cli(self, *arguments):
        shim = Path(__file__).resolve().parents[2] / "bench_check.py"
        return subprocess.run([sys.executable, str(shim), "workload", *arguments], capture_output=True, text=True)

    def test_cli_usage_error_is_one_structured_refusal(self):
        completed = self.cli("check")
        self.assertEqual(completed.returncode, 2)
        result = json.loads(completed.stdout)
        self.assertEqual(result["schema"], "sub0perf-workload-outcome-1")
        self.assertEqual(result["status"], "refused")
        self.assertIn("--results", result["error"])

    def test_cli_budget_status_exit_codes_and_unchanged_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "record.json"
            raw = json.dumps(self.record())
            path.write_text(raw)
            for budget, code, status in ((200, 0, "within-budget"), (99, 1, "over-budget")):
                completed = self.cli("check", "--results", str(path), "--max-rss-mib", str(budget))
                self.assertEqual(completed.returncode, code, completed.stderr)
                self.assertEqual(json.loads(completed.stdout)["status"], status)
                self.assertEqual(json.loads(completed.stdout)["elapsed_verdict"], "unjudged")
            self.assertEqual(path.read_text(), raw)

    @unittest.skipUnless(sys.platform == "linux", "Linux RSS adapter")
    @patch.object(w, "power_state", return_value=PowerState(state="ac"))
    @patch.object(w, "wait_for_quiet", return_value=(0., 0., True))
    def test_hashing_holds_shared_lock_and_cli_records_one_json_outcome(self, quiet, power):
        import contextlib
        import io
        with tempfile.TemporaryDirectory() as directory:
            spec = self.measure_case(directory)
            real_digest = w.digest
            def locked_digest(path):
                lock = Path(directory) / ".git/sub0perf/lock"
                if Path(path).name in ("input", "spec.json"):
                    self.assertTrue(lock.is_file())
                return real_digest(path)
            output = Path(directory) / "evidence.json"
            stream = io.StringIO()
            with patch.object(w, "digest", side_effect=locked_digest), contextlib.redirect_stdout(stream):
                self.assertEqual(w.main(["measure", "--spec", str(spec), "--repo", directory, "--out", str(output)]), 0)
            self.assertEqual(json.loads(stream.getvalue())["status"], "recorded")
            before = output.read_bytes()
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as error:
                w.main(["measure", "--spec", str(spec), "--repo", directory, "--out", str(output)])
            self.assertEqual(error.exception.code, 2)
            self.assertEqual(output.read_bytes(), before)
