"""Tests for the speed check: a mains machine pinned slow is deferred before any build, like a machine on battery."""

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from .. import accept, cli, common, deferral, gate, gitstate, power, report, speed
from ..baseline import make_record
from ..judge import evaluate
from .helpers import RepoTestCase, git_run, record
from .test_power import captured, no_build


class LimitTest(unittest.TestCase):
    def test_default_environment_and_config(self):
        self.assertEqual(speed.limit({}), 1.4)
        self.assertEqual(speed.limit({}, {"throttle_ratio": 1.2}), 1.2)
        self.assertEqual(speed.limit({speed.LIMIT_VAR: "1.7"}, {"throttle_ratio": 1.2}), 1.7)
        self.assertEqual(speed.limit({speed.LIMIT_VAR: "nonsense"}), 1.4)
        self.assertEqual(speed.limit({speed.LIMIT_VAR: "0.5"}), 1.4)  # a limit under 1 would defer everything


class MeasureTest(unittest.TestCase):
    def test_a_normal_reading_is_taken_once(self):
        reads = []
        ratio, now = speed.measure_ratio("exe", 100.0, 1.4, lambda exe: reads.append(1) or 105.0)
        self.assertEqual(len(reads), 1)
        self.assertAlmostEqual(ratio, 1.05)

    def test_a_cold_clock_reading_is_repeated_until_two_agree(self):
        readings = iter([190.0, 192.0, 191.0, 101.0, 100.5, 100.5])
        # the first (cold, 1.9x) reading is over the limit; the steady re-read starts from the next ones and ends at 190/190: still slow
        ratio, _ = speed.measure_ratio("exe", 100.0, 1.4, lambda exe: next(readings))
        self.assertGreater(ratio, 1.8)
        warming = iter([190.0, 150.0, 105.0, 104.0])
        ratio, _ = speed.measure_ratio("exe", 100.0, 1.4, lambda exe: next(warming))
        self.assertAlmostEqual(ratio, 1.04)  # the clock came up: not throttled

    def test_a_pinned_slow_machine_reads_slow(self):
        ratio, _ = speed.measure_ratio("exe", 100.0, 1.4, lambda exe: 188.0)
        self.assertGreater(ratio, 1.4)


class CheckTargetsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.exe = self.dir / "bench.exe"
        self.exe.write_text("x", encoding="utf-8")
        self.baseline = self.dir / "base.json"
        self.baseline.write_text(json.dumps({"calibration_ns": 4800.0}), encoding="utf-8")

    def check(self, read, environ=None, **kw):
        return speed.check_targets([("t", str(self.exe), str(self.baseline))], environ={speed.WARM_VAR: "0", **(environ or {})},
                                   read=read, **kw)

    def test_compares_with_the_baselines_calibration_and_says_so(self):
        found = self.check(lambda exe: 4900.0)
        self.assertAlmostEqual(found.ratio, 4900 / 4800)
        self.assertIn("speed check t: calibration 4,900 ns against the baseline's 4,800 ns = 1.02x", found.notes[0])

    def test_no_baseline_calibration_proceeds_and_says_so(self):
        self.baseline.write_text("{}", encoding="utf-8")
        found = self.check(mock.Mock(side_effect=AssertionError("nothing to compare with")))
        self.assertIsNone(found.ratio)
        self.assertIn("no calibration (a first run); proceeding", found.notes[0])

    def test_no_baseline_file_or_no_built_benchmark_proceeds(self):
        self.baseline.unlink()
        self.assertIsNone(self.check(lambda exe: 1.0).ratio)
        self.baseline.write_text(json.dumps({"calibration_ns": 4800.0}), encoding="utf-8")
        self.exe.unlink()
        found = self.check(lambda exe: 1.0)
        self.assertIsNone(found.ratio)
        self.assertIn("no built benchmark", found.notes[0])

    def test_a_failing_calibration_run_is_a_note_not_a_crash(self):
        def broken(exe):
            raise OSError("cannot run")
        found = self.check(broken)
        self.assertIsNone(found.ratio)
        self.assertIn("cannot run", found.notes[0])

    def test_the_slowest_target_decides(self):
        other = self.dir / "other.json"
        other.write_text(json.dumps({"calibration_ns": 1000.0}), encoding="utf-8")
        found = speed.check_targets([("a", str(self.exe), str(self.baseline)), ("b", str(self.exe), str(other))],
                                    environ={speed.WARM_VAR: "0"}, read=lambda exe: 2000.0)
        self.assertAlmostEqual(found.ratio, 2.0)

    def sustained(self, ns, calls=None):
        """A fake sustained reader (exe, warm) -> ns; records the warm seconds asked for."""
        return lambda exe, warm: (calls.append(warm) if calls is not None else None) or ns

    def test_a_sustained_throttle_with_a_fine_burst_is_found(self):
        """The 2026-11-04 laptop: the first 3 s run at boost (0.97x), then every core falls to 0.42x (2.4x the time)."""
        self.baseline.write_text(json.dumps({"calibration_ns": 4800.0, "calibration_sustained_ns": 4850.0}), encoding="utf-8")
        calls = []
        found = self.check(lambda exe: 4650.0, {speed.WARM_VAR: "4.5"}, read_sustained=self.sustained(11600.0, calls))
        self.assertEqual(calls, [4.5])
        self.assertAlmostEqual(found.ratio, 11600 / 4850)
        self.assertTrue(found.sustained)
        self.assertIn("sustained calibration 11,600 ns over 4.5 s of load, against the baseline's 4,850 ns = 2.39x", found.notes[1])

    def test_a_healthy_machine_passes_both_readings(self):
        found = self.check(lambda exe: 4700.0, {speed.WARM_VAR: "4.5"}, read_sustained=self.sustained(4700.0))
        self.assertLess(found.ratio, 1.4)
        self.assertEqual(len(found.notes), 2)

    def test_an_old_baseline_compares_the_sustained_reading_with_this_runs_burst(self):
        found = self.check(lambda exe: 4700.0, {speed.WARM_VAR: "4.5"}, read_sustained=self.sustained(11600.0))
        self.assertAlmostEqual(found.ratio, 11600 / 4700)  # the drop within the run
        self.assertTrue(found.sustained)
        self.assertIn("the baseline has no sustained calibration", found.notes[1])

    def test_a_slow_burst_skips_the_warm_load(self):
        found = self.check(lambda exe: 11000.0, {speed.WARM_VAR: "4.5"},
                           read_sustained=mock.Mock(side_effect=AssertionError("burst already slow")))
        self.assertGreater(found.ratio, 1.4)
        self.assertFalse(found.sustained)
        self.assertEqual(len(found.notes), 1)

    def unavailable(self, message):
        def read(exe, warm):
            raise speed.SustainedUnavailable(message)
        return read

    def test_an_executable_without_soak_support_falls_back_to_the_burst_reading_and_says_so(self):
        found = self.check(lambda exe: 4700.0, {speed.WARM_VAR: "4.5"},
                           read_sustained=self.unavailable("the executable wrote no output for `--calibration --soak` (it predates `--soak`); " + speed.REBUILD_ADVICE))
        self.assertAlmostEqual(found.ratio, 4700 / 4800)
        self.assertFalse(found.sustained)
        self.assertEqual(len(found.notes), 2)
        self.assertIn("no sustained reading: the executable wrote no output", found.notes[1])
        self.assertIn("rebuild the benchmark executable", found.notes[1])
        self.assertIn("Only the burst reading is used", found.notes[1])

    def test_a_failing_sustained_run_is_a_note_not_a_crash(self):
        def broken(exe, warm):
            raise OSError("cannot run")
        found = self.check(lambda exe: 4700.0, {speed.WARM_VAR: "4.5"}, read_sustained=broken)
        self.assertIn("cannot run", found.notes[-1])
        self.assertIn("rebuild the benchmark executable", found.notes[-1])
        self.assertAlmostEqual(found.ratio, 4700 / 4800)  # the burst reading still counts

    def test_warm_seconds_from_environment_config_default(self):
        self.assertEqual(speed.warm_seconds({}), 4.5)
        self.assertEqual(speed.warm_seconds({}, {"warm_seconds": 6}), 6)
        self.assertEqual(speed.warm_seconds({speed.WARM_VAR: "0"}, {"warm_seconds": 6}), 0)
        self.assertEqual(speed.warm_seconds({speed.WARM_VAR: "junk"}), 4.5)

    def test_the_sustained_fake_alone_means_burst_fine_sustained_throttled(self):
        found = speed.check_targets([], environ={speed.FAKE_SUSTAINED_VAR: "2.4"}, read=mock.Mock(side_effect=AssertionError("faked")))
        self.assertEqual(found.ratio, 2.4)
        self.assertTrue(found.sustained)
        self.assertIn(speed.FAKE_SUSTAINED_VAR, found.notes[0])

    def test_recording_a_sustained_calibration_for_a_baseline(self):
        self.assertEqual(speed.record_sustained("exe", {}, None, self.sustained(4700.1234)), 4700.123)
        self.assertIsNone(speed.record_sustained("exe", {}, None, self.unavailable("x")))
        self.assertIsNone(speed.record_sustained("exe", {speed.WARM_VAR: "0"}))
        self.assertIsNone(speed.record_sustained("exe", {speed.FAKE_VAR: "1.0"}))

    def test_the_fake_replaces_the_measurement_and_is_warned_about(self):
        found = speed.check_targets([("t", "missing", "missing")], environ={speed.FAKE_VAR: "1.88"},
                                    read=mock.Mock(side_effect=AssertionError("faked")))
        self.assertEqual(found.ratio, 1.88)
        self.assertIn("machine speed is being faked", found.notes[0])
        self.assertIsNone(speed.check_targets([], environ={speed.FAKE_VAR: "x"}).ratio)


class StateTest(unittest.TestCase):
    def test_over_the_limit_is_throttled_and_untrusted(self):
        mains = power.PowerState(power.AC, source="fake")
        slow = power.with_speed(mains, 1.88, 1.4)
        self.assertTrue(slow.throttled and slow.untrusted)
        self.assertIn("machine throttled (calibration 1.88x slower", slow.reason())
        self.assertEqual(slow.as_dict(), {"state": "ac", "saver": False, "speed_ratio": 1.88, "throttled": True})
        ok = power.with_speed(mains, 1.02, 1.4)
        self.assertFalse(ok.untrusted)
        self.assertEqual(ok.as_dict()["speed_ratio"], 1.02)  # recorded even when fine
        self.assertIs(power.with_speed(mains, None, 1.4), mains)

    def test_the_deferral_names_the_ratio_the_causes_and_the_override(self):
        slow = power.with_speed(power.PowerState(power.AC), 1.88, 1.4)
        text = power.deferral_text(slow, False)
        for part in ("1.88x slower", "vendor power mode", "charging power cap", "thermal", "SUB0MATH_PERF_ON_BATTERY=1",
                     "performance not measured"):
            self.assertIn(part, text)
        self.assertIn("refused", power.deferral_text(slow, True))
        self.assertNotIn("throttled", power.deferral_text(power.PowerState(power.BATTERY), False))

    def test_a_throttled_record_differs_and_cannot_become_a_baseline(self):
        info = {"state": "ac", "saver": False, "speed_ratio": 1.9, "throttled": True}
        self.assertNotEqual(power.key(info), power.key(None))
        self.assertEqual(power.key({"state": "ac", "speed_ratio": 1.02}), power.key(None))  # a measured-fine ratio changes nothing
        self.assertIn("throttled", power.label(info))
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(accept.power_refusal({"power": info}, "x", force=False))
            self.assertFalse(accept.power_refusal({"power": info}, "x", force=True))
            self.assertTrue(accept.power_refusal({"power": info, "power_override": True}, "x", force=True))
            self.assertFalse(accept.power_refusal({"power": {"state": "ac", "speed_ratio": 1.02}}, "x", force=False))

    def test_the_override_text_names_the_throttle(self):
        throttled = power.with_speed(power.PowerState(power.AC), 1.9, 1.4)
        self.assertIn("throttled machine", power.override_text(throttled))
        self.assertIn("throttled machine", power.override_text(throttled.as_dict()))
        self.assertIn("on battery", power.override_text(power.PowerState(power.BATTERY)))
        self.assertIn("on battery", power.override_text(power.PowerState(power.BATTERY, throttled=True)))

    def test_a_throttled_override_report_is_flagged(self):
        rec = record({"A": 100, "B": 100, "C": 100})
        rec["power"] = {"state": "ac", "saver": False, "speed_ratio": 1.9, "throttled": True}
        rec["power_override"] = True
        rows, pairs, env = evaluate(rec, {"benchmarks": {}, "calibration_ns": 1.0}, 0.10, 0.05, 0.05, False)
        text = report.render(rows, pairs, env, "b", 0.1, 0.05)
        self.assertIn("throttled machine", text)


class ModeTest(unittest.TestCase):
    """The Windows power-mode overlay is reported by name, and named with its fix when the machine is throttled in the efficiency mode."""

    def test_the_overlay_guid_is_named(self):
        self.assertEqual(power._OVERLAYS["961cc777-2547-4f9d-8174-7d86181b8a7a"], "Best power efficiency")
        self.assertEqual(power._OVERLAYS["00000000-0000-0000-0000-000000000000"], "Balanced")

    def test_describe_and_the_record_carry_both_modes(self):
        state = power.PowerState(power.AC, scheme="Balanced", source="windows", mode_ac="Best power efficiency", mode_dc="Balanced")
        self.assertIn("power mode Best power efficiency (mains) / Balanced (battery)", power.describe(state))
        self.assertEqual(state.as_dict()["mode"], {"ac": "Best power efficiency", "dc": "Balanced"})
        self.assertEqual(power.PowerState(power.BATTERY, mode_ac="A", mode_dc="B").mode, "B")

    def test_the_deferral_names_the_mode_and_the_fix(self):
        state = power.with_speed(power.PowerState(power.AC, mode_ac="Best power efficiency"), 2.4, 1.4, sustained=True)
        text = power.deferral_text(state, False)
        self.assertIn("sustained calibration 2.40x slower", text)
        self.assertIn("Windows power mode: Best power efficiency; set Balanced or Best performance", text)
        plain = power.deferral_text(power.with_speed(power.PowerState(power.AC, mode_ac="Balanced"), 2.4, 1.4), False)
        self.assertNotIn("Windows power mode", plain)
        self.assertIn("vendor power mode", plain)


class GateTest(RepoTestCase):
    """The real gate_command with the speed faked: a throttled mains machine defers before building."""

    def setUp(self):
        super().setUp()
        (self.repo / "f.txt").write_text("one", encoding="utf-8")
        git_run(self.repo, "add", ".")
        git_run(self.repo, "commit", "-q", "-m", "one")
        self.head = gitstate.git_head(self.repo)
        self.perf = gitstate.perf_dir(self.repo)
        target = {"build_dir": "b", "exe": "e", "baseline": str(self.repo / "base.json"), "config": ""}
        common.atomic_write_text(self.perf / "config.json", json.dumps({"targets": {"t": target}}))
        (self.repo / "base.json").write_text("{}", encoding="utf-8")
        self.push = f"refs/heads/main {self.head} refs/heads/main {'1' * 40}\n"

    def gate(self, ratio, run_gate=None, **env):
        args = types.SimpleNamespace(repo=str(self.repo), refs_from_stdin=True, lock_timeout=1.0, level="quick", budget=None, rounds=3)
        real = sys.stdin
        sys.stdin = io.StringIO(self.push)
        env = {power.FAKE_VAR: "ac", speed.FAKE_VAR: str(ratio), **env}
        patches = [mock.patch.dict(os.environ, env),
                   mock.patch.object(gate, "changed_paths_between", lambda *a: ["include/x.hpp"]),
                   mock.patch.object(gate.subprocess, "run", side_effect=no_build),
                   mock.patch.object(gate, "run_gate", run_gate or mock.Mock(side_effect=AssertionError("must not measure")))]
        try:
            with contextlib.ExitStack() as stack:
                for p in patches:
                    stack.enter_context(p)
                return captured(gate.gate_command, args)
        finally:
            sys.stdin = real

    def events(self):
        return [json.loads(line)["event"] for line in (self.perf / "log.jsonl").read_text(encoding="utf-8").splitlines()]

    def test_throttled_defers_without_building_and_says_why(self):
        code, out = self.gate(1.88)
        self.assertEqual(code, 0)
        self.assertIn("machine throttled (calibration 1.88x slower than the baseline's)", out)
        self.assertIn("vendor power mode", out)
        self.assertIn("To measure anyway: set SUB0MATH_PERF_ON_BATTERY=1", out)
        pending = deferral.load(self.perf)
        self.assertIn("throttled", pending[0]["reason"])
        self.assertIn("power-deferred", self.events())

    def test_strict_refuses(self):
        code, out = self.gate(1.88, **{power.STRICT_VAR: "1"})
        self.assertEqual(code, 1)
        self.assertIn("strict power mode", out)
        self.assertEqual(deferral.load(self.perf), [])

    def test_the_override_measures_flags_the_state_and_keeps_the_ratio(self):
        run = mock.Mock(return_value=0)
        code, out = self.gate(1.88, run_gate=run, **{power.OVERRIDE_VAR: "1"})
        self.assertEqual(code, 0)
        run.assert_called_once()
        self.assertIn("measuring on a throttled machine", out)
        state = run.call_args.args[0].power_state
        self.assertTrue(state.throttled)
        self.assertEqual(state.as_dict()["speed_ratio"], 1.88)

    def test_a_normal_speed_measures_and_records_the_ratio(self):
        run = mock.Mock(return_value=0)
        code, out = self.gate(1.05, run_gate=run)
        self.assertEqual(code, 0)
        state = run.call_args.args[0].power_state
        self.assertFalse(state.untrusted)
        self.assertEqual(state.as_dict()["speed_ratio"], 1.05)
        rec = make_record("e", {"A": [1.0]}, [{"load": 0.0, "calibration_start_ns": 1.0, "calibration_end_ns": 1.0}], {"A": [0.5]}, {}, 0, 1, state)
        self.assertEqual(rec["power"]["speed_ratio"], 1.05)

    def test_the_configured_limit_applies(self):
        common.atomic_write_text(self.perf / "config.json", json.dumps({"targets": {}, "throttle_ratio": 2.0}))
        self.assertEqual(self.gate(1.88, run_gate=mock.Mock(return_value=0))[0], 0)  # under the configured limit: it measures
        self.assertEqual(deferral.load(self.perf), [])
        self.assertEqual(self.gate(2.5)[0], 0)
        self.assertEqual(len(deferral.load(self.perf)), 1)

    def test_no_calibration_to_compare_proceeds_and_says_so(self):
        run = mock.Mock(return_value=0)
        code, out = self.gate("", run_gate=run, **{speed.FAKE_VAR: ""})
        self.assertEqual(code, 0)
        run.assert_called_once()
        self.assertIn("speed check skipped for t", out)

    def test_battery_does_not_run_the_speed_check(self):
        with mock.patch.object(speed, "check_targets", side_effect=AssertionError("battery defers without timing")):
            code, out = self.gate(1.0, **{power.FAKE_VAR: "battery"})
        self.assertEqual(code, 0)
        self.assertIn("on battery", out)


class AutoCommandTest(unittest.TestCase):
    """The after-build `auto` path: the speed state reaches the record, and does not shadow the coverage state (a crash once)."""

    def run_auto(self, ratio, **env):
        from ..runner import list_names  # noqa: F401  (the module imports it inside measure_and_judge)
        rec = record({"A": 100, "B": 100, "C": 100})
        args = types.SimpleNamespace(exe="e", baseline="b", target=None, repo=None, report_only=True, lock_timeout=1.0, level="quick",
                                     budget=None, rounds=3, max_rounds=3, filter="", epochs=0, seed=1, out=None, record=False,
                                     confirm_rounds=0, tolerance=0.1, pair_tolerance=0.05, instr_tolerance=0.02, max_cv=0.05,
                                     absolute=False, max_load=0.2, settle=0.0, subset=False)
        env = {power.FAKE_VAR: "ac", speed.FAKE_VAR: str(ratio), **env}
        measured = (rec["samples"], rec["rounds"], rec["positions"], rec["counters"])
        with mock.patch.dict(os.environ, env), mock.patch.object(cli, "measure_all", return_value=measured),                 mock.patch("sub0perf.runner.list_names", return_value=["A", "B", "C"]),                 mock.patch.object(cli, "choose_benchmarks", return_value=(["A", "B", "C"], 3, 0.0)):
            return captured(cli.auto_command, args)

    def test_a_throttled_machine_defers_the_after_build_check(self):
        code, out = self.run_auto(1.9)
        self.assertEqual(code, 0)
        self.assertIn("machine throttled", out)
        self.assertIn("performance not measured", out)

    def test_a_normal_speed_measures_and_the_record_carries_the_ratio(self):
        seen = []
        with mock.patch.object(cli, "report", side_effect=lambda rec, args: seen.append(rec) or ([], [], [])):
            code, out = self.run_auto(1.02)
        self.assertEqual(code, 0)
        self.assertEqual(seen[0]["power"]["speed_ratio"], 1.02)
        self.assertNotIn("power_override", seen[0])

    def test_the_override_flags_the_record(self):
        seen = []
        with mock.patch.object(cli, "report", side_effect=lambda rec, args: seen.append(rec) or ([], [], [])):
            code, out = self.run_auto(1.9, **{power.OVERRIDE_VAR: "1"})
        self.assertTrue(seen[0]["power_override"])
        self.assertTrue(seen[0]["power"]["throttled"])
        self.assertIn("measuring on a throttled machine", out)


class PowerCommandTest(unittest.TestCase):
    def test_the_power_command_reports_a_throttled_machine_with_exit_3(self):
        args = types.SimpleNamespace(exe="e", baseline="b", target=None, repo=None)
        with mock.patch.dict(os.environ, {power.FAKE_VAR: "ac", speed.FAKE_VAR: "1.9"}):
            code, out = captured(cli.power_command, args)
        self.assertEqual(code, 3)
        self.assertIn("throttled", out)

    def test_without_exe_and_baseline_no_speed_check_runs(self):
        args = types.SimpleNamespace(exe=None, baseline=None, target=None, repo=None)
        with mock.patch.dict(os.environ, {power.FAKE_VAR: "ac", speed.FAKE_VAR: "1.9"}):
            code, _ = captured(cli.power_command, args)
        self.assertEqual(code, 0)


if __name__ == "__main__":
    unittest.main()


class GateRecordTest(unittest.TestCase):
    """run_gate hands make_record the power state, not the coverage state of the same name (a crash once, on a real push)."""

    def test_the_gate_record_gets_the_power_state(self):
        from .. import gate
        seen = []

        class Stop(Exception):
            pass

        def fake_record(*a, **k):
            seen.append(a[-1])
            raise Stop()

        with tempfile.TemporaryDirectory() as d:
            config = Path(d) / "config.json"
            config.write_text(json.dumps({"targets": {"T": {"build_dir": "b", "exe": "e", "baseline": "b.json"}}}), encoding="utf-8")
            with mock.patch.dict(os.environ, {power.FAKE_VAR: "ac"}):
                now = power.power_state()
            args = types.SimpleNamespace(power_state=now, power_policy=power.policy(config={}), level="quick", budget_seconds=90.0,
                                         epochs=0, seed=1)
            with mock.patch.object(gate, "git", return_value="up"), \
                    mock.patch.object(gate.subprocess, "run", return_value=types.SimpleNamespace(returncode=0)), \
                    mock.patch.object(gate, "list_names", return_value=["A"]), \
                    mock.patch.object(gate, "choose_benchmarks", return_value=(["A"], 3, 0.0)), \
                    mock.patch.object(gate, "measure_all", return_value=({}, [], {}, {})), \
                    mock.patch.object(gate, "make_record", side_effect=fake_record):
                with self.assertRaises(Stop):
                    gate.run_gate(args, d, Path(d), config, "abc123456", True)
        self.assertIs(seen[0], now)


class ReadSustainedTest(unittest.TestCase):
    """read_sustained_ns turns every way the sustained reading can fail into SustainedUnavailable with a plain message."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.exe = Path(tmp.name) / "bench.exe"
        self.exe.write_text("x", encoding="utf-8")

    def read(self, side_effect=None, return_value=None, warm=4.5, elapsed=None):
        from .. import runner
        clock = iter([0.0, warm if elapsed is None else elapsed])
        with mock.patch.object(runner, "measure", side_effect=side_effect, return_value=return_value),                 mock.patch.object(speed.time, "monotonic", lambda: next(clock)):
            return speed.read_sustained_ns(str(self.exe), warm)

    def assertUnavailable(self, contains, **kw):
        with self.assertRaises(speed.SustainedUnavailable) as caught:
            self.read(**kw)
        for text in contains:
            self.assertIn(text, str(caught.exception))

    def test_a_good_reading(self):
        self.assertEqual(self.read(return_value={common.CALIBRATION: 4700.0}), 4700.0)

    def test_an_old_executable_that_writes_no_output_file(self):
        """The 2026-11-04 push: FileNotFoundError for one.json crashed the check."""
        self.assertUnavailable(["wrote no output", "predates `--soak`", "rebuild"], side_effect=FileNotFoundError("one.json"))

    def test_an_old_executable_that_returns_at_once(self):
        self.assertUnavailable(["returned after 0.3 s", "ignores `--soak`", "rebuild"],
                               return_value={common.CALIBRATION: 4700.0}, elapsed=0.3)

    def test_unparseable_output(self):
        self.assertUnavailable(["could not be read", "JSONDecodeError", "rebuild"], side_effect=json.JSONDecodeError("bad", "{", 1))
        self.assertUnavailable(["could not be read", "KeyError"], side_effect=KeyError("__calibration"))

    def test_a_failing_process(self):
        self.assertUnavailable(["exited with code 3", "run it by hand"], side_effect=subprocess.CalledProcessError(3, "x"))

    def test_a_hung_process(self):
        self.assertUnavailable(["did not finish within 48 s", "retry when it is idle"], side_effect=subprocess.TimeoutExpired("x", 48))

    def test_a_missing_executable_is_not_blamed_on_soak(self):
        self.exe.unlink()
        self.assertUnavailable(["cannot be found", "build it"], side_effect=FileNotFoundError("x"))

    def test_an_executable_that_cannot_start(self):
        self.assertUnavailable(["could not be run", "check the build"], side_effect=PermissionError("denied"))

    def test_recording_a_baseline_stores_nothing_when_unavailable(self):
        def read(exe, warm):
            raise speed.SustainedUnavailable("x")
        self.assertIsNone(speed.record_sustained("exe", {}, None, read))
