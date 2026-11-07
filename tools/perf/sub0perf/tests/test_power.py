"""Tests for the power-state guard: the query, the gate's decisions, the records and verdicts, and the loud diagnostics."""

import contextlib
import io
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from .. import accept, cli, common, deferral, gate, gitstate, power, report, runner
from ..baseline import baseline_data, make_record
from ..judge import evaluate
from .helpers import RepoTestCase, git_run, record

BATTERY = power.PowerState(power.BATTERY, source="fake")
MAINS = power.PowerState(power.AC, source="fake")


def captured(call, *args, **kw):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        result = call(*args, **kw)
    return result, out.getvalue()


real_run = __import__("subprocess").run


def no_build(command, *args, **kw):
    """subprocess.run that lets git through and fails the test on a build."""
    if command[0] == "cmake":
        raise AssertionError("nothing may be built on battery")
    return real_run(command, *args, **kw)


class QueryTest(unittest.TestCase):
    def test_fake_values(self):
        for value, state, saver in (("ac", "ac", False), ("battery", "battery", False), ("unknown", "unknown", False),
                                    ("saver", "ac", True), ("garbage", "unknown", False)):
            found = power.power_state({power.FAKE_VAR: value})
            self.assertEqual((found.state, found.saver, found.source), (state, saver, "fake"))

    def test_unknown_counts_as_mains_but_is_reported(self):
        unknown = power.power_state({power.FAKE_VAR: "unknown"})
        self.assertFalse(unknown.untrusted)
        self.assertEqual(unknown.as_dict()["state"], "unknown")
        self.assertEqual(power.key({"state": "unknown"}), power.key(None))

    def test_a_saver_on_mains_is_untrusted(self):
        saver = power.power_state({power.FAKE_VAR: "saver"})
        self.assertTrue(saver.untrusted)
        self.assertIn("saver", saver.reason())

    def test_linux_sysfs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            def supply(name, kind, **files):
                (root / name).mkdir()
                for key, value in {"type": kind, **files}.items():
                    (root / name / key).write_text(value + "\n")
            supply("BAT0", "Battery", status="Discharging")
            self.assertEqual(power._linux(root).state, "battery")
            supply("AC", "Mains", online="0")
            self.assertEqual(power._linux(root).state, "battery")
            (root / "AC" / "online").write_text("1\n")
            self.assertEqual(power._linux(root).state, "ac")
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "AC").mkdir()
            (Path(tmp) / "AC" / "type").write_text("Mains\n")
            self.assertEqual(power._linux(Path(tmp)).state, "ac")  # no battery: a desktop
            self.assertEqual(power._linux(Path(tmp) / "missing").state, "unknown")

    def test_linux_low_power_profile_is_a_saver(self):
        with tempfile.TemporaryDirectory() as tmp:
            profile = Path(tmp) / "profile"
            profile.write_text("low-power\n")
            (Path(tmp) / "psu").mkdir()
            self.assertTrue(power._linux(Path(tmp) / "psu", profile).saver)


class DiagnosticsTest(unittest.TestCase):
    def test_override_on_battery_is_loud(self):
        notes = power.environment_notes(BATTERY, power.Policy(override=True), {})
        self.assertEqual(notes, [f"sub0perf: {power.OVERRIDE_TEXT}"])
        self.assertIn("not comparable with the mains baseline", notes[0])
        self.assertIn("cannot become one", notes[0])

    def test_override_set_on_mains_is_reported_as_ignored(self):
        notes = power.environment_notes(MAINS, power.Policy(override=True), {})
        self.assertEqual(len(notes), 1)
        self.assertIn("set but ignored (on mains)", notes[0])

    def test_strict_is_reported(self):
        self.assertIn("strict power mode", power.environment_notes(MAINS, power.Policy(strict=True), {})[0])

    def test_a_fake_is_a_warning(self):
        notes = power.environment_notes(BATTERY, power.Policy(), {power.FAKE_VAR: "battery"})
        self.assertIn("WARNING", notes[0])
        self.assertIn("power state is being faked", notes[0])

    def test_nothing_set_nothing_said(self):
        self.assertEqual(power.environment_notes(MAINS, power.Policy(), {}), [])

    def test_deferral_hint_names_the_override(self):
        text = power.deferral_text(BATTERY, strict=False)
        self.assertIn("on battery: performance not measured", text)
        self.assertIn("the next push on mains measures against the unchanged baseline", text)
        self.assertIn("To measure anyway: set SUB0MATH_PERF_ON_BATTERY=1. Results are flagged and can't become a baseline.", text)

    def test_strict_hint_names_the_override_and_how_to_turn_strict_off(self):
        text = power.deferral_text(BATTERY, strict=True)
        self.assertIn("refused", text)
        self.assertIn("SUB0MATH_PERF_ON_BATTERY=1", text)
        self.assertIn("unset SUB0MATH_PERF_POWER_STRICT", text)
        self.assertIn("strict_power", text)

    def test_policy_reads_environment_and_config(self):
        self.assertEqual(power.policy({}, None), power.Policy(False, False))
        self.assertEqual(power.policy({power.OVERRIDE_VAR: "1"}, None), power.Policy(True, False))
        self.assertTrue(power.policy({}, {"strict_power": True}).strict)
        self.assertFalse(power.policy({power.STRICT_VAR: "0"}, {"strict_power": True}).strict)  # the environment wins

    def test_power_command_prints_state_and_switches(self):
        with mock.patch.dict(os.environ, {power.FAKE_VAR: "battery", power.OVERRIDE_VAR: "1"}):
            code, out = captured(cli.power_command, None)
        self.assertEqual(code, 3)
        self.assertIn("power: battery", out)
        self.assertIn(power.OVERRIDE_TEXT, out)
        self.assertIn("WARNING", out)


class GateDecisionTest(RepoTestCase):
    """The real gate_command, not plan-only, with the power state faked. Building (cmake) must never be reached on battery."""

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

    def gate(self, fake, run_gate=None, **env):
        args = types.SimpleNamespace(repo=str(self.repo), refs_from_stdin=True, lock_timeout=1.0, level="quick", budget=None, rounds=3)
        real = sys.stdin
        sys.stdin = io.StringIO(self.push)
        patches = [mock.patch.dict(os.environ, {power.FAKE_VAR: fake, **env}),
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

    def test_battery_defers_without_building_and_records_the_range(self):
        code, out = self.gate("battery")
        self.assertEqual(code, 0)
        self.assertIn("on battery: performance not measured", out)
        self.assertIn("To measure anyway: set SUB0MATH_PERF_ON_BATTERY=1", out)
        pending = deferral.load(self.perf)
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0]["head"], self.head)
        self.assertIn(self.head[:9], pending[0]["ranges"][0])
        self.assertIn("power-deferred", self.events())

    def test_battery_saver_defers(self):
        self.assertEqual(self.gate("saver")[0], 0)
        self.assertIn("battery saver", deferral.load(self.perf)[0]["reason"])

    def test_documentation_only_still_skips_first(self):
        with mock.patch.object(gate, "documentation_only", lambda *a: True):
            code, out = self.gate("battery")
        self.assertEqual(code, 0)
        self.assertNotIn("on battery", out)
        self.assertEqual(deferral.load(self.perf), [])

    def test_strict_fails_with_hints_and_defers_nothing(self):
        code, out = self.gate("battery", **{power.STRICT_VAR: "1"})
        self.assertEqual(code, 1)
        self.assertIn("SUB0MATH_PERF_ON_BATTERY=1", out)
        self.assertIn("unset SUB0MATH_PERF_POWER_STRICT", out)
        self.assertIn("strict power mode", out)
        self.assertEqual(deferral.load(self.perf), [])
        self.assertIn("power-refused", self.events())

    def test_strict_from_the_config(self):
        common.atomic_write_text(self.perf / "config.json", json.dumps({"targets": {}, "strict_power": True}))
        self.assertEqual(self.gate("battery")[0], 1)

    def test_override_measures_and_says_so(self):
        run = mock.Mock(return_value=0)
        code, out = self.gate("battery", run_gate=run, **{power.OVERRIDE_VAR: "1"})
        self.assertEqual(code, 0)
        run.assert_called_once()
        self.assertIn(f"sub0perf: {power.OVERRIDE_TEXT}", out)
        self.assertEqual(run.call_args.args[0].power_state.state, "battery")

    def test_override_on_mains_is_reported_as_ignored(self):
        run = mock.Mock(return_value=0)
        code, out = self.gate("ac", run_gate=run, **{power.OVERRIDE_VAR: "1"})
        self.assertEqual(code, 0)
        self.assertIn("set but ignored (on mains)", out)
        self.assertNotIn("measuring on battery", out)

    def test_unknown_proceeds(self):
        run = mock.Mock(return_value=0)
        self.assertEqual(self.gate("unknown", run_gate=run)[0], 0)
        run.assert_called_once()

    def test_a_fake_outside_tests_is_warned_about(self):
        run = mock.Mock(return_value=0)
        self.assertIn("power state is being faked", self.gate("ac", run_gate=run)[1])

    def test_a_switch_to_battery_mid_run_is_inconclusive_and_deferred(self):
        run = mock.Mock(side_effect=runner.PowerChanged(BATTERY))
        code, out = self.gate("ac", run_gate=run)
        self.assertEqual(code, 0)
        self.assertIn("inconclusive", out)
        self.assertIn("baseline untouched", out)
        self.assertIn("To measure anyway", out)
        self.assertTrue(deferral.load(self.perf)[0]["partial"])
        self.assertIn("power-stopped", self.events())

    def test_a_switch_mid_run_in_strict_mode_fails(self):
        run = mock.Mock(side_effect=runner.PowerChanged(BATTERY))
        self.assertEqual(self.gate("ac", run_gate=run, **{power.STRICT_VAR: "1"})[0], 1)

    def test_the_round_guard_is_installed_only_without_the_override(self):
        seen = []
        run = mock.Mock(side_effect=lambda *a: seen.append(runner._round_guard) or 0)
        self.gate("ac", run_gate=run)
        self.gate("battery", run_gate=run, **{power.OVERRIDE_VAR: "1"})
        self.assertIs(seen[0], gate.power_guard)
        self.assertIsNone(seen[1])
        self.assertIsNone(runner._round_guard)  # and removed afterwards


class RoundGuardTest(unittest.TestCase):
    def test_a_switch_between_rounds_stops_the_run(self):
        states = iter([MAINS, MAINS, BATTERY])
        measured = []
        with mock.patch.object(runner, "wait_for_quiet", lambda *a: (0.0, 0.0, True)), \
             mock.patch.object(runner, "measure", lambda *a: {common.CALIBRATION: 1.0}), \
             mock.patch.object(runner, "measure_full", lambda *a: measured.append(1) or {"B": (1.0, {})}), \
             mock.patch.object(power, "power_state", lambda *a: next(states)):
            runner.set_round_guard(gate.power_guard)
            try:
                with self.assertRaises(runner.PowerChanged):
                    runner.run_rounds("exe", ["B"], 5, 0, 0.2, 1.0, 1)
            finally:
                runner.set_round_guard(None)
        self.assertEqual(len(measured), 2)  # two rounds ran; the third was stopped before it started


class RecordTest(unittest.TestCase):
    def test_records_and_baselines_carry_the_power_state(self):
        made = make_record("exe", {}, [], {}, {}, state=BATTERY)
        self.assertEqual(made["power"], {"state": "battery", "saver": False})
        full = {**record({"A": 100.0}), "power": {"state": "ac", "saver": False, "scheme": "Balanced"}}
        self.assertEqual(baseline_data(full)["power"]["scheme"], "Balanced")
        self.assertNotIn("power", baseline_data(record({"A": 100.0})))  # an old record stays without the field

    def judged(self, base_power, now_power, now_ns=100.0):
        base = {**record({"A": 100.0, "B, hand-written": 100.0, "B, lib": 120.0}, seed=1)}
        if base_power:
            base["power"] = base_power
        now = {**record({"A": now_ns, "B, hand-written": 100.0, "B, lib": 120.0}, seed=2)}
        if now_power:
            now["power"] = now_power
        return evaluate(now, baseline_data(base), 0.10, 0.05, 0.05, False)

    def test_a_different_power_state_makes_absolute_verdicts_environment(self):
        rows, pairs, env = self.judged({"state": "ac"}, {"state": "battery"}, now_ns=300.0)
        self.assertEqual({r[0]: r[5] for r in rows}["A"], "ENVIRONMENT")
        self.assertIn("power state differs", env["method"])
        self.assertEqual({p[0]: p[4] for p in pairs}["B, lib"], "ok")  # pair ratios still judge

    def test_a_saver_on_mains_differs_too(self):
        rows, _, env = self.judged({"state": "ac"}, {"state": "ac", "saver": True}, now_ns=300.0)
        self.assertEqual({r[0]: r[5] for r in rows}["A"], "ENVIRONMENT")

    def test_old_baselines_and_unknown_judge_as_mains(self):
        for base, now in ((None, {"state": "ac"}), ({"state": "ac"}, None), ({"state": "unknown"}, {"state": "ac"})):
            rows, _, env = self.judged(base, now, now_ns=300.0)
            self.assertEqual({r[0]: r[5] for r in rows}["A"], "REGRESSED", (base, now))
            self.assertIsNone(env["method"])

    def test_the_report_shows_the_override(self):
        _, _, env = self.judged({"state": "ac"}, {"state": "battery"})
        env["power_override"] = True
        text = report.render([], [], {**env, "baseline": True}, "b.json", 0.1, 0.05)
        self.assertIn(power.OVERRIDE_TEXT, text)
        self.assertIn("override: true", text)


class BaselineRefusalTest(unittest.TestCase):
    def refusal(self, rec, force=False):
        return captured(accept.power_refusal, rec, "samples.json", force)

    def test_mains_old_and_unknown_are_allowed(self):
        self.assertFalse(self.refusal({"power": {"state": "ac"}})[0])
        self.assertFalse(self.refusal({})[0])
        refused, out = self.refusal({"power": {"state": "unknown"}})
        self.assertFalse(refused)
        self.assertIn("unknown", out)

    def test_battery_and_saver_are_refused_unless_forced(self):
        for info in ({"state": "battery"}, {"state": "ac", "saver": True}):
            self.assertTrue(self.refusal({"power": info})[0])
            refused, out = self.refusal({"power": info}, force=True)
            self.assertFalse(refused)
            self.assertIn("WARNING", out)

    def test_an_override_run_is_never_allowed(self):
        refused, out = self.refusal({"power": {"state": "battery"}, "power_override": True}, force=True)
        self.assertTrue(refused)
        self.assertIn("can never become a baseline", out)

    def test_baseline_and_accept_commands_refuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            results, base = Path(tmp) / "r.json", Path(tmp) / "b.json"
            results.write_text(json.dumps({**record({"A": 100.0}), "power": {"state": "battery"}}), encoding="utf-8")
            args = types.SimpleNamespace(results=str(results), baseline=str(base), reason="x", force_power=False, all_changed=False,
                                         benchmark=["A"])
            self.assertEqual(captured(accept.baseline_command, args)[0], 2)
            self.assertEqual(captured(accept.accept_command, args)[0], 2)
            self.assertFalse(base.exists())


class DeferralTest(RepoTestCase):
    def test_describe_lists_the_unmeasured_ranges_and_clear_empties_them(self):
        perf = gitstate.perf_dir(self.repo)
        deferral.add(perf, "a" * 40, [("r", "a" * 40, "refs/heads/main", "b" * 40)], "on battery")
        deferral.add(perf, "c" * 40, [], "battery saver is on", partial=True)
        lines = deferral.describe(deferral.load(perf))
        self.assertIn("2 earlier push(es) were not measured", lines[0])
        self.assertIn("bbbbbbbbb..aaaaaaaaa", lines[1])
        self.assertIn("stopped mid-run", lines[2])
        deferral.clear(perf)
        self.assertEqual(deferral.describe(deferral.load(perf)), [])
