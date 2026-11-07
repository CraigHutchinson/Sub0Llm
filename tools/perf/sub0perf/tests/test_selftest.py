"""Tests for selftest: the wiring checks and the plan-only gate decisions."""

import json
import os
import sys
import unittest
from pathlib import Path

from .. import common, gate, gitstate, hooks as hk, plan as planning, selftest as st
from .helpers import RepoTestCase, git_run

HAND = "Pair A, hand-written"
NAMES = [HAND, "Pair A, library", "Solo 1", "Solo 2"]


class SelftestCase(RepoTestCase):
    def setUp(self):
        super().setUp()
        (self.repo / "f.txt").write_text("one", encoding="utf-8")
        git_run(self.repo, "add", ".")
        git_run(self.repo, "commit", "-q", "-m", "one")
        git_run(self.repo, "commit", "-q", "--allow-empty", "-m", "two")
        self.baseline = self.repo / "base.json"
        self.baseline.write_text(json.dumps({"benchmarks": {n: {"median": 1.0} for n in NAMES}}), encoding="utf-8")
        self.build = self.repo / "build"

    def install(self, **extra):
        args = type("A", (), {"repo": str(self.repo), "build_dir": str(self.build), "target": "tgt", "exe": str(self.build / "tgt.exe"),
                              "baseline": str(self.baseline), "config": "", **extra})
        hk.install_hooks_command(args)

    def costs(self, **by_name):
        state = planning.default_state()
        state["costs"] = {n: by_name.get(n, 1.0) for n in NAMES}
        planning.save_state(gitstate.perf_dir(self.repo), "tgt", state)

    def rows(self, **kw):
        return {check: (status, detail) for status, check, detail in st.run_selftest(self.repo, **kw)}

    def status(self, check, **kw):
        return self.rows(**kw)[check][0]

    def hook_path(self, name="pre-push"):
        return st.hooks_dir(self.repo) / name


class WiringTest(SelftestCase):
    def test_a_fresh_install_has_no_failure(self):
        self.install()
        self.costs()
        rows = st.run_selftest(self.repo)
        self.assertEqual([r for r in rows if r[0] == st.FAIL], [])
        self.assertEqual(self.status("hook pre-push"), st.PASS)

    def test_unbuilt_target_is_a_warning_not_a_failure(self):
        self.install()
        self.costs()
        status, detail = self.rows()["target tgt"]
        self.assertEqual(status, st.WARN)
        self.assertIn("not built yet", detail)
        self.build.mkdir()
        self.assertEqual(self.status("target tgt"), st.WARN)  # build dir but no exe
        (self.build / "tgt.exe").write_text("x", encoding="utf-8")
        self.assertEqual(self.status("target tgt"), st.PASS)

    def test_missing_hook_fails(self):
        self.install()
        self.hook_path().unlink()
        self.assertEqual(self.status("hook pre-push"), st.FAIL)

    def test_hook_in_the_wrong_place_fails(self):
        self.install()
        git_run(self.repo, "config", "core.hooksPath", "elsewhere")  # git now looks elsewhere: the installed hook is inactive
        status, detail = self.rows()["hook pre-push"]
        self.assertEqual(status, st.FAIL)
        self.assertIn("elsewhere", detail)

    def test_foreign_hook_fails(self):
        self.install()
        self.hook_path().write_text("#!/bin/sh\nexit 0\n", encoding="utf-8", newline="\n")
        self.assertEqual(self.status("hook pre-push"), st.FAIL)

    def test_missing_script_fails(self):
        self.install()
        path = self.hook_path()
        path.write_text(hk.hook_body("pre-push", sys.executable, "/no/such/bench_check.py", hk.HOOK_COMMENTS["pre-push"]),
                        encoding="utf-8", newline="\n")
        status, detail = self.rows()["hook pre-push"]
        self.assertEqual(status, st.FAIL)
        self.assertIn("not an existing bench_check.py", detail)

    def test_drifted_logic_fails_and_a_changed_path_only_warns(self):
        self.install()
        path = self.hook_path()
        good = path.read_text(encoding="utf-8")
        path.write_text(good + "echo extra\n", encoding="utf-8", newline="\n")
        self.assertEqual(self.status("hook pre-push"), st.FAIL)
        python_line = next(l for l in good.splitlines() if l.startswith("python="))
        path.write_text(good.replace(python_line, "python=" + Path(__file__).as_posix()), encoding="utf-8", newline="\n")
        self.assertEqual(self.status("hook pre-push"), st.WARN)  # another python that exists: same logic, other paths

    def test_missing_post_checkout_only_warns(self):
        self.install()
        self.hook_path("post-checkout").unlink()
        self.assertEqual(self.status("hook post-checkout"), st.WARN)

    def test_invalid_config_fails(self):
        self.install()
        (gitstate.perf_dir(self.repo) / "config.json").write_text("{not json", encoding="utf-8")
        self.assertEqual(self.status("config"), st.FAIL)

    def test_no_config_is_a_warning(self):
        self.assertEqual(self.status("config"), st.WARN)

    def test_missing_baseline_fails_and_bad_baseline_fails(self):
        self.install()
        self.baseline.unlink()
        self.assertEqual(self.status("target tgt baseline"), st.FAIL)
        self.baseline.write_text("{}", encoding="utf-8")
        self.assertEqual(self.status("target tgt baseline"), st.FAIL)

    def test_pins_are_parsed(self):
        (self.repo / "deps.cmake").write_text("set(X_REV abcdef1234567)\n", encoding="utf-8")
        self.install(pin=["file=deps.cmake,var=X_REV,repo=o/n", "file=deps.cmake,var=NOPE,repo=o/n", "file=gone.cmake,var=Y,repo=o/n"])
        rows = {r[1]: r[0] for r in st.run_selftest(self.repo)}
        self.assertEqual(rows["pin X_REV"], st.PASS)
        self.assertEqual(rows["pin NOPE"], st.WARN)
        self.assertEqual(rows["pin Y"], st.FAIL)

    def test_only_target_restricts_and_unknown_target_fails(self):
        self.install()
        self.assertEqual(self.status("target nope", target="nope"), st.FAIL)

    def test_stale_lock_fails_and_live_lock_warns(self):
        self.install()
        lock = gitstate.perf_dir(self.repo) / "lock"
        lock.write_text(json.dumps({"pid": 2 ** 22 + 12345, "time": 0, "what": "gate"}), encoding="utf-8")
        self.assertEqual(self.status("lock"), st.FAIL)
        import time
        lock.write_text(json.dumps({"pid": os.getpid(), "time": time.time(), "what": "gate"}), encoding="utf-8")
        self.assertEqual(self.status("lock"), st.WARN)
        lock.unlink()
        self.assertEqual(self.status("lock"), st.PASS)


class DecisionTest(SelftestCase):
    def setUp(self):
        super().setUp()
        self.install()

    def test_decisions_pass_and_nothing_is_built_logged_or_locked(self):
        self.costs()
        rows = self.rows()
        for check in ("gate: empty stdin (up to date)", "gate: deletion only", "gate: docs-only range", "gate: code range tgt"):
            self.assertEqual(rows[check][0], st.PASS, rows[check])
        self.assertFalse((gitstate.perf_dir(self.repo) / "log.jsonl").exists())
        self.assertFalse((gitstate.perf_dir(self.repo) / "lock").exists())

    def test_plan_only_never_runs_a_subprocess_for_building(self):
        self.costs()
        head = gitstate.git_head(self.repo)
        real, started = gate.subprocess, []
        gate.subprocess = type("Spy", (), {"run": staticmethod(lambda *a, **k: started.append(a)),
                                           "Popen": staticmethod(lambda *a, **k: started.append(a))})
        try:
            code, outcome, previews, _ = st.drive(self.repo, f"refs/heads/b {head} refs/heads/b {'1' * 40}\n",
                                                  lambda r, l: ["src/a.cpp"], 90.0)
        finally:
            gate.subprocess = real
        self.assertEqual(started, [])  # gate.py itself starts no process: no cmake build, no benchmark
        self.assertEqual((code, outcome), (0, "measure"))
        self.assertEqual(previews[0]["target"], "tgt")

    def test_bypass_variables_and_a_dirty_tree_do_not_change_decisions(self):
        self.costs()
        (self.repo / "f.txt").write_text("dirty", encoding="utf-8")
        os.environ["SUB0MATH_PERF_SKIP"] = "x"
        try:
            rows = self.rows()
        finally:
            del os.environ["SUB0MATH_PERF_SKIP"]
        self.assertEqual(rows["gate: code range tgt"][0], st.PASS)
        self.assertEqual(rows["environment"][0], st.WARN)

    def test_a_plan_over_the_budget_fails(self):
        self.costs(**{HAND: 200.0})  # the control alone costs minutes per round
        status, detail = self.rows()["gate: code range tgt"]
        self.assertEqual(status, st.FAIL)
        self.assertIn("OVER", detail)

    def test_unknown_and_slow_benchmarks_are_reported_and_the_allowance_counted(self):
        state = planning.default_state()
        state["costs"] = {HAND: 1.0, "Pair A, library": 1.0}
        state["slow"] = {"Solo 2": 500.0}
        planning.save_state(gitstate.perf_dir(self.repo), "tgt", state)
        rows = self.rows()
        status, detail = rows["coverage tgt"]
        self.assertEqual(status, st.WARN)
        self.assertIn("1 unknown", detail)
        self.assertIn("1 slow", detail)
        self.assertIn("stalest group", detail)
        self.assertIn("18 s probe allowance", rows["gate: code range tgt"][1])  # a fifth of 90 s for the unknown Solo 1

    def test_a_budget_can_be_given(self):
        self.costs()
        status, detail = self.rows(budget=5.0)["gate: code range tgt"]
        self.assertIn("5 s budget", detail)

    def test_no_config_has_no_decisions_to_test(self):
        (gitstate.perf_dir(self.repo) / "config.json").unlink()
        self.assertEqual(self.status("gate decisions"), st.WARN)


class CommandTest(SelftestCase):
    def test_exit_code_follows_failures(self):
        self.install()
        self.costs()
        args = type("A", (), {"repo": str(self.repo), "target": None, "budget": None})
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(st.selftest_command(args), 0)
            self.hook_path().unlink()
            self.assertEqual(st.selftest_command(args), 1)
        self.assertIn("failed", out.getvalue())


if __name__ == "__main__":
    unittest.main()
