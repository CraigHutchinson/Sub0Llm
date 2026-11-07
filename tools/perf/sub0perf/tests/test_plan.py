"""Tests for plan: fitting a quick gate to a time budget, rotation and coverage, and what overruns the budget."""

import tempfile
import unittest
from pathlib import Path

from .. import confirm, judge, plan
from ..baseline import baseline_data
from .helpers import record

# Five pair groups (a library benchmark and its hand-written reference), then standalone benchmarks.
PAIRS = [n for i in range(5) for n in (f"p{i}, Length", f"p{i}, hand-written")]
LOOSE = [f"Loose {i:02d}" for i in range(10)]
ALL = PAIRS + LOOSE


def state_with(cost=1.0, overhead=2.0, measured=None, pushes=0, recheck=()):
    return {"pushes": pushes, "costs": {n: cost for n in ALL}, "round_overhead": overhead, "benchmarks": measured or {},
            "recheck": list(recheck)}


class UnitsTest(unittest.TestCase):
    def test_pair_groups_are_units_and_the_rest_stand_alone(self):
        units = plan.units_of(ALL)
        self.assertEqual(len(units), 5 + 10)
        self.assertEqual(sorted(units["p0"]), ["p0, Length", "p0, hand-written"])
        self.assertEqual(units["Loose 03"], ["Loose 03"])

    def test_a_group_without_reference_is_not_one_unit(self):
        self.assertEqual(sorted(plan.units_of(["a, x", "a, y", "b, hand-written", "b, z"])), ["a, x", "a, y", "b"])

    def test_controls_are_the_cheapest_references_and_avoid_bimodal(self):
        costs = {n: 1.0 for n in ALL}
        costs["p3, hand-written"] = 0.1
        costs["p1, hand-written"] = 0.2
        self.assertEqual(plan.choose_controls(ALL, costs), ["p3, hand-written", "p1, hand-written"])
        self.assertNotIn("p3, hand-written", plan.choose_controls(ALL, costs, avoid=["p3, hand-written"]))
        self.assertEqual(plan.choose_controls(ALL, costs, avoid=["p3, hand-written"])[0], "p1, hand-written")


class PlanFitTest(unittest.TestCase):
    def test_fits_the_budget_and_keeps_the_reserve(self):
        p = plan.plan_quick(ALL, state_with(), "c1", 60.0)
        self.assertLessEqual(p.estimate, 60.0 * (1 - plan.RESERVE))
        self.assertEqual(p.rounds, 3)
        self.assertEqual(p.overrun, 0.0)
        self.assertTrue(p.rotating)
        self.assertEqual(len(p.rechecks) + len(p.rotating) + len(p.skipped), len(p.units))

    def test_only_the_controls_are_always_measured_not_every_pair(self):
        p = plan.plan_quick(ALL, state_with(), "c1", 20.0)  # 3 x (2 + 2 controls) = 12 s; 15 s usable
        self.assertEqual(len(p.controls), plan.CONTROLS)
        self.assertTrue(set(p.controls) <= set(p.selected))
        self.assertLess(len([n for n in p.selected if n in PAIRS]), len(PAIRS))

    def test_a_pair_group_is_measured_whole_unless_only_its_control_is(self):
        for k in range(8):
            p = plan.plan_quick(ALL, state_with(), f"c{k}", 45.0)
            for i in range(5):
                both = {f"p{i}, Length", f"p{i}, hand-written"}
                allowed = (0, 1, 2) if f"p{i}, hand-written" in p.controls else (0, 2)
                self.assertIn(len(both & set(p.selected)), allowed)
                if f"p{i}, Length" in p.selected:
                    self.assertIn(f"p{i}, hand-written", p.selected)

    def test_a_big_budget_measures_everything(self):
        p = plan.plan_quick(ALL, state_with(), "c1", 10000.0)
        self.assertEqual(sorted(p.selected), sorted(ALL))
        self.assertEqual(p.skipped, [])
        self.assertEqual(p.pushes_to_cover, 1)

    def test_one_group_over_the_budget_is_measured_alone_with_the_overrun_reported(self):
        state = state_with()
        for n in ("p0, Length", "p0, hand-written"):
            state["costs"][n] = 8.0
        state["benchmarks"] = {n: {"push": 5} for n in ALL if not n.startswith("p0")}  # p0 is the stalest
        p = plan.plan_quick(ALL, state, "c1", 70.0)  # controls 12 s + p0 48 s: over the 52 s usable, within the 70 s budget
        self.assertEqual(p.rotating, ["p0"])
        self.assertIn("p0, Length", p.selected)
        self.assertGreater(p.estimate, 52.5)

    def test_a_group_over_the_whole_budget_is_deferred_not_measured_alone(self):
        state = state_with()
        for n in ("p0, Length", "p0, hand-written"):
            state["costs"][n] = 100.0
        state["benchmarks"] = {n: {"push": 5} for n in ALL if not n.startswith("p0")}  # p0 is the stalest
        p = plan.plan_quick(ALL, state, "c1", 30.0)
        self.assertEqual(p.deferred, ["p0"])
        self.assertNotIn("p0, Length", p.selected)
        self.assertEqual(p.overrun, 0.0)
        text = chr(10).join(plan.describe_plan(p, len(ALL), state, "c1"))
        self.assertIn("not measured yet: 1 group(s)", text)

    def test_expensive_groups_are_skipped_but_cheaper_ones_still_fit(self):
        state = state_with()
        state["costs"]["Loose 00"] = 500.0
        p = plan.plan_quick(ALL, state, "c1", 60.0)
        self.assertNotIn("Loose 00", p.selected)
        self.assertGreater(len(p.rotating), 3)

    def test_unknown_costs_use_the_probe_default(self):
        state = {"pushes": 0, "costs": {}, "benchmarks": {}, "round_overhead": 2.0, "recheck": []}
        p = plan.plan_quick(ALL, state, "c1", 60.0, default_cost=2.0)  # controls 3 x (2 + 4) = 18 s of 45 s usable
        self.assertTrue(p.rotating)
        self.assertLessEqual(p.estimate, 45.0)

    def test_describe_reports_counts_and_coverage(self):
        state = state_with()
        p = plan.plan_quick(ALL, state, "c1", 60.0)
        text = "\n".join(plan.describe_plan(p, len(ALL), state, "c1"))
        self.assertIn(f"{len(p.selected)}/{len(ALL)}", text)
        self.assertIn(f"covered within {p.pushes_to_cover} push(es)", text)
        self.assertIn("controls, always", text)
        self.assertIn("stalest", text)


class RecheckTest(unittest.TestCase):
    def test_groups_awaiting_a_recheck_go_first(self):
        state = state_with(recheck=["p4", "Loose 07"])
        fresh = ("p4, Length", "p4, hand-written", "Loose 07")  # measured last push, so otherwise last in line
        state["benchmarks"] = {n: {"push": 9 if n in fresh else 1} for n in ALL}
        p = plan.plan_quick(ALL, state, "c1", 60.0)
        self.assertEqual(sorted(p.rechecks), ["Loose 07", "p4"])

    def test_rechecks_cannot_take_more_than_their_share(self):
        state = state_with(recheck=[f"p{i}" for i in range(5)])
        p = plan.plan_quick(ALL, state, "c1", 60.0)  # 45 s usable, 12 s of controls: rechecks may use half of the rest
        extra = sum(state["costs"][n] for k in p.rechecks for n in p.units[k] if n not in p.controls) * 3
        self.assertLessEqual(extra, (45.0 - 12.0) * plan.RECHECK_SHARE)
        self.assertLess(len(p.rechecks), 5)
        self.assertTrue(p.rotating)

    def test_update_recheck_sets_and_clears(self):
        units = plan.units_of(ALL)
        state = plan.default_state()
        plan.update_recheck(state, units, ["p1, Length", "p1, hand-written", "Loose 01"], ["p1, Length", "Loose 01"])
        self.assertEqual(state["recheck"], ["Loose 01", "p1"])
        plan.update_recheck(state, units, ["p1, Length", "p1, hand-written"], [])  # p1 measured and clean now
        self.assertEqual(state["recheck"], ["Loose 01"])
        plan.update_recheck(state, units, ["p2, Length", "p2, hand-written"], [])  # Loose 01 not measured: stays
        self.assertEqual(state["recheck"], ["Loose 01"])


class RotationTest(unittest.TestCase):
    def run_pushes(self, budget, pushes):
        state = state_with()
        seen = []
        for k in range(pushes):
            commit = f"commit{k}"
            p = plan.plan_quick(ALL, state, commit, budget)
            plan.mark_measured(state, p.selected, commit, date="2026-10-27T10:00:00")
            seen.append(p)
        return state, seen

    def test_everything_is_covered_within_the_announced_number_of_pushes(self):
        for budget in (25.0, 40.0, 60.0):
            bound = plan.plan_quick(ALL, state_with(), "commit0", budget).pushes_to_cover
            self.assertGreater(bound, 1, budget)
            _, seen = self.run_pushes(budget, bound)
            self.assertEqual(set().union(*(p.selected for p in seen)), set(ALL), budget)

    def test_staleness_priority_goes_to_what_was_skipped_last_time(self):
        state, seen = self.run_pushes(60.0, 1)
        skipped = set(seen[0].skipped)
        second = plan.plan_quick(ALL, state, "commit1", 60.0)
        self.assertTrue(skipped <= set(second.rotating))  # all of them fit, so none waits twice

    def test_deterministic_for_a_commit_and_different_between_commits(self):
        a = plan.plan_quick(ALL, state_with(), "same", 60.0)
        b = plan.plan_quick(ALL, state_with(), "same", 60.0)
        c = plan.plan_quick(ALL, state_with(), "other", 60.0)
        self.assertEqual(a.selected, b.selected)
        self.assertNotEqual(a.rotating, c.rotating)

    def test_mark_measured_counts_pushes_and_records_commit_and_date(self):
        state = plan.default_state()
        plan.mark_measured(state, ["x"], "abc", date="2026-10-27T09:00:00")
        plan.mark_measured(state, ["y"], "def", date="2026-10-28T09:00:00")
        self.assertEqual(state["pushes"], 2)
        self.assertEqual(state["benchmarks"]["x"], {"push": 1, "commit": "abc", "date": "2026-10-27T09:00:00"})
        self.assertEqual(state["benchmarks"]["y"]["push"], 2)

    def test_state_round_trips(self):
        with tempfile.TemporaryDirectory() as tmp:
            perf = Path(tmp)
            self.assertEqual(plan.load_state(perf, "T")["pushes"], 0)
            state = plan.default_state()
            plan.mark_measured(state, ["x"], "abc")
            plan.save_state(perf, "T", state)
            self.assertEqual(plan.load_state(perf, "T")["benchmarks"]["x"]["commit"], "abc")

    def test_absorb_timings_keeps_median_costs(self):
        state = plan.default_state()
        plan.absorb_timings(state, {"bench": {"a": [1.0, 3.0, 2.0]}, "overhead": [4.0, 6.0]})
        self.assertEqual(state["costs"]["a"], 2.0)
        self.assertEqual(state["round_overhead"], 5.0)


class BudgetTest(unittest.TestCase):
    def test_remaining_and_overrun_follow_the_clock(self):
        now = [100.0]
        b = plan.Budget(60.0, clock=lambda: now[0])
        self.assertTrue(b.affords(60.0))
        now[0] = 150.0
        self.assertAlmostEqual(b.remaining(), 10.0)
        self.assertFalse(b.affords(11.0))
        self.assertEqual(b.overrun(), 0.0)
        now[0] = 175.0
        self.assertAlmostEqual(b.overrun(), 15.0)

    def test_no_budget_affords_everything(self):
        b = plan.Budget(None, clock=lambda: 0.0)
        self.assertTrue(b.affords(1e9))
        self.assertEqual(b.overrun(), 0.0)


class OverrunTest(unittest.TestCase):
    """A quick gate never skips confirming a regression because of the budget, and skips an inconclusive retry that does not fit."""

    def setUp(self):
        self.baseline = baseline_data(record({"Bench 0": 1000.0, "Bench 1": 1000.0}))
        self.rec = record({"Bench 0": 1500.0, "Bench 1": 1000.0})  # Bench 0 looks regressed

    def evaluate_all(self):
        return [judge.evaluate(self.rec, self.baseline, 0.10, 0.05, 0.05, False)]

    def test_regression_is_confirmed_even_with_no_budget_left(self):
        calls = []
        now = [0.0]
        budget = plan.Budget(10.0, clock=lambda: now[0])
        now[0] = 50.0  # already 40 s over

        def rerun(names, rounds):
            calls.append((sorted(names), rounds))
            now[0] += 30.0

        first, _ = confirm.confirm_regressions(self.evaluate_all, rerun, 4, None)
        self.assertEqual(first, {"Bench 0"})
        self.assertEqual(len(calls), 1)  # re-measured despite the exhausted budget
        self.assertGreater(budget.overrun(), 40.0)  # and the overrun is there to report

    def test_inconclusive_retry_is_skipped_when_it_does_not_fit(self):
        noisy = record({"Bench 0": 1000.0, "Bench 1": 1000.0}, jitter=0.30)
        evaluate_all = lambda: [judge.evaluate(noisy, self.baseline, 0.10, 0.05, 0.05, False)]
        ran = []
        first, final = confirm.retry_inconclusive(evaluate_all, lambda n, r: ran.append(r), 3, None, lambda names, r: False)
        self.assertTrue(first)
        self.assertEqual(ran, [])
        self.assertEqual(first, final)
        confirm.retry_inconclusive(evaluate_all, lambda n, r: ran.append(r), 3, None, lambda names, r: True)
        self.assertEqual(ran, [3])


class SubsetJudgingTest(unittest.TestCase):
    def test_benchmarks_not_measured_this_push_are_not_missing(self):
        base = baseline_data(record({"a": 1000.0, "b": 1000.0, "c": 1000.0}))
        rec = record({"a": 1000.0})
        args = type("A", (), {"tolerance": 0.1, "pair_tolerance": 0.05, "max_cv": 0.05, "absolute": False, "instr_tolerance": 0.02,
                              "filter": "", "subset": True})
        rows, _, _ = judge.evaluate_with(rec, base, args)
        self.assertEqual({r[0]: r[5] for r in rows}, {"a": "ok"})
        args.subset = False
        rows, _, _ = judge.evaluate_with(rec, base, args)
        self.assertIn("missing", {r[5] for r in rows})


if __name__ == "__main__":
    unittest.main()


class DeferTest(unittest.TestCase):
    """Benchmarks with no recorded cost, or found too slow, are not planned at quick; they never justify an overrun."""

    def test_unplanned_lists_unknown_and_slow(self):
        state = state_with()
        del state["costs"]["Loose 00"]
        state["slow"] = {"Loose 01": 9.0}
        self.assertEqual(plan.unplanned(ALL, state), ["Loose 00", "Loose 01"])

    def test_deferred_benchmarks_are_never_selected_or_made_controls(self):
        state = state_with()
        new = ["Trail bike whole run, hand-written", "Trail bike whole run"]
        state["slow"] = {n: 9.0 for n in new}
        p = plan.plan_quick(ALL + new, state, "c1", 60.0, defer=plan.unplanned(ALL + new, state))
        self.assertFalse(set(new) & set(p.selected))
        self.assertFalse(set(new) & set(p.controls))
        self.assertEqual(p.deferred, ["Trail bike whole run"])
        self.assertLessEqual(p.estimate, 60.0)

    def test_new_expensive_benchmarks_do_not_displace_the_budget(self):
        # The 2026-11-03 case: 8 never-measured benchmarks are the stalest and the median cost is tiny.
        state = state_with(cost=0.05)
        new = [f"Whole run {i}" for i in range(8)]
        everything = ALL + new
        state["slow"] = {n: 6.0 for n in new}
        p = plan.plan_quick(everything, state, "c1", 90.0, defer=plan.unplanned(everything, state))
        self.assertFalse(set(new) & set(p.selected))
        self.assertEqual(sorted(p.deferred), sorted(new))
        self.assertLessEqual(p.estimate, 90.0)

    def test_absorbing_a_timing_clears_slow(self):
        state = state_with()
        state["slow"] = {"Loose 01": 9.0}
        timings = plan.new_timings()
        timings["bench"]["Loose 01"] = [20.0]
        plan.absorb_timings(state, timings)
        self.assertEqual(state["slow"], {})
        self.assertEqual(state["costs"]["Loose 01"], 20.0)


class ProbeTest(unittest.TestCase):
    def setUp(self):
        from .. import runner
        self.runner = runner
        self.now = [0.0]
        self.durations = {}
        self.args = type("A", (), {"epochs": 0})()
        self.original = runner.measure_full

        def fake(exe, extra, scratch, timeout=None):
            import subprocess
            name = extra[1]
            took = self.durations[name]
            if timeout is not None and took > timeout:
                self.now[0] += timeout
                raise subprocess.TimeoutExpired(extra, timeout)
            self.now[0] += took
            return {}
        runner.measure_full = fake

    def tearDown(self):
        self.runner.measure_full = self.original

    def probe(self, names, allowance, per_probe):
        return self.runner.probe_unknown("exe", names, self.args, allowance, per_probe, clock=lambda: self.now[0])

    def test_a_slow_benchmark_is_cut_off_and_recorded_as_slow(self):
        self.durations = {"fast": 1.0, "whole run": 600.0}
        costs, slow, spent = self.probe(["fast", "whole run"], 20.0, 5.0)
        self.assertEqual(costs, {"fast": 1.0})
        self.assertEqual(slow, {"whole run": 5.0})
        self.assertAlmostEqual(spent, 6.0)  # not 600

    def test_the_allowance_bounds_the_total_and_leaves_the_rest_unknown(self):
        self.durations = {f"b{i}": 4.0 for i in range(10)}
        costs, slow, spent = self.probe(list(self.durations), 10.0, 5.0)
        self.assertLessEqual(spent, 10.0)
        self.assertLess(len(costs), 10)
        self.assertEqual(slow, {})  # an allowance cut-off says nothing about the benchmark

    def test_launch_kills_a_process_that_outlives_its_timeout(self):
        import subprocess
        import sys
        import time
        from .. import platform_proc
        began = time.monotonic()
        with self.assertRaises(subprocess.TimeoutExpired):
            platform_proc.launch([sys.executable, "-c", "import time; time.sleep(60)"], 0.5)
        self.assertLess(time.monotonic() - began, 20)


class ChooseBenchmarksTest(unittest.TestCase):
    """The gate probes benchmarks with no recorded cost before planning, and plans none that are slow or unreached."""

    def test_new_slow_benchmarks_are_probed_cut_off_and_left_out(self):
        from .. import gate
        new = [f"Trail bike {i}" for i in range(3)]
        everything = ALL + new
        state = state_with()
        probed = []

        def fake_probe(exe, names, args, allowance, per_probe):
            probed.append((list(names), allowance, per_probe))
            return {}, {n: per_probe for n in names}, per_probe * len(names)

        original = gate.probe_unknown
        gate.probe_unknown = fake_probe
        try:
            args = type("A", (), {"level": "quick", "budget_seconds": 90.0, "rounds": 3, "epochs": 0})()
            chosen, rounds, spent = gate.choose_benchmarks(args, "exe", "t", state, "c1", everything)
        finally:
            gate.probe_unknown = original
        self.assertEqual(probed[0][0], new)  # only the unknown ones, within 20% of the budget, 10% each
        self.assertEqual(probed[0][1:], (18.0, 9.0))
        self.assertFalse(set(new) & set(chosen))
        self.assertEqual(set(state["slow"]), set(new))
        self.assertAlmostEqual(spent, 27.0)
