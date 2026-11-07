"""Tests for confirm."""

import unittest

from ..baseline import baseline_data
from .. import confirm, judge, runner, stats
from .helpers import record, ROUNDS, NAMES


class ConfirmationTest(unittest.TestCase):
    def setUp(self):
        self.base = {n: 1000.0 + 50 * i for i, n in enumerate(NAMES)}
        self.baseline = baseline_data(record(self.base))
        self.now = dict(self.base)
        self.now["Bench 1"] *= 1.50  # looks regressed on the first pass
        self.now["Bench 4"] *= 1.50
        self.rec = record(self.now)

    def evaluate_all(self):
        return [judge.evaluate(self.rec, self.baseline, 0.10, 0.05, 0.05, False)]

    def stub(self, truth):
        """A run function that appends samples from `truth` (name -> ns) for the requested names only."""
        calls = []

        def run(names, rounds, first_round, samples, meta, positions, counters):
            calls.append((list(names), first_round))
            for n in names:
                samples[n].append(truth[n])
                positions[n].append(0.5)
            meta.append({"round": first_round, "load": 0.0, "waited_s": 0.0, "quiet": True,
                         "calibration_start_ns": 1000.0, "calibration_end_ns": 1000.0})
        return run, calls

    def test_extend_rounds_keeps_alignment(self):
        run, calls = self.stub(self.base)
        rec = self.rec
        runner.extend_rounds(run, {"Bench 1"}, 2, rec["samples"], rec["rounds"], rec["positions"], rec["counters"])
        self.assertEqual(len(rec["rounds"]), ROUNDS + 2)
        self.assertEqual([c[1] for c in calls], [ROUNDS, ROUNDS + 1])
        self.assertEqual(calls[0][0], ["Bench 1"])
        for n, v in rec["samples"].items():
            self.assertEqual(len(v), ROUNDS + 2, n)
            self.assertEqual(len(rec["positions"][n]), ROUNDS + 2, n)

    def test_noise_is_not_confirmed(self):
        run, calls = self.stub(self.base)  # the re-measurement shows no regression
        rec = self.rec
        first, final = confirm.confirm_regressions(
            self.evaluate_all,
            lambda names, n: runner.extend_rounds(run, names, n, rec["samples"], rec["rounds"], rec["positions"], rec["counters"]), 8)
        self.assertEqual(first, {"Bench 1", "Bench 4"})
        self.assertEqual(final, set())
        self.assertEqual({n for c in calls for n in c[0]}, {"Bench 1", "Bench 4"})

    def test_real_regression_is_confirmed(self):
        run, _ = self.stub(self.now)
        rec = self.rec
        first, final = confirm.confirm_regressions(
            self.evaluate_all,
            lambda names, n: runner.extend_rounds(run, names, n, rec["samples"], rec["rounds"], rec["positions"], rec["counters"]), 4)
        self.assertEqual(final, {"Bench 1", "Bench 4"})

    def test_nothing_regressed_measures_nothing(self):
        self.rec = record(self.base)
        first, final = confirm.confirm_regressions(self.evaluate_all, lambda *a: self.fail("re-measured"), 4)
        self.assertEqual((first, final), (set(), set()))

    def test_pair_remeasures_its_reference(self):
        base = {"G, hand-written": 1000.0, "G, vocab": 1000.0, "H, hand-written": 500.0, "H, vocab": 500.0}
        baseline = baseline_data(record(base))
        now = dict(base)
        now["G, vocab"] *= 1.30
        found = confirm.regressions_in([judge.evaluate(record(now), baseline, 0.10, 0.05, 0.05, False)])
        self.assertEqual(found[0], {"G, vocab"})
        self.assertEqual(found[1], {"G, vocab", "G, hand-written"})


class ModesRoundsTest(unittest.TestCase):
    """The confirmation pass gathers enough rounds for the two-level test before anything blocks."""

    def setUp(self):
        self.baseline = baseline_data(record({"G, hand-written": 1000.0, "G, vocab": 1000.0}))
        self.rec = record({"G, hand-written": 1000.0, "G, vocab": 1000.0}, jitter=0.002)
        for key in ("rounds",):
            self.rec[key] = self.rec[key][:3]
        for n in self.rec["samples"]:
            self.rec["samples"][n] = self.rec["samples"][n][:3]
            self.rec["positions"][n] = self.rec["positions"][n][:3]
        self.rec["samples"]["G, vocab"] = [1150.0, 1152.0, 1000.0]  # the slow level has the median: looks +15%
        self.requested = []

    def evaluate_all(self):
        return [judge.evaluate(self.rec, self.baseline, 0.10, 0.05, 0.05, False)]

    def rerun(self, names, n):
        """Adds n rounds: the benchmark alternates between its two levels, the reference is steady."""
        self.requested.append((set(names), n))
        for _ in range(n):
            k = len(self.rec["rounds"])
            self.rec["rounds"].append({"round": k, "load": 0.0, "waited_s": 0.0, "quiet": True,
                                       "calibration_start_ns": 1000.0, "calibration_end_ns": 1000.0})
            for name in self.rec["samples"]:
                self.rec["samples"][name].append((1000.0 if k % 2 else 1151.0) if name == "G, vocab" else 1000.0)
                self.rec["positions"][name].append(0.5)

    def test_rounds_needed_counts_measured_samples_only(self):
        self.assertEqual(stats.rounds_needed(self.rec, ["G, vocab"]), stats.MODES_MIN_SAMPLES - 3)
        self.rec["positions"]["G, vocab"][2] = None  # a filled-in copy is not a measurement
        self.assertEqual(stats.rounds_needed(self.rec, ["G, vocab", "G, hand-written"]), stats.MODES_MIN_SAMPLES - 2)
        self.assertEqual(stats.rounds_needed(self.rec, ["no such"]), 0)

    def test_looks_regressed_on_the_first_pass(self):
        self.assertEqual(confirm.regressions_in(self.evaluate_all())[0], {"G, vocab"})

    def test_confirmation_runs_long_enough_for_the_modes_test_and_the_bimodal_benchmark_does_not_block(self):
        needed = lambda names: stats.rounds_needed(self.rec, names)
        first, final = confirm.confirm_regressions(self.evaluate_all, self.rerun, 4, needed)
        self.assertEqual(first, {"G, vocab"})
        self.assertEqual(self.requested[0][1], stats.MODES_MIN_SAMPLES - 3)  # not just the 4 that were asked for
        self.assertEqual(final, set())

    def test_a_longer_request_is_kept(self):
        confirm.confirm_regressions(self.evaluate_all, self.rerun, 20, lambda names: 2)
        self.assertEqual(self.requested[0][1], 20)

    def test_retrying_the_inconclusive_also_reaches_the_minimum(self):
        self.rec["samples"]["G, vocab"] = [1000.0, 1300.0, 900.0]  # noisy
        confirm.retry_inconclusive(self.evaluate_all, self.rerun, 3, lambda names: stats.rounds_needed(self.rec, names))
        self.assertEqual(self.requested[0][1], stats.MODES_MIN_SAMPLES - 3)


class InconclusiveTest(unittest.TestCase):
    def setUp(self):
        self.base = {"G, hand-written": 1000.0, "G, vocab": 1000.0, "H, hand-written": 500.0, "H, vocab": 500.0}
        self.baseline = baseline_data(record(self.base))

    def noisy_record(self):
        rec = record(self.base)
        rec["samples"]["H, vocab"] = [500.0, 800.0, 300.0, 650.0, 420.0]  # far above max_cv
        return rec

    def evaluations(self, rec):
        return [judge.evaluate(rec, self.baseline, 0.10, 0.05, 0.05, False)]

    def test_noisy_benchmark_and_its_group_are_found(self):
        names, remeasure = confirm.inconclusive_in(self.evaluations(self.noisy_record()))
        self.assertIn("H, vocab", names)
        self.assertEqual(remeasure, {"H, vocab", "H, hand-written"})

    def test_clean_run_is_conclusive(self):
        self.assertEqual(confirm.inconclusive_in(self.evaluations(record(self.base))), (set(), set()))

    def test_retry_remeasures_once_and_reports_what_stays(self):
        rec = self.noisy_record()
        calls = []
        first, final = confirm.retry_inconclusive(lambda: self.evaluations(rec), lambda names, n: calls.append((set(names), n)), 3)
        self.assertEqual(calls, [({"H, vocab", "H, hand-written"}, 3)])
        self.assertIn("H, vocab", first)
        self.assertEqual(final, first)  # the stub added nothing, so it is still inconclusive

    def test_retry_that_cures_the_noise_reports_none(self):
        rec = self.noisy_record()

        def fix(names, n):
            rec["samples"]["H, vocab"] = [500.0] * ROUNDS
        first, final = confirm.retry_inconclusive(lambda: self.evaluations(rec), fix, 3)
        self.assertTrue(first)
        self.assertEqual(final, set())

    def test_conclusive_run_is_not_remeasured(self):
        confirm.retry_inconclusive(lambda: self.evaluations(record(self.base)), lambda *a: self.fail("re-measured"), 3)

    def test_method_mismatch_is_reported_but_not_retried(self):
        rec = record(self.base)
        rec["cpu"] = 3
        baseline = baseline_data(record(self.base))
        baseline["cpu"] = 5
        rec["samples"]["G, vocab"] = [1300.0] * ROUNDS
        evaluations = [judge.evaluate(rec, baseline, 0.10, 0.05, 0.05, False)]
        names, remeasure = confirm.inconclusive_in(evaluations)
        self.assertEqual(names, {"G, vocab"})
        self.assertEqual(remeasure, set())

