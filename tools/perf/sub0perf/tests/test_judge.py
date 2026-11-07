"""Tests for judge."""

import unittest

from ..baseline import baseline_data
from .. import judge, report
from .. import stats as stats_mod
from .helpers import record, judge_against, NAMES


class JudgingTest(unittest.TestCase):
    def test_single_shift_with_controls_regresses(self):
        base = {f"Group {i}, vocabulary": 1000.0 + 100 * i for i in range(6)}
        base["Group 0, hand-written"] = 900.0
        now = dict(base)
        now["Group 3, vocabulary"] *= 1.30
        verdicts, _, env = judge_against(base, now)
        self.assertEqual(verdicts["Group 3, vocabulary"], "REGRESSED")
        self.assertEqual(sum(v == "REGRESSED" for v in verdicts.values()), 1)
        self.assertIsNone(env["suite_spread"])

    def test_large_environment_shift_makes_absolute_verdicts_inconclusive(self):
        base = {f"Group {i}, vocabulary": 1000.0 + 100 * i for i in range(4)}
        base.update({f"Group {i}, hand-written": 900.0 + 100 * i for i in range(4)})
        now = {n: t * 1.40 for n, t in base.items()}  # everything, controls included, 40% slower
        now["Group 2, vocabulary"] *= 1.15  # plus a residual that would otherwise count
        verdicts, pairs, env = judge_against(base, now)
        self.assertGreater(env["environment_shift"], 0.10)
        self.assertNotIn("REGRESSED", verdicts.values())
        self.assertEqual(verdicts["Group 2, vocabulary"], "ENVIRONMENT")
        # The pair ratio for group 2 is relative to the same environment and still judges.
        self.assertEqual({p[0]: p[4] for p in pairs}["Group 2, vocabulary"], "REGRESSED")

    def test_pair_ratio_ignores_filled_rounds(self):
        rec = record({"G, vocabulary": 1000.0, "G, hand-written": 1000.0}, jitter=0.0)
        # Two extra rounds in a 30% slower environment: only the vocabulary member was re-measured,
        # the reference got its median copied in (as alignment filling does).
        for _ in range(2):
            rec["rounds"].append(dict(rec["rounds"][0]))
            rec["samples"]["G, vocabulary"].append(1300.0)
            rec["positions"]["G, vocabulary"].append(0.5)
            rec["samples"]["G, hand-written"].append(1000.0)
            rec["positions"]["G, hand-written"].append(None)
        stats, _, _, _ = stats_mod.summarise(rec)
        pairs = stats_mod.pair_ratios(stats)
        self.assertAlmostEqual(pairs["G, vocabulary"]["ratio"], 1.0, places=6)

    def test_random_shifts_without_controls_do_not_block(self):
        shifts = [-0.15, 0.15, 0.12, -0.14, 0.15, -0.10, 0.13, -0.12]  # as if random, +-15%
        base = {n: 1000.0 + 50 * i for i, n in enumerate(NAMES)}
        now = {n: t * (1.0 + d) for (n, t), d in zip(base.items(), shifts)}
        verdicts, _, env = judge_against(base, now)
        self.assertGreater(env["suite_spread"], 0.05)
        self.assertNotIn("REGRESSED", verdicts.values())
        # The same data with the floor disabled would have blocked something.
        self.assertGreater(max(now[n] / base[n] - 1.0 for n in base), 0.10)

    def test_genuine_shift_without_controls_regresses(self):
        base = {n: 1000.0 + 50 * i for i, n in enumerate(NAMES)}
        now = dict(base)
        now["Bench 2"] *= 1.40
        verdicts, _, env = judge_against(base, now)
        self.assertEqual(verdicts["Bench 2"], "REGRESSED")
        self.assertEqual(sum(v == "REGRESSED" for v in verdicts.values()), 1)
        self.assertIsNotNone(env["suite_spread"])
        self.assertIn("No controls", report.render(*judge.evaluate(record(now), baseline_data(record(base)), 0.10, 0.05, 0.05, False),
                                              "baseline", 0.10, 0.05))


class BimodalTest(unittest.TestCase):
    """A benchmark that runs at one of two levels from process to process is inconclusive, not regressed."""

    def setUp(self):
        self.base = {"G, hand-written": 1000.0, "G, vocab": 1000.0, "H, hand-written": 500.0, "H, vocab": 500.0}
        self.baseline = baseline_data(record(self.base))

    def judge(self, vocab_samples):
        rec = record(self.base, jitter=0.002)
        rec["samples"]["G, vocab"] = vocab_samples
        rows, pairs, env = judge.evaluate(rec, self.baseline, 0.10, 0.05, 0.05, False)
        return {r[0]: r[5] for r in rows}["G, vocab"], {p[0]: p[4] for p in pairs}["G, vocab"]

    def test_one_level_moving_is_noisy_not_regressed(self):
        # Baseline 1000. Samples at 1000 and 1150 (five rounds): the median is in the slow level (+15%) but the fast level is the baseline.
        verdict, _ = self.judge([1000.0, 1002.0, 1150.0, 1155.0, 1152.0])
        self.assertEqual(verdict, "NOISY")

    def test_both_levels_moving_is_still_a_regression(self):
        verdict, _ = self.judge([1300.0, 1302.0, 1450.0, 1455.0, 1452.0])
        self.assertEqual(verdict, "REGRESSED")

    def test_one_level_faster_is_not_an_improvement(self):
        verdict, _ = self.judge([1000.0, 1002.0, 850.0, 852.0, 848.0])
        self.assertEqual(verdict, "NOISY")

    def test_a_steady_regression_is_unaffected(self):
        verdict, pair = self.judge([1300.0, 1305.0, 1297.0, 1302.0, 1299.0])
        self.assertEqual((verdict, pair), ("REGRESSED", "REGRESSED"))

    def test_the_pair_ratio_of_a_bimodal_benchmark_is_noisy(self):
        _, pair = self.judge([1000.0, 1002.0, 1150.0, 1155.0, 1152.0])
        self.assertEqual(pair, "NOISY")

    def test_report_marks_bimodal_rows(self):
        rec = record(self.base, jitter=0.002)
        rec["samples"]["G, vocab"] = [1000.0, 1002.0, 1150.0, 1155.0, 1152.0]
        rows, pairs, env = judge.evaluate(rec, self.baseline, 0.10, 0.05, 0.05, False)
        self.assertIn("bimodal", report.render(rows, pairs, env, "b.json", 0.10, 0.05))


class BimodalGateTest(unittest.TestCase):
    """What the gate saw: a bimodal benchmark (or its bimodal control) read as a regression, with the label saying bimodal."""

    LOW = [1000.0, 1001.0, 1002.0, 1003.0]
    HIGH = [1180.0, 1182.0, 1181.0]

    def evaluate(self, base_rec, now_rec):
        return judge.evaluate(now_rec, baseline_data(base_rec), 0.10, 0.05, 0.05, False)

    def test_bimodal_control_is_not_used_to_correct_the_environment(self):
        # The control sat at its high level when the baseline was made and at its low level now: -15%, which would make
        # the steady library benchmark look 18% slower once "corrected".
        base = record({"G, hand-written": 1180.0, "G, vocab": 1000.0, "H, vocab": 700.0}, jitter=0.002)
        base["samples"]["G, hand-written"] = [1180.0, 1182.0, 1181.0, 1000.0, 1001.0]
        now = record({"G, hand-written": 1000.0, "G, vocab": 1000.0, "H, vocab": 700.0}, jitter=0.002)
        now["samples"]["G, hand-written"] = self.LOW + self.HIGH + [1000.0, 1002.0]
        for k in ("positions",):
            now[k]["G, hand-written"] = [0.5] * 9
        now["rounds"] = now["rounds"] + [dict(now["rounds"][0]) for _ in range(4)]
        for n in ("G, vocab", "H, vocab"):
            now["samples"][n] = now["samples"][n] + [now["samples"][n][0]] * 4
            now["positions"][n] = [0.5] * 9
        rows, _, env = self.evaluate(base, now)
        self.assertEqual({r[0]: r[5] for r in rows}["G, vocab"], "ok")
        self.assertIsNone(env["environment_shift"])

    def test_steady_control_is_still_used(self):
        base = record({"G, hand-written": 1000.0, "G, vocab": 1000.0, "H, vocab": 700.0}, jitter=0.002)
        now = record({"G, hand-written": 1050.0, "G, vocab": 1050.0, "H, vocab": 700.0}, jitter=0.002)
        rows, _, env = self.evaluate(base, now)
        self.assertAlmostEqual(env["environment_shift"], 0.05, places=2)

    def test_baseline_remembers_its_two_levels(self):
        rec = record({"G, vocab": 1000.0}, jitter=0.002)
        rec["samples"]["G, vocab"] = [1000.0, 1002.0, 1150.0, 1155.0, 1152.0]
        entry = baseline_data(rec)["benchmarks"]["G, vocab"]
        self.assertEqual(len(entry["modes"]), 2)
        self.assertLess(entry["modes"][0], entry["modes"][1])
        steady = baseline_data(record({"G, vocab": 1000.0}))["benchmarks"]["G, vocab"]
        self.assertNotIn("modes", steady)

    def baseline_with_levels(self):
        base = record({"G, hand-written": 1000.0, "G, vocab": 1000.0, "H, hand-written": 500.0, "H, vocab": 500.0}, jitter=0.002)
        base["samples"]["G, vocab"] = [1000.0, 1002.0, 1150.0, 1155.0, 1152.0]  # median in the slow level
        return base

    def judge_now(self, vocab_samples):
        now = record({"G, hand-written": 1000.0, "G, vocab": 1000.0, "H, hand-written": 500.0, "H, vocab": 500.0}, jitter=0.002)
        now["samples"]["G, vocab"] = vocab_samples
        rows, pairs, _ = self.evaluate(self.baseline_with_levels(), now)
        return {r[0]: r[5] for r in rows}["G, vocab"]

    def test_steady_value_inside_the_baselines_two_levels_is_not_a_change(self):
        # The baseline median is ~1152; a steady run at the fast level is -13%, at the slow level 0.
        self.assertNotIn(self.judge_now([1000.0, 1001.0, 1002.0, 1000.0, 1001.0]), ("IMPROVED", "REGRESSED"))

    def test_steady_value_above_both_baseline_levels_is_a_regression(self):
        self.assertEqual(self.judge_now([1400.0, 1402.0, 1399.0, 1401.0, 1400.0]), "REGRESSED")

    def test_bimodal_value_above_both_levels_is_a_regression_and_says_so(self):
        now = record({"G, hand-written": 1000.0, "G, vocab": 1000.0, "H, hand-written": 500.0, "H, vocab": 500.0}, jitter=0.002)
        now["samples"]["G, vocab"] = [1400.0, 1402.0, 1600.0, 1605.0, 1602.0]
        rows, pairs, env = self.evaluate(self.baseline_with_levels(), now)
        self.assertEqual({r[0]: r[5] for r in rows}["G, vocab"], "REGRESSED")
        self.assertIn("bimodal, both levels moved", report.render(rows, pairs, env, "b.json", 0.10, 0.05))

    def test_a_shift_must_survive_every_reference(self):
        # Against 1000 the slow level (1150) is a regression; against 1100 (the other scaling) the fast level is not.
        self.assertEqual(judge.judge_modes("REGRESSED", (1100.0, 1300.0, 0.2), [1000.0, 1150.0], 0.10), "NOISY")
        self.assertEqual(judge.judge_modes("REGRESSED", (1300.0, 1400.0, 0.1), [1000.0, 1150.0], 0.10), "REGRESSED")

    def test_a_bimodal_reference_range_is_used(self):
        self.assertEqual(judge.judge_modes("REGRESSED", (1150.0, 1150.0, 0.0), (1000.0, 1150.0), 0.10), "NOISY")
        self.assertEqual(judge.judge_modes("IMPROVED", (1000.0, 1000.0, 0.0), (1000.0, 1150.0), 0.10), "NOISY")

    def test_other_verdicts_pass_through(self):
        self.assertEqual(judge.judge_modes("ok", (1.0, 2.0, 1.0), 1.0, 0.1), "ok")
        self.assertEqual(judge.judge_modes("REGRESSED", None, 1.0, 0.1), "REGRESSED")


class MethodMismatchTest(unittest.TestCase):
    def method(self, record_cpu, baseline_cpu):
        times = {"G, hand-written": 1000.0, "G, vocab": 1000.0}
        rec = record(times)
        base = baseline_data(record(times))
        if record_cpu != "absent":
            rec["cpu"] = record_cpu
        if baseline_cpu == "absent":
            del base["cpu"]
        else:
            base["cpu"] = baseline_cpu
        return judge.evaluate(rec, base, 0.10, 0.05, 0.05, False)[2]["method"]

    def test_unpinned_baseline_against_pinned_run_is_flagged(self):
        note = self.method(7, None)
        self.assertIn("pinned CPU differs", note)
        self.assertIn("baseline unpinned", note)

    def test_pinned_baseline_against_unpinned_run_is_flagged(self):
        self.assertIn("now unpinned", self.method(None, 7))

    def test_different_cpus_are_flagged_and_equal_ones_are_not(self):
        self.assertIn("pinned CPU differs", self.method(1, 2))
        self.assertIsNone(self.method(2, 2))
        self.assertIsNone(self.method(None, None))

    def test_unknown_cpu_is_not_a_mismatch(self):
        self.assertIsNone(self.method("absent", 3))
        self.assertIsNone(self.method(3, "absent"))

