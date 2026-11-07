"""Tests for stats: robust spread, bimodal detection, measured counts."""

import random
import unittest

from ..stats import measured_count, modes, robust_cv
from .helpers import record


class ModesTest(unittest.TestCase):
    def test_two_levels_are_found(self):
        low, high, gap = modes([1000, 1010, 995, 1080, 1090, 1075, 1005])
        self.assertAlmostEqual(low, 1000, delta=10)
        self.assertAlmostEqual(high, 1080, delta=10)
        self.assertGreater(gap, 0.05)

    def test_one_level_is_not_split(self):
        self.assertIsNone(modes([1000, 1004, 996, 1010, 990, 1002, 998]))

    def test_one_outlier_is_not_a_level(self):
        self.assertIsNone(modes([1000, 1004, 996, 1010, 990, 1002, 1300]))  # a level needs two samples

    def test_too_few_samples_say_nothing(self):
        self.assertIsNone(modes([1000, 1000, 1100, 1100]))

    def test_gaussian_noise_is_rarely_taken_for_two_levels(self):
        rng = random.Random(3)
        hits = sum(modes([1000 * (1 + rng.gauss(0, 0.01)) for _ in range(7)]) is not None for _ in range(2000))
        self.assertLess(hits / 2000, 0.01)

    def test_summarise_reports_modes_of_measured_rounds_only(self):
        rec = record({"A": 1000.0}, jitter=0.0)
        rec["samples"]["A"] = [1000.0, 1000.0, 1080.0, 1080.0, 1000.0]
        from ..stats import summarise
        self.assertIsNotNone(summarise(rec)[0]["A"]["modes"])
        rec["positions"]["A"] = [0.5, 0.5, None, None, None]  # three filled-in copies: only two are measured
        self.assertIsNone(summarise(rec)[0]["A"]["modes"])


class MeasuredCountTest(unittest.TestCase):
    def test_counts_unfilled_samples(self):
        rec = record({"A": 1000.0})
        self.assertEqual(measured_count(rec, "A"), 5)
        rec["positions"]["A"][1] = None
        self.assertEqual(measured_count(rec, "A"), 4)
        self.assertEqual(measured_count(rec, "absent"), 0)

    def test_robust_cv_of_constant_is_zero(self):
        self.assertEqual(robust_cv([5.0, 5.0, 5.0]), 0.0)


if __name__ == "__main__":
    unittest.main()
