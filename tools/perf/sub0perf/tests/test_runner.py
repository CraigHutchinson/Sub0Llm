"""Tests for runner: steady calibration readings at the edges of a round."""

import unittest

from .. import runner


class SteadyCalibrationTest(unittest.TestCase):
    def test_discards_cold_readings_until_two_agree(self):
        readings = iter([9030.0, 6100.0, 4910.0, 4890.0, 4880.0])
        self.assertEqual(runner.steady_calibration(lambda: next(readings)), 4890.0)

    def test_gives_up_after_the_attempt_limit(self):
        readings = iter([1000.0 * (i + 1) for i in range(20)])
        self.assertEqual(runner.steady_calibration(lambda: next(readings)), 1000.0 * runner.CALIBRATION_ATTEMPTS)
