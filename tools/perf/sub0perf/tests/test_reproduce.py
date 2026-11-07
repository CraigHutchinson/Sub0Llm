"""Tests for reproduce: a difference from an earlier record counts only if it shows again side by side."""

import types
import unittest

from .. import gate, reproduce
from .helpers import record

ARGS = types.SimpleNamespace(tolerance=0.10, pair_tolerance=0.05, max_cv=0.05, absolute=False, instr_tolerance=0.02,
                             epochs=0, max_load=0.2, settle=0.0, seed=1)
TIMES = {"G, hand-written": 1000.0, "G, vocab": 1000.0, "H, vocab": 700.0}


def measuring(now_times, ref_times):
    """A stand-in for side_by_side that returns synthetic records of the two builds."""
    return lambda exe_now, exe_ref, names, rounds, args: (record({n: now_times[n] for n in names}, seed=3),
                                                         record({n: ref_times[n] for n in names}, seed=4))


def listing(names):
    return lambda exe, text: list(names)


class ReproduceTest(unittest.TestCase):
    def run_it(self, now, ref, names=("G, vocab",), have=TIMES, **kw):
        return reproduce.reproduced_against("new", "old", list(names), ARGS, measure=measuring(now, ref), listing=listing(have), **kw)

    def test_identical_builds_do_not_reproduce(self):
        again, gone, why = self.run_it(TIMES, TIMES)
        self.assertEqual((again, gone, why), (set(), {"G, vocab"}, None))

    def test_a_real_slowdown_reproduces(self):
        again, gone, why = self.run_it({**TIMES, "G, vocab": 1400.0}, TIMES)
        self.assertEqual((again, gone), ({"G, vocab"}, set()))

    def test_the_pair_with_its_reference_is_measured_together(self):
        asked = []

        def measure(exe_now, exe_ref, names, rounds, args):
            asked.append(list(names))
            return record({n: TIMES[n] for n in names}), record({n: TIMES[n] for n in names})
        reproduce.reproduced_against("new", "old", ["G, vocab"], ARGS, measure=measure, listing=listing(TIMES))
        self.assertEqual(asked, [["G, hand-written", "G, vocab"]])

    def test_a_benchmark_the_old_build_lacks_cannot_reproduce(self):
        again, gone, why = self.run_it({**TIMES, "G, vocab": 1400.0}, TIMES, have=["H, vocab"])
        self.assertEqual((again, gone), (set(), {"G, vocab"}))
        self.assertIn("none of these", why)

    def test_a_build_that_cannot_run_is_not_a_verdict(self):
        def broken(*a):
            raise OSError("no such library")
        again, gone, why = reproduce.reproduced_against("new", "old", ["G, vocab"], ARGS, measure=broken, listing=listing(TIMES))
        self.assertEqual((again, gone), (set(), {"G, vocab"}))
        self.assertIn("could not be measured", why)

    def test_rounds_alternate_which_build_goes_first(self):
        order = []
        real = reproduce.run_rounds
        reproduce.run_rounds = lambda exe, *a, **k: order.append(exe)
        try:
            reproduce.side_by_side("new", "old", ["a"], 4, ARGS)
        finally:
            reproduce.run_rounds = real
        self.assertEqual(order, ["new", "old", "old", "new", "new", "old", "old", "new"])


class GateUsesItTest(unittest.TestCase):
    def test_without_the_kept_build_nothing_blocks(self):
        real = gate.reference_binary
        gate.reference_binary = lambda *a: None
        try:
            self.assertEqual(gate.reproduce_vs_record("repo", "t", "exe", "a" * 40, ["X", "Y"], ARGS), ([], ["X", "Y"]))
        finally:
            gate.reference_binary = real

    def test_with_it_only_what_reproduces_blocks(self):
        real_bin, real_rep = gate.reference_binary, gate.reproduced_against
        gate.reference_binary = lambda *a: "old.exe"
        gate.reproduced_against = lambda exe, old, names, args: ({"Y"}, {"X"}, None)
        try:
            self.assertEqual(gate.reproduce_vs_record("repo", "t", "exe", "a" * 40, ["X", "Y"], ARGS), (["Y"], ["X"]))
        finally:
            gate.reference_binary, gate.reproduced_against = real_bin, real_rep


if __name__ == "__main__":
    unittest.main()
