"""Tests for levels: which level and budget a gate runs with."""

import types
import unittest

from .. import levels

PATTERNS = levels.DEFAULT_FULL_REFS
ZERO = "0" * 40
SHA = "a" * 40


def push(ref):
    return (ref, SHA, ref, ZERO)


class SelectLevelTest(unittest.TestCase):
    def test_default_is_quick(self):
        self.assertEqual(levels.select_level(None, None, [push("refs/heads/main")], PATTERNS)[0], "quick")

    def test_milestone_tag_selects_full(self):
        for tag in ("refs/tags/v1.2.0", "refs/tags/milestone/M2"):
            level, why = levels.select_level(None, None, [push("refs/heads/main"), push(tag)], PATTERNS)
            self.assertEqual(level, "full", tag)
            self.assertIn(tag.split("refs/tags/")[1], why)

    def test_other_tags_and_branches_do_not(self):
        for ref in ("refs/tags/nightly", "refs/heads/v1", "refs/heads/milestone/x"):
            self.assertEqual(levels.select_level(None, None, [push(ref)], PATTERNS)[0], "quick", ref)

    def test_deleting_a_milestone_tag_does_not(self):
        deleted = ("refs/tags/v1", ZERO, "refs/tags/v1", SHA)
        self.assertEqual(levels.select_level(None, None, [deleted], PATTERNS)[0], "quick")

    def test_patterns_are_configurable(self):
        refs = [push("refs/tags/rel-3")]
        self.assertEqual(levels.select_level(None, None, refs, PATTERNS)[0], "quick")
        self.assertEqual(levels.select_level(None, None, refs, levels.parse_patterns("rel-*;v*"))[0], "full")
        self.assertEqual(levels.full_ref_patterns({"full_refs": ["rel-*"]}), ("rel-*",))
        self.assertEqual(levels.full_ref_patterns({}), PATTERNS)

    def test_precedence_cli_over_env_over_refs_over_default(self):
        tag = [push("refs/tags/v1")]
        self.assertEqual(levels.select_level("quick", "full", tag, PATTERNS)[0], "quick")   # cli beats env and refs
        self.assertEqual(levels.select_level(None, "quick", tag, PATTERNS)[0], "quick")     # env beats refs
        self.assertEqual(levels.select_level(None, "full", [], PATTERNS)[0], "full")        # env beats default
        self.assertEqual(levels.select_level(None, None, tag, PATTERNS)[0], "full")         # refs beat default
        self.assertEqual(levels.select_level("full", "quick", [], PATTERNS)[0], "full")
        self.assertIn("--level", levels.select_level("full", None, [], PATTERNS)[1])
        self.assertIn("SUB0MATH_PERF_LEVEL", levels.select_level(None, "full", [], PATTERNS)[1])

    def test_unknown_level_is_an_error(self):
        with self.assertRaises(ValueError):
            levels.select_level(None, "thorough", [], PATTERNS)


class SelectBudgetTest(unittest.TestCase):
    def test_quick_defaults_and_overrides(self):
        self.assertEqual(levels.select_budget("quick", None, None)[0], 90.0)
        self.assertEqual(levels.select_budget("quick", None, None, 60)[0], 60.0)
        self.assertEqual(levels.select_budget("quick", None, "45", 60)[0], 45.0)
        self.assertEqual(levels.select_budget("quick", 30.0, "45", 60)[0], 30.0)
        self.assertIsNone(levels.select_budget("quick", 0.0, None)[0])

    def test_full_has_none_unless_given(self):
        self.assertIsNone(levels.select_budget("full", None, None, 60)[0])
        self.assertEqual(levels.select_budget("full", 600.0, None)[0], 600.0)
        self.assertEqual(levels.select_budget("full", None, "300")[0], 300.0)


class ResolveTest(unittest.TestCase):
    def test_sets_the_attributes(self):
        args = types.SimpleNamespace(level=None, budget=None)
        levels.resolve(args, [push("refs/tags/v2")], {"quick_budget": 50}, environ={})
        self.assertEqual((args.level, args.budget_seconds), ("full", None))
        args = types.SimpleNamespace(level=None, budget=None)
        levels.resolve(args, [], {"quick_budget": 50}, environ={"SUB0MATH_PERF_LEVEL": "quick"})
        self.assertEqual((args.level, args.budget_seconds), ("quick", 50.0))


if __name__ == "__main__":
    unittest.main()
