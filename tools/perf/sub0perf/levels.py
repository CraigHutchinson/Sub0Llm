"""Gate levels: `quick` for everyday runs, `full` for milestones; which one applies, and the time budget that goes with it."""

import fnmatch
import os

QUICK, FULL = "quick", "full"
LEVELS = (QUICK, FULL)
DEFAULT_FULL_REFS = ("v*", "milestone/*")  # tag names whose push selects the full level
DEFAULT_QUICK_BUDGET = 90.0                # seconds of measuring for a quick gate
TAGS = "refs/tags/"


def parse_patterns(text):
    """A comma- or semicolon-separated pattern list (CMake lists use ';') as a tuple; empty entries are dropped."""
    return tuple(p.strip() for p in str(text).replace(";", ",").split(",") if p.strip())


def full_ref_patterns(config=None):
    """The tag-name patterns that select full: the repository config's `full_refs`, else the defaults."""
    found = (config or {}).get("full_refs")
    return tuple(found) if found else DEFAULT_FULL_REFS


def milestone_refs(refs, patterns):
    """The pushed tags (names) that match a pattern. `refs` are (local ref, local sha, remote ref, remote sha); a delete is not a push."""
    names = []
    for local_ref, local_sha, *_ in refs:
        if set(local_sha) == {"0"} or not local_ref.startswith(TAGS):
            continue
        tag = local_ref[len(TAGS):]
        if any(fnmatch.fnmatchcase(tag, p) for p in patterns):
            names.append(tag)
    return names


def select_level(cli_level, env_level, refs, patterns):
    """(level, why). Precedence: --level, then SUB0MATH_PERF_LEVEL, then a pushed milestone tag (full), then quick."""
    for source, value in (("--level", cli_level), ("SUB0MATH_PERF_LEVEL", env_level)):
        if value:
            value = value.strip().lower()
            if value not in LEVELS:
                raise ValueError(f"{source}={value!r}: the level is one of {', '.join(LEVELS)}")
            return value, f"{source}={value}"
    tags = milestone_refs(refs, patterns)
    if tags:
        return FULL, f"the push includes milestone tag {', '.join(tags)} (patterns: {', '.join(patterns)})"
    return QUICK, "the default for everyday pushes"


def select_budget(level, cli_budget, env_budget, config_budget=None):
    """(seconds or None, why). A quick gate has a budget (--budget, SUB0MATH_PERF_BUDGET, the config's, else 90 s); full has
    none unless one is given with --budget or SUB0MATH_PERF_BUDGET. 0 or less means no budget."""
    for source, value in (("--budget", cli_budget), ("SUB0MATH_PERF_BUDGET", env_budget)):
        if value not in (None, ""):
            seconds = float(value)
            return (seconds if seconds > 0 else None), f"{source}={seconds:g}"
    if level == QUICK:
        if config_budget not in (None, ""):
            seconds = float(config_budget)
            return (seconds if seconds > 0 else None), f"configured quick budget {seconds:g} s"
        return DEFAULT_QUICK_BUDGET, f"default quick budget {DEFAULT_QUICK_BUDGET:g} s"
    return None, "full has no budget"


def resolve(args, refs, config=None, environ=None):
    """Sets args.level, args.level_why, args.budget_seconds and args.budget_why from the command line, environment, refs and config."""
    environ = os.environ if environ is None else environ
    level, why = select_level(getattr(args, "level", None), environ.get("SUB0MATH_PERF_LEVEL"), refs, full_ref_patterns(config))
    budget, budget_why = select_budget(level, getattr(args, "budget", None), environ.get("SUB0MATH_PERF_BUDGET"),
                                       (config or {}).get("quick_budget"))
    args.level, args.level_why, args.budget_seconds, args.budget_why = level, why, budget, budget_why
    return level, budget
