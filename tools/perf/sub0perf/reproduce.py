"""Reproducing a verdict against an earlier record: the earlier build is measured again beside the new one.

A local record was measured in an earlier session, so a difference from it can be the session (clock and power state,
other load, which of its two levels a bimodal benchmark landed on) and not the code. Only a difference that survives
measuring both builds again, alternately, in this session is evidence about the code.
"""

from .baseline import baseline_data, make_record
from .common import group_of
from .judge import evaluate_with, names_with
from .runner import list_names, run_rounds
from .stats import MODES_MIN_SAMPLES


def groups_of(names, available):
    """Every name of `available` in the groups of `names`, so pairs are re-measured with their references."""
    wanted = {group_of(n) for n in names}
    return sorted(n for n in available if group_of(n) in wanted)


def side_by_side(exe_now, exe_ref, names, rounds, args):
    """Measures `names` with both executables, alternating which goes first each round: (now record, reference record)."""
    sides = [({n: [] for n in names}, [], {n: [] for n in names}, {}) for _ in range(2)]
    for r in range(rounds):
        for which in ((0, 1) if r % 2 == 0 else (1, 0)):
            exe = (exe_now, exe_ref)[which]
            samples, meta, positions, counters = sides[which]
            run_rounds(exe, names, 1, args.epochs, args.max_load, args.settle, args.seed, r, samples, meta, positions, counters)
    return tuple(make_record(exe, *side, args.epochs, args.seed) for exe, side in zip((exe_now, exe_ref), sides))


def reproduced_against(exe_now, exe_ref, names, args, rounds=MODES_MIN_SAMPLES, measure=side_by_side, listing=list_names):
    """Which of `names` (benchmarks, pairs, counters) still regress when both builds are measured again side by side.

    Returns (names that reproduce, names that do not, why nothing could be checked or None). Names the earlier
    executable does not have cannot be compared, and count as not reproduced. `measure` and `listing` are
    replaceable for tests.
    """
    try:
        have = set(listing(exe_ref, "")) & set(listing(exe_now, ""))
        targets = groups_of(names, have)
        if not targets:
            return set(), set(names), "the earlier build has none of these benchmarks"
        now, ref = measure(exe_now, exe_ref, targets, rounds, args)
    except Exception as error:  # a retained executable that no longer starts (missing library, moved) is not a verdict
        return set(), set(names), f"the earlier build could not be measured ({type(error).__name__}: {error})"
    rows, pair_rows, env = evaluate_with(now, baseline_data(ref), args, True)
    again = set(names_with(rows, pair_rows, env, "REGRESSED")) & set(names)
    return again, set(names) - again, None
