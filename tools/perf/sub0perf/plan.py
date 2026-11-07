"""Planning a quick gate: which benchmarks, and how many rounds, fit a time budget.

The unit of rotation is a group: a pair group (a library benchmark and its hand-written twin, measured back to back in
the same rounds, so its ratio is a self-contained relative check, valid whenever it is selected) or a standalone
benchmark. A quick gate measures:

1. the controls, always: the CONTROLS cheapest hand-written references (not known to be bimodal), which absolute
   verdicts need to factor the environment out. Two is the least that lets one disagree with the other;
2. the groups that need a re-check: their last result was a regression that did not confirm, or inconclusive
   (at most half of the spendable budget, stalest first);
3. then whole groups, stalest first (never measured, then longest since last measured; ties are broken by a hash of the
   commit and the group, so the choice for a commit is repeatable and differs between commits), while they fit.

Every group is therefore covered within a bounded number of pushes, which the plan reports. If the stalest group alone
is over the budget it is measured alone and the overrun is reported. What was measured when is kept in
`.git/sub0perf/coverage/<target>.json` together with the measured cost of a sample, so the next plan is made from
timings, not guesses.
"""

import hashlib
import json
import math
import statistics
import time
from dataclasses import dataclass, field

from .common import REFERENCE_MARK, atomic_write_text, group_of, read_json

QUICK_ROUNDS = 3        # interleaved rounds: the least that gives a median and a spread
CONTROLS = 2            # hand-written references always measured, for the environment correction of absolute verdicts
RESERVE = 0.25          # share of the budget kept back for confirmation and retry passes
RECHECK_SHARE = 0.5     # share of the spendable budget that groups awaiting a re-check may take
FALLBACK_COST = 1.0     # seconds per sample when nothing is known and no probe was made
FALLBACK_ROUND_OVERHEAD = 2.0  # two steady calibrations per round, seconds
PROBE_SHARE = 0.2       # share of the budget that timing benchmarks with no recorded cost may take, all together
PROBE_CAP_SHARE = 0.1   # share of the budget one such probe may take; a benchmark that needs more is too slow for quick
MAX_COVER_PUSHES = 1000


def units_of(names):
    """{unit key: [members]}: a group with a hand-written reference is one unit (its members run side by side);
    every other benchmark is a unit of its own."""
    paired = {group_of(n) for n in names if REFERENCE_MARK in n}
    units = {}
    for n in names:
        units.setdefault(group_of(n) if group_of(n) in paired else n, []).append(n)
    return units


def choose_controls(names, costs, avoid=(), count=CONTROLS, default_cost=FALLBACK_COST):
    """The `count` cheapest hand-written references, skipping `avoid` (known bimodal) unless too few remain."""
    refs = sorted((n for n in names if REFERENCE_MARK in n), key=lambda n: (costs.get(n, default_cost), n))
    good = [n for n in refs if n not in set(avoid)]
    return (good + [n for n in refs if n not in good])[:count]


def tiebreak(commit, key):
    return hashlib.sha1(f"{commit}:{key}".encode("utf-8")).hexdigest()


def unit_staleness(members, benchmarks):
    """The push at which the stalest member was last measured (-1 = never)."""
    return min(benchmarks.get(n, {}).get("push", -1) for n in members)


def rotation_order(units, benchmarks, commit):
    """Unit keys by staleness: never measured first, then the longest since measured; deterministic for a commit."""
    return sorted(units, key=lambda k: (unit_staleness(units[k], benchmarks), tiebreak(commit, k)))


@dataclass
class Plan:
    rounds: int
    selected: list
    units: dict                 # every unit key -> its members
    controls: list
    rechecks: list              # unit keys chosen because their last result needed a re-check
    rotating: list              # unit keys chosen by staleness
    skipped: list               # unit keys not chosen this time, stalest first
    estimate: float             # predicted seconds of the planned pass
    budget: float
    reserve: float              # seconds kept for confirmation and retry passes
    overrun: float = 0.0        # seconds by which the plan itself exceeds the budget
    pushes_to_cover: int = 0    # pushes until every unit has had a turn, simulated from the costs
    deferred: list = field(default_factory=list)  # unit keys left to the full level: cost unknown, too slow, or over the whole budget
    notes: list = field(default_factory=list)


def round_cost(names, costs, overhead, default_cost):
    return overhead + sum(costs.get(n, default_cost) for n in names)


def default_cost_of(costs):
    return statistics.median(costs.values()) if costs else FALLBACK_COST


def select(names, state, commit, budget, rounds, reserve, default_cost, avoid, defer=()):
    """(units, controls, rechecks, rotating, skipped, estimate, chosen benchmark names, deferred) for one push.

    `defer` names benchmarks a quick gate must not measure (cost unknown, or too slow); their groups are deferred, as is
    a group that costs more than the whole budget beside the controls, since no overrun is justified for it.
    """
    costs = state.get("costs", {})
    overhead = state.get("round_overhead", FALLBACK_ROUND_OVERHEAD)
    units = units_of(names)
    defer = set(defer)
    controls = choose_controls([n for n in names if n not in defer], costs, avoid, default_cost=default_cost)
    chosen = set(controls)
    cost_of = lambda n: costs.get(n, default_cost)

    def added(key):
        return rounds * sum(cost_of(n) for n in units[key] if n not in chosen)

    estimate = rounds * (overhead + sum(cost_of(n) for n in controls))
    spend = budget * (1 - reserve)
    base = estimate
    order = rotation_order(units, state.get("benchmarks", {}), commit)
    deferred = [k for k in order if set(units[k]) & defer or base + added(k) > budget]
    order = [k for k in order if k not in deferred]
    rechecks, rotating, skipped = [], [], []
    limit = estimate + max(0.0, spend - estimate) * RECHECK_SHARE
    for k in (k for k in order if k in set(state.get("recheck", []))):
        if estimate + added(k) <= limit:
            rechecks.append(k)
            estimate += added(k)
            chosen |= set(units[k])
    waiting = [k for k in order if k not in rechecks]
    if waiting and not rechecks and estimate + added(waiting[0]) > spend:
        # The stalest group does not fit even beside the controls: measure it alone (otherwise cheaper groups would keep
        # it waiting for ever) and report the overrun.
        k = waiting.pop(0)
        estimate += added(k)
        chosen |= set(units[k])
        rotating.append(k)
        skipped = waiting
    else:
        for k in waiting:
            if estimate + added(k) <= spend:
                rotating.append(k)
                estimate += added(k)
                chosen |= set(units[k])
            else:
                skipped.append(k)
    return units, controls, rechecks, rotating, skipped, estimate, chosen, deferred


def plan_quick(names, state, commit, budget, rounds=QUICK_ROUNDS, reserve=RESERVE, default_cost=None, avoid=(), defer=()):
    """A Plan for measuring `names` within `budget` seconds. `state` is the coverage state (see load_state).

    `defer`: benchmarks not to measure now (see select); their groups are listed in Plan.deferred."""
    costs = state.get("costs", {})
    if default_cost is None:
        default_cost = default_cost_of(costs)
    units, controls, rechecks, rotating, skipped, estimate, chosen, deferred = select(names, state, commit, budget, rounds,
                                                                                      reserve, default_cost, avoid, defer)
    overrun = max(0.0, estimate - budget)
    notes = []
    if overrun:
        first = (rechecks + rotating)[0] if rechecks + rotating else "-"
        notes.append(f"the controls and the first group ({first}) are estimated at {estimate:.0f} s, over the "
                     f"{budget:.0f} s budget by {overrun:.0f} s; they are measured anyway")
    # Replay the following pushes (no re-checks) to say how many it takes to give every group a turn.
    sim = {"benchmarks": {n: dict(v) for n, v in state.get("benchmarks", {}).items()}, "costs": costs,
           "round_overhead": state.get("round_overhead", FALLBACK_ROUND_OVERHEAD), "pushes": state.get("pushes", 0)}
    covered, cover = set(deferred), 0  # deferred groups wait for full (or a cheaper probe), so they do not lengthen the cycle
    picked = chosen
    while cover < MAX_COVER_PUSHES:
        cover += 1
        covered |= {k for k, members in units.items() if set(members) & picked}
        mark_measured(sim, picked, "sim", date="")
        if len(covered) == len(units):
            break
        picked = select(names, sim, f"sim{cover}:{commit}", budget, rounds, reserve, default_cost, avoid, defer)[-2]
    return Plan(rounds, sorted(chosen), units, controls, rechecks, rotating, skipped, estimate, budget, budget * reserve, overrun,
                cover, deferred, notes)


def describe_plan(plan, total, state, commit):
    """The lines printed at the start of a quick gate: the plan and the coverage it gives."""
    benchmarks = state.get("benchmarks", {})
    pushes = state.get("pushes", 0)
    groups = len(plan.units)
    lines = [f"sub0perf: plan: {len(plan.selected)}/{total} benchmarks x {plan.rounds} rounds, estimated {plan.estimate:.0f} s of a "
             f"{plan.budget:.0f} s budget ({plan.reserve:.0f} s kept for confirmation and retries)",
             f"sub0perf:   controls, always: {len(plan.controls)} hand-written reference(s) ({'; '.join(plan.controls)}); "
             f"groups this push: {len(plan.rechecks) + len(plan.rotating)} of {groups} "
             f"({len(plan.rechecks)} awaiting a re-check)"]
    lines += [f"sub0perf:   {note}" for note in plan.notes]
    lines.append(f"sub0perf:   coverage: measured {len(plan.selected)}/{total} now; every group is covered within "
                 f"{plan.pushes_to_cover} push(es)")
    if plan.deferred:
        shown = ", ".join(plan.deferred[:5]) + (f" and {len(plan.deferred) - 5} more" if len(plan.deferred) > 5 else "")
        lines.append(f"sub0perf:   not measured yet: {len(plan.deferred)} group(s) too slow for the quick budget or not yet timed "
                     f"({shown}); the full level measures them")
    if plan.skipped:
        stalest = plan.skipped[0]
        last = unit_staleness(plan.units[stalest], benchmarks)
        when = "never measured" if last < 0 else f"last measured {pushes - last} push(es) ago"
        lines.append(f"sub0perf:   stalest after this push: {stalest} ({when})")
    return lines


def default_state():
    return {"pushes": 0, "benchmarks": {}, "costs": {}, "slow": {}, "round_overhead": FALLBACK_ROUND_OVERHEAD, "recheck": []}


def state_path(perf, target):
    return perf / "coverage" / f"{target}.json"


def load_state(perf, target):
    path = state_path(perf, target) if perf else None
    if path is not None and path.is_file():
        try:
            return {**default_state(), **read_json(path)}
        except (OSError, ValueError):
            pass
    return default_state()


def save_state(perf, target, state):
    if perf:
        atomic_write_text(state_path(perf, target), json.dumps(state, indent=1) + "\n")


def mark_measured(state, names, commit, date=None):
    """Counts a push and notes `names` as measured by it. Returns the push number."""
    state["pushes"] = state.get("pushes", 0) + 1
    stamp = date or time.strftime("%Y-%m-%dT%H:%M:%S")
    for n in names:
        state.setdefault("benchmarks", {})[n] = {"push": state["pushes"], "commit": commit, "date": stamp}
    return state["pushes"]


def absorb_timings(state, timings):
    """Keeps the measured cost of a sample (median wall seconds of its process) and of a round's calibration overhead."""
    for name, seconds in (timings.get("bench") or {}).items():
        if seconds:
            state.setdefault("costs", {})[name] = round(statistics.median(seconds), 3)
            state.get("slow", {}).pop(name, None)  # measured after all (a full run): it has a cost now
    if timings.get("overhead"):
        state["round_overhead"] = round(statistics.median(timings["overhead"]), 3)


def update_recheck(state, units, measured, flagged):
    """Groups whose result this push was a regression (confirmed or not) or inconclusive are re-checked first next time;
    a group measured this push without being flagged is cleared."""
    measured, flagged = set(measured), set(flagged)
    keep = {k for k in state.get("recheck", []) if k in units and not (set(units[k]) & measured)}
    state["recheck"] = sorted(keep | {k for k, members in units.items() if set(members) & flagged})


def unplanned(names, state):
    """Benchmarks a quick gate must not plan yet: those with no recorded cost, and those a probe found too slow."""
    costs, slow = state.get("costs", {}), state.get("slow", {})
    return [n for n in names if n not in costs or n in slow]


def new_timings():
    return {"bench": {}, "overhead": []}


class Budget:
    """A wall-clock allowance. `seconds` None means unlimited. The clock is replaceable for tests."""

    def __init__(self, seconds, clock=time.monotonic):
        self.seconds, self.clock, self.start = seconds, clock, clock()

    def elapsed(self):
        return self.clock() - self.start

    def remaining(self):
        return math.inf if self.seconds is None else self.seconds - self.elapsed()

    def affords(self, seconds):
        return seconds <= self.remaining()

    def overrun(self):
        return 0.0 if self.seconds is None else max(0.0, self.elapsed() - self.seconds)


def pass_estimate(names, rounds, state, default_cost=None):
    """Predicted seconds of `rounds` more rounds over just `names`."""
    costs = state.get("costs", {})
    if default_cost is None:
        default_cost = default_cost_of(costs)
    return rounds * round_cost(names, costs, state.get("round_overhead", FALLBACK_ROUND_OVERHEAD), default_cost)
