"""Measuring: one process per benchmark, interleaved rounds, calibration at the edges of every round."""

import json
import random
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

from . import platform_proc
from .common import CALIBRATION, group_of
from .load import wait_for_quiet

COUNTERS = {"instructions": "median(instructions)", "cycles": "median(cpucycles)",
            "branches": "median(branchinstructions)", "branch_misses": "median(branchmisses)"}


def measure_full(exe, args, scratch, timeout=None):
    """Runs the benchmark executable once; returns {name: (ns per op, {counter: per op})}.

    Counters are included only when the platform provides them (nanobench reports 0 otherwise).
    """
    out = Path(scratch) / "one.json"
    platform_proc.launch([exe, *args, "--json", str(out)], timeout)
    data = json.loads(out.read_text(encoding="utf-8"))
    results = {}
    for r in data["results"]:
        batch = r["batch"] or 1.0
        counters = {k: r[f] / batch for k, f in COUNTERS.items() if r.get(f)}
        results[r["name"]] = (r["median(elapsed)"] / batch * 1e9, counters)
    return results


def measure(exe, args, scratch, timeout=None):
    """Runs the benchmark executable once; returns {name: ns per op}."""
    return {name: ns for name, (ns, _) in measure_full(exe, args, scratch, timeout).items()}


def list_names(exe, text_filter):
    args = [exe, "--list"] + (["--filter", text_filter] if text_filter else [])
    out = subprocess.run(args, check=True, capture_output=True, text=True).stdout
    return [line.strip() for line in out.splitlines() if line.strip()]


# The quiet-wait idles the pinned CPU, and its clock drops: a calibration taken straight after it read about 1.8x the
# warm value (9,030 vs 4,900 ns, every round on 2026-10-27), and interpolating samples towards it distorted every
# scaled median. Repeat until two consecutive readings agree, so the reading is taken at the working clock.
CALIBRATION_AGREEMENT = 0.05
CALIBRATION_ATTEMPTS = 6


def steady_calibration(read):
    """Calls `read()` (one calibration measurement) until two consecutive values agree within CALIBRATION_AGREEMENT,
    at most CALIBRATION_ATTEMPTS times, and returns the last value."""
    previous = read()
    for _ in range(CALIBRATION_ATTEMPTS - 1):
        value = read()
        if abs(value - previous) <= CALIBRATION_AGREEMENT * min(value, previous):
            return value
        previous = value
    return previous


class PowerChanged(Exception):
    """The machine went on battery (or into a battery saver) during a run: what was measured is inconclusive."""


_round_guard = None  # called before every round; raises PowerChanged (see gate.py). Like the pin CPU, set once per command.


def set_round_guard(guard):
    global _round_guard
    _round_guard = guard


def run_rounds(exe, names, rounds, epochs, max_load, settle, seed, first_round=0, samples=None, meta=None, positions=None,
               counters=None, timings=None):
    """Interleaved rounds. `timings`, when given ({"bench": {name: [seconds]}, "overhead": [seconds]}), collects the wall time of
    each benchmark's process and of each round's two calibrations, which is what planning a budget needs."""
    samples = samples if samples is not None else {n: [] for n in names}
    counters = counters if counters is not None else {}
    positions = positions if positions is not None else {n: [] for n in names}
    meta = meta if meta is not None else []
    extra = ["--epochs", str(epochs)] if epochs else []
    groups = {}
    for n in names:
        groups.setdefault(group_of(n), []).append(n)
    with tempfile.TemporaryDirectory() as scratch:
        for r in range(first_round, first_round + rounds):
            if _round_guard:
                _round_guard()
            rng = random.Random(seed * 1000003 + r)
            load, waited, quiet = wait_for_quiet(max_load, settle)
            began = time.monotonic()
            cal_start = steady_calibration(lambda: measure(exe, ["--calibration", *extra], scratch)[CALIBRATION])
            overhead = time.monotonic() - began
            order = list(groups)
            rng.shuffle(order)
            sequence = []
            for g in order:
                members = list(groups[g])
                rng.shuffle(members)
                sequence += members
            for k, n in enumerate(sequence):
                began = time.monotonic()
                ns, found = measure_full(exe, ["--exact", n, *extra], scratch)[n]
                if timings is not None:
                    timings["bench"].setdefault(n, []).append(time.monotonic() - began)
                samples.setdefault(n, []).append(ns)
                for key, value in found.items():
                    counters.setdefault(n, {}).setdefault(key, []).append(value)
                # Where in the round it ran (0 = start, 1 = end), to interpolate the clock between calibrations.
                positions.setdefault(n, []).append(round((k + 0.5) / len(sequence), 4))
            began = time.monotonic()
            cal_end = steady_calibration(lambda: measure(exe, ["--calibration", *extra], scratch)[CALIBRATION])
            if timings is not None:
                timings["overhead"].append(overhead + time.monotonic() - began)
            meta.append({"round": r, "load": round(load, 4), "waited_s": round(waited, 1), "quiet": quiet,
                         "calibration_start_ns": cal_start, "calibration_end_ns": cal_end})
            print(f"round {r + 1}: load {load:.0%}{'' if quiet else ' (busy)'}, calibration {cal_start:,.0f}/{cal_end:,.0f} ns,"
                  f" {sum(len(m) for m in groups.values())} benchmarks", flush=True)
    return samples, meta, positions, counters


def fill_unmeasured(samples, positions, lengths_before):
    """Gives every benchmark that got no new sample its median again, marked filled, so all keep one sample per round."""
    for n, v in samples.items():
        if len(v) == lengths_before[n]:
            v.append(statistics.median(v))
            positions.setdefault(n, []).append(None)  # filled, not measured


def extend_rounds(run, names, rounds, samples, meta, positions, counters):
    """Adds `rounds` rounds measuring only `names`; the other benchmarks repeat their median so rounds stay aligned.

    `run` has run_rounds' signature minus the process arguments, so a test can pass a stub that measures nothing.
    """
    for _ in range(rounds):
        before = {n: len(v) for n, v in samples.items()}
        run(sorted(names), 1, len(meta), samples, meta, positions, counters)
        fill_unmeasured(samples, positions, before)


def measure_all(exe, args, names=None, rounds=None, timings=None):
    """A run with the measuring options of parsed command-line `args`: (samples, meta, positions, counters).

    Every benchmark (those matching --filter), or just `names` when a plan chose them; `rounds` overrides args.rounds.
    """
    return run_rounds(exe, names if names is not None else list_names(exe, args.filter), rounds or args.rounds, args.epochs,
                      args.max_load, args.settle, args.seed, timings=timings)


def probe_cost(exe, name, args):
    """A cheap probe when no timings are known: the wall seconds of one benchmark process and of one calibration.

    Returns (seconds per benchmark sample, seconds of a round's calibrations).
    """
    extra = ["--epochs", str(args.epochs)] if args.epochs else []
    with tempfile.TemporaryDirectory() as scratch:
        began = time.monotonic()
        measure(exe, ["--calibration", *extra], scratch)
        calibration = time.monotonic() - began
        began = time.monotonic()
        measure(exe, ["--exact", name, *extra], scratch)
        return time.monotonic() - began, 2 * calibration


def probe_unknown(exe, names, args, allowance, per_probe, clock=time.monotonic):
    """Times one sample of each of `names` (benchmarks with no recorded cost), within `allowance` seconds in all.

    A benchmark is cut off after `per_probe` seconds (or what is left of the allowance): it is then too slow for a
    quick gate and is reported as slow, with that many seconds as the least one sample costs. The allowance running
    out leaves the rest unprobed. Returns (costs, slow, seconds spent).
    """
    extra = ["--epochs", str(args.epochs)] if args.epochs else []
    costs, slow = {}, {}
    began = clock()
    with tempfile.TemporaryDirectory() as scratch:
        for n in names:
            left = allowance - (clock() - began)
            if left <= 0:
                break
            limit = min(per_probe, left)
            started = clock()
            try:
                measure_full(exe, ["--exact", n, *extra], scratch, limit)
                costs[n] = round(clock() - started, 3)
            except subprocess.TimeoutExpired:
                if limit < per_probe:
                    break  # the allowance ended it, which shows nothing about this benchmark
                slow[n] = round(limit, 3)
            except (subprocess.CalledProcessError, OSError, ValueError, KeyError):
                pass  # unprobed: stays unplanned
    return costs, slow, clock() - began


def rerunner(exe, args, samples, meta, positions, counters, timings=None):
    """`rerun(names, rounds)` that adds rounds for just `names` to these samples (see extend_rounds)."""
    def rerun(names, rounds):
        extend_rounds(lambda ns, r, first, *rest: run_rounds(exe, ns, r, args.epochs, args.max_load, args.settle, args.seed, first,
                                                              *rest, timings=timings), names, rounds, samples, meta, positions,
                      counters)
    return rerun
