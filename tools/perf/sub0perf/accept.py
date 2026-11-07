"""Recording a baseline, and accepting changes into it with a logged reason (the baseline's `accepted` list)."""

import time
from pathlib import Path

from .baseline import baseline_data, write_baseline
from . import power
from .common import read_json
from .gitstate import git_head
from .judge import evaluate_with, names_with
from .runner import COUNTERS
from .stats import measured_count


# A run whose calibration kernel swung by more than this between rounds was measured in an unsteady machine state
# (seen: 5,046 vs 9,042 ns in one run, which scaled every median by up to 40%); a baseline must not come from it.
MAX_CALIBRATION_SPREAD = 1.25


def calibration_spread(record):
    """Largest over smallest calibration time across the run's rounds (1.0 when there is nothing to compare)."""
    values = [v for r in record.get("rounds", []) for v in (r.get("calibration_start_ns"), r.get("calibration_end_ns"))
              if isinstance(v, (int, float)) and v > 0]
    return max(values) / min(values) if len(values) >= 2 else 1.0


def unsteady(record, source):
    """Prints why and returns True when the record's calibration was too unsteady to record a baseline from."""
    spread = calibration_spread(record)
    if spread > MAX_CALIBRATION_SPREAD:
        print(f"bench_check: calibration varied {spread:.2f}x across the rounds of {source} (limit {MAX_CALIBRATION_SPREAD}x);"
              f" the machine was not steady, so it cannot become a baseline. Re-measure on a quiet machine")
        return True
    return False


def power_refusal(record, source, force=False):
    """Prints why and returns True when the record's power state rules it out as a baseline.

    A run measured on battery (or in a battery saver) is refused; `--force-power` allows one made by hand (`run`), never one
    the gate measured under SUB0MATH_PERF_ON_BATTERY (flagged `power_override`). `unknown` is allowed, and said so.
    A record without the field (made before it existed) counts as ac.
    """
    info = record.get("power")
    if record.get("power_override"):
        print(f"bench_check: {source} was measured under {power.OVERRIDE_VAR}=1 (power: {power.label(info)}); it can never become a baseline")
        return True
    if power.key(info) != (power.AC, False, False):
        if force:
            print(f"bench_check: WARNING: recording a baseline from a run measured on {power.label(info)} (--force-power)")
            return False
        print(f"bench_check: {source} was measured on {power.label(info)}, where clocks and boost differ from mains; it cannot become a "
              f"baseline. Re-measure on mains (--force-power overrides, for a run made by hand)")
        return True
    if (info or {}).get("state") == power.UNKNOWN:
        print(f"bench_check: note: the power state of {source} is unknown; it is accepted as mains")
    return False


def baseline_command(args):
    record = read_json(args.results)
    if power_refusal(record, args.results, getattr(args, "force_power", False)) or unsteady(record, args.results):
        return 2
    data = baseline_data(record)
    path = Path(args.baseline)
    if path.is_file():
        old = read_json(path)
        if not args.reason:
            print("bench_check: a baseline exists; re-recording it needs --reason (it is logged in the file)")
            return 2
        data["accepted"] = old.get("accepted", []) + [{
            "date": time.strftime("%Y-%m-%d"), "commit": git_head(path.parent) or "",
            "reason": args.reason, "benchmarks": "all (re-recorded)"}]
    write_baseline(path, data)
    print(f"wrote {len(data['benchmarks'])} benchmarks and {len(data['pairs'])} pairs to {args.baseline}")
    return 0


def without_data(record, names):
    """The names the samples record has no measured sample for (absent, empty, or only filled-in copies)."""
    return [n for n in names if measured_count(record, n) == 0]


def accept_command(args):
    """Moves the named (or every changed) benchmark's baseline to the measured value, with a logged reason."""
    record = read_json(args.results)
    fresh = baseline_data(record)
    path = Path(args.baseline)
    data = read_json(path) if path.is_file() else baseline_data(record)
    if not isinstance(next(iter(data.get("benchmarks", {}).values()), {}), dict):
        print("bench_check: the baseline is in the old format; re-record it with `baseline --reason ...` first")
        return 2
    if args.all_changed:
        rows, pair_rows, env = evaluate_with(record, path, args, False)
        names = names_with(rows, pair_rows, env, "REGRESSED", "IMPROVED")
    else:
        names = args.benchmark
    if not names:
        print("bench_check: nothing to accept")
        return 2
    # Without a measured sample there is nothing to move the baseline to: it would keep its old value and log no change.
    empty = without_data(record, names)
    if empty:
        print(f"bench_check: no measured samples in {args.results} for: {', '.join(empty)}\n"
              f"  measure them first, e.g. `run --exe ... --filter \"<name or group>\" --out samples.json`, then accept from that file")
        return 2
    if power_refusal(record, args.results, getattr(args, "force_power", False)) or unsteady(record, args.results):
        return 2
    scale = (data.get("calibration_ns") or fresh["calibration_ns"]) / fresh["calibration_ns"]
    changes = {}
    for n in names:
        new = dict(fresh["benchmarks"][n])
        new["median_ns"] = round(new["median_ns"] * scale, 3)  # expressed at the baseline's clock
        old = data["benchmarks"].get(n, {}).get("median_ns")
        changes[n] = {"from_ns": old, "to_ns": new["median_ns"],
                      "change": None if not old else round(new["median_ns"] / old - 1.0, 4)}
        for key in COUNTERS:
            if key in new:
                before = data["benchmarks"].get(n, {}).get(key)
                changes[n][key] = {"from": before, "to": new[key],
                                   "change": None if not before else round(new[key] / before - 1.0, 4)}
        data["benchmarks"][n] = new
        if n in fresh["pairs"]:
            changes[n]["pair_ratio"] = {"from": data.get("pairs", {}).get(n, {}).get("ratio"), "to": fresh["pairs"][n]["ratio"]}
            data.setdefault("pairs", {})[n] = fresh["pairs"][n]
    data.setdefault("accepted", []).append({"date": time.strftime("%Y-%m-%d"), "commit": git_head(path.parent) or "",
                                            "reason": args.reason, "benchmarks": changes})
    write_baseline(path, data)
    for n, c in changes.items():
        delta = "new" if c["change"] is None else f"time {c['change']:+.1%}"
        if c.get("instructions", {}).get("change") is not None:
            delta += f", instructions {c['instructions']['change']:+.1%}"
        print(f"accepted {n}: {delta}")
    print(f"logged in {path}; commit it with the change, and say why")
    return 0
