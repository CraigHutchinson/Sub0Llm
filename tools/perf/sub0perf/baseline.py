"""The two file formats: the samples record (written by run, auto and gate) and the baseline (format 2)."""

import json
import platform
from pathlib import Path

from . import platform_proc, power
from .common import atomic_write_text
from .stats import pair_ratios, summarise


def make_record(exe, samples, meta, positions, counters, epochs=0, seed=None, state=None):
    """The one samples record written by run, auto and gate (format 2; `seed` is the shuffle seed of the rounds).

    `power` is the power state it was measured in (read now unless `state`, a power.PowerState, is given).
    """
    return {"format": 2, "exe": Path(exe).name, "power": (state or power.power_state()).as_dict(), "epochs": epochs,
            "cpu": platform_proc.pin_cpu(), "seed": seed,
            "counters": counters or {}, "host": platform.node(), "machine": platform.machine(),
            "system": f"{platform.system()} {platform.release()}",
            "rounds": meta, "samples": samples, "positions": positions}


def write_samples(path, record):
    atomic_write_text(path, json.dumps(record, indent=1) + "\n")


def baseline_tuple(data):
    """(benchmarks, calibration ns, pairs) of a baseline object; the earlier format (a number per benchmark) is lifted."""
    benchmarks = {}
    for name, v in data.get("benchmarks", {}).items():
        benchmarks[name] = v if isinstance(v, dict) else {"median_ns": v, "cv": None}
    return benchmarks, data.get("calibration_ns"), data.get("pairs", {})


def baseline_data(record):
    """A format-2 baseline object built from a samples record."""
    stats, cal, _, _ = summarise(record)
    pairs = pair_ratios(stats)
    return {"format": 2, "host": record.get("host", platform.node()), "machine": record.get("machine", platform.machine()),
            "system": record.get("system", f"{platform.system()} {platform.release()}"),
            "note": "Medians per op after calibration scaling, with robust CV across interleaved rounds. "
                    "Regenerate deliberately; say why in the commit. Regressions are accepted with `accept`.",
            **({"power": record["power"]} if "power" in record else {}),
            "calibration_ns": round(cal, 3),
            **({"calibration_sustained_ns": record["calibration_sustained_ns"]} if record.get("calibration_sustained_ns") else {}),
            "epochs": record.get("epochs", 0), "cpu": record.get("cpu"),
            "benchmarks": {n: {"median_ns": round(s["median_ns"], 3), "cv": round(s["cv"], 4), "rounds": s["n"],
                               **({"modes": [round(s["modes"][0], 3), round(s["modes"][1], 3)]} if s["modes"] else {}),
                               **{k: round(c["median"], 2) for k, c in s["counters"].items()}}
                           for n, s in sorted(stats.items())},
            "pairs": {n: {"reference": p["reference"], "ratio": round(p["ratio"], 4),
                          **({"modes": [round(p["modes"][0], 4), round(p["modes"][1], 4)]} if p["modes"] else {}),
                          **({"instruction_ratio": round(p["instruction_ratio"], 4)} if "instruction_ratio" in p else {})}
                      for n, p in sorted(pairs.items())},
            "accepted": []}


def write_baseline(path, data):
    atomic_write_text(path, json.dumps(data, indent=2) + "\n")
