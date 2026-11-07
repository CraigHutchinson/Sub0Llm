"""Pushes the gate did not measure (on battery): kept in `.git/sub0perf/deferred.json` until a run on mains has measured them.

The baseline and the coverage state are never touched by a deferral, so the stalest-first rotation resumes where it
stopped; this list only says which commits went unmeasured, so a regression found later is attributed to the right range.
"""

import json
import time

from .common import atomic_write_text, read_json

KEEP = 50  # entries kept, newest last


def path(perf):
    return perf / "deferred.json" if perf else None


def load(perf):
    p = path(perf)
    try:
        found = read_json(p) if p and p.is_file() else []
    except (OSError, ValueError):
        found = []
    return found if isinstance(found, list) else []


def add(perf, head, refs, reason, partial=False):
    """Records a push that was not measured: its head, ranges (remote..local per pushed ref) and why. `partial` = stopped mid-run."""
    if not perf:
        return
    ranges = [f"{r[2]}: {r[3][:9]}..{r[1][:9]}" for r in refs if set(r[1]) != {"0"}]
    entry = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "head": head, "ranges": ranges, "reason": reason, "partial": partial}
    atomic_write_text(path(perf), json.dumps((load(perf) + [entry])[-KEEP:], indent=1) + "\n")


def clear(perf):
    p = path(perf)
    if p and p.is_file():
        p.unlink()


def describe(entries):
    """Lines for the next run on mains: which pushes went unmeasured."""
    if not entries:
        return []
    lines = [f"sub0perf: {len(entries)} earlier push(es) were not measured (deferred on battery); this run measures against the "
             f"unchanged baseline, so a regression may come from any of them:"]
    for e in entries:
        lines.append(f"  - {e['time']} {e['head'][:9]} ({e['reason']}{', stopped mid-run' if e.get('partial') else ''})"
                     + (f": {'; '.join(e['ranges'])}" if e.get("ranges") else ""))
    return lines
