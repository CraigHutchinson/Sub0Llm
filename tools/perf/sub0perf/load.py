"""System load: how busy the machine is, and waiting for it to quieten."""

import os
import time

from . import platform_proc


def system_load(seconds=0.5):
    """Fraction of all CPU time spent busy over a short window (our own process is idle meanwhile)."""
    first = platform_proc.cpu_times()
    if first is None:
        if hasattr(os, "getloadavg"):
            return min(1.0, os.getloadavg()[0] / (os.cpu_count() or 1))
        return 0.0
    time.sleep(seconds)
    second = platform_proc.cpu_times()
    busy, total = second[0] - first[0], second[1] - first[1]
    return busy / total if total > 0 else 0.0


def wait_for_quiet(max_load, settle):
    """Waits up to `settle` seconds for the load to fall to `max_load`; returns (load, waited, quiet)."""
    start = time.monotonic()
    load = system_load()
    while load > max_load and time.monotonic() - start < settle:
        time.sleep(1.0)
        load = system_load()
    return load, time.monotonic() - start, load <= max_load
