"""A cheap speed check before any build or measuring: is this machine as fast as it was when the baseline was recorded?

Battery is not the only way a machine runs slow. On mains a vendor power mode, a charging power cap or thermal limits can pin
the CPU at its base clock with no boost; every timing then comes out a uniform factor slower, and the gate would measure
everything just to call the run ENVIRONMENT at the end. The check times the benchmark executable's own calibration kernel
(`--calibration`: a fixed dependent arithmetic chain whose time tracks the clock; about 0.3 s, one thread, no build) and
compares it with the `calibration_ns` stored in the baseline, which was timed by the same kernel. Slower than the limit
means `throttled`, handled like battery (see power.py, gate.py).

The default limit is 1.4x. In the 32 local records of the reference host, the median calibration of a run sat within
0.97x..1.29x of the baseline's (the high end is runs that overlapped other load) and a single reading is up to 2.2x after the
quiet-wait idled the CPU; a machine pinned at base clock read 1.88x. 1.4x clears all the normal spread and catches the pinned
machine; a lone reading over the limit is re-read until two agree (runner.steady_calibration), so a cold clock is not a verdict.

That reading is a burst. A laptop in the Windows "Best power efficiency" power mode (or under a vendor mode) boosts normally for about
3 s of load and then falls to about 0.42x (P-cores) and 0.33x (E-cores) of the boost speed, which made every timing 2.4x slower
while a 0.3 s kernel still read 0.97x. So the check also takes a sustained reading: one process runs the kernel (a single
thread, as in the measurement) continuously for `warm_seconds` (default 4.5 s: past the 3 s onset with a margin; `--calibration --soak
<seconds>` of bench_runner.hpp) and reports the median of its last quarter. It has to be ONE process: the slowdown does not carry over
to a new process (measured 2026-11-04: 15 short readings back to back all read 1.00x, while a single 32 s process read 2.4x), so
repeating the 0.3 s reading does not find it. The result is compared with the baseline's `calibration_sustained_ns` (recorded the
same way by `sub0perf run`). An executable built before `--soak` existed ignores it (and writes no output file for it, or returns after 0.3 s): that is noticed, noted with the
remedy (rebuild), and only the burst figure is used. Every failure of the sustained reading (see SustainedUnavailable) is such a note, never an exception. A baseline without it (recorded before this existed) has only the burst figure, so the sustained reading is compared
with this run's own burst reading instead (a machine that drops by the limit within seconds is throttled whatever the baseline
says) and the note says so. The ratio the gate acts on is the slowest of the two. The sustained part is skipped when the burst
reading is already over the limit (deferring anyway). Its cost is warm_seconds plus a process start per push that has something to measure (the
empty-push and docs-only skips come first), against the gate's 90 s budget: about 5 %. `warm_seconds` 0 turns it off.

Without a built executable or a baseline calibration (a first run on a host) nothing can be compared: the gate proceeds and says so.
`SUB0PERF_FAKE_SPEED=<ratio>` replaces the burst measurement and `SUB0PERF_FAKE_SUSTAINED=<ratio>` the sustained one (alone: burst
fine, sustained throttled); they are for tests only.
"""

import os
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from .common import CALIBRATION, read_json

DEFAULT_LIMIT = 1.4
LIMIT_VAR = "SUB0MATH_PERF_THROTTLE_RATIO"   # slower than the baseline's calibration by more than this: throttled
FAKE_VAR = "SUB0PERF_FAKE_SPEED"             # tests only: the burst ratio to report
FAKE_SUSTAINED_VAR = "SUB0PERF_FAKE_SUSTAINED"   # tests only: the sustained ratio to report
DEFAULT_WARM = 4.5                           # seconds of load before the sustained reading (the throttle sets in after about 3 s)
WARM_VAR = "SUB0MATH_PERF_WARM_SECONDS"      # or "warm_seconds" in the config; 0 = burst reading only


@dataclass
class Check:
    ratio: float | None = None      # the slowest ratio over the targets and readings; None when nothing could be compared
    sustained: bool = False         # the slowest reading was a sustained one (it exceeded its burst reading)
    limit: float = DEFAULT_LIMIT
    notes: list = field(default_factory=list)   # one line per target (or why it was not compared) and warnings


def limit(environ=None, config=None):
    """The ratio above which the machine counts as throttled: the environment, else `throttle_ratio` of the config, else 1.4."""
    environ = os.environ if environ is None else environ
    for raw in (environ.get(LIMIT_VAR), (config or {}).get("throttle_ratio")):
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if value > 1.0:
            return value
    return DEFAULT_LIMIT


def warm_seconds(environ=None, config=None):
    """Seconds of load before the sustained reading: the environment, else `warm_seconds` of the config, else 4.5. 0 turns it off."""
    environ = os.environ if environ is None else environ
    for raw in (environ.get(WARM_VAR), (config or {}).get("warm_seconds")):
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if value >= 0:
            return value
    return DEFAULT_WARM


def read_ns(exe):
    """One reading of the calibration kernel, ns per iteration of its chain."""
    from .runner import measure
    with tempfile.TemporaryDirectory() as scratch:
        return measure(exe, ["--calibration"], scratch)[CALIBRATION]


def measure_ratio(exe, base_ns, bound, read=read_ns):
    """(ratio now / baseline, ns now). One reading; only when it is over `bound` are the readings repeated until two agree."""
    now = read(exe)
    if now / base_ns > bound:
        from .runner import steady_calibration
        now = steady_calibration(lambda: read(exe))
    return now / base_ns, now


class SustainedUnavailable(Exception):
    """The sustained reading could not be taken; the message says what happened and what to do. The burst figure is used alone."""


REBUILD_ADVICE = "rebuild the benchmark executable (against a Sub0Math with `--calibration --soak`) to get the sustained reading"


def read_sustained_ns(exe, warm):
    """One sustained reading: the calibration kernel run continuously for `warm` seconds in one process, the median of its last quarter
    (ns per iteration of its chain). Raises SustainedUnavailable, with the reason and the remedy, when it cannot be had:
    an executable built before `--soak` existed either returns at once or writes no output for it; the output can be unreadable;
    the process can fail to start, fail or hang."""
    from .runner import measure
    began = time.monotonic()
    with tempfile.TemporaryDirectory() as scratch:
        try:
            value = measure(exe, ["--calibration", "--soak", f"{warm:g}"], scratch, timeout=warm * 4 + 30)[CALIBRATION]
        except FileNotFoundError as error:
            if Path(str(exe)).is_file():  # the executable is there, so what is missing is its output file
                raise SustainedUnavailable(f"the executable wrote no output for `--calibration --soak` (it predates `--soak`); {REBUILD_ADVICE}") from error
            raise SustainedUnavailable(f"the benchmark executable {exe} cannot be found; build it") from error
        except subprocess.TimeoutExpired as error:
            raise SustainedUnavailable(f"the sustained run did not finish within {error.timeout:g} s and was stopped; "
                                       f"the machine may be heavily loaded: retry when it is idle") from error
        except subprocess.CalledProcessError as error:
            raise SustainedUnavailable(f"the executable exited with code {error.returncode} for `--calibration --soak {warm:g}`; "
                                       f"run it by hand to see why, or {REBUILD_ADVICE}") from error
        except (ValueError, KeyError, TypeError) as error:  # JSONDecodeError is a ValueError
            raise SustainedUnavailable(f"the executable's `--soak` output could not be read ({type(error).__name__}: {error}); "
                                       f"{REBUILD_ADVICE}") from error
        except OSError as error:
            raise SustainedUnavailable(f"the executable could not be run ({type(error).__name__}: {error}); check the build") from error
    took = time.monotonic() - began
    if took < 0.8 * warm:
        raise SustainedUnavailable(f"the executable returned after {took:.1f} s instead of {warm:g} s (it ignores `--soak`); {REBUILD_ADVICE}")
    return value


def record_sustained(exe, environ=None, config=None, read=read_sustained_ns):
    """The sustained calibration to store with a run (for a baseline made from it). None when the machine speed is faked, the warm time is 0 or the executable cannot be run. Costs the warm seconds."""
    environ = os.environ if environ is None else environ
    warm = warm_seconds(environ, config)
    if warm <= 0 or environ.get(FAKE_VAR) or environ.get(FAKE_SUSTAINED_VAR):
        return None
    try:
        return round(read(exe, warm), 3)
    except (SustainedUnavailable, OSError, ValueError, KeyError, subprocess.SubprocessError):
        return None


def check_targets(targets, environ=None, config=None, read=read_ns, read_sustained=read_sustained_ns):
    """Checks (label, exe, baseline path) triples; the result's ratio is the slowest. Never raises: what cannot be run is a note."""
    environ = os.environ if environ is None else environ
    result = Check(limit=limit(environ, config))
    warm = warm_seconds(environ, config)
    ratios = []  # (ratio, is sustained)
    fakes = {}
    for var in (FAKE_VAR, FAKE_SUSTAINED_VAR):
        if environ.get(var):
            try:
                fakes[var] = float(environ[var])
            except ValueError:
                result.notes.append(f"sub0perf: WARNING: {var}={environ[var]} is not a number; ignored")
            else:
                result.notes.append(f"sub0perf: WARNING: {var}={environ[var]} is set: the machine speed is being faked (for tests only)")
    if fakes:
        ratios = [(fakes.get(FAKE_VAR, 1.0), False)] + ([(fakes[FAKE_SUSTAINED_VAR], True)] if FAKE_SUSTAINED_VAR in fakes else [])
    elif not any(environ.get(v) for v in (FAKE_VAR, FAKE_SUSTAINED_VAR)):
        for label, exe, baseline in targets:
            try:
                data = read_json(baseline) if baseline and Path(baseline).is_file() else {}
                base_ns = data.get("calibration_ns")
                if not base_ns:
                    result.notes.append(f"sub0perf: speed check skipped for {label}: the baseline has no calibration (a first run); proceeding")
                elif not (exe and Path(exe).is_file()):
                    result.notes.append(f"sub0perf: speed check skipped for {label}: no built benchmark to time yet; proceeding")
                else:
                    ratio, now = measure_ratio(exe, base_ns, result.limit, read)
                    ratios.append((ratio, False))
                    result.notes.append(f"sub0perf: speed check {label}: calibration {now:,.0f} ns against the baseline's {base_ns:,.0f} ns "
                                        f"= {ratio:.2f}x (throttled above {result.limit:.2f}x)")
                    if warm > 0 and ratio <= result.limit:  # a burst that is already slow defers anyway: no need to wait for more
                        try:
                            steady = read_sustained(exe, warm)
                        except SustainedUnavailable as problem:
                            steady = None
                            result.notes.append(f"sub0perf: speed check {label}: no sustained reading: {problem}. Only the burst reading is used")
                        except (OSError, ValueError, KeyError, subprocess.SubprocessError) as problem:
                            steady = None
                            result.notes.append(f"sub0perf: speed check {label}: no sustained reading: {type(problem).__name__}: {problem}. "
                                                f"Only the burst reading is used; {REBUILD_ADVICE}")
                        base_sustained = data.get("calibration_sustained_ns")
                        against = base_sustained or now
                        if steady is not None:
                            ratios.append((steady / against, True))
                            result.notes.append(
                                f"sub0perf: speed check {label}: sustained calibration {steady:,.0f} ns over {warm:g} s of load, against "
                                + (f"the baseline's {base_sustained:,.0f} ns" if base_sustained else
                                   "this run's burst reading (the baseline has no sustained calibration; re-record it with `run`)")
                                + f" = {steady / against:.2f}x")
            except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
                result.notes.append(f"sub0perf: speed check skipped for {label}: {type(error).__name__}: {error}; proceeding")
    if ratios:
        result.ratio, result.sustained = max(ratios, key=lambda r: (r[0], r[1]))
    return result
