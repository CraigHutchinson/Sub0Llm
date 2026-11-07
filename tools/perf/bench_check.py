#!/usr/bin/env python3
"""Runs benchmarks interleaved and judges them against a per-host baseline: regressions, progressions,
noise, and paired comparisons with hand-written references.

  bench_check.py run      --exe BENCH --out samples.json [--rounds 5] [--epochs N] [--filter TEXT]
  bench_check.py check    --results samples.json --baseline perf/baselines/<host>.json
  bench_check.py baseline --results samples.json --baseline perf/baselines/<host>.json
  bench_check.py auto     --exe BENCH --baseline ... [--rounds 3] [--epochs 5] [--report-only] [--record]
  bench_check.py accept   --results samples.json --baseline ... --reason TEXT (--benchmark NAME ... | --all-changed)
  bench_check.py gate     [--repo DIR] [--confirm-rounds 4]                      (the pre-push hook; reads .git/sub0perf/config.json)
  bench_check.py install-hooks --repo DIR --build-dir DIR --target NAME --exe PATH --baseline PATH

How a run is made robust (`run`, and `auto` which runs then checks):

- Interleaved rounds. Every benchmark is measured once per round, each in its own short process
  (`--exact NAME`), in a freshly shuffled order every round (seeded, so a run can be repeated). Members of
  a group (names sharing the text before the first ", ") run back to back, in shuffled order, so a pair's
  ratio is taken from adjacent measurements. Slow drift (thermal, turbo, background work) then spreads
  over every benchmark instead of biasing the ones measured last.
- Calibration. A fixed dependent arithmetic chain (`BENCH --calibration`) is timed at the start and end
  of every round. Its time tracks the core's clock, so each round's timings are scaled by it, and its
  change from the first to the last round is reported as clock drift. A baseline records the
  calibration time too, so a whole run on a slower or throttled host is corrected before judging.
- System load. Before each round the CPU time used by other processes is sampled. Above --max-load the
  run waits (up to --settle seconds) for it to fall, then proceeds and marks the round as loaded.
- Variance. Each benchmark's spread across rounds is measured as a robust coefficient of variation
  (median absolute deviation x 1.4826 / median). A change counts only if it exceeds both the tolerance
  and three times the noise. A benchmark noisier than --max-cv is NOISY: `auto` gives it extra rounds
  (up to --max-rounds) and reports it as inconclusive if it stays noisy, rather than guessing.

Hardware counters. Where the OS exposes them (Linux perf_event; not Windows), nanobench reports
instructions, CPU cycles, branches and branch misses per op. They are kept with every record and
baseline. Instructions per op are nearly deterministic, so they are judged with a tight tolerance
(--instr-tolerance, 2%): a change that adds work shows even when timing noise would hide it. Cycles per
op are judged like time but do not depend on the clock speed. The report adds IPC and the branch-miss
rate, and pairs add the instruction ratio against the hand-written reference: exactly how much more
work the vocabulary does. A counter regression counts like a time regression.

Verdicts: REGRESSED (slower, significant), IMPROVED (faster, significant: a progression to record with
`baseline`), ok, NOISY, new, missing. Pairs: in every group containing a name with "hand-written" (the
fastest-known reference, never weakened), each other member's time is divided by the reference's in the
same round. That ratio is nearly independent of the host, so it is reported even with no baseline, and
compared with the baseline's ratio when there is one.

Local records and the pre-push gate. Every measurement of a clean working tree (the after-build check
with --record, and the gate) is kept as `.git/sub0perf/records/<target>/<commit>.json`, so the state a
branch was checked out at and the state about to be pushed are both on record, measured on this
machine. The pre-push hook (`gate`) builds each configured benchmark, measures HEAD, and blocks the
push on a regression against the committed baseline, or against the nearest earlier commit with a
local record (like for like: same machine, same session conditions), unless the regression has been
accepted. Accepting is explicit: `accept` moves the named benchmarks' baseline to the new measurement
and appends an entry (date, commit, reason, before and after) to the baseline's `accepted` log; commit
that file with the change. `git push --no-verify` bypasses the hook, as with any git hook.

Self-test. `selftest` proves in seconds that a repository's gate is wired as intended, without building or measuring: the
pre-push hook is installed where git looks (core.hooksPath honoured), calls an existing bench_check.py and matches what
`install-hooks` writes (drift is reported); `.git/sub0perf/config.json`, the targets' build dirs, executables and baselines,
and the pins are valid (an unbuilt target is a warning); the real gate, in a plan-only mode, skips empty stdin, deletions and
documentation-only ranges and plans a quick run for a code range whose predicted seconds plus the probe allowance fit the
budget (also reports the unknown and slow benchmarks and the stalest group); and no stale lock is left. Exit 1 on any FAIL.

Confirmation. A single short measurement is not enough evidence to block. In `auto` and `gate`, anything
judged REGRESSED on the first pass (a benchmark, a pair or a counter, against the baseline or the earlier
record) is measured again for --confirm-rounds more rounds (default 4; calibration as usual), and the
verdicts are recomputed from all rounds. Only regressions that persist block; the rest are reported as
"not confirmed". Where there are no controls (no hand-written reference), a change must also exceed 3x
the suite's own spread of changes (median absolute deviation across all benchmarks) to count, since an
environment shift cannot otherwise be told from a regression.

`check` and `auto` exit 1 on any regression unless --report-only. With GITHUB_STEP_SUMMARY set, the
report is appended to the job summary. Baselines from the earlier format (a number per benchmark) are
still read; they lack calibration and spread, so judging falls back to the suite's median drift.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sub0perf.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
