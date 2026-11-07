#!/usr/bin/env python3
"""Runs benchmarks interleaved and judges them against a per-host baseline: regressions, progressions,
noise, and paired comparisons with hand-written references.

  bench_check.py run      --exe BENCH --out samples.json [--rounds 5] [--epochs N] [--filter TEXT]
  bench_check.py check    --results samples.json --baseline perf/baselines/<host>.json
  bench_check.py baseline --results samples.json --baseline perf/baselines/<host>.json
  bench_check.py auto     --exe BENCH --baseline ... [--rounds 3] [--epochs 5] [--report-only] [--record]
  bench_check.py accept   --results samples.json --baseline ... --reason TEXT (--benchmark NAME ... | --all-changed)
  bench_check.py gate     [--repo DIR] [--level quick|full] [--budget SECONDS] [--confirm-rounds 4]                      (the pre-push hook; reads .git/sub0perf/config.json)
  bench_check.py install-hooks --repo DIR --build-dir DIR --target NAME --exe PATH --baseline PATH  [--pin file=F,var=V,repo=owner/name ...]

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

Levels and the time budget. `gate` and `auto` run at the `quick` level by default: 2 controls (hand-written references) always, plus as
many whole groups (a pair group, or a standalone benchmark), stalest first, as fit --budget (default 90 s, SUB0MATH_PERF_BUDGET, or the configured quick budget). A regression
is still confirmed in full even if that overruns the budget; inconclusive and unmeasured benchmarks never block.
A benchmark with no recorded cost is timed first (one sample, cut off at a tenth of the budget, a fifth in all); a slower one, or a group costing more than the whole budget, is
reported as not measured yet and left to the full level, and never justifies an overrun.
`--level full` (or SUB0MATH_PERF_LEVEL=full, or pushing a tag matching `v*` / `milestone/*`, configurable with
install-hooks --full-refs) measures everything with the full method and no budget. The level and why are the first
line of the gate output. See docs/PERFORMANCE.md, "Levels, time budget and coverage".

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

import argparse
import contextlib
import os
import sys
from pathlib import Path

from . import levels, platform_proc, plan as planning, power, speed
from .accept import accept_command, baseline_command
from .baseline import make_record, write_samples
from .common import read_json
from .confirm import confirm_regressions, noisy_names
from .gate import choose_benchmarks, gate_command, power_guard
from .gitstate import git_head, perf_dir, save_record, tree_clean
from .hooks import install_hooks_command, note_checkout_command
from .judge import evaluate_with
from .lock import LockTimeout, PerfLock
from .report import report
from .selftest import selftest_command
from .runner import PowerChanged, fill_unmeasured, measure_all, rerunner, run_rounds, set_round_guard
from .stats import rounds_needed


def checked_state(args):
    """The power state including the speed check against args.baseline (see speed.py), printed. Skipped on battery, which defers anyway."""
    state = power.power_state()
    if state.untrusted or not (args.exe and args.baseline):
        return state
    config = None
    if getattr(args, "repo", None):
        perf = perf_dir(Path(args.repo))
        config = read_json(perf / "config.json") if perf and (perf / "config.json").is_file() else None
    found = speed.check_targets([(args.target or Path(args.exe).stem, args.exe, args.baseline)], config=config)
    for note in found.notes:
        print(note)
    return power.with_speed(state, found.ratio, found.limit, found.sustained)


def power_command(args):
    """Prints the power state (and the switches in effect); exit 0 when timings are trustworthy, 3 on battery, in a saver or throttled
    (the speed check runs when --exe and --baseline are given)."""
    state = checked_state(args)
    print(f"sub0perf: power: {power.describe(state)}")
    for note in power.environment_notes(state, power.policy()):
        print(note)
    return 3 if state.untrusted else 0


def run_command(args):
    state = checked_state(args)  # with --baseline: a throttled machine is noticed, and the record cannot become a baseline
    if state.untrusted:
        print(f"sub0perf: WARNING: measuring {state.reason()}; the record carries the power state, and cannot become a baseline "
              f"without --force-power")
    samples, meta, positions, counters = measure_all(args.exe, args)
    record = make_record(args.exe, samples, meta, positions, counters, args.epochs, args.seed, state)
    if not state.untrusted:  # the sustained speed of this host, for a baseline made from the record (the guard compares with it)
        sustained = speed.record_sustained(args.exe)
        if sustained:
            record["calibration_sustained_ns"] = sustained
    write_samples(args.out, record)
    return 0


def check_command(args):
    record = read_json(args.results)
    _, _, regressed = report(record, args)
    return 0 if args.report_only or not regressed else 1


def auto_command(args):
    if os.environ.get("SUB0MATH_PERF_SKIP"):
        return 0  # the gate builds benchmarks itself and measures them once
    state, pol = checked_state(args), power.policy()
    args.power_state = state
    for note in power.environment_notes(state, pol):
        print(note)
    if state.untrusted and not pol.override:  # the after-build check burns battery for a verdict that cannot be trusted
        print(power.deferral_text(state, False).replace("the next push on mains measures", "the next gate run on mains measures"))
        return 0
    set_round_guard(None if pol.override else power_guard)
    try:
        with PerfLock(Path(args.repo), "auto", args.lock_timeout) if args.repo else contextlib.nullcontext():
            return measure_and_judge(args)
    except LockTimeout as e:
        print(f"sub0perf: {e}")
        return 0 if args.report_only else 1
    except PowerChanged as changed:
        print(f"sub0perf: the power state changed during the run ({changed.args[0].reason()}); inconclusive, nothing judged or kept")
        return 0
    finally:
        set_round_guard(None)


def measure_and_judge(args):
    from .runner import list_names
    levels.resolve(args, [])  # --level / SUB0MATH_PERF_LEVEL, else quick; refs only matter to the pre-push gate
    repo = Path(args.repo) if args.repo else None
    perf = perf_dir(repo) if repo else None
    head = (git_head(repo) if repo else None) or str(args.seed)
    clean = bool(repo) and tree_clean(repo)
    state = planning.load_state(perf, args.target) if args.target else planning.default_state()
    timings = planning.new_timings()
    print(f"sub0perf: level {args.level} ({args.level_why})")
    if args.filter:  # the user chose the benchmarks
        names, rounds = None, args.rounds
    else:
        names, rounds, _ = choose_benchmarks(args, args.exe, args.target or Path(args.exe).stem, state, head, list_names(args.exe, ""),
                                      args.baseline)
        args.subset = len(names) < len(list_names(args.exe, ""))
    budget = planning.Budget(args.budget_seconds if args.level == levels.QUICK else None)
    samples, meta, positions, counters = measure_all(args.exe, args, names=names, rounds=rounds, timings=timings)
    now = getattr(args, "power_state", None) or power.power_state()  # (`state` above is the coverage state)
    record = make_record(args.exe, samples, meta, positions, counters, args.epochs, args.seed, now)
    if now.untrusted:  # only under SUB0MATH_PERF_ON_BATTERY
        record["power_override"] = True
        print(f"sub0perf: {power.override_text(now)}")
    # Benchmarks too noisy to judge get more rounds, up to --max-rounds in total (a quick run only while the budget allows).
    while len(meta) < args.max_rounds:
        rows, pair_rows, _ = evaluate_with(record, args.baseline, args, bool(args.filter))
        noisy = noisy_names(rows, pair_rows)
        if not noisy:
            break
        if args.level == levels.QUICK and args.budget_seconds and not budget.affords(planning.pass_estimate(noisy, 1, state)):
            print(f"sub0perf: {len(noisy)} benchmark(s) too noisy; another round does not fit the time budget")
            break
        print(f"{len(noisy)} benchmark(s) too noisy; adding a round for them", flush=True)
        # Extra rounds re-measure only the noisy set; the others repeat their median so rounds stay aligned.
        before = {n: len(v) for n, v in samples.items()}
        run_rounds(args.exe, sorted(noisy), 1, args.epochs, args.max_load, args.settle, args.seed, len(meta), samples, meta,
                   positions, counters, timings)
        fill_unmeasured(samples, positions, before)
    rerun = rerunner(args.exe, args, samples, meta, positions, counters, timings)
    confirm_regressions(lambda: [evaluate_with(record, args.baseline, args, bool(args.filter))], rerun, args.confirm_rounds,
                        lambda names: rounds_needed(record, names))
    if args.out:
        write_samples(args.out, record)
    if args.record and args.target:
        save_record(Path(args.repo or "."), args.target, record, only_if_missing=True, exe=args.exe)
    if args.target and perf and clean and not args.filter:  # keep what was measured when, and what it cost, for the next plan
        planning.absorb_timings(state, timings)
        planning.mark_measured(state, names if names is not None else list_names(args.exe, ""), head)
        planning.save_state(perf, args.target, state)
    if budget.overrun():
        print(f"sub0perf: the time budget was exceeded by {budget.overrun():.0f} s")
    _, _, regressed = report(record, args)
    if record.get("power_override"):
        print(f"\nsub0perf: {power.override_text(record.get('power') or {})}")
    return 0 if args.report_only or not regressed else 1


def main():
    # Keep the legacy benchmark parser/commands unchanged; workload evidence has its own
    # schema and budgets, reached through the same entry point used by all consumers.
    if sys.argv[1:2] == ["workload"]:
        from .workload import main as workload_main
        return workload_main(sys.argv[2:])
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["run", "check", "baseline", "auto", "accept", "gate", "install-hooks", "note-checkout", "selftest", "power", "workload"])
    parser.add_argument("--exe", help="benchmark executable (run, auto)")
    parser.add_argument("--out", help="samples file to write (run; optional for auto)")
    parser.add_argument("--results", help="samples file from `run` (check, baseline)")
    parser.add_argument("--baseline", help="perf/baselines/<host>.json")
    parser.add_argument("--filter", default="", help="only benchmarks whose name contains this text")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--max-rounds", type=int, default=0, help="auto: total rounds allowed for noisy benchmarks (default 2x)")
    parser.add_argument("--epochs", type=int, default=0, help="override nanobench epochs (quicker, noisier)")
    parser.add_argument("--level", choices=list(levels.LEVELS), default=None,
                        help="gate, auto: quick (default; a time-budgeted baseline of coverage) or full (everything, full rigour). "
                             "Precedence: --level, then SUB0MATH_PERF_LEVEL, then a pushed tag matching the full-ref patterns "
                             "(default v*, milestone/*), then quick")
    parser.add_argument("--budget", type=float, default=None, metavar="SECONDS",
                        help="gate, auto: target seconds of measuring. Quick defaults to SUB0MATH_PERF_BUDGET, the configured "
                             "quick budget, else 90; full has none unless given. 0 = no budget")
    parser.add_argument("--full-refs", action="append", default=[], metavar="PATTERNS",
                        help="install-hooks: comma-separated tag-name patterns whose push selects the full level (default v*,milestone/*)")
    parser.add_argument("--quick-budget", type=float, default=None, metavar="SECONDS",
                        help="install-hooks: the quick gate's budget for this repository")
    parser.add_argument("--confirm-rounds", type=int, default=4,
                        help="auto, gate: extra rounds for just the benchmarks that regressed; only regressions that persist count (0 = off)")
    parser.add_argument("--seed", type=int, default=1, help="shuffle seed of the rounds (recorded in the records)")
    parser.add_argument("--lock-timeout", type=float, default=900.0,
                        help="auto, gate: seconds to wait for another measuring run on this repository before giving up")
    parser.add_argument("--refs-from-stdin", action="store_true",
                        help="gate: read the pushed refs from stdin (pre-push hook); refuse a dirty tree or a HEAD that is not pushed")
    parser.add_argument("--max-load", type=float, default=0.20, help="busy fraction of all CPUs above which a round waits")
    parser.add_argument("--settle", type=float, default=30.0, help="seconds to wait for load to fall")
    parser.add_argument("--tolerance", type=float, default=0.10)
    parser.add_argument("--pair-tolerance", type=float, default=0.05)
    parser.add_argument("--instr-tolerance", type=float, default=0.02, help="instructions per op change that counts")
    parser.add_argument("--max-cv", type=float, default=0.05, help="spread above which a benchmark is inconclusive")
    parser.add_argument("--absolute", action="store_true", help="no calibration or drift correction")
    parser.add_argument("--report-only", action="store_true", help="never fail; report only")
    parser.add_argument("--reason", help="why a baseline is re-recorded or a change accepted (logged in the file)")
    parser.add_argument("--benchmark", nargs="+", default=[], help="accept: benchmarks to move to the new measurement")
    parser.add_argument("--all-changed", action="store_true", help="accept: every regressed or improved benchmark")
    parser.add_argument("--force-power", action="store_true",
                        help="baseline, accept: allow a record measured by hand on battery (never one measured under SUB0MATH_PERF_ON_BATTERY)")
    parser.add_argument("--record", action="store_true", help="auto: keep a clean tree's measurement as its commit's record")
    parser.add_argument("--repo", help="git repository (auto --record, gate, install-hooks, selftest)")
    parser.add_argument("--target", help="benchmark target name (auto --record, install-hooks; selftest: check just this target)")
    parser.add_argument("--build-dir", help="install-hooks: build directory the gate builds in")
    parser.add_argument("--pin", action="append", default=[], metavar="file=F,var=V[,repo=owner/name][,checkout=DIR][,name=N]",
                        help="install-hooks: a dependency pin (CMake `set(V <sha>)` in file F) the gate checks is on the dependency's "
                             "remote before a push that changes F; repeatable")
    parser.add_argument("--config", help="install-hooks: build configuration for multi-config generators")
    parser.add_argument("--cpu", type=int, default=-1,
                        help="logical CPU to pin measurements to (default: the last one; -2 = do not pin)")
    args = parser.parse_args()
    if not args.max_rounds:
        args.max_rounds = args.rounds * 2
    needs = {"run": ["exe", "out"], "check": ["results", "baseline"], "baseline": ["results", "baseline"], "auto": ["exe", "baseline"],
             "accept": ["results", "baseline", "reason"], "gate": [], "install-hooks": ["repo", "build_dir", "target", "exe", "baseline"],
             "note-checkout": [], "selftest": [], "power": []}
    if args.command == "accept" and not args.reason:
        args.reason = os.environ.get("SUB0MATH_PERF_REASON")
    missing = [f"--{k.replace('_', '-')}" for k in needs[args.command] if not getattr(args, k)]
    if missing:
        parser.error(f"{args.command} needs {' '.join(missing)}")
    if args.exe:
        args.exe = str(Path(args.exe).resolve())
    args.reason = args.reason or os.environ.get("SUB0MATH_PERF_REASON")  # lets build targets pass a reason
    platform_proc.set_pin_cpu(platform_proc.resolve_pin_cpu(args.cpu))
    if args.command == "gate":
        args.rounds = args.rounds if args.rounds != 5 else 3
        # Same method as baselines and records (default epochs), so absolute times compare like for like.
    commands = {"run": run_command, "check": check_command, "baseline": baseline_command, "auto": auto_command,
                "accept": accept_command, "gate": gate_command, "install-hooks": install_hooks_command,
                "note-checkout": note_checkout_command, "selftest": selftest_command, "power": power_command}
    return commands[args.command](args)
