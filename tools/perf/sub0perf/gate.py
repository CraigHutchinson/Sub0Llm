"""The pre-push gate: refuse a push that is not what was built, measure HEAD, block on regressions that persist."""

import os
import subprocess
import sys
from pathlib import Path

from .baseline import baseline_data, make_record
from . import deferral, levels, plan as planning, power, speed
from .common import REFERENCE_MARK, group_of, read_json, tooling_script
from .confirm import confirm_regressions, inconclusive_in, retry_inconclusive
from .gitstate import (ZERO_SHA_CHARS, accepted_since, changed_paths_between, documentation_only, git, git_head, log_event,
                       parse_push_refs, perf_dir, pushed_refs, push_problems, reference_binary, reference_record, save_record, tree_clean)
from .judge import evaluate_with, names_with
from .lock import LockTimeout, PerfLock
from .pins import pin_problems, show_file_at
from .report import render, report
from .reproduce import reproduced_against
from .runner import PowerChanged, list_names, measure_all, probe_cost, probe_unknown, rerunner, set_round_guard
from .stats import rounds_needed


def check_pins(repo, config, refs):
    """Refuses a push whose pinned dependency commits are not on the dependencies' remotes. True when the push may go on.

    Runs before everything else, also when the performance gate is bypassed or the push is documentation only (a pin
    file is not documentation): `SUB0MATH_PINS_SKIP=<reason>` bypasses it, logged.
    """
    pins = config.get("pins", [])
    if not pins or not pushed_refs(refs):
        return True
    reason = os.environ.get("SUB0MATH_PINS_SKIP")
    if reason:
        print(f"sub0perf: dependency pin check skipped (SUB0MATH_PINS_SKIP={reason})")
        log_event(repo, {"event": "pins-skip", "reason": reason})
        return True
    problems, warnings = pin_problems(repo, pins, refs, lambda remote, local: changed_paths_between(repo, remote, local),
                                      show_file_at(repo))
    for warning in warnings:
        print(f"sub0perf: WARNING: {warning}")
    if warnings:
        log_event(repo, {"event": "pins-unverified", "warnings": warnings})
    if problems:
        print("sub0perf: push refused: a pinned dependency is not on its remote:")
        for problem in problems:
            print(f"  - {problem}")
        print("  (SUB0MATH_PINS_SKIP=<reason> bypasses this check and logs the reason; git push --no-verify skips all hooks)")
        log_event(repo, {"event": "pins-refused", "problems": problems})
        return False
    return True


def gate_command(args):
    repo = Path(args.repo or git(".", "rev-parse", "--show-toplevel") or ".")
    d = perf_dir(repo)
    config_path = d / "config.json" if d else None
    config = read_json(config_path) if config_path and config_path.is_file() else None
    refs = parse_push_refs(sys.stdin.read()) if getattr(args, "refs_from_stdin", False) else []
    # plan_only (selftest): take every decision the gate takes, then stop before the pins' network check, the build and any
    # measuring; sets args.outcome (why it stopped) and args.previews (the quick plan per target). `args.changed_paths`
    # may replace the git diff of a pushed range; the environment bypass and the tree state are not part of a decision test.
    plan_only = getattr(args, "plan_only", False)
    args.outcome = ""
    changed_paths = getattr(args, "changed_paths", None) or (lambda remote, local: changed_paths_between(repo, remote, local))
    log = (lambda *_: None) if plan_only else (lambda event: log_event(repo, event))
    if config and not plan_only and not check_pins(repo, config, refs):
        return 1
    skip = None if plan_only else os.environ.get("SUB0MATH_PERF_SKIP")
    if skip:
        print(f"sub0perf: pre-push gate skipped (SUB0MATH_PERF_SKIP={skip})")
        log({"event": "skip", "commit": git_head(repo), "reason": skip})
        args.outcome = "skip-env"
        return 0
    if config is None:
        print("sub0perf: no benchmarks configured for this repository; nothing to gate")
        args.outcome = "no-config"
        return 0
    try:
        levels.resolve(args, refs, config)
    except ValueError as error:
        print(f"sub0perf: {error}; push blocked")
        return 1
    print(f"sub0perf: level {args.level} ({args.level_why}); "
          f"{'budget ' + format(args.budget_seconds, 'g') + ' s (' + args.budget_why + ')' if args.budget_seconds else 'no time budget'}")
    head = git_head(repo)
    clean = True if plan_only else tree_clean(repo)
    if args.refs_from_stdin:  # a push: what is measured must be what is pushed
        if not refs:
            print("sub0perf: nothing is pushed (everything up to date); nothing to measure")
            args.outcome = "skip-empty"
            return 0
        if all(set(r[1]) == ZERO_SHA_CHARS for r in refs):
            print("sub0perf: only deletions are pushed; nothing to measure")
            args.outcome = "skip-deletions"
            return 0
        problems = push_problems(refs, head, clean, lambda sha: git(repo, "rev-parse", f"{sha}^{{commit}}") or sha)
        if problems:
            print("sub0perf: push refused by the performance gate:")
            for problem in problems:
                print(f"  - {problem}")
            print("  (SUB0MATH_PERF_SKIP=<reason> bypasses the gate and logs the reason; git push --no-verify skips all hooks)")
            log({"event": "gate-refused", "commit": head, "problems": problems})
            args.outcome = "refused"
            return 1

        if documentation_only(refs, changed_paths):
            print("sub0perf: the push changes documentation only; nothing to measure")
            log({"event": "skip", "commit": head, "reason": "documentation only"})
            args.outcome = "skip-docs"
            return 0
    pol, state = power.policy(config=config), power.power_state()
    if not state.untrusted:  # battery defers anyway; a throttled mains machine is found by timing the calibration kernel, before any build
        found = speed.check_targets([(name, t["exe"], t["baseline"]) for name, t in config.get("targets", {}).items()], config=config)
        for note in found.notes:
            print(note)
        state = power.with_speed(state, found.ratio, found.limit, found.sustained)
    for note in power.environment_notes(state, pol):
        print(note)
    if state.untrusted and not pol.override:  # before any build or measuring: a battery timing is not comparable with the baseline
        print(power.deferral_text(state, pol.strict))
        if not plan_only:
            if not pol.strict:
                deferral.add(d, head, refs, state.reason())
            log({"event": "power-refused" if pol.strict else "power-deferred", "commit": head, "power": state.as_dict()})
        args.outcome = "refused-power" if pol.strict else "defer-power"
        return 1 if pol.strict else 0
    args.power_state, args.power_policy = state, pol
    if plan_only:
        args.outcome = "measure"
        args.previews = preview_targets(args, repo, d, config, head)
        return 0
    if not pol.override:
        set_round_guard(power_guard)
    try:
        with PerfLock(repo, "gate", args.lock_timeout):
            return run_gate(args, repo, d, config_path, head, clean)
    except LockTimeout as e:
        print(f"sub0perf: {e}; push blocked")
        return 1
    except PowerChanged as changed:
        return stopped_by_power(changed.args[0], pol, repo, d, head, refs, log)
    finally:
        set_round_guard(None)


def power_guard():
    """Called before every round of a gate run: stops the run when the machine has gone on battery since it began."""
    now = power.power_state()
    if now.untrusted:
        raise PowerChanged(now)


def stopped_by_power(now, pol, repo, d, head, refs, log):
    """Exit code after the power state changed mid-run: the run is inconclusive and nothing it measured is kept or judged."""
    print(f"\nsub0perf: the power state changed during the run ({now.reason()}); the measurements are "
          f"inconclusive and are discarded (no verdicts, no record, baseline untouched)")
    print(power.deferral_text(now, pol.strict))
    if not pol.strict:
        deferral.add(d, head, refs, now.reason(), partial=True)
    log({"event": "power-stopped", "commit": head, "power": now.as_dict(), "strict": pol.strict})
    return 1 if pol.strict else 0


def probe_unknown_costs(exe, args, state, all_names):
    """Times each benchmark that has no recorded cost, within a share of the budget. Returns the seconds spent.

    A new benchmark may cost minutes per sample, and its cost is not known until it has run; so one sample of each is
    timed first, cut off at a tenth of the budget. What is slower is recorded as slow and what the allowance did not
    reach stays unknown; neither is planned (see planning.unplanned), both wait for the full level or a later probe.
    """
    unknown = [n for n in all_names if n not in state["costs"] and n not in state.get("slow", {})]
    if not unknown:
        return 0.0
    first = not state["costs"]
    spent = 0.0
    if first:  # nothing timed yet: also time a round's calibration
        _, overhead = probe_cost(exe, (planning.choose_controls(all_names, {}, (), 1) or all_names)[0], args)
        state["round_overhead"] = overhead
        spent += overhead / 2
    refs_first = sorted(unknown, key=lambda n: REFERENCE_MARK not in n)  # controls come from the references
    costs, slow, took = probe_unknown(exe, refs_first, args, args.budget_seconds * planning.PROBE_SHARE,
                                      args.budget_seconds * planning.PROBE_CAP_SHARE)
    state["costs"].update(costs)
    state.setdefault("slow", {}).update(slow)
    print(f"sub0perf: timed {len(costs)} of {len(unknown)} benchmark(s) with no recorded cost in {took:.0f} s"
          + (f"; {len(slow)} slower than {args.budget_seconds * planning.PROBE_CAP_SHARE:.0f} s per sample, left to the full level" if slow else ""))
    return spent + took


def choose_benchmarks(args, exe, target, state, head, all_names, baseline=None):
    """(benchmarks to measure, rounds, seconds spent probing) for this level: everything at full, a plan that fits the time budget at quick."""
    rounds = args.rounds
    if args.level != levels.QUICK:
        return all_names, rounds, 0.0
    if not args.budget_seconds:
        print(f"sub0perf: quick level without a time budget: all {len(all_names)} benchmarks, {rounds} rounds")
        return all_names, rounds, 0.0
    spent = probe_unknown_costs(exe, args, state, all_names)
    plan, rounds = make_quick_plan(args, state, head, all_names, spent, baseline)
    for line in planning.describe_plan(plan, len(all_names), state, head):
        print(line)
    return plan.selected, rounds, spent


def make_quick_plan(args, state, head, all_names, spent, baseline=None):
    """(Plan, rounds) for a quick gate that has spent `spent` seconds probing: the part of choose_benchmarks that needs no benchmark."""
    rounds = max(args.rounds, planning.QUICK_ROUNDS)
    known = read_json(baseline).get("benchmarks", {}) if baseline and Path(baseline).is_file() else {}
    bimodal = [n for n, b in known.items() if isinstance(b, dict) and b.get("modes")]
    plan = planning.plan_quick(all_names, state, head, max(args.budget_seconds - spent, 1.0), rounds, avoid=bimodal,
                               defer=planning.unplanned(all_names, state))
    return plan, rounds


def probe_allowance(budget, state, all_names):
    """The most seconds the probe of benchmarks with no recorded cost can take: its share of the budget, plus half a
    round's calibration when nothing has been timed yet (see probe_unknown_costs). 0 when every benchmark has a cost."""
    unknown = [n for n in all_names if n not in state["costs"] and n not in state.get("slow", {})]
    if not unknown:
        return 0.0
    extra = state.get("round_overhead", planning.FALLBACK_ROUND_OVERHEAD) / 2 if not state["costs"] else 0.0
    return budget * planning.PROBE_SHARE + extra


def preview_targets(args, repo, d, config, head):
    """One dict per target: the quick plan the gate would make now, from the recorded costs and coverage, without
    building, listing, probing or measuring. The benchmark names are those the baseline and the coverage state know."""
    previews = []
    for target, t in config.get("targets", {}).items():
        state = planning.load_state(d, target)
        known = read_json(t["baseline"]).get("benchmarks", {}) if Path(t["baseline"]).is_file() else {}
        names = sorted(set(known) | set(state["costs"]) | set(state.get("slow", {})) | set(state["benchmarks"]))
        preview = {"target": target, "state": state, "names": names, "level": args.level, "budget": args.budget_seconds,
                   "allowance": 0.0, "plan": None, "rounds": args.rounds}
        if args.level == levels.QUICK and args.budget_seconds and names:
            preview["allowance"] = probe_allowance(args.budget_seconds, state, names)
            preview["plan"], preview["rounds"] = make_quick_plan(args, state, head, names, preview["allowance"], t["baseline"])
            for line in planning.describe_plan(preview["plan"], len(names), state, head):
                print(line)
        previews.append(preview)
    return previews


def run_gate(args, repo, d, config_path, head, clean):
    config = read_json(config_path)
    upstream = git(repo, "rev-parse", "@{upstream}")
    blocked = []
    power_now, pol = args.power_state, args.power_policy
    pending = deferral.load(d)
    for line in deferral.describe(pending):
        print(line)
    for target, t in config.get("targets", {}).items():
        print(f"sub0perf: pre-push gate for {target} at {head[:9]}{'' if clean else ' (working tree has uncommitted changes)'}")
        command = ["cmake", "--build", t["build_dir"], "--target", target] + (["--config", t["config"]] if t.get("config") else [])
        build = subprocess.run(command, env={**os.environ, "SUB0MATH_PERF_SKIP": "1"})
        if build.returncode != 0:
            print(f"sub0perf: building {target} failed; push blocked")
            blocked.append(target)
            continue
        exe = t["exe"]
        args.filter = ""
        quick = args.level == levels.QUICK
        state = planning.load_state(d, target)
        timings = planning.new_timings()
        all_names = list_names(exe, "")
        chosen, rounds, probed_s = choose_benchmarks(args, exe, target, state, head, all_names, t["baseline"])
        planning.save_state(d, target, state)  # what the probe learned survives a gate that is stopped
        args.subset = len(chosen) < len(all_names)
        # Measuring time, probing included; the build above does not count.
        budget = planning.Budget(max(args.budget_seconds - probed_s, 0.0) if args.budget_seconds else args.budget_seconds)
        samples, meta, positions, counters = measure_all(exe, args, names=chosen, rounds=rounds, timings=timings)
        record = make_record(exe, samples, meta, positions, counters, args.epochs, args.seed, power_now)
        record["level"] = args.level
        if power_now.untrusted:  # only under SUB0MATH_PERF_ON_BATTERY: never a baseline, and every report says so
            record["power_override"] = True
        record["unmeasured"] = sorted(set(all_names) - set(chosen))
        args.baseline = t["baseline"]
        ref_sha, ref = reference_record(repo, target, upstream)

        def both(record=record, ref=ref, baseline=t["baseline"]):
            found = [evaluate_with(record, baseline, args)]
            if ref:
                found.append(evaluate_with(record, baseline_data(ref), args))
            return found

        rerun = rerunner(exe, args, samples, meta, positions, counters, timings=timings)
        more_for = lambda names, record=record: rounds_needed(record, names)

        def affordable(names, extra, record=record, state=state, budget=budget):
            return budget.affords(planning.pass_estimate(names, max(extra, rounds_needed(record, names)), state))

        # A benchmark the baseline knows to sit at two levels is measured for enough rounds from the start for the
        # two-level test to work, instead of finding out only when it looks regressed.
        known = read_json(t["baseline"]).get("benchmarks", {}) if Path(t["baseline"]).is_file() else {}
        groups = {group_of(n) for n, b in known.items() if isinstance(b, dict) and b.get("modes") and n in samples}
        bimodal = {n for n in samples if group_of(n) in groups}
        # Full measures them that long from the start; quick only when a verdict turns out to depend on it (the
        # confirmation and retry passes below always gather that many samples for what they re-measure).
        if args.level == levels.FULL and bimodal and more_for(bimodal):
            print(f"sub0perf: {len(bimodal)} benchmark(s) are known to be bimodal; measuring them for {more_for(bimodal)} more round(s)")
            rerun(bimodal, more_for(bimodal))

        # Inconclusive (noisy, environment-shifted) results are measured once more with twice the rounds, and a push
        # is never blocked by what stays inconclusive. One noisy measurement never blocks either: whatever regressed
        # is re-measured, and only what persists counts.
        retried, _ = retry_inconclusive(both, rerun, args.rounds, more_for, affordable if quick and args.budget_seconds else None)
        # A regression is always confirmed before it blocks, even if that overruns a quick budget.
        first_regressed, _ = confirm_regressions(both, rerun, args.confirm_rounds, more_for)
        inconclusive = sorted(inconclusive_in(both())[0])
        measured_s = budget.elapsed() + probed_s
        planning.absorb_timings(state, timings)
        planning.update_recheck(state, planning.units_of(all_names), chosen, set(first_regressed) | set(inconclusive))
        planning.mark_measured(state, chosen, head)
        planning.save_state(d, target, state)
        print(f"\nsub0perf: measured {len(chosen)}/{len(all_names)} benchmarks at the {args.level} level in {measured_s:.0f} s"
              + (f" (budget {args.budget_seconds:g} s)" if args.budget_seconds else ""))
        if budget.overrun():
            print(f"sub0perf: the time budget was exceeded by {budget.overrun():.0f} s (confirming a regression, or the always-measured "
                  f"pairs and controls, take precedence over the budget)")
        record["inconclusive"] = inconclusive
        record["inconclusive_retried"] = sorted(retried)
        save_record(repo, target, record, exe=exe)  # the record before push (with any extra rounds), and the build that made it
        print("\n### Against the committed baseline\n")
        rows, pair_rows, regressed_base = report(record, args)
        regressed_ref = []
        if ref:
            print(f"\n### Against the local record of {ref_sha[:9]} (like for like on this machine)\n")
            ref_rows, ref_pairs, env = evaluate_with(record, baseline_data(ref), args)
            print(render(ref_rows, ref_pairs, env, f"record {ref_sha[:9]}", args.tolerance, args.pair_tolerance))
            regressed_ref = names_with(ref_rows, ref_pairs, env, "REGRESSED")
        else:
            print("\nsub0perf: no local record of an earlier commit; it is made by the first clean build after a checkout")
        baseline_rel = os.path.relpath(t["baseline"], repo).replace("\\", "/")
        accepted = accepted_since(repo, baseline_rel, ref_sha)
        unaccepted = [n for n in regressed_ref if n not in accepted and "*" not in accepted]
        unreproduced = []
        if unaccepted:
            unaccepted, unreproduced = reproduce_vs_record(repo, target, exe, ref_sha, unaccepted, args)
        if regressed_base or unaccepted:
            blocked.append(target)
            names_text = ", ".join(sorted(set(unaccepted) | {r[0] for r in rows if r[5] == "REGRESSED"}
                                         | {p[0] for p in pair_rows if p[4] == "REGRESSED"}))
            print(f"\nsub0perf: push blocked by regressions in {target}: {names_text}")
            print("  Fix them, or accept them explicitly and commit the baseline:")
            print(f'    python "{tooling_script()}" accept --results "{d / "records" / target / (head + ".json")}"'
                  f' --baseline "{t["baseline"]}" --all-changed --reason "why this is acceptable"')
        if unreproduced:
            print(f"\nsub0perf: WARNING: {len(unreproduced)} difference(s) from the record of {ref_sha[:9]} were not reproduced in "
                  f"this session, so they do not block the push: {', '.join(sorted(unreproduced))}")
        if inconclusive:
            print(f"\nsub0perf: WARNING: {len(inconclusive)} benchmark(s) of {target} stayed inconclusive after re-measuring with "
                  f"{args.rounds} more round(s); nothing was proven about them{'' if target in blocked else ', and the push is allowed'}:")
            for name in inconclusive:
                print(f"  - {name}")
            print("  (recorded in the local record and log; re-run the check on a quieter machine to judge them)")
        log_event(repo, {"event": "gate", "target": target, "commit": head, "reference": ref_sha,
                         "blocked": target in blocked, "regressed_vs_baseline": regressed_base, "unaccepted_vs_record": unaccepted, "unreproduced_vs_record": unreproduced,
                         "inconclusive": inconclusive, "power": power_now.as_dict(), "power_override": power_now.untrusted, "seed": args.seed, "level": args.level,
                         "measured": len(chosen), "benchmarks": len(all_names), "seconds": round(measured_s, 1)})
    if blocked and pending:
        print("\nsub0perf: the push is blocked by a regression; earlier pushes were not measured, so it may come from any of them:")
        for line in deferral.describe(pending)[1:]:
            print(line)
    elif not blocked and not power_now.untrusted:
        deferral.clear(d)  # measured on mains and fine: the unmeasured range is covered
    if power_now.untrusted:  # repeated at the end, where the verdict is read
        print(f"\nsub0perf: {power.override_text(power_now)}")
    return 1 if blocked else 0


def reproduce_vs_record(repo, target, exe, ref_sha, names, args):
    """Splits regressions against the local record into (those that reproduce, those that do not).

    The record was measured in an earlier session, so a difference from it can be the session and not the code. It
    blocks only when the record's own build, kept beside the record, is measured again in this session alternately with
    the new build and the difference is still there. Without that build nothing can be shown, so none of them block.
    """
    old = reference_binary(repo, target, ref_sha)
    if old is None:
        print(f"\nsub0perf: the build of {ref_sha[:9]} was not kept, so its differences cannot be re-measured side by side")
        return [], list(names)
    print(f"\nsub0perf: re-measuring the kept build of {ref_sha[:9]} beside the new one for {len(names)} difference(s) from its record")
    again, gone, why = reproduced_against(exe, str(old), names, args)
    if why:
        print(f"sub0perf: {why}")
    print(f"sub0perf: reproduced side by side: {', '.join(sorted(again)) or 'none'}")
    print(f"sub0perf: not reproduced (same times in this session): {', '.join(sorted(gone)) or 'none'}")
    return sorted(again), sorted(gone)
