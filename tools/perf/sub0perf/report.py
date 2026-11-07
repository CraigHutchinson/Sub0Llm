"""The tables: the markdown report of a judged run, and the job summary."""

import os
from pathlib import Path

from . import power
from .judge import evaluate_with, verdict_hits


def _bimodal_label(modes, verdict):
    """' bimodal' for samples at two levels; a regression or improvement that stays says both levels moved."""
    if not modes:
        return ""
    return " bimodal, both levels moved" if verdict in ("REGRESSED", "IMPROVED") else " bimodal"


def render(rows, pair_rows, env, baseline_path, tolerance, pair_tolerance):
    pct = lambda x: "-" if x is None else f"{x:+.1%}"
    num = lambda x: "-" if x is None else f"{x:,.1f}"
    lines = [f"**Environment:** {env['rounds']} interleaved rounds; load max {env['max_load']:.0%}"
             f" ({env['busy_rounds']} round(s) above the limit); calibration {env['calibration_ns']:,.0f} ns,"
             f" spread {env['calibration_cv']:.1%}, clock drift first to last {env['clock_drift']:+.1%}."]
    shift = env.get("environment_shift")
    if shift is not None and abs(shift) > env.get("tolerance", 0.1) / 2:
        lines.append(f"> The environment shifted {shift:+.1%} beyond the clock between the two measurements (seen in the"
                     " controls, which do not use the library). It is factored out, but absolute verdicts are less"
                     " certain than usual; pair ratios and instruction counts are unaffected.")
    if env.get("power_override"):
        lines.append(f"> **{power.override_text(env.get('power') or {})}**  (power: {power.label(env.get('power'))}, override: true)")
    if env.get("method"):
        lines.append(f"> Method mismatch: {env['method']}.")
    if abs(env["clock_drift"]) > 0.05:
        lines.append(f"> Clock drift of {env['clock_drift']:+.1%} during the run (thermal or power state); rounds are"
                     " scaled by calibration, but treat close calls with care.")
    if not env["baseline"]:
        lines.append(f"No baseline at `{baseline_path}`: run `bench_check.py baseline` on this host to adopt the gate."
                     " Pair ratios below are still meaningful without one.")
    else:
        lines.append(f"Judged {env['mode']}; tolerance {tolerance:.0%} or 3x the noise, whichever is larger.")
        if env.get("suite_spread") is not None:
            lines.append(f"No controls: suite spread {env['suite_spread']:.1%} used as an extra noise floor"
                         f" (a change must also exceed 3x it, {3 * env['suite_spread']:.1%}).")

    def section(title, wanted):
        picked = [r for r in rows if r[5] in wanted]
        if not picked:
            return
        lines.append(f"\n**{title}**\n")
        lines.append("| Benchmark | Baseline ns | Now ns | Raw | Judged | Spread | Verdict |")
        lines.append("|---|---:|---:|---:|---:|---:|---|")
        for name, b, s, raw, change, v in picked:
            spread = "-" if not s else format(s["cv"], ".1%") + _bimodal_label(s.get("modes"), v)
            base_ns = num(b["median_ns"] if b else None)
            now_ns = num(s["median_ns"] if s else None)
            lines.append(f"| {name} | {base_ns} | {now_ns} | {pct(raw)} | {pct(change)} | {spread} | {v} |")

    section("Regressions", {"REGRESSED"})
    section("Progressions (record them: run the baseline target and commit it, saying why)", {"IMPROVED"})
    section("Inconclusive: too noisy to judge, or at two timing levels (bimodal)", {"NOISY"})
    section("Inconclusive: the environment shifted beyond the tolerance (see the controls); pairs and counters still judge",
            {"ENVIRONMENT"})
    section("Unchanged, new or missing", {"ok", "new", "missing"})

    counter_rows = env.get("counter_rows", [])
    if counter_rows:
        lines.append(f"\n**Hardware counters per op** (instructions tolerance {env['instr_tolerance']:.0%};"
                     " cycles do not depend on clock speed)\n")
        lines.append("| Benchmark | Instructions | Change | Cycles | Change | IPC | Branch misses | Verdict |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
        for row in counter_rows:
            now = row["now"]
            ins, cyc = now.get("instructions"), now.get("cycles")
            ipc = f"{ins / cyc:.2f}" if ins and cyc else "-"
            miss = f"{now['branch_misses'] / now['branches']:.2%}" if now.get("branches") and "branch_misses" in now else "-"
            lines.append(f"| {row['name']} | {num(ins)} | {pct(row.get('instructions_change'))} | {num(cyc)}"
                         f" | {pct(row.get('cycles_change'))} | {ipc} | {miss} | {', '.join(row['verdicts']) or 'ok'} |")
    else:
        lines.append("\nHardware counters: not available on this platform (Linux perf_event provides instructions and cycles).")

    if pair_rows:
        lines.append(f"\n**Pairs against hand-written references** (ratio > 1 means slower than the reference;"
                     f" pair tolerance {pair_tolerance:.0%})\n")
        lines.append("| Benchmark | Reference | Ratio now | Baseline ratio | Change | Spread | Instruction ratio | Verdict |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---|")
        for name, p, bp, change, v in pair_rows:
            base_ratio = "-" if not bp else format(bp["ratio"], ".3f") + "x"
            instr = format(p["instruction_ratio"], ".3f") + "x" if "instruction_ratio" in p else "-"
            lines.append(f"| {name} | {p['reference']} | {p['ratio']:.3f}x | {base_ratio} | {pct(change)} | {p['cv']:.1%}{_bimodal_label(p.get('modes'), v)} | {instr} | {v} |")
    return "\n".join(lines)


def report(record, args):
    rows, pair_rows, env = evaluate_with(record, args.baseline, args)
    text = render(rows, pair_rows, env, args.baseline, args.tolerance, args.pair_tolerance)
    print(text)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as f:
            f.write(f"### {record.get('exe', 'Benchmarks')} against {Path(args.baseline).name}\n\n{text}\n\n")
    regressed = len(verdict_hits(rows, pair_rows, env, ["REGRESSED"]))
    improved = len(verdict_hits(rows, pair_rows, env, ["IMPROVED"]))
    noisy = len(verdict_hits(rows, pair_rows, env, ["NOISY", "ENVIRONMENT"]))
    print(f"\nperf: {regressed} regressed, {improved} improved, {noisy} inconclusive")
    return rows, pair_rows, regressed
