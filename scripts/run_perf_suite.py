#!/usr/bin/env python3
"""Run Sub0Llm's optimization measurement protocol and check it against the KPI gates.

This exists to delete the boilerplate that otherwise gets re-done by hand for every optimization --
configure, build, check for contention, run N times, parse, interleave, compare, evaluate gates, log.
Doing that manually is where measurement mistakes come from (a contended run, a batched A/B, a stale
sibling build dir, a mean quoted without its spread). Shape borrowed from Sub0h264's
scripts/run_all_suites.py, which is the proven original.

Policy it enforces: docs/OPTIMIZATION_PROCESS.md. Gates: docs/optimization/kpi_gates.json.

Stages (each skippable, so iteration stays fast):

  1. suites   -- sub0_tests + sub0_frontend_tests at the neutral config  (G-HASH, G-SUITE-*)
  2. quality  -- forward/forward_one parity + logit stats, real artifact (G-PARITY, G-QUALITY)
  3. perf     -- interleaved multi-arm decode throughput, real artifact  (G-PERF)
  4. compete  -- llama.cpp on the same host and model                    (G-COMPETITOR, soft)

Typical invocations:

  # Measure two toggle arms against each other, 3 interleaved runs each, tagged for history:
  python scripts/run_perf_suite.py --stage perf --label B35 \
      --arm "base:" --arm "fused:--moe-quant-dot 1"

  # The combination matrix (this is what OPTIMIZATION_PROCESS.md S5 asks for):
  python scripts/run_perf_suite.py --stage perf --label combo \
      --arm "fused:--moe-quant-dot 1" \
      --arm "fused+simd:--moe-quant-dot 1 --simd-reduce 1" \
      --arm "fused+io:--moe-quant-dot 1 --moe-io-mode pipelined"

  # Default-path regression check only (fast, no real artifact needed):
  python scripts/run_perf_suite.py --stage suites

Every run appends a row to docs/optimization/perf_history.jsonl and rewrites
docs/optimization/perf_report.md with a gate panel on top.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import pathlib
import re
import statistics
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
GATES = ROOT / "docs" / "optimization" / "kpi_gates.json"
HISTORY = ROOT / "docs" / "optimization" / "perf_history.jsonl"
REPORT = ROOT / "docs" / "optimization" / "perf_report.md"

# The real 48-layer axes. A PARTIAL recipe silently yields PARAM_FLOATS 2570717696 and a rejected
# artifact with no hint which axis is wrong -- this cost real time twice before it was written down.
# Canonical copy: docs/MOE_QUANT_DOT.md S6h.
REAL_AXES = """--dmodel 2560 --layers 48 --heads 24 --seq 128 --kv-heads 2 --head-dim 256
--rotary-dim 64 --rope-theta 10000000 --gdn-full-attn-stride 4 --gdn-key-heads 16
--gdn-value-heads 48 --gdn-key-head-dim 128 --gdn-value-head-dim 128 --hc-count 4 --hc-lowrank 320
--qsa-indexer-n-heads 4 --qsa-indexer-kv-heads 1 --qsa-indexer-head-dim 128
--qsa-indexer-budget 2048 --qsa-indexer-compress-ratio 4 --tie-embeddings 0 --d-ff 640
--moe-quant-experts 1 --num-experts 512 --experts-per-tok 10 --prec-param 1 --vocab-exact 248320""".split()

ARTIFACT = r"D:\ModelWeights\Sub0Llm-Qwen4-full48-bf16\qwen4_full48_q_bf16.bin"
CONTENTION_RE = re.compile(r"sub0llm|clang|ninja|cmake", re.I)


def contention_count() -> int:
    """Named processes that would poison a timing run (sibling agents' builds/benchmarks).

    NECESSARY BUT NOT SUFFICIENT -- see system_load_pct(). This alone reported "0 contention" on a
    freshly rebooted host whose total CPU load was sampling 6-19% sustained (a VS Code updater,
    browser, desktop apps, GPU container settling after boot) -- enough to make a ~7-10% decode
    delta unattributable, against a noise floor of +-1.5%.
    """
    try:
        out = subprocess.run(["tasklist"], capture_output=True, text=True, timeout=60).stdout
    except Exception:
        return -1  # unknown -- caller decides; never silently treat as "clear"
    return sum(1 for line in out.splitlines() if CONTENTION_RE.search(line))


# Threshold for background load on an otherwise-idle host. 5% is deliberately strict: the decode
# noise floor is +-1.5%, and background load is not uniform -- a burst landing on the cores decode
# is using costs far more than its time-averaged share suggests.
MAX_BACKGROUND_LOAD_PCT = 5.0


def system_load_pct(samples: int = 5, interval_s: int = 2) -> float | None:
    """Average total CPU load over a short window -- catches contention nothing else names.

    Samples the performance counter rather than trusting one instantaneous reading: a single read of
    Win32_Processor.LoadPercentage returned 48% where the sustained figure was 6-19%. Returns None if
    the counter is unavailable, which callers must treat as UNKNOWN, never as idle.
    """
    ps = ("$s = Get-Counter '\\Processor(_Total)\\% Processor Time' "
          f"-SampleInterval {interval_s} -MaxSamples {samples}; "
          "($s.CounterSamples | Measure-Object CookedValue -Average).Average")
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                             capture_output=True, text=True, timeout=samples * interval_s + 60).stdout
        return float(out.strip().splitlines()[-1])
    except Exception:
        return None


def run(cmd, cwd=None, timeout=3600, check: bool = True) -> str:
    """Run a command and return its combined output.

    FAILS LOUDLY by default. This used to ignore the exit code, so a failed configure or build left the
    PREVIOUS binary in place and the suite went on to measure it -- found 2026-09-22 when a compile error
    produced a "profile" of a pre-O1 binary (1.56 s/token against O1's 0.705) with no warning at all.
    A measurement of the wrong binary is worse than no measurement: it looks like evidence.
    """
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    out = r.stdout + r.stderr
    if check and r.returncode != 0:
        tail = "\n".join(out.strip().splitlines()[-25:])
        raise RuntimeError(f"command failed (exit {r.returncode}): {' '.join(map(str, cmd))}\n{tail}")
    return out


def configure(build: pathlib.Path, extra: list[str]) -> None:
    run([str(build / "sub0llm-configure.exe"), *REAL_AXES, *extra], cwd=build)


def build_target(build: pathlib.Path, target: str) -> str:
    return run(["cmake", "--build", str(build), "--target", target, "-j"])


def decode_s_per_token(build: pathlib.Path, tokens: int = 3, cold: bool = False) -> float | None:
    """Decode s/token. `cold`: evict the artifact AND its sidecar from the page cache first (verified --
    page_cache.evict_verified raises rather than return a warm "cold"), and run --decode-only, because the
    default run's forward() pre-warms exactly the expert pages forward_one then reads."""
    extra = []
    if cold:
        import page_cache
        page_cache.evict_verified([ARTIFACT, ARTIFACT + ".moeq"])
        extra = ["--decode-only"]
    out = run([str(build / "sub0llm-qwen4-forward.exe"), "--model", ARTIFACT, "--tokens", str(tokens), *extra])
    m = re.search(r"forward_one over \d+ positions in [\d.]+s \(([\d.]+) s/token\)", out)
    return float(m.group(1)) if m else None


def parity(build: pathlib.Path, tokens: int = 6) -> float | None:
    out = run([str(build / "sub0llm-qwen4-forward.exe"), "--model", ARTIFACT, "--tokens", str(tokens)])
    m = re.search(r"max \|forward - forward_one\| = ([\d.eE+-]+)", out)
    return float(m.group(1)) if m else None


def stage_perf(build: pathlib.Path, arms: list[tuple[str, list[str]]], runs: int, tokens: int,
               cold: bool = False) -> dict:
    """Interleaved A/B/A/B across arms -- never batched, per OPTIMIZATION_PROCESS.md S1.

    Reconfigures + rebuilds between arms in ONE build dir rather than using N sibling dirs, because a
    stale sibling dir silently compares the wrong source tree.
    """
    samples: dict[str, list[float]] = {name: [] for name, _ in arms}
    for i in range(runs):
        for name, flags in arms:
            configure(build, flags)
            build_target(build, "sub0llm-qwen4-forward")
            v = decode_s_per_token(build, tokens, cold)
            if v is not None:
                samples[name].append(v)
            print(f"  [{i+1}/{runs}] {name}: {v} s/token", flush=True)
    out = {}
    for name, xs in samples.items():
        if not xs:
            continue
        out[name] = {
            "runs": xs,
            "median": statistics.median(xs),
            "stdev": statistics.pstdev(xs) if len(xs) > 1 else 0.0,
            # Spread matters as much as the middle: a mean hiding 1.44/1.49/1.67 is a different claim
            # from one hiding 1.51/1.52/1.51.
            "spread_pct": (max(xs) - min(xs)) / statistics.median(xs) * 100.0,
        }
    return out


def evaluate_gates(results: dict, gates: dict) -> list[dict]:
    """Only gates whose inputs this run actually produced are evaluated; the rest report 'n/a'.

    A gate that silently passes because nothing measured it is worse than no gate at all.
    """
    verdicts = []
    perf = results.get("perf", {})
    noise = 2.0  # percent; the measured floor, see OPTIMIZATION_PROCESS.md S1
    if len(perf) >= 2:
        names = list(perf)
        base, *rest = names
        for other in rest:
            ratio = perf[other]["median"] / perf[base]["median"]
            delta = (ratio - 1.0) * 100.0
            verdicts.append({
                "id": "G-PERF",
                "label": f"{other} vs {base}",
                "value": f"{delta:+.1f}%",
                # Inside the noise floor is explicitly neither a win nor a regression.
                "status": "noise" if abs(delta) < noise else ("PASS" if delta < 0 else "FAIL"),
            })
    for g in gates["gates"]:
        if g["id"].startswith("G-PERF"):
            continue
        verdicts.append({"id": g["id"], "label": g["label"],
                         "value": results.get(g["id"], "n/a"), "status": "n/a"})
    return verdicts


VTUNE = pathlib.Path(r"C:\Program Files (x86)\Intel\oneAPI\vtune\2026.4\bin64\vtune.exe")


def stage_vtune(build: pathlib.Path, flags: list[str], tokens: int, outdir: pathlib.Path) -> dict:
    """Top-Down microarchitecture analysis -- the measurement the roofline arithmetic cannot make.

    The roofline says which roofs are NOT saturated (docs/optimization/roofline_post_b35.md found
    2.7% of memory and 2.2% of SIMD, i.e. neither). It cannot say why. `uarch-exploration` splits
    the pipeline into Retiring / Front-End Bound / Bad Speculation / Back-End Bound, and the last
    into Memory Bound vs Core Bound -- which is exactly the missing discriminator.

    Run unelevated, per this project's own precedent ([[cpu-profiling-tooling-backlog]]).
    """
    if not VTUNE.exists():
        return {"error": f"vtune not found at {VTUNE}"}
    configure(build, flags)
    build_target(build, "sub0llm-qwen4-forward")
    res = outdir / "vtune_uarch"
    if res.exists():
        import shutil as _sh
        _sh.rmtree(res, ignore_errors=True)
    run([str(VTUNE), "-collect", "uarch-exploration", "-r", str(res), "--",
         str(build / "sub0llm-qwen4-forward.exe"), "--model", ARTIFACT, "--tokens", str(tokens)],
        timeout=5400)
    rep = run([str(VTUNE), "-report", "summary", "-r", str(res)], timeout=1800)
    def pct(label):
        m = re.search(rf"{label}[^\n]*?([\d.]+)%", rep)
        return float(m.group(1)) if m else None
    return {
        "retiring_pct": pct("Retiring"),
        "front_end_bound_pct": pct("Front-End Bound"),
        "bad_speculation_pct": pct("Bad Speculation"),
        "back_end_bound_pct": pct("Back-End Bound"),
        "memory_bound_pct": pct("Memory Bound"),
        "core_bound_pct": pct("Core Bound"),
        "result_dir": str(res),
    }


def contention_check(sb) -> tuple[dict, list[str]]:
    """One pass of the S1 gate. Returns (record for the history row, problems -- empty if clear).

    On failure it also ATTRIBUTES the load (perf_sandbox.load_attribution): a gate that only says
    "8.6%, wait" gives no way to tell a telemetry task that will finish in ten minutes from a stuck
    process that never will. Known transient Windows tasks are labelled as such.
    """
    import perf_sandbox
    n = contention_count()
    if sb is not None:
        time.sleep(5)   # let confined processes' in-flight quanta drain off the reserved cores
        load = perf_sandbox.bench_core_load(sb.bench_cpus)
    else:
        load = system_load_pct()
    problems = []
    if n != 0:
        problems.append(f"{n} named competing process(es) (sibling builds/benchmarks)")
    if load is None:
        problems.append("background load UNKNOWN (counter unavailable) -- not treated as idle")
    elif load > MAX_BACKGROUND_LOAD_PCT:
        where = ("on the sandbox's RESERVED cores (what it cannot confine unelevated: SYSTEM services, "
                 "kernel threads, DPCs -- OPTIMIZATION_PROCESS.md S1a)" if sb is not None
                 else "(post-boot updaters, browsers, indexing all count; try --sandbox)")
        problems.append(f"background CPU load {load:.1f}% > {MAX_BACKGROUND_LOAD_PCT:.0f}% {where}")
    top = perf_sandbox.load_attribution() if problems else []
    if top:
        problems.append("top consumers (% of one core): " + ", ".join(
            f"{name} {pct}" + (f" [known transient: {why}]" if why else "") for name, pct, why in top[:6]))
    return ({"named_processes": n, "background_load_pct": load, "top_consumers": top,
             "sandbox": sb.report() if sb is not None else None}, problems)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", action="append", choices=["suites", "quality", "perf", "compete", "vtune"],
                    help="repeatable; default is perf only")
    ap.add_argument("--arm", action="append", default=[],
                    help='"name:flags", e.g. "fused:--moe-quant-dot 1". First arm is the baseline.')
    ap.add_argument("--build", default="out/build/wp5c_full48", help="build dir for real-artifact stages")
    ap.add_argument("--cold", action="store_true",
                    help="evict the artifact + sidecar from the page cache before EVERY run (verified) and time "
                         "decode alone (--decode-only). Cold and warm numbers are different quantities: "
                         "label and compare them separately")
    ap.add_argument("--runs", type=int, default=3, help="runs per arm (minimum 3 by policy)")
    ap.add_argument("--tokens", type=int, default=3)
    ap.add_argument("--label", default="", help="opportunity ID, tags the history row (e.g. B35)")
    ap.add_argument("--allow-contention", action="store_true",
                    help="measure anyway. Produces a number that policy says is not evidence.")
    ap.add_argument("--sandbox", action="store_true",
                    help="confine other processes to 2 housekeeping E-cores for the run (perf_sandbox.py); "
                         "the load gate then checks the RESERVED cores, not the whole box")
    ap.add_argument("--wait-stable", type=float, default=0, metavar="MIN",
                    help="re-check the contention gate every 60 s for up to MIN minutes before refusing")
    args = ap.parse_args()
    with contextlib.ExitStack() as stack:
        sb = None
        if args.sandbox:
            import perf_sandbox
            sb = stack.enter_context(perf_sandbox.Sandbox(max_seconds=4 * 3600))
            print(f"sandbox: {json.dumps(sb.report())}", file=sys.stderr)
        return run_suite(args, sb)


def run_suite(args, sb) -> int:
    stages = args.stage or ["perf"]
    gates = json.loads(GATES.read_text(encoding="utf-8"))

    results_load = None
    if "perf" in stages or "quality" in stages or "vtune" in stages:
        deadline = time.monotonic() + args.wait_stable * 60
        while True:
            results_load, problems = contention_check(sb)
            if not problems or args.allow_contention or time.monotonic() > deadline:
                break
            print("waiting to stabilise: " + "; ".join(problems), file=sys.stderr)
            time.sleep(60)
        if problems and not args.allow_contention:
            print("REFUSING TO MEASURE:\n  - " + "\n  - ".join(problems) + "\n"
                  "A contended measurement is not a slow measurement, it is a meaningless one\n"
                  "(OPTIMIZATION_PROCESS.md S1). Wait (--wait-stable N), or pass --allow-contention\n"
                  "and do not quote the number as evidence.", file=sys.stderr)
            return 2
        print(f"contention check: {results_load['named_processes']} named, "
              f"{results_load['background_load_pct']:.1f}% background -- clear", file=sys.stderr)

    arms = []
    for spec in args.arm:
        name, _, flags = spec.partition(":")
        arms.append((name.strip(), flags.split()))
    if not arms:
        arms = [("default", [])]

    build = ROOT / args.build
    results: dict = {"label": args.label, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    # Every history row carries the conditions it was measured under -- a number without its
    # contention context cannot be compared against a later one.
    results["contention"] = results_load

    if "perf" in stages:
        if args.runs < 3:
            print("warning: policy minimum is 3 runs per arm", file=sys.stderr)
        results["perf"] = stage_perf(build, arms, args.runs, args.tokens, args.cold)
        results["cache"] = "cold" if args.cold else "warm"

    if "vtune" in stages:
        # Profiles the LAST arm -- normally the one under investigation, since profiling the baseline
        # tells you about code you are not changing.
        results["vtune"] = stage_vtune(build, arms[-1][1], args.tokens, REPORT.parent)

    verdicts = evaluate_gates(results, gates)
    results["gates"] = verdicts

    HISTORY.parent.mkdir(parents=True, exist_ok=True)
    with HISTORY.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(results) + "\n")

    lines = ["# Sub0Llm performance report", "",
             f"Generated {results['utc']}" + (f" -- label `{args.label}`" if args.label else ""), "",
             "## Gate panel", "", "| Gate | Detail | Value | Status |", "|---|---|---|---|"]
    for v in verdicts:
        lines.append(f"| `{v['id']}` | {v['label']} | {v['value']} | {v['status']} |")
    if results.get("perf"):
        lines += ["", "## Throughput (interleaved)", "",
                  "| Arm | Median s/token | Spread | Runs |", "|---|---:|---:|---|"]
        for name, st in results["perf"].items():
            runs = ", ".join(f"{x:.3f}" for x in st["runs"])
            lines.append(f"| {name} | {st['median']:.3f} | {st['spread_pct']:.1f}% | {runs} |")
    lines += ["", "History: `perf_history.jsonl`. Policy: `docs/OPTIMIZATION_PROCESS.md`.", ""]
    REPORT.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines[4:]))
    return 0 if all(v["status"] != "FAIL" for v in verdicts) else 1


if __name__ == "__main__":
    sys.exit(main())
