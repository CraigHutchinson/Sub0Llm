#!/usr/bin/env python3
"""Runs a benchmark executable under a profiler and prints the summary.

  profile.py <analysis> --exe <bench> [--filter <name>] [--seconds 10] [--out <dir>]

Analyses:
  vtune-hotspots     Intel VTune user-mode sampling: where time goes, by function and source line
  vtune-uarch        Intel VTune microarchitecture exploration: front/back-end bound, ports, memory
  vtune-memory       Intel VTune memory access: bandwidth, latency, cache misses
  advisor-roofline   Intel Advisor survey + FLOP trip counts: vectorization and roofline (HTML report)
  advisor-survey     Intel Advisor survey only: which loops vectorized, and why not
  perf-record        Linux perf sampling with call graphs
  perf-stat          Linux perf hardware counters (cycles, IPC, branch and cache misses)

The benchmark runs with `--soak <seconds>` so the selected hot path dominates the samples. Build with
the `profile` preset first: optimised code with debug info and frame pointers (and ITT task markers,
which VTune shows on its timeline). Results go to <out>/<analysis>-<filter>-<timestamp>.

Tools are found on PATH, then via VTUNE_PROFILER_DIR / ADVISOR_DIR / ONEAPI_ROOT, then in the default
oneAPI install locations.
"""

import argparse
import datetime
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path


def find_intel_tool(name, env_var, product):
    found = shutil.which(name)
    if found:
        return found
    roots = [os.environ.get(env_var)]
    if os.environ.get("ONEAPI_ROOT"):
        roots.append(str(Path(os.environ["ONEAPI_ROOT"]) / product / "latest"))
    if platform.system() == "Windows":
        roots.append(rf"C:\Program Files (x86)\Intel\oneAPI\{product}\latest")
    else:
        roots += [f"/opt/intel/oneapi/{product}/latest", str(Path.home() / f"intel/oneapi/{product}/latest")]
    for root in filter(None, roots):
        for sub in ("bin64", "bin"):
            candidate = Path(root) / sub / (name + (".exe" if platform.system() == "Windows" else ""))
            if candidate.is_file():
                return str(candidate)
    return None


def run(cmd, cwd=None):
    print("+ " + " ".join(f'"{c}"' if " " in c else c for c in cmd), flush=True)
    return subprocess.run(cmd, cwd=cwd).returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("analysis", choices=["vtune-hotspots", "vtune-uarch", "vtune-memory", "advisor-roofline",
                                             "advisor-survey", "perf-record", "perf-stat"])
    parser.add_argument("--exe", required=True, help="benchmark executable built with tools/perf/bench_runner.hpp")
    parser.add_argument("--filter", default="", help="benchmark name substring; empty runs every benchmark")
    parser.add_argument("--seconds", type=float, default=10.0, help="soak time per selected benchmark")
    parser.add_argument("--out", default="profiles")
    args = parser.parse_args()

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    label = re.sub(r"[^A-Za-z0-9]+", "_", args.filter).strip("_") or "all"
    result = Path(args.out).resolve() / f"{args.analysis}-{label}-{stamp}"
    result.parent.mkdir(parents=True, exist_ok=True)
    workload = [str(Path(args.exe).resolve()), "--soak", str(args.seconds)]
    # Profilers drop state files (e.g. Advisor's config/state.cfg) in their working directory.
    os.chdir(result.parent)
    if args.filter:
        workload += ["--filter", args.filter]

    if args.analysis.startswith("vtune"):
        vtune = find_intel_tool("vtune", "VTUNE_PROFILER_DIR", "vtune")
        if not vtune:
            sys.exit("VTune not found: install Intel VTune Profiler or put `vtune` on PATH")
        collect = {"vtune-hotspots": "hotspots", "vtune-uarch": "uarch-exploration",
                   "vtune-memory": "memory-access"}[args.analysis]
        extra = ["-knob", "sampling-mode=sw"] if collect == "hotspots" else []
        code = run([vtune, "-collect", collect, *extra, "-result-dir", str(result), "--", *workload])
        if code == 0:
            run([vtune, "-report", "summary", "-result-dir", str(result)])
            run([vtune, "-report", "hotspots", "-result-dir", str(result), "-format", "text",
                 "-report-output", str(result / "hotspots.txt"), "-group-by", "function"])
            print(f"\nOpen in the GUI: vtune-gui {result}")
        return code

    if args.analysis.startswith("advisor"):
        advisor = find_intel_tool("advisor", "ADVISOR_DIR", "advisor")
        if not advisor:
            sys.exit("Intel Advisor not found: install it or put `advisor` on PATH")
        code = run([advisor, "--collect=survey", f"--project-dir={result}", "--", *workload])
        if code != 0:
            return code
        if args.analysis == "advisor-roofline":
            # Advisor exits non-zero on warnings (e.g. modules without debug info); the data is still usable.
            if run([advisor, "--collect=tripcounts", "--flop", f"--project-dir={result}", "--", *workload]) != 0:
                print("advisor: trip-count collection reported problems; roofline may be incomplete")
            run([advisor, "--report=roofline", f"--project-dir={result}", f"--report-output={result / 'roofline.html'}"])
        run([advisor, "--report=survey", f"--project-dir={result}", "--format=text",
             f"--report-output={result / 'survey.txt'}"])
        print(f"\nSurvey: {result / 'survey.txt'}\nOpen in the GUI: advisor-gui {result}")
        return code

    perf = shutil.which("perf")
    if not perf:
        sys.exit("perf not found (Linux only): install linux-tools for this kernel")
    if args.analysis == "perf-stat":
        return run([perf, "stat", "-d", "--", *workload])
    result.mkdir(parents=True, exist_ok=True)
    code = run([perf, "record", "-g", "--call-graph", "dwarf", "-o", str(result / "perf.data"), "--", *workload])
    if code == 0:
        run([perf, "report", "--stdio", "--no-children", "-i", str(result / "perf.data"), "--percent-limit", "1"])
    return code


if __name__ == "__main__":
    sys.exit(main())
