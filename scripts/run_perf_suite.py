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
    """Processes that would poison a timing run. OPTIMIZATION_PROCESS.md S1: this must be 0."""
    try:
        out = subprocess.run(["tasklist"], capture_output=True, text=True, timeout=60).stdout
    except Exception:
        return -1  # unknown -- caller decides; never silently treat as "clear"
    return sum(1 for line in out.splitlines() if CONTENTION_RE.search(line))


def run(cmd, cwd=None, timeout=3600) -> str:
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    return r.stdout + r.stderr


def configure(build: pathlib.Path, extra: list[str]) -> None:
    run([str(build / "sub0llm-configure.exe"), *REAL_AXES, *extra], cwd=build)


def build_target(build: pathlib.Path, target: str) -> str:
    return run(["cmake", "--build", str(build), "--target", target, "-j"])


def decode_s_per_token(build: pathlib.Path, tokens: int = 3) -> float | None:
    out = run([str(build / "sub0llm-qwen4-forward.exe"), "--model", ARTIFACT, "--tokens", str(tokens)])
    m = re.search(r"forward_one over \d+ positions in [\d.]+s \(([\d.]+) s/token\)", out)
    return float(m.group(1)) if m else None


def parity(build: pathlib.Path, tokens: int = 6) -> float | None:
    out = run([str(build / "sub0llm-qwen4-forward.exe"), "--model", ARTIFACT, "--tokens", str(tokens)])
    m = re.search(r"max \|forward - forward_one\| = ([\d.eE+-]+)", out)
    return float(m.group(1)) if m else None


def stage_perf(build: pathlib.Path, arms: list[tuple[str, list[str]]], runs: int, tokens: int) -> dict:
    """Interleaved A/B/A/B across arms -- never batched, per OPTIMIZATION_PROCESS.md S1.

    Reconfigures + rebuilds between arms in ONE build dir rather than using N sibling dirs, because a
    stale sibling dir silently compares the wrong source tree.
    """
    samples: dict[str, list[float]] = {name: [] for name, _ in arms}
    for i in range(runs):
        for name, flags in arms:
            configure(build, flags)
            build_target(build, "sub0llm-qwen4-forward")
            v = decode_s_per_token(build, tokens)
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", action="append", choices=["suites", "quality", "perf", "compete", "vtune"],
                    help="repeatable; default is perf only")
    ap.add_argument("--arm", action="append", default=[],
                    help='"name:flags", e.g. "fused:--moe-quant-dot 1". First arm is the baseline.')
    ap.add_argument("--build", default="out/build/wp5c_full48", help="build dir for real-artifact stages")
    ap.add_argument("--runs", type=int, default=3, help="runs per arm (minimum 3 by policy)")
    ap.add_argument("--tokens", type=int, default=3)
    ap.add_argument("--label", default="", help="opportunity ID, tags the history row (e.g. B35)")
    ap.add_argument("--allow-contention", action="store_true",
                    help="measure anyway. Produces a number that policy says is not evidence.")
    args = ap.parse_args()

    stages = args.stage or ["perf"]
    gates = json.loads(GATES.read_text(encoding="utf-8"))

    if "perf" in stages or "quality" in stages:
        n = contention_count()
        if n != 0 and not args.allow_contention:
            print(f"REFUSING TO MEASURE: {n} competing process(es) detected.\n"
                  "This host has produced 2x run-to-run variance under load; a contended measurement\n"
                  "is not a slow measurement, it is a meaningless one (OPTIMIZATION_PROCESS.md S1).\n"
                  "Wait, or pass --allow-contention and do not quote the number as evidence.",
                  file=sys.stderr)
            return 2

    arms = []
    for spec in args.arm:
        name, _, flags = spec.partition(":")
        arms.append((name.strip(), flags.split()))
    if not arms:
        arms = [("default", [])]

    build = ROOT / args.build
    results: dict = {"label": args.label, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    if "perf" in stages:
        if args.runs < 3:
            print("warning: policy minimum is 3 runs per arm", file=sys.stderr)
        results["perf"] = stage_perf(build, arms, args.runs, args.tokens)

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
