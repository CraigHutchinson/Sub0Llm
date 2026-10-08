#!/usr/bin/env python3
"""Run Sub0Llm's optimization measurement protocol and check it against the KPI gates.

This exists to delete the boilerplate that otherwise gets re-done by hand for every optimization --
configure, build, check for contention, run N times, parse, interleave, compare, evaluate gates, log.
Doing that manually is where measurement mistakes come from (a contended run, a batched A/B, a stale
sibling build dir, a mean quoted without its spread). Shape borrowed from Sub0h264's
scripts/run_all_suites.py, which is the proven original.

Policy it enforces: docs/OPTIMIZATION_PROCESS.md. Gates: docs/optimization/kpi_gates.json.

Stages (each skippable, so iteration stays fast):

  1. suites   -- sub0llm_tests + sub0llm_frontend_tests at the neutral config  (G-HASH, G-SUITE-*)
  2. quality  -- forward/forward_one parity + logit stats, real artifact (G-PARITY, G-QUALITY)
  3. perf     -- interleaved multi-arm decode throughput, real artifact  (G-PERF)
  4. compete  -- llama.cpp on the same host and model                    (G-COMPETITOR, soft)
  5. ppl      -- teacher-forced perplexity of the decode path on a pinned blended text, one run per
                 arm (it is deterministic), each arm compared token by token with the first (G-PPL).
                 The quality gate for any change to decode-path precision.

  # Does a precision change cost quality? (fixture: scripts/make_ppl_fixture.py)
  python scripts/run_perf_suite.py --stage ppl --label O5 \
      --arm "bf16:--backbone-quant-dot 0" --arm "native:"

Typical invocations:

  # REAL_AXES configures the recommended decode options by default (auto), so an arm names only
  # what it changes from that best build -- e.g. "--backbone-act-super 0" to measure O9 off.
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
import os
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
    # Rebuild the configurator FIRST: a new generated constant (a new --flag) is otherwise unknown to the
    # build dir's stale configurator binary, and the engine then fails to compile against a header that
    # lacks it. A no-op when it is already current.
    build_target(build, "sub0llm-configure")
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


def stage_arm_binaries(build: pathlib.Path, arms: list[tuple[str, list[str]]],
                       targets: tuple[str, ...] = ("sub0llm-qwen4-forward",)) -> dict[str, pathlib.Path]:
    """Configure + build every arm ONCE, copying its executable and DLLs into build/arms/<name>/.

    Timing then never follows a compile. The old loop rebuilt before every sample, so each arm was
    measured on a CPU a compile had just heated, a thermal confound that tracked the arm schedule
    (S1b 2026-09-29: one reactive arm spanned 0.111-0.170 s/token over three runs). The generated-config
    hash proves the arms really are different builds (AGENTS.md S10.2), since their banners can be identical.
    """
    import hashlib
    import shutil
    staged: dict[str, pathlib.Path] = {}
    hashes: dict[str, str] = {}
    for name, flags in arms:
        configure(build, flags)
        for target in targets:
            build_target(build, target)
        dst = build / "arms" / name
        shutil.rmtree(dst, ignore_errors=True)
        dst.mkdir(parents=True)
        for f in [*(build / f"{t}.exe" for t in targets), *build.glob("*.dll")]:
            shutil.copy2(f, dst / f.name)
        # The generated headers ARE the build's compile-time identity. The DLL is not: the linker stamps
        # it, so identical flags hash differently on every build and a DLL comparison can never fire.
        h = hashlib.sha256()
        for header in sorted((build / "generated").glob("*.hpp")):
            h.update(header.name.encode())
            h.update(header.read_bytes())
        digest = h.hexdigest()
        clash = [other for other, d in hashes.items() if d == digest]
        if clash:
            raise RuntimeError(f"arm {name!r} generated the same configuration as {clash[0]!r}: "
                               "the flags did not change the build, so the A/B would compare a build with itself")
        hashes[name] = digest
        staged[name] = dst
        print(f"  staged {name}: generated-config sha256 {digest[:12]}", flush=True)
    return staged


def stage_perf(build: pathlib.Path, arms: list[tuple[str, list[str]]], runs: int, tokens: int,
               cold: bool = False, cooldown_s: float = 20.0) -> dict:
    """Interleaved across arms -- never batched, per OPTIMIZATION_PROCESS.md S1.

    Every arm is built once up front (stage_arm_binaries), in ONE build dir rather than N sibling dirs,
    because a stale sibling dir silently compares the wrong source tree. Round r then runs the arms
    starting from arm r % n, so no arm always takes the first, coolest slot, and a fixed cooldown
    precedes every sample so thermal state does not accumulate along the schedule.
    """
    staged = stage_arm_binaries(build, arms)
    samples: dict[str, list[float]] = {name: [] for name, _ in arms}
    for i in range(runs):
        order = arms[i % len(arms):] + arms[:i % len(arms)]
        for name, _ in order:
            time.sleep(cooldown_s)
            # The start-of-run gate cannot see a sibling build that begins mid-run; a cheap named-process
            # check before EVERY sample does, and holds the schedule rather than record a contended number.
            while (n := contention_count()) > 0:
                print(f"  paused before {name}: {n} named competing process(es) running", flush=True)
                time.sleep(30)
            v = decode_s_per_token(staged[name], tokens, cold)
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


def qwen_tokenizer_dir() -> pathlib.Path:
    """$SUB0_QWEN_TOKENIZER_DIR, else the Hugging Face cache snapshot holding the real tokenizer files."""
    env = os.environ.get("SUB0_QWEN_TOKENIZER_DIR")
    if env:
        return pathlib.Path(env)
    hub = pathlib.Path.home() / ".cache" / "huggingface" / "hub" / "models--Qwen--Qwen3.8-Flash-Next" / "snapshots"
    for merges in sorted(hub.glob("*/merges.txt")):
        return merges.parent
    raise FileNotFoundError("no Qwen tokenizer: set SUB0_QWEN_TOKENIZER_DIR")


PPL_GATE_NATS = 0.03   # per token; ~3% perplexity. See the G-PPL entry in kpi_gates.json.


def stage_ppl(build: pathlib.Path, arms: list[tuple[str, list[str]]], max_tokens: int,
              version: str, speed_rounds: int = 0, speed_tokens: int = 2000,
              cooldown_s: float = 20.0, ballast_room_gib: float | None = None) -> dict:
    """Score the pinned fixture through forward_one per arm, then compare each arm with the first.

    Perplexity is deterministic for a given build and text, so one scoring run per arm suffices;
    contention changes only the reported decode speed, never the score. The comparison is PAIRED
    (per-token NLL differences on identical tokens), which cancels the text's own difficulty and is far
    more sensitive than comparing two perplexities.

    Every arm is built once up front (stage_arm_binaries), so no run follows a compile. `speed_rounds`
    then adds that many shorter runs per arm in ROTATED order with a cooldown before each: the long-run
    decode tok/s of an A/B. The scoring run's own tok/s is reported but is one sample in a fixed order.

    `ballast_room_gib` measures under memory pressure (regime 2, docs/STORAGE_STACK_PLAN.md): a locked RAM
    ballast leaves only that much room for expert bytes beyond the engine, and the model files are evicted
    before EVERY run, so each starts cold and no arm inherits another's OS cache. Windows only.
    """
    import make_ppl_fixture
    fixture = make_ppl_fixture.ensure(make_ppl_fixture.default_path(version), version)
    tok = qwen_tokenizer_dir()
    staged = stage_arm_binaries(build, arms, targets=("sub0llm-qwen4-gen",))
    files = [ARTIFACT, ARTIFACT + ".moeq"]
    ballast = None
    if ballast_room_gib is not None:
        sys.path.insert(0, str(ROOT / "scripts" / "storage_ab"))
        import common as storage_ab
        import page_cache
        ballast = storage_ab.start_memory_pressure(ballast_room_gib, files)

    def score(name: str, tokens: int, dump: pathlib.Path | None) -> dict:
        time.sleep(cooldown_s)
        while (n := contention_count()) > 0:
            print(f"  paused before {name}: {n} named competing process(es) running", flush=True)
            time.sleep(30)
        if ballast is not None:
            page_cache.evict_verified(files)
        cmd = [str(staged[name] / "sub0llm-qwen4-gen.exe"), "--model", ARTIFACT, "--tokenizer-dir", str(tok),
               "--ppl", str(fixture), "--ppl-tokens", str(tokens)]
        out = run(cmd + (["--ppl-dump", str(dump)] if dump else []), timeout=4 * 3600)
        m = re.search(r"PPL-RESULT (.*)", out)
        if not m:
            raise RuntimeError(f"arm {name}: no PPL-RESULT line")
        result = {k: float(v) for k, v in (kv.split("=") for kv in m.group(1).split())}
        # What the engine reports about its own memory, when it does (cache mode; any --ppl run for the rest).
        for key, pattern in (("cache_hit_pct", r"resident-hit rate ([\d.]+)%"),
                             ("wait_ms_total", r"expert cache waits: \d+ blocking acquires, ([\d.]+) ms total"),
                             ("wait_us_each", r"ms total, ([\d.]+) us each"),
                             ("peak_working_set_gib", r"\[mem\] final\s+peak working set ([\d.]+) GiB"),
                             ("params_resident_pct", r"end-of-run residency: parameters ([\d.]+)%"),
                             ("backbone_resident_pct", r"native backbone mapping ([\d.]+)%")):
            if (found := re.search(pattern, out)):
                result[key] = float(found.group(1))
        return result

    def memory_note(a: dict) -> str:
        parts = [f"{label} {a[key]:{fmt}}{unit}" for key, label, fmt, unit in (
            ("cache_hit_pct", "cache hit", ".1f", "%"), ("wait_ms_total", "waits", ".0f", " ms"),
            ("peak_working_set_gib", "peak", ".1f", " GiB"), ("backbone_resident_pct", "backbone resident", ".0f", "%"),
            ("params_resident_pct", "parameters resident", ".0f", "%")) if key in a]
        return ("; " + ", ".join(parts)) if parts else ""

    try:
        return _score_arms(build, arms, max_tokens, version, speed_rounds, speed_tokens, score, memory_note,
                           ballast_room_gib)
    finally:
        if ballast is not None:
            ballast.kill()


def _score_arms(build, arms, max_tokens, version, speed_rounds, speed_tokens, score, memory_note,
                ballast_room_gib) -> dict:
    """stage_ppl's scoring pass, speed rounds and paired comparison, split out so the ballast is always released."""
    per_arm: dict = {}
    nll: dict[str, list[float]] = {}
    argmax: dict[str, list[str]] = {}
    for name, _ in arms:
        dump = build / f"ppl_{name}.tsv"
        per_arm[name] = score(name, max_tokens, dump)
        rows = [line.split("\t") for line in dump.read_text(encoding="utf-8").splitlines() if line]
        nll[name] = [float(r[2]) for r in rows]
        argmax[name] = [r[3] for r in rows]
        print(f"  {name}: ppl {per_arm[name]['ppl']:.4f} over {int(per_arm[name]['tokens'])} tokens, "
              f"{per_arm[name]['decode_tok_s']:.2f} tok/s{memory_note(per_arm[name])}", flush=True)
    for i in range(speed_rounds):
        shift = (i + 1) % len(arms)
        for name, _ in arms[shift:] + arms[:shift]:
            r = score(name, speed_tokens, None)
            v = r["decode_tok_s"]
            per_arm[name].setdefault("speed_tok_s", []).append(v)
            print(f"  speed [{i+1}/{speed_rounds}] {name}: {v:.2f} tok/s over {speed_tokens} tokens{memory_note(r)}",
                  flush=True)
    base, *rest = [a for a, _ in arms]
    paired = {}
    for other in rest:
        d = [b - a for a, b in zip(nll[base], nll[other])]
        mean = statistics.fmean(d)
        se = statistics.stdev(d) / len(d) ** 0.5
        paired[other] = {"vs": base, "tokens": len(d), "mean_dnll": mean, "se": se,
                         "ci95": [mean - 1.96 * se, mean + 1.96 * se],
                         "ppl_ratio": per_arm[other]["ppl"] / per_arm[base]["ppl"],
                         "top1_agreement": sum(x == y for x, y in zip(argmax[base], argmax[other])) / len(d)}
    return {"fixture": version, "arms": per_arm, "paired": paired, "ballast_room_gib": ballast_room_gib}


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
    for other, p in results.get("ppl", {}).get("paired", {}).items():
        lo, hi = p["ci95"]
        verdicts.append({
            "id": "G-PPL",
            "label": f"{other} vs {p['vs']}, {p['tokens']} tokens",
            "value": f"{p['mean_dnll']:+.4f} nats/token (95% CI {lo:+.4f}..{hi:+.4f}), "
                     f"ppl x{p['ppl_ratio']:.4f}, top-1 agree {p['top1_agreement']:.1%}",
            # Confidently no worse than the threshold -> PASS; confidently worse -> FAIL; else more tokens.
            "status": "PASS" if hi <= PPL_GATE_NATS else ("FAIL" if lo > PPL_GATE_NATS else "inconclusive"),
        })
    for g in gates["gates"]:
        if g["id"].startswith("G-PERF") or (g["id"] == "G-PPL" and "ppl" in results):
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
    ap.add_argument("--stage", action="append", choices=["suites", "quality", "perf", "compete", "vtune", "ppl"],
                    help="repeatable; default is perf only")
    ap.add_argument("--ppl-tokens", type=int, default=12000,
                    help="ppl stage: cap on scored tokens (v2 has ~9,400, v1 ~2,400)")
    ap.add_argument("--ppl-fixture", choices=["ppl_blend_v1", "ppl_blend_v2"], default="ppl_blend_v2",
                    help="ppl stage: pinned text (scripts/make_ppl_fixture.py); v1 only to compare with old history")
    ap.add_argument("--ppl-speed-rounds", type=int, default=0, metavar="N",
                    help="ppl stage: after scoring, N more runs per arm in rotated order (long-run decode tok/s)")
    ap.add_argument("--ppl-speed-tokens", type=int, default=2000, help="ppl stage: tokens per speed-round run")
    ap.add_argument("--ballast-room", type=float, default=None, metavar="GIB",
                    help="ppl stage: measure under memory pressure. Locks RAM so only GIB is left for expert bytes "
                         "beyond the engine, and evicts the model files before every run (regime 2; Windows)")
    ap.add_argument("--arm", action="append", default=[],
                    help='"name:flags", e.g. "fused:--moe-quant-dot 1". First arm is the baseline.')
    ap.add_argument("--build", default="out/build/wp5c_full48", help="build dir for real-artifact stages")
    ap.add_argument("--cold", action="store_true",
                    help="evict the artifact + sidecar from the page cache before EVERY run (verified) and time "
                         "decode alone (--decode-only). Cold and warm numbers are different quantities: "
                         "label and compare them separately")
    ap.add_argument("--runs", type=int, default=3, help="runs per arm (minimum 3 by policy)")
    ap.add_argument("--tokens", type=int, default=3)
    ap.add_argument("--cooldown", type=float, default=20.0, metavar="S",
                    help="perf stage: idle seconds before every timed run, so heat does not build up along the schedule")
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
        results["perf"] = stage_perf(build, arms, args.runs, args.tokens, args.cold, args.cooldown)
        results["cache"] = "cold" if args.cold else "warm"

    if "ppl" in stages:
        results["ppl"] = stage_ppl(build, arms, args.ppl_tokens, args.ppl_fixture, args.ppl_speed_rounds,
                                   args.ppl_speed_tokens, args.cooldown, args.ballast_room)

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
    if results.get("ppl"):
        room = results["ppl"].get("ballast_room_gib")
        pressure = f", under a RAM ballast leaving {room:g} GiB for experts" if room is not None else ""
        lines += ["", f"## Perplexity ({results['ppl']['fixture']}, decode path{pressure})", "",
                  "| Arm | Perplexity | Mean NLL | Top-1 | Decode tok/s (scoring run) | Speed rounds, tok/s |",
                  "|---|---:|---:|---:|---:|---|"]
        for name, a in results["ppl"]["arms"].items():
            speed = ", ".join(f"{x:.2f}" for x in a.get("speed_tok_s", [])) or "-"
            lines.append(f"| {name} | {a['ppl']:.4f} | {a['mean_nll']:.4f} | {a['top1']:.4f} | "
                         f"{a['decode_tok_s']:.2f} | {speed} |")
    lines += ["", "History: `perf_history.jsonl`. Policy: `docs/OPTIMIZATION_PROCESS.md`.", ""]
    REPORT.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines[4:]))
    return 0 if all(v["status"] != "FAIL" for v in verdicts) else 1


if __name__ == "__main__":
    sys.exit(main())
