#!/usr/bin/env python3
"""Static + dynamic analysis of a Sub0Llm hot kernel. The 'why', to the perf suite's 'whether'.

docs/OPTIMIZATION_PROCESS.md S1 requires every perf claim to state achieved bandwidth against the bus
ceiling AND achieved IOPS/FLOPS against the ISA-WIDTH ceiling. When neither saturates, the kernel is
latency- or dependency-bound and no amount of tuning either roof helps -- at which point you need to
know WHY, and wall-clock cannot tell you. This script automates the tools that can.

Analyzers are a registry (see ANALYZERS below). Each is independent, each returns a dict, and adding a
new one is a single decorated function -- the intent is that this grows as we learn what else informs
a decision. Current set:

  vectorize  did the loop vectorize, at what width, and if not WHY (clang optimization records)
  asm        instruction census: SIMD vs scalar mix, the specific MAC shapes, code size
  mca        llvm-mca static port pressure / IPC / bottleneck, without running anything
  alias      pointers that lack __restrict -- the classic silent vectorization blocker
  align      SIMD-consumed arrays that lack alignas (the B39 lesson: std::array has alignment 1)

Deliberately NOT here: wall-clock throughput (that is run_perf_suite.py) and VTune collection (that
needs the real artifact and minutes per run -- run_perf_suite.py --stage vtune).

Usage:
  python scripts/analyze_kernel.py --probe include/sub0/moe_quant_dot.hpp
  python scripts/analyze_kernel.py --probe include/sub0/simd_reduce.hpp --only vectorize,asm
  python scripts/analyze_kernel.py --list
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
CLANG = shutil.which("clang++") or r"C:\Program Files\LLVM\bin\clang++.exe"
MCA = shutil.which("llvm-mca") or r"C:\Program Files\LLVM\bin\llvm-mca.exe"

# Per docs/optimization/roofline_post_b35.md. Used to turn raw counts into "% of the roof", which is
# the only form of the number that supports a decision.
ISA = {
    "avx2_int8_mac_per_instr": 16,   # vpmaddwd: 16 int8->int16 pairs per instruction
    "vector_ports": 2,
    "ghz": 3.5,
    "dram_gb_s": 30.0,
}

ANALYZERS: dict[str, dict] = {}


def analyzer(name: str, doc: str):
    def deco(fn):
        ANALYZERS[name] = {"fn": fn, "doc": doc}
        return fn
    return deco


def _compile(src: pathlib.Path, out: pathlib.Path, extra: list[str]) -> str:
    cmd = [CLANG, "-O3", "-march=native", "-std=c++23", f"-I{ROOT / 'include'}", str(src), *extra]
    if out.suffix == ".s":
        cmd += ["-S", "-o", str(out)]
    else:
        cmd += ["-c", "-o", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.stdout + r.stderr


@analyzer("vectorize", "did each loop vectorize, at what width, and if not why")
def a_vectorize(ctx) -> dict:
    """Clang's optimization records are structured YAML -- far better than scraping stderr, because a
    'not vectorized' remark carries the ANALYSIS string saying which dependence or cost stopped it."""
    rec = ctx["tmp"] / "opt.yaml"
    log = _compile(ctx["probe"], ctx["tmp"] / "v.o",
                   ["-fsave-optimization-record", f"-foptimization-record-file={rec}",
                    "-Rpass=loop-vectorize", "-Rpass-missed=loop-vectorize",
                    "-Rpass-analysis=loop-vectorize"])
    widths = [int(w) for w in re.findall(r"vectorization width:?\s*(\d+)", log, re.I)]
    missed = re.findall(r"loop not vectorized:?\s*([^\n\[]+)", log, re.I)
    return {
        "vectorized_loops": len(re.findall(r"vectorized loop", log, re.I)),
        "widths": sorted(set(widths), reverse=True),
        # Distinct reasons only -- the same reason repeated N times is one finding, not N.
        "not_vectorized_reasons": sorted({m.strip() for m in missed})[:8],
    }


@analyzer("asm", "instruction census: SIMD vs scalar mix and the specific MAC shapes")
def a_asm(ctx) -> dict:
    s = ctx["tmp"] / "k.s"
    _compile(ctx["probe"], s, [])
    text = s.read_text(errors="ignore") if s.exists() else ""
    def n(pat): return len(re.findall(pat, text))
    ymm, zmm, xmm = n(r"%ymm"), n(r"%zmm"), n(r"%xmm")
    # The shapes that matter for an int8 dot, called out by name so the report is readable without
    # knowing x86: vpmaddwd is the 16-lane form we want, vpmulld the 8-lane fallback (see B35).
    shapes = {k: n(rf"\b{k}\b") for k in
              ("vpmaddwd", "vpmaddubsw", "vpmulld", "vpmovsxbw", "vpmovsxbd", "vphaddd", "vpdpbusd")}
    return {"ymm": ymm, "zmm": zmm, "xmm": xmm,
            "simd_share_pct": round(100.0 * (ymm + zmm) / max(1, ymm + zmm + xmm), 1),
            "mac_shapes": {k: v for k, v in shapes.items() if v},
            "text_bytes": len(text)}


@analyzer("mca", "llvm-mca static port pressure / IPC / bottleneck, without executing")
def a_mca(ctx) -> dict:
    """Static analysis of the hottest basic block. Cheap enough to run on every candidate variant
    before committing to one -- which is the point, since running each variant costs a rebuild."""
    s = ctx["tmp"] / "k.s"
    if not s.exists():
        _compile(ctx["probe"], s, [])
    if not pathlib.Path(MCA).exists() and not shutil.which("llvm-mca"):
        return {"error": "llvm-mca not found"}
    r = subprocess.run([MCA, "-mcpu=native", str(s)], capture_output=True, text=True)
    out = r.stdout + r.stderr
    def grab(pat, cast=float):
        m = re.search(pat, out)
        return cast(m.group(1)) if m else None
    return {
        "ipc": grab(r"IPC:\s*([\d.]+)"),
        "uops_per_cycle": grab(r"uOps Per Cycle:\s*([\d.]+)"),
        "block_rthroughput": grab(r"Block RThroughput:\s*([\d.]+)"),
        # Port pressure is where a "neither roof saturated" kernel usually confesses: a single port
        # over-subscribed while the vector units idle is a different fix from a dependent-load chain.
        # Reported as the busiest ports so the answer is one line, not a matrix.
        "busiest_ports": _mca_ports(out),
    }


def _mca_ports(out: str) -> list[str] | None:
    """Pull the 'Resource pressure per iteration' row and name its top entries.

    llvm-mca prints a legend ([0] -> SKLPort0, ...) then a row of per-port pressures. Reporting the
    raw matrix is unreadable; the decision only needs 'which units are saturated'.
    """
    legend = dict(re.findall(r"\[(\d+(?:\.\d+)?)\]\s*-\s*(\w+)", out))
    m = re.search(r"Resource pressure per iteration:\s*\n[^\n]*\n([^\n]*)", out)
    if not m:
        return None
    vals = []
    for i, tok in enumerate(m.group(1).split()):
        try:
            vals.append((float(tok), legend.get(str(i), f"port{i}")))
        except ValueError:
            continue
    vals.sort(reverse=True)
    return [f"{name}={v:.2f}" for v, name in vals[:4]] or None


@analyzer("alias", "pointers lacking __restrict -- the classic silent vectorization blocker")
def a_alias(ctx) -> dict:
    """Without __restrict the compiler must assume the output may alias an input, which forces a
    reload per iteration and frequently blocks vectorization outright. This is a heuristic scan, not
    a proof -- it reports candidates for a human to judge, it does not assert a defect."""
    src = ctx["source"].read_text(errors="ignore")
    hits = []
    for m in re.finditer(r"^[ \t]*(?:template[^\n]*\n[ \t]*)?(?:\[\[nodiscard\]\]\s*)?"
                         r"(?:inline\s+)?\w[\w:<>,\s*&]*\s(\w+)\(([^)]*)\)", src, re.M):
        name, params = m.group(1), m.group(2)
        ptrs = re.findall(r"(?:const\s+)?[\w:]+\s*\*\s*(?!__restrict)(\w+)", params)
        if len(ptrs) >= 2:   # one pointer cannot alias itself into a problem; two or more can
            hits.append({"fn": name, "unrestricted_ptrs": ptrs[:6]})
    return {"candidates": hits[:10], "note": "heuristic -- >=2 raw pointer params without __restrict"}


@analyzer("align", "SIMD-consumed arrays lacking alignas (the B39 lesson)")
def a_align(ctx) -> dict:
    """B39: swapping std::vector for std::array to satisfy the no-heap rule silently forfeited the
    heap's alignment (std::array<int8_t,N> has natural alignment 1), costing ~5% in a SIMD kernel."""
    src = ctx["source"].read_text(errors="ignore")
    bad = []
    for m in re.finditer(r"^([ \t]*)(?!.*alignas)(std::array<\s*(?:std::)?(?:u?int8_t|float|int32_t)[^;]*)\s(\w+)\s*\{",
                         src, re.M):
        bad.append({"decl": m.group(2).strip()[:60], "member": m.group(3)})
    return {"unaligned_arrays": bad[:10],
            "note": "std::array has the alignment of its element type (1 for int8) unless alignas is given"}


# A header alone generates no code, so the codegen analyzers (vectorize/asm/mca) need a TU that
# actually INSTANTIATES the hot path. One entry per analyzable kernel; add a new one when a new hot
# kernel appears. Keep the instantiation shaped like the real call site -- analyzing a differently
# shaped instantiation measures a kernel the engine never runs.
PROBE_BODIES: dict[str, str] = {
    "include/sub0/moe_quant_dot.hpp": (
        "using namespace sub0::moeqd;\n"
        "// Real decode shapes: gate/up are 640 rows x 2560 elems (docs/MOE_QUANT_DOT.md).\n"
        "extern \"C\" void probe_iq1(const std::uint8_t* r, const ActBlocks& x, float* o)\n"
        "{ detail::gemv(Iq1SPlane{r}, 640, 2560, x, o); }\n"
        "extern \"C\" void probe_iq2(const std::uint8_t* r, const ActBlocks& x, float* o)\n"
        "{ detail::gemv(Iq2XxsPlane{r}, 640, 2560, x, o); }\n"
        "extern \"C\" void probe_iq4(const std::uint8_t* r, const ActBlocks& x, float* o)\n"
        "{ detail::gemv(Iq4NlPlane{r}, 640, 2560, x, o); }\n"
    ),
    "include/sub0/simd_reduce.hpp": (
        "extern \"C\" float probe_dot(const float* a, const float* b, int n)\n"
        "{ return sub0::simd::dot(a, b, n); }\n"
        "extern \"C\" float probe_sum(const float* a, int n)\n"
        "{ return sub0::simd::sum(a, n); }\n"
    ),
}


def make_probe(header: pathlib.Path, tmp: pathlib.Path, rel: str) -> pathlib.Path:
    """A minimal TU that includes the header AND instantiates its hot path.

    Without an instantiation the codegen analyzers silently report zeros -- which reads as "no SIMD"
    when the truth is "no code". If a header has no PROBE_BODIES entry, say so rather than emitting a
    confident zero.
    """
    body = PROBE_BODIES.get(rel.replace("\\", "/"))
    p = tmp / "probe.cpp"
    p.write_text(f'#include "{header.as_posix()}"\n' + (body or ""), encoding="utf-8")
    if not body:
        print(f"note: no PROBE_BODIES entry for {rel} -- codegen analyzers will see an empty TU.\n"
              f"      Add one to scripts/analyze_kernel.py to analyze it.", file=sys.stderr)
    return p


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probe", help="header to analyze, repo-relative")
    ap.add_argument("--only", help="comma-separated analyzer subset")
    ap.add_argument("--list", action="store_true", help="list analyzers and exit")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args()

    if args.list:
        for k, v in ANALYZERS.items():
            print(f"  {k:10s} {v['doc']}")
        return 0
    if not args.probe:
        ap.error("--probe is required (or --list)")

    source = ROOT / args.probe
    if not source.exists():
        print(f"no such file: {source}", file=sys.stderr)
        return 2

    names = [n.strip() for n in args.only.split(",")] if args.only else list(ANALYZERS)
    results = {}
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)
        ctx = {"source": source, "tmp": tmp, "probe": make_probe(source, tmp, args.probe)}
        for n in names:
            if n not in ANALYZERS:
                results[n] = {"error": "unknown analyzer"}
                continue
            try:
                results[n] = ANALYZERS[n]["fn"](ctx)
            except Exception as e:   # one analyzer failing must not lose the others' findings
                results[n] = {"error": f"{type(e).__name__}: {e}"}

    if args.json:
        print(json.dumps({"source": args.probe, "isa": ISA, "results": results}, indent=2))
    else:
        print(f"\n=== {args.probe} ===")
        for n, r in results.items():
            print(f"\n[{n}] {ANALYZERS.get(n, {}).get('doc', '')}")
            for k, v in r.items():
                print(f"    {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
