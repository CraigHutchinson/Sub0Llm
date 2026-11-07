#!/usr/bin/env python3
"""Rebuilds with compiler vectorization remarks and summarises them for this project's own sources.

  vectorization_report.py --build-dir build-vec [--target <t>] [--source-root .] [--output vec.md]

Configure the build directory with the `vectorization` preset (SUB0MATH_VECTORIZATION_REPORT=ON),
which turns on GCC -fopt-info-vec-*, Clang -Rpass*=loop-vectorize|slp-vectorizer, or MSVC
/Qvec-report:2. The build is run clean so every remark is emitted, then remarks are grouped by file
and line, keeping only files under --source-root (vendored and system headers are dropped).

Use it to check that a hot loop vectorized after a change, and why one did not (aliasing, unknown
trip count, calls, non-contiguous access).
"""

import argparse
import collections
import re
import subprocess
import sys
from pathlib import Path

PATTERNS = [
    # GCC: file:line:col: optimized: loop vectorized using 32 byte vectors
    (re.compile(r"^(?P<file>[^:]+(?::[\\/][^:]+)?):(?P<line>\d+):\d+: (?P<kind>optimized|missed): (?P<msg>.*)$"), None),
    # Clang: file:line:col: remark: vectorized loop ... [-Rpass=loop-vectorize]
    (re.compile(r"^(?P<file>[^:]+(?::[\\/][^:]+)?):(?P<line>\d+):\d+: remark: (?P<msg>.*)$"), "clang"),
    # MSVC: file(line) : info C5001: loop vectorized / C5002: loop not vectorized due to reason '1301'
    (re.compile(r"^(?P<file>.+?)\((?P<line>\d+)\)\s*: info (?P<code>C500[12]): (?P<msg>.*)$"), "msvc"),
]


def classify(match, flavour):
    msg = match.group("msg")
    if flavour == "msvc":
        return ("optimized" if match.group("code") == "C5001" else "missed"), msg
    if flavour == "clang":
        return ("optimized" if re.match(r"(vectorized|SLP vectorized)", msg) else "missed"), msg
    return match.group("kind"), msg


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--build-dir", required=True)
    parser.add_argument("--target", action="append", default=[])
    parser.add_argument("--source-root", default=".")
    parser.add_argument("--output", default=None, help="also write the Markdown report here")
    args = parser.parse_args()

    cmd = ["cmake", "--build", args.build_dir, "--clean-first"]
    for t in args.target:
        cmd += ["--target", t]
    proc = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    log = proc.stdout + proc.stderr
    if proc.returncode != 0:
        print(log[-4000:], file=sys.stderr)
        return proc.returncode

    root = Path(args.source_root).resolve()
    remarks = collections.defaultdict(lambda: collections.Counter())
    for raw in log.splitlines():
        for pattern, flavour in PATTERNS:
            m = pattern.match(raw.strip())
            if not m:
                continue
            path = Path(m.group("file"))
            try:
                rel = path.resolve().relative_to(root)
            except (ValueError, OSError):
                break
            if any(part in ("vendor", "_deps") or part.startswith("build") for part in rel.parts):
                break
            kind, msg = classify(m, flavour)
            remarks[(rel.as_posix(), int(m.group("line")))][(kind, msg[:140])] += 1
            break

    lines = ["| Location | Result | Remark |", "|---|---|---|"]
    vectorized = missed = 0
    for (file, line), counter in sorted(remarks.items()):
        for (kind, msg), _ in counter.most_common():
            vectorized += kind == "optimized"
            missed += kind == "missed"
            lines.append(f"| `{file}:{line}` | {'vectorized' if kind == 'optimized' else 'missed'} | {msg.replace('|', '/')} |")
    report = f"Vectorization remarks under {root.name}: {vectorized} vectorized, {missed} missed.\n\n" + "\n".join(lines)
    print(report)
    if args.output:
        Path(args.output).write_text(report + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
