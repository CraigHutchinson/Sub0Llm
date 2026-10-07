#!/usr/bin/env python3
"""How many of the expert cache's fills really come from disk in deep regime 2?

One cache run under the usual ballast while typeperf samples the physical disks' read rate once a second.
Fill bytes follow from the engine's fetch count (~1.65 MB each); the disk bytes read during decode say how
much of that was cold. 2026-10-06: 94 GiB read from D: against ~80 GiB of fills -- every fill is cold.

    python cold_fraction.py <ballast.exe> [--arm cache17] [--room-gib 10] [--build out/build/s1b]

The arm must already be staged under <build>/arms/ (deep_regime.py stages them).
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import re
import subprocess
import time

import common
import make_ppl_fixture
import page_cache
import run_perf_suite as r

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("ballast_exe")
parser.add_argument("--arm", default="cache17")
parser.add_argument("--room-gib", type=int, default=10)
parser.add_argument("--build", default=str(common.REPO / "out" / "build" / "s1b"))
args = parser.parse_args()

build = pathlib.Path(args.build)
exe = build / "arms" / args.arm / "sub0llm-qwen4-gen.exe"
fixture = make_ppl_fixture.ensure(make_ppl_fixture.default_path("ppl_blend_v2"), "ppl_blend_v2")
tokenizer = r.qwen_tokenizer_dir()
files = [r.ARTIFACT, r.ARTIFACT + ".moeq"]
while (problems := r.contention_check(None)[1]):
    print("waiting: " + "; ".join(problems), flush=True)
    time.sleep(60)
page_cache.evict_verified(files)
available = common.available_gib()
ballast = common.start_ballast(args.ballast_exe, max(1, int(available - 20 - args.room_gib - 2)))
log = build / f"cold-fraction-{args.arm}.csv"
log.unlink(missing_ok=True)
try:
    page_cache.evict_verified(files)
    perf = subprocess.Popen(["typeperf", r"\PhysicalDisk(*)\Disk Read Bytes/sec", r"\Memory\Standby Cache Normal Priority Bytes",
                             r"\Memory\Available Bytes", "-si", "1", "-o", str(log), "-y"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(3)
    out = r.run([str(exe), "--model", r.ARTIFACT, "--tokenizer-dir", str(tokenizer), "--ppl", str(fixture), "--ppl-tokens", "2000"],
                timeout=4 * 3600)
    time.sleep(2)
    perf.kill()
finally:
    ballast.kill()
for pattern in (r"PPL-RESULT (.*)", r"expert cache: (.*)", r"expert cache waits: (.*)", r"\[mem\] final\s+(.*)"):
    match = re.search(pattern, out)
    print(match.group(0) if match else f"(no {pattern})")
rows = list(csv.reader(open(log, encoding="utf-8", errors="replace")))
head, data = rows[0], [row for row in rows[1:] if len(row) == len(rows[0])]


def number(text: str) -> float:
    try:
        return float(text)
    except ValueError:
        return 0.0


for i, name in enumerate(head):
    if "Disk Read Bytes/sec" in name:
        series = [number(row[i]) for row in data]
        sixth = max(1, len(series) // 6)
        buckets = [sum(series[k:k + sixth]) / sixth / 1e6 for k in range(0, len(series), sixth)]
        print(f"{name}: total {sum(series) / 2**30:.1f} GiB over {len(series)} s; MB/s by sixth: " + " ".join(f"{b:.0f}" for b in buckets))
    elif "Standby" in name or "Available" in name:
        series = [number(row[i]) / 2**30 for row in data]
        print(f"{name}: start {series[0]:.1f} mid {series[len(series) // 2]:.1f} end {series[-1]:.1f} GiB")
