#!/usr/bin/env python3
"""Sandbox A/B under memory pressure: buffered against uncached fills, without the engine.

Replays a recorded expert-access trace (sub0llm-qwen4-gen --expert-trace PREFIX) through Sub0TieredCache's
trace-replay benchmark, with a locked ballast leaving only a few GiB beside the cache so buffered fills
cannot be served from the OS standby cache. The sidecar is evicted before every run, and each run then
waits: cached-file state left by a buffered run makes non-cached reads queue for seconds on Windows.

    python replay_ab.py <ballast.exe> <sub0tieredcache-trace-replay.exe> <trace-prefix> [--batches 30000]
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time

import common
import page_cache
import run_perf_suite as r

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("ballast_exe")
parser.add_argument("replay_exe")
parser.add_argument("trace_prefix", help="PREFIX of PREFIX.extents / PREFIX.trace")
parser.add_argument("--batches", default="30000")
parser.add_argument("--cache-gib", type=int, default=17)
parser.add_argument("--room-gib", type=int, default=4)
parser.add_argument("--rounds", type=int, default=2)
args = parser.parse_args()

data = r.ARTIFACT + ".moeq"
base = ["--data", data, "--extents", args.trace_prefix + ".extents", "--trace", args.trace_prefix + ".trace",
        "--budget-mib", str(args.cache_gib * 1024), "--chunk-kib", "256", "--readers", "10", "--size-classed",
        "--compute-us", "2800", "--limit-batches", args.batches]
variants = [("buffered", []), ("uncached", ["--uncached"])]
while (problems := r.contention_check(None)[1]):
    print("waiting: " + "; ".join(problems), flush=True)
    time.sleep(60)
page_cache.evict_verified([data])
available = common.available_gib()
ballast = common.start_ballast(args.ballast_exe, max(1, int(available - args.cache_gib - args.room_gib)))
results: dict[str, list[dict]] = {name: [] for name, _ in variants}
try:
    for round_index in range(args.rounds):
        for name, extra in variants[round_index % 2:] + variants[:round_index % 2]:
            while r.contention_count() > 0:
                time.sleep(30)
            page_cache.evict_verified([data])
            time.sleep(15)
            lines = subprocess.run([args.replay_exe, *base, *extra], capture_output=True, text=True).stdout.strip().splitlines()
            if not lines or not lines[-1].startswith("{"):
                print(f"[{round_index + 1}] {name}: FAILED", flush=True)
                continue
            row = json.loads(lines[-1])
            results[name].append(row)
            print(f"[{round_index + 1}/{args.rounds}] {name:9}: rows {row['budget_rows']} hit {100 * row['hit_rate']:.2f}% "
                  f"misses {row['misses']} p50 {row['wait_p50_us']:.0f} p90 {row['wait_p90_us']:.0f} us "
                  f"stall {row['stall_s']:.1f}s wall {row['wall_s']:.1f}s", flush=True)
finally:
    ballast.kill()
print("--- medians")
for name, rows in results.items():
    if rows:
        def median(key: str) -> float:
            return statistics.median(row[key] for row in rows)
        print(f"{name:9}: misses {median('misses'):.0f} p50 {median('wait_p50_us'):.0f} p90 {median('wait_p90_us'):.0f} us "
              f"stall {median('stall_s'):.1f}s wall {median('wall_s'):.1f}s")
