#!/usr/bin/env python3
"""Deep regime 2 engine A/B: reactive mmap against the owned expert cache (buffered and uncached fills).

A locked RAM ballast shrinks free memory so the 37 GiB sidecar cannot sit in the OS cache, then each arm
decodes 2,000 G-PPL tokens from an evicted start, in rotated rounds. Perplexity must be identical across
arms. Absolute tok/s moves with how much memory is free on the day: compare arms within one run only.

    python deep_regime.py <ballast.exe> [--room-gib 10] [--budget-gib 17] [--rounds 2] [--build out/build/s1b]
"""
from __future__ import annotations

import argparse
import pathlib
import re
import subprocess
import sys
import time

import common
import make_ppl_fixture
import page_cache
import run_perf_suite as r

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("ballast_exe")
parser.add_argument("--room-gib", type=int, default=10, help="memory left for expert caching beyond the engine")
parser.add_argument("--budget-gib", default="17", help="--moe-cache-gib for the cache arms")
parser.add_argument("--rounds", type=int, default=2)
parser.add_argument("--tokens", type=int, default=2000)
parser.add_argument("--build", default=str(common.REPO / "out" / "build" / "s1b"))
args = parser.parse_args()

build = pathlib.Path(args.build)
cache = ["--moe-io-mode", "cache", "--moe-cache-gib", args.budget_gib]
arms = [("reactive", ["--moe-io-mode", "reactive"]),
        (f"cache{args.budget_gib}b", [*cache, "--moe-cache-io", "buffered"]),
        (f"cache{args.budget_gib}", cache)]  # the cache's default fills: uncached
deadline = time.monotonic() + 1800
while (problems := r.contention_check(None)[1]):
    if time.monotonic() > deadline:
        sys.exit("REFUSING: " + "; ".join(problems))
    print("waiting: " + "; ".join(problems), flush=True)
    time.sleep(60)
# A stale configurator silently produces the wrong arms (it does not know newer flags): build it first.
subprocess.run(["cmake", "--build", str(build), "--target", "sub0llm-configure"], check=True, capture_output=True)
staged = r.stage_arm_binaries(build, arms, ("sub0llm-qwen4-gen",))
fixture = make_ppl_fixture.ensure(make_ppl_fixture.default_path("ppl_blend_v2"), "ppl_blend_v2")
tokenizer = r.qwen_tokenizer_dir()
files = [r.ARTIFACT, r.ARTIFACT + ".moeq"]
ballast = common.start_memory_pressure(args.room_gib, files, args.ballast_exe)
try:
    for i in range(args.rounds):
        for name, _ in arms[i % len(arms):] + arms[:i % len(arms)]:
            time.sleep(20)
            while r.contention_count() > 0:
                time.sleep(30)
            page_cache.evict_verified(files)
            out = r.run([str(staged[name] / "sub0llm-qwen4-gen.exe"), "--model", r.ARTIFACT, "--tokenizer-dir", str(tokenizer),
                         "--ppl", str(fixture), "--ppl-tokens", str(args.tokens)], timeout=4 * 3600)
            (build / f"deep-{name}-{i}.log").write_text(out, encoding="utf-8")
            result = re.search(r"PPL-RESULT (.*)", out)
            memory = re.search(r"\[mem\] final\s+(.*)", out)
            hits = re.search(r"resident-hit rate ([\d.]+%)", out)
            waits = re.search(r"expert cache waits: (.*)", out)
            print(f"[{i + 1}/{args.rounds}] {name}: {result.group(1) if result else 'NO RESULT'} | "
                  f"{memory.group(1) if memory else ''} | cache hit {hits.group(1) if hits else '-'} | "
                  f"waits {waits.group(1) if waits else '-'}", flush=True)
finally:
    ballast.kill()
