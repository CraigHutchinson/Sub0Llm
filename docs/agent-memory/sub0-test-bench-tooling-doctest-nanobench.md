---
name: sub0-test-bench-tooling-doctest-nanobench
description: "Sub0 family standard — doctest for tests, nanobench for benchmarks, latest versions fetched via CPM; adopt incrementally, no forced migrations"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 09648518-580b-4f7b-ba75-d6985e369c04
  modified: 2026-09-30T23:37:34.518Z
---

The Sub0 projects have unified on **doctest** (unit tests) and **nanobench** (benchmarks), latest
releases, fetched with CPM (Sub0Log's pattern; Sub0Pipeline/Sub0Pub still vendor them). Sub0Llm is on
Catch2 and Sub0MemPage uses hand-rolled tests; that is fine.

**Why:** user direction 2026-10-01, after surveying the family: "we have unified on using doctest and
nanobench ... not necessary to migrate but a notable consideration to incrementally update as and where
viable". Sub0ECS v2 moved from Google Benchmark to nanobench 4.6 and doctest 2.5.3 on that basis.

**How to apply:** new test/bench code in any Sub0 project uses doctest/nanobench. When touching an
existing Catch2/gtest/Google Benchmark/hand-rolled suite for other reasons, consider converting the part
you touch — but do not start a migration nobody asked for. nanobench 4.6's `Bench::compare()` (paired,
interleaved, CI on the ratio) is the preferred way to compare alternatives; set `epochIterations` when
alternatives mutate an evolving fixture, since calibration otherwise runs each a different number of
times. Related: [[single-responsibility-headers]].
