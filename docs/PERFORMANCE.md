# Codified performance iteration

Sub0Llm adopts Sub0Math's shared `sub0perf` implementation and measurement methodology. This is the
entry point for new performance work. The preserved [upstream methodology](performance/Sub0Math-PERFORMANCE.md)
is pinned at `1d20313cbd2ff2753620246b81934d0489a0636f`; code, tests, runner, CMake helpers and nanobench
are vendored unchanged with hashes in `tools/perf/UPSTREAM.json` and the upstream MIT license. No
Sub0Math runtime library or network fetch is needed. `scripts/check_sub0perf_vendor.py` verifies the
snapshot; update source, hashes and documentation together in a reviewed PR rather than silently forking it.

## Everyday workflow

```sh
cmake --preset perf
cmake --build --preset perf -j2
ctest --preset perf
python3 tools/perf/bench_check.py selftest --repo .
python3 tools/perf/bench_check.py gate --repo . --level quick
```

The perf preset builds only `sub0llm-bench-denseq`, without corpus/model generation or dependencies
fetched by CPM. `SUB0_BUILD_BENCHMARKS=ON` also includes the shared rig in the normal engine build.
Defaults match upstream: a quick report after relinking, 90-second target budget, three initial
rounds, and hook installation outside CI. `SUB0MATH_PERF_ON_BUILD=ENFORCE` makes a confirmed regression
fail the after-build check; `OFF` disables that report. Hook installation chains existing hooks and
records its target/build/executable/baseline under the git common directory's `sub0perf/config.json`.
The hook/config are local; configure the perf preset in each clone to adopt them. Worktrees share
that common directory: configure one measuring build at a time and run `selftest` after changing it.
`SUB0MATH_PERF_GIT_HOOKS=OFF` disables installation (it does not remove hooks previously installed).

The pre-push gate builds the configured target, verifies the pushed HEAD/clean tracked tree, measures,
and compares with the committed host baseline and earlier local records. It skips documentation-only
updates and empty/deletion pushes. It keeps measured binaries for side-by-side reproduction. Confirmed
baseline regressions block; a regression against a local record blocks only when reproduced. The lock
serializes this repository's measuring runs; it cannot serialize other repositories' CPU work.
A bypass/deferred/noisy/environment/method-mismatch result is not a performance pass. Read the output.

Quick rotates whole groups while retaining controls and reports missing coverage. Full measures all:

```sh
python3 tools/perf/bench_check.py gate --repo . --level full
# Or select the milestone gate on a push:
SUB0MATH_PERF_LEVEL=full git push
```

Matching release/milestone tags (`v*`, `milestone/*`) also select full. Confirmation may exceed a quick
budget. The shared rig uses shuffled interleaved rounds, adjacent reference pairs, calibrated clocks,
load checks, CPU pinning, robust spread, bimodal detection, hardware counters where available, and
explicit confirmation/reproduction. Battery/saver/throttle deferrals are reported; unknown power is
labelled rather than treated as known mains. No root power-policy or affinity changes to other processes
are made. On hybrid CPUs explicitly choose a representative logical CPU with `--cpu N` for direct
`run`/`auto` commands; default pinning is the last logical CPU. Never compare methods/CPUs as equivalent.

## Correctness and the three pillars

Use [the iteration evidence template](../perf/ITERATION_TEMPLATE.md) in the linked issue/PR. Profile first,
state one falsifiable hypothesis, check independent correctness before measuring, then report correctness,
performance **and memory**. Give a mechanism three optimization passes before parking it; preserve
meaningfully different paths for later contexts. Review each increment and merge through an issue-linked PR.

The initial suite checks `denseq::project` against an independent direct packed-byte Q8_0 scalar dot at
256 and Qwen's 5120 input width, with 32 output rows, before timing. Both use the same deterministic
fixed-scale fixtures; the reference exploits the known scale and is not weakened. The groups are small,
warm-cache output-row slices, not DRAM-streamed model projections. They cover Q8_0 only; new quant
modules need corresponding representative cases. Their ratios are comparisons with that reference,
not universal abstraction-overhead budgets. A green fixture/gate is **not** 27B correctness, memory
scaling, prefill/decode throughput, or full-context quality. The nine-position numerical failure in
#29/#43 and draft PR #47 remains unresolved and must not be excused by kernel speed.

For model changes, run the independent model/reference checks at unchanged tolerances and the relevant
regressions before performance interpretation. Quality uses the existing paired-perplexity contract
where supported; shorter logit checks remain smoke checks. Record real-model peak RSS separately;
sub0perf's kernel JSON does not measure RSS or certify model memory. Stage all A/B binaries/configs
before timed samples; never rebuild between them. Preserve identity hashes, prompts/token IDs, context,
workers and cache regime. Report tok/s for decode and ms/token for phase costs, every sample/spread,
and dated measured bandwidth/compute roofs for this host. Do not reuse old Windows bus/noise estimates
as Linux facts. Cold-cache runs require verified eviction; warmups in this microbenchmark are warm.

The detailed end-to-end recipes in [OPTIMIZATION_PROCESS.md](OPTIMIZATION_PROCESS.md) and
`scripts/run_perf_suite.py` remain for real models, cold/warm separation and memory pressure. Their
Windows sandbox/eviction mechanisms are not automatically portable to Linux. Keep their model-specific
correctness gates and use staged, interleaved builds; shared sub0perf supplies the new kernel iteration
rig. Do not label an ad-hoc wall-clock observation as a controlled performance claim.

## Baselines, acceptance and profiles

No Sub0Math timing baseline is reused. Before claiming a local regression check, record a full, steady
run and review the generated host baseline (never from a quick subset or contaminated measurement):

```sh
cmake --build out/build/perf --target sub0llm-bench-denseq_runs
python3 tools/perf/bench_check.py baseline \
  --results out/build/perf/perf/sub0llm-bench-denseq.samples.json \
  --baseline perf/baselines/HOST.json
```

Use the same hostname as CMake's `SUB0MATH_PERF_HOST`; re-recording requires `--reason`. A `new` row
means unbaselined. Shared default tolerances are 10% absolute time, 5% pair change, 2% instructions;
these are harness defaults, not acceptable model slowdown targets or changes to numerical tolerances.
Use stricter predeclared per-experiment criteria where relevant. `accept` requires actual measured
samples and a reason; commit its baseline change with the corresponding reviewed issue/PR. Never
accept a regression automatically to get a push through. Check raw rounds, coverage, environment and
memory/quality before accepting any timing result.

```sh
cmake --preset perf-profile -DSUB0MATH_PROFILE_FILTER=5120
cmake --build --preset perf-profile --target sub0llm-bench-denseq_perf_record
# Also available: _vtune_hotspots, _vtune_uarch, _advisor_roofline, _json, _check, _baseline
```

Profiler tools must be installed separately; unavailable counters/profilers are reported as unavailable.
`tools/perf/profile.py` and `vectorization_report.py` are the shared drivers. `.github/workflows/perf-tools.yml`
verifies vendored provenance, runs upstream synthetic gate tests, builds/checks the real fixture protocol,
and uploads full-suite three-round samples. Shared hosted CI samples are diagnostics, not a host baseline
or an enforced hardware performance verdict.
