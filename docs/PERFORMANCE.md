# Codified performance iteration

Sub0Llm adopts Sub0Math's shared `sub0perf` implementation and measurement methodology. This is the
entry point for new performance work. The preserved [upstream methodology](performance/Sub0Math-PERFORMANCE.md)
is pinned at `3152af9f92ff9abb3f53eaa6487392ea94ae53a1`; code, tests, runner, CMake helpers and nanobench
are vendored unchanged with hashes in `tools/perf/UPSTREAM.json` and the upstream MIT license. No
Sub0Math runtime library or network fetch is needed. `scripts/check_sub0perf_vendor.py` verifies the
snapshot; update source, hashes and documentation together in a reviewed PR rather than silently forking it.

For the local dense 27B objective and staged milestones, see [the performance campaign](performance/QWEN27B_CAMPAIGN.md).

The dense 27B default-promotion gate is documented in [full-context qualification](performance/QWEN27B_QUALIFICATION.md).

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
scaling, prefill/decode throughput, or full-context quality. The earlier nine-position discrepancy was resolved by #54/PR #55: automatic reference flash
attention cast F32 K/V to F16. PR #47 records the corrected nine-position check and a complete
16-position reply/EOS check at unchanged tolerances. Future runtime changes still need their own
independent numerical/quality evidence; these samples do not establish maximum-context correctness.

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

## Sub0Llm workload extensions

The implementation now lives upstream in Sub0Math's `sub0perf.workload`, reached with
`python3 tools/perf/bench_check.py workload measure|check`. The former
`tools/perf_extensions/workload.py` path is a compatibility shim. The pinned shared package remains
byte-identical to the pushed upstream commit; local code owns only workload specs and model policy.
This extension adds correctness-gated workload identities and Linux child peak-RSS budgets, which
the kernel gate cannot certify. Use the checked-in process smoke example after building the perf preset:

```sh
python3 tools/perf_extensions/workload.py measure \
  --spec perf/workloads/denseq-process.json --repo . --out out/denseq-process.json && \
python3 tools/perf_extensions/workload.py check \
  --results out/denseq-process.json --max-rss-mib 256
```

Use a new output path for each run; existing evidence is never overwritten. Choose the RSS budget before running. `measure` requires three or more samples, runs the declared
correctness command first, and refuses command failure, timeout, contention above 5%, simulated power,
battery/saver, power changes, or input/binary changes during the run. An optional priming invocation
supports `cache: warm`; `uncontrolled` makes no cache guarantee. Verified cold capture is refused.
All paths are relative to the spec file except absolute paths; executables require explicit paths.
Commands run without a shell, with bounded runtime. Specify model/config/tokenizer/prompt/oracle-script
files under `inputs` and hash their complete bytes before/after (large GGUFs incur that I/O cost outside
timing). Put workers, context length, seed, precision and implementation route in `identity`. The
chosen correctness command must implement an independent check at unchanged thresholds; exit zero
alone does not prove the chosen check is adequate. For #29, use the independent numerical gate, not
`--inspect-context` or successful token generation. A failed oracle prevents any measured invocation.

Records include all elapsed/RSS samples, output digests, measured/oracle binary SHA256, input hashes,
command arguments, adapter/spec identities, relevant threading environment, allowed CPU affinity,
source commit/dirty flag, host and power. Caller-declared environment values are stored: do not put
secrets there. Other inherited environment is not exhaustively captured; declare consequential
settings. Non-executable scripts/interpreted programs and correctness data must appear in `inputs`.
With `check --baseline PRIOR.json`, different workload/model/config/prompt hashes, oracle identity or
arguments, cache regime, adapter/timer method, affinity, environment, host and power are incompatible.
The measured binary hash may differ (it is an A/B arm); workload arguments must match. Use the same
script and input paths across arms. RSS uses each child's own Linux `wait4` high-water mark, so a
large correctness check or warmup does not contaminate measured samples. It includes process startup;
for commands with descendants it is not a sum of simultaneous process-tree memory. Invoke the engine
directly and use separate tools for aggregate service/process-tree/GPU memory.

**Elapsed time is deliberately unjudged.** It includes startup, model loading and all work inside
the invocation, sampled with a 10 ms polling bound. It is not calibrated decode/prefill/turn latency,
model tok/s or a new time-regression gate. `check` only gates the maximum observed RSS against the
explicit absolute budget and optionally validates compatibility; success is not a performance pass.
Continue using shared sub0perf for calibrated timing, confirmation and reproduction. A record from
a dirty tree remains labelled as such and is diagnostic, not a reviewed baseline.

Remaining extensions to evaluate in subsequent issues are phase-native prefill/decode metrics and
token denominators, multi-worker topology/memory-pressure metadata, and verified Linux cache control.
The generic evidence layer has been contributed in [Sub0Math PR #2](https://github.com/CraigHutchinson/Sub0Math/pull/2);
engine-specific token/context/cache policy stays here. Extend the
existing shared confirmation/acceptance path when integrating timing metrics, rather than introducing
a weaker parallel time judge. The original #50 and upstream-adoption #52 add the identity/RSS layer only. Reference correction
#55 and chat validation #47 supply separate finite numerical evidence; new performance claims still
need the appropriate runtime/quality checks and model-specific measurement methodology.


Shared workload outcomes use `sub0perf-workload-outcome-1` with `status`, `next_action` and stable
exit codes (0 recorded/within-budget, 1 over-budget, 2 refused). Helpers log to stderr; stdout contains
one JSON outcome for normal completion/refusal. Shared records use `sub0perf-workload-1`. Regenerate
older local `sub0llm-workload-1` records; they are deliberately refused as incompatible rather than
silently compared across implementations. No-clobber atomic publication requires hard-link support
on the output filesystem and refuses if unavailable. See the pinned upstream method for the complete
schema/validation contract. This removes duplicated implementation and behavioral tests; consumer
checks verify the compatibility entry point while shared tests verify the behavior.
