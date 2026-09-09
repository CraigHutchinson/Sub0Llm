# Spike execution and evidence policy

Date: 2026-09-08. Applies to the isolated Intel groundwork branch and subsequent spike work.

Each further implementation spike is delegated to a `gpt-5.6-sol` agent with bounded file ownership,
question, budget and completion criteria. Data gathering and external research go to Sonnet 5 or `gpt-5.6-terra`, per the updated user preference.
The integration owner reviews evidence and code, reconciles cross-spike assumptions and schedules
hardware runs. Parallel preparation does not authorize concurrent benchmarks. Explicit user steering
supersedes this default. If a requested model is unavailable, record the fallback.

## Evidence required before a design choice becomes a fact

Every report distinguishes:

- **Measured:** exact hardware/compiler/driver, source and input identity, command, raw result and scope.
- **Documented:** primary source/version and the actual contract; explicitly state whether tested here.
- **Hypothesis:** plausible mechanism or expected benefit, with an experiment that could disprove it.
- **Unknown:** missing support, fidelity, observability or integration evidence, named consumer and revisit gate.

A declaration in an installed header is not an executed capability. A successful API call is not a
speedup. Shared addressability is not proof of physical zero-copy. A component speedup is not request
latency. An inconclusive or unsupported result is useful and must not become a silent fallback.

## Preserve choices without speculative production machinery

Implement the smallest standalone experiment that discriminates viable options. Keep production
interfaces private until a real consumer and measured need exist. Prefer a simple baseline; complexity
must earn its place through correctness and representative measurements, including preparation costs.
Keep alternative kernels/allocation/submission paths in the benchmark harness when an integration A/B
is still needed. Do not expand the engine's configuration surface merely to retain an experiment.

For each unresolved choice, record alternatives, changed variables, constants, correctness oracle,
measured metrics, known confounds, reversal cost and next decision gate. Change one variable at a time
where possible, then cross interacting choices (especially quantization, packing, allocation and
submission). Full-integration acceptance must include startup, TTFT, decode distribution, memory
headroom and representative state/expert behavior, not just isolated throughput.

R0 checks API feasibility; R1 ranks component alternatives; R2 reviews consumed interfaces; R3 repeats
A/B choices on the correct integrated path; R4 qualifies the useful workload. Earlier winners remain
provisional when those later workloads can change the ranking. Retire scaffolding per AGENTS §11 only
when its finding has a consumer and its remaining measurement purpose has ended.

## Current preparation wave

| Sol assignment | Owns | Current gate / boundary |
|---|---|---|
| USM capabilities | `tools/intel_probe/usm_capabilities.cpp`, `scripts/intel/inventory/run-usm-capabilities.ps1`, its report | Prepare static probe; runtime and compiler execution deferred pending a measurement/build window |
| Prepared copies | `benchmarks/intel/mechanisms/prepared_copy.cpp`, `scripts/intel/inventory/run-prepared-copy.ps1`, its report | Ordinary copy vs prepared copy vs reusable host staging; no claimed winner before measured correctness/performance |

Mapped import follows capability results and Terra's contract audit. Hint/scratch comparisons follow
the baseline allocation controls. Real IQ and representative dependency chains retain their existing
S2/S4 gates; this wave does not close them or authorize full backend integration.

## Stable, preview and experimental candidates

Include directly relevant preview releases, experimental extensions and upstream proposals in research.
Record the source date, exact release/commit, maturity, OS/driver/device prerequisites, local test status
and fallback. A proposal, merged implementation and released supported API are distinct evidence levels.
Do not assume an experimental feature will stabilize by integration time. Recheck at R0/R2 and before
qualification; repeat affected correctness/performance checks after changing the runtime tuple.

Evaluate a promising unstable feature in a private opt-in spike when its potential benefit addresses a
measured or credible workload bottleneck. Retain a viable baseline until the exact production tuple is
qualified. Version pinning, packaging/licensing, maintenance burden and migration cost belong in the
later adoption decision alongside request latency and memory use. No SDK/driver upgrade is implied by
adding an option to the watchlist.