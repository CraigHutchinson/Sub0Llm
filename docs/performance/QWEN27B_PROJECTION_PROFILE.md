# Real 27B projection attribution

Issue #58, 2026-11-07, CrogBuntu1 Ryzen 9 8945HX; eight physical cores 0–7, native Clang build.
Two diagnostic runs use the same 13-token prompt and eight fixed-count decode evaluations as
[the reference comparison](QWEN27B_SPEED_COMPARISON.md). These attribute cost and do not claim a speed win.

Projections account for 99.203%, 99.200% of complete decode phase time.
IQ1_S is the first optimization target, accounting for approximately 71% of evaluation time in both runs.
Even eliminating every non-projection cost would leave nearly the entire reference gap.

| Format | Prefill share, runs 1 / 2 | Decode share, runs 1 / 2 |
|---|---|---|
| IQ1_S | 71.042%, 71.084% | 71.015%, 71.066% |
| IQ2_XXS | 9.507%, 9.532% | 9.495%, 9.521% |
| IQ1_M | 5.498%, 5.437% | 5.450%, 5.450% |
| Q3_K | 5.260%, 5.295% | 5.305%, 5.268% |
| IQ2_XS | 3.941%, 3.889% | 3.934%, 3.887% |
| IQ2_S | 2.596%, 2.604% | 2.604%, 2.601% |
| IQ3_XXS | 0.733%, 0.734% | 0.730%, 0.734% |
| Q2_K | 0.545%, 0.548% | 0.544%, 0.547% |
| Q8_0 | 0.067%, 0.067% | 0.066%, 0.066% |
| IQ4_XS | 0.038%, 0.038% | 0.038%, 0.038% |
| IQ3_S | 0.023%, 0.023% | 0.023%, 0.023% |

The optional `--projection-profile FILE --timings-file PHASES` measures each `Layer::project` call
around validation, dispatch and all OpenMP workers. Wall times are measured on the caller and summed
across sequential projections, not summed across worker threads. No profiler allocations occur in the
projection loop; fixed per-phase/type counters are owned by the invocation. Disabled profiling reads
no clock in the projection wrapper. Quantized bytes and logical operations are accounting proxies.

The remainder includes normalization, recurrent state/attention, preparation, allocations and bookkeeping;
it is not assigned to an invented kernel category. This profiler cannot distinguish dequantization,
scalar dot accumulation, memory stalls or OpenMP overhead inside a projection. Use targeted kernel
experiments to separate them. The result prioritizes a format, not a proven microarchitectural cause.

The shared sub0perf lock, quiet checks and process measurement helper were used. Both runs have
less than 5% foreign CPU load globally and on selected cores. Peak process RSS and commands are retained
in [raw evidence](qwen27b_projection_profile.json), together with native phase durations, per-format call
counts/bytes/operations, source and binary hashes. No rebuild occurred between runs. These are warm
working-weight diagnostics; no cache eviction or memory pressure was introduced.

Correctness: the new binary’s sixteen logit vectors are bit-identical to the previous native binary,
and pass the independent reference gate at unchanged relative L2 0.001 (worst 0.0000845936).
Both instrumented runs predict the same nine fixed-count IDs as the previous native measurement.
The profile collector’s separate prefill/decode aggregation, format separation, failure accounting and
unknown-type rejection are independently checked by the native fixture. All seven CTest checks pass.

Next increment: optimize IQ1_S dot execution under #18/#30, retaining packed weights, bounded reusable
scratch and a scalar reference. Check independent format/tensor fixtures and complete-model numerical
behavior before measuring speed. Re-run the controlled interleaved comparison; do not promote a kernel
on this attribution result alone. Broader goals and acceptance gates: [campaign](QWEN27B_CAMPAIGN.md).
