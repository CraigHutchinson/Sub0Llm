# Intel inference: benchmark and optimization contract

2026-09-08. Design requirements, not measured results. Consumed by
[I05/I06/I18–I23 and integration packages](INTEL_IGPU_WORK_PACKAGES.md).
I21 owns the shared harness; each kernel/execution package owns its measured implementation.

## Milestones and evidence levels

| Level | Required workload | What may be concluded |
|---|---|---|
| Instruction | Checked arithmetic/unpack and memory probes on the exact adapter | Capability and mechanism costs |
| Component | Real non-square dense/expert planes, GDN/GR/QSA and control shapes | Local speed/quality, not token latency |
| Chain | Actual dependencies, intermediate layouts, dynamic expert indices and state | Submission/fusion/transfer value |
| Prefix | Correct four-layer artifact and encoded weights; all exclusions declared | Architecture-path correctness and prefix latency |
| Full | Complete memory-feasible artifact, real tokenizer/chat/PLE and generation caller | Useful-model latency and quality |

I06 makes a component-based investment decision. It cannot require native full-request timing before
I11/I15 exist. I15 establishes a correct baseline; I22 optimizes it; I23 qualifies each implemented
level independently. I17a capacity analysis starts early so prefix investment does not hide an
infeasible full-model target. Kernel, prefix and full-model records must never share an unlabeled score.

## Reproducible workload matrix

Freeze the minimum matrix before comparing variants, then expand only for a stated question:

- Batch one first; M=1 dense/expert decode and M=32/128 prefill microkernels. Batch four is diagnostic,
  not a serving capability claim. Test realistic small and exact model geometry, non-square and tail
  shapes, vocabulary head, routing/top-k, state/reset and all three real IQ expert encodings.
- Prefix/full prompts at 32/128/512/2048/4096 tokens when supported by a recorded generated config,
  with 128–256 decode steps. Include compression/chunk boundaries and near-context-limit cases.
  Do not exceed baked capacity; a shorter configured run is labeled explicitly.
- Use a deterministic token replay for numerical comparisons and identical MoE routing workload.
  Also run ordinary free generation with real prompts: activation/quantization changes can alter
  routing and EOS, so report divergent workloads rather than attributing all timing to faster code.
- Separate weights/kernel-cache cold start, prepared first request, warmed repeated request, cold
  expert-cache misses and steady reuse. Document the cold-cache procedure; if filesystem coldness
  cannot be established, label it uncontrolled. Never flush caches by memory exhaustion.
- Keep CPU thread count/affinity, AC/power mode, display workload, RAM use, temperature and compiler
  flags recorded. Use explicit, isolated configurations for CPU, Intel and eligible CUDA comparators.
  The CUDA feature guards remain binding; unsupported comparisons are not zero-speed results.

## Measurement definitions

| Metric | Timing boundary / denominator |
|---|---|
| Preparation | Process/model loading, mapping, allocations, conversion/packing, compilation and state reservation, reported as separate phases |
| Warm engine TTFT | Prepared backend receives token IDs until first logits are available to the CPU sampler |
| User-visible TTFT | Request accepted, including tokenizer/chat handling, prefill, sampling, detokenization and first streamed output; cold startup reported separately |
| Prefill throughput | Input tokens actually processed divided by completed prefill duration; state updated before timing ends |
| Inter-token latency | Consecutive token-ready timestamps at the chosen engine or user-visible boundary; report which, p50/p95 and maximum |
| Decode throughput | Actual generated tokens over the corresponding completed decode interval; first-token inclusion and EOS policy explicit |
| Component/chain time | Submission through completion plus separately measured host submission/wait and device intervals |
| Memory | Unique ownership/accounting categories, process commit/working set, runtime budget and preparation peak; distinguish estimates from observable resident bytes |
| Energy | Supported counter scope and duration, idle baseline and joules per generated token; shared package energy is not isolated iGPU energy |

Never time enqueue alone. Drain earlier work before isolated trials and wait for the measured work's
completion. Use monotonic host timing; interpret device timestamp units, valid bits and wrap only from
queried properties. Do not subtract unrelated host/device clock epochs. Concurrent kernel intervals
are not additive wall time. For very short probes repeat a dependent workload, report repetition and
empty-harness overhead, and preserve memory/cache behavior representative of the claim.

Record raw samples per request and token position. Start with at least five interleaved paired trials;
report trial count, spread and paired change. Tokens within one request are correlated: uncertainty
estimates use independent request trials, not thousands of token samples as independent experiments.
If the interval overlaps the promotion boundary, report inconclusive and take a predeclared bounded
number of extra trials. Do not remove slow runs merely because they worsen p95. Record failures and
external interruptions with an exclusion reason before recomputing results.

Instrumented traces diagnose causes; final performance comes from separate unprofiled trials. Intel
documents differing collection overhead for hardware events and API tracing; available metrics depend
on the installed tool/device. Missing counters are marked unavailable, never zero.
[Intel profiling guidance](https://www.intel.com/content/www/us/en/docs/vtune-profiler/user-guide/2026-1/gpu-application-analysis.html).

I21's result schema includes commit plus dirty/source identity, executable/config/artifact checksums,
backend/compiler/runtime/driver/device identity, mode/precision/packing/tuning recipe, fixture origin,
context/shape, seed/token replay, all metric units/boundaries, raw samples, placement, allocation/copy
counts, correctness status and limits. Reuse B09 provenance rather than another SHA generator.
GPU-required jobs fail on missing required fixtures or unexpected CPU execution. Intentional fallback
is a separately labeled mode. Use preallocated tracing/sample storage or collect outside the hot path.

## Optimization sequence and bounded search

1. Establish correctness, residency and the full dependency timeline. Attribute wall time to dense
   projections/head, GDN, QSA, router/expert work, memory faults/copies, host waits and sampling.
2. Calculate useful work and bytes for the actual shapes and quantization. Compare achieved bandwidth
   against I19's measured access-pattern bandwidth, not advertised TOPS. Cache and reuse assumptions
   must be visible. Use a simple compute/bandwidth bound to prioritize experiments, not predict a win.
3. Tune the largest demonstrated contribution first. Each hypothesis records baseline, expected
   bottleneck, bounded variant/time budget, fixture set, memory budget and rejection rule.
4. Validate every candidate numerically before timing. Inspect generated instructions/register spills
   where available. Keep the fastest correct candidates and recheck the integrated chain with ablation.
5. Freeze the winning recipe and remeasure independent prompts, shapes and cold/repeated sessions.
   Reopen only for a changed workload, driver/compiler, failure or demonstrated bottleneck.

| Candidate | Sweep only where applicable | Required cost/quality evidence |
|---|---|---|
| Dense decode/prefill | GEMV versus GEMM; subgroup/vector width, tile/workgroup, layout, oneDNN/oneMKL versus custom | Tail correctness, register/spills, useful bandwidth and inclusive latency |
| Encoded MoE | Fused decode-dot versus bounded tiles; selected-expert grouping and codebook/scales layout | Routing-dependent reads, cold/warm cache, physical copies and quantization error |
| Recurrent/sparse ops | GDN chunk strategy; QSA selection/attention fusion; norm/gate fusion | State drift, masking/boundaries, scratch growth and dependency critical path |
| Memory | Shared access versus resident pack/staging, bounded double buffering and cache capacity | Copy/fault/eviction stalls, CPU contention and peak host budget |
| Submission | Ordinary submission versus graph/command-list replay, batch of dependent work | Dynamic index/argument correctness, host CPU occupancy, cancellation granularity |
| End-to-end host work | Final-logit transfer, tokenizer/detokenizer, sampler and callback buffering | User-visible TTFT/inter-token gains with unchanged user contracts |

No unrestricted autotuning during decode. Benchmark-only switches stay in benchmark programs.
Model/precision/tuning choices known at configure time use generated constants and specialized kernels
under AGENTS §2; installed-device capability validation occurs at setup. A runtime dev exception needs
the required TODO and memory note. Token-dependent routes/lengths are runtime data, not configuration.
If compiled artifacts or packed caches are retained, key by source, shape, encoding, math mode,
compiler/runtime compatibility and tuning version; reject stale derivatives and rebuild before decode.

## Promotion, regression and operational limits

Use the existing provisional promotion rule: repeatable 20% improvement in warm decode latency or
TTFT with no material regression in the other (provisional 5%), against the best *comparable* route.
Also report absolute latency: a relative improvement does not itself make a model interactive.
No user response-time SLO has been supplied; I06 publishes observed ranges and I23 must avoid claiming
an accepted SLO. Quality-preserving memory/energy benefits need a separate measured justification.

Each retained change passes I01's predeclared operator/precision tolerances, nonfinite/state/reset
checks and two-scale validation. For lossy modes, include fixed-token logit metrics and, when I17b
exists, a held-out real-text quality evaluation with a frozen corpus/tokenizer and baseline. Free-text
examples alone are not a quality gate. Record routing/selected-token disagreements explicitly.

CPU-only CI checks contracts/default behavior; Intel correctness and performance require a reserved
hardware run. Establish the hardware noise envelope before fixing regression thresholds. Store both
previous accepted baseline and candidate; an unexplained repeatable loss beyond the noise envelope
blocks optimization promotion. It need not erase a correct experimental implementation.

I23 checks repeated sessions, memory plateau after warmup, graph/cache lifetime, bounded chunk duration
and cancel/device-failure behavior. Bound outstanding work so cancellation is observed between chunks
or tokens; do not promise preemption of an in-flight kernel. Long-running kernels on the display GPU
must be split if they compromise the qualified responsiveness envelope. Never change OS watchdog
settings as the default solution. Record unsupported suspend/device-loss recovery as a limitation.
