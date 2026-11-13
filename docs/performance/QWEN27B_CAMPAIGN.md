# Closing the Qwen 27B reference gap

The goal is to iteratively match or exceed clean optimized llama.cpp on this computer, then approach
the highest throughput its compute, memory and execution dependencies permit. The final hardware
limit is aspirational: theoretical peak instructions or bandwidth alone do not guarantee achievable
model throughput. A faster result must retain the required correctness, context and memory behavior.

Parent tracking: #30 (parity and tokens/s), #29 (runtime integration), #18 (quant format modules).
Review, push and merge every accepted increment through a linked PR. Use the shared Sub0Math
`sub0perf` workflow in [PERFORMANCE.md](../PERFORMANCE.md); propose reusable workflow improvements
upstream rather than duplicating the rig here.

## Baseline and gates

[The controlled 2026-11-07 comparison](QWEN27B_SPEED_COMPARISON.md), #56 / PR #57, measured native
Sub0Llm at 0.249 prompt tokens/s and 0.249 generated tokens/s, versus clean native llama.cpp at
17.027 and 4.284. Same model and eight physical cores; the gap is 68.42× and 17.24×. Native compiler
flags alone improved Sub0Llm generation by 4.04%. These are short-workload backend measurements.

Before each timed iteration: state one falsifiable hypothesis, stage all binaries, run the independent
numerical checks at unchanged tolerances and validate memory. Preserve all accepted/rejected samples,
input/build identities, worker topology, power/load state and native phase boundaries. Report throughput,
spread, latency and RSS together. Keep the existing 8 GiB proving-ground memory budget unless a separate
reviewed issue explicitly changes the deployment target. A three-round short-prompt smoke result does
not establish long-context correctness or model quality; extend the workload and quality coverage as
the resident backend grows. Arithmetic changes require the appropriate independent quality gate.

## Milestones

1. **Attribute the gap (#58).** Measure inclusive projection wall time by quant format on the actual
   mixed-format model, separately for prefill and decode. Compare the sum with complete phase time;
   leave GDN, attention, setup and allocation costs visible as the remainder. This is a profiler,
   not a throughput improvement claim.
2. **Accelerate generation (#18 / #30).** Prioritize formats by measured cost. Integrate format modules
   incrementally; implement packed or vectorized dots with reused scratch and explicit activation
   precision. Start with representative kernel fixtures and real tensor slices, then validate complete
   model logits/quality and measure end-to-end benefit. Review each format as its own increment.
3. **Accelerate prompt processing (#29 / #30).** Reuse weights across batched tokens and avoid unused
   output-head evaluations. Preserve causal GDN/attention state and chat/context behavior. Remove
   repeated allocation and preparation where the phase profile justifies it.
4. **Close the reference gap.** Re-run clean native llama.cpp alongside every candidate. Match or exceed
   reference generation and prompt processing, separately, under equivalent deployment constraints and
   beyond the initial short prompt. Improvements in one phase cannot conceal regressions in another.
   Declare parity only after repeated measurements resolve the difference within measured uncertainty.
5. **Pursue the machine limit.** Sweep core count, both CCDs, SMT, affinity and batching with staged builds.
   Measure sustainable memory traffic and compute for the actual kernel precision and access pattern.
   Build a workload-specific roofline from actual required bytes/operations and serial dependencies;
   compare achieved throughput with that bound. Test whether fusion, layout, repacking or reduced
   redundant traffic can beat llama.cpp. Record exhausted hypotheses and remaining headroom honestly.

The existing eight-core probes are starting evidence, not the final system roof. Encoded payload bytes
and logical projection FLOPs are proxies; they must not be presented as measured DRAM traffic or retired
instructions. The current host disallows hardware counters. Do not silently substitute historical Intel
measurements or change system policy to manufacture a roofline claim.

## Increment record

| Increment | Status | Evidence |
|---|---|---|
| Baseline #56 / PR #57 | merged | Controlled four-arm comparison and raw samples |
| Projection attribution #58 | complete | [Two real-model runs](QWEN27B_PROJECTION_PROFILE.md): ~99.2% projections, ~71% IQ1_S; unchanged logits |
| IQ1_S #60 | measured candidate | [2.95× end-to-end gain](IQ1S_ITERATION.md); unchanged activations, broader promotion qualification remains |
| Remaining projection reduction [#62](https://github.com/CraigHutchinson/Sub0Llm/issues/62) | measured candidate | [Another 1.72× gain](FLOAT_REDUCTION_ITERATION.md), 1.264 generated tokens/s; quality/promotion remains #63 |
| IQ2_XXS [#66](https://github.com/CraigHutchinson/Sub0Llm/issues/66) | measured candidate, draft PR #67 | [Another 18% generation gain](IQ2XXS_ITERATION.md), 1.503 tokens/s; corpus/default qualification pending |
| Context/quality and promotion [#63](https://github.com/CraigHutchinson/Sub0Llm/issues/63) | active | Full supported window, independent quality gates, generated defaults and artifact promotion |
| Packed integer activations #84 / PR #104 | merged explicit candidate | [Q16 passes the 16-position gate](PACKED_ACTIVATION_BACKEND.md); Q8 fails; no default promotion |
| Concurrency reference #90 / PR #106 | merged | [Matched llama.cpp concurrency baseline](QWEN_CONCURRENCY_BASELINE.md), bounded pooled Sub0 engine still pending |
| IQ1 Q16 row/session tile #105 | reviewed candidate | [2.16× two-session prepared kernel gain](Q16_SESSION_TILE.md); single-session model ~1.58 gen tok/s vs clean llama.cpp ~4.13; full corpus pending |

Next execution package: [#107](https://github.com/CraigHutchinson/Sub0Llm/issues/107), bounded
two-session whole-model proving ground with shared projection coordination and isolated KV/GDN/RNG.
The tile result establishes a kernel opportunity; it does not establish concurrent-model parity.
