# How optimization work is done in Sub0Llm

**Status: PROCESS. Binding on all performance work.** Adapted from `Sub0h264`'s proven practice
(`docs/optimization/execution_plan.md` and `docs/optimization/opportunities/` in that repo), which
carries substantially more optimization investment than this project does, plus the two rules this
project learned the hard way across the B24-B39 thread (`AGENTS.md` §12, §13).

That thread produced four real wins and five parked attempts. It also produced four avoidable mistakes,
each of which this document exists to prevent: a mechanism retired on a single measurement; a lever
chosen against a profile that a 2.4x change had already invalidated; code merged on correctness+perf
gates that a quality pass then found real defects in; and measurements taken while sibling agents were
saturating the same host.

---

## 1. The measurement protocol

A number that does not follow this protocol is not evidence, and must not be quoted in a commit message.

- **Check for contention first — BOTH named processes AND total system load.** Named sibling tools
  (`sub0llm|clang|ninja|cmake`) must be 0, *and* sustained total CPU load must be under **5%**
  (performance counter, averaged over ~10 s — never one instantaneous read, which reported 48% where
  the sustained figure was 6-19%). This host produced **2x run-to-run variance** under sibling-agent
  load (B36/B37). The named-process check alone is **not sufficient**: after a reboot it reported "0
  contention" while a VS Code updater, browser, desktop apps and a GPU container held total load at
  6-19% — enough to make a ~7-10% decode delta unattributable against a ±1.5% noise floor.
  `scripts/run_perf_suite.py` enforces both and refuses to measure otherwise. A contended measurement
  is not a slow measurement, it is a meaningless one.
- **`sub0llm-qwen4-forward`'s `forward_one` timing can never be cold-cache.** The tool runs `forward()`
  first on the same tokens, and identical tokens route to identical experts — so `forward()` pre-warms
  exactly the sidecar pages `forward_one` then reads. A "cold" `forward_one` number is really warm.
  Measuring genuine cold-cache decode (the only case `--moe-io-mode pipelined` exists for) needs a
  forward_one-only mode, or a flushed standby list plus distinct tokens. **Not yet built** — noted here
  so nobody quotes a cold number from the existing tool.
- **Interleave arms, never batch them.** A/B/A/B, not AAA then BBB — thermal drift is real on this part
  (`[[thermal-confounds-ab-wallclock-testing]]`).
- **Minimum 3 runs per arm**, and report every one, not just the mean. A mean hiding 1.44/1.49/1.67 is
  a different claim from one hiding 1.51/1.52/1.51.
- **Noise floor on this host: ±1.5% for decode s/token** at zero contention (measured: the tightest
  arms reproduce to ±0.2%, the loosest to ±1.6%). **A delta under 2% is "no measurable change"** — not
  a win, and not a regression. Sub-floor changes are still kept and logged; see §4.
- **Prefer ONE build directory reconfigured between arms** over two directories, unless you have
  verified both were built from the same source tree. A stale sibling build dir silently compares the
  wrong thing.
- **Report both roofs, not wall-clock alone.** A delta in seconds says something got faster; it does
  not say whether anything is left. Every perf claim must state achieved **GB/s against the ~30 GB/s
  measured bus ceiling** and achieved **IOPS/FLOPS against the ISA-width ceiling for the data type in
  question**. The logic is decisive: if bandwidth is not saturated the limit must be compute; if
  neither is saturated the kernel is latency- or dependency-bound and tuning either roof is wasted
  effort. **Use the right unit width** — measuring an int8 kernel against a scalar ALU ceiling reports
  "35% utilised, little left" where measuring against AVX2 reports 2.2% and ~45x of headroom. See
  `docs/optimization/roofline_post_b35.md` for the worked example, which overturned this thread's
  governing assumption.
- State the artifact and the axes. The real 48-layer recipe is in `docs/MOE_QUANT_DOT.md` §6h — a
  partial recipe yields `PARAM_FLOATS 2570717696` and a rejected model, with no hint which axis is wrong.

## 2. KPI gates

Machine-readable in `docs/optimization/kpi_gates.json`. **Hard** gates block landing on `main`; **soft**
gates warn. A hard-gate failure means park (§3), never revert.

| Gate | Threshold | Severity | Why |
|---|---|---|---|
| `G-HASH` neutral-build decode fingerprint | unchanged (`816c4a54ad49b8cf`) | hard | AGENTS.md §4 — proves nothing leaked into the default path |
| `G-SUITE` suite assertion counts | exact, or delta fully explained by added tests | hard | a changed count nobody can attribute is an unexplained behaviour change |
| `G-PARITY` `forward`/`forward_one` | exactly 0 in any arm not deliberately breaking it | hard | the invariant B34 broke and had to repair |
| `G-QUALITY` logit L2 vs the unfused path | ≤ 0.43 (the FP8 precedent), argmax ≥ 3/6 | hard | below that an arm is not shippable even default-off |
| `G-PERF` decode s/token | no regression > 2% (the noise floor) | hard | |
| `G-COMPETITOR` vs llama.cpp, same host + model | tracked, not gated | soft | the external bar; see §6 |

## 3. Park, never revert

**Verbatim from Sub0h264's policy, and the reason is load-bearing**: *"some optimisations that look
slower in isolation enable bigger wins downstream once paired with other changes."*

Sub0Llm parks **better than a branch allows**, and this is a deliberate improvement on the source
practice: a parked attempt lands on `main` behind its own `constexpr` toggle, default-off (the
B36/B37/B38 pattern). A parked *branch* bit-rots — it stops compiling the moment `main` moves, and B36
had to be re-derived from scratch against a changed `ExpertCacheSource` for exactly that reason. A
parked *toggle* stays compiled, stays exercised by the normal suite, and can be measured in combination
with any other toggle by anyone, at any time, with no archaeology.

**Procedure when an attempt fails a hard perf gate:**

1. Integrate it as a default-off `constexpr` toggle via the configurator (the existing four:
   `--moe-io-mode`, `--prec-param fp8`, `--simd-reduce`, `--moe-quant-dot`).
2. Document the measured result **at the toggle's own CLI help text**, not only in the backlog — anyone
   choosing the flag then sees the real tradeoff before building with it.
3. Record the **hypothesis for revisit**: what would have to change for this to pay. Sub0h264's own
   example: *"pair with L4.5, which removes the bounds-check overhead this attempt introduced."*
4. Only when a toggle is genuinely unshippable (not merely unhelpful) does it stay on a branch —
   `feature/b39-constexpr-shape` is the current example, parked with two of its three passes unspent.

## 4. Iterate three times (AGENTS.md §13)

A first implementation's number is a property of *that implementation*, not of the mechanism. Take
**three passes** before judging. Break-even or better → continue. Only park after three.

Sub0h264 encodes the same idea as attempt numbering (`perf/L3.1-attempt-1`, `-attempt-2`, …). Two
worked examples from this repo:

- **B39** was parked at −3.3% after exactly ONE fix — an `alignas` bug the change itself had
  introduced — which had already recovered 60% of its original −8.3%. Two passes unspent.
- **B34** was retired at −20% while B31's parity invariant pinned its dominant call site unvectorized.
  B35 later removed that call site from decode's fused path entirely. The original number said nothing
  about the mechanism in its current context.

**Sub-noise-floor wins are kept, not discarded.** Sub0h264's own gate carries this explicitly
(threshold 11.9 rather than 12.0, σ=0.05 fps: *"marginal optimisations within 0.1 fps are kept and
tracked … for holistic review"*). Isolated sub-2% wins that compound are the entire thesis of §5;
discarding each one because it alone cannot clear the noise floor guarantees the compound never happens.

## 5. Measure combinations, and re-profile after any large win

Optimizations interact. Measure the matrix, not just single arms — every parked toggle is live on
`main` precisely so this is cheap.

**And re-derive the profile after any change over ~20%.** This is the mistake that cost this thread
most: B27's I/O-vs-compute split, B30's per-resolve byte accounting, and B31/B34's dominant-call-site
analysis were all measured when decode was ~3.6 s/token and MoE resolve was ~80% of it. B35 moved that
by 2.4x, and lever selection continued against the stale profile — including one combination experiment
that was pure guesswork as a result. **A large win invalidates the map that justified it.**

## 5a. The layered optimization review (helicopter down to the line)

The design review this project runs (`cpp-review`) descends L0 intent -> L1 structural -> L2 API -> L3
implementation, and stops at the first layer that fails. Optimization needs the same discipline, and
this thread proves it by counter-example.

| Layer | Question | Evidence needed |
|---|---|---|
| **O0 System** | Is this workload even bound by the thing I am about to improve? | A profile: bandwidth vs compute vs I/O split |
| **O1 Subsystem** | Is this the component that dominates? | Share-of-total per component |
| **O2 Data flow** | Are the right bytes moving, once? | Byte accounting: round-trips, residency, layout, alignment |
| **O3 Function/call shape** | Is the algorithm and call frequency right? | Call counts, fusion opportunities, redundant passes |
| **O4 Line/ISA** | Is the instruction sequence right? | Generated assembly, unrolling, SIMD width, branches |

**The rule: an optimization at layer N is only valid if the binding constraint is AT layer N. Verify
upward before optimizing downward.** Optimizing O4 while the constraint sits at O0 cannot help, and
routinely hurts by adding scaffolding to a path that was already waiting on memory.

**This thread's record, sorted by the layer each change actually attacked:**

| Change | Layer attacked | Constraint's real layer | Result |
|---|---|---|---|
| B29 thread pinning | O0/O1 (bandwidth contention) | O0 | **+2-9%** |
| B31 transpose elimination | O2 (a whole DRAM round-trip) | O2 | **+7.9%** |
| B35 fused quantized dot | O2 (never materialise 18.75 MiB) | O2 | **+2.4x** |
| B28 `dequantize_iq2_xxs` | O4, but repairing a *pathological defect* (6x-slow branch) | — | **+18-20%** |
| B28 prefetch hints | O4 | O0 (bandwidth) | null |
| B33 FP8 backbone | O2 in intent, O4 in cost (branchy widen) | O2 | −40..60% |
| B34 SIMD reductions | O4 | O0 (bandwidth) | −20% |
| B39 compile-time shape | O3/O4 | O2 (layout) | −3.3% |

**Every win attacked O0-O2. Every loss attacked O3-O4 while the constraint lived higher.** The single
apparent exception, B28's dequant fix, is not a micro-optimization at all — it removed a per-element
data-dependent branch that made one format 6x slower than its siblings. Repairing a pathology is not
the same activity as tuning a healthy loop.

**Read that table as history, not as a standing law — it has already expired once.** "O3/O4 levers
lose here" was a property of the pre-B35 constraint (a path moving ~39 MB per resolve), not a property
of this codebase. B35 cut that 25x, and the phase that is now 68.7% of decode sits at 2.7% of the
memory roof and 2.2% of the integer-SIMD roof: **unpack-bound, at roughly scalar issue rate**. ISA-level
work is now aimed at the right layer for the first time in this thread. See
`docs/optimization/roofline_post_b35.md`.

The corollary, and the reason §5's re-profiling rule exists: **the layer of the constraint moves as you
fix it.** B35 shifted the binding constraint off the MoE resolve path; every O3/O4 lever selected
against the pre-B35 profile is now aimed at the wrong layer by construction. Re-derive O0/O1 after any
change over ~20%, before picking the next lever.

## 5b. Analytical tooling

Wall-clock A/B says *whether*; these say *why*, and they are what stop a lever being chosen by
intuition. Both are installed on this host and both were unused for the whole B24-B39 thread.

- **VTune 2026.4** (`C:\Program Files (x86)\Intel\oneAPItune6.4in64tune.exe`) — real core
  utilisation and the memory-bound vs core-bound classification. Run unelevated with the
  standalone-benchmark + `-target-pid` pattern (`[[cpu-profiling-tooling-backlog]]`); that pattern has
  already found one real MoE defect in this repo that code reading missed.
- **llvm-mca** (`C:\Program Files\LLVMin\llvm-mca.exe`) — static port-pressure and IPC analysis of
  an instruction sequence, **without running it**. The cheapest way to compare two candidate kernel
  bodies before committing to either.

Both are **automated**, so neither is a manual ritual anyone has to remember:

| Command | What it answers |
|---|---|
| `python scripts/analyze_kernel.py --probe <header>` | static: vectorization, instruction mix, port pressure, aliasing, alignment |
| `python scripts/run_perf_suite.py --stage vtune --arm "x:<flags>"` | dynamic: Top-Down split (Retiring / Front-End / Bad-Spec / Back-End, and Memory Bound vs Core Bound) |

`analyze_kernel.py` carries a registry of independent analyzers; **adding one is a single decorated
function**, and the set is expected to grow as we learn what else informs a decision. The current five
and why each earns its place:

- **vectorize** — clang optimization records: did the loop vectorize, at what width, and if not, the
  compiler's own stated reason. A kernel at 2% of the SIMD roof usually has an answer waiting here.
- **asm** — instruction census. The `ymm` vs `xmm` ratio catches half-width vectorization (real code
  running at 128-bit when 256 is available), and the named MAC shapes (`vpmaddwd` 16-lane vs `vpmulld`
  8-lane) say which form the compiler actually picked.
- **mca** — static port pressure and IPC without executing. The cheapest way to compare candidate
  kernel bodies *before* committing to a rebuild, and the place a "neither roof saturated" kernel
  usually confesses: one port oversubscribed while the vector units idle is a different fix from a
  dependent-load chain.
- **alias** — raw pointer parameters lacking `__restrict`. Without it the compiler must assume the
  output may alias an input, which forces a reload per iteration and frequently blocks vectorization
  outright. Heuristic: it reports candidates, it does not assert a defect.
- **align** — SIMD-consumed arrays lacking `alignas`. B39's lesson, automated: swapping `std::vector`
  for `std::array` to satisfy §1's no-heap rule silently forfeits heap alignment, because
  `std::array<int8_t, N>` has natural alignment **1**. That cost ~5% and was found by hand.

Candidates for future analyzers, when a decision needs them: loop-carried dependence distance;
gather/scatter density; L1/L2 miss rates from VTune's memory-access collection; branch-misprediction
rate; unroll factor vs I-cache pressure; `-Rpass=inline` for hot calls that failed to inline (the
`vectorize` analyzer already surfaces "call instruction cannot be vectorized", which is that problem
wearing a different hat).

Use them when the roofline says neither roof is saturated: the arithmetic tells you *where* to look,
but only a core-utilisation measurement distinguishes dependent-load latency from port contention from
issue-width limits.

## 6. The external bar

Track a real competitor on the same host and model, built from source, under fair-comparison controls.
Sub0h264 does this with libavc (static-linked, single-core pinned, per-stream matrix, losses reported
alongside wins). Sub0Llm's equivalent is **llama.cpp** (`D:\Craig\llama.cpp-qwen4exp`,
`llama-bench -ngl 0 -t 24`, the same GGUF).

It is a **soft** gate: it sets the bar, and it tells you when an internal "we are near the floor"
conclusion is wrong. B30 concluded decode was near its DRAM-bandwidth floor; B32's llama.cpp
measurement showed a ~4-6x gap and directly motivated both B31 and B35. Re-measure it whenever a
bandwidth or floor argument is about to close down a line of work.

## 7. Opportunity briefs

One self-contained, agent-executable brief per opportunity under `docs/optimization/opportunities/`,
following Sub0h264's template. See that directory's own `README.md` for the shape, the status legend,
and the doc-stays-in-sync convention.
