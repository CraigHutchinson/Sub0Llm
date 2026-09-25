# Optimisation opportunity briefs

Self-contained, agent-executable briefs — one doc per opportunity. **Read the brief before writing
code.** Template and conventions taken from `Sub0h264`'s `docs/optimization/opportunities/README.md`,
which is the proven original; the status legend below adds one state that project does not need.

## Doc template

1. **Header** — expected gain, risk, complexity (S/M/L), and which KPI gates apply
2. **Context** — why this opportunity exists, **with the profile data that says so** (and when that
   profile was measured — see `docs/OPTIMIZATION_PROCESS.md` §5 on stale profiles)
3. **Problem** — the specific inefficiency
4. **Options** — 1-3 reasoned alternatives with pros/cons
5. **Chosen approach** — pick + rationale (this is the contract)
6. **Implementation** — files, line ranges, data structures
7. **Validation** — which gates in `../kpi_gates.json`, and the exact commands
8. **Risks & mitigations** — numerical impact, parity, edge cases
9. **Follow-ups / compounding opportunities** — what this unlocks

Section 9 is not optional and is not a courtesy. It is where the compounding thesis lives
(`AGENTS.md` §13): B34's brief, had one existed, would have named "blocked by B31's parity invariant on
`expert_ffn_row_source`" — and B35 removing that call site would then have been an obvious trigger to
re-run it, rather than something noticed months later.

## Convention

- If the implementation diverges from the brief, **update the brief first**, so doc and code stay in
  sync.
- Commit messages cite the opportunity ID and reference the brief.
- Every brief's validation section names the specific gates that must hold for the change to land.
- **Attempts are numbered.** A second pass at the same opportunity is attempt 2, recorded in the brief
  under `## Attempts`, with its own measured delta and the hypothesis that motivated it. Three attempts
  before parking (§4 of the process doc).

## Status legend

- **pending** — no work started
- **spiked** — prototyping underway, doc may be stale
- **implemented** — merged and default-on; doc is a historical record
- **parked-toggle** — merged behind a default-off `constexpr` toggle, measured non-positive so far,
  live and combinable. *This is Sub0Llm's preferred park state* — see process doc §3 for why it beats a
  parked branch.
- **parked-branch** — kept on a branch, not integrated (only for work that is genuinely unshippable
  rather than merely unhelpful)
- **retired** — reviewed across three attempts and not proceeding (doc explains why)

## Index — current state

| ID | Opportunity | Status | Measured (latest) | Attempts spent |
|---|---|---|---|---|
| B25/B36 | Pipelined overlapped I/O for MoE resolve (`--moe-io-mode pipelined`) | parked-toggle | ~1.4% worse (warm page cache) | 1 of 3 |
| B33/B37 | FP8 (E4M3) backbone storage (`--prec-param fp8`) | parked-toggle | −40..60%; −4.63 GiB memory | 1 of 3 |
| B34/B38 | Multi-accumulator SIMD reductions (`--simd-reduce`) | parked-toggle | −20% alone; −4% stacked on B35 | 2 of 3 |
| B35 | Fused quantized MoE dot products (`--moe-quant-dot`) | implemented (default-off by quality, not perf) | **+2.4x** (3.63 → 1.51 s/token) | 1 |
| B39 | Compile-time kernel shape | parked-branch | −3.3% after 1 fix (was −8.3%) | 1 of 3 |

**Current profile: [`../profile_post_o1.md`](../profile_post_o1.md)** (supersedes post-B35; also corrects the DRAM ceiling to ~79 GB/s P-cores / ~91 GB/s all-core).** Previous:  [`../profile_post_b35.md`](../profile_post_b35.md) (measured 2026-09-21 at the fused
operating point). MoE **68.7%**, mixer GDN/QSA **21.2%**, lm_head 5.4%, gated-residual 4.6%. Lever
selection must cite this, not the pre-B35 numbers.

### Briefed and ready

| ID | Opportunity | Status | Evidence |
|---|---|---|---|
| [O1](O1_iq2xxs_sign_fold.md) | AVX2 fused MoE GEMV: vector accumulator per row (widened from "fold IQ2_XXS signs") | **merged** | **+2.14x decode** (1.508 → 0.705 s/token); L2 0.2938 → 0.2325 | 1 |
| [O2](O2_backbone_gemv.md) | One GEMV primitive for the bf16 backbone: vectorized + threaded; B40 pin fix | **merged** | **+2.42x decode** (0.757 → 0.313 s/token), bit-identical | 2 |
| [O3](O3_gr_threads_and_spin.md) | Gated Residual via the GEMV primitive; `--decode-omp-spin` | **merged** | **0.313 → 0.213 s/token** (spin alone −20%, spread halved) | 1 |
| [O4](O4_gdn_threads_and_moe_schedule.md) | Thread GDN across heads/channels; `schedule(dynamic)` for routed experts | **merged** | **0.236 → 0.211 s/token**; GDN phase −18.5%, bit-exact. MoE lever did NOT reproduce (−2.4%, in noise) | 1 |
| [O5](../../BACKBONE_NATIVE_QUANT.md) | Native-quant backbone: keep the unsloth GGUF's Q8_0/Q4_K/Q5_K/Q6_K bytes resident, fused int8-activation dot | head + GDN wired, **opt-in** `--backbone-quant-dot 1`; GR/QSA/shared expert not yet wired | **0.218 → 0.202 s/token (−7.3%)**: GDN −25%, lm_head −46%; untouched bf16 phases +5–13% (open, §17c). L2 0.2325 → 0.2788 | 3 |

**Candidates, not yet briefed — ordered and costed in [`../../SPECULATION_NGRAM_MOE_DESIGN.md`](../../SPECULATION_NGRAM_MOE_DESIGN.md).**

- **O6 — speculative decoding** (prompt-lookup first, then the model's own MTP head, which the local GGUF
  lacks and must come from the HF checkpoint). Verification reads the backbone once for K tokens, but
  routed-expert compute scales with K on a 512-expert model, so the realistic CPU ceiling is ~1.2–1.5x.
  Needs the batched `forward()` on decode's kernels first. Gate: greedy output token-identical.
- **O7 — routed-expert kernels.** The expert phase is ~46 ms/token for ~0.25 GB read (~5 GB/s against a
  79 GB/s roof): compute-bound on IQ1_S/IQ2_XXS/IQ4_NL unpacking, the shape O5 phase 2a fixed for the
  K-quants. Raises plain decode AND speculation's ceiling, because expert work is what verification
  cannot amortize.
- **Measure first (design doc §6):** E1, an expert-route log (union growth vs K on 512 experts, and
  predictability from n-gram context); E2, offline prompt-lookup acceptance; E3, the memory budget once
  the n-gram table is enabled.

**Retracted**: *IQ1_S narrowing*, previously named by B35 as the next lever on the grounds that its dot
uses the 8-lane `vpmulld` shape. The per-format profile refutes it — IQ1_S is the **cheapest** format
per plane (326 µs vs IQ4_NL 757 and IQ2_XXS 1227). Lane width is not the binding constraint. Do not
brief it until something measures it as a problem.

**Open follow-ups not yet briefed**: IQ4_NL is 39.1% of MoE dot time — the largest single slice and
unexamined (though already at the best GB/s of the three); the mixer (GDN/QSA) at 21.2% of total needs
its own O1 split (48 GDN layers vs 12 QSA) before any lever is picked there; a
branchless/lookup-table `fp8_widen` (the named fix for B33's regression); reducing B35's
activation-quantization error, which is the whole of its quality cost and the only reason it ships
default-off.

## Process backlog

| ID | Item | Status |
|---|---|---|
| P1 | **Make the codified optimization workflow lean enough to be part of the DEFAULT development workflow** — do after the O2 work lands | backlog |
| P2 | **Rename the `sub0_tests` target to `sub0llm_tests`**, matching the `sub0llm-*` tool naming | backlog |

**P2 — rename `sub0_tests` → `sub0llm_tests`.** Do it in one commit, on a quiet tree, with no delegated
track mid-flight: every agent brief, gate and script names the target. The build-affecting consumers are
`tests/CMakeLists.txt` (the target itself), `scripts/run_perf_suite.py`, `docs/optimization/kpi_gates.json`
(G-HASH and G-SUITE-ENGINE name the stage) and `.vscode/settings.json`. About 27 more docs mention it by
name: update the living ones (AGENTS.md, OPTIMIZATION_PROCESS.md, the briefs) and leave dated historical
records as written. Verify with `git grep sub0_tests`, which should then show only historical records.
Open question for the user: rename `sub0_frontend_tests` → `sub0llm_frontend_tests` at the same time
(22 files), so the two suites stay consistent?

**P1 — what to fold in, from what the O1/O2 session actually needed** (keep it lean: one command per
question, defaults that are right, no ritual):

- **One command, sensible defaults.** `run_perf_suite.py` should default to what is almost always wanted:
  sandbox on, `--wait-stable` on, stable-then-measure, the steady-state (default-path) number AND the
  phase table in one run. Today each of those is a separate flag a session has to rediscover.
- **Fail loudly everywhere, by construction.** The harness silently measured a stale binary after a
  failed build until 2026-09-22. Every stage must refuse rather than degrade (already true of `run()`,
  the contention gate and `page_cache.evict_verified`; audit the rest once).
- **Built-in instruments, not scaffolds.** `--profile-phases` replaced a hand-written, reverted
  scaffold. Apply the same to anything else re-typed per session (per-format MoE timing, page-fault
  counts, bandwidth/roofline numbers — `bw.cpp` lives in a scratchpad and should be a tool).
- **Inner loop vs gate.** Kernel microbenchmarks (`sub0llm-bench-moeqd`, a GEMV equivalent) are the
  seconds-long inner loop; the real-artifact decode is the gate. Make that split explicit and cheap to
  follow, so the 37 GiB decode is run a few times per change, not per iteration.
- **Roofline inputs measured, not remembered.** The ~30 GB/s "ceiling" was a single-core number used as
  the machine ceiling for weeks. A `--stage roofline` should measure 1-core and all-core bandwidth and
  the ISA-width MAC ceiling on the current host and write them where every report reads them.
- **A quick tier for everyday work.** A `--quick` preset (build + suites + G-HASH + one warm decode, no
  sandbox wait) that is cheap enough to run on every engine change, so perf regressions are caught in
  normal development rather than by a dedicated optimization session.
- **Control thermal noise, or measure around it.** By O3 iteration 2 (decode ~0.21-0.25 s/token), back-to-
  back real-artifact runs on this laptop-class Arrow Lake-HX part spread 16-30% even at <5% background
  load, which swamps changes of a few percent. Options: longer interleaved series with outlier-robust
  statistics, a cool-down between runs, fixed clocks (power-plan max-processor-state < 100% disables
  turbo), and making the kernel microbenchmarks, not the 37 GiB decode, the evidence for small changes.
- **Trim the docs to match.** OPTIMIZATION_PROCESS.md has grown by accretion; once the tooling carries
  the rules, the doc should shrink to the WHY and point at the commands.
