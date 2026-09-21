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

**Current profile: [`../profile_post_b35.md`](../profile_post_b35.md)** (measured 2026-09-21 at the fused
operating point). MoE **68.7%**, mixer GDN/QSA **21.2%**, lm_head 5.4%, gated-residual 4.6%. Lever
selection must cite this, not the pre-B35 numbers.

**Open follow-ups named but not yet briefed**: IQ1_S narrowing (47% of planes, the one format landing on the wider `vpmulld` shape); a
branchless/lookup-table `fp8_widen` (the named fix for B33's regression); reducing B35's
activation-quantization error, which is the whole of its quality cost and the only reason it ships
default-off.
