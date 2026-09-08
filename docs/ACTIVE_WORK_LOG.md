# Active work log — cross-agent coordination

This repo now has more than one agent working on it concurrently (at minimum: Claude Code on the
Qwen4-preview/WP4 track, and a separate independent-review agent working through
`docs/INDEPENDENT_REVIEW_BACKLOG.md`'s items). Two failure modes this file exists to prevent:

1. **Two agents editing the same file at the same time** — not just a merge conflict, but one agent's
   in-flight investigation silently invalidated by the other's concurrent change to a shared surface
   (`backend_cpu.cpp`, `layout.hpp`, `gdn_math.hpp`, etc. are read/written by nearly every work package).
   **2026-09-07 decision: both agents share ONE working tree today** (not separate git worktrees) —
   deliberate, for efficient work separation without cross-worktree sync overhead. This makes the log's
   own discipline load-bearing, not just good hygiene: `git add -A`/`-u` or any blanket-stage operation
   is unsafe while another agent's uncommitted edits are present — always `git add <specific paths>`,
   check `git status --short` before AND after every commit, and never `git checkout --`/`git stash`/
   `git reset` across a file you did not personally add to the log.
2. **CPU-contended perf/build work running alongside another agent's perf-sensitive work** — a heavy
   `cmake --build`, test suite run, or benchmark on one track can add real noise to timing measurements
   on another track (this project's own established lesson,
   `[[thermal-confounds-ab-wallclock-testing]]` — interleave trials, don't contend). This applies even
   when the two tracks touch disjoint files.

**Protocol, deliberately lightweight** — a table row, not a lock file:

- **Before starting work that touches a shared file**, check the table below for an overlapping "Files/
  areas" entry with `status: active`. If one exists, either pick different work or coordinate directly
  (leave a note in this file, or ask the user).
- **Before running a CPU-heavy build, full test suite, or benchmark**, check whether another row is
  `status: active` with a perf-sensitive workload (training, Muon optimizer work, CPU profiling). If so,
  ask before running — don't just proceed and hope the numbers are still trustworthy.
- **When you start a work package that will touch files for more than a few minutes**, add a row.
  **When you finish (merge, abandon, or hand off)**, update its status rather than leaving it stale.
- Status values: `active` (currently being worked, files below are hot), `paused` (started, not
  currently running — safe to touch but coordinate if resuming), `done` (merged/complete, historical).

## Log

2026-09-08 Claude Code update: verified Codex's B14 Muon work survived GitHub Copilot's backend-file
relocation. `src/backends/cpu/backend.cpp` (the new home of the old `src/backend_cpu.cpp`) already
contains the Muon scratch-reuse code (`g_muon_scratch`, `MUON_SCRATCH_FLOATS`, per-thread scratch prep)
as committed content — nothing was dropped by the move. `include/sub0/muon.hpp` still carries the same
78-line uncommitted diff it had before the reorg, byte-for-byte unchanged, so Codex's own in-progress
edit there is undisturbed and still waiting on Codex to commit it. WP5a and WP5b (rows below) have both
been resumed now that quota renewed, each told about the new `src/backends/cpu/`+`src/backends/cuda/`
layout and `cmake/Backends.cmake` so their eventual merges don't assume the old monolithic paths.

2026-09-08 Codex update: B14 implementation and validation are complete; remaining Muon files committed in `8a72c67`.
The CPU backend is released from B14 ownership. Current work is Intel iGPU research/design only:
`docs/INTEL_IGPU_BACKEND_DESIGN.md`, `docs/INTEL_IGPU_WORK_PACKAGES.md`, and a link in the independent
backlog. No engine edits, builds, or performance runs are part of this research pass.

| Started | Agent | Work package | Branch | Files/areas | Status | Notes |
|---|---|---|---|---|---|---|
| 2026-09-08 | Codex | Revise Intel plan for native execution, ISA/quantization and UMA research | — | `docs/INTEL_IGPU_BACKEND_DESIGN.md`, `docs/INTEL_IGPU_WORK_PACKAGES.md`, new `docs/INTEL_IGPU_ISA_MEMORY_RESEARCH.md`, independent backlog status links | done | Revised to 21 packages with native ISA/quantization, memory/residency and Level Zero/SYCL submission gates; Vulkan parked. Reconciled landed extractions and WP4f/WP5 status. No builds or device stress. |
| 2026-09-08 | GitHub Copilot | I08/I09 backend area layout extraction | — | `src/backends/cpu/`, `src/backends/cuda/`, `cmake/Backends.cmake`, `CMakeLists.txt`, `docs/ACTIVE_WORK_LOG.md` | active | Mechanical source relocation and CPU API façade extraction; preserve behavior, public APIs, CUDA C++20/no-RDC boundary, and validate/commit each bounded move. Native configure remains blocked by nvcc missing cl.exe in this shell. |
| 2026-09-08 | Claude Code | WP5a: real Qwen4 tokenizer (byte-level BPE, real `vocab.json`/`merges.txt`, gated against `transformers`) | `feature/wp5a-qwen-tokenizer` | new file, likely `include/sub0/qwen_tokenizer.hpp` (+`.cpp`) — deliberately NOT touching the existing `tokenizer.hpp`/`tokenizer.cpp` | active | New goal beyond the original WP1-4 plan: real interactive inference with the transplanted model. First of two parallel prerequisites (this one has zero file overlap with the other). Resumed 2026-09-08 after rate-limit; briefed on the backend reorg below. |
| 2026-09-08 | Claude Code | WP5b: scale the transplant to the full 48-layer model (`mmap` the MoE sidecar first, then attempt the real transplant + load, measure real RSS) | `feature/wp5b-full-scale-transplant` | `include/sub0/moe_quant.hpp`, `tools/sub0llm-transplant.cpp`, `tests/qwen4_real_axes/*.hpp`, possibly `tools/sub0llm-qwen4-forward.cpp` | active | Second of the two parallel prerequisites for interactive inference. Real, unresolved risk: full 48-layer sidecar is ~38GiB even quantized-resident — feasibility on this 63GB machine is not yet known, agent told to report honestly rather than force a fit. Resumed 2026-09-08 after rate-limit; briefed on the backend reorg below. |
| 2026-09-08 | Codex | Intel iGPU platform research and backend decomposition plan | — | `docs/INTEL_IGPU_BACKEND_DESIGN.md`, `docs/INTEL_IGPU_WORK_PACKAGES.md`, `docs/INDEPENDENT_REVIEW_BACKLOG.md` | done | Research/design complete, 18 planned packages. No engine edits or performance runs. Platform selection awaits experiments. |
| 2026-09-07 | Claude Code | WP4f: root-cause the layer-0 Sub0Llm-vs-llama.cpp divergence | `fix/wp4f-layer0-divergence` (merged `296f2a1`, branch deleted) | `include/sub0/transplant.hpp`, `tools/sub0llm-transplant.cpp`, `tests/transplant_tests.cpp`, `tests/transplant_fixture_tests.cpp` | done | Root cause found: `transplant.hpp` read raw GGUF bytes as if they were the HF checkpoint verbatim, but llama.cpp's own converter pre-transforms three things on the way in (RMSNorm gains folded to `1+w`, `ssm_a` stored as `-exp(A_log)`, GDN value-heads reordered grouped→tiled) — fixed with explicit inverses, independently reviewed and reverified before merge. Residual ~2.2% gap at `blk.0.out` after the fix is proven (independent float64 reimplementation) to be llama.cpp's own quantized-activation matmul noise, not a Sub0Llm defect. `backend_cpu.cpp`/`gdn_math.hpp`/`gated_residual_math.hpp` were NOT touched — the fix stayed fully clear of Codex's files throughout. |
| (backlog) | Independent review agent (Codex) | B14: finish Muon scratch reuse + profile matrix products | — | `include/sub0/muon.hpp`, `src/backend_cpu.cpp` (Muon step), `benchmarks/muon_bench.cpp`, `tests/muon_tests.cpp`, `tests/cuda_tests.cpp`, `benchmarks/CMakeLists.txt`, `docs/MUON_CPU_OPTIMIZATION.md`, `docs/CPU_PERF_BACKLOG.md` | done | Completed 2026-09-08; CPU wiring landed during relocation, remaining seven Muon files committed in `8a72c67`. No ongoing file ownership or performance run. Results in `docs/MUON_CPU_OPTIMIZATION.md`. |

## Cleared for the independent-review agent to pick up next (after B14), per this file's own read of WP4's active scope

Cross-checked `docs/INDEPENDENT_REVIEW_BACKLOG.md`'s own "Suggested execution order" against WP4's
actual currently-hot files (`transplant.hpp`, `gdn_math.hpp`, `gated_residual_math.hpp`,
`sub0llm-transplant.cpp`, and — if the divergence investigation's root cause turns out to be on the
Sub0Llm side — potentially `backend_cpu.cpp`'s GDN/GR forward path). The backlog's own scheduling
already reasoned about this carefully and mostly agrees with the read below; noted where this adds
detail.

**Safe to start now, no file overlap with WP4's active work**:
- **B04** (harden `tokmap.hpp` corpus parsing) — isolated parser, no shared file.
- **B05** (reject truncated tokenizer semantics tables, `tokenizer.cpp`) — isolated; the backlog's own
  "coordinate with tokenizer work" caveat doesn't apply, nothing in WP4 touches the tokenizer.
- **B09** (build provenance / `SUB0_GIT_SHA`, `CMakeLists.txt`) — small, additive, `CMakeLists.txt`
  changes so far in WP4 have been new `add_executable`/`target_*` blocks for new tools, not the
  provenance-capture logic B09 targets — low collision risk, but announce the specific lines touched
  before merging since `CMakeLists.txt` is a genuinely shared file.
- **B11** (verify `cmake/get_cpm.cmake` bootstrap) — fully independent.
- **B17** (`window.hpp` document-window fallback) — independent data-path work, no WP4 overlap.
- **B07/B08 design work** (test harness design, validation-gate design) — design-only per the backlog's
  own scope; fine to start, just don't wire it into `scripts/wp4b_check.sh` (already correctly excluded
  by the backlog itself) or any other currently-active WP4 script.
- **B18** (document process/thread-ownership contract) — **mostly safe as pure documentation**, but
  `core.hpp` now carries WP4f's `forward_capture` declaration (merged, stable, not actively changing) —
  fine to document around it; just don't add code/assertions to `core.hpp` itself without checking this
  log first, since a fresh WP4 stage could still add another entry point there.

**Wait until the current WP4f investigation resolves (its outcome determines whether `backend_cpu.cpp`'s
GDN/GR forward path or `gdn_math.hpp`/`gated_residual_math.hpp` need a real fix)**:
- **B12** (forward-only arena overhead in `backend_cpu.cpp`) — the backlog already correctly scheduled
  this as "wait for Claude's stable comparison baseline"; sharpened here: specifically wait for the
  layer-0 divergence investigation above to resolve, since a real fix there may itself touch
  `backend_cpu.cpp`'s per-layer forward path.
- **B15** (CPU affinity / gradient-reduction) — same file, same reasoning; the backlog already says "do
  not run competing CPU benchmarks alongside Claude's performance runs," which also means: not while
  the divergence investigation's own (likely CPU-heavy) verification work is active.
- **B06** (validate before committing loaded state, `engine_core.cpp`) — the backlog already scheduled
  this for "after Claude's checkpoint"; `engine_core.cpp` also carries WP4's `model_file.hpp`
  extraction and `forward_capture` wiring (both merged/stable), so this is safe once the current active
  branch above resolves, same as the backlog's own read.
- **B01/B02/B03/B13/B16** — the backlog's own scheduling for these already looks sound from WP4's side;
  no additional WP4-specific conflict found, so defer to the backlog document's own ordering.

**This section should be re-checked, not assumed stale-safe** — re-read this file's own Log table above
before actually starting any of the "safe now" items, in case a new WP4 row has appeared since this was
written.
