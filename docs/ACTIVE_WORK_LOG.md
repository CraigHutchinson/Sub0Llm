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

**2026-09-08 Claude Code — shared-tree state flag from earlier today, now RESOLVED.** Found while merging
WP5c: the SHARED working tree (`D:\Craig\GitHub\Sub0Llm`, not a `.claude/worktrees/*` one) had
`feature/wp5c-qwen-generation` checked out (not `main`) — apparently WP5c's own resumed subagent session
worked directly in it rather than in its own isolated worktree — with an uncommitted `AGENTS.md` edit
present (a further refinement of the "delegate to Terra" research-delegation note). `main` itself was
never at risk: its branch ref stayed correctly at `4468c26` throughout, and the WP5c merge was done in a
separate throwaway worktree specifically to avoid disturbing the shared tree's dirty state. The user
confirmed the `AGENTS.md` edit was theirs (broadening delegation to Sonnet-class agents, not just Terra)
— committed and cherry-picked onto `main` (`4eb471e`), the shared tree switched back to `main`, and the
stale `feature/wp5c-qwen-generation` branch deleted (local + already-deleted-on-origin). Shared tree is
clean and on `main` again as of this entry.

2026-09-08 Claude Code update: **`main` did not compile** (`cb85d5a`, pushed) — three mechanical build
breaks left by the CPU/CUDA backend split, found by actually building, not just reading the diff: (1)
`PARENT_SCOPE` on `SUB0_BACKEND_SOURCES`/`SUB0_CUDA_BACKEND_SOURCES` in `src/backends/{cpu,cuda}/
CMakeLists.txt` was a no-op (the whole chain is `include()`, never `add_subdirectory()`, so there's no
child scope to pop out of) — `sub0_core` was linking with zero backend translation units; (2) internal
call sites inside `backend.cpp`'s own `cpu_detail` namespace (the per-window bind/unbind in `train_batch`,
the `backward()` call) used unqualified names that ADL made ambiguous against the outer `SUB0_API sub0::`
declarations, since the argument types live in `sub0::` — qualified those specific calls with
`cpu_detail::` to match `api.cpp`'s own established forwarding pattern; (3) `trainable_floats()` was
correctly defined in `cpu_detail` but was the one function missing from `api.hpp`'s declaration list and
`api.cpp`'s facade, only surfacing at final link. All three fixed, full `d196check` rebuild (71/71
targets) + both suites green (28,755,032 + 118,193 assertions). Only `src/backends/{cpu,cuda}/*` touched.
**If GitHub Copilot's I08/I09 row below is still mid-edit on these same files, re-check for a collision
before your next commit there** — this fix landed on top of whatever was on `main` at `cbcb82d`.

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

**2026-09-08 Claude Code (WP5b) — `main` DOES NOT COMPILE, and it needs BOTH agents to fix it.** Found
while building the engine at the real 48-layer Qwen4 axes. Two independent breakages, one fixed here and
one that is not this work package's to fix:

1. **FIXED on `feature/wp5b-full-scale-transplant` (commit `ad2c818`).** `src/backends/{cpu,cuda}/
   CMakeLists.txt` set `SUB0_BACKEND_SOURCES` / `SUB0_CUDA_BACKEND_SOURCES` with `PARENT_SCOPE`, but
   `cmake/Backends.cmake` pulls them in with `include()`, which runs in the CALLER'S scope rather than
   creating a child one — so both variables were empty at their `add_library()` sites. The configure
   banner said `(engine: )` and `sub0_core` was built with NO backend at all; the symptom was a link
   failure on `params_ptr`/`sync_params_to_*`/`load_moe_quant_sidecar`. Fixed by a plain `set()`.
2. **NOT FIXED — needs Codex, and it is a real cross-agent hazard worth recording.** The note above says
   Codex's B14 Muon work "survived the relocation" because `src/backends/cpu/backend.cpp` contains the
   scratch-reuse code as committed content. That is exactly the problem: the reorg committed the
   BACKEND half of B14 while `include/sub0/muon.hpp`'s matching half is still uncommitted in the main
   working tree. So `main`'s committed `backend.cpp` calls `muon::scratch_floats(...)` (which the
   committed `muon.hpp` does not declare) and a 6-argument `newton_schulz5` (the committed one takes 5).
   Three further call sites are ambiguous between `sub0::` and `sub0::cpu_detail::` after the facade
   split. Six compile errors; the engine cannot be built from `main` by anyone who does not also have
   Codex's uncommitted `muon.hpp` in their tree. **Codex committing that header is the fix**; nothing
   was attempted here, because reconstructing another agent's in-flight API by guesswork is precisely
   the failure mode this log exists to prevent.

   WP5b's own engine measurements were therefore taken on the pre-reorg tree (this branch's own
   `4005819`, i.e. `b855451` + WP5b's two commits), which builds cleanly. The reorg is mechanical and
   does not touch any file WP5b changed, so the numbers carry — but they were NOT taken on today's
   `main`, and that is stated rather than glossed.

| Started | Agent | Work package | Branch | Files/areas | Status | Notes |
|---|---|---|---|---|---|---|
| 2026-09-09 | Codex + Terra | S0a/S0b/S0c tandem-backend spike groundwork | `research/intel-groundwork` | Tandem leaf report; plan/provider design/log | prepared; execution pending | Findings and A/B design reviewed with explicit provenance, native API lifetime constraints and dependency order. Native dependencies, second-target coverage and new measurements remain open. No installs, hardware runs or main edits. |
| 2026-09-09 | Codex + Terra | Backend naming and portable/native composition review | `research/intel-groundwork` | Design/work-package documentation only | done | Primary-source portability/naming review integrated in `BACKEND_PROVIDER_DESIGN.md` and I07. Mixed Intel-native/oneDNN providers and a conditional portable SYCL executor remain distinct; no production interfaces or hardware runs. |
| 2026-09-08 | Codex + Sol agents + Terra | Intel USM next-spike preparation | `research/intel-groundwork` | Capability/prepared-copy leaf sources, runners and reports; execution policy and package links | prepared; runtime pending | Sol-authored probes reviewed; Terra release research integrated. On 2026-09-09 all three compile/link gates passed with DPC++ 2025.3.3; runtime correctness and timing remain pending. No installs or main engine changes. |
| 2026-09-08 | Codex + Terra | Windows USM official-doc applicability review | `research/intel-groundwork` | Isolated Intel worktree documentation only | done | Primary-source review and installed-header audit integrated in `INTEL_IGPU_WINDOWS_USM.md` and S1/I19 follow-ups. No hardware runs or main Qwen changes. |
| 2026-09-08 | Codex | Intel I00 and S0/S1 groundwork | `research/intel-groundwork` | Isolated worktree `out/worktrees/intel-groundwork`; standalone probes/scripts/reports | paused | Both groundwork sizes and five negative cases pass; ten f32 projection shapes and three USM modes measured, with raw records archived. Partial I00/S0/S1/I05/I19 only; direct submission, IQ, chains and capacity remain open. No active benchmark and no main Qwen source edits. |
| 2026-09-09 | Claude Code | B20: cache-block `transplant::transpose_out_in`'s naive strided transpose, then parallelize per-expert resolve across a layer's `EXPERTS_PER_TOK` selection | `feature/b20-decode-perf` | `include/sub0/transplant.hpp`, `include/sub0/moe_quant.hpp`, `src/backends/cpu/decode.cpp`/`internal.hpp` (the `moe_resolve` call site) | active | I08's decode/API split (merged `cb9e2b1`) is this item's own named prerequisite -- both target files/areas are otherwise untouched by any other active row. Coordinate here before touching `moe_quant.hpp`/`transplant.hpp`, both hot shared files per B20's own text. |
| 2026-09-08 | GitHub Copilot / Claude Code | B19: `--vocab-exact` for `sub0llm-configure` | merged `a2ce70b` | `tools/configurator.cpp`, `docs/INDEPENDENT_REVIEW_BACKLOG.md` | done | Copilot implemented, left uncommitted on `main`. Claude Code reviewed (traced the `reused=exact_vocab` control-flow interaction with `tok_stamp_matches` line by line before trusting it), fixed two indentation slips, and independently verified before committing: every error path exercised, the real Qwen4 48-layer axes configured both ways into separate output dirs and `diff`'d byte-for-byte identical (bar the three intentionally-empty tokenizer paths) at **0.033s vs ~120s**, regression suites rebuilt clean afterward. First real cross-agent handoff this session where Claude Code reviewed and shipped another agent's own implementation rather than its own. |
| 2026-09-08 | Claude Code | I08 (decode/API slice only): extract `forward_one`/`kv_reset`/MoE-resolve decode path out of `src/backends/cpu/backend.cpp`'s training-heavy monolith, behavior-preserving, as the structural prerequisite to landing B20's two fixes | merged `cb9e2b1` (`feature/i08-decode-api-split`, branch deleted local+origin) | `src/backends/cpu/backend.cpp` (3,146 → 2,170 lines), new `src/backends/cpu/decode.cpp` (607) + `src/backends/cpu/internal.hpp` (546), `src/backends/cpu/CMakeLists.txt` (+1 source), and two comment-only lines in `tools/sub0llm-qwen4-{gen,forward}.cpp`. NOTHING else — no `moe_quant.hpp`, no `transplant.hpp`, no test file, no public header | done | **Pure code motion, and the evidence says so.** Three files now: `internal.hpp` (546 lines) holds only what BOTH TUs must share as ONE instance (`Worker`/`g_workers`/`W`, the fast-math helpers, the binding `thread_local`s, `QsaRopeTables`, `Layer`, `moe_resolve`, `Model`, `ensure_thread_built`) — verbatim, the only edits being `static`→`inline` on the leaf helpers that must keep inlining into both TUs and definition→`extern` on the single-instance objects. A self-review pass caught the header being **over-shared** on the first cut and moved `arena_alloc`/`mk_param`/`mk_node`/`Mat` and the four parameter arenas back to `static` in backend.cpp: `grep` proves decode.cpp names none of them, because forward_one runs one row through stack buffers and allocates neither an arena slot nor a node-pool entry — keeping them private makes "decode allocates nothing" (AGENTS.md §1) checkable mechanically. The fast-math block stayed whole on purpose even though decode never calls `dgelu_fast`/`dsilu_fast`: AGENTS.md §6's standing invariant is that each value and its derivative stay mutually consistent, and splitting `gelu_fast` from `dgelu_fast` across files would scatter the pair a gradient check exists to protect. `decode.cpp` holds `Model::forward_one`, the eight per-row kernels (verified decode-only — nothing in the batched Node-graph path calls them), the three decode-persistent caches (still anonymous-namespace, still their own reset lifetime), and the `kv_*` entry points. **No new abstraction layer** (I08 forbids a polymorphic tensor class) and **no B20 work** — no `#pragma omp parallel` added, `transpose_out_in` untouched. Proof of motion rather than rewrite: all four moved function bodies (`build_layout`/`init_weights`/`forward`/`forward_one`, 601 lines) `diff` **byte-identical** to the originals modulo a 4-space de-indent and three deliberate comment fixes; a whole-file line-set comparison found exactly 14 changed lines, all of them the intended signature transforms. **Neutral d196 L11 H7 (identical before/after):** `sub0_tests` 28,755,032/147, `sub0_frontend_tests` 120,889/244 and 121,457/244, `sub0llm-gen --n 250 --seed 1234` output SHA-256 identical (`E2AE75A6…6779`), decode 0.724 → 0.695 ms/token, clean build 42.3 → 43.3 s, `sub0_core.dll` 1,136,128 → 1,137,152 B (+0.09%), CPU-backend object bytes 1,928,837 → 1,854,091 + 310,556, and the backend TU itself compiles *faster* (4.49 → ~4.2 s) with decode.cpp's 1.5 s alongside it in parallel. **Real 48-layer Qwen4:** the WP5c determinism fixture reproduces **bit-for-bit** (all 30 ids, same continuation text), `forward` vs `forward_one` parity **0 exactly** in all 6 post-split runs, every logits-row statistic and the independent-replay/un-normed-readout figures identical to all printed digits, peak working set unchanged (18.36/32.40/35.51 GiB; 40.87 GiB for the gen run), `sub0_core.dll` 2,848,768 → 2,870,272 B (+0.75%), backend TU 5.09 → 5.3 s plus decode.cpp's 3.2 s. **Decode latency: no measurable change, and that took a real experiment to establish.** A single before/after pair showed `forward`/`forward_one` at 58–61 s against ~50 s — a 15–20% apparent regression on a pass that adds no optimization, so it was chased rather than waved away. A **counterbalanced interleaved A/B (12 runs: pre/post `sub0_core.dll` swapped with no rebuild between trials, the same .exe and model throughout, run order reversed for half the trials)** settled it: pre 54.40 s / 52.25 s mean against post 52.83 s / 51.13 s, i.e. post *faster* on both metrics and both medians, with the unmodified PRE code itself reaching 59.89 s / 58.74 s — the "regression" range lies inside the pre-split arm's own spread. The clincher: **in every one of the 6 trials whichever arm ran SECOND was the slower one, and that flipped sign exactly with the ordering.** Between-arm differences are far below the 24% within-arm spread. `[[thermal-confounds-ab-wallclock-testing]]` reproduced textbook-clean, and the lesson is worth restating for whoever measures B20 next: **a single before/after pair on this machine is not evidence at this timescale — interleave, and counterbalance the order.** The 30-token generation figure moved 8.61 → 8.59 → 9.22 s/token across three separate sessions of the same binaries, which bounds that metric's own noise. One real defect found and fixed on the way, in my own diff: `decode.cpp` inherited `tied_head_row`'s `#pragma omp simd reduction(+ : s)` but not `backend.cpp`'s hard `#error` on a missing `_OPENMP` — a silently *weakened* guard, since that pragma is what permits the float reassociation, so a TU that lost OpenMP would decode to different last bits than `forward()` with the parity check as the only symptom. Guard restored. Left alone deliberately: `docs/INTEL_IGPU_WORK_PACKAGES.md`'s I08 status text (whoever integrates this owns that bookkeeping), I08's optimizer/backward/reduction phases, I07/I09, and the pre-existing `backend_cpu.cpp` references scattered through `tests/`/`tools/`/`CMakeLists.txt` left by the earlier area relocation — not this change's drift, and a repo-wide sweep of shared files would collide with other agents. **B20 is now unblocked**: its two fixes land in a 607-line decode TU, not a 3,146-line monolith. **Merged, independently reverified before merging** — read both new files in full, traced the shared/private boundary reasoning and the restored OpenMP guard by hand, then rebuilt from a clean isolated worktree and reproduced the two claims that matter most myself: neutral-config suites 0 failures on my own reconstructed baseline (matching my own independently-run figures exactly), and the real 48-layer WP5c determinism fixture reproducing bit-for-bit — all 30 ids and the full continuation text — from a from-scratch rebuild of this branch. Rebuilt+retested again at the actual merge commit before pushing. |
| 2026-09-08 | Claude Code | WP5c: the generation loop -- real Qwen tokenizer + real transplanted model + sampling, end to end | merged (`feature/wp5c-qwen-generation`, branch deleted local+origin) | new `tools/sub0llm-qwen4-gen.cpp` + one additive `CMakeLists.txt` block. NOTHING else: `qwen_tokenizer.*`, `moe_quant.hpp`, `transplant.hpp`, `engine_core.cpp`, `src/backends/*` and both test suites are untouched, by design | done | **Merged and independently reverified** -- rebuilt from source in an isolated worktree (own `sub0llm-configure` run at the real 48-layer axes, VOCAB=248320 exactly) and reproduced the headline generation, the exact ids, and determinism myself before merging; also independently re-ran the seed-varied case (seed 99 diverges after 2 tokens) to confirm the seed is genuinely wired, not just replayed. **THE WHOLE PIPELINE COMPOSES, and the real 48-layer model produced real English**: `"The capital of France is"` -> `" Paris. How many countries have capitals with"` (156.8s wall, prefill 10.39 s/token, generation 9.65 s/token, peak 37.11 GiB). Determinism at fixed seed proven at BOTH 4 and 48 layers (the 48-layer check matters on its own: it shows WP4e's resolve-pool residency does not leak into the answer). The `eos_id()` stop branch fired **for real** -- a chat-shaped prompt typed with literal special tokens (no templating layer) made the model's argmax `<\|im_end\|>` immediately after a completed turn. Regression counts reproduced EXACTLY from a clean reconfigure: 28,755,032/147 and 120,889 / 121,457 of 244. **One real defect found in engine code this WP did not touch and did NOT fix: `sub0::sample_token` (`engine_core.cpp`) holds 2.84 MiB of `std::array` locals at VOCAB 248,320 and dies with `STATUS_STACK_OVERFLOW` (0xC00000FD) on the 1 MiB Windows default -- measured by relinking without the workaround. The new tool takes a 32 MiB stack at its own link line; the engine-side fix (reused `thread_local` scratch, per AGENTS.md S1) is somebody's separate change.** SEQ_LEN finding: 128, and deliberately NOT raised -- throughput binds first by a wide margin (filling the existing window would take ~20 min), and WP5b's `act_grad`-elision is the real prerequisite to a longer one. Full writeup in `docs/WP4_SCOPE.md` S6 "WP5c". |
| 2026-09-08 | Codex | Whole-plan Intel review and benchmark/optimization scope | — | Intel design/work packages/research/performance docs, review report, independent backlog links | done | Whole-plan review closed eight scope gaps in documentation; 24 packages, six bounded spikes and five review checkpoints. Links/package-map/whitespace checks passed; no engine edits or hardware runs. |
| 2026-09-08 | Codex | Revise Intel plan for native execution, ISA/quantization and UMA research | — | `docs/INTEL_IGPU_BACKEND_DESIGN.md`, `docs/INTEL_IGPU_WORK_PACKAGES.md`, new `docs/INTEL_IGPU_ISA_MEMORY_RESEARCH.md`, independent backlog status links | done | Revised to 21 packages with native ISA/quantization, memory/residency and Level Zero/SYCL submission gates; Vulkan parked. Reconciled landed extractions and WP4f/WP5 status. No builds or device stress. |
| 2026-09-08 | GitHub Copilot | I08/I09 backend area layout extraction | — | `src/backends/cpu/`, `src/backends/cuda/`, `cmake/Backends.cmake`, `CMakeLists.txt`, `docs/ACTIVE_WORK_LOG.md` | paused | Area relocation + CPU API façade landed (`ba76503`/`fae914a`/`220afaf`), the phase `docs/INTEL_IGPU_WORK_PACKAGES.md` itself records as done. No commit has touched `src/backends/` since — marked `paused` rather than left `active` since no activity is observably in flight; re-check before assuming this is stale if Copilot resumes. Claude Code is taking I08's decode/API extraction phase only (row below) as a B20 prerequisite; I08's optimizer/backward/reduction phases and I09 (CUDA) remain open for whoever picks them up next. |
| 2026-09-08 | Claude Code | WP5a: real Qwen4 tokenizer (byte-level BPE, real `vocab.json`/`merges.txt`, gated against the real reference) | merged (`feature/wp5a-qwen-tokenizer`, branch deleted local+origin) | ALL NEW except two additive lines: `include/sub0/qwen_tokenizer.hpp`, `include/sub0/qwen_unicode.hpp`, `include/sub0/qwen_unicode_tables.hpp`, `src/qwen_tokenizer.cpp`, `tests/qwen_tokenizer_tests.cpp`, `tests/fixtures/qwen_tokenizer/`, `scripts/qwen_tokenizer_{tables,fixture}.py`, `docs/QWEN_TOKENIZER.md`; plus one source line in `CMakeLists.txt` and one in `tests/CMakeLists.txt`, and one `.gitignore` entry. Existing `tokenizer.hpp`/`tokenizer.cpp`/`casing.hpp` deliberately NOT touched. | done | **Merged and independently reverified** — rebuilt in an isolated worktree, hand-traced the NFC composition-blocking algorithm and the regex-alternation transliteration against the real regex quoted in the header, then reproduced the headline finding myself in a fresh Python session against the real HF-cached files (not just re-running the branch's own tests): `AutoTokenizer` loads this model as `Qwen2Tokenizer` (its hardcoded regex lacks `\p{M}`) and mis-tokenizes Thai `"กัน"` as `[24400, 64020]` where the model's own `tokenizer.json` pipeline gives `[148783]` — matched the branch's report to the id, both sides. Reran the suite both without the real files present (2473/9, three file-gated cases correctly skip) and with them (`SUB0_QWEN_TOKENIZER_DIR` pointed at the real cache) — 3041/9, matching exactly. Full suite 120,889/244, default-build baseline (118,416/235) unchanged. 1936 fixture rows all produced BY the real tokenizer. See `docs/QWEN_TOKENIZER.md` §2.1. Zero file overlap with WP5b, as planned. |
| 2026-09-08 | Claude Code | WP5b: scale the transplant to the full 48-layer model (`mmap` the MoE sidecar first, then attempt the real transplant + load, measure real RSS) | merged `e1bea23` (`feature/wp5b-full-scale-transplant`, branch deleted local+origin) | `include/sub0/moe_quant.hpp`, new `include/sub0/file_map.hpp`, `include/sub0/moe_math.hpp` (one missing `<algorithm>`), `tools/sub0llm-transplant.cpp`, `tests/qwen4_real_axes/*.hpp`, `tests/moe_quant_tests.cpp`, new `tests/qwen4_full48_quant_shape_tests.cpp`, `CMakeLists.txt`+`tests/CMakeLists.txt` (new targets only), `src/backends/{cpu,cuda}/CMakeLists.txt` (the build fix above) | done | **Merged and independently reverified** — rebuilt in an isolated worktree from the branch tip (not on trust), hand-checked `full48_totals.hpp`'s census arithmetic against my own computation, reran `sub0_frontend_tests` and the new `sub0_qwen4_full48_quant_shape_tests` and reproduced every number exactly. The branch's own independent PARENT_SCOPE fix (`ad2c818`) was byte-for-byte the same diagnosis as the one below, found in two concurrent sessions — merge conflict resolved by keeping the more detailed comment. Post-merge, the previously-blocking Muon `backend.cpp`/`muon.hpp` mismatch (item 2 below) is ALSO now resolved (Codex committed the remaining Muon files in `8a72c67`), so `main` builds clean end-to-end again: full `d196check` rebuild (123/123 targets) + `sub0_tests` (28,755,032 assertions/147 cases) + `sub0_frontend_tests` (118,416/235) + both shape-test targets, all green at the actual merge commit. **The full model runs.** 48-layer quantized-resident transplant produced (18.31 GiB blob + 37.11 GiB sidecar = 55.42 GiB on disk, 162.8 s, every gate green incl. `--verify` 0/1,074 bit-for-bit); loads in 19.0 s; `Model::forward` [6 x 248320] in 50.35 s with zero non-finite; `forward`/`forward_one` parity **0 exactly**; **peak working set 35.51 GiB** against 63.43 GiB. The `mmap` was the enabler, not a nicety: eager-read the same run needs ≥69.45 GiB, i.e. more than total physical RAM. Verdict: memory and disk are fine, **throughput (≈8.2 s/token) is now the only binding constraint** for interactive inference. Full writeup in `docs/WP4_SCOPE.md` §6 "WP5b". `tools/sub0llm-qwen4-forward.cpp` needed NO change. |
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
