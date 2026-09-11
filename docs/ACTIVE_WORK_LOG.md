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
| 2026-09-10 | Claude Code (Sonnet subagent, in-flight) | B25: explicit pipelined overlapped I/O for `ParallelExperts`' MoE resolve, replacing reactive mmap faults | `feature/b25-explicit-io-resolve`, isolated worktree, not yet merged | `include/sub0/moe_quant.hpp`, `src/backends/cpu/decode.cpp` (`ParallelExperts`), possibly `include/sub0/file_map.hpp` | in-flight, awaiting independent review before merge | **Directly targets a user-set session goal: "actual token-generation near the CPU's theoretical performance."** Computed this session: if `ParallelExperts`' 10 threads achieved their full requested I/O concurrency, decode would floor near ~847 ms/token against the current measured ~5.48 s/token (~6.5x gap) — B21's own still-open concurrency-ceiling finding is the single largest identified cause. This work implements a scoped, Sub0Llm-local version of the fix Sub0MemPage's own design already specifies (explicit batched overlapped I/O instead of reactive `mmap` page faults), pipelined so a completed expert's compute starts immediately rather than waiting for the slowest of the batch (a mid-dispatch correction from an initial two-phase design). Also considering CPU-cache prefetch hints in the dequant/transpose compute path itself (~55% of a cold resolve's own cost, WP6b) as a secondary, time-permitting lever. Correctness gate is BIT-EXACT (pure I/O-mechanism swap, no numerical change), unlike B24's BF16 tolerance-based gate. |
| 2026-09-10 | Claude Code | B24 Phase 2: native-quant backbone dequant-vs-resident crossover, measurement only | merged `7e59457` (`research/b24-native-quant-backbone`) | new `benchmarks/backbone_dequant_bench.cpp` + `benchmarks/CMakeLists.txt` entry, `docs/BACKBONE_PRECISION.md` (new §2c) | done | **The 2a-vs-2b fork §2 left open is resolved, decisively, not close.** Measured on the real Qwen3.8-Flash-Next UD-IQ1_S shards (not synthetic): dequantizing `Q5_K`/`Q6_K`/`Q8_0` backbone tensors inline on every read (2b) costs 5-12x more than the DRAM bandwidth it saves at 100% access density (unlike the MoE sidecar's own ~2%-density case, WP6b), because native-quant formats only save ~3-6x bytes vs F32 here (not the sidecar's blended ~12x) while the per-element dequant cost stays roughly the same. **Recommendation: build 2a (dequantize once into a resident buffer) — Phase 2 collapses to a resident-format choice for `sub0llm-transplant`'s output, not a new inline per-token dequant engine path.** Also surfaced: this host's real achieved DRAM bandwidth (~28-35 GB/s, measured) is ~2x lower than `docs/BACKBONE_PRECISION.md` §0's assumed 70 GB/s figure — doesn't change the 2a/2b verdict (both sides share the same measured BW) but changes Phase 1's own absolute win size. Honesty gap flagged not closed: not re-run under B21-style memory-pressure ballast; judged unlikely to flip a 5-30x margin, Q8_0's ~5-7x noted as the narrowest and worth rechecking first if revisited. **Independently rebuilt and rerun before merging** — my own run against the same real GGUF shards reproduced the same decisive verdict and direction on every format (my own margins: Q8_0 ~7.5x, Q5_K ~10.6x, Q6_K ~12.4x, within the branch's own reported 5-30x range). No production code touched. Ran in parallel with a separate Phase 1 (BF16 implementation) agent, still in flight as of this row. |
| 2026-09-10 | Claude Code (Sonnet subagent, resumed after an interrupting rate limit, not a task failure) | B24 Phase 1: BF16 backbone storage (implementation) — COMPLETE, merged | `feature/b24-bf16-backbone` (commit `8e4d3cf`) | `tools/sub0llm-transplant.cpp`, `include/sub0/model_file.hpp`, `src/engine_core.cpp`, `include/sub0/core.hpp`, `src/backends/cpu/backend.cpp`, `src/backends/cpu/{api.cpp,api.hpp,decode.cpp}`, `tools/configurator.cpp`, `tools/sub0llm-qwen4-forward.cpp`, `include/sub0/{bf16.hpp,param_store.hpp}` (new), every `*_math.hpp`, `tests/{scratch_embed_tests.cpp,transplant_fixture_tests.cpp}` | done | Full spec + measured results in `docs/BACKBONE_PRECISION.md` §1 (now §1c/1d/1e) and `docs/INDEPENDENT_REVIEW_BACKLOG.md` B24. The resuming agent found and fixed 4 real defects the interrupted pass left uncaught (an `if constexpr`-in-a-non-template gotcha that broke the first real build, a missing namespace qualification, two pre-existing test files broken by the new templated kernel signatures) — see §1e for detail. Measured on the REAL 48-layer artifact: forward-pass L2-relative logit diff ~0.199 (5/6 argmax-identical, most likely MoE top-k routing discontinuity, not a defect), `forward()`/`forward_one()` parity bit-exact, coherent English preserved, decode ~9% faster, load time roughly halved, peak memory −9.24 GiB. `sub0_tests` confirmed architecturally incompatible with ANY `FORWARD_ONLY` config (BF16 requires it by design) — pre-existing, not new; `sub0_frontend_tests` green under both dtypes (120,889/244). **Independently re-verified before merging** — hand-tested `bf16.hpp`'s RNE rounding (both tie cases, NaN, inf), reviewed and confirmed the `if constexpr` fix, independently reran `forward`/`forward_one` on both real 48-layer artifacts (bit-exact each), independently recomputed the L2 diff from my own logit dumps (0.19889, exact match) and argmax agreement (5/6, same disagreeing row), independently reran `sub0_frontend_tests` (120,889/244, exact match). Every reported number reproduced exactly. |
| 2026-09-10 | Claude Code | B21: root-cause `ParallelExperts`' decode concurrency shortfall | `main` (docs only, no engine change) | `docs/INDEPENDENT_REVIEW_BACKLOG.md` (B21 follow-up section); temporary instrumentation added to and reverted from `src/backends/cpu/decode.cpp` (no diff against `main`) | open, findings updated | Also: initialized and pushed `github.com/CraigHutchinson/Sub0MemPage` (design-only spec repo, consolidating three parallel research passes) as the general-purpose async-paging library this whole investigation motivates. Direct wall-clock thread instrumentation of `ParallelExperts` refines the original "looks like one thread" finding: `omp_get_num_threads()` reports the real team size (10) every call; only the very first call of the whole process is fully serial (one-time `libomp` pool cold-start); every call after that shows real but PARTIAL concurrency, plateauing at 6 of 10 threads running with genuinely overlapping wall-clock durations, the remaining 4 starting ~90-120ms later as the first group finishes. Windows core parking tested directly (duplicated+activated "High performance" power scheme, reverted after) and ruled out — throughput and the 6-then-4 pattern were unchanged. Per-expert duration during the "concurrent" bursts (30-99ms) is 3-9x the isolated harness's own single-thread baselines (cold ~11.4ms, warm ~6.5ms) — real contention under concurrent load in the live engine that the isolated harness's smaller memory footprint does not reproduce. Root cause still open; see B21 in the backlog for the updated acceptance criteria. |
| 2026-09-09 | Codex + Sol/Terra agents | Intel Phase 1/R0 groundwork and serialized USM runtime window | merged to `main` through the Intel-groundwork landing | Intel plans/evidence, I01 admission/checksums, I07b.0 boundary, I21 schema/orchestrator; no production engine edits | paused at explicit gate | R0 runtime passed on Intel `0x7d67`; direct ZE is unavailable on the installed development tuple. I01 records named blockers. I21 synthetic validation passes but performance comparison remains ineligible until `prepared_copy` can run one selected arm per process. Resume there, then reserve hardware for I21; S1.5 and S2/I18 remain the next independent Sol spikes. |
| 2026-09-09 | Codex + Sol/Terra reviewers | Intel groundwork PR handoff | `research/intel-groundwork` | Intel design/work packages/findings/runbook and standalone runners | done; superseded by R0 checkpoint | Initial review and compile gates completed; the later R0 row records the executed runtime controls and authoritative resume point. |
| 2026-09-09 | Codex + Terra | S0a/S0b/S0c tandem-backend spike groundwork | `research/intel-groundwork` | Tandem leaf report; plan/provider design/log | prepared; execution pending | Findings and A/B design reviewed with explicit provenance, native API lifetime constraints and dependency order. Native dependencies, second-target coverage and new measurements remain open. No installs, hardware runs or main edits. |
| 2026-09-09 | Codex + Terra | Backend naming and portable/native composition review | `research/intel-groundwork` | Design/work-package documentation only | done | Primary-source portability/naming review integrated in `BACKEND_PROVIDER_DESIGN.md` and I07. Mixed Intel-native/oneDNN providers and a conditional portable SYCL executor remain distinct; no production interfaces or hardware runs. |
| 2026-09-08 | Codex + Sol agents + Terra | Intel USM next-spike preparation | `research/intel-groundwork` | Capability/prepared-copy leaf sources, runners and reports; execution policy and package links | done; superseded by R0 checkpoint | Sol-authored probes and Terra research were integrated; later R0 evidence records successful runtime correctness and bounded exploratory timings. No installs or production engine changes. |
| 2026-09-08 | Codex + Terra | Windows USM official-doc applicability review | `research/intel-groundwork` | Isolated Intel worktree documentation only | done | Primary-source review and installed-header audit integrated in `INTEL_IGPU_WINDOWS_USM.md` and S1/I19 follow-ups. No hardware runs or main Qwen changes. |
| 2026-09-08 | Codex | Intel I00 and S0/S1 groundwork | `research/intel-groundwork` | Isolated worktree `out/worktrees/intel-groundwork`; standalone probes/scripts/reports | paused | Both groundwork sizes and five negative cases pass; ten f32 projection shapes and three USM modes measured, with raw records archived. Partial I00/S0/S1/I05/I19 only; direct submission, IQ, chains and capacity remain open. No active benchmark and no main Qwen source edits. |
| 2026-09-09 | Claude Code | Memory audit (precise accounting of the ~41 GiB decode peak, incl. the already-known `act_grad` waste) + an isolated single-expert micro-benchmark decomposing dequant/transpose/FFN/disk-I/O costs separately | merged `23ec79e` (`feature/moe-mem-audit-microbench`, branch deleted local+origin) | `src/backends/cpu/internal.hpp`, `src/backends/cpu/backend.cpp`, `src/backends/cpu/decode.cpp` (comment only), new `benchmarks/moe_expert_bench.cpp` + one additive block in `benchmarks/CMakeLists.txt`, `docs/WP4_SCOPE.md` (new §6 WP6a/WP6b), `docs/CPU_PERF_BACKLOG.md` (2a closed, 2d added). **`docs/INDEPENDENT_REVIEW_BACKLOG.md` deliberately NOT touched** -- the shared tree had another agent's uncommitted edits to it | done | **The whole ~41 GiB is now accounted for line by line, with real tooling, and the one dead item in it is gone.** VMMap is not installed on this host, so the breakdown came from a purpose-built `VirtualQueryEx` + `QueryWorkingSetEx` walk of the LIVE gen process (classifying every committed page as private / mapped-file / image and naming each mapping's backing file), cross-checked against an independently compiled probe TU printing `sizeof(Worker)` and every arena's `constexpr` byte count -- the two agree to the byte. **Measured mid-decode, before the fix: private 32.835 GiB (18.310 `g_param_data` + 14.045 one `Worker` + 0.480 everything else), sidecar mapping 37.113 GiB committed but only 4.809 GiB RESIDENT, images 0.014.** Confirmed as part of the audit rather than assumed: `g_param_grad`/`g_param_m`/`g_param_vel` really are unallocated (no such region exists; 4x would have been 73.24 GiB), and `Worker::moe_cache`'s 150 MiB batched pool is **never allocated at all** in a `forward_one`-only run -- `op_moe` is its only `allocate()` caller. Decode's own per-thread pools are 250.0 MiB across 10 threads and the decode-persistent caches 177.4 MiB (KV 24.0, **GDN 149.6**, QSA 3.75). **The fix**: `ACT_GRAD_FLOATS = FORWARD_ONLY ? 1 : ACT_CAP`, mirroring `WORKER_GRAD_FLOATS` exactly, plus an EMPTY grad span out of `arena_alloc` (the treatment `mk_param` already gave a parameter leaf) and a refusal at `backward()`'s own seam. **`sizeof(Worker)` 14.045 -> 7.023 GiB; peak working set 40.98 -> 33.96 GiB (-7.02, -17.1%); `[mem] after graph_reset` 32.41 -> 25.39.** The point is the row after: **the sidecar's resident share at a comparable point in the SAME run rose 4.809 -> 7.634 GiB** -- the reclaimed pages went straight to the file cache B20 found decode to be bound by. Throughput 5.61 -> 5.53 s/token, i.e. unchanged, exactly as that finding predicts. **Verified at BOTH configurations the shared `Worker` layout serves, each baseline re-taken on this tree by stashing the diff rather than cited from an earlier session**: real 48 layers reproduces the WP5c determinism fixture's **all 30 ids and full continuation byte-for-byte**; neutral d196 **28,875,042/147** and **120,889/244** identical before/after; and a GDN-only training-capable build (`--gdn-full-attn-stride 2`, no GR/MoE/QSA, so `FORWARD_ONLY` is false and the full-size `act_grad` MUST survive) gives identical `[grad]` 13/3 and `[engine]` 24,030,273 counts before and after. **The benchmark** (`sub0_moe_expert_bench`, own target, links nothing, reads all dims from the real sidecar's own header, plain `main()` because Catch2's warm-up destroys the cold-read arm) puts real numbers on the split for the first time -- per expert, 4 runs at fresh seeds: cold read 2.98 ms, dequant **IQ1_S 2.21 / IQ4_NL 2.39 / IQ2_XXS 13.73**, transpose 1.53/1.59, `expert_ffn_row` **0.50**, full pipeline cold 10.58 / warm 6.80. **Of a cold resolve: 35.7% page-in, 40.2% dequantize, 14.7% transpose, 4.7% the FFN.** That is the answer to WP4e's open re-encode question. **Two real findings reported and NOT fixed, per this project's discipline**: (1) `dequantize_iq2_xxs` is ~6x slower per element than either neighbour and 14,336 of the file's 73,728 planes use it -- `docs/CPU_PERF_BACKLOG.md` 2d names the concrete inner-loop difference against `dequantize_iq1_s`, because "it is a harder format" does not survive reading the two side by side; (2) 480 resolves/token x 10.58 ms = 5.08 s/token against a measured 5.53, i.e. the real decode behaves almost as if its ten threads were one -- which explains rather than contradicts B20's 2% parallelization result. Also noticed, unrelated and unfixed: a `--vocab-exact` (B19) build emits no `tokenizer.tok`, so three `[engine]` cases FAIL (not skip) on `load_tokenizer(default_tokenizer())` -- identically before and after this change, so pre-existing. **The disk-residency problem itself was deliberately left alone** (no prefetch, no `SetProcessWorkingSetSize`, no re-encode): that design is separate work, and it should now be informed by WP6b's per-arm table rather than by a single conflated figure. **Merged, independently reverified before merging** — read the `act_grad` fix and `arena_alloc`/`backward()` changes in full, confirmed the empty-span handling and the new refusal are both correct AND necessary (not just defensive: without the refusal, `backward()`'s `loss->grad[0]=seed` would write out-of-bounds into the now-empty span). Rebuilt from scratch and reproduced the two claims that matter most myself: a GDN-only trainable config (own `--gdn-full-attn-stride 1` config, MoE/GR/QSA off) reproduces its full suite incl. `[grad]`-tagged tests exactly, and the real 48-layer determinism fixture reproduces bit-for-bit with the exact reported peak-memory drop (40.98→33.96 GiB) and the exact `after graph_reset` figure (32.41→25.39). Independently reran the new benchmark against the real sidecar and reproduced both headline numbers within normal run-to-run variance (37.6/45.9/14.7/4.9% vs the branch's 35.7/40.2/14.7/4.7%; IQ2_XXS ~5.6x slower, matching). Rebuilt+retested again at the actual merge commit before pushing. |
| 2026-09-09 | Claude Code | B20: cache-block `transplant::transpose_out_in`'s naive strided transpose, then parallelize per-expert resolve across a layer's `EXPERTS_PER_TOK` selection | merged `d2bbea4` (`feature/b20-decode-perf`, branch deleted local+origin) | `include/sub0/transplant.hpp` (part 1, alone); `include/sub0/moe_math.hpp`, `src/backends/cpu/{decode.cpp,internal.hpp,backend.cpp}` (part 2). **`include/sub0/moe_quant.hpp` was NOT touched at all** -- neither fix needed it, so the S0Q1 format and its reader are untouched by this branch | done | **Both fixes landed, each profiled independently as B20 asked, and the honest answer is that one of them worked and the other found a wall.** Baseline reproduced first on this tree (not taken on trust): `transpose_out_in` **52.9%** of sampled CPU (B20 recorded 52.5%), `Total Thread Count: 1`, **9.14 s/token**. **Part 1 (cache-blocked transpose, `0759fd9`): 52.9% -> 21.5%, 9.14 -> 5.68 s/token (1.61x).** All three transposes now share ONE blocked core (`transpose_block`) differing only in destination stride/column offset. Tile order and size were MEASURED (both real expert-plane shapes, 3 interleaved trials, 8/16/32/64/128 x both loop orders): contiguous STORES win, and B=16 is not "best by a little" -- B=32 collapses to 1.3-1.8x on one shape only, because rows 2560 floats apart map onto just two L1 set positions here, so at B=32 a collision group is 16 rows against 12/8-way associativity. A guessed constant would have shipped that. **Part 2 (parallel per-expert resolve, `dec5273`): `Total Thread Count: 1 -> 10`, every hot function's CPU time down ~2-4x in the same window -- and 5.77 (1 thread) vs 5.65 (10 threads) s/token, i.e. 2%.** That is the finding, not a footnote: **decode is now DISK-bound, not CPU-bound.** Measured in steady-state decode -- **13,610 hard page faults/sec** (`\Memory\Pages Input/sec`), 55.7 MB/s of matching physical disk reads, **16.3% processor utility**; ~319 MB faulted per token against the ~253 MB of encoded expert bytes a token's 480 routed-expert reads need, with no warm-up trend over 30 tokens. Part 1 left decode 94% CPU-busy (56.4s CPU / 60.0s elapsed); part 1 is what exposed the wall, and part 2 proves the CPU side is no longer it. **The S0Q1 sidecar's demand-paging behaviour is the real next item** (37.11 GiB mapping that does not stay cached beside the 18.31 GiB f32 blob in 63.4 GiB of RAM) -- reported, not fixed, per this project's own discipline. **Resolve-pool concurrency design, stated because B20 asked for the reasoning and not just the choice**: neither of B20's two suggestions. More `Worker`s is unavailable at these axes for a measured reason -- a Worker owns two ACT_CAP arenas, ~14 GiB here (the gen tool's `[mem]` line moves 18.36 -> 32.40 GiB on first touch), and decode uses neither, since `forward_one` allocates no arena slot and no node. A concurrency-safe SHARED pool would need per-slot reservation plus pinning against round-robin overwrite -- refcounting on the hottest path -- for a hit rate that is provably ZERO in decode (top-k indices are distinct, the key includes the layer, and 480 resolves/token round-robin an 8-slot pool clean before the next token asks). So each decode thread brings its own ONE-slot pool + FFN accumulators, ~25 MiB vs a Worker's 14 GiB; `op_moe`'s batched path keeps its 8-slot pool untouched, since T rows through one layer DO re-select experts. **Bit-for-bit throughout, verified not asserted**: `--verify` against the REAL 48-layer artifact **0 mismatches of 1074 tensors** (zero new tolerance); `forward` vs `forward_one` at the real 48 layers **max |diff| = 0 EXACTLY** with the 10-thread decode path against the serial batched one; the WP5c determinism fixture reproduces all 30 ids and the full continuation text at every stage; neutral d196 suites **28,875,042 / 147** and **120,889 / 244**, identical to pre-change runs taken on THIS tree by stashing the diff and rebuilding. **One real correctness trap found and closed**: FTZ/DAZ is per-thread MXCSR set in `ensure_thread_built`, which decode's new threads never call -- a thread that missed it computes DIFFERENT floats once an intermediate goes subnormal, so the bit-for-bit claim would have been quietly false. `set_flush_denormals` moved backend.cpp -> internal.hpp and every decode worker sets it. **Two things left alone deliberately**: `gguf::to_f32`'s redundant re-zeroing (B20's own optional third item -- skipped BECAUSE of the finding above: no CPU-side micro-optimisation is measurable at the token level while decode is disk-latency bound, so it would be an unmeasurable change), and `_kmpc_barrier` spin (25.4s of 58.5s CPU across the 48 short regions a token opens -- costs ~0.4 of 24 cores and, measured, zero wall clock: KMP_BLOCKTIME=0 gives 5.64 vs 5.65 s/token, so tuning the OpenMP blocktime globally, which would also hit train_batch's long regions, is not justified on this evidence). **`docs/INDEPENDENT_REVIEW_BACKLOG.md` was NOT edited** -- the shared tree had uncommitted changes to it from another agent when this work started, so B20's own entry still reads as open; whoever merges this branch owns that bookkeeping. **Merged, independently reverified before merging** -- hand-verified the transpose index algebra (all three call sites) and the two-phase MoE combine's summation-order-preservation myself before trusting either claim, then reproduced every empirical number from a from-scratch rebuild in an isolated worktree: `--verify` 0/1074, `forward`/`forward_one` parity 0 exactly, the determinism fixture byte-for-byte, neutral suites + the transplant/moequant-tagged tests specifically all green, and independently measured 5.57-6.00 s/token myself. Rebuilt+retested again at the actual merge commit before pushing. `docs/INDEPENDENT_REVIEW_BACKLOG.md`'s B20 entry still needs updating to Done -- next available slot in this session. |
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

---

**2026-09-11 Claude Code — B28 dispatched (active).** Following B27's finding that MoE resolve is now
~100% compute-bound on this machine (I/O wait 0.02%), dispatching a Sonnet subagent (isolated worktree,
`feature/b28-iq2xxs-dequant-fix`) to: (1) fix `gguf::dequantize_iq2_xxs`'s known ~6x-slower-than-neighbours
defect (`docs/CPU_PERF_BACKLOG.md` §2d — hoist the tail-bound check out of the inner loop per that
section's own diagnosed hypothesis), and (2) add software-prefetch hints to the dequant/transpose compute
path in `moe_quant.hpp`/`moe_math.hpp` (the user's own suggestion). Files: `include/sub0/gguf.hpp`,
`include/sub0/moe_quant.hpp`, `include/sub0/moe_math.hpp`, `tests/gguf_tests.cpp`,
`benchmarks/moe_expert_bench.cpp`. Gate: bit-exact vs `tests/gguf_tests.cpp` + `--verify` against the real
48-layer artifact (numerical decode, must not change), plus a real before/after decode-throughput
measurement. Do not touch `docs/ACTIVE_WORK_LOG.md`'s shared rows above.

---

**2026-09-11 Claude Code — B28 done, merged, pushed.** Fixed `dequantize_iq2_xxs`'s ~6x-slower decode
(real cause: per-element branch on a sign bit, not just the tail-bound guard). Independently reverified
(numbers, tests, and a real 48-layer decode rerun all reproduced) before merging `--no-ff` to `main`
(`8ed590d`), full `d196check` rebuild + both suites green, pushed. Decode: ~4.78-4.85 s/token ->
~3.94-3.98 s/token (~18-20% faster). Prefetch-hint task attempted, measured no benefit, reverted. See
`docs/INDEPENDENT_REVIEW_BACKLOG.md` B28. Session `/goal` ("token-gen near CPU theoretical performance")
still open -- theoretical floor under B27's compute-bound model is ~837 ms/token; next lever is B21's own
concurrency-ceiling finding, now suspected to be a compute-side bandwidth/cache-contention ceiling rather
than disk-related.

---

**2026-09-11 Claude Code — B29 dispatched (active).** Following B28, re-measure `ParallelExperts`' real
10-thread concurrency in the LIVE engine post-dequant-fix, and directly test B27's hypothesis that B21's
own "~2-3x ceiling, independent of thread count" is a compute-side memory-bandwidth/cache-contention
effect against the engine's 25+ GiB resident footprint (not disk-I/O concurrency, now ruled out by B27).
Files: temporary instrumentation only in `src/backends/cpu/decode.cpp` (reverted before commit, per B21's
and B27's own precedent), plus a possible new isolated harness variant with a large resident-ballast
footprint if the existing one (B21's own harness, referenced in this file's B21 section) doesn't already
cover the post-B28 numbers. Do not touch other agents' shared files without checking this log first.

---

**2026-09-11 Claude Code (subagent) — B29 done, on `feature/b29-concurrency-ceiling`, NOT merged.**
Confirmed the bandwidth-ceiling hypothesis directly: `moe_expert_bench`'s own real per-arm numbers imply
a single thread's resolve+FFN already demands ~12-15 GB/s against this host's own measured ~28-33 GB/s
ceiling (`docs/BACKBONE_PRECISION.md` §2c). Live-engine wall-clock instrumentation (reverted, same
technique as B21/B27) found a REGIME CHANGE from B21's own finding: post-B28, every sampled
`ParallelExperts` call shows FULL serialization (zero measured overlap, not B21's earlier "~2-3x
partial"), with per-expert cost close to the isolated single-thread baseline (no more of B21's 3-9x
inflation). Direct A/B on `sub0llm-qwen4-forward --tokens 6` (real 48-layer BF16 artifact, interleaved
runs): `MOE_DECODE_THREADS` 10 -> 3.75-3.79 s/token; forced to 3 -> 3.79 s/token (no change); forced to 1
-> **3.42-3.47 s/token, ~9% FASTER** — concurrency above 1 has a real, measured, negative cost here, not
just no benefit. `forward`/`forward_one` parity stayed exactly 0 at every thread count. **Fix merged into
this branch**: `MOE_DECODE_THREADS` pinned to `1` in `src/backends/cpu/internal.hpp` (was
`min(DEFAULT_THREADS, EXPERTS_PER_TOK)`), also freeing ~225 MiB of decode-thread pools. Regression:
`sub0_tests` 9,540,077/147 (bit-for-bit identical with/without the change at the same toy config, checked
not assumed) and `sub0_frontend_tests` 120,889/244 (exact match to the session's established baseline),
both green. Instrumentation confirmed reverted (`git diff --stat` on `decode.cpp` clean). See
`docs/INDEPENDENT_REVIEW_BACKLOG.md` B29 (new) and B21 (closed by B29). Branch not merged — awaiting
this session's own independent reverification per this session's standing practice.

---

**2026-09-11 Claude Code — B29 done, merged, pushed; B30/B31 filed, session goal status synthesized.**
`MOE_DECODE_THREADS` pinned to 1 (`src/backends/cpu/internal.hpp`) -- independently reverified (own
rebuild+rerun of both suites, own A/B on the real 48-layer artifact confirming direction if not exact
magnitude), merged `--no-ff` (`b79c85c`), full `d196check` rebuild + suites green, decode hash unchanged
from B28's merge (confirms numeric inertness), pushed. B21 closed. B30 synthesizes B27->B28->B29 into a
single honest assessment of the session's `/goal` ("token-gen near CPU theoretical performance"): decode
is DRAM-bandwidth-bound, a single thread already achieves ~44-52% of this host's real bandwidth ceiling,
and the bottom-up computed floor (~2.8-3.0 s/token) now matches measured decode (~3.57-3.67 s/token)
closely -- no more free/cheap levers of the kind found this session remain. Combined result:
~5.5-6.0 s/token baseline -> ~3.57-3.67 s/token, ~35-40% faster, real and independently verified at every
step. B31 filed (not started, scoped, Large effort): fusing the resolve pipeline's dequant/transpose/FFN
stages to cut ~3 redundant DRAM round-trips per expert -- the one remaining structural lever, a genuine
redesign not a same-pattern fix, deliberately not attempted opportunistically this pass.

---

**2026-09-11 Claude Code — B32: real llama.cpp comparison on the same model/host.** User asked "how fast
is llama.cpp latest with this model" -- built `llama-bench` from the local `D:\Craig\llama.cpp-qwen4exp`
fork (commit ccc3646, 2026-09-07) and ran it CPU-only against the real source GGUF
(`D:\ModelWeights\Qwen3.8-Flash-Next-GGUF\UD-IQ1_S`). Result: llama.cpp generates at ~1.0-1.7 t/s
(~0.59-0.94 s/token) vs Sub0Llm's current ~3.57-3.67 s/token -- **llama.cpp is ~4-6x faster on the same
hardware/model**. This revises B30's "near the floor" conclusion: B31 (fusing the resolve pipeline's
dequant->transpose->FFN stages to cut redundant DRAM round-trips) is now directly evidenced as real,
available headroom, not speculative. See docs/INDEPENDENT_REVIEW_BACKLOG.md B32 for full numbers/caveats.

---

**2026-09-11 Claude Code — B33 dispatched (active).** Continuing docs/BACKBONE_PRECISION.md's own
originally-planned Phase 2, per the user's explicit direction, now scoped precisely by B24 Phase 2's own
decisive finding (inline block-dequant loses 5-30x -- do NOT build that): add a new resident `PARAM_DTYPE`
value for a flat, block-free 8-bit float (FP8 E4M3), promoted exactly like `bf16.hpp`'s `Bf16CPtr` (cheap
bit-manipulation, no per-block scale lookup, no auxiliary state) -- NOT a GGUF-style per-block-scaled
Q8_0/Q4 format, which would repeat the already-ruled-out 2b shape. Files: new `include/sub0/fp8.hpp`
(mirrors `bf16.hpp`'s structure exactly), `include/sub0/param_store.hpp` extension, `include/sub0/
model_file.hpp` (`ParamDtype::FP8`), `tools/sub0llm-transplant.cpp` (`--prec-param fp8` output mode),
`tools/configurator.cpp` (generated `Dtype`/`PARAM_DTYPE` plumbing). Kernels are already templated on
weight-pointer type from B24 Phase 1 -- no `*_math.hpp` changes expected unless the proxy type needs a new
capability. Correctness gate: same tolerance-based precedent as B24 Phase 1 (Q8_0-vs-F32 ~3.5e-3, BF16's
own ~0.199 L2 logit diff as the most recent real comparator), full suite green, real measured decode
throughput AND peak resident memory before/after on the real 48-layer artifact. Do not touch other agents'
shared files without checking this log first.

---

**2026-09-11 Claude Code — B31 dispatched (active), in parallel with B33 (no file overlap).** Cache-tiled
fusion of the MoE resolve pipeline (dequant->transpose->FFN), per B30/B32's findings and the user's own
explicit direction to make this L1/L2/L3-cache-aware with constexpr-derived tile sizes rather than merely
reducing pass count. Files: `include/sub0/moe_quant.hpp`, `include/sub0/gguf.hpp` (dequant output shape),
`src/sub0/transplant/*` or wherever `transpose_out_in` lives, `include/sub0/moe_math.hpp`
(`expert_ffn_row`'s input contract), possibly `src/backends/cpu/decode.cpp`. Does NOT touch
`param_store.hpp`/`bf16.hpp`/`fp8.hpp`/`model_file.hpp`/`configurator.cpp`/`sub0llm-transplant.cpp` --
those are B33's. Gate: bit-exact numerical output (pure data-flow restructuring, no precision change),
full suite green, real before/after decode throughput on the real 48-layer artifact.

---

**2026-09-11 Claude Code — B31 DONE, branch `feature/b31-cache-tiled-resolve`, NOT merged (session owner
reverifies, same as every prior package).** Full writeup: `docs/INDEPENDENT_REVIEW_BACKLOG.md` B31 row,
`docs/CPU_PERF_BACKLOG.md` §2e. Design decision made and documented inline: ELIMINATED the transpose stage
for decode's hot path (not a 3-stage cache-tiled pipeline) — `expert_ffn_row`'s existing per-output
accumulation order is reproducible term-for-term as a direct dot product against the plane's UNTRANSPOSED
GGUF source order, so the transpose was never mathematically necessary for this consumer. New, additive
surface only: `moeq::dequantize_expert_source`/`ExpertCacheSource` (moe_quant.hpp),
`moe::expert_ffn_row_source`/`forward_row_via_run_ex` (moe_math.hpp, `forward_row_via_run` now a thin
wrapper), `MoeDecodeExpertCacheSource` (internal.hpp), decode.cpp's `MoeDecodeThread`/forward_one MoE call
site switched to the fused path under `USE_MOE_QUANT` only — op_moe's batched path and every existing
test/tool keep the ORIGINAL `ExpertCache`/`dequantize_expert`/`expert_ffn_row` contract untouched (AGENTS.md
S10). Bit-exact: proven in `expert_ffn_row_source`'s own comment, checked by a new `tests/moe_quant_tests.cpp`
case (both paths, same encoded bytes, bit-for-bit output), and by `forward`/`forward_one` parity staying
exactly 0 on the real 48-layer artifact with the fused path live. `sub0_frontend_tests` 120,923/245 (was
120,889/244 — +1 case/+34 assertions, the new test only). Real measured on the real 48-layer BF16 artifact
(`sub0llm-qwen4-forward --tokens 6`, thermal-confound-aware interleaved A/B against a sibling worktree at
the pre-B31 commit, 4 runs each): **3.675 s/token before -> 3.385 s/token after, ~7.9% faster**, every
AFTER run faster than every BEFORE run. Reported honestly as smaller than the ~40-50% a pure
bandwidth-utilization estimate implied — MoE resolve is only part of decode's per-token cost, and the
eliminated transpose traffic was evidently not the dominant single-thread bandwidth cost. A
`constexpr`/L2-cache-derived row tile was added per the task's cache-aware brief but measured to make no
reliable difference (same shape of honest null result as B28's own prefetch finding), kept anyway since it
costs nothing and documents the real cache budget. Session goal ("actual token-generation near CPU
theoretical performance") is NOT fully met by this package alone — real progress, gap narrows further but
does not close; see B31's own backlog row for the full honest accounting.

---

**2026-09-11 Claude Code — B31 done, merged, pushed.** Eliminated the MoE resolve's transpose stage
entirely (not just tiled it) -- a proven bit-exact reordering (expert_ffn_row's accumulation order is
reproducible as a direct dot product against GGUF's untransposed source layout, since transpose_out_in is
a pure permutation). Independently reverified: hand-traced the permutation semantics myself, reran the
real 48-layer model (max diff 0, logits identical to my own earlier recorded numbers), suites match exactly
(120,923/245). Merged `--no-ff` (`d75449b`), full d196check rebuild + suites green, pushed. Real measured
decode: ~3.675 s/token -> ~3.385 s/token (~7.9%, agent's controlled A/B) -- smaller than the theoretical
estimate implied, honestly reported (MoE resolve is only part of decode's cost; the eliminated transpose
traffic wasn't as dominant as the byte-counting arithmetic suggested). B33 (FP8 backbone) still running in
parallel, no file overlap confirmed.

---

**2026-09-11 Claude Code — B33 done, NOT merged (real negative result).** FP8 (E4M3) as a third resident
`PARAM_DTYPE` (following BF16's own architecture exactly) is correctly implemented and correctness-gated
(bit-exact parity, exhaustive round-trip tests, zero kernel changes needed), but real-model measurement
showed a ~60% decode SLOWDOWN vs BF16 (agent's number) and a ~40-46% slowdown (my own independent
interleaved A/B on the real 48-layer artifacts) -- the opposite of the modest win expected, most likely
because `Fp8CPtr`'s multi-branch exponent-remap widen costs more per-element CPU than the DRAM bandwidth
it saves. Quality also markedly worse than BF16 (L2 0.43 vs 0.199). Kept unmerged on
`feature/b33-fp8-backbone` for a future branchless/lookup-table `fp8_widen` follow-up. See
docs/BACKBONE_PRECISION.md S2d and docs/INDEPENDENT_REVIEW_BACKLOG.md B33 for full detail. Session total
this thread: B27->B28->B29->B31 real wins (baseline ~5.5-6.0 s/token -> ~3.38 s/token); B25 and B33 real,
correctness-clean, but not currently worth shipping.

---

**2026-09-11 Claude Code — B34 dispatched (active).** User asked us to cross-check llama.cpp's own
implementation for concrete techniques. Two real findings from directly reading `ggml`'s x86 quant kernels
and probing our own codegen: (1) llama.cpp's core technique is quantized-activation x quantized-weight
integer SIMD dot products (a genuinely different kernel shape, filed for a later dedicated pass, NOT this
package); (2) more immediately actionable -- Sub0Llm's own hot reduction loops (`s += x[i]*w[i]` in
moe_math.hpp/gdn_math.hpp/qsa_math.hpp/gated_residual_math.hpp) do NOT auto-vectorize AT ALL, confirmed by
compiling the exact loop shape standalone (Clang: "cannot prove it is safe to reorder floating-point
operations") -- this is true regardless of `-march=native`/AVX2 already being enabled in our build configs
(SUB0_NATIVE=ON), the compiler simply refuses to vectorize a strict scalar FP reduction. A quick multi-
accumulator restructuring test got real SIMD codegen (SLP vectorizer kicked in, ymm registers appeared)
immediately. B34 dispatched to apply this restructuring to every real hot reduction loop across the four
`*_math.hpp` kernel files, `consteval`-derived lane count matching this host's AVX2 width (8 floats/ymm --
confirmed via B24/B31's own host facts, no AVX-512 on Arrow Lake-HX). This WILL change summation order
(reassociation across lanes), so it is NOT expected to be bit-exact -- gate is the honest ULP/L2-diff
discipline this session has used throughout (B24/B31 precedent), not a bit-exact requirement. Files:
`include/sub0/moe_math.hpp`, `include/sub0/gdn_math.hpp`, `include/sub0/qsa_math.hpp`,
`include/sub0/gated_residual_math.hpp`. Does not touch param_store/fp8/bf16/model_file/configurator/
transplant (B33's files, unmerged) or moe_quant.hpp/decode.cpp (B31's files, already merged).

---

**2026-09-11 Claude Code — B34 done, NOT merged (real negative result, independently confirmed).**
Multi-accumulator SIMD restructuring of the engine's hot reduction loops is correctly built and
correctness-gated (parity 0, logit L2-diff 6.76e-06, vectorization independently confirmed via assembly),
but real measured decode throughput is ~20% SLOWER (agent's thermal-confound-aware A/B) -- independently
reproduced by me (own rebuild+rerun, same direction, ~3.65 vs ~3.38 s/token post-B31 baseline). First
implementation attempt hit an API rate limit mid-task; resumed in the SAME worktree by a fresh agent per
this session's established precedent, which also found and fixed a real correctness bug (a reordered dot
product inside expert_ffn_row_source broke B31's own bit-exactness invariant with expert_ffn_row -- fixed
with a second, unreordered `dot_seq` primitive). Kept unmerged on `feature/b34-simd-unlock`, matching the
B25/B33 precedent. This reinforces (alongside B28's prefetch null and B33's FP8 regression) that this
workload is DRAM-bandwidth-bound, not CPU-bound -- compute-side speedups don't help. B35 (quantized-dot-
product, reduces bytes moved not just compute) remains the more promising next lever. See
docs/INDEPENDENT_REVIEW_BACKLOG.md B34 for full detail.

---

**2026-09-11 Claude Code — B36/B37/B38 dispatched (active, in parallel).** User: "integrate the unmerged
branches to main - but make them A/B shootout compliant so they are visible improvements and cleanly
maintained." Integrating B25/B33/B34's real, correctness-clean-but-negative-result work into `main` as
proper compile-time-selectable toggles (matching this project's existing `--prec-param`/`PARAM_DTYPE`
idiom) rather than leaving them stranded on unmerged local branches -- each keeps its arm buildable,
testable, and honestly documented, defaulting to today's best-known-good behavior.

- **B36** (from `feature/b25-explicit-io-resolve`): a `--moe-io-mode reactive|pipelined` configurator flag
  (`constexpr bool MOE_IO_PIPELINED`, default `false`/reactive == today's behavior). Must RECONCILE with
  B31's already-merged `ExpertCacheSource`/`expert_ffn_row_source` fused resolve (B25's own branch predates
  B31 -- rebase onto current `main`, don't just replay the old diff). Files: `include/sub0/moe_io.hpp`
  (already exists on the branch), `include/sub0/moe_quant.hpp`, `src/backends/cpu/decode.cpp`,
  `tools/configurator.cpp` (new flag only, near `--prec-param`).
- **B37** (from `feature/b33-fp8-backbone`): merge FP8 as a real third `--prec-param fp8`/`PARAM_DTYPE::FP8`
  option, default stays BF16. Files: `include/sub0/fp8.hpp`, `include/sub0/param_store.hpp`,
  `include/sub0/model_file.hpp`, `tools/configurator.cpp` (new option value only), `tools/sub0llm-transplant.cpp`,
  `src/backends/cpu/backend.cpp`, `src/engine_core.cpp`, `tests/fp8_tests.cpp` (merged in as permanent
  regression coverage, runs regardless of the build's own PARAM_DTYPE).
- **B38** (from `feature/b34-simd-unlock`): a `--simd-reduce` configurator flag (`constexpr bool
  USE_SIMD_REDUCE`, default `false` == today's scalar behavior, bit-exact identical to current `main`).
  Prefer a SHARED, templated kernel body (parameterized on the reduction primitive) over duplicating each
  kernel twice, to avoid drift risk. Files: `include/sub0/simd_reduce.hpp`, `include/sub0/moe_math.hpp`,
  `include/sub0/gdn_math.hpp`, `include/sub0/qsa_math.hpp`, `include/sub0/gated_residual_math.hpp`,
  `tools/configurator.cpp` (new flag only).

**Shared discipline across all three**: default arm must reproduce TODAY's exact behavior (bit-exact decode
hash where applicable) -- these are additive capabilities, not replacements (AGENTS.md S4/S10). The
alternate arm must still build, pass the existing correctness suite, and reproduce (or update, if the
session owner's own host state has changed) the already-documented A/B result honestly inline near the
toggle's own definition, not just buried in the backlog. `tools/configurator.cpp` is touched by all three --
each keeps its own addition small/localized (near `--prec-param`) to minimize merge friction; the session
owner resolves any 3-way conflict at merge time.

---

**2026-09-11 Claude Code -- B38 done, on `feature/b38-simd-integration`, NOT merged.** Integrated B34's
SIMD-reduce work as a real `--simd-reduce` configurator flag (`constexpr bool USE_SIMD_REDUCE`, default
`false`). `include/sub0/simd_reduce.hpp` carries B34's `dot()`/`dot_seq()`/`sumsq()`/`sum()` unchanged plus
three new `*_choice<UseSimd>()` dispatchers -- the one seam every kernel's reduction call site goes
through via `if constexpr`, so `gdn_math.hpp`/`gated_residual_math.hpp`/`qsa_math.hpp`/`moe_math.hpp`'s
surrounding logic (loop bounds, tiling, gating) stays written once and only the innermost reduction call
differs by template arg. Every affected function (`gdn::forward`/`backward`, `gr::hc_norm`,
`qsa::rms_norm_row`/`indexer_select_row`/`attn_project_row`/`attn_row`/`forward`,
`moe::forward_row_via_run_ex` + its thin wrappers) gained a leading `bool UseSimd = false` template
parameter, threaded explicitly from `USE_SIMD_REDUCE` at every real engine call site in
`src/backends/cpu/{backend,decode}.cpp`. No file needed a duplicated-body fallback. `expert_ffn_row_source`'s
three call sites stay on `simd::dot_seq` UNCONDITIONALLY -- outside the `UseSimd` parameter entirely --
preserving B31's forward()/forward_one() bit-exactness pairing regardless of the flag, per this task's own
explicit instruction and B34's own found-and-fixed correctness constraint.

**Verification, real numbers**: `d196check`'s own historical corpora are git-LFS/untracked and absent from
this isolated worktree, so the literal `816c4a54ad49b8cf` hash could not be reproduced directly. Used two
independent routes instead: (a) `sub0_frontend_tests` (fixture-driven, needs no corpus, exercises all four
kernel files at the REAL Qwen4 axes) reproduced the exact pre-existing 120,923/245 before any new test was
added; (b) a synthetic `--vocab-exact` engine config with GDN+MoE+QSA+GR ALL live, built+run through the
real compiled `sub0_core`/`sub0_tests`: `arch_identity_tests.cpp`'s forward/decode fingerprint reproduced
identically (forward `0xdbf452a2d188e5b2`, decode `0x7510a105755db578`) across a from-scratch rebuild at
`USE_SIMD_REDUCE=false`, and `forward_one`-vs-`forward` parity held (1.63e-7, inside the existing 1e-3
tolerance). At `--simd-reduce 1` on the same axes: builds clean, parity still holds (1.60e-7), and BOTH
fingerprint hashes CHANGE as expected (forward `0xbfeb0ff9602e504`, decode `0xbc4a564ef53a97b4`) --
confirming the flag has real, live, deterministic effect through the compiled engine, not dead code. A new
permanent test (`moe_quant_tests.cpp`, "moeq (B38)") instantiates `forward_row_via_run_ex<true>` directly
against real sidecar-format bytes and requires both arms finite, close (loose tolerance, not bit-exact),
and `expert_ffn_row_source`'s own output IDENTICAL between the two runs -- proving the flag stays scoped
and never leaks into the routed-expert path. `sub0_frontend_tests` with the new case: 120,945/246, all
green. **Real-artifact throughput spot-check NOT reproduced this session**: the 48-layer BF16 artifact (+
~37 GiB MoE sidecar) B34's own ~20% number depends on was not present anywhere in this worktree or its
accessible filesystem (searched, absent) -- flagged honestly rather than assumed; B34's own already
independently-reconfirmed ~20% decode slowdown (3.09s -> 3.72s mean s/token) stands as the citable number
pending a future session with access to that artifact re-measuring post-merge. `tools/configurator.cpp`
touched ONLY for the one `--simd-reduce` option, nothing else restructured. Committed on
`feature/b38-simd-integration`, not merged -- session owner reverifies before merging, matching every prior
package this session. See `docs/INDEPENDENT_REVIEW_BACKLOG.md` B38 for full detail.
