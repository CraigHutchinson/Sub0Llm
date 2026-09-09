# Independent project review and work backlog

Review date: 2026-09-07. Source snapshot: `7a188e88e5694716839664281f5aea2b8f2781f5`.

Follow-up 2026-09-08: Intel planning now prioritizes native Sub0Llm execution and interactive inference.
See [backend design](INTEL_IGPU_BACKEND_DESIGN.md), [ISA/quantization and memory research](INTEL_IGPU_ISA_MEMORY_RESEARCH.md)
and [24 packages, I00–I23](INTEL_IGPU_WORK_PACKAGES.md). Vulkan is parked; instruction proof, usable
UMA memory and native submission measurements lead the research. CPU/CUDA source-area moves,
backend manifests and CPU API facade have landed through `220afaf`; deeper extraction and runtime
qualification remain open. No Intel backend has been implemented or measured by this research pass.
Whole-plan review adds [benchmark/optimization requirements](INTEL_IGPU_PERFORMANCE.md), six bounded
spikes and five evidence-driven review checkpoints; [findings and closures](INTEL_IGPU_PLAN_REVIEW.md).

Follow-up 2026-09-08: the user requested B14 immediately. CPU Muon scratch reuse and Gram-product
optimization are implemented and validated; see [results and limits](MUON_CPU_OPTIMIZATION.md).
The review below describes the original snapshot; B14's historical evidence is retained for context.

## Worklog

### 2026-09-08 — B11 dependency bootstrap hardening

- **Status:** Done; commit `8321c66`.
- **Base SHA:** `6c8563906bdd52428d6c8b4fb59be6fd7ead62a5`
- **Change:** `cmake/get_cpm.cmake` now validates the pinned CPM SHA-256, downloads through a
   temporary file, checks download status, and retries a corrupted cache cleanly.
- **Validation:** CMake detected and replaced the existing invalid cache, then verified the downloaded
   artifact as `C8CDC32C03816538CE22781ED72964DC864B2A34A310D3B7104812A5CA2D835D`. Configure later
   stopped at the known environment issue where `nvcc` cannot find `cl.exe`.

### 2026-09-08 — B05 tokenizer table validation

- **Status:** Done; commit `8321c66`.
- **Change:** `src/tokenizer.cpp` now rejects any tokenizer whose required 256-byte glue table is
   truncated or absent instead of restoring defaults and accepting changed semantics.
- **Test:** `tests/tok_lib_tests.cpp` serializes a real tokenizer and rejects every suffix truncation
   from 1 through 256 bytes.
- **Validation:** `src/tokenizer.cpp` passes a C++26 syntax-only compile. The focused CMake test could
   not be run because the active native configure stops when `nvcc` cannot find `cl.exe`.

### 2026-09-08 — B04 corpus parser hardening

- **Status:** Done; commit `d8c49bf`.
- **Change:** `include/sub0/tokmap.hpp` now checks arithmetic and file extents before mapping data,
   accepts only supported v2 widths/flags, validates document-start ordering and bounds, and reads
   document tables safely when their offsets are not naturally aligned.
- **Test:** `tests/tok_lib_tests.cpp` covers an unaligned valid table, overflow-sized counts, malformed
   widths/flags, invalid document starts, and a truncated legacy header.
- **Validation:** The focused test translation unit compiled cleanly, and a standalone runtime probe
   passed all malformed-input assertions. The full CMake test remains blocked by `nvcc` not finding
   `cl.exe` during the active native configure.

### 2026-09-08 — B09 build provenance refresh

- **Status:** Done; commit `3dbad05`.
- **Change:** `sub0_train` now depends on a build-time generated provenance header. It records the
   current short Git SHA and appends `-dirty` when the source tree has uncommitted changes, so a normal
   rebuild refreshes resume metadata without requiring CMake reconfiguration.
- **Validation:** The generator emitted `d8c49bf-dirty` and produced identical output on a repeated run.
   A train-stage syntax compile was unavailable because the active native build has no generated
   `sub0_config.hpp` and CMake configure is blocked by `nvcc` not finding `cl.exe`.

### 2026-09-08 — B17 document-window fallback

- **Status:** Done; commit `ff7d4fd`.
- **Change:** `include/sub0/window.hpp` now rejects empty/short sources before unsigned range
   arithmetic and returns `{0, 0}` when no selected document can provide an input/target pair.
- **Test:** `tests/tok_lib_tests.cpp` covers singleton documents, short corpora, split-truncated
   documents, and a subset with no trainable selected document.
- **Validation:** The focused test translation unit compiled cleanly, and the standalone runtime probe
   passed with `compile=0 run=0`. Full CMake tests remain blocked by `nvcc` not finding `cl.exe`.

### 2026-09-08 — B18 engine ownership contract

- **Status:** Done; commit `32931f7`.
- **Change:** `include/sub0/core.hpp` now documents the process-global model lifecycle, required call
   sequencing, returned-pointer lifetimes, model-replacement restrictions, and external thread/OpenMP
   ownership limits. No serving abstraction was added.
- **Validation:** Public-header diff review and whitespace validation completed. A full compile was not
   available because the active native build lacks generated `sub0_config.hpp`.

### 2026-09-08 — B19 exact external vocabulary configuration

- **Status:** Done; committed and independently verified 2026-09-09.
- **Change:** `tools/configurator.cpp` now accepts `--vocab-exact N` without `--corpus`, requires
   explicit core dimensions, skips corpus scanning/Unigram learning/tokenization, and emits an engine
   config with empty corpus/tokenizer artifact paths for an externally owned vocabulary.
- **Compatibility:** Existing `--corpus` invocations and learned-tokenizer behavior remain unchanged;
   `--vocab` conflicts are rejected in exact mode.
- **Validation, independently reproduced (not on trust):** built `sub0llm-configure`, exercised every
   error path (`--vocab-exact` with missing dims, `--corpus`+`--vocab-exact` together, neither given —
   all correctly rejected). Ran the real Qwen4 48-layer axes both ways into separate output
   directories: `--vocab-exact 248320` took **0.033 s** against the corpus-driven path's **~120 s**
   (matching this item's own "a fraction of that" target); `diff`ing the two runs' real generated
   `sub0_corpus.hpp`/`sub0_system.hpp` (not the small umbrella header) showed **zero differences except
   the three intentionally-empty `DEFAULT_CORPUS`/`DEFAULT_CORPUS_TOK`/`DEFAULT_TOKENIZER` paths** —
   `VOCAB=248320` and every model axis byte-identical to the corpus-driven path, confirmed by direct
   `diff`, not just a printed summary line. Fixed two small indentation inconsistencies in the new CLI
   option registrations before committing. Rebuilt `sub0_tests`/`sub0_frontend_tests` from the neutral
   small config afterward: both pass with 0 failures (`sub0_frontend_tests` exactly 120,889/244,
   matching every prior count this session — confirms the change doesn't touch the engine-independent
   path at all).

This is a repository review, not a review of Claude's current changes. The pre-existing edit to
`TOKENIZER_V2_IDEAS.md` was read for context and left untouched. No engine, test, configuration,
model, corpus, or build-tree files were changed. Findings below come from source inspection and
consumer tracing; no builds, training runs, benchmark runs, or numerical comparisons were executed.
Consequently, performance opportunities are hypotheses unless explicitly described as historical
measurements in an existing project document.

## Understanding of the project

The current direction is **supporting existing models, inference, and distillation**. The opening
status in `ROADMAP.md` explicitly closes out further from-scratch pretraining research. Training
maintenance below protects existing capabilities and artifacts; it is not a proposal to restart that
research program.

The defining architecture is a staged compiler-like workflow: the config-independent frontend and
configurator process a corpus and resolve settings; generated corpus/system headers specialize the
engine; separate train/gen/tune executables share stage implementations. `layout.hpp` supplies the
parameter layout and architecture identity. CPU execution uses shared parameter arenas, per-worker
activation/gradient arenas, and incremental decode caches; CUDA is a separately compiled device
backend behind a neutral seam. Compile-time shape specialization is fundamental, not accidental
complexity to replace with a general dynamic tensor framework.

The main strengths are the explicit format/identity rules, independent numerical fixtures and mutation
checks, backend parity tests, frontend tests that do not require an engine build, and an unusually
useful record of failed assumptions. The main weakness is that those rules are enforced unevenly at
artifact and process boundaries. Detailed local mathematical tests coexist with much thinner coverage
of actual save/resume/failure workflows. Historical commentary also sometimes contradicts current
code, making it expensive to establish what is implemented and what remains open.

The recorded WP4f result agrees exactly on embedding lookup and residual tiling, then diverges at
layer 0. Root-cause investigation, weight mapping, GDN/GR/QSA/MoE fidelity, and the llama.cpp comparison
remain Claude's work. This review draws no conclusion about which implementation causes that divergence.

## How to use this backlog

- **P1:** concrete integrity, correctness, or misleading-success problem. Address before relying on the
  affected operation. This is impact priority, not permission to interrupt the active Qwen work.
- **P2:** reproducibility, test coverage, maintainability, or a material performance investigation.
- **P3:** conditional improvement; do it only when its workload or integration is actually needed.
- **Confirmed:** the behavior is directly visible in the inspected source. It does not imply a runtime
  reproduction was performed. **Candidate:** the cost or operational consequence needs measurement.
- Effort is relative: **S** is a narrow change; **M** crosses a few consumers; **L** needs design and
  multiple configurations. These are scoping estimates, not promised durations.

### Ordered index

| ID | Priority | Work item | Evidence | Effort | Scheduling |
|---|---|---|---|---|---|
| B01 | P1 | Preserve tokenizer/corpus identity on training resume | Confirmed | M | Training maintenance; avoid overlapping stage edits |
| B02 | P1 | Publish model and recipe files without destroying the last good copy | Confirmed | M | Shared serialization; after Claude's checkpoint |
| B03 | P1 | Propagate unsuccessful final saves and decode failures | Confirmed | M | Shared stage/API work; after Claude's checkpoint |
| B04 | Done | Bound and validate corpus-file parsing | Validated by focused probe and test compile | M | Completed on user request, 2026-09-08 |
| B05 | Done | Reject truncated tokenizer semantics tables | Validated by source compile; focused test added | S–M | Completed on user request, 2026-09-08 |
| B06 | P2 | Validate model/checkpoint input before mutating live state | Confirmed | M | Shared serialization; after Claude's checkpoint |
| B07 | P2 | Test the real persistence and resume workflow | Confirmed gap | M | New harness can be designed independently |
| B08 | P2 | Make validation failures fail the command; record reproducible matrices | Confirmed gap | M | New infrastructure independent; WP4 script owned by active work |
| B09 | Done | Make build provenance match the actual binary | Validated by generator stability check | S–M | Completed on user request, 2026-09-08 |
| B10 | P2 | Reconcile onboarding, support status, and existing backlogs | Confirmed | M | Start independently; leave active tokenizer/WP4 docs alone |
| B11 | Done | Make dependency bootstrap fail reliably and verify its download | Confirmed | S | Independent CMake helper |
| B12 | P2 | Benchmark and reduce forward-only arena overhead | Confirmed cost; gain unmeasured | M | Shared CPU backend; wait for numerical baseline |
| B13 | P2 | Bound sampler stack usage and measure large-vocabulary sampling | Confirmed storage; failure unmeasured | M | After large-vocabulary inference baseline |
| B14 | Done | Remove remaining Muon allocation and optimize its matrix products | Validated; 1.9–3.5x kernel speedup | M | Completed on user request, 2026-09-08; linked report above |
| B15 | P3 | Reassess CPU affinity and gradient-reduction bottlenecks | Partly implemented; candidate gains | M | Only for a measured CPU workload |
| B16 | P2 | Make generation performance transitions and prefill measurable | Confirmed algorithm choice | L | After correctness work; benchmark first |
| B17 | Done | Keep document-window fallback inside its declared sampling contract | Validated by focused probe and test compile | M | Completed on user request, 2026-09-08 |
| B18 | Done | State the engine's process/thread ownership contract | Validated by public-header review | S initially | Completed on user request, 2026-09-08 |
| B19 | Done | Configure externally defined Qwen4 vocabularies without corpus learning | Confirmed; measured ~100 s avoidable learner cost | M | After WP5 tokenizer/configuration path is stable |
| B20 | Done | Reduce MoE decode transpose cost and use available CPU parallelism | Merged `d2bbea4`, 1.61x + exposed decode is now disk-bound | L | Completed 2026-09-09; both parts independently reverified, bit-for-bit |
| B21 | P1 | Find why `ParallelExperts`' 10 decode threads show 1-thread disk-queue depth in production | Confirmed by direct `\PhysicalDisk\Avg. Disk Queue Length` measurement, root cause open | M | Blocks trusting any further decode-throughput number as I/O-bound-and-therefore-fixed |

## Findings and acceptance criteria

### B01 — Preserve tokenizer/corpus identity on resume

`src/train_stage.cpp:1548` bundles the current build's tokenizer and corpus before the checkpoint is
loaded at line 1747. `bundle_into_model_dir` at line 580 removes an existing destination before
hardlinking/copying the new source. The checkpoint reader validates dimensions and architecture
fingerprints, but carries no tokenizer fingerprint. A different token mapping with the same vocabulary
size can therefore replace the bundle and still pass checkpoint shape checks. Even a subsequently
rejected resume may already have replaced the user's previously bundled artifacts. The schedule path
explicitly preserves existing pins on resume; the tokenizer/corpus path does not.

**Work:** determine fresh/resume mode before publishing artifacts. Reuse the existing resume bundle;
validate identity before any replacement. First examine existing model trailers and metadata as identity
sources; do not casually add checkpoint fields. Preserve deliberate data-changing workflows explicitly.

**Done when:** same-vocabulary/different-mapping and changed-build-corpus integration cases cannot
silently resume with new semantics; a rejected resume leaves the old bundle byte-identical; ordinary
resume uses the original artifacts. Include explicit output paths, not just registry-selected models.

### B02 — Publish artifacts safely

`src/engine_core.cpp:83` opens the destination model directly and writes into it. A failed or interrupted
save can truncate the existing usable model. `include/sub0/registry.hpp:292,422` likewise truncates
`config.json` and `state.json` in place, with void writers. Those files are operational inputs: the resume
path reads the optimizer from config, so they are not merely cosmetic logs. Checkpoints use a temporary
file, but `atomic_replace` at `src/train_stage.cpp:549` falls back to remove-then-rename: there is a gap
where no destination exists, and a second rename failure leaves it absent. `save_model` also returns
stream state before an explicit flush/close check.

**Work:** reuse one checked publication policy for these writers, preserving the byte formats. On
replacement failure retain the last good destination and report the recoverable temporary file. Define
whether the guarantee covers process interruption or also power loss; a rename alone is not a stated
power-loss durability protocol. Account for paired model/sidecar publication where applicable without
changing Claude's in-progress importer.

**Done when:** open/write/flush/close/replace failures are exercised; readers see either the old complete
artifact or the new complete artifact; JSON publication failures are observable to callers; current
checkpoint/model compatibility remains unchanged.

### B03 — Return failure when the requested result was not produced

At `src/train_stage.cpp:2908–2932`, failure of the final model save, final checkpoint save, and rescue
checkpoint save only produces log messages. Execution still prunes checkpoints, writes a terminal
`trained`/`plateaued` status, and returns zero. Separately, `kv_decode_generate` in
`include/sub0/decode.hpp:112` returns void and handles device failures by returning early;
`src/gen_stage.cpp:292–300` then prints the partial context and returns zero. An automation cannot
distinguish a device failure from a valid short completion by the exit status.

**Work:** propagate a small explicit completion/failure result through actual consumers. Preserve
partial generated text, but distinguish EOS, requested length, and device failure. For training, record
which final artifact actually saved and avoid advertising unsaved progress as safely completed.

**Done when:** injected GPU seam failures and exhausted save retries produce a nonzero command status
and useful recovery information; EOS remains success; retention does not discard recovery material
on an unsuccessful final publication. Enumerate all decode consumers, including evaluation/tuning.

### B04 — Harden the corpus parser before exposing views

`include/sub0/tokmap.hpp:174` multiplies and adds untrusted u64 counts without overflow checks.
A concrete arithmetic counterexample is a 32-byte v2 header with `ntok=2^63`, 16-bit tokens, and
`ndoc=0`: `ntok*2` wraps to zero, so the file-size check accepts a huge token count with no payload.
Document indices are copied without checking that the first is zero, that indices are ordered, or
that they lie within the token extent. `sample_window` uses `upper_bound(...)-1`, so those invariants
matter to memory safety. The document table is read through a u64 pointer even when the packed-token
extent does not align its start to eight bytes. The legacy document header also reads its fourth u32
before checking that the file has the required 16 bytes.

**Work:** validate each byte extent against remaining file bytes before arithmetic/conversion; require
supported widths/flags and representable counts; decode potentially unaligned records safely; enforce
document-index invariants before returning `ok()`. Audit token-ID validation at the ingest boundary too.

**Done when:** tiny malformed fixtures cover wraparound, truncated legacy headers, unsupported widths,
misaligned but valid tables, and invalid document indices. Current valid files remain readable. Run
the parser under available memory/undefined-behavior instrumentation in an isolated test build.

### B05 — Treat a missing tokenizer semantics table as corruption

`src/tokenizer.cpp:974–987` acknowledges that current-version files always contain the 256-byte glue
table, but replaces a short or absent table with defaults, clears stream errors, and reports success.
For a tokenizer with corpus-calibrated defaults, this substitutes different encoding/decoding semantics.
It is not legacy compatibility: the version gate already rejects older schemes. The same parser resizes
containers from file-supplied counts before validating a consistent vocabulary/base relationship or
bounding the allocation by the input size.

**Work:** require the complete table for accepted schemes and validate count relationships and input
extents before allocation. Keep any intentionally permissive inspection/recovery behavior separate
from loading a tokenizer for training or inference.

**Done when:** removing each possible suffix of the table fails loading without changing the output
tokenizer; a non-default calibrated table round-trips identically; malformed counts fail promptly;
existing round-trip and fingerprint tests stay green. No scheme bump is needed merely to reject
malformed files that the current writer never emits.

### B06 — Validate before committing loaded state

`src/engine_core.cpp:143–159` overwrites parameters and uploads them before validating tokenizer and
architecture trailers. A later mismatch returns false after live state has changed. Checkpoint loading
at `src/train_stage.cpp:751–798` similarly modifies run state, RNG, and parameter/moment arrays as it
reads; its comment promises that malformed input leaves the fresh model untouched. Current training
callers do abort on a failed load, which limits the immediate consequence, but the API contract and
behavior disagree. Checkpoint strings/history counts also lack file-size-derived allocation bounds,
and RNG parsing success is not checked separately.

**Work:** validate metadata, extents, and trailers first; publish identity and device state only after
successful loading. Choose between transactional loading and an explicitly unusable state on failure.
Do not blindly double resident model memory with a full-size staging copy: inspect bounded preflight,
mapping, or recovery/reinitialization options against real model size.

**Done when:** truncated payloads, bad trailers, malformed RNG, and implausible lengths have deterministic
failure behavior; callers cannot mistake rejected weights for a valid model; the documented state
guarantee is tested; peak RAM is recorded for the chosen design.

### B07 — Cover the actual save/resume workflow

`tests/CMakeLists.txt` links engine tests to core/frontend, not the train stage. Existing CUDA tests
explicitly simulate checkpointing by downloading and uploading memory, while `run_config_tests.cpp`
explicitly says resume reconciliation itself is not covered. Those are useful tests, but they do not
exercise the ordering defects in B01–B03.

**Work:** add a small process-level harness around real stage tools, or extract a narrowly scoped
persistence unit with real production callers. Cover run creation, save, restart, compatibility refusal,
optimizer reconciliation, metadata publication, and checkpoint retention. Avoid a wholesale train-stage
rewrite just to make functions testable.

**Done when:** a deterministic CPU continuation matches an uninterrupted run under the same recipe;
failure cases leave original artifacts intact; all files live in isolated temporary model directories;
no test touches a user's build-generated corpus or live model registry.

### B08 — Turn test policy into executable, trustworthy gates

`scripts/wp4b_check.sh:20` runs the test executable with `|| true`, then prints selected output. It can
finish successfully despite test failures. It also assumes particular pre-existing build directories
and a machine-specific corpus path. No tracked `.github` workflow was found. The configured presets
provide debug/native test entries but no sanitizer matrix. Fixture suites such as
`tests/gdn_qwen4_fixture_tests.cpp:67` use WARN-and-return when required reference data is absent.
That is acceptable for an optional developer run, but insufficient for a mandatory correctness gate.

**Work:** preserve useful diagnostic collection while aggregating failures into the final exit code.
Create an isolated, reproducible runner for frontend, default engine, and selected realistic feature
configurations. Record config/derived dimensions, exact assertion and case counts, skipped gates, build
identity, and available instrumentation. Make fixtures mandatory in the validation job that claims them.

**Done when:** intentionally failing a case or removing a required fixture makes the gate fail; a clean
checkout can reproduce the selected matrix; default/feature-on runs are unfiltered as AGENTS.md requires.
Do not execute or edit the active WP4 script while Claude owns that workflow.

### B09 — Make provenance describe the binary being run

`CMakeLists.txt:374` captures `SUB0_GIT_SHA` only during CMake configure. The normal development workflow
can rebuild changed source after a commit without rerunning CMake, retaining the old SHA. Dirty source
is not represented. This value is not just display text: `src/train_stage.cpp:1466` uses SHA equality
when deciding whether an unfinished run can resume automatically.

**Work:** generate small, build-refreshed provenance metadata with explicit clean/dirty and generated
configuration identity. Keep it out of heavyweight engine recompilation where possible. Distinguish
source provenance from architectural compatibility rather than treating a commit label as proof of both.

**Done when:** commit-only, source-edit, and generated-config changes produce truthful metadata;
an incremental no-op build stays cheap; offline/source-archive builds report an explicit unknown state.

### B10 — Reconcile the documentation against current code

Concrete drift includes README's C++23/CMake >=3.20 prerequisites versus CMake's C++26/minimum 3.28;
the preset file separately advertises 3.21. The README's static/BSS/no-heap description is too broad:
the CPU backend now deliberately allocates shared arenas and lazy Workers on the heap to avoid the
Windows image-size limit. `CPU_PERF_BACKLOG.md` says P-core affinity is absent, but
`ensure_thread_built` calls `pin_current_thread_p_first`. AGENTS.md's historical Muon example no longer
describes all current allocations. `TOKENIZER_V2_IDEAS.md` describes older/future work while the current
scheme constant is 5. That file is actively edited and must not be "cleaned up" under this review.

**Work:** give readers one current entry point covering supported host/toolchain, configure/build/run,
architecture-by-backend support, and current versus experimental scope. Keep incident history, but mark
superseded findings closed or partially closed. Replace private `[[memory-note]]` dependencies with
repo-readable explanations when they are necessary to perform a task. Verify Windows environment setup
instructions in a fresh shell rather than trusting an inherited developer session.

**Done when:** a new contributor can identify the supported build, current tokenizer scheme, and working
inference scope without reconciling historical paragraphs; every carried-forward performance ticket has
a present-day source location and status. Do not expand this into a Linux port: explicitly decide and
document support first, given unconditional Windows includes in current production sources.

### B11 — Verify the dependency bootstrap

`cmake/get_cpm.cmake` downloads CPM when a path does not exist, without an expected hash or checking
the download status. An interrupted/failed attempt can leave a file that suppresses the next download;
the later include fails confusingly. This script is executable build input.

**Work:** download to a temporary file, verify a pinned digest and successful status, then publish it.
Retain a documented offline/cache path. Keep dependency updates separate from this robustness fix.

**Done when:** interruption, HTTP failure, and incorrect bytes produce a clear diagnostic and a clean
retry; a valid cached download works without network access.

### B12 — Measure and reduce forward-only arena overhead

`src/backend_cpu.cpp:311` clears both value and gradient spans for every allocation. The Worker at
line 287 still contains a full activation-gradient arena even in forward-only builds, although parameter
gradient storage already has a forward-only specialization. This confirms the existing CPU backlog
item and suggests a memory saving as well as a zero-fill saving. Some forward operations rely on cleared
state, so removing all initialization is not a safe mechanical optimization.

**Work:** count allocated/cleared bytes by operation and separate scratch lifetime from backward need.
Start with proven-unread gradient storage in the existing compile-time forward-only mode; consider
ordinary inference separately. Preserve value initialization wherever recurrence/accumulation needs it.

**Done when:** default hashes/assertion counts and feature fixtures remain unchanged; realistic tiny and
larger configurations report wall time, peak committed memory, and bytes initialized. Reprofile after
the change. Wait for Claude's stable comparison baseline before modifying this shared backend.

### B13 — Bound sampler stack usage; benchmark vocabulary scaling

**Update 2026-09-08: the overflow this item declined to claim has since been directly observed.**
WP5c's generation harness (`tools/sub0llm-qwen4-gen.cpp`) is the first real-VOCAB consumer to actually
call `sample_token`, at the real model's `VOCAB=248,320` — measured stack usage there is 2.84 MiB in
one frame (two `std::array<float, VOCAB>` plus one `std::array<int, VOCAB>`), and relinking without a
workaround reproduces `STATUS_STACK_OVERFLOW` (`0xC00000FD`) on the very first sampling call, against
Windows' 1 MiB default. Currently worked around at that one tool's own link line (`/STACK:33554432`)
rather than in `engine_core.cpp` itself — see `docs/WP4_SCOPE.md` §6 WP5c. This item's own proposed fix
(reused, bounded `thread_local` scratch) is the correct one; it just wasn't done yet when this was
written.

`src/engine_core.cpp:343` places full-vocabulary float logits, int indices, and float keep arrays on
the stack in the top-k path: approximately `12*VOCAB` bytes at source level, before other call frames.
That is about 3 MB at a 250k vocabulary. Actual stack reservation/failure depends on the generated
code and linker stack setting and has not been measured here. It also computes exponentials across
the entire vocabulary after top-k masking, so small k still pays a full-vocabulary softmax pass.

**Work:** inspect actual stack usage and use reused, bounded scratch appropriate to this DLL's TLS
constraints. Benchmark top-k selection plus probability sampling independently of the forward pass.
If restricting softmax to selected entries, preserve or explicitly account for seeded sampling order
and numerical behavior rather than silently changing reproducibility.

**Done when:** small and imported-scale vocabularies run under the documented stack budget with no
per-token allocation; seeded behavior and distribution checks cover top-k boundaries and invalid logits;
report latency separately from model forward time. Do not claim an observed stack overflow yet.

### B14 — Finish Muon scratch reuse, only if the workload warrants it

**Completed 2026-09-08 on user request.** See [the implementation and validation report](MUON_CPU_OPTIMIZATION.md).
The following is the original finding, not remaining work.

`include/sub0/muon.hpp:54` now grows thread-local X/A/AA/BX buffers and reuses them. The historical
"all buffers allocate every call" diagnosis is stale. However, `src/backend_cpu.cpp:3061` still creates
`std::vector<float> upd(n)` per matrix per optimizer step. The A@A multiply in `muon.hpp` also reads one
operand with a strided inner loop; whether it dominates after allocation removal requires profiling.

**Work:** prepare per-worker update and Newton–Schulz scratch for the maximum eligible matrix before
the optimizer loop, respecting dynamic OpenMP assignment. Then profile matrix products before choosing
blocking, SIMD, or a library. Retain the smaller-dimension orientation optimization unless measurement
justifies an alternative; the existing transpose TODO is not evidence that transposition is wasteful.

**Done when:** allocation instrumentation shows zero hot-loop allocations, including the first
optimizer step; buffer-reuse and numerical-property tests pass; a real workload reports time and memory
at two scales. Existing training research remains closed; prioritize this only for a real consumer.

### B15 — Re-measure affinity and reduction rather than reimplementing closed work

P-core-first pinning already exists in `include/sub0/cpu_affinity.hpp` and is called at
`src/backend_cpu.cpp:2732`. It ranks logical processors by efficiency class; it does not independently
prioritize distinct physical cores ahead of SMT siblings within a tier. The gradient reduction at
line 3041 still reads every worker's complete gradient array. Neither observation proves the best
thread-placement or reduction strategy on the current workload.

**Work:** compare current affinity against OS placement and a physical-core-aware order, measuring
the actual OpenMP teams used by forward, evaluation, and optimizer work. Profile reduction bandwidth
and SIMD efficiency. A tree reduction does not automatically eliminate the need to read the gradients
and may add passes, so do not assume the old backlog's proposed fix wins.

**Done when:** measurements include exact topology, threads, batch, model, warmup, and repeated samples;
new scheduling preserves numerical tolerances; the existing backlog is updated with implemented versus
still-hypothetical work. Do not run competing CPU benchmarks alongside Claude's performance runs.

### B16 — Characterize generation's performance cliff and prefill

`src/gen_stage.cpp:279` chooses incremental decode only if prompt length plus requested generation
fits the window. Otherwise the whole request uses full-forward generation from the outset, even while
the initial tokens would still fit. The warning is helpful but the abrupt cost transition remains.
`include/sub0/decode.hpp:142,164` also prefills by repeatedly calling the single-token path. These are
real implementation choices; the gain from alternatives has not been measured here.

**Work:** benchmark time to first token, prefill tokens/sec, decode tokens/sec, and peak RAM across the
window boundary. First consider retaining the existing incremental path until the boundary and switching
to the existing full-forward implementation there. Separately investigate batched/chunked prefill. A
sliding/ring KV cache is a larger algorithmic change because RoPE position, sinks, recurrent state, and
interceptors must retain their meanings.

**Done when:** outputs are compared against the current path around exact-boundary and over-boundary
cases; performance reports separate prefill from decode; any cache change is gated at the necessary
feature configurations after external fidelity work completes. No opportunistic Qwen cache rewrite.

### B17 — Make document-window fallback honor document/subset boundaries

`include/sub0/window.hpp:74` promises a window within a single document. At fraction=1, exhausting
eight rejection attempts falls back to an unrestricted flat window, which can cross document boundaries.
With a subset active and no trainable selected document, the scan instead returns `{0,1}`, which need
not belong to the selected subset. Repeated single-token documents and very small selected subsets
are useful triggers. These are contract violations even if uncommon in current prose corpora.

**Work:** establish trainable-document availability at source initialization, then make exhaustion return
a valid selected document or an explicit unusable-source result. Preserve the declared weighting policy;
do not replace token-weighted sampling with uniform document sampling without making that change clear.

**Done when:** adversarial corpora containing singleton documents, no selected trainable documents,
and split-straddling documents never produce an out-of-contract window; zero/short extents are handled
before unsigned subtraction; ordinary seeded behavior remains unchanged where no fallback occurs.

### B18 — Document process and thread ownership before adding consumers

`include/sub0/core.hpp` exposes process-global build/load/forward operations, with some cache pointers
documented as thread-local. `ensure_thread_built` maps storage using an OpenMP thread number. Two
unrelated host threads outside an OpenMP team are not thereby assigned distinct Worker slots, and
multiple models do not have separate ownership handles. This is consistent with the current single-model
CLI architecture, but it is not an adequate contract for a future embedded or concurrent caller.

**Work:** state supported calling order, returned-pointer lifetime, single-model ownership, and external
concurrency restrictions at the public API. Add assertions for unsupported calls only where justified.
Introduce contexts or session ownership only when an actual production consumer needs them.

**Done when:** a reader can determine whether concurrent generation, model replacement, and nested
OpenMP calls are supported without inspecting backend internals. No generic serving framework is added.

### B19 — Configuring a real-Qwen4-axes build should not require learning a vocabulary from a corpus

`tools/configurator.cpp` treats `--corpus <path>` as required and always runs the full unigram BPE
learner (`unigram::learn`, this project's own from-scratch scheme) over it before it will emit
`sub0_config.hpp` — including for a Qwen4-axes build, where the learned result is never actually used.
The runtime tokenizer for those builds is WP5a's own `sub0::qwen_tok::Tokenizer`
(`include/sub0/qwen_tokenizer.hpp`), which loads the real model's `vocab.json`/`merges.txt` directly;
the configurator's own learned `tokenizer.tok` is written to `${GEN_DIR}` and then never consulted by
`sub0llm-qwen4-gen`/`sub0llm-qwen4-forward`. Its only real job in a Qwen4 build is landing the
compile-time `VOCAB` constant on 248,320 — and even that is indirect: `docs/WP4_SCOPE.md` §6 WP4d's own
`--vocab 248202` line is a hand-derived correction (`248320 - 288 + S`, where `S` is this specific
corpus's own distinct-single-byte count) chosen to make the *learned* count land on the *real* target,
which is fragile (tied to whichever corpus is passed) and slow: **102.7–106.2 s per run**, measured
repeatedly this session (WP5c's own independent verification build, and a fresh VTune profiling build),
each time just to reconfirm the same already-known `VOCAB=248,320`.

**Work:** give the configurator a path that does not require a corpus or the unigram learner at all
for an externally-defined vocabulary — e.g. a `--vocab-exact N` mode (or reading the axes from a file,
mirroring how `tests/qwen4_real_axes/sub0_config.hpp` already hand-writes these constants for the
engine-free transplant/shape-test tools, extended to the CONFIGURE step so linked-engine tools get the
same shortcut). Existing from-scratch-corpus builds, which genuinely need the learned tokenizer, must be
unaffected — this is additive, not a replacement for the learner.

**Done when:** a Qwen4-axes engine build (`sub0_core`-linking targets, not just the engine-free ones)
can be configured without `--corpus` and without running `unigram::learn`, producing a `sub0_config.hpp`
identical to today's corpus-driven path at the same axes; the real measured configure time drops from
~100 s to a fraction of that; every existing non-Qwen configure invocation is unchanged.

### B20 — MoE resolve's own transpose dominates decode, and decode never uses more than one core

**Real, measured, not inferred.** VTune hotspots against a live `sub0llm-qwen4-gen` decode run (real
48-layer weights, `-collect hotspots -target-pid <pid> -duration 60`, 2026-09-08) found two independent
findings in the same result, both concrete enough to act on directly:

1. **`sub0::transplant::transpose_out_in` (`include/sub0/transplant.hpp:370`) is 52.5% of ALL sampled
   CPU time — 28.9s of 55.0s — more than the three format-specific dequantizers it calls
   (`dequantize_iq2_xxs`/`dequantize_iq1_s`/`dequantize_iq4_nl`, 22.7% combined) put together.** The
   call chain, from VTune's own top-down report: `moe_resolve` → `ExpertCache::resolve` →
   `moeq::dequantize_expert` (`include/sub0/moe_quant.hpp:154`) → `transpose_out_in`, which then calls
   the per-format decoders as its own children. `transpose_out_in`'s body is a plain nested loop —
   `dst[i*out_f+o] = src[o*in_f+i]` — contiguous read, but every write strides by `out_f` floats; at
   `out_f`/`in_f` on the order of 640/2560 the destination plane is 6.55 MiB, far past any per-core
   cache, so the write side thrashes on every element. Its own header comment ("called once per
   destination tensor from a loop") describes the offline transplant tool's original one-shot use —
   `moe_quant.hpp` reuses the exact same function, unmodified, once per expert per layer per token
   (480 calls/token at these axes), the hottest path this engine has.
2. **`Total Thread Count: 1` for the entire 60s window** (`Elapsed Time: 60.017s` ≈ `CPU Time: 55.025s`)
   on a 24-core host. Confirmed architecturally, not just observed: `src/backends/cpu/backend.cpp` has
   exactly two `#pragma omp parallel` sites in the whole file (`train_batch`'s window loop and
   `AdamW::step`'s per-matrix loop) — nothing parallelizes `forward_one`/decode at all. This was a
   reasonable design at this project's original from-scratch-model scale (multi-threading the BATCH
   dimension, not a single token's own work); it was never revisited for a single real-scale decode
   step, where the 10 experts a layer resolves are independent work that a single core processes
   serially.

Both findings sit inside `include/sub0/moe_quant.hpp`'s `ExpertCache::resolve`/`dequantize_expert` path
and `include/sub0/transplant.hpp`, both shared code also used by the offline transplant tool and by
WP4e/WP4f's own correctness gates — any change here must keep those bit-for-bit, not just the live tool.

**Work:** (a) give `transpose_out_in` (and its siblings in `transplant.hpp`, e.g. `per_head_half_transpose`)
a cache-blocked/tiled implementation instead of the naive nested loop — a pure locality fix, no
algorithmic or numerical change, so the existing bit-for-bit gates (`tests/transplant_tests.cpp`,
WP4e's `--verify`) should need no new tolerance; (b) parallelize the per-expert resolve work across the
`EXPERTS_PER_TOK` selected experts within one layer's decode step (they are independent by construction
— no shared mutable state beyond `ExpertCache`'s own resolve pool, which would need a read path safe
under concurrent resolves, or a per-thread pool). Re-profile after each change independently rather than
landing both at once, so the report can say which one bought what.

**Done when:** a repeat of this same VTune session shows `transpose_out_in` no longer dominant and
`Total Thread Count` above 1 during decode; the measured seconds/token for the real 48-layer model
(currently ~9.65–10.4s, `docs/WP4_SCOPE.md` §6 WP5c) is reported before and after each change; every
existing transplant/MoE-quant correctness test still passes bit-for-bit. Coordinate through
`docs/ACTIVE_WORK_LOG.md` before touching `moe_quant.hpp`/`transplant.hpp` — both are hot, shared files.

**Status: Done; merged `d2bbea4` (2026-09-09), independently reverified before merging.** Both parts
landed, each re-profiled with VTune independently as required above. Part (a): `transpose_out_in` and
its siblings share one measured (tile=16, both loop orders swept, not guessed) cache-blocked core —
52.9% → 21.5% of sampled CPU time, 9.14 → 5.68 s/token (1.61x). Part (b): resolve now runs a layer's
selected experts across threads (`Total Thread Count` 1 → 10), via a two-phase compute-then-combine
restructuring of `moe_math.hpp`'s `forward_row_via` that keeps the weighted-sum accumulation order
identical for both the serial and parallel paths — but bought only ~2% net, because it exposed that
decode is now DISK-bound: ~13,610 hard page faults/sec, ~56 MB/s matching physical disk reads, no
warm-up trend over 30 tokens, because the 37.11 GiB S0Q1 mapping doesn't stay resident alongside the
18.31 GiB f32 backbone in 63.4 GiB of RAM. **That demand-paging behavior is the real next item**, not
a CPU optimization — reported, not fixed, here. `include/sub0/moe_quant.hpp` was not touched by
either fix. Bit-for-bit verified independently (not on trust) after merging: `--verify` against the
real 48-layer artifact 0 mismatches of 1074, `forward`/`forward_one` parity 0 exactly, the WP5c
determinism fixture byte-for-byte, and the neutral suites plus the transplant/moequant-tagged tests
specifically all green. One real defect found and fixed along the way: decode's new worker threads
never called `set_flush_denormals` (FTZ/DAZ is per-thread MXCSR state) — a silently false bit-exactness
claim waiting to happen the first time an intermediate went subnormal on a thread that missed it. Full
writeup: `docs/ACTIVE_WORK_LOG.md`'s B20 row.

### B21 — `ParallelExperts`' 10 decode threads are not concurrently in flight on the disk in production

Follow-up to B20, filed 2026-09-09 during Sub0MemPage research. B20's own row above already flagged decode
as disk-bound; this item is the next layer down — **whether B20's own fix (fanning MoE resolve across
`MOE_DECODE_THREADS` = 10 OpenMP threads) is actually buying real concurrent disk I/O in the live engine.**

**Measured, not inferred.** Two independent lines of evidence, both from this session:

1. An isolated harness running the engine's own real resolve code path (`moeq::ExpertCache::resolve` →
   `gguf::to_f32` → `transplant::transpose_out_in` → `moe::expert_ffn_row`), one thread per cold expert,
   one private `ExpertCache<1,...>` per thread — a byte-for-byte structural match to `decode.cpp`'s
   `ParallelExperts` — scales 4.2–5.2x from 1 to 10 threads (11.05 ms/expert → 2.13 ms/expert), and its
   **1-thread** projection (5.31–5.67 s/token) matches the live engine's actual measured throughput
   (5.53–5.85 s/token) almost exactly, while its **10-thread** projection (1.02–1.27 s/token) does not.
2. Directly measuring `\PhysicalDisk(1 D:)\Avg. Disk Queue Length` during a real `sub0llm-qwen4-gen`
   decode run (current `main`, includes B20): **0.04–0.09 throughout steady-state decode** — the
   harness's own single-thread figure (0.12–0.20), nowhere near its ten-thread figure (0.98–1.21).

Both point the same way: **the live engine's ten `ParallelExperts` threads are not concurrently issuing
disk reads**, even though B20's `--verify`/parity/determinism-fixture gates (correctly) show the fix is
bit-for-bit correct — this is a performance defect, not a correctness one.

**What was ruled out by code review**: no lock, no shared mutable state, and no structural serialization
in the actual call site (`decode.cpp`'s `ParallelExperts`, `moe_quant.hpp`'s `Store`/`ExpertCache`). Each
of the (confirmed, from the real generated config) 10 `MOE_DECODE_THREADS` gets its own heap-allocated
`MoeDecodeThread` (private single-slot `ExpertCache` + FFN accumulators), reads through one shared
read-only `FileMap`, and nothing between a thread's `resolve()` and `dequantize_expert()`'s raw byte
access can block on another thread — the `#pragma omp parallel num_threads(...)` / `#pragma omp for
schedule(static)` structure is architecturally identical to the harness shape that DID scale.

**What was ruled out by direct measurement** (same research session): mmap section-object serialization on
this file (10-thread mmap-only fault throughput scales 5.1–5.6x in isolation); NVMe saturation (device
sustains 6.34 GB/s vs. the engine's measured 55–108 MB/s); the 48-region fork/join barrier tail cost
(measured 1.2–1.3x, not 4–5x); page-cache pressure under a 25 GiB resident-ballast simulation of the
engine's own real private footprint (measured 1.3x, not 4–5x).

**Still open, not yet tested**: P-core/E-core thread-affinity placement of the real OpenMP team vs. the
harness's `std::thread` pool on this 8P+16E part; whether per-region OpenMP wake-up cost differs
materially once the team is reused across 48 short regions inside a much larger, much busier process
already holding 25+ GiB resident (TLB/cache contention the isolated harness's smaller footprint would not
reproduce); whether something upstream of the resolve call itself (the router's own work, or the two-phase
compute-then-combine restructuring) serializes the ten resolves in practice despite no lock being visible.

**Acceptance criteria**: instrument or re-measure to identify the actual serialization point (VTune's
thread-concurrency histogram over the resolve region, not its hotspot list, is the next instrument named
by this research); fix it; re-verify decode throughput moves toward the harness's own 10-thread projection
(~1.0–1.3 s/token) while `--verify`/parity/the determinism fixture stay exactly unchanged (this is once
again a pure scheduling question — the two-phase compute-then-sum structure means answer correctness
cannot depend on which thread computes which expert or how quickly).

Full research trail: session scratchpad `sub0mempage-research-empirical-concurrency.md` (not part of this
repo) — every number above is reproducible from the commands logged in that file's appendix.

## Suggested execution order

1. **With WP5a/b active:** prepare B07/B08 harness design and Intel I00/I01/I18–I20 research in
   isolated files. Coordinate benchmarks and large-memory probes through the active log. Copilot owns
   continuing I08/I09 extraction. WP4f's converter correction is merged and available as an input.
2. **Persistence and truthful validation:** make B02/B03/B06 concrete with failure-injection tests;
   close B01 before trusting training resume. B09 provenance implementation has landed; B08's broader
   validation contract remains. Schedule outstanding full-suite checks for B04/B05/B09/B11/B17 once
   the recorded build-environment blocker is resolved; their focused evidence is in the worklog above.
3. **Performance after frozen baselines:** measure B12/B13/B16 and implement demonstrated wins.
   Keep QSA GEMV profiling linked to `CPU_PERF_BACKLOG.md`, coordinating WP5's current consumers.
4. **Later training:** B15 and Muon's host settings/scratch separation need a real training consumer.
   B14 is implemented and validated; B17 and B18's original hardening/documentation scope has landed.
   B18 documentation does not complete the future Intel session implementation.

Do not turn the historical spike list into a blanket deletion campaign. AGENTS.md requires consumer
enumeration and gate round trips; `spellspike`, `scratchspike`, and `wordspike` have production includes
today. Likewise, do not substitute higher assertion counts for boundary coverage, quote historical
speedups as current measurements, or declare full Qwen support from synthetic/reference-component
tests; WP4f prefix validation does not establish full-model inference.
