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
