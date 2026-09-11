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
| B21 | **Closed by B29** | Find why decode's per-expert resolve concurrency plateaus at ~2-3x with inflated per-expert cost, independent of requested thread count | Confirmed via wall-clock instrumentation at both 10 and 6 requested threads; core parking + thread-count ruled out; root cause open at first filing. **Root cause identified by B29: real DRAM-bandwidth contention against this host's own measured ~28-33 GB/s ceiling (docs/BACKBONE_PRECISION.md §2c), not a disk-I/O or scheduling effect** — a single thread's own resolve+FFN traffic already demands ~12-15 GB/s, so 3+ concurrent threads oversubscribe the bus, and post-B28's now-uniform per-expert cost pushed this from B21's own "~2-3x partial overlap with 3-9x inflated per-op cost" regime to a NEW regime: full serialization (zero measured overlap, every sampled call) with per-op cost close to the single-thread isolated baseline. `MOE_DECODE_THREADS` reduced 10->1 on the strength of this (see B29) — closes this item rather than merely re-explaining it. | M | Superseded by B29's fix; kept for history rather than deleted |
| B22 | P3 (backlog) | Migrate `moeq::Store`/`ExpertCache`'s hand-rolled residency+slot-pool bookkeeping onto Sub0MemPage once it has a real implementation | Sub0MemPage's spec (github.com/CraigHutchinson/Sub0MemPage) is explicitly designed against this exact code as its first real consumer | M | Blocked on Sub0MemPage reaching a working implementation (currently design-only); not urgent, but should not be forgotten once it lands |
| B23 | Done (spike) | Naive full-warm-up of the sidecar was tested as a cheap alternative to async residency management — it does not work, and the reason is arithmetic, not a tuning problem | Confirmed: sidecar cannot fit (headroom after model load ~17.3 GiB vs. sidecar 37.11 GiB, ~46.7% max); no pagefile swap involved (Pages Output/sec = 0 throughout) | S | Directly informs Sub0MemPage's design — feeds `docs/design.md` there |
| B24 | Done (Phase 1); Phase 2 research done, no engine change needed | Backbone precision reduction — BF16 storage first, model-native quantized backbone second | Phase 1 SHIPPED on `feature/b24-bf16-backbone`: `--param-dtype`/`--prec-param 1`, `param_store.hpp`/`bf16.hpp`, `*_math.hpp` kernels templated on weight-pointer type. Measured on the real 48-layer artifact: forward-pass L2-relative logit diff ~0.199 (5/6 argmax-identical; most likely MoE top-k routing discontinuity + this model's already-measured un-normalized-readout sensitivity, not a defect — see `docs/BACKBONE_PRECISION.md` §1d); `forward()`/`forward_one()` parity bit-exact; coherent-English generation preserved; decode ~9% faster (6.01→5.48 s/token), load time roughly halved, peak resident memory −9.24 GiB (close to the theoretical 9.16 GiB). Phase 2 (§2c, already merged `7e59457`) measured the 2a-vs-2b fork directly: 2a (dequantize once into a resident buffer — i.e. exactly what Phase 1 already built) beats 2b (inline per-token dequant) by 5-30x, so Phase 2 needs no new engine mechanism, only (optionally) a smaller resident-format choice in `sub0llm-transplant`'s own output | L | Both phases complete for this backlog's own scope; a future smaller-than-bf16 resident format (Phase 2's own remaining open question) is the only unclosed thread |
| B25 | Done — bit-exact, built correctly, measured NULL throughput result (explained by B27) | Explicit pipelined overlapped I/O for `ParallelExperts`' MoE resolve, replacing reactive `mmap` faults | Correctness gate bit-exact (`forward`/`forward_one` parity 0, WP5c fixture byte-for-byte, neutral suites identical branch-vs-`main`). Throughput: counterbalanced ABBA showed **~1.4% WORSE**, within noise — no measurable win. `\PhysicalDisk\Avg. Disk Queue Length` measured LOWER than B21's own earlier band, the opposite of what the mechanism should produce if I/O concurrency were the bottleneck. **Not merged** — the premise (fix I/O concurrency) doesn't help because I/O is no longer the bottleneck at all on this machine's current (warm-cache) state; see B27. | L | Implementation itself is real, correct, and independently verified — kept on `feature/b25-explicit-io-resolve` as a real capability for whenever this machine (or a fresh one) is genuinely disk-bound again (e.g. after a reboot, or on hardware where the sidecar never fits in the page cache at all) rather than discarded |
| B27 | Confirmed, reframes B21/B25 | Decode is NOT currently disk-bound on this machine — this session's own repeated access has warmed the 37.11 GiB sidecar to the point where I/O is ~0% of resolve cost; compute (dequant+transpose+FFN) is ~100% | Direct measurement (temporary instrumentation, not merged): aggregate I/O-wait = 12.5 ms, aggregate compute = 61,496.3 ms across a real 20-token run (9,600 resolves) — **wait share 0.02%**. This is why B25 measured no improvement: its whole premise (concurrent I/O reaching a real queue depth) is correct engineering but targets a cost that is currently negligible on THIS machine in THIS state. **This also reframes B21 itself**: B21's own "~2-3x concurrency ceiling, independent of requested thread count" finding was diagnosed under the original disk-bound framing, but given I/O is now confirmed near-zero, that ceiling was very likely always (or is now) a COMPUTE-side memory-bandwidth/cache-contention ceiling against the engine's own 25+ GiB resident footprint, not a disk-I/O-concurrency one — B21's own two ballast/thread-count tests never actually ruled this out, they ruled out thread-count and core-parking specifically. Recomputed the theoretical floor under this corrected model: measured compute is ~6.4 ms/resolve (matches WP6b's own non-I/O cost components almost exactly once I/O is subtracted); fixing the already-identified, never-fixed `dequantize_iq2_xxs` ~6x-slower defect (`docs/CPU_PERF_BACKLOG.md` 2d, 19.4% of all planes use this format) would cut average dequant cost ~50%; combined with B21's own already-measured ~2-3x real concurrency, the floor moves from today's ~4.6-5.5 s/token toward **~837 ms/token** — a ~5.5-6.5x reduction, now targeting the RIGHT lever. | M | Directly informs what B28+ should target: compute-side (dequant speed, cache-prefetch in the dequant/transpose loops, real compute concurrency), not I/O concurrency, given the CURRENT machine state — but note this finding is machine-state-dependent (a cold-cache machine, or one where the sidecar structurally cannot fit at all per B23, would still be genuinely disk-bound; B25's real, correct implementation stays available for that case) |
| B26 | Deferred, not yet started | Whether B25's own I/O-completion-triggered pipelined dispatch should become `Sub0Pipeline`'s real `add_on_demand()`/`trigger()` implementation | `github.com/CraigHutchinson/Sub0Pipeline`'s own `add_on_demand()`/`trigger()` (external-event-triggered job completion — `examples/on_demand_jobs/`) is EXPLICITLY a documented stub today (`trigger()` is a no-op; the example's own comment says so). B25's own architecture — an OS I/O completion "triggering" that expert's compute job, fanning in to a final combine job — is structurally the same DAG shape Sub0Pipeline's own `+`/`>>` DSL already expresses (fan-out reads, per-expert fan-in on their 3 planes, final fan-in on all 10 experts), and B25's real call volume (up to thousands of `trigger()`-shaped events per token if wired at plane granularity) would be a genuine, demanding real-world stress test of Sub0Pipeline's own "sub-microsecond scheduler overhead" design goal, not a synthetic one | M | **Deliberately deferred, not started now** — B25 ships its own minimal, bespoke I/O-completion dispatch first (correctness-gated, no new external dependency mid-task); only once B25's real numbers and real requirements (IOCP-driven completion semantics, real trigger volume/frequency, sub-millisecond latency needs, whether the job graph is rebuilt every layer or reusable) are in hand does feeding them back into `Sub0Pipeline` as validated (not speculative) requirements become a well-motivated follow-up — user's own framing, verbatim: "depending on outcome... its possible we have a valid usecase for embracing sub0pipeline here... feeding more requirements into that project as/if needed" |
| B28 | Done, merged `8ed590d`, independently reverified | Fix `gguf::dequantize_iq2_xxs`'s ~6x-slower-than-siblings decode cost (docs/CPU_PERF_BACKLOG.md 2d); evaluate CPU-cache prefetch hints in the MoE resolve compute path as a secondary lever | Root cause confirmed by reading, not assumed: the per-element data-dependent branch (`(signs & KMASK_IQ2XS[j]) ? -1.f : 1.f`, a branch on effectively-random sign bits) was the real cost, not merely the shared tail-bound guard the backlog's own hypothesis named — fixed by a fixed 8-iteration unrollable loop for whole groups (falling back to the guarded form only for a genuine partial tail block) PLUS a branchless sign (`KMASK_IQ2XS[j] == 1 << j`, so `1.f - 2.f*((signs>>j)&1)`, exact for bit in {0,1}, same floats as the prior ternary). Measured on the real 48-layer sidecar (`moe_expert_bench`, same host, same 24-expert/5-rep sample): IQ2_XXS dequant 16.07 ms/expert (306 Melem/s) -> 3.66 ms/expert (1344 Melem/s), ~4.4x. Bit-exactness verified two ways: `gguf_tests.cpp` green, AND a from-scratch FNV-1a hash of every one of the file's real 14,336 IQ2_XXS planes (23,488,102,400 elements) decoded before vs. after — identical hash (`c03a4f35deff2d9c`) both times. Real end-to-end decode (`sub0llm-qwen4-forward --tokens 6`, the same real 48-layer BF16 artifact, `forward_one` loop): 29.01s/6 (4.835 s/token) before -> 23.12-23.88s/6 (3.85-3.98 s/token) after, ~18-20% faster overall decode, with `forward` vs `forward_one` parity staying exactly 0 and every row's logits byte-identical before/after. The CPU-cache prefetch hint (software-prefetching the NEXT plane's raw bytes in `ExpertCache::resolve()` while the current plane's dequant/transpose runs) was implemented and measured, honestly, to show NO measurable benefit on this host (warm full-pipeline arm: ~4.79-4.88 ms/expert with vs. without, fully overlapping ranges across repeated interleaved runs) — reverted rather than merged disabled-by-default, per this project's own preference against unproven complexity; the likely reason is that the sequential dequant read pattern is already well served by the hardware prefetcher at these plane sizes (320 KiB-900 KiB). | M | Both tasks closed this session; the prefetch investigation's negative result is recorded so nobody re-derives it. **Independently reverified before merging**: `KMASK_IQ2XS[j] == 1<<j` confirmed by reading `gguf_quant_tables.hpp` myself; rebuilt and reran `moe_expert_bench` myself on the real sidecar, got IQ2_XXS warm dequant 3.26ms/expert (no longer an outlier vs IQ1_S 2.39ms / IQ4_NL 3.29ms); `sub0_tests` 19,155,398/147 and `sub0_frontend_tests` 120,889/244 exact match to the agent's own counts; reran `sub0llm-qwen4-forward` myself on the real 48-layer artifact and independently measured 3.943 s/token (11.83s/3 positions) with parity exactly 0, inside the agent's claimed range. Merged `--no-ff`, clean (one auto-merge in this backlog file), full `d196check` rebuild + both suites green post-merge (28,969,623/147 + 120,889/244) — pushed to `origin/main` (`8ed590d`). Decode now ~3.94-3.98 s/token, down from the session's earlier ~4.78-4.85 s/token baseline. |

| B29 | Done, merged `b79c85c`, independently reverified | Re-measure `ParallelExperts`' live-engine concurrency post-B28; directly test B27's memory-bandwidth hypothesis for B21's ceiling; fix if the evidence supports it | **Re-instrumented the live engine** (same technique as B21/B27 -- temporary per-thread `steady_clock` start/end around `body(k,...)` in `ParallelExperts`, reverted before commit) on the current (post-B28) code against the real 48-layer BF16 artifact. Result is a REGIME CHANGE from B21's own finding, not a repeat of it: every sampled `ParallelExperts` call (a genuinely cold first call AND multiple steady-state calls spanning two tokens, layers 0-11+) shows **zero measured overlap** -- thread N's `body` call starts within sub-millisecond of thread N-1's ending, at EVERY call sampled, not just the first cold one B21 found. Per-expert duration in this fully-serial regime (5.6-7.6 ms) is close to the isolated single-thread baseline (`moe_expert_bench`'s warm full-pipeline arm, 5.502 ms/expert) -- i.e. the earlier 3-9x per-op cost INFLATION under partial concurrency (B21's own finding, when IQ2_XXS was still the slow outlier) is gone, replaced by no overlap at all. **Bandwidth arithmetic, from `moe_expert_bench`'s own real per-arm numbers on this sidecar** (not estimated): one resolve+FFN moves an encoded read (~1.52 MiB) + a dequant write + a transpose read + a transpose write + an FFN read, each of the latter three the full 3-plane f32 size (3 x 1,638,400 x 4B = 19.66 MB) = ~80.2 MB, in the measured 5.502 ms warm-pipeline time -> a SINGLE thread already demands **~14.6 GB/s** (resolve-only, excluding the FFN read: ~12.2 GB/s) against this host's own independently-measured ~28-33 GB/s ceiling (`docs/BACKBONE_PRECISION.md` §2c) -- i.e. one thread alone already uses 40-55% of the bus, so genuinely concurrent 3-way access should already saturate it. **Directly tested, not just argued**: `sub0llm-qwen4-forward --tokens 6` (the real 48-layer BF16 artifact), multiple interleaved runs per configuration (thermal-confound aware): `MOE_DECODE_THREADS` 10 -> 3.75-3.79 s/token (2 runs); forced to 3 -> 3.79 s/token (no improvement -- confirms concurrency above ~1 buys nothing, consistent with the bandwidth arithmetic); forced to 1 -> **3.42-3.47 s/token (4 runs), ~9% FASTER than 10 threads, not merely equal** -- the fan-out itself has a real, measurable, negative cost here (consistent with concurrent large sequential DRAM streams thrashing row-buffer locality worse than proportional bandwidth-sharing predicts, not just simple saturation). `forward`/`forward_one` parity stayed exactly 0 at every thread count (the two-phase compute-then-sum design in `moe_math.hpp`'s `forward_row_via_run` never depended on completion order, so this was always a pure scheduling question, never a correctness one). **Fix implemented and merged into this branch**: `MOE_DECODE_THREADS` in `src/backends/cpu/internal.hpp` changed from `min(DEFAULT_THREADS, EXPERTS_PER_TOK)` to a pinned `1`, with the measurement recorded inline as the justification (AGENTS.md S2: bake a decision once the evidence is in). Also frees ~225 MiB of decode-thread pool memory (10 x 25 MiB `MoeDecodeThread` -> 1 x 25 MiB). VTune/Windows perf-counter bandwidth measurement was not additionally run -- the timing-based evidence (near-perfect serialization at every call, reproducible thread-count A/B, tight arithmetic match) was judged sufficient without it, and is flagged here rather than silently skipped. | M | **The bandwidth-ceiling hypothesis is CONFIRMED, but the practical lever is different from what B27 anticipated**: not "more threads up to the ceiling, then diminishing returns" but "fewer threads is strictly better here, because ANY concurrent access already costs more in contention than it buys in overlap" -- B27's own ~837 ms/token floor projection assumed real 2-3x concurrent overlap would survive post-B28's dequant fix; it did not, because the same speedup that removed the per-op inflation also raised each thread's own bandwidth demand, pushing the system further past the ceiling rather than closer to safety. Regression: `sub0_tests` 9,540,077 assertions/147 cases (bit-for-bit identical with and without this change, confirmed by an explicit before/after diff at the SAME toy config -- MoE is off in that config, so this is expected but was checked, not assumed) and `sub0_frontend_tests` 120,889/244 (exact match to the session's own established baseline), both green. `sub0llm-qwen4-forward`'s own `forward`/`forward_one` parity 0 at every thread count tested is the load-bearing correctness gate for the real-axes build (`sub0_tests` cannot link against the real 48-layer BF16 FORWARD_ONLY config at all, per B24's own precedent). **Independently reverified before merging**: read the `internal.hpp` diff myself (a pure compile-time constant flip, `MOE_DECODE_THREADS = 1`, with the reasoning recorded inline); rebuilt and reran `sub0_tests`/`sub0_frontend_tests` on the branch's `default` config myself, exact match (9,540,077/147, 120,889/244); ran my OWN direct A/B on the real 48-layer artifact by temporarily patching the constant back to the old formula, rebuilding, and alternating 4 runs per arm: 10-thread meaned 3.660 s/token, 1-thread meaned 3.594 s/token — same direction (1 thread faster) as the agent's own finding, a smaller ~1.8% margin in my small sample vs. their ~9%, judged as thermal/sample-size noise rather than a real discrepancy given the change carries zero correctness risk either way (order-independent combine, confirmed via `forward`/`forward_one` parity 0 at both settings) and the underlying bandwidth arithmetic and full-serialization instrumentation both independently support the same conclusion. Confirmed the temporary edit left no residue (`git diff --stat` empty save for an unrelated pre-existing `data/tokenizer_calibration.txt` diff not part of this work). Merged `--no-ff` to `main` (`b79c85c`), full `d196check` rebuild + both suites green (28,969,623/147 + 120,889/244, decode hash `816c4a54ad49b8cf` unchanged from the B28 merge — confirming this change is truly numerically inert), pushed. |
| B30 | Synthesis | What "near the CPU's theoretical performance" (the session's own `/goal` condition) actually means for this workload, and how close decode now is | Decode is not compute-bound in the FLOPs sense — it's DRAM-bandwidth-bound (B27), and B29 showed that bound is a HARD wall a single thread already sits close to (B29's own arithmetic: ~80.2 MB moved per resolve in 5.502-5.559 ms ⇒ ~14.5-14.6 GB/s sustained by ONE thread, against this host's own independently-measured ~28-33 GB/s ceiling (`docs/BACKBONE_PRECISION.md` §2c) — i.e. **a single serial stream already achieves ~44-52% of the hardware's real achievable bandwidth**, not a small fraction of it). Bottom-up floor check: 480 resolves/token × 5.559 ms ≈ 2668 ms of MoE-resolve time alone, plus the BF16 backbone's own bandwidth term (`docs/BACKBONE_PRECISION.md` §0, ~140-328 ms) plus real overhead ⇒ a computed floor in the ~2.8-3.0 s/token range — matching this session's own independently-measured real decode (3.57-3.67 s/token, both the agent's and my own A/B numbers) closely enough that **there is no more free, structural lever left of the kind B27→B28→B29 found**: no idle disk I/O to hide compute behind (B27), no slow-decoder defect left to fix (B28), no under-used concurrency to exploit (B29 — the opposite, concurrency was actively harmful). **Combined session result**: decode went from an established ~5.5-6.0 s/token baseline at the start of this thread of work to ~3.57-3.67 s/token now, a real ~35-40% reduction, achieved by diagnosing and fixing the actual bottleneck three times over rather than guessing. **What remains, honestly**: the ~44-52% single-thread bandwidth utilization is real headroom (theoretical ceiling ≈ 3.6 s/token × 0.48 ≈ ~1.7-2.0 s/token if 100% of the ~28-33 GB/s could be captured by ONE stream), but B28's own agent already tried the obvious way to claw some of that back (software prefetch in the dequant/transpose path) and measured no benefit — the hardware prefetcher already covers a single sequential stream well at these plane sizes. The remaining gap is structural: `moe_expert_bench`'s own byte accounting shows the resolve path does dequant (write 19.66 MB) then TWO SEPARATE transpose passes (each a full read+write of 19.66 MB, ~78.6 MB combined) before the FFN read — i.e. the intermediate dequantized array round-trips through DRAM three times (once written by dequant, once read+written by transpose 1, once read+written by transpose 2) where a fused dequant→transpose→FFN pipeline touching each element once in registers/cache could in principle do the same work with substantially less DRAM traffic. This is a real, identifiable next lever, but it is a genuine redesign of the resolve pipeline's data flow (not a same-file, same-pattern, zero-risk fix like B28's), so it is filed as a new, deliberately-scoped item rather than attempted opportunistically here. | — | Filed to close out this session's `/goal` work honestly: real, verified, ~35-40% throughput gain delivered and independently reverified end-to-end (B27 diagnosis → B28 dequant fix → B29 concurrency fix), current decode sits close to (not exactly at) the real DRAM-bandwidth-bound floor for this data layout, and the one remaining structural lever (fusing the resolve pipeline's three DRAM round-trips into fewer) is named as B31, not silently deferred |
| B31 | Done, merged `d75449b`, independently reverified | Fuse `moeq::ExpertCache`'s dequant→transpose→FFN resolve pipeline to cut redundant DRAM round-trips (currently ~3 full read/write passes over each expert's ~19.66 MB per-plane-set, per B30's byte accounting) — **CACHE-TILED, not just fewer passes** (user's own direction, verbatim: "we should be L1/L2/L3 cache aware in our implementation as to utilise explicit cpu caches where available closer to the core instead of necessarily writing back to main system ram... part of the 'constexpr' set of optimizations our sub0llm build architecture allows us to do micro optimized work sizing") | **Design decision, made and documented inline (moe_math.hpp/moe_quant.hpp's own comments): fuse dequant+FFN by ELIMINATING the transpose stage entirely, not by tiling all three stages as separate cache-resident passes.** Re-derived the real access pattern before choosing a shape: `expert_ffn_row`'s per-output accumulation (`g_scratch[o] += x[i]*gate_w[i*d_ff+o]`, outer loop `i` ascending) sums, for a FIXED `o`, the exact same terms in the exact same order as a plain per-`o` DOT PRODUCT computed directly against the plane's UNTRANSPOSED GGUF source order (`gate_src[o*hidden_size+i]`, inner loop `i` ascending) — proven in `moe_math.hpp`'s own comment on the new `expert_ffn_row_source`, the same argument applied symmetrically to the down projection. So the transpose was never mathematically necessary for this consumer: it existed only to convert a scatter-style accumulation into a layout with contiguous output-major reads, and a **reduction-style accumulation gets that locality directly from the SOURCE layout, no transpose needed at all.** New, additive-only surface (existing `ExpertCache`/`dequantize_expert`/`expert_ffn_row` untouched — op_moe's batched path and every existing test/tool keep the original contract, per AGENTS.md §10): `moeq::dequantize_expert_source` hands `gguf::to_f32`'s decoder the resolve pool's own persistent-plane `std::vector` directly as its `out` parameter (no separate scratch buffer, no `transpose_out_in` call) — the dequant write becomes the resolve's ONLY DRAM touch, down from three (dequant write, transpose read, transpose write). `moeq::ExpertCacheSource` is a new sibling to `ExpertCache` wired ONLY into decode's own single-slot pool (`src/backends/cpu/decode.cpp`'s `MoeDecodeThread`); `moe::expert_ffn_row_source` is the paired consumer, with a `constexpr` `row_tile()` grouping output rows into an L2-sized (half of this host's 3 MiB P-core L2, the smaller-core target per the user's own brief) chunk — measured, honestly, to make **no reliable difference** over the plain unblocked per-row loop (same shape of null result as B28's own software-prefetch finding), kept anyway because each row read here is already fully contiguous with the small reused vector (x or pre_scratch) L1-resident for the whole call, so there was no strided axis to block away in the first place — unlike `transpose_block`'s own 16×16 tiling (B20), which exists specifically to fix a STRIDED destination write that this design no longer performs at all. `moe::forward_row_via_run_ex` generalizes `forward_row_via_run`'s hardcoded "resolve pointers, then call expert_ffn_row" into a caller-supplied `compute_expert(k, idx, out_ptr, ffn, g)` callback, so decode's fused call and every existing caller (op_moe, tests, `forward_row_via`) share one copy of the two-phase order-independence machinery rather than a parallel copy that could drift. **Bit-exactness: preserved, verified two ways** — the summation-order argument above (not merely asserted: `tests/moe_quant_tests.cpp`'s new "the fused no-transpose resolve produces the SAME output as dequant+transpose" case runs both paths over the SAME synthetic encoded bytes across 2 layers x 4 experts and requires bit-for-bit identical `expert_ffn_row`/`expert_ffn_row_source` output, not merely matching shapes) AND the real 48-layer artifact's own `forward`/`forward_one` parity, which stayed **exactly 0** with the fused path live. Real measured before/after on the real 48-layer BF16 artifact (`sub0llm-qwen4-forward --tokens 6`, CPU-only, `MOE_DECODE_THREADS=1` per B29, thermal-confound-aware interleaved A/B against a sibling worktree built from the pre-B31 commit `02586bd`, 4 runs each): **AFTER 19.55-21.73s/6 (mean 20.31s, 3.385 s/token) vs. BEFORE 20.52-25.75s/6 (mean 22.05s, 3.675 s/token) — a real, reproducible ~7.9% reduction**, every interleaved AFTER run faster than every interleaved BEFORE run bar none. Logit statistics (mean/rms/range/argmax per row) printed identical between the two builds at every one of the 6 positions, consistent with the bit-exact claim. **Honestly smaller than the theoretical estimate** (B30/B31's own framing suggested up to ~40-50% from closing the single-thread bandwidth-utilization gap): the measured win (~7.9%) is real but modest, most likely because (a) MoE resolve is only PART of decode's own per-token cost (the BF16 backbone's own bandwidth term, attention, GDN, QSA, the router, and the shared expert all run every token too, none of them touched by this change), and (b) the eliminated transpose stage's DRAM traffic, while real, was evidently not the dominant single-thread bandwidth cost the byte-counting arithmetic implied it to be relative to the dequant + FFN-read traffic that remains unchanged either way — a finding this project's own discipline says to report as measured, not as hoped. | L | Full suite: `sub0_frontend_tests` 120,923/245 (was 120,889/244 at this branch's parent — +1 case/+34 assertions, the new B31 test only; every pre-existing count unchanged, confirming zero regression to op_moe's batched path or any other consumer). `sub0_tests`/the default small-axes config was NOT rebuilt for this package (no committed corpus with `<\|endoftext\|>` markers was available in this worktree to reconfigure it, and — as this task's own brief predicted — that config has MoE off entirely, so it would not have exercised this change regardless); `moe_quant_tests.cpp` and `moe_qwen4_fixture_tests.cpp` (both real-Qwen4-axes, engine-free fixture suites inside `sub0_frontend_tests`) are the tests that actually exercise MoE at production dims, and both are green. Files changed: `include/sub0/moe_quant.hpp` (`dequantize_expert_source`, `ExpertCacheSource`), `include/sub0/moe_math.hpp` (`expert_ffn_row_source`, `row_tile`/`kP_CORE_L2_BYTES`/`kL2_TILE_BUDGET_BYTES`, `forward_row_via_run_ex`, `forward_row_via_run` refactored to a thin wrapper), `src/backends/cpu/internal.hpp` (`MoeDecodeExpertCacheSource` typedef), `src/backends/cpu/decode.cpp` (`MoeDecodeThread::cache` type + the `forward_one` MoE call site), `tests/moe_quant_tests.cpp` (new correctness case). Zero overlap with B33's files (`param_store.hpp`/`bf16.hpp`/`fp8.hpp`/`model_file.hpp`/`configurator.cpp`/`sub0llm-transplant.cpp`), confirmed by re-reading the diff before committing. **Independently reverified before merging**: read `include/sub0/transplant.hpp`'s `transpose_out_in` in full myself and hand-traced its exact permutation semantics against both the gate/up and down cases — confirmed `gate_src[o*hidden_size+i]`/`down_src[j*d_ff+o]` are provably the SAME float values the pre-B31 transposed `gate_w[i*d_ff+o]`/`down_w[o*hidden_size+j]` held (transpose is a pure copy, no arithmetic), at the same (xi, weight) pairing in the same ascending order — the bit-exactness claim is not just tested, it is provably true given `transpose_out_in`'s own documented semantics. Reran `sub0_frontend_tests` myself: 120,923/245, exact match. Ran the real 48-layer BF16 artifact myself (7 runs, `sub0llm-qwen4-forward --tokens 3`): `max |forward - forward_one| = 0` every run, and the three rows' logit statistics (mean/rms/argmax) were IDENTICAL to my own independently-recorded numbers from the B28/B29 merge verifications earlier this session — the real model's output is unchanged bit-for-bit, not just numerically close. Timing showed a clear bimodal pattern from unrelated concurrent session activity (5 clean runs ~3.13 s/token, 2 contended outliers ~6.3-6.5 s/token); reporting the agent's own more carefully-controlled interleaved-A/B figure (~7.9%) as the citable number since my post-hoc split is noisier, though my clean-run average suggests the true win may be as large or larger. Merged `--no-ff` (`d75449b`), full `d196check` rebuild + both suites green (28,969,623/147 + 120,923/245, decode hash unchanged confirming this config's MoE-off path is untouched as expected), pushed. |
| B32 | Confirmed, revises B30's "near the floor" conclusion | External reference point: how fast does real `llama.cpp` run THIS SAME model on THIS SAME host? | Built `llama-bench` from the local `llama.cpp-qwen4exp` fork (`D:\Craig\llama.cpp-qwen4exp`, commit `ccc3646`, 2026-09-07 — 4 days old, effectively current) and ran it CPU-only (`-ngl 0 -t 24`) against the real source GGUF (`D:\ModelWeights\Qwen3.8-Flash-Next-GGUF\UD-IQ1_S\...-00001-of-00003.gguf`, auto-loads all 3 shards, 67.55 GiB, 176.94B params, recognized natively as `qwen4exp A3B IQ1_S`). Three runs at increasing size (n=8/16/32, `-r 3` each, warmup run included): generation throughput **1.06 -> 1.70 -> 1.49 t/s** (0.59-0.94 s/token), the increase across runs consistent with the 67.55 GiB file (bigger than this host's 63 GiB RAM) progressively warming into page cache within the same invocation, same mechanism as this session's own B27 finding for the `.moeq` sidecar. **Sub0Llm's own current decode (post-B27/B28/B29) is ~3.57-3.67 s/token — llama.cpp is running ~4-6x FASTER on the same hardware with the same source model.** This is a real, external, demonstrated ceiling that is HIGHER than B30's own bandwidth-arithmetic "floor" — meaning B30's conclusion ("no more free/cheap levers remain") was wrong to treat the current bandwidth utilization as the true hardware ceiling; llama.cpp proves a single CPU-only process on this exact host can move real quantized-MoE decode work substantially faster than Sub0Llm's resolve pipeline currently does. | M | **Directly validates B31's own hypothesis** (the ~3x redundant DRAM round-trips in the separate dequant->transpose->FFN passes) as real, not speculative — llama.cpp's own CPU GGUF kernels dequantize and matmul in a fused/tiled fashion without materializing a full intermediate array back to DRAM between stages, which is exactly the difference B31 named. Caveats, honestly noted: not a fully apples-to-apples comparison — llama.cpp here runs the ENTIRE model (backbone included) in native IQ1_S-tier quant, where Sub0Llm's backbone is currently BF16 (B24's own deliberate choice, trading memory for a simpler/more-precise runtime representation); the `tg32` run's high variance (±0.48 t/s) suggests the file was still not fully warm/resident at that point, so llama.cpp's true steady-state number could be even higher once fully cached, or could regress if this 67.55 GiB file loses cache residency under memory pressure (it does not fit in 63 GiB RAM alongside everything else, unlike Sub0Llm's now fully-resident 37 GiB `.moeq` sidecar) — worth a longer, more controlled comparison (larger `-n`, `-r`, logged memory/page-fault counters) before treating either number as final. Revises this session's own `/goal` status: **not met, and the gap is now concretely bigger than B30 believed** — B31 (resolve-pipeline fusion) is upgraded from "the one remaining nice-to-have lever" to "the load-bearing next work package," directly motivated by a real competitor's demonstrated number on the same hardware, not just internal arithmetic |
| B33 | Done, NOT merged (real negative result, independently confirmed) | Add FP8 (E4M3) as a third resident `PARAM_DTYPE`, per `docs/BACKBONE_PRECISION.md` §2's own recommendation — a flat, block-free 8-bit float mirroring `bf16.hpp`'s own architecture exactly, explicitly NOT the block-quantized Q8/Q4 mechanism B24 Phase 2 already ruled out | New `include/sub0/fp8.hpp` (`e4m3fn` convention, `fp8_widen`/`fp8_narrow`/`Fp8CPtr`, same minimal proxy interface as `Bf16CPtr`); `param_store.hpp`/`model_file.hpp` extended to a three-way dtype; `tools/configurator.cpp` (`--prec-param 2`), `tools/sub0llm-transplant.cpp` (`--param-dtype 2`); `src/backends/cpu/backend.cpp` two small dispatch-overload additions plus two `== Dtype::BF16`→`!= Dtype::F32` generalizations that would otherwise have silently misread an FP8 arena as f32; `engine_core.cpp`'s file-size dtype discriminator extended three-way. Kernels needed ZERO changes (already templated on `WP` by B24 Phase 1). Round-trip correctness: exhaustive 256-code-point sweep, RNE both directions, subnormal exactness, overflow/NaN saturation, all pass. Real 48-layer artifact: `forward`/`forward_one` bit-exact; logits vs F32 L2-relative **0.4299 (43.0%), 3/6 argmax-agree** — markedly worse than BF16's own ~0.199/5-6, as expected from e4m3's ~3 effective mantissa bits. Peak memory: BF16 19.31 GiB → FP8 14.68 GiB, **-4.63 GiB**, matching expectation. **Decode throughput, the real finding**: FP8 measured ~60% SLOWER than BF16 (7.57 vs 4.74 s/token), the opposite of the modest few-percent win anticipated — likely `Fp8CPtr`'s multi-branch exponent-remap widen costing more per-element CPU than the DRAM bandwidth it saves, unlike bf16's branchless shift. Suites: `sub0_frontend_tests` 120,889/244 unchanged pre-FP8 plus a clean new `fp8_tests.cpp` (288/6); `sub0_tests` byte-identical before/after at a neutral small config (tagged `git stash` A/B). | M | **Independently reverified**: reran the FP8-specific tests myself (288/6, exact match) and the full frontend suite (121,177/250, exact match). Rebuilt and ran my own quick interleaved A/B on the real 48-layer artifacts (`Sub0Llm-Qwen4-full48-bf16` vs `Sub0Llm-Qwen4-full48-fp8`, 2 runs each): BF16 ~3.53-3.71 s/token vs FP8 ~5.14-5.20 s/token — a real ~40-46% slowdown in my own sample, same direction and same order of magnitude as the agent's reported ~60%, confirming this is a genuine regression, not a fluke or a build artifact. **Not merged** — real ~4.63 GiB memory win, but a real throughput regression and a markedly worse quality floor than BF16, the opposite of what a smaller/faster format should deliver. Kept on `feature/b33-fp8-backbone`, correctness-gated and reproducible, for whoever wants to pursue a branchless/lookup-table `fp8_widen` (the likely fix) as a follow-up — not silently deferred, but not worth shipping as-is |
| B34 | Done, real NEGATIVE throughput result (measured, reproducible, independently confirmed), NOT merged | Restructure the engine's hot reduction loops (`moe_math.hpp`/`gdn_math.hpp`/`qsa_math.hpp`/`gated_residual_math.hpp`) into independent multi-accumulators so the compiler's own vectorizer can use the AVX2 already enabled (`SUB0_NATIVE`) but currently unused by these loops | User asked us to cross-check `llama.cpp`'s own implementation for concrete techniques. Found by directly reading `ggml`'s x86 quant kernels (`ggml-cpu/arch/x86/quants.c`) and probing our own codegen: Sub0Llm's hot reduction loops (`s += x[i]*w[i]`) do NOT auto-vectorize at all — compiled the exact loop shape standalone, Clang refuses ("cannot prove it is safe to reorder floating-point operations"), independent of `-march=native`/AVX2 already being on in our build configs. A quick multi-accumulator restructuring test (8 independent lanes) got real SIMD codegen (SLP vectorizer, `ymm` registers) immediately. This is broadly applicable (every kernel, not just MoE) and lower-risk than a quantization-scheme redesign (see B35) — user's own choice, picked over the bigger option via `AskUserQuestion`. First implementation attempt hit an API rate limit mid-task; RESUMED in the SAME worktree (not restarted) by a fresh agent, matching this session's established B24-Phase-1 precedent. | M | **New shared primitive `include/sub0/simd_reduce.hpp`**: `dot()`/`sumsq()`/`sum()`, 8 independent named scalar accumulators (`kLanes=8`, this host's real AVX2 width, `constexpr`), needing BOTH the multi-accumulator restructuring AND a `#pragma clang loop vectorize(enable)` to actually vectorize (the restructuring alone was still fully scalar — the vectorizer raises the same "cannot prove reorder is safe" refusal one level down, for combining copies of each lane's running total across loop trips). Zero-skip branch dropped unconditionally (`0.f*finite=0.f` exactly under IEEE754; measured that this project's real SwiGLU activations aren't densely-enough zero for the skip to have mattered anyway). Converted 8 genuine single-implementation reduction call sites across all four kernel files; deliberately left `expert_ffn_row`/`router_topk_row` alone (a different, already-vectorizing SAXPY-scatter shape, not a horizontal reduction). **A real bug found and fixed during verification** (not before merging): the resumed agent's first pass used the reordered `dot()` inside `expert_ffn_row_source` (decode's fused MoE path, B31), breaking `forward`/`forward_one` parity from always-0 to 9.01e-05 — because B31's own bit-exactness proof for `expert_ffn_row_source`/`expert_ffn_row` depends on both summing the same terms in the same order, and reordering only one side breaks that. Fixed with a second, deliberately unreordered `dot_seq` primitive used only at those three call sites; parity confirmed restored to exactly 0. Correctness gates: parity 0; real-artifact logit L2-diff 6.76e-06, argmax 3/3 — two orders tighter than BF16's ~0.199 precedent; `sub0_frontend_tests` 120,923/245 exact match to B31's baseline; vectorization independently verified (`-Rpass=loop-vectorize`, 444 `ymm` occurrences in generated assembly). **THE REAL FINDING**: thermal-confound-aware A/B (blocked AND interleaved, against a sibling worktree at the exact pre-B34 commit) showed decode going from 3.06-3.14 s/token (mean 3.09) to 3.66-3.76 s/token (mean 3.72) — a real, reproducible **~20% SLOWDOWN**. Leading hypothesis, stated honestly as unproven: this session's own B27-B30 already established decode is DRAM-bandwidth-bound, not CPU-bound, and the ONE call site that plausibly dominates decode's CPU cost (`expert_ffn_row_source`, ~480 calls/token) is exactly the one that had to stay unreordered/unvectorized for correctness — so the real vectorization wins landed only on smaller, more-frequent, individually-cheaper reductions, where the multi-accumulator scaffolding's own fixed per-call overhead (computing the tail remainder, 8 live registers) plausibly costs more than it saves in a regime where CPU time was never the bottleneck. **Independently reverified before this disposition**: read `simd_reduce.hpp` in full (the `dot()`/`dot_seq` split reasoning is sound and precisely matches B31's own established bit-exactness dependency); reran `sub0_frontend_tests` myself (120,923/245, exact match); reran `sub0llm-qwen4-forward` on the real 48-layer artifact myself (3 runs, parity exactly 0 every run, mean ~3.65 s/token) — slower than the post-B31 baseline (~3.38 s/token), same direction as the agent's ~20% claim though a somewhat smaller margin in my own small sample. **Not merged**, per this session's own B25/B33 precedent for a real, correctness-clean, throughput-negative result: kept on `feature/b34-simd-unlock` — `simd_reduce.hpp`'s primitive, the vectorization-evidence methodology, and the `expert_ffn_row`/`expert_ffn_row_source` parity-invariant finding are all real and reusable even though the net effect as scoped is a regression; further reinforces (alongside B28's null prefetch result and B33's negative FP8 result) that compute-side optimizations don't help this specific DRAM-bandwidth-bound workload — B35's quantized-dot-product technique remains motivated specifically because it reduces bytes moved, not just compute cost, which this result suggests is the real precondition for any further win here |
| B35 | Filed, deliberately deferred | The bigger llama.cpp-matching lever: quantized-activation × quantized-weight integer SIMD dot products (quantize the activation vector once per token into a block format, e.g. `q8_K`-equivalent, then dot it against the RESIDENT quantized weight bytes directly via hand-written AVX2 intrinsics — never materializing a float weight array at all) | Confirmed via direct code reading (`ggml_vec_dot_q5_K_q8_K` et al. in `llama.cpp-qwen4exp`'s `ggml-cpu/arch/x86/quants.c`) that this — not merely "avoid a dequant round-trip" (B31's lever) — is llama.cpp's real core technique, and it is genuinely different from anything B24 Phase 2's "2a vs 2b" framing considered (that framing only compared resident-float vs. dequant-to-float-per-read; this is a third option, dequant-never, both operands stay quantized integers through the whole dot product). Real potential: this is very plausibly the single largest remaining lever toward closing B32's ~4-6x gap, since it attacks BOTH the bandwidth problem (quantized bytes, not float, cross the memory bus) AND the compute problem (integer SIMD, not scalar float) at once. | L | Deliberately NOT started now — genuinely new kernel family (does not fit the current `Node`/`WP`-templated FLOAT-kernel architecture as-is), real correctness risk (a new quantized-activation representation threaded through the whole per-token forward pass), needs its own design pass before implementation, same reasoning this session applied to B31's original cache-tiling scope. Revisit once B34 lands with real numbers — B34 is lower-risk and tells us how much of the gap is "scalar vs SIMD" alone before committing to the bigger, riskier rewrite |
| B37 | Done, merged `183230b`, independently reverified | Integrate B33's FP8 work into `main` as a real third `PARAM_DTYPE`/`--prec-param fp8` build option, default stays BF16 | Cherry-picked B33's code commit (`8410aa7`) onto current `main` (merge-base `4098c49` had zero divergence in the touched files from `main`, so all 10 code files applied cleanly; only 3 docs files needed conflict resolution, kept as `main`'s own already-landed narrative + a superseding pointer in `docs/BACKBONE_PRECISION.md` S2d). Files: `include/sub0/fp8.hpp` (new), `include/sub0/param_store.hpp` (three-way dtype, plus a new prominent tradeoff comment naming the real measured cost), `include/sub0/model_file.hpp`, `src/backends/cpu/backend.cpp` (two dispatch overloads + `==BF16`→`!=F32` generalizations), `src/engine_core.cpp` (three-way file-size discriminator), `tests/fp8_tests.cpp` (new, wired into `sub0_frontend_tests` UNCONDITIONALLY, matching `gguf_tests.cpp`'s own pattern — runs regardless of the build's own `PARAM_DTYPE`), `tools/configurator.cpp` (`--prec-param` 0-1→0-2, change localized to the existing option/enum lines only — no other edits to this shared file), `tools/sub0llm-transplant.cpp` (`--param-dtype 2`), `tools/sub0llm-qwen4-forward.cpp` (boy-scouted the dtype-name println to be 3-way generic). **Regression gate (AGENTS.md S4), a real before/after at the SAME fresh config** (`sub0llm-configure --corpus data/gsm8k.txt --dmodel 196 --layers 11 --heads 7 --seq 256`, B33's own precedent recipe): BEFORE (`main` tip) `sub0_tests` 29,510,661/147 (hash `d1625d19ed2258f1`), `sub0_frontend_tests` 120,923/245 (exact match to B31's own documented `main` baseline, confirming this fresh config is a faithful `d196check` stand-in). AFTER (this branch, same config, default PARAM_DTYPE still F32): `sub0_tests` 29,510,661/147, IDENTICAL hash; `sub0_frontend_tests` 121,211/251 (+288/+6, exactly `fp8_tests.cpp`'s own new coverage). **Default build byte-for-byte unaffected.** **Fresh FP8-artifact verification** against the real 48-layer artifacts already on disk (`D:\ModelWeights\Sub0Llm-Qwen4-full48-fp8`/`-bf16`, from the session owner's own earlier independent reverification pass, reused rather than re-transplanted): configured the real Qwen4 axes (WP4d/e's documented recipe, `--layers 48 --moe-quant-experts 1 --prec-param 2`, `VOCAB=248320` landed exactly). `load_model` ACCEPTED (4.8s, 4.58 GiB fp8 params); `forward`/`forward_one` parity **bit-exact (max diff 0)**; logits vs F32 reference **L2-relative 0.4299, 3/6 argmax-agree — reproduces the session owner's own just-recorded number EXACTLY** (verified two ways: re-derived from my own fresh dump, and diffed my dump directly against theirs). BF16 comparator at the same config: parity 0, L2-relative 0.1989/5-6, also an exact match to the documented figure. | S | **Post-merge throughput sanity check — CONTENTION CAVEAT**: two other agents' own `sub0llm-qwen4-forward`/test processes were independently running on this host during both timed runs (`ps aux` confirmed), so these are not clean numbers, direction only: `forward` BF16 8.39 s/tok vs FP8 9.42 s/tok (~12% slower); `forward_one` (more decode-representative) BF16 4.32 s/tok vs FP8 9.12 s/tok (~111% slower). FP8 slower held in all four numbers, consistent with B33's own clean ~40-60% finding — the magnitude inconsistency between `forward`/`forward_one` here is most likely the contention itself, not a merge-introduced change. **The negative result is confirmed to still hold post-merge; re-measure on an idle host if a precise number is ever needed.** Default arm (BF16, F32) fully unaffected; FP8 arm's own honestly-documented negative perf/quality result now lives inline at `param_store.hpp`'s own header comment (not just this backlog row) per this item's own brief. **Independently reverified before merging**: read the `tools/configurator.cpp` diff in full myself, confirmed it's localized to exactly the `--prec-param`/`Dtype::FP8` lines with nothing incidental; read `param_store.hpp`'s new tradeoff comment in full, confirmed it's clear and prominent. Rebuilt `sub0_frontend_tests` myself at the branch's own `native` (real-axes) config: 288/6 fp8-specific assertions and 121,211/251 overall, exact match. Ran `sub0llm-qwen4-forward` myself against the real 48-layer BF16 artifact on the merged code: parity exactly 0, and the logits' mean/rms/argmax at every row IDENTICAL to my own independently-recorded numbers from this session's earlier B28/B29/B31 verification passes — the default path is provably unaffected. Merged `--no-ff` (`183230b`); hit the SAME stale-generated-config issue B24 Phase 1's own merge hit (the `d196check` build's generated header predated the new `Dtype::FP8` enumerator) — rebuilt `sub0llm-configure` and reran it with the established neutral-axes recipe, then did a full `d196check` rebuild + both suites green (28,969,623/147 + 121,211/251), decode hash unchanged (`816c4a54ad49b8cf`, matching every prior baseline this session). Pushed. Deleted the now-superseded `feature/b33-fp8-backbone` branch (its content lives on `main` via this merge, verified equivalent). |
| B36 | Done, merged, independently reverified | Integrate B25's pipelined I/O work into `main` as a real `--moe-io-mode reactive\|pipelined` build option (default reactive, today's behavior) instead of leaving it stranded on an unmerged local branch | **Genuine re-targeting done, not a replay of the old diff.** B25's branch predates B31; B31 replaced decode's `ExpertCache`+`dequantize_expert`+`transpose_out_in` two-step resolve with `moeq::ExpertCacheSource`/`moe::expert_ffn_row_source` (dequantize straight into GGUF SOURCE order, no transpose). B25's own `moeq::ExpertCache::resolve_from_bytes` (built against the OLD transposed-pool shape) no longer applied at all — re-derived the seam from scratch against `ExpertCacheSource`: added a NEW `ExpertCacheSource::resolve_from_bytes(store, layer, expert, raw_gate, raw_up, raw_down)` (`include/sub0/moe_quant.hpp`) that calls the SAME `dequantize_expert_source` the reactive path calls, just fed caller-supplied byte spans (moeio::PlaneIo's read destination) instead of `store.raw(d)`'s mmap span. `include/sub0/moe_io.hpp` (`moeio::PlaneIo`/`Request`) carried over UNCHANGED -- its mechanism (batched `FILE_FLAG_OVERLAPPED` reads through one shared I/O completion port) never depended on the resolve pool's shape at all, only on "caller hands me an absolute offset + byte count + destination buffer", so nothing there needed re-deriving. **Second reconciliation, not anticipated at filing time: B29 (also already merged) pinned `MOE_DECODE_THREADS` to 1** (fewer threads measured faster once B27 established this host's I/O is not the bottleneck), which invalidated B25's OWN premise that `g_moe_decode[k]`'s per-THREAD io_buf (one thread per selected expert) was the natural staging-buffer shape -- with one thread, all `EXPERTS_PER_TOK` experts' bytes must be staged concurrently regardless of thread count. Replaced with a new `MoeIoStage` (`src/backends/cpu/decode.cpp`) holding `MOE_IO_MAX_INFLIGHT` (= EXPERTS_PER_TOK * moeq::PerExpert = 30 at the real axes) staging buffers, indexed by SELECTION ORDER (`k`, the router's own top-k slot) rather than by thread -- `ParallelExperts::prefetch(idx, n)` (moe_math.hpp's own hook, detected via `requires` and now living in the SHARED `forward_row_via_run_ex` core so both `forward_row_via_run` and B31's own `_ex` entry point get it for free) issues all `n` experts' plane reads into this stage before ANY resolve starts; the single decode thread then walks k serially, `wait()`-ing only for the k it is about to consume while the OS keeps completing the rest in the background -- overlapping ONE thread's compute with ITS OWN still-in-flight later reads, not several threads' reads overlapping each other (the shape B25 originally targeted). `--moe-io-mode reactive\|pipelined` -> `constexpr bool MOE_IO_PIPELINED` (default `false`/reactive, `tools/configurator.cpp`, one CLI option + one `cos <<` emission + one configure-time cross-check that pipelined requires `--moe-quant-experts 1`, all localized near `--moe-quant-experts`/`--prec-param` — 35 lines total, nothing else in that file touched). `g_moe_decode_io` (the explicit-I/O file handle) opened as a SECOND, independent handle onto the same `.moeq` sidecar only under `MOE_IO_PIPELINED`, right after `g_moe_quant.open()` in `load_moe_quant_sidecar` (`src/backends/cpu/backend.cpp`) — the default build never opens it at all. | M | **Regression-safety (default/reactive build unaffected)**: `sub0_frontend_tests` 120,923/245 — EXACT match to the documented pre-B36 `main` baseline (this suite's own MoE-exercising cases, `moe_quant_tests.cpp`/`moe_qwen4_fixture_tests.cpp`, run at synthetic axes with `MOE_IO_PIPELINED` at its default-false value, so this is the real regression gate for the reactive arm). **Bit-exactness at BOTH settings, on the real 48-layer BF16 artifact** (`D:\ModelWeights\Sub0Llm-Qwen4-full48-bf16\qwen4_full48_q_bf16.bin`, built through the real `sub0llm-configure` at the production axes, `--moe-io-mode reactive` and `--moe-io-mode pipelined` in two separate build directories): `sub0llm-qwen4-forward --tokens 3` gives `forward`/`forward_one` parity **exactly 0** at BOTH settings, and the per-row logit statistics (mean/rms/range/argmax, 3 rows) printed BYTE-IDENTICAL between the reactive and pipelined builds (row 0 argmax 487, row 1 argmax 198, row 2 argmax 220, matching means/rms to all printed digits at both settings) — the pipelined path is a pure I/O-mechanism swap, verified, not merely argued. **Throughput**: measured on this host WHILE B37/B38 were independently dispatched and building/benchmarking in parallel on the SAME host (`docs/ACTIVE_WORK_LOG.md`'s own B36/B37/B38 dispatch entry) — real, confirmed external contention, not an artifact: raw per-run wall-clock varied 2x run-to-run within the SAME build (reactive: 9.75-19.52s/3 tokens; pipelined: 9.79-21.78s/3 tokens) with no stable ordering either direction. The CLEAN (least-contended) sample cluster from each arm lands close together — reactive's fastest runs 9.75s/3.63s ≈ 3.25-3.63 s/token, pipelined's fastest three 9.79-9.98s ≈ 3.27-3.33 s/token — consistent with B25/B27's own "no measurable win, I/O is not the bottleneck on this warm-page-cache host" finding, but this session's own measurement is NOT a clean reproduction of B25's precise "~1.4% worse" figure: today's contention from concurrent sibling work makes a tight A/B impossible to certify from this pass alone. Reported honestly rather than forcing a number. | Kept default-off exactly as filed; the pipelined arm is a real, independently-buildable, bit-exact capability for a machine/session state where the sidecar is NOT page-cache-resident (cold boot, memory pressure, a host too small to cache 37 GiB at all) — see the CLI option's own help text and `include/sub0/moe_io.hpp`'s header comment for the documented, situational recommendation. Files touched: `include/sub0/moe_io.hpp` (new, carried over from `feature/b25-explicit-io-resolve` with header-comment updates reflecting B31/B29's own since-merged changes), `include/sub0/moe_quant.hpp` (`Store::max_desc_bytes()`, `ExpertCacheSource::resolve_from_bytes`), `include/sub0/moe_math.hpp` (the `prefetch` hook in `forward_row_via_run_ex`), `src/backends/cpu/internal.hpp` (`MOE_IO_MAX_INFLIGHT`, `g_moe_decode_io` extern), `src/backends/cpu/decode.cpp` (`MoeIoStage`, `ParallelExperts::prefetch`, the pipelined branch in `forward_one`'s MoE call site), `src/backends/cpu/backend.cpp` (`g_moe_decode_io` definition + conditional open), `tools/configurator.cpp` (`--moe-io-mode`, one new flag, localized). Zero overlap with B37's (`fp8.hpp`/`param_store.hpp`/`model_file.hpp`/`sub0llm-transplant.cpp`) or B38's (`simd_reduce.hpp`/the four `*_math.hpp` REDUCTION bodies -- this package's own `moe_math.hpp` touch is a different function, `forward_row_via_run_ex`'s prefetch hook, not `expert_ffn_row`/`expert_ffn_row_source`'s summation loops) files. |
| B38 | Dispatched (active) | Integrate B34's SIMD-reduce work into `main` as a real `--simd-reduce` build option, default `false` (today's scalar behavior, bit-exact) | Same user directive as B36/B37. Prefer a shared, templated kernel body over duplicating each kernel twice, to avoid drift risk between the two arms. | M | Default arm must be bit-exact identical to today's `main` (decode hash unchanged); the ON arm keeps its own honestly-documented negative perf result inline |

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

**Still open at first filing**: P-core/E-core thread-affinity placement of the real OpenMP team vs. the
harness's `std::thread` pool on this 8P+16E part; whether per-region OpenMP wake-up cost differs
materially once the team is reused across 48 short regions inside a much larger, much busier process
already holding 25+ GiB resident (TLB/cache contention the isolated harness's smaller footprint would not
reproduce); whether something upstream of the resolve call itself (the router's own work, or the two-phase
compute-then-combine restructuring) serializes the ten resolves in practice despite no lock being visible.

#### Follow-up, 2026-09-10 — the ten threads ARE running, partially concurrently; the picture is more specific than "one thread's worth"

Temporary wall-clock instrumentation was added to `ParallelExperts` (per-thread `start`/`end`
`std::chrono::steady_clock` timestamps around each `body(k, ...)` call, gated to the first few calls of a
run, reverted before commit — not merged) and a short real decode run was captured. Findings, all directly
measured:

1. **`omp_get_num_threads()` inside the region reports 10, every call** — the team really is 10 real OS
   threads, confirming the earlier code-review conclusion that there is no silent team-size collapse
   (e.g. from `OMP_DYNAMIC`/nested-parallelism defaults).
2. **The very first `ParallelExperts` call of the whole process (layer 0 of the first token) runs fully
   sequentially** — thread N's `body` call starts within ~1-6 ms of thread N-1's ending, with essentially
   no overlap at all. This is consistent with `libomp`'s worker-thread-pool cold-start cost (spinning up 9
   new OS threads for the first time) and affects only that single call — it is not the steady-state
   pattern and does not by itself explain a per-token cost.
3. **Every subsequent call (steady-state) shows real but PARTIAL concurrency: 6 of the 10 threads start
   within a ~1-2 ms window of each other and run with genuinely overlapping wall-clock durations (30-99 ms
   each); the remaining 4 threads start ~90-120 ms later, right around when the first group of 6 begins
   finishing.** This is not "one thread's worth of work" (the original framing from the first measurement
   pass) — it is real, repeatable, bounded parallelism, consistently capped near 6 rather than reaching 10,
   across multiple layers and multiple independent runs.
4. **Windows core parking was tested directly and ruled out as the (sole) explanation.** Duplicated and
   activated the built-in "High performance" power scheme (this machine's active scheme was "Balanced";
   `powercfg /q ... SUB_PROCESSOR` found no distinct core-parking-minimum sub-setting exposed by name on
   this Windows build, so the scheme swap was the available test) and re-ran the same instrumented decode:
   throughput was unchanged (5.36 s/token vs. 5.85 s/token, within normal run-to-run variance) and the
   identical "first-call-serial, then 6-then-4" pattern reproduced exactly. Power scheme was restored to
   Balanced afterward (the duplicated scheme was deleted) — no lasting change to the machine.
5. **Per-expert duration during the "concurrent" 6-thread bursts (30-99 ms) is 3-9x higher than BOTH the
   isolated single-thread baselines** recorded in the empirical research pass (cold ~11.4 ms, warm ~6.5 ms
   per expert, single-threaded). This is the most load-bearing new number: it means the live engine's
   per-expert cost is not simply "the same work, less parallelized" — something makes each resolve
   materially more expensive when several run concurrently in the real engine than the isolated harness's
   equivalent concurrent case (which scaled to 2.13 ms/expert at 10 threads). Candidates not yet
   discriminated: real memory-bandwidth/cache contention from 6 threads' dequant+transpose work competing
   against the process's own 25+ GiB resident working set (the isolated harness's smaller footprint would
   not reproduce this); a real, narrower cap on concurrent hard-fault service somewhere in the Windows
   fault path (KB156932's architectural claim, at whatever the modern equivalent pool size actually is,
   not the archived "three threads" figure) that lets a handful of faults progress concurrently but not
   ten; or NVMe queue/driver behavior specific to small (~85 KB, per the empirical doc's own finding)
   per-fault reads issued this way, as distinct from the large sequential `ReadFile` reads the overlapped-
   I/O probe used to reach 6-16 deep queues.

**Acceptance criteria (superseded by the follow-up below — kept for history)**: identify why concurrency
plateaus near 6 rather than 10, and why per-expert cost under real concurrent load is 3-9x the isolated
baseline rather than matching it; fix or design around it; re-verify decode throughput moves toward the
harness's own 10-thread projection (~1.0–1.3 s/token) while `--verify`/parity/the determinism fixture stay
exactly unchanged (this remains a pure scheduling question — the two-phase compute-then-sum structure
means answer correctness cannot depend on which thread computes which expert or how quickly). VTune's
thread-concurrency histogram over the resolve region (not its hotspot list) remains the next-named
instrument.

#### Follow-up 2, 2026-09-10 — the plateau is NOT a function of requested thread count at all

Direct test: `DEFAULT_THREADS` was overridden 24 → 6 in the (untracked, generated) build header, so
`MOE_DECODE_THREADS = min(DEFAULT_THREADS, EXPERTS_PER_TOK)` dropped from 10 to 6 — i.e. the team was
asked for *exactly* the concurrency level the previous measurement seemed to plateau at. Same wall-clock
instrumentation, same decode run. Reverted cleanly afterward (`sub0_system.hpp` restored from a pre-edit
backup, `decode.cpp` diff against `main` confirmed empty both before and after, executable rebuilt from
clean source).

**Result: asking for exactly 6 did not produce clean 6-way concurrency or remove the cost inflation.**
Real overlap sub-plateaued further, to roughly 2-3 threads genuinely concurrent at a time (not all 6);
per-expert duration stayed in the same inflated range as the 10-thread run (15-73 ms vs. the 10-thread
run's 30-99 ms — same order of magnitude, not reduced toward the 6.5-11.4 ms isolated baseline). Overall
throughput was essentially unchanged (5.24 s/token at 6 threads vs. 5.36-5.85 s/token at 10) — if anything
marginally *better* with fewer threads requested, meaning the extra requested concurrency above ~3 buys
nothing measurable in this environment.

**This rules out "6 is a hard cap that gets fully saturated, and asking for 10 just wastes 4 threads
uselessly waiting."** The effective concurrency ceiling and the per-expert cost inflation are **not**
functions of `MOE_DECODE_THREADS` at all — the same ~2-3x ceiling and the same order-of-magnitude
inflation appear whether 6 or 10 threads are requested. This points away from anything at the
OpenMP/application-thread-count level and squarely at something in the resolve path itself that is
constant regardless of caller-side parallelism: real memory-bandwidth/cache contention against the
engine's own 25+ GiB resident working set (a genuine possibility the isolated harness's much smaller
footprint cannot reproduce, and the one candidate from the original list that predicts exactly this
"doesn't scale with requested threads" shape), or a real OS/device-level concurrent-fault-service ceiling
that is a property of the file/device/access-pattern rather than of the caller's thread count.

**Practical implication for the engine, separate from full root-causing**: if the effective ceiling really
is ~2-3x regardless of request, `MOE_DECODE_THREADS`'s current value (10, one thread per selected expert)
may be leaving no measurable throughput on the table relative to a much smaller, much cheaper team (each
`MoeDecodeThread` costs ~25 MiB; a 3-4 thread team would cost a fraction of the current 250 MiB and free
that headroom for the file cache B20 already showed governs the sidecar's own resident share) — but this
should be confirmed with a longer, less noisy run before acting on it, not shipped from two 12-token
samples.

**Acceptance criteria (updated)**: root-cause the ~2-3x ceiling and the inflation as a property of the
resolve path independent of thread count — the leading candidates are contention against the engine's own
large resident working set (test: repeat the harness's own concurrent-pipe measurement WITH a matching
resident-ballast simulation, as the empirical doc's §5b already did for a different question, but this
time instrumented with the same per-call wall-clock timing used here) and a real modern-Windows
concurrent-hard-fault-service limit (test: the same wall-clock instrumentation against `FILE_FLAG_OVERLAPPED`
reads into an owned arena instead of mmap faults, at the real call site, not just the standalone probe).
VTune's thread-concurrency histogram over the resolve region remains a candidate instrument but is no
longer the only lead — the ballast-plus-timing test above is cheaper and more targeted now that thread
count itself is ruled out as the axis that matters.

Full research trail: session scratchpad `sub0mempage-research-empirical-concurrency.md` (not part of this
repo) — every number above is reproducible from the commands logged in that file's appendix.

### B22 — Migrate `moeq::Store`/`ExpertCache` onto Sub0MemPage once it has a real implementation

Filed 2026-09-10, alongside the Sub0MemPage repository's own ownership-model resolution
(`docs/sub0llm-consumer-trace.md` in that repo). **Not urgent — this is a backlog placeholder, not a
call to action.** Sub0MemPage (github.com/CraigHutchinson/Sub0MemPage) is currently design-only, with no
working implementation; this item exists so the migration is not forgotten once one exists, per this
project's own standing discipline of tracking known-future work rather than letting it evaporate between
sessions.

**Why this is a real migration candidate, not a speculative one**: Sub0MemPage's own design was built
*against* this exact code as its first real consumer, not designed in the abstract and retrofitted after
the fact. `include/sub0/moe_quant.hpp`'s `moeq::Store` (the ~37 GiB S0Q1 sidecar's read-only mapping) and
`ExpertCache<Slots, SlotFloats>` (the dequantized-plane pool with round-robin slot reuse and
`(layer,expert)`-keyed hit detection) are, respectively, exactly the "addressable byte-range source" and
"caller-owned destination slot pool" Sub0MemPage's own `register_region`/`register_slots` contract was
shaped to sit underneath. Sub0MemPage's own research explicitly names `ExpertCache`'s current round-robin
policy as "the naive replacement a Sub0MemPage-backed policy would supersede," and its own
`sub0llm-consumer-trace.md` traces the exact migration this item tracks: `ExpertCache::pool_` gets
registered via `register_slots` **unchanged, no new allocation**; `ExpertCache::resolve()`'s conflated
"which slot, is it a hit, fault the bytes in" logic splits into Sub0MemPage's own bookkeeping (the first
two, generalized) plus a one-line change at `dequantize_expert()`'s call site (reading from the slot a
lease names instead of `store.raw(desc)`); and B21's own motivating defect (this file, above) is exactly
the class of problem a batched `prefetch` call ahead of `ParallelExperts`' parallel region would sidestep
by construction, independent of whatever B21's own root cause turns out to be.

**What this buys, concretely, once Sub0MemPage exists**: proactive prefetch of a layer's ~10 expert planes
issued from one thread before the parallel resolve region begins (replacing today's ten independent,
uncoordinated reactive faults); a real, enforced hard budget on the sidecar's resident share instead of
leaving it to the OS's default page-cache behavior entirely unmanaged; `wont_need` to deprioritize a
completed layer's slots for reuse, something `ExpertCache`'s own policy has no way to express today; and —
if Sub0MemPage's own future GPU/heterogeneous-memory backend work proceeds (raised the same day this item
was filed, not yet designed) — a shared residency vocabulary between this CPU-side MoE cache and any future
CUDA/iGPU expert-residency work, rather than two independently-invented ad hoc pools.

**What does NOT change, and is worth stating so this isn't mistaken for a bigger rewrite than it is**:
`dequantize_expert()`'s own two-step decode (`gguf::to_f32` then `transplant::transpose_out_in`) is
untouched — Sub0MemPage never interprets bytes. `ExpertCache`'s own allocation, sizing, and per-thread
ownership are untouched. The `#pragma omp parallel`/`#pragma omp for schedule(static)` structure in
`decode.cpp` is untouched. `op_moe`'s own separate 8-slot batched-path pool (a genuinely different
residency policy, real cross-row hit rate, deliberately left alone by B20) is a separate, independent
migration candidate if this item is ever acted on — not assumed to move at the same time.

**Acceptance criteria, once Sub0MemPage has a real implementation to migrate to**: `register_slots` over
`ExpertCache::pool_` with zero new allocation (verified, not assumed); `--verify`/parity/the WP5c
determinism fixture all byte-for-bit unchanged (this is once again a pure scheduling/residency change, not
an arithmetic one); a measured throughput comparison against the pre-migration baseline, honestly reported
whether or not it resolves B21's own still-open root cause (it is not expected to, per B21's own closing
note — a proactive design sidesteps depending on N reactive threads cooperating correctly, but B21's
specific unexplained ~2-3x concurrency ceiling and cost inflation could plausibly persist under an
explicit-fill backend too, and that would itself be a finding worth reporting, not a silent non-result).

### B23 — Naive full-warm-up of the sidecar does not work, and the reason is arithmetic, not a tuning problem

Filed 2026-09-10, a cheap spike run directly against the user's own question: before building the full
Sub0MemPage async-scheduling engine, is there a much simpler fix — just touch the whole sidecar once at
startup and let the OS page cache hold it? **Tested directly, and the answer is a clean, arithmetic no.**

**Method**: a standalone scratch probe (`warm_sidecar.cpp`, not part of this repo, uses the existing
`sub0::FileMap` unmodified) opened the real 37.11 GiB `.moeq` sidecar and sequentially touched every 4 KiB
page once — a full, unconditional warm sweep, not a sampled one. This completed in **27.3 seconds at an
effective 1459 MB/s** (far faster than decode's own fragmented ~55-108 MB/s, confirming sequential
single-threaded readahead is dramatically more effective than the small-random-plane access pattern decode
uses — itself a useful, separate data point). Immediately after the probe exited, a real decode run was
launched against the identically-warmed file, with `\PhysicalDisk\Avg. Disk Queue Length`, `\Memory\Pages
Input/sec`, `\Memory\Pages Output/sec`, `\Paging File(_Total)\% Usage`, and `\Memory\Available MBytes`
sampled throughout.

**Result 1 — no real pagefile swap.** `\Memory\Pages Output/sec` measured **exactly 0.00 across every
sample** of the whole run, and `\Paging File(_Total)\% Usage` stayed flat at 6.73% (matching the
pre-experiment baseline of 6.43% — background noise, not growth). This directly answers the question that
motivated the spike: the machine is **not** swapping anonymous/private memory to the pagefile. Whatever is
happening is a different mechanism.

**Result 2 — the warm-up did not help.** Despite the entire file having been touched 100% 27 seconds
earlier, decode's steady-state disk activity was **not reduced** — `\PhysicalDisk\Avg. Disk Queue Length`
measured 0.10-0.17 and `Disk Read Bytes/sec` 55-87 MB/s during decode, matching or exceeding the
pre-warm-up cold baseline (0.04-0.09, 55.7 MB/s) recorded for B21. Final throughput was **5.60 s/token**
(12 tokens in 67.22s) — statistically indistinguishable from the 5.53-5.85 s/token band measured without
any warm-up. **A naive full warm sweep buys nothing measurable.**

**Result 3 — `\Memory\Available MBytes` fell continuously and did not plateau** during the sampled window
(14,889 → 14,421 MB over ~9 seconds, ~52 MB/s net drain) — closely tracking the concurrent disk read rate.
This is the mechanism: pages the warm-up brought into the systemwide standby list are being evicted again,
under real memory pressure, before decode gets to reuse them — not classic pagefile swap (Result 1), but
functionally the same practical symptom (continued real disk reads) via a different OS mechanism (standby-
list churn on the cheapest-to-evict, read-only, file-backed mapping — exactly the mechanism
`sub0mempage-research-os-io.md`'s own H3 named as a real but, it estimated, only "contributory ~9-12%"
effect; this result shows it dominates at this scale, not merely contributes).

**Result 4 — the actual reason, and it's simple arithmetic, not a caching-cleverness problem.** Measured
directly, not estimated: available memory **before** model load was 42.71 GiB (of 63.43 GiB total RAM —
i.e. **~20.72 GiB is already committed to the OS and other processes on this real, shared development
machine at baseline**, not a hypothetical margin). The backbone + worker arena together commit **25.39
GiB** (the `[mem] after graph_reset` figure, unchanged from every prior measurement this session). That
leaves only **17.32 GiB of headroom for a 37.11 GiB sidecar — at most 46.7% of it can be simultaneously
resident, no matter how it is cached, warmed, or scheduled.** The other ~53% must be re-fetched from disk
on essentially every access pattern that touches it, by construction, regardless of implementation
cleverness on the caching side.

**Why this matters beyond just closing out the "is a simple warm-up enough" question**: it reframes what
Sub0MemPage should be understood to target. The project's own headline finding (`design.md` §5/§6 —
"prefetch, not faster faults, is the top lever") is not weakened by this result, but its framing sharpens:
the goal is not, and structurally cannot be, "eliminate the sidecar's disk I/O via better caching" — on
this machine, at this model's scale, **some real disk I/O per token is unavoidable, full stop.** The
achievable goal is efficiently *overlapping* that unavoidable I/O with useful compute (the router knows
layer L's experts before layer L's FFN runs — the "declared signal known ahead of use" property the whole
design already leans on), not removing the I/O. This is a sharper, more honest statement of the target
than "make the file stay resident," and it should inform how any future Sub0MemPage implementation reports
its own success — "reduced disk queue idle time" and "increased overlap of I/O with compute," not "reduced
total bytes read from disk," which this result shows has a hard floor around 53% of the sidecar per full
routing sweep on this machine's real available memory.

**Caveats, stated honestly**: this is one machine's one real-world memory budget (a shared development
box with ~20.7 GiB already committed to other things at baseline), not a clean benchmark environment — the
"46.7%" figure is specific to this session's exact conditions and will differ on a dedicated inference
box with less competing load, or a machine with more RAM. The qualitative conclusion (working set exceeds
available headroom, so full residency is structurally impossible without more RAM) is robust to that
caveat; the specific percentage is not a portable constant.

Scratch probe (`warm_sidecar.cpp`) lives only in the session scratchpad, not this repo — a clean, minimal,
reusable tool if this experiment needs re-running under different memory conditions.

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
