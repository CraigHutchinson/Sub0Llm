# Coordinated storage stack: Sub0Llm, Sub0TieredCache, Sub0MemPage

Revision S1, 2026-09-25 (checkpoint updated the same day; see "Current checkpoint"). **Plan, not an
implemented integration.** Source audit starts at Sub0Llm `5eede7d`, Sub0TieredCache `e1639f8`,
Sub0MemPage `2ce9306`. Lower libraries are now at Sub0TieredCache `e4da6a7` and Sub0MemPage `213acdd`. Existing compute and checkpoint behavior
remain the baseline. This document owns application acceptance and cross-project sequencing;
[cache requirements](../../Sub0TieredCache/REQUIREMENTS.md) own row semantics and
[transfer requirements](../../Sub0MemPage/REQUIREMENTS.md) own byte/lifetime semantics.
Workspace links assume sibling checkouts; shipped builds use explicit pinned dependencies.

## Responsibility matrix

| Concern | Sub0Llm | Sub0TieredCache | Sub0MemPage |
|---|---|---|---|
| Token/ngram IDs, router top-k, model layout | Owns | Receives row IDs + adapters | Receives byte extents |
| GGUF/safetensors/sidecar parsing | Owns adapters | Calls registered extent resolver | No format knowledge |
| Frozen row versions and cached representations | Supplies source generation/codec | Owns key, conversion scheduling, admission and row leases | Preserves registered immutable source lifetime |
| Quantization, tensor transpose and model kernels | Owns math and codec implementation | Schedules codec; caches requested representation | Never transforms content |
| Local/remote tier choice, HTTP mirror | Sets deployment policy | Owns policy, HTTP and versioned disk cache | Moves local-file/host/device bytes |
| Slot/transfer safety | Holds outputs through compute | Reserves row outputs; holds raw leases through conversion | Enforces transfer claims/completion; optional raw byte cache |
| Memory | Sets total host/pinned/device budgets | Partitions cache/output budgets | Enforces bounded transfers/staging within supplied pools |
| Accelerator execution | Owns compute stream/schedule | Bridges row readiness to compute | Optional CUDA/cuFile/SYCL transport adapters |

Dependencies point down: Llm -> TieredCache -> MemPage -> OS/vendor SDK. Llm can also consume MemPage
for byte-only transfers with no row-cache semantics (e.g. whole-artifact streaming); it must not fork
an alternative I/O engine. MoE representation caches move behind TieredCache only where their contract
fits and earns its cost; no wrapper is imposed on an existing native-quant hot loop without a measured
reason. Homogeneous expert planes can be separate fixed-width tables; variable-shape objects must not
be smuggled into a row-width contract. Record the chosen mapping per adapter.

## Top-down workload contracts

| ID | Need and integration point | Required lower-layer property | Acceptance |
|---|---|---|---|
| E1 | Frozen external ngram/PLE rows before embedding | Ordered batch resolve, duplicate coalescing, explicit source/output dtype and width | Same resolved rows and model outputs as resident reference; trainable tables unchanged |
| E2 | Routed MoE encoded planes after router selection | Known-batch prefetch, contiguous or explicitly gathered planes, event-safe leases | Encoded bytes identical; same selected experts/math; compare current pipelined/native-quant paths |
| E3 | Frozen quantized backbone/GDN sidecars | Native encoded representation retained, role/extent bounds, bounded staging | No forced float expansion; sidecar role and numerical parity gates remain authoritative |
| E4 | CUDA model/row input on a particular device | Qualified file-to-CUDA path, stream readiness and final-use retirement | Staged path correct first; GDS direct path separately proves route and parity |
| E5 | Budget pressure, cancellation and repeated sessions | Bounded queues/pools, explicit errors, safe teardown and generations | No live-pointer reuse or silent extra allocation; recover/restart without stale data |

The [source audit](../../Sub0TieredCache/docs/reference-consumer-sub0llm.md) identifies the current
`moeq::Store`, decoded caches, `g_moe_decode_io`/`g_moe_io_stage` and `g_backbone_quant` seams.
`src/backends/cuda/backend.cu` currently rejects `NGRAM_EMBED` and `USE_MOE` at compile time;
these need separate compute packages before E1/E2 can qualify on CUDA.

These are requirements, not claims that current CUDA kernels consume every encoded format or that a
PLE adapter already exists. Reject unsupported consumer/representation pairs rather than pretending
storage support implies compute support. Frozen external rows are eligible; per-step learned embedding
or parameter updates remain in engine-owned arenas. Any checkpoint/config change follows AGENTS.md's
compatibility and fingerprint rules, separately from transport selection.

## End-to-end data and lifetime trace

1. At startup, Llm parses metadata, selects output representations/device, sizes budgets and registers
   immutable sources and format/codec adapters with TieredCache. TieredCache allocates/borrows bounded
   buffers and registers transport endpoints with MemPage. Vendor initialization stays out of decode.
2. Llm computes row IDs/router choices. TieredCache resolves keys and byte extents, checks resident
   representations, then reserves output capacity and requests lower transfers. Known future input is
   declared; next-token/next-layer predictions remain speculative.
3. MemPage completes exact byte transfers into host or device storage. TieredCache gathers/converts if
   needed, with source and destination claims held until the last conversion event. Only then is a row
   published. Native-encoded identity paths bypass conversion; GDS cannot repair an absent GPU codec.
4. Llm acquires RowLeases or resolves into its own preallocated output buffer. CPU compute reads host
   bytes; GPU compute uses an explicit readiness dependency. Input/output storage survives through the
   last compute event. Completion tickets alone do not protect residency.
5. Release/retirement makes storage reusable. Error, timeout and cancellation do not free in-flight
   destinations. Source invalidation creates a new generation; old readers drain safely.

Raw staging, converted rows and activations are different live sets. Budget all concurrently live
allocations once, including host pinned memory, VRAM, queues/events, registration and driver headroom.
No two layers independently evict the same allocation. No hidden "load entire model" fallback.

## Joined-up development sequence

| Slice | Bottom-up delivery | Middle-layer consumer | Top-down feedback/gate |
|---|---|---|---|
| S0 contract/fixtures | MemPage M2 design + deterministic fake backend | TieredCache T0 in-memory oracle | Llm defines E1–E5 fixture shapes, IDs, layouts and required errors now |
| S1 local CPU vertical slice | M2 state machine + M3 real-file backend | T1 frozen-row cache on real MemPage | E1 small external table vs resident oracle, then E2 byte adapter; no GPU/network needed |
| S2 staged CUDA vertical slice | M4 host-pinned -> CUDA transfer | T2 GPU row/identity and codec lifecycle | E4 exact bytes/events, then compute parity on supported shapes; preserve CPU path |
| S3 accelerated transports | M5 Intel USM, M6 native-Linux NVIDIA cuFile qualification | T4 same row contract | Require exact route evidence; replay S1/S2 fixtures and model parity |
| S4 remote + deployment scale | Existing local transport remains | T3 HTTP -> versioned local mirror | E1 real shard samples; bounded large-working-set traces; optional remote tier |
| S5 measured optimization | Tune chunking/queue depth and transport | Tune row admission/layout with trace evidence | Full application latency/throughput/memory gates; retain fallback and parked toggles |

M5/M6 can be researched in parallel with S1, but neither blocks CPU correctness or justifies skipping
S2. Work in all three repositories against one slice: upper acceptance fixtures are written early,
lower implementations land first, then the middle layer consumes a pinned lower revision, then the
engine consumes the pinned pair. Do not merge a consumer relying on an unpublished/unavailable revision.
No automatic fetch of latest main. Ordinary builds/tests are offline and do not need sibling checkouts.

## Shared fixture and change protocol

The fixture schema records source bytes/hash, source generation, extents, row IDs/order/duplicates,
source/output representation and width, target domain, expected bytes/errors and deterministic event
schedule. Put byte/state fixtures in MemPage, row/version/codec fixtures in TieredCache, model/format
fixtures in Llm. Upper suites reuse released lower fixtures where relevant and add their own oracle;
no copied implementation becomes its own correctness oracle. Large external artifacts are optional,
identified by hashes and provenance, not required in ordinary CI.

Each slice's acceptance manifest records all three git SHAs, contract revision S1, fixture hashes,
compiler/build configuration, OS/device/driver/filesystem facts, test counts and explicit skips, fallback
policy, observed transfer path and memory high-water marks. The manifest is a planned test output,
not a new runtime wire format or current tool. Its first consumer is the S1 integration test runner.

If a caller exposes a missing guarantee: reproduce it in a small upper test, translate the minimal
byte-level requirement into a lower fixture, review both contracts, implement below, rerun upward.
If a backend cannot meet a guarantee: report a capability/failure below, define the correct fallback
above, update the application acceptance case. Never paper over a constraint in one repository only.
A change to domains, lifetimes, row keys or fallback behavior requires a three-plan review in the same
work package. Dependency pins advance only after consumer CI is green.

## Correctness, performance and release gates

- MemPage standalone: deterministic failures/concurrency, exact real-file bytes, boundary/EOF/alignment,
  budget pressure and no hot-path allocation. Optional hardware suites cannot stand in for these.
- TieredCache standalone with pinned MemPage: row ordering, dtype/native encoding, versions, leases,
  real lower transport under pressure, and remote local-server tests. No Llm link dependency.
- Llm: default configuration retains existing exact assertion counts; enabled integration runs full
  unfiltered tests at tiny-realistic and production-like axes. Exact byte/storage parity first;
  model logits/reference metrics preserve the current numerical gates and token-output checks.
- Hardware: qualify staged CUDA on supported Windows/Linux; qualify GDS only on a supported native-Linux
  tuple. macOS host tests qualify portable cache semantics, not NVIDIA/Intel capability. Missing hardware
  is a recorded unqualified path, never a mocked performance pass.
- Measure current baseline before selecting cache/transfer defaults: per-row latency, queue wait, I/O and
  H2D bytes, cache hits, staging/VRAM peaks and end-to-end token/prefill times. Keep cold/warm distinction,
  interleaved trials and uncontended runs per OPTIMIZATION_PROCESS.md. No timing claim in this plan.
- Release requires each standalone library gate plus one CPU and each advertised accelerator vertical
  slice. C++ authoring/review gates apply before merging implementation. Keep new engine paths off until
  consumer correctness and measured benefit are established; no speculative unused CLI/config knobs.

## Current checkpoint

Updated 2026-09-25. Llm has **not** adopted either library yet: no Sub0Llm build, engine or test
file depends on Sub0TieredCache or Sub0MemPage. This plan authorizes staged design/implementation
work. It does not claim that NVIDIA GDS, a CUDA row cache or the end-to-end stack is available.

### Pinned lower revisions (both on `main`)

| Repository | Revision | Content | Evidence |
|---|---|---|---|
| Sub0MemPage | `213acdd2121cec369c6b9606c514db2231e94f27` | M2 state machines (`slot_pool.hpp`, `transfer_set.hpp`); M3 slice 1 `local_file_backend.hpp`, a portable worker-pool local-file backend (POSIX `pread`, Windows positional overlapped `ReadFile`); `Sub0MemPage::testing` CMake target (deterministic fake backend); optional Intel capability inventory | `docs/validation/2026-09-25/README.md` (M2); `docs/implementation-plan.md` "Checkpoint: dev loop + M3 slice 1" |
| Sub0TieredCache | `e4da6a7e87bcbc65f79257bef9843d41adbec66d` (pins MemPage `213acdd`) | T0 bounded row cache `row_cache.hpp` (`Table`/`RowCache`/`RowLease`, CLOCK eviction, coalescing, generation-bound `invalidate`, bf16->f32 codec); T1 sharded local-file sources `local_file_source.hpp` (`FlatFileResolver`, `register_local_file_shard`); T3 remote HTTP mirror plus `remote/mirror_backend.hpp` | `docs/validation/2026-09-25/README.md`: 9/9 CTest on GCC/clang, ASan+UBSan and TSan clean, mingw/Wine with the same counts, GitHub Actions on Linux, macOS and Windows MSVC |

Not started: MemPage M3 native async (io_uring, IOCP), M4 staged CUDA, and M5/M6 (Intel USM,
cuFile); TieredCache T2 (GPU rows/codec lifecycle) and T4 (Intel, NVIDIA). T3 exists ahead of its S4
slice. It is not an S1 dependency and gives no S4 acceptance yet: its fixtures are synthetic and it
supports plain HTTP only. No timing claim exists in any of the three repositories.

### Slice status

| Slice | Status |
|---|---|
| S0 contract/fixtures | **Lower layers done**: MemPage M2 plus the fake backend, and the TieredCache T0 oracle. The Llm-side E1–E5 fixture shapes are still unwritten; E1 is being written first, in S1. |
| S1 local CPU vertical slice | **Lower half done**: MemPage M3 slice 1 and TieredCache T1 run on real MemPage at the pins above. **Llm half not started**: the E1 fixture, the adapter, the resident-reference comparison, then E2's byte adapter. |
| S2–S5 | Not started. T3 remote code exists early; see above. |

### Next: S1 on the Sub0Llm side (handoff for a local session)

The steps are the "Joined-up development sequence" S1 row, applied to E1. None of it has landed in
Sub0Llm yet: a cloud session studied the APIs and stopped before writing code.

1. **Coordinate.** Check `docs/ACTIVE_WORK_LOG.md` and add a row for the files below. The 2026-09-26
   entry released the host and lists no active storage/ngram work. Take no performance measurements
   for S1; correctness only. Any later timing follows `OPTIMIZATION_PROCESS.md` on the dedicated machine.
2. **E1 fixture (Llm owns it; see "Shared fixture and change protocol").** A small, deterministic,
   generated external table, written to a temp file by the test:
   - row count;
   - source dtype bf16, output dtype f32 (width x2);
   - row width in elements;
   - an ordered request of row IDs with duplicates and non-monotonic order, including row 0 and the
     last row;
   - an out-of-range ID case that must fail as a whole (all-or-nothing);
   - an expected-bytes oracle derived from the generator formula, never from the adapter;
   - source hash, generation and extents.

   Mirror the real Qwen4 shape in miniature, 16 rows per position with 160 elements per row (see
   `NGRAM_TABLE_TIERED_STORAGE.md` §0 and `tests/fixtures/qwen4_preview/ngram_embedding_manifest.json`),
   but keep the row count small.
3. **Resident reference.** The existing resident path is the plain branch of `op_embed` in
   `src/backends/cpu/backend.cpp`: `o[t, j] = tab[ids[t] * C + j]` over a row-major table. The
   `ParamCPtr` read widens bf16 through `sub0::bf16_widen` in `include/sub0/bf16.hpp`. `op_embed` is a
   `static` function inside the engine, so there are two options:
   - re-express the plain gather engine-free, as `tests/ngram_qwen4_fixture_tests.cpp` already does for
     the concat convention, and state that choice;
   - or expose a small shared gather so the reference is the engine's own code. This option touches
     `backend.cpp` (a shared file), so coordinate first.

   Compare the adapter byte-for-byte against the reference and the generator oracle. Model-output
   parity (E1's second acceptance clause) needs engine wiring and is a later step. Trainable ngram
   tables (`ngram_tab`) stay in engine arenas and are not replaced.
4. **Thin adapter.** Build it over `sub0tieredcache::Table`:
   - `Representation::bf16_to_f32` with `source_row_bytes = width * 2` and
     `output_row_bytes = width * 4`;
   - `FlatFileResolver` plus `register_local_file_shard` on a caller-owned
     `sub0mempage::LocalFileBackend`;
   - `resolve_into(rows, leases)` in request order, then copy each lease's bytes into the caller's
     preallocated output (AGENTS.md §1: size the lease and output buffers once, outside the per-call
     path).

   Relevant API facts:
   - `TableConfig` needs `output_storage` sized `budget_rows * output_row_bytes`, `scratch_storage`
     sized `scratch_rows * source_row_bytes`, and nonzero `max_tickets` and `max_batch_rows`.
   - `LocalFileBackend::create(LocalFileBackendConfig{.workers, .queue_capacity, .max_sources})`.
   - Duplicate rows in one call get separate leases on one slot and cause exactly one fetch; check this
     with `Table::stats().fetches` and `.coalesced`.
5. **Default-off build toggle, no default dependency.** Add a CMake option, for example
   `SUB0_STORAGE_TIEREDCACHE`, default OFF. It is a build-dependency switch and not a model capability,
   so it follows the `SUB0_BUILD_SPIKE_TESTS` pattern rather than a configurator `constexpr`. When ON,
   it `FetchContent`s both repositories at the full SHAs above and builds a separate test target; no
   existing target changes. Integration pitfalls found while reading the lower `CMakeLists.txt` files:
   - When Sub0TieredCache is a subproject, its `CMakeLists.txt` does **not** fetch or link Sub0MemPage.
     Sub0Llm must bring in Sub0MemPage itself at `213acdd` and link `Sub0MemPage::Sub0MemPage`.
   - Set `SUB0MEMPAGE_BUILD_TESTING=OFF` and `SUB0TIEREDCACHE_BUILD_TESTING=OFF` before
     `FetchContent_MakeAvailable`. Sub0MemPage builds its tests even as a subproject when that option
     is ON.
   - For offline builds, support `FETCHCONTENT_SOURCE_DIR_SUB0MEMPAGE`,
     `FETCHCONTENT_SOURCE_DIR_SUB0TIEREDCACHE` and `FETCHCONTENT_FULLY_DISCONNECTED=ON`.
   - Both libraries are C++23 header-only interface targets. On Windows, Sub0TieredCache links
     `ws2_32`.
6. **Gates** (AGENTS.md §4, §10, §12):
   - Rebuild the unmodified default configuration and confirm `sub0_tests` and `sub0_frontend_tests`
     assertion/case counts are identical. The latest recorded baselines are 29,510,661/147 and
     208,910/293 (ACTIVE_WORK_LOG O5 2b-3). Re-record them on the local machine before the change.
   - Run the new target with the toggle ON on Windows and Linux.
   - Run a `cpp-review` pass over the diff.
   - Write an S1 acceptance manifest recording the three SHAs, contract S1, fixture hash, compilers,
     OS and test counts.
