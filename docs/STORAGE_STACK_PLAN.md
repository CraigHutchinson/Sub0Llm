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

Updated 2026-09-29. **S1a (E1 adapter fixture) and S1b (E2 routed-MoE byte transport) are done; S1c
(E1 frozen-table model wiring) remains** -- see "S1b closure record" and "S1 acceptance manifest (Llm
side)" below. Llm's first production consumer of Sub0MemPage is the opt-in decode transport
`--moe-io-mode mempage` (`include/sub0/moe_io_mempage.hpp`); CMake fetches and links MemPage only for a
build configured that way. The E1 TieredCache adapter is still test-only behind the default-OFF
`SUB0_STORAGE_TIEREDCACHE` option. The default build depends on neither library. This plan authorizes staged
design/implementation work. It does not claim that NVIDIA GDS, a CUDA row cache or the end-to-end
stack is available.

### Core use case: the n-gram table, which never fits (user direction, 2026-09-29)

The stack serves two workload regimes:

1. **Everything fits in RAM.** The OS page cache already does the job, and reactive mmap is the right
   transport. An owned cache is not needed there; it only needs to be close enough.
2. **The working set exceeds RAM.** A generic OS cache thrashes. MemPage owns the cache, bypassing OS
   paging, so it can prefetch, pin and evict for the specific workload.

MemPage targets regime 2. **Its core consumer is the real n-gram embedding table (E1)**: 102.4 GB bf16
(`docs/NGRAM_TABLE_TIERED_STORAGE.md`) against 63 GB of RAM, touched as 16 hash-scattered 320-byte rows
per token. It is regime 2 by construction. It is also the worst case for OS paging: a 4 KB fault per
320-byte row is about 12x read amplification, and the page cache fills with neighbouring rows nobody
asked for. That makes it the design driver for TieredCache (rows) over MemPage (owned slots and fills).
Routed-MoE experts (E2, S1b) are the second consumer of the same substrate.

Owned-cache capabilities should be designed for E1's rows first and then reused for E2's variable
expert extents, never as a MoE-only special case. These are extent- or row-keyed slots, unbuffered
aligned fills (no double caching through the OS), and access-aware admission and prefetch. Benchmarks
for these features must run in regime 2: long decodes, or tables larger than RAM. A 6-token rerun of a
cached prompt measures regime 1.

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
| S0 contract/fixtures | **Lower layers done**: MemPage M2 plus the fake backend, and the TieredCache T0 oracle. Llm E1 is implemented in S1a; E2–E5 fixtures remain to be written. |
| S1 local CPU vertical slice | **Partial: S1a E1 fixture and S1b E2 transport done; S1c open.** S1b: see "S1b closure record" below. S1a: MemPage M3 slice 1 and TieredCache T1 run on real MemPage at the pins above; Llm's E1 fixture, adapter (`include/sub0/ngram_tiered_storage.hpp`) and resident-reference comparison are merged behind `SUB0_STORAGE_TIEREDCACHE` (default OFF). See "S1 acceptance manifest (Llm side)" below. |
| S2–S5 | Not started. T3 remote code exists early; see above. |

### S1b closure record -- DONE 2026-09-29

Opt-in `--moe-io-mode mempage` routes decode's selected-expert plane reads through
`moeio::MemPagePlaneIo`: one `LocalFileBackend` registration plus one explicit-destination
`TransferSet` per staged plane, set up once at load, with one reader per selected expert
(`EXPERTS_PER_TOK`). Claims are retired at the next layer's prefetch, after the OpenMP region has joined
every consumer. No per-token allocation. The reactive default and the pipelined IOCP reader are unchanged.

| Evidence | Result |
|---|---|
| Revisions | Sub0Llm `baf33b9` (local main); runtime MemPage pin `213acdd` (unchanged). Packaging-only lower commits, not pinned: MemPage `ef40370`, TieredCache `53f91e2` (5/5 and 11/11 CTest on Windows Clang and WSL GCC 15). |
| Artifact | `qwen4_full48_q_bf16.bin` + `.moeq` (39,848,247,352 bytes, 73,728 planes, 512 experts, top-10). |
| Output parity | Reactive, pipelined and mempage give byte-identical 6-token `forward_one` logits: SHA-256 `99BFC58A...` with `--backbone-quant-dot 0`, and `D1F29B19...` under the recommended defaults (mempage at both 30 and 10 readers). |
| Adapter suite | `sub0_storage_moe_io_tests`: 642 assertions / 5 cases on Windows Clang and WSL GCC 15, including the real-sidecar case (encoded bytes and dequantized values vs `moeq::Store`). |
| Default build | Neutral d196 suites exact: `sub0_tests` 29,510,661/147, `sub0_frontend_tests` 231,180/304. |
| Review | `cpp-review` pass: no MUST findings. The retire-at-prefetch invariant is documented at its call site. |

Performance (`run_perf_suite.py`, arms built once, rotated, 20 s cooldown, per-sample contention gate;
median s/token over 6 tokens, 8 warm rounds and 4 cold rounds with verified eviction):

| Arm | Warm | Cold |
|---|---:|---:|
| reactive (mmap, default) | 0.115 | 0.396 |
| pipelined (IOCP) | 0.196 | 0.267 |
| mempage | 0.145 | **0.241** |

Readers: depth 2 forfeits the cold benefit (0.404 s/token cold). Depths 8 and 30 tie at the median, and
30 is much noisier, so the implementation uses one reader per selected expert.

**Default decision: unchanged (reactive).** MemPage is the fastest cold-cache reader (-39% vs reactive,
-10% vs IOCP). This host's normal state is warm, though: the 37 GiB sidecar fits in 63 GiB of RAM, and
there reactive is still 26% faster. Closing the warm gap is the next optimization target. Its likely
cost centres are the per-claim mutex/condition-variable completions and the copy into staging that
reactive avoids. A scoped AUTO default, e.g. for hosts whose RAM cannot hold the sidecar, needs evidence
from such a host first.

Not measured: peak memory (staging is `EXPERTS_PER_TOK x 3 x max_desc_bytes`, as for pipelined) and
macOS. Fixed along the way: a quantized-MoE build given `build_model()` without its sidecar now refuses
instead of crashing with SIGSEGV; the 32-bit plane-size check now guards pipelined too; real-axes
layout/tokenizer test defects are fixed; and `run_perf_suite.py` no longer rebuilds before every sample
(thermal confound).

Next: S1c (E1 frozen-table model wiring, below). M4/T2 staged CUDA is S2.

### Owned expert cache (`--moe-io-mode cache`) -- first measurement, 2026-10-01

One Sub0TieredCache row per (layer, expert) holds the expert's three contiguous encoded planes. The rows
are `RowExtent::bounded`, a TieredCache addition at `65e59d5`, because expert sizes differ between
layers. The cache uses caller-owned RAM committed on first touch, with a budget from `--moe-cache-gib`
(default: half of physical RAM). Decode reads planes in place. Fills are buffered for now, because
unbuffered reads plateau on this host: Sub0MemPage `docs/investigations/unbuffered-read-ceiling.md`.

| Evidence | Result |
|---|---|
| Revisions | Sub0Llm `f048990`; Sub0TieredCache `65e59d5` (local, not yet published); Sub0MemPage pin `213acdd` |
| Output parity | 6-token logits byte-identical to reactive (`D1F29B19...`); perplexity 11.930422 across every arm below |
| Suites | `sub0_storage_moe_cache_tests` 124/4 (includes the real sidecar); TieredCache 11/11 on Windows and Linux |

Long decode (2,000 G-PPL tokens) from an evicted cache, 2 rotated rounds:

| Arm | Round 1 | Round 2 | Mean tok/s | Peak working set |
|---|---:|---:|---:|---:|
| cache, 31.7 GiB (auto) | 6.65 | 6.82 | **6.73** | 49.2 GiB |
| reactive | 6.31 | 6.76 | 6.53 | 50.7 GiB |
| cache, 20 GiB | 6.22 | 6.44 | 6.33 | 39.0 GiB |
| mempage | 4.82 | 5.03 | 4.93 | 19.7 GiB |

Reading:
- At equal memory the cache is at parity with reactive, slightly ahead (+3%). That is within run-to-run
  noise; reactive's own two rounds differ by 7%.
- Reactive's speed depends on how much RAM the OS happens to spare. The same arm peaked at 27.5 GiB and
  decoded at 4.93 tok/s in an earlier session.
- The owned budget is fixed and predictable. At 20 GiB it gives up 3% for 11.6 GiB less memory.
- This idle host is only shallowly in regime 2. The deciding test is deep regime 2: hold RAM so only
  ~20 GB is free, then compare at a matched budget. The default stays reactive until that test.

### S1b package definition (historical, as planned)

Refreshed 2026-09-29. The E1 row fixture is complete; the former instructions to implement it
were stale. S1 is **partially complete against its original acceptance**: E2 is absent and E1 model
output parity still needs engine wiring. Keep the historical E1 test evidence below, without treating
17 assertions as end-to-end model acceptance.

The next implementation package is **S1b: route the existing optional CPU MoE pipelined read path
through MemPage**, first proving the adapter independently, then wiring the same adapter into decode.
It needs no GPU, remote mirror or new cache policy. This exercises the real encoded-plane workload
before adding GPU transport to an unconsumed interface.

1. **Publishable baseline.** Llm's S1 merge is local, absent from origin/main. Review and publish the
   intended local commit range through the normal integration workflow before advertising a remote
   reproducible three-repository release. Do not push the entire ahead-of-origin stack merely to publish
   storage work. Lower dependency pins already match their published main branches and need no bump.
2. **Trace current consumers.** Use `moeio::PlaneIo` (`include/sub0/moe_io.hpp`),
   `ParallelExperts::prefetch`, `g_moe_decode_io`, `g_moe_io_stage` and `resolve_from_bytes` in CPU
   decode/internal headers as the wiring points. Check current O8 row-split behavior and native-quant
   consumers before editing: the original 2026-09-25 seam has evolved. Coordinate shared files and CPU
   time in ACTIVE_WORK_LOG first. Preserve the existing reader as the comparison implementation.
3. **Prove the byte adapter.** Llm owns sidecar identity/descriptors, expert selection and plane extents.
   Register a bounded `LocalFileBackend` and explicit-destination `TransferSet` once. Preallocate claims,
   tickets and raw staging for the known selected-expert batch. Retain each claim through its last
   dequantization or native-quant consumer, including parallel workers. Destruction drains before freeing
   buffers/backend. MemPage never writes decoded `ExpertCache` storage. Do not route variable plane
   shapes through TieredCache's fixed-width table contract just to force a three-layer dependency.
4. **Acceptance before wiring.** Deterministic three-plane files plus real S0Q1 sidecar descriptors:
   exact bytes versus independent positional reads; selected order/duplicates; extent/overflow and short
   read failures; queue exhaustion; delayed/out-of-order completion; cancellation/drain; repeated sessions;
   no request-time allocation or reuse before the last reader. Compare decoded and fused outputs with
   current `moeq::Store`/reader oracles. Record real artifact identity/hash; synthetic-only does not close
   interop. Raise a minimal lower test for any missing guarantee before changing the adapter.
5. **Wire and qualify.** Consume the tested adapter in the existing optional pipelined CPU path;
   follow the configurator for any evaluated transport choice and keep the old default until measured.
   No speculative standalone CLI. Run default before/after exact counts, full enabled suite, Windows and
   Linux correctness, real-model output/quality checks and C++ review. Measure current reader versus
   adapter under equal raw/decoded budgets, warm/cold conditions and uncontended trials using the
   optimization protocol. Treat portable workers versus IOCP as an implementation difference to measure,
   not an assumed speedup. Proven recommendations become scoped AUTO defaults per AGENTS.md section 4.
6. **Close the slice explicitly.** Record exact three SHAs, fixture hashes, counts/skips, output parity,
   memory peaks and performance evidence. S1b completion does not imply E1 frozen-table model wiring.
   The latter remains S1c, requiring a real frozen imported-table consumer while mutable tables stay in
   engine arenas. M4/T2 staged CUDA is S2 after CPU consumer lifetime requirements are demonstrated.

TieredCache work for this package is contract regression/feedback, not a redundant MoE wrapper:
re-run its pinned standalone row tests if MemPage semantics change. R18 overlap enforcement must account
for TieredCache's retiring/current bindings sharing storage with exclusive per-range claims; do not add
an allocation-wide rejection that breaks this existing invalidation pattern. A missing lower guarantee
blocks its consumer, and is fixed/tested below before dependency pins advance.

### Remote, branches and evidence snapshot (2026-09-29)

| Repository | Local / published state | PR and CI evidence |
|---|---|---|
| Sub0MemPage | main/origin main `213acdd`; `feature/usm-plan-groundwork` has two unmerged historical commits (`bd1fe75`, `b26b93a`), not an automatic merge candidate | [PR 1 merged](https://github.com/CraigHutchinson/Sub0MemPage/pull/1); [CI green at pinned head](https://github.com/CraigHutchinson/Sub0MemPage/actions/runs/36129269678) |
| Sub0TieredCache | main/origin main `e4da6a7`; cloud branch equals main | [PR 1](https://github.com/CraigHutchinson/Sub0TieredCache/pull/1) and [PR 2](https://github.com/CraigHutchinson/Sub0TieredCache/pull/2) merged; [CI green at pinned head](https://github.com/CraigHutchinson/Sub0TieredCache/actions/runs/36132307886) |
| Sub0Llm | local main `9ac5e05`, 46 commits ahead of origin/main `56ec59f` before this doc update; S1 merge `a320b06`, follow-up `7641aad` included locally | No open PRs or Actions runs returned; S1 evidence below is recorded local validation, not fresh remote CI |

All remotes fetched with prune and open PRs checked again after the session interruption. No open PRs
in any repository. No merge, push, branch deletion or worktree cleanup is part of this coordination pass.
Llm's S1 worktree branch and remote storage handoff branch are already ancestors of local main. O12
also landed locally during this audit. Remaining historical B39/O5/archive branches require their own
owner review; branch divergence alone does not establish missing implementation. The Intel branch
likewise needs a diff/evidence review against current M1/M5 before reuse; it does not block S1b.
Unrelated dirty performance-tool files were present at the resumed snapshot and are excluded.

### S1 acceptance manifest (Llm side) -- DONE 2026-09-26

Historical S1a fixture steps completed; this manifest does not certify the new S1b package above. Adapter: `include/sub0/ngram_tiered_storage.hpp`
(`sub0::storage::NgramTieredAdapter`, isolated -- nothing in `src/`/`tools/` includes it). Test:
`tests/ngram_tiered_storage_tests.cpp` (2 cases, 17 assertions). Build toggle:
`SUB0_STORAGE_TIEREDCACHE` (default OFF), `tests/CMakeLists.txt`.

| Field | Value |
|---|---|
| Contract revision | S1 |
| Sub0Llm SHA (base this manifest was built from) | `0179e9e` (O8 merge on `main`), plus this session's own commits on top |
| Sub0MemPage pin | `213acdd2121cec369c6b9606c514db2231e94f27` |
| Sub0TieredCache pin | `e4da6a7e87bcbc65f79257bef9843d41adbec66d` |
| E1 fixture shape | 24 rows x 160 elements (mirrors the real Qwen4 `head_dim_per_ngram=160` in miniature, `NGRAM_TABLE_TIERED_STORAGE.md` sec 0), source dtype bf16 (2 B/elem), output dtype f32 (4 B/elem, `Representation::bf16_to_f32`), single flat-file shard |
| E1 fixture hash | `930e8b911cdcaaa9ee017ef6f2f4fb01fb0503e01429a374102f3d3b6e2f56d6` (SHA-256 of the 7,680-byte generated file; deterministic from `gen_bits()`'s formula in the test, not a checked-in binary -- reproducible from source, recorded here for the manifest) |
| Compilers | Windows: `clang++` 22.1.6 (LLVM), `-march=native`, `-std=gnu++26`. Linux (WSL): `g++-15` (Ubuntu 15.2.0-14ubuntu1~24~ppa1), `-std=gnu++26` |
| Build tools | CMake 4.2.3, Ninja |
| OS | Windows 11 Home 10.0.26220 (host: Core Ultra 9 275HX); WSL2 Ubuntu 24.04 on the same host |
| Toggle-OFF gate (before, this machine) | `sub0_tests` 29,510,661/147 (fingerprints forward `5a7382ea70d3913b`, grad `7f44bdae18c313dd`, decode `d1625d19ed2258f1`); `sub0_frontend_tests` 228,198/295 |
| Toggle-OFF gate (after, same machine) | Identical: `sub0_tests` 29,510,661/147, same three fingerprints; `sub0_frontend_tests` 228,198/295 |
| Toggle-ON gate, Windows | `sub0_storage_tiered_cache_tests`: 17 assertions / 2 test cases, all passed. `sub0_tests` re-run in the same (toggle-ON) build tree: unchanged, 29,510,661/147, same fingerprints |
| Toggle-ON gate, Linux (WSL/g++-15) | `sub0_storage_tiered_cache_tests`: 17 assertions / 2 test cases, all passed |
| Mutation check | Deliberate off-by-one row offset in `FlatFileResolver`'s `base_offset` (adapter's `create()`) -> rebuilt -> test correctly FAILED (11/12 assertions, 1/2 cases, the out-of-bounds last-row read); reverted -> rebuilt -> back to 17/2 |
| `cpp-review` | One real gap found and fixed (doc-only): `resolve_rows()` silently narrows `Table`'s own thread-safety guarantee (the adapter's `lease_scratch_` has no lock of its own) -- documented with an explicit `@note`; also documented why `table_`'s member-declaration position relative to `backend_`/`resolver_`/the storage vectors is load-bearing for destruction order. No behavior change |
| Boy-scout fix (pre-existing, found while bringing up the Linux run) | Root `CMakeLists.txt` applied Clang's `-fconstexpr-steps` unconditionally under `if(NOT MSVC)`, which fails outright on GCC (`unrecognized command-line option`) for every TU in the tree. Scoped to `$<CXX_COMPILER_ID:Clang>`; verified a no-op for the existing Clang/Windows build (both suites reproduce exactly) and verified positively unblocking the WSL/GCC 15 configure+build |
| Not verified this pass | E2 (routed-MoE byte adapter), model-output parity (E1's second acceptance clause -- needs engine wiring, a later step), macOS, any accelerator path |

Slice status: **S1a fixture done; overall S1 partial**. Next: S1b E2's byte adapter, then S1c engine wiring
(`docs/NGRAM_TABLE_TIERED_STORAGE.md` sec 5) once a real consumer is ready to depend on
`ngram_tiered_storage.hpp` from `src/`.
