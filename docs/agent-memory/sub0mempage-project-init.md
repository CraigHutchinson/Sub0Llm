---
name: sub0mempage-project-init
description: "Sub0MemPage (async paging orchestration lib): M2 state machines + dev.py workflow landed 2026-09-25; START from docs/implementation-plan.md M2 checkpoint"
metadata: 
  node_type: memory
  type: project
  originSessionId: 5cbd8e2b-5223-4a4c-ad2b-a3a3e9b3d19d
  modified: 2026-09-10T11:50:43.795Z
---

New repo `github.com/CraigHutchinson/Sub0MemPage`, initialized+pushed 2026-09-10 (spec-only, no
implementation yet, mirroring how Sub0Firn itself started). Local clone: `D:\Craig\GitHub\Sub0MemPage`.

**Scope, decided directly by the user, not inferred**: Sub0MemPage is a general-purpose, LOWER-LEVEL
library for asynchronous, proactive paging orchestration over memory-mapped regions (prefetch
scheduling, working-set budget enforcement, OS mmap/async-I/O mechanics -- Windows
PrefetchVirtualMemory/overlapped I/O/IOCP, Linux io_uring/madvise/posix_fadvise). It owns RESIDENCY,
not addressing and not content. [[sub0log-adoption-backlog]]'s sibling-project pattern applies here too.

**Sub0Firn** (`D:\Craig\GitHub\Sub0Firn`, tiered row-indexed `(table_id,row_index)` table cache) and
**Sub0Llm's own MoE-expert sidecar cache** (`include/sub0/moe_quant.hpp`) are both CONSUMERS of
Sub0MemPage, not the same project. User's own words: "Sub0Firn could be another use-case for the
Sub0MemPage library." Reconciliation doc (`docs/sub0firn-reconciliation.md`) shows Sub0Firn could adopt
it with ZERO changes to its own R1-R10.

**Genesis**: grew directly out of investigating whether Sub0Llm should self-manage proactive MoE-expert
paging instead of relying on 100%-reactive Windows mmap (`FileMap` in `file_map.hpp`). Three parallel
research agents produced: OS-mechanics prior art (Windows has NO async page-fault mechanism -- KB156932;
CIDR 2022 paper measures mmap 2-20x worse than fio under load), empirical on-this-machine measurements
(mmap DOES scale 5-7x with threads in isolation, contradicting the naive "OS serializes it" theory --
the real Sub0Llm decode engine's own bottleneck turned out to be a SEPARATE, still-unexplained defect,
filed as B21 in Sub0Llm's own `docs/INDEPENDENT_REVIEW_BACKLOG.md`), and API-design prior art (CUDA
Unified Memory advise/prefetch, RocksDB async I/O, PostgreSQL Read Stream, DPDK rte_mempool hard-budget
semantics, Redis/Caffeine/MGLRU O(1) eviction bookkeeping).

**Key design conclusion carried into the spec**: prefetch/scheduling is the top lever "by an order of
magnitude" -- the real decode workload measured queue-depth 0.04-0.09 against a demonstrated 6.34 GB/s
device ceiling at queue-depth 9-12. Explicit overlapped I/O beats mmap by only 1.5-2.9x (a real but
secondary effect) -- NOT the primary story.

First build-out was delegated to a background agent that hit a session rate-limit mid-task but had
already written all 17 files to disk uncommitted; recovered by verifying file completeness directly and
committing/pushing manually rather than re-running the agent.

**Next steps, not yet done**: no implementation exists yet (header is a pure skeleton, all calls
commented out). See [[b21-decode-concurrency-plateau]] for B21's own status.

**2026-09-10 — major scope decision, resolved two open questions at once**: user asked directly "do we
want/need OS file paging or do we want to pre-allocate pinned/scratch memory regions... to avoid any and
all disk IO inside the hot loops?" and "should the client app allocate the buffer and mempage just deals
with populating/managing it?" Resolved BOTH as one decision (they turned out to be the same fork from two
angles), pushed as commit `418dc4e`:

(a) **OQ3 resolved**: self-managed pinned scratch + explicit async fill (io_uring/overlapped I/O) is now
the PRIMARY hot-path-safe mode; OS-reactive mmap paging (even `PrefetchVirtualMemory`-hinted) is kept only
as a secondary, opportunistic mode. Decisive evidence beyond PostgreSQL's own 15-year fadvise-to-owned-
buffers migration precedent: B21 itself -- Sub0Llm's real `ParallelExperts` already tried the "many
threads fault a shared mapping" approach and measurably fails to achieve the concurrency an isolated test
achieves on the same file/OS/machine.

(b) **Ownership model resolved: the CALLER owns and allocates ALL destination memory; Sub0MemPage never
allocates bulk data, only small bookkeeping metadata.** Extended prior-art research (fresh WebFetch pulls,
all High confidence): io_uring registered buffers ("the application always supplies the memory pointer"),
POSIX `aio_read` (caller-owned `aio_buf`), NVIDIA GPUDirect Storage `cuFile` ("cuFile relies on users to
complete their own allocation"), Microsoft DirectStorage ("the hardware will write directly into the
buffer that's provided by the title", explicit warning against per-request allocation) -- every real
storage/transport precedent takes this shape. **Not new design** -- it's the general form of code Sub0Llm's
own `ExpertCache<Slots, SlotFloats>` already independently wrote. New `register_slots(region, slot_bytes,
num_slots, slots_ptr) -> pool_handle` call registers the caller's own pre-allocated array; `resolve` now
returns leases over caller-owned slots, not library-owned pointers -- and this makes Sub0MemPage's
`resolve` literally the same shape Sub0Firn's own `resolve_into` already has, strengthening (not weakening)
the Sub0Firn-reconciliation argument.

All 7 core docs updated to match (README/REQUIREMENTS/design.md/prior-art.md/sub0firn-reconciliation.md/
sub0llm-consumer-trace.md/the skeleton header) -- REQUIREMENTS.md gained R14, revised R1/R2/R7/R8/R13.

**Consolidation review, same day**: found and fixed a real defect from the ownership-model pass -- R14
(the new caller-owned-storage requirement) had landed physically after R8 but kept the number "14",
leaving the doc reading R1..R8,R14,R9..R13 out of order. Renumbered sequentially (new req is now R9, old
R9-R13 shifted to R10-R14); fixed every cross-reference across all 7 docs + AGENTS.md/STYLE_GUIDE.md
individually (NOT a blind find-replace -- several mentions are Sub0Firn's own EXTERNAL R-numbers that
must not be touched). Also found design.md sec 1/3/4 had never been updated for sec 8's ownership
resolution (still stated the universal "always legally readable" guarantee and pure OS-eviction framing)
-- fixed to match README's already-corrected sec 1.

**Sub0Llm backlog migration item filed**: B22 in `Sub0Llm/docs/INDEPENDENT_REVIEW_BACKLOG.md` -- migrate
`moeq::Store`/`ExpertCache` onto Sub0MemPage once it has a real implementation. Backlog placeholder, not
urgent (Sub0MemPage is design-only), filed so it isn't forgotten. Checked for other FileMap consumers in
Sub0Llm needing the same treatment -- none found; `ExpertCache`/`Store` is the only match.

**B23 spike, same day — cheap alternative tested and ruled out by arithmetic, feeds directly into design**:
user asked to spike Sub0Llm-side experiments before committing further to Sub0MemPage's full build, and
floated: what if a naive full warm-up of the sidecar at startup is enough (checking for real swap if it
isn't)? Built a standalone probe (`warm_sidecar.cpp`, session scratchpad only, uses `sub0::FileMap`
unmodified), touched the whole 37.11 GiB sidecar once (27.3s @ 1459 MB/s -- itself useful: sequential
single-threaded readahead beats decode's own fragmented small-plane pattern by ~10x), then ran real decode
immediately after. **Result: no improvement at all** (throughput unchanged, disk queue depth unchanged or
worse) **and no real pagefile swap either** (`\Memory\Pages Output/sec` = 0.00 throughout, ruling out
classic swap thrashing). Root cause measured directly: available memory before model load was 42.71 GiB
of 63.43 GiB total (this real dev machine already commits ~20.72 GiB to the OS/other processes at
baseline); backbone+worker commit 25.39 GiB; only 17.32 GiB of headroom remains for the 37.11 GiB sidecar
-- **at most 46.7% of it can be simultaneously resident, full stop, regardless of caching cleverness.**
Filed as B23 in `Sub0Llm/docs/INDEPENDENT_REVIEW_BACKLOG.md` and fed into Sub0MemPage's own `design.md`
§6a: sharpens the whole project's target from "eliminate disk I/O" (shown structurally impossible here)
to "efficiently overlap the unavoidable I/O with compute."

**Multi-agent coordination note, discovered mid-session**: Sub0MemPage is no longer solely mine -- Codex
is actively working on it too (a `feature/usm-plan-groundwork` local branch, "Plan reconciliation; optional
tools/intel_usm" per their own `docs/ACTIVE_WORK_LOG.md` entry, which they introduced -- that file does
NOT exist on `main` yet). Their branch already picked up and acted on the heterogeneous-scope direction
(Intel USM inventory tooling, `docs/intel-usm/`, a new `implementation-plan.md`, status moved to
"IMPLEMENTATION STARTED"). When pushing new findings to `main`, check `git branch -a` first and use
`git cherry-pick` (not a same-branch commit) if accidentally on their branch -- this happened once this
session, resolved by cherry-picking the clean docs-only change onto `main` and resetting their branch back
to its own tip untouched.

**Future scope recorded, not yet designed, same day**: user raised whether Sub0MemPage should extend to
heterogeneous CPU/GPU/iGPU memory backends (and whether the project needs a rename). Recommended: keep
the name ("page" is the literal common unit across CPU virtual memory, CUDA UM, and iGPU/USM pages --
broader scope makes it MORE apt); yes to the scope extension as a documented future direction (not
designed yet) -- grounded in real, not speculative, need: `Sub0Llm/docs/INTEL_IGPU_USM_CAPABILITY_SPIKE.md`
is live research whose `prepare_for_device_copy`/`release_from_device_copy` pair already reinvents part
of Sub0MemPage's own resolve/release vocabulary independently, and CUDA's `cudaMemAdvise`/
`cudaMemPrefetchAsync` is already Sub0MemPage's #1 cited precedent. Recorded as design.md sec 9 + a README
non-goals pointer -- explicitly no new API surface or GPU-specific research done yet; that's the next
step if pursued (would mirror the original 3-agent parallel research pass sec 8 itself came from).

**2026-09-25 — M2 draft + codified workflow LANDED on main (2275463, 2340409), pushed.** Next session:
read `docs/implementation-plan.md` "Checkpoint: M2 draft" first; it has the ordered to-do (bench --aa noise
floor, contended try_resolve mutex lead, QEMU aarch64 stage, MSVC ASan, CI TSan, M3 file transport,
decide Codex's unmerged `feature/usm-plan-groundwork` — pushed as-is, NOT merged, would regress main).
Loop = `python scripts/dev.py check` (MSVC + WSL GCC, ASan/UBSan, TSan, 11 mutants); `dev.py bench` built
but never run end-to-end. Gotchas: invoke WSL with `wsl -d Ubuntu-24.04 --exec ...` (plain `wsl -- bash -c`
re-expands `$vars`); lldb on this host is broken (missing python311.dll) — use WSL ASan for crashes.


**2026-09-29 — S1b DONE: MemPage is Sub0Llm's first production consumer** (Sub0Llm `baf33b9`/`e281b66`,
local, unpushed). Opt-in `--moe-io-mode mempage` routes decode's expert-plane reads through
`moeio::MemPagePlaneIo` (MemPage pin `213acdd`). Byte-identical logits vs reactive/pipelined. Cold:
0.241 s/token vs IOCP 0.267 vs reactive mmap 0.396 (fastest cold reader). Warm: reactive 0.115 stays
default, mempage 0.145 (user's view: a strong baseline; next target is closing the warm gap, likely
per-claim mutex/cv completions + the staging copy). Reader depth matters: 2 workers = no cold benefit;
8 and 30 tie at the median, 30 noisier, so one reader per selected expert. Full record:
`Sub0Llm/docs/STORAGE_STACK_PLAN.md` "S1b closure record". Next: S1c (E1 frozen-table model wiring).
