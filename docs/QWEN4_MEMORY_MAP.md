# Qwen4 real-48-layer decode: every allocation, mapped

**Purpose.** A complete, ground-truth inventory of what the gen tool (`sub0llm-qwen4-gen`) actually
allocates at the real 48-layer Qwen4 axes — what each thing is, why it exists, WHEN it is created in the
process's lifetime, and WHERE it lives (private committed heap vs. reserved-then-faulted mapped file,
thread-local vs. process-shared, one-time vs. per-thread). Requested directly: "make sure every allocation
is mapped out and documented for the need - timing - positioning." Every number below is either taken
verbatim from `docs/WP4_SCOPE.md`'s own WP6a audit (independently cross-checked there via two separate
measurement methods agreeing to the byte) or newly derived in this pass and marked **NEW**. Nothing here
is re-estimated from memory of an old summary — every figure was re-read from the current source or
re-derived from the real generated config (`out/build/qwen4gen-vtune/generated/`) during this review.

Companion reading: `docs/WP4_SCOPE.md` §6 (WP6a/WP6b, the original byte-exact audit and the single-expert
cost decomposition), `docs/INDEPENDENT_REVIEW_BACKLOG.md` B20/B21/B23 (the disk-bound finding, the
concurrency-ceiling investigation, and the memory-budget arithmetic that shows the sidecar cannot fully
fit), and `Sub0MemPage/docs/design.md` (where B23's arithmetic now also lives, since it's the concrete
evidence motivating that project).

Axes this document is written against (the real Qwen4-preview build, `out/build/qwen4gen-vtune`):
`D_MODEL=2560, N_LAYERS=48, N_HEADS=24, N_KV_HEADS=2, D_HEAD=256, D_FF=640, VOCAB=248320, SEQ_LEN=128,
NUM_EXPERTS=512, EXPERTS_PER_TOK=10, MOE_QUANT_EXPERTS=true, DEFAULT_THREADS=24, MAX_WORKERS=24`. Machine:
`croglegion`, 63.43 GiB total RAM, Core Ultra 9 275HX (8P+16E).

---

## 0. The picture at a glance

| Bucket | Size | Backing | Resident when |
|---|---:|---|---|
| **Backbone** (`g_param_data`) | 18.310 GiB | private heap, fully committed | 100% resident, always, from `load_model` onward |
| **Worker** (one instance) | 7.023 GiB | private heap, fully committed | 100% resident, always, from first `ensure_thread_built()` onward |
| **Sidecar** (`.moeq` mapping) | 37.113 GiB committed-as-reserved, but only **4.8–7.6 GiB actually resident** (measured, varies) | mapped file, reserve-then-fault | partially and continuously resident; never all of it at once, by construction (§6) |
| Decode per-thread pools + persistent caches | ~0.43 GiB | private heap | resident once decode's first token runs |
| Everything else (tokenizer, CRT heap, stacks, DLL images) | ~0.07 GiB | private heap + mapped image | resident from early startup |

**The arithmetic that governs all of it** (B23, `docs/INDEPENDENT_REVIEW_BACKLOG.md`): this machine has
63.43 GiB total RAM, and this is a real, shared development machine, not a dedicated inference box — the
"available before this process starts" figure has been measured at two different values on two different
occasions this session (47.29 GiB in WP6a's own earlier pass, 42.71 GiB in B23's) simply because background
load on a real desktop varies; both are honest measurements, neither is "the" constant, and B23's own text
already flags this. Taking B23's more recent, more conservative figure: ~20.72 GiB already committed to
the OS and other processes at baseline. Backbone + Worker together commit 25.33 GiB the moment the model is
loaded and the first Worker is built — non-negotiable, always resident. That leaves **17.32 GiB of real
headroom for a 37.11 GiB sidecar (or, at WP6a's own less-loaded baseline, closer to 22 GiB / ~59%) — at
most roughly 47–59% of the sidecar can be resident at any one time, on this machine, depending on
background load, and NEVER all of it.** Every other number in this document sits inside that hard ceiling.

---

## 1. The backbone — `g_param_data`

**What it is.** One flat `float[PARAM_FLOATS]` array holding every backbone parameter tensor: token
embeddings, every layer's attention/GDN/Gated-Residual weights, the QSA indexer weights on QSA layers,
`lm_head` (when not tied), and the model-level Gated-Residual exit collapse. **It does NOT hold the 512
routed experts per layer** — those are `MOE_QUANT_EXPERTS`-gated out of `PARAM_LAYOUT` entirely and live in
the sidecar instead (§3). `PARAM_FLOATS = 4,915,107,200` at these axes = exactly **18.3102 GiB**, and
`NUM_PARAMS = 1074` (the tensor/`Node` count this array is sliced into).

**Need.** These weights are read on every single forward step, dense — there is no sparsity to exploit
here, unlike the MoE experts. Full residency is not a choice; it's the only sane placement for something
every token touches in full.

**Timing.** `ensure_shared_params()` (`backend.cpp:109`), called once via `std::call_once`, the first time
any code path needs a parameter node — in the gen tool, this is triggered from inside `load_model`'s own
setup before the read. `std::make_unique<float[]>(PARAM_FLOATS)` runs first (**value-initializes the whole
array to zero** — the code comment already says so explicitly: `// value-initialized -> zeroed`), then
`load_model` (`engine_core.cpp:144`) does one `std::ifstream::read()` call straight into `params_ptr()` for
the full `PARAM_FLOATS * sizeof(float)` span, overwriting every byte just zeroed.

**NEW, this pass — a real, small, previously-unflagged redundancy.** The zero-fill and the immediate full
overwrite are each individually well-motivated on their own (value-initialization is `make_unique<T[]>`'s
documented behavior; a plain streamed read is the simplest correct load path) but **nothing currently
avoids paying for both** — 18.31 GiB gets written twice in a row, once by the allocator's zero-fill and
once by the loader, before a single token is processed. At the measured ~1.05 GB/s effective load rate,
the redundant zero-fill costs on the order of a few seconds of the ~17.4s `load_model: ACCEPTED` time —
not free, but small relative to the whole startup, and genuinely safe to leave alone unless startup latency
becomes a real target: an uninitialized-array variant (`std::make_unique_for_overwrite<float[]>` in C++20,
already the project's own language standard) would remove it, at the cost of the array carrying garbage in
the (extremely unlikely, load-failure-only) code paths that read it before or if the read fails partway.
Named here as a finding, not filed as an action — flag it if startup latency ever becomes a real metric.

**Positioning.** Fully committed private heap memory (`std::unique_ptr<float[]>`, not a container that
could grow). Never touches the mapped-file/paging machinery at all — this is the one big buffer that is
deliberately NOT treated the way the sidecar is, because it must always be fully resident, so there is
nothing "residency management" could usefully do for it. **One instance, process-lifetime, never freed
until process exit.**

**What's conditionally NOT there.** `g_param_grad`, `g_param_m`, `g_param_vel` — the training-only gradient
and AdamW-moment arenas, each also `PARAM_FLOATS` long (43.4 GiB apiece at these axes — 130+ GiB together)
— are gated by `FORWARD_ONLY` (`internal.hpp:204`, true whenever Gated Residual, MoE, or QSA is on, all
three of which this build has). `ensure_shared_params()` simply never allocates them; the WP6a walk
confirmed no such region exists in the live process (not "allocated and empty" — genuinely absent).

---

## 2. The Worker — one instance, 7.023 GiB

**What it is.** All per-thread compute state for the Node-graph (batched) forward/backward path: the
activation arena, the parameter-node table, the forward-graph node pool, the optimizer parameter-view
table, and this thread's own MoE batched-resolve cache. `decode.cpp`'s `forward_one` (the actual hot path
for token generation) does **not** use a Worker's arena at all — it runs one row through stack buffers —
but `Model::build_layout()` still needs one Worker's `param_nodes` to give every parameter `Node` a
`data`/`grad` span, so exactly one gets built regardless of whether decode ever touches its scratch.

**Need, field by field, at these axes** (byte-exact, from WP6a — this document does not re-derive these,
only re-states them with the timing/positioning framing this review adds):

| Field | Size | Why it exists | Why this size |
|---|---:|---|---|
| `act_data` (`std::array<float, ACT_CAP>`) | **7.022 GiB** | the batched forward's activation arena — every intermediate tensor a `forward()` call (not `forward_one`) produces | `ACT_CAP` = a `consteval` worst-case sum over every op this config could execute (`calc_act_cap()`, `internal.hpp:81`), `* 3/2 + 8192` headroom |
| `act_grad` (`std::array<float, ACT_GRAD_FLOATS>`) | **1 float** (was 7.022 GiB) | the matching backward-pass gradient arena | `ACT_GRAD_FLOATS = FORWARD_ONLY ? 1 : ACT_CAP` — **provably dead** in this build: `backward_node` `abort()`s for Gated Residual/MoE/QSA before ever writing here (the WP6a fix, merged) |
| `grad` (`std::array<float, WORKER_GRAD_FLOATS>`) | 1 float | this thread's per-parameter gradient accumulator | `WORKER_GRAD_FLOATS = FORWARD_ONLY ? 1 : PARAM_FLOATS` — same dead-in-this-build proof as `act_grad` |
| `param_nodes` (`std::array<Node, NUM_PARAMS>`) | ~129 KiB | one `Node` per backbone parameter tensor, `data`/`grad` spans into `g_param_data`/`grad` | 1074 × 120 B (measured `sizeof(Node)`) |
| `pool` (`std::array<Node, MAX_NODES>`) | ~169 KiB | scratch for every intermediate node a `forward()` call constructs | 1410 × 120 B (`MAX_NODES` derived the same way `ACT_CAP` is) |
| `views` (`std::array<ParamView, NUM_PARAMS>`) | ~26 KiB | optimizer (AdamW) per-parameter offset/decay-flag table | 1074 × 24 B |
| `moe_cache` (`MoeExpertCache`, 8 slots) | **0 B allocated**, ~0.13 KiB struct overhead | the BATCHED path's (`op_moe`) dequantize-on-demand pool, real cross-row hit rate across a window's T rows | lazily heap-allocated on first `.allocate()` call — `op_moe` is the ONLY caller, and a `forward_one`-only decode/gen run **never calls `op_moe` at all**, so this 150 MiB pool costs exactly zero bytes in the gen tool. It only materializes in a `--verify`/training/eval run that exercises the batched forward path. |

**Sum**: 7.022 GiB (`act_data`) + ~0.31 MiB (everything else) ≈ **7.023 GiB**, matching `sizeof(Worker)`
measured directly (7,540,701,272 bytes).

**Timing.** Lazily built, exactly once, the first time any thread calls `ensure_thread_built()`
(`internal.hpp:627`) — for the gen tool, this is triggered by the `[mem] after graph_reset` step (the
`18.36 → 25.39 GiB` jump in every `[mem]` log this whole session cites is precisely this one allocation).
Decode's own parallel resolve threads (§4) do **not** trigger a second Worker — they deliberately bring
their own much smaller per-thread state instead, exactly because ten Workers (70+ GiB) would not fit.

**Positioning.** One `std::unique_ptr<Worker>` out of a `std::array<..., MAX_WORKERS>` pool (`MAX_WORKERS =
24` here, but only the slots a run actually uses ever allocate — the gen tool uses exactly one). Private,
fully committed heap memory, heap-allocated (not a static array) specifically because a zero-init BSS array
this size would push the DLL's `SizeOfImage` past what the Windows loader accepts
(`STATUS_INVALID_IMAGE_FORMAT`) — the same reason `g_param_data` is heap-allocated rather than `static`.

---

## 3. The sidecar — `moeq::Store` + `ExpertCache`, and why it can never be fully resident

**What it is.** The 512 routed experts × 48 layers × 3 planes (gate/up/down) = 73,728 tensors, in their
native quantized GGUF encoding, in one file (`qwen4_full48_q.bin.moeq`, 37.11 GiB payload) that
`moeq::Store` maps whole and read-only. This is the ONE thing in the whole process explicitly NOT designed
for full residency — the header comment (`file_map.hpp:1-9`) states the arithmetic directly: an eager read
of this alongside the backbone and a Worker "does not fit in this machine's ~48 GiB of free RAM," which is
why it is a mapping rather than an owned buffer at all.

### 3a. `Store`'s own footprint — small, and separate from the mapping's payload

| Item | Size | Backing | Timing |
|---|---:|---|---|
| `Header` struct | 56 B | stack/inline, copied out of the mapping's first bytes | `Store::open`, once |
| `descs_` (`std::vector<Desc>`) | **2.25 MiB** (73,728 × 32 B, `static_assert`ed) | private heap | `Store::open`, once — copied OUT of the mapped header into owned memory so no resolve ever dereferences through the mapping for metadata, only for payload bytes |
| the mapping itself (`FileMap`) | 37.11 GiB **reserved**, resident on demand | mapped file | `Store::open`, once — `CreateFileMapping(PAGE_READONLY)` + `MapViewOfFile` |

**NEW, this pass — precisely what "reserved" means here, stated explicitly rather than assumed.**
`CreateFileMapping`/`MapViewOfFile` (the exact Win32 calls `FileMap::open` makes, `file_map.hpp:75-89`)
reserve address space and associate it with the file; they do **not** commit any physical memory or
pagefile space up front, and — critically, and worth stating because it's easy to conflate with
`g_param_data`'s eager commit — a read-only file-backed mapping's resident pages are never charged against
the Windows page-file commit limit at all, unlike private/anonymous memory. Every byte of the 37.11 GiB
becomes resident only the first time some code actually reads it, one page (page-fault granularity, or a
small cluster — B21's own research measured ~85 KB per hard fault here) at a time — and, just as
importantly, the OS is free to evict any of those resident pages again under memory pressure, silently,
with no notification to this process at all. This is the entire reason B21's disk activity never goes to
zero and B23's warm-up experiment didn't help: there is no code path anywhere that asks Windows to keep a
range resident once faulted in, and per §0's arithmetic, Windows could not honor that request for the full
file even if something did ask.

### 3b. The resolve pools that dequantize experts out of the mapping

Two genuinely different pools exist, deliberately not unified (`internal.hpp:237-285` has the full
reasoning — summarized here with timing/positioning added):

| Pool | Slots | Size | Used by | Timing | Real hit rate |
|---|---:|---:|---|---|---|
| `MoeExpertCache` (`Worker::moe_cache`) | 8 | 150.0 MiB | `op_moe`, the BATCHED forward path only | lazily allocated on first `.allocate()` — **never** in a pure gen/decode run (§2) | real: many rows of one window's T rows re-select the same expert |
| `MoeDecodeExpertCache` (one per decode thread, `MoeDecodeThread::cache`) | 1 | 18.75 MiB pool + 6.25 MiB `raw_scratch_` = 25.0 MiB **per thread** | `forward_one`'s decode path (`ParallelExperts`, B20 part 2) | lazily allocated the first time each of the `MOE_DECODE_THREADS` (10, at these axes) OpenMP threads enters the resolve region — i.e., during the FIRST token's FIRST layer, not at startup | provably zero by construction: a token's top-10 experts are distinct indices, the key includes the layer, and 480 resolves/token round-robin any 1-slot pool clean well before the next token asks |

Decode's ten threads together: **250.0 MiB**, all private heap, all allocated within the first token's
first layer (not at process startup — worth being precise about, since every `[mem]` log line this session
has cited stops at `graph_reset`, before this pool exists at all).

### 3c. The full resolve lifecycle, one expert

1. Router selects 10 experts for this layer (known before any of their bytes are touched — the "declared
   signal" `Sub0MemPage`'s whole design leans on).
2. `moe_resolve` (`internal.hpp:511`) checks the calling thread's own `MoeDecodeExpertCache` for a
   `(layer, expert)` key hit — always a miss in decode, per §3b's own proof.
3. On a miss, `cache.resolve(g_moe_quant, layer, expert)` (`moe_quant.hpp`) reads three `Desc`s out of the
   already-resident `descs_` table (§3a, no I/O), then calls `store.raw(desc)` for each of the three
   planes — this is the moment a page fault can happen, synchronously, on the calling thread, if that
   range of the mapping isn't already resident.
4. `dequantize_expert` decodes the raw quantized bytes (format-specific, `gguf::to_f32`) into the pool
   slot's plane, through `raw_scratch_` as intermediate scratch.
5. `transplant::transpose_out_in` (the B20 cache-blocked transpose) reorders it into the layout
   `expert_ffn_row` expects.
6. `moe::expert_ffn_row` consumes the resolved expert; the slot is then dead until the next resolve
   overwrites it (single-slot pool, so this is unconditional, not policy-driven).

---

## 4. Decode-persistent caches — KV, GDN, QSA

All three are `thread_local`, single instance (the gen tool is single-threaded at the top level; decode's
own `ParallelExperts` fan-out threads share these read/write-free, since only the calling thread's `pos`
advances). All three are **lazily sized on first `reset()`**, called from `kv_reset()`
(`decode.cpp:673`) once at the start of each generation, and held for the generation's whole lifetime —
never resized mid-generation, never freed until the thread exits or a new generation calls `reset()` again.

| Cache | Size (measured) | What it holds | Growth pattern |
|---|---:|---|---|
| `KVCache` | 24.0 MiB | every execution's K/V history, `[LOOP_EXEC_COUNT][SEQ_LEN][D_KV]` for k and v separately (`LOOP_EXEC_COUNT = N_LAYERS = 48` here, no LoopSplit) | fixed-size on `reset()`, written in-place per position, no further allocation |
| `GdnCache` | 149.6 MiB | Gated DeltaNet's recurrent state + conv history — an ACCUMULATOR, not a per-position row store, so it does not grow with position at all | fixed-size on `reset()`, **unconditionally re-zeroed every reset** (unlike KVCache's assign-on-size-change) since a stale accumulator would leak a previous generation's state |
| `QsaCache` | 3.75 MiB (3.0 raw keys + 0.75 pooled block-key cache) | the QSA indexer's raw, unnormed keys (grows with position, real per-position store) plus a pooled block-key cache (O(T/ratio), not O(T²) — the whole reason the pooled cache exists) | fixed-size on `reset()`; `n_cached` counts unconditionally reset (a stale nonzero count would wrongly trust a previous generation's pooled blocks) |

Sum: **177.4 MiB**, matching WP6a exactly.

---

## 5. Everything else — the previously-uncatalogued small items

WP6a's own walk lumped these into one residual line ("everything else private (tokenizer tables, CRT heap,
stacks) ~53 MiB"). **NEW, this pass — broken out individually** rather than left as one unexplained residual:

| Item | Estimated size | Backing | Timing |
|---|---:|---|---|
| Tokenizer tables (`qwen_tok::Tokenizer`) | ~25–35 MiB | private heap | loaded once at startup, before `load_model`, from `vocab.json`/`merges.txt`/`tokenizer_config.json` on `SUB0_QWEN_TOKENIZER_DIR` |
| — `tok_blob_`/`dec_blob_` (byte-alphabet + decoded-byte text, one blob + offsets rather than ~248K separate `std::string`s) | ~2–4 MiB | | |
| — `id_of_` (`unordered_map<string_view, int>`, 248,077 entries) | ~13–14 MiB | | node-based hash map overhead dominates over the (view-only, non-owning) key/value payload itself |
| — `merges_` (`unordered_map<uint64_t, Merge>`, one entry per BPE merge rule, comparable in count to the vocabulary) | ~10–14 MiB | | |
| `sub0llm-qwen4-gen.exe` + `sub0_core.dll` + system DLLs | 0.014 GiB | mapped image (`MEM_IMAGE`) | process/DLL load, before `main()` |
| OS thread stacks (main + up to 10 decode threads, default ~1 MiB each on Windows unless configured) | ~11 MiB | private heap (kernel-managed) | one per thread, as each is created — main at process start, decode threads on first entry into `ParallelExperts` |
| Page-table overhead for the sidecar mapping, once substantially faulted in | on the order of tens of MiB (4-level x86-64 page tables, ~1/512 of resident bytes at the lowest level alone) | kernel memory, not this process's own working set | grows as more of the mapping is faulted in; never directly visible to this process, not part of any `[mem]` line this session has measured |
| `TokMap` (`include/sub0/tokmap.hpp`) | N/A — **out of scope for the gen tool entirely** | | a second, hand-rolled mapping utility that exists for the TRAINING corpus path (`corpus.tok`); the gen tool never opens one. Named here only so a reader searching for "another mapping in this codebase" finds the answer rather than the silence. |

None of these change the picture in §0 — they are collectively well under 100 MiB against multi-GiB
players — but "every allocation" was the ask, and a ~53 MiB residual with no breakdown was the one
remaining unexplained bucket in the whole inventory.

---

## 6. Full lifecycle timeline

1. **Compile time** (`sub0llm-configure`, once, before any binary runs): every axis in the "axes" line
   above is baked as a `constexpr`/`consteval` — `PARAM_FLOATS`, `ACT_CAP`, `MAX_NODES`, `MOE_EXPERT_SLOT_FLOATS`,
   etc. Nothing here costs runtime memory; it determines the SIZES everything below will be.
2. **Process start, before `main()`**: the exe + `sub0_core.dll` + system DLLs map in (~14 MiB, `MEM_IMAGE`).
3. **Tokenizer load** (~25–35 MiB, private heap): `vocab.json`/`merges.txt`/`tokenizer_config.json` parsed
   into `tok_blob_`/`id_of_`/`merges_`.
4. **`load_model`**: `ensure_shared_params()` allocates and zeroes `g_param_data` (18.31 GiB, private,
   fully committed) — `[mem] before load: 0.07 GiB` → then one streamed read overwrites all 18.31 GiB —
   `[mem] after load: 18.36 GiB`.
5. **`moeq::Store::open`**: reads the `.moeq` header + 2.25 MiB descriptor table into private heap, then
   `CreateFileMapping`/`MapViewOfFile` reserves the 37.11 GiB mapping — near-zero committed bytes at this
   point, since nothing has been touched yet.
6. **`graph_reset`** (first `ensure_thread_built()`): the one Worker is built — `act_data` (7.02 GiB) +
   everything else (§2) — `[mem] after graph_reset: 25.39 GiB`. This is the last allocation that happens
   before a single token is generated.
7. **Generation begins, token 1, layer 0**: the FIRST `ParallelExperts` call lazily builds each
   participating decode thread's `MoeDecodeThread` (§3b) — 250.0 MiB total, spread across the first layer's
   worth of resolves, not all at once. `kv_reset()` has already sized KV/GDN/QSA (§4, 177.4 MiB) before
   the first token.
8. **Every subsequent resolve, every token, every layer, for the rest of the run**: no further allocation
   happens anywhere in this list. What DOES happen, continuously, is the sidecar mapping's own pages
   faulting in and being evicted again under the §0 memory-pressure ceiling — the only ongoing "allocation"
   activity in the whole process, and the one this document's whole §3 exists to explain.
9. **Process exit**: everything above is released by the OS; nothing here has an explicit teardown path
   worth documenting (the gen tool is a short-lived CLI process, not a long-running server).

---

## 7. Summary — findings from this review pass specifically

Consolidating what's genuinely **NEW** in this document versus what it faithfully carries forward from
WP6a:

1. **The zero-then-overwrite redundancy in `g_param_data`'s load path** (§1) — real, small (a few seconds
   inside a 17.4s load), safe to leave alone unless startup latency becomes a real target.
2. **`Worker::moe_cache`'s 150 MiB pool costs exactly zero bytes in a pure gen/decode run** (§2) — it is a
   struct member that exists, but its lazy heap buffer is never allocated because `op_moe` (its only
   caller) never runs outside the batched forward path. Worth stating explicitly since every `[mem]` log
   this session has produced could otherwise be misread as including it.
3. **`FileMap`'s reserve-then-fault semantics, and specifically that resident pages never count against
   the pagefile commit limit** (§3a) — stated precisely rather than assumed, and it's the exact mechanism
   B21/B23 already measured the symptoms of.
4. **The tokenizer's own ~25–35 MiB footprint, broken out** (§5) — previously folded into WP6a's
   undifferentiated "~53 MiB everything else" residual.
5. **`TokMap` exists as a second mapping utility but is entirely out of scope for the gen tool** (§5) —
   named so it isn't mistaken for something this document missed.
6. **The full lifecycle ordering** (§6) — WP6a measured a snapshot mid-decode; this document adds the
   step-by-step "what allocates when" a snapshot alone can't show, which is what the user's "timing"
   request specifically asked for.

Nothing in this pass found a missing multi-GiB allocation, a leak, or a double-count — WP6a's own
`32.835 GiB` private-total figure and the two independent measurement methods that produced it still stand.
This document's contribution is completeness (every allocation named, not just the large ones) and
lifecycle framing (when, not just how much) — both explicitly requested, neither fully covered before.
