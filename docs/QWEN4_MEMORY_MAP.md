# Qwen4 real-48-layer decode: every allocation, mapped

**Purpose.** A complete, ground-truth inventory of what the gen tool (`sub0llm-qwen4-gen`) actually
allocates at the real 48-layer Qwen4 axes — what each thing is, why it exists, WHEN it is created in the
process's lifetime, WHERE it lives (private committed heap vs. reserved-then-faulted mapped file,
thread-local vs. process-shared, one-time vs. per-thread), and WHAT DTYPE it holds (§7 — every area's
actual numeric type, checked against source rather than assumed, and whether a smaller one would help).
Requested directly: "make sure every allocation is mapped out and documented for the need - timing -
positioning" and, in a follow-up pass, "check the types used for the different areas... should we use more
quantized or smaller types." Every number below is either taken
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

| Bucket | Size | Dtype in memory | On disk | Backing | Resident when |
|---|---:|---|---|---|---|
| **Backbone** (`g_param_data`) | 18.310 GiB | **F32** (`MASTER_DTYPE`) | F32 (the `.bin` checkpoint's own S0L5 format) | private heap, fully committed | 100% resident, always, from `load_model` onward |
| **Worker** (one instance) | 7.023 GiB | **F32** activations (no reduced-precision CPU path exists — §8) | n/a (never persisted) | private heap, fully committed | 100% resident, always, from first `ensure_thread_built()` onward |
| **Sidecar** (`.moeq` mapping) | 37.113 GiB committed-as-reserved, but only **4.8–7.6 GiB actually resident** (measured, varies) | **F32 once resolved into a pool slot** (§3b) — the ONLY area in this whole process with real at-rest compression | **mixed GGUF quant: IQ1_S/IQ2_XXS/IQ4_NL, ~2.64 bits/weight blended average** (§8) | mapped file, reserve-then-fault | partially and continuously resident; never all of it at once, by construction (§6) |
| Decode per-thread pools + persistent caches | ~0.43 GiB | **F32** throughout | n/a | private heap | resident once decode's first token runs |
| Everything else (tokenizer, CRT heap, stacks, DLL images) | ~0.07 GiB | mixed (tokenizer tables are UTF-8 text + `int32`; images are machine code) | n/a | private heap + mapped image | resident from early startup |

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
`NUM_PARAMS = 1074` (the tensor/`Node` count this array is sliced into). **This F32 array is this
project's own checkpoint format, not a description of the original source model** — the real downloaded
Qwen3.8-Flash-Next file is genuinely quantized for these same tensors (`Q5_K`/`Q6_K`/`Q8_0`, mixed per
tensor role); §7d has the full trace of where that precision goes and when.

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

| Field | Dtype | Size | Why it exists | Why this size |
|---|---|---:|---|---|
| `act_data` (`std::array<float, ACT_CAP>`) | **F32** — CPU backend has no other option (§8) | **7.022 GiB** | the batched forward's activation arena — every intermediate tensor a `forward()` call (not `forward_one`) produces | `ACT_CAP` = a `consteval` worst-case sum over every op this config could execute (`calc_act_cap()`, `internal.hpp:81`), `* 3/2 + 8192` headroom |
| `act_grad` (`std::array<float, ACT_GRAD_FLOATS>`) | F32 (moot — dead) | **1 float** (was 7.022 GiB) | the matching backward-pass gradient arena | `ACT_GRAD_FLOATS = FORWARD_ONLY ? 1 : ACT_CAP` — **provably dead** in this build: `backward_node` `abort()`s for Gated Residual/MoE/QSA before ever writing here (the WP6a fix, merged) |
| `grad` (`std::array<float, WORKER_GRAD_FLOATS>`) | F32 (moot — dead) | 1 float | this thread's per-parameter gradient accumulator | `WORKER_GRAD_FLOATS = FORWARD_ONLY ? 1 : PARAM_FLOATS` — same dead-in-this-build proof as `act_grad` |
| `param_nodes` (`std::array<Node, NUM_PARAMS>`) | struct (spans + metadata, not tensor data itself) | ~129 KiB | one `Node` per backbone parameter tensor, `data`/`grad` spans into `g_param_data`/`grad` | 1074 × 120 B (measured `sizeof(Node)`) |
| `pool` (`std::array<Node, MAX_NODES>`) | struct | ~169 KiB | scratch for every intermediate node a `forward()` call constructs | 1410 × 120 B (`MAX_NODES` derived the same way `ACT_CAP` is) |
| `views` (`std::array<ParamView, NUM_PARAMS>`) | `{size_t off, n; bool decay}` | ~26 KiB | optimizer (AdamW) per-parameter offset/decay-flag table | 1074 × 24 B |
| `moe_cache` (`MoeExpertCache`, 8 slots) | **F32** (dequantized output — the source planes are quantized, §3b/§8) | **0 B allocated**, ~0.13 KiB struct overhead | the BATCHED path's (`op_moe`) dequantize-on-demand pool, real cross-row hit rate across a window's T rows | lazily heap-allocated on first `.allocate()` call — `op_moe` is the ONLY caller, and a `forward_one`-only decode/gen run **never calls `op_moe` at all**, so this 150 MiB pool costs exactly zero bytes in the gen tool. It only materializes in a `--verify`/training/eval run that exercises the batched forward path. |

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

| Pool | Slots | Dtype | Size | Used by | Timing | Real hit rate |
|---|---:|---|---:|---|---|---|
| `MoeExpertCache` (`Worker::moe_cache`) | 8 | **F32** — always, regardless of the source plane's on-disk format (§3c step 4 dequantizes unconditionally) | 150.0 MiB | `op_moe`, the BATCHED forward path only | lazily allocated on first `.allocate()` — **never** in a pure gen/decode run (§2) | real: many rows of one window's T rows re-select the same expert |
| `MoeDecodeExpertCache` (one per decode thread, `MoeDecodeThread::cache`) | 1 | **F32** (pool) + **F32** (`raw_scratch_` — source-order intermediate before the transpose, also full-width, not the on-disk quant format) | 18.75 MiB pool + 6.25 MiB `raw_scratch_` = 25.0 MiB **per thread** | `forward_one`'s decode path (`ParallelExperts`, B20 part 2) | lazily allocated the first time each of the `MOE_DECODE_THREADS` (10, at these axes) OpenMP threads enters the resolve region — i.e., during the FIRST token's FIRST layer, not at startup | provably zero by construction: a token's top-10 experts are distinct indices, the key includes the layer, and 480 resolves/token round-robin any 1-slot pool clean well before the next token asks |

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

| Cache | Dtype | Size (measured) | What it holds | Growth pattern |
|---|---|---:|---|---|
| `KVCache` | `std::vector<float>` — **F32** | 24.0 MiB | every execution's K/V history, `[LOOP_EXEC_COUNT][SEQ_LEN][D_KV]` for k and v separately (`LOOP_EXEC_COUNT = N_LAYERS = 48` here, no LoopSplit) | fixed-size on `reset()`, written in-place per position, no further allocation |
| `GdnCache` | **F32** | 149.6 MiB | Gated DeltaNet's recurrent state + conv history — an ACCUMULATOR, not a per-position row store, so it does not grow with position at all | fixed-size on `reset()`, **unconditionally re-zeroed every reset** (unlike KVCache's assign-on-size-change) since a stale accumulator would leak a previous generation's state |
| `QsaCache` | **F32** (`raw_k`/`block_k`); `n_cached` is `std::vector<int>` (32-bit, negligible) | 3.75 MiB (3.0 raw keys + 0.75 pooled block-key cache) | the QSA indexer's raw, unnormed keys (grows with position, real per-position store) plus a pooled block-key cache (O(T/ratio), not O(T²) — the whole reason the pooled cache exists) | fixed-size on `reset()`; `n_cached` counts unconditionally reset (a stale nonzero count would wrongly trust a previous generation's pooled blocks) |

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

## 7. Types used per area, and whether smaller ones would help

Checked directly against the source rather than assumed, because the answer turned out to be more
lopsided than "everything could use a bit of quantization" — **exactly one area in this entire process has
any real at-rest compression today, and it is not the one carrying the most bytes.**

### 7a. The dtype landscape, area by area

| Area | Dtype today | Reduced-precision path exists? | Where |
|---|---|---|---|
| Backbone (`g_param_data`) | **F32** | **No — corrected below, §7d.** `MASTER_DTYPE`/`HEAD_DTYPE` are hardcoded `Dtype::F32` by the configurator (`tools/configurator.cpp:1759-1760`) — not even a build-time choice, unlike `GEMM_DTYPE`/`ACT_DTYPE` below. **The SOURCE model is not F32 for these tensors at all** (mixed `Q5_K`/`Q6_K`/`Q8_0`/`F32` — §7d); this project's own offline transplant tool fully dequantizes every backbone tensor to F32, once, before the gen tool ever runs. | ON DISK, in `qwen4_full48_q.bin`: never quantized — but that file is itself the OUTPUT of a one-time dequantization, not the original model (§7d) |
| Worker `act_data`/activations | **F32** | **Only on the CUDA backend.** `GEMM_DTYPE`/`ACT_DTYPE` can bake `Dtype::BF16` (`sub0_config.hpp`, configurator-selected), and `backend.cu` genuinely uses it (`act_t = std::conditional_t<ACT_DTYPE == Dtype::BF16, __nv_bfloat16, float>`, `backend.cu:183`). **The CPU backend never reads `GEMM_DTYPE`/`ACT_DTYPE` at all** — confirmed by grep: those two symbols appear nowhere in `src/backends/cpu/*.cpp` except a comment explaining that the BF16 selection is GPU-only (`backend.cpp:19`). | CUDA only; the CPU path this whole document is about has no reduced-precision activation storage, period |
| MoE experts, **on disk / in the mapping** | **Mixed GGUF quant: IQ1_S, IQ2_XXS, IQ4_NL** (§7b) | **Yes — this is the one place it already happened**, and it's why the sidecar is 37.11 GiB instead of the 450 GiB the same experts would be at F32 | `moe_quant.hpp`/`gguf.hpp`, WP4e |
| MoE experts, **once resolved into a pool slot** | **F32**, unconditionally | No — `dequantize_expert` always produces F32 output regardless of the source format, because `expert_ffn_row` is an F32 compute kernel | `moe_quant.hpp::dequantize_expert`, `moe_math.hpp::expert_ffn_row` |
| KV / GDN / QSA decode caches | **F32** | No — plain `std::vector<float>` throughout | `decode.cpp` |
| Backbone weights under `USE_TERNARY` | **Still F32** | **This is the counter-intuitive finding of this pass.** `USE_TERNARY` sounds like a storage format but is not one: `ternarize_into(std::span<const float> w, std::span<float> q)` (`backend.cpp:276`) takes F32 in and produces F32 out — it restricts VALUES to a ternary domain (BitNet-style straight-through estimator, for training) and re-derives them on every `op_linear` call from a still-fully-F32-sized `g_param_data`; it does not shrink the array. `USE_TERNARY` is also `false` in this build. | `backend.cpp:213` itself documents the re-quantization hazard ("absmean re-quantization is not idempotent") that this scheme is built around, which is itself evidence it's a compute-path restriction, not a storage format |
| Node graph metadata (`Node`, `ParamView`) | structs, not tensor dtype | n/a | these are bookkeeping, not weights |

### 7b. The sidecar's real, measured compression — the one genuine data point in this area

Computed directly from `gguf.hpp`'s own `block_spec()` table (bytes per block, elements per block) and the
real plane-format census already taken this session (`IQ1_S` 34,816 planes, `IQ2_XXS` 14,336 planes,
`IQ4_NL` 24,576 planes, out of 73,728 total — every plane is `D_MODEL × D_FF` = 1,638,400 elements):

| Format | Block shape | Bits/element | Planes | Share of sidecar bytes |
|---|---|---:|---:|---:|
| `IQ1_S` | 50 B / 256 elements | **1.5625** | 34,816 | ~27.9% |
| `IQ2_XXS` | 66 B / 256 elements | **2.0625** | 14,336 | ~15.2% |
| `IQ4_NL` | 18 B / 32 elements | **4.5** | 24,576 | ~56.9% |
| **Blended average** | — | **~2.64** | 73,728 | — |

That blended 2.64 bits/element against F32's 32 bits/element is exactly the **12.1×** compression that
makes 37.11 GiB out of what would otherwise be ~450 GiB — already the single biggest memory decision in
this whole system, made once, at WP4e, and it is the reason a sidecar exists as a mapping at all rather
than the question being moot.

### 7c. So — should other areas use smaller types too? Ranked by real leverage

**Yes, and the backbone is the obvious next target, by a wide margin** — it is F32, it is the
second-largest resident item (18.31 GiB, only 7.02 GiB behind the whole reason this document's §0
arithmetic is tight), and unlike the sidecar it currently has **zero** reduced-precision path of any kind,
not even the GPU-only one `act_data` at least has. Ranked by expected memory win against implementation
risk/cost, most to least attractive:

1. **Backbone → BF16 storage (not GGUF-style variable quant).** Halves 18.31 GiB → ~9.15 GiB. This is the
   single highest-leverage, lowest-risk lever available: BF16 shares F32's exponent range (so weight
   magnitudes that already trained fine in F32 don't need re-calibration the way an INT format would), and
   — per §7d — **the hard part is already built and already exercised**: `sub0llm-transplant` already
   dequantizes every backbone tensor's real source format (`Q5_K`/`Q6_K`/`Q8_0`/native `BF16`/F32, mixed
   per tensor role) via `gguf::to_f32`; adding a BF16 OUTPUT path would only need a round/pack step added
   at the point that tool already writes bytes, not a new decoder for anything. Critically, unlike the
   sidecar's per-token sparse access pattern, the backbone is touched **densely, every layer, every
   token**, so a real GGUF-style variable-bit-rate format (needing per-block dequant work on every read)
   would add real, unavoidable CPU cost to the hottest, most-frequently-touched weights in the whole
   system — BF16 sidesteps that: a BF16→F32 promote is a single shift, cheap enough to not be a real tax
   on a dense, every-token path the way format-specific block dequant would be.
   **Directly reopens B23's own arithmetic**: freeing ~9.15 GiB of backbone headroom would move this
   machine's sidecar-residency ceiling from ~47–59% (§0) toward roughly 68–83% of the sidecar simultaneously
   resident — the single biggest lever available anywhere in this document, bigger than anything Sub0MemPage
   itself can buy through scheduling alone, because it changes the ceiling, not just how efficiently the
   existing ceiling is used.
2. **Worker `act_data` → BF16, following the CUDA backend's own already-validated precedent.** Halves 7.02
   GiB → ~3.5 GiB. Lower priority than the backbone only because it's a smaller absolute number, but it is
   genuinely the SAME change the CUDA backend already made and presumably already validated — porting an
   existing, proven pattern to the CPU backend, not inventing a new one. Also frees real headroom for the
   sidecar (§0), same mechanism as point 1.
3. **KV/GDN/QSA caches → FP16 or BF16.** Halves 177.4 MiB → ~89 MiB. Small in absolute terms (this is the
   smallest of the three levers) but it is the most standard, most battle-tested technique of the three —
   FP16 KV-cache is close to the default assumption in most modern LLM-serving engines — and the lowest
   correctness risk, since these values pass through comparatively few subsequent operations before being
   consumed.
4. **MoE resolve pools → store the dequantized slot in BF16 instead of F32.** Modest (~75–125 MiB across
   the batched + decode pools combined) — lowest priority of the four, both because the absolute bytes are
   small and because it sits directly in the hottest per-token compute path (`expert_ffn_row`), so any
   promote/demote cost here is paid far more often, per byte saved, than points 1–3.

**What this review deliberately does NOT recommend**: extending the sidecar's own GGUF-style variable-bit
quantization to the backbone. The sidecar's format was the right choice *because* only ~10 of 512 experts
per layer are touched per token — the per-block dequant cost is paid rarely, amortized over a huge unused
majority. The backbone has no such sparsity to hide behind; every weight is read every token, so a format
needing real per-read dequant work (rather than BF16's near-free promote) would trade a memory win for a
recurring, dense CPU cost this document has no measurement to justify yet. If this direction is pursued,
say so as an explicit follow-up experiment, not assumed free — the same "measure before spending it"
discipline `docs/MEMORY_AUDIT.md`'s own §4 lever table already applies to a different backend's bf16
gradient-accumulator question.

### 7d. Correction — the backbone is not "never quantized," it is "already dequantized, once, offline"

**Directly asked, worth stating precisely rather than leaving §7a's table row to be misread.** The
backbone being F32 in `g_param_data`/on disk in `qwen4_full48_q.bin` does NOT mean the real source model
is F32 for these tensors, and it does NOT mean `load_model` does any dequantizing at runtime — both would
be wrong readings of §7a as originally stated. What actually happens:

1. **The real, downloaded Qwen3.8-Flash-Next GGUF file is quantized for backbone tensors too, not only the
   routed experts.** Per `docs/WP4_SCOPE.md`'s own real per-tensor dump (a full read of `blk.0.*`/`blk.23.*`,
   a GDN and a QSA layer, `type_raw` read directly from the file, never assumed from a tensor's name/role):
   GDN's `in_proj_qkv` is `Q5_K`, GDN's `out_proj` is `Q6_K`, QSA's `q_proj`/`k_proj`/`v_proj` are `Q5_K`,
   Gated Residual's `down`/`up` are `Q8_0` — only small vectors (norms, `q_norm`/`k_norm`, GR's own
   `norm`/`inject`) are natively `F32`. This is a real, deliberate, non-uniform mix — "unsloth's own
   `Dynamic` (`UD`) per-layer importance-based mixed quantization," per that document's own words.
2. **`sub0llm-transplant` (`tools/sub0llm-transplant.cpp`, WP4c) is a separate, OFFLINE tool** — real GGUF
   file in, this project's own `S0L5` checkpoint out — run once, ahead of time, to PRODUCE
   `qwen4_full48_q.bin`. For every backbone tensor, whatever its native format, it reads that tensor's own
   `type_raw` and calls `gguf::to_f32(slice, raw, out)` (line 264 of that file) — the exact same generic
   dequantize dispatcher the sidecar's own `dequantize_expert` uses — and writes the F32 result into the
   checkpoint.
3. **The 512 routed experts per layer are the one deliberate, documented exception** (WP4e,
   `SUB0_MOE_QUANT_EXPERTS`): transplant does NOT dequantize those into the main blob at all — it copies
   them still in their native GGUF-quantized bytes into the separate `.moeq` sidecar, unchanged. That is
   the ONLY place the source model's own compactness survives into what the gen tool actually loads.
4. **`load_model`'s runtime read (§1) is therefore loading an already-fully-dequantized file.** No
   dequantization of any kind happens when the gen tool starts — it happened once, offline, whenever
   `sub0llm-transplant` was last run to produce this checkpoint, entirely decoupled from every `[mem]`
   number and every timing figure elsewhere in this document.

**Why this strengthens, not just corrects, §7c's own recommendation 1**: a BF16-backbone lever is not
proposing something the pipeline has never done before — it is proposing that the ALREADY-EXISTING,
ALREADY-TESTED `sub0llm-transplant` tool stop fully discarding the source model's own reduced precision at
the exact point it already has every backbone tensor's real value in hand (as the intermediate F32 result
of `gguf::to_f32`, before it's written to disk). The real engineering work this implies is narrower than
"add BF16 support to the engine" — it is "teach one existing offline tool to round its already-computed F32
values down to BF16 before writing them," which is a materially smaller, more contained change than §7c's
original framing implied, and worth stating as such rather than leaving it looking like new infrastructure.

**Correctness discipline, stated because it applies here exactly as everywhere else in this project**: any
of the above is a real numerical change (BF16 has ~8 fewer significant bits than F32), not a free
`sizeof()` relabeling — the same "verify correctness against reference before performance" standard this
whole session's B19/B20/B21 work has followed (parity vs. a reference, the WP5c determinism fixture,
`--verify` against the real artifact) would apply in full before any of this ships, exactly as it did for
every other memory-shape change WP4e/WP6a already made.

## 8. Summary — findings from this review pass specifically

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
7. **The dtype landscape (§7) — one real finding, one counter-intuitive one.** Real: the backbone (18.31
   GiB, the second-largest resident item in the whole process) has literally zero reduced-precision path
   today, not even the GPU-only one activations at least have — `MASTER_DTYPE`/`HEAD_DTYPE` are hardcoded
   F32 by the configurator, not even a build-time choice. Counter-intuitive: `USE_TERNARY`, the one
   quantization-sounding knob this codebase already has, does not shrink `g_param_data` at all — it
   restricts weight VALUES to a ternary domain for training purposes and re-derives them from a still-fully-
   F32 array on every `op_linear` call, never touching storage size. BF16 backbone storage is named as the
   single highest-leverage lever in this whole document — bigger than anything scheduling alone (Sub0MemPage
   included) can buy, because it moves B23's own residency ceiling rather than just using the existing one
   more efficiently — but is explicitly NOT yet implemented, measured, or validated; it's a ranked
   recommendation, not a finding of something already done.
8. **A direct correction, prompted by a user question rather than found unprompted (§7d).** §7a's original
   framing ("backbone: never quantized, never has been") was accurate about `g_param_data`/the on-disk
   checkpoint but misleading about the SOURCE model: the real Qwen3.8-Flash-Next GGUF file genuinely
   quantizes these same tensors (`Q5_K`/`Q6_K`/`Q8_0`, mixed per tensor role, verified against
   `docs/WP4_SCOPE.md`'s own real per-tensor dump), and `sub0llm-transplant` (a separate offline tool)
   already dequantizes every one of them via `gguf::to_f32` before ever writing the checkpoint —
   `load_model` itself does zero dequantization at runtime. This sharpens, not weakens, recommendation 1
   above: a BF16-backbone lever would extend an already-built, already-tested tool's existing dequant step
   by one rounding operation, not build new infrastructure.

Nothing in this pass found a missing multi-GiB allocation, a leak, or a double-count — WP6a's own
`32.835 GiB` private-total figure and the two independent measurement methods that produced it still stand.
This document's contribution is completeness (every allocation named, not just the large ones), lifecycle
framing (when, not just how much), and now the dtype question (what precision, and where the real leverage
for reducing it actually is) — all three explicitly requested, none fully covered before.
