# Qwen3.8-Flash-Next (Qwen4-preview) — deployment feasibility on this machine

Status: **RESEARCH + DESIGN ONLY.** No engine code lands this pass, per the task's own explicit
fallback ("design it, leave implementation for later" unless a change is trivial AND safe AND
isolated AND off-by-default — none of the three threads below meet that bar). Follows
`docs/QWEN4_PREVIEW_REFERENCE.md`'s and `docs/NGRAM_TABLE_TIERED_STORAGE.md`'s citation discipline:
every quantitative claim below is either (a) a byte fetched directly from the real, live repository
this pass and computed from first principles, or (b) tagged with a confidence level the way
`QWEN4_PREVIEW_REFERENCE.md`'s facts table already does. This document does not re-derive the
architecture facts already established in `docs/QWEN4_PREVIEW_REFERENCE.md` and `docs/GATED_DELTANET.md`
(48 layers, 36 GDN : 12 QSA, 512+1 experts top-10, 51.2B-param n-gram/PLE table, 125B total / 6B
active) — read those first; this document is the next layer up: **can this machine actually run any
form of this model, and if so, how, and is that worth this project's time relative to what it has
already built.**

**This machine**: 63GB RAM, one RTX 5070 Laptop (~8GB VRAM, sm_120), 589GB free on `D:`, 179GB free on
`C:` (`[[host-cpu-arrow-lake-hx]]`).

**Companion docs, not duplicated here**: `docs/SUB0FIRN_SPEC.md` (the real, spec'd tiered-cache
project for the n-gram table — do not re-litigate its design; §3 below only asks whether a much
smaller, throwaway alternative also has a place) and `docs/NGRAM_TABLE_TIERED_STORAGE.md` (the prior
art and staged plan Sub0Firn implements — §2 below adds llama.cpp-specific and MoE-specific prior art
that doc did not need, since it was scoped to embedding-table serving, not expert-weight serving).

---

## 1. Real quantization sizing — verified against the live repository, not a summarized page

### 1a. Method

Per `AGENTS.md` §5/§9 (verify against the real primary source, not an AI summary of it), the repo file
listing was fetched via the Hugging Face **model API with `?blobs=true`**
(`https://huggingface.co/api/models/unsloth/Qwen3.8-Flash-Next-GGUF?blobs=true`), which returns the
exact byte size and LFS blob hash of every file — not a rendered tree page. A first pass at this task
*did* try the rendered `tree/main` page through a page-to-markdown summarizer first, and it came back
exactly as `AGENTS.md` §5 warns: quantization tags and shard counts were right, but every file size
came back as "not specified in metadata" — the summarizer simply couldn't extract the numbers from the
rendered page. The raw API JSON has them exactly, byte-for-byte.

For the n-gram/PLE table's own internal representation *inside* each `.gguf` file, this document reused
this project's own already-verified GGUF format understanding (`include/sub0/gguf.hpp`'s `Reader`:
magic `"GGUF"` + `u32` version + `u64` tensor_count + `u64` metadata_kv_count, then that many KV pairs
— each a length-prefixed string key, a `u32` type tag, and a typed value, with `Array` requiring an
extra element-type + length before its payload — then that many `TensorInfo` entries — name, `n_dims`,
`dims[]`, `u32` ggml type, `u64` offset) and re-implemented the identical byte-cursor logic in a small
Python script, then fetched only the first few hundred KB to ~256KB of each candidate `.gguf` shard via
**HTTP Range requests** (`Range: bytes=0-N`) against `.../resolve/main/<tier>/<file>.gguf` — the same
technique `docs/QWEN4_PREVIEW_REFERENCE.md`'s Stage 1 already validated against the real safetensors
shards, now confirmed to work identically against GGUF: **zero full-file downloads**, ~1.3MB of HTTP
traffic total across every probe below.

### 1b. Real, byte-exact per-tier totals

| Tier | Total bytes | Decimal GB | Binary GiB | Fits in 63GB RAM? |
|---|---:|---:|---:|---|
| UD-IQ1_S | 72,548,733,472 | 72.55 | 67.56 | **No** — 4.56 GiB over |
| UD-IQ1_M | 74,536,835,616 | 74.54 | 69.42 | No |
| UD-Q2_K_XL | 78,865,128,864 | 78.87 | 73.45 | No |
| UD-IQ3_XXS | 81,957,769,984 | 81.96 | 76.33 | No |
| UD-Q3_K_XL | 89,986,354,624 | 89.99 | 83.81 | No |
| UD-IQ4_XS | 93,682,584,224 | 93.68 | 87.25 | No |
| UD-Q4_K_XL | 111,323,654,464 | 111.33 | 103.69 | No |
| UD-Q5_K_XL | 158,285,806,384 | 158.29 | 147.42 | No |
| UD-Q6_K_XL | 169,165,033,088 | 169.17 | 157.55 | No |
| Q8_0 | 188,228,347,712 | 188.23 | 175.30 | No |
| BF16 | 354,027,133,432 | 354.03 | 329.72 | No |

(Totals exclude the separate `MTP/` 4B-param head files and the two `mmproj-*.gguf` vision-projector
files, which are optional/separate loads, not part of the base text decoder.)

**This confirms the task brief's stated starting point directly, byte-for-byte, not just "at a high
level": every listed quant exceeds 63GB, so nothing fits fully resident regardless of quantization
level.** The gap at the smallest tier is real but not enormous (4.56 GiB) — worth remembering for §4's
recommendation, since "doesn't fit fully resident" and "cannot be touched at all" are different claims
(mmap + OS page cache degrades rather than refuses when a file exceeds RAM — see §4).

### 1c. The n-gram/PLE table's real tensor identity in GGUF

The 51.2B-param n-gram embedding table (`docs/QWEN4_PREVIEW_REFERENCE.md`'s
`Qwen4ExpTextNGramEmbedding`) is exposed in every GGUF conversion as **one single tensor**,
`per_layer_token_embd.weight`, shape `[160, 320001536]` (GGUF's `ne[0]` is the fastest-varying axis,
so this is 320,001,536 rows × 160 columns — exactly the real, already-verified `nn.Embedding` shape) —
**confirmed by direct header parse, not inferred from file size**. The name itself (llama.cpp's own
"per-layer embedding"/PLE convention, borrowed from the Gemma-3n/Qwen3-Next family this mechanism
descends from) is a useful cross-reference: this project's own docs call it "n-gram embedding," the
GGUF ecosystem calls the identical mechanism "PLE" — same 320M-row, 160-dim, hash-addressed table,
different name.

Verified directly, per tier (fetched this pass, `abs_offset`/`type` read from each file's own header):

| Tier | Shard holding the tensor | ggml type | Table-only size | Confidence |
|---|---|---|---:|---|
| BF16 | `BF16-00003-of-00008.gguf` (**entire file is this one tensor**, 102,400,491,712 B) | `BF16` (30) | 102.40 GB | **High — direct header read** |
| Q8_0 | `Q8_0-00003-of-00006.gguf` (**entire file is this one tensor**, 54,400,261,312 B) | `Q8_0` (8) | 54.40 GB | **High — direct header read** |
| UD-Q6_K_XL | `UD-Q6_K_XL-00003-of-00006.gguf` (**entire file**, byte-identical LFS blob to Q8_0's shard 3) | `Q8_0` (8) | 54.40 GB | **High — direct header read + identical blob hash** |
| UD-Q5_K_XL | (byte-identical LFS blob to Q8_0's shard 3, same file role) | `Q8_0` (8) | 54.40 GB | **High — identical blob hash, not independently re-parsed** |
| UD-Q4_K_XL | inside `UD-Q4_K_XL-00002-of-00004.gguf`, alongside other tensors | `IQ4_NL` (20) | 28.80 GB | **High — direct header read** |
| UD-Q2_K_XL | inside `UD-Q2_K_XL-00002-of-00003.gguf` | `IQ4_NL` (20) | 28.80 GB | **High — direct header read** |
| UD-IQ1_M | inside `UD-IQ1_M-00002-of-00003.gguf` | `IQ4_NL` (20) | 28.80 GB | **High — direct header read** |
| UD-IQ1_S | inside `UD-IQ1_S-00002-of-00003.gguf` | `IQ4_NL` (20) | 28.80 GB | **High — direct header read** |
| UD-IQ3_XXS, UD-IQ4_XS, UD-Q3_K_XL | not individually re-probed this pass | presumed `IQ4_NL`, by the exact pattern of every other UD tier above | ~28.80 GB (extrapolated) | Medium — pattern-consistent, not directly re-verified for these three |

(`IQ4_NL` size derived from the format's own definition — 32-element blocks, 4 bits/element + one
16-bit block scale = 18 bytes per 32 elements: `51,200,245,760 × 18/32 = 28,800,138,240` bytes,
matching the observed in-file placement to within header/alignment padding.)

**This directly and conclusively answers the task's question 1(a).** Unsloth's "UD" (Unsloth Dynamic)
quantization does **not** shrink the n-gram/PLE table proportionally to the overall quant label at all
— it clamps the table to one of exactly two fixed floors, `Q8_0` (~8.5 bits/element) for the three
least-aggressive tiers or `IQ4_NL` (~4.5 bits/element) for every tier from UD-Q4_K_XL down through the
"1-bit" `UD-IQ1_S`/`UD-IQ1_M` tiers — while the model's *other* embedding (`token_embd.weight`, the
ordinary 2560-wide vocab embedding) is genuinely let down to the tier's nominal precision (`Q8_0` at
UD-Q4_K_XL, down to `Q4_K` at UD-IQ1_S/UD-IQ1_M). The "1-bit"/"2-bit" label on these tiers describes
the **MoE expert weights** (the actual bulk of the 125B params), not this table. This matches Unsloth's
own publicly-documented "Dynamic Quants" methodology (protecting sensitive/high-impact tensors at
higher precision while quantizing the bulk aggressively) — but this is now a **directly verified fact
about this specific model's specific files**, not an assumption carried over from that methodology's
general reputation.

**A concrete consequence worth stating plainly**: at the smallest real tier (`UD-IQ1_S`, 72.55GB
total), the n-gram/PLE table alone is 28.80GB — **39.7% of the entire download**, for a component that
this engine's own text decoder touches at a sparsity ratio of roughly 5×10⁻⁸ per token position
(`docs/NGRAM_TABLE_TIERED_STORAGE.md` §0). Whatever inference strategy this machine ends up using, this
one tensor dominates both the size problem and the "table row indices are hash-addressed, not
locality-friendly" access-pattern problem `NGRAM_TABLE_TIERED_STORAGE.md` already worked out in detail
for the *safetensors* form of this same table — none of that reasoning changes for the GGUF form; only
the on-disk byte layout does.

### 1d. Can the table be split off / fetched independently, the way the safetensors form's index.json allows?

**Yes, in two different ways depending on tier, both verified concretely this pass:**

- **BF16, Q8_0, UD-Q5_K_XL, UD-Q6_K_XL**: the table is the **entire contents of one whole GGUF shard
  file** (confirmed above — `tensor_count=1` in that file's own header). A caller can simply **not
  download that one file** and get every other tensor in the model from the remaining shards — no byte
  surgery needed at all, this is already as clean a split as the safetensors form's per-tensor shard
  index gives. Concretely: `Q8_0-00003-of-00006.gguf` (54.4GB) is skippable outright, leaving 133.8GB
  for the rest of the Q8_0-quantized model — still far over 63GB RAM on its own, but a real, immediately
  actionable reduction.
- **UD-Q4_K_XL and below (every tier that uses `IQ4_NL` for this tensor)**: the table shares a shard
  file with other tensors, so a whole-file skip is not available — but `gguf.hpp`'s own existing
  offset-computation logic (parse the header, walk the KV section, walk the tensor-info array, each
  `TensorInfo` already carries the exact `offset`/`dims`/`type` needed) computes the tensor's exact byte
  range within that shard with **zero new code**, the same as it already does for any other tensor in a
  file this reader is pointed at. A caller wanting to *exclude* the table at these tiers would need a
  byte-range-aware fetch/skip (download everything in the shard except that one tensor's byte range) —
  more surgery than "don't download this file," but still a computed, not guessed, byte range, using
  logic this project already has.

**A real, concrete gap found while doing this, worth recording as a scoped follow-up rather than fixed
this pass**: `gguf.hpp`'s `TensorType` enum recognizes only `F32`, `F16`, and `Q8_0`
(`include/sub0/gguf.hpp` lines ~38 and ~124-131) — `tensor_byte_size()` returns `0` (its own documented
"unsupported" signal) for `BF16` (30), `IQ4_NL` (20), and every K-quant (`Q4_K`/`Q5_K`/`Q6_K`, 12/13/14)
used above. Concretely, **today's reader can compute the exact byte length of this table only at the
plain `Q8_0` tier** — at `BF16` (despite being an even simpler, non-block format — literally `n × 2`
bytes, the same shape as the already-supported `F16` case) and at every `IQ4_NL`/K-quant tier, it can
locate the tensor's *offset* correctly (that part of the header format doesn't care about the type) but
cannot yet report *how many bytes* to read there or decode them. Adding a `BF16` entry is a one-line,
same-shape addition to an existing `switch`-like dispatch (mirroring `F16`'s own `n * 2`); adding
`IQ4_NL`/K-quant *size* rules is a small, mechanical addition per format (each has a publicly documented
block layout); *decoding* those formats into floats (what `dequantize_q8_0` already does for `Q8_0`) is
separate, larger work this reader has never attempted for anything but `Q8_0` and isn't proposed here.
Not fixed this pass — named because it is the concrete blocker between "this project's tooling can
locate this tensor in the wild" (true today, demonstrated live above) and "this project's tooling can
read its actual values" (true only for the `Q8_0` tier today).

---

## 2. MoE expert placement / GPU-offload prior art

### 2a. `llama.cpp`'s own, real, shipped support — the single most directly relevant option

These GGUF files are built for `llama.cpp` (and forks). Two real, current, documented mechanisms exist
today, verified via `llama.cpp`'s own PR/docs and independent write-ups:

- **`-ot` / `--override-tensor <regex>=<device>`** (landed via
  [ggml-org/llama.cpp#11397](https://github.com/ggml-org/llama.cpp/pull/11397)): overrides which
  device buffer a tensor is allocated on, matched by a regex against the tensor's GGUF name. The
  standard MoE idiom is `-ngl 99 -ot exps=CPU` (keep everything on GPU except any tensor whose name
  contains `exps`, i.e. every expert's `ffn_*_exps` weight) or a layer-ranged regex like
  `-ot "[2-9][0-9]\.ffn_.*_exps\.=CPU"` to keep only some layers' experts off-GPU. Multiple `-ot` rules
  compose, so a caller can also place different layer ranges on different GPUs
  (`blk\.([0-9])\.=CUDA0,blk\.(1[0-9])\.=CUDA1,exps=CPU`).
- **`--n-cpu-moe N`** (a newer, simpler convenience wrapping the same mechanism): offload the N
  *highest-numbered* layers' expert blocks to CPU RAM without hand-writing a regex, while attention/KV
  cache stay on GPU. Reported real-world effect (multiple independent guides, e.g.
  [Doctor-Shotgun's llama.cpp MoE offload guide](https://huggingface.co/blog/Doctor-Shotgun/llamacpp-moe-offload-guide),
  [turbollm.dev's flags reference](https://turbollm.dev/guides/llama-cpp-flags-that-matter)): can turn
  an unusable (OOM) load into a working one, at a cost proportional to how many tokens hit an
  off-GPU expert (a PCIe round-trip per hit).

**The core, load-bearing fact about this mechanism, worth stating precisely for the "how do they
predict which experts are needed" question the task asks**: **`-ot`/`--n-cpu-moe` do not predict
anything.** They are a **static, load-time-only** placement decision — which *layers'* expert blocks
live on which device is fixed for the whole run, decided once from a regex/count, with no per-token,
per-request, or per-prompt adaptivity at all. Every token that happens to route to an off-GPU expert
pays the same PCIe-transfer cost, every time, regardless of how "hot" that expert has been recently.
This is the most important single distinction between `llama.cpp`'s real, shipped mechanism and every
research system in §2b below — llama.cpp solves *placement under a fixed budget*, not *prediction of
what will be needed next*.

**Directly reusable, not conceptual**: since §1 already established the exact GGUF files this task
would use, `-ot`/`--n-cpu-moe` are literally zero-additional-engineering — they are command-line flags
against files that already exist on Hugging Face, usable today with a stock `llama.cpp` build. This is
the strongest "directly reusable" claim of anything surveyed in this section.

### 2b. Academic/industry prior art — verified real, with confidence and reuse assessment

Every entry below was independently re-confirmed this pass (title, arXiv id, and at minimum an
abstract fetch) rather than cited from recall, per the task's explicit instruction not to trust a
plausible-sounding title:

| System | Verified real? | What it actually does | Predicts "hot" experts how | Reusable or conceptual |
|---|---|---|---|---|
| **Eliseev & Mazur, "Fast Inference of MoE LMs with Offloading"** ([arXiv:2312.17238](https://arxiv.org/abs/2312.17238)) | **Yes** — real paper, Dec 2023 | Mixtral-8x7B on a single consumer GPU: an LRU cache of recently-used experts on GPU, plus **speculative expert loading** — since the router for layer *L+1* can be evaluated as soon as layer *L*'s hidden state is known, it prefetches layer *L+1*'s likely experts while layer *L* is still computing, overlapping the fetch with real compute | One-layer-ahead router-logit lookahead (structural, not learned) + LRU as the fallback cache policy | **Conceptual for this project** — no maintained general-purpose library ships this as a drop-in; the technique (not the code) is what would transfer |
| **Fiddler** ([arXiv:2402.07033](https://arxiv.org/abs/2402.07033), ICLR 2025, [github.com/efeslab/fiddler](https://github.com/efeslab/fiddler)) | **Yes** — real paper + real, maintained open-source repo | CPU-GPU orchestration for uncompressed Mixtral-8x7B (>90GB) on one 24GB GPU: a latency model decides, per layer, whether to compute an expert's FFN on CPU (weights stay resident there) or move it to GPU, based on how many of the current batch's tokens actually need it | A cost model over "how many tokens in this batch route here," not a cache/eviction policy at all — closer to Fiddler's own per-step scheduling decision than to prefetching | **Directly reusable as an external tool** (real installable repo) if this project ever wanted to run an *uncompressed* MoE model outside `llama.cpp`'s GGUF ecosystem — not reusable as C++ library code inside this engine |
| **PowerInfer** ([arXiv:2312.12456](https://arxiv.org/abs/2312.12456)) | **Yes** — real paper | Not MoE-specific — a **dense**-model technique (hot/cold *neuron* activation sparsity, not expert routing), but the same shape of problem: an offline-profiled activation-frequency predictor keeps "hot" neurons resident on GPU, computes "cold" ones on CPU | Offline profiling (a power-law activation-frequency histogram, computed ahead of serving) + adaptive online refinement | **Conceptual only** — the predictor's specific mechanism (neuron-level activation sparsity) doesn't transfer to expert-level MoE routing, but the general "profile once, place by frequency" idea is the same one DLRM's hot/cold splitting (`NGRAM_TABLE_TIERED_STORAGE.md` §1) already uses for embeddings |
| **MoE-Infinity** ([arXiv:2401.14361](https://arxiv.org/abs/2401.14361), [github.com/EfficientMoE/MoE-Infinity](https://github.com/EfficientMoE/MoE-Infinity)) | **Yes** — real paper + real, actively maintained open-source repo (confirmed this pass: HuggingFace-compatible MoE class, OpenAI-compatible serving engine, explicitly lists Qwen-family model support) | An **activation-aware expert cache**: traces which experts a model tends to activate together/in sequence across requests, keeps the structurally-predicted-hot set resident on GPU, and prefetches the rest from host RAM/SSD ahead of need rather than reacting to a miss | Structure-aware prediction from observed expert-activation traces (not plain LRU/LFU) — the same qualitative claim `NGRAM_TABLE_TIERED_STORAGE.md` §1 already flagged as this system's core contribution, now with its GitHub repo independently confirmed real and Qwen-compatible | **The most directly reusable *system* surveyed** if this project ever wanted to actually *serve* this model (as opposed to running it inside Sub0Llm's own engine) — a real, installable alternative to `llama.cpp` for this exact use case, at the cost of leaving the GGUF/`llama.cpp` ecosystem for a PyTorch-based one |
| **FlexGen** ([arXiv:2303.06865](https://arxiv.org/abs/2303.06865), ICML 2023) | **Yes** — real paper | Not MoE-specific — solves GPU/CPU/disk tensor placement via linear programming, optimized for **large-batch throughput** (its own headline result is batch size 144 on a single 16GB GPU), not single-request latency | An LP solve over a static cost model, not a runtime predictor at all | **Conceptual only, and a weak fit** — this project's own likely use pattern (occasional single-request interaction, §4) is exactly the regime FlexGen is *not* optimized for; cited to rule it out as a template, not to reuse, the same way `NGRAM_TABLE_TIERED_STORAGE.md` already ruled out DeepSpeed ZeRO-Infinity for a structurally similar reason |
| **DeepSpeed ZeRO-Infinity / ZeRO-Inference** | **Yes** — already verified in `NGRAM_TABLE_TIERED_STORAGE.md` §1 | Bulk, sequential NVMe streaming of whole parameter/optimizer-state shards — not sparse, hash- or router-addressed random lookup | N/A — no prediction, streams everything in a fixed order | **Negative precedent, restated from the companion doc** — cited again here only because the task brief named it explicitly; nothing new to add beyond what `NGRAM_TABLE_TIERED_STORAGE.md` already established |
| **NVIDIA Merlin HugeCTR HPS, Bandana, DLRM hot/cold splitting** | **Yes** — already verified in `NGRAM_TABLE_TIERED_STORAGE.md` §1 | Embedding-*row* serving under tiered memory, not expert-*weight* serving — cited there for the n-gram table, restated here only to note the underlying tiered-cache shape (GPU→RAM→disk, frequency-leaning eviction, measure-then-size) is the same shape MoE-Infinity/Fiddler independently arrived at for experts | Frequency-based (LFU-leaning), not recency-based | Already assessed in the companion doc; not re-litigated here |

**The core technical question, answered directly**: across every real system above that *does*
predict (Eliseev & Mazur, Fiddler, MoE-Infinity, and — for a different sparsity axis — PowerInfer), the
winning idea is **structure-aware prediction from the model's own computation graph or observed
access traces, not a generic cache-replacement policy applied blind**: either exploit that the router
for the *next* layer/token is computable slightly ahead of when its experts are actually needed
(Eliseev & Mazur's one-layer lookahead), or profile/trace real access patterns offline or across
requests and place by that (PowerInfer, MoE-Infinity, DLRM/HugeCTR's frequency-aware caches). Plain
LRU/no-prediction-at-all (`llama.cpp`'s own `-ot`/`--n-cpu-moe`, and the null case of "just always keep
the same layers on CPU") is the one strategy every research system above is implicitly or explicitly
positioned *against* — but it is also the only one that ships today as a zero-effort, already-available
command-line flag against the exact files §1 already found.

### 2c. What's actually reusable vs. purely conceptual, restated as one list

- **Directly usable today, zero new code**: `llama.cpp`'s `-ot`/`--n-cpu-moe` against the GGUF files
  from §1.
- **Directly usable as an alternative tool/stack** (not as library code inside this engine):
  MoE-Infinity (most mature, Qwen-compatible, real repo) and Fiddler (real repo, narrower model
  coverage) — both would mean running this model *outside* Sub0Llm entirely, via their own serving
  stack, not integrating anything into `src/`.
  - **A caveat worth stating plainly, not glossed over**: neither system's documentation, as fetched
    this pass, specifically confirms support for `Qwen4ExpForConditionalGeneration`
    (`docs/QWEN4_PREVIEW_REFERENCE.md`'s real, verified `model_type`) — MoE-Infinity's own listed model
    support was framed generically ("Qwen models") in what was fetched, not this exact, very-recently
    released architecture by name. Treat "MoE-Infinity/Fiddler could serve this specific model" as
    plausible given how recently the model shipped and how actively both projects are maintained, not
    as independently confirmed for this exact `model_type` this pass.
- **Conceptual only** (the idea transfers, no code to reuse): Eliseev & Mazur's router-lookahead
  prefetch, PowerInfer's offline activation-frequency profiling, FlexGen's LP-based placement,
  DeepSpeed ZeRO-Infinity (ruled out, not adopted).

---

## 3. Naive n-gram/PLE table streaming PoC — design-only, and here is exactly why

### 3a. Restating the scope boundary from the task brief

This section is **not** Sub0Firn. Sub0Firn (`docs/SUB0FIRN_SPEC.md`, `docs/NGRAM_TABLE_TIERED_STORAGE.md`,
now a real spun-off repository at
[github.com/CraigHutchinson/Sub0Firn](https://github.com/CraigHutchinson/Sub0Firn)) is the real,
staged, multi-tier design for this exact problem. This section asks a narrower question: is there a
*simpler-than-that*, throwaway, zero-new-caching-logic option — plain `mmap` + let the OS page cache do
whatever it does — and does it fit `docs/NGRAM_EMBEDDING.md`'s own already-identified integration seam
(`ngram_tab[e]` becoming "a thin client issuing `resolve_into` calls instead of a raw parameter
pointer") as a genuinely minimal special case.

### 3b. The one genuinely encouraging structural fact, found this pass

`include/sub0/core.hpp`'s `Node::data` is a **`std::span<float>`, not an owning container**
(`std::span<float> data, grad;` — `core.hpp` line 52). `backend_cpu.cpp`'s `op_embed` (the function
`docs/NGRAM_EMBEDDING.md` §4's `op_embed(ngram_tab[e], ngram_ids[e], T)` calls) only ever does
`mat(table->data, table->rows, C)` and indexes into it — **it has no idea, and no way to tell, whether
`table->data` points at an arena slot, a `std::vector`'s backing store, or a raw `mmap`'d region.** This
is a real, load-bearing consequence of a design decision this project already made for other reasons
(non-owning spans throughout the `Node` graph) — not something built for this task, but something this
task benefits from directly: **`op_embed` itself needs zero changes to read from an mmap'd table.**

### 3c. Why this does not make the naive PoC trivial anyway — the two cases split differently

`docs/NGRAM_TABLE_TIERED_STORAGE.md` §0 already drew this exact distinction and it matters again here:

- **Case 2 (a hypothetical future Sub0Llm-*trained*, oversized n-gram table)**: for this case, §3b's
  finding really does mean a naive whole-table `mmap` is close to "a few lines" — bind a leaf `Node`'s
  `data` span to an `mmap`'d flat-fp32 sidecar file instead of an arena slot, and the entire rest of
  `op_embed`/`op_linear`/`op_add`'s composition (`docs/NGRAM_EMBEDDING.md` §4) is unaffected. **But this
  case does not exist yet** — `docs/NGRAM_TABLE_TIERED_STORAGE.md` §0 already established Sub0Llm's own
  table would need `NGRAM_TABLE_SIZE` (or `D_MODEL`/`K`/`NGRAM_MAX_N`) to grow 3-4 orders of magnitude
  past today's toy configs before residency is even a question — there is no code path today that
  produces a table this would apply to, so implementing it now would be exactly the "speculative
  surface area nothing reads yet" `AGENTS.md` §8 warns against.
- **Case 1 (the real, external Qwen n-gram/PLE table — the actual present pressure, per
  `NGRAM_TABLE_TIERED_STORAGE.md` §0's own honest disagreement with treating both cases as equally
  urgent)**: this is where the naive PoC is asked for, and here it is **not** trivial, for a reason
  §3b's finding does not fix — **this table has no existing consumption seam in `Model::forward` at
  all.** It has never been a `PARAM_LAYOUT` leaf, was never wired into any `op_embed` call, and
  `docs/QWEN4_PREVIEW_REFERENCE.md`'s own Stage 1 extraction built a **standalone, offline correctness
  fixture** from a handful of real rows — it never ran the table live inside this engine's graph, and
  nothing about that has changed. "Point an existing leaf's span at an mmap" presumes an existing leaf;
  there isn't one to redirect. Building one from scratch runs immediately into three real, non-trivial
  requirements, none of which are "a few lines":
  1. **Dtype conversion.** The real table's on-disk bytes are `BF16` (or `Q8_0`/`IQ4_NL` in the GGUF
     forms, §1c) — `op_embed`'s `Mat`/`std::span<float>` machinery reads plain `float`. A row-at-a-time
     read would need to decode into a small scratch buffer before the span could expose it as `float`,
     which is no longer "point the span at the mmap" but "point the span at a decode buffer this project
     would still need to write" — and per §1d, this project's *own* GGUF reader cannot yet even compute
     `BF16`'s byte length, let alone decode it.
  2. **A leaf-construction path outside `PARAM_LAYOUT`.** This tensor is untrained and uncheckpointed by
     this project — it cannot join `PARAM_LAYOUT`/`ARCH_FINGERPRINT` the way Sub0Llm's own table does
     (`docs/NGRAM_TABLE_TIERED_STORAGE.md` §2e already worked through exactly why case 1 "isn't really
     the same knob" as case 2's checkpoint-format question) — so `Model` would need an entirely new,
     parallel way to construct a leaf `Node` that is neither an arena-allocated trainable parameter nor
     a `PARAM_LAYOUT` entry. Nothing in `Model`'s current structure does this today.
  3. **The resolve-pass wiring `NGRAM_TABLE_TIERED_STORAGE.md` §2a already specified.** Per `AGENTS.md`
     §1 (no runtime-variable-latency branch in a hot path), even a "naive" mmap cannot just let a page
     fault happen inline inside the batched forward loop on a cold page backed by a *remote* HTTP range
     source or a not-yet-paged-in disk file without becoming exactly the unbounded-latency hazard that
     rule exists to prevent — a resolve pass ahead of the hot op is still required, which is real,
     already-designed plumbing (`NGRAM_TABLE_TIERED_STORAGE.md` §2a/Stage 3), not a "let the OS handle
     it and walk away" shortcut. (A **local, already-fully-downloaded** flat file is the one sub-case
     where a page fault's latency is bounded enough — tens of µs, NVMe-class — that skipping an explicit
     resolve pass might be defensible; this document does not resolve that judgment call, since case 1
     doesn't have a local file to mmap without first solving requirement 1 above.)

### 3d. Conclusion for this thread

**Design-only, not implemented this pass** — matching the task's own explicit fallback instruction.
The naive mmap idea is real and worth keeping (§3b's `std::span<float>` finding is a genuine, positive
discovery worth remembering), but it applies cleanly to a case (Sub0Llm's own future table) that does
not exist yet, and does not apply cleanly to the case that actually matters today (the external Qwen
table) without first doing dtype-decode and leaf-construction work that is squarely `Sub0Firn`/Stage-3
territory, not a PoC-sized shortcut around it. **This is not a reason to abandon a lightweight option
in principle** — it is a reason to build the *decode* step (§1d's `gguf.hpp` gap: `BF16`/`IQ4_NL`/K-quant
byte-size + dequantization support) as the actual next concrete unit of work, since that is the one
piece every path through this problem — Sub0Firn's real design, and any future "naive" shortcut alike —
needs regardless of which cache architecture eventually sits on top of it.

---

## 4. Recommendation — what is actually achievable here, and what to do next

### 4a. What is achievable, stated plainly and hedged where the evidence is thin

- **Training this model here: not on the table**, full stop — unchanged from
  `docs/QWEN4_PREVIEW_REFERENCE.md`'s own framing, and §1's real numbers only reinforce it (even the
  *smallest* GGUF quant is 67.56 GiB, before accounting for activations, KV cache, or any training
  state at all).
- **Full, comfortable serving on this machine: not achievable at any quant level.** Every real tier in
  §1b exceeds 63GB RAM; the 8GB GPU is 8-12% of even the smallest tier's footprint, so GPU offload via
  `-ngl`/`-ot` (§2a) can only ever shift a small slice of the model — it does not change the fundamental
  "the working set does not fit in fast memory" problem.
- **Occasional, single-request CPU-mostly inference at the smallest tier (`UD-IQ1_S`, 67.56 GiB) is
  *plausible*, not comfortably usable** — this is a hedged claim, not a confident "yes": `llama.cpp`
  memory-maps GGUF files by default, so the model does not need to fully load before generation starts,
  and the OS page cache absorbs re-reads of whatever fits; but 67.56 GiB against 63GB RAM means the
  *entire* file cannot stay resident simultaneously even once "warmed," so some amount of re-paging from
  the 589GB-free `D:` drive on every generation is close to guaranteed, not an edge case — expect
  noticeably slow, possibly quite slow, single-token latency, not a number this document can respectably
  estimate without actually running it (no measurement exists this pass). `-ot exps=CPU`/`--n-cpu-moe`
  (§2a) plus a handful of GPU-resident attention/GDN layers (`-ngl`, a small number) is the concrete,
  actionable starting recipe if this is ever tried — it costs nothing to attempt with a downloaded
  `UD-IQ1_S` file and 589GB of free disk comfortably covers the ~72.5GB download.
- **Every quant tier from `UD-IQ1_M` upward is further from fitting, not closer** — there is no quant
  level on the menu that makes this a comfortable fit; `UD-IQ1_S` is the only tier worth attempting at
  all on this hardware, and even it is a "technically launchable" claim, not a "runs well" one.

### 4b. The highest-value next step — specific to this project, not generic Qwen-serving advice

**Do not acquire any GGUF quant tier wholesale, including `UD-IQ1_S`, as this project's next action.**
§1's own numbers argue against it directly: even the smallest real tier is a 72.5GB download for a
"technically launchable, not comfortably usable" result (§4a), and this project has already
demonstrated, twice now (`docs/QWEN4_PREVIEW_REFERENCE.md`'s Stage 0/1 safetensors extraction; this
document's own §1 GGUF header probes), that **surgical HTTP-range extraction of exactly the bytes
needed answers real questions at a cost of megabytes, not gigabytes.** If a genuine need for live
access to this model's real weights arises (e.g. a distillation-teacher signal from the real n-gram/PLE
table, or a further GDN/QSA correctness fixture at a different layer), the right next step is **more of
the same targeted extraction this project already validated**, not a bulk download — this holds
regardless of which quant tier would technically be smallest, because the smallest tier is still
inconveniently large and the extraction technique doesn't care about tier at all (§1c's byte-exact
header reads worked identically across every tier probed).

**Where this project's actual leverage already is, and where it should stay:**

1. **The GDN and n-gram-embedding CPU (+ CUDA, for GDN) implementations are already the load-bearing
   deliverable of this whole line of work** — both are validated end-to-end against real, extracted
   Qwen weights at small scale (`tests/fixtures/qwen4_preview/*`, `docs/GATED_DELTANET.md` §6's
   `4.37e-11`-level fixture match). The natural next step for *this* codebase is not "run the 125B
   model," it is **using these already-verified small-scale correctness fixtures as the validation
   oracle for Sub0Llm's own from-scratch GDN/n-gram mechanisms at Sub0Llm's own realistic training
   scales** — which is exactly the stated purpose of the pivot this whole research thread sits inside
   (`[[pivot-to-existing-model-support-qwen4-preview]]`: "Qwen3.8-Flash-Next as validation oracle," not
   as a model to deploy). This document's findings do not change that conclusion; they confirm it more
   strongly, by showing concretely how far out of reach *actually running* the real model is on this
   hardware at every quant level.
2. **If an external-table consumption use case becomes real** (§3's case 1), the correct venue is
   **Sub0Firn**, not a bespoke shortcut in this repo — it is already spec'd through Stage 2 (standalone,
   zero Sub0Llm dependency) and Stage 3-5 (Sub0Llm integration), and §3d's finding (the `gguf.hpp`
   dtype-decode gap) is squarely useful prerequisite work for *either* Sub0Firn's real design or any
   future naive shortcut, so it is worth doing regardless of which one comes first. This document does
   not schedule that work — only names it as the concrete, scoped, connects-to-what-exists next
   candidate, per §1d/§3d.
3. **`llama.cpp`'s `-ot`/`--n-cpu-moe` (§2a) and MoE-Infinity (§2b) are both real, usable-today options
   if this project (or the user) ever wants to actually *talk to* this model** for reasons other than
   validation-oracle extraction — e.g. using it as a live reference to sanity-check a specific output
   this project's own engine produces. Neither requires any engine code in this repository; both are
   external-tool decisions, not Sub0Llm feature work.

### 4c. One thing this document deliberately does not claim

No token-per-second number, no wall-clock generation latency estimate, and no confident "this will/will
not work" verdict for `UD-IQ1_S` on this exact machine is given anywhere above, because **no actual run
was attempted this pass** — every number in §1 is a verified, static, byte-level fact (file sizes,
tensor shapes, quant types); every claim in §2/§4a about *behavior* under memory pressure is drawn from
cited prior art and general `mmap`/page-cache reasoning, not a measurement taken on this hardware
against this model. Per this project's own standing policy
(`[[measure-tokenizer-on-the-full-corpus-blend]]`'s sibling spirit — don't present an estimate as a
measurement), if a performance number for actually running `UD-IQ1_S` here becomes load-bearing for a
future decision, it should be measured, not further estimated from this document's numbers.
