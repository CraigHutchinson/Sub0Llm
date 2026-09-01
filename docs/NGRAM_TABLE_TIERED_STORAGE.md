# N-gram / sparse embedding table tiered storage — design + staged plan

Status: **DESIGN ONLY. No engine code this pass.** Follows the staging convention of
`docs/DEPTH_ATTENTION.md`/`docs/GATED_DELTANET.md`/`docs/NGRAM_EMBEDDING.md` (numbered findings, a
staged plan with checkable exit conditions, an explicit "novel vs reuses an existing idiom" split,
sources cited with a confidence tag) but this doc's own "Stage 0" has no landed code — the staging
below is the deliverable, not a record of what shipped.

**Companion doc**: [SUB0FIRN_SPEC.md](SUB0FIRN_SPEC.md) — the spec/requirements/README for **Sub0Firn**,
the standalone sideline project this design spins the tiered-storage engine out into (see §7). This doc
covers the problem, the prior art, and how Sub0Llm's own engine reconciles with it; that doc covers what
Sub0Firn itself promises as an independent, engine-agnostic library.

**2026-09-01: the Sub0Firn repository now exists**, and has since grown past its initial seed —
[github.com/CraigHutchinson/Sub0Firn](https://github.com/CraigHutchinson/Sub0Firn) now has its own
`REQUIREMENTS.md` (a normative R1–R10 contract), `AGENTS.md`, `STYLE_GUIDE.md`, a `docs/` split into
`prior-art.md`/`tiered-storage-design.md`/`reference-consumer-sub0llm.md`, and a buildable
(header-only-skeleton) `CMakeLists.txt` — brought up to the same requirements rigor as the sibling
`Sub0Log` project. This doc (originally seeded there as `DESIGN_RATIONALE.md`, now
`docs/tiered-storage-design.md` in that repo) stays here as Sub0Llm's own historical record of how/why
the spin-off happened, not kept in lockstep with every subsequent edit there; the Sub0Firn repo is now
where any actual implementation work on the tiered-cache engine itself belongs, not here. No code has
landed in
either repo yet — both documents are still design-only.

## 0. The problem, stated at real scale

Two motivating cases, per the task brief — kept distinct throughout because they hit the resource wall
at very different scales:

1. **Using the real `Qwen/Qwen3.8-Flash-Next` n-gram table as a frozen external resource** — a
   distillation-teacher signal source, or an imported/frozen component a Sub0Llm model reuses. Real,
   verified numbers (`docs/QWEN4_PREVIEW_REFERENCE.md`, `tests/fixtures/qwen4_preview/ngram_embedding_manifest.json`):
   `nn.Embedding(320_001_536, 160)`, **51.2B params exactly, ~102.4GB in bf16**, stored as 128 on-disk
   `ngram_embedding.shard_N.weight` tensors. A single forward pass touches exactly **16 rows per token
   position** (8 bigram heads + 8 trigram heads, `ngram_size=3`, `heads_per_ngram=8`) — at bf16, `16 ×
   320 bytes = 5,120 bytes` touched per position out of 102.4GB (a per-token sparsity ratio of roughly
   `5×10⁻⁸`). This machine has **63GB RAM and one ~8GB-VRAM laptop GPU**
   (`[[host-cpu-arrow-lake-hx]]`) — the full table cannot be resident in either, and downloading it
   whole (once, ~102GB) is a real option but a large one-time cost most workflows won't want to pay
   before knowing the signal is useful.
2. **Sub0Llm's own trained n-gram tables**, if ever scaled up from today's toy sizes. `docs/NGRAM_EMBEDDING.md`
   §6's own worked example (`D_MODEL=448, NGRAM_MAX_N=3, K=2`, `NGRAM_TABLE_SIZE≈8000`) is **≈3.78M
   floats (~15.1MB fp32)** — four to five orders of magnitude below case 1. Today these tables are
   fully-resident `PARAM_LAYOUT` leaves, exactly like `tok_emb` (`docs/NGRAM_EMBEDDING.md` §4), and
   nothing about that is wrong at this scale.

**Where this doc disagrees with the brief's framing, stated up front (elaborated in §5c and the final
report)**: the brief motivates both cases as roughly co-equal pressure toward the same design. Worked
through concretely, they are not close. Growing Sub0Llm's own table to a size that actually breaks full
residency on a 63GB-RAM machine needs `NGRAM_TABLE_SIZE` (or `D_MODEL`/`K`/`NGRAM_MAX_N`) to grow by
roughly **3–4 orders of magnitude** past today's toy configs — a real but *distant* future, not an
imminent one, and this project's own `VOCAB` (a few thousand to tens of thousands at realistic corpus
sizes) caps how large `NGRAM_TABLE_SIZE` naturally wants to be without deliberately padding it beyond
the hashing scheme's own real vocabulary anyway. Case 2 is designed for below (§4, §6) because the brief
asks for it and because the interface should not need a second redesign later, but the actual, present
pressure is 100% case 1. This doc does not treat case 2 as equally urgent, only as equally *served* by
the same design.

## 1. Real prior art (fetched and cited, not recalled — `AGENTS.md` §5's discipline applied to an
infrastructure design)

Confidence tags follow `docs/QWEN4_PREVIEW_REFERENCE.md`'s own convention: **High** = primary source
fetched and quoted directly this pass; **Medium** = fetched, but an automated summarizer rendered the
content rather than raw extracted text (a real limitation hit twice below — both PDFs came back as
undecoded binary to the fetch tool, so those two citations rest on the tool's own summary, not a
verbatim quote independently re-checked); **title-only** = a real, existing, citable source located via
search, not independently fetched this pass.

| System | What it actually does | Confidence |
|---|---|---|
| **NVIDIA Merlin HugeCTR — Hierarchical Parameter Server (HPS)** | Real quote, `hugectr_parameter_server.html`: *"a three-level storage architecture. The first and highest performing level is GPU memory and is followed by CPU memory. The third layer can be high-speed local SSDs with or without a distributed database."* Backend technologies, also quoted verbatim: local CPU-memory tier = `hash_map`/`parallel_hash_map`/`multi_process_hash_map`, or `redis_cluster` for a **distributed** CPU-memory tier; disk tier = `rocks_db` ("Create or connect to a RocksDB database"). GPU-tier eviction is explicitly LRU: *"Prune embeddings starting from the oldest embedding until the partition contains at most `overflow_margin * overflow_resolution_target` embeddings. This policy implements the least-recently used (LRU) algorithm."* A `hit_rate_threshold` config knob controls whether a GPU-cache miss triggers a **synchronous** or **asynchronous** insert: below threshold, synchronous (correctness-first, accept the stall); above threshold, asynchronous (throughput-first, the rare miss doesn't block). **Caveat found, not hidden**: the standalone HPS module has been **deprecated since v25.03** per its own current docs page — real production evidence that a bespoke multi-backend DB-serving layer for this exact problem is expensive enough to maintain that even NVIDIA's own team retired the standalone form (folded into a newer serving stack rather than kept as a separable component) — worth weighing against building something similarly broad. | **High** |
| **Bandana (Facebook/Meta, MLSys 2019)** | Real abstract, quoted in full: *"We present Bandana, a storage system that reduces the DRAM footprint of embeddings, by using Non-volatile Memory (NVM) as the primary storage medium, with a small amount of DRAM as cache. The main challenge in storing embeddings on NVM is its limited read bandwidth compared to DRAM. Bandana uses two primary techniques to address this limitation: first, it stores embedding vectors that are likely to be read together in the same physical location, using hypergraph partitioning, and second, it decides the number of embedding vectors to cache in DRAM by simulating dozens of small caches. These techniques allow Bandana to increase the effective read bandwidth of NVM by 2-3x."* **This is the single most load-bearing citation for this design** — see §1's discussion below. | **High** (abstract verbatim; body text came back corrupted from the PDF fetch, so the hypergraph-partitioning mechanism's exact algorithm is not independently re-verified beyond the abstract's own description) |
| **TT-Rec (Meta, MLSys 2021, arXiv:2101.11714)** | Tensor-Train factorization replacing a dense embedding table with a product of small "core" tensors. Real reported result (search-confirmed, title/authors Yin, Acun, Liu, Wu): **112x model-size reduction on a terabyte-scale table with no accuracy loss and no training-time overhead** vs. the uncompressed baseline, plus an optimized `TT-EmbeddingBag` kernel reported 3x faster than prior TT implementations. | Medium (title/result confirmed via search summary, not a direct fetch of the paper body) |
| **DeepSpeed ZeRO-Infinity — NVMe parameter/optimizer-state offload** | Real, documented config surface (`deepspeed.readthedocs.io`): NVMe offload is configured with a **buffer pool** (`nvme_path`, buffer count, buffer size defaulting to 100,000,000 elements) and a **fixed CPU-resident element count** (default 1,000,000,000) — i.e. it is designed around large **sequential, bulk** reads of contiguous parameter/optimizer-state shards streamed to/from NVMe at high aggregate bandwidth, reported able to fit a 33.3B-param model's states across CPU+NVMe. **Deliberately a negative precedent for this problem**: nothing in this design does per-row, hash-addressed random lookup — it offloads whole parameter groups per training step, not sparse rows on demand. Citing it mainly to rule it out as a template, not to reuse its mechanism. | **High** (docs quoted directly) |
| **Frequency-aware GPU software cache for DLRM (arXiv:2208.05321)** | Real abstract, quoted in full: *"We propose a GPU-based software cache approach to dynamically manage the embedding table in the CPU and GPU memory space by leveraging the id's frequency statistics of the target dataset... Evaluating our prototype system shows that we can keep only 1.5% of the embedding parameters in the GPU to obtain a decent end-to-end training speed."* Concrete anchor number used in §3's VRAM-tier estimate. Frequency-based (LFU-shaped), not recency-based (LRU) — the literature in this space consistently prefers frequency over recency because embedding-id access in recommendation workloads is Zipfian/skewed rather than recency-correlated. | **High** (abstract verbatim) |
| **DLRM hot/cold popularity-based splitting (Mahajan et al., VLDB 2022, "Accelerating Recommendation System Training by Leveraging Popular Choices")** | Same family as the row above: split embedding rows into a small "popular"/hot partition kept in fast memory and a large "cold" partition kept in slow memory, sized from real access-frequency statistics rather than a fixed ratio. | Title-only (located, not fetched) |
| **MoE-Infinity (arXiv:2401.14361) — sparsity-aware expert cache for MoE inference on personal machines** | **Directly analogous, not just adjacent**: this is a huge sparse model (mixture-of-experts) routing to a tiny active subset per token, run on a resource-constrained personal machine, with a disk-tier cache. Core claim (tool-summarized, not independently re-quoted verbatim — see confidence note): a routing-structure-aware predictive cache beats plain LRU because MoE expert activation is *structured*, not i.i.d. — the paper models expert-transition patterns to prefetch before a request, rather than reacting to a miss after it happens. Reported magnitude numbers (memory reduction, hit-rate deltas vs. LRU) came back from the fetch tool as a paraphrase with specific percentages attached; **these exact percentages are not independently verified against the paper's raw text this pass and should be treated as approximate, not load-bearing** — the qualitative claim (structure-aware prediction beats generic recency-based eviction for this exact shape of workload) is the part being relied on here, consistent with `AGENTS.md`'s "quote the source, don't paraphrase from training data" discipline extended to "don't over-trust a paraphrase you couldn't independently verify either." | **Medium** (qualitative claim); **Low** (specific percentages) |
| **FlashMoe (arXiv, 2026) — ML-based cache replacement for SSD-offloaded MoE experts** | Recent (2026), same problem shape as MoE-Infinity, reportedly combining recency and frequency signals adaptively for expert-cache replacement rather than committing to one policy. Notable mainly as evidence this exact problem class (huge sparse table on personal-scale hardware, disk-tier cache, prediction beats a single fixed eviction policy) is still an active, unsettled research area as of this pass, not a solved one. | Title-only |
| **mmap + OS page cache as an implicit cache tier** | General finding across multiple sources: for large-file random access, `mmap` avoids `read()`'s per-call syscall/seek overhead and lets the kernel's own LRU-based page cache absorb re-reads for free — "the OS already does most of this job" is a real, load-bearing null hypothesis worth taking seriously before building a bespoke tier (§1's discussion below works out exactly where it stops applying at this table's real row size). | Medium (general finding, several corroborating but non-primary sources) |
| **LMDB vs. RocksDB as the disk-tier KV format** | LMDB: B+tree over an `mmap`'d file, true zero-copy reads (a lookup returns a pointer straight into the mapping, no deserialization/copy step), MVCC with a single writer and unlimited concurrent lock-free readers, **no background compaction** (so no compaction-induced latency spikes — "consistently fast reads and predictable latency," per the sourced summary). RocksDB: LSM-tree, write-optimized, real write amplification, and background compaction that **can** spike tail read latency. | Medium (general design summary from multiple docs/comparison sources, not a single primary spec fetched verbatim) |

**The single most load-bearing citation, and why**: Bandana. It is the only source above solving
*exactly* this problem — a huge, mostly-frozen embedding table, a small DRAM budget, real read-bandwidth
pressure on the slow tier — rather than a training-time or MoE-routing variant of it. Its two techniques
map onto this design directly: (a) **co-access-aware physical placement** ("store vectors likely to be
read together in the same physical location") is the mechanism that makes a disk/NVM tier's coarse read
granularity (a page, a block, an OS readahead window) actually pay for itself instead of wasting most of
every fetch on unrelated rows, and (b) **measure-then-size the cache, don't guess it** ("simulating
dozens of small caches") is a concrete, checkable technique this design should reuse directly once a real
access trace exists (§3's hit-rate numbers below are estimates for exactly this reason — no trace exists
yet to run Bandana's own sizing technique against).

**Where the literature disagrees or evidence is thin, stated honestly**: HugeCTR's own team retired the
standalone HPS module (§1's table) even though the 3-tier shape it implemented is echoed by nearly every
other source here — that is evidence the *shape* is right but a *fully general, multi-backend, always-on
service* is expensive to keep alive, not evidence against tiering itself. TT-Rec's compression approach
and every caching approach above are not competitors so much as orthogonal axes (§5c) and nothing in the
literature located this pass directly compares "compress the table" against "tier the serving of an
uncompressed table" head-to-head on the same workload — that comparison, if it matters, is open.
MoE-Infinity's and FlashMoe's exact quantitative claims are the thinnest evidence in this table (§1
above) precisely because both PDF fetches came back corrupted/summarized rather than raw text; their
qualitative direction (prediction beats naive recency for structured sparse access) is corroborated by
every other source in this table that discusses eviction policy at all (HugeCTR's LRU-by-default is the
one outlier, and even it adds a frequency-adjacent `hit_rate_threshold` knob on top).

## 2. Reconciling with this engine's real constraints — the part generic advice won't get right

### 2a. No heap allocation / no branch-on-miss inside the hot path (`AGENTS.md` §1)

A cache-miss-triggered disk or network read cannot happen inline inside the batched forward loop —
not because "it would be slow" in the abstract, but because this engine's entire per-step cost model
assumes fixed-size, pre-sized, reused scratch buffers with no runtime-variable-latency branch anywhere
in `backend_cpu.cpp`'s hot loop (`op_embed`, `op_linear`, etc. all read/write fixed-shape spans). The
design answer is exactly what the brief proposes and this section works out precisely: **an explicit
prefetch/resolve pass that runs BEFORE the batched op**, populating a pre-sized working-set buffer the
hot loop then reads from unconditionally — uniform latency, no branch on hit/miss inside the compute
itself. Concretely, this becomes a new pipeline stage inserted before `Model::forward`'s existing
n-gram block (`docs/NGRAM_EMBEDDING.md` §4's `op_embed(ngram_tab[e], ngram_ids[e], T)` calls): compute
`ngram_ids[e]` for the whole window exactly as today (pure index arithmetic, no I/O), then **resolve**
every row those ids address into a flat, pre-sized arena buffer, THEN run the existing composed
`op_embed`/`op_linear`/`op_add` pipeline reading from that buffer instead of from a resident
`PARAM_LAYOUT` tensor. `op_embed` itself does not need to change — only what it's an embedding lookup
*into* changes, from "the resident table" to "this call's already-resolved working-set buffer," which
is exactly the kind of substitution the thin-client interface (§7, `SUB0FIRN_SPEC.md`) exists to make
clean.

**How the resolve pass gets its row-ids differs sharply between training and decode — the brief is right
to flag this as the crux, and §2b below is the actual answer for training.**

### 2b. Training: the corpus is known upfront — a strictly stronger opportunity than reactive caching

This project already has a directly-relevant, *working* precedent for exactly this shape of problem:
the out-of-core corpus pipeline (`[[out-of-core-corpus-pipeline]]`, `[[corpus-tok-reuse-stamp]]`).
`corpus.tok` is mmap'd rather than resident, the configurator computes it in one streaming pass, and
a stamp (`generated/tokenizer.stamp`, keyed on corpus size+mtime plus the exact inputs the pass depends
on: `vocab_target`/`min_merge`/`emit_tok`) lets a re-run skip recomputation entirely when nothing
relevant changed. **The n-gram working set is the same shape of derived, cacheable artifact.**

Concretely: `docs/NGRAM_EMBEDDING.md` §5 already establishes that a training window's local start `t=0`
either coincides with a real document start or a point the model has no other access to anyway — so
every position of every document in the corpus is a potential window-local position some future draw
might sample. That means the **complete set of n-gram hash row-ids a training run could ever touch** is
not "whatever a particular sequence of sampled windows happened to touch" (which a reactive cache would
have to discover step by step, and might still miss on a later, differently-sampled epoch) — it is the
closure of the hash formula over the **entire tokenized corpus**, computed once:

```
for each document d in corpus.tok:
    for each position t in d:
        for each table e in 0..E-1:
            working_set.insert(hash_e(t))     # exact same formula op_embed already evaluates
```

This is:
- **Computable in one pass, at the same asymptotic cost as tokenization itself** (`O(corpus_length ×
  E)` hash evaluations, no I/O beyond reading `corpus.tok`, which is already mmap'd) — directly parallel
  to how `corpus.tok` itself is a one-time derived artifact of the raw corpus.
- **Naturally stampable** the same way (`corpus.tok`'s mtime+size, plus the config axes the hash
  formula actually depends on: `NGRAM_MAX_N`/`NGRAM_TABLES_PER_ORDER`/`NGRAM_TABLE_SIZE` for Sub0Llm's
  own table, or a table/version identifier for an external table like Qwen's) — a `ngram_workingset.stamp`
  sitting right next to `tokenizer.stamp`, invalidated by exactly the inputs that change the answer and
  nothing else, following `corpus-tok-reuse-stamp`'s own validated pattern (size+mtime+the three inputs
  that matter, not a blanket "any config change" invalidation).
- **A strictly stronger prefetching signal than a generic LRU cache gets**, because the working set is
  the exhaustive closure, not a sample: if it fits the RAM tier's budget, **the entire training run can
  be warmed once before the first step and then never miss again** — every subsequent access during
  training is a guaranteed RAM hit, not merely a probable one. This is the headline result worth stating
  plainly: for training, under a corpus that's small enough for its own working set to fit the RAM
  budget, tiered storage degenerates to "one bulk warm-up, then resident for the run" — no ongoing
  eviction decisions to get wrong at all. When the working set does *not* fit (the realistic case at
  Qwen's real table scale against a large corpus), the precomputed set is still the right input to
  Bandana's own "simulate dozens of small caches" sizing technique (§1) — it is the exact access
  population that technique wants to be handed, not an approximation of it.
- **A genuine admission check worth stating explicitly**: if the computed working-set size approaches
  the full table size (plausible for a very large/diverse corpus against a table with excellent
  discriminating hash coverage), tiering was never going to help much for that corpus regardless of
  policy — the honest conclusion in that case is "this corpus doesn't have enough n-gram repetition
  for a cache tier to pay for itself," not a reason to build a smarter eviction policy.

### 2c. Decode/inference: no corpus, genuinely reactive — but with one free structural insight

A live prompt's n-grams are not known in advance, so decode cannot reuse §2b's precompute. This is
exactly the shape MoE-Infinity/FlashMoe (§1) are built for: reactive, frequency/recency-aware caching
with predictive prefetching where the *structure* of the access pattern (not the specific future values)
is knowable ahead of any single request. Two points worth making precisely, not left as "just use LRU":

- **The hash formula's own structure gives the resolve pass a genuine, if narrow, prefetch window for
  free, distinct from anything MoE-Infinity needs machine learning to predict.** `docs/NGRAM_EMBEDDING.md`
  §8's Stage 1 `forward_one` already carries a rolling context-history buffer across sequential
  single-token decode calls. The current input token id at decode step `t` — the very token being
  embedded that step — is known the instant it is sampled at the *end* of step `t-1`, strictly before
  step `t`'s `forward_one` call begins. So the resolve pass for step `t`'s 16 row-ids can be issued as an
  explicit call **between** "sample the next token" and "call `forward_one` for it" — the exact same
  "prefetch pass before the hot op" shape §2a establishes for training, just with a one-step-deep window
  instead of a whole-corpus one, and satisfying the same "no branch inside compute" rule for decode that
  §2b's precompute satisfies for training via a completely different mechanism. This is **not** free
  latency-hiding via overlap, though — worth being honest about the limit: because n-gram injection
  happens only at the input embedding (`docs/NGRAM_EMBEDDING.md` §2, single-layer injection), there is no
  independent per-step compute to overlap the resolve against on a single CPU thread; the benefit here is
  purely architectural (an explicit call site instead of a hidden branch), not a throughput win. **If**
  the deferred multi-layer re-injection (`NanbeigeNgramLayerFusion`, `docs/NGRAM_EMBEDDING.md` §7) is
  ever implemented, real overlap becomes possible — issue the resolve at token-sample time, let the
  attention/FFN of the intervening layers run while any disk/network I/O completes in the background,
  inject only once resolved. Flagged as a real future opportunity, not claimed as available today.
- **Eviction policy for the reactive RAM/VRAM tiers should be frequency-leaning, not plain LRU**, per
  §1's DLRM/frequency-aware-cache citations — natural-language n-gram recurrence within one session is
  bursty (the same words/phrases recur, per the task brief's own framing) but not strictly
  recency-correlated (a phrase used at the start of a long conversation may recur at the end, having
  been "cold" throughout the middle). An LFU-with-decay or ARC-style policy (adapts between
  recency-favoring and frequency-favoring based on observed workload, rather than committing to one) is
  the better-motivated default than plain LRU, though no measurement exists yet to confirm this at this
  project's own access patterns (§3 states this as an estimate).

### 2d. Relation to `gguf.hpp` and Stage 1's HTTP Range extraction — extend the existing seam, don't parallel it

`include/sub0/gguf.hpp` already establishes "read exactly the bytes a caller names, from an external
weight-file format, without loading the whole file" as a working, engine-free, testable pattern (it
parses a GGUF header/tensor-table and computes exact byte offsets, though it does not itself fetch —
it operates over an in-memory buffer the caller already has). `docs/QWEN4_PREVIEW_REFERENCE.md`'s own
Stage 1 extraction (real HTTP Range requests against safetensors shards, offsets located from each
shard's own header, never downloading the full table) is the same idea, proven against a real remote
source rather than a local buffer. **The tiered store's "resolve a table-id + row-index to bytes"
contract is this same primitive, generalized and made persistent/reusable rather than reinvented**:

| Source shape | What "resolve a row" means | Relationship to existing precedent |
|---|---|---|
| **(a) Fully-local downloaded safetensors shards** (user chose to pay the ~102GB one-time cost) | `mmap` the shard set, compute the byte offset from each shard's own header (exactly `gguf.hpp`'s offset-table logic, applied to the safetensors format instead of GGUF), read directly — no network at all after the initial download. | Direct reuse of `gguf.hpp`'s "parse header, compute exact offset" logic, retargeted at safetensors' JSON header instead of GGUF's binary one. |
| **(b) Remote-only, HTTP Range + local disk cache in front** | Exactly Stage 1's proven extraction script's shape, made **persistent and reusable across runs** instead of one-off: a disk cache tier stores previously-fetched rows keyed by `(table_id, row_index)`, consulted before any network round-trip; a miss issues the same Range request Stage 1 already validated. | Extends Stage 1's script from "one-off extraction tool" into "the remote leg of a standing cache tier." |
| **(c) Sub0Llm's own from-scratch-trained table** | Today: fully resident `PARAM_LAYOUT`, no lookup/resolve step needed at all — `op_embed` reads it directly, same as `tok_emb`. | **This is the case that stays exactly as-is by this design, on purpose** (§0's scale argument) — nothing here proposes changing the default until a real Sub0Llm-trained table's size actually crosses a threshold worth naming (§5c works out where). When it does, it becomes a fourth row in this table: a *local*, *self-produced* source, structurally closest to (a) since the whole table already lives on this machine, just no longer resident in RAM. |

A local disk-cache tier populated from (b) or a full local copy under (a) are the **same on-disk format
and the same resolve code path** in this design — the only difference is whether every possible row is
already present (a) or the cache is partial and can still miss out to the network (b). This is a
deliberate simplification: don't design two disk-tier formats, design one, and let "how much of it is
populated" be the only difference between "cold cache in front of a remote source" and "a complete local
mirror."

### 2e. Interaction with `NGRAM_EMBED`'s compile-time toggle — mostly orthogonal, and here is the reasoning

`layout.hpp`'s `NGRAM_EMBED` (`if constexpr`-gated, per `AGENTS.md` §2) governs whether **Sub0Llm's own
n-gram mechanism is architecturally present at all** — it changes `PARAM_LAYOUT`, `trainable_floats()`,
and joins `MODEL_ARCH_ID` (`docs/NGRAM_EMBEDDING.md` §6). The brief asks whether resident-vs-tiered
serving of a *fixed* table should be a similar compile-time axis or a legitimate runtime exception
(the `--optimizer adamw|muon` precedent). Worked through concretely, the honest answer is **it's not
really the same knob at all, for case 1**:

- **Case 1 (external frozen table, e.g. real Qwen weights)** is not, and was never proposed to be, a
  `PARAM_LAYOUT` entry in this engine's terms — it isn't trained, isn't checkpointed by this project,
  and is consumed through the new thin-client interface (§7), not through `op_embed`'s existing
  resident-tensor path. There is no `NGRAM_EMBED`-shaped axis to be a runtime alternative *to*, because
  nothing about `ARCH_FINGERPRINT`/checkpoint shape is at stake — a Sub0Llm build that consumes an
  external tiered table is choosing a **different data source for a `Node`'s input**, structurally the
  same kind of decision as which corpus file to point `--corpus` at, not an architecture axis at all.
  So: **no new `if constexpr` toggle is needed for case 1**, and none should be added — it would be
  exactly the "speculative knob nothing reads yet" `AGENTS.md` §8 warns against, since case 1 doesn't
  touch `PARAM_LAYOUT` in the first place.
- **Case 2 (Sub0Llm's own table, hypothetically outgrowing residency)** is the case where this question
  actually bites, and here the answer leans the *other* way from the `--optimizer` precedent: switching
  a *trained, checkpointed* table from resident to externally-tiered **does** change what "the model"
  even means on disk — today `model.bin`/`.ckpt` embed the table's floats directly; a tiered table would
  need `model.bin` to instead carry a *reference* (content hash / table id / cache-tier location) with
  the real floats living outside the checkpoint entirely. That is squarely `AGENTS.md` §3 territory
  (checkpoint-format blast radius), not §2's constexpr-vs-runtime question — and it resolves the same way
  §3's own preferred pattern already does: an **additive, gracefully-degrading trailer field** (the exact
  precedent `docs/NGRAM_EMBEDDING.md` and `docs/GATED_DELTANET.md` both already used for their own new
  trailers), read as "absent = fully resident, the only mode that has ever existed" on every file
  predating this feature. This is explicitly **not scoped for this pass** (§0 already argues case 2 is
  distant), but naming it here matters because it means the eventual toggle, when it exists, is a
  checkpoint-format decision gated behind `AGENTS.md` §3's process, not a `constexpr`-vs-runtime
  argument at all — a different section of the checklist applies, not this one.

### 2f. On-disk location conventions — which convention is Sub0Llm's, which is Sub0Firn's

Sub0Llm already has two real, load-bearing on-disk conventions: per-build-directory generated artifacts
(`out/build/<name>/generated/`, e.g. `tokenizer.stamp`) and per-corpus sidecars living next to the corpus
file itself (`corpus.tok`, `<corpus>.words`). A Sub0Llm-side disk cache tier should sit in the same
family — most naturally a sibling of `corpus.tok` (keyed by the SAME corpus identity `corpus-tok-reuse-stamp`
already tracks, since case-2's working set is a function of the corpus) for the training-time
precomputed-working-set case, or a `generated/`-relative cache directory for anything build-config-scoped
rather than corpus-scoped. **This convention is Sub0Llm's own and must not leak into Sub0Firn** — per the
brief's own instruction and `SUB0FIRN_SPEC.md`'s explicit non-goal, Sub0Firn (a spun-off, portable
library) instead defaults to a platform-appropriate user/system cache directory (`XDG_CACHE_HOME` on
Linux, `%LOCALAPPDATA%` on Windows, `~/Library/Caches` on macOS — the same three-way split most portable
cache libraries already use), configurable, with Sub0Llm's own build simply pointing its Sub0Firn client
at a Sub0Llm-chosen path (most naturally under `out/build/<name>/generated/` or a corpus-adjacent
directory) rather than Sub0Firn inventing or assuming that structure itself.

## 3. Concrete ladder proposal

All hit-rate/capacity numbers below are **back-of-envelope estimates**, explicitly not measurements — no
real access trace exists yet for either the real Qwen table or a hypothetical scaled-up Sub0Llm table,
matching the brief's own instruction to state them as estimates.

| Rung | What lives there | What evicts what | Miss cost | Capacity @ this machine | Estimated hit rate |
|---|---|---|---|---|---|
| **VRAM hot subset** | The most-recently/most-frequently touched rows, for a build actually running inference/training on the GPU backend | Frequency-leaning (§2c), evicted to the RAM tier below (never dropped outright — the RAM tier is the backstop) | A miss here just means "read from RAM tier instead" — a normal PCIe copy, not a stall the CPU-only default build ever sees at all (this rung only matters once a CUDA consumer exists — none does yet for `NGRAM_EMBED`, per `docs/NGRAM_EMBEDDING.md` §7's CUDA guard) | ~8GB VRAM total, shared with activations/model weights — plausibly **0.5–2GB** available for this rung. At Qwen's real bf16 row size (320B), 1GB ≈ **3.3M rows (~1% of the 320M-row table)** | The frequency-aware-cache citation (§1) reports "decent" DLRM training speed at exactly ~1.5% GPU residency — a real anchor for this rung's rough size, but that number comes from Zipfian categorical-id access; splitmix64-hashed n-gram buckets (§3 below) are deliberately spread more uniformly to avoid collisions, so treat 1% as an **optimistic upper bound**, not an expected outcome, for this table specifically |
| **RAM working-set cache** | The precomputed corpus working set (§2b, training) or the reactive frequency/recency cache (§2c, decode) | Same policy as VRAM tier, one level coarser; for training, ideally **nothing evicts anything** because the working set was sized to fit (§2b's "one bulk warm-up, then resident for the run" outcome) | A miss here falls through to disk or network — tens of µs (NVMe) to tens of ms (network RTT), see below | 63GB total machine RAM, shared with the OS, corpus mmap, activations, everything else in the process — plausibly **16–24GB** available. At 24GB / 320B/row ≈ **~75M rows (~23% of the table)** | For a training corpus whose true n-gram working set is smaller than this budget: **effectively 100%** after warm-up (§2b). For decode against an open-ended prompt distribution: unmeasured, plausibly in the 60–85% range by loose analogy to the MoE-Infinity/frequency-cache citations (§1), but this specific number is the least trustworthy in this table |
| **Local disk/NVMe cache tier** | Either (a) a complete local mirror of the source table (§2d's case (a)), or (b) a growing partial cache fed by remote fetches (§2d's case (b)) | LRU or LFU over disk-tier entries once the tier is deliberately kept smaller than the full table (case (b) only); no eviction needed for case (a) | NVMe random 4KB-class read: tens of µs per row at this row's real size (320–640 bytes, well under one NVMe page) — cheap enough that a resolve pass hitting this tier for a whole window's worth of misses is still fast relative to a training step | Bounded only by disk space the user allocates — could be the **full 102.4GB** (a one-time download, case (a)) or a deliberately smaller cache (case (b)) | Case (a): 100% by construction (no miss possible below this tier). Case (b): depends entirely on how the disk-tier cache itself is sized/evicted — same open question as the RAM tier, one level down |
| **Remote source** | The real, unmodified `Qwen/Qwen3.8-Flash-Next` safetensors shards on Hugging Face, accessed via HTTP Range (§2d's proven Stage 1 mechanism) | N/A — this is the source of truth, nothing evicts it | Network-RTT-dominated (tens of ms per request), **not** payload-size-dominated (a 320-byte row is trivial next to typical RTT) — and see the honest limitation below | N/A (the full table, by definition) | N/A |

**One honest limitation worth flagging explicitly**: Bandana's core technique (co-locate co-accessed
rows physically, §1) cannot be applied to the **remote** tier as-is, because the real Qwen checkpoint's
on-disk row order is hash-bucket order (a row's physical position is a function of the hash formula's
output, unrelated to which OTHER rows tend to be read alongside it) — reading a batch of Range requests
for rows a training window actually needs will typically scatter across many of the 128 shards, not
cluster into a few. Bandana-style co-access reordering only becomes available **after** a local copy
exists to rewrite (i.e. it is an optimization available to disk-tier case (a)/(b) once populated, not to
the remote tier itself) — worth stating as a genuine limitation this design does not solve for the
network leg, not glossed over.

## 4. Interaction with `NGRAM_EMBED` Stage 1 and the real-weight fixtures — what changes, what doesn't

- **`docs/NGRAM_EMBEDDING.md`'s Stage 1 CPU op (landed on `main`) does not change.** `op_embed` still
  reads a flat span of floats by index; this design only proposes changing *what span that is* — a
  resolved working-set buffer instead of a resident `PARAM_LAYOUT` tensor, for case-1-shaped consumers
  only. Sub0Llm's own trained table, and every existing test, is unaffected.
- **`tests/fixtures/qwen4_preview/ngram_embedding_*` is the natural first prototype target**, not because
  this design proposes building the real 102GB pipeline first, but because it already has real row
  values, real row indices, and a real known-correct flattened output (`docs/QWEN4_PREVIEW_REFERENCE.md`'s
  Stage 1) — a future Stage 1 of *this* plan (§6) could build the resolve-pass/thin-client contract
  against exactly those 96 real rows (16 heads × 6 positions) as a closed, offline correctness fixture,
  with zero network dependency and zero need for the real 320M-row table to exist anywhere, before ever
  touching a live HTTP Range request.
- **`persistent_slots_engine_tests.cpp`'s guard (`docs/NGRAM_EMBEDDING.md` §5, ids `>= VOCAB` hash as "no
  signal")** is unaffected — it is a property of the hash-id computation, which stays exactly where it is
  today (pure index arithmetic in `forward()`/`forward_one()`), upstream of wherever the resolve step
  gets inserted.

## 5. Staged implementation plan (design only)

Each stage names whether it is buildable as **Sub0Firn standalone, zero Sub0Llm dependency** or requires
**Sub0Llm integration**, per the brief's spin-off framing.

- **Stage 0 — Sub0Firn: the engine-agnostic interface + an in-memory reference implementation.**
  `resolve`/`resolve_many`/`prefetch`/`try_get` (exact signatures in `SUB0FIRN_SPEC.md`) implemented
  against a plain in-process hash map with no tiering at all — i.e. "the contract, proven with the
  simplest possible backend." Exit condition: a unit test registers a small synthetic table, resolves a
  batch of rows, and gets back exactly what was registered — no I/O, no async, no tiers yet. **Buildable
  standalone.**
- **Stage 1 — Sub0Firn: local-file tiers (RAM cache + local disk mirror), no network.** Implements the
  RAM working-set cache and a local-disk case-(a)-shaped tier (§2d) against a synthetic or
  hand-constructed flat table file. Exit condition: a working set that exceeds the RAM budget correctly
  falls through to the disk tier and returns identical values to Stage 0's reference; a working set that
  fits the RAM budget never touches disk after warm-up (§2b's headline claim, made checkable against a
  synthetic corpus of known working-set size). **Buildable standalone.**
- **Stage 2 — Sub0Firn: remote tier (HTTP Range) + disk-tier-as-cache-in-front (case (b)).** Generalizes
  `docs/QWEN4_PREVIEW_REFERENCE.md`'s proven Stage 1 extraction script (§2d) into the reusable remote leg.
  Exit condition: resolving the real 96 rows in `tests/fixtures/qwen4_preview/ngram_embedding_*` via live
  HTTP Range requests against the real Qwen checkpoint reproduces the fixture's `ngram_embedding_per_head.bin`
  bit-for-bit — the same "real-weight fixture as correctness gate" discipline `docs/GATED_DELTANET.md`
  §5 step 2 already established, applied to a serving-infra correctness question instead of a math one.
  **Buildable standalone** (needs network access and the public Hugging Face repo, nothing Sub0Llm-side).
- **Stage 3 — Sub0Llm integration: thin-client op + resolve-pass wiring.** Sub0Llm vendors the Sub0Firn
  client (§7's "how Sub0Llm consumes it"), and `Model::forward`'s existing n-gram block (§2a) gets an
  explicit resolve call inserted before its `op_embed`/`op_linear`/`op_add` composition, reading from the
  resolved buffer instead of a resident tensor — for an EXTERNAL table only; Sub0Llm's own trained table
  is untouched (§2e). Exit condition: with an external-table build pointed at a small local test table
  (not the real 102GB one), the composed pipeline's output matches what the SAME table, if it had been a
  resident `PARAM_LAYOUT` tensor instead, would have produced — i.e. a differential test proving the
  resolve indirection is transparent to the math, the direct analogue of `docs/NGRAM_EMBEDDING.md` §8's
  own "neutral-setting" bit-identical checks. **Requires Sub0Llm.**
- **Stage 4 — Sub0Llm integration: corpus-aware working-set precompute (§2b).** The `ngram_workingset.stamp`
  mechanism, wired into `sub0llm-configure` next to `tokenizer.stamp`. Exit condition: re-running configure
  with only an unrelated flag changed hits the stamp and skips recomputation (mirroring
  `corpus-tok-reuse-stamp`'s own validated re-run timing test); changing `NGRAM_TABLE_SIZE`/`NGRAM_MAX_N`/
  `NGRAM_TABLES_PER_ORDER` (or, for an external table, its version identifier) correctly invalidates it.
  **Requires Sub0Llm.**
- **Stage 5 — Sub0Llm integration: decode-path resolve (§2c).** The one-step-ahead resolve call inserted
  between token-sample and `forward_one`, using Stage 3's same thin-client contract with a reactive
  (frequency-leaning) cache policy instead of Stage 4's precomputed one. Exit condition: a generation
  session against a small local test table produces token-for-token identical output to the same session
  run with the table fully resident (again, a differential/transparency test, not a new numerical
  property — the math doesn't change, only where the bytes come from). **Requires Sub0Llm.**
- **Not scoped by this plan at all**: actually pointing Stage 3/5 at the real 102GB Qwen table end-to-end
  (a real distillation-teacher-signal use case, not an infra correctness question) — that is downstream
  of this plan being built, not part of it; a bulk local mirror vs. remain-remote-with-cache decision
  (§2d/§3) is a user/workflow choice to make once Stage 2 exists, not a design decision this doc needs to
  pre-commit to.

## 6. Explicitly deferred (not silently dropped)

- **Case 2's checkpoint-format work** (§2e) — naming the additive-trailer shape is in scope; actually
  implementing it is not, per §0's distance argument. Revisit once a real Sub0Llm `NGRAM_TABLE_SIZE`
  sweep shows residency actually becoming a problem, not before.
- **An out-of-process daemon / system-level shared cache** (§7's integration-seam discussion) — named as
  a real future option, not designed in detail or scheduled into the staged plan above; the staged plan
  only ever needs an in-process, vendored client through Stage 5.
- **Compression (TT-Rec-style) as an alternative to tiering** — noted as orthogonal (§1, §5c reasoning)
  but not designed; would only become relevant to Sub0Llm's *own* future large table (case 2, TT-Rec's
  own regime), not to serving an already-dense, already-trained external table like Qwen's without a
  separate, nontrivial dense→TT decomposition step this doc does not scope.
- **Bandana-style co-access physical reordering of a local disk-tier copy** — named as the right long-term
  answer for the disk tier's read amplification (§3's limitation note) but explicitly not designed this
  pass; needs a real access trace to drive the hypergraph partitioning Bandana itself depends on, which
  does not exist without a working Stage 1–2 prototype first.
- **A live measurement of any hit-rate number in §3** — every number in that table is an estimate,
  restated here for emphasis per the brief's own instruction not to present them as anything else.

## 7. Novel to this engine vs. reuses an existing idiom

**Reuses an existing idiom, directly**: the "resolve pass before the hot op, not a branch inside it"
shape (`AGENTS.md` §1, already the pattern for e.g. `op_attn`'s scratch allocation); the
stamp-invalidation pattern for a derived, cacheable artifact (`corpus-tok-reuse-stamp`, extended in §2b
rather than reinvented); the "read exact bytes via a computed offset from a header" pattern (`gguf.hpp`,
extended in §2d); the additive/gracefully-degrading trailer-field pattern for a hypothetical future
checkpoint change (§2e, already used twice by `docs/NGRAM_EMBEDDING.md` and `docs/GATED_DELTANET.md`).

**Novel to this engine, this pass**: the corpus-aware working-set precompute as a *first-class
alternative to reactive caching* (§2b) — not present in any of the cited prior art, all of which assumes
online/reactive access patterns because none of those systems' training data is knowable upfront the way
a fixed, already-tokenized training corpus is; the observation that n-gram-hash lookup's fixed,
input-derived (not learned-router-derived) sparsity pattern makes its prefetch problem strictly *easier*
than MoE expert routing's (§1's MoE-Infinity discussion) despite both being cited as the same shape of
problem; and the decode-time "free" one-step resolve window derived from `forward_one`'s existing
rolling-history buffer (§2c) — a genuine, if narrow, structural consequence of code this project already
has, not present in any external source.

## 8. Where this doc deliberately disagreed with or refined the user's own framing

1. **§0**: the two motivating cases are not equally urgent. Case 1 (the real Qwen table) is the entire
   present pressure; case 2 (Sub0Llm's own table) is 3–4 orders of magnitude away at today's realistic
   `VOCAB` sizes. Both are still designed for (the interface doesn't distinguish them structurally), but
   this doc does not pretend they are equally near-term.
2. **§2a/§2c**: the brief's "prefetch/resolve pass... async/upfront rather than synchronous-in-the-hot-path"
   framing is exactly right for training (§2b), but for decode the honest characterization is narrower —
   there IS a clean explicit call site (satisfying the architectural rule), but not, with the current
   single-layer injection point, genuine latency-hiding via overlap (no independent compute exists yet to
   hide behind it). Said explicitly rather than implied by reusing the training case's stronger framing.
3. **§2e**: the ladder's "resident vs. tiered" choice is *not* a clean instance of the
   `--optimizer adamw|muon` runtime-exception pattern the brief asked to weigh it against — for case 1 it
   isn't an architecture axis at all (no `PARAM_LAYOUT` entry exists to switch), and for case 2 it's a
   checkpoint-format decision (`AGENTS.md` §3), not a constexpr-vs-runtime one. Both conclusions are more
   specific than "it's a legitimate runtime exception," which is what a surface reading of the brief's own
   question might suggest.
