---
name: scratch-slots-vs-rag-analysis
description: "ANSWERED: scratch/persistent-slot mechanism is NOT a RAG precursor (exact-id lookup, no similarity search) -- closer to Memory-Augmented NNs (NTM/Memory Networks). Permanence should be a CONTINUOUS priority score (PER/ARC/Compressive-Transformer-style), not MemGPT's discrete tiers -- user correction. Open question: pure compression cache vs. embellished semantic memory, unresolved."
metadata:
  type: project
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
---

BACKLOG question, answered (user, 2026-07-16): "Is the idea of the scratch token database effectively a
precursor/partial RAG implementation? Hopefully it can tackle more than RAG as the db is a changing set
with permanent, mid, and short term entries potentially." Context: [[persistent-slot-range-engine-substrate]],
[[multi-token-pattern-compression-backlog]], [[scratch-tokens-context-translation-layer]],
[[deterministic-mechanisms-thesis]].

## Short answer

**No, not structurally a RAG precursor** — RAG's defining mechanism is *similarity-based retrieval* (a
query embedding matched via nearest-neighbor search against a large, typically-static external corpus).
Sub0Llm's scratch/persistent-slot mechanism has no similarity search anywhere: a slot is looked up by
**exact token-id match** (a symbol-table / key-value lookup), not approximate nearest-neighbor. It's much
closer in lineage to **Memory-Augmented Neural Networks** (Neural Turing Machines, Memory Networks) — an
addressable external memory the model/harness explicitly **writes to and reads from** — than to RAG's
(typically) read-only retrieval over a fixed knowledge base. The "compose bound fragments into a vector"
step (`encode_slot`) IS functionally analogous to RAG/RETRO's "embed the retrieved content" step, but that
one shared step doesn't make the whole mechanism RAG — the addressing/retrieval half is fundamentally
different (exact-symbolic vs. approximate-semantic).

## Prior art (verified via web search 2026-07-16)

- **RAG** (Lewis et al. 2020, NeurIPS, arXiv:2005.11401): a pretrained seq2seq generator + a dense vector
  index over a corpus (Wikipedia in the original paper) accessed via a neural retriever (DPR-style). Two
  formulations: same retrieved passages for a whole generated sequence, or different passages per token.
  The knowledge base is **external, dense-vector-indexed, and not modified by generation**.
- **RETRO** (Borgeaud et al. 2021, DeepMind, arXiv:2112.04426) — the closest "RAG done via a dedicated
  architectural mechanism" comparison: a retrieval database of frozen-BERT-embedded text chunks (key =
  embedding, value = chunk + its continuation), nearest-neighbor search, and **chunked cross-attention** to
  incorporate retrieved chunks (not just prepending text — an actual architectural integration point,
  scaled to trillions of tokens). Still fundamentally similarity-based retrieval over a static corpus,
  same core distinction from sub0's exact-id lookup.
- **Neural Turing Machines** (Graves et al. 2014, arXiv:1410.5401) — a controller network + an external
  memory MATRIX + read/write heads with differentiable, content-AND-location-based addressing. The memory
  is dynamically read AND WRITTEN during processing, not a static pre-built index. This is the closest
  architectural lineage to sub0's scratch mechanism: an addressable store the model (or here, the harness)
  actively mutates within a session.
- **Memory Networks** (Weston et al. 2014) — explicitly stores facts as continuous vectors in an external
  memory, attention-indexed for retrieval; unlike NTM, doesn't modify memory contents within an episode
  (more read-mostly, closer to RAG's retrieval half, but still an explicit addressable store, not a
  similarity index over an external corpus).
- **MemGPT** (2025-era; see also the broader "Memory for Autonomous LLM Agents" survey literature) — the
  most directly relevant prior art for the user's "permanent/mid/short-term" framing. A TWO-TIER
  architecture: Tier 1 = main/working context (in-context, small, always visible — "core memory"); Tier 2 =
  external context, itself split into **recall storage** (a searchable log of past interactions) and
  **archival storage** (a vector-indexed long-term store). The agent (the LLM itself) manages promotion/
  demotion between tiers via explicit function calls (`archival_memory_search`, `core_memory_append`, etc.),
  described as an OS-inspired virtual-memory analogy (RAM vs. disk). This is a live, actively-researched
  design pattern for agentic LLM memory, not a one-off idea.

## What matches the user's intuition, and what would need genuinely new work

**CORRECTION (user, 2026-07-16, same day): MemGPT's discrete tiers are the WRONG reference model.** The
user's actual intuition is NOT "three discrete buckets" (short/mid/long) — it's that permanence should be a
**continuous scale**, more like a priority queue: entries that are important, get reinforced, or get reused
stay resident; everything else decays/gets evicted, with no hard tier boundary. MemGPT's core/recall/archival
split is a real, citable design (kept above for reference), but it's an *engineering* discretization of
what the user is describing conceptually — worth knowing it exists, wrong model to build toward here.

**Better-matched prior art for the CONTINUOUS-priority framing:**
- **Prioritized Experience Replay** (Schaul et al. 2016, ICLR, arXiv:1511.05952) — the cleanest existing
  example of "importance, not age, drives retention": each memory entry gets a continuous priority
  (originally TD-error magnitude) that determines how likely it is to be kept/resampled, no discrete tiers,
  just a priority-weighted queue. The mechanism (not the RL-specific TD-error signal) is the transferable
  idea: some importance SCORE, continuously updated, drives an eviction/promotion decision.
- **Compressive Transformer** (Rae et al. 2019, DeepMind, arXiv:1911.05507) — directly relevant on TWO
  fronts. (1) Structurally, it's the closest existing "aging memory" design: a primary (recent, full-
  fidelity) memory and a compressed (older) memory, with entries COMPRESSED rather than discarded as they
  age out of the primary window — a graceful-degradation model of permanence rather than a hard cliff (the
  entry doesn't vanish, it loses fidelity). (2) The paper explicitly tried a **"most-used" compression
  criterion (sorted by attention usage)** alongside plain age-based FIFO — i.e., it already explored exactly
  the "used/reinforced entries are kept [at higher fidelity]" idea the user is describing. (3) Its
  compression functions were **mean/max pooling and 1D convolution** — the SAME primitive operations as
  sub0's own `SlotEncoding::MeanPool` and the just-built `SlotEncoding::ConvPool` (see
  [[meanpool-alternatives-prior-art-and-math]]) — a concrete, unplanned unification point: the encoders
  built for order-sensitive CONTENT embedding are structurally the same class of function a continuous-
  permanence memory scheme would use to compact an aging entry. Worth remembering if this is ever pursued.
- **Classical cache eviction (LFU / ARC / decay-weighted LFU)** — the boring-but-correct engineering
  vocabulary for "keep what's used, forget what isn't, continuously, not in discrete tiers." An
  Adaptive-Replacement-Cache-style scheme (balances recency AND frequency, adapts the balance over time) is
  probably the most directly implementable STARTING POINT if/when this gets built — it doesn't need any
  ML machinery, just a per-entry decaying usage counter driving eviction order.
- **Spaced-repetition / forgetting-curve models** (Ebbinghaus; SM-2/Anki-style scheduling) — cognitive-
  science grounding for "reinforcement extends persistence continuously," if a psychologically-motivated
  decay function (rather than a pure LRU/LFU counter) is ever wanted.

So: short/mid/long-term is a useful ROUGH VOCABULARY for talking about the substrate that already exists
(ephemeral pool = de facto short-lived; persistent range = de facto permanent-once-written), but the actual
DESIGN, if pursued, should be a continuous priority/decay score per persistent entry (closer to ARC/PER),
not a fixed set of MemGPT-style promotion tiers.

**What's missing to actually subsume/extend RAG** (not just sit adjacent to it): RAG's real value is
answering queries when the exact reference ISN'T known in advance — "find me something relevant to X" via
semantic similarity, over a large candidate set. Sub0's scratch/persistent mechanism has **no similarity
search at all** — every lookup is by exact id, decided in advance by the harness or the model emitting a
specific known symbol. To genuinely extend beyond RAG (a superset, not just a different, complementary
mechanism) would need an ADDED approximate-retrieval layer: e.g., nearest-neighbor search over the
persistent range's content-derived embeddings (the `encode_slot` vectors already computed for
[[meanpool-alternatives-prior-art-and-math]]'s work could plausibly double as retrieval KEYS for such a
search) to find "the closest known persistent entry to this query," not just deref a named one. That
similarity-search mechanism does not exist anywhere in this codebase today — this is real, unscoped future
work, not a small gap.

## OPEN QUESTION (user, 2026-07-16, explicitly unresolved -- do not force an answer): does the scratch
## database embellish entries with useful extra information, or is it purely an optimization/compression cache?

The user is genuinely uncertain here, not asking for a decision. Two poles worth naming so a future
decision has structure to start from, without resolving which one is right:

- **Pure optimization-cache pole**: a persistent id ALWAYS expands back to its exact original fragment
  sequence, losslessly — a memoization/dictionary device (like a compiler's symbol-interning table). No
  semantic value beyond efficiency; the model's knowledge is unchanged, only the cost of referencing
  recurring content drops. Simple, safe, easy to verify (round-trip equality is the whole correctness bar),
  fits this project's [[deterministic-mechanisms-thesis]] "exact compute, verify against reference" culture
  cleanly.
- **Embellished/semantic-memory pole**: a persistent entry carries something BEYOND its literal expansion —
  an aggregated/learned representation of what it means in context, usage statistics, links to related
  entries, or model-derived annotations. Closer to a genuine memory record than a compression symbol. More
  powerful, but introduces real open questions this project would want to scrutinize before committing to:
  what exactly gets embellished, how staleness/drift is verified, whether an enriched entry can silently
  become WRONG in a way a pure lossless cache structurally cannot.

These aren't mutually exclusive (a system could do both — lossless exact expansion plus an attached
enriched vector) but they carry different complexity/risk profiles, and the user has NOT decided which
direction (or blend) is wanted. Revisit this fork explicitly if/when the persistent-slot substrate's actual
write/mining policy ([[persistent-slot-range-engine-substrate]]'s "still not built" list) gets picked up —
don't default to one pole without re-surfacing the question.

## How to apply

Don't describe the scratch/persistent-slot mechanism as "RAG" or "a RAG implementation" in docs/memory — it
overclaims a similarity-retrieval capability that doesn't exist and undersells what it actually is (an
exact-addressable read-write memory, closer to NTM/Memory-Networks lineage). DO use the MemGPT-style
tiered-memory framing (core/working, recall, archival) as the reference architecture if a genuine multi-tier
scratch-slot design is ever pursued — it's real, current prior art that already answers many of the design
questions (promotion/demotion policy, what triggers a write to a given tier, how the model interacts with
tiers it can't see directly) that would otherwise need re-deriving from scratch. Treat "add a similarity-
search layer over the persistent range" as the concrete, scoped next step if the goal is genuinely to reach
RAG-like capability (semantic retrieval), rather than assuming the existing exact-id substrate gets there
for free.
