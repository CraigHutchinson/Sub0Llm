# Qwen3.8-Flash-Next / Qwen4-preview architecture — reference + validation plan

Status: **research handoff, nothing implemented yet.** This doc exists so the mechanism-replication work
(N-gram embeddings, Gated DeltaNet) has a real, cited source to build against, per `AGENTS.md` §5
("verify precise algorithms against their actual reference source — never from recall alone"). Follow
the `docs/DEPTH_ATTENTION.md` precedent: this is the handover doc, staged status lives here, updated as
each stage lands.

## Why this model, and what "supporting" it means here

Alibaba released `Qwen3.8-Flash-Next` on 2026-08-26, explicitly framed as an early, open-weight preview
of the architecture Qwen4 will be built on — the same pattern used for Qwen3-Next ahead of Qwen3.5. See
[docs/ROADMAP.md](ROADMAP.md)'s 2026-08-30 status entry for why the project is pivoting toward
supporting existing released models rather than further from-scratch training.

**Scale reality check, stated plainly so no later step forgets it**: 125B total / 6B activated params
(MoE) plus a separate 51B-param n-gram embedding table — roughly 176B params stored, ~350GB in bf16.
This engine trains dense/GQA models in the tens-to-hundreds-of-millions-of-params range on a single
Windows workstation (see [[host-cpu-arrow-lake-hx]]; 63GB RAM, one laptop-class ~8GB-VRAM GPU, per the
2026-08-30 hardware check). Training or fully running this model here is not on the table. **What is on
the table**: extracting a handful of real weight tensors (one Gated-DeltaNet layer, the n-gram embedding
table, one QSA layer) and using them — via the model's own real published forward-pass math, run on just
those tensors — as a correctness oracle for this engine's own from-scratch implementations of the same
mechanisms at its own scale. That is a few-hundred-MB extraction and a CPU-only numpy/PyTorch reference
script, not a deployment of the real model.

## Verified facts, with sourcing and confidence

**Caution carried over from [[nanbeige-architecture-reference]]**: an AI-summarized WebFetch of a
marketing/announcement page is not a verified source — several of the numbers below came back consistent
across multiple independent write-ups (a good sign) but **none has been cross-checked against the raw
`config.json` / tech report PDF yet**. Treat every number here as "reported, not yet directly verified"
until Stage 0 below re-derives it from the primary source.

| Fact | Value | Source confidence |
|---|---|---|
| Release date | 2026-08-26 | High — multiple independent news write-ups agree |
| Total / activated params | 125B total, 6B activated per token (MoE) | High — consistent across sources |
| N-gram embedding table | 51B params, 20M bigrams/trigrams, injected at an early layer | Medium — one detailed source ([TechNode](https://technode.com/2026/08/26/alibabas-qwen-to-open-source-qwen3-8-flash-next-previewing-qwen4-architecture/)), matches the *mechanism* (not necessarily every number) of Nanbeige's verified `NanbeigeNgramEmbedding`, see [[nanbeige-architecture-reference]] |
| MoE experts | 512 total (10 routed + 1 shared) | Low-medium — from one AI-summarized model-card fetch, NOT yet cross-checked against `config.json` |
| Layers | 48, hybrid pattern `12 × (3 × (Gated DeltaNet → MoE) → 1 × (QSA → MoE))` (36 linear-attention layers : 12 full-attention layers, 3:1 ratio) | Medium — the 3:1 ratio and layer count are corroborated by the general Qwen3-Next/Qwen3.5 hybrid pattern (independently documented by Raschka, vLLM's blog, and Maxime Labonne — see links below), which this preview is stated to continue |
| Context length | 262,144 native, ~1M via YaRN | High — consistent across sources |
| License | Qwen Community License 1.0 (not MIT/Apache) | Medium — one source |
| Weights | Hugging Face `Qwen/Qwen3.8-Flash-Next`, safetensors (bf16) + ~139 pre-made GGUF quantizations for llama.cpp/LM Studio/Jan/Ollama | Medium — re-verify the exact repo id and file list directly before downloading anything |
| Tech report | Said to be at `github.com/QwenLM/Qwen3.8-Flash-Next/blob/main/tech_report.pdf` | **Low — unverified, re-check this URL exists before citing it further** |

**Gated DeltaNet mechanism (higher confidence — corroborated by real, independent technical write-ups
of the same mechanism in Qwen3-Next/Qwen3.5, which this preview is stated to carry forward)**: linear
attention derived from Mamba2 + the delta rule. Two gates: α (decay gate, controls how fast the memory
state decays/resets) and β (update gate, controls how strongly a new token's input overwrites the
existing state) — the delta rule computes an error-correction term (difference between the new value and
what the current state already predicts) rather than pure additive accumulation, which is what
distinguishes it from plain Mamba2. Scales O(1) per token (recurrent state, not a growing KV cache) vs.
O(n) for softmax attention. Sources: [Sebastian Raschka's Gated DeltaNet writeup](https://sebastianraschka.com/llms-from-scratch/ch04/08_deltanet/),
[vLLM's Qwen3-Next architecture blog](https://vllm.ai/blog/2025-09-11-qwen3-next), [Maxime Labonne's Qwen3.5 attention writeup](https://huggingface.co/blog/mlabonne/qwen35).
**Not yet fetched**: the actual `modeling_qwen3_8_next.py` (or equivalent) source — required before
writing a single line of the engine implementation, per AGENTS.md §5. Do not implement from this
doc's prose description alone; this table is a starting map for where to look, not the reference itself.

**Qwen Sparse Attention (QSA)** — least-verified piece here. Reported to run at "micro-block level"
rather than per-token selection, replacing full softmax attention in the non-GDN layers. No independent
technical write-up found yet (unlike GDN, which several people have already reverse-engineered/explained
for Qwen3-Next). **Treat as open research, not a spec** — deprioritized relative to GDN/n-gram-embeddings
per the 2026-08-30 scope decision (block-sparse attention's benefit is long-context latency, which is not
this engine's current focus; GDN and n-gram embeddings both have a direct existing backlog item and a
verified real reference already in hand via Nanbeige).

## Staged plan

**Stage 0 — verify the real source (do this before any C++).**
1. Fetch the real `config.json` for `Qwen/Qwen3.8-Flash-Next` from Hugging Face directly (raw JSON, not
   an AI-summarized model card) and re-derive every number in the table above from it.
2. Locate and fetch the real modeling code — check `transformers` (if landed), the model repo's own
   `modeling_*.py` if it ships custom code (`trust_remote_code`-style), or the tech-report PDF. Quote the
   real GDN/QSA/n-gram-embedding forward-pass code, the way [[nanbeige-architecture-reference]] quotes
   `modeling_nanbeige.py`.
3. Get the `model.safetensors.index.json` (or GGUF equivalent) and identify exactly which shard(s)
   contain one full Gated-DeltaNet layer's weights and the n-gram embedding table — download ONLY those
   files, not the full ~350GB model. A single small GGUF quant's per-tensor layout works too if metadata
   parsing (this repo's `include/sub0/gguf.hpp`, currently header/metadata-only, see
   [[gguf-import-feasibility-review]]) is extended to read tensor data.

**Stage 1 — build the real-weight validation harness (Python, per [[python-not-committed-cpp-library]] —
local scripts only, never committed).** Load the extracted real tensors into the *real* published module
logic (or a byte-for-byte faithful transcription of it, if the real class can't be instantiated
standalone), run it on a few hand-chosen inputs, and save the (input, output) pairs as small fixture
files this engine's tests can load.

**Stage 2 — implement, off by default, gated against Stage 1's fixtures.** Follow the
`docs/DEPTH_ATTENTION.md` staged precedent exactly: constexpr toggle + `RunConfig` X-macro entry +
`ARCH_FINGERPRINT` bit (Stage 0 there), CPU op + forward/backward (Stage 1), CUDA (Stage 2), each gated
on bit-identical-when-off at two scales before the feature is exercised. **Remember the depth-attention
lesson**: a gradient check alone cannot validate a new op whose no-op form is still differentiable — the
Stage-1 fixtures from real Qwen weights are the actual correctness gate here, stronger than what
depth-attention had available.

- **N-gram embeddings** — start here. Already the next queued Nanbeige-features item
  ([[nanbeige-features-progress]]), already has a verified real reference
  (`NanbeigeNgramEmbedding`/`NanbeigeNgramLayerFusion`, see [[nanbeige-architecture-reference]] §1) that
  predates this preview and matches its stated mechanism closely. Lowest architectural risk of the three.
- **Gated DeltaNet** — new recurrent-state mechanism, no existing Sub0Llm precedent (attention here is
  currently softmax-only). Higher risk: touches the per-layer state/cache design, likely needs new
  checkpoint fields (additive, per AGENTS.md §3) if the recurrent state is meant to persist across
  generation steps like the KV cache does. Scope Stage 0/1 (CPU forward, no training) before committing
  to a backward-pass design.
- **QSA** — documented above as open research; not scheduled yet.
