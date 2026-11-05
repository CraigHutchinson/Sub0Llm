---
name: deepseek-line-is-orthogonal-to-depth-axis
description: "STRATEGIC — DeepSeek's whole V2->V4 arc is SEQUENCE-axis KV compression + sparsity (MLA, DSA, CSA/HCA), not the depth axis depth-attention/DenseFormer/mHC operate on. Mostly NOT actionable at this project's scale"
metadata: 
  node_type: memory
  type: reference
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
  modified: 2026-07-29T18:48:14.369Z
---

Checked 2026-07-29 while deciding whether depth attention (arm D) is the right bet.

## What DeepSeek actually shipped

A single consistent trajectory, all on the **sequence axis** of attention:

- **MLA** (V2) — projects Q/K/V into a compact latent space; cuts KV-cache size and decode memory
  bandwidth. The strictly-stronger successor to GQA, which this project adopted as GQA-2.
- **DSA** (V3.2) — a lightweight "lightning indexer" scores preceding tokens with a multi-head
  ReLU-gated dot product, takes top-k (k=2048), and runs main attention only over those. O(L²) -> O(Lk).
- **CSA + HCA** (V4, 2026) — interleaves Compressed Sparse Attention (consolidate every m KV entries,
  then sparse top-k over the compressed entries) with Heavily Compressed Attention (larger compression
  rate, dense over the result).

## The point for this project

**None of it touches the depth axis.** Depth attention, DenseFormer's depth-weighted averaging, and
mHC all mix information ACROSS LAYERS. DeepSeek compresses and sparsifies ACROSS POSITIONS. They are
complementary, not competing -- and the frontier's revealed preference is that KV/context cost is where
the returns are AT SCALE.

**At this project's scale that axis is not the constraint.** d192, 512-token window, 3 KV heads: the KV
cache is trivial. So the DeepSeek line is mostly not actionable here -- it is a scale-driven
optimization for a problem Sub0Llm does not have yet. Do not read "DeepSeek didn't do depth attention"
as evidence depth attention is worthless; read it as "they had a bigger fire to put out".

## Two threads that ARE worth following

1. **mHC in V4 -- UNVERIFIED, and high value if true.** A secondary source
   (techjacksolutions.com, a blog, NOT a paper) describes V4 as "MoE & mHC". If DeepSeek really adopted
   manifold-constrained hyper-connections, that is the one place the frontier touches the depth axis,
   and it would promote mHC from a deferred backlog item ([[nanbeige-features-progress]]) to
   frontier-validated. **Verify against the actual V4 paper/model card before acting on it** -- this
   came from a blog, and the project's own rule is to fetch the real source (`AGENTS.md` §5).
2. **Context compression** (DeepSeek-OCR's optical context compression, and the broader
   "end-to-end context compression at scale" line). This is the closest thing in DeepSeek's work to
   [[scratch-tokens-context-translation-layer]] and [[deterministic-mechanisms-thesis]] -- compressing
   context into far fewer carriers. Much more relevant to this project's actual thesis than MLA is.

## Recommendation recorded at the time

Depth attention stays a reasonable bet BECAUSE it is cheap here (zero parameters,
[[depth-attention-stage0-done]]), not because the frontier endorses it. But **DenseFormer is cheaper
still** (learned scalars, no softmax over depth) and has published gains, so it is arguably the better
first experiment on the depth axis. Decide that deliberately rather than defaulting to Nanbeige's
choice because it is the one we happened to read.

Sources: arxiv 2506.02523 (hardware analysis of MLA), Raschka's "Big LLM Architecture Comparison".
