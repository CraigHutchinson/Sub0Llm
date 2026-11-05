---
name: ch28-diffusion-track-plan
description: Ch28+ builds a text-diffusion LM track (DiffusionGemma) as a new specialized project volume
metadata: 
  node_type: memory
  type: project
  originSessionId: 9f29e15e-c786-4872-a953-a7472888b707
---

Ch28 starts a NEW architectural axis: **text diffusion** (DiffusionGemma 26B-A4B-it — Gemma-4
backbone, refines a 256-token canvas in parallel with bidirectional attention, low-confidence
remasking, `entropy_bound` early-stop, block-autoregressive chaining for long outputs).

Locked decisions (2026-06-11):
- **Educational small-model arc FIRST** (Ch28→Ch31); real 26B engine is the Ch32 milestone (won't
  fit RTX 5070's 8GB → CPU/hybrid offload like Ch27's Gemma-12B).
- Lives in a **separate specialized project track** = new top-level `diffusion/` volume linking
  libsub0llm, built around compile-time monomorphization.
- **Hardware-constexpr + `-verify` included now**: extend `sub0llm-specialize` to emit a `HostSpec`
  (P/E core split, RAM, SIMD width) and validate the running machine; see [[ch27-specialized-build-plan]].

Reuse: attention already has a `causal` flag → bidirectional = `causal=false`; MoE (Ch18), autograd
losses, tokenizer, TextCorpus, checkpoint/GGUF, sampler, and the `StaticSpec`/`sub0llm-specialize`
backbone all carry over. New work: forward corruption (noising), time embedding, mixture-of-MLM loss,
iterative reverse sampler, block chaining.

**How to apply:** keep diffusion work ADDITIVE — another agent owns Ch27 hot files
(modern_gpt.hpp, src/backends/cuda/, tools/qbench). Don't refactor those; add new paths in the
diffusion tree. Teach absorbing-state masking first, switch to uniform-state for the real model.
Plan file: C:\Users\craig\.claude\plans\mighty-wishing-boot.md
