---
name: gist-is-compute-context-primitive
description: "ch32 gist/coarsening is a COMPUTE+CONTEXT scaling primitive (coarse plan -> parallel fine windows), NOT a within-window accuracy fix; judge by runtime/memory/context, not single-window NLL"
metadata: 
  node_type: memory
  type: project
  originSessionId: f6f73404-ddcc-4bc1-b640-2b89a8b5c475
---

The ch32 P2 "gist" (and P3 MERA coarsening — they are the SAME operator at different depths) is a
**coarsening primitive for coarse-to-fine, PARALLEL sub-window generation**, not a per-token accuracy
fix. Coarse pass builds a low-res plan `G` over the full width N; then `M = N/w` fine sub-windows are
denoised with full local `O(w²)` attention **conditioned on the shared `G`, independent given `G` →
embarrassingly parallel.**

**Why it matters (the value proposition):** work `O(N²) → O((N/c)² + N·w)` (~15× less at N=1024,w=64,
c=16); attention memory `O(N²) → O(w²)` per window (~256× less); effective context = N at per-window
cost `w²` (context-window extension). The M windows are the **batch dim of the existing batched
block-diagonal forward** (GPU parallel in one call) and map onto cpu_topology per-window workers — so
it largely reuses existing infra ([[cpu-threading-model]], [[cuda-first-class-iteration-time]]).
Diffusion is GPU-favorable, and batch-of-windows is more parallel than flat decode.

**The 2b retrospective (the miss):** I first measured the gist by single-window NLL (N=w=64) and found
it ~redundant with self-attention (real gist beats a capacity-matched shuffled control by only ~2% in
3/3 seeds, doesn't net-beat baseline). That's correct but tested the WRONG thing — in a **bidirectional
denoiser every position already attends to all visible tokens**, so a gist of *visible* tokens is
redundant *within a window*. Its value is **cross-window** coherence + compute/context at N≫w, which a
single-window test can't see. The within-window redundancy is the FLOOR, not a verdict. Full write-up:
`diffusion/chapters/ch32_tree_predictor/DESIGN_REVIEW_3.md` + `2B_RESULTS.md`.

**It's COMPUTE/CONTEXT only — NOT a coherence win (measured).** 2e benchmark (N 128/256/512):
attn-ops 3.6→10.7× fewer, attn-mem 4→64× smaller, ~2× wall-clock, all growing with N — the win is
real and scales. But accuracy/quality: hier is at-best-parity, currently slightly worse. 2c mask-aware
coarse pooling (pool over visible non-[MASK] tokens) closed the CONTENT-NLL gap (+0.5% at N=512);
remaining NLL gap = FUNCTION words across window seams (→ 2d). And generation M2 (ch32_hier_gen, 3
seeds): flat ≥ hier content-recurrence in 3/3 — **the gist does NOT make generation more coherent**.
Don't claim coherence gains for the hierarchy; its value is efficiency/context.

**How to judge it (corrected gate, BUILD_PLAN Phase 2 re-scoped):** build the coarse-to-fine generator
(step 2e) and benchmark tok/s + memory + max-N vs a flat denoiser at N≫w, plus the flat−hierarchical
NLL / M2-content-recurrence gap. Levers to shrink the gap: 2c IB-pooling (retain long-range info in G)
and boundary/seam fixes (halo overlap, edge conditioning, global refine — Ch28 measured edges recover
~40% vs 62% interior, [[ch28-curriculum-findings]]). Accuracy story fits M3 power-law (local attention
+ thin long-range tail carried by G). Relates to [[ch28-diffusion-track-plan]],
[[tree-predictor-parked-research-diffusion-training]].
