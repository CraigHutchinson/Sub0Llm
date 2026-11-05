---
name: verbosity-time-slider-from-gist
description: "future extension — the gist/coarse-to-fine split enables a meaning-preserving verbosity/time slider (terseness vs elaboration vs latency); global knob now, per-window content-adaptive later"
metadata: 
  node_type: memory
  type: project
  originSessionId: f6f73404-ddcc-4bc1-b640-2b89a8b5c475
---

User idea (2026-06): the ch32 gist/coarse-to-fine hierarchy enables a **verbosity/time slider** that a
flat model can't offer cleanly, because the gist separates *what to say* (`G`, invariant meaning) from
*how much surface to spend* (the fine realization). Two ORTHOGONAL axes:

1. **Refinement depth (iterations K)** — time↑, quality↑, length fixed. Already exists as sampler knobs
   (`max_iters`, `conf_threshold`, `entropy_bound`, `min_commit_frac`). Ceiling: weak models lose to
   one-step refinement (error compounding, [[ch30-iterative-refinement-precondition]]) — gate on
   strength.
2. **Realization length (verbosity N per gist)** — words↑, filler↑, clarity↑(to a point), time↑ but
   PARALLEL across windows. NEW: fix `G`, infill into a terse short canvas or a verbose long one — same
   meaning, different elaboration (plan→surface-realization NLG split, native to diffusion since it
   picks canvas length up front).

Forms: **global slider** (one 0..1 dial → (K, N/Nc); a meaning-preserving `max_tokens`) and the user's
**per-window dynamic** extension — allocate length+iterations per fine window by content density (`G`
already scores each coarse slot): content-rich slots get more tokens/passes, filler regions fewer =
per-window adaptive computation time (Graves ACT), natural since windows are independent and the sampler
already has per-window entropy stopping. Connects to LoopedGPT `forward_k` (Ch17) + thinking budget (Ch16).

Risks: verbosity ≠ quality monotonically (too long → filler/dilution/M2 looping) — use M2
content-recurrence + distinct-n as guardrails to find the info-density knee. Ragged per-window lengths
need padding/bucketing (block-diagonal forward assumes equal T).

Status: **recorded design axis, not built.** Needs HierDenoiser GENERATION first (sampler is
`Denoiser`-only today). Test once it exists: sweep (K,N) for a fixed gist → verbosity/clarity/time
surface (tokens, wall-time, M2, info density), find the knee + default mapping. Full writeup:
`diffusion/chapters/ch32_tree_predictor/DESIGN_REVIEW_3.md` §8. Builds on
[[gist-is-compute-context-primitive]].
