---
name: tree-predictor-parked-research-diffusion-training
description: Tree-predictor chapter PARKED — flat diffusion works at scale (DiffusionGemma) so it's optional not necessary; research how DiffusionGemma/LLaDA/MDLM train (long-range recipe) before building; balanced-binary-tree prior is theoretically weak for language
metadata:
  node_type: memory
  type: project
  originSessionId: f6f73404-ddcc-4bc1-b640-2b89a8b5c475
---

Decided 2026-06-19. The §13.8 result (flat parallel denoiser is a short-range local interpolator; objective can't fix it — see [[diffusion-contiguous-masking-lever]]) green-lit a structural fix, and a design skeleton was written (`diffusion/chapters/ch32_tree_predictor/README.md`, committed 5f85419). **But then PARKED** — do NOT start building. Reasons (user, three caveats):
1. **Not necessary.** DiffusionGemma-26B ([[diffusiongemma-reference-run]]) is a coherent FLAT masked-diffusion LM ⇒ our local-interpolator limit is a property of OUR small-scale setup/recipe, not of flat diffusion. The tree is a possible NOVEL improvement, not a required fix.
2. **Research prerequisite [DO THIS BEFORE ANY TREE WORK].** Find public info on HOW flat text-diffusion is actually trained at scale — DiffusionGemma, **LLaDA**, **MDLM** (Sahoo), **block/semi-autoregressive diffusion**, SEDD. Their recipe (scale, noise schedule, per-token vs per-window weighting, block/suffix masking, semi-AR ordering) may already supply the long-range coordination we're missing — far cheaper than a new paradigm. Likely the real lever is in there.
3. **Balanced-binary-pyramid prior (ch32 §5-A) is theoretically weak for language** — syntax is unbalanced + n-ary, not a balanced binary tree. Pushes toward real/induced trees (§4-B/C) or the soft-structure variant (§5-C); also a reason to validate the idea against the literature first.

**Unpark when:** the diffusion-training research is done AND (a) the known recipes don't close the §13.8 gap on our setup, and (b) a structural prior still looks like a promising novel lever. ch32 README header records the parked status + unpark criteria.

**REDESIGNED 2026-06-20 (commit 6370b09): the doc is no longer a syntax tree — it's "Unified Multi-Resolution Language."** Distilled from §13.6-13.9 + char-vs-word experiments: language has THREE coexisting levels of abstraction at once — (A) gist/keyword = content-word semantic skeleton ("girl. walk. ball."); (B) prose/word = grammar+function words (fast to learn, OOV-blind); (C) surface/char = spelling+morphology (no OOV, slow to grammar). Meaning: in-vocab word = KNOWN; OOV word = FOUND via char-composition (CharCNN/fastText-style, morphology generalizes) + context/proximity. Level C is the OOV net for B; level A is the semantic plan fighting topic-drift. The HARD part = ONE coherent expression not three bolted models. Candidate unifiers: multi-resolution residual stream (hourglass/CANINE/ByT5/Charformer), multi-scale per-position embedding, or diffusion at WORD level + char-composition embeddings + gist conditioning vector (fits our engine). Still PARKED pending the flat-diffusion training research. Empirically grounded: char (C) works/OOV-proof but slow; word (B) fast grammar but OOV-blind; subword (B↔C bridge) coherent once decoded right (§13.9); remaining gap = semantics/topic-drift = what level A targets.

**RESUME-HERE for the hierarchy track:** `diffusion/chapters/ch32_tree_predictor/BUILD_PLAN.md` → "STATUS / RESUME HERE" section (updated 2026-06-20). Premise VALIDATED (M3 power-law), 4b spread-commit DONE (~30% less looping); NOT started: P1 char-codec → P2 gist (IB-pooling+feudal) → P3 MERA. GPU training now available ([[cuda-first-class-iteration-time]]) so P-phases can iterate fast once ch29 --device cuda is wired.

**Currently running instead:** EXP#2 char-diffusion baseline on TinyStories (`/d/tmp/ch29_tinystories`, 90-char vocab, 21M tokens, scattered masking) — tests whether the local interpolator produces COHERENT text on simple/coherent data (where small models are known to work), separating "long-range modelling" from "Shakespeare is hard." See [[ch29-char-level-run-results]].
