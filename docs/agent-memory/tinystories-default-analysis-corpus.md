---
name: tinystories-default-analysis-corpus
description: "Use TinyStories as the DEFAULT corpus for analysis/experiments; modern coherent English is much easier to verify than Shakespeare. Use Shakespeare only where TinyStories can't cover the property."
metadata: 
  node_type: memory
  type: feedback
  originSessionId: f6f73404-ddcc-4bc1-b640-2b89a8b5c475
---

User preference (2026-06-20): for analysis, experiments, and generation reads, **default to TinyStories** (`data/tinystories_clean.txt`, char-level vocab ~90 / word-level ~12k), not Shakespeare.

**Why:** TinyStories is deliberately simple, coherent modern English designed for small models, so coherence/grammar/quality are **much easier to verify by eye** than Elizabethan verse. Where a result is available on TinyStories, prefer it.

**How to apply:** run new probes/baselines/samples on TinyStories first; only fall back to `data/complete_shakespeare.txt` when TinyStories genuinely can't express the property being tested (e.g. a register/structure it lacks), and say so explicitly. The corpus M3 criticality probe ran on both, but ongoing model-side analysis (M3b generated-text MI, ch32 build-out increments) should be TinyStories-centric. See [[ch29-char-level-run-results]], [[tree-predictor-parked-research-diffusion-training]].
