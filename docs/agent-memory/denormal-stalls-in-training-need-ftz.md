---
name: denormal-stalls-in-training-need-ftz
description: Training binaries slow down 4x+ over a run from x86 denormal stalls in Adam state; set FTZ+DAZ at startup
metadata: 
  node_type: memory
  type: project
  originSessionId: 9f29e15e-c786-4872-a953-a7472888b707
---

CPU training throughput decays as loss falls (Ch28 diffusion: 254 → 65 steps/s over 19 min) because
Adam's second-moment EMA (×0.999) drifts into denormal float range and x86 denormal ops cost
~50-100× microcode penalty. The native preset uses `-ffp-contract=fast` but NOT `-ffast-math`, so
flush-to-zero is off by default.

**Why:** the slowdown tracks training *progress*, not time or memory — flat RSS rules out leaks.
Fix verified 2026-06-12: `_mm_setcsr(_mm_getcsr() | 0x8040)` (FTZ+DAZ) at top of main() made the
Ch28 run flat ~390 steps/s, 1140s → 256s (4.4×), identical 14/14 recovery.

**How to apply:** any new training main() needs FTZ+DAZ set at startup; better long-term home is a
shared `sub0llm::runtime_init()` (or the trainer) so all chapters get it. Inference parity binaries
should consider whether FTZ affects logit-exactness vs llama.cpp before enabling ([[verify-correctness-against-reference-before-perf]]).
