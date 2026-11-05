---
name: boundary-aware-coarsening-falsified
description: "Gated (content-weighted) MERA coarsening LOSES to rigid mean-pool (NELBO 1.668 vs 1.514) — word-level spaces are boundary structure, not noise to suppress; keep mean-pool"
metadata: 
  node_type: memory
  type: project
  originSessionId: f6f73404-ddcc-4bc1-b640-2b89a8b5c475
---

FALSIFIED (2026-06-23, full A/B). The boundary-aware / content-weighted coarsening idea — replace MERA's
rigid mean-pool with a learned softmax-over-block gate (`--mera-gated-pool`, zero-init so it starts ==
mean-pool) to down-weight uninformative tokens — **loses to plain mean-pool**.

A/B (identical config: MERA seq128/win16 → levels 128·32·8, truecase, cuda-native, B8): gated tracked
**~0.15-0.30 NELBO worse the ENTIRE run** and plateaued earlier; **best NELBO gated 1.668 vs mean-pool
1.514** (+0.15). The cosine-gist comparison gives the MECHANISM: the gate DID its job — gist top-1==space
fell 25% → 1% (content-only coarse plan) — but that HURT: gist sharpness dropped (mean top-1 cosine 0.366
→ 0.208, blurrier coarse vectors) and NELBO rose. **In word-level tokenization the space token is NOT
noise — it is word-BOUNDARY structure.** The rigid mean-pool's space-aware coarse representation predicts
better than the gated pool's content-only one; suppressing the boundary signal removes useful structure.

Verdict: **keep rigid mean-pool**; don't rebuild the pooling. (Joins [[ch32-char-composition-oov-falsified]]
as a documented negative — both are "an intuitively-appealing representational tweak that removes signal
the simple baseline was using.") The `--mera-gated-pool` flag stays in-tree (off by default) as the
reproducible experiment. The cosine GIST readout ([[gist-is-compute-context-primitive]]) was the
instrument that revealed the mechanism — keep using it to judge coarsening changes. If boundary alignment
is ever revisited, the lever is NOT down-weighting within a rigid block but VARIABLE block boundaries
(syntax-aligned splits) — a different, harder change that preserves the boundary signal.
