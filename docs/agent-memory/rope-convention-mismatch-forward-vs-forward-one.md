---
name: rope-convention-mismatch-forward-vs-forward-one
description: RESOLVED — ModernGPT RoPE unified on half-split; checkpoints are versioned (v2) and legacy ones are rejected
metadata: 
  node_type: memory
  type: project
  originSessionId: 2c2568c0-ff36-4396-a4a1-461041aa521f
---

**Resolved.** ModernGPT's training `forward()` and inference `forward_one()` once
used different RoPE conventions (interleaved `(2i,2i+1)` vs half-split `(i,i+Dh/2)`),
so their logits diverged for the same weights.

**Fix:** the autograd `rope()` in `src/autograd/ops.cpp` was switched to the
**half-split (NeoX/HF `rotate_half`)** convention — the one GGUF/Qwen3 expects and
that `forward_one()` already used. Both paths now agree (regression test:
`[modern_gpt][rope]` "forward and forward_one agree" in `test_modern_gpt.cpp`).

**Checkpoint guard:** `kCheckpointFormatVersion = 2` in `include/sub0llm/nn/checkpoint.hpp`.
`save_checkpoint` writes `format_version`; `load_checkpoint` rejects anything not
equal to the current version. Pre-versioned files are treated as legacy v1
(interleaved RoPE) and fail to load with a clear message, so old weights can't be
silently restored under the new convention. (No local trained checkpoints existed
when this changed, so breaking v1 compatibility was safe.)

Bump `kCheckpointFormatVersion` again on any future weight-incompatible change.
Related: [[resolve-preexisting-issues-blocking-our-work]].
