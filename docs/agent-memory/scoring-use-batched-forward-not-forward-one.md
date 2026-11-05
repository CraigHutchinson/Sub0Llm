---
name: scoring-use-batched-forward-not-forward-one
description: "For scoring a known token sequence (NLL/perplexity), use batched forward(), not token-by-token forward_one()"
metadata: 
  node_type: memory
  type: reference
  originSessionId: 2c2568c0-ff36-4396-a4a1-461041aa521f
---

When you need logits for a *known* sequence (surprisal/NLL scoring, e.g.
`comprehension_pass` in episodic memory), use the batched `ModernGPT::forward()`,
not a `forward_one()` loop.

**Why:** `forward_one()` recomputes the full-vocab LM head (V≈152k) once *per
token* — N separate GEMVs. `forward()` does the LM head for all T positions in a
single `(T,D)·(D,V)` matmul. For scoring, batched is clearly faster. Measured:
swapping `comprehension_pass` to a `forward_one` loop made the episodic probe
*slower* (130s → 175s), so it was reverted.

`forward_one()` + KV cache only wins for **generation**, where future tokens
aren't available and each step is genuinely a single new token. Don't reach for it
just because it skips the autograd graph — the per-token vocab projection costs
more than the graph it avoids. The real speed lever for the scoring path is the
autograd `forward()` itself (its LM head materialises a 155M-element transpose of
the tied embedding per call via `transpose2d`).
