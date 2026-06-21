# Ch32 P1 — Character-composition codec: results

**Status: 1a + 1b DONE and validated (round-trip + order-sensitivity). 1c (wire into the
word Denoiser, OOV A/B) next.**

P1 is the headline increment of the hierarchy build (BUILD_PLAN §Phase 1): fold level-C
(sub-word spelling) into the word representation so a word-level diffusion model is no longer
helpless on a word it has never seen. The codec gives every spelling — in-vocab or OOV — a
content-addressed vector.

Code: [`sub0diff/nn/char_codec.hpp`](../../include/sub0diff/nn/char_codec.hpp).
Test (validation): `CharComposer/CharDecoder round-trip spellings (P1 1a/1b)` in
[`test_diffusion.cpp`](../../tests/test_diffusion.cpp) — tag `[char_codec]`.

## What was built

| step | block | design | status |
|------|-------|--------|--------|
| 1a | `CharComposer` | char-ids → bidirectional blocks (order-sensitive, RoPE) → **mean-pool** → (1,D). Reuses the Denoiser's `BidirectionalBlock`; pooling/broadcast via `matmul` with constant ones-vectors → **no new autograd ops**. | ✅ |
| 1b | `CharDecoder` | (1,D) word vector **+ learned positional queries** → bidirectional blocks → char logits (weight-tied to a char embedding). Non-autoregressive (the diffusion idiom). | ✅ |
| —  | `char_recon_loss` | autoencoder loss for one word; device-agnostic (`weighted_cross_entropy`) so it trains on CPU **or** GPU. | ✅ |

## Validation (actual test output)

Overfit a 10-word set (`cat dog god run sun the star tree moon fish`), D=32, 2 layers,
Adam 3e-3, 400 steps. Asserted and **passing** (`All tests passed (3 assertions)`):

1. **Learns** — final loss < first loss (autoencoder converges).
2. **Round-trips the spelling** — argmax of the decoded char logits matches the input
   spelling at **> 90 %** of positions (vs 3.7 % chance for 27 symbols).
3. **Order-sensitive** — `dog` and `god` (same letters, different order) compose to
   **distinct** vectors (cosine < 0.999). The composed vector is a real word
   representation, not a bag of letters — the property the OOV fix depends on.

## Key finding — the decoder needs position queries

First attempt (decoder = broadcast the single word vector to L slots, differentiate only by
RoPE) reconstructed just **35 %** — RoPE on identical rows is too weak to spell. Adding a
**learned positional query** per slot (each output position gets a distinct learnable input,
conditioned by the word vector) took it to **> 90 %**. This is the standard parallel-decode-
from-a-latent pattern and is the load-bearing design choice for non-autoregressive spelling.

## Cost note (why this is now cheap to iterate)

The codec trains on GPU via the same engine as the Denoiser. GPU training became practical
this milestone: the **CUDA caching allocator** removed the per-op `cudaMalloc`/`cudaFree`
sync that made GPU training allocation-bound — controlled bench **0.0635 → 0.0158 s/step
(4× faster, 23.9× vs CPU)**. See [`SPECIALIZATION_ROADMAP.md`](../../SPECIALIZATION_ROADMAP.md).

## Next — 1c (the OOV A/B, the actual kill-test)

Wire the codec around the word `Denoiser` (compose-in for the embedding, decode-out for OOV
words), with a lookup/composed **gated blend**, and measure on TinyStories:

- **M1 (OOV-cliff)** must move toward ~1: `NLL_oov / NLL_invocab` for held-out word types.
- In-vocab NLL must **not regress** vs plain word-level.
- Grammar speed ≈ word-level (~2–4k steps), not char-level pace.

Kills the design if: the OOV cliff persists, OR in-vocab regresses, OR grammar slows to
char-level pace. Phase-0 **M1** metric is the gate and is the immediate next build.
