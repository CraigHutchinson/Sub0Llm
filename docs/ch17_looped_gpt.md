# Chapter 17 — LoopedGPT: Thinking as Computation Depth

## Overview

LoopedGPT is a Universal Transformer variant that applies a **single
`ModernTransformerBlock` K times** per forward pass rather than stacking K
independent blocks. Every loop reuses the same weights, so the model has
dramatically fewer parameters than a K-layer `ModernGPT` while performing the
same amount of computation per token.

The key idea: **K becomes a reasoning budget**. At training time K is fixed. At
inference time `forward_k(ids, k)` accepts any k ≥ 1, letting you trade latency
for quality without touching the weights.

## Architecture

```
token_ids (T,) Int32
     │
     ▼
tok_emb                  ← Embedding  (V, D), weight-tied with LM head
     │
     ▼  ┌───────────────────────────────────────┐
     └─►│  ModernTransformerBlock (shared)      │◄─ loops K times
        │    RMSNorm → GQA-RoPE → residual      │
        │    RMSNorm → SwiGLU   → residual      │
        └───────────────────────────────────────┘
     │
     ▼
final_norm  (RMSNorm)
     │
     ▼
logits = x · tok_emb_weightᵀ    (T, V)   weight-tied LM head
```

## API

```cpp
#include "sub0llm/nn/looped_gpt.hpp"
using namespace sub0llm::nn;

// Construct: vocab=98, embed=48, 2 query heads, 1 KV head, K=4 loops
LoopedGPT model(98, 48, /*n_heads=*/2, /*n_kv_heads=*/1, /*n_loops=*/4);

// Standard forward — uses training-time loop count (K=4)
auto logits = model.forward(token_ids);       // returns Variable (T, V)

// Inference-time reasoning budget — run with K=8 loops (no retraining needed)
auto logits8 = model.forward_k(token_ids, 8);

// Accessors
model.vocab_size();   // 98
model.embed_dim();    // 48
model.n_loops();      // 4

// Parameters for optimiser
Adam adam(model.parameters(), 3e-3f);
```

## Parameter Efficiency

With `V=98`, `D=48`, `n_heads=2`, `n_kv_heads=1`:

| K | LoopedGPT params | ModernGPT params (K layers) | Ratio |
|---|------------------|-----------------------------|-------|
| 1 |           30,496 |                      30,496 | 100%  |
| 2 |           30,496 |                      56,240 |  54%  |
| 4 |           30,496 |                     107,728 |  28%  |
| 8 |           30,496 |                     210,704 |  14%  |

LoopedGPT's parameter count is constant regardless of K. At K=8 it uses only
14.5% of the parameters of an equivalent 8-layer ModernGPT.

## Example Run

```
Chapter 17 — LoopedGPT: Thinking as Computation Depth
============================================================

=== §17.1  Parameter Count: LoopedGPT vs ModernGPT ===
  K=1  looped=  30496  stacked=  30496  looped/stacked=100.0%
  K=2  looped=  30496  stacked=  56240  looped/stacked=54.2%
  K=4  looped=  30496  stacked= 107728  looped/stacked=28.3%
  K=8  looped=  30496  stacked= 210704  looped/stacked=14.5%
  LoopedGPT amortises the same weights K times — fewer
  parameters, same compute depth per forward pass.

=== §17.2  Training LoopedGPT (K=4 loops) ===
  vocab_size    : 98
  corpus tokens : 403
  parameters    : 30496
  loops         : 4
  step   0  loss=4.6340
  step 100  loss=3.4559
  step 200  loss=3.6783
  step 299  loss=1.9496
  Training: 4.6340 → 1.9496

=== §17.3  Final Loss vs Loop Count (300 steps each) ===
  K=1  params= 30496  final_loss=0.8079
  K=2  params= 30496  final_loss=1.3752
  K=4  params= 30496  final_loss=1.6866
  K=8  params= 30496  final_loss=2.5886
  With a fixed step budget, fewer loops train faster (K=1
  overfits most readily). More loops need more steps to
  propagate gradients through deeper unrolled iterations.

=== §17.4  Inference Loop Budget (forward_k) ===
  Prompt: "Alice was beg"
  K= 1: " so onl m thought Alice we mui"
  K= 2: "inking aveonncWh o oon p"
  K= 4: "inking avingd pink tister o"
  K= 8: "inkingnde twiceeeened"
  K=16: "eeeeeeeeeeeeeee"
  Higher K = more refined hidden state = (potentially) better
  output without changing any weights.

=== §17.5  Loops vs Thinking Tokens (Ch16) ===
  Ch16 (thinking tokens)  Ch17 (looped GPT)
  ─────────────────────   ─────────────────────────────────
  Budget = token count    Budget = loop count K
  Thinking is visible     Thinking is internal (hidden state)
  Uses full vocab space   Uses fixed computation path
  Interpretable trace     Opaque iterative refinement
  Larger KV cache         No extra KV cache
  Supported by RLHF       RLHF trains loop count implicitly

  Both approaches trade off extra compute for quality.
  Thinking tokens expose the chain-of-thought; looped GPT
  hides it inside repeated weight application.

Done.
```

## Training Notes

- **With a fixed step budget, fewer loops converge faster.** K=1 overfits most
  quickly; K=8 needs many more gradient steps to learn good iterative
  refinement. The gradient flows back through all K unrolled applications of the
  block, making the effective depth K× deeper than a single-layer model.

- **More loops at inference-time than at training-time** (e.g., train K=4,
  infer K=16) can diverge — the block was not trained to be applied that many
  times and the hidden state may spiral. See §17.4: K=16 degenerates to
  repeating the same token.

- **Weight tying** between the token embedding and the LM head applies the same
  as in `ModernGPT`: `logits = x · tok_emb_weight^T`.

## Comparison: LoopedGPT vs Thinking Tokens (Ch16)

| Dimension              | Ch16 thinking tokens          | Ch17 LoopedGPT                 |
|------------------------|-------------------------------|--------------------------------|
| Budget unit            | Token count                   | Loop count K                   |
| Thinking visibility    | Visible in sequence           | Internal (hidden state only)   |
| KV cache overhead      | Grows with thinking length    | None                           |
| Interpretability       | Readable chain-of-thought     | Opaque iterative refinement    |
| RLHF compatibility     | Natural (reward on tokens)    | Implicit (reward on output)    |
| Parameter cost         | None (same model)             | None (K does not add params)   |

Both approaches spend extra compute to improve output quality. Thinking tokens
make that computation visible and verifiable; LoopedGPT keeps it internal.

## Files

| Path | Description |
|------|-------------|
| `include/sub0llm/nn/looped_gpt.hpp` | Class declaration |
| `src/nn/looped_gpt.cpp` | Implementation |
| `chapters/ch17_looped_gpt/main.cpp` | Demo program (§17.1–§17.5) |
| `tests/test_looped_gpt.cpp` | 12 Catch2 tests |
