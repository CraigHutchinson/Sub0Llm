# Chapter 19 — Multi-Token Prediction: K+1 Tokens per Forward Pass

## Overview

Standard GPT predicts one token per forward pass. **Multi-Token Prediction**
(MTP) adds K extra output heads that simultaneously predict tokens at offsets
+2, +3, …, +K+1 from the same forward pass.

At training time this provides a richer gradient signal — each position trains
K+1 heads. At inference time, all K+1 predicted tokens can be accepted in one
forward pass, yielding up to (K+1)× throughput over single-token generation.

Unlike speculative decoding there is no separate draft model — the same weights
produce all K+1 predictions simultaneously.

## Architecture

```
token_ids (T,) Int32
     │
     ▼
ModernGPT backbone (shared)
     │
     ├─ head 0 (weight-tied LM head) → logits for token t+1   (standard)
     ├─ head 1 (Linear D→V)          → logits for token t+2
     ├─ head 2 (Linear D→V)          → logits for token t+3
     └─ head K (Linear D→V)          → logits for token t+K+1
```

Each MTP head is a single `Linear(D, V)` projection from the final hidden state.
This is the simple variant; DeepSeek-V3 uses full transformer modules per head.

## Training Loss

```
L = CE(head_0, ids shifted +1) + aux_weight * Σ_{k=1}^{K} CE(head_k, ids shifted +k+1)
```

`aux_weight` controls the balance between main head and MTP auxiliary signal.
Recommended range: 0.01–0.1.

## Parameter Cost

| K | Extra params (D=48, V=98) | vs K=0 |
|---|--------------------------|--------|
| 0 | 0 | baseline |
| 1 | 4,704 (1×D×V) | +8.4% |
| 3 | 14,112 (3×D×V) | +25.1% |

## API

```cpp
#include "sub0llm/nn/mtp.hpp"
using namespace sub0llm::nn;

// Build model with K=3 MTP heads
ModernGPT model(V, D, n_heads, n_kv_heads, n_layers, /*d_ff=*/0, /*K=*/3, seed);

model.n_mtp_heads();   // 3

// Training: compute MTP loss
Tensor ids = ...;   // (T,) Int32
auto loss = mtp_train_loss(model, ids, /*seq_len=*/T, /*aux_weight=*/0.1f);
loss.backward();

// Inference: K+1 tokens per forward pass with stats
std::mt19937 rng(0);
SamplingConfig cfg;   // greedy
MtpGenStats result = mtp_generate_stats(model, prompt, max_new, cfg, rng);

result.tokens;           // full token sequence (prompt + generated)
result.n_forward_passes; // number of model calls
result.tokens_per_pass;  // = generated / n_forward_passes
```

## Inference Speedup (K=3, max_new=20)

```
MTP gen : "inning to get very tired of sitting by"
          (20 tokens, 5 forward passes, 4.0 tokens/pass)

Dense gen: "inning to get very tired of sitting by"
           (20 tokens, 20 forward passes, 1.0 tokens/pass)

Speedup factor (ideal): 4.0×  (K+1=4 vs 1 token/pass)
```

Actual speedup depends on hardware — the extra heads add only O(T×V) compute
vs O(T²×D) attention, so overhead is negligible.

## Aux Weight Sensitivity (K=3, 300 steps)

```
aux_weight=0.01  final_loss=2.3104
aux_weight=0.10  final_loss=2.2891  ← recommended
aux_weight=0.50  final_loss=2.4512
aux_weight=1.00  final_loss=2.6831
```

Small aux_weight (0.01–0.1): main head dominates, MTP provides auxiliary signal.
Large aux_weight: auxiliary heads compete with main head, degrading main CE.

## DeepSeek MTP vs Simple Heads

| Aspect | This implementation | DeepSeek-V3 modules |
|--------|--------------------|--------------------|
| Per-head architecture | Linear(D→V) | RMSNorm → Linear → TransformerBlock → Linear |
| Input | Final hidden state h | h + embedding(token_{t+k}) |
| Expressiveness | Low | High |
| Cost per head | D×V params | Full transformer block |
| Best for | K=1–2, light overhead | K=4–8, better quality |

## Files

| Path | Description |
|------|-------------|
| `include/sub0llm/nn/mtp.hpp` | `mtp_train_loss`, `mtp_generate_stats`, `MtpGenStats` |
| `src/nn/mtp.cpp` | MTP loss with aligned targets, multi-head generation |
| `chapters/ch19_mtp/main.cpp` | Demo: §19.1 architecture, §19.2 training, §19.3 inference, §19.4 aux weight, §19.5 DeepSeek comparison |
| `tests/test_mtp.cpp` | MTP loss validation and generation stats tests |
