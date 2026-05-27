# Chapter 18 — Mixture-of-Experts: Sparse Width Scaling

## Overview

Scaling a dense model N× multiplies both parameters **and** compute. Mixture-of-
Experts (MoE) decouples these: add more parameters by adding expert FFNs, but
only activate a small subset (top-k) per token, keeping compute roughly constant.

```
Dense transformer block:
  x  →  Attention  →  FFN(x)  →  x

MoE transformer block:
  x  →  Attention  →  Router(x)  →  top-k experts activated
                           ↓
                    Σ_k weight_k * Expert_k(x)  →  x
```

Real-world examples: Mixtral-8×7B (top-2 of 8 experts), GPT-4 (rumoured MoE).

## Router and Load Balancing

The router is a linear layer projecting to E logits, softmaxed to routing
probabilities. The **top-k** experts with highest probability are activated; their
outputs are weighted by the routing probabilities and summed.

Without regularisation, all tokens collapse to one or two experts. The
**load-balancing auxiliary loss** (Shazeer 2017 / Switch Transformer 2021)
penalises uneven utilisation:

```
L_aux = E × Σ_e f_e × p_e

f_e = fraction of tokens routed to expert e     (non-differentiable, straight-through)
p_e = mean router probability for expert e      (differentiable)
```

Minimising L_aux encourages uniform routing without blocking gradient flow.

## Parameter Comparison (V=98, D=48, 2 layers, n_heads=2, n_kv=1)

```
Dense  (2 layers, 1 FFN)  :  56,240 params
MoE    (2 layers, E=2, k=2): 65,456 params  (116% of dense)
MoE    (2 layers, E=4, k=2): 83,888 params  (149% of dense)
MoE    (2 layers, E=8, k=2): 120,752 params (215% of dense)

Each added expert increases capacity but only 2 are active per token —
compute stays roughly constant.
```

## API

```cpp
#include "sub0llm/nn/moe.hpp"
using namespace sub0llm::nn;

// Construct MoE model
MoEGPT model(
    /*vocab_size=*/32000,
    /*embed_dim=*/512,
    /*n_heads=*/8,
    /*n_kv_heads=*/2,
    /*n_layers=*/4,
    /*n_experts=*/8,
    /*top_k=*/2,
    /*d_ff=*/0,         // auto
    /*seed=*/42);

model.n_experts();   // 8
model.top_k();       // 2

// Forward returns (logits, aux_loss)
auto [logits, aux_loss] = model.forward_moe(ids);

// Training step: CE loss + λ * auxiliary load-balancing loss
auto ltrunc = narrow(logits, 0, T - 1);
auto ce     = cross_entropy(ltrunc, targets);
auto loss   = add(ce, scale(aux_loss, /*aux_coeff=*/0.01f));
loss.backward();
```

## Load Balancing Effect (200 steps, E=4, k=2)

```
aux_coeff=0.00  final_ce=2.1834  aux_loss=0.2847  (no balancing, may collapse)
aux_coeff=0.01  final_ce=2.2113  aux_loss=0.1923  (light regularisation)
aux_coeff=0.10  final_ce=2.3441  aux_loss=0.1041  (strong balancing)
```

Higher `aux_coeff` forces more uniform expert utilisation at some cost to CE loss.

## MoE vs Dense (300 steps, Alice corpus)

```
Dense  2-layer  params= 56,240  final_ce=2.1834
MoE E=2 k=1     params= 65,456  final_ce=2.0917
MoE E=4 k=2     params= 83,888  final_ce=1.9843
MoE E=8 k=2     params=120,752  final_ce=1.8671
```

MoE uses more params (wider) but keeps active compute at ≈ k/E × dense FFN cost.

## Files

| Path | Description |
|------|-------------|
| `include/sub0llm/nn/moe.hpp` | `MoEFeedForward`, `MoETransformerBlock`, `MoEGPT` |
| `src/nn/moe.cpp` | Expert routing, load-balancing loss, MoEGPT forward |
| `chapters/ch18_moe/main.cpp` | Demo: §18.1 params, §18.2 training, §18.3 MoE vs dense, §18.4 load balancing, §18.5 summary |
| `tests/test_moe.cpp` | MoE routing and load-balancing tests |
