# Chapter 12 — LoRA Fine-Tuning

## Overview

Full fine-tuning updates all parameters of a pre-trained model — expensive for
large models (7B+ parameters). **LoRA** (Low-Rank Adaptation, Hu et al., 2021)
freezes the base weights and adds a pair of tiny low-rank matrices that are
the only parameters that receive gradients:

```
output = base(x) + (α/r) * B @ A @ x

A: (in, r)   — down-project
B: (r, out)  — up-project
ΔW = (α/r) * B @ A
```

B is initialised to **zero**, so the LoRA contribution is exactly zero at the
start of fine-tuning — the model begins from precisely the pre-trained weights.

## Parameter Efficiency

For a D×D weight matrix (D=256):

| Rank | LoRA params | % of full |
|------|-------------|-----------|
| 1 | 512 | 0.78% |
| 2 | 1,024 | 1.56% |
| 4 | 2,048 | 3.13% |
| 8 | 4,096 | 6.25% |
| 16 | 8,192 | 12.50% |
| 32 | 16,384 | 25.00% |

## API

```cpp
#include "sub0llm/nn/lora.hpp"
using namespace sub0llm::nn;

// Create a LoRA-adapted linear layer
LoRALinear layer(
    /*in_features=*/512,
    /*out_features=*/512,
    /*rank=*/4,
    /*alpha=*/1.0f,
    /*seed=*/42);

// Forward: output = base(x) + (alpha/rank) * B @ A @ x
auto out = layer.forward(x);   // (T, out_features)

// Only LoRA parameters receive gradients
auto lora_params = layer.lora_parameters();   // [A, B] — trainable
auto all_params  = layer.all_parameters();    // [base, A, B] — base is frozen

// Verify gradient isolation
sum(out).backward();
layer.all_parameters()[0]->grad().numel();  // == 0  (base frozen)
layer.all_parameters()[1]->grad().numel();  // >  0  (lora_A trainable)

// Train only LoRA params
SGD optim(layer.lora_parameters(), /*lr=*/5e-2f);
```

## Rank Ablation (D=32, V=16, T=8, 100 steps)

```
rank     params  loss[0]   loss[end]  converged?
     1      128   2.7723      0.9124  yes
     2      256   2.7723      0.7831  yes
     4      512   2.7723      0.5942  yes
     8    1,024   2.7723      0.4211  yes
```

Lower rank can converge faster on simple tasks (simpler loss landscape).
Higher rank adds capacity for complex adaptation tasks.

## Comparison with Full Fine-Tuning

| Aspect | Full FT | LoRA |
|--------|---------|------|
| Trainable params | All (100%) | rank×(in+out) (~1–5%) |
| Memory for gradients | Full model | LoRA matrices only |
| Base weight modification | Yes | No (base frozen) |
| Catastrophic forgetting | Risk | Reduced |
| Merge at inference | N/A | ΔW added to base |

## Files

| Path | Description |
|------|-------------|
| `include/sub0llm/nn/lora.hpp` | `LoRALinear` class |
| `src/nn/lora.cpp` | LoRA forward with frozen base |
| `chapters/ch12_finetuning/main.cpp` | Demo: §12.1 param counts, §12.2 zero delta, §12.3 grad isolation, §12.4 fine-tuning, §12.5 rank ablation |
| `tests/test_lora.cpp` | LoRA tests |
