# Chapter 11 — Pretraining with MTP

## Overview

With the modern architecture (Ch10) and Adam (Ch09) in place, this chapter
assembles a complete pretraining loop. Three new components are added:

1. **Learning-rate schedules** — cosine decay with linear warmup.
2. **`narrow` op** — differentiable row-slicing of a Variable, needed to align
   logits and targets when using MTP heads.
3. **`mtp_cross_entropy`** — weighted sum of CE losses across all K+1 MTP heads.

## Learning Rate Schedules

### Cosine with Linear Warmup

```
step < warmup_steps:  lr = max_lr * (step / warmup_steps)
step ≥ warmup_steps:  lr = min_lr + 0.5*(max_lr-min_lr) * (1 + cos(π*t))
                        where t = (step - warmup) / (total - warmup)
```

Linear warmup prevents loss spikes from large gradients at the start of training.
Cosine decay gives a smooth final approach to `min_lr`.

```
  step       lr
     0     0.00e+00  (warmup start)
     5     1.50e-04  (mid warmup)
    10     3.00e-04  (peak)
    25     2.57e-04  (cosine decay)
    50     1.65e-04
    75     3.43e-05
    99     3.00e-05  (min_lr)
```

## `narrow` Op

```cpp
// Extract rows [start, start+length) from a (T, D) Variable
auto y = narrow(x, /*start=*/1, /*length=*/3);   // (3, D) from (5, D)

// Backward: scatter gradient back to rows [1, 4); rows outside get grad=0
```

Used to align logit rows with target tokens when the sequence has T tokens but
only T−1 prediction pairs.

## MTP Cross-Entropy

```
mtp_cross_entropy(heads, ids, weights=[1/K, 1/K, ...])

L = Σ_k weights[k] * CE(head_k_logits, ids shifted by k+1)
```

Main head (k=0) predicts t+1, MTP head k predicts t+k+1. Default weights
distribute equally across all K+1 heads.

## API

```cpp
#include "sub0llm/nn/scheduler.hpp"
#include "sub0llm/nn/trainer.hpp"
using namespace sub0llm::nn;

// LR schedule
CosineWithWarmup sched(/*max_lr=*/3e-4f, /*min_lr=*/3e-5f,
                        /*warmup_steps=*/100, /*total_steps=*/1000);
float lr = sched(step);   // current learning rate

ConstantLR const_sched(3e-4f);   // no decay

// Narrow op
auto y = narrow(x, /*start=*/0, /*length=*/T - 1);   // drop last row

// MTP loss
auto heads = model.forward_mtp(ids);        // K+1 Variables
auto loss  = mtp_cross_entropy(heads, ids); // scalar Variable

// Custom head weights (main head gets more weight)
std::vector<float> weights = {1.0f, 0.1f, 0.1f};
auto loss2 = mtp_cross_entropy(heads, ids, weights);
```

## MTP Head Ablation (30 steps, V=32, D=32, T=12)

```
n_mtp=0: params= 49,728  loss 3.2891 → 2.3044
n_mtp=1: params= 52,736  loss 3.2891 → 2.1923
n_mtp=2: params= 55,744  loss 3.2891 → 1.9876
```

MTP typically accelerates early convergence at minimal parameter cost.

## Files

| Path | Description |
|------|-------------|
| `include/sub0llm/nn/scheduler.hpp` | `CosineWithWarmup`, `ConstantLR` |
| `include/sub0llm/nn/trainer.hpp` | `Trainer` helper (wraps training loop) |
| `src/nn/mtp.cpp` | `mtp_cross_entropy`, `narrow` backward |
| `chapters/ch11_pretraining/main.cpp` | Demo: §11.1 LR schedule, §11.2 narrow, §11.3 MTP loss, §11.4 full loop, §11.5 ablation |
| `tests/test_pretraining.cpp` | Scheduler and MTP loss tests |
