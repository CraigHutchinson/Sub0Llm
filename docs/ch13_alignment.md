# Chapter 13 — DPO Alignment

## Overview

RLHF (reinforcement learning from human feedback) with PPO requires training a
separate reward model and running an RL loop. **Direct Preference Optimization**
(Rafailov et al., 2023) simplifies alignment: given pairs of (chosen, rejected)
responses, DPO directly optimises the policy without a separate reward model.

The key insight is that the Bradley-Terry preference model can be expressed as a
classification loss on policy log-probabilities, with the reference policy as
an implicit baseline.

## DPO Loss

```
L_DPO = -log σ(β * ((lp_w - ref_lp_w) - (lp_l - ref_lp_l)))

lp_w     = log π_θ(y_w | x)    — log-prob of chosen response under policy
lp_l     = log π_θ(y_l | x)    — log-prob of rejected response under policy
ref_lp_w = log π_ref(y_w | x)  — log-prob of chosen under reference (frozen)
ref_lp_l = log π_ref(y_l | x)  — log-prob of rejected under reference (frozen)
β        — temperature controlling deviation from reference
```

| Margin | Loss | Interpretation |
|--------|------|----------------|
| 0.0 | 0.6931 | Policy treats winner = loser |
| 1.0 | 0.3133 | Policy correctly prefers winner |
| 5.0 | 0.0067 | Strong preference → loss → 0 |
| −2.0 | 1.1269 | Policy incorrectly prefers loser |

## Implicit Reward

After DPO training, the implicit reward function is:
```
r(x, y) = β * (log π_θ(y|x) − log π_ref(y|x))
```

The DPO-trained policy implicitly encodes a reward signal — no separate reward
model is needed.

## API

```cpp
#include "sub0llm/nn/dpo.hpp"
using namespace sub0llm::nn;

// Pre-compute reference log-probs (frozen — computed once, no gradient)
auto ref_logits_w = reference.forward(ids_w);
float ref_lp_w    = log_prob_sequence(ref_logits_w, ids_w)
                        .data().data_as<float>()[0];

// DPO training step
auto logits_w = policy.forward(ids_w);
auto logits_l = policy.forward(ids_l);

auto loss = dpo_loss(
    logits_w, ids_w,        // chosen
    logits_l, ids_l,        // rejected
    ref_lp_w, ref_lp_l,     // reference log-probs (floats, no grad)
    /*beta=*/0.1f);

loss.backward();
adam.step();

// Numerically stable log sigmoid (used internally by dpo_loss)
auto ls = log_sigmoid(variable);   // log(1 / (1 + exp(-x)))
```

## Training Demo (30 steps, V=16, D=16)

Winner: tokens [0,1,2,3,4,5]; Loser: tokens [5,4,3,2,1,0]

```
ref_lp_w = -2.918  ref_lp_l = -2.718

step  0  loss=0.6978
step  5  loss=0.6514
step 10  loss=0.6052
step 20  loss=0.5231
step 29  loss=0.4618  (decreasing ✓)
Final margin (winner − loser relative to ref): +0.3241
```

## DPO vs PPO

| Aspect | PPO | DPO |
|--------|-----|-----|
| Reward model | Explicit (separate model) | Implicit (in loss) |
| RL loop | Yes (policy + value + reward + ref) | No |
| Training complexity | High | Low |
| Reference policy | Needed for KL | Needed for margin |
| Stability | Requires clipping | Directly supervised |

## Files

| Path | Description |
|------|-------------|
| `include/sub0llm/nn/dpo.hpp` | `dpo_loss`, `log_prob_sequence` |
| `src/nn/dpo.cpp` | DPO implementation with numerically stable log-sigmoid |
| `chapters/ch13_alignment/main.cpp` | Demo: §13.1 log_sigmoid, §13.2 DPO anatomy, §13.3 reference log-probs, §13.4 training, §13.5 implicit reward |
| `tests/test_dpo.cpp` | DPO loss and gradient tests |
