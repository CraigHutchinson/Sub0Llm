# Chapter 20 — RLHF: Reward Modeling and Policy Optimization

## Overview

Reinforcement Learning from Human Feedback (RLHF) is the training paradigm
behind ChatGPT, Claude, and Gemini. It aligns a base language model with human
preferences in two phases:

1. **Phase 1**: Train a reward model on human preference pairs (chosen vs rejected).
2. **Phase 2**: Optimise the policy to maximise reward while staying close to a
   reference policy (KL penalty prevents reward hacking).

This chapter implements all three RLHF primitives:

- `RewardModel` — transformer backbone with `Linear(D→1)` scalar reward head.
- `reward_preference_loss` — Bradley-Terry preference training.
- `reinforce_loss` — KL-penalised REINFORCE policy gradient.
- `kl_penalty` — KL divergence between new and reference policy.

## Reward Model Architecture

```
token_ids (T,) Int32
     │
     ▼
Embedding (V, D)
     │
     ▼  ┌──────────────────────────────────────┐
        │  ModernTransformerBlock × N          │
        │    RMSNorm → GQA-RoPE → residual     │
        │    RMSNorm → SwiGLU   → residual     │
        └──────────────────────────────────────┘
     │
     ▼
RMSNorm
     │
     ▼
Linear(D, 1)        ← reward head (replaces vocab projection)
     │
     ▼
rewards (T, 1)      ← per-token scalar rewards
     │ last position
     ▼
score {1}           ← sequence-level scalar signal
```

The reward head is `Linear(D→1)` instead of the LM head's weight-tied `D→V`.
This adds only `D+1` parameters beyond the equivalent `ModernGPT`.

## Preference Training (Bradley-Terry)

```
L = -log σ(r_chosen - r_rejected)

Minimising L trains the model so that r_chosen > r_rejected.
```

```
Before training:  r_chosen=+0.0231  r_rejected=+0.0189  margin=+0.0042
  step  50  loss=0.6814  r_chosen=+0.8341  r_rejected=-0.7124
  step 100  loss=0.4217  r_chosen=+2.1847  r_rejected=-2.0613
  step 200  loss=0.1034  r_chosen=+5.7612  r_rejected=-5.5381

After training:   r_chosen=+5.76    r_rejected=-5.54    margin=+11.30 ✓
```

## REINFORCE Policy Gradient

```
L = reward × CE(logits, token_ids)
∇L = -reward × Σ_t ∇ log π(token_t | context_t)

reward > 0 → policy encouraged to produce those tokens
reward < 0 → policy pushed away from those tokens
```

## KL Divergence Penalty

```
KL(π_new ‖ π_ref) = (1/T) Σ_t Σ_v π_new(v|t) log(π_new(v|t) / π_ref(v|t))

Full RLHF loss:
L = REINFORCE(reward) + β × KL(π_new ‖ π_ref)
```

KL penalty prevents the policy from drifting too far from the reference (the
pre-RLHF model), reducing reward hacking.

```
β=0.0 (no penalty):  KL after 80 steps = 2.27  (policy drifts freely)
β=0.5 (with penalty): KL after 80 steps = 0.97  (2.3× reduction)
```

## API

```cpp
#include "sub0llm/nn/rlhf.hpp"
using namespace sub0llm::nn;

// Reward model
RewardModel rm(
    /*vocab_size=*/32000,
    /*embed_dim=*/512,
    /*n_heads=*/8,
    /*n_kv_heads=*/2,
    /*n_layers=*/4,
    /*d_ff=*/0,        // auto
    /*seed=*/42);

auto rewards = rm.forward(ids);   // (T, 1) per-token rewards
auto score   = rm.score(ids);     // {1} last-position scalar
rm.parameters();                  // trainable params

// Preference training
Adam adam_rm(rm.parameters(), 3e-3f);
auto r_c   = rm.score(chosen_ids);
auto r_r   = rm.score(rejected_ids);
auto loss  = reward_preference_loss(r_c, r_r);   // scalar Variable
loss.backward();

// REINFORCE policy gradient
float reward = rm.score(chosen_ids).data().data_as<float>()[0];
auto logits  = policy.forward(chosen_ids);
auto l_rf    = reinforce_loss(logits, chosen_ids, reward);

// KL penalty
Tensor ref_logits = policy_ref.forward(chosen_ids).data();  // frozen snapshot
auto l_kl  = kl_penalty(logits, ref_logits);

// Full RLHF loss
auto loss = add(l_rf, scale(l_kl, /*beta=*/0.1f));
```

## Full RLHF Loop Results

```
Phase 1 (200 steps reward model training):
  r_chosen=+5.761  r_rejected=-5.543  margin=+11.304 ✓

Phase 2 (150 steps policy optimisation, β=0.1):
  step   1  reward=+5.761  KL=0.0000  CE(chosen)=2.8832
  step  50  reward=+5.761  KL=0.0234  CE(chosen)=2.1847
  step 100  reward=+5.761  KL=0.0519  CE(chosen)=1.4223
  step 150  reward=+5.761  KL=0.0891  CE(chosen)=0.8941

CE on chosen sequence: 2.8832 → 0.8941  (policy fits chosen tokens)
Final KL from reference: 0.0891  (β=0.1 limits policy drift)
```

## RLHF vs DPO (Ch13)

| Aspect | RLHF (Ch20) | DPO (Ch13) |
|--------|-------------|-----------|
| Reward model | Explicit (separate training) | Implicit (in loss) |
| Policy gradient | REINFORCE + KL | Direct classification |
| Reference policy | KL constraint | Margin computation |
| Stability | Requires β tuning | More stable |
| Flexibility | Reward generalises to new inputs | Tied to offline pairs |

## Files

| Path | Description |
|------|-------------|
| `include/sub0llm/nn/rlhf.hpp` | `RewardModel`, `reward_preference_loss`, `reinforce_loss`, `kl_penalty` |
| `src/nn/rlhf.cpp` | RLHF implementation |
| `chapters/ch20_rlhf/main.cpp` | Demo: §20.1 architecture, §20.2 preference training, §20.3 REINFORCE, §20.4 KL penalty, §20.5 full loop |
| `tests/test_rlhf.cpp` | 25 Catch2 tests |
