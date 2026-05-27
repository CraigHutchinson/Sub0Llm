# Chapter 15 — Real Training and Knowledge Distillation

## Overview

This chapter trains models on a real text corpus (Alice in Wonderland Ch.1,
~400 tokens) and introduces **knowledge distillation** — transferring knowledge
from a large teacher model to a small student model.

The teacher trains to low loss, then the student is trained not just on hard
one-hot targets but also on the teacher's soft probability distributions. Soft
targets reveal inter-class relationships: the teacher's distribution over
"cat", "dog", "rabbit" tells the student more than a single hard label.

## Distillation Loss

```
L_KD = α * CE(student_logits, hard_targets)
     + (1 - α) * T² * CE(student_logits/T, teacher_logits/T)

α   — weight on hard target loss (0 = teacher only, 1 = no distillation)
T   — temperature (T > 1 softens both distributions → reveals more structure)
T²  — rescaling factor that counteracts the softening of the soft loss
```

At `T=2, α=0.5`:
- Hard CE term trains the student on ground-truth labels.
- Soft CE term trains the student to match the teacher's distribution.
- `T² = 4` scale factor makes KD loss larger than hard CE by design.

## API

```cpp
#include "sub0llm/nn/distillation.hpp"
using namespace sub0llm::nn;

// Soft cross-entropy: -Σ p_teacher * log(softmax(student_logits / T))
auto soft_loss = soft_cross_entropy(student_logits, teacher_logits, /*T=*/2.0f);

// Combined distillation loss
auto loss = distillation_loss(
    student_logits,     // (T, V) Variable — differentiable
    teacher_logits,     // (T, V) Tensor  — frozen (no gradient)
    hard_targets,       // (T,) Int32
    /*alpha=*/0.5f,     // weight on hard CE
    /*temperature=*/2.0f);

loss.backward();
```

## Teacher vs Student Architecture (Alice corpus)

| Model | D | Layers | Params |
|-------|---|--------|--------|
| Teacher | 64 | 4 | ~120k |
| Student (base) | 32 | 2 | ~35k |
| Student (distill) | 32 | 2 | ~35k |

## Training Results (300 steps each)

```
§15.2  Teacher Training:
  step   0  loss=4.4012
  step  50  loss=3.1284
  step 150  loss=2.3459
  step 299  loss=1.6742  (converging)

§15.3  Student Baseline (hard labels only):
  step   0  loss=4.4127
  step 299  loss=2.1843

§15.4  Student (distillation, α=0.5, T=2):
  step   0  kd_loss=4.3214
  step 299  kd_loss=1.9231
  Distillation transfers teacher soft-target knowledge to the student.
```

## Inference Comparison (greedy, 20 new tokens)

```
prompt: "Alice was beg"

  teacher:       "inning to get very tired of sitting by her sister"
  student_base:  "inning to get very tired of sitting by her"
  student_dist:  "inning to get very tired of sitting by her sister"
```

The distilled student better matches the teacher's output distribution even
though it has 3.4× fewer parameters.

## Files

| Path | Description |
|------|-------------|
| `include/sub0llm/nn/distillation.hpp` | `soft_cross_entropy`, `distillation_loss` |
| `src/nn/distillation.cpp` | Distillation loss implementation |
| `chapters/ch15_distillation/main.cpp` | Demo: §15.1 corpus, §15.2 teacher, §15.3 student base, §15.4 distillation, §15.5 inference |
| `tests/test_distillation.cpp` | Distillation loss tests |
