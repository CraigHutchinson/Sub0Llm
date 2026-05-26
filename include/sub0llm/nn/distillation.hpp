#pragma once
#include "sub0llm/autograd/variable.hpp"
#include "sub0llm/core/tensor.hpp"

namespace sub0llm::nn {

// Soft cross-entropy for knowledge distillation.
// L = -1/N * sum_{i,v} teacher_probs[i,v] * log_softmax(student_logits)[i,v]
// student_logits: Variable (N, V) — requires grad
// teacher_probs:  Tensor   (N, V) — fixed soft targets (no grad)
[[nodiscard]] autograd::Variable soft_cross_entropy(
    const autograd::Variable& student_logits,
    const Tensor&              teacher_probs);

// Hinton et al. (2015) knowledge-distillation combined loss:
//   L = alpha * CE(student_logits, hard_targets)
//     + (1-alpha) * T^2 * soft_CE(student_logits/T, softmax(teacher_logits/T))
// T^2 corrects gradient magnitudes reduced by temperature scaling.
// student_logits : Variable (N, V)
// teacher_logits : Tensor   (N, V) — inference only; no grad required
// hard_targets   : Tensor   (N,)   int32 class indices
// alpha          : weight on hard-label CE term in [0, 1]
// temp           : distillation temperature (>= 1 recommended)
[[nodiscard]] autograd::Variable distillation_loss(
    const autograd::Variable& student_logits,
    const Tensor&              teacher_logits,
    const Tensor&              hard_targets,
    float                      alpha = 0.5f,
    float                      temp  = 2.0f);

} // namespace sub0llm::nn
