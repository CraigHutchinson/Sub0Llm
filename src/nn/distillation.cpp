#include "sub0llm/nn/distillation.hpp"

#include "sub0llm/autograd/ops.hpp"

#include <format>
#include <stdexcept>

namespace sub0llm::nn {

using namespace sub0llm::autograd;

autograd::Variable soft_cross_entropy(
    const autograd::Variable& student_logits,
    const Tensor&              teacher_probs)
{
    const auto& sd = student_logits.data();
    if (sd.ndim() != 2)
        throw std::runtime_error(std::format(
            "soft_cross_entropy: student_logits must be 2D (N,V), got {}D", sd.ndim()));
    if (teacher_probs.ndim() != 2)
        throw std::runtime_error(std::format(
            "soft_cross_entropy: teacher_probs must be 2D (N,V), got {}D", teacher_probs.ndim()));

    const int64_t N = sd.shape(0);
    const int64_t V = sd.shape(1);

    if (N == 0)
        throw std::runtime_error("soft_cross_entropy: batch size N must be > 0");
    if (teacher_probs.shape(0) != N || teacher_probs.shape(1) != V)
        throw std::runtime_error(std::format(
            "soft_cross_entropy: shape mismatch — student ({},{}) vs teacher ({},{})",
            N, V, teacher_probs.shape(0), teacher_probs.shape(1)));

    Variable teacher_var(teacher_probs, false);
    // L = -1/N * sum_{i,v} teacher[i,v] * log_softmax(student)[i,v]
    Variable log_sm    = log_softmax(student_logits);
    Variable products  = mul(teacher_var, log_sm);
    Variable total     = sum(products);
    return scale(total, -1.0f / static_cast<float>(N));
}

autograd::Variable distillation_loss(
    const autograd::Variable& student_logits,
    const Tensor&              teacher_logits,
    const Tensor&              hard_targets,
    float                      alpha,
    float                      temp)
{
    if (alpha < 0.0f || alpha > 1.0f)
        throw std::runtime_error(std::format(
            "distillation_loss: alpha={} must be in [0, 1]", alpha));
    if (temp <= 0.0f)
        throw std::runtime_error(std::format(
            "distillation_loss: temp={} must be positive", temp));

    if (teacher_logits.ndim() != 2)
        throw std::runtime_error(std::format(
            "distillation_loss: teacher_logits must be 2D (N,V), got {}D",
            teacher_logits.ndim()));

    // Short-circuit: pure hard CE
    if (alpha == 1.0f)
        return cross_entropy(student_logits, hard_targets);

    // Soft targets: softmax(teacher_logits / temp)
    Variable teacher_var(teacher_logits, false);
    Variable teacher_scaled = scale(teacher_var, 1.0f / temp);
    // Use softmax to get soft probabilities, then extract as Tensor (no grad needed)
    Variable soft_probs_var  = softmax(teacher_scaled);
    Tensor   teacher_probs   = soft_probs_var.data();

    // Scale student logits by 1/T
    Variable student_scaled = scale(student_logits, 1.0f / temp);

    // Soft loss: T^2 * soft_CE(student/T, soft_targets)
    Variable soft_term = scale(
        soft_cross_entropy(student_scaled, teacher_probs),
        temp * temp);

    // Short-circuit: pure soft KD
    if (alpha == 0.0f)
        return soft_term;

    // Hard CE term
    Variable hard_term = cross_entropy(student_logits, hard_targets);

    // Combined: alpha * hard + (1-alpha) * soft
    return add(scale(hard_term, alpha), scale(soft_term, 1.0f - alpha));
}

} // namespace sub0llm::nn
