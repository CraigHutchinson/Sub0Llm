#include "sub0llm/nn/scheduler.hpp"

#include <cmath>
#include <stdexcept>
#include <format>

namespace sub0llm::nn {

CosineWithWarmup::CosineWithWarmup(float   max_lr,
                                   float   min_lr,
                                   int64_t warmup_steps,
                                   int64_t total_steps)
    : max_lr_(max_lr), min_lr_(min_lr),
      warmup_steps_(warmup_steps), total_steps_(total_steps) {
    if (max_lr <= 0.0f)
        throw std::runtime_error(
            std::format("CosineWithWarmup: max_lr={} must be positive", max_lr));
    if (min_lr < 0.0f)
        throw std::runtime_error(
            std::format("CosineWithWarmup: min_lr={} must be >= 0", min_lr));
    if (min_lr > max_lr)
        throw std::runtime_error(
            std::format("CosineWithWarmup: min_lr={} must be <= max_lr={}", min_lr, max_lr));
    if (warmup_steps < 0)
        throw std::runtime_error(
            std::format("CosineWithWarmup: warmup_steps={} must be >= 0", warmup_steps));
    if (total_steps <= 0)
        throw std::runtime_error(
            std::format("CosineWithWarmup: total_steps={} must be positive", total_steps));
    if (warmup_steps >= total_steps)
        throw std::runtime_error(std::format(
            "CosineWithWarmup: warmup_steps={} must be < total_steps={}",
            warmup_steps, total_steps));
}

float CosineWithWarmup::operator()(int64_t step) const {
    if (step < 0) step = 0;

    // Linear warmup phase.
    if (step < warmup_steps_)
        return max_lr_ * static_cast<float>(step) / static_cast<float>(warmup_steps_);

    // After full schedule: floor.
    if (step >= total_steps_) return min_lr_;

    // Cosine decay phase.
    const float progress = static_cast<float>(step - warmup_steps_) /
                           static_cast<float>(total_steps_ - warmup_steps_);
    constexpr float pi = 3.14159265358979323846f;
    return min_lr_ + 0.5f * (max_lr_ - min_lr_) * (1.0f + std::cos(pi * progress));
}

} // namespace sub0llm::nn
