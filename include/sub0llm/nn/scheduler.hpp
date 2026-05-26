#pragma once

#include <cstdint>
#include <stdexcept>
#include <format>

namespace sub0llm::nn {

// ── ConstantLR ────────────────────────────────────────────────────────────────
//
// Returns the same learning rate for every step.
class ConstantLR {
public:
    explicit ConstantLR(float lr) : lr_(lr) {
        if (lr <= 0.0f)
            throw std::runtime_error(
                std::format("ConstantLR: lr={} must be positive", lr));
    }

    [[nodiscard]] float operator()(int64_t /*step*/) const noexcept { return lr_; }
    [[nodiscard]] float peak_lr() const noexcept { return lr_; }

private:
    float lr_;
};

// ── CosineWithWarmup ──────────────────────────────────────────────────────────
//
// Standard LLM pretraining schedule:
//   step < warmup_steps : linear ramp from 0 → max_lr
//   step ∈ [warmup, total] : cosine decay from max_lr → min_lr
//   step > total_steps  : held at min_lr
//
// Matches the schedule used in LLaMA, GPT-3, PaLM, etc.
class CosineWithWarmup {
public:
    // max_lr:        peak learning rate after warmup
    // min_lr:        floor (typically max_lr / 10)
    // warmup_steps:  linear warmup duration (e.g. 100–2000 steps)
    // total_steps:   full schedule length
    CosineWithWarmup(float   max_lr,
                     float   min_lr,
                     int64_t warmup_steps,
                     int64_t total_steps);

    [[nodiscard]] float operator()(int64_t step) const;
    [[nodiscard]] float peak_lr()  const noexcept { return max_lr_; }
    [[nodiscard]] float floor_lr() const noexcept { return min_lr_; }

private:
    float   max_lr_, min_lr_;
    int64_t warmup_steps_, total_steps_;
};

} // namespace sub0llm::nn
