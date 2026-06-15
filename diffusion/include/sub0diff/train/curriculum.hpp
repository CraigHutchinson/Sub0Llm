#pragma once

// curriculum.hpp — frontier-point noise curriculum with a token-gated mastery ceiling.
//
// Ch28's measured winner (frontier-point + token-gated, ~60% sweep recall vs the
// uniform "formal objective"): instead of sampling the noise level t uniformly over
// (0,1] — which spends most steps on high-noise, low-signal samples a small model
// can't yet learn — train at the FRONTIER the model can currently handle, and raise
// that frontier only once it has been MASTERED. Mastery is judged in masked TOKENS
// (the true signal unit, so dwell time per level scales like 1/noise), comparing a
// noise bucket's loss EMA only against ITSELF so a difficulty shift can't masquerade
// as learning. Ch29 used the uniform objective and underfit the easy regime; this
// brings the curriculum back as a reusable, shared component.

#include <algorithm>
#include <cstdint>
#include <vector>

namespace sub0diff::train {

class NoiseCurriculum {
public:
    struct Config {
        float         start        = 0.05f;   // initial ceiling — start easy (5% masked)
        float         end          = 1.00f;   // target ceiling (full noise = formal objective's top)
        float         step         = 0.05f;   // bucket width AND ceiling increment
        std::uint64_t min_tokens   = 1500;    // masked-token evidence required per decision
        float         bucket_alpha = 0.02f;   // EMA horizon (~50 samples)
        float         raise_thresh = 0.08f;   // EMA improvement to raise the ceiling
        float         lower_thresh = -0.06f;  // EMA regression to lower it
    };

    explicit NoiseCurriculum(Config c)
        : cfg_(c), ceiling_(c.start),
          buckets_(bucket_of(c.end) + 1) {}

    // The frontier noise level to train at this step (point curriculum: AT the ceiling).
    [[nodiscard]] float frontier() const noexcept { return ceiling_; }
    [[nodiscard]] float ceiling()  const noexcept { return ceiling_; }
    [[nodiscard]] bool  converged() const noexcept { return converged_; }

    // Feed back one step's outcome (actual_noise = realised mask fraction, its mean
    // CE loss, and how many tokens were masked). Advances/lowers the ceiling when the
    // ceiling bucket has accumulated enough evidence. Safe to call with aggregated
    // results from W data-parallel windows (pass the mean loss and summed tokens).
    void observe(float actual_noise, float loss, std::uint64_t masked_tokens) {
        NoiseBucket& b = buckets_[bucket_of(actual_noise)];
        b.ema = b.ema_init ? cfg_.bucket_alpha * loss + (1.0f - cfg_.bucket_alpha) * b.ema
                           : (b.ema_init = true, loss);

        NoiseBucket& top = buckets_[bucket_of(ceiling_)];
        top.tokens += masked_tokens;
        if (top.tokens < cfg_.min_tokens) return;

        if (!top.baseline_set) {                 // first full evidence window at this ceiling
            top.baseline = top.ema;
            top.baseline_set = true;
            top.tokens = 0;
            return;
        }
        const float improvement = top.baseline - top.ema;   // >0 = learning
        if (improvement > cfg_.raise_thresh) {
            ceiling_ = std::min(cfg_.end, ceiling_ + cfg_.step);
            // Re-baseline the (possibly new) ceiling bucket against fresh evidence.
            buckets_[bucket_of(ceiling_)].baseline_set = false;
            buckets_[bucket_of(ceiling_)].tokens = 0;
        } else if (improvement < cfg_.lower_thresh) {
            ceiling_ = std::max(cfg_.start, ceiling_ - cfg_.step);
            buckets_[bucket_of(ceiling_)].baseline_set = false;
            buckets_[bucket_of(ceiling_)].tokens = 0;
        } else {                                 // plateau
            if (ceiling_ >= cfg_.end - 0.001f) converged_ = true;
            top.tokens = 0;                      // re-judge after another evidence window
        }
    }

private:
    struct NoiseBucket {
        float         ema = 0.0f, baseline = 0.0f;
        std::uint64_t tokens = 0;
        bool          ema_init = false, baseline_set = false;
    };
    [[nodiscard]] std::size_t bucket_of(float noise) const noexcept {
        const auto idx = static_cast<std::size_t>(noise / cfg_.step + 0.5f);
        return std::min(idx, static_cast<std::size_t>(cfg_.end / cfg_.step + 0.5f));
    }

    Config                   cfg_;
    float                    ceiling_;
    bool                     converged_ = false;
    std::vector<NoiseBucket> buckets_;
};

} // namespace sub0diff::train
