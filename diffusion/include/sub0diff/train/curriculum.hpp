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
#include <limits>
#include <vector>

namespace sub0diff::train {

// ── Convergence-gated frontier curriculum (integer-k, decided per EPOCH) ─────────────
// A second, simpler curriculum motivated by the Ch29 minimal-case findings (TRAINING_DESIGN
// §12): a single EXACT masked count k is the lowest-variance gradient there is (every window
// masks the same k → the consistency lever that flipped N=64 from 3% to 99%). So train at
// EXACTLY k masked tokens, starting at the easiest (k=1: recover one token from T-1 visible),
// and raise k only once the model has MASTERED the current level — judged by the held-out
// NELBO at that level plateauing, measured at the per-epoch eval cadence (the one place the
// model's standing is read cleanly; the within-step training loss is far too noisy to gate on).
// Like a child mastering 1-token infilling, then 2-token, … before facing the full noise range.
// When k reaches k_max (= T-1, the min-1-visible cap), the curriculum is converged and the
// trainer switches to the full formal objective (all noise levels, t∈(0.02,t_max]).
//
// Contrast with NoiseCurriculum: that one gates on a within-step loss EMA in masked-TOKEN time
// and moves a fractional ceiling; this gates on a per-EPOCH held-out NELBO plateau and moves an
// integer k. This matches the rule "model parameters and the curriculum level only change on a
// full-epoch boundary," so a difficulty step never contaminates the convergence signal.
//
// OPEN CONSIDERATION — forgetting the easy levels. Frontier-POINT trains at EXACTLY k, so while
// climbing, the model stops seeing the easy k=1 regime (recover from 63 visible). In THEORY it
// shouldn't forget — and the post-convergence phase trains the full objective, revisiting every
// level — but the easy levels could degrade DURING the climb (a window at k=20 conditions each
// prediction on far less context than at k=1, so it is not a strict superset). ch29 prints a live
// "base(k=1)-NELBO" forgetting watch each epoch; per-`t` recall at any saved checkpoint is the
// retrospective check. IF forgetting is observed, the fix is to make the curriculum a CAP/BIAS
// rather than a point: train the RANGE t∈[t_min, k/T] up to the frontier (so every easier level
// stays in the mix and is continually reinforced) instead of exactly k — a one-line change to the
// trainer's t-range. Kept open pending evidence; the point variant is the lowest-variance default.
class FrontierCurriculum {
public:
    struct Config {
        std::int64_t seq_len     = 64;    // window length T (frontier t = k/T)
        std::int64_t k_start     = 1;     // easiest level: exactly 1 masked token
        std::int64_t k_max       = 0;     // 0 = derive T-1 (the min-1-visible cap)
        std::int64_t k_step      = 1;     // masked tokens added per advancement (user's 1/64→2/64)
        int          patience    = 2;     // epochs of no NELBO improvement ⇒ level mastered
        float        min_improve = 0.01f; // held-out NELBO drop that still counts as learning
    };

    explicit FrontierCurriculum(Config c)
        : cfg_(c), k_(std::max<std::int64_t>(1, c.k_start)),
          k_max_(c.k_max > 0 ? c.k_max : std::max<std::int64_t>(1, c.seq_len - 1)) {
        k_ = std::min(k_, k_max_);
    }

    // Noise level the trainer masks at this step: t = k/T (with exact-count ⇒ exactly k masked).
    [[nodiscard]] float        frontier() const noexcept {
        return static_cast<float>(k_) / static_cast<float>(cfg_.seq_len);
    }
    [[nodiscard]] std::int64_t level()   const noexcept { return k_; }
    [[nodiscard]] std::int64_t max_level() const noexcept { return k_max_; }
    [[nodiscard]] bool         converged() const noexcept { return converged_; }
    [[nodiscard]] float        best()    const noexcept { return best_; }
    [[nodiscard]] int          stalls()  const noexcept { return stalls_; }

    // Call ONCE per epoch (at the held-out eval) with the held-out NELBO measured at the CURRENT
    // frontier level (mask exactly k). Returns true iff the level advanced on this call.
    bool observe_epoch(float nelbo_at_frontier) {
        if (converged_) return false;
        if (nelbo_at_frontier < best_ - cfg_.min_improve) {   // still descending at this level
            best_ = nelbo_at_frontier;
            stalls_ = 0;
            return false;
        }
        if (++stalls_ < cfg_.patience) return false;          // plateau not yet confirmed
        if (k_ >= k_max_) { converged_ = true; return false; }  // mastered the hardest level
        k_ = std::min(k_max_, k_ + cfg_.k_step);              // advance to the next difficulty
        best_ = std::numeric_limits<float>::max();
        stalls_ = 0;
        return true;
    }

private:
    Config       cfg_;
    std::int64_t k_, k_max_;
    float        best_ = std::numeric_limits<float>::max();
    int          stalls_ = 0;
    bool         converged_ = false;
};

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
