#pragma once

// recovery.hpp (Ch29) — shared one-step recovery evaluation for denoisers.
//
// Generalizes Ch28's chapter-local evaluation: mask chosen positions of a clean
// window, run a single denoiser forward, count exact recoveries. The corpus-wide
// sweep produces a recall-vs-noise curve, and optional per-position accumulators
// expose window-edge effects (Ch28 measured ~40% edge vs ~62% interior recall —
// one-sided context; motivates Ch30/31 block overlap / committed-block context).

#include "sub0diff/nn/denoiser.hpp"
#include "sub0diff/nn/noise_schedule.hpp"

#include <cstdint>
#include <random>
#include <span>
#include <vector>

namespace sub0diff::eval {

struct RecoveryResult {
    int hits   = 0;   // masked positions filled correctly
    int masked = 0;   // total masked positions
    [[nodiscard]] float recall() const noexcept {
        return masked > 0 ? static_cast<float>(hits) / static_cast<float>(masked) : 0.0f;
    }
};

// Per-window-position accumulators (index = position): reveals whether edge
// positions train/recover worse than the interior.
struct PositionStats {
    std::vector<int> hits, masked;
    explicit PositionStats(std::size_t T) : hits(T, 0), masked(T, 0) {}
    [[nodiscard]] float recall_at(std::size_t i) const noexcept {
        return masked[i] > 0 ? static_cast<float>(hits[i]) / static_cast<float>(masked[i]) : 0.0f;
    }
};

// Mask the flagged positions, denoise once, count exact recoveries (greedy argmax
// over REAL tokens only — the [MASK] row is never a valid prediction).
[[nodiscard]] RecoveryResult evaluate_recovery(const nn::Denoiser& model,
                                               std::span<const std::int32_t> clean_ids,
                                               std::span<const std::uint8_t> masked,
                                               PositionStats* pos = nullptr);

// Corrupt EVERY window of `corpus_ids` at `noise` and accumulate recall.
[[nodiscard]] RecoveryResult evaluate_corpus_recall(const nn::Denoiser& model,
                                                    std::span<const std::int32_t> corpus_ids,
                                                    std::int64_t T, float noise,
                                                    std::mt19937& rng,
                                                    PositionStats* pos = nullptr);

} // namespace sub0diff::eval
