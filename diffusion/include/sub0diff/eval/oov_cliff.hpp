#pragma once

// oov_cliff.hpp (Ch32 Phase 0, metric M1) — the OOV-CLIFF metric.
//
// A word-level model is fast at grammar but helpless on words it barely saw in training (a rare
// type has a poorly-trained embedding row; a never-seen word has none at all). M1 quantifies that
// weakness WITHOUT retraining: split the per-masked-token NELBO of a held-out stream by whether the
// target is a RARE type (the rarest fraction of the vocabulary by train-frequency — a proxy for
// OOV-at-test). NLL_rare / NLL_common is the "cliff": ~1 means the model predicts rare and common
// words equally well; >>1 means it falls off a cliff on rare words. The P1 char-composition codec
// (a content-addressed word vector built from spelling) targets exactly this gap — M1 is its gate:
// after 1c the ratio should move toward ~1 while in-vocab NLL does not regress (BUILD_PLAN §Phase 0).

#include "sub0diff/nn/denoiser.hpp"
#include "sub0diff/nn/noise_schedule.hpp"

#include <cstdint>
#include <random>
#include <span>
#include <vector>

namespace sub0diff::eval {

struct OovCliffResult {
    double        sum_ce_rare = 0.0, sum_ce_common = 0.0;   // summed CE (nats) over masked tokens
    std::uint64_t n_rare = 0, n_common = 0;                 // masked-token counts per bucket
    [[nodiscard]] double nll_rare()   const { return n_rare   ? sum_ce_rare   / static_cast<double>(n_rare)   : 0.0; }
    [[nodiscard]] double nll_common() const { return n_common ? sum_ce_common / static_cast<double>(n_common) : 0.0; }
    [[nodiscard]] double ratio()      const { return nll_common() > 0.0 ? nll_rare() / nll_common() : 0.0; }
};

// Mark the rarest `rare_frac` fraction of token TYPES (by train-frequency, ties broken by id) as
// "rare". Returns is_rare indexed by token id (size = vocab). Types absent from the train stream
// (freq 0 — the OOV-est) sort first and are marked rare.
[[nodiscard]] std::vector<std::uint8_t>
rare_type_mask(std::span<const std::int32_t> train_ids, std::int64_t vocab, double rare_frac);

// Mask positions across the eval stream, denoise once per window, and split the per-masked-token
// NELBO (CE in nats, over the full model vocab) by whether the target is rare. Batched forward
// (B=32) so it runs efficiently on the model's device (GPU when training on cuda).
[[nodiscard]] OovCliffResult
evaluate_oov_cliff(const nn::Denoiser& model, std::span<const std::int32_t> eval_ids,
                   std::span<const std::uint8_t> is_rare, std::int64_t T, float noise,
                   std::mt19937& rng, std::size_t max_windows);

}  // namespace sub0diff::eval
