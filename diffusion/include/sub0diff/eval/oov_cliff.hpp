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

#include "sub0llm/core/tensor.hpp"

#include <algorithm>
#include <cmath>
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

namespace detail {
// Per-masked-token CE (nats) from one window's (T,C) host logits, split by target rarity.
// CE = logsumexp(row) - row[target] — a numerically-stable softmax NLL over the full model vocab.
inline void score_ce(const float* lz, std::size_t C, std::span<const std::int32_t> clean_ids,
                     std::span<const std::uint8_t> masked, std::span<const std::uint8_t> is_rare,
                     OovCliffResult& out) {
    for (std::size_t t = 0; t < masked.size(); ++t) {
        if (!masked[t]) continue;
        const float* row = lz + t * C;
        float mx = row[0];
        for (std::size_t c = 1; c < C; ++c) mx = std::max(mx, row[c]);
        double sum = 0.0;
        for (std::size_t c = 0; c < C; ++c) sum += std::exp(static_cast<double>(row[c] - mx));
        const auto   tgt = static_cast<std::size_t>(clean_ids[t]);
        const double ce  = static_cast<double>(mx) + std::log(sum) - static_cast<double>(row[tgt]);
        if (is_rare[tgt]) { out.sum_ce_rare   += ce; ++out.n_rare; }
        else              { out.sum_ce_common += ce; ++out.n_common; }
    }
}
}  // namespace detail

// Mask positions across the eval stream, denoise once per window, and split the per-masked-token
// NELBO (CE in nats, over the full model vocab) by whether the target is rare. Batched forward
// (B=32) so it runs efficiently on the model's device (GPU when training on cuda). Templated on the
// model type — works for nn::Denoiser AND nn::CodecDenoiser (same forward/mask_id/vocab interface).
template <class Model>
[[nodiscard]] OovCliffResult
evaluate_oov_cliff(const Model& model, std::span<const std::int32_t> eval_ids,
                   std::span<const std::uint8_t> is_rare, std::int64_t T, float noise,
                   std::mt19937& rng, std::size_t max_windows) {
    OovCliffResult out;
    nn::Corruption corr;
    const std::size_t n_positions = eval_ids.size() - static_cast<std::size_t>(T) + 1;
    const std::size_t n_eval = (max_windows == 0) ? n_positions : std::min(max_windows, n_positions);
    const double stride = static_cast<double>(n_positions) / static_cast<double>(n_eval);

    constexpr std::int64_t B = 32;
    std::vector<std::int32_t>                  batch_tokens;
    std::vector<float>                         noises;
    std::vector<std::vector<std::uint8_t>>     masks;
    std::vector<std::span<const std::int32_t>> windows;
    batch_tokens.reserve(static_cast<std::size_t>(B * T));
    noises.reserve(static_cast<std::size_t>(B));

    auto flush = [&] {
        const std::int64_t bn = static_cast<std::int64_t>(noises.size());
        if (bn == 0) return;
        sub0llm::Tensor input({bn * T}, sub0llm::DType::Int32);
        std::ranges::copy(batch_tokens, input.data_as<std::int32_t>().begin());
        auto                  logits = model.forward(input, noises, bn, T);
        const sub0llm::Tensor lh     = logits.data().device().is_cpu()
                                           ? logits.data()
                                           : logits.data().to(sub0llm::Device::cpu());
        const float*      lz = lh.data_as<float>().data();
        const std::size_t C  = static_cast<std::size_t>(model.model_vocab());
        for (std::int64_t b = 0; b < bn; ++b)
            detail::score_ce(lz + static_cast<std::size_t>(b) * static_cast<std::size_t>(T) * C, C,
                             windows[static_cast<std::size_t>(b)], masks[static_cast<std::size_t>(b)],
                             is_rare, out);
        batch_tokens.clear(); noises.clear(); masks.clear(); windows.clear();
    };

    for (std::size_t i = 0; i < n_eval; ++i) {
        const auto off = static_cast<std::size_t>(static_cast<double>(i) * stride);
        auto window = eval_ids.subspan(off, static_cast<std::size_t>(T));
        nn::corrupt_into(window, noise, nn::NoiseSchedule::Absorbing,
                         model.mask_id(), model.real_vocab(), rng, corr);
        batch_tokens.insert(batch_tokens.end(), corr.tokens.begin(), corr.tokens.end());
        masks.emplace_back(corr.corrupted.begin(), corr.corrupted.end());
        windows.push_back(window);
        noises.push_back(static_cast<float>(corr.n_corrupted) / static_cast<float>(T));
        if (static_cast<std::int64_t>(noises.size()) == B) flush();
    }
    flush();
    return out;
}

}  // namespace sub0diff::eval
