#include "sub0diff/eval/oov_cliff.hpp"

#include "sub0llm/core/tensor.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <span>
#include <vector>

namespace sub0diff::eval {

using sub0llm::DType;
using sub0llm::Tensor;

std::vector<std::uint8_t> rare_type_mask(std::span<const std::int32_t> train_ids,
                                         std::int64_t vocab, double rare_frac) {
    std::vector<std::uint64_t> freq(static_cast<std::size_t>(vocab), 0);
    for (auto id : train_ids)
        if (id >= 0 && id < vocab) ++freq[static_cast<std::size_t>(id)];

    std::vector<std::int64_t> order(static_cast<std::size_t>(vocab));
    std::iota(order.begin(), order.end(), 0);
    std::ranges::sort(order, [&](std::int64_t a, std::int64_t b) {
        const auto fa = freq[static_cast<std::size_t>(a)], fb = freq[static_cast<std::size_t>(b)];
        return fa != fb ? fa < fb : a < b;          // rarest types first, ties by id
    });

    std::vector<std::uint8_t> is_rare(static_cast<std::size_t>(vocab), 0);
    const std::size_t n_rare = static_cast<std::size_t>(rare_frac * static_cast<double>(vocab));
    for (std::size_t i = 0; i < n_rare && i < order.size(); ++i)
        is_rare[static_cast<std::size_t>(order[i])] = 1;
    return is_rare;
}

namespace {
// Per-masked-token CE (nats) from one window's (T,C) host logits, split by target rarity.
// CE = logsumexp(row) - row[target], a numerically-stable softmax NLL over the full model vocab.
void score_ce(const float* lz, std::size_t C, std::span<const std::int32_t> clean_ids,
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
}  // namespace

OovCliffResult evaluate_oov_cliff(const nn::Denoiser& model, std::span<const std::int32_t> eval_ids,
                                  std::span<const std::uint8_t> is_rare, std::int64_t T, float noise,
                                  std::mt19937& rng, std::size_t max_windows) {
    OovCliffResult out;
    nn::Corruption corr;
    const std::size_t n_positions = eval_ids.size() - static_cast<std::size_t>(T) + 1;
    const std::size_t n_eval = (max_windows == 0) ? n_positions : std::min(max_windows, n_positions);
    const double stride = static_cast<double>(n_positions) / static_cast<double>(n_eval);

    // Batch B windows per Denoiser forward (block-diagonal — identical to B single forwards),
    // matching the recall sweep so M1 runs efficiently on the model's device.
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
        Tensor input({bn * T}, DType::Int32);
        std::ranges::copy(batch_tokens, input.data_as<std::int32_t>().begin());
        auto         logits = model.forward(input, noises, bn, T);
        const Tensor lh     = logits.data().device().is_cpu()
                                  ? logits.data()
                                  : logits.data().to(sub0llm::Device::cpu());
        const float*      lz = lh.data_as<float>().data();
        const std::size_t C  = static_cast<std::size_t>(model.model_vocab());
        for (std::int64_t b = 0; b < bn; ++b)
            score_ce(lz + static_cast<std::size_t>(b) * static_cast<std::size_t>(T) * C, C,
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
    flush();   // trailing partial batch
    return out;
}

}  // namespace sub0diff::eval
