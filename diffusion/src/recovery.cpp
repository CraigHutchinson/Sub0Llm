#include "sub0diff/eval/recovery.hpp"

#include "sub0llm/core/tensor.hpp"

#include <algorithm>

namespace sub0diff::eval {

using sub0llm::Tensor;
using sub0llm::DType;

RecoveryResult evaluate_recovery(const nn::Denoiser& model,
                                 std::span<const std::int32_t> clean_ids,
                                 std::span<const std::uint8_t> masked,
                                 PositionStats* pos) {
    const auto T = static_cast<std::int64_t>(clean_ids.size());
    Tensor input({T}, DType::Int32);
    auto in = input.data_as<std::int32_t>();
    std::ranges::copy(clean_ids, in.begin());

    int n_masked = 0;
    for (std::size_t i = 0; i < masked.size(); ++i)
        if (masked[i]) { in[i] = model.mask_id(); ++n_masked; }

    const float noise = static_cast<float>(n_masked) / static_cast<float>(T);
    auto logits = model.forward(input, noise);
    const auto lz = logits.data().data_as<float>();
    const auto C = static_cast<std::size_t>(model.model_vocab());
    const auto V = model.real_vocab();   // argmax over real tokens only

    RecoveryResult res{0, n_masked};
    for (std::size_t t = 0; t < masked.size(); ++t) {
        if (!masked[t]) continue;
        std::int32_t best = 0;
        float best_v = -1e30f;
        for (std::int64_t c = 0; c < V; ++c) {
            const float v = lz[t * C + static_cast<std::size_t>(c)];
            if (v > best_v) { best_v = v; best = static_cast<std::int32_t>(c); }
        }
        const bool hit = (best == clean_ids[t]);
        res.hits += hit;
        if (pos) { pos->masked[t] += 1; pos->hits[t] += hit; }
    }
    return res;
}

RecoveryResult evaluate_corpus_recall(const nn::Denoiser& model,
                                      std::span<const std::int32_t> corpus_ids,
                                      std::int64_t T, float noise,
                                      std::mt19937& rng,
                                      PositionStats* pos,
                                      std::size_t max_windows) {
    RecoveryResult total;
    nn::Corruption corr;
    // A stream of N tokens has N-T+1 sliding positions; sample them on a uniform
    // stride when a budget is set so coverage spans the whole stream.
    const std::size_t n_positions = corpus_ids.size() - static_cast<std::size_t>(T) + 1;
    const std::size_t n_eval = (max_windows == 0) ? n_positions
                                                  : std::min(max_windows, n_positions);
    const double stride = static_cast<double>(n_positions) / static_cast<double>(n_eval);
    for (std::size_t i = 0; i < n_eval; ++i) {
        const auto off = static_cast<std::size_t>(static_cast<double>(i) * stride);
        auto window = corpus_ids.subspan(off, static_cast<std::size_t>(T));
        nn::corrupt_into(window, noise, spec::NoiseSchedule::Absorbing,
                         model.mask_id(), model.real_vocab(), rng, corr);
        auto r = evaluate_recovery(model, window, corr.corrupted, pos);
        total.hits   += r.hits;
        total.masked += r.masked;
    }
    return total;
}

} // namespace sub0diff::eval
