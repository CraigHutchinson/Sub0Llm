#include "sub0diff/eval/recovery.hpp"

#include "sub0llm/core/tensor.hpp"

#include <algorithm>
#include <span>
#include <vector>

namespace sub0diff::eval {

using sub0llm::Tensor;
using sub0llm::DType;

namespace {
// Score one window's masked positions from its (T, C) host logits: greedy argmax over the real
// vocab, token recall + word-start/continuation/whole-word breakdown. Shared by the single-window
// and batched paths so they stay bit-identical. `lz` points at this window's first logit row.
RecoveryResult score_window(const float* lz, std::size_t C, std::int64_t V,
                            std::span<const std::int32_t> clean_ids,
                            std::span<const std::uint8_t> masked,
                            PositionStats* pos,
                            std::span<const std::uint8_t> is_word_start) {
    const bool word_aware = !is_word_start.empty();
    auto starts_word = [&](std::size_t t) {
        return t == 0 || is_word_start[static_cast<std::size_t>(clean_ids[t])] != 0;
    };
    RecoveryResult res{};
    for (auto m : masked) res.masked += (m != 0);
    int cur_word_masked = 0, cur_word_hits = 0;
    auto flush_word = [&] {
        if (cur_word_masked > 0) {
            res.word_total += 1;
            res.word_hits  += (cur_word_hits == cur_word_masked);
        }
        cur_word_masked = cur_word_hits = 0;
    };
    for (std::size_t t = 0; t < masked.size(); ++t) {
        if (word_aware && starts_word(t)) flush_word();
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
        if (word_aware) {
            if (starts_word(t)) { res.ws_masked += 1; res.ws_hits += hit; }
            else                { res.wc_masked += 1; res.wc_hits += hit; }
            cur_word_masked += 1; cur_word_hits += hit;
        }
    }
    if (word_aware) flush_word();
    return res;
}
}  // namespace

RecoveryResult evaluate_recovery(const nn::Denoiser& model,
                                 std::span<const std::int32_t> clean_ids,
                                 std::span<const std::uint8_t> masked,
                                 PositionStats* pos,
                                 std::span<const std::uint8_t> is_word_start) {
    const auto T = static_cast<std::int64_t>(clean_ids.size());
    Tensor input({T}, DType::Int32);
    auto in = input.data_as<std::int32_t>();
    std::ranges::copy(clean_ids, in.begin());

    int n_masked = 0;
    for (std::size_t i = 0; i < masked.size(); ++i)
        if (masked[i]) { in[i] = model.mask_id(); ++n_masked; }

    const float noise = static_cast<float>(n_masked) / static_cast<float>(T);
    auto logits = model.forward(input, noise);
    // The forward runs on the model's device (GPU when training on cuda — the expensive part stays
    // on the GPU); bring only the (T, vocab) logits to host for the argmax below.
    const Tensor lh = logits.data().device().is_cpu()
                          ? logits.data()
                          : logits.data().to(sub0llm::Device::cpu());
    return score_window(lh.data_as<float>().data(), static_cast<std::size_t>(model.model_vocab()),
                        model.real_vocab(), clean_ids, masked, pos, is_word_start);
}

RecoveryResult evaluate_corpus_recall(const nn::Denoiser& model,
                                      std::span<const std::int32_t> corpus_ids,
                                      std::int64_t T, float noise,
                                      std::mt19937& rng,
                                      PositionStats* pos,
                                      std::size_t max_windows,
                                      std::span<const std::uint8_t> is_word_start,
                                      bool whole_word) {
    RecoveryResult total;
    nn::Corruption corr;
    // A stream of N tokens has N-T+1 sliding positions; sample them on a uniform
    // stride when a budget is set so coverage spans the whole stream.
    const std::size_t n_positions = corpus_ids.size() - static_cast<std::size_t>(T) + 1;
    const std::size_t n_eval = (max_windows == 0) ? n_positions
                                                  : std::min(max_windows, n_positions);
    const double stride = static_cast<double>(n_positions) / static_cast<double>(n_eval);

    // Batch B windows per Denoiser forward. The sweep is forward-bound, and a batched (block-
    // diagonal) forward is mathematically identical to B single-window forwards but with far better
    // GPU utilisation — fewer launches, one D2H per batch. Recall numbers are unchanged (same masks,
    // same per-window scoring) — only the wall time drops.
    constexpr std::int64_t B = 32;
    std::vector<std::int32_t>                  batch_tokens;   // (bn·T) corrupted ids, row-major
    std::vector<float>                         noises;         // per-window n_masked/T
    std::vector<std::vector<std::uint8_t>>     masks;          // per-window masked flags (copied)
    std::vector<std::span<const std::int32_t>> windows;        // per-window clean span
    batch_tokens.reserve(static_cast<std::size_t>(B * T));
    noises.reserve(static_cast<std::size_t>(B));

    auto flush = [&] {
        const std::int64_t bn = static_cast<std::int64_t>(noises.size());
        if (bn == 0) return;
        Tensor input({bn * T}, DType::Int32);
        std::ranges::copy(batch_tokens, input.data_as<std::int32_t>().begin());
        auto         logits = model.forward(input, noises, bn, T);   // (bn·T, model_vocab)
        const Tensor lh     = logits.data().device().is_cpu()
                                  ? logits.data()
                                  : logits.data().to(sub0llm::Device::cpu());
        const float*      lz = lh.data_as<float>().data();
        const std::size_t C  = static_cast<std::size_t>(model.model_vocab());
        for (std::int64_t b = 0; b < bn; ++b)
            total.accumulate(score_window(
                lz + static_cast<std::size_t>(b) * static_cast<std::size_t>(T) * C, C,
                model.real_vocab(), windows[static_cast<std::size_t>(b)],
                masks[static_cast<std::size_t>(b)], pos, is_word_start));
        batch_tokens.clear(); noises.clear(); masks.clear(); windows.clear();
    };

    for (std::size_t i = 0; i < n_eval; ++i) {
        const auto off = static_cast<std::size_t>(static_cast<double>(i) * stride);
        auto window = corpus_ids.subspan(off, static_cast<std::size_t>(T));
        if (whole_word)
            nn::corrupt_whole_word_into(window, is_word_start, noise, spec::NoiseSchedule::Absorbing,
                                        model.mask_id(), model.real_vocab(), rng, corr);
        else
            nn::corrupt_into(window, noise, spec::NoiseSchedule::Absorbing,
                             model.mask_id(), model.real_vocab(), rng, corr);
        batch_tokens.insert(batch_tokens.end(), corr.tokens.begin(), corr.tokens.end());
        masks.emplace_back(corr.corrupted.begin(), corr.corrupted.end());
        windows.push_back(window);
        noises.push_back(static_cast<float>(corr.n_corrupted) / static_cast<float>(T));
        if (static_cast<std::int64_t>(noises.size()) == B) flush();
    }
    flush();   // trailing partial batch
    return total;
}

} // namespace sub0diff::eval
