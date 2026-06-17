#pragma once

// inspect.hpp — per-sample recovery inspection for a masked-diffusion denoiser.
//
// Dumps the N best- and N worst-recovered held-out windows at a fixed noise level with
// per-masked-position truth→prediction detail, so a headline recall number (e.g. "15%")
// becomes legible: which words the model nails, which it misses, and whether misses are
// word-STARTS (real prediction) or continuations (partial-word completion). A reusable
// diagnostic for any chapter training the Denoiser.

#include <algorithm>
#include <cstdint>
#include <random>
#include <span>
#include <string>
#include <vector>

#include "sub0diff/nn/denoiser.hpp"
#include "sub0diff/nn/noise_schedule.hpp"
#include "sub0diff/spec/diffusion_spec.hpp"

#include "sub0llm/compat/print.hpp"
#include "sub0llm/core/tensor.hpp"
#include "sub0llm/tokenizer/bpe.hpp"

namespace sub0diff::eval {

inline void inspect_recovery(const sub0diff::nn::Denoiser& model, const sub0llm::BPETokenizer& tok,
                             std::span<const std::int32_t> stream, std::int64_t T,
                             std::span<const std::uint8_t> ws, bool whole_word,
                             int n_show, float noise) {
    namespace dn = sub0diff::nn;
    using sub0llm::Tensor;
    using sub0llm::DType;

    // Render a single token id legibly: the Ġ word-start marker (UTF-8 C4 A0) → '·'.
    auto disp = [&](std::int32_t id) {
        std::string s(tok.token_str(id));
        if (s.size() >= 2 && static_cast<unsigned char>(s[0]) == 0xC4 &&
            static_cast<unsigned char>(s[1]) == 0xA0)
            return "·" + s.substr(2);
        return s;
    };
    struct Miss { std::int32_t pos, truth, pred; bool hit, word_start; };
    struct Sample { std::size_t off; float recall; std::vector<Miss> masked; };

    const std::size_t n_positions = stream.size() - static_cast<std::size_t>(T) + 1;
    const std::size_t budget = std::min<std::size_t>(400, n_positions);
    const double step = static_cast<double>(n_positions) / static_cast<double>(budget);
    std::mt19937 rng(20240614);
    dn::Corruption corr;
    std::vector<Sample> samples;
    samples.reserve(budget);

    for (std::size_t i = 0; i < budget; ++i) {
        const auto off = static_cast<std::size_t>(static_cast<double>(i) * step);
        auto window = stream.subspan(off, static_cast<std::size_t>(T));
        if (whole_word)
            dn::corrupt_whole_word_into(window, ws, noise, sub0diff::spec::NoiseSchedule::Absorbing,
                                        model.mask_id(), model.real_vocab(), rng, corr);
        else
            dn::corrupt_into(window, noise, sub0diff::spec::NoiseSchedule::Absorbing,
                             model.mask_id(), model.real_vocab(), rng, corr);
        // Forward on the corrupted window, argmax over real tokens at each masked position.
        Tensor input({T}, DType::Int32);
        std::ranges::copy(corr.tokens, input.data_as<std::int32_t>().begin());
        const float actual = static_cast<float>(corr.n_corrupted) / static_cast<float>(T);
        auto logits = model.forward(input, actual);
        const auto lz = logits.data().data_as<float>();
        const auto C  = static_cast<std::size_t>(model.model_vocab());
        const auto V  = model.real_vocab();
        Sample s{off, 0.0f, {}};
        int hits = 0;
        for (std::int64_t t = 0; t < T; ++t) {
            if (!corr.corrupted[static_cast<std::size_t>(t)]) continue;
            std::int32_t best = 0; float bv = -1e30f;
            for (std::int64_t c = 0; c < V; ++c) {
                const float v = lz[static_cast<std::size_t>(t) * C + static_cast<std::size_t>(c)];
                if (v > bv) { bv = v; best = static_cast<std::int32_t>(c); }
            }
            const auto truth = window[static_cast<std::size_t>(t)];
            const bool hit = (best == truth);
            hits += hit;
            const bool wstart = (t == 0) || ws[static_cast<std::size_t>(truth)] != 0;
            s.masked.push_back({static_cast<std::int32_t>(t), truth, best, hit, wstart});
        }
        s.recall = s.masked.empty() ? 0.0f
                                    : static_cast<float>(hits) / static_cast<float>(s.masked.size());
        samples.push_back(std::move(s));
    }

    std::ranges::sort(samples, [](const Sample& a, const Sample& b) { return a.recall > b.recall; });

    auto dump = [&](const Sample& s) {
        auto window = stream.subspan(s.off, static_cast<std::size_t>(T));
        std::vector<sub0llm::BPETokenizer::TokenId> ids(window.begin(), window.end());
        std::string text = tok.decode(ids);
        if (text.size() > 200) text = text.substr(0, 200) + "…";
        std::println("  ── window @ {}  recall {:.0f}% ({} masked) ──",
                     s.off, s.recall * 100.0f, s.masked.size());
        std::println("    text: {}", text);
        for (const auto& m : s.masked) {
            std::println("      [{:>2}] {:<4} {:>10} → {:<10} {}",
                         m.pos, m.word_start ? "WORD" : "cont",
                         "'" + disp(m.truth) + "'", "'" + disp(m.pred) + "'",
                         m.hit ? "✓" : "✗");
        }
    };

    std::println("\n── --inspect: per-sample recovery at noise {:.0f}%{} ──",
                 noise * 100.0f, whole_word ? " (whole-word)" : "");
    const int n = std::min(n_show, static_cast<int>(samples.size()));
    std::println("Best {} recovered windows:", n);
    for (int i = 0; i < n; ++i) dump(samples[static_cast<std::size_t>(i)]);
    std::println("\nWorst {} recovered windows:", n);
    for (int i = 0; i < n; ++i)
        dump(samples[samples.size() - 1 - static_cast<std::size_t>(i)]);
}

}  // namespace sub0diff::eval
