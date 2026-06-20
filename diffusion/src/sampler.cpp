#include "sub0diff/nn/sampler.hpp"

#include "sub0llm/core/tensor.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>

namespace sub0diff::nn {

using sub0llm::Tensor;
using sub0llm::DType;

std::vector<std::int32_t> make_canvas(const Denoiser& model, std::int64_t T,
                                      std::span<const std::int32_t> prompt) {
    if (static_cast<std::int64_t>(prompt.size()) > T)
        throw std::runtime_error("make_canvas: prompt longer than canvas");
    std::vector<std::int32_t> canvas(static_cast<std::size_t>(T), model.mask_id());
    std::ranges::copy(prompt, canvas.begin());
    return canvas;
}

SamplerStats refine_canvas(const Denoiser& model,
                           std::span<std::int32_t> canvas,
                           const SamplerConfig& cfg,
                           std::mt19937& rng,
                           const std::function<void(std::span<const std::int32_t>,
                                                    std::size_t)>& on_iter) {
    const auto t0 = std::chrono::steady_clock::now();
    const std::size_t T = canvas.size();
    const std::int32_t mask_id = model.mask_id();
    const auto C = static_cast<std::size_t>(model.model_vocab());
    const auto V = static_cast<std::size_t>(model.real_vocab());  // never predict [MASK]
    const std::size_t max_iters = cfg.max_iters ? cfg.max_iters : T;

    Tensor input({static_cast<std::int64_t>(T)}, DType::Int32);
    const auto in = input.data_as<std::int32_t>();

    // Per-position scratch for the current iteration's masked predictions.
    struct Pred { std::size_t pos; std::int32_t token; float conf; float entropy; };
    std::vector<Pred> preds;
    preds.reserve(T);
    std::vector<float> probs(V);

    // Positions that were already filled at entry are FIXED (prompt / surviving
    // corruption context) — remasking may only ever touch sampler-generated tokens.
    std::vector<std::uint8_t> fixed(T);
    for (std::size_t i = 0; i < T; ++i) fixed[i] = (canvas[i] != mask_id);

    SamplerStats stats;
    for (std::size_t iter = 0; iter < max_iters; ++iter) {
        std::size_t n_masked = 0;
        for (std::size_t i = 0; i < T; ++i) {
            in[i] = canvas[i];
            n_masked += (canvas[i] == mask_id);
        }
        if (n_masked == 0) break;
        ++stats.iterations;

        const float noise = static_cast<float>(n_masked) / static_cast<float>(T);
        auto logits = model.forward(input, noise);
        const float* lz = logits.data().data_as<float>().data();

        // Softmax over the REAL vocab at each masked position; collect prediction,
        // confidence (max prob) and entropy.
        preds.clear();
        double entropy_sum = 0.0;
        for (std::size_t i = 0; i < T; ++i) {
            if (canvas[i] != mask_id) continue;
            const float* row = lz + i * C;
            float mx = row[0];
            for (std::size_t c = 1; c < V; ++c) mx = std::max(mx, row[c]);
            float z = 0.0f;
            for (std::size_t c = 0; c < V; ++c) { probs[c] = std::exp(row[c] - mx); z += probs[c]; }
            float entropy = 0.0f, best_p = 0.0f;
            std::size_t best_c = 0;
            for (std::size_t c = 0; c < V; ++c) {
                const float p = probs[c] / z;
                probs[c] = p;
                if (p > best_p) { best_p = p; best_c = c; }
                if (p > 0.0f) entropy -= p * std::log(p);
            }
            std::int32_t tok = static_cast<std::int32_t>(best_c);
            if (cfg.temperature > 0.0f) {
                // Temperature sampling over the (already computed) distribution.
                // For temperature != 1 re-weight; T=1 uses probs as-is.
                if (cfg.temperature != 1.0f) {
                    float zz = 0.0f;
                    for (std::size_t c = 0; c < V; ++c) {
                        probs[c] = std::pow(probs[c], 1.0f / cfg.temperature);
                        zz += probs[c];
                    }
                    for (std::size_t c = 0; c < V; ++c) probs[c] /= zz;
                }
                std::discrete_distribution<std::size_t> dd(probs.begin(), probs.end());
                tok = static_cast<std::int32_t>(dd(rng));
            }
            preds.push_back({i, tok, best_p, entropy});
            entropy_sum += entropy;
        }

        // DiffusionGemma-style early stop: the whole remaining canvas is already
        // low-entropy — commit everything and finish.
        const double mean_entropy = entropy_sum / static_cast<double>(preds.size());
        if (mean_entropy < cfg.entropy_bound) {
            for (const auto& p : preds) canvas[p.pos] = p.token;
            stats.committed += preds.size();
            stats.entropy_stopped = true;
            if (on_iter) on_iter(canvas, stats.iterations);
            break;
        }

        // Commit all positions above the confidence threshold, but always at least
        // ceil(min_commit_frac · n_masked) — guaranteed progress (most-confident first).
        std::ranges::sort(preds, [](const Pred& a, const Pred& b) { return a.conf > b.conf; });
        const std::size_t min_commit = std::max<std::size_t>(
            1, static_cast<std::size_t>(std::ceil(cfg.min_commit_frac
                                                  * static_cast<float>(n_masked))));
        // How many the confidence rule commits this iter (all ≥ threshold, but ≥ min_commit).
        std::size_t target = 0;
        for (const auto& p : preds) {
            if (target >= min_commit && p.conf < cfg.conf_threshold) break;
            ++target;
        }
        std::size_t committed = 0;
        if (cfg.commit_order == CommitOrder::Spread) {
            // Spread the `target` commits across the canvas (gist-field analog): greedy
            // farthest-point in confidence order, then fill any shortfall ignoring the gap.
            const std::size_t gap = std::max<std::size_t>(1, T / std::max<std::size_t>(1, target));
            std::vector<std::size_t> chosen;
            chosen.reserve(target);
            auto far_enough = [&](std::size_t pos) {
                for (auto q : chosen)
                    if ((pos > q ? pos - q : q - pos) < gap) return false;
                return true;
            };
            for (const auto& p : preds) {           // pass 1: spread, most-confident first
                if (committed >= target) break;
                if (far_enough(p.pos)) {
                    canvas[p.pos] = p.token; chosen.push_back(p.pos); ++committed;
                }
            }
            for (const auto& p : preds) {           // pass 2: fill to target, gap relaxed
                if (committed >= target) break;
                if (canvas[p.pos] == mask_id) { canvas[p.pos] = p.token; ++committed; }
            }
        } else {
            for (const auto& p : preds) {
                if (committed >= min_commit && p.conf < cfg.conf_threshold) break;
                canvas[p.pos] = p.token;
                ++committed;
            }
        }
        stats.committed += committed;

        // Low-confidence remasking: re-open generated tokens the model now doubts.
        // Uses this iteration's logits — prob of the token CURRENTLY in the canvas
        // given the context it was predicted from. Capped at committed-1 so each
        // iteration makes net progress.
        if (cfg.remask_threshold > 0.0f && committed > 1) {
            struct Doubt { std::size_t pos; float p; };
            std::vector<Doubt> doubts;
            for (std::size_t i = 0; i < T; ++i) {
                if (fixed[i] || canvas[i] == mask_id) continue;
                const float* row = lz + i * C;
                float mx = row[0];
                for (std::size_t c = 1; c < V; ++c) mx = std::max(mx, row[c]);
                float z = 0.0f;
                for (std::size_t c = 0; c < V; ++c) z += std::exp(row[c] - mx);
                const float p_cur = std::exp(row[static_cast<std::size_t>(canvas[i])] - mx) / z;
                if (p_cur < cfg.remask_threshold) doubts.push_back({i, p_cur});
            }
            std::ranges::sort(doubts, [](const Doubt& a, const Doubt& b) { return a.p < b.p; });
            const std::size_t cap = committed - 1;
            for (std::size_t k = 0; k < std::min(cap, doubts.size()); ++k)
                canvas[doubts[k].pos] = mask_id;
        }
        if (on_iter) on_iter(canvas, stats.iterations);
    }

    stats.seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    return stats;
}

} // namespace sub0diff::nn
