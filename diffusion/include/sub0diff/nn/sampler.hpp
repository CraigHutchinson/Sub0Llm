#pragma once

// sampler.hpp (Ch30) — the REVERSE process: iterative canvas refinement.
//
// Ch28/29 trained a denoiser that predicts every masked position in ONE forward
// pass — good when most of the canvas is context, hopeless when most of it is
// masked. The reverse process closes that gap by iterating:
//
//   1. denoise the whole canvas (bidirectional, all positions in parallel)
//   2. COMMIT only the predictions the model is confident about
//   3. keep the rest masked and denoise again — every committed token is new
//      context that sharpens the next round ("the canvas snaps into focus")
//
// Confidence-based commitment (MaskGIT / LLaDA low-confidence remasking) plus an
// entropy bound for early stopping (DiffusionGemma's `entropy_bound`): when every
// remaining masked position is already low-entropy, finish in one final commit.
//
// The loop is SELF-TERMINATING and adaptive: each iteration commits at least
// ceil(n_masked/T)·guaranteed progress, confident canvases finish in few rounds,
// hard ones take more — iteration count is an output, not an input.
//
// The same machinery doubles as ITERATIVE RECOVERY: seed the canvas with a
// corrupted window (unmasked positions fixed) and refine — directly comparable
// to Ch29's one-step recovery numbers.

#include "sub0diff/nn/denoiser.hpp"
#include "sub0diff/viz/trace.hpp"
#include "sub0llm/core/tensor.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <random>
#include <span>
#include <stdexcept>
#include <vector>

namespace sub0diff::nn {

// Viz Phase B hook: per-frame TRUE per-level internal activations. Generic fallback = none (flat and
// other non-hierarchical models). A hierarchical model (MeraDenoiser) supplies a same-namespace overload
// (declared with the model) selected by ADL at this template's instantiation — so the sampler stays
// model-agnostic and flat builds don't depend on the MERA header.
template <class Model>
[[nodiscard]] inline std::vector<std::vector<float>> capture_levels(const Model&,
                                                                    std::span<const std::int32_t>) {
    return {};
}

// Commit ORDER within an iteration (which confident positions to commit first).
//   Confidence — pure most-confident-first (MaskGIT/LLaDA default).
//   Spread     — 4b experiment (ch32): commit a spatially SPREAD coarse skeleton first (a
//                training-free "gist field" — confident anchors distributed across the canvas),
//                then fill, so the model can't greedily extend one region into a repetition loop.
//                Tests whether topic-drift/looping is fixable at DECODE time or needs training.
enum class CommitOrder : std::uint8_t { Confidence, Spread };

struct SamplerConfig {
    float conf_threshold = 0.9f;   // commit positions with max-prob ≥ this
    float entropy_bound  = 0.1f;   // mean masked entropy below this → commit all & stop
    // Progress guarantee: commit ≥ this fraction of the still-masked positions each iter,
    // even if none clear conf_threshold. KEEP THIS SMALL. At 0.10 the EARLY all-masked steps
    // force-commit ~10% of the canvas from near-unigram statistics (the model has no context
    // yet), seeding garbage the bootstrap can't recover (measured: coherent prose → char-salad
    // on the SAME model). 0.03 ≈ a handful of the most-confident positions per iter — matches
    // tiny-diffusion's conservative "commit the confident few, or at least 1" and recovers the
    // model's real quality. Lower (0.01) = cleaner but ~3× the iterations; --min-commit overrides.
    float min_commit_frac = 0.03f;
    float temperature    = 1.0f;   // 0 = greedy argmax; >0 = sample from softmax
    std::size_t max_iters = 0;     // 0 = canvas length (one-per-iter worst case)

    // Low-confidence REMASKING: each iteration, generated (non-fixed) tokens whose
    // probability under the CURRENT context falls below this are returned to [MASK]
    // for reconsideration — commitments are revisable, so early errors don't
    // permanently poison the context (the error-compounding failure of pure
    // commit-only refinement, measured in Ch30). 0 = off. Net progress is enforced
    // (never remask more than was committed this iteration, minus one).
    float remask_threshold = 0.0f;

    CommitOrder commit_order = CommitOrder::Confidence;
};

struct SamplerStats {
    std::size_t iterations = 0;    // denoiser forwards used (self-chosen)
    std::size_t committed  = 0;    // positions filled by the sampler
    bool entropy_stopped   = false;
    double seconds         = 0.0;
};

// Convenience: an all-masked canvas of length T with `prompt` fixed at the front.
// Templated on the model type — any denoiser exposing mask_id() works (Denoiser, HierDenoiser).
template <class Model>
[[nodiscard]] std::vector<std::int32_t> make_canvas(const Model& model, std::int64_t T,
                                                    std::span<const std::int32_t> prompt = {}) {
    if (static_cast<std::int64_t>(prompt.size()) > T)
        throw std::runtime_error("make_canvas: prompt longer than canvas");
    std::vector<std::int32_t> canvas(static_cast<std::size_t>(T), model.mask_id());
    std::ranges::copy(prompt, canvas.begin());
    return canvas;
}

// Refine `canvas` in place. Positions equal to model.mask_id() are generated; all other positions
// are FIXED context (prompt / surviving corruption). on_iter (optional) observes each iteration.
// Templated on the model type so the SAME tuned sampler (confidence/spread commit, entropy stop,
// remask) drives both the flat Denoiser and the coarse-to-fine HierDenoiser — a fair generation A/B.
// Only uses model.forward(input, noise), mask_id(), model_vocab(), real_vocab().
template <class Model>
SamplerStats refine_canvas(const Model& model, std::span<std::int32_t> canvas,
                           const SamplerConfig& cfg, std::mt19937& rng,
                           const std::function<void(std::span<const std::int32_t>,
                                                    std::size_t)>& on_iter = {},
                           viz::GenerationTrace* trace = nullptr) {
    const auto t0 = std::chrono::steady_clock::now();
    const std::size_t T = canvas.size();
    const std::int32_t mask_id = model.mask_id();
    const auto C = static_cast<std::size_t>(model.model_vocab());
    const auto V = static_cast<std::size_t>(model.real_vocab());  // never predict [MASK]
    const std::size_t max_iters = cfg.max_iters ? cfg.max_iters : T;

    sub0llm::Tensor input({static_cast<std::int64_t>(T)}, sub0llm::DType::Int32);
    const auto in = input.data_as<std::int32_t>();

    struct Pred { std::size_t pos; std::int32_t token; float conf; float entropy; };
    std::vector<Pred> preds;
    preds.reserve(T);
    std::vector<float> probs(V);

    std::vector<std::uint8_t> fixed(T);
    for (std::size_t i = 0; i < T; ++i) fixed[i] = (canvas[i] != mask_id);

    std::vector<std::int32_t> prev;   // canvas at the START of an iter (for the trace before/after diff)
    SamplerStats stats;
    for (std::size_t iter = 0; iter < max_iters; ++iter) {
        std::size_t n_masked = 0;
        for (std::size_t i = 0; i < T; ++i) {
            in[i] = canvas[i];
            n_masked += (canvas[i] == mask_id);
        }
        if (n_masked == 0) break;
        ++stats.iterations;
        const auto it0 = std::chrono::steady_clock::now();
        if (trace) prev.assign(canvas.begin(), canvas.end());

        const float noise = static_cast<float>(n_masked) / static_cast<float>(T);
        auto logits = model.forward(input, noise);
        const sub0llm::Tensor lh = logits.data().device().is_cpu()
                                       ? logits.data()
                                       : logits.data().to(sub0llm::Device::cpu());
        const float* lz = lh.data_as<float>().data();

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

        // Build a trace Frame from the before(prev)/after(canvas) diff + this iter's per-position
        // confidence/entropy/prediction (the data the loop already has). committed = newly filled this
        // iter; remasked = re-opened this iter.
        auto record_frame = [&] {
            if (!trace) return;
            viz::Frame fr;
            fr.iter = static_cast<int>(stats.iterations);
            fr.ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - it0).count();
            fr.tokens.assign(canvas.begin(), canvas.end());
            fr.predicted.assign(canvas.begin(), canvas.end());
            fr.masked.resize(T); fr.committed.resize(T); fr.remasked.resize(T);
            fr.confidence.assign(T, 0.0f); fr.entropy.assign(T, 0.0f);
            for (std::size_t i = 0; i < T; ++i) {
                fr.masked[i]    = static_cast<std::uint8_t>(canvas[i] == mask_id);
                fr.committed[i] = static_cast<std::uint8_t>(prev[i] == mask_id && canvas[i] != mask_id);
                fr.remasked[i]  = static_cast<std::uint8_t>(prev[i] != mask_id && canvas[i] == mask_id);
                fr.committed_count += fr.committed[i];
            }
            for (const auto& p : preds) {
                fr.confidence[p.pos] = p.conf;
                fr.entropy[p.pos]    = p.entropy;
                fr.predicted[p.pos]  = p.token;
            }
            // Phase B: TRUE per-level internals for THIS frame's canvas (empty for flat — viewer then
            // falls back to the derived settled-fraction view). One extra forward per frame; fine for
            // the single traced passage the Viz records.
            fr.level_rms = capture_levels(model, std::span<const std::int32_t>(canvas));
            trace->frames.push_back(std::move(fr));
        };

        const double mean_entropy = entropy_sum / static_cast<double>(preds.size());
        if (mean_entropy < cfg.entropy_bound) {
            for (const auto& p : preds) canvas[p.pos] = p.token;
            stats.committed += preds.size();
            stats.entropy_stopped = true;
            record_frame();
            if (on_iter) on_iter(canvas, stats.iterations);
            break;
        }

        std::ranges::sort(preds, [](const Pred& a, const Pred& b) { return a.conf > b.conf; });
        const std::size_t min_commit = std::max<std::size_t>(
            1, static_cast<std::size_t>(std::ceil(cfg.min_commit_frac * static_cast<float>(n_masked))));
        std::size_t target = 0;
        for (const auto& p : preds) {
            if (target >= min_commit && p.conf < cfg.conf_threshold) break;
            ++target;
        }
        std::size_t committed = 0;
        if (cfg.commit_order == CommitOrder::Spread) {
            const std::size_t gap = std::max<std::size_t>(1, T / std::max<std::size_t>(1, target));
            std::vector<std::size_t> chosen;
            chosen.reserve(target);
            auto far_enough = [&](std::size_t pos) {
                for (auto q : chosen)
                    if ((pos > q ? pos - q : q - pos) < gap) return false;
                return true;
            };
            for (const auto& p : preds) {
                if (committed >= target) break;
                if (far_enough(p.pos)) { canvas[p.pos] = p.token; chosen.push_back(p.pos); ++committed; }
            }
            for (const auto& p : preds) {
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
        record_frame();
        if (on_iter) on_iter(canvas, stats.iterations);
    }

    stats.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return stats;
}

} // namespace sub0diff::nn
