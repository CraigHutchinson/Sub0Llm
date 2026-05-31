#include "sub0llm/nn/episodic_memory.hpp"
#include "sub0llm/autograd/ops.hpp"
#include "sub0llm/core/tensor.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace sub0llm::nn {

// ── EpisodicState ─────────────────────────────────────────────────────────────

void EpisodicState::reset() {
    for (auto& d : deltas) {
        auto dd = d.data_as<float>();
        std::fill(dd.begin(), dd.end(), 0.0f);
    }
    merged = false;
}

void EpisodicState::merge(ModernGPT& model) {
    if (merged) return;
    auto params = model.parameters();
    const std::size_t n = std::min(params.size(), deltas.size());
    for (std::size_t i = 0; i < n; ++i) {
        if (deltas[i].numel() == 0) continue;
        auto wd = params[i]->data().data_as<float>();
        auto dd = deltas[i].data_as<float>();
        for (std::size_t j = 0; j < static_cast<std::size_t>(deltas[i].numel()); ++j)
            wd[j] += dd[j];
    }
    merged = true;
}

void EpisodicState::unmerge(ModernGPT& model) {
    if (!merged) return;
    auto params = model.parameters();
    const std::size_t n = std::min(params.size(), deltas.size());
    for (std::size_t i = 0; i < n; ++i) {
        if (deltas[i].numel() == 0) continue;
        auto wd = params[i]->data().data_as<float>();
        auto dd = deltas[i].data_as<float>();
        for (std::size_t j = 0; j < static_cast<std::size_t>(deltas[i].numel()); ++j)
            wd[j] -= dd[j];
    }
    merged = false;
}

// ── make_episodic_state ───────────────────────────────────────────────────────

EpisodicState make_episodic_state(ModernGPT& model) {
    EpisodicState state;
    auto params = model.parameters();
    state.deltas.reserve(params.size());
    for (auto* p : params)
        state.deltas.emplace_back(zeros(p->data().shape()));
    return state;
}

// ── comprehension_pass ────────────────────────────────────────────────────────

std::vector<float> comprehension_pass(
    const ModernGPT& model, const std::vector<int32_t>& tokens)
{
    if (tokens.size() < 2) return {};

    const int64_t T = static_cast<int64_t>(tokens.size());
    Tensor ids({T}, DType::Int32);
    auto id_data = ids.data_as<int32_t>();
    for (std::size_t i = 0; i < tokens.size(); ++i)
        id_data[i] = tokens[i];

    // Forward pass — autograd graph is built but we discard it; we only need logits.
    auto logits_var = model.forward(ids);
    const Tensor& logits = logits_var.data();   // (T, V)
    const int64_t V = logits.shape()[1];
    auto ld = logits.data_as<float>();

    std::vector<float> losses;
    losses.reserve(static_cast<std::size_t>(T - 1));

    for (int64_t i = 0; i < T - 1; ++i) {
        const int32_t target = tokens[static_cast<std::size_t>(i + 1)];
        // Numerically-stable log-sum-exp
        float max_l = -std::numeric_limits<float>::infinity();
        for (int64_t v = 0; v < V; ++v)
            max_l = std::max(max_l, ld[static_cast<std::size_t>(i * V + v)]);
        float sum_exp = 0.0f;
        for (int64_t v = 0; v < V; ++v)
            sum_exp += std::exp(ld[static_cast<std::size_t>(i * V + v)] - max_l);
        float log_sum = max_l + std::log(sum_exp);
        losses.push_back(log_sum - ld[static_cast<std::size_t>(i * V + target)]);
    }
    return losses;
}

// ── episodic_encode ───────────────────────────────────────────────────────────

void episodic_encode(ModernGPT& model, EpisodicState& state,
                     const std::vector<int32_t>& tokens,
                     const EpisodicConfig& cfg)
{
    if (!cfg.accumulate) state.reset();
    if (tokens.size() < 2) return;

    auto losses = comprehension_pass(model, tokens);
    auto params  = model.parameters();
    const int64_t T = static_cast<int64_t>(tokens.size()) - 1;  // # of predictions

    // Walk the surprisal signal; greedily extend spans above the threshold.
    int64_t i = 0;
    while (i < T) {
        if (losses[static_cast<std::size_t>(i)] <= cfg.surprise_threshold) {
            ++i;
            continue;
        }

        // Extend span while still surprising and within the length cap.
        int64_t j = i + 1;
        while (j < T
               && losses[static_cast<std::size_t>(j)] > cfg.surprise_threshold
               && j - i < static_cast<int64_t>(cfg.max_span_len))
            ++j;

        // span uses input tokens [i .. j-1] to predict [i+1 .. j].
        const int64_t Ts = j - i;
        if (Ts < static_cast<int64_t>(cfg.min_span_len)) {
            i = j;
            continue;
        }

        Tensor span_ids({Ts}, DType::Int32);
        Tensor span_tgt({Ts}, DType::Int32);
        auto sdi = span_ids.data_as<int32_t>();
        auto sdt = span_tgt.data_as<int32_t>();
        for (int64_t k = 0; k < Ts; ++k) {
            sdi[static_cast<std::size_t>(k)] = tokens[static_cast<std::size_t>(i + k)];
            sdt[static_cast<std::size_t>(k)] = tokens[static_cast<std::size_t>(i + k + 1)];
        }

        // Elaborative rehearsal: cfg.think_steps gradient steps on this span.
        for (int t = 0; t < cfg.think_steps; ++t) {
            for (auto* p : params) p->zero_grad();

            auto logits = model.forward(span_ids);
            auto loss   = autograd::cross_entropy(logits, span_tgt);
            loss.backward();

            // Negative gradient → delta (gradient descent direction)
            for (std::size_t pi = 0; pi < params.size(); ++pi) {
                if (!params[pi]->requires_grad()) continue;
                const Tensor& g = params[pi]->grad();
                if (g.numel() == 0) continue;
                auto gd = g.data_as<float>();
                auto dd = state.deltas[pi].data_as<float>();
                const std::size_t n = static_cast<std::size_t>(g.numel());
                for (std::size_t k = 0; k < n; ++k)
                    dd[k] -= cfg.learning_rate * gd[k];
            }
        }
        i = j;
    }
}

} // namespace sub0llm::nn
