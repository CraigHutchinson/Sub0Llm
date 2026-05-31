#include "sub0llm/nn/episodic_memory.hpp"
#include "sub0llm/autograd/ops.hpp"
#include "sub0llm/core/tensor.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <limits>
#include <stdexcept>

namespace sub0llm::nn {

// ── internal helpers ──────────────────────────────────────────────────────────

// Apply (+1) or remove (-1) deltas from model weights. sign = +1.0 or -1.0.
static void apply_delta(ModernGPT& model,
                         std::vector<Tensor>& deltas,
                         float sign)
{
    auto params = model.parameters();
    if (params.size() != deltas.size())
        throw std::runtime_error(std::format(
            "EpisodicState: parameter count mismatch — state has {} deltas, "
            "model has {} parameters",
            deltas.size(), params.size()));

    for (std::size_t i = 0; i < deltas.size(); ++i) {
        if (deltas[i].numel() == 0) continue;
        auto wd = params[i]->data().data_as<float>();
        auto dd = deltas[i].data_as<float>();
        for (std::size_t j = 0; j < static_cast<std::size_t>(deltas[i].numel()); ++j)
            wd[j] += sign * dd[j];
    }
}

// ── EpisodicState ─────────────────────────────────────────────────────────────

void EpisodicState::reset() {
    // Caller must unmerge before resetting; resetting while merged would leave
    // the old deltas permanently baked into the model weights with no way to
    // remove them (merged=false would make subsequent unmerge() a no-op).
    if (merged)
        throw std::runtime_error(
            "EpisodicState::reset() called while state is merged into model — "
            "call unmerge() first");
    for (auto& d : deltas) {
        auto dd = d.data_as<float>();
        std::fill(dd.begin(), dd.end(), 0.0f);
    }
}

void EpisodicState::merge(ModernGPT& model) {
    if (merged) return;
    apply_delta(model, deltas, +1.0f);
    merged = true;
}

void EpisodicState::unmerge(ModernGPT& model) {
    if (!merged) return;
    apply_delta(model, deltas, -1.0f);
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
    if (!cfg.accumulate) {
        // Safe reset: unmerge first if the state is currently baked into the model.
        if (state.merged) state.unmerge(model);
        state.reset();
    }
    if (tokens.size() < 2) return;

    auto losses = comprehension_pass(model, tokens);
    auto params  = model.parameters();

    if (params.size() != state.deltas.size())
        throw std::runtime_error(std::format(
            "episodic_encode: state has {} deltas but model has {} parameters",
            state.deltas.size(), params.size()));

    const int64_t T = static_cast<int64_t>(tokens.size()) - 1;  // # of predictions

    // Pre-compute which parameters require gradients — invariant across spans/steps.
    std::vector<bool> needs_grad(params.size());
    for (std::size_t pi = 0; pi < params.size(); ++pi)
        needs_grad[pi] = params[pi]->requires_grad();

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

        // Elaborative rehearsal: cfg.think_steps true gradient-descent steps.
        //
        // Each step applies the current step's gradient to the model weights
        // before computing the next gradient, so each step builds on the
        // previous. After the loop the temporary weight changes are undone and
        // the net delta is accumulated into state.deltas.
        //
        // Per-span accumulator (tracks only this span's contribution so that
        // we can undo the temporary model changes after the loop).
        std::vector<float> span_acc;  // flat, indexed per param via offsets

        // Build a flat buffer and per-param offset table for efficient undo.
        std::vector<std::size_t> offsets(params.size() + 1, 0);
        for (std::size_t pi = 0; pi < params.size(); ++pi)
            offsets[pi + 1] = offsets[pi] + (needs_grad[pi]
                ? static_cast<std::size_t>(params[pi]->data().numel()) : 0);
        span_acc.assign(offsets.back(), 0.0f);

        for (int t = 0; t < cfg.think_steps; ++t) {
            for (auto* p : params) p->zero_grad();

            auto logits = model.forward(span_ids);
            auto loss   = autograd::cross_entropy(logits, span_tgt);
            loss.backward();

            // Apply this step's gradient to BOTH the model (so next step sees
            // updated weights) and the per-span accumulator (for final undo).
            for (std::size_t pi = 0; pi < params.size(); ++pi) {
                if (!needs_grad[pi]) continue;
                const Tensor& g = params[pi]->grad();
                if (g.numel() == 0) continue;
                auto gd = g.data_as<float>();
                auto wd = params[pi]->data().data_as<float>();
                const std::size_t n = static_cast<std::size_t>(g.numel());
                const float lr = cfg.learning_rate;
                float* acc = span_acc.data() + offsets[pi];
                for (std::size_t k = 0; k < n; ++k) {
                    const float delta = -lr * gd[k];
                    acc[k]  += delta;   // track for undo
                    wd[k]   += delta;   // apply to model → next step sees updated weights
                }
            }
        }

        // Restore model weights and fold per-span delta into state.deltas.
        for (std::size_t pi = 0; pi < params.size(); ++pi) {
            if (!needs_grad[pi]) continue;
            const std::size_t n = offsets[pi + 1] - offsets[pi];
            if (n == 0) continue;
            auto wd = params[pi]->data().data_as<float>();
            auto dd = state.deltas[pi].data_as<float>();
            const float* acc = span_acc.data() + offsets[pi];
            for (std::size_t k = 0; k < n; ++k) {
                wd[k] -= acc[k];   // undo temporary model change
                dd[k] += acc[k];   // accumulate into persistent episodic delta
            }
        }

        i = j;
    }
}

} // namespace sub0llm::nn
