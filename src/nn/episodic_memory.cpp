#include "sub0llm/nn/episodic_memory.hpp"
#include "sub0llm/autograd/ops.hpp"
#include "sub0llm/core/tensor.hpp"
#include "sub0llm/nn/optimizer.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace sub0llm::nn {

// ── internal helpers ──────────────────────────────────────────────────────────

static void check_size_match(const std::vector<Tensor>& deltas,
                              const std::vector<autograd::Variable*>& params)
{
    if (params.size() != deltas.size())
        throw std::runtime_error(std::format(
            "EpisodicState: parameter count mismatch — state has {} deltas, "
            "model has {} parameters",
            deltas.size(), params.size()));
}

// Apply (+1) or remove (-1) deltas from model weights.
static void apply_delta(ModernGPT& model,
                         std::vector<Tensor>& deltas,
                         float sign)
{
    auto params = model.parameters();
    check_size_match(deltas, params);

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
        if (state.merged) state.unmerge(model);
        state.reset();
    }
    if (tokens.size() < 2) return;

    auto losses = comprehension_pass(model, tokens);
    auto params  = model.parameters();
    check_size_match(state.deltas, params);

    const int64_t T = static_cast<int64_t>(tokens.size()) - 1;  // # of predictions

    // Pre-compute which parameters require gradients — invariant across spans.
    std::vector<bool> needs_grad(params.size());
    for (std::size_t pi = 0; pi < params.size(); ++pi)
        needs_grad[pi] = params[pi]->requires_grad();

    // One SGD optimizer per episodic_encode call; optimizer state persists
    // across spans within the same document.
    SGD span_opt(params, cfg.learning_rate);

    // Save base weights ONCE before the span loop.
    // Every span restores to these same base weights, so a single copy suffices.
    // Only grad-enabled parameters are saved; non-grad params are never modified.
    std::vector<std::vector<float>> orig_vals;
    orig_vals.reserve(params.size());
    for (std::size_t pi = 0; pi < params.size(); ++pi) {
        if (needs_grad[pi]) {
            auto pd = params[pi]->data().data_as<float>();
            orig_vals.emplace_back(pd.begin(), pd.end());
        } else {
            orig_vals.emplace_back();  // empty placeholder — never read
        }
    }

    // Restore all grad-enabled params to orig_vals.
    // Called after each span (normal completion) and on exception (safety).
    auto restore_weights = [&] {
        for (std::size_t pi = 0; pi < params.size(); ++pi) {
            if (!needs_grad[pi]) continue;
            auto wd = params[pi]->data().data_as<float>();
            const auto& ov = orig_vals[pi];
            for (std::size_t k = 0; k < ov.size(); ++k)
                wd[k] = ov[k];
        }
    };

    // Walk the surprisal signal; greedily extend spans above the threshold.
    int64_t i = 0;
    while (i < T) {
        if (losses[static_cast<std::size_t>(i)] <= cfg.surprise_threshold) {
            ++i;
            continue;
        }

        int64_t j = i + 1;
        while (j < T
               && losses[static_cast<std::size_t>(j)] > cfg.surprise_threshold
               && j - i < static_cast<int64_t>(cfg.max_span_len))
            ++j;

        // span uses input tokens [i..j-1] to predict [i+1..j].
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

        // ── Elaborative rehearsal ─────────────────────────────────────────
        //
        // Run think_steps SGD steps on the span — each step sees the model
        // updated by the previous one (true gradient descent).  Then fold the
        // net delta into state.deltas and restore the model to base weights.
        //
        // SGD is used rather than Adam because Adam normalises updates to
        // approximately ±lr per step regardless of gradient magnitude.  For
        // short write sessions (think_steps ≤ 20) this makes the effective
        // update sensitive to the lr choice and can overshoot badly.  SGD's
        // update scales with gradient magnitude, keeping smaller-gradient
        // parameters small and larger-gradient parameters larger.
        //
        // If forward/backward throws, restore_weights() ensures base model
        // weights are never left in a partially-updated state.
        try {
            for (int t = 0; t < cfg.think_steps; ++t) {
                span_opt.zero_grad();
                auto logits = model.forward(span_ids);
                auto loss   = autograd::cross_entropy(logits, span_tgt);
                loss.backward();
                span_opt.step();
            }
        } catch (...) {
            restore_weights();
            throw;
        }

        // delta += (params_after - orig); restore model to original weights.
        for (std::size_t pi = 0; pi < params.size(); ++pi) {
            if (!needs_grad[pi]) continue;
            const std::size_t n = orig_vals[pi].size();
            if (n == 0) continue;
            auto wd = params[pi]->data().data_as<float>();
            auto dd = state.deltas[pi].data_as<float>();
            for (std::size_t k = 0; k < n; ++k) {
                dd[k] += wd[k] - orig_vals[pi][k];  // accumulate net span delta
                wd[k]  = orig_vals[pi][k];           // restore model weights
            }
        }

        i = j;
    }
}

// ── Session persistence ───────────────────────────────────────────────────────

static constexpr std::array<char, 8> k_epis_magic = {'S','U','B','0','E','P','I','S'};
static constexpr uint32_t            k_epis_version = 1;

void save_episodic_state(const EpisodicState& state, const std::string& path)
{
    std::ofstream f(path, std::ios::binary);
    if (!f.is_open())
        throw std::runtime_error(
            std::format("save_episodic_state: cannot open '{}' for writing", path));

    // Magic + version
    f.write(k_epis_magic.data(), 8);
    f.write(reinterpret_cast<const char*>(&k_epis_version), sizeof(k_epis_version));

    // n_deltas
    const uint64_t n_deltas = static_cast<uint64_t>(state.deltas.size());
    f.write(reinterpret_cast<const char*>(&n_deltas), sizeof(n_deltas));

    for (const Tensor& d : state.deltas) {
        // n_dims + shape (each dim as int64 = 8 bytes)
        const uint32_t n_dims = static_cast<uint32_t>(d.ndim());
        f.write(reinterpret_cast<const char*>(&n_dims), sizeof(n_dims));
        for (std::size_t i = 0; i < n_dims; ++i) {
            const int64_t dim = d.shape(i);
            f.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
        }

        // float32 data
        const auto sp = d.data_as<float>();
        f.write(reinterpret_cast<const char*>(sp.data()),
                static_cast<std::streamsize>(sp.size() * sizeof(float)));
    }

    // merged flag
    const uint8_t merged_byte = state.merged ? 1u : 0u;
    f.write(reinterpret_cast<const char*>(&merged_byte), sizeof(merged_byte));

    if (!f)
        throw std::runtime_error(
            std::format("save_episodic_state: I/O error writing '{}'", path));
}

[[nodiscard]] EpisodicState load_episodic_state(const std::string& path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open())
        throw std::runtime_error(
            std::format("load_episodic_state: cannot open '{}'", path));

    // Validate magic
    std::array<char, 8> magic{};
    f.read(magic.data(), 8);
    if (!f || magic != k_epis_magic)
        throw std::runtime_error(
            std::format("load_episodic_state: bad magic in '{}'", path));

    // Validate version
    uint32_t version = 0;
    f.read(reinterpret_cast<char*>(&version), sizeof(version));
    if (!f || version != k_epis_version)
        throw std::runtime_error(
            std::format("load_episodic_state: unsupported version {} in '{}'",
                        version, path));

    // n_deltas
    uint64_t n_deltas = 0;
    f.read(reinterpret_cast<char*>(&n_deltas), sizeof(n_deltas));
    if (!f)
        throw std::runtime_error(
            std::format("load_episodic_state: truncated header in '{}'", path));

    EpisodicState state;
    state.deltas.reserve(n_deltas);

    for (uint64_t di = 0; di < n_deltas; ++di) {
        uint32_t n_dims = 0;
        f.read(reinterpret_cast<char*>(&n_dims), sizeof(n_dims));
        if (!f)
            throw std::runtime_error(
                std::format("load_episodic_state: truncated dims for delta {} in '{}'",
                            di, path));

        std::vector<int64_t> shape(n_dims);
        for (uint32_t i = 0; i < n_dims; ++i) {
            f.read(reinterpret_cast<char*>(&shape[i]), sizeof(int64_t));
            if (!f)
                throw std::runtime_error(
                    std::format("load_episodic_state: truncated shape[{}] for delta {} in '{}'",
                                i, di, path));
        }

        Tensor t(shape, DType::Float32);
        auto sp = t.data_as<float>();
        f.read(reinterpret_cast<char*>(sp.data()),
               static_cast<std::streamsize>(sp.size() * sizeof(float)));
        if (!f)
            throw std::runtime_error(
                std::format("load_episodic_state: truncated data for delta {} in '{}'",
                            di, path));

        state.deltas.emplace_back(std::move(t));
    }

    // merged flag
    uint8_t merged_byte = 0;
    f.read(reinterpret_cast<char*>(&merged_byte), sizeof(merged_byte));
    if (!f)
        throw std::runtime_error(
            std::format("load_episodic_state: truncated merged flag in '{}'", path));
    state.merged = (merged_byte != 0);

    return state;
}

} // namespace sub0llm::nn
