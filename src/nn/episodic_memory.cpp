#include "sub0llm/nn/episodic_memory.hpp"
#include "sub0llm/autograd/ops.hpp"
#include "sub0llm/core/tensor.hpp"
#include "sub0llm/nn/optimizer.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <format>
#include <fstream>
#include <limits>
#include <numeric>
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

    // Scoring a known sequence: use the batched forward() — it computes the LM
    // head for all T positions in a single (T,D)·(D,V) matmul.  (Token-by-token
    // forward_one() would redo the full-vocab projection per position, which is
    // slower here; it only wins for generation, where future tokens are unknown.)
    // The autograd graph it builds is discarded — we read logits only.
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
                     const std::vector<std::vector<int32_t>>& docs,
                     const EpisodicConfig& cfg)
{
    if (!cfg.accumulate) {
        if (state.merged) state.unmerge(model);
        state.reset();
    }

    auto params = model.parameters();
    check_size_match(state.deltas, params);

    // ── Comprehension + span detection across ALL documents ─────────────────────
    // Each surprising span becomes one rehearsal sequence (input ids, target ids).
    // Detecting them up front lets the whole set be trained jointly below.
    struct Seq { Tensor input; Tensor tgt; Tensor weight; };  // weight: per-token novelty
    std::vector<Seq> seqs;
    double nll_sum = 0.0;
    std::size_t nll_n = 0;

    for (const auto& tokens : docs) {
        if (tokens.size() < 2) continue;
        const auto losses = comprehension_pass(model, tokens);
        for (float l : losses) { nll_sum += l; ++nll_n; }

        const int64_t T = static_cast<int64_t>(tokens.size()) - 1;
        for (int64_t i = 0; i < T; ) {
            if (losses[static_cast<std::size_t>(i)] <= cfg.surprise_threshold) { ++i; continue; }
            int64_t j = i + 1;
            while (j < T
                   && losses[static_cast<std::size_t>(j)] > cfg.surprise_threshold
                   && j - i < static_cast<int64_t>(cfg.max_span_len))
                ++j;
            if (j - i >= static_cast<int64_t>(cfg.min_span_len)) {
                const int64_t Ts = j - i;
                Tensor in({Ts}, DType::Int32), tg({Ts}, DType::Int32), wt({Ts}, DType::Float32);
                auto id = in.data_as<int32_t>();
                auto td = tg.data_as<int32_t>();
                auto wd = wt.data_as<float>();
                for (int64_t k = 0; k < Ts; ++k) {
                    id[static_cast<std::size_t>(k)] = tokens[static_cast<std::size_t>(i + k)];
                    td[static_cast<std::size_t>(k)] = tokens[static_cast<std::size_t>(i + k + 1)];
                    // Novelty weight = base surprisal of predicting this token
                    // (floored so predictable tokens still get a little signal).
                    wd[static_cast<std::size_t>(k)] =
                        std::max(0.1f, losses[static_cast<std::size_t>(i + k)]);
                }
                seqs.push_back({std::move(in), std::move(tg), std::move(wt)});
            }
            i = j;
        }
    }
    if (seqs.empty()) return;

    const float mean_nll =
        nll_n ? static_cast<float>(nll_sum / static_cast<double>(nll_n)) : 0.0f;

    // Novelty-scaled effort: less rehearsal (fewer steps and/or a gentler lr) when
    // the content is close to what the model already predicts well (low mean NLL).
    const float novelty_frac =
        cfg.novelty_ref > 0.0f ? std::min(1.0f, mean_nll / cfg.novelty_ref) : 1.0f;
    int eff_steps = cfg.think_steps;
    if (cfg.adaptive_steps)
        eff_steps = std::clamp(
            static_cast<int>(std::lround(static_cast<float>(cfg.think_steps) * novelty_frac)),
            1, cfg.think_steps);
    const float eff_lr =
        cfg.adaptive_lr ? cfg.learning_rate * std::max(0.2f, novelty_frac)
                        : cfg.learning_rate;

    if (cfg.on_progress) {
        EpisodicProgress p;
        p.phase      = EpisodicProgress::Phase::Comprehension;
        p.mean_nll   = mean_nll;
        p.span_count = static_cast<int>(seqs.size());
        cfg.on_progress(p);
    }

    // ── Trainable-parameter policy ──────────────────────────────────────────────
    // Save/restore requires_grad; LoRA freezes externally (see enable_episodic_lora),
    // otherwise restrict to the last-N layers here.
    std::vector<bool> orig_rg(params.size());
    for (std::size_t pi = 0; pi < params.size(); ++pi)
        orig_rg[pi] = params[pi]->requires_grad();
    struct RgRestore {
        std::vector<autograd::Variable*>& ps;
        const std::vector<bool>&          rg;
        ~RgRestore() { for (std::size_t i = 0; i < ps.size(); ++i)
                           ps[i]->set_requires_grad(rg[i]); }
    } rg_restore{params, orig_rg};
    if (cfg.lora_rank <= 0)
        model.set_trainable_last_layers(cfg.trainable_last_layers);

    std::vector<bool> needs_grad(params.size());
    for (std::size_t pi = 0; pi < params.size(); ++pi)
        needs_grad[pi] = params[pi]->requires_grad();

    std::vector<autograd::Variable*> trainable_params;
    for (std::size_t pi = 0; pi < params.size(); ++pi)
        if (needs_grad[pi]) trainable_params.push_back(params[pi]);

    SGD opt(trainable_params, eff_lr);

    // Snapshot base weights once; the whole session's net change becomes the delta.
    std::vector<std::vector<float>> orig_vals;
    orig_vals.reserve(params.size());
    for (std::size_t pi = 0; pi < params.size(); ++pi) {
        if (needs_grad[pi]) {
            auto pd = params[pi]->data().data_as<float>();
            orig_vals.emplace_back(pd.begin(), pd.end());
        } else {
            orig_vals.emplace_back();
        }
    }
    auto restore_weights = [&] {
        for (std::size_t pi = 0; pi < params.size(); ++pi) {
            if (!needs_grad[pi]) continue;
            auto wd = params[pi]->data().data_as<float>();
            const auto& ov = orig_vals[pi];
            for (std::size_t k = 0; k < ov.size(); ++k) wd[k] = ov[k];
        }
    };

    // Locality regularisation ("light on existing memory"): snapshot base logits on
    // the generic anchor once; each step is penalised for moving them, so the whole
    // joint write stays specific.  (Model is at base weights here.)
    const bool locality = cfg.locality_weight > 0.0f && cfg.locality_anchor.size() >= 2;
    Tensor             anchor_ids;
    autograd::Variable base_anchor;
    float              anchor_inv_n = 1.0f;
    if (locality) {
        const int64_t Ta = static_cast<int64_t>(cfg.locality_anchor.size());
        anchor_ids = Tensor({Ta}, DType::Int32);
        auto ad = anchor_ids.data_as<int32_t>();
        for (std::size_t k = 0; k < cfg.locality_anchor.size(); ++k)
            ad[k] = cfg.locality_anchor[k];
        Tensor base_logits = copy(model.forward(anchor_ids).data());
        anchor_inv_n = 1.0f / static_cast<float>(base_logits.numel());
        base_anchor  = autograd::Variable(std::move(base_logits), /*requires_grad=*/false);
    }

    // ── Joint rehearsal ─────────────────────────────────────────────────────────
    // One trajectory for the whole document set: each step accumulates gradients
    // from every sequence PLUS the locality penalty, then takes a single clipped
    // SGD step.  All sequences co-train (so a reworded query links to the fact) and
    // the locality term constrains the TOTAL movement (so unrelated text is spared)
    // — neither holds when sequences are trained in independent sessions.
    // SGD (not Adam): its step scales with gradient magnitude, avoiding the lr
    // sensitivity Adam's normalisation causes over short write sessions.
    try {
        for (int t = 0; t < eff_steps; ++t) {
            opt.zero_grad();
            float step_loss = 0.0f;
            for (const auto& s : seqs) {
                auto logits = model.forward(s.input);
                auto loss   = cfg.token_novelty_weight
                            ? autograd::weighted_cross_entropy(logits, s.tgt, s.weight)
                            : autograd::cross_entropy(logits, s.tgt);
                loss.backward();                                   // accumulate grads
                step_loss += loss.data().data_as<float>()[0];
            }
            if (locality) {
                using namespace autograd;
                auto diff = sub(model.forward(anchor_ids), base_anchor);
                auto mse  = scale(sum(mul(diff, diff)), anchor_inv_n);
                scale(mse, cfg.locality_weight).backward();        // accumulate grads
            }
            float gnorm = 0.0f;
            if (cfg.max_grad_norm > 0.0f)
                gnorm = clip_grad_norm(trainable_params, cfg.max_grad_norm);
            opt.step();

            if (cfg.on_progress) {
                EpisodicProgress p;
                p.phase       = EpisodicProgress::Phase::Rehearsal;
                p.span_count  = static_cast<int>(seqs.size());
                p.step        = t;
                p.total_steps = eff_steps;
                p.loss        = step_loss / static_cast<float>(seqs.size());
                p.grad_norm   = gnorm;
                cfg.on_progress(p);
            }
        }
    } catch (...) {
        restore_weights();
        throw;
    }

    // Fold the whole session's net change into the delta, then restore the base.
    for (std::size_t pi = 0; pi < params.size(); ++pi) {
        if (!needs_grad[pi]) continue;
        const std::size_t n = orig_vals[pi].size();
        if (n == 0) continue;
        auto wd = params[pi]->data().data_as<float>();
        auto dd = state.deltas[pi].data_as<float>();
        for (std::size_t k = 0; k < n; ++k) {
            dd[k] += wd[k] - orig_vals[pi][k];
            wd[k]  = orig_vals[pi][k];
        }
    }
}

// Single-document convenience overload.
void episodic_encode(ModernGPT& model, EpisodicState& state,
                     const std::vector<int32_t>& tokens,
                     const EpisodicConfig& cfg)
{
    episodic_encode(model, state, std::vector<std::vector<int32_t>>{tokens}, cfg);
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
