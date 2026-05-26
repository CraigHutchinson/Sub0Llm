#include "sub0llm/nn/mtp.hpp"

#include "sub0llm/autograd/ops.hpp"

#include <format>
#include <stdexcept>

namespace sub0llm::nn {

using namespace autograd;

// ── helpers ───────────────────────────────────────────────────────────────────

namespace {

// Extract last row of a (T, V) logit tensor → (V,) Tensor.
static Tensor last_row(const Tensor& logits) {
    if (logits.ndim() != 2)
        throw std::runtime_error(std::format(
            "mtp: last_row expected 2D (T,V) tensor, got {}D", logits.ndim()));
    const int64_t T = logits.shape(0);
    const int64_t V = logits.shape(1);
    Tensor out({V}, DType::Float32);
    auto src = logits.data_as<float>();
    auto dst = out.data_as<float>();
    const std::size_t off = static_cast<std::size_t>((T - 1) * V);
    for (std::size_t j = 0; j < static_cast<std::size_t>(V); ++j)
        dst[j] = src[off + j];
    return out;
}

// Build Int32 token tensor from a vector slice [begin, begin+len).
static Tensor make_target(const std::vector<int32_t>& toks,
                           std::size_t begin, int64_t len) {
    Tensor t({len}, DType::Int32);
    auto sp = t.data_as<int32_t>();
    for (std::size_t i = 0; i < static_cast<std::size_t>(len); ++i)
        sp[i] = toks[begin + i];
    return t;
}

// Build Int32 Tensor from raw token_ids span starting at 0.
static Tensor make_ids_tensor(const std::vector<int32_t>& toks, int64_t len) {
    Tensor t({len}, DType::Int32);
    auto sp = t.data_as<int32_t>();
    for (int64_t i = 0; i < len; ++i)
        sp[static_cast<std::size_t>(i)] = toks[static_cast<std::size_t>(i)];
    return t;
}

static int32_t sample_last(const Tensor& logits, const SamplingConfig& cfg,
                             std::mt19937& rng) {
    Tensor row = last_row(logits);
    switch (cfg.mode) {
        case SamplingMode::Greedy:      return greedy_sample(row);
        case SamplingMode::Temperature: return temperature_sample(row, cfg.temperature, rng);
        case SamplingMode::TopK:        return top_k_sample(row, cfg.top_k, cfg.temperature, rng);
        case SamplingMode::TopP:        return top_p_sample(row, cfg.top_p, cfg.temperature, rng);
        default: throw std::runtime_error(
            std::format("mtp: unknown SamplingMode {}", static_cast<int>(cfg.mode)));
    }
}

} // anonymous namespace

// ── mtp_train_loss ────────────────────────────────────────────────────────────

Variable mtp_train_loss(ModernGPT& model, const Tensor& token_ids,
                         int64_t seq_len, float aux_weight)
{
    if (model.n_mtp_heads() < 1)
        throw std::runtime_error(
            std::format("mtp_train_loss: model has no MTP heads "
                        "(n_mtp_heads={}); construct with n_mtp_heads >= 1",
                        model.n_mtp_heads()));
    if (token_ids.ndim() != 1)
        throw std::runtime_error("mtp_train_loss: token_ids must be 1D (T,)");
    const int64_t T  = token_ids.shape(0);
    const int64_t K  = model.n_mtp_heads();
    if (T != seq_len)
        throw std::runtime_error(std::format(
            "mtp_train_loss: token_ids length {} != seq_len {}", T, seq_len));
    const int64_t min_len = 2 + K;
    if (T < min_len)
        throw std::runtime_error(std::format(
            "mtp_train_loss: seq_len={} must be >= 2+K={}", T, min_len));
    if (aux_weight < 0.f || aux_weight > 1.f)
        throw std::runtime_error(std::format(
            "mtp_train_loss: aux_weight={} must be in [0,1]", aux_weight));

    // Build raw token id vector for target extraction.
    std::vector<int32_t> id_vec(static_cast<std::size_t>(T));
    auto id_sp = token_ids.data_as<int32_t>();
    for (std::size_t i = 0; i < static_cast<std::size_t>(T); ++i)
        id_vec[i] = id_sp[i];

    // Forward through all heads.
    auto heads = model.forward_mtp(token_ids);  // size = 1 + K

    // Accumulate losses.
    // Head k predicts the token at position t+k+1 for context position t.
    // Usable positions: 0 … T-k-2  (length T-k-1).
    Variable total_loss;
    bool first = true;
    for (int64_t k = 0; k <= K; ++k) {
        const int64_t n_pos = T - k - 1;  // number of prediction positions
        if (n_pos < 1) break;

        // Logits for positions 0 … n_pos-1.
        Variable logits_k = narrow(heads[static_cast<std::size_t>(k)], 0, n_pos);

        // Targets: token_ids[k+1 … T-1].
        Tensor targets_k = make_target(id_vec, static_cast<std::size_t>(k + 1), n_pos);

        Variable ce_k = cross_entropy(logits_k, targets_k);

        float w = (k == 0) ? 1.0f : aux_weight;
        Variable weighted = scale(ce_k, w);

        total_loss = first ? weighted : add(total_loss, weighted);
        first = false;
    }
    return total_loss;
}

// ── mtp_generate_stats ────────────────────────────────────────────────────────

MtpGenStats mtp_generate_stats(ModernGPT& model,
                                 const std::vector<int32_t>& prompt,
                                 int64_t max_new_tokens,
                                 const SamplingConfig& cfg,
                                 std::mt19937& rng)
{
    if (prompt.empty())
        throw std::runtime_error("mtp_generate: prompt must not be empty");
    if (max_new_tokens < 1)
        throw std::runtime_error(std::format(
            "mtp_generate: max_new_tokens={} must be >= 1", max_new_tokens));

    const int64_t K = model.n_mtp_heads();

    MtpGenStats stats;
    stats.tokens = prompt;
    stats.tokens.reserve(prompt.size() + static_cast<std::size_t>(max_new_tokens));

    int64_t remaining = max_new_tokens;

    if (K == 0) {
        // Fallback: standard single-token generation.
        while (remaining > 0) {
            Tensor ids = make_ids_tensor(stats.tokens,
                                          static_cast<int64_t>(stats.tokens.size()));
            Tensor logits = model.forward(ids).data();
            stats.tokens.push_back(sample_last(logits, cfg, rng));
            --remaining;
            ++stats.n_forward_passes;
        }
    } else {
        // MTP: produce up to K+1 tokens per forward pass.
        while (remaining > 0) {
            Tensor ids = make_ids_tensor(stats.tokens,
                                          static_cast<int64_t>(stats.tokens.size()));
            auto heads = model.forward_mtp(ids);  // 1 + K logit variables

            const int64_t n_draft = std::min(K + 1, remaining);
            for (int64_t k = 0; k < n_draft; ++k) {
                Tensor logits = heads[static_cast<std::size_t>(k)].data();
                stats.tokens.push_back(sample_last(logits, cfg, rng));
                --remaining;
            }
            ++stats.n_forward_passes;
        }
    }

    const int64_t n_new = static_cast<int64_t>(stats.tokens.size())
                          - static_cast<int64_t>(prompt.size());
    stats.tokens_per_pass = (stats.n_forward_passes > 0)
        ? static_cast<double>(n_new) / static_cast<double>(stats.n_forward_passes)
        : 0.0;

    return stats;
}

std::vector<int32_t> mtp_generate(ModernGPT& model,
                                    const std::vector<int32_t>& prompt,
                                    int64_t max_new_tokens,
                                    const SamplingConfig& cfg,
                                    std::mt19937& rng)
{
    return mtp_generate_stats(model, prompt, max_new_tokens, cfg, rng).tokens;
}

} // namespace sub0llm::nn
