#include "sub0llm/nn/rlhf.hpp"

#include "sub0llm/autograd/ops.hpp"

#include <cmath>
#include <format>
#include <stdexcept>

namespace sub0llm::nn {

using namespace autograd;

// ── helpers ───────────────────────────────────────────────────────────────────

namespace {

int64_t rm_validate(int64_t vocab_size, int64_t embed_dim) {
    if (vocab_size <= 0)
        throw std::runtime_error(
            std::format("RewardModel: vocab_size={} must be positive", vocab_size));
    if (embed_dim <= 0)
        throw std::runtime_error(
            std::format("RewardModel: embed_dim={} must be positive", embed_dim));
    return embed_dim;
}

} // anonymous namespace

// ── RewardModel ───────────────────────────────────────────────────────────────

RewardModel::RewardModel(int64_t     vocab_size,
                         int64_t     embed_dim,
                         std::size_t n_heads,
                         std::size_t n_kv_heads,
                         int64_t     n_layers,
                         int64_t     d_ff,
                         std::uint64_t seed)
    : tok_emb_(vocab_size, rm_validate(vocab_size, embed_dim), seed),
      ln_f_(embed_dim),
      reward_head_(embed_dim, 1,
                   seed + 2 + static_cast<std::uint64_t>(n_layers) * 2000 + 1),
      vocab_size_(vocab_size),
      embed_dim_(embed_dim)
{
    if (n_heads == 0)
        throw std::runtime_error(std::format(
            "RewardModel: n_heads={} must be positive", n_heads));
    if (n_kv_heads == 0 || n_kv_heads > n_heads)
        throw std::runtime_error(std::format(
            "RewardModel: n_kv_heads={} must be in [1, n_heads={}]",
            n_kv_heads, n_heads));
    if (n_heads % n_kv_heads != 0)
        throw std::runtime_error(std::format(
            "RewardModel: n_heads={} must be divisible by n_kv_heads={}",
            n_heads, n_kv_heads));
    if (n_layers <= 0)
        throw std::runtime_error(std::format(
            "RewardModel: n_layers={} must be positive", n_layers));

    blocks_.reserve(static_cast<std::size_t>(n_layers));
    for (int64_t l = 0; l < n_layers; ++l)
        blocks_.emplace_back(embed_dim, n_heads, n_kv_heads, d_ff,
                             seed + 2 + static_cast<std::uint64_t>(l) * 2000);
}

Variable RewardModel::forward(const Tensor& token_ids) const {
    if (token_ids.ndim() != 1)
        throw std::runtime_error(
            "RewardModel::forward: token_ids must be 1D (T,)");
    if (token_ids.shape(0) < 1)
        throw std::runtime_error(
            "RewardModel::forward: token_ids must be non-empty");
    auto x = tok_emb_.forward(token_ids);            // (T, D)
    for (const auto& block : blocks_) x = block.forward(x);
    x = ln_f_.forward(x);
    return reward_head_.forward(x);                  // (T, 1)
}

Variable RewardModel::score(const Tensor& token_ids) const {
    auto rewards = forward(token_ids);               // (T, 1)
    const int64_t T = rewards.data().shape(0);       // use actual output shape
    return sum(narrow(rewards, T - 1, 1));           // scalar {1}
}

std::vector<Variable*> RewardModel::parameters() {
    std::vector<Variable*> p;
    p.push_back(&tok_emb_.weight());
    for (auto& block : blocks_)
        for (auto* v : block.parameters())    p.push_back(v);
    for (auto* v : ln_f_.parameters())        p.push_back(v);
    for (auto* v : reward_head_.parameters()) p.push_back(v);
    return p;
}

int64_t RewardModel::vocab_size() const noexcept { return vocab_size_; }
int64_t RewardModel::embed_dim()  const noexcept { return embed_dim_; }
int64_t RewardModel::num_layers() const noexcept {
    return static_cast<int64_t>(blocks_.size());
}

// ── reward_preference_loss ────────────────────────────────────────────────────

Variable reward_preference_loss(const Variable& r_chosen,
                                 const Variable& r_rejected)
{
    if (r_chosen.data().numel() != 1)
        throw std::runtime_error(std::format(
            "reward_preference_loss: r_chosen must be scalar (numel=1), got {}",
            r_chosen.data().numel()));
    if (r_rejected.data().numel() != 1)
        throw std::runtime_error(std::format(
            "reward_preference_loss: r_rejected must be scalar (numel=1), got {}",
            r_rejected.data().numel()));
    // L = -log σ(r_chosen - r_rejected)
    return scale(log_sigmoid(sub(r_chosen, r_rejected)), -1.0f);
}

// ── reinforce_loss ────────────────────────────────────────────────────────────

Variable reinforce_loss(const Variable& logits,
                         const Tensor&   token_ids,
                         float           reward)
{
    if (logits.data().ndim() != 2)
        throw std::runtime_error(std::format(
            "reinforce_loss: logits must be 2D (T,V), got {}D",
            logits.data().ndim()));
    if (token_ids.ndim() != 1)
        throw std::runtime_error(
            "reinforce_loss: token_ids must be 1D (T,)");
    if (logits.data().shape(0) != token_ids.shape(0))
        throw std::runtime_error(std::format(
            "reinforce_loss: logits T={} != token_ids T={}",
            logits.data().shape(0), token_ids.shape(0)));
    if (!std::isfinite(reward))
        throw std::runtime_error(std::format(
            "reinforce_loss: reward={} must be finite", reward));
    return scale(cross_entropy(logits, token_ids), reward);
}

// ── kl_penalty ────────────────────────────────────────────────────────────────

Variable kl_penalty(const Variable& logits_new, const Tensor& logits_ref) {
    if (logits_new.data().ndim() != 2)
        throw std::runtime_error(std::format(
            "kl_penalty: logits_new must be 2D (T,V), got {}D",
            logits_new.data().ndim()));
    if (logits_ref.ndim() != 2)
        throw std::runtime_error(std::format(
            "kl_penalty: logits_ref must be 2D (T,V), got {}D",
            logits_ref.ndim()));
    if (logits_new.data().shape(0) != logits_ref.shape(0) ||
        logits_new.data().shape(1) != logits_ref.shape(1))
        throw std::runtime_error(std::format(
            "kl_penalty: shape mismatch ({},{}) vs ({},{})",
            logits_new.data().shape(0), logits_new.data().shape(1),
            logits_ref.shape(0), logits_ref.shape(1)));

    const int64_t T = logits_new.data().shape(0);
    if (T < 1)
        throw std::runtime_error(std::format(
            "kl_penalty: sequence length T={} must be >= 1", T));

    // Differentiable terms from the new policy
    auto log_p_new = log_softmax(logits_new);           // (T, V)
    auto p_new     = softmax(logits_new);               // (T, V)

    // Reference terms are constant — no gradient flows through them
    Variable logits_ref_var(logits_ref, false);
    auto log_p_ref = log_softmax(logits_ref_var);       // (T, V)

    // KL(new || ref) = sum_v p_new * (log_p_new - log_p_ref)
    auto kl_pts   = mul(p_new, sub(log_p_new, log_p_ref));  // (T, V)
    auto kl_total = sum(kl_pts);                             // scalar (sum T×V)

    // Normalise by T: mean per-position KL
    return scale(kl_total, 1.0f / static_cast<float>(T));
}

} // namespace sub0llm::nn
