#pragma once

#include "sub0llm/autograd/variable.hpp"
#include "sub0llm/core/tensor.hpp"
#include "sub0llm/nn/embedding.hpp"
#include "sub0llm/nn/gpt.hpp"
#include "sub0llm/nn/modern_gpt.hpp"

#include <cstdint>
#include <vector>

namespace sub0llm::nn {

// ── RewardModel ───────────────────────────────────────────────────────────────
//
// LLaMA-style transformer backbone + scalar reward head (Linear D→1).
// Same architecture as ModernGPT but the output head is a scalar projection
// rather than the weight-tied vocabulary projection.
//
// Used in RLHF to score token sequences:
//   forward(ids) → (T, 1)  per-token scalar rewards
//   score(ids)   → {1}     sequence-level reward at the last position
class RewardModel {
public:
    RewardModel(int64_t     vocab_size,
                int64_t     embed_dim,
                std::size_t n_heads,
                std::size_t n_kv_heads,
                int64_t     n_layers,
                int64_t     d_ff      = 0,
                std::uint64_t seed    = 42);

    // Per-token scalar rewards.  Returns (T, 1) Variable.
    [[nodiscard]] autograd::Variable forward(const Tensor& token_ids) const;

    // Sequence-level reward: last-position scalar, shape {1}.
    [[nodiscard]] autograd::Variable score(const Tensor& token_ids) const;

    [[nodiscard]] std::vector<autograd::Variable*> parameters();

    [[nodiscard]] int64_t vocab_size() const noexcept;
    [[nodiscard]] int64_t embed_dim()  const noexcept;
    [[nodiscard]] int64_t num_layers() const noexcept;

private:
    Embedding                           tok_emb_;
    std::vector<ModernTransformerBlock> blocks_;
    RMSNorm                             ln_f_;
    Linear                              reward_head_;  // D → 1
    int64_t                             vocab_size_;
    int64_t                             embed_dim_;
};

// ── Preference loss (Bradley-Terry) ──────────────────────────────────────────
//
// L = -log σ(r_chosen - r_rejected)
//
// Trains the reward model to satisfy r_chosen > r_rejected for preferred
// completions.  Both arguments must be scalar Variables (numel == 1).
[[nodiscard]] autograd::Variable reward_preference_loss(
    const autograd::Variable& r_chosen,
    const autograd::Variable& r_rejected);

// ── REINFORCE policy gradient loss ────────────────────────────────────────────
//
// L = reward × CE(logits, token_ids)
//
// Gradient:  ∇L = reward × ∇CE = −reward × Σ_t ∇ log π(token_t | context_t)
//
// reward > 0 → encourages the policy to assign higher probability to token_ids
//              (acts like scaled supervised fine-tuning on those tokens).
// reward < 0 → discourages those tokens.
//
// Requirements:
//   logits   : (T, vocab_size) Variable from the policy being trained
//   token_ids: (T,) int32 Tensor
//   reward   : scalar signal from the frozen reward model
[[nodiscard]] autograd::Variable reinforce_loss(
    const autograd::Variable& logits,
    const Tensor&              token_ids,
    float                      reward);

// ── KL divergence penalty ─────────────────────────────────────────────────────
//
//   KL(π_new ‖ π_ref) = (1/T) Σ_t Σ_v π_new(v|t) (log π_new(v|t) − log π_ref(v|t))
//
// logits_new is differentiable; logits_ref is a frozen constant Tensor.
//
// Returns a non-negative scalar Variable.
// Weight with scale(kl_penalty(...), β) before adding to the policy loss.
//
// Requirements:
//   logits_new: (T, V) Variable
//   logits_ref: (T, V) Tensor with the same shape
[[nodiscard]] autograd::Variable kl_penalty(
    const autograd::Variable& logits_new,
    const Tensor&              logits_ref);

} // namespace sub0llm::nn
