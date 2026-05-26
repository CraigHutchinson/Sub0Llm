#pragma once

#include "sub0llm/autograd/variable.hpp"
#include "sub0llm/core/tensor.hpp"
#include "sub0llm/nn/modern_gpt.hpp"

#include <cstdint>
#include <vector>

namespace sub0llm::nn {

// ── MoEFeedForward ────────────────────────────────────────────────────────────
//
// Mixture-of-Experts FFN layer (Shazeer et al. 2017 / Switch Transformer 2021).
// E independent SwiGLU expert FFNs share the same input dimension D.
// A learned router (Linear D→E) selects the top-k experts per token:
//
//   router_logits = x · W_router^T          (T, E)
//   gates, indices = top_k(softmax(router_logits), k)  per token
//   output[t] = sum_{i in top_k} gates[t,i] * expert_i(x[t])
//
// Auxiliary load-balancing loss (optional, for training):
//   L_aux = E * sum_e f_e * p_e
//   where f_e = fraction of tokens routed to expert e,
//         p_e = mean router probability for expert e.
//
// n_experts : total number of expert FFNs (E)
// top_k     : how many experts each token uses (typically 1 or 2)
// d_ff      : each expert's hidden dim (0 = auto: 8*D/3 rounded to 64)
class MoEFeedForward {
public:
    MoEFeedForward(int64_t     D,
                   int64_t     n_experts,
                   int64_t     top_k,
                   int64_t     d_ff  = 0,
                   std::uint64_t seed = 42);

    // x: Variable (T, D)
    // Returns {output (T, D), aux_loss scalar}.
    // aux_loss is the load-balancing loss; add it scaled by a small coefficient
    // (e.g. 0.01) to the main CE loss during training.
    struct ForwardResult {
        autograd::Variable output;
        autograd::Variable aux_loss;
    };
    [[nodiscard]] ForwardResult forward(const autograd::Variable& x) const;

    [[nodiscard]] std::vector<autograd::Variable*> parameters();

    [[nodiscard]] int64_t n_experts() const noexcept { return n_experts_; }
    [[nodiscard]] int64_t top_k()     const noexcept { return top_k_; }

private:
    int64_t                           D_;
    int64_t                           n_experts_;
    int64_t                           top_k_;
    Linear                            router_;
    std::vector<SwiGLUFeedForward>    experts_;
};

// ── MoETransformerBlock ───────────────────────────────────────────────────────
//
// Pre-norm GPT block identical to ModernTransformerBlock but with the SwiGLU
// FFN replaced by a MoEFeedForward layer.
//
//   h = x + GQA_RoPE(RMSNorm(x))
//   y, aux = MoE(RMSNorm(h))
//   output = h + y
class MoETransformerBlock {
public:
    MoETransformerBlock(int64_t     D,
                        std::size_t n_heads,
                        std::size_t n_kv_heads,
                        int64_t     n_experts,
                        int64_t     top_k,
                        int64_t     d_ff   = 0,
                        std::uint64_t seed = 42);

    struct ForwardResult {
        autograd::Variable output;
        autograd::Variable aux_loss;
    };
    [[nodiscard]] ForwardResult forward(const autograd::Variable& x) const;

    [[nodiscard]] std::vector<autograd::Variable*> parameters();

private:
    RMSNorm              norm1_;
    GroupedQueryAttention attn_;
    RMSNorm              norm2_;
    MoEFeedForward       moe_;
};

// ── MoEGPT ───────────────────────────────────────────────────────────────────
//
// LLaMA-style GPT with all FFN layers replaced by Mixture-of-Experts.
//
//   • Token embedding (weight-tied LM head)
//   • N × MoETransformerBlock
//   • Final RMSNorm
//   • LM head = transpose(tok_emb weight)
//
// forward() returns only the main logits.
// forward_moe() also returns the mean auxiliary load-balancing loss across all
// blocks so callers can add it to their training objective.
class MoEGPT {
public:
    MoEGPT(int64_t     vocab_size,
           int64_t     embed_dim,
           std::size_t n_heads,
           std::size_t n_kv_heads,
           int64_t     n_layers,
           int64_t     n_experts,
           int64_t     top_k,
           int64_t     d_ff   = 0,
           std::uint64_t seed = 42);

    // Returns logits (T, vocab_size). Auxiliary losses are discarded — use for
    // inference or when you don't need load-balancing regularisation.
    [[nodiscard]] autograd::Variable forward(const Tensor& token_ids) const;

    // Returns {logits (T, V), aux_loss scalar (mean across blocks)}.
    // Use aux_loss during training: total_loss = ce_loss + aux_coeff * aux_loss.
    struct MoEForwardResult {
        autograd::Variable logits;
        autograd::Variable aux_loss;
    };
    [[nodiscard]] MoEForwardResult forward_moe(const Tensor& token_ids) const;

    [[nodiscard]] std::vector<autograd::Variable*> parameters();

    [[nodiscard]] int64_t     vocab_size()  const noexcept;
    [[nodiscard]] int64_t     embed_dim()   const noexcept;
    [[nodiscard]] std::size_t num_layers()  const noexcept;
    [[nodiscard]] int64_t     n_experts()   const noexcept;
    [[nodiscard]] int64_t     top_k()       const noexcept;

private:
    Embedding                       tok_emb_;
    std::vector<MoETransformerBlock> blocks_;
    RMSNorm                         ln_f_;
    int64_t                         n_experts_;
    int64_t                         top_k_;
};

} // namespace sub0llm::nn
