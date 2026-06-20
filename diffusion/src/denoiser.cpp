#include "sub0diff/nn/denoiser.hpp"
#include "sub0diff/nn/time_embedding.hpp"

#include "sub0llm/autograd/ops.hpp"

#include <algorithm>

namespace sub0diff::nn {

using sub0llm::autograd::Variable;
namespace ag = sub0llm::autograd;

// ── BidirectionalBlock ───────────────────────────────────────────────────────────

BidirectionalBlock::BidirectionalBlock(std::int64_t D, std::size_t n_heads,
                                       std::size_t n_kv_heads, std::int64_t d_ff,
                                       std::uint64_t seed, float rope_base)
    : norm1_(D),
      attn_(static_cast<std::size_t>(D), n_heads, n_kv_heads, rope_base, seed,
            /*window_size=*/-1, /*head_dim=*/0, /*use_qk_norm=*/false, /*alloc_weights=*/true),
      norm2_(D),
      ffn_(D, d_ff, seed + 1) {}

Variable BidirectionalBlock::forward(const Variable& x) const {
    // Bidirectional: causal=false lets every canvas position attend to every other.
    Variable h = ag::add(x, attn_.forward(norm1_.forward(x), /*causal=*/false));
    Variable y = ag::add(h, ffn_.forward(norm2_.forward(h)));
    return y;
}

Variable BidirectionalBlock::forward(const Variable& x, std::int64_t B, std::int64_t T) const {
    // Same residual structure as the 2D path; attention runs block-diagonal over the
    // B windows while the row-wise norm/FFN see all B·T rows at once.
    Variable h = ag::add(x, attn_.forward(norm1_.forward(x), B, T, /*causal=*/false));
    Variable y = ag::add(h, ffn_.forward(norm2_.forward(h)));
    return y;
}

std::vector<Variable*> BidirectionalBlock::parameters() {
    std::vector<Variable*> p;
    auto take = [&](std::vector<Variable*> v) { p.insert(p.end(), v.begin(), v.end()); };
    take(norm1_.parameters());
    take(attn_.parameters());
    take(norm2_.parameters());
    take(ffn_.parameters());
    return p;
}

// ── Denoiser ───────────────────────────────────────────────────────────────────

Denoiser::Denoiser(std::int64_t vocab_size, std::int64_t embed_dim,
                   std::size_t n_heads, std::size_t n_kv_heads, std::int64_t n_layers,
                   std::int64_t d_ff, std::uint64_t seed)
    : real_vocab_(vocab_size),
      embed_dim_(embed_dim),
      tok_emb_(vocab_size + 1, embed_dim, seed),   // +1 row for the [MASK] id
      ln_f_(embed_dim) {
    blocks_.reserve(static_cast<std::size_t>(n_layers));
    for (std::int64_t l = 0; l < n_layers; ++l)
        blocks_.emplace_back(embed_dim, n_heads, n_kv_heads, d_ff,
                             seed + 100 + static_cast<std::uint64_t>(l) * 17);
}

Variable Denoiser::forward(const sub0llm::Tensor& token_ids, float noise_level) const {
    const std::int64_t T = token_ids.numel();

    // 1. token embedding (T, D) + noise-level conditioning (broadcast over positions)
    Variable x = tok_emb_.forward(token_ids);
    // Conditioning is a host-built constant (no grad); move it to x's device so the add stays
    // on-device when the model lives on CUDA (.to is a no-op on CPU). Stage 4 Phase 7.
    Variable t_emb{time_embedding_rows(noise_level, T, embed_dim_).to(x.data().device())};
    x = ag::add(x, t_emb);

    // 2. bidirectional transformer stack
    for (const auto& blk : blocks_) x = blk.forward(x);

    // 3. final norm + weight-tied LM head: logits = x · Wᵀ, W = tok_emb (Vm, D)
    x = ln_f_.forward(x);
    return ag::matmul_bt(x, tok_emb_.weight());  // (T, Vm) — fused x · Wᵀ, W row-major (Vm, D)
}

Variable Denoiser::forward(const sub0llm::Tensor& token_ids,
                           std::span<const float> noise_levels,
                           std::int64_t B, std::int64_t T) const {
    if (token_ids.numel() != B * T)
        throw std::runtime_error("Denoiser batched forward: token_ids must have B*T elements");
    if (static_cast<std::int64_t>(noise_levels.size()) != B)
        throw std::runtime_error("Denoiser batched forward: need one noise level per window (B)");

    // 1. token embedding (B·T, D)
    Variable x = tok_emb_.forward(token_ids);

    // 2. per-window noise conditioning: row b·T+t uses window b's level (broadcast over t).
    sub0llm::Tensor cond({B * T, embed_dim_}, sub0llm::DType::Float32);
    auto cs = cond.data_as<float>();
    for (std::int64_t b = 0; b < B; ++b) {
        sub0llm::Tensor rows = time_embedding_rows(noise_levels[static_cast<std::size_t>(b)],
                                                   T, embed_dim_);  // (T, D)
        auto rs = rows.data_as<float>();
        std::copy_n(rs.begin(), T * embed_dim_, cs.begin() + b * T * embed_dim_);
    }
    // Host-built constant (no grad) → move to x's device for an on-device add (Stage 4 Phase 7).
    x = ag::add(x, Variable{cond.to(x.data().device())});

    // 3. bidirectional transformer stack (batched, block-diagonal attention)
    for (const auto& blk : blocks_) x = blk.forward(x, B, T);

    // 4. final norm + weight-tied LM head → (B·T, Vm)
    x = ln_f_.forward(x);
    return ag::matmul_bt(x, tok_emb_.weight());
}

std::vector<Variable*> Denoiser::parameters() {
    std::vector<Variable*> p;
    p.push_back(&tok_emb_.weight());
    for (auto& blk : blocks_) {
        auto bp = blk.parameters();
        p.insert(p.end(), bp.begin(), bp.end());
    }
    auto np = ln_f_.parameters();
    p.insert(p.end(), np.begin(), np.end());
    return p;
}

Denoiser& Denoiser::to(sub0llm::Device device) {
    for (auto* p : parameters())
        p->to(device);
    return *this;
}

} // namespace sub0diff::nn
