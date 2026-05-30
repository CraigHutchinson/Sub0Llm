#include "sub0llm/nn/modern_gpt.hpp"

#include "sub0llm/autograd/ops.hpp"
#include "sub0llm/core/ops.hpp"

#include <cmath>
#include <format>
#include <limits>
#include <random>
#include <stdexcept>

namespace sub0llm::nn {

namespace {

int64_t swiglu_d_ff(int64_t D, int64_t d_ff) {
    if (d_ff > 0) return d_ff;
    const int64_t raw = (8 * D + 2) / 3;
    const int64_t r   = ((raw + 63) / 64) * 64;
    return r > 0 ? r : 64;
}

// Validates D>0 and returns D; used as first argument in SwiGLUFeedForward
// initializer list so the guard fires before any Linear is constructed.
int64_t swiglu_check_D(int64_t D) {
    if (D <= 0)
        throw std::runtime_error(
            std::format("SwiGLUFeedForward: D={} must be positive", D));
    return D;
}

// Validates vocab_size>0 and embed_dim>0; returns embed_dim for use in the
// ModernGPT initializer list before tok_emb_/ln_f_ are constructed.
int64_t mgpt_validate(int64_t vocab_size, int64_t embed_dim) {
    if (vocab_size <= 0)
        throw std::runtime_error(
            std::format("ModernGPT: vocab_size={} must be positive", vocab_size));
    if (embed_dim <= 0)
        throw std::runtime_error(
            std::format("ModernGPT: embed_dim={} must be positive", embed_dim));
    return embed_dim;
}

autograd::Variable xavier_var(int64_t rows, int64_t cols,
                               std::mt19937_64& rng, std::string name) {
    const float a = std::sqrt(6.0f / static_cast<float>(rows + cols));
    std::uniform_real_distribution<float> dist(-a, a);
    Tensor w = zeros({rows, cols});
    for (auto& v : w.data_as<float>()) v = dist(rng);
    return autograd::Variable(std::move(w), true, std::move(name));
}

// Precompute RoPE cos/sin frequency tables for T positions and head dim Dh.
// Returns tables of shape (T, Dh/2).
void compute_rope_freqs(int64_t T, int64_t Dh, float base,
                        Tensor& cos_out, Tensor& sin_out) {
    const int64_t D2 = Dh / 2;
    cos_out = zeros({T, D2});
    sin_out = zeros({T, D2});
    auto cd = cos_out.data_as<float>();
    auto sd = sin_out.data_as<float>();
    for (int64_t t = 0; t < T; ++t)
        for (int64_t i = 0; i < D2; ++i) {
            const float theta = std::pow(base, -2.0f * static_cast<float>(i) /
                                               static_cast<float>(Dh));
            const float angle = static_cast<float>(t) * theta;
            const auto ti = static_cast<std::size_t>(t * D2 + i);
            cd[ti] = std::cos(angle);
            sd[ti] = std::sin(angle);
        }
}

// Lower-triangular causal mask: 0 on/below diagonal, −∞ above.
Tensor causal_mask(int64_t T) {
    Tensor m = zeros({T, T});
    auto   md = m.data_as<float>();
    const float neg_inf = -std::numeric_limits<float>::infinity();
    const auto  Ts = static_cast<std::size_t>(T);
    for (std::size_t i = 0; i < Ts; ++i)
        for (std::size_t j = i + 1; j < Ts; ++j)
            md[i * Ts + j] = neg_inf;
    return m;
}

} // anonymous namespace

// ── RMSNorm ───────────────────────────────────────────────────────────────────

RMSNorm::RMSNorm(int64_t D, float eps) : eps_(eps) {
    if (D <= 0)
        throw std::runtime_error(std::format("RMSNorm: D={} must be positive", D));
    Tensor wt = zeros({D});
    for (auto& v : wt.data_as<float>()) v = 1.0f;
    weight_ = autograd::Variable(std::move(wt), true, "rmsnorm.weight");
}

autograd::Variable RMSNorm::forward(const autograd::Variable& x) const {
    return autograd::rms_norm(x, weight_, eps_);
}

std::vector<autograd::Variable*> RMSNorm::parameters() {
    return {&weight_};
}

int64_t RMSNorm::dim() const noexcept {
    return weight_.data().numel();
}

// ── SwiGLUFeedForward ─────────────────────────────────────────────────────────

SwiGLUFeedForward::SwiGLUFeedForward(int64_t D, int64_t d_ff, std::uint64_t seed)
    : gate_(swiglu_check_D(D), swiglu_d_ff(D, d_ff), seed),
      up_  (D, swiglu_d_ff(D, d_ff), seed + 1),
      down_(swiglu_d_ff(D, d_ff), D, seed + 2) {}

autograd::Variable SwiGLUFeedForward::forward(const autograd::Variable& x) const {
    using namespace autograd;
    auto gate_out = silu(gate_.forward(x));  // (T, d_ff)
    auto up_out   = up_.forward(x);          // (T, d_ff)
    return down_.forward(mul(gate_out, up_out));
}

std::vector<autograd::Variable*> SwiGLUFeedForward::parameters() {
    std::vector<autograd::Variable*> p;
    for (auto* v : gate_.parameters()) p.push_back(v);
    for (auto* v : up_.parameters())   p.push_back(v);
    for (auto* v : down_.parameters()) p.push_back(v);
    return p;
}

// ── GroupedQueryAttention ─────────────────────────────────────────────────────

GroupedQueryAttention::GroupedQueryAttention(std::size_t embed_dim,
                                             std::size_t n_heads,
                                             std::size_t n_kv_heads,
                                             float       rope_base,
                                             std::uint64_t seed)
    : embed_dim_(embed_dim), n_heads_(n_heads),
      n_kv_heads_(n_kv_heads), rope_base_(rope_base) {
    if (embed_dim == 0 || n_heads == 0 || n_kv_heads == 0)
        throw std::runtime_error(std::format(
            "GroupedQueryAttention: embed_dim={}, n_heads={}, n_kv_heads={} "
            "must all be positive", embed_dim, n_heads, n_kv_heads));
    if (embed_dim % n_heads != 0)
        throw std::runtime_error(std::format(
            "GroupedQueryAttention: embed_dim={} must be divisible by n_heads={}",
            embed_dim, n_heads));
    if (n_heads % n_kv_heads != 0)
        throw std::runtime_error(std::format(
            "GroupedQueryAttention: n_heads={} must be divisible by n_kv_heads={}",
            n_heads, n_kv_heads));

    head_dim_ = embed_dim / n_heads;
    if (head_dim_ % 2 != 0)
        throw std::runtime_error(std::format(
            "GroupedQueryAttention: head_dim={} must be even for RoPE", head_dim_));

    const int64_t D  = static_cast<int64_t>(embed_dim_);
    const int64_t Dh = static_cast<int64_t>(head_dim_);
    std::mt19937_64 rng(seed);

    W_Q_.reserve(n_heads_);
    W_O_.reserve(n_heads_);
    for (std::size_t h = 0; h < n_heads_; ++h) {
        W_Q_.push_back(xavier_var(D, Dh, rng, std::format("gqa.W_Q.{}", h)));
        W_O_.push_back(xavier_var(Dh, D, rng, std::format("gqa.W_O.{}", h)));
    }
    W_K_.reserve(n_kv_heads_);
    W_V_.reserve(n_kv_heads_);
    for (std::size_t g = 0; g < n_kv_heads_; ++g) {
        W_K_.push_back(xavier_var(D, Dh, rng, std::format("gqa.W_K.{}", g)));
        W_V_.push_back(xavier_var(D, Dh, rng, std::format("gqa.W_V.{}", g)));
    }
}

autograd::Variable GroupedQueryAttention::forward(const autograd::Variable& x,
                                                   bool causal) const {
    using namespace autograd;
    if (x.data().ndim() != 2)
        throw std::runtime_error(
            "GroupedQueryAttention::forward: x must be 2D (T, embed_dim)");
    const int64_t T = x.data().shape()[0];
    if (T < 1)
        throw std::runtime_error(std::format(
            "GroupedQueryAttention::forward: T={} must be >= 1", T));

    const float    scale_fac = 1.0f / std::sqrt(static_cast<float>(head_dim_));
    const std::size_t r    = n_heads_ / n_kv_heads_;  // heads per KV group

    // Precompute RoPE frequencies — cached by T to avoid per-step trig ops.
    if (T != rope_cache_T_) {
        compute_rope_freqs(T, static_cast<int64_t>(head_dim_), rope_base_,
                           rope_cache_cos_, rope_cache_sin_);
        rope_cache_T_ = T;
    }
    const Tensor& cos_f = rope_cache_cos_;
    const Tensor& sin_f = rope_cache_sin_;

    // Precompute K and V for each KV head (shared across query groups).
    std::vector<Variable> K_heads(n_kv_heads_), V_heads(n_kv_heads_);
    for (std::size_t g = 0; g < n_kv_heads_; ++g) {
        auto k_raw = matmul(x, W_K_[g]);          // (T, head_dim)
        auto v_raw = matmul(x, W_V_[g]);          // (T, head_dim)
        K_heads[g] = rope(k_raw, cos_f, sin_f);
        V_heads[g] = v_raw;
    }

    // Build causal mask once per T — cached to avoid repeated allocation.
    Variable mask_var;
    if (causal) {
        if (T != mask_cache_T_) {
            mask_cache_   = causal_mask(T);
            mask_cache_T_ = T;
        }
        mask_var = Variable(mask_cache_, false);
    }

    // Accumulate output across all query heads.
    auto compute_head = [&](std::size_t h) -> Variable {
        const std::size_t g = h / r;
        auto q_raw = matmul(x, W_Q_[h]);                        // (T, head_dim)
        auto q     = rope(q_raw, cos_f, sin_f);
        auto scores = scale(matmul(q, transpose2d(K_heads[g])), scale_fac); // (T,T)
        if (causal) scores = add(scores, mask_var);
        auto attn   = softmax(scores);                           // (T, T)
        auto ctx    = matmul(attn, V_heads[g]);                  // (T, head_dim)
        return matmul(ctx, W_O_[h]);                             // (T, D)
    };

    Variable out = compute_head(0);
    for (std::size_t h = 1; h < n_heads_; ++h)
        out = add(out, compute_head(h));
    return out;
}

std::vector<autograd::Variable*> GroupedQueryAttention::parameters() {
    std::vector<autograd::Variable*> params;
    params.reserve(n_heads_ * 2 + n_kv_heads_ * 2);
    for (std::size_t h = 0; h < n_heads_; ++h) {
        params.push_back(&W_Q_[h]);
        params.push_back(&W_O_[h]);
    }
    for (std::size_t g = 0; g < n_kv_heads_; ++g) {
        params.push_back(&W_K_[g]);
        params.push_back(&W_V_[g]);
    }
    return params;
}

// ── ModernTransformerBlock ────────────────────────────────────────────────────

ModernTransformerBlock::ModernTransformerBlock(int64_t     D,
                                               std::size_t n_heads,
                                               std::size_t n_kv_heads,
                                               int64_t     d_ff,
                                               std::uint64_t seed)
    : norm1_(D),
      attn_(static_cast<std::size_t>(D), n_heads, n_kv_heads, 10000.0f, seed),
      norm2_(D),
      ffn_(D, d_ff, seed + 1000) {}

autograd::Variable ModernTransformerBlock::forward(const autograd::Variable& x) const {
    using namespace autograd;
    auto h = add(x, attn_.forward(norm1_.forward(x), /*causal=*/true));
    return add(h, ffn_.forward(norm2_.forward(h)));
}

std::vector<autograd::Variable*> ModernTransformerBlock::parameters() {
    std::vector<autograd::Variable*> p;
    for (auto* v : norm1_.parameters()) p.push_back(v);
    for (auto* v : attn_.parameters())  p.push_back(v);
    for (auto* v : norm2_.parameters()) p.push_back(v);
    for (auto* v : ffn_.parameters())   p.push_back(v);
    return p;
}

// ── ModernGPT ─────────────────────────────────────────────────────────────────

ModernGPT::ModernGPT(int64_t     vocab_size,
                     int64_t     embed_dim,
                     std::size_t n_heads,
                     std::size_t n_kv_heads,
                     int64_t     n_layers,
                     int64_t     d_ff,
                     int64_t     n_mtp,
                     std::uint64_t seed)
    : tok_emb_(vocab_size, mgpt_validate(vocab_size, embed_dim), seed),
      ln_f_(embed_dim) {
    if (n_heads == 0)
        throw std::runtime_error(std::format(
            "ModernGPT: n_heads={} must be positive", n_heads));
    if (n_kv_heads == 0 || n_kv_heads > n_heads)
        throw std::runtime_error(std::format(
            "ModernGPT: n_kv_heads={} must be in [1, n_heads={}]",
            n_kv_heads, n_heads));
    if (n_layers <= 0)
        throw std::runtime_error(std::format(
            "ModernGPT: n_layers={} must be positive", n_layers));
    if (n_mtp < 0)
        throw std::runtime_error(std::format(
            "ModernGPT: n_mtp_heads={} must be >= 0", n_mtp));

    blocks_.reserve(static_cast<std::size_t>(n_layers));
    for (int64_t l = 0; l < n_layers; ++l)
        blocks_.emplace_back(embed_dim, n_heads, n_kv_heads, d_ff,
                             seed + 2 + static_cast<std::uint64_t>(l) * 2000);

    mtp_heads_.reserve(static_cast<std::size_t>(n_mtp));
    for (int64_t k = 0; k < n_mtp; ++k)
        mtp_heads_.emplace_back(embed_dim, vocab_size,
                                seed + 2 + static_cast<std::uint64_t>(n_layers) * 2000
                                         + static_cast<std::uint64_t>(k));
}

autograd::Variable ModernGPT::forward(const Tensor& token_ids) const {
    using namespace autograd;
    if (token_ids.ndim() != 1)
        throw std::runtime_error("ModernGPT::forward: token_ids must be 1D (T,)");
    if (token_ids.shape()[0] < 1)
        throw std::runtime_error("ModernGPT::forward: token_ids must be non-empty");
    auto x = tok_emb_.forward(token_ids);
    for (const auto& block : blocks_) x = block.forward(x);
    x = ln_f_.forward(x);
    return matmul(x, transpose2d(tok_emb_.weight()));
}

std::vector<autograd::Variable> ModernGPT::forward_mtp(const Tensor& token_ids) const {
    using namespace autograd;
    if (token_ids.ndim() != 1)
        throw std::runtime_error("ModernGPT::forward_mtp: token_ids must be 1D (T,)");
    if (token_ids.shape()[0] < 1)
        throw std::runtime_error("ModernGPT::forward_mtp: token_ids must be non-empty");
    auto x = tok_emb_.forward(token_ids);
    for (const auto& block : blocks_) x = block.forward(x);
    x = ln_f_.forward(x);

    std::vector<Variable> logits;
    logits.reserve(1 + mtp_heads_.size());
    // Head 0: weight-tied main LM head
    logits.push_back(matmul(x, transpose2d(tok_emb_.weight())));
    // Heads 1…k: additional MTP heads
    for (const auto& head : mtp_heads_)
        logits.push_back(head.forward(x));
    return logits;
}

std::vector<autograd::Variable*> ModernGPT::parameters() {
    std::vector<autograd::Variable*> p;
    p.push_back(&tok_emb_.weight());
    for (auto& block : blocks_)
        for (auto* v : block.parameters()) p.push_back(v);
    for (auto* v : ln_f_.parameters()) p.push_back(v);
    for (auto& head : mtp_heads_)
        for (auto* v : head.parameters()) p.push_back(v);
    return p;
}

int64_t     ModernGPT::vocab_size()  const noexcept { return tok_emb_.vocab_size(); }
int64_t     ModernGPT::embed_dim()   const noexcept { return tok_emb_.embed_dim(); }
std::size_t ModernGPT::num_layers()  const noexcept { return blocks_.size(); }
int64_t     ModernGPT::n_mtp_heads() const noexcept {
    return static_cast<int64_t>(mtp_heads_.size());
}

} // namespace sub0llm::nn
