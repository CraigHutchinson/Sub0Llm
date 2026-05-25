#include "sub0llm/nn/gpt.hpp"

#include <cmath>
#include <format>
#include <random>
#include <stdexcept>

namespace sub0llm::nn {

namespace {

autograd::Variable xavier_var(int64_t rows, int64_t cols,
                               std::mt19937_64& rng, std::string name) {
    const float a = std::sqrt(6.0f / static_cast<float>(rows + cols));
    std::uniform_real_distribution<float> dist(-a, a);
    Tensor w = zeros({rows, cols});
    for (auto& v : w.data_as<float>()) v = dist(rng);
    return autograd::Variable(std::move(w), true, std::move(name));
}

} // anonymous namespace

// ── Linear ────────────────────────────────────────────────────────────────────

Linear::Linear(int64_t in_features, int64_t out_features, std::uint64_t seed) {
    if (in_features <= 0 || out_features <= 0)
        throw std::runtime_error(std::format(
            "Linear: in_features={} and out_features={} must be positive",
            in_features, out_features));
    std::mt19937_64 rng(seed);
    W_ = xavier_var(out_features, in_features, rng, "linear.W");
    b_ = autograd::Variable(zeros({out_features}), true, "linear.b");
}

autograd::Variable Linear::forward(const autograd::Variable& x) const {
    using namespace autograd;
    return bias_add(matmul(x, transpose2d(W_)), b_);
}

std::vector<autograd::Variable*> Linear::parameters() {
    return {&W_, &b_};
}

// ── LayerNorm ─────────────────────────────────────────────────────────────────

LayerNorm::LayerNorm(int64_t D, float eps) : eps_(eps) {
    if (D <= 0)
        throw std::runtime_error(std::format(
            "LayerNorm: D={} must be positive", D));
    Tensor wt = zeros({D});
    for (auto& v : wt.data_as<float>()) v = 1.0f;
    weight_ = autograd::Variable(std::move(wt), true, "ln.weight");
    bias_   = autograd::Variable(zeros({D}), true, "ln.bias");
}

autograd::Variable LayerNorm::forward(const autograd::Variable& x) const {
    return autograd::layer_norm(x, weight_, bias_, eps_);
}

std::vector<autograd::Variable*> LayerNorm::parameters() {
    return {&weight_, &bias_};
}

// ── FeedForward ───────────────────────────────────────────────────────────────

FeedForward::FeedForward(int64_t D, std::uint64_t seed)
    : fc1_(D, 4 * D, seed), fc2_(4 * D, D, seed + 1) {}

autograd::Variable FeedForward::forward(const autograd::Variable& x) const {
    return fc2_.forward(autograd::gelu(fc1_.forward(x)));
}

std::vector<autograd::Variable*> FeedForward::parameters() {
    auto p = fc1_.parameters();
    for (auto* v : fc2_.parameters()) p.push_back(v);
    return p;
}

// ── TransformerBlock ──────────────────────────────────────────────────────────

TransformerBlock::TransformerBlock(int64_t D, int64_t num_heads,
                                   std::uint64_t seed)
    : norm1_(D),
      attn_(static_cast<std::size_t>(D), static_cast<std::size_t>(num_heads),
            seed),
      norm2_(D),
      ffn_(D, seed + 1000) {}

autograd::Variable TransformerBlock::forward(const autograd::Variable& x) const {
    using namespace autograd;
    auto h = add(x, attn_.forward(norm1_.forward(x), /*causal=*/true));
    return add(h, ffn_.forward(norm2_.forward(h)));
}

std::vector<autograd::Variable*> TransformerBlock::parameters() {
    std::vector<autograd::Variable*> p;
    for (auto* v : norm1_.parameters()) p.push_back(v);
    for (auto* v : attn_.parameters())  p.push_back(v);
    for (auto* v : norm2_.parameters()) p.push_back(v);
    for (auto* v : ffn_.parameters())   p.push_back(v);
    return p;
}

// ── GPT ───────────────────────────────────────────────────────────────────────

GPT::GPT(int64_t vocab_size, int64_t embed_dim, int64_t num_heads,
         int64_t num_layers, int64_t max_seq_len, std::uint64_t seed)
    : tok_emb_(vocab_size, embed_dim, seed),
      pos_emb_(max_seq_len, embed_dim, seed + 1),
      ln_f_(embed_dim) {
    if (num_heads <= 0)
        throw std::runtime_error(std::format(
            "GPT: num_heads={} must be positive", num_heads));
    if (num_layers <= 0)
        throw std::runtime_error(std::format(
            "GPT: num_layers={} must be positive", num_layers));
    blocks_.reserve(static_cast<std::size_t>(num_layers));
    for (int64_t l = 0; l < num_layers; ++l)
        blocks_.emplace_back(embed_dim, num_heads,
                             seed + 2 + static_cast<std::uint64_t>(l) * 2000);
}

autograd::Variable GPT::forward(const Tensor& token_ids) const {
    using namespace autograd;
    if (token_ids.ndim() != 1)
        throw std::runtime_error(
            "GPT::forward: token_ids must be 1D (T,)");
    const int64_t T = token_ids.shape()[0];
    auto x = add(tok_emb_.forward(token_ids), pos_emb_.forward(T));
    for (const auto& block : blocks_)
        x = block.forward(x);
    x = ln_f_.forward(x);
    return matmul(x, transpose2d(tok_emb_.weight()));
}

std::vector<autograd::Variable*> GPT::parameters() {
    std::vector<autograd::Variable*> p;
    p.push_back(&tok_emb_.weight());
    p.push_back(&pos_emb_.weight());
    for (auto& block : blocks_)
        for (auto* v : block.parameters()) p.push_back(v);
    for (auto* v : ln_f_.parameters()) p.push_back(v);
    return p;
}

} // namespace sub0llm::nn
