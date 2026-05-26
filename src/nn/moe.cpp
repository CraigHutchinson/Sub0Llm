#include "sub0llm/nn/moe.hpp"

#include "sub0llm/autograd/ops.hpp"
#include "sub0llm/core/tensor.hpp"

#include <algorithm>
#include <format>
#include <numeric>
#include <stdexcept>

namespace sub0llm::nn {

using namespace autograd;

// ── MoEFeedForward ────────────────────────────────────────────────────────────

MoEFeedForward::MoEFeedForward(int64_t D, int64_t n_experts, int64_t top_k,
                                int64_t d_ff, std::uint64_t seed)
    : D_(D),
      n_experts_(n_experts),
      top_k_(top_k),
      router_(D, n_experts, seed)
{
    if (D <= 0)
        throw std::runtime_error(
            std::format("MoEFeedForward: D={} must be positive", D));
    if (n_experts < 2)
        throw std::runtime_error(
            std::format("MoEFeedForward: n_experts={} must be >= 2", n_experts));
    if (top_k < 1 || top_k > n_experts)
        throw std::runtime_error(
            std::format("MoEFeedForward: top_k={} must be in [1, n_experts={}]",
                         top_k, n_experts));

    experts_.reserve(static_cast<std::size_t>(n_experts));
    for (int64_t e = 0; e < n_experts; ++e)
        experts_.emplace_back(D, d_ff, seed + 1 + static_cast<std::uint64_t>(e));
}

MoEFeedForward::ForwardResult MoEFeedForward::forward(const Variable& x) const {
    const auto& xd = x.data();
    if (xd.ndim() != 2)
        throw std::runtime_error("MoEFeedForward::forward: x must be 2D (T, D)");
    if (xd.shape(1) != D_)
        throw std::runtime_error(std::format(
            "MoEFeedForward::forward: x has D={} but expected D={}", xd.shape(1), D_));

    const int64_t T = xd.shape(0);
    const int64_t E = n_experts_;
    const int64_t k = top_k_;

    // ── Router ────────────────────────────────────────────────────────────────
    Variable router_logits = router_.forward(x);       // (T, E)
    Variable probs         = softmax(router_logits);   // (T, E)

    // ── Top-k routing decisions (non-differentiable) ──────────────────────────
    auto prob_raw = probs.data().data_as<float>();

    Tensor routing_mask({T, E}, DType::Float32);
    auto mask_sp = routing_mask.data_as<float>();
    std::fill(mask_sp.begin(), mask_sp.end(), 0.f);

    const auto T_sz = static_cast<std::size_t>(T);
    const auto E_sz = static_cast<std::size_t>(E);
    const auto k_sz = static_cast<std::size_t>(k);

    for (std::size_t t = 0; t < T_sz; ++t) {
        std::vector<std::size_t> order(E_sz);
        std::iota(order.begin(), order.end(), std::size_t{0});
        std::partial_sort(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(k_sz),
            order.end(),
            [&](std::size_t a, std::size_t b) {
                return prob_raw[t * E_sz + a] > prob_raw[t * E_sz + b];
            });
        for (std::size_t i = 0; i < k_sz; ++i)
            mask_sp[t * E_sz + order[i]] = 1.f;
    }

    // ── Sparse gates (differentiable through probs) ───────────────────────────
    Variable gates = mul(probs, Variable(routing_mask, false));  // (T, E)

    // ── Auxiliary load-balancing loss ─────────────────────────────────────────
    // L_aux = E * sum_e f_e * p_e,   where f_e = count_e/T,  p_e = mean(probs[:,e])
    // = (E / T) * sum_{t,e} routing_mask[t,e] * probs[t,e]
    // = (E / T) * sum(gates)
    Variable aux_loss = scale(sum(gates),
        static_cast<float>(E) / static_cast<float>(T));

    // ── Weighted expert outputs ────────────────────────────────────────────────
    // output = sum_e row_scale(expert_e(x), gate_e_col)
    // gate_e_col = transpose2d(narrow(transpose2d(gates), e, 1))  shape (T, 1)
    Variable gates_T = transpose2d(gates);  // (E, T)

    Variable output(zeros({T, D_}, DType::Float32), false);
    for (int64_t e = 0; e < E; ++e) {
        Variable expert_out  = experts_[static_cast<std::size_t>(e)].forward(x);
        Variable gate_row    = narrow(gates_T, e, 1);        // (1, T)
        Variable gate_col    = transpose2d(gate_row);        // (T, 1)
        output = add(output, row_scale(expert_out, gate_col));
    }

    return {std::move(output), std::move(aux_loss)};
}

std::vector<autograd::Variable*> MoEFeedForward::parameters() {
    std::vector<Variable*> p;
    for (auto* v : router_.parameters()) p.push_back(v);
    for (auto& exp : experts_)
        for (auto* v : exp.parameters()) p.push_back(v);
    return p;
}

// ── MoETransformerBlock ───────────────────────────────────────────────────────

MoETransformerBlock::MoETransformerBlock(int64_t D, std::size_t n_heads,
                                          std::size_t n_kv_heads,
                                          int64_t n_experts, int64_t top_k,
                                          int64_t d_ff, std::uint64_t seed)
    : norm1_(D),
      attn_(static_cast<std::size_t>(D), n_heads, n_kv_heads, 10000.f, seed),
      norm2_(D),
      moe_(D, n_experts, top_k, d_ff, seed + 1)
{}

MoETransformerBlock::ForwardResult MoETransformerBlock::forward(const Variable& x) const {
    // h = x + attn(norm1(x))
    auto h = add(x, attn_.forward(norm2_.forward(x)));
    // y, aux = moe(norm2(h))
    auto [moe_out, aux_loss] = moe_.forward(norm1_.forward(h));
    // output = h + moe_out
    return {add(h, moe_out), std::move(aux_loss)};
}

std::vector<autograd::Variable*> MoETransformerBlock::parameters() {
    std::vector<Variable*> p;
    for (auto* v : norm1_.parameters()) p.push_back(v);
    for (auto* v : attn_.parameters())  p.push_back(v);
    for (auto* v : norm2_.parameters()) p.push_back(v);
    for (auto* v : moe_.parameters())   p.push_back(v);
    return p;
}

// ── MoEGPT ───────────────────────────────────────────────────────────────────

MoEGPT::MoEGPT(int64_t vocab_size, int64_t embed_dim,
                std::size_t n_heads, std::size_t n_kv_heads,
                int64_t n_layers, int64_t n_experts, int64_t top_k,
                int64_t d_ff, std::uint64_t seed)
    : tok_emb_(vocab_size, embed_dim, seed),
      ln_f_(embed_dim),
      n_experts_(n_experts),
      top_k_(top_k)
{
    if (vocab_size <= 0)
        throw std::runtime_error(
            std::format("MoEGPT: vocab_size={} must be positive", vocab_size));
    if (embed_dim <= 0)
        throw std::runtime_error(
            std::format("MoEGPT: embed_dim={} must be positive", embed_dim));
    if (n_layers <= 0)
        throw std::runtime_error(
            std::format("MoEGPT: n_layers={} must be positive", n_layers));

    blocks_.reserve(static_cast<std::size_t>(n_layers));
    for (int64_t l = 0; l < n_layers; ++l)
        blocks_.emplace_back(embed_dim, n_heads, n_kv_heads, n_experts, top_k,
                              d_ff, seed + 1 + static_cast<std::uint64_t>(l) * 100);
}

MoEGPT::MoEForwardResult MoEGPT::forward_moe(const Tensor& token_ids) const {
    if (token_ids.ndim() != 1)
        throw std::runtime_error("MoEGPT::forward: token_ids must be 1D (T,)");
    if (token_ids.shape(0) < 1)
        throw std::runtime_error("MoEGPT::forward: token_ids must be non-empty");

    auto x = tok_emb_.forward(token_ids);

    // Accumulate aux losses across all blocks.
    // Start with a zero scalar variable so we can add into it.
    Variable aux_sum(zeros({1}), false);
    for (auto& block : blocks_) {
        auto [out, aux] = block.forward(x);
        x       = std::move(out);
        aux_sum = add(aux_sum, aux);
    }

    x = ln_f_.forward(x);
    Variable logits   = matmul(x, transpose2d(tok_emb_.weight()));
    Variable aux_mean = scale(aux_sum, 1.0f / static_cast<float>(blocks_.size()));

    return {std::move(logits), std::move(aux_mean)};
}

autograd::Variable MoEGPT::forward(const Tensor& token_ids) const {
    return forward_moe(token_ids).logits;
}

std::vector<autograd::Variable*> MoEGPT::parameters() {
    std::vector<Variable*> p;
    p.push_back(&tok_emb_.weight());
    for (auto& block : blocks_)
        for (auto* v : block.parameters()) p.push_back(v);
    for (auto* v : ln_f_.parameters()) p.push_back(v);
    return p;
}

int64_t     MoEGPT::vocab_size()  const noexcept { return tok_emb_.vocab_size(); }
int64_t     MoEGPT::embed_dim()   const noexcept { return tok_emb_.embed_dim(); }
std::size_t MoEGPT::num_layers()  const noexcept { return blocks_.size(); }
int64_t     MoEGPT::n_experts()   const noexcept { return n_experts_; }
int64_t     MoEGPT::top_k()       const noexcept { return top_k_; }

} // namespace sub0llm::nn
