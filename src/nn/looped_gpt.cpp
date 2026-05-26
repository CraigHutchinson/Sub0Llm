#include "sub0llm/nn/looped_gpt.hpp"

#include "sub0llm/autograd/ops.hpp"

#include <format>
#include <stdexcept>

namespace sub0llm::nn {

using namespace autograd;

LoopedGPT::LoopedGPT(int64_t     vocab_size,
                     int64_t     embed_dim,
                     std::size_t n_heads,
                     std::size_t n_kv_heads,
                     int64_t     n_loops,
                     int64_t     d_ff,
                     std::uint64_t seed)
    : tok_emb_(vocab_size, embed_dim, seed),
      block_(embed_dim, n_heads, n_kv_heads, d_ff, seed + 1),
      ln_f_(embed_dim),
      n_loops_(n_loops)
{
    if (vocab_size <= 0)
        throw std::runtime_error(
            std::format("LoopedGPT: vocab_size={} must be positive", vocab_size));
    if (embed_dim <= 0)
        throw std::runtime_error(
            std::format("LoopedGPT: embed_dim={} must be positive", embed_dim));
    if (n_loops <= 0)
        throw std::runtime_error(
            std::format("LoopedGPT: n_loops={} must be positive", n_loops));
}

autograd::Variable LoopedGPT::forward_impl(const Tensor& token_ids, int64_t k) const {
    if (token_ids.ndim() != 1)
        throw std::runtime_error("LoopedGPT::forward: token_ids must be 1D (T,)");
    if (token_ids.shape()[0] < 1)
        throw std::runtime_error("LoopedGPT::forward: token_ids must be non-empty");
    if (k < 1)
        throw std::runtime_error(
            std::format("LoopedGPT::forward_k: k={} must be >= 1", k));

    auto x = tok_emb_.forward(token_ids);
    for (int64_t i = 0; i < k; ++i)
        x = block_.forward(x);
    x = ln_f_.forward(x);
    return matmul(x, transpose2d(tok_emb_.weight()));
}

autograd::Variable LoopedGPT::forward(const Tensor& token_ids) const {
    return forward_impl(token_ids, n_loops_);
}

autograd::Variable LoopedGPT::forward_k(const Tensor& token_ids, int64_t k) const {
    return forward_impl(token_ids, k);
}

std::vector<autograd::Variable*> LoopedGPT::parameters() {
    std::vector<autograd::Variable*> p;
    p.push_back(&tok_emb_.weight());
    for (auto* v : block_.parameters()) p.push_back(v);
    for (auto* v : ln_f_.parameters())  p.push_back(v);
    return p;
}

int64_t LoopedGPT::vocab_size() const noexcept { return tok_emb_.vocab_size(); }
int64_t LoopedGPT::embed_dim()  const noexcept { return tok_emb_.embed_dim(); }
int64_t LoopedGPT::n_loops()    const noexcept { return n_loops_; }

} // namespace sub0llm::nn
