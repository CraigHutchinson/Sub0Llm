#include "sub0llm/nn/attention.hpp"

#include "sub0llm/autograd/ops.hpp"

#include <cmath>
#include <format>
#include <limits>
#include <random>
#include <stdexcept>

namespace sub0llm::nn {

namespace {

// Glorot/Xavier uniform: U[-a, a], a = sqrt(6 / (fan_in + fan_out)).
autograd::Variable xavier_var(int64_t fan_in, int64_t fan_out,
                               std::mt19937_64& rng, std::string name) {
    const float a = std::sqrt(6.0f / static_cast<float>(fan_in + fan_out));
    std::uniform_real_distribution<float> dist(-a, a);
    Tensor w = zeros({fan_in, fan_out});
    for (auto& v : w.data_as<float>()) v = dist(rng);
    return autograd::Variable(std::move(w), true, std::move(name));
}

// Lower-triangular mask: 0 on and below diagonal, -inf above.
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

// ── MultiHeadSelfAttention ────────────────────────────────────────────────────

MultiHeadSelfAttention::MultiHeadSelfAttention(std::size_t embed_dim,
                                               std::size_t num_heads,
                                               std::uint64_t seed) {
    if (embed_dim == 0 || num_heads == 0)
        throw std::runtime_error(std::format(
            "MultiHeadSelfAttention: embed_dim={} and num_heads={} must be positive",
            embed_dim, num_heads));
    if (embed_dim % num_heads != 0)
        throw std::runtime_error(std::format(
            "MultiHeadSelfAttention: embed_dim={} must be divisible by num_heads={}",
            embed_dim, num_heads));

    embed_dim_ = embed_dim;
    num_heads_ = num_heads;
    head_dim_  = embed_dim / num_heads;

    const int64_t D  = static_cast<int64_t>(embed_dim_);
    const int64_t Dh = static_cast<int64_t>(head_dim_);

    std::mt19937_64 rng(seed);
    W_Q_.reserve(num_heads_);
    W_K_.reserve(num_heads_);
    W_V_.reserve(num_heads_);
    W_O_.reserve(num_heads_);

    for (std::size_t h = 0; h < num_heads_; ++h) {
        W_Q_.push_back(xavier_var(D, Dh, rng, std::format("attn.W_Q.{}", h)));
        W_K_.push_back(xavier_var(D, Dh, rng, std::format("attn.W_K.{}", h)));
        W_V_.push_back(xavier_var(D, Dh, rng, std::format("attn.W_V.{}", h)));
        W_O_.push_back(xavier_var(Dh, D, rng, std::format("attn.W_O.{}", h)));
    }
}

autograd::Variable MultiHeadSelfAttention::forward(const autograd::Variable& x,
                                                    bool causal) const {
    using namespace autograd;
    if (x.data().ndim() != 2)
        throw std::runtime_error(
            "MultiHeadSelfAttention::forward: x must be 2D (T, embed_dim)");
    const auto D_in = static_cast<std::size_t>(x.data().shape()[1]);
    if (D_in != embed_dim_)
        throw std::runtime_error(std::format(
            "MultiHeadSelfAttention::forward: input dim {} != embed_dim {}",
            D_in, embed_dim_));

    const int64_t T = x.data().shape()[0];
    if (T < 1)
        throw std::runtime_error(std::format(
            "MultiHeadSelfAttention::forward: sequence length T={} must be >= 1", T));
    const float scale_fac = 1.0f / std::sqrt(static_cast<float>(head_dim_));

    // One head → Variable (T, embed_dim).
    auto compute_head = [&](std::size_t h) -> Variable {
        auto Q = matmul(x, W_Q_[h]);                          // (T, Dh)
        auto K = matmul(x, W_K_[h]);                          // (T, Dh)
        auto V = matmul(x, W_V_[h]);                          // (T, Dh)
        auto scores = scale(matmul(Q, transpose2d(K)), scale_fac);  // (T, T)
        if (causal) {
            const Tensor m = causal_mask(T);
            scores = add(scores, Variable(m, false));
        }
        auto attn    = softmax(scores);                        // (T, T)
        auto ctx     = matmul(attn, V);                        // (T, Dh)
        return matmul(ctx, W_O_[h]);                           // (T, D)
    };

    Variable out = compute_head(0);
    for (std::size_t h = 1; h < num_heads_; ++h)
        out = add(out, compute_head(h));

    return out;
}

std::vector<autograd::Variable*> MultiHeadSelfAttention::parameters() {
    std::vector<autograd::Variable*> params;
    params.reserve(num_heads_ * 4);
    for (std::size_t h = 0; h < num_heads_; ++h) {
        params.push_back(&W_Q_[h]);
        params.push_back(&W_K_[h]);
        params.push_back(&W_V_[h]);
        params.push_back(&W_O_[h]);
    }
    return params;
}

} // namespace sub0llm::nn
