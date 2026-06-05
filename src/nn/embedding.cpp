#include "sub0llm/nn/embedding.hpp"

#include "sub0llm/autograd/embedding_ops.hpp"

#include <format>
#include <random>
#include <stdexcept>

namespace sub0llm::nn {

Embedding::Embedding(std::int64_t vocab_size, std::int64_t embed_dim,
                     std::uint64_t seed, bool alloc_weights) {
    if (vocab_size <= 0 || embed_dim <= 0)
        throw std::runtime_error(std::format(
            "Embedding: vocab_size={} and embed_dim={} must both be positive",
            vocab_size, embed_dim));
    vocab_size_ = vocab_size;
    embed_dim_  = embed_dim;

    if (!alloc_weights) {   // Q8 inference: table elided, served from Q8 elsewhere
        weight_ = autograd::Variable(zeros({1, 1}), /*requires_grad=*/false, "emb.weight.elided");
        return;
    }

    // Initialise with N(0, 1/sqrt(embed_dim)) — same as PyTorch default.
    const float std_dev = 1.0f / std::sqrt(static_cast<float>(embed_dim));
    std::mt19937_64                    rng(seed);
    std::normal_distribution<float>    dist(0.0f, std_dev);

    Tensor w = zeros({vocab_size, embed_dim});
    auto   s = w.data_as<float>();
    for (auto& v : s) v = dist(rng);

    weight_ = autograd::Variable(std::move(w), /*requires_grad=*/true, "emb.weight");
}

autograd::Variable Embedding::forward(const Tensor& indices) const {
    return autograd::embedding_lookup(weight_, indices);
}

std::int64_t Embedding::vocab_size() const noexcept { return vocab_size_; }
std::int64_t Embedding::embed_dim()  const noexcept { return embed_dim_; }

} // namespace sub0llm::nn
