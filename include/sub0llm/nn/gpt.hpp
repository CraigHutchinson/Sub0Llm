#pragma once

#include "sub0llm/autograd/ops.hpp"
#include "sub0llm/autograd/variable.hpp"
#include "sub0llm/core/tensor.hpp"
#include "sub0llm/nn/attention.hpp"
#include "sub0llm/nn/embedding.hpp"
#include "sub0llm/nn/positional_encoding.hpp"

#include <cstdint>
#include <vector>

namespace sub0llm::nn {

// Linear projection: y = x @ W^T + b
//   x: (T, in_features) → (T, out_features)
//   W: (out_features, in_features), Xavier uniform init
//   b: (out_features,), zero init
class Linear {
public:
    Linear(int64_t in_features, int64_t out_features, std::uint64_t seed = 42);

    [[nodiscard]] autograd::Variable forward(const autograd::Variable& x) const;

    // Inference-only: x @ W^T + b using pure Tensor ops (no autograd graph).
    // x: (*, in_features) → (*, out_features).
    [[nodiscard]] Tensor apply_one(const Tensor& x) const;

    [[nodiscard]] std::vector<autograd::Variable*> parameters();

private:
    autograd::Variable W_;
    autograd::Variable b_;
};

// Layer normalisation with learnable affine parameters.
//   weight (gamma): (D,), initialised to 1
//   bias   (beta):  (D,), initialised to 0
class LayerNorm {
public:
    explicit LayerNorm(int64_t D, float eps = 1e-5f);

    [[nodiscard]] autograd::Variable forward(const autograd::Variable& x) const;
    [[nodiscard]] std::vector<autograd::Variable*> parameters();

private:
    autograd::Variable weight_;
    autograd::Variable bias_;
    float              eps_;
};

// Position-wise feed-forward network.
//   FFN(x) = gelu(x @ W1^T + b1) @ W2^T + b2
//   Expansion factor 4: W1 (4D, D), W2 (D, 4D)
class FeedForward {
public:
    FeedForward(int64_t D, std::uint64_t seed = 42);

    [[nodiscard]] autograd::Variable forward(const autograd::Variable& x) const;
    [[nodiscard]] std::vector<autograd::Variable*> parameters();

private:
    Linear fc1_;
    Linear fc2_;
};

// GPT-style transformer block with pre-norm.
//   x = x + attn(norm1(x))   (causal self-attention)
//   x = x + ffn(norm2(x))    (feed-forward network)
class TransformerBlock {
public:
    TransformerBlock(int64_t D, int64_t num_heads, std::uint64_t seed = 42);

    [[nodiscard]] autograd::Variable forward(const autograd::Variable& x) const;
    [[nodiscard]] std::vector<autograd::Variable*> parameters();

private:
    LayerNorm              norm1_;
    MultiHeadSelfAttention attn_;
    LayerNorm              norm2_;
    FeedForward            ffn_;
};

// Minimal GPT model: token embedding + positional encoding + N transformer
// blocks + final layer norm + weight-tied language model head.
//
//   forward(ids) : (T,) int32 → (T, vocab_size) logits
//   logits = ln_f(x) @ W_emb^T   (weight tying with tok_emb)
class GPT {
public:
    GPT(int64_t vocab_size, int64_t embed_dim, int64_t num_heads,
        int64_t num_layers, int64_t max_seq_len, std::uint64_t seed = 42);

    // Returns logits of shape (T, vocab_size).
    [[nodiscard]] autograd::Variable forward(const Tensor& token_ids) const;

    // All trainable parameters (tok_emb weight appears once; it is shared with
    // the LM head via weight tying).
    [[nodiscard]] std::vector<autograd::Variable*> parameters();

    [[nodiscard]] int64_t vocab_size()  const noexcept { return tok_emb_.vocab_size(); }
    [[nodiscard]] int64_t embed_dim()   const noexcept { return tok_emb_.embed_dim(); }
    [[nodiscard]] std::size_t num_layers() const noexcept { return blocks_.size(); }

private:
    Embedding              tok_emb_;
    LearnedPositionalEncoding pos_emb_;
    std::vector<TransformerBlock> blocks_;
    LayerNorm              ln_f_;
};

} // namespace sub0llm::nn
