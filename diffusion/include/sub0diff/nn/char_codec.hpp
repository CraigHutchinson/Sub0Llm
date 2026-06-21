#pragma once

// char_codec.hpp (Ch32, Phase P1) — the CHARACTER-COMPOSITION CODEC: the OOV fix for word-level
// diffusion. A word-level model is fast at grammar but helpless on a word it has never seen (no
// embedding row). The codec folds level-C (sub-word spelling) into the word representation:
//
//   CharComposer:  word = (char-id sequence)  →  ONE D-dim word vector   (compose-in)
//   CharDecoder:   D-dim word vector           →  char logits per position (decode-out)
//
// Trained as an autoencoder (compose→decode→reconstruct the spelling), the pair learns a
// CONTENT-ADDRESSED word vector: any spelling — in-vocab OR out-of-vocab — maps to/from a vector
// by its characters, so an unseen word is no longer a hole. (DESIGN_REVIEW §6.)
//
// Reuse-first (BUILD_PLAN principle #3): the order-sensitive mixing is the SAME bidirectional
// transformer block the Denoiser uses (RoPE gives char positions); pooling and broadcast are done
// with constant ones-vectors through the existing autograd matmul, so the module adds NO new ops.

#include "sub0diff/nn/denoiser.hpp"   // BidirectionalBlock

#include "sub0llm/autograd/ops.hpp"
#include "sub0llm/autograd/variable.hpp"
#include "sub0llm/core/ops.hpp"       // ones
#include "sub0llm/core/tensor.hpp"
#include "sub0llm/nn/embedding.hpp"

#include <cstdint>
#include <vector>

namespace sub0diff::nn {

namespace detail {
// An (r×c) ones matrix on `dev`, wrapped as a non-trainable constant Variable. Used as a
// mean-pool selector (1×L) or a broadcast selector (L×1) through the existing autograd matmul.
[[nodiscard]] inline sub0llm::autograd::Variable ones_const(std::int64_t r, std::int64_t c,
                                                            sub0llm::Device dev) {
    return sub0llm::autograd::Variable(
        sub0llm::ones(sub0llm::Tensor::Shape{r, c}, sub0llm::DType::Float32, dev),
        /*requires_grad=*/false);
}
}  // namespace detail

// Word (L char-ids) → one D-dim vector. Chars are embedded, mixed by bidirectional attention
// (order-sensitive — "dog" and "god" diverge), then mean-pooled to a single vector.
class CharComposer {
public:
    CharComposer(std::int64_t n_chars, std::int64_t D, std::size_t n_heads, std::size_t n_kv_heads,
                 std::int64_t n_layers, std::int64_t d_ff = 0, std::uint64_t seed = 42)
        : char_emb_(n_chars, D, seed) {
        blocks_.reserve(static_cast<std::size_t>(n_layers));
        for (std::int64_t i = 0; i < n_layers; ++i)
            blocks_.emplace_back(D, n_heads, n_kv_heads, d_ff,
                                 seed + static_cast<std::uint64_t>(i) + 1);
    }

    // char_ids: (L,) int32 → (1, D).
    [[nodiscard]] sub0llm::autograd::Variable forward(const sub0llm::Tensor& char_ids) const {
        namespace ag = sub0llm::autograd;
        ag::Variable e = char_emb_.forward(char_ids);     // (L, D)
        for (const auto& b : blocks_) e = b.forward(e);   // (L, D), full bidirectional mixing
        const std::int64_t L = char_ids.shape(0);
        auto sel = detail::ones_const(1, L, e.data().device());        // (1, L)
        return ag::scale(ag::matmul(sel, e), 1.0f / static_cast<float>(L));  // (1, D) mean-pool
    }

    [[nodiscard]] std::vector<sub0llm::autograd::Variable*> parameters() {
        std::vector<sub0llm::autograd::Variable*> p{&char_emb_.weight()};
        for (auto& b : blocks_) {
            auto bp = b.parameters();
            p.insert(p.end(), bp.begin(), bp.end());
        }
        return p;
    }
    void to(sub0llm::Device dev) { for (auto* p : parameters()) p->to(dev); }

private:
    sub0llm::nn::Embedding          char_emb_;
    std::vector<BidirectionalBlock> blocks_;
};

// Word vector (1, D) → char logits (L, n_chars), non-autoregressive (the diffusion idiom). Each
// output position starts from a LEARNED positional query plus the (broadcast) word vector, so the
// decode is "fill in the spelling of THIS word at each slot" — position queries give the slots
// distinct inputs (RoPE alone, on identical rows, is too weak to spell). Bidirectional blocks mix.
class CharDecoder {
public:
    CharDecoder(std::int64_t n_chars, std::int64_t D, std::size_t n_heads, std::size_t n_kv_heads,
                std::int64_t n_layers, std::int64_t d_ff = 0, std::uint64_t seed = 1042,
                std::int64_t max_len = 32)
        : out_emb_(n_chars, D, seed),
          pos_(sub0llm::ops::mul(sub0llm::randn(sub0llm::Tensor::Shape{max_len, D}), 0.02f),
               /*requires_grad=*/true) {
        blocks_.reserve(static_cast<std::size_t>(n_layers));
        for (std::int64_t i = 0; i < n_layers; ++i)
            blocks_.emplace_back(D, n_heads, n_kv_heads, d_ff,
                                 seed + static_cast<std::uint64_t>(i) + 1);
    }

    // word_vec: (1, D); L = spelling length → (L, n_chars) logits (weight-tied to out_emb).
    [[nodiscard]] sub0llm::autograd::Variable forward(const sub0llm::autograd::Variable& word_vec,
                                                      std::int64_t L) const {
        namespace ag = sub0llm::autograd;
        auto sel = detail::ones_const(L, 1, word_vec.data().device());   // (L, 1)
        ag::Variable h = ag::matmul(sel, word_vec);                      // (L, D) broadcast word vec
        h = ag::add(h, ag::narrow(pos_, 0, L));                          // + learned position queries
        for (const auto& b : blocks_) h = b.forward(h);                  // (L, D) bidirectional
        return ag::matmul_bt(h, out_emb_.weight());                     // (L, n_chars)
    }

    [[nodiscard]] std::vector<sub0llm::autograd::Variable*> parameters() {
        std::vector<sub0llm::autograd::Variable*> p{&out_emb_.weight(), &pos_};
        for (auto& b : blocks_) {
            auto bp = b.parameters();
            p.insert(p.end(), bp.begin(), bp.end());
        }
        return p;
    }
    void to(sub0llm::Device dev) { for (auto* p : parameters()) p->to(dev); }

private:
    sub0llm::nn::Embedding          out_emb_;
    sub0llm::autograd::Variable     pos_;     // (max_len, D) learned positional queries
    std::vector<BidirectionalBlock> blocks_;
};

// Autoencoder reconstruction loss for one word: compose its chars, decode, score the spelling.
// Uniform weights → weighted_cross_entropy is plain mean CE, but stays on whatever device the
// model lives on (it is the CUDA-supported loss; plain cross_entropy is CPU-only).
[[nodiscard]] inline sub0llm::autograd::Variable
char_recon_loss(const CharComposer& comp, const CharDecoder& dec, const sub0llm::Tensor& char_ids) {
    namespace ag = sub0llm::autograd;
    const std::int64_t L      = char_ids.shape(0);
    ag::Variable       wv     = comp.forward(char_ids);   // (1, D)
    ag::Variable       logits = dec.forward(wv, L);       // (L, n_chars)
    const sub0llm::Device dev = logits.data().device();
    sub0llm::Tensor    w(sub0llm::Tensor::Shape{L}, sub0llm::DType::Float32, sub0llm::Device::cpu());
    for (auto& v : w.data_as<float>()) v = 1.0f;
    return ag::weighted_cross_entropy(logits, char_ids,
                                      dev.is_cpu() ? w : w.to(dev));
}

}  // namespace sub0diff::nn
