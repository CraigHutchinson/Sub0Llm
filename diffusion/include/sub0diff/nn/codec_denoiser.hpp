#pragma once

// codec_denoiser.hpp (Ch32, P1 step 1c) — the word Denoiser with a CHARACTER-COMPOSED embedding.
//
// Identical to nn::Denoiser except the (model_vocab, D) embedding table E — used for BOTH the input
// embedding and the weight-tied LM head — is a blend of a per-word LOOKUP row and the word's CHAR-
// COMPOSED vector (CharComposer over its spelling):
//
//     E = lookup  +  alpha ⊙ compose_vocab(spellings)          (alpha per-word, init 0)
//
// alpha starts at 0, so the model is IDENTICAL to plain word-level at init (the no-regression
// baseline the BUILD_PLAN 1c requires). As training proceeds, the composer's gradient comes from
// EVERY word (it is shared), so it learns well even for words that are individually rare — and a
// rare word's E row gains a meaningful, content-addressed component. That is the mechanism by which
// the M1 OOV cliff (rare types predicted ~8× worse, M1_RESULTS.md) should shrink.
//
// The forward signatures match Denoiser exactly so the model can drop into a generalised loss/eval
// once those are templated on the model type (the remaining 1c wiring).

#include "sub0diff/nn/char_codec.hpp"      // CharComposer
#include "sub0diff/nn/denoiser.hpp"        // BidirectionalBlock
#include "sub0diff/nn/time_embedding.hpp"  // time_embedding_rows

#include "sub0llm/autograd/embedding_ops.hpp"  // embedding_lookup on an arbitrary Variable weight
#include "sub0llm/autograd/ops.hpp"
#include "sub0llm/autograd/variable.hpp"
#include "sub0llm/core/ops.hpp"
#include "sub0llm/core/tensor.hpp"
#include "sub0llm/nn/embedding.hpp"
#include "sub0llm/nn/modern_gpt.hpp"       // RMSNorm

#include <cstdint>
#include <span>
#include <vector>

namespace sub0diff::nn {

class CodecDenoiser {
public:
    // word_chars: (model_vocab · max_len,) int32 — each word id's spelling, padded to max_len.
    // Blend of the per-word lookup row and the char-composed vector:
    //   Additive  (default): E = lookup + alpha ⊙ composed   (alpha learned, init 0)
    //   ConvexFixed        : E = g ⊙ lookup + (1−g) ⊙ composed, g FIXED per word — 1 for common
    //     words (pure lookup, no regression), 0 for rare words (pure composed = its spelling vector).
    // The convex-fixed mode is the 1C_RESULTS.md follow-up (2): it DROPS the noisy lookup for rare
    // words instead of adding composed on top, isolating whether composed_rare beats lookup_rare. It
    // only helps if composed_rare is meaningful — pretrain the composer first (pretrain_composer).
    enum class Blend { Additive, ConvexFixed };

    CodecDenoiser(std::int64_t vocab_size, std::int64_t embed_dim, std::size_t n_heads,
                  std::size_t n_kv_heads, std::int64_t n_layers, std::int64_t d_ff,
                  std::int64_t n_chars, std::int64_t max_len, sub0llm::Tensor word_chars,
                  std::uint64_t seed = 42, Blend blend = Blend::Additive,
                  std::span<const std::uint8_t> is_rare = {})
        : real_vocab_(vocab_size), embed_dim_(embed_dim), max_len_(max_len), blend_(blend),
          tok_emb_(vocab_size + 1, embed_dim, seed),
          composer_(n_chars, embed_dim, n_heads, n_kv_heads, /*codec depth=*/2, 0, seed + 7000),
          word_chars_(std::move(word_chars)),
          alpha_(sub0llm::zeros(sub0llm::Tensor::Shape{vocab_size + 1, embed_dim}), /*requires_grad=*/true),
          ln_f_(embed_dim) {
        blocks_.reserve(static_cast<std::size_t>(n_layers));
        for (std::int64_t l = 0; l < n_layers; ++l)
            blocks_.emplace_back(embed_dim, n_heads, n_kv_heads, d_ff,
                                 seed + 100 + static_cast<std::uint64_t>(l) * 17);
        if (blend_ == Blend::ConvexFixed) {
            // lk_ = g (1 for common, 0 for rare); ck_ = 1−g. Non-trainable; word ids beyond is_rare
            // (the mask id at index vocab_size) default to common (pure lookup).
            const std::int64_t Vm = vocab_size + 1;
            lk_ = sub0llm::zeros(sub0llm::Tensor::Shape{Vm, embed_dim});
            ck_ = sub0llm::zeros(sub0llm::Tensor::Shape{Vm, embed_dim});
            auto lk = lk_.data_as<float>();
            auto ck = ck_.data_as<float>();
            for (std::int64_t id = 0; id < Vm; ++id) {
                const bool rare = static_cast<std::size_t>(id) < is_rare.size() && is_rare[static_cast<std::size_t>(id)];
                const float g = rare ? 0.0f : 1.0f;
                for (std::int64_t d = 0; d < embed_dim; ++d) {
                    lk[static_cast<std::size_t>(id * embed_dim + d)] = g;
                    ck[static_cast<std::size_t>(id * embed_dim + d)] = 1.0f - g;
                }
            }
        }
    }

    [[nodiscard]] sub0llm::autograd::Variable forward(const sub0llm::Tensor& token_ids,
                                                      float noise_level) const {
        namespace ag = sub0llm::autograd;
        const std::int64_t T = token_ids.numel();
        ag::Variable E = effective_embedding();                 // (Vm, D)
        ag::Variable x = ag::embedding_lookup(E, token_ids);    // (T, D)
        ag::Variable t_emb{time_embedding_rows(noise_level, T, embed_dim_).to(x.data().device())};
        x = ag::add(x, t_emb);
        for (const auto& blk : blocks_) x = blk.forward(x);
        x = ln_f_.forward(x);
        return ag::matmul_bt(x, E);                             // (T, Vm), weight-tied to E
    }

    [[nodiscard]] sub0llm::autograd::Variable forward(const sub0llm::Tensor& token_ids,
                                                      std::span<const float> noise_levels,
                                                      std::int64_t B, std::int64_t T) const {
        namespace ag = sub0llm::autograd;
        ag::Variable E = effective_embedding();                 // (Vm, D) — computed once, used twice
        ag::Variable x = ag::embedding_lookup(E, token_ids);    // (B·T, D)
        sub0llm::Tensor cond({B * T, embed_dim_}, sub0llm::DType::Float32);
        auto cs = cond.data_as<float>();
        for (std::int64_t b = 0; b < B; ++b) {
            sub0llm::Tensor rows = time_embedding_rows(noise_levels[static_cast<std::size_t>(b)],
                                                       T, embed_dim_);
            auto rs = rows.data_as<float>();
            std::copy_n(rs.begin(), T * embed_dim_, cs.begin() + b * T * embed_dim_);
        }
        x = ag::add(x, ag::Variable{cond.to(x.data().device())});
        for (const auto& blk : blocks_) x = blk.forward(x, B, T);
        x = ln_f_.forward(x);
        return ag::matmul_bt(x, E);                             // (B·T, Vm)
    }

    [[nodiscard]] std::vector<sub0llm::autograd::Variable*> parameters() {
        std::vector<sub0llm::autograd::Variable*> p{&tok_emb_.weight(), &alpha_};
        auto cp = composer_.parameters();
        p.insert(p.end(), cp.begin(), cp.end());
        for (auto& blk : blocks_) {
            auto bp = blk.parameters();
            p.insert(p.end(), bp.begin(), bp.end());
        }
        auto lp = ln_f_.parameters();
        p.insert(p.end(), lp.begin(), lp.end());
        return p;
    }

    void to(sub0llm::Device dev) {
        for (auto* p : parameters()) p->to(dev);
        if (blend_ == Blend::ConvexFixed) { lk_ = lk_.to(dev); ck_ = ck_.to(dev); }  // constants
    }

    // Access the composer to pretrain it as a char autoencoder before LM training (follow-up 3).
    [[nodiscard]] CharComposer& composer() noexcept { return composer_; }

    [[nodiscard]] std::int32_t mask_id()     const noexcept { return static_cast<std::int32_t>(real_vocab_); }
    [[nodiscard]] std::int64_t real_vocab()  const noexcept { return real_vocab_; }
    [[nodiscard]] std::int64_t model_vocab() const noexcept { return real_vocab_ + 1; }
    [[nodiscard]] std::int64_t embed_dim()   const noexcept { return embed_dim_; }
    // Per-word lookup↔composed gate (init 0 = pure lookup). Inspect to see how far the composed
    // path has activated (mean |alpha|), and whether it concentrates on rare words.
    [[nodiscard]] const sub0llm::autograd::Variable& alpha() const noexcept { return alpha_; }

private:
    // E = lookup + alpha ⊙ compose_vocab(spellings). compose_vocab is the one expensive call; it is
    // evaluated once per forward and shared by the input embedding and the LM head.
    [[nodiscard]] sub0llm::autograd::Variable effective_embedding() const {
        namespace ag = sub0llm::autograd;
        ag::Variable composed = composer_.compose_vocab(word_chars_, model_vocab(), max_len_);  // (Vm,D)
        if (blend_ == Blend::ConvexFixed) {
            // E = g ⊙ lookup + (1−g) ⊙ composed. g=1 (common) ⇒ pure lookup; g=0 (rare) ⇒ pure
            // composed spelling vector. lk_/ck_ are non-trainable constants on the model's device.
            ag::Variable g{lk_, /*requires_grad=*/false};
            ag::Variable gc{ck_, /*requires_grad=*/false};
            return ag::add(ag::mul(tok_emb_.weight(), g), ag::mul(composed, gc));
        }
        // Additive: E = lookup + alpha ⊙ composed (per-element gate; mul+add are both on CUDA —
        // row_scale is not). alpha inits 0 ⇒ E = lookup at the start (no-regression baseline).
        return ag::add(tok_emb_.weight(), ag::mul(composed, alpha_));
    }

    std::int64_t                    real_vocab_, embed_dim_, max_len_;
    Blend                           blend_;
    sub0llm::nn::Embedding          tok_emb_;       // per-word lookup (Vm × D)
    CharComposer                    composer_;      // spelling → word vector
    sub0llm::Tensor                 word_chars_;    // (Vm·max_len,) int32 — fixed spellings
    sub0llm::autograd::Variable     alpha_;         // (Vm, D) per-element lookup↔composed gate (init 0)
    sub0llm::Tensor                 lk_, ck_;       // ConvexFixed: (Vm,D) g and 1−g constants
    std::vector<BidirectionalBlock> blocks_;
    sub0llm::nn::RMSNorm            ln_f_;
};

}  // namespace sub0diff::nn
