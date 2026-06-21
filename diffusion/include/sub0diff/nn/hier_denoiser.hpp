#pragma once

// hier_denoiser.hpp (Ch32, P2 step 2e) — the COARSE-TO-FINE hierarchical denoiser.
//
// The gist is a coarsening operator (DESIGN_REVIEW_3): instead of one O(N²) attention over the whole
// canvas, do a CHEAP coarse pass over N/c pooled slots (the global plan G), then a FINE pass over
// M = N/w sub-windows with window-local O(w²) attention conditioned on G. The fine windows are
// independent given G, so the fine attention is block-diagonal — exactly the denoiser's existing
// batched forward(x, B·M, w). Attention cost O((N/c)² + N·w) ≪ flat O(N²); the fine pass parallelises
// across the M windows (the batch dim). This is the single-level case of the P3 MERA stack.
//
//   emb ──+time──► x0 ──pool(c)──► coarse blocks (attn over N/c) ──► broadcast(c) ──► coarse_ctx
//                   └────────────────────────────── x = x0 + coarse_ctx ──► fine blocks (local w) ──► head
//
// The fine blocks RMSNorm their input, so the coarse signal needs no explicit scale gate (any
// reasonable amplitude is normalised before attention). Pool/broadcast are the reshape + batched-
// matmul trick from CharComposer::compose_vocab — O(N·D), no B² dense selector. forward signatures
// match Denoiser so the generalised loss/eval templates accept it.

#include "sub0diff/nn/denoiser.hpp"        // BidirectionalBlock
#include "sub0diff/nn/time_embedding.hpp"

#include "sub0llm/autograd/ops.hpp"
#include "sub0llm/autograd/variable.hpp"
#include "sub0llm/core/ops.hpp"
#include "sub0llm/core/tensor.hpp"
#include "sub0llm/nn/embedding.hpp"
#include "sub0llm/nn/modern_gpt.hpp"       // RMSNorm

#include <array>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

namespace sub0diff::nn {

class HierDenoiser {
public:
    // coarsen: c (tokens per coarse slot); window: w (fine window width). N must be a multiple of
    // both w and c (and w a multiple of c is not required). n_coarse_layers + n_fine_layers ≈ a flat
    // model's depth for a parameter-matched A/B.
    HierDenoiser(std::int64_t vocab_size, std::int64_t embed_dim, std::size_t n_heads,
                 std::size_t n_kv_heads, std::int64_t n_coarse_layers, std::int64_t n_fine_layers,
                 std::int64_t coarsen, std::int64_t window, std::int64_t d_ff = 0,
                 std::uint64_t seed = 42)
        : real_vocab_(vocab_size), embed_dim_(embed_dim), coarsen_(coarsen), window_(window),
          tok_emb_(vocab_size + 1, embed_dim, seed),
          ln_f_(embed_dim) {
        coarse_.reserve(static_cast<std::size_t>(n_coarse_layers));
        for (std::int64_t l = 0; l < n_coarse_layers; ++l)
            coarse_.emplace_back(embed_dim, n_heads, n_kv_heads, d_ff,
                                 seed + 200 + static_cast<std::uint64_t>(l) * 17);
        fine_.reserve(static_cast<std::size_t>(n_fine_layers));
        for (std::int64_t l = 0; l < n_fine_layers; ++l)
            fine_.emplace_back(embed_dim, n_heads, n_kv_heads, d_ff,
                               seed + 500 + static_cast<std::uint64_t>(l) * 17);
    }

    [[nodiscard]] sub0llm::autograd::Variable forward(const sub0llm::Tensor& token_ids,
                                                      float noise_level) const {
        const std::int64_t N = token_ids.numel();
        std::array<float, 1> nl{noise_level};
        return forward(token_ids, std::span<const float>(nl), 1, N);
    }

    // token_ids: (B·N,) — B sequences of length N. Each sequence is coarsened over its own N tokens
    // and fined over its own M = N/w sub-windows (block-diagonal), so the whole batch is one pair of
    // batched forwards.
    [[nodiscard]] sub0llm::autograd::Variable forward(const sub0llm::Tensor& token_ids,
                                                      std::span<const float> noise_levels,
                                                      std::int64_t B, std::int64_t N) const {
        namespace ag = sub0llm::autograd;
        if (N % coarsen_ != 0 || N % window_ != 0)
            throw std::runtime_error("HierDenoiser: N must be divisible by coarsen and window");
        const std::int64_t Nc = N / coarsen_;        // coarse slots per sequence
        const std::int64_t M  = N / window_;          // fine windows per sequence
        const sub0llm::Device dev = tok_emb_.weight().data().device();

        ag::Variable emb = tok_emb_.forward(token_ids);            // (B·N, D)
        // time conditioning (per sequence, broadcast over its N positions)
        sub0llm::Tensor cond({B * N, embed_dim_}, sub0llm::DType::Float32);
        auto cs = cond.data_as<float>();
        for (std::int64_t b = 0; b < B; ++b) {
            sub0llm::Tensor rows = time_embedding_rows(noise_levels[static_cast<std::size_t>(b)],
                                                       N, embed_dim_);
            std::copy_n(rows.data_as<float>().begin(), N * embed_dim_, cs.begin() + b * N * embed_dim_);
        }
        ag::Variable x0 = ag::add(emb, ag::Variable{cond.to(dev)});

        // ── coarse pass: mean-pool every `coarsen_` rows → (B·Nc, D), attend within each sequence ──
        ag::Variable g3 = ag::reshape(x0, {B * Nc, coarsen_, embed_dim_});         // group c rows
        ag::Variable psel(sub0llm::ops::mul(
            sub0llm::ones({B * Nc, 1, coarsen_}, sub0llm::DType::Float32, dev),
            1.0f / static_cast<float>(coarsen_)), /*requires_grad=*/false);
        ag::Variable coarse = ag::reshape(ag::matmul(psel, g3), {B * Nc, embed_dim_});  // (B·Nc, D)
        for (const auto& blk : coarse_) coarse = blk.forward(coarse, B, Nc);     // attn over Nc slots

        // ── broadcast each slot back to its `coarsen_` fine positions → (B·N, D) ──
        ag::Variable c3 = ag::reshape(coarse, {B * Nc, 1, embed_dim_});
        ag::Variable bsel(sub0llm::ones({B * Nc, coarsen_, 1}, sub0llm::DType::Float32, dev),
                          /*requires_grad=*/false);
        ag::Variable coarse_ctx = ag::reshape(ag::matmul(bsel, c3), {B * N, embed_dim_});

        // ── fine pass: x = x0 + coarse_ctx, then window-local (block-diagonal) attention ──
        ag::Variable x = ag::add(x0, coarse_ctx);
        for (const auto& blk : fine_) x = blk.forward(x, B * M, window_);   // M·B windows of w
        x = ln_f_.forward(x);
        return ag::matmul_bt(x, tok_emb_.weight());                 // (B·N, Vm)
    }

    [[nodiscard]] std::vector<sub0llm::autograd::Variable*> parameters() {
        std::vector<sub0llm::autograd::Variable*> p{&tok_emb_.weight()};
        for (auto& blk : coarse_) { auto bp = blk.parameters(); p.insert(p.end(), bp.begin(), bp.end()); }
        for (auto& blk : fine_)   { auto bp = blk.parameters(); p.insert(p.end(), bp.begin(), bp.end()); }
        auto lp = ln_f_.parameters();
        p.insert(p.end(), lp.begin(), lp.end());
        return p;
    }

    void to(sub0llm::Device dev) { for (auto* p : parameters()) p->to(dev); }

    [[nodiscard]] std::int32_t mask_id()     const noexcept { return static_cast<std::int32_t>(real_vocab_); }
    [[nodiscard]] std::int64_t real_vocab()  const noexcept { return real_vocab_; }
    [[nodiscard]] std::int64_t model_vocab() const noexcept { return real_vocab_ + 1; }
    [[nodiscard]] std::int64_t embed_dim()   const noexcept { return embed_dim_; }

private:
    std::int64_t                    real_vocab_, embed_dim_, coarsen_, window_;
    sub0llm::nn::Embedding          tok_emb_;
    std::vector<BidirectionalBlock> coarse_, fine_;
    sub0llm::nn::RMSNorm            ln_f_;
};

}  // namespace sub0diff::nn
