#pragma once

// gist_denoiser.hpp (Ch32, Phase P2 step 2b) — the word Denoiser with GIST CONDITIONING.
//
// Topic-drift (M2): a flat denoiser reuses the entities it introduces no more than chance — it has
// no plan for the passage. P2 adds a "manager" signal: a single gist vector g summarising the
// passage's CONTENT words, broadcast to every position so the "worker" denoiser can condition its
// predictions on the topic (the feudal idea, DESIGN_REVIEW §5 / Review II §5).
//
// g is pooled from the VISIBLE content words only — input positions that are (a) not [MASK] and
// (b) content types (is_content). So g is a pure function of what the model can already see: it is
// LEAK-FREE (never reads a masked answer) and inference-compatible (at generation it pools the
// committed canvas). It is projected through a learnable W_g (init 0 ⇒ g contributes nothing at the
// start ⇒ the model is IDENTICAL to the flat Denoiser baseline — the no-regression init the
// BUILD_PLAN requires), then added to every position alongside the time conditioning.
//
// Kill-test (2b): does conditioning on g LOWER masked-token NLL (esp. on content words) vs the flat
// baseline? If g gives no improvement it is being ignored and the gist idea is dead (BUILD_PLAN §P2).
// forward signatures match Denoiser exactly so the generalised loss/eval templates accept it.

#include "sub0diff/nn/denoiser.hpp"        // BidirectionalBlock
#include "sub0diff/nn/time_embedding.hpp"  // time_embedding_rows

#include "sub0llm/autograd/ops.hpp"
#include "sub0llm/autograd/variable.hpp"
#include "sub0llm/core/ops.hpp"
#include "sub0llm/core/tensor.hpp"
#include "sub0llm/nn/embedding.hpp"
#include "sub0llm/nn/modern_gpt.hpp"       // RMSNorm

#include <algorithm>
#include <array>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

namespace sub0diff::nn {

class GistDenoiser {
public:
    // shuffle_gist: route each window's positions to a DIFFERENT window's gist (a fixed b→(b+1)%B
    // derangement). Same architecture + params, but the gist↔window correspondence is broken — the
    // signal is destroyed while capacity is held fixed. The 2b control: if the real gist beats the
    // shuffled one, the NLL win is the gist SIGNAL, not the extra W_g capacity.
    GistDenoiser(std::int64_t vocab_size, std::int64_t embed_dim, std::size_t n_heads,
                 std::size_t n_kv_heads, std::int64_t n_layers, std::int64_t d_ff,
                 std::span<const std::uint8_t> is_content, std::uint64_t seed = 42,
                 bool shuffle_gist = false)
        : real_vocab_(vocab_size), embed_dim_(embed_dim), shuffle_gist_(shuffle_gist),
          tok_emb_(vocab_size + 1, embed_dim, seed),
          gist_w_(sub0llm::zeros(sub0llm::Tensor::Shape{embed_dim, embed_dim}), /*requires_grad=*/true),
          is_content_(is_content.begin(), is_content.end()),
          ln_f_(embed_dim) {
        blocks_.reserve(static_cast<std::size_t>(n_layers));
        for (std::int64_t l = 0; l < n_layers; ++l)
            blocks_.emplace_back(embed_dim, n_heads, n_kv_heads, d_ff,
                                 seed + 100 + static_cast<std::uint64_t>(l) * 17);
    }

    [[nodiscard]] sub0llm::autograd::Variable forward(const sub0llm::Tensor& token_ids,
                                                      float noise_level) const {
        const std::int64_t T = token_ids.numel();
        std::array<float, 1> nl{noise_level};
        return forward(token_ids, std::span<const float>(nl), 1, T);
    }

    [[nodiscard]] sub0llm::autograd::Variable forward(const sub0llm::Tensor& token_ids,
                                                      std::span<const float> noise_levels,
                                                      std::int64_t B, std::int64_t T) const {
        namespace ag = sub0llm::autograd;
        const sub0llm::Device dev = tok_emb_.weight().data().device();
        ag::Variable emb = tok_emb_.forward(token_ids);           // (B·T, D)

        // time conditioning (per-window noise level, broadcast over positions)
        sub0llm::Tensor cond({B * T, embed_dim_}, sub0llm::DType::Float32);
        auto cs = cond.data_as<float>();
        for (std::int64_t b = 0; b < B; ++b) {
            sub0llm::Tensor rows = time_embedding_rows(noise_levels[static_cast<std::size_t>(b)],
                                                       T, embed_dim_);
            std::copy_n(rows.data_as<float>().begin(), T * embed_dim_, cs.begin() + b * T * embed_dim_);
        }
        ag::Variable x = ag::add(emb, ag::Variable{cond.to(dev)});

        // gist conditioning: pool VISIBLE content-word embeddings per window → g (B,D), project,
        // broadcast back to (B·T, D). pool_/bcast_ are host-built constants (no grad) on `dev`.
        auto [pool, bcast] = gist_selectors(token_ids, B, T);     // (B, B·T) and (B·T, B)
        ag::Variable g      = ag::matmul(ag::Variable{pool.to(dev), false}, emb);   // (B, D) mean-pool
        ag::Variable g_proj = ag::matmul(g, gist_w_);                               // (B, D) learned use
        ag::Variable g_bc   = ag::matmul(ag::Variable{bcast.to(dev), false}, g_proj);  // (B·T, D)
        x = ag::add(x, g_bc);

        for (const auto& blk : blocks_) x = blk.forward(x, B, T);
        x = ln_f_.forward(x);
        return ag::matmul_bt(x, tok_emb_.weight());               // (B·T, Vm), weight-tied
    }

    [[nodiscard]] std::vector<sub0llm::autograd::Variable*> parameters() {
        std::vector<sub0llm::autograd::Variable*> p{&tok_emb_.weight(), &gist_w_};
        for (auto& blk : blocks_) {
            auto bp = blk.parameters();
            p.insert(p.end(), bp.begin(), bp.end());
        }
        auto lp = ln_f_.parameters();
        p.insert(p.end(), lp.begin(), lp.end());
        return p;
    }

    void to(sub0llm::Device dev) { for (auto* p : parameters()) p->to(dev); }

    [[nodiscard]] std::int32_t mask_id()     const noexcept { return static_cast<std::int32_t>(real_vocab_); }
    [[nodiscard]] std::int64_t real_vocab()  const noexcept { return real_vocab_; }
    [[nodiscard]] std::int64_t model_vocab() const noexcept { return real_vocab_ + 1; }
    [[nodiscard]] std::int64_t embed_dim()   const noexcept { return embed_dim_; }
    // mean |gist_w| — inspect how far the gist path has activated from its zero init.
    [[nodiscard]] const sub0llm::autograd::Variable& gist_w() const noexcept { return gist_w_; }

private:
    // pool (B, B·T): row b is the mean selector over window b's VISIBLE content positions (1/count,
    // or 0 if none). bcast (B·T, B): one-hot of each position's window — broadcasts g back per row.
    [[nodiscard]] std::pair<sub0llm::Tensor, sub0llm::Tensor>
    gist_selectors(const sub0llm::Tensor& token_ids, std::int64_t B, std::int64_t T) const {
        const sub0llm::Tensor ids_h =
            token_ids.device().is_cpu() ? token_ids : token_ids.to(sub0llm::Device::cpu());
        const auto ids = ids_h.data_as<std::int32_t>();
        const std::int32_t mask = mask_id();

        sub0llm::Tensor pool({B, B * T}, sub0llm::DType::Float32);
        sub0llm::Tensor bcast({B * T, B}, sub0llm::DType::Float32);
        auto pd = pool.data_as<float>();
        auto bd = bcast.data_as<float>();
        std::fill(pd.begin(), pd.end(), 0.0f);
        std::fill(bd.begin(), bd.end(), 0.0f);

        for (std::int64_t b = 0; b < B; ++b) {
            std::int64_t cnt = 0;
            for (std::int64_t i = 0; i < T; ++i) {
                const std::int32_t id = ids[static_cast<std::size_t>(b * T + i)];
                if (id != mask && id >= 0 && static_cast<std::size_t>(id) < is_content_.size()
                    && is_content_[static_cast<std::size_t>(id)])
                    ++cnt;
                // position → its window's gist, or (control) a different window's gist.
                const std::int64_t src = shuffle_gist_ ? (b + 1) % B : b;
                bd[static_cast<std::size_t>((b * T + i) * B + src)] = 1.0f;
            }
            if (cnt == 0) continue;
            const float w = 1.0f / static_cast<float>(cnt);
            for (std::int64_t i = 0; i < T; ++i) {
                const std::int32_t id = ids[static_cast<std::size_t>(b * T + i)];
                if (id != mask && id >= 0 && static_cast<std::size_t>(id) < is_content_.size()
                    && is_content_[static_cast<std::size_t>(id)])
                    pd[static_cast<std::size_t>(b * (B * T) + b * T + i)] = w;
            }
        }
        return {std::move(pool), std::move(bcast)};
    }

    std::int64_t                    real_vocab_, embed_dim_;
    bool                            shuffle_gist_ = false;  // control: break gist↔window correspondence
    sub0llm::nn::Embedding          tok_emb_;
    sub0llm::autograd::Variable     gist_w_;        // (D, D) gist projection (init 0 = flat baseline)
    std::vector<std::uint8_t>       is_content_;    // content-type mask (by token id)
    std::vector<BidirectionalBlock> blocks_;
    sub0llm::nn::RMSNorm            ln_f_;
};

}  // namespace sub0diff::nn
