#pragma once

// mera_denoiser.hpp (Ch32, P3) — the RECURSIVE log-depth coarsening denoiser (MERA / U-Net).
//
// The single-level HierDenoiser (P2 2e) extends context only ~c× because its coarse pass is still
// O((N/c)²) — its own N² term, scaled by c² (2E_RESULTS §Context-ceiling). MERA removes that wall by
// coarsening RECURSIVELY until the top level is tiny, so NO level ever carries a large N²:
//
//   encode (up):  level k has length N/cᵏ. disentangle (windowed local attn, O(len·w)) then coarsen
//                 (mean-pool by c). Repeat while len > w; the top level (len ≤ w) gets ONE full attn.
//   decode (down): from the top, broadcast each level's context to the next finer level, add the
//                 encoder skip, refine (windowed local attn). Repeat down to length N.
//
// Total attention work Σ_k O((N/cᵏ)·w) = O(N·w·(1+1/c+1/c²+…)) = O(N·w) — LINEAR in N, no level large.
// That is the ceiling-removing property the context sweep motivated. Each level has its own blocks, so
// the level structure is baked in at construction from N (one model per N — fine for the benchmark and
// for fixed-length training). forward signatures match Denoiser so the loss/eval templates accept it.

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

class MeraDenoiser {
public:
    // c: coarsen factor per level; w: local window width; max_seq_len: the LONGEST sequence the model
    // will see — it sizes the block pyramid (max levels). forward() then accepts ANY valid N ≤
    // max_seq_len (the pyramid is rebuilt per call; blocks are indexed by depth-from-finest, so enc[k]
    // always processes level k and the single top block handles whatever level falls ≤ w). blocks_per_
    // level = 1 (one disentangle + one refine per level + one top); depth ≈ 2·levels + 1.
    MeraDenoiser(std::int64_t vocab_size, std::int64_t embed_dim, std::size_t n_heads,
                 std::size_t n_kv_heads, std::int64_t coarsen, std::int64_t window,
                 std::int64_t max_seq_len, std::int64_t d_ff = 0, std::uint64_t seed = 42)
        : real_vocab_(vocab_size), embed_dim_(embed_dim), coarsen_(coarsen), window_(window),
          tok_emb_(vocab_size + 1, embed_dim, seed), ln_f_(embed_dim) {
        max_levels_ = static_cast<std::int64_t>(compute_lens(max_seq_len).size()) - 1;  // coarsen steps at max N
        std::uint64_t s = seed + 1000;
        enc_.reserve(static_cast<std::size_t>(max_levels_));
        dec_.reserve(static_cast<std::size_t>(max_levels_));
        for (std::int64_t k = 0; k < max_levels_; ++k) enc_.emplace_back(embed_dim, n_heads, n_kv_heads, d_ff, s += 17);
        top_blk_.emplace_back(embed_dim, n_heads, n_kv_heads, d_ff, s += 17);
        for (std::int64_t k = 0; k < max_levels_; ++k) dec_.emplace_back(embed_dim, n_heads, n_kv_heads, d_ff, s += 17);
    }

    [[nodiscard]] sub0llm::autograd::Variable forward(const sub0llm::Tensor& token_ids,
                                                      float noise_level) const {
        std::array<float, 1> nl{noise_level};
        return forward(token_ids, std::span<const float>(nl), 1, token_ids.numel());
    }

    [[nodiscard]] sub0llm::autograd::Variable forward(const sub0llm::Tensor& token_ids,
                                                      std::span<const float> noise_levels,
                                                      std::int64_t B, std::int64_t N) const {
        namespace ag = sub0llm::autograd;
        const auto lens = compute_lens(N);                        // pyramid for THIS N (any valid N)
        const std::int64_t K = static_cast<std::int64_t>(lens.size()) - 1;
        if (K > max_levels_)
            throw std::runtime_error("MeraDenoiser: N exceeds the constructed max_seq_len pyramid depth");
        const sub0llm::Device dev = tok_emb_.weight().data().device();

        ag::Variable emb = tok_emb_.forward(token_ids);            // (B·N, D)
        sub0llm::Tensor cond({B * N, embed_dim_}, sub0llm::DType::Float32);
        auto cs = cond.data_as<float>();
        for (std::int64_t b = 0; b < B; ++b) {
            sub0llm::Tensor rows = time_embedding_rows(noise_levels[static_cast<std::size_t>(b)], N, embed_dim_);
            std::copy_n(rows.data_as<float>().begin(), N * embed_dim_, cs.begin() + b * N * embed_dim_);
        }
        ag::Variable cur = ag::add(emb, ag::Variable{cond.to(dev)});

        // encode: disentangle (windowed) + coarsen, saving a skip per level
        std::vector<ag::Variable> skips(static_cast<std::size_t>(K));
        for (std::int64_t k = 0; k < K; ++k) {
            cur = attn_level(enc_[static_cast<std::size_t>(k)], cur, B, lens[static_cast<std::size_t>(k)]);
            skips[static_cast<std::size_t>(k)] = cur;
            cur = pool(cur, B, lens[static_cast<std::size_t>(k)], dev);
        }
        // top: one full-attention pass at the smallest level (len ≤ w)
        cur = attn_level(top_blk_[0], cur, B, lens[static_cast<std::size_t>(K)]);
        // decode: broadcast up, add skip, refine (windowed)
        for (std::int64_t k = K - 1; k >= 0; --k) {
            cur = broadcast(cur, B, lens[static_cast<std::size_t>(k + 1)], dev);    // → len_k
            cur = ag::add(skips[static_cast<std::size_t>(k)], cur);
            cur = attn_level(dec_[static_cast<std::size_t>(k)], cur, B, lens[static_cast<std::size_t>(k)]);
        }
        cur = ln_f_.forward(cur);
        return ag::matmul_bt(cur, tok_emb_.weight());             // (B·N, Vm)
    }

    [[nodiscard]] std::vector<sub0llm::autograd::Variable*> parameters() {
        std::vector<sub0llm::autograd::Variable*> p{&tok_emb_.weight()};
        auto take = [&](std::vector<BidirectionalBlock>& v) {
            for (auto& b : v) { auto bp = b.parameters(); p.insert(p.end(), bp.begin(), bp.end()); }
        };
        take(enc_); take(top_blk_); take(dec_);
        auto lp = ln_f_.parameters();
        p.insert(p.end(), lp.begin(), lp.end());
        return p;
    }

    void to(sub0llm::Device dev) { for (auto* p : parameters()) p->to(dev); }

    [[nodiscard]] std::int32_t mask_id()     const noexcept { return static_cast<std::int32_t>(real_vocab_); }
    [[nodiscard]] std::int64_t real_vocab()  const noexcept { return real_vocab_; }
    [[nodiscard]] std::int64_t model_vocab() const noexcept { return real_vocab_ + 1; }
    [[nodiscard]] std::int64_t embed_dim()   const noexcept { return embed_dim_; }
    // levels at the constructed max_seq_len (the deepest pyramid this model supports).
    [[nodiscard]] std::int64_t n_levels()    const noexcept { return max_levels_ + 1; }
    // levels for a specific N (≤ max) — what forward(N) actually uses.
    [[nodiscard]] std::int64_t n_levels_for(std::int64_t N) const { return static_cast<std::int64_t>(compute_lens(N).size()); }

private:
    // The level-length pyramid for a sequence of length N: [N, N/c, N/c², …] down to the first ≤ w.
    [[nodiscard]] std::vector<std::int64_t> compute_lens(std::int64_t N) const {
        std::vector<std::int64_t> lens{N};
        while (lens.back() > window_) {
            if (lens.back() % coarsen_ != 0)
                throw std::runtime_error("MeraDenoiser: level length not divisible by coarsen");
            lens.push_back(lens.back() / coarsen_);
        }
        return lens;
    }

    // Run one block over a level: full attention if len ≤ w (one window per sequence), else windowed
    // block-diagonal attention (len/w windows of w per sequence).
    [[nodiscard]] sub0llm::autograd::Variable attn_level(const BidirectionalBlock& blk,
                                                         const sub0llm::autograd::Variable& x,
                                                         std::int64_t B, std::int64_t len) const {
        return len <= window_ ? blk.forward(x, B, len) : blk.forward(x, B * (len / window_), window_);
    }

    // mean-pool every `coarsen_` rows: (B·len, D) → (B·len/c, D), via reshape + batched matmul.
    [[nodiscard]] sub0llm::autograd::Variable pool(const sub0llm::autograd::Variable& x, std::int64_t B,
                                                   std::int64_t len, sub0llm::Device dev) const {
        namespace ag = sub0llm::autograd;
        const std::int64_t slots = B * len / coarsen_;
        ag::Variable g3 = ag::reshape(x, {slots, coarsen_, embed_dim_});
        ag::Variable sel(sub0llm::ops::mul(sub0llm::ones({slots, 1, coarsen_}, sub0llm::DType::Float32, dev),
                                           1.0f / static_cast<float>(coarsen_)), false);
        return ag::reshape(ag::matmul(sel, g3), {slots, embed_dim_});
    }

    // broadcast each coarse slot to its `coarsen_` finer positions: (B·lc, D) → (B·lc·c, D).
    [[nodiscard]] sub0llm::autograd::Variable broadcast(const sub0llm::autograd::Variable& x, std::int64_t B,
                                                        std::int64_t lc, sub0llm::Device dev) const {
        namespace ag = sub0llm::autograd;
        const std::int64_t slots = B * lc;
        ag::Variable c3 = ag::reshape(x, {slots, 1, embed_dim_});
        ag::Variable sel(sub0llm::ones({slots, coarsen_, 1}, sub0llm::DType::Float32, dev), false);
        return ag::reshape(ag::matmul(sel, c3), {slots * coarsen_, embed_dim_});
    }

    std::int64_t                    real_vocab_, embed_dim_, coarsen_, window_, max_levels_ = 0;
    sub0llm::nn::Embedding          tok_emb_;
    std::vector<BidirectionalBlock> enc_, top_blk_, dec_;
    sub0llm::nn::RMSNorm            ln_f_;
};

}  // namespace sub0diff::nn
