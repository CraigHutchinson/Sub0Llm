#pragma once

#include "sub0llm/autograd/variable.hpp"
#include "sub0llm/nn/modern_gpt.hpp"
#include "sub0llm/nn/numeric_router.hpp"
#include "sub0llm/tokenizer/numeric_tokenizer.hpp"

#include <cstdint>
#include <vector>

namespace sub0llm::nn {

struct MathResult {
    float value;
    bool  is_nan;
    bool  is_overflow;
};

// How numeric token IDs are presented to the transformer embedding layer.
// The reg side-channel always uses original token IDs; only the embedding input changes.
enum class TokenMode {
    Real,              // unchanged — transformer sees actual numeric token IDs
    Anon,              // all numeric → single NUM placeholder; cannot distinguish any values
    Algebraic,         // distinct values per expression → X0, X1, … (same value = same Xn)
    AlgebraicSpecial,  // like Algebraic but {-1, 0, 1} stay visible — structural constants
                       // (loop terminators, boolean outputs, identity elements) remain distinct
};

struct RouteInfo {
    std::vector<RouteType> routes;   // per-token hard route decision
    std::vector<float>     entropy;  // per-token softmax entropy (nats, max=ln(8)≈2.08)
};

// Apply exact arithmetic for the given RouteType.
// FFN / default returns {0, true, false} (NaN placeholder).
[[nodiscard]] MathResult apply_math_op(RouteType op, float a, float b);

// ── MathLayer ─────────────────────────────────────────────────────────────────
//
// Replaces the norm2 + SwiGLU FFN in a transformer block at depth l_math.
// Uses a NumericRouter to blend exact arithmetic embeddings with the FFN output.
// Routing uses STE: forward uses argmax hard mask, backward flows through softmax.
class MathLayer {
public:
    // D: embed dim; d_ff: FFN hidden dim (0 = auto-compute).
    MathLayer(int64_t D, int64_t d_ff = 0, std::uint64_t seed = 42);

    // h:      (T, D) post-attention residual stream
    // reg[t]: float value if t is a numeric token, NaN otherwise
    // Returns (T, D) — FFN-gated residual (no math injection; use forward_with_boost
    //                  for the full logit-space math path).
    [[nodiscard]] autograd::Variable forward(
        const autograd::Variable& h,
        const std::vector<float>& reg) const;

    // Like forward(), but separates the FFN residual from the math-result boost.
    // Returns {ffn_residual (T,D), math_logit_boost (T,V)}.
    //   ffn_residual      — add to h for the block residual connection (FFN branch only)
    //   math_logit_boost  — sparse (T, vocab_size) direct logit boost; caller adds this
    //                       to logits AFTER the final matmul.  Prediction is thereby
    //                       independent of result-token embedding norm — a one-hot at
    //                       position encode_int(result) is used instead of emb[result].
    [[nodiscard]] std::pair<autograd::Variable, autograd::Variable>
    forward_with_boost(
        const autograd::Variable& h,
        const std::vector<float>& reg,
        const NumericTokenizer&   ntok,
        float                     kMathBoost,
        int64_t                   vocab_size) const;

    [[nodiscard]] RouteInfo route_info(const autograd::Variable& h) const;
    // Pre-softmax router logits (T, n_types) — feed into cross_entropy for supervision.
    [[nodiscard]] autograd::Variable router_logits(const autograd::Variable& h) const;

    [[nodiscard]] std::vector<autograd::Variable*> parameters();

private:
    int64_t            D_;
    RMSNorm            norm_;
    NumericRouter      router_;
    SwiGLUFeedForward  ffn_;
};

// ── MathTransformerBlock ──────────────────────────────────────────────────────
//
// Same as ModernTransformerBlock but with MathLayer instead of SwiGLU FFN.
class MathTransformerBlock {
public:
    MathTransformerBlock(int64_t D, std::size_t n_heads, std::size_t n_kv_heads,
                         int64_t d_ff = 0, std::uint64_t seed = 42);

    // Returns {hidden_out (T,D), math_logit_boost (T,V)} — see MathLayer::forward_with_boost.
    [[nodiscard]] std::pair<autograd::Variable, autograd::Variable>
    forward_math_with_boost(
        const autograd::Variable& x,
        const std::vector<float>& reg,
        const NumericTokenizer&   ntok,
        float                     kMathBoost,
        int64_t                   vocab_size) const;

    [[nodiscard]] RouteInfo route_info(const autograd::Variable& x) const;
    [[nodiscard]] autograd::Variable router_logits(const autograd::Variable& x) const;

    [[nodiscard]] std::vector<autograd::Variable*> parameters();

private:
    RMSNorm               norm1_;
    GroupedQueryAttention  attn_;
    MathLayer              math_ffn_;
};

// ── MathGPT ───────────────────────────────────────────────────────────────────
//
// ModernGPT-style architecture with a MathTransformerBlock at depth l_math_.
//   forward():      uses all ModernTransformerBlocks (math-unaware)
//   forward_math(): substitutes MathTransformerBlock at l_math_
//
// l_math = -1 → default round(0.7 * n_layers)
class MathGPT {
public:
    MathGPT(int64_t       total_vocab,
            int64_t       embed_dim,
            std::size_t   n_heads,
            std::size_t   n_kv_heads,
            int64_t       n_layers,
            int64_t       l_math    = -1,
            int64_t       d_ff      = 0,
            std::uint64_t seed      = 42,
            float         math_boost = 15.0f);

    // Returns logits (T, total_vocab) using all ModernTransformerBlocks.
    [[nodiscard]] autograd::Variable forward(const Tensor& token_ids) const;

    // Returns logits (T, total_vocab) using MathTransformerBlock at l_math_.
    // The transformer embedding sees remap_tokens(token_ids, mode); reg is built
    // from the original token_ids so math neurons always read exact float values.
    [[nodiscard]] autograd::Variable forward_math(
        const Tensor& token_ids, const NumericTokenizer& ntok,
        TokenMode mode = TokenMode::Real) const;

    [[nodiscard]] RouteInfo route_info(
        const Tensor& token_ids, const NumericTokenizer& ntok,
        TokenMode mode = TokenMode::Real) const;

    // Pre-softmax router logits (T, n_types) at l_math_ — for supervision loss.
    // Must use the same TokenMode and ntok as the paired forward_math() call so
    // the embedding representations seen by the router are consistent.
    [[nodiscard]] autograd::Variable router_logits(
        const Tensor& token_ids,
        const NumericTokenizer& ntok,
        TokenMode mode = TokenMode::Real) const;
    // Convenience overload for Real mode (no remapping needed).
    [[nodiscard]] autograd::Variable router_logits(const Tensor& token_ids) const;

    [[nodiscard]] std::vector<autograd::Variable*> parameters();
    // Router + math_block parameters only; includes tok_emb weight.
    [[nodiscard]] std::vector<autograd::Variable*> math_parameters();
    // math_block_ parameters only — excludes tok_emb, blocks_, ln_f.
    // Use as the optimizer target in Phase 1 curriculum so tok_emb stays
    // consistent with Phase 2 initialisation, enabling clean weight transfer.
    [[nodiscard]] std::vector<autograd::Variable*> math_block_only_parameters();

    // Copy math_block_ weights from source into this model (curriculum learning).
    // source and *this must have the same embed_dim, n_heads, n_kv_heads, d_ff.
    void import_math_block(MathGPT& source);

    [[nodiscard]] int64_t     l_math()     const noexcept;
    [[nodiscard]] int64_t     vocab_size() const noexcept;
    [[nodiscard]] int64_t     embed_dim()  const noexcept;
    [[nodiscard]] std::size_t num_layers() const noexcept;

private:
    Embedding                           tok_emb_;
    std::vector<ModernTransformerBlock>  blocks_;      // n_layers blocks
    MathTransformerBlock                 math_block_;  // replaces blocks_[l_math_] in forward_math
    RMSNorm                              ln_f_;
    int64_t                              l_math_;
    float                                math_boost_;  // post-ln_f injection scale

    // Remap numeric token IDs for the embedding lookup (leaves non-numeric unchanged).
    // Real → identity; Anon → all numeric to num_placeholder_token();
    // Algebraic → first distinct value → X0, second → X1, etc. (per expression).
    [[nodiscard]] static Tensor remap_tokens(
        const Tensor& token_ids, const NumericTokenizer& ntok, TokenMode mode);

    [[nodiscard]] static std::vector<float> build_register(
        const Tensor& token_ids, const NumericTokenizer& ntok);
};

} // namespace sub0llm::nn
