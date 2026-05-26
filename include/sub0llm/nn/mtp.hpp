#pragma once

#include "sub0llm/autograd/variable.hpp"
#include "sub0llm/core/tensor.hpp"
#include "sub0llm/nn/modern_gpt.hpp"
#include "sub0llm/nn/sampler.hpp"

#include <random>
#include <vector>

namespace sub0llm::nn {

// ── MTP training loss ─────────────────────────────────────────────────────────
//
// Computes the combined Multi-Token Prediction training loss:
//
//   L = sum_{k=0}^{K}  w_k * CE(narrow(head_k_logits, 0, T-k-1),
//                                tokens[start+k+1 .. start+seq_len-1])
//
// where K = model.n_mtp_heads() and w_0 = 1, w_{k>0} = aux_weight.
//
// The main head (k=0) is the standard next-token CE.  Each auxiliary head k
// predicts the token k+1 positions ahead from each context position.
// Reducing seq_len positions are used for higher-offset heads (e.g. head k
// is computed over positions 0 … T-k-2, giving T-k-1 predictions).
//
// Requirements:
//   • model.n_mtp_heads() >= 1
//   • token_ids.shape(0) == seq_len  (single contiguous window)
//   • seq_len >= 2 + model.n_mtp_heads()
[[nodiscard]] autograd::Variable mtp_train_loss(
    ModernGPT&    model,
    const Tensor& token_ids,
    int64_t       seq_len,
    float         aux_weight = 0.1f);

// ── MTP inference ─────────────────────────────────────────────────────────────
//
// Generates tokens using all K+1 heads in a single forward call:
//
//   Each forward pass over context of length T produces K+1 candidate tokens:
//     head_k last-row logits → sample token at position T+k  (k = 0…K)
//   All K+1 tokens are appended to the context and the next pass begins.
//
// With K MTP heads this delivers up to K+1 tokens per forward pass vs the
// standard 1-token-per-pass approach, at no extra parameter cost.
//
// Quality degrades slightly for k>0 because head k was trained assuming
// tokens [T..T+k-1] were available; at inference they are not yet seen.
// In practice this is small for moderate K (2–4).
//
// Returns: prompt + max_new_tokens newly generated tokens.
// If model.n_mtp_heads() == 0, falls back to standard single-token generation.
[[nodiscard]] std::vector<int32_t> mtp_generate(
    ModernGPT&                  model,
    const std::vector<int32_t>& prompt,
    int64_t                     max_new_tokens,
    const SamplingConfig&       cfg,
    std::mt19937&               rng);

// Statistics returned by mtp_generate_stats.
struct MtpGenStats {
    std::vector<int32_t> tokens;
    int64_t              n_forward_passes{0};
    double               tokens_per_pass{0.0};
};

// Same as mtp_generate but also reports forward-pass efficiency.
[[nodiscard]] MtpGenStats mtp_generate_stats(
    ModernGPT&                  model,
    const std::vector<int32_t>& prompt,
    int64_t                     max_new_tokens,
    const SamplingConfig&       cfg,
    std::mt19937&               rng);

} // namespace sub0llm::nn
