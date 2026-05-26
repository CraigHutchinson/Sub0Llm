#pragma once

#include "sub0llm/autograd/ops.hpp"
#include "sub0llm/autograd/variable.hpp"
#include "sub0llm/core/tensor.hpp"

#include <cstdint>
#include <vector>

namespace sub0llm::nn {

// ── mtp_cross_entropy ─────────────────────────────────────────────────────────
//
// Compute the combined MTP training loss from the output of
// ModernGPT::forward_mtp(token_ids).
//
//   logits_heads[k]: Variable of shape (T, vocab_size)
//     k=0  — weight-tied main LM head, predicts token at position t+1
//     k>0  — MTP head k, predicts token at position t+k+1
//
//   token_ids: Tensor (T,) int32 — the full input sequence
//
//   weights: per-head scaling factors (default: 1/n_heads for each head)
//
// For head k the valid positions are t = 0 … T-k-2:
//   logits slice  = logits_heads[k][0 : T-k-1]   shape (T-k-1, V)
//   target slice  = token_ids[k+1 : T]            shape (T-k-1,)
//
// Returns: scalar Variable = Σ_k weights[k] * CE(logits_k, targets_k)
// If the sequence is too short for any head (T <= 1) throws std::runtime_error.
[[nodiscard]] autograd::Variable mtp_cross_entropy(
    const std::vector<autograd::Variable>& logits_heads,
    const Tensor&                          token_ids,
    const std::vector<float>&              weights = {});

} // namespace sub0llm::nn
