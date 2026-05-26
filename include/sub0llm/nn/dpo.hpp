#pragma once

#include "sub0llm/autograd/variable.hpp"
#include "sub0llm/core/tensor.hpp"
#include <vector>

namespace sub0llm::nn {

// log_prob_sequence: mean of log P(token_ids[t+1] | logits[t]) for t=0..T-2.
// logits: Variable (T, V), token_ids: Tensor (T,) int32 — must have numel >= T.
// Returns: scalar Variable (per-token mean log-prob, <= 0; more negative = worse)
// Using per-token mean avoids length bias when comparing sequences of different lengths.
// Throws if T < 2 or token_ids.numel() < T.
[[nodiscard]] autograd::Variable log_prob_sequence(
    const autograd::Variable& logits,
    const Tensor&              token_ids);

// dpo_loss: Direct Preference Optimization loss for a single (winner, loser) pair.
//
// L = -log σ(β * ((lp_w - ref_lp_w) - (lp_l - ref_lp_l)))
//
// where lp_w = log_prob_sequence(logits_w, ids_w) under the current policy,
//       lp_l = log_prob_sequence(logits_l, ids_l) under the current policy,
//       ref_lp_w / ref_lp_l = pre-computed log-probs under the FROZEN reference model.
//
// ref_lp_w and ref_lp_l are plain floats (pre-computed, no gradient).
// beta: DPO temperature (typically 0.1).
[[nodiscard]] autograd::Variable dpo_loss(
    const autograd::Variable& logits_w,  const Tensor& ids_w,
    const autograd::Variable& logits_l,  const Tensor& ids_l,
    float ref_lp_w, float ref_lp_l,
    float beta = 0.1f);

} // namespace sub0llm::nn
