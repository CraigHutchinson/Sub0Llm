#include "sub0llm/nn/dpo.hpp"

#include "sub0llm/autograd/ops.hpp"
#include "sub0llm/core/tensor.hpp"

#include <format>
#include <stdexcept>

namespace sub0llm::nn {

using namespace sub0llm::autograd;

autograd::Variable log_prob_sequence(
    const autograd::Variable& logits,
    const Tensor&              token_ids)
{
    const int64_t T = logits.data().shape(0);
    if (T < 2)
        throw std::runtime_error(std::format(
            "log_prob_sequence: sequence length T={} must be >= 2", T));
    if (static_cast<int64_t>(token_ids.numel()) < T)
        throw std::runtime_error(std::format(
            "log_prob_sequence: token_ids has {} elements but logits has {} rows",
            token_ids.numel(), T));

    // Valid input positions: rows [0, T-1)
    Variable logits_trunc = narrow(logits, 0, T - 1);

    // Targets: token_ids[1..T-1]
    Tensor targets({T - 1}, DType::Int32, token_ids.device());
    {
        const auto src = token_ids.data_as<int32_t>();
        auto       dst = targets.data_as<int32_t>();
        for (int64_t i = 0; i < T - 1; ++i)
            dst[static_cast<std::size_t>(i)] = src[static_cast<std::size_t>(i + 1)];
    }

    // cross_entropy returns mean NLL; return negated mean = mean log-prob.
    // Using per-token average avoids length bias when sequences differ in length.
    Variable ce_mean = cross_entropy(logits_trunc, targets);
    return scale(ce_mean, -1.0f);
}

autograd::Variable dpo_loss(
    const autograd::Variable& logits_w,  const Tensor& ids_w,
    const autograd::Variable& logits_l,  const Tensor& ids_l,
    float ref_lp_w, float ref_lp_l,
    float beta)
{
    if (beta <= 0.0f)
        throw std::runtime_error(std::format(
            "dpo_loss: beta={} must be positive", beta));

    Variable lp_w = log_prob_sequence(logits_w, ids_w);
    Variable lp_l = log_prob_sequence(logits_l, ids_l);

    Variable policy_diff = sub(lp_w, lp_l);

    // ref_offset = ref_lp_l - ref_lp_w  (constant, no gradient)
    const float ref_offset = ref_lp_l - ref_lp_w;

    // Add constant offset as a non-grad Variable
    Tensor offset_t({1}, DType::Float32);
    offset_t.data_as<float>()[0] = beta * ref_offset;
    Variable offset_var(std::move(offset_t), false);

    // margin = beta * policy_diff + beta * ref_offset
    Variable margin = add(scale(policy_diff, beta), offset_var);

    // L = -log σ(margin)
    return scale(log_sigmoid(margin), -1.0f);
}

} // namespace sub0llm::nn
