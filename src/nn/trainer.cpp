#include "sub0llm/nn/trainer.hpp"

#include "sub0llm/autograd/ops.hpp"
#include "sub0llm/core/ops.hpp"
#include "sub0llm/core/tensor.hpp"

#include <algorithm>
#include <format>
#include <stdexcept>

namespace sub0llm::nn {

autograd::Variable mtp_cross_entropy(
    const std::vector<autograd::Variable>& logits_heads,
    const Tensor&                          token_ids,
    const std::vector<float>&              weights)
{
    if (logits_heads.empty())
        throw std::runtime_error("mtp_cross_entropy: logits_heads must be non-empty");
    if (token_ids.ndim() != 1)
        throw std::runtime_error("mtp_cross_entropy: token_ids must be 1D (T,)");

    const int64_t T     = token_ids.shape(0);
    const int64_t nhead = static_cast<int64_t>(logits_heads.size());

    if (T < 2)
        throw std::runtime_error(std::format(
            "mtp_cross_entropy: sequence length T={} must be >= 2", T));

    // Number of heads that will actually fire (head k needs T - k - 1 > 0).
    const int64_t active = std::min(nhead, T - 1);

    // Build per-head weights (default: 1/active each, 0 for skipped heads).
    std::vector<float> w(static_cast<std::size_t>(nhead), 0.0f);
    if (weights.empty()) {
        const float inv = 1.0f / static_cast<float>(active);
        for (int64_t k = 0; k < active; ++k)
            w[static_cast<std::size_t>(k)] = inv;
    } else {
        if (static_cast<int64_t>(weights.size()) != nhead)
            throw std::runtime_error(std::format(
                "mtp_cross_entropy: weights.size()={} != nhead={}",
                weights.size(), nhead));
        w = weights;
    }

    using namespace autograd;

    Variable total_loss;
    bool first = true;

    for (int64_t k = 0; k < nhead; ++k) {
        const int64_t offset = k + 1;             // target offset = position + offset
        const int64_t len    = T - offset;        // number of valid (input, target) pairs
        if (len <= 0) break;                      // sequence too short for this head

        // Slice logits for this head: rows [0, len).
        Variable logits_k = narrow(logits_heads[static_cast<std::size_t>(k)], 0, len);

        // Build target tensor: token_ids[offset .. T-1].
        Tensor tgt({len}, DType::Int32);
        {
            const auto src = token_ids.data_as<int32_t>();
            auto       dst = tgt.data_as<int32_t>();
            for (int64_t i = 0; i < len; ++i)
                dst[static_cast<std::size_t>(i)] = src[static_cast<std::size_t>(i + offset)];
        }

        auto ce = cross_entropy(logits_k, tgt);

        // Scale by this head's weight.
        if (w[static_cast<std::size_t>(k)] != 1.0f)
            ce = scale(ce, w[static_cast<std::size_t>(k)]);

        if (first) { total_loss = ce;                  first = false; }
        else         total_loss = add(total_loss, ce);
    }

    if (!total_loss.defined())
        throw std::runtime_error(std::format(
            "mtp_cross_entropy: sequence length T={} too short for any head", T));

    return total_loss;
}

} // namespace sub0llm::nn
