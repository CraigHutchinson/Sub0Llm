#pragma once

#include "sub0llm/autograd/variable.hpp"
#include "sub0llm/core/tensor.hpp"
#include "sub0llm/nn/gpt.hpp"

#include <cstdint>
#include <vector>

namespace sub0llm::nn {

enum class RouteType : int {
    FFN          = 0,
    Add          = 1,
    Sub          = 2,
    Mul          = 3,
    Div          = 4,
    IsLessThan   = 5,
    IsGreaterThan = 6,
    IsEqual      = 7,
    N_TYPES      = 8
};

inline constexpr int kNumRouteTypes = static_cast<int>(RouteType::N_TYPES);

// Linear(D, n_types) + softmax router with STE hard routing.
// Forward uses argmax (hard), backward flows through the softmax probabilities.
class NumericRouter {
public:
    explicit NumericRouter(int64_t D, std::uint64_t seed = 42);

    struct ForwardResult {
        autograd::Variable soft_probs;  // (T, n_types) softmax probabilities
        Tensor             hard_mask;   // (T, n_types) argmax one-hot 0/1
    };

    [[nodiscard]] ForwardResult forward(const autograd::Variable& x) const;
    [[nodiscard]] std::vector<RouteType> route_hard(const autograd::Variable& x) const;
    // Returns pre-softmax logits (T, n_types) — use for supervision cross-entropy loss.
    [[nodiscard]] autograd::Variable router_logits(const autograd::Variable& x) const;
    [[nodiscard]] std::vector<autograd::Variable*> parameters();

private:
    int64_t D_;
    Linear  gate_;   // D → n_types
};

} // namespace sub0llm::nn
