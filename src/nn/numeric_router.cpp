#include "sub0llm/nn/numeric_router.hpp"

#include "sub0llm/autograd/ops.hpp"
#include "sub0llm/core/tensor.hpp"

#include <algorithm>
#include <cstddef>
#include <format>
#include <numeric>
#include <stdexcept>

namespace sub0llm::nn {

using namespace autograd;

NumericRouter::NumericRouter(int64_t D, std::uint64_t seed)
    : D_(D), gate_(D, static_cast<int64_t>(kNumRouteTypes), seed) {
    if (D <= 0)
        throw std::runtime_error(
            std::format("NumericRouter: D={} must be positive", D));
}

NumericRouter::ForwardResult NumericRouter::forward(const autograd::Variable& x) const {
    const auto& xd = x.data();
    if (xd.ndim() != 2)
        throw std::runtime_error("NumericRouter::forward: x must be 2D (T, D)");

    const int64_t T = xd.shape(0);
    const int64_t K = static_cast<int64_t>(kNumRouteTypes);

    Variable logits = gate_.forward(x);          // (T, K)
    Variable probs  = softmax(logits);            // (T, K)

    // Build hard argmax mask
    auto prob_raw = probs.data().data_as<float>();
    Tensor mask({T, K}, DType::Float32);
    auto mask_sp = mask.data_as<float>();
    std::fill(mask_sp.begin(), mask_sp.end(), 0.0f);

    const auto T_sz = static_cast<std::size_t>(T);
    const auto K_sz = static_cast<std::size_t>(K);

    for (std::size_t t = 0; t < T_sz; ++t) {
        std::size_t best = 0;
        float best_val = prob_raw[t * K_sz];
        for (std::size_t k = 1; k < K_sz; ++k) {
            if (prob_raw[t * K_sz + k] > best_val) {
                best_val = prob_raw[t * K_sz + k];
                best = k;
            }
        }
        mask_sp[t * K_sz + best] = 1.0f;
    }

    return {std::move(probs), std::move(mask)};
}

Variable NumericRouter::router_logits(const autograd::Variable& x) const {
    return gate_.forward(x);   // (T, K) pre-softmax
}

std::vector<RouteType> NumericRouter::route_hard(const autograd::Variable& x) const {
    auto [probs, mask] = forward(x);

    const int64_t T = x.data().shape(0);
    const int64_t K = static_cast<int64_t>(kNumRouteTypes);
    auto mask_sp = mask.data_as<float>();

    std::vector<RouteType> routes;
    routes.reserve(static_cast<std::size_t>(T));

    const auto T_sz = static_cast<std::size_t>(T);
    const auto K_sz = static_cast<std::size_t>(K);

    for (std::size_t t = 0; t < T_sz; ++t) {
        int best = 0;
        for (std::size_t k = 1; k < K_sz; ++k) {
            if (mask_sp[t * K_sz + k] > mask_sp[t * K_sz + static_cast<std::size_t>(best)])
                best = static_cast<int>(k);
        }
        routes.push_back(static_cast<RouteType>(best));
    }

    return routes;
}

std::vector<autograd::Variable*> NumericRouter::parameters() {
    return gate_.parameters();
}

} // namespace sub0llm::nn
