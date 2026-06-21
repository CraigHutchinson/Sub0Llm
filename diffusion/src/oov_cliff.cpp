#include "sub0diff/eval/oov_cliff.hpp"

#include <algorithm>
#include <numeric>
#include <vector>

namespace sub0diff::eval {

// evaluate_oov_cliff is a header template (model-generic). Only the model-independent
// frequency split lives here.
std::vector<std::uint8_t> rare_type_mask(std::span<const std::int32_t> train_ids,
                                         std::int64_t vocab, double rare_frac) {
    std::vector<std::uint64_t> freq(static_cast<std::size_t>(vocab), 0);
    for (auto id : train_ids)
        if (id >= 0 && id < vocab) ++freq[static_cast<std::size_t>(id)];

    std::vector<std::int64_t> order(static_cast<std::size_t>(vocab));
    std::iota(order.begin(), order.end(), 0);
    std::ranges::sort(order, [&](std::int64_t a, std::int64_t b) {
        const auto fa = freq[static_cast<std::size_t>(a)], fb = freq[static_cast<std::size_t>(b)];
        return fa != fb ? fa < fb : a < b;          // rarest types first, ties by id
    });

    std::vector<std::uint8_t> is_rare(static_cast<std::size_t>(vocab), 0);
    const std::size_t n_rare = static_cast<std::size_t>(rare_frac * static_cast<double>(vocab));
    for (std::size_t i = 0; i < n_rare && i < order.size(); ++i)
        is_rare[static_cast<std::size_t>(order[i])] = 1;
    return is_rare;
}

}  // namespace sub0diff::eval
