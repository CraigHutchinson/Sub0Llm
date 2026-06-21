#include "sub0diff/eval/topic_drift.hpp"

#include <algorithm>
#include <array>
#include <numeric>
#include <unordered_map>
#include <unordered_set>

namespace sub0diff::eval {

std::vector<std::uint8_t> content_type_mask(std::span<const std::int32_t> train_ids,
                                            std::int64_t vocab, std::int64_t stop_k) {
    std::vector<std::uint64_t> freq(static_cast<std::size_t>(vocab), 0);
    for (auto id : train_ids)
        if (id >= 0 && id < vocab) ++freq[static_cast<std::size_t>(id)];

    std::vector<std::int64_t> order(static_cast<std::size_t>(vocab));
    std::iota(order.begin(), order.end(), 0);
    std::ranges::sort(order, [&](std::int64_t a, std::int64_t b) {
        const auto fa = freq[static_cast<std::size_t>(a)], fb = freq[static_cast<std::size_t>(b)];
        return fa != fb ? fa > fb : a < b;          // most-frequent types first, ties by id
    });

    std::vector<std::uint8_t> is_content(static_cast<std::size_t>(vocab), 1);
    const std::size_t k = std::min<std::size_t>(static_cast<std::size_t>(std::max<std::int64_t>(0, stop_k)),
                                                order.size());
    for (std::size_t i = 0; i < k; ++i)             // the stop_k most frequent = function words
        is_content[static_cast<std::size_t>(order[i])] = 0;
    return is_content;
}

namespace {

// FNV-1a over n consecutive int32 tokens — a low-collision key for an n-gram set.
[[nodiscard]] std::uint64_t ngram_key(const std::int32_t* p, std::int64_t n) {
    std::uint64_t h = 1469598103934665603ull;
    for (std::int64_t i = 0; i < n; ++i) {
        h ^= static_cast<std::uint64_t>(static_cast<std::uint32_t>(p[i]));
        h *= 1099511628211ull;
    }
    return h;
}

// unique n-grams / total n-grams for one passage (0 total → returns -1 = undefined).
[[nodiscard]] double distinct_n(std::span<const std::int32_t> t, std::int64_t n) {
    const std::int64_t total = static_cast<std::int64_t>(t.size()) - n + 1;
    if (total <= 0) return -1.0;
    std::unordered_set<std::uint64_t> seen;
    seen.reserve(static_cast<std::size_t>(total));
    for (std::int64_t i = 0; i < total; ++i) seen.insert(ngram_key(t.data() + i, n));
    return static_cast<double>(seen.size()) / static_cast<double>(total);
}

// Fraction of a passage's distinct content types that occur ≥2× (entity recurrence). -1 = no
// content tokens (undefined).
[[nodiscard]] double content_recurrence(std::span<const std::int32_t> t,
                                        std::span<const std::uint8_t> is_content) {
    const auto vocab = static_cast<std::int32_t>(is_content.size());
    std::unordered_map<std::int32_t, std::uint32_t> cnt;
    for (auto id : t)
        if (id >= 0 && id < vocab && is_content[static_cast<std::size_t>(id)]) ++cnt[id];
    if (cnt.empty()) return -1.0;
    std::size_t recurring = 0;
    for (const auto& [id, c] : cnt) if (c >= 2) ++recurring;
    return static_cast<double>(recurring) / static_cast<double>(cnt.size());
}

[[nodiscard]] double jaccard(const std::unordered_set<std::int32_t>& a,
                             const std::unordered_set<std::int32_t>& b) {
    if (a.empty() && b.empty()) return -1.0;                 // undefined (no content either side)
    std::size_t inter = 0;
    const auto& small = a.size() < b.size() ? a : b;
    const auto& big   = a.size() < b.size() ? b : a;
    for (auto x : small) if (big.contains(x)) ++inter;
    const std::size_t uni = a.size() + b.size() - inter;
    return uni ? static_cast<double>(inter) / static_cast<double>(uni) : -1.0;
}

}  // namespace

TopicDriftResult evaluate_topic_drift(std::span<const std::span<const std::int32_t>> passages,
                                      std::span<const std::uint8_t> is_content,
                                      std::int64_t window, std::int64_t near_d, std::int64_t far_d) {
    TopicDriftResult out;
    std::array<double, 3> d_sum{0, 0, 0};
    std::array<std::uint64_t, 3> d_cnt{0, 0, 0};
    double rec_sum = 0.0;
    std::uint64_t rec_cnt = 0;
    double near_sum = 0.0, far_sum = 0.0;
    std::uint64_t near_cnt = 0, far_cnt = 0;
    const auto vocab = static_cast<std::int32_t>(is_content.size());

    for (const auto& t : passages) {
        if (t.empty()) continue;
        ++out.n_passages;
        out.n_tokens += t.size();

        for (std::int64_t k = 0; k < 3; ++k) {
            const double dn = distinct_n(t, k + 2);
            if (dn >= 0.0) { d_sum[static_cast<std::size_t>(k)] += dn; ++d_cnt[static_cast<std::size_t>(k)]; }
        }
        const double rec = content_recurrence(t, is_content);
        if (rec >= 0.0) { rec_sum += rec; ++rec_cnt; }

        // Content-window sets, then pairwise Jaccard bucketed by window distance.
        const std::int64_t n_win = static_cast<std::int64_t>(t.size()) / window;
        std::vector<std::unordered_set<std::int32_t>> sets(static_cast<std::size_t>(std::max<std::int64_t>(0, n_win)));
        for (std::int64_t w = 0; w < n_win; ++w)
            for (std::int64_t i = w * window; i < (w + 1) * window; ++i) {
                const std::int32_t id = t[static_cast<std::size_t>(i)];
                if (id >= 0 && id < vocab && is_content[static_cast<std::size_t>(id)])
                    sets[static_cast<std::size_t>(w)].insert(id);
            }
        for (std::int64_t i = 0; i < n_win; ++i)
            for (std::int64_t j = i + 1; j < n_win; ++j) {
                const double jc = jaccard(sets[static_cast<std::size_t>(i)], sets[static_cast<std::size_t>(j)]);
                if (jc < 0.0) continue;
                const std::int64_t d = j - i;
                if (d <= near_d)      { near_sum += jc; ++near_cnt; }
                else if (d >= far_d)  { far_sum  += jc; ++far_cnt; }
            }
    }

    out.content_recurrence = rec_cnt ? rec_sum / static_cast<double>(rec_cnt) : 0.0;
    out.distinct2 = d_cnt[0] ? d_sum[0] / static_cast<double>(d_cnt[0]) : 0.0;
    out.distinct3 = d_cnt[1] ? d_sum[1] / static_cast<double>(d_cnt[1]) : 0.0;
    out.distinct4 = d_cnt[2] ? d_sum[2] / static_cast<double>(d_cnt[2]) : 0.0;
    out.persistence_near = near_cnt ? near_sum / static_cast<double>(near_cnt) : 0.0;
    out.persistence_far  = far_cnt  ? far_sum  / static_cast<double>(far_cnt)  : 0.0;
    out.drift = out.persistence_near > 0.0 ? 1.0 - out.persistence_far / out.persistence_near : 0.0;
    return out;
}

TopicDriftResult evaluate_topic_drift(std::span<const std::int32_t> passage,
                                      std::span<const std::uint8_t> is_content,
                                      std::int64_t window, std::int64_t near_d, std::int64_t far_d) {
    std::array<std::span<const std::int32_t>, 1> one{passage};
    return evaluate_topic_drift(one, is_content, window, near_d, far_d);
}

}  // namespace sub0diff::eval
