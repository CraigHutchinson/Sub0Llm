// unigram.cpp — Unigram LM vocabulariser (see sub0/unigram.hpp).

#include "sub0/unigram.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_set>

namespace sub0::tok {

namespace {
constexpr double INF = std::numeric_limits<double>::infinity();

// Core Viterbi over `word`: min Σ −log p. `exclude` (>=0) disables that token id (for the loss-delta
// prune's "best alternative without t"). `want_path` fills `path` with the chosen ids. Returns the
// total cost (INF if unreachable, which can happen only when a needed single byte is excluded).
double viterbi(const Unigram& u, std::string_view word, int exclude, bool want_path, std::vector<int>* path) {
    const int L = static_cast<int>(word.size());
    std::vector<double> dp(static_cast<std::size_t>(L) + 1, INF);
    std::vector<int>    bi(static_cast<std::size_t>(L) + 1, -1), bid(static_cast<std::size_t>(L) + 1, -1);
    dp[0] = 0.0;
    std::string key;
    for (int i = 0; i < L; ++i) {
        if (dp[static_cast<std::size_t>(i)] == INF) continue;
        const int lmax = std::min(u.max_len, L - i);
        for (int l = 1; l <= lmax; ++l) {
            key.assign(word.data() + i, static_cast<std::size_t>(l));
            const auto it = u.index.find(key);
            if (it == u.index.end() || it->second == exclude) continue;
            const double cost = dp[static_cast<std::size_t>(i)] - u.logp[static_cast<std::size_t>(it->second)];
            if (cost < dp[static_cast<std::size_t>(i + l)]) {
                dp[static_cast<std::size_t>(i + l)]  = cost;
                bi[static_cast<std::size_t>(i + l)]  = i;
                bid[static_cast<std::size_t>(i + l)] = it->second;
            }
        }
    }
    const double total = dp[static_cast<std::size_t>(L)];
    if (want_path && path) {
        path->clear();
        if (total != INF)
            for (int j = L; j > 0;) { path->push_back(bid[static_cast<std::size_t>(j)]); j = bi[static_cast<std::size_t>(j)]; }
        std::reverse(path->begin(), path->end());
    }
    return total;
}
}  // namespace

std::vector<int> Unigram::segment(std::string_view word) const {
    std::vector<int> path;
    viterbi(*this, word, /*exclude=*/-1, /*want_path=*/true, &path);
    return path;
}

long long corpus_tokens(const Unigram& u, const std::vector<std::pair<std::string, long long>>& words,
                        long long* bytes) {
    long long total = 0, by = 0;
    for (const auto& [w, f] : words) {
        total += f * static_cast<long long>(u.segment(w).size());
        by    += f * static_cast<long long>(w.size());
    }
    if (bytes) *bytes = by;
    return total;
}

Unigram learn_unigram(const std::vector<std::pair<std::string, long long>>& words, const UnigramOptions& opt) {
    // 1. Count every substring (k=1..max_piece) weighted by word frequency -- the candidate pool.
    std::unordered_map<std::string, long long> sub;
    sub.reserve(1u << 16);
    for (const auto& [w, f] : words) {
        const int n = static_cast<int>(w.size());
        const int kmax = std::min(n, opt.max_piece);
        for (int k = 1; k <= kmax; ++k)
            for (int i = 0; i + k <= n; ++i) sub[w.substr(static_cast<std::size_t>(i), static_cast<std::size_t>(k))] += f;
    }

    // 2. Seed: all single bytes (MANDATORY -> every word stays encodable) + the most frequent
    //    multi-byte pieces (count >= min_count) up to ~target*seed_mult candidates.
    std::vector<std::pair<long long, std::string>> singles, multi;
    for (const auto& [s, c] : sub) {
        if (s.size() == 1)              singles.push_back({c, s});
        else if (c >= opt.min_count)    multi.push_back({c, s});
    }
    std::sort(multi.begin(), multi.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    const std::size_t seed_n = static_cast<std::size_t>(std::max(opt.target, 1)) * static_cast<std::size_t>(std::max(1, opt.seed_mult));
    if (seed_n > singles.size() && multi.size() > seed_n - singles.size())
        multi.resize(seed_n - singles.size());

    Unigram u;
    std::vector<double> counts;                      // parallel to u.token (expected usage)
    auto add = [&](const std::string& s, double c) {
        u.index.emplace(s, u.size());
        u.token.push_back(s);
        u.logp.push_back(0.0);
        counts.push_back(c);
        u.max_len = std::max(u.max_len, static_cast<int>(s.size()));
    };
    for (const auto& [c, s] : singles) add(s, static_cast<double>(c));
    for (const auto& [c, s] : multi)   add(s, static_cast<double>(c));

    auto set_logp = [&] {
        double tot = 0.0;
        for (double c : counts) tot += c;
        const double lt = std::log(tot > 0.0 ? tot : 1.0);
        for (std::size_t i = 0; i < counts.size(); ++i)
            u.logp[i] = std::log(counts[i] > 1e-9 ? counts[i] : 1e-9) - lt;
    };
    auto run_em = [&](int iters) {
        for (int it = 0; it < iters; ++it) {
            std::fill(counts.begin(), counts.end(), 0.0);
            for (const auto& [w, f] : words)
                for (int id : u.segment(w)) counts[static_cast<std::size_t>(id)] += static_cast<double>(f);
            set_logp();
        }
    };

    set_logp();
    run_em(opt.em_iters);

    // 3. Prune to target by LOSS-DELTA (the SentencePiece criterion): a token's value is how much the
    //    corpus encoding cost WORSENS if it is removed -- for each word using it, the gap between its
    //    best segmentation and the best one that avoids the token, summed and frequency-weighted. This
    //    keeps a rarely-used token that is the only compact way to encode some words (count-based
    //    pruning wrongly drops those), and removes tokens whose spans have a cheap alternative. Single
    //    bytes are never pruned, so every word stays encodable and no surviving slot is dead.
    while (u.size() > opt.target) {
        std::vector<double> loss(static_cast<std::size_t>(u.size()), 0.0);
        std::vector<int> path;
        for (const auto& [w, f] : words) {
            const double best = viterbi(u, w, -1, true, &path);
            if (best == INF) continue;
            std::unordered_set<int> seen(path.begin(), path.end());
            for (int t : seen) {
                if (u.token[static_cast<std::size_t>(t)].size() == 1) continue;     // singles are mandatory
                const double alt = viterbi(u, w, t, false, nullptr);               // best without token t
                if (alt != INF) loss[static_cast<std::size_t>(t)] += static_cast<double>(f) * (alt - best);
            }
        }
        std::vector<int> prunable;
        for (int id = 0; id < u.size(); ++id) if (u.token[static_cast<std::size_t>(id)].size() > 1) prunable.push_back(id);
        std::sort(prunable.begin(), prunable.end(),
                  [&](int a, int b) { return loss[static_cast<std::size_t>(a)] < loss[static_cast<std::size_t>(b)]; });
        const int target_size = std::max(opt.target, static_cast<int>(static_cast<double>(u.size()) * (1.0 - opt.drop_frac)));
        const int to_remove   = std::min(static_cast<int>(prunable.size()), u.size() - target_size);
        if (to_remove <= 0) break;

        std::vector<char> drop(static_cast<std::size_t>(u.size()), 0);
        for (int i = 0; i < to_remove; ++i) drop[static_cast<std::size_t>(prunable[static_cast<std::size_t>(i)])] = 1;

        Unigram nu;
        std::vector<double> nc;
        for (int id = 0; id < u.size(); ++id) {
            if (drop[static_cast<std::size_t>(id)]) continue;
            const std::string& s = u.token[static_cast<std::size_t>(id)];
            nu.index.emplace(s, nu.size());
            nu.token.push_back(s);
            nu.logp.push_back(0.0);
            nc.push_back(counts[static_cast<std::size_t>(id)]);
            nu.max_len = std::max(nu.max_len, static_cast<int>(s.size()));
        }
        u = std::move(nu);
        counts = std::move(nc);
        set_logp();
        run_em(opt.em_iters);
    }
    return u;
}

}  // namespace sub0::tok
