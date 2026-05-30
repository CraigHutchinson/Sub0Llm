#include "sub0llm/nn/long_context.hpp"

#include "sub0llm/core/ops.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>

namespace sub0llm::nn {

namespace {

// Sample next token from logits (1, V) using temperature + optional top-k.
int32_t sample_token(const Tensor& logits, float temperature, int32_t top_k,
                     std::mt19937& rng) {
    auto ld = logits.data_as<float>();
    const int64_t V = static_cast<int64_t>(logits.numel());

    if (temperature <= 0.0f || (top_k == 1)) {
        // Greedy
        int64_t best = 0;
        for (int64_t v = 1; v < V; ++v)
            if (ld[static_cast<std::size_t>(v)] > ld[static_cast<std::size_t>(best)])
                best = v;
        return static_cast<int32_t>(best);
    }

    // Apply temperature
    std::vector<float> probs(static_cast<std::size_t>(V));
    for (std::size_t v = 0; v < static_cast<std::size_t>(V); ++v)
        probs[v] = ld[v] / temperature;

    // Optionally restrict to top-k
    int32_t k = (top_k > 0 && top_k < static_cast<int32_t>(V)) ? top_k
                                                                   : static_cast<int32_t>(V);
    if (k < static_cast<int32_t>(V)) {
        // Find k-th largest value threshold
        std::vector<float> sorted_probs = probs;
        std::nth_element(sorted_probs.begin(),
                         sorted_probs.begin() + (static_cast<std::size_t>(V) -
                                                 static_cast<std::size_t>(k)),
                         sorted_probs.end());
        const float threshold =
            sorted_probs[static_cast<std::size_t>(V) - static_cast<std::size_t>(k)];
        const float neg_inf = -std::numeric_limits<float>::infinity();
        for (std::size_t v = 0; v < static_cast<std::size_t>(V); ++v)
            if (probs[v] < threshold) probs[v] = neg_inf;
    }

    // Softmax
    float max_val = *std::max_element(probs.begin(), probs.end());
    float sum     = 0.0f;
    for (auto& p : probs) { p = std::exp(p - max_val); sum += p; }
    for (auto& p : probs) p /= sum;

    std::discrete_distribution<int32_t> dist(probs.begin(), probs.end());
    return dist(rng);
}

} // anonymous namespace

std::vector<int32_t> generate_cached(const ModernGPT&         model,
                                      const std::vector<int32_t>& prompt_ids,
                                      KVCache&                  cache,
                                      const LongContextConfig&  config) {
    if (prompt_ids.empty())
        throw std::runtime_error("generate_cached: prompt_ids must not be empty");

    std::mt19937 rng(42);
    std::vector<int32_t> generated;

    // Encode prompt — fill the KV cache one token at a time
    Tensor last_logits;
    for (int32_t tok : prompt_ids) {
        if (cache.full()) break;
        last_logits = model.forward_one(tok, cache, config.rope_scaling);
    }

    // Generate new tokens
    int32_t next_tok = sample_token(last_logits, config.temperature, config.top_k, rng);
    for (int64_t step = 0; step < config.max_new_tokens; ++step) {
        generated.push_back(next_tok);
        if (config.on_token && !config.on_token(next_tok)) break;
        if (cache.full()) break;
        last_logits = model.forward_one(next_tok, cache, config.rope_scaling);
        next_tok = sample_token(last_logits, config.temperature, config.top_k, rng);
    }

    return generated;
}

} // namespace sub0llm::nn
