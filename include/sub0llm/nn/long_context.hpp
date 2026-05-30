#pragma once

#include "sub0llm/nn/kv_cache.hpp"
#include "sub0llm/nn/modern_gpt.hpp"

#include <cstdint>
#include <functional>
#include <vector>

namespace sub0llm::nn {

// Configuration for long-context KV-cached generation.
struct LongContextConfig {
    int64_t max_new_tokens = 200;       // tokens to generate after the prompt
    float   temperature    = 1.0f;      // sampling temperature (1 = greedy at limit)
    int32_t top_k          = 0;         // top-k sampling; 0 = greedy / temperature only
    float   rope_scaling   = 1.0f;      // NTK-aware RoPE scale (>1 extends context)
    // Called after every generated token; return false to stop early.
    std::function<bool(int32_t tok)> on_token = nullptr;
};

// Autoregressive generation with a pre-allocated KV cache.
//
// Encodes the prompt token-by-token (filling the cache), then generates up to
// config.max_new_tokens new tokens.  The combined sequence must fit within
// cache.max_seq; tokens beyond that are silently skipped.
//
// Returns the generated token ids (prompt NOT included).
[[nodiscard]] std::vector<int32_t> generate_cached(
    const ModernGPT&         model,
    const std::vector<int32_t>& prompt_ids,
    KVCache&                  cache,
    const LongContextConfig&  config = {});

} // namespace sub0llm::nn
