#pragma once
#include "sub0llm/core/tensor.hpp"
#include <cstddef>
#include <cstdint>
#include <vector>

namespace sub0llm::nn {

// ── KVCache ───────────────────────────────────────────────────────────────────
//
// Pre-allocated key/value buffers for autoregressive inference.
// Avoids the O(n²) total recomputation that naive generate() causes.
//
// Layout: cache.K[layer][kv_head]  — Tensor (max_seq, head_dim), row-major.
//         cache.V[layer][kv_head]  — same shape.
//         cache.filled             — how many token positions are occupied.
//
// Usage:
//   auto cache = model.make_kv_cache(4096);
//   for (int32_t tok : prompt_ids)
//       auto logits = model.forward_one(tok, cache.filled, cache);
//   for (int64_t step = 0; step < max_new; ++step) {
//       auto tok = sample(logits);
//       logits = model.forward_one(tok, cache.filled, cache);
//   }

struct KVCache {
    std::vector<std::vector<Tensor>> K;  // [n_layers][n_kv_heads] → (max_seq, head_dim)
    std::vector<std::vector<Tensor>> V;  // same shape
    int64_t filled   = 0;               // tokens currently in cache
    int64_t max_seq  = 0;               // pre-allocated capacity

    [[nodiscard]] bool full()     const noexcept { return filled >= max_seq; }
    [[nodiscard]] bool empty()    const noexcept { return filled == 0; }
    [[nodiscard]] int64_t remain() const noexcept { return max_seq - filled; }

    void reset() noexcept { filled = 0; }
};

} // namespace sub0llm::nn
