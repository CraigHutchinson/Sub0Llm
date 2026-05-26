#pragma once

#include "sub0llm/core/tensor.hpp"
#include "sub0llm/nn/modern_gpt.hpp"

#include <cstdint>
#include <random>
#include <vector>

namespace sub0llm::nn {

// ── SamplingConfig ────────────────────────────────────────────────────────────
//
// Controls how the next token is drawn from the logit distribution.
// Defaults produce greedy sampling (temperature=1, top_k=0, top_p=1 → argmax).

enum class SamplingMode {
    Greedy,      // deterministic argmax
    Temperature, // multinomial with temperature rescaling
    TopK,        // restrict to top-k logits, then temperature-multinomial
    TopP,        // restrict to smallest set with cumulative prob ≥ p, then sample
};

struct SamplingConfig {
    SamplingMode mode        = SamplingMode::Greedy;
    float        temperature = 1.0f;   // > 0; <1 sharpens, >1 flattens
    int64_t      top_k       = 0;      // 0 = disabled; must be > 0 when mode=TopK
    float        top_p       = 1.0f;   // (0,1]; 1.0 = disabled; nucleus threshold
};

// ── Sampling functions ────────────────────────────────────────────────────────
//
// All samplers accept a 1-D logits tensor of shape (V,) or a 2-D logits tensor
// of shape (T, V) — in the 2-D case only the last row (position T-1) is used.
// DType must be Float32.  Throws std::runtime_error on precondition violation.

// Argmax over the vocabulary dimension.  Deterministic; ignores temperature.
[[nodiscard]] int32_t greedy_sample(const Tensor& logits);

// Multinomial sampling after dividing logits by `temp`.
// Requires temp > 0.
[[nodiscard]] int32_t temperature_sample(const Tensor& logits,
                                         float          temp,
                                         std::mt19937&  rng);

// Keep the top-k logits, zero out the rest, then temperature-multinomial.
// Requires k >= 1.  If k >= V the full distribution is used (equivalent to
// temperature sampling).
[[nodiscard]] int32_t top_k_sample(const Tensor& logits,
                                   int64_t        k,
                                   float          temp,
                                   std::mt19937&  rng);

// Nucleus (top-p) sampling: sort by descending probability, keep the smallest
// prefix whose cumulative probability meets or exceeds p, then sample.
// Requires 0 < p <= 1 and temp > 0.
[[nodiscard]] int32_t top_p_sample(const Tensor& logits,
                                   float          p,
                                   float          temp,
                                   std::mt19937&  rng);

// ── Autoregressive generation loop ───────────────────────────────────────────
//
// Feeds `prompt` into `model`, then samples `max_new` additional tokens one at
// a time using the supplied SamplingConfig and rng.
//
// Returns the full token sequence (prompt + generated).
// The model is called in inference mode — Variable.data() is accessed directly;
// no backward pass is triggered.
//
// max_new must be >= 1.
[[nodiscard]] std::vector<int32_t> generate(ModernGPT&                   model,
                                             const std::vector<int32_t>&  prompt,
                                             int64_t                      max_new,
                                             const SamplingConfig&        cfg,
                                             std::mt19937&                rng);

} // namespace sub0llm::nn
