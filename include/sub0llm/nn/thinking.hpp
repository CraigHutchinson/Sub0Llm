#pragma once
#include "sub0llm/nn/modern_gpt.hpp"
#include "sub0llm/nn/sampler.hpp"
#include "sub0llm/core/tensor.hpp"
#include <cstdint>
#include <random>
#include <vector>

namespace sub0llm::nn {

// Configuration for thinking-token generation.
struct ThinkingConfig {
    SamplingConfig inner_cfg;          // sampling strategy for the reasoning scratchpad
    SamplingConfig answer_cfg;         // sampling strategy for the visible answer
    int32_t think_bos_id   = -1;       // token ID of <think>  (must be set; -1 = disabled)
    int32_t think_eos_id   = -1;       // token ID of </think>
    int64_t max_think_tokens  = 64;    // reasoning budget (hard cap)
    int64_t max_answer_tokens = 32;    // answer budget (hard cap)
};

// Output of one thinking-token generation pass.
struct ThinkingResult {
    std::vector<int32_t> thinking_tokens; // tokens emitted inside <think>…</think>
    std::vector<int32_t> answer_tokens;   // tokens emitted after </think>
    std::vector<int32_t> full_sequence;   // prompt + <think> + thinking + </think> + answer
    bool thinking_complete = false;       // true if </think> was found within budget
};

// Generate with a chain-of-thought scratchpad.
//
// Appends think_bos_id to the prompt, samples up to max_think_tokens thinking
// tokens (stopping early if think_eos_id is produced), then forces think_eos_id
// if the budget is exhausted, and finally samples up to max_answer_tokens answer
// tokens.
//
// Throws std::runtime_error if think_bos_id or think_eos_id is -1, or if
// prompt is empty, or if budgets are < 1.
[[nodiscard]] ThinkingResult generate_with_thinking(
    ModernGPT&                  model,
    const std::vector<int32_t>& prompt,
    const ThinkingConfig&       cfg,
    std::mt19937&               rng);

// Self-consistency decoding over multiple thinking traces.
//
// Runs generate_with_thinking n_samples times, collects the FIRST answer token
// from each run, and returns the plurality-vote winner (most frequent token).
// On a tie, the lowest token id wins.
// Throws if n_samples < 1 or any run produces an empty answer.
[[nodiscard]] int32_t think_self_consistency(
    ModernGPT&                  model,
    const std::vector<int32_t>& prompt,
    const ThinkingConfig&       cfg,
    int                         n_samples,
    std::mt19937&               rng);

} // namespace sub0llm::nn
