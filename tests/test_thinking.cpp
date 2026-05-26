#include <catch2/catch_test_macros.hpp>

#include "sub0llm/nn/thinking.hpp"
#include "sub0llm/nn/modern_gpt.hpp"
#include "sub0llm/nn/sampler.hpp"

#include <cstdint>
#include <random>
#include <vector>

using namespace sub0llm;
using namespace sub0llm::nn;

// ── Helpers ───────────────────────────────────────────────────────────────────

static ThinkingConfig make_cfg(int32_t bos, int32_t eos,
                                int64_t max_think = 8, int64_t max_answer = 4) {
    ThinkingConfig cfg;
    cfg.think_bos_id = bos;
    cfg.think_eos_id = eos;
    cfg.max_think_tokens  = max_think;
    cfg.max_answer_tokens = max_answer;
    cfg.inner_cfg.mode  = SamplingMode::Greedy;
    cfg.answer_cfg.mode = SamplingMode::Greedy;
    return cfg;
}

// ── generate_with_thinking tests ──────────────────────────────────────────────

TEST_CASE("generate_with_thinking — full_sequence length correct") {
    ModernGPT model(/*vocab_size=*/16, /*embed_dim=*/16,
                    /*n_heads=*/2, /*n_kv_heads=*/1, /*n_layers=*/1,
                    /*d_ff=*/0, /*n_mtp_heads=*/0, /*seed=*/42);

    std::vector<int32_t> prompt = {1, 2, 3};
    auto cfg = make_cfg(/*bos=*/14, /*eos=*/15);
    std::mt19937 rng(42);

    auto result = generate_with_thinking(model, prompt, cfg, rng);

    const std::size_t expected =
        prompt.size() + 1 +
        result.thinking_tokens.size() + 1 +
        result.answer_tokens.size();
    REQUIRE(result.full_sequence.size() == expected);
}

TEST_CASE("generate_with_thinking — answer tokens within budget") {
    ModernGPT model(16, 16, 2, 1, 1, 0, 0, 42);

    std::vector<int32_t> prompt = {0, 1};
    auto cfg = make_cfg(14, 15, /*max_think=*/8, /*max_answer=*/4);
    std::mt19937 rng(1);

    auto result = generate_with_thinking(model, prompt, cfg, rng);

    REQUIRE(result.answer_tokens.size() <= static_cast<std::size_t>(cfg.max_answer_tokens));
}

TEST_CASE("generate_with_thinking — thinking tokens within budget") {
    ModernGPT model(16, 16, 2, 1, 1, 0, 0, 42);

    std::vector<int32_t> prompt = {0, 1};
    auto cfg = make_cfg(14, 15, /*max_think=*/8, /*max_answer=*/4);
    std::mt19937 rng(1);

    auto result = generate_with_thinking(model, prompt, cfg, rng);

    REQUIRE(result.thinking_tokens.size() <= static_cast<std::size_t>(cfg.max_think_tokens));
}

TEST_CASE("generate_with_thinking — think_bos in full_sequence") {
    ModernGPT model(16, 16, 2, 1, 1, 0, 0, 42);

    std::vector<int32_t> prompt = {1, 2, 3};
    const int32_t bos_id = 14;
    auto cfg = make_cfg(bos_id, /*eos=*/15);
    std::mt19937 rng(0);

    auto result = generate_with_thinking(model, prompt, cfg, rng);

    REQUIRE(result.full_sequence[prompt.size()] == bos_id);
}

TEST_CASE("generate_with_thinking — think_eos in full_sequence") {
    ModernGPT model(16, 16, 2, 1, 1, 0, 0, 42);

    std::vector<int32_t> prompt = {1, 2, 3};
    const int32_t eos_id = 15;
    auto cfg = make_cfg(/*bos=*/14, eos_id);
    std::mt19937 rng(0);

    auto result = generate_with_thinking(model, prompt, cfg, rng);

    // think_eos must appear right after prompt + bos + thinking_tokens
    const std::size_t eos_pos = prompt.size() + 1 + result.thinking_tokens.size();
    REQUIRE(result.full_sequence[eos_pos] == eos_id);
}

TEST_CASE("generate_with_thinking — budget invariant holds") {
    // Verify that either:
    //   (a) thinking_complete=true and thinking_tokens.size() < max_think, or
    //   (b) thinking_complete=false and thinking_tokens.size() == max_think
    // Use a small budget so the test runs fast.
    ModernGPT model(16, 16, 2, 1, 1, 0, 0, 42);

    std::vector<int32_t> prompt = {0, 1};
    const int64_t max_think = 3;
    auto cfg = make_cfg(/*bos=*/14, /*eos=*/15, max_think, /*max_answer=*/2);
    std::mt19937 rng(0);

    auto result = generate_with_thinking(model, prompt, cfg, rng);

    if (result.thinking_complete) {
        REQUIRE(result.thinking_tokens.size() < static_cast<std::size_t>(max_think));
    } else {
        REQUIRE(result.thinking_tokens.size() == static_cast<std::size_t>(max_think));
    }
}

TEST_CASE("generate_with_thinking — empty prompt throws") {
    ModernGPT model(16, 16, 2, 1, 1, 0, 0, 42);
    auto cfg = make_cfg(14, 15);
    std::mt19937 rng(0);

    REQUIRE_THROWS_AS(generate_with_thinking(model, {}, cfg, rng), std::runtime_error);
}

TEST_CASE("generate_with_thinking — negative bos_id throws") {
    ModernGPT model(16, 16, 2, 1, 1, 0, 0, 42);
    auto cfg = make_cfg(/*bos=*/-1, /*eos=*/15);
    std::mt19937 rng(0);

    REQUIRE_THROWS_AS(generate_with_thinking(model, {1}, cfg, rng), std::runtime_error);
}

TEST_CASE("generate_with_thinking — zero budget throws") {
    ModernGPT model(16, 16, 2, 1, 1, 0, 0, 42);
    auto cfg = make_cfg(14, 15);
    cfg.max_think_tokens = 0;
    std::mt19937 rng(0);

    REQUIRE_THROWS_AS(generate_with_thinking(model, {1}, cfg, rng), std::runtime_error);
}

// ── think_self_consistency tests ──────────────────────────────────────────────

TEST_CASE("think_self_consistency — returns valid vocab token") {
    constexpr int64_t V = 16;
    ModernGPT model(V, 16, 2, 1, 1, 0, 0, 42);

    std::vector<int32_t> prompt = {0, 1, 2};
    auto cfg = make_cfg(14, 15, 4, 3);
    std::mt19937 rng(7);

    int32_t tok = think_self_consistency(model, prompt, cfg, 3, rng);

    REQUIRE(tok >= 0);
    REQUIRE(tok < static_cast<int32_t>(V));
}

TEST_CASE("think_self_consistency — zero samples throws") {
    ModernGPT model(16, 16, 2, 1, 1, 0, 0, 42);
    auto cfg = make_cfg(14, 15);
    std::mt19937 rng(0);

    REQUIRE_THROWS_AS(think_self_consistency(model, {1}, cfg, 0, rng), std::runtime_error);
}

TEST_CASE("think_self_consistency — deterministic on greedy") {
    ModernGPT model(16, 16, 2, 1, 1, 0, 0, 42);

    std::vector<int32_t> prompt = {0, 1};
    // Greedy cfg — deterministic, so all 3 calls must return the same token
    auto cfg = make_cfg(14, 15, 4, 3);
    // inner and answer are already greedy from make_cfg

    int32_t results[3];
    for (int i = 0; i < 3; ++i) {
        std::mt19937 rng(0);  // same seed each time
        results[i] = think_self_consistency(model, prompt, cfg, 3, rng);
    }

    REQUIRE(results[0] == results[1]);
    REQUIRE(results[1] == results[2]);
}
