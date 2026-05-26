#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "sub0llm/nn/sampler.hpp"
#include "sub0llm/core/tensor.hpp"

#include <algorithm>
#include <cstdint>
#include <map>
#include <numeric>
#include <random>

using namespace sub0llm;
using namespace sub0llm::nn;
using Catch::Matchers::WithinAbs;

// ── Helpers ───────────────────────────────────────────────────────────────────

static Tensor make_logits_1d(std::initializer_list<float> vals) {
    Tensor t({static_cast<int64_t>(vals.size())}, DType::Float32);
    auto sp = t.data_as<float>();
    std::size_t i = 0;
    for (float v : vals) sp[i++] = v;
    return t;
}

static Tensor make_logits_2d(int64_t T, int64_t V, std::initializer_list<float> last_row_vals) {
    // Fill earlier rows with zeros; last row with supplied values.
    Tensor t({T, V}, DType::Float32);
    auto sp = t.data_as<float>();
    std::fill(sp.begin(), sp.end(), 0.0f);
    std::size_t offset = static_cast<std::size_t>((T - 1) * V);
    std::size_t i      = 0;
    for (float v : last_row_vals) sp[offset + i++] = v;
    return t;
}

// ── greedy_sample ─────────────────────────────────────────────────────────────

TEST_CASE("greedy_sample — argmax on 1-D logits", "[sampler]") {
    // Highest logit is index 2.
    Tensor logits = make_logits_1d({0.1f, 0.5f, 3.0f, 1.2f});
    REQUIRE(greedy_sample(logits) == 2);
}

TEST_CASE("greedy_sample — uses last row of 2-D logits", "[sampler]") {
    // First row has max at index 0; last row has max at index 3.
    Tensor logits = make_logits_2d(3, 4, {-1.0f, -2.0f, 0.0f, 5.0f});
    REQUIRE(greedy_sample(logits) == 3);
}

TEST_CASE("greedy_sample — wrong dtype throws", "[sampler]") {
    Tensor t({4}, DType::Int32);
    REQUIRE_THROWS_AS(greedy_sample(t), std::runtime_error);
}

// ── temperature_sample ────────────────────────────────────────────────────────

TEST_CASE("temperature_sample — very low temperature approaches greedy", "[sampler]") {
    // With temp=0.01 the distribution is extremely peaked; the argmax token
    // should be selected for the overwhelming majority of draws.
    Tensor logits = make_logits_1d({0.0f, 0.0f, 10.0f, 0.0f});
    std::mt19937 rng(42);

    int wins = 0;
    for (int i = 0; i < 100; ++i)
        if (temperature_sample(logits, 0.01f, rng) == 2) ++wins;

    REQUIRE(wins == 100);
}

TEST_CASE("temperature_sample — non-positive temperature throws", "[sampler]") {
    Tensor logits = make_logits_1d({1.0f, 2.0f, 3.0f});
    std::mt19937 rng(1);
    REQUIRE_THROWS_AS(temperature_sample(logits, 0.0f, rng), std::runtime_error);
    REQUIRE_THROWS_AS(temperature_sample(logits, -1.0f, rng), std::runtime_error);
}

TEST_CASE("temperature_sample — output always within vocab range", "[sampler]") {
    Tensor logits = make_logits_1d({1.0f, 2.0f, 3.0f, 4.0f});
    std::mt19937 rng(99);
    for (int i = 0; i < 200; ++i) {
        int32_t tok = temperature_sample(logits, 1.0f, rng);
        REQUIRE(tok >= 0);
        REQUIRE(tok < 4);
    }
}

// ── top_k_sample ──────────────────────────────────────────────────────────────

TEST_CASE("top_k_sample — k=1 always picks the max", "[sampler]") {
    // k=1 collapses to greedy.
    Tensor logits = make_logits_1d({1.0f, 9.0f, 2.0f, 3.0f});
    std::mt19937 rng(7);
    for (int i = 0; i < 50; ++i)
        REQUIRE(top_k_sample(logits, 1, 1.0f, rng) == 1);
}

TEST_CASE("top_k_sample — never samples below the top-k threshold", "[sampler]") {
    // Logits: indices 0,1 are clearly highest; 2,3 are suppressed.
    // With k=2, tokens 2 and 3 should never appear.
    Tensor logits = make_logits_1d({5.0f, 4.0f, -10.0f, -10.0f});
    std::mt19937 rng(13);
    for (int i = 0; i < 200; ++i) {
        int32_t tok = top_k_sample(logits, 2, 1.0f, rng);
        REQUIRE((tok == 0 || tok == 1));
    }
}

TEST_CASE("top_k_sample — k < 1 throws", "[sampler]") {
    Tensor logits = make_logits_1d({1.0f, 2.0f});
    std::mt19937 rng(0);
    REQUIRE_THROWS_AS(top_k_sample(logits, 0, 1.0f, rng), std::runtime_error);
}

// ── top_p_sample ──────────────────────────────────────────────────────────────

TEST_CASE("top_p_sample — p=1.0 uses the full distribution", "[sampler]") {
    // With p=1.0 all tokens are eligible; just verify no crash and valid range.
    Tensor logits = make_logits_1d({1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
    std::mt19937 rng(21);
    for (int i = 0; i < 200; ++i) {
        int32_t tok = top_p_sample(logits, 1.0f, 1.0f, rng);
        REQUIRE(tok >= 0);
        REQUIRE(tok < 5);
    }
}

TEST_CASE("top_p_sample — very small p selects only the top token", "[sampler]") {
    // The highest-prob token alone exceeds p=0.01, so it must always be picked.
    Tensor logits = make_logits_1d({0.0f, 0.0f, 0.0f, 50.0f});
    std::mt19937 rng(55);
    for (int i = 0; i < 100; ++i)
        REQUIRE(top_p_sample(logits, 0.01f, 1.0f, rng) == 3);
}

TEST_CASE("top_p_sample — out-of-range p throws", "[sampler]") {
    Tensor logits = make_logits_1d({1.0f, 2.0f});
    std::mt19937 rng(0);
    REQUIRE_THROWS_AS(top_p_sample(logits, 0.0f, 1.0f, rng), std::runtime_error);
    REQUIRE_THROWS_AS(top_p_sample(logits, 1.1f, 1.0f, rng), std::runtime_error);
}

// ── generate ─────────────────────────────────────────────────────────────────

TEST_CASE("generate — output length is prompt + max_new", "[sampler]") {
    // Tiny model: V=8, D=8, 1 head, 1 layer — just needs to run without crashing.
    ModernGPT model(/*vocab_size=*/8, /*embed_dim=*/8,
                    /*n_heads=*/1, /*n_kv_heads=*/1, /*n_layers=*/1,
                    /*d_ff=*/0, /*n_mtp_heads=*/0, /*seed=*/7);

    std::vector<int32_t> prompt = {0, 1, 2};
    std::mt19937 rng(42);
    SamplingConfig cfg;
    cfg.mode = SamplingMode::Greedy;

    auto out = generate(model, prompt, 4, cfg, rng);

    REQUIRE(out.size() == 7u);   // 3 prompt + 4 generated
    // Prompt tokens are preserved at the front.
    REQUIRE(out[0] == 0);
    REQUIRE(out[1] == 1);
    REQUIRE(out[2] == 2);
}

TEST_CASE("generate — all generated tokens are within vocab range", "[sampler]") {
    constexpr int64_t V = 16;
    ModernGPT model(V, 16, 2, 1, 1, 0, 0, /*seed=*/3);

    std::vector<int32_t> prompt = {0};
    std::mt19937 rng(0);
    SamplingConfig cfg;
    cfg.mode        = SamplingMode::TopP;
    cfg.top_p       = 0.9f;
    cfg.temperature = 0.8f;

    auto out = generate(model, prompt, 10, cfg, rng);

    REQUIRE(out.size() == 11u);
    for (int32_t tok : out) {
        REQUIRE(tok >= 0);
        REQUIRE(tok < static_cast<int32_t>(V));
    }
}

TEST_CASE("generate — empty prompt throws", "[sampler]") {
    ModernGPT model(8, 8, 1, 1, 1, 0, 0, 1);
    std::mt19937 rng(0);
    SamplingConfig cfg;
    REQUIRE_THROWS_AS(generate(model, {}, 1, cfg, rng), std::runtime_error);
}

TEST_CASE("generate — max_new < 1 throws", "[sampler]") {
    ModernGPT model(8, 8, 1, 1, 1, 0, 0, 1);
    std::mt19937 rng(0);
    SamplingConfig cfg;
    REQUIRE_THROWS_AS(generate(model, {0}, 0, cfg, rng), std::runtime_error);
}
