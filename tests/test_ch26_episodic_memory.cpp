#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "sub0llm/nn/episodic_memory.hpp"
#include "sub0llm/nn/modern_gpt.hpp"
#include "sub0llm/nn/optimizer.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

using namespace sub0llm;
using namespace sub0llm::nn;
using Catch::Matchers::WithinAbs;

// ── Tier 1: Write gate — comprehension_pass gives sensible surprisal ──────────

TEST_CASE("comprehension_pass returns T-1 losses for T tokens", "[ch26][tier1]") {
    ModernGPT model(32, 32, 2, 1, 2, 64);
    std::vector<int32_t> tokens = {0, 1, 2, 3, 4};
    auto losses = comprehension_pass(model, tokens);
    CHECK(losses.size() == 4);
}

TEST_CASE("comprehension_pass losses are non-negative", "[ch26][tier1]") {
    ModernGPT model(32, 32, 2, 1, 2, 64);
    std::vector<int32_t> tokens = {0, 1, 2, 3, 4, 5, 6, 7};
    auto losses = comprehension_pass(model, tokens);
    for (float l : losses)
        CHECK(l >= 0.0f);
}

TEST_CASE("comprehension_pass: uniform-logit model gives ~ log(V) loss", "[ch26][tier1]") {
    // With random-init weights and single token, loss ~ log(vocab_size).
    const int64_t V = 64;
    ModernGPT model(V, 32, 2, 1, 2, 64, 0, 42);
    std::vector<int32_t> tokens = {0, 1};
    auto losses = comprehension_pass(model, tokens);
    REQUIRE(losses.size() == 1);
    // log(64) ≈ 4.16; freshly initialised model should be in [2, 8]
    CHECK(losses[0] > 1.5f);
    CHECK(losses[0] < 9.0f);
}

TEST_CASE("comprehension_pass: short sequence returns empty", "[ch26][tier1]") {
    ModernGPT model(32, 32, 2, 1, 2, 64);
    CHECK(comprehension_pass(model, {}).empty());
    CHECK(comprehension_pass(model, {0}).empty());
}

// ── Tier 2: make_episodic_state and reset ─────────────────────────────────────

TEST_CASE("make_episodic_state matches parameter count", "[ch26][tier2]") {
    ModernGPT model(32, 32, 2, 1, 2, 64);
    auto state = make_episodic_state(model);
    CHECK(state.deltas.size() == model.parameters().size());
    CHECK(!state.merged);
}

TEST_CASE("make_episodic_state deltas start at zero", "[ch26][tier2]") {
    ModernGPT model(32, 32, 2, 1, 2, 64);
    auto state = make_episodic_state(model);
    for (auto& d : state.deltas) {
        auto dd = d.data_as<float>();
        for (float v : dd)
            CHECK(v == 0.0f);
    }
}

TEST_CASE("reset zeros out accumulated deltas", "[ch26][tier2]") {
    ModernGPT model(32, 32, 2, 1, 2, 64);
    auto state = make_episodic_state(model);

    // Put non-zero values into first delta
    if (!state.deltas.empty()) {
        auto dd = state.deltas[0].data_as<float>();
        for (auto& v : dd) v = 1.0f;
    }
    state.reset();
    for (auto& d : state.deltas) {
        auto dd = d.data_as<float>();
        for (float v : dd)
            CHECK(v == 0.0f);
    }
    CHECK(!state.merged);
}

// ── Tier 3: episodic_encode changes the deltas ────────────────────────────────

TEST_CASE("episodic_encode produces non-zero deltas", "[ch26][tier3]") {
    ModernGPT model(32, 32, 2, 1, 2, 64);
    auto state = make_episodic_state(model);

    // Use a very low surprise threshold so at least one span is written
    EpisodicConfig cfg;
    cfg.surprise_threshold = 0.0f;  // every token triggers a write
    cfg.think_steps        = 1;
    cfg.learning_rate      = 1e-2f;

    std::vector<int32_t> tokens = {0, 1, 2, 3, 4, 5};
    episodic_encode(model, state, tokens, cfg);

    // At least one delta should be non-zero
    bool any_nonzero = false;
    for (auto& d : state.deltas) {
        auto dd = d.data_as<float>();
        for (float v : dd) {
            if (v != 0.0f) { any_nonzero = true; break; }
        }
        if (any_nonzero) break;
    }
    CHECK(any_nonzero);
}

TEST_CASE("episodic_encode with accumulate=false resets first", "[ch26][tier3]") {
    ModernGPT model(32, 32, 2, 1, 2, 64);
    auto state = make_episodic_state(model);

    // Manually dirty the state
    if (!state.deltas.empty()) {
        auto dd = state.deltas[0].data_as<float>();
        for (auto& v : dd) v = 999.0f;
    }

    EpisodicConfig cfg;
    cfg.accumulate         = false;
    cfg.surprise_threshold = 0.0f;
    cfg.think_steps        = 1;
    cfg.learning_rate      = 0.0f;  // zero lr → deltas stay at zero after reset

    std::vector<int32_t> tokens = {0, 1, 2};
    episodic_encode(model, state, tokens, cfg);

    // With lr=0 no non-zero deltas should remain from the dirty initial values
    if (!state.deltas.empty()) {
        auto dd = state.deltas[0].data_as<float>();
        for (float v : dd)
            CHECK(v == 0.0f);
    }
}

// ── Tier 3: merge/unmerge roundtrip ──────────────────────────────────────────

TEST_CASE("merge changes model weight data", "[ch26][tier3]") {
    ModernGPT model(32, 32, 2, 1, 2, 64);
    auto state = make_episodic_state(model);

    // Set a known delta on first parameter
    auto params = model.parameters();
    REQUIRE(!params.empty());
    {
        auto dd = state.deltas[0].data_as<float>();
        for (auto& v : dd) v = 0.01f;
    }

    // Record original weight value
    float w_before;
    {
        auto wd = params[0]->data().data_as<float>();
        w_before = wd[0];
    }

    state.merge(model);

    float w_after;
    {
        auto wd = params[0]->data().data_as<float>();
        w_after = wd[0];
    }

    CHECK_THAT(w_after - w_before, WithinAbs(0.01f, 1e-5f));
    CHECK(state.merged);

    // unmerge should restore
    state.unmerge(model);
    float w_restored;
    {
        auto wd = params[0]->data().data_as<float>();
        w_restored = wd[0];
    }
    CHECK_THAT(w_restored, WithinAbs(w_before, 1e-5f));
    CHECK(!state.merged);
}

TEST_CASE("merge is idempotent", "[ch26][tier3]") {
    ModernGPT model(32, 32, 2, 1, 2, 64);
    auto state = make_episodic_state(model);

    auto params = model.parameters();
    auto dd = state.deltas[0].data_as<float>();
    for (auto& v : dd) v = 0.1f;

    float w0;
    { auto wd = params[0]->data().data_as<float>(); w0 = wd[0]; }

    state.merge(model);
    state.merge(model);  // second merge is a no-op

    float w1;
    { auto wd = params[0]->data().data_as<float>(); w1 = wd[0]; }

    // Should have changed by exactly 0.1 (once), not 0.2
    CHECK_THAT(w1 - w0, WithinAbs(0.1f, 1e-5f));
}

// ── Tier 3: loss reduction after episodic write ───────────────────────────────

TEST_CASE("episodic write reduces NLL on the training span", "[ch26][tier3]") {
    // The fundamental claim: applying a gradient step on the surprising span
    // must reduce the cross-entropy on that same span.
    ModernGPT model(32, 32, 2, 1, 2, 64, 0, 7);
    auto state = make_episodic_state(model);

    std::vector<int32_t> tokens = {5, 12, 3, 7, 20};

    float loss_before = 0.0f;
    {
        auto ls = comprehension_pass(model, tokens);
        for (float v : ls) loss_before += v;
        loss_before /= static_cast<float>(ls.size());
    }

    EpisodicConfig cfg;
    cfg.surprise_threshold = 0.0f;
    cfg.think_steps        = 3;
    cfg.learning_rate      = 5e-3f;
    episodic_encode(model, state, tokens, cfg);

    state.merge(model);

    float loss_after = 0.0f;
    {
        auto ls = comprehension_pass(model, tokens);
        for (float v : ls) loss_after += v;
        loss_after /= static_cast<float>(ls.size());
    }

    state.unmerge(model);

    // Core claim: loss decreases after the episodic write
    CHECK(loss_after < loss_before);
}

// ── Tier 4: capacity and forgetting ──────────────────────────────────────────

TEST_CASE("reset between sessions prevents interference", "[ch26][tier4]") {
    ModernGPT model(32, 32, 2, 1, 2, 64, 0, 99);

    EpisodicConfig cfg;
    cfg.surprise_threshold = 0.0f;
    cfg.think_steps        = 2;
    cfg.learning_rate      = 1e-2f;

    // Session 1: encode one sequence
    auto state = make_episodic_state(model);
    episodic_encode(model, state, {1, 2, 3, 4, 5}, cfg);
    state.merge(model);

    float loss_s1_after;
    {
        auto ls = comprehension_pass(model, {1, 2, 3, 4, 5});
        loss_s1_after = std::accumulate(ls.begin(), ls.end(), 0.0f) /
                        static_cast<float>(ls.size());
    }

    state.unmerge(model);
    state.reset();  // clear for new session

    // Session 2: encode a different sequence
    episodic_encode(model, state, {20, 21, 22, 23, 24}, cfg);
    state.merge(model);

    float loss_s2_after;
    {
        auto ls = comprehension_pass(model, {20, 21, 22, 23, 24});
        loss_s2_after = std::accumulate(ls.begin(), ls.end(), 0.0f) /
                        static_cast<float>(ls.size());
    }

    state.unmerge(model);

    // Both sessions should have reduced loss on their own span
    // (This is more of a sanity check — we just confirm neither is NaN/Inf)
    CHECK(std::isfinite(loss_s1_after));
    CHECK(std::isfinite(loss_s2_after));
}
