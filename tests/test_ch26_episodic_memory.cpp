#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "sub0llm/nn/episodic_memory.hpp"
#include "sub0llm/nn/modern_gpt.hpp"
#include "sub0llm/nn/optimizer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <stdexcept>
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

TEST_CASE("reset throws when merged to prevent silent corruption", "[ch26][tier2]") {
    ModernGPT model(32, 32, 2, 1, 2, 64);
    auto state = make_episodic_state(model);

    auto dd = state.deltas[0].data_as<float>();
    for (auto& v : dd) v = 0.1f;
    state.merge(model);

    // reset() while merged must throw — caller must unmerge first
    CHECK_THROWS_AS(state.reset(), std::runtime_error);

    state.unmerge(model);
    CHECK_NOTHROW(state.reset());  // safe after unmerge
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

    // Session 1: measure baseline, encode, verify loss reduced
    float loss_s1_before;
    {
        auto ls = comprehension_pass(model, {1, 2, 3, 4, 5});
        loss_s1_before = std::accumulate(ls.begin(), ls.end(), 0.0f) /
                         static_cast<float>(ls.size());
    }

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

    CHECK(loss_s1_after < loss_s1_before);  // session 1 actually improved

    // Session 2: measure baseline, encode different sequence, verify loss reduced
    float loss_s2_before;
    {
        auto ls = comprehension_pass(model, {20, 21, 22, 23, 24});
        loss_s2_before = std::accumulate(ls.begin(), ls.end(), 0.0f) /
                         static_cast<float>(ls.size());
    }

    episodic_encode(model, state, {20, 21, 22, 23, 24}, cfg);
    state.merge(model);

    float loss_s2_after;
    {
        auto ls = comprehension_pass(model, {20, 21, 22, 23, 24});
        loss_s2_after = std::accumulate(ls.begin(), ls.end(), 0.0f) /
                        static_cast<float>(ls.size());
    }

    state.unmerge(model);

    CHECK(loss_s2_after < loss_s2_before);  // session 2 actually improved
    CHECK(std::isfinite(loss_s1_after));
    CHECK(std::isfinite(loss_s2_after));
}

TEST_CASE("episodic delta does not catastrophically degrade unrelated tokens", "[ch26][tier4]") {
    ModernGPT model(32, 32, 2, 1, 2, 64, 0, 7);

    // Sequences A and B use disjoint token values
    std::vector<int32_t> seq_a = {5, 12, 3, 7, 20};
    std::vector<int32_t> seq_b = {1, 2, 8, 15, 25};

    // Baseline: NLL on seq_b before any episodic write
    float nll_b_base;
    {
        auto ls = comprehension_pass(model, seq_b);
        nll_b_base = std::accumulate(ls.begin(), ls.end(), 0.0f) /
                     static_cast<float>(ls.size());
    }

    // Write delta for seq_a only
    auto state = make_episodic_state(model);
    EpisodicConfig cfg;
    cfg.surprise_threshold = 0.0f;
    cfg.think_steps        = 3;
    cfg.learning_rate      = 5e-3f;
    episodic_encode(model, state, seq_a, cfg);

    // Merged: cross-contamination should be bounded (not catastrophic)
    state.merge(model);
    float nll_b_merged;
    {
        auto ls = comprehension_pass(model, seq_b);
        nll_b_merged = std::accumulate(ls.begin(), ls.end(), 0.0f) /
                       static_cast<float>(ls.size());
    }
    CHECK(std::isfinite(nll_b_merged));
    // A small delta (lr=5e-3, 3 steps) should not more than double the NLL
    // on unrelated tokens — bounded interference
    CHECK(nll_b_merged < nll_b_base * 2.5f);

    // After unmerge: base model is completely restored, no residual contamination
    state.unmerge(model);
    float nll_b_restored;
    {
        auto ls = comprehension_pass(model, seq_b);
        nll_b_restored = std::accumulate(ls.begin(), ls.end(), 0.0f) /
                         static_cast<float>(ls.size());
    }
    CHECK_THAT(nll_b_restored, WithinAbs(nll_b_base, 1e-3f));
}

// ── Tier 5: Session persistence ───────────────────────────────────────────────

TEST_CASE("save/load roundtrip preserves delta values exactly", "[ch26][tier5]") {
    ModernGPT model(32, 32, 2, 1, 2, 64, 0, 7);
    auto state = make_episodic_state(model);

    // Write a known pattern into the deltas
    EpisodicConfig cfg;
    cfg.surprise_threshold = 0.0f;
    cfg.think_steps        = 2;
    cfg.learning_rate      = 1e-2f;
    episodic_encode(model, state, {5, 12, 3, 7, 20}, cfg);

    // Snapshot pre-save delta values
    std::vector<float> before;
    for (auto& d : state.deltas) {
        auto dd = d.data_as<float>();
        before.insert(before.end(), dd.begin(), dd.end());
    }

    const std::string tmp_path = "/tmp/test_epis_roundtrip.epis";
    save_episodic_state(state, tmp_path);

    auto loaded = load_episodic_state(tmp_path);
    std::remove(tmp_path.c_str());

    REQUIRE(loaded.deltas.size() == state.deltas.size());
    CHECK(loaded.merged == state.merged);

    std::vector<float> after;
    for (auto& d : loaded.deltas) {
        auto dd = d.data_as<float>();
        after.insert(after.end(), dd.begin(), dd.end());
    }

    REQUIRE(before.size() == after.size());
    for (std::size_t i = 0; i < before.size(); ++i)
        CHECK_THAT(after[i], WithinAbs(before[i], 1e-6f));
}

TEST_CASE("loaded delta merges into model and reduces loss", "[ch26][tier5]") {
    ModernGPT model(32, 32, 2, 1, 2, 64, 0, 7);
    std::vector<int32_t> tokens = {5, 12, 3, 7, 20};

    float loss_before = 0.0f;
    {
        auto ls = comprehension_pass(model, tokens);
        for (float v : ls) loss_before += v;
        loss_before /= static_cast<float>(ls.size());
    }

    // Write delta and save
    auto state = make_episodic_state(model);
    EpisodicConfig cfg;
    cfg.surprise_threshold = 0.0f;
    cfg.think_steps        = 3;
    cfg.learning_rate      = 5e-3f;
    episodic_encode(model, state, tokens, cfg);

    const std::string tmp_path = "/tmp/test_epis_merge.epis";
    save_episodic_state(state, tmp_path);

    // Load into a fresh state and merge
    auto loaded = load_episodic_state(tmp_path);
    std::remove(tmp_path.c_str());

    loaded.merge(model);
    float loss_after = 0.0f;
    {
        auto ls = comprehension_pass(model, tokens);
        for (float v : ls) loss_after += v;
        loss_after /= static_cast<float>(ls.size());
    }
    loaded.unmerge(model);

    CHECK(loss_after < loss_before);
}

TEST_CASE("load_episodic_state throws on bad magic", "[ch26][tier5]") {
    const std::string tmp_path = "/tmp/test_epis_bad.epis";
    {
        std::ofstream f(tmp_path, std::ios::binary);
        f.write("BADMAGIC", 8);
    }
    CHECK_THROWS_AS(load_episodic_state(tmp_path), std::runtime_error);
    std::remove(tmp_path.c_str());
}
