#include "sub0llm/nn/mtp.hpp"
#include "sub0llm/nn/optimizer.hpp"
#include "sub0llm/autograd/ops.hpp"
#include "sub0llm/core/tensor.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

using namespace sub0llm;
using namespace sub0llm::nn;
using namespace sub0llm::autograd;

// ── helpers ───────────────────────────────────────────────────────────────────

static Tensor make_ids(std::initializer_list<int32_t> toks) {
    Tensor t({static_cast<int64_t>(toks.size())}, DType::Int32);
    auto sp = t.data_as<int32_t>();
    std::size_t i = 0;
    for (auto v : toks) sp[i++] = v;
    return t;
}

static Tensor make_ids_seq(int64_t len, int64_t vocab_size) {
    Tensor t({len}, DType::Int32);
    auto sp = t.data_as<int32_t>();
    for (int64_t i = 0; i < len; ++i)
        sp[static_cast<std::size_t>(i)] = static_cast<int32_t>(i % vocab_size);
    return t;
}

// ── mtp_train_loss ────────────────────────────────────────────────────────────

// ModernGPT(vocab_size, embed_dim, n_heads, n_kv_heads, n_layers, d_ff=0, n_mtp_heads=0, seed=42)

TEST_CASE("mtp_train_loss — rejects model with no MTP heads") {
    ModernGPT m(20, 16, 2, 1, 2);  // n_mtp_heads defaults to 0
    auto ids = make_ids({0, 1, 2, 3, 4});
    REQUIRE_THROWS_AS(mtp_train_loss(m, ids, 5), std::runtime_error);
}

TEST_CASE("mtp_train_loss — rejects 2D token_ids") {
    ModernGPT m(20, 16, 2, 1, 2, 0, 1 /*n_mtp_heads=1*/);
    Tensor ids2d({3, 4}, DType::Int32);
    REQUIRE_THROWS_AS(mtp_train_loss(m, ids2d, 12), std::runtime_error);
}

TEST_CASE("mtp_train_loss — rejects mismatched seq_len") {
    ModernGPT m(20, 16, 2, 1, 2, 0, 1 /*n_mtp_heads=1*/);
    auto ids = make_ids({0, 1, 2, 3, 4});  // T=5
    REQUIRE_THROWS_AS(mtp_train_loss(m, ids, 4), std::runtime_error);  // seq_len=4 != T
}

TEST_CASE("mtp_train_loss — rejects seq too short (T < 2+K)") {
    ModernGPT m(20, 16, 2, 1, 2, 0, 2 /*n_mtp_heads=2*/);
    // Need T >= 4, provide T=3
    auto ids = make_ids({0, 1, 2});
    REQUIRE_THROWS_AS(mtp_train_loss(m, ids, 3), std::runtime_error);
}

TEST_CASE("mtp_train_loss — rejects invalid aux_weight") {
    ModernGPT m(20, 16, 2, 1, 2, 0, 1 /*n_mtp_heads=1*/);
    auto ids = make_ids({0, 1, 2, 3, 4});
    REQUIRE_THROWS_AS(mtp_train_loss(m, ids, 5, -0.1f), std::runtime_error);
    REQUIRE_THROWS_AS(mtp_train_loss(m, ids, 5,  1.1f), std::runtime_error);
}

TEST_CASE("mtp_train_loss — returns scalar Variable") {
    ModernGPT m(20, 16, 2, 1, 2, 0, 1 /*n_mtp_heads=1*/);
    auto ids = make_ids_seq(6, 20);
    auto loss = mtp_train_loss(m, ids, 6, 0.1f);
    REQUIRE(loss.data().numel() == 1);
    REQUIRE(loss.data().data_as<float>()[0] > 0.f);
}

TEST_CASE("mtp_train_loss — loss is finite and positive") {
    ModernGPT m(20, 16, 2, 1, 2, 0, 2 /*n_mtp_heads=2*/);
    auto ids = make_ids_seq(8, 20);
    auto loss = mtp_train_loss(m, ids, 8, 0.1f);
    float lv = loss.data().data_as<float>()[0];
    REQUIRE(std::isfinite(lv));
    REQUIRE(lv > 0.f);
}

TEST_CASE("mtp_train_loss — gradient flows to model parameters") {
    ModernGPT m(20, 16, 2, 1, 2, 0, 1 /*n_mtp_heads=1*/);
    auto ids = make_ids_seq(6, 20);
    auto loss = mtp_train_loss(m, ids, 6, 0.1f);
    loss.backward();

    bool any_grad = false;
    for (auto* v : m.parameters()) {
        if (v->grad().numel() == 0) continue;
        for (auto f : v->grad().data_as<float>())
            if (f != 0.f) { any_grad = true; break; }
        if (any_grad) break;
    }
    REQUIRE(any_grad);
}

TEST_CASE("mtp_train_loss — aux_weight=0 gives same loss as aux_weight with all zeros auxiliary heads") {
    // aux_weight=0 means only the main head (k=0) contributes.
    ModernGPT m1(20, 16, 2, 1, 2, 0, 1, 42 /*n_mtp_heads=1*/);
    ModernGPT m2(20, 16, 2, 1, 2, 0, 1, 42 /*n_mtp_heads=1*/);
    auto ids = make_ids_seq(6, 20);
    auto loss0 = mtp_train_loss(m1, ids, 6, 0.0f);
    auto loss1 = mtp_train_loss(m2, ids, 6, 0.1f);
    // With same weights, aux_weight=0 loss should be <= aux_weight=0.1 loss
    // (main CE is same, aux adds positive term)
    REQUIRE(loss0.data().data_as<float>()[0] <= loss1.data().data_as<float>()[0] + 1e-4f);
}

TEST_CASE("mtp_train_loss — training step reduces loss (K=1)") {
    const int64_t V = 20, T = 8;
    ModernGPT m(V, 16, 2, 1, 1, 0, 1 /*n_mtp_heads=1*/, 42);
    Adam adam(m.parameters(), 5e-3f);

    auto ids = make_ids_seq(T, V);

    float first = -1.f, last = -1.f;
    for (int step = 0; step < 40; ++step) {
        adam.zero_grad();
        auto loss = mtp_train_loss(m, ids, T, 0.1f);
        loss.backward();
        adam.step();
        float lv = loss.data().data_as<float>()[0];
        if (step == 0) first = lv;
        last = lv;
    }
    REQUIRE(last < first);
}

// ── mtp_generate ─────────────────────────────────────────────────────────────

TEST_CASE("mtp_generate — rejects empty prompt") {
    ModernGPT m(20, 16, 2, 1, 2, 0, 1 /*n_mtp_heads=1*/);
    std::mt19937 rng(42);
    SamplingConfig cfg;
    REQUIRE_THROWS_AS(mtp_generate(m, {}, 5, cfg, rng), std::runtime_error);
}

TEST_CASE("mtp_generate — rejects max_new_tokens < 1") {
    ModernGPT m(20, 16, 2, 1, 2, 0, 1 /*n_mtp_heads=1*/);
    std::mt19937 rng(42);
    SamplingConfig cfg;
    REQUIRE_THROWS_AS(mtp_generate(m, {0, 1, 2}, 0, cfg, rng), std::runtime_error);
}

TEST_CASE("mtp_generate — produces correct number of tokens (K=0 fallback)") {
    ModernGPT m(20, 16, 2, 1, 2);  // n_mtp_heads=0
    std::mt19937 rng(42);
    SamplingConfig cfg{SamplingMode::Greedy};
    auto result = mtp_generate(m, {0, 1, 2}, 5, cfg, rng);
    REQUIRE(static_cast<int64_t>(result.size()) == 3 + 5);
}

TEST_CASE("mtp_generate — produces correct number of tokens (K=2)") {
    ModernGPT m(20, 16, 2, 1, 2, 0, 2 /*n_mtp_heads=2*/);
    std::mt19937 rng(42);
    SamplingConfig cfg{SamplingMode::Greedy};
    auto result = mtp_generate(m, {0, 1, 2}, 7, cfg, rng);
    REQUIRE(static_cast<int64_t>(result.size()) == 3 + 7);
}

TEST_CASE("mtp_generate — prompt tokens preserved at front") {
    ModernGPT m(20, 16, 2, 1, 2, 0, 1 /*n_mtp_heads=1*/);
    std::mt19937 rng(42);
    SamplingConfig cfg{SamplingMode::Greedy};
    std::vector<int32_t> prompt = {3, 7, 11};
    auto result = mtp_generate(m, prompt, 4, cfg, rng);
    REQUIRE(result[0] == 3);
    REQUIRE(result[1] == 7);
    REQUIRE(result[2] == 11);
}

// ── mtp_generate_stats ────────────────────────────────────────────────────────

TEST_CASE("mtp_generate_stats — K=0 uses 1 pass per token") {
    ModernGPT m(20, 16, 2, 1, 2);  // n_mtp_heads=0
    std::mt19937 rng(42);
    SamplingConfig cfg{SamplingMode::Greedy};
    auto stats = mtp_generate_stats(m, {0, 1}, 4, cfg, rng);
    REQUIRE(stats.n_forward_passes == 4);
    REQUIRE(stats.tokens_per_pass == Catch::Approx(1.0));
}

TEST_CASE("mtp_generate_stats — K=2 uses fewer passes than tokens") {
    ModernGPT m(20, 16, 2, 1, 2, 0, 2 /*n_mtp_heads=2, 3 tokens/pass*/);
    std::mt19937 rng(42);
    SamplingConfig cfg{SamplingMode::Greedy};
    // 9 new tokens with K=2 → ceil(9/3)=3 passes
    auto stats = mtp_generate_stats(m, {0, 1}, 9, cfg, rng);
    REQUIRE(stats.n_forward_passes == 3);
    REQUIRE(stats.tokens_per_pass == Catch::Approx(3.0));
    REQUIRE(static_cast<int64_t>(stats.tokens.size()) == 2 + 9);
}

TEST_CASE("mtp_generate_stats — tokens_per_pass > 1 for K > 0") {
    ModernGPT m(20, 16, 2, 1, 2, 0, 1 /*n_mtp_heads=1, 2 tokens/pass*/);
    std::mt19937 rng(42);
    SamplingConfig cfg{SamplingMode::Greedy};
    auto stats = mtp_generate_stats(m, {0, 1}, 6, cfg, rng);
    REQUIRE(stats.tokens_per_pass > 1.0);
}
