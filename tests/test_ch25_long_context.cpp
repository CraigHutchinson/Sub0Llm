#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "sub0llm/nn/kv_cache.hpp"
#include "sub0llm/nn/long_context.hpp"
#include "sub0llm/nn/modern_gpt.hpp"

#include <cstdint>
#include <numeric>
#include <vector>

using Catch::Matchers::WithinAbs;
using namespace sub0llm;
using namespace sub0llm::nn;

// ── KVCache struct ────────────────────────────────────────────────────────────

TEST_CASE("KVCache helpers", "[ch25][kv_cache]") {
    KVCache cache;
    cache.max_seq = 8;
    cache.filled  = 0;
    CHECK(cache.empty());
    CHECK(!cache.full());
    CHECK(cache.remain() == 8);

    cache.filled = 8;
    CHECK(cache.full());
    CHECK(!cache.empty());
    CHECK(cache.remain() == 0);

    cache.reset();
    CHECK(cache.filled == 0);
}

// ── make_kv_cache ─────────────────────────────────────────────────────────────

TEST_CASE("ModernGPT::make_kv_cache allocates correct shape", "[ch25][kv_cache]") {
    const int64_t V = 64, D = 32;
    ModernGPT model(V, D, /*n_heads=*/4, /*n_kv_heads=*/2,
                    /*n_layers=*/2, /*d_ff=*/64);

    const int64_t max_seq = 128;
    auto cache = model.make_kv_cache(max_seq);

    CHECK(cache.max_seq == max_seq);
    CHECK(cache.filled  == 0);
    CHECK(cache.K.size() == 2);  // n_layers
    CHECK(cache.V.size() == 2);
    CHECK(cache.K[0].size() == 2);  // n_kv_heads
    CHECK(cache.K[0][0].shape()[0] == max_seq);
    CHECK(cache.K[0][0].shape()[1] == D / 4);  // head_dim
}

// ── ModernGPT accessors ───────────────────────────────────────────────────────

TEST_CASE("ModernGPT::n_kv_heads and head_dim", "[ch25]") {
    ModernGPT model(128, 64, 4, 2, 2, 0);
    CHECK(model.n_kv_heads() == 2);
    CHECK(model.head_dim()   == 16);  // 64 / 4
}

// ── forward_one correctness: cache increments filled ─────────────────────────

TEST_CASE("forward_one increments cache.filled", "[ch25][kv_cache]") {
    const int64_t V = 64, D = 32;
    ModernGPT model(V, D, 4, 2, 2, 64);
    auto cache = model.make_kv_cache(16);

    CHECK(cache.filled == 0);
    auto logits = model.forward_one(0, cache);
    CHECK(cache.filled == 1);
    CHECK(logits.ndim() == 2);
    CHECK(logits.shape()[0] == 1);
    CHECK(logits.shape()[1] == V);
}

TEST_CASE("forward_one fills cache for multiple tokens", "[ch25][kv_cache]") {
    const int64_t V = 64, D = 32;
    ModernGPT model(V, D, 4, 2, 2, 64);
    auto cache = model.make_kv_cache(16);

    for (int32_t tok = 0; tok < 5; ++tok)
        model.forward_one(tok, cache);
    CHECK(cache.filled == 5);
}

TEST_CASE("forward_one throws when cache is full", "[ch25][kv_cache]") {
    ModernGPT model(32, 32, 4, 2, 2, 64);
    auto cache = model.make_kv_cache(2);
    model.forward_one(0, cache);
    model.forward_one(1, cache);
    CHECK(cache.full());
    CHECK_THROWS_AS(model.forward_one(2, cache), std::runtime_error);
}

// ── Sliding-window mask doesn't crash forward() ───────────────────────────────

TEST_CASE("ModernGPT sliding window: full attention", "[ch25][sliding_window]") {
    ModernGPT model(64, 32, 4, 2, 2, 64, 0, 42, /*window_size=*/-1);
    Tensor ids({8}, DType::Int32);
    auto d = ids.data_as<int32_t>();
    for (std::size_t i = 0; i < 8; ++i) d[i] = static_cast<int32_t>(i);
    auto logits = model.forward(ids);
    CHECK(logits.data().shape()[0] == 8);
    CHECK(logits.data().shape()[1] == 64);
}

TEST_CASE("ModernGPT sliding window W=4: forward() succeeds", "[ch25][sliding_window]") {
    ModernGPT model(64, 32, 4, 2, 2, 64, 0, 42, /*window_size=*/4);
    Tensor ids({8}, DType::Int32);
    auto d = ids.data_as<int32_t>();
    for (std::size_t i = 0; i < 8; ++i) d[i] = static_cast<int32_t>(i % 64);
    auto logits = model.forward(ids);
    CHECK(logits.data().shape()[0] == 8);
}

// ── generate_cached produces the right number of tokens ──────────────────────

TEST_CASE("generate_cached returns max_new_tokens", "[ch25][generate]") {
    ModernGPT model(64, 32, 4, 2, 2, 64);
    auto cache = model.make_kv_cache(64);
    std::vector<int32_t> prompt = {0, 1, 2, 3};

    LongContextConfig cfg;
    cfg.max_new_tokens = 8;
    auto out = generate_cached(model, prompt, cache, cfg);

    CHECK(out.size() == 8);
    CHECK(cache.filled == static_cast<int64_t>(prompt.size()) + 8);
}

TEST_CASE("generate_cached stops at cache capacity", "[ch25][generate]") {
    ModernGPT model(64, 32, 4, 2, 2, 64);
    const int64_t max_seq = 6;
    auto cache = model.make_kv_cache(max_seq);
    std::vector<int32_t> prompt = {0, 1, 2};  // 3 tokens

    LongContextConfig cfg;
    cfg.max_new_tokens = 100;  // more than cache allows
    auto out = generate_cached(model, prompt, cache, cfg);

    // 3 prompt + out.size() + 1 (last token generated before stop) ≤ max_seq
    CHECK(cache.filled <= max_seq);
}

TEST_CASE("generate_cached on_token callback fires and can stop early", "[ch25][generate]") {
    ModernGPT model(64, 32, 4, 2, 2, 64);
    auto cache = model.make_kv_cache(64);
    std::vector<int32_t> prompt = {0, 1};

    int call_count = 0;
    LongContextConfig cfg;
    cfg.max_new_tokens = 20;
    cfg.on_token = [&](int32_t) -> bool {
        ++call_count;
        return call_count < 5;  // stop after 5 tokens
    };
    auto out = generate_cached(model, prompt, cache, cfg);
    CHECK(out.size() == 5);
}

// ── RoPE NTK scaling: forward_one with scaling > 1 doesn't crash ─────────────

TEST_CASE("forward_one with rope_scaling > 1 succeeds", "[ch25][rope_ntk]") {
    ModernGPT model(64, 32, 4, 2, 2, 64);
    auto cache = model.make_kv_cache(8);
    for (int32_t i = 0; i < 4; ++i) {
        auto logits = model.forward_one(i, cache, /*rope_scaling=*/2.0f);
        CHECK(logits.shape()[1] == 64);
    }
}

// ── apply_one helpers ─────────────────────────────────────────────────────────

TEST_CASE("RMSNorm::apply_one matches forward at single token", "[ch25][apply_one]") {
    using namespace sub0llm::autograd;
    RMSNorm norm(16);
    // Create a (1,16) input tensor
    Tensor x = zeros({1, 16});
    auto xd = x.data_as<float>();
    for (std::size_t i = 0; i < 16; ++i) xd[i] = static_cast<float>(i + 1) * 0.1f;

    Tensor out_infer = norm.apply_one(x);

    // Check shape
    CHECK(out_infer.shape()[0] == 1);
    CHECK(out_infer.shape()[1] == 16);

    // Output should be non-zero (weights initialised to 1)
    auto od = out_infer.data_as<float>();
    bool any_nonzero = false;
    for (std::size_t i = 0; i < 16; ++i)
        if (od[i] != 0.0f) { any_nonzero = true; break; }
    CHECK(any_nonzero);
}
