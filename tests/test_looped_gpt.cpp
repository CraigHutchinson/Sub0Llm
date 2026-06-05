#include "sub0llm/nn/looped_gpt.hpp"
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

static int64_t count_params(LoopedGPT& m) {
    int64_t n = 0;
    for (auto* v : m.parameters())
        n += static_cast<int64_t>(v->data().numel());
    return n;
}

// ── construction & accessors ──────────────────────────────────────────────────

TEST_CASE("LoopedGPT - accessors") {
    LoopedGPT m(20, 16, 2, 1, 3);
    REQUIRE(m.vocab_size() == 20);
    REQUIRE(m.embed_dim()  == 16);
    REQUIRE(m.n_loops()    == 3);
}

TEST_CASE("LoopedGPT - invalid construction throws") {
    REQUIRE_THROWS_AS(LoopedGPT(0,  16, 2, 1, 2), std::runtime_error);
    REQUIRE_THROWS_AS(LoopedGPT(20,  0, 2, 1, 2), std::runtime_error);
    REQUIRE_THROWS_AS(LoopedGPT(20, 16, 2, 1, 0), std::runtime_error);
}

// ── forward shape ─────────────────────────────────────────────────────────────

TEST_CASE("LoopedGPT - forward output shape (T,V)") {
    LoopedGPT m(20, 16, 2, 1, 2, 0, 1);
    auto ids    = make_ids({0, 1, 2, 3});
    auto logits = m.forward(ids);
    REQUIRE(logits.data().ndim() == 2);
    REQUIRE(logits.data().shape(0) == 4);
    REQUIRE(logits.data().shape(1) == 20);
}

TEST_CASE("LoopedGPT - forward single token") {
    LoopedGPT m(10, 8, 2, 1, 1, 0, 2);
    auto ids    = make_ids({3});
    auto logits = m.forward(ids);
    REQUIRE(logits.data().shape(0) == 1);
    REQUIRE(logits.data().shape(1) == 10);
}

// ── forward_k ────────────────────────────────────────────────────────────────

TEST_CASE("LoopedGPT - forward_k output shape matches forward") {
    LoopedGPT m(20, 16, 2, 1, 3, 0, 5);
    auto ids = make_ids({0, 1, 2});
    auto l3  = m.forward(ids);            // uses n_loops=3
    auto lk3 = m.forward_k(ids, 3);      // explicit 3
    auto sp3  = l3.data().data_as<float>();
    auto spk3 = lk3.data().data_as<float>();
    REQUIRE(l3.data().numel() == lk3.data().numel());
    for (std::size_t i = 0; i < static_cast<std::size_t>(l3.data().numel()); ++i)
        REQUIRE(sp3[i] == Catch::Approx(spk3[i]));
}

TEST_CASE("LoopedGPT - forward_k k=0 throws") {
    LoopedGPT m(10, 8, 2, 1, 2);
    auto ids = make_ids({0});
    REQUIRE_THROWS_AS(m.forward_k(ids, 0), std::runtime_error);
}

TEST_CASE("LoopedGPT - forward_k different k produces different logits") {
    // With different loop counts the hidden state refines differently.
    LoopedGPT m(10, 8, 2, 1, 2, 0, 7);
    auto ids = make_ids({0, 1});
    auto l1 = m.forward_k(ids, 1);
    auto l4 = m.forward_k(ids, 4);
    bool any_diff = false;
    auto sp1 = l1.data().data_as<float>();
    auto sp4 = l4.data().data_as<float>();
    for (std::size_t i = 0; i < static_cast<std::size_t>(l1.data().numel()); ++i)
        if (sp1[i] != sp4[i]) { any_diff = true; break; }
    REQUIRE(any_diff);
}

// ── parameter count ──────────────────────────────────────────────────────────

TEST_CASE("LoopedGPT - fewer parameters than equivalent multi-layer ModernGPT") {
    // K-loop LoopedGPT has 1 block; K-layer ModernGPT has K blocks.
    const int64_t V = 20, D = 16, K = 4;
    LoopedGPT looped(V, D, 2, 1, K);
    ModernGPT  stacked(V, D, 2, 1, K);

    int64_t p_looped  = count_params(looped);
    auto stacked_p    = stacked.parameters();
    int64_t p_stacked = 0;
    for (auto* v : stacked_p) p_stacked += static_cast<int64_t>(v->data().numel());

    REQUIRE(p_looped < p_stacked);
}

// ── gradient flow ─────────────────────────────────────────────────────────────

TEST_CASE("LoopedGPT - backward populates gradients") {
    LoopedGPT m(10, 8, 2, 1, 2, 0, 3);
    Tensor targets({3}, DType::Int32);
    auto tgt = targets.data_as<int32_t>();
    tgt[0] = 1; tgt[1] = 2; tgt[2] = 3;

    auto ids    = make_ids({0, 1, 2, 3});
    auto logits = m.forward(ids);
    auto ltrunc = narrow(logits, 0, 3);
    auto loss   = cross_entropy(ltrunc, targets);
    loss.backward();

    bool any_grad = false;
    for (auto* v : m.parameters()) {
        if (v->grad().numel() == 0) continue;
        auto gs = v->grad().data_as<float>();
        for (std::size_t i = 0; i < static_cast<std::size_t>(v->grad().numel()); ++i)
            if (gs[i] != 0.0f) { any_grad = true; break; }
        if (any_grad) break;
    }
    REQUIRE(any_grad);
}

// ── training step ─────────────────────────────────────────────────────────────

TEST_CASE("LoopedGPT - loss decreases over 50 training steps") {
    const int64_t V = 20, T = 8;
    LoopedGPT m(V, 16, 2, 1, 2, 0, 99);
    Adam adam(m.parameters(), 5e-3f);

    Tensor ids({T}, DType::Int32);
    Tensor tgt({T - 1}, DType::Int32);
    auto id_sp = ids.data_as<int32_t>();
    auto tg_sp = tgt.data_as<int32_t>();
    for (int64_t i = 0; i < T;     ++i) id_sp[static_cast<std::size_t>(i)] = static_cast<int32_t>(i % V);
    for (int64_t i = 0; i < T - 1; ++i) tg_sp[static_cast<std::size_t>(i)] = static_cast<int32_t>((i + 1) % V);

    float first_loss = -1.f, last_loss = -1.f;
    for (int step = 0; step < 50; ++step) {
        adam.zero_grad();
        auto logits = m.forward(ids);
        auto ltrunc = narrow(logits, 0, T - 1);
        auto loss   = cross_entropy(ltrunc, tgt);
        loss.backward();
        adam.step();
        float lv = loss.data().data_as<float>()[0];
        if (step == 0) first_loss = lv;
        last_loss = lv;
    }
    REQUIRE(last_loss < first_loss);
}

// ── input validation ──────────────────────────────────────────────────────────

TEST_CASE("LoopedGPT - forward rejects 2D token tensor") {
    LoopedGPT m(10, 8, 2, 1, 1);
    Tensor bad({2, 3}, DType::Int32);
    REQUIRE_THROWS_AS(m.forward(bad), std::runtime_error);
}

TEST_CASE("LoopedGPT - forward rejects empty token tensor") {
    LoopedGPT m(10, 8, 2, 1, 1);
    Tensor empty({0}, DType::Int32);
    REQUIRE_THROWS_AS(m.forward(empty), std::runtime_error);
}
