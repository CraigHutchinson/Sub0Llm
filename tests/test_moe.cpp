#include "sub0llm/nn/moe.hpp"
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

static Variable make_input(int64_t T, int64_t D, float fill = 0.5f) {
    Tensor t({T, D}, DType::Float32);
    for (auto& v : t.data_as<float>()) v = fill;
    return Variable(t, true);
}

// ── row_scale (autograd op) ───────────────────────────────────────────────────

TEST_CASE("row_scale - forward correctness") {
    // x[i,j] = i+1,  v[i,0] = 2.0f  → y[i,j] = 2*(i+1)
    Tensor xt({3, 4}, DType::Float32);
    Tensor vt({3, 1}, DType::Float32);
    auto xsp = xt.data_as<float>();
    auto vsp = vt.data_as<float>();
    for (int64_t i = 0; i < 3; ++i) {
        vsp[static_cast<std::size_t>(i)] = static_cast<float>(i + 1);
        for (int64_t j = 0; j < 4; ++j)
            xsp[static_cast<std::size_t>(i * 4 + j)] = 1.0f;
    }
    Variable xv(xt, true);
    Variable vv(vt, true);
    auto y = row_scale(xv, vv);

    for (int64_t i = 0; i < 3; ++i)
        for (int64_t j = 0; j < 4; ++j)
            REQUIRE(y.data().data_as<float>()[static_cast<std::size_t>(i * 4 + j)]
                    == Catch::Approx(static_cast<float>(i + 1)));
}

TEST_CASE("row_scale - backward x-gradient") {
    // y[i,j] = x[i,j]*v[i,0]; grad_x[i,j] = upstream[i,j]*v[i,0]
    Tensor xt({2, 3}, DType::Float32);
    Tensor vt({2, 1}, DType::Float32);
    for (auto& f : xt.data_as<float>()) f = 1.0f;
    vt.data_as<float>()[0] = 2.0f;
    vt.data_as<float>()[1] = 3.0f;

    Variable xv(xt, true);
    Variable vv(vt, false);
    auto loss = sum(row_scale(xv, vv));
    loss.backward();

    // sum reduces to scalar; upstream is 1 everywhere
    // grad_x[0,*] = 1 * v[0,0] = 2
    REQUIRE(xv.grad().data_as<float>()[0] == Catch::Approx(2.0f));
    // grad_x[1,*] = 1 * v[1,0] = 3
    REQUIRE(xv.grad().data_as<float>()[3] == Catch::Approx(3.0f));
}

TEST_CASE("row_scale - backward v-gradient") {
    // grad_v[i,0] = sum_j(upstream[i,j] * x[i,j])
    Tensor xt({2, 3}, DType::Float32);
    Tensor vt({2, 1}, DType::Float32);
    for (auto& f : xt.data_as<float>()) f = 1.0f;  // x all ones
    for (auto& f : vt.data_as<float>()) f = 1.0f;  // v = 1

    Variable xv(xt, false);
    Variable vv(vt, true);
    auto loss = sum(row_scale(xv, vv));
    loss.backward();

    // grad_v[i,0] = sum_j(1 * 1) = 3 (D=3 columns)
    REQUIRE(vv.grad().data_as<float>()[0] == Catch::Approx(3.0f));
    REQUIRE(vv.grad().data_as<float>()[1] == Catch::Approx(3.0f));
}

TEST_CASE("row_scale - rejects wrong v shape") {
    auto x = make_input(3, 4);
    Variable vbad(zeros({3, 2}), false);
    REQUIRE_THROWS_AS(row_scale(x, vbad), std::runtime_error);
}

// ── MoEFeedForward ────────────────────────────────────────────────────────────

TEST_CASE("MoEFeedForward - output shape (T, D)") {
    MoEFeedForward moe(16, 4, 2);
    auto x = make_input(5, 16);
    auto [out, aux] = moe.forward(x);
    REQUIRE(out.data().shape(0) == 5);
    REQUIRE(out.data().shape(1) == 16);
    REQUIRE(aux.data().numel() == 1);
}

TEST_CASE("MoEFeedForward - aux loss is non-negative") {
    MoEFeedForward moe(16, 4, 1, 0, 99);
    auto x = make_input(4, 16);
    auto [out, aux] = moe.forward(x);
    REQUIRE(aux.data().data_as<float>()[0] >= 0.0f);
}

TEST_CASE("MoEFeedForward - invalid construction throws") {
    REQUIRE_THROWS_AS(MoEFeedForward(16, 1, 1),   std::runtime_error);  // n_experts < 2
    REQUIRE_THROWS_AS(MoEFeedForward(16, 4, 0),   std::runtime_error);  // top_k < 1
    REQUIRE_THROWS_AS(MoEFeedForward(16, 4, 5),   std::runtime_error);  // top_k > n_experts
    REQUIRE_THROWS_AS(MoEFeedForward(0,  4, 1),   std::runtime_error);  // D <= 0
}

TEST_CASE("MoEFeedForward - gradient flows through router and experts") {
    MoEFeedForward moe(8, 2, 1, 0, 7);
    auto x = make_input(3, 8);
    auto [out, aux] = moe.forward(x);
    auto loss = add(sum(out), aux);
    loss.backward();

    bool any_grad = false;
    for (auto* v : moe.parameters()) {
        if (v->grad().numel() == 0) continue;
        for (auto f : v->grad().data_as<float>())
            if (f != 0.f) { any_grad = true; break; }
        if (any_grad) break;
    }
    REQUIRE(any_grad);
}

// ── MoEGPT ────────────────────────────────────────────────────────────────────

TEST_CASE("MoEGPT - forward output shape (T, V)") {
    MoEGPT m(20, 16, 2, 1, 2, 4, 2);
    auto ids    = make_ids({0, 1, 2, 3});
    auto logits = m.forward(ids);
    REQUIRE(logits.data().shape(0) == 4);
    REQUIRE(logits.data().shape(1) == 20);
}

TEST_CASE("MoEGPT - forward_moe returns aux loss") {
    MoEGPT m(20, 16, 2, 1, 2, 4, 2);
    auto ids = make_ids({0, 1, 2});
    auto [logits, aux] = m.forward_moe(ids);
    REQUIRE(logits.data().ndim() == 2);
    REQUIRE(aux.data().numel() == 1);
    REQUIRE(aux.data().data_as<float>()[0] >= 0.f);
}

TEST_CASE("MoEGPT - accessors") {
    MoEGPT m(30, 16, 2, 1, 3, 4, 2);
    REQUIRE(m.vocab_size() == 30);
    REQUIRE(m.embed_dim()  == 16);
    REQUIRE(m.num_layers() == 3);
    REQUIRE(m.n_experts()  == 4);
    REQUIRE(m.top_k()      == 2);
}

TEST_CASE("MoEGPT - invalid construction throws") {
    REQUIRE_THROWS_AS(MoEGPT(0,  16, 2, 1, 2, 4, 2), std::runtime_error);
    REQUIRE_THROWS_AS(MoEGPT(20,  0, 2, 1, 2, 4, 2), std::runtime_error);
    REQUIRE_THROWS_AS(MoEGPT(20, 16, 2, 1, 0, 4, 2), std::runtime_error);
}

TEST_CASE("MoEGPT - more parameters than dense ModernGPT (E experts vs 1)") {
    // MoEGPT has E SwiGLU experts per block; ModernGPT has 1.
    // With E=4, MoE should have more params.
    MoEGPT moe(20, 16, 2, 1, 2, 4, 2);
    ModernGPT dense(20, 16, 2, 1, 2);

    auto count = [](auto& m) {
        int64_t n = 0;
        for (auto* v : m.parameters()) n += static_cast<int64_t>(v->data().numel());
        return n;
    };
    REQUIRE(count(moe) > count(dense));
}

TEST_CASE("MoEGPT - training step reduces loss") {
    const int64_t V = 20, T = 6;
    MoEGPT m(V, 16, 2, 1, 1, 2, 1, 0, 42);
    Adam adam(m.parameters(), 5e-3f);

    Tensor ids({T}, DType::Int32);
    Tensor tgt({T - 1}, DType::Int32);
    for (int64_t i = 0; i < T;     ++i) ids.data_as<int32_t>()[static_cast<std::size_t>(i)] = static_cast<int32_t>(i % V);
    for (int64_t i = 0; i < T - 1; ++i) tgt.data_as<int32_t>()[static_cast<std::size_t>(i)] = static_cast<int32_t>((i + 1) % V);

    float first = -1.f, last = -1.f;
    for (int step = 0; step < 40; ++step) {
        adam.zero_grad();
        auto [logits, aux] = m.forward_moe(ids);
        auto ltrunc = narrow(logits, 0, T - 1);
        auto loss   = add(cross_entropy(ltrunc, tgt), scale(aux, 0.01f));
        loss.backward();
        adam.step();
        float lv = loss.data().data_as<float>()[0];
        if (step == 0) first = lv;
        last = lv;
    }
    REQUIRE(last < first);
}
