#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "sub0llm/autograd/ops.hpp"
#include "sub0llm/autograd/variable.hpp"
#include "sub0llm/nn/gpt.hpp"

#include <cmath>
#include <vector>

using Catch::Matchers::WithinAbs;
using namespace sub0llm;
using namespace sub0llm::autograd;
using namespace sub0llm::nn;

// ── Helpers ───────────────────────────────────────────────────────────────────

static Variable leaf2d(std::vector<float> vals, int64_t rows, int64_t cols,
                       bool rg = true) {
    Tensor t = zeros({rows, cols});
    auto s = t.data_as<float>();
    for (std::size_t i = 0; i < vals.size(); ++i) s[i] = vals[i];
    return Variable(std::move(t), rg);
}

// ── autograd::gelu ────────────────────────────────────────────────────────────

TEST_CASE("gelu - forward matches ops::gelu", "[gpt][gelu]") {
    auto x = leaf2d({-1.f, 0.f, 1.f, 2.f}, 2, 2);
    auto y = gelu(x);
    const auto yd = y.data().data_as<float>();
    // gelu(0) = 0, gelu(-1) < 0, gelu(1) > 0.5, gelu(2) close to 2
    REQUIRE_THAT(yd[1], WithinAbs(0.0f, 1e-5f));           // gelu(0)=0
    REQUIRE(yd[0] < 0.0f);                                   // gelu(-1)<0
    REQUIRE_THAT(yd[2], WithinAbs(0.8413f, 1e-3f));         // gelu(1)≈0.841
}

TEST_CASE("gelu - backward numerical gradient check", "[gpt][gelu]") {
    const float eps = 1e-3f;
    Tensor xt = zeros({1, 4});
    { auto s = xt.data_as<float>(); s[0]=-1.f; s[1]=0.f; s[2]=0.5f; s[3]=2.f; }

    Variable x(xt, true);
    sum(gelu(x)).backward();
    const auto ag = x.grad().data_as<float>();

    const auto xd = x.data().data_as<float>();
    for (std::size_t k = 0; k < 4u; ++k) {
        auto eval = [&](float delta) {
            Tensor xk = zeros({1, 4});
            auto s = xk.data_as<float>();
            for (std::size_t i = 0; i < 4u; ++i) s[i] = xd[i];
            s[k] += delta;
            Variable vk(std::move(xk), false);
            auto yk = gelu(vk);
            float val = 0.0f;
            for (float v : yk.data().data_as<float>()) val += v;
            return val;
        };
        const float ng = (eval(eps) - eval(-eps)) / (2.0f * eps);
        REQUIRE_THAT(ag[k], WithinAbs(ng, 2e-3f));
    }
}

// ── autograd::layer_norm ──────────────────────────────────────────────────────

TEST_CASE("layer_norm - output is zero-mean unit-variance with identity affine",
          "[gpt][ln]") {
    // weight=1, bias=0 → y = x_hat
    Tensor xt = zeros({2, 4});
    { auto s = xt.data_as<float>();
      s[0]=1.f; s[1]=2.f; s[2]=3.f; s[3]=4.f;
      s[4]=10.f; s[5]=20.f; s[6]=30.f; s[7]=40.f; }

    Tensor wt = zeros({4}); for (auto& v : wt.data_as<float>()) v = 1.0f;
    Tensor bt = zeros({4});

    Variable x(xt, false), w(wt, false), b(bt, false);
    auto y  = layer_norm(x, w, b);
    const auto yd = y.data().data_as<float>();

    for (std::size_t row = 0; row < 2u; ++row) {
        float mn = 0.f, ms = 0.f;
        for (std::size_t j = 0; j < 4u; ++j) mn += yd[row * 4u + j];
        mn /= 4.f;
        for (std::size_t j = 0; j < 4u; ++j) ms += yd[row * 4u + j] * yd[row * 4u + j];
        ms /= 4.f;
        REQUIRE_THAT(mn, WithinAbs(0.0f, 1e-5f));
        REQUIRE_THAT(ms, WithinAbs(1.0f, 1e-4f));
    }
}

TEST_CASE("layer_norm - bias shifts output by exact amount", "[gpt][ln]") {
    // With weight=1, adding bias=k to feature j shifts y[j] by exactly k.
    Tensor xt = zeros({1, 3});
    { auto s = xt.data_as<float>(); s[0]=1.f; s[1]=2.f; s[2]=3.f; }
    Tensor wt = zeros({3}); for (auto& v : wt.data_as<float>()) v = 1.0f;
    Tensor bt = zeros({3}); { auto s = bt.data_as<float>(); s[0]=0.f; s[1]=5.f; s[2]=-3.f; }

    Variable x(xt, false), w(wt, false);
    Variable b0(zeros({3}), false);
    Variable b1(bt, false);
    const auto y0v = layer_norm(x, w, b0);
    const auto y1v = layer_norm(x, w, b1);
    const auto y0 = y0v.data().data_as<float>();
    const auto y1 = y1v.data().data_as<float>();
    // bias shifts output by the bias value
    REQUIRE_THAT(y1[0] - y0[0], WithinAbs( 0.0f, 1e-5f));
    REQUIRE_THAT(y1[1] - y0[1], WithinAbs( 5.0f, 1e-5f));
    REQUIRE_THAT(y1[2] - y0[2], WithinAbs(-3.0f, 1e-5f));
}

TEST_CASE("layer_norm - weight scales x_hat proportionally", "[gpt][ln]") {
    // weight=2 doubles each normalised feature.
    Tensor xt = zeros({1, 3});
    { auto s = xt.data_as<float>(); s[0]=1.f; s[1]=2.f; s[2]=3.f; }
    Tensor w1t = zeros({3}); for (auto& v : w1t.data_as<float>()) v = 1.0f;
    Tensor w2t = zeros({3}); for (auto& v : w2t.data_as<float>()) v = 2.0f;

    Variable x(xt, false), w1(w1t, false), w2(w2t, false), b(zeros({3}), false);
    const auto r1 = layer_norm(x, w1, b);
    const auto r2 = layer_norm(x, w2, b);
    const auto y1 = r1.data().data_as<float>();
    const auto y2 = r2.data().data_as<float>();
    for (std::size_t j = 0; j < 3u; ++j)
        REQUIRE_THAT(y2[j], WithinAbs(2.0f * y1[j], 1e-5f));
}

TEST_CASE("layer_norm - backward gradient check for x", "[gpt][ln]") {
    const float eps = 1e-3f;
    Tensor xt = zeros({2, 3});
    { auto s = xt.data_as<float>();
      s[0]=0.3f; s[1]=-0.1f; s[2]=0.8f;
      s[3]=1.2f; s[4]=0.0f;  s[5]=-0.5f; }
    Tensor wt = zeros({3}); { auto s = wt.data_as<float>(); s[0]=1.f; s[1]=2.f; s[2]=0.5f; }
    Tensor bt = zeros({3}); { auto s = bt.data_as<float>(); s[0]=0.1f; s[1]=-0.2f; s[2]=0.f; }

    Variable x(xt, true), w(wt, false), b(bt, false);
    sum(layer_norm(x, w, b)).backward();
    const auto ag = x.grad().data_as<float>();

    const auto xd = x.data().data_as<float>();
    const auto wd = wt.data_as<float>();
    const auto bd = bt.data_as<float>();
    for (std::size_t k = 0; k < 6u; ++k) {
        auto eval = [&](float delta) {
            Tensor xk = zeros({2, 3});
            auto s = xk.data_as<float>();
            for (std::size_t i = 0; i < 6u; ++i) s[i] = xd[i];
            s[k] += delta;
            Variable vx(std::move(xk), false), vw(wt, false), vb(bt, false);
            auto yk = layer_norm(vx, vw, vb);
            float val = 0.0f;
            for (float v : yk.data().data_as<float>()) val += v;
            return val;
        };
        const float ng = (eval(eps) - eval(-eps)) / (2.0f * eps);
        REQUIRE_THAT(ag[k], WithinAbs(ng, 5e-3f));
    }
}

TEST_CASE("layer_norm - backward gradient check for weight", "[gpt][ln]") {
    const float eps = 1e-3f;
    Tensor xt = zeros({2, 3});
    { auto s = xt.data_as<float>();
      s[0]=0.3f; s[1]=-0.1f; s[2]=0.8f;
      s[3]=1.2f; s[4]=0.0f;  s[5]=-0.5f; }
    Tensor wt = zeros({3}); { auto s = wt.data_as<float>(); s[0]=1.f; s[1]=2.f; s[2]=0.5f; }
    Tensor bt = zeros({3});

    Variable x(xt, false), w(wt, true), b(bt, false);
    sum(layer_norm(x, w, b)).backward();
    const auto ag = w.grad().data_as<float>();

    const auto wd = wt.data_as<float>();
    for (std::size_t k = 0; k < 3u; ++k) {
        auto eval = [&](float delta) {
            Tensor wk = zeros({3});
            auto s = wk.data_as<float>();
            for (std::size_t i = 0; i < 3u; ++i) s[i] = wd[i];
            s[k] += delta;
            Variable vx(xt, false), vw(std::move(wk), false), vb(bt, false);
            auto yk = layer_norm(vx, vw, vb);
            float val = 0.0f;
            for (float v : yk.data().data_as<float>()) val += v;
            return val;
        };
        const float ng = (eval(eps) - eval(-eps)) / (2.0f * eps);
        REQUIRE_THAT(ag[k], WithinAbs(ng, 5e-3f));
    }
}

TEST_CASE("layer_norm - non-2D input throws", "[gpt][ln]") {
    Variable x(zeros({4}), false), w(zeros({4}), false), b(zeros({4}), false);
    REQUIRE_THROWS_AS(layer_norm(x, w, b), std::runtime_error);
}

// ── Linear ────────────────────────────────────────────────────────────────────

TEST_CASE("Linear - output shape (T, out)", "[gpt][linear]") {
    Linear lin(8, 4);
    auto x = leaf2d(std::vector<float>(3 * 8, 0.1f), 3, 8);
    auto y = lin.forward(x);
    REQUIRE(y.data().shape()[0] == 3);
    REQUIRE(y.data().shape()[1] == 4);
}

TEST_CASE("Linear - gradient flows to W and b", "[gpt][linear]") {
    Linear lin(4, 3, /*seed=*/1);
    auto x = leaf2d(std::vector<float>(2 * 4, 0.5f), 2, 4);
    sum(lin.forward(x)).backward();
    for (auto* p : lin.parameters()) {
        REQUIRE(p->grad().numel() > 0);
        bool any_nonzero = false;
        for (float v : p->grad().data_as<float>())
            if (v != 0.0f) { any_nonzero = true; break; }
        REQUIRE(any_nonzero);
    }
}

TEST_CASE("Linear - invalid dimensions throw", "[gpt][linear]") {
    REQUIRE_THROWS_AS(Linear(0, 4), std::runtime_error);
    REQUIRE_THROWS_AS(Linear(4, 0), std::runtime_error);
}

TEST_CASE("Linear - parameters() returns 2 pointers", "[gpt][linear]") {
    Linear lin(4, 8);
    REQUIRE(lin.parameters().size() == 2u);
}

// ── LayerNorm ─────────────────────────────────────────────────────────────────

TEST_CASE("LayerNorm - default weight=1 bias=0", "[gpt][layernorm]") {
    LayerNorm ln(4);
    const auto wp = ln.parameters();
    REQUIRE(wp.size() == 2u);
    for (float v : wp[0]->data().data_as<float>())
        REQUIRE_THAT(v, WithinAbs(1.0f, 1e-7f));
    for (float v : wp[1]->data().data_as<float>())
        REQUIRE_THAT(v, WithinAbs(0.0f, 1e-7f));
}

TEST_CASE("LayerNorm - gradient flows to weight and bias", "[gpt][layernorm]") {
    LayerNorm ln(6);
    auto x = leaf2d(std::vector<float>(3 * 6, 0.3f), 3, 6, false);
    sum(ln.forward(x)).backward();
    for (auto* p : ln.parameters()) {
        REQUIRE(p->grad().numel() > 0);
    }
}

// ── FeedForward ───────────────────────────────────────────────────────────────

TEST_CASE("FeedForward - output shape (T, D)", "[gpt][ffn]") {
    FeedForward ffn(8, /*seed=*/0);
    auto x = leaf2d(std::vector<float>(3 * 8, 0.1f), 3, 8);
    auto y = ffn.forward(x);
    REQUIRE(y.data().shape()[0] == 3);
    REQUIRE(y.data().shape()[1] == 8);
}

TEST_CASE("FeedForward - gradient flows to all 4 parameters", "[gpt][ffn]") {
    FeedForward ffn(4, /*seed=*/1);
    auto x = leaf2d(std::vector<float>(2 * 4, 0.1f), 2, 4);
    sum(ffn.forward(x)).backward();
    REQUIRE(ffn.parameters().size() == 4u);
    for (auto* p : ffn.parameters()) {
        REQUIRE(p->grad().numel() > 0);
        bool any_nonzero = false;
        for (float v : p->grad().data_as<float>())
            if (v != 0.0f) { any_nonzero = true; break; }
        REQUIRE(any_nonzero);
    }
}

// ── TransformerBlock ──────────────────────────────────────────────────────────

TEST_CASE("TransformerBlock - output shape (T, D)", "[gpt][block]") {
    TransformerBlock block(8, 2, /*seed=*/0);
    auto x = leaf2d(std::vector<float>(4 * 8, 0.1f), 4, 8);
    auto y = block.forward(x);
    REQUIRE(y.data().shape()[0] == 4);
    REQUIRE(y.data().shape()[1] == 8);
}

TEST_CASE("TransformerBlock - gradient flows to all parameters", "[gpt][block]") {
    TransformerBlock block(4, 2, /*seed=*/3);
    Variable x(randn({3, 4}), true);
    sum(block.forward(x)).backward();
    for (auto* p : block.parameters()) {
        REQUIRE(p->grad().numel() > 0);
        bool any_nonzero = false;
        for (float v : p->grad().data_as<float>())
            if (v != 0.0f) { any_nonzero = true; break; }
        REQUIRE(any_nonzero);
    }
}

// ── GPT ───────────────────────────────────────────────────────────────────────

TEST_CASE("GPT - forward output shape (T, vocab_size)", "[gpt][model]") {
    GPT model(/*vocab=*/32, /*embed=*/8, /*heads=*/2, /*layers=*/2,
              /*max_seq=*/16, /*seed=*/0);
    Tensor ids = zeros({5}, DType::Int32);
    { auto s = ids.data_as<int32_t>(); s[0]=1; s[1]=3; s[2]=7; s[3]=2; s[4]=0; }
    auto logits = model.forward(ids);
    REQUIRE(logits.data().shape()[0] == 5);
    REQUIRE(logits.data().shape()[1] == 32);
}

TEST_CASE("GPT - backward populates gradients on all parameters", "[gpt][model]") {
    GPT model(/*vocab=*/16, /*embed=*/4, /*heads=*/2, /*layers=*/1,
              /*max_seq=*/8, /*seed=*/1);
    Tensor ids = zeros({3}, DType::Int32);
    { auto s = ids.data_as<int32_t>(); s[0]=0; s[1]=5; s[2]=3; }

    auto logits = model.forward(ids);
    sum(logits).backward();

    std::size_t params_with_grad = 0;
    for (auto* p : model.parameters())
        if (p->grad().numel() > 0) ++params_with_grad;
    REQUIRE(params_with_grad == model.parameters().size());
}

TEST_CASE("GPT - invalid arguments throw", "[gpt][model]") {
    REQUIRE_THROWS_AS(GPT(32, 8, 2,  0, 16), std::runtime_error);  // num_layers=0
    REQUIRE_THROWS_AS(GPT(32, 8, 0,  1, 16), std::runtime_error);  // num_heads=0
    REQUIRE_THROWS_AS(GPT(32, 8, -1, 1, 16), std::runtime_error);  // num_heads<0
}

TEST_CASE("GPT - weight tying: tok_emb appears once in parameters()", "[gpt][model]") {
    GPT model(32, 8, 2, 1, 16, /*seed=*/0);
    const auto params = model.parameters();
    // Count occurrences of the tok_emb weight pointer
    std::size_t count = 0;
    for (const auto* p : params)
        if (p == params[0]) ++count;
    REQUIRE(count == 1u);
}

// ── Ch27: Linear Q8 inference fast path ─────────────────────────────────────────
TEST_CASE("Linear Q8 apply_one matches f32 within quant tolerance", "[gpt][quant]") {
    const int64_t in = 256, out = 64;
    Linear lin(in, out, /*seed=*/7);

    Tensor x = zeros({1, in});
    auto xd = x.data_as<float>();
    uint32_t s = 12345u;
    for (int64_t i = 0; i < in; ++i) {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        xd[static_cast<std::size_t>(i)] = (static_cast<float>(s >> 8) / 16777216.0f - 0.5f);
    }

    const Tensor y_f32 = lin.apply_one(x);
    lin.quantize_weights();
    REQUIRE(lin.q8_enabled());
    const Tensor y_q8 = lin.apply_one(x);

    REQUIRE(y_q8.shape() == y_f32.shape());
    auto a = y_f32.data_as<float>();
    auto b = y_q8.data_as<float>();
    double se = 0.0, sr = 0.0;
    for (int64_t i = 0; i < out; ++i) {
        const double d = static_cast<double>(b[static_cast<std::size_t>(i)]) - a[static_cast<std::size_t>(i)];
        se += d * d; sr += static_cast<double>(a[static_cast<std::size_t>(i)]) * a[static_cast<std::size_t>(i)];
    }
    REQUIRE(std::sqrt(se / (sr + 1e-12)) < 0.05);   // < 5% relRMS
}
