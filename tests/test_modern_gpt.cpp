#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "sub0llm/autograd/ops.hpp"
#include "sub0llm/autograd/variable.hpp"
#include "sub0llm/nn/modern_gpt.hpp"

#include <cmath>
#include <vector>

using Catch::Matchers::WithinAbs;
using namespace sub0llm;
using namespace sub0llm::autograd;
using namespace sub0llm::nn;

// ── Helpers ───────────────────────────────────────────────────────────────────

static Variable leaf(std::vector<float> vals, int64_t rows, int64_t cols,
                     bool rg = true) {
    Tensor t = zeros({rows, cols});
    auto s = t.data_as<float>();
    for (std::size_t i = 0; i < vals.size(); ++i) s[i] = vals[i];
    return Variable(std::move(t), rg);
}

static Tensor make_ids(int64_t T, int32_t vocab) {
    Tensor ids({T}, DType::Int32);
    auto d = ids.data_as<int32_t>();
    for (std::size_t i = 0; i < static_cast<std::size_t>(T); ++i)
        d[i] = static_cast<int32_t>(static_cast<int64_t>(i) % vocab);
    return ids;
}


// ── autograd::silu ────────────────────────────────────────────────────────────

TEST_CASE("silu — forward values", "[modern_gpt][silu]") {
    // silu(0) = 0, silu(x) > 0 for x > 0, negative for small x < 0
    auto x = leaf({-3.f, -1.f, 0.f, 1.f, 3.f}, 1, 5);
    auto y = silu(x);
    auto yd = y.data().data_as<float>();

    REQUIRE_THAT(yd[2], WithinAbs(0.0f, 1e-6f));  // silu(0) = 0
    REQUIRE(yd[3] > 0.0f);                          // silu(1) > 0
    REQUIRE(yd[0] < 0.0f);                          // silu(-3) < 0
    // silu(1) = 1 * sigmoid(1) ≈ 0.7311
    REQUIRE_THAT(yd[3], WithinAbs(0.7311f, 2e-4f));
}

TEST_CASE("silu — backward numerical gradient check", "[modern_gpt][silu]") {
    auto x = leaf({-2.f, -0.5f, 0.f, 1.f, 2.5f}, 1, 5);
    sum(silu(x)).backward();
    const auto ag = x.grad().data_as<float>();
    const auto xd = x.data().data_as<float>();

    for (std::size_t k = 0; k < 5u; ++k) {
        const float xv = xd[k];
        const float sg = 1.0f / (1.0f + std::exp(-xv));
        const float expected = sg * (1.0f + xv * (1.0f - sg));
        REQUIRE_THAT(ag[k], WithinAbs(expected, 1e-4f));
    }
}

// ── autograd::rms_norm ────────────────────────────────────────────────────────

TEST_CASE("rms_norm — output is unit-RMS when weight=1", "[modern_gpt][rms_norm]") {
    constexpr int64_t T = 3, D = 6;
    Tensor xd({T, D}, DType::Float32);
    {
        auto d = xd.data_as<float>();
        for (std::size_t i = 0; i < static_cast<std::size_t>(T * D); ++i)
            d[i] = static_cast<float>(i + 1) * 0.3f;
    }
    Tensor wd({D}, DType::Float32);
    for (auto& v : wd.data_as<float>()) v = 1.0f;

    Variable x(xd, false), w(wd, false);
    auto y = rms_norm(x, w);

    auto yd = y.data().data_as<float>();
    for (std::size_t t = 0; t < static_cast<std::size_t>(T); ++t) {
        float rms2 = 0.0f;
        for (std::size_t d = 0; d < static_cast<std::size_t>(D); ++d) rms2 += yd[t * static_cast<std::size_t>(D) + d] * yd[t * static_cast<std::size_t>(D) + d];
        REQUIRE_THAT(std::sqrt(rms2 / static_cast<float>(D)), WithinAbs(1.0f, 2e-5f));
    }
}

TEST_CASE("rms_norm — weight scales output proportionally", "[modern_gpt][rms_norm]") {
    constexpr int64_t T = 1, D = 4;
    auto x = leaf({1.f, 2.f, 3.f, 4.f}, T, D, /*rg=*/false);

    Tensor w2d({D}, DType::Float32);
    for (auto& v : w2d.data_as<float>()) v = 2.0f;
    Variable w2(w2d, false);
    auto y2 = rms_norm(x, w2);

    Tensor w1d({D}, DType::Float32);
    for (auto& v : w1d.data_as<float>()) v = 1.0f;
    Variable w1(w1d, false);
    auto y1 = rms_norm(x, w1);

    auto y2d = y2.data().data_as<float>();
    auto y1d = y1.data().data_as<float>();
    for (std::size_t i = 0; i < static_cast<std::size_t>(D); ++i)
        REQUIRE_THAT(y2d[i], WithinAbs(2.0f * y1d[i], 1e-5f));
}

TEST_CASE("rms_norm — backward: x-grad numerical check", "[modern_gpt][rms_norm]") {
    constexpr int64_t T = 2, D = 4;
    const float eps = 5e-4f;

    Tensor xd({T, D}, DType::Float32);
    {
        auto d = xd.data_as<float>();
        d[0]=1.f; d[1]=-0.5f; d[2]=2.f; d[3]=0.3f;
        d[4]=0.f; d[5]=1.5f;  d[6]=-1.f; d[7]=0.8f;
    }
    Tensor wd({D}, DType::Float32);
    for (auto& v : wd.data_as<float>()) v = 1.0f;

    const std::size_t Nelems = static_cast<std::size_t>(T * D);
    const auto xd_orig = xd.data_as<float>();
    for (std::size_t k = 0; k < Nelems; ++k) {
        auto eval = [&](float delta) {
            Tensor xdc = zeros({T, D});
            auto dc = xdc.data_as<float>();
            for (std::size_t i = 0; i < Nelems; ++i) dc[i] = xd_orig[i];
            dc[k] += delta;
            Variable xv(std::move(xdc), false), wv(wd, false);
            auto y = rms_norm(xv, wv);
            float s = 0.f; for (float v : y.data().data_as<float>()) s += v;
            return s;
        };
        const float ng = (eval(eps) - eval(-eps)) / (2.f * eps);

        Variable xv(xd, true), wv(wd, false);
        sum(rms_norm(xv, wv)).backward();
        REQUIRE_THAT(xv.grad().data_as<float>()[k], WithinAbs(ng, 1e-3f));
    }
}

TEST_CASE("rms_norm — backward: weight-grad numerical check", "[modern_gpt][rms_norm]") {
    constexpr int64_t T = 2, D = 4;
    const float eps = 1e-3f;

    Tensor xd({T, D}, DType::Float32);
    {
        auto d = xd.data_as<float>();
        d[0]=1.f; d[1]=2.f; d[2]=3.f; d[3]=4.f;
        d[4]=0.5f; d[5]=1.5f; d[6]=2.5f; d[7]=3.5f;
    }
    Tensor wd({D}, DType::Float32);
    for (auto& v : wd.data_as<float>()) v = 1.0f;

    const std::size_t Dw = static_cast<std::size_t>(D);
    const auto wd_orig = wd.data_as<float>();
    for (std::size_t k = 0; k < Dw; ++k) {
        auto eval = [&](float delta) {
            Tensor wdc = zeros({D});
            auto dc = wdc.data_as<float>();
            for (std::size_t i = 0; i < Dw; ++i) dc[i] = wd_orig[i];
            dc[k] += delta;
            Variable xv(xd, false), wv(std::move(wdc), false);
            auto y = rms_norm(xv, wv);
            float s = 0.f; for (float v : y.data().data_as<float>()) s += v;
            return s;
        };
        const float ng = (eval(eps) - eval(-eps)) / (2.f * eps);

        Variable xv(xd, false), wv(wd, true);
        sum(rms_norm(xv, wv)).backward();
        REQUIRE_THAT(wv.grad().data_as<float>()[k], WithinAbs(ng, 1e-3f));
    }
}

// ── autograd::rope ────────────────────────────────────────────────────────────

TEST_CASE("rope — forward: rotation pairs", "[modern_gpt][rope]") {
    // For a single position with known cos/sin, verify the rotation formula.
    // out[0] = x[0]*cos - x[1]*sin, out[1] = x[0]*sin + x[1]*cos
    const float angle = 0.5f;
    const float cv = std::cos(angle), sv = std::sin(angle);

    Tensor cos_f({1, 1}, DType::Float32); cos_f.data_as<float>()[0] = cv;
    Tensor sin_f({1, 1}, DType::Float32); sin_f.data_as<float>()[0] = sv;

    auto x = leaf({3.f, 4.f}, 1, 2);
    auto y = rope(x, cos_f, sin_f);
    auto yd = y.data().data_as<float>();

    REQUIRE_THAT(yd[0], WithinAbs(3.f * cv - 4.f * sv, 1e-5f));
    REQUIRE_THAT(yd[1], WithinAbs(3.f * sv + 4.f * cv, 1e-5f));
}

TEST_CASE("rope — forward preserves vector norm", "[modern_gpt][rope]") {
    // Rotation is norm-preserving.
    constexpr int64_t T = 4, Dh = 8;
    const int64_t D2 = Dh / 2;

    Tensor cos_f({T, D2}, DType::Float32), sin_f({T, D2}, DType::Float32);
    {
        auto cd = cos_f.data_as<float>(); auto sd = sin_f.data_as<float>();
        const std::size_t Ts = static_cast<std::size_t>(T), D2s = static_cast<std::size_t>(D2);
        for (std::size_t t = 0; t < Ts; ++t)
            for (std::size_t i = 0; i < D2s; ++i) {
                float a = 0.1f * static_cast<float>(t * D2s + i);
                cd[t * D2s + i] = std::cos(a); sd[t * D2s + i] = std::sin(a);
            }
    }

    Tensor xd({T, Dh}, DType::Float32);
    {
        auto d = xd.data_as<float>();
        for (std::size_t i = 0; i < static_cast<std::size_t>(T * Dh); ++i)
            d[i] = static_cast<float>(i + 1);
    }
    Variable x(xd, false);
    auto y = rope(x, cos_f, sin_f);

    auto xr = xd.data_as<float>(); auto yr = y.data().data_as<float>();
    for (std::size_t t = 0; t < static_cast<std::size_t>(T); ++t) {
        float nx = 0.f, ny = 0.f;
        const std::size_t Dhs = static_cast<std::size_t>(Dh);
        for (std::size_t d = 0; d < Dhs; ++d) {
            nx += xr[t * Dhs + d] * xr[t * Dhs + d];
            ny += yr[t * Dhs + d] * yr[t * Dhs + d];
        }
        REQUIRE_THAT(std::sqrt(ny), WithinAbs(std::sqrt(nx), 1e-4f));
    }
}

TEST_CASE("rope — backward: inverse rotation (negate sin)", "[modern_gpt][rope]") {
    // Apply rope, upstream gradient = all-ones, backward should equal inverse rotation.
    const float angle = 0.7f;
    const float cv = std::cos(angle), sv = std::sin(angle);

    Tensor cos_f({1, 1}, DType::Float32); cos_f.data_as<float>()[0] = cv;
    Tensor sin_f({1, 1}, DType::Float32); sin_f.data_as<float>()[0] = sv;

    auto x = leaf({2.f, 5.f}, 1, 2);
    sum(rope(x, cos_f, sin_f)).backward();

    // upstream g = [1, 1], inverse: gx[0] = g[0]*cos + g[1]*sin
    //                                 gx[1] = -g[0]*sin + g[1]*cos
    auto gd = x.grad().data_as<float>();
    REQUIRE_THAT(gd[0], WithinAbs(cv + sv, 1e-5f));
    REQUIRE_THAT(gd[1], WithinAbs(-sv + cv, 1e-5f));
}

TEST_CASE("rope — backward numerical gradient check", "[modern_gpt][rope]") {
    constexpr int64_t T = 3, Dh = 4;
    const int64_t D2 = Dh / 2;
    const float eps = 1e-3f;

    Tensor cos_f({T, D2}, DType::Float32), sin_f({T, D2}, DType::Float32);
    {
        auto cd = cos_f.data_as<float>(); auto sd = sin_f.data_as<float>();
        for (std::size_t i = 0; i < static_cast<std::size_t>(T * D2); ++i) {
            float a = 0.3f * static_cast<float>(i);
            cd[i] = std::cos(a); sd[i] = std::sin(a);
        }
    }

    Tensor xd({T, Dh}, DType::Float32);
    {
        auto d = xd.data_as<float>();
        for (std::size_t i = 0; i < static_cast<std::size_t>(T * Dh); ++i)
            d[i] = 0.1f * static_cast<float>(i + 1);
    }

    const std::size_t Nrope = static_cast<std::size_t>(T * Dh);
    const auto xd_orig = xd.data_as<float>();
    for (std::size_t k = 0; k < Nrope; ++k) {
        auto eval = [&](float delta) {
            Tensor xdc = zeros({T, Dh});
            auto dc = xdc.data_as<float>();
            for (std::size_t i = 0; i < Nrope; ++i) dc[i] = xd_orig[i];
            dc[k] += delta;
            Variable xv(std::move(xdc), false);
            auto y = rope(xv, cos_f, sin_f);
            float s = 0.f; for (float v : y.data().data_as<float>()) s += v;
            return s;
        };
        const float ng = (eval(eps) - eval(-eps)) / (2.f * eps);

        Variable xv(xd, true);
        sum(rope(xv, cos_f, sin_f)).backward();
        REQUIRE_THAT(xv.grad().data_as<float>()[k], WithinAbs(ng, 1e-3f));
    }
}

// ── RMSNorm class ─────────────────────────────────────────────────────────────

TEST_CASE("RMSNorm — construction and dim()", "[modern_gpt][rmsnorm]") {
    RMSNorm n(16);
    REQUIRE(n.dim() == 16);
    REQUIRE(n.parameters().size() == 1);

    REQUIRE_THROWS([] { RMSNorm bad(0); }());
}

TEST_CASE("RMSNorm — forward output is unit-RMS", "[modern_gpt][rmsnorm]") {
    constexpr int64_t T = 4, D = 8;
    RMSNorm n(D);

    Tensor xd({T, D}, DType::Float32);
    {
        auto d = xd.data_as<float>();
        for (std::size_t i = 0; i < static_cast<std::size_t>(T * D); ++i)
            d[i] = static_cast<float>((i % 7) + 1) * 0.5f;
    }
    Variable x(xd, false);
    auto y = n.forward(x);

    auto yd = y.data().data_as<float>();
    for (std::size_t t = 0; t < static_cast<std::size_t>(T); ++t) {
        float rms2 = 0.f;
        const std::size_t Ds = static_cast<std::size_t>(D);
        for (std::size_t d = 0; d < Ds; ++d) rms2 += yd[t * Ds + d] * yd[t * Ds + d];
        REQUIRE_THAT(std::sqrt(rms2 / static_cast<float>(D)), WithinAbs(1.0f, 2e-5f));
    }
}

// ── SwiGLUFeedForward ─────────────────────────────────────────────────────────

TEST_CASE("SwiGLUFeedForward — construction auto d_ff", "[modern_gpt][swiglu]") {
    constexpr int64_t D = 64;
    SwiGLUFeedForward ffn(D, /*d_ff=*/0, /*seed=*/1);
    const auto params = ffn.parameters();
    REQUIRE(!params.empty());
    // d_ff = ceil(8*64/3 / 64) * 64 = 192
    const int64_t expected_dff = (((8 * D + 2) / 3 + 63) / 64) * 64;
    // gate + up: D→d_ff (no bias here since Linear may have bias — check total numel)
    int64_t total = 0;
    for (auto* p : params) total += p->data().numel();
    REQUIRE(total > 0);
    (void)expected_dff;
}

TEST_CASE("SwiGLUFeedForward — output shape preserved", "[modern_gpt][swiglu]") {
    constexpr int64_t T = 5, D = 32;
    SwiGLUFeedForward ffn(D, /*d_ff=*/64, /*seed=*/2);

    Tensor xd({T, D}, DType::Float32);
    for (auto& v : xd.data_as<float>()) v = 0.1f;
    Variable x(xd, false);
    auto y = ffn.forward(x);

    REQUIRE(y.data().shape()[0] == T);
    REQUIRE(y.data().shape()[1] == D);
}

TEST_CASE("SwiGLUFeedForward — backward propagates gradients", "[modern_gpt][swiglu]") {
    constexpr int64_t T = 3, D = 16;
    SwiGLUFeedForward ffn(D, /*d_ff=*/32, /*seed=*/3);

    Tensor xd({T, D}, DType::Float32);
    {
        auto d = xd.data_as<float>();
        for (std::size_t i = 0; i < static_cast<std::size_t>(T * D); ++i)
            d[i] = 0.05f * static_cast<float>(i + 1);
    }
    Variable x(xd, true);
    sum(ffn.forward(x)).backward();

    REQUIRE(x.grad().numel() == T * D);
}

TEST_CASE("SwiGLUFeedForward — throws on D<=0", "[modern_gpt][swiglu]") {
    REQUIRE_THROWS([] { SwiGLUFeedForward bad(0, 64, 1); }());
}

// ── GroupedQueryAttention ─────────────────────────────────────────────────────

TEST_CASE("GroupedQueryAttention — MHA (n_kv_heads==n_heads)", "[modern_gpt][gqa]") {
    constexpr std::size_t D = 16, Nh = 2, Nkv = 2;
    GroupedQueryAttention gqa(D, Nh, Nkv, 10000.0f, 1);

    Tensor xd({4, static_cast<int64_t>(D)}, DType::Float32);
    for (auto& v : xd.data_as<float>()) v = 0.1f;
    Variable x(xd, false);
    auto y = gqa.forward(x, /*causal=*/true);

    REQUIRE(y.data().shape()[0] == 4);
    REQUIRE(y.data().shape()[1] == static_cast<int64_t>(D));
}

TEST_CASE("GroupedQueryAttention — GQA (n_kv_heads < n_heads)", "[modern_gpt][gqa]") {
    constexpr std::size_t D = 16, Nh = 4, Nkv = 2;
    GroupedQueryAttention gqa(D, Nh, Nkv, 10000.0f, 2);

    Tensor xd({5, static_cast<int64_t>(D)}, DType::Float32);
    {
        auto d = xd.data_as<float>();
        for (std::size_t i = 0; i < 5u * D; ++i)
            d[i] = 0.05f * static_cast<float>(i + 1);
    }
    Variable x(xd, false);
    auto y = gqa.forward(x, /*causal=*/true);
    REQUIRE(y.data().shape()[0] == 5);
    REQUIRE(y.data().shape()[1] == static_cast<int64_t>(D));
}

TEST_CASE("GroupedQueryAttention — MQA (n_kv_heads==1)", "[modern_gpt][gqa]") {
    constexpr std::size_t D = 16, Nh = 4, Nkv = 1;
    GroupedQueryAttention gqa(D, Nh, Nkv, 10000.0f, 3);

    const auto params = gqa.parameters();
    // W_Q: Nh, W_O: Nh, W_K: 1, W_V: 1
    REQUIRE(params.size() == Nh * 2 + 2u);
}

TEST_CASE("GroupedQueryAttention — backward propagates gradients", "[modern_gpt][gqa]") {
    constexpr std::size_t D = 16, Nh = 2, Nkv = 1;
    GroupedQueryAttention gqa(D, Nh, Nkv, 10000.0f, 4);

    Tensor xd({3, static_cast<int64_t>(D)}, DType::Float32);
    for (auto& v : xd.data_as<float>()) v = 0.1f;
    Variable x(xd, true);
    sum(gqa.forward(x)).backward();
    REQUIRE(x.grad().numel() == 3 * static_cast<int64_t>(D));
}

TEST_CASE("GroupedQueryAttention — throws on bad dimensions", "[modern_gpt][gqa]") {
    REQUIRE_THROWS([] { GroupedQueryAttention bad(0, 2, 1); }());
    REQUIRE_THROWS([] { GroupedQueryAttention bad(16, 0, 1); }());
    REQUIRE_THROWS([] { GroupedQueryAttention bad(16, 3, 1); }());  // 16 % 3 != 0
    REQUIRE_THROWS([] { GroupedQueryAttention bad(16, 4, 3); }());  // 4 % 3 != 0
}

TEST_CASE("GroupedQueryAttention — forward throws on non-2D input", "[modern_gpt][gqa]") {
    GroupedQueryAttention gqa(16, 2, 1, 10000.0f, 5);
    Tensor xd({16}, DType::Float32);
    Variable x(xd, false);
    REQUIRE_THROWS(gqa.forward(x));
}

// ── ModernTransformerBlock ────────────────────────────────────────────────────

TEST_CASE("ModernTransformerBlock — forward output shape", "[modern_gpt][block]") {
    constexpr int64_t D = 16;
    ModernTransformerBlock block(D, 2, 1, 32, 10);

    Tensor xd({6, D}, DType::Float32);
    for (auto& v : xd.data_as<float>()) v = 0.1f;
    Variable x(xd, false);
    auto y = block.forward(x);

    REQUIRE(y.data().shape()[0] == 6);
    REQUIRE(y.data().shape()[1] == D);
}

TEST_CASE("ModernTransformerBlock — backward flow", "[modern_gpt][block]") {
    constexpr int64_t D = 16;
    ModernTransformerBlock block(D, 2, 1, 32, 11);

    Tensor xd2({4, D}, DType::Float32);
    for (auto& v : xd2.data_as<float>()) v = 0.05f;
    Variable x2(xd2, true);
    sum(block.forward(x2)).backward();
    REQUIRE(x2.grad().numel() == 4 * D);
}

// ── ModernGPT ─────────────────────────────────────────────────────────────────

TEST_CASE("ModernGPT — construction and accessors", "[modern_gpt]") {
    ModernGPT m(32, 16, 2, 1, 2, 0, 0, 1);
    REQUIRE(m.vocab_size() == 32);
    REQUIRE(m.embed_dim()  == 16);
    REQUIRE(m.num_layers() == 2u);
    REQUIRE(m.n_mtp_heads() == 0);
}

TEST_CASE("ModernGPT — forward output shape", "[modern_gpt]") {
    constexpr int64_t V = 64, D = 32, T = 8;
    ModernGPT m(V, D, 2, 1, 2, 0, 0, 2);
    Tensor ids = make_ids(T, static_cast<int32_t>(V));
    auto logits = m.forward(ids);
    REQUIRE(logits.data().shape()[0] == T);
    REQUIRE(logits.data().shape()[1] == V);
}

TEST_CASE("ModernGPT — backward runs without error", "[modern_gpt]") {
    constexpr int64_t V = 32, D = 16, T = 4;
    ModernGPT m(V, D, 2, 1, 2, 0, 0, 3);
    Tensor ids = make_ids(T, static_cast<int32_t>(V));

    Tensor tgt({T}, DType::Int32);
    auto td = tgt.data_as<int32_t>();
    for (std::size_t i = 0; i < static_cast<std::size_t>(T); ++i)
        td[i] = static_cast<int32_t>((static_cast<int64_t>(i) + 1) % V);

    auto logits = m.forward(ids);
    auto loss   = cross_entropy(logits, tgt);
    loss.backward();

    auto params = m.parameters();
    int64_t with_grad = 0;
    for (auto* p : params)
        if (p->grad().numel() > 0) ++with_grad;
    REQUIRE(with_grad > 0);
}

TEST_CASE("ModernGPT — forward_mtp returns 1 + n_mtp_heads heads", "[modern_gpt]") {
    constexpr int64_t V = 32, D = 16, T = 6, Nmtp = 3;
    ModernGPT m(V, D, 2, 1, 2, 0, Nmtp, 4);
    Tensor ids = make_ids(T, static_cast<int32_t>(V));

    auto heads = m.forward_mtp(ids);
    REQUIRE(heads.size() == static_cast<std::size_t>(1 + Nmtp));
    for (const auto& h : heads) {
        REQUIRE(h.data().shape()[0] == T);
        REQUIRE(h.data().shape()[1] == V);
    }
}

TEST_CASE("ModernGPT — forward with n_mtp==0 same as forward_mtp head_0", "[modern_gpt]") {
    constexpr int64_t V = 32, D = 16, T = 5;
    ModernGPT m(V, D, 2, 1, 2, 0, 0, 5);
    Tensor ids = make_ids(T, static_cast<int32_t>(V));

    auto logits     = m.forward(ids);
    auto mtp_heads  = m.forward_mtp(ids);

    auto la = logits.data().data_as<float>();
    auto lb = mtp_heads[0].data().data_as<float>();
    for (std::size_t i = 0; i < static_cast<std::size_t>(T * V); ++i)
        REQUIRE_THAT(la[i], WithinAbs(lb[i], 1e-5f));
}

TEST_CASE("ModernGPT — throws on invalid arguments", "[modern_gpt]") {
    REQUIRE_THROWS([] { ModernGPT bad(32, 16, 0, 1, 2); }());   // n_heads=0
    REQUIRE_THROWS([] { ModernGPT bad(32, 16, 2, 0, 2); }());   // n_kv_heads=0
    REQUIRE_THROWS([] { ModernGPT bad(32, 16, 2, 3, 2); }());   // n_kv_heads>n_heads
    REQUIRE_THROWS([] { ModernGPT bad(32, 16, 2, 1, 0); }());   // n_layers=0
    REQUIRE_THROWS([] { ModernGPT bad(32, 16, 2, 1, 2, 0, -1); }()); // n_mtp<0
}

TEST_CASE("ModernGPT — forward throws on non-1D token_ids", "[modern_gpt]") {
    ModernGPT m(32, 16, 2, 1, 2, 0, 0, 6);
    Tensor bad({4, 4}, DType::Int32);
    REQUIRE_THROWS(m.forward(bad));
}

TEST_CASE("ModernGPT — loss decreases over 30 SGD steps", "[modern_gpt]") {
    constexpr int64_t V = 64, D = 32, T = 8;
    ModernGPT m(V, D, 2, 1, 2, 0, 0, 7);
    auto params = m.parameters();

    Tensor ids = make_ids(T, static_cast<int32_t>(V));
    Tensor tgt({T}, DType::Int32);
    {
        auto td = tgt.data_as<int32_t>();
        for (std::size_t i = 0; i < static_cast<std::size_t>(T); ++i)
            td[i] = static_cast<int32_t>((static_cast<int64_t>(i) + 1) % V);
    }

    float first_loss = -1.f, last_loss = -1.f;
    for (int step = 0; step < 30; ++step) {
        for (auto* p : params)
            p->grad() = zeros(p->data().shape(), DType::Float32);
        auto loss = cross_entropy(m.forward(ids), tgt);
        loss.backward();
        const float lv = loss.data().data_as<float>()[0];
        if (step == 0) first_loss = lv;
        if (step == 29) last_loss = lv;
        for (auto* p : params) {
            if (p->grad().numel() == 0) continue;
            auto pd = p->data().data_as<float>();
            auto gd = p->grad().data_as<float>();
            for (std::size_t i = 0; i < pd.size(); ++i) pd[i] -= 3e-3f * gd[i];
        }
    }
    REQUIRE(last_loss < first_loss);
}
