#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "sub0llm/autograd/ops.hpp"
#include "sub0llm/autograd/variable.hpp"
#include "sub0llm/core/ops.hpp"
#include "sub0llm/nn/modern_gpt.hpp"
#include "sub0llm/nn/scheduler.hpp"
#include "sub0llm/nn/trainer.hpp"

#include <cmath>
#include <vector>

using Catch::Matchers::WithinAbs;
using namespace sub0llm;
using namespace sub0llm::autograd;
using namespace sub0llm::nn;
namespace ops = sub0llm::ops;

// ── Helpers ───────────────────────────────────────────────────────────────────

static Tensor make_ids(int64_t T, int32_t V) {
    Tensor ids({T}, DType::Int32);
    auto d = ids.data_as<int32_t>();
    for (std::size_t i = 0; i < static_cast<std::size_t>(T); ++i)
        d[i] = static_cast<int32_t>(i % static_cast<std::size_t>(V));
    return ids;
}

// ── ops::narrow ───────────────────────────────────────────────────────────────

TEST_CASE("ops::narrow - 2D slice values", "[pretraining][narrow]") {
    Tensor t({4, 3}, DType::Float32);
    auto d = t.data_as<float>();
    for (std::size_t i = 0; i < 12u; ++i) d[i] = static_cast<float>(i);

    auto s = ops::narrow(t, 1, 2);
    REQUIRE(s.shape()[0] == 2);
    REQUIRE(s.shape()[1] == 3);

    const auto sd = s.data_as<float>();
    REQUIRE_THAT(sd[0], WithinAbs(3.f, 1e-6f));  // row 1 col 0
    REQUIRE_THAT(sd[1], WithinAbs(4.f, 1e-6f));  // row 1 col 1
    REQUIRE_THAT(sd[3], WithinAbs(6.f, 1e-6f));  // row 2 col 0
}

TEST_CASE("ops::narrow - 1D slice", "[pretraining][narrow]") {
    Tensor t({6}, DType::Float32);
    auto d = t.data_as<float>();
    for (std::size_t i = 0; i < 6u; ++i) d[i] = static_cast<float>(i) * 2.f;

    auto s = ops::narrow(t, 2, 3);
    REQUIRE(s.shape()[0] == 3);
    REQUIRE(s.ndim() == 1);
    const auto sd = s.data_as<float>();
    REQUIRE_THAT(sd[0], WithinAbs(4.f, 1e-6f));
    REQUIRE_THAT(sd[2], WithinAbs(8.f, 1e-6f));
}

TEST_CASE("ops::narrow - throws on bad bounds", "[pretraining][narrow]") {
    Tensor t({5, 2}, DType::Float32);
    REQUIRE_THROWS(ops::narrow(t, -1, 2));    // negative start
    REQUIRE_THROWS(ops::narrow(t, 3, 3));     // start+length > T
    REQUIRE_THROWS(ops::narrow(t, 0, 0));     // length == 0
    REQUIRE_THROWS(ops::narrow(t, 5, 1));     // start == T
}

// ── autograd::narrow ──────────────────────────────────────────────────────────

TEST_CASE("autograd::narrow - forward matches ops::narrow", "[pretraining][narrow]") {
    Tensor xd({5, 3}, DType::Float32);
    auto d = xd.data_as<float>();
    for (std::size_t i = 0; i < 15u; ++i) d[i] = static_cast<float>(i);

    Variable x(xd, false);
    auto y = narrow(x, 1, 3);
    REQUIRE(y.data().shape()[0] == 3);
    REQUIRE(y.data().shape()[1] == 3);

    const auto yd = y.data().data_as<float>();
    REQUIRE_THAT(yd[0], WithinAbs(3.f, 1e-6f));
    REQUIRE_THAT(yd[8], WithinAbs(11.f, 1e-6f));  // row 2 (=index 3), col 2
}

TEST_CASE("autograd::narrow - backward scatters to correct rows", "[pretraining][narrow]") {
    Tensor xd({5, 3}, DType::Float32);
    for (auto& v : xd.data_as<float>()) v = 1.f;

    Variable x(xd, true);
    sum(narrow(x, 1, 3)).backward();

    const auto gd = x.grad().data_as<float>();
    // rows 0 and 4 should have 0 grad; rows 1,2,3 should have 1 grad
    for (std::size_t c = 0; c < 3u; ++c) {
        REQUIRE_THAT(gd[c],           WithinAbs(0.f, 1e-6f));  // row 0
        REQUIRE_THAT(gd[3 + c],       WithinAbs(1.f, 1e-6f));  // row 1
        REQUIRE_THAT(gd[6 + c],       WithinAbs(1.f, 1e-6f));  // row 2
        REQUIRE_THAT(gd[9 + c],       WithinAbs(1.f, 1e-6f));  // row 3
        REQUIRE_THAT(gd[12 + c],      WithinAbs(0.f, 1e-6f));  // row 4
    }
}

TEST_CASE("autograd::narrow - backward numerical gradient check", "[pretraining][narrow]") {
    const float eps = 1e-3f;
    Tensor xd({4, 3}, DType::Float32);
    {
        auto d = xd.data_as<float>();
        for (std::size_t i = 0; i < 12u; ++i) d[i] = 0.1f * static_cast<float>(i + 1);
    }
    const auto xd_orig = xd.data_as<float>();

    for (std::size_t k = 0; k < 12u; ++k) {
        auto eval = [&](float delta) {
            Tensor xdc = zeros({4, 3});
            auto dc = xdc.data_as<float>();
            for (std::size_t i = 0; i < 12u; ++i) dc[i] = xd_orig[i];
            dc[k] += delta;
            Variable xv(std::move(xdc), false);
            auto y = narrow(xv, 1, 2);  // rows [1,2]
            float s = 0.f; for (float v : y.data().data_as<float>()) s += v;
            return s;
        };
        const float ng = (eval(eps) - eval(-eps)) / (2.f * eps);

        Variable xv(xd, true);
        sum(narrow(xv, 1, 2)).backward();
        REQUIRE_THAT(xv.grad().data_as<float>()[k], WithinAbs(ng, 1e-3f));
    }
}

TEST_CASE("autograd::narrow - throws on bad range", "[pretraining][narrow]") {
    Tensor xd({4, 2}, DType::Float32);
    Variable x(xd, false);
    REQUIRE_THROWS(narrow(x, -1, 2));
    REQUIRE_THROWS(narrow(x, 3, 2));   // 3+2=5 > 4
    REQUIRE_THROWS(narrow(x, 0, 0));
}

// ── ConstantLR ────────────────────────────────────────────────────────────────

TEST_CASE("ConstantLR - always returns lr", "[pretraining][scheduler]") {
    ConstantLR s(1e-3f);
    REQUIRE_THAT(s(0),   WithinAbs(1e-3f, 1e-7f));
    REQUIRE_THAT(s(100), WithinAbs(1e-3f, 1e-7f));
    REQUIRE_THAT(s(-5),  WithinAbs(1e-3f, 1e-7f));
    REQUIRE_THAT(s.peak_lr(), WithinAbs(1e-3f, 1e-7f));
}

TEST_CASE("ConstantLR - throws on non-positive lr", "[pretraining][scheduler]") {
    REQUIRE_THROWS([] { ConstantLR bad(0.f); }());
    REQUIRE_THROWS([] { ConstantLR bad(-1.f); }());
}

// ── CosineWithWarmup ──────────────────────────────────────────────────────────

TEST_CASE("CosineWithWarmup - warmup ramps linearly", "[pretraining][scheduler]") {
    CosineWithWarmup s(1e-3f, 1e-4f, 10, 100);
    // step 0: lr should be ~0 (0/10 * max_lr = 0)
    REQUIRE_THAT(s(0),  WithinAbs(0.f, 1e-7f));
    // step 5: half warmup
    REQUIRE_THAT(s(5),  WithinAbs(5e-4f, 1e-7f));
    // step 10: exactly max_lr
    REQUIRE_THAT(s(10), WithinAbs(1e-3f, 2e-6f));
}

TEST_CASE("CosineWithWarmup - cosine decay at midpoint", "[pretraining][scheduler]") {
    CosineWithWarmup s(1e-3f, 1e-4f, 0, 100);
    // step 0: warmup=0 so immediately at max_lr
    REQUIRE_THAT(s(0),  WithinAbs(1e-3f, 1e-7f));
    // step 50: midpoint → lr = min + 0.5*(max-min)*(1+cos(pi*0.5)) = min + 0.5*(max-min)
    const float expected_mid = 1e-4f + 0.5f * (1e-3f - 1e-4f);
    REQUIRE_THAT(s(50), WithinAbs(expected_mid, 1e-6f));
    // step 100: at/past end → min_lr
    REQUIRE_THAT(s(100), WithinAbs(1e-4f, 1e-6f));
    REQUIRE_THAT(s(200), WithinAbs(1e-4f, 1e-6f));
}

TEST_CASE("CosineWithWarmup - lr is monotone in cosine phase", "[pretraining][scheduler]") {
    CosineWithWarmup s(1e-3f, 0.f, 5, 50);
    float prev = s(5);
    for (int64_t step = 6; step <= 50; ++step) {
        const float cur = s(step);
        REQUIRE(cur <= prev + 1e-7f);  // non-increasing
        prev = cur;
    }
}

TEST_CASE("CosineWithWarmup - throws on invalid args", "[pretraining][scheduler]") {
    REQUIRE_THROWS([] { CosineWithWarmup bad(0.f,  0.f, 5, 100); }()); // max_lr=0
    REQUIRE_THROWS([] { CosineWithWarmup bad(1e-3f, -1.f, 5, 100); }()); // min_lr<0
    REQUIRE_THROWS([] { CosineWithWarmup bad(1e-3f, 2e-3f, 5, 100); }()); // min>max
    REQUIRE_THROWS([] { CosineWithWarmup bad(1e-3f, 0.f, -1, 100); }()); // warmup<0
    REQUIRE_THROWS([] { CosineWithWarmup bad(1e-3f, 0.f, 5, 0); }()); // total=0
    REQUIRE_THROWS([] { CosineWithWarmup bad(1e-3f, 0.f, 5, 5); }()); // warmup==total
    REQUIRE_THROWS([] { CosineWithWarmup bad(1e-3f, 0.f, 50, 1); }()); // warmup>total, total==1
}

TEST_CASE("CosineWithWarmup - valid boundary: warmup=0, total=1", "[pretraining][scheduler]") {
    // warmup_steps=0 < total_steps=1 is valid (no cosine phase, goes straight to min_lr)
    CosineWithWarmup s(1e-3f, 1e-4f, 0, 1);
    REQUIRE_THAT(s(0), WithinAbs(1e-3f, 1e-6f)); // step 0 at or past total → cosine(0) = max_lr
    REQUIRE_THAT(s(1), WithinAbs(1e-4f, 1e-6f)); // step 1 >= total_steps → min_lr
}

// ── mtp_cross_entropy ─────────────────────────────────────────────────────────

TEST_CASE("mtp_cross_entropy - single head (n_mtp=0) matches plain CE", "[pretraining][mtp]") {
    constexpr int64_t V = 16, T = 6;
    ModernGPT m(V, 16, 2, 1, 1, 0, 0, 1);
    Tensor ids = make_ids(T, V);

    auto heads  = m.forward_mtp(ids);
    REQUIRE(heads.size() == 1u);

    // mtp_cross_entropy with one head (k=0, predicts t+1 for t=0..T-2)
    // Plain cross_entropy on logits[0..T-2] vs ids[1..T-1]
    auto mtp_l = mtp_cross_entropy(heads, ids);

    // Manual: narrow to T-1 rows, targets = ids[1..T-1]
    auto logits_trunc = narrow(heads[0], 0, T - 1);
    Tensor tgt({T - 1}, DType::Int32);
    {
        const auto src = ids.data_as<int32_t>();
        auto       dst = tgt.data_as<int32_t>();
        for (std::size_t i = 0; i + 1 < static_cast<std::size_t>(T); ++i)
            dst[i] = src[i + 1];
    }
    auto plain_l = cross_entropy(logits_trunc, tgt);

    // mtp_cross_entropy default weight = 1/1 = 1.0, same as plain CE
    REQUIRE_THAT(mtp_l.data().data_as<float>()[0],
                 WithinAbs(plain_l.data().data_as<float>()[0], 1e-4f));
}

TEST_CASE("mtp_cross_entropy - multiple heads produce scalar loss", "[pretraining][mtp]") {
    constexpr int64_t V = 16, T = 8, Nmtp = 2;
    ModernGPT m(V, 16, 2, 1, 1, 0, Nmtp, 2);
    Tensor ids = make_ids(T, V);

    auto heads = m.forward_mtp(ids);
    auto loss  = mtp_cross_entropy(heads, ids);

    REQUIRE(loss.data().ndim() == 1);
    REQUIRE(loss.data().numel() == 1);
    REQUIRE(std::isfinite(loss.data().data_as<float>()[0]));
}

TEST_CASE("mtp_cross_entropy - custom weights scale loss", "[pretraining][mtp]") {
    constexpr int64_t V = 16, T = 6, Nmtp = 1;
    ModernGPT m(V, 16, 2, 1, 1, 0, Nmtp, 3);
    Tensor ids = make_ids(T, V);

    auto heads = m.forward_mtp(ids);

    // Equal weights = {0.5, 0.5}
    auto loss_eq = mtp_cross_entropy(heads, ids, {0.5f, 0.5f});
    // Main-only weight = {1.0, 0.0}
    auto loss_main = mtp_cross_entropy(heads, ids, {1.0f, 0.0f});

    // With weight {1,0} we should get the same as CE on main head alone.
    auto logits_trunc = narrow(heads[0], 0, T - 1);
    Tensor tgt({T - 1}, DType::Int32);
    {
        const auto src = ids.data_as<int32_t>();
        auto       dst = tgt.data_as<int32_t>();
        for (std::size_t i = 0; i + 1 < static_cast<std::size_t>(T); ++i)
            dst[i] = src[i + 1];
    }
    auto plain = cross_entropy(logits_trunc, tgt);

    REQUIRE_THAT(loss_main.data().data_as<float>()[0],
                 WithinAbs(plain.data().data_as<float>()[0], 1e-4f));

    // Equal-weight loss should differ from main-only.
    REQUIRE(std::abs(loss_eq.data().data_as<float>()[0] -
                     plain.data().data_as<float>()[0]) > 1e-5f);
}

TEST_CASE("mtp_cross_entropy - backward flows through all heads", "[pretraining][mtp]") {
    constexpr int64_t V = 16, T = 6, Nmtp = 2;
    ModernGPT m(V, 16, 2, 1, 1, 0, Nmtp, 4);
    Tensor ids = make_ids(T, V);

    auto heads = m.forward_mtp(ids);
    auto loss  = mtp_cross_entropy(heads, ids);
    loss.backward();

    auto params = m.parameters();
    int64_t n_grad = 0;
    for (auto* p : params) if (p->grad().numel() > 0) ++n_grad;
    REQUIRE(n_grad == static_cast<int64_t>(params.size()));
}

TEST_CASE("mtp_cross_entropy - throws on empty heads", "[pretraining][mtp]") {
    Tensor ids = make_ids(5, 8);
    REQUIRE_THROWS(mtp_cross_entropy({}, ids));
}

TEST_CASE("mtp_cross_entropy - throws on T < 2", "[pretraining][mtp]") {
    constexpr int64_t V = 8;
    ModernGPT m(V, 8, 2, 1, 1, 0, 0, 5);
    Tensor ids = make_ids(1, V);
    auto heads = m.forward_mtp(ids);
    REQUIRE_THROWS(mtp_cross_entropy(heads, ids));
}

TEST_CASE("mtp_cross_entropy - wrong weights size throws", "[pretraining][mtp]") {
    constexpr int64_t V = 8, T = 4;
    ModernGPT m(V, 8, 2, 1, 1, 0, 1, 6);
    Tensor ids = make_ids(T, V);
    auto heads = m.forward_mtp(ids);
    // heads.size() == 2, passing 3 weights should throw
    REQUIRE_THROWS(mtp_cross_entropy(heads, ids, {1.f, 0.5f, 0.1f}));
}

// ── Integration: training loop converges ─────────────────────────────────────

TEST_CASE("Pretraining loop - loss decreases with MTP", "[pretraining][integration]") {
    constexpr int64_t V = 32, T = 10, Nmtp = 1;
    ModernGPT model(V, 32, 2, 1, 2, 0, Nmtp, 99);
    CosineWithWarmup sched(3e-3f, 3e-4f, 3, 30);
    auto params = model.parameters();

    Tensor ids = make_ids(T, V);

    float first_loss = -1.f, last_loss = -1.f;

    for (int64_t step = 0; step < 30; ++step) {
        for (auto* p : params) p->grad() = zeros(p->data().shape(), DType::Float32);

        auto heads = model.forward_mtp(ids);
        auto loss  = mtp_cross_entropy(heads, ids);
        loss.backward();

        const float lr = sched(step);
        const float lv = loss.data().data_as<float>()[0];
        if (step == 0)  first_loss = lv;
        if (step == 29) last_loss  = lv;

        for (auto* p : params) {
            if (p->grad().numel() == 0) continue;
            auto pd = p->data().data_as<float>();
            auto gd = p->grad().data_as<float>();
            for (std::size_t i = 0; i < pd.size(); ++i) pd[i] -= lr * gd[i];
        }
    }

    REQUIRE(last_loss < first_loss);
}
