#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "sub0llm/autograd/ops.hpp"
#include "sub0llm/autograd/variable.hpp"
#include "sub0llm/nn/dpo.hpp"
#include "sub0llm/nn/modern_gpt.hpp"
#include <cmath>
#include <vector>

using Catch::Matchers::WithinAbs;
using namespace sub0llm;
using namespace sub0llm::autograd;
using namespace sub0llm::nn;

TEST_CASE("log_sigmoid - forward values") {
    auto eval_log_sig = [](float x_val) -> float {
        Tensor xd({1}, DType::Float32);
        xd.data_as<float>()[0] = x_val;
        Variable xv(std::move(xd), false);
        return log_sigmoid(xv).data().data_as<float>()[0];
    };

    // log_sigmoid(0) = log(0.5) = -log(2)
    REQUIRE_THAT(eval_log_sig(0.0f), WithinAbs(-std::log(2.0f), 1e-5f));

    // log_sigmoid(10) ≈ 0 (near-zero)
    REQUIRE_THAT(eval_log_sig(10.0f), WithinAbs(0.0f, 1e-4f));

    // log_sigmoid(-10) ≈ -10
    REQUIRE_THAT(eval_log_sig(-10.0f), WithinAbs(-10.0f, 1e-2f));
}

TEST_CASE("log_sigmoid - backward") {
    Tensor xd({1}, DType::Float32);
    xd.data_as<float>()[0] = 0.5f;
    const float eps = 1e-3f;

    auto eval = [&](float delta) {
        Tensor xdc({1}, DType::Float32);
        xdc.data_as<float>()[0] = xd.data_as<float>()[0] + delta;
        Variable xv(std::move(xdc), false);
        return log_sigmoid(xv).data().data_as<float>()[0];
    };
    const float ng = (eval(eps) - eval(-eps)) / (2.f * eps);

    Variable xv(xd, true);
    log_sigmoid(xv).backward();
    REQUIRE_THAT(xv.grad().data_as<float>()[0], WithinAbs(ng, 1e-3f));
}

TEST_CASE("log_prob_sequence - throws on T<2") {
    // Build trivial logits (1, V) - T=1 should throw
    Tensor ld({1, 4}, DType::Float32);
    auto ls = ld.data_as<float>();
    for (std::size_t i = 0; i < ls.size(); ++i) ls[i] = 0.25f;
    Variable logits(std::move(ld), false);

    Tensor ids({1}, DType::Int32);
    ids.data_as<int32_t>()[0] = 0;

    REQUIRE_THROWS_AS(log_prob_sequence(logits, ids), std::runtime_error);
}

TEST_CASE("log_prob_sequence - returns scalar") {
    const int64_t T = 4;
    const int64_t V = 8;
    Tensor ld({T, V}, DType::Float32);
    {
        auto ls = ld.data_as<float>();
        for (std::size_t i = 0; i < ls.size(); ++i)
            ls[i] = static_cast<float>(i % V) * 0.1f;
    }
    Variable logits(std::move(ld), false);

    Tensor ids({T}, DType::Int32);
    {
        auto is = ids.data_as<int32_t>();
        for (int64_t i = 0; i < T; ++i)
            is[static_cast<std::size_t>(i)] = static_cast<int32_t>(i % V);
    }

    Variable lp = log_prob_sequence(logits, ids);
    REQUIRE(lp.data().numel() == 1);
    REQUIRE(std::isfinite(lp.data().data_as<float>()[0]));
}

TEST_CASE("log_prob_sequence - backward works") {
    const int64_t T = 4;
    const int64_t V = 8;
    Tensor ld({T, V}, DType::Float32);
    {
        auto ls = ld.data_as<float>();
        for (std::size_t i = 0; i < ls.size(); ++i)
            ls[i] = static_cast<float>(i % V) * 0.1f;
    }
    Variable logits(std::move(ld), true);

    Tensor ids({T}, DType::Int32);
    {
        auto is = ids.data_as<int32_t>();
        for (int64_t i = 0; i < T; ++i)
            is[static_cast<std::size_t>(i)] = static_cast<int32_t>(i % V);
    }

    Variable lp = log_prob_sequence(logits, ids);
    REQUIRE(std::isfinite(lp.data().data_as<float>()[0]));
    lp.backward();
    // Gradient should have been computed - logits grad should exist
    REQUIRE(logits.grad().numel() == T * V);
}

TEST_CASE("dpo_loss - returns scalar") {
    const int64_t T = 4;
    const int64_t V = 8;

    auto make_logits = [&](bool rg) -> Variable {
        Tensor ld({T, V}, DType::Float32);
        auto ls = ld.data_as<float>();
        for (std::size_t i = 0; i < ls.size(); ++i)
            ls[i] = static_cast<float>(i % V) * 0.1f;
        return Variable(std::move(ld), rg);
    };

    auto make_ids = [&]() -> Tensor {
        Tensor ids({T}, DType::Int32);
        auto is = ids.data_as<int32_t>();
        for (int64_t i = 0; i < T; ++i)
            is[static_cast<std::size_t>(i)] = static_cast<int32_t>(i % V);
        return ids;
    };

    Variable logits_w = make_logits(true);
    Variable logits_l = make_logits(true);
    Tensor   ids_w    = make_ids();
    Tensor   ids_l    = make_ids();

    Variable loss = dpo_loss(logits_w, ids_w, logits_l, ids_l, -2.0f, -3.0f, 0.1f);

    REQUIRE(loss.data().numel() == 1);
    REQUIRE(std::isfinite(loss.data().data_as<float>()[0]));

    loss.backward();
    REQUIRE(logits_w.grad().numel() == T * V);
    REQUIRE(logits_l.grad().numel() == T * V);
}

TEST_CASE("dpo_loss - equal sequences give zero margin") {
    const int64_t T = 4;
    const int64_t V = 8;

    // Same logits for winner and loser
    Tensor ld({T, V}, DType::Float32);
    {
        auto ls = ld.data_as<float>();
        for (std::size_t i = 0; i < ls.size(); ++i)
            ls[i] = static_cast<float>(i % V) * 0.1f;
    }
    Variable logits_w(ld, false);
    Variable logits_l(ld, false);

    Tensor ids({T}, DType::Int32);
    {
        auto is = ids.data_as<int32_t>();
        for (int64_t i = 0; i < T; ++i)
            is[static_cast<std::size_t>(i)] = static_cast<int32_t>(i % V);
    }

    // Same reference log-probs, same sequences → margin = 0 → loss = log(2)
    Variable loss = dpo_loss(logits_w, ids, logits_l, ids, -2.0f, -2.0f, 0.1f);
    REQUIRE_THAT(loss.data().data_as<float>()[0],
                 WithinAbs(std::log(2.0f), 1e-4f));
}

TEST_CASE("dpo_loss - backward flows to model params") {
    const int64_t V = 16;
    const int64_t D = 16;

    ModernGPT policy(V, D, 2, 1, 1, 0, 0, /*seed=*/42);

    // Winner sequence: [0,1,2,3,4,5]
    Tensor ids_w({6}, DType::Int32);
    {
        auto is = ids_w.data_as<int32_t>();
        for (int i = 0; i < 6; ++i) is[static_cast<std::size_t>(i)] = i;
    }
    // Loser sequence: [5,4,3,2,1,0]
    Tensor ids_l({6}, DType::Int32);
    {
        auto is = ids_l.data_as<int32_t>();
        for (int i = 0; i < 6; ++i) is[static_cast<std::size_t>(i)] = 5 - i;
    }

    auto logits_w = policy.forward(ids_w);
    auto logits_l = policy.forward(ids_l);

    Variable loss = dpo_loss(logits_w, ids_w, logits_l, ids_l, -5.0f, -6.0f, 0.1f);
    REQUIRE(std::isfinite(loss.data().data_as<float>()[0]));

    loss.backward();

    // At least one parameter should have a non-zero gradient
    bool any_grad = false;
    for (auto* p : policy.parameters()) {
        if (p->grad().numel() > 0) {
            const auto gs = p->grad().data_as<float>();
            for (float g : gs) {
                if (g != 0.0f) { any_grad = true; break; }
            }
        }
        if (any_grad) break;
    }
    REQUIRE(any_grad);
}
