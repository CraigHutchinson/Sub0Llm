#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "sub0llm/nn/lora.hpp"
#include "sub0llm/nn/optimizer.hpp"
#include "sub0llm/autograd/ops.hpp"
#include "sub0llm/autograd/variable.hpp"
#include "sub0llm/core/tensor.hpp"

#include <cmath>
#include <vector>

using Catch::Matchers::WithinAbs;
using namespace sub0llm;
using namespace sub0llm::nn;
using namespace sub0llm::autograd;

// ── Helpers ───────────────────────────────────────────────────────────────────

static Variable make_input(int64_t rows, int64_t cols, float fill = 1.0f) {
    Tensor t = zeros({rows, cols}, DType::Float32);
    auto   s = t.data_as<float>();
    for (std::size_t i = 0; i < s.size(); ++i) s[i] = fill;
    return Variable(std::move(t), /*requires_grad=*/false);
}

// ── Tests ─────────────────────────────────────────────────────────────────────

TEST_CASE("LoRALinear - throws on bad args") {
    CHECK_THROWS_AS(LoRALinear(0, 4, 0),  std::runtime_error);  // in <= 0
    CHECK_THROWS_AS(LoRALinear(-1, 4, 0), std::runtime_error);  // in < 0
    CHECK_THROWS_AS(LoRALinear(4, 0, 0),  std::runtime_error);  // out <= 0
    CHECK_THROWS_AS(LoRALinear(4, 4, -1), std::runtime_error);  // rank < 0
    CHECK_THROWS_AS(LoRALinear(4, 4, 5),  std::runtime_error);  // rank > min(in,out)
    CHECK_THROWS_AS(LoRALinear(4, 8, 5),  std::runtime_error);  // rank > min(4,8)=4
    CHECK_THROWS_AS(LoRALinear(4, 4, 2, 0.0f), std::runtime_error);  // alpha <= 0
    CHECK_THROWS_AS(LoRALinear(4, 4, 2, -1.0f), std::runtime_error); // alpha < 0
}

TEST_CASE("LoRALinear - rank=0 is frozen linear") {
    LoRALinear layer(8, 4, /*rank=*/0);

    CHECK(layer.lora_parameters().empty());
    CHECK(layer.all_parameters().size() == 1);
    CHECK_FALSE(layer.all_parameters()[0]->requires_grad());

    auto x   = make_input(3, 8);
    auto out = layer.forward(x);
    CHECK(out.data().shape()[0] == 3);
    CHECK(out.data().shape()[1] == 4);
}

TEST_CASE("LoRALinear - delta is zero at init") {
    // B is initialised to zeros, so ΔW = (alpha/r)*A@B = 0 regardless of A.
    // Therefore forward(x) == x @ W^T at initialisation.
    LoRALinear layer(8, 4, /*rank=*/2, /*alpha=*/1.0f, /*seed=*/42);

    auto params = layer.all_parameters();
    REQUIRE(params.size() == 3);

    // Verify parameter layout.
    CHECK_FALSE(params[0]->requires_grad());  // base
    CHECK(params[1]->requires_grad());        // A
    CHECK(params[2]->requires_grad());        // B

    // B must be exactly zero.
    const auto b_data = params[2]->data().data_as<float>();
    for (float v : b_data) CHECK(v == 0.0f);
}

TEST_CASE("LoRALinear - base has no gradient after backward") {
    LoRALinear layer(8, 4, /*rank=*/2);
    auto x   = make_input(3, 8);
    auto out = layer.forward(x);
    sum(out).backward();

    // Base weight: requires_grad=false → should accumulate no gradient.
    CHECK(layer.all_parameters()[0]->grad().numel() == 0);
}

TEST_CASE("LoRALinear - lora params have gradients after backward") {
    LoRALinear layer(8, 4, /*rank=*/2);
    auto x   = make_input(3, 8);
    auto out = layer.forward(x);
    sum(out).backward();

    auto params = layer.all_parameters();
    REQUIRE(params.size() == 3);

    CHECK(params[1]->grad().numel() > 0);  // A
    CHECK(params[2]->grad().numel() > 0);  // B
}

TEST_CASE("LoRALinear - lora_parameters count") {
    {
        LoRALinear layer(8, 4, /*rank=*/4);
        CHECK(layer.lora_parameters().size() == 2);
    }
    {
        LoRALinear layer(8, 4, /*rank=*/0);
        CHECK(layer.lora_parameters().size() == 0);
    }
}

TEST_CASE("LoRALinear - grad norm is non-zero for A and B after backward") {
    // Sanity: after a real backward pass with non-trivial input, A and B
    // accumulate non-zero gradients (B gets grad through the A path since
    // upstream != 0; A gets grad once B is updated - but here B starts at 0
    // so dL/dA = upstream @ B^T which is 0.  That is expected at init.)
    //
    // To get a non-trivial A gradient, perturb B before the forward pass.
    LoRALinear layer(4, 3, /*rank=*/2, /*alpha=*/1.0f, /*seed=*/7);

    // Manually set B to something non-zero.
    auto params = layer.all_parameters();
    {
        auto b_data = params[2]->data().data_as<float>();
        for (std::size_t i = 0; i < b_data.size(); ++i)
            b_data[i] = static_cast<float>(i + 1) * 0.1f;
    }

    auto x   = make_input(2, 4, 0.5f);
    auto out = layer.forward(x);
    sum(out).backward();

    // Both A and B should now have gradients.
    float norm_A = 0.0f, norm_B = 0.0f;
    for (float v : params[1]->grad().data_as<float>()) norm_A += v * v;
    for (float v : params[2]->grad().data_as<float>()) norm_B += v * v;
    CHECK(norm_A > 0.0f);
    CHECK(norm_B > 0.0f);
}

TEST_CASE("LoRALinear - fine-tuning decreases loss") {
    // Simple regression-style task via cross_entropy:
    // embed integers → LoRALinear → logits → cross_entropy.
    // Train only lora_parameters() and verify loss decreases.
    constexpr int64_t T  = 8;
    constexpr int64_t V  = 16;
    constexpr int64_t D  = 32;
    constexpr int64_t R  = 4;
    constexpr float   LR = 1e-2f;
    constexpr int     Steps = 20;

    // Fixed embedding matrix (frozen).
    Tensor emb_data = zeros({V, D}, DType::Float32);
    {
        auto s = emb_data.data_as<float>();
        for (std::size_t i = 0; i < s.size(); ++i)
            s[i] = static_cast<float>(i % 7) * 0.1f - 0.3f;
    }
    Variable emb_weight(emb_data, /*requires_grad=*/false);

    // Token ids: 0..T-1 mod V.
    Tensor ids({T}, DType::Int32);
    {
        auto d = ids.data_as<int32_t>();
        for (std::size_t i = 0; i < static_cast<std::size_t>(T); ++i)
            d[i] = static_cast<int32_t>(i % V);
    }

    // Targets: ids shifted by 1 (next-token prediction).
    Tensor targets({T}, DType::Int32);
    {
        const auto src = ids.data_as<int32_t>();
        auto       dst = targets.data_as<int32_t>();
        for (std::size_t i = 0; i < static_cast<std::size_t>(T); ++i)
            dst[i] = src[(i + 1) % static_cast<std::size_t>(T)];
    }

    LoRALinear layer(D, V, R, /*alpha=*/1.0f, /*seed=*/11);
    SGD        optim(layer.lora_parameters(), LR);

    float first_loss = -1.0f, last_loss = -1.0f;

    for (int step = 0; step < Steps; ++step) {
        optim.zero_grad();

        // Manual embedding lookup (gather rows).
        Tensor x_data = zeros({T, D}, DType::Float32);
        {
            const auto emb = emb_data.data_as<float>();
            auto       xd  = x_data.data_as<float>();
            const auto id  = ids.data_as<int32_t>();
            for (std::size_t i = 0; i < static_cast<std::size_t>(T); ++i) {
                const std::size_t row = static_cast<std::size_t>(id[i]);
                for (std::size_t j = 0; j < static_cast<std::size_t>(D); ++j)
                    xd[i * static_cast<std::size_t>(D) + j] =
                        emb[row * static_cast<std::size_t>(D) + j];
            }
        }
        Variable x(x_data, /*requires_grad=*/false);

        auto logits = layer.forward(x);
        auto loss   = cross_entropy(logits, targets);
        loss.backward();
        optim.step();

        const float lv = loss.data().data_as<float>()[0];
        if (step == 0)        first_loss = lv;
        if (step == Steps - 1) last_loss = lv;
    }

    CHECK(last_loss < first_loss);
}
