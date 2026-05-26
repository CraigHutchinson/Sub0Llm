#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "sub0llm/nn/distillation.hpp"
#include "sub0llm/core/tensor.hpp"
#include "sub0llm/autograd/variable.hpp"
#include "sub0llm/autograd/ops.hpp"
#include <cmath>
#include <vector>

using Catch::Matchers::WithinAbs;
using namespace sub0llm;
using namespace sub0llm::autograd;
using namespace sub0llm::nn;

// ── soft_cross_entropy tests ──────────────────────────────────────────────────

TEST_CASE("soft_cross_entropy — uniform teacher probs give approx log(V)") {
    // With uniform teacher distribution (1/V each), soft CE = log(V)
    // because: -sum_v (1/V) * log_softmax(logits)[v] = -avg log_softmax = log(V) when logits uniform
    constexpr int64_t N = 4, V = 8;

    Tensor student_data({N, V}, DType::Float32);
    {
        auto sp = student_data.data_as<float>();
        for (std::size_t i = 0; i < sp.size(); ++i) sp[i] = 0.0f;  // uniform logits
    }
    Variable student(std::move(student_data), false);

    Tensor teacher({N, V}, DType::Float32);
    {
        auto tp = teacher.data_as<float>();
        for (std::size_t i = 0; i < tp.size(); ++i) tp[i] = 1.0f / static_cast<float>(V);
    }

    Variable loss = soft_cross_entropy(student, teacher);
    REQUIRE(loss.data().numel() == 1);
    float val = loss.data().data_as<float>()[0];
    // Should equal log(V)
    REQUIRE_THAT(val, WithinAbs(std::log(static_cast<float>(V)), 1e-4f));
}

TEST_CASE("soft_cross_entropy — identical distributions gives lower loss") {
    // When teacher_probs = softmax(same logits as student), soft CE < log(V)
    constexpr int64_t N = 4, V = 8;

    Tensor logit_data({N, V}, DType::Float32);
    {
        auto sp = logit_data.data_as<float>();
        for (int64_t i = 0; i < N; ++i)
            for (int64_t j = 0; j < V; ++j)
                sp[static_cast<std::size_t>(i * V + j)] = static_cast<float>(j) * 0.5f;
    }

    // Student logits
    Variable student(logit_data, false);

    // Teacher probs = softmax(same logits)
    Variable teacher_var(logit_data, false);
    Variable soft_probs = softmax(teacher_var);
    Tensor teacher_probs = soft_probs.data();

    Variable loss = soft_cross_entropy(student, teacher_probs);
    float val = loss.data().data_as<float>()[0];

    REQUIRE(std::isfinite(val));
    REQUIRE(val < std::log(static_cast<float>(V)));
}

TEST_CASE("soft_cross_entropy — gradient flows to student") {
    constexpr int64_t N = 4, V = 8;

    Tensor student_data({N, V}, DType::Float32);
    {
        auto sp = student_data.data_as<float>();
        for (std::size_t i = 0; i < sp.size(); ++i)
            sp[i] = static_cast<float>(i % V) * 0.1f;
    }
    Variable student(std::move(student_data), true);

    Tensor teacher({N, V}, DType::Float32);
    {
        auto tp = teacher.data_as<float>();
        for (std::size_t i = 0; i < tp.size(); ++i) tp[i] = 1.0f / static_cast<float>(V);
    }

    Variable loss = soft_cross_entropy(student, teacher);
    loss.backward();

    REQUIRE(student.grad().numel() == N * V);
    bool any_nonzero = false;
    for (float g : student.grad().data_as<float>()) {
        if (g != 0.0f) { any_nonzero = true; break; }
    }
    REQUIRE(any_nonzero);
}

TEST_CASE("soft_cross_entropy — shape mismatch throws") {
    constexpr int64_t N = 4, M = 3, V = 8;

    Tensor student_data({N, V}, DType::Float32);
    Variable student(std::move(student_data), false);

    Tensor teacher({M, V}, DType::Float32);  // M != N

    REQUIRE_THROWS_AS(soft_cross_entropy(student, teacher), std::runtime_error);
}

// ── distillation_loss tests ───────────────────────────────────────────────────

TEST_CASE("distillation_loss — alpha=1 equals hard CE") {
    constexpr int64_t N = 4, V = 8;

    Tensor logit_data({N, V}, DType::Float32);
    {
        auto sp = logit_data.data_as<float>();
        for (std::size_t i = 0; i < sp.size(); ++i)
            sp[i] = static_cast<float>(i % V) * 0.2f;
    }

    Tensor teacher_data({N, V}, DType::Float32);
    {
        auto tp = teacher_data.data_as<float>();
        for (std::size_t i = 0; i < tp.size(); ++i) tp[i] = 1.0f / static_cast<float>(V);
    }

    Tensor targets({N}, DType::Int32);
    {
        auto ts = targets.data_as<int32_t>();
        for (int64_t i = 0; i < N; ++i) ts[static_cast<std::size_t>(i)] = static_cast<int32_t>(i % V);
    }

    Variable student1(logit_data, false);
    Variable student2(logit_data, false);

    Variable dist = distillation_loss(student1, teacher_data, targets, 1.0f, 2.0f);
    Variable hard = cross_entropy(student2, targets);

    float dist_val = dist.data().data_as<float>()[0];
    float hard_val = hard.data().data_as<float>()[0];

    REQUIRE_THAT(dist_val, WithinAbs(hard_val, 1e-5f));
}

TEST_CASE("distillation_loss — alpha=0 uses only soft term") {
    constexpr int64_t N = 4, V = 8;

    Tensor logit_data({N, V}, DType::Float32);
    {
        auto sp = logit_data.data_as<float>();
        for (std::size_t i = 0; i < sp.size(); ++i)
            sp[i] = static_cast<float>(i % V) * 0.2f;
    }

    Tensor teacher_data({N, V}, DType::Float32);
    {
        auto tp = teacher_data.data_as<float>();
        for (std::size_t i = 0; i < tp.size(); ++i)
            tp[i] = static_cast<float>(i % V + 1) * 0.3f;
    }

    Tensor targets({N}, DType::Int32);
    {
        auto ts = targets.data_as<int32_t>();
        for (int64_t i = 0; i < N; ++i) ts[static_cast<std::size_t>(i)] = static_cast<int32_t>(i % V);
    }

    Variable student1(logit_data, false);
    Variable student2(logit_data, false);

    Variable soft_only = distillation_loss(student1, teacher_data, targets, 0.0f, 2.0f);
    Variable hard_only = cross_entropy(student2, targets);

    float soft_val = soft_only.data().data_as<float>()[0];
    float hard_val = hard_only.data().data_as<float>()[0];

    // They should differ since teacher != one-hot hard targets
    REQUIRE(std::isfinite(soft_val));
    REQUIRE(std::abs(soft_val - hard_val) > 1e-6f);
}

TEST_CASE("distillation_loss — invalid alpha throws") {
    constexpr int64_t N = 2, V = 4;

    Tensor logit_data({N, V}, DType::Float32);
    Tensor teacher_data({N, V}, DType::Float32);
    Tensor targets({N}, DType::Int32);

    Variable student1(logit_data, false);
    Variable student2(logit_data, false);

    REQUIRE_THROWS_AS(distillation_loss(student1, teacher_data, targets, -0.1f, 2.0f),
                      std::runtime_error);
    REQUIRE_THROWS_AS(distillation_loss(student2, teacher_data, targets, 1.1f, 2.0f),
                      std::runtime_error);
}

TEST_CASE("distillation_loss — invalid temp throws") {
    constexpr int64_t N = 2, V = 4;

    Tensor logit_data({N, V}, DType::Float32);
    Tensor teacher_data({N, V}, DType::Float32);
    Tensor targets({N}, DType::Int32);

    Variable student1(logit_data, false);
    Variable student2(logit_data, false);

    REQUIRE_THROWS_AS(distillation_loss(student1, teacher_data, targets, 0.5f, 0.0f),
                      std::runtime_error);
    REQUIRE_THROWS_AS(distillation_loss(student2, teacher_data, targets, 0.5f, -1.0f),
                      std::runtime_error);
}

TEST_CASE("distillation_loss — gradient flows to student") {
    constexpr int64_t N = 4, V = 8;

    Tensor logit_data({N, V}, DType::Float32);
    {
        auto sp = logit_data.data_as<float>();
        for (std::size_t i = 0; i < sp.size(); ++i)
            sp[i] = static_cast<float>(i % V) * 0.15f;
    }

    Tensor teacher_data({N, V}, DType::Float32);
    {
        auto tp = teacher_data.data_as<float>();
        for (std::size_t i = 0; i < tp.size(); ++i) tp[i] = 1.0f / static_cast<float>(V);
    }

    Tensor targets({N}, DType::Int32);
    {
        auto ts = targets.data_as<int32_t>();
        for (int64_t i = 0; i < N; ++i) ts[static_cast<std::size_t>(i)] = static_cast<int32_t>(i % V);
    }

    Variable student(std::move(logit_data), true);
    Variable loss = distillation_loss(student, teacher_data, targets, 0.5f, 2.0f);
    loss.backward();

    REQUIRE(student.grad().numel() == N * V);
    bool any_nonzero = false;
    for (float g : student.grad().data_as<float>()) {
        if (g != 0.0f) { any_nonzero = true; break; }
    }
    REQUIRE(any_nonzero);
}
