#include "sub0llm/nn/rlhf.hpp"
#include "sub0llm/nn/optimizer.hpp"
#include "sub0llm/autograd/ops.hpp"
#include "sub0llm/core/tensor.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <cmath>

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

static Tensor make_logits(int64_t T, int64_t V, float fill = 0.0f) {
    Tensor t({T, V}, DType::Float32);
    for (auto& f : t.data_as<float>()) f = fill;
    return t;
}

// ── RewardModel ───────────────────────────────────────────────────────────────

TEST_CASE("RewardModel - forward output shape (T, 1)") {
    RewardModel rm(20, 16, 2, 1, 2);
    auto ids = make_ids({0, 1, 2, 3});
    auto out = rm.forward(ids);
    REQUIRE(out.data().shape(0) == 4);
    REQUIRE(out.data().shape(1) == 1);
}

TEST_CASE("RewardModel - score returns scalar") {
    RewardModel rm(20, 16, 2, 1, 2);
    auto ids = make_ids({0, 1, 2, 3});
    auto s = rm.score(ids);
    REQUIRE(s.data().numel() == 1);
    REQUIRE(std::isfinite(s.data().data_as<float>()[0]));
}

TEST_CASE("RewardModel - accessors") {
    RewardModel rm(30, 24, 2, 1, 3);
    REQUIRE(rm.vocab_size() == 30);
    REQUIRE(rm.embed_dim()  == 24);
    REQUIRE(rm.num_layers() == 3);
}

TEST_CASE("RewardModel - invalid construction throws") {
    REQUIRE_THROWS_AS(RewardModel(0,  16, 2, 1, 2), std::runtime_error);  // vocab=0
    REQUIRE_THROWS_AS(RewardModel(20,  0, 2, 1, 2), std::runtime_error);  // embed=0
    REQUIRE_THROWS_AS(RewardModel(20, 16, 0, 1, 2), std::runtime_error);  // heads=0
    REQUIRE_THROWS_AS(RewardModel(20, 16, 2, 0, 2), std::runtime_error);  // kv_heads=0
    REQUIRE_THROWS_AS(RewardModel(20, 16, 2, 1, 0), std::runtime_error);  // layers=0
    REQUIRE_THROWS_AS(RewardModel(20, 16, 4, 3, 2), std::runtime_error);  // 4 % 3 != 0
}

TEST_CASE("RewardModel - forward rejects non-1D ids") {
    RewardModel rm(20, 16, 2, 1, 2);
    Tensor ids2d({3, 4}, DType::Int32);
    REQUIRE_THROWS_AS(rm.forward(ids2d), std::runtime_error);
}

TEST_CASE("RewardModel - gradient flows to parameters") {
    RewardModel rm(20, 16, 2, 1, 1, 0, 42);
    auto ids = make_ids({0, 1, 2, 3});
    auto score = rm.score(ids);
    score.backward();

    bool any_grad = false;
    for (auto* v : rm.parameters()) {
        if (v->grad().numel() == 0) continue;
        for (auto f : v->grad().data_as<float>())
            if (f != 0.f) { any_grad = true; break; }
        if (any_grad) break;
    }
    REQUIRE(any_grad);
}

// ── reward_preference_loss ────────────────────────────────────────────────────

TEST_CASE("reward_preference_loss - scalar Variable returned") {
    Tensor rc({1}, DType::Float32); rc.data_as<float>()[0] =  1.0f;
    Tensor rr({1}, DType::Float32); rr.data_as<float>()[0] = -1.0f;
    Variable vc(rc, true), vr(rr, true);
    auto loss = reward_preference_loss(vc, vr);
    REQUIRE(loss.data().numel() == 1);
    REQUIRE(loss.data().data_as<float>()[0] > 0.f);
}

TEST_CASE("reward_preference_loss - margin=0 gives log(2) loss") {
    // r_chosen == r_rejected → L = -log σ(0) = log(2)
    Tensor rc({1}, DType::Float32); rc.data_as<float>()[0] = 0.f;
    Tensor rr({1}, DType::Float32); rr.data_as<float>()[0] = 0.f;
    Variable vc(rc, true), vr(rr, true);
    auto loss = reward_preference_loss(vc, vr);
    REQUIRE(loss.data().data_as<float>()[0] ==
            Catch::Approx(std::log(2.0f)).epsilon(1e-4));
}

TEST_CASE("reward_preference_loss - large positive margin gives low loss") {
    Tensor rc({1}, DType::Float32); rc.data_as<float>()[0] =  10.f;
    Tensor rr({1}, DType::Float32); rr.data_as<float>()[0] = -10.f;
    Variable vc(rc, false), vr(rr, false);
    auto loss = reward_preference_loss(vc, vr);
    REQUIRE(loss.data().data_as<float>()[0] < 0.001f);
}

TEST_CASE("reward_preference_loss - gradient flows to r_chosen and r_rejected") {
    Tensor rc({1}, DType::Float32); rc.data_as<float>()[0] = 0.5f;
    Tensor rr({1}, DType::Float32); rr.data_as<float>()[0] = -0.5f;
    Variable vc(rc, true), vr(rr, true);
    auto loss = reward_preference_loss(vc, vr);
    loss.backward();
    // Gradient w.r.t. r_chosen should be negative (increasing r_chosen reduces loss)
    REQUIRE(vc.grad().data_as<float>()[0] < 0.f);
    // Gradient w.r.t. r_rejected should be positive (increasing r_rejected increases loss)
    REQUIRE(vr.grad().data_as<float>()[0] > 0.f);
}

TEST_CASE("reward_preference_loss - rejects non-scalar inputs") {
    Tensor t2({2}, DType::Float32);
    Variable v2(t2, true), v1(Tensor({1}, DType::Float32), true);
    REQUIRE_THROWS_AS(reward_preference_loss(v2, v1), std::runtime_error);
    REQUIRE_THROWS_AS(reward_preference_loss(v1, v2), std::runtime_error);
}

TEST_CASE("reward_preference_loss - training reduces loss") {
    RewardModel rm(20, 16, 2, 1, 1, 0, 42);
    Adam adam(rm.parameters(), 1e-2f);

    auto chosen_ids   = make_ids({0, 2, 4, 6});
    auto rejected_ids = make_ids({1, 3, 5, 7});

    float first = -1.f, last = -1.f;
    for (int step = 0; step < 60; ++step) {
        adam.zero_grad();
        auto r_c = rm.score(chosen_ids);
        auto r_r = rm.score(rejected_ids);
        auto loss = reward_preference_loss(r_c, r_r);
        loss.backward();
        adam.step();
        float lv = loss.data().data_as<float>()[0];
        if (step == 0) first = lv;
        last = lv;
    }
    REQUIRE(last < first);
}

// ── reinforce_loss ────────────────────────────────────────────────────────────

TEST_CASE("reinforce_loss - returns scalar Variable") {
    Tensor logits = make_logits(4, 20);
    Variable lv(logits, true);
    auto ids = make_ids({0, 1, 2, 3});
    auto loss = reinforce_loss(lv, ids, 1.0f);
    REQUIRE(loss.data().numel() == 1);
}

TEST_CASE("reinforce_loss - positive reward scales CE positively") {
    Tensor logits = make_logits(4, 20);
    Variable lv(logits, false);
    auto ids = make_ids({0, 1, 2, 3});
    float ce_val = cross_entropy(lv, ids).data().data_as<float>()[0];
    float rl_val = reinforce_loss(lv, ids, 2.0f).data().data_as<float>()[0];
    REQUIRE(rl_val == Catch::Approx(2.0f * ce_val).epsilon(1e-5));
}

TEST_CASE("reinforce_loss - reward=0 gives zero loss") {
    Tensor logits = make_logits(4, 20);
    Variable lv(logits, false);
    auto ids = make_ids({0, 1, 2, 3});
    auto loss = reinforce_loss(lv, ids, 0.0f);
    REQUIRE(loss.data().data_as<float>()[0] == Catch::Approx(0.0f).margin(1e-6));
}

TEST_CASE("reinforce_loss - gradient flows to logits") {
    Tensor logits = make_logits(4, 20);
    Variable lv(logits, true);
    auto ids = make_ids({0, 1, 2, 3});
    reinforce_loss(lv, ids, 1.5f).backward();
    bool any_grad = false;
    for (auto f : lv.grad().data_as<float>())
        if (f != 0.f) { any_grad = true; break; }
    REQUIRE(any_grad);
}

TEST_CASE("reinforce_loss - rejects mismatched shapes") {
    Variable lv(make_logits(4, 20), true);
    auto ids = make_ids({0, 1, 2});  // T=3, logits T=4
    REQUIRE_THROWS_AS(reinforce_loss(lv, ids, 1.0f), std::runtime_error);
}

TEST_CASE("reinforce_loss - rejects non-finite reward") {
    Variable lv(make_logits(4, 20), true);
    auto ids = make_ids({0, 1, 2, 3});
    REQUIRE_THROWS_AS(
        reinforce_loss(lv, ids, std::numeric_limits<float>::infinity()),
        std::runtime_error);
    REQUIRE_THROWS_AS(
        reinforce_loss(lv, ids, std::numeric_limits<float>::quiet_NaN()),
        std::runtime_error);
}

// ── kl_penalty ────────────────────────────────────────────────────────────────

TEST_CASE("kl_penalty - same logits gives zero KL") {
    Tensor logits = make_logits(4, 20, 0.0f);
    Variable lv(logits, true);
    auto kl = kl_penalty(lv, logits);
    REQUIRE(kl.data().data_as<float>()[0] == Catch::Approx(0.0f).margin(1e-5));
}

TEST_CASE("kl_penalty - different logits gives positive KL") {
    Tensor logits_new = make_logits(4, 20, 0.0f);
    Tensor logits_ref = make_logits(4, 20, 0.0f);
    // Shift new logits so distributions differ
    logits_new.data_as<float>()[0] = 5.0f;
    Variable lv(logits_new, true);
    auto kl = kl_penalty(lv, logits_ref);
    REQUIRE(kl.data().data_as<float>()[0] > 0.f);
}

TEST_CASE("kl_penalty - gradient flows through logits_new") {
    // Non-uniform logits so that KL gradient is non-zero.
    Tensor logits_new({3, 10}, DType::Float32);
    Tensor logits_ref({3, 10}, DType::Float32);
    auto ns = logits_new.data_as<float>();
    auto rs = logits_ref.data_as<float>();
    for (std::size_t i = 0; i < 30; ++i) {
        ns[i] = static_cast<float>(i % 5);         // varying
        rs[i] = static_cast<float>((i + 3) % 7);   // different
    }
    Variable lv(logits_new, true);
    kl_penalty(lv, logits_ref).backward();
    bool any_grad = false;
    for (auto f : lv.grad().data_as<float>())
        if (f != 0.f) { any_grad = true; break; }
    REQUIRE(any_grad);
}

TEST_CASE("kl_penalty - rejects shape mismatch") {
    Variable lv(make_logits(4, 20), true);
    Tensor ref_wrong = make_logits(3, 20);  // T mismatch
    REQUIRE_THROWS_AS(kl_penalty(lv, ref_wrong), std::runtime_error);
}

TEST_CASE("kl_penalty - rejects T=0 input") {
    Tensor empty({0, 20}, DType::Float32);
    Variable lv(empty, true);
    REQUIRE_THROWS_AS(kl_penalty(lv, empty), std::runtime_error);
}
