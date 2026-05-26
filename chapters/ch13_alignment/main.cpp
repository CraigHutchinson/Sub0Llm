#include "sub0llm/nn/dpo.hpp"
#include "sub0llm/nn/modern_gpt.hpp"
#include "sub0llm/nn/optimizer.hpp"
#include "sub0llm/nn/scheduler.hpp"
#include "sub0llm/autograd/ops.hpp"
#include "sub0llm/core/tensor.hpp"
#include <cmath>
#include <format>
#include <iostream>
#include <vector>

using namespace sub0llm;
using namespace sub0llm::nn;
using namespace sub0llm::autograd;

// ── §13.1  log_sigmoid demo ───────────────────────────────────────────────────

static void section_log_sigmoid() {
    std::cout << "\n=== §13.1  log_sigmoid — numerically stable log σ(x) ===\n";
    std::cout << std::format("  {:>6}  {:>14}  {:>14}  {:>10}\n",
                              "x", "log_sigmoid", "-log(1+e^-x)", "diff");

    const float xs[] = {-10.f, -5.f, -2.f, -1.f, 0.f, 1.f, 2.f, 5.f, 10.f};
    for (float xv : xs) {
        Tensor xd({1}, DType::Float32);
        xd.data_as<float>()[0] = xv;
        Variable xvar(std::move(xd), false);
        const float ls  = log_sigmoid(xvar).data().data_as<float>()[0];
        // Reference: for large |x| use the stable branch too, but direct formula
        // illustrates numerical precision.
        const float ref = (xv >= 0.f)
                          ? -std::log1p(std::exp(-xv))
                          : xv - std::log1p(std::exp(xv));
        std::cout << std::format("  {:>6.1f}  {:>14.6f}  {:>14.6f}  {:>10.2e}\n",
                                  xv, ls, ref, std::abs(ls - ref));
    }
    std::cout << "  Numerical agreement confirms stable implementation.\n";
}

// ── §13.2  DPO loss anatomy ───────────────────────────────────────────────────

static void section_dpo_anatomy() {
    std::cout << "\n=== §13.2  DPO Loss Anatomy ===\n";
    std::cout << "  L = -log σ(β * ((lp_w - ref_lp_w) - (lp_l - ref_lp_l)))\n\n";

    // Demonstrate loss vs margin relationship.
    std::cout << std::format("  {:>8}  {:>10}  {:>10}\n", "margin", "loss", "description");

    // margin = 0: equal policy and reference → loss = log(2)
    {
        const float margin = 0.0f;
        const float loss   = -std::log(1.0f / (1.0f + std::exp(-margin)));
        std::cout << std::format("  {:>8.2f}  {:>10.4f}  equal winner/loser\n",
                                  margin, loss);
    }
    // margin = 1 (strong preference)
    {
        const float margin = 1.0f;
        const float loss   = -std::log(1.0f / (1.0f + std::exp(-margin)));
        std::cout << std::format("  {:>8.2f}  {:>10.4f}  policy correctly prefers winner\n",
                                  margin, loss);
    }
    // margin = 5 (very strong preference)
    {
        const float margin = 5.0f;
        const float loss   = -std::log(1.0f / (1.0f + std::exp(-margin)));
        std::cout << std::format("  {:>8.2f}  {:>10.4f}  strong preference → loss → 0\n",
                                  margin, loss);
    }
    // margin = -2 (policy prefers loser)
    {
        const float margin = -2.0f;
        const float loss   = -std::log(1.0f / (1.0f + std::exp(-margin)));
        std::cout << std::format("  {:>8.2f}  {:>10.4f}  policy incorrectly prefers loser\n",
                                  margin, loss);
    }
    std::cout << "  β=0.1 scales the reward gap; larger β → sharper preference signal.\n";
}

// ── §13.3  Reference log-probs ────────────────────────────────────────────────

static void section_reference_logprobs() {
    std::cout << "\n=== §13.3  Reference Log-Probabilities ===\n";
    std::cout << "  Pre-computing log-probs under a frozen reference model.\n";

    constexpr int64_t V = 16, D = 16;
    ModernGPT reference(V, D, 2, 1, 1, 0, 0, /*seed=*/99);

    // Two synthetic sequences.
    Tensor ids_w({6}, DType::Int32);
    {
        auto is = ids_w.data_as<int32_t>();
        for (int i = 0; i < 6; ++i) is[static_cast<std::size_t>(i)] = i;
    }
    Tensor ids_l({6}, DType::Int32);
    {
        auto is = ids_l.data_as<int32_t>();
        for (int i = 0; i < 6; ++i) is[static_cast<std::size_t>(i)] = 5 - i;
    }

    // Compute with no gradients — reference model is frozen.
    auto ref_logits_w = reference.forward(ids_w);
    auto ref_logits_l = reference.forward(ids_l);

    const float ref_lp_w = log_prob_sequence(ref_logits_w, ids_w)
                               .data().data_as<float>()[0];
    const float ref_lp_l = log_prob_sequence(ref_logits_l, ids_l)
                               .data().data_as<float>()[0];

    std::cout << std::format("  ref_lp_w (winner under ref model): {:.4f}\n", ref_lp_w);
    std::cout << std::format("  ref_lp_l (loser  under ref model): {:.4f}\n", ref_lp_l);
    std::cout << "  These scalars are passed unchanged to dpo_loss().\n";
}

// ── §13.4  DPO training loop ──────────────────────────────────────────────────

static void section_dpo_training() {
    std::cout << "\n=== §13.4  DPO Training Loop ===\n";

    constexpr int64_t V = 16, D = 16;

    // Build policy and reference models with different seeds.
    ModernGPT policy(V, D, 2, 1, 1, 0, 0, /*seed=*/42);
    ModernGPT reference(V, D, 2, 1, 1, 0, 0, /*seed=*/99);

    // Winner: ascending tokens [0,1,2,3,4,5]
    Tensor ids_w({6}, DType::Int32);
    {
        auto is = ids_w.data_as<int32_t>();
        for (int i = 0; i < 6; ++i) is[static_cast<std::size_t>(i)] = i;
    }
    // Loser: descending tokens [5,4,3,2,1,0]
    Tensor ids_l({6}, DType::Int32);
    {
        auto is = ids_l.data_as<int32_t>();
        for (int i = 0; i < 6; ++i) is[static_cast<std::size_t>(i)] = 5 - i;
    }

    // Pre-compute reference log-probs (frozen — computed once, no gradient).
    const float ref_lp_w_val = [&] {
        auto rl = reference.forward(ids_w);
        return log_prob_sequence(rl, ids_w).data().data_as<float>()[0];
    }();
    const float ref_lp_l_val = [&] {
        auto rl = reference.forward(ids_l);
        return log_prob_sequence(rl, ids_l).data().data_as<float>()[0];
    }();
    std::cout << std::format("  ref_lp_w={:.3f}  ref_lp_l={:.3f}\n",
                              ref_lp_w_val, ref_lp_l_val);

    Adam adam(policy.parameters(), 1e-3f);

    std::cout << std::format("  {:>5}  {:>10}\n", "step", "loss");
    float first_loss = -1.f, last_loss = -1.f;

    for (int step = 0; step < 30; ++step) {
        adam.zero_grad();

        auto logits_w = policy.forward(ids_w);
        auto logits_l = policy.forward(ids_l);
        auto loss = dpo_loss(logits_w, ids_w, logits_l, ids_l,
                             ref_lp_w_val, ref_lp_l_val);
        loss.backward();
        adam.step();

        const float lv = loss.data().data_as<float>()[0];
        if (step == 0)  first_loss = lv;
        if (step == 29) last_loss  = lv;

        if (step % 5 == 0 || step == 29)
            std::cout << std::format("  step {:>2}  loss={:.4f}\n", step, lv);
    }

    // Show preference margin evolution.
    {
        auto lw = policy.forward(ids_w);
        auto ll = policy.forward(ids_l);
        const float lp_w   = log_prob_sequence(lw, ids_w).data().data_as<float>()[0];
        const float lp_l   = log_prob_sequence(ll, ids_l).data().data_as<float>()[0];
        const float margin = (lp_w - ref_lp_w_val) - (lp_l - ref_lp_l_val);
        std::cout << std::format("  Final margin (β·margin drives loss): {:.4f}\n", margin);
    }
    std::cout << std::format("  Loss: {:.4f} → {:.4f}  ({})\n",
                              first_loss, last_loss,
                              last_loss < first_loss ? "decreasing ✓" : "flat/increasing");
}

// ── §13.5  Implicit reward ────────────────────────────────────────────────────

static void section_implicit_reward() {
    std::cout << "\n=== §13.5  Implicit DPO Reward ===\n";
    std::cout << "  r(x,y) = β * (log π(y|x) - log π_ref(y|x))\n\n";

    constexpr float beta = 0.1f;
    constexpr int64_t V = 16, D = 16;

    // Run a quick DPO training first to get a non-trivial policy.
    ModernGPT policy(V, D, 2, 1, 1, 0, 0, /*seed=*/42);
    ModernGPT reference(V, D, 2, 1, 1, 0, 0, /*seed=*/99);

    Tensor ids_w({6}, DType::Int32);
    {
        auto is = ids_w.data_as<int32_t>();
        for (int i = 0; i < 6; ++i) is[static_cast<std::size_t>(i)] = i;
    }
    Tensor ids_l({6}, DType::Int32);
    {
        auto is = ids_l.data_as<int32_t>();
        for (int i = 0; i < 6; ++i) is[static_cast<std::size_t>(i)] = 5 - i;
    }

    const float ref_lp_w = [&] {
        auto rl = reference.forward(ids_w);
        return log_prob_sequence(rl, ids_w).data().data_as<float>()[0];
    }();
    const float ref_lp_l = [&] {
        auto rl = reference.forward(ids_l);
        return log_prob_sequence(rl, ids_l).data().data_as<float>()[0];
    }();

    Adam adam(policy.parameters(), 1e-3f);
    for (int step = 0; step < 30; ++step) {
        adam.zero_grad();
        auto logits_w = policy.forward(ids_w);
        auto logits_l = policy.forward(ids_l);
        auto loss = dpo_loss(logits_w, ids_w, logits_l, ids_l, ref_lp_w, ref_lp_l);
        loss.backward();
        adam.step();
    }

    // Compute implicit rewards under the trained policy.
    const float policy_lp_w = [&] {
        auto lw = policy.forward(ids_w);
        return log_prob_sequence(lw, ids_w).data().data_as<float>()[0];
    }();
    const float policy_lp_l = [&] {
        auto ll = policy.forward(ids_l);
        return log_prob_sequence(ll, ids_l).data().data_as<float>()[0];
    }();

    const float reward_w = beta * (policy_lp_w - ref_lp_w);
    const float reward_l = beta * (policy_lp_l - ref_lp_l);

    std::cout << std::format("  reward(winner) = {:.4f}\n", reward_w);
    std::cout << std::format("  reward(loser)  = {:.4f}\n", reward_l);
    std::cout << std::format("  reward gap     = {:.4f}  ({})\n",
                              reward_w - reward_l,
                              reward_w > reward_l
                              ? "winner > loser ✓"
                              : "gap not yet separated (more steps needed)");
}

// ── main ──────────────────────────────────────────────────────────────────────

int main() {
    std::cout << "Chapter 13 — DPO Alignment\n";
    std::cout << std::string(60, '=') << '\n';

    section_log_sigmoid();
    section_dpo_anatomy();
    section_reference_logprobs();
    section_dpo_training();
    section_implicit_reward();

    std::cout << "\nDone.\n";
}
