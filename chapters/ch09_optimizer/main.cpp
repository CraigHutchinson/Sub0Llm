// Chapter 09 — Optimizers
//
// Now that we have a full GPT-style model with autograd, we need optimizers
// to actually train it.  This chapter adds:
//
//   • Gradient clipping (global L2 norm) — prevents exploding gradients.
//   • SGD with momentum — simple, but effective with careful tuning.
//   • Adam (Kingma & Ba, 2015) — adaptive per-parameter learning rates with
//     first- and second-moment estimates and bias correction.
//
// The demo trains a tiny GPT model on a circular next-token prediction task
// and shows the cross-entropy loss decreasing from ≈log(vocab) toward zero.

#include <cmath>
#include <format>
#include <vector>

#include "sub0llm/autograd/ops.hpp"
#include "sub0llm/compat/print.hpp"
#include "sub0llm/core/ops.hpp"
#include "sub0llm/nn/gpt.hpp"
#include "sub0llm/nn/optimizer.hpp"
#include "sub0llm/version.hpp"

using namespace sub0llm;
using namespace sub0llm::autograd;
using namespace sub0llm::nn;

static void section(std::string_view title) {
    std::println("\n── {} ──", title);
}

// ─────────────────────────────────────────────────────────────────────────────
int main() {
    std::println("sub0llm v{} — Chapter 09: Optimizers", version_string);

    // ── 1. Gradient clipping ──────────────────────────────────────────────────
    section("1. Gradient clipping");
    std::println(
        "clip_grad_norm(params, max_norm):\n"
        "  global_norm = sqrt(Σ ||g_i||²)\n"
        "  if global_norm > max_norm: g_i *= max_norm / global_norm\n"
        "Prevents exploding gradients during early training."
    );
    {
        Tensor t1 = zeros({2}); { auto d = t1.data_as<float>(); d[0]=3.f; d[1]=0.f; }
        Tensor t2 = zeros({2}); { auto d = t2.data_as<float>(); d[0]=0.f; d[1]=4.f; }
        Variable p1(zeros({2}), true); p1.grad() = t1;
        Variable p2(zeros({2}), true); p2.grad() = t2;

        std::vector<Variable*> params = {&p1, &p2};
        const float norm_before = std::sqrt(9.f + 16.f);  // 5
        const float norm_after = clip_grad_norm(params, 2.5f);

        std::println("  Global grad norm: {:.3f}  →  clipped to {:.1f}",
                     norm_after, 2.5f);
        std::println("  Scale applied: {:.3f}",  2.5f / norm_before);
        const auto g1 = p1.grad().data_as<float>();
        const auto g2 = p2.grad().data_as<float>();
        std::println("  p1 grad after clip: [{:.3f}, {:.3f}]", g1[0], g1[1]);
        std::println("  p2 grad after clip: [{:.3f}, {:.3f}]", g2[0], g2[1]);
    }

    // ── 2. SGD with momentum ──────────────────────────────────────────────────
    section("2. SGD with momentum");
    std::println(
        "  v_{{t+1}} = momentum * v_t - lr * g_t\n"
        "  p_{{t+1}} = p_t + v_{{t+1}}\n"
        "Momentum accumulates gradient history, dampening oscillations."
    );
    {
        // Minimise f(p) = 0.5*p^2 with gradient g = p
        Variable p(zeros({1}), true);
        p.data().data_as<float>()[0] = 2.0f;

        SGD sgd({&p}, 0.1f, 0.9f);
        std::println("  Starting p = {:.4f}", 2.0f);
        for (int step = 0; step < 5; ++step) {
            const float pv = p.data().data_as<float>()[0];
            Tensor g = zeros({1}); g.data_as<float>()[0] = pv; p.grad() = g;
            sgd.step();
            std::println("  step {:2d}: p = {:.6f}", step + 1,
                         p.data().data_as<float>()[0]);
        }
    }

    // ── 3. Adam ───────────────────────────────────────────────────────────────
    section("3. Adam (Kingma & Ba, 2015)");
    std::println(
        "  m_{{t+1}} = b1*m_t + (1-b1)*g           (first moment)\n"
        "  v_{{t+1}} = b2*v_t + (1-b2)*g^2         (second moment)\n"
        "  m_hat   = m_{{t+1}} / (1 - b1^t)        (bias correction)\n"
        "  v_hat   = v_{{t+1}} / (1 - b2^t)\n"
        "  p_{{t+1}} = p_t - lr * m_hat / (sqrt(v_hat) + eps)"
    );
    {
        Variable p(zeros({1}), true);
        p.data().data_as<float>()[0] = 2.0f;

        Adam adam({&p}, 0.1f);
        std::println("  Starting p = {:.4f}", 2.0f);
        for (int step = 0; step < 5; ++step) {
            const float pv = p.data().data_as<float>()[0];
            Tensor g = zeros({1}); g.data_as<float>()[0] = pv; p.grad() = g;
            adam.step();
            std::println("  step {:2d}: p = {:.6f}", step + 1,
                         p.data().data_as<float>()[0]);
        }
    }

    // ── 4. Training a tiny GPT with Adam ─────────────────────────────────────
    section("4. Training a tiny GPT with Adam");

    constexpr int64_t VOCAB   = 32;
    constexpr int64_t EMBED   = 16;
    constexpr int64_t HEADS   = 2;
    constexpr int64_t LAYERS  = 1;
    constexpr int64_t MAX_SEQ = 16;
    constexpr int64_t T       = 8;   // sequence length per batch
    constexpr float   MAX_NORM = 1.0f;

    GPT model(VOCAB, EMBED, HEADS, LAYERS, MAX_SEQ, /*seed=*/42);
    auto params = model.parameters();
    Adam adam(params, /*lr=*/1e-2f);

    std::println(
        "Model: vocab={} embed={} heads={} layers={}\n"
        "Task:  predict next token in circular sequence 0→1→…→{}→0",
        VOCAB, EMBED, HEADS, LAYERS, T - 1
    );

    // Fixed input: tokens 0,1,2,…,T-1; targets: 1,2,…,T-1,0
    Tensor ids     = zeros({T}, DType::Int32);
    Tensor targets = zeros({T}, DType::Int32);
    {
        auto si = ids.data_as<int32_t>();
        auto st = targets.data_as<int32_t>();
        for (std::size_t i = 0; i < static_cast<std::size_t>(T); ++i) {
            si[i] = static_cast<int32_t>(i);
            st[i] = static_cast<int32_t>((static_cast<int64_t>(i) + 1) % VOCAB);
        }
    }

    const float loss_random = std::log(static_cast<float>(VOCAB));
    std::println("Random-guess loss (log {}): {:.4f}\n", VOCAB, loss_random);

    for (int step = 0; step < 50; ++step) {
        adam.zero_grad();
        auto logits = model.forward(ids);
        auto loss   = cross_entropy(logits, targets);
        loss.backward();
        (void)clip_grad_norm(params, MAX_NORM);
        adam.step();

        if (step == 0 || (step + 1) % 10 == 0)
            std::println("  step {:3d}: loss = {:.4f}", step + 1,
                         loss.data().data_as<float>()[0]);
    }

    // ── 5. What's next ────────────────────────────────────────────────────────
    section("5. What's next — Ch10: Pretraining");
    std::println(
        "Ch10 adds:\n"
        "  • Real text dataset + BPE tokenization\n"
        "  • Mini-batch DataLoader with random shuffling\n"
        "  • Cosine learning-rate schedule with linear warm-up\n"
        "  • Checkpointing (save / restore model weights)\n"
        "  • A complete pretraining loop on a small corpus"
    );

    return 0;
}
