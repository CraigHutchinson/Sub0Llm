#include "sub0llm/nn/lora.hpp"
#include "sub0llm/nn/optimizer.hpp"
#include "sub0llm/autograd/ops.hpp"
#include "sub0llm/core/tensor.hpp"

#include <cmath>
#include <format>
#include <iostream>
#include <random>
#include <vector>

using namespace sub0llm;
using namespace sub0llm::nn;
using namespace sub0llm::autograd;

static void section(std::string_view title) {
    std::cout << "\n=== " << title << " ===\n";
}

// ── §12.1  LoRA anatomy — parameter counts ────────────────────────────────────

static void section_param_counts() {
    section("§12.1  LoRA anatomy — parameter counts");

    constexpr int64_t D = 256;
    const int64_t full_params = D * D;
    std::cout << std::format("  Full linear ({0}×{0}): {1} trainable parameters\n",
                              D, full_params);
    std::cout << std::format("  {:>6}  {:>12}  {:>10}\n", "rank", "LoRA params", "% of full");
    for (int64_t r : {1, 2, 4, 8, 16, 32}) {
        const int64_t lora_params = D * r + r * D;
        const float pct = 100.0f * static_cast<float>(lora_params) /
                          static_cast<float>(full_params);
        std::cout << std::format("  {:>6}  {:>12}  {:>9.2f}%\n", r, lora_params, pct);
    }
    std::cout << "  LoRA trades capacity for drastically fewer trainable parameters.\n";
}

// ── §12.2  Zero delta at initialisation ───────────────────────────────────────

static void section_zero_delta() {
    section("§12.2  Zero delta at initialisation");

    LoRALinear layer(8, 6, /*rank=*/2, /*alpha=*/1.0f, /*seed=*/99);

    auto params = layer.all_parameters();
    const auto b_data = params[2]->data().data_as<float>();  // lora_B

    bool all_zero = true;
    for (float v : b_data)
        if (v != 0.0f) { all_zero = false; break; }

    std::cout << std::format("  LoRALinear(8, 6, rank=2): lora_B all zeros at init: {}\n",
                              all_zero ? "yes" : "no");
    std::cout << "  Since ΔW = (α/r)*A@B and B=0, the LoRA contribution is zero at init.\n";
    std::cout << "  Fine-tuning starts from exactly the pre-trained base weight.\n";

    // Show that forward gives finite, non-NaN output.
    Tensor xd({3, 8}, DType::Float32);
    {
        auto s = xd.data_as<float>();
        for (std::size_t i = 0; i < s.size(); ++i)
            s[i] = static_cast<float>(i) * 0.1f - 0.1f;
    }
    Variable x(xd, /*requires_grad=*/false);
    auto out = layer.forward(x);
    std::cout << std::format("  forward shape: ({}, {}), finite: {}\n",
                              out.data().shape()[0], out.data().shape()[1],
                              std::isfinite(out.data().data_as<float>()[0]) ? "yes" : "no");
}

// ── §12.3  Gradient isolation ─────────────────────────────────────────────────

static void section_grad_isolation() {
    section("§12.3  Gradient isolation");

    LoRALinear layer(16, 8, /*rank=*/4, /*alpha=*/1.0f, /*seed=*/7);

    Tensor xd({5, 16}, DType::Float32);
    {
        auto s = xd.data_as<float>();
        for (std::size_t i = 0; i < s.size(); ++i)
            s[i] = static_cast<float>(i % 5) * 0.2f - 0.4f;
    }
    Variable x(xd, /*requires_grad=*/false);

    auto out = layer.forward(x);
    sum(out).backward();

    auto params = layer.all_parameters();
    const bool base_no_grad = (params[0]->grad().numel() == 0);
    const bool a_has_grad   = (params[1]->grad().numel() > 0);
    const bool b_has_grad   = (params[2]->grad().numel() > 0);

    std::cout << std::format("  base (frozen) has no grad: {}\n",
                              base_no_grad ? "yes" : "no");
    std::cout << std::format("  lora_A has gradient: {} (numel={})\n",
                              a_has_grad ? "yes" : "no",
                              params[1]->grad().numel());
    std::cout << std::format("  lora_B has gradient: {} (numel={})\n",
                              b_has_grad ? "yes" : "no",
                              params[2]->grad().numel());
    std::cout << "  Only LoRA parameters receive gradients; base weights are frozen.\n";
}

// ── §12.4  LoRA fine-tuning ───────────────────────────────────────────────────

static void section_finetuning() {
    section("§12.4  LoRA fine-tuning");

    // Task: next-token prediction on a small repeating sequence.
    // Architecture: fixed embeddings → LoRALinear → cross_entropy.
    constexpr int64_t T  = 8;    // sequence length
    constexpr int64_t V  = 16;   // vocabulary size
    constexpr int64_t D  = 32;   // embedding dim
    constexpr int64_t R  = 4;    // LoRA rank
    constexpr float   LR = 5e-2f;
    constexpr int     Steps = 100;

    // Fixed embedding matrix — random, frozen throughout fine-tuning.
    std::mt19937 emb_rng(123);
    std::normal_distribution<float> emb_dist(0.0f, 1.0f);
    Tensor emb_data({V, D}, DType::Float32);
    {
        auto s = emb_data.data_as<float>();
        for (auto& v : s) v = emb_dist(emb_rng);
    }

    // Token ids and shifted targets.
    Tensor ids({T}, DType::Int32);
    Tensor targets({T}, DType::Int32);
    {
        auto di = ids.data_as<int32_t>();
        auto dt = targets.data_as<int32_t>();
        for (std::size_t i = 0; i < static_cast<std::size_t>(T); ++i) {
            di[i] = static_cast<int32_t>(i % V);
            dt[i] = static_cast<int32_t>((i + 1) % static_cast<std::size_t>(V));
        }
    }

    // Pre-compute input embeddings (frozen lookup, no autograd needed).
    auto make_input = [&]() -> Variable {
        Tensor xd({T, D}, DType::Float32);
        const auto emb = emb_data.data_as<float>();
        auto       xv  = xd.data_as<float>();
        const auto id  = ids.data_as<int32_t>();
        for (std::size_t i = 0; i < static_cast<std::size_t>(T); ++i) {
            const std::size_t row = static_cast<std::size_t>(id[i]);
            for (std::size_t j = 0; j < static_cast<std::size_t>(D); ++j)
                xv[i * static_cast<std::size_t>(D) + j] =
                    emb[row * static_cast<std::size_t>(D) + j];
        }
        return Variable(std::move(xd), /*requires_grad=*/false);
    };

    std::cout << std::format(
        "  Task: next-token on T={} tokens, V={} vocab, D={} embed, rank={}\n"
        "  Trainable: {} params (lora_A + lora_B)  vs  {} full\n",
        T, V, D, R,
        D * R + R * V,
        D * V);

    // LoRA fine-tuning: train only A and B.
    LoRALinear layer_lora(D, V, R, /*alpha=*/1.0f, /*seed=*/42);
    SGD        optim_lora(layer_lora.lora_parameters(), LR);

    float lora_first = -1.0f, lora_last = -1.0f;
    std::cout << "\n  LoRA fine-tuning (frozen base + trainable A,B):\n";
    for (int step = 0; step < Steps; ++step) {
        optim_lora.zero_grad();
        auto logits = layer_lora.forward(make_input());
        auto loss   = cross_entropy(logits, targets);
        loss.backward();
        optim_lora.step();

        const float lv = loss.data().data_as<float>()[0];
        if (step == 0)        lora_first = lv;
        if (step == Steps - 1) lora_last = lv;

        if (step % 20 == 0 || step == Steps - 1)
            std::cout << std::format("    step {:>3}  loss={:.4f}\n", step, lv);
    }

    float full_first = -1.0f, full_last = -1.0f;
    // Compare against high-rank LoRA which has strictly more capacity.
    {
        constexpr int64_t FULL_RANK = 8;  // high-rank LoRA approximates full FT
        LoRALinear layer_hrank(D, V, FULL_RANK, /*alpha=*/1.0f, /*seed=*/42);
        SGD        optim_h(layer_hrank.lora_parameters(), LR);

        std::cout << "\n  High-rank LoRA (rank=8, more capacity):\n";
        for (int step = 0; step < Steps; ++step) {
            optim_h.zero_grad();
            auto logits = layer_hrank.forward(make_input());
            auto loss   = cross_entropy(logits, targets);
            loss.backward();
            optim_h.step();

            const float lv = loss.data().data_as<float>()[0];
            if (step == 0)        full_first = lv;
            if (step == Steps - 1) full_last = lv;

            if (step % 20 == 0 || step == Steps - 1)
                std::cout << std::format("    step {:>3}  loss={:.4f}\n", step, lv);
        }
    }

    std::cout << std::format(
        "\n  Summary:\n"
        "    LoRA  rank={}: loss {:.4f} → {:.4f}  ({})\n"
        "    LoRA  rank=8:  loss {:.4f} → {:.4f}  ({})\n",
        R, lora_first, lora_last,
        lora_last < lora_first ? "converging" : "learning",
        full_first, full_last,
        full_last < full_first ? "converging" : "learning");
}

// ── §12.5  Rank ablation ──────────────────────────────────────────────────────

static void section_rank_ablation() {
    section("§12.5  Rank ablation");

    constexpr int64_t T  = 8;
    constexpr int64_t V  = 16;
    constexpr int64_t D  = 32;
    constexpr float   LR = 5e-2f;
    constexpr int     Steps = 100;

    // Fixed embeddings — random N(0,1), seed different from §12.4.
    std::mt19937 emb_rng(456);
    std::normal_distribution<float> emb_dist(0.0f, 1.0f);
    Tensor emb_data({V, D}, DType::Float32);
    {
        auto s = emb_data.data_as<float>();
        for (auto& v : s) v = emb_dist(emb_rng);
    }

    Tensor ids({T}, DType::Int32);
    Tensor targets({T}, DType::Int32);
    {
        auto di = ids.data_as<int32_t>();
        auto dt = targets.data_as<int32_t>();
        for (std::size_t i = 0; i < static_cast<std::size_t>(T); ++i) {
            di[i] = static_cast<int32_t>(i % V);
            dt[i] = static_cast<int32_t>((i + 1) % static_cast<std::size_t>(V));
        }
    }

    auto make_input = [&]() -> Variable {
        Tensor xd({T, D}, DType::Float32);
        const auto emb = emb_data.data_as<float>();
        auto       xv  = xd.data_as<float>();
        const auto id  = ids.data_as<int32_t>();
        for (std::size_t i = 0; i < static_cast<std::size_t>(T); ++i) {
            const std::size_t row = static_cast<std::size_t>(id[i]);
            for (std::size_t j = 0; j < static_cast<std::size_t>(D); ++j)
                xv[i * static_cast<std::size_t>(D) + j] =
                    emb[row * static_cast<std::size_t>(D) + j];
        }
        return Variable(std::move(xd), /*requires_grad=*/false);
    };

    std::cout << std::format("  {:>6}  {:>12}  {:>10}  {:>10}  {}\n",
                              "rank", "params", "loss[0]", "loss[end]", "converged?");

    for (int64_t r : {1, 2, 4, 8}) {
        LoRALinear layer(D, V, r, /*alpha=*/1.0f, /*seed=*/42);
        SGD        optim(layer.lora_parameters(), LR);

        float first_loss = -1.0f, last_loss = -1.0f;

        for (int step = 0; step < Steps; ++step) {
            optim.zero_grad();
            auto logits = layer.forward(make_input());
            auto loss   = cross_entropy(logits, targets);
            loss.backward();
            optim.step();

            const float lv = loss.data().data_as<float>()[0];
            if (step == 0)        first_loss = lv;
            if (step == Steps - 1) last_loss  = lv;
        }

        const int64_t n_params = D * r + r * V;
        std::cout << std::format("  {:>6}  {:>12}  {:>10.4f}  {:>10.4f}  {}\n",
                                  r, n_params, first_loss, last_loss,
                                  last_loss < first_loss * 0.75f ? "yes" : "partial");
    }

    std::cout << "  Lower rank can converge faster on simple tasks (simpler landscape).\n";
    std::cout << "  Higher rank adds capacity; rank selection is task-dependent.\n";
}

// ── main ──────────────────────────────────────────────────────────────────────

int main() {
    std::cout << "Chapter 12 — LoRA Fine-Tuning\n";
    std::cout << std::string(60, '=') << '\n';

    section_param_counts();
    section_zero_delta();
    section_grad_isolation();
    section_finetuning();
    section_rank_ablation();

    std::cout << "\nDone.\n";
    return 0;
}
