#include "sub0llm/nn/sampler.hpp"
#include "sub0llm/nn/modern_gpt.hpp"
#include "sub0llm/core/tensor.hpp"

#include <cmath>
#include <format>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <vector>

using namespace sub0llm;
using namespace sub0llm::nn;

static void section(std::string_view title) {
    std::cout << "\n=== " << title << " ===\n";
}

// Build a 1-D float tensor from a vector of values.
static Tensor make_logits(std::vector<float> vals) {
    Tensor t({static_cast<int64_t>(vals.size())}, DType::Float32);
    auto sp = t.data_as<float>();
    for (std::size_t i = 0; i < vals.size(); ++i) sp[i] = vals[i];
    return t;
}

// ── §14.1  Temperature scaling ────────────────────────────────────────────────

static void section_temperature() {
    section("§14.1  Temperature Sampling");

    // Peaked logits: token 2 is dominant.
    const std::vector<float> logit_vals = {0.5f, 1.0f, 4.0f, 0.8f, 0.2f};
    Tensor logits = make_logits(logit_vals);

    const int N = 2000;
    std::cout << std::format("  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}\n",
                              "temp", "tok0", "tok1", "tok2", "tok3", "tok4");

    for (float temp : {0.1f, 0.5f, 1.0f, 2.0f, 5.0f}) {
        std::mt19937 rng(42);
        std::vector<int> counts(5, 0);
        for (int i = 0; i < N; ++i)
            ++counts[static_cast<std::size_t>(temperature_sample(logits, temp, rng))];

        std::cout << std::format("  {:>8.1f}  {:>7.1f}%  {:>7.1f}%  {:>7.1f}%  {:>7.1f}%  {:>7.1f}%\n",
                                  temp,
                                  100.0f * static_cast<float>(counts[0]) / static_cast<float>(N), 100.0f * static_cast<float>(counts[1]) / static_cast<float>(N),
                                  100.0f * static_cast<float>(counts[2]) / static_cast<float>(N), 100.0f * static_cast<float>(counts[3]) / static_cast<float>(N),
                                  100.0f * static_cast<float>(counts[4]) / static_cast<float>(N));
    }

    std::cout << "  Low temperature concentrates mass on the peak; high flattens it.\n";
    std::cout << "  Greedy (T→0) = argmax: " << greedy_sample(logits) << '\n';
}

// ── §14.2  Top-k sampling ─────────────────────────────────────────────────────

static void section_top_k() {
    section("§14.2  Top-k Sampling");

    // Clear winner at index 4, then decreasing logits.
    const std::vector<float> logit_vals = {-1.0f, 0.5f, 1.5f, 3.0f, 6.0f};
    Tensor logits = make_logits(logit_vals);

    const int N = 2000;
    std::cout << std::format("  {:>4}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}  tokens allowed\n",
                              "k", "tok0", "tok1", "tok2", "tok3", "tok4");

    for (int64_t k : {1, 2, 3, 5}) {
        std::mt19937 rng(7);
        std::vector<int> counts(5, 0);
        for (int i = 0; i < N; ++i)
            ++counts[static_cast<std::size_t>(top_k_sample(logits, k, 1.0f, rng))];

        std::cout << std::format("  {:>4}  {:>7.1f}%  {:>7.1f}%  {:>7.1f}%  {:>7.1f}%  {:>7.1f}%"
                                  "  [top-{}]\n",
                                  k,
                                  100.0f * static_cast<float>(counts[0]) / static_cast<float>(N), 100.0f * static_cast<float>(counts[1]) / static_cast<float>(N),
                                  100.0f * static_cast<float>(counts[2]) / static_cast<float>(N), 100.0f * static_cast<float>(counts[3]) / static_cast<float>(N),
                                  100.0f * static_cast<float>(counts[4]) / static_cast<float>(N), k);
    }

    std::cout << "  Top-k prevents probability mass leaking to very low-probability tokens.\n";
}

// ── §14.3  Top-p (nucleus) sampling ──────────────────────────────────────────

static void section_top_p() {
    section("§14.3  Top-p (Nucleus) Sampling");

    // Logits with a long tail.
    const std::vector<float> logit_vals = {0.1f, 0.3f, 1.0f, 3.5f, 5.0f};
    Tensor logits = make_logits(logit_vals);

    const int N = 2000;
    std::cout << std::format("  {:>6}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}\n",
                              "p", "tok0", "tok1", "tok2", "tok3", "tok4");

    for (float p : {0.5f, 0.8f, 0.95f, 1.0f}) {
        std::mt19937 rng(21);
        std::vector<int> counts(5, 0);
        for (int i = 0; i < N; ++i)
            ++counts[static_cast<std::size_t>(top_p_sample(logits, p, 1.0f, rng))];

        std::cout << std::format("  {:>6.2f}  {:>7.1f}%  {:>7.1f}%  {:>7.1f}%  {:>7.1f}%  {:>7.1f}%\n",
                                  p,
                                  100.0f * static_cast<float>(counts[0]) / static_cast<float>(N), 100.0f * static_cast<float>(counts[1]) / static_cast<float>(N),
                                  100.0f * static_cast<float>(counts[2]) / static_cast<float>(N), 100.0f * static_cast<float>(counts[3]) / static_cast<float>(N),
                                  100.0f * static_cast<float>(counts[4]) / static_cast<float>(N));
    }

    std::cout << "  Nucleus sampling adapts the candidate set to the distribution shape.\n";
    std::cout << "  Small p → narrow nucleus; p=1.0 → full distribution.\n";
}

// ── §14.4  Autoregressive generation loop ─────────────────────────────────────

static void section_generation() {
    section("§14.4  Autoregressive Generation Loop");

    constexpr int64_t V = 16, D = 32;
    ModernGPT model(V, D, 2, 1, 2, 0, 0, /*seed=*/42);

    std::vector<int32_t> prompt = {1, 2, 3};
    const int64_t max_new       = 12;

    // Greedy
    {
        std::mt19937 rng(0);
        SamplingConfig cfg;  // defaults to Greedy
        auto out = generate(model, prompt, max_new, cfg, rng);

        std::cout << "  Greedy:      [";
        for (std::size_t i = 0; i < out.size(); ++i) {
            if (i == prompt.size()) std::cout << "| ";
            std::cout << out[i];
            if (i + 1 < out.size()) std::cout << ' ';
        }
        std::cout << "]\n";
        std::cout << std::format("  Generated {} tokens after {} prompt tokens\n",
                                  out.size() - prompt.size(), prompt.size());
    }

    // Temperature
    {
        std::mt19937 rng(42);
        SamplingConfig cfg;
        cfg.mode        = SamplingMode::Temperature;
        cfg.temperature = 0.7f;
        auto out = generate(model, prompt, max_new, cfg, rng);

        std::cout << "  Temp=0.7:    [";
        for (std::size_t i = 0; i < out.size(); ++i) {
            if (i == prompt.size()) std::cout << "| ";
            std::cout << out[i];
            if (i + 1 < out.size()) std::cout << ' ';
        }
        std::cout << "]\n";
    }

    // Top-p
    {
        std::mt19937 rng(7);
        SamplingConfig cfg;
        cfg.mode        = SamplingMode::TopP;
        cfg.top_p       = 0.9f;
        cfg.temperature = 0.8f;
        auto out = generate(model, prompt, max_new, cfg, rng);

        std::cout << "  Top-p=0.9:   [";
        for (std::size_t i = 0; i < out.size(); ++i) {
            if (i == prompt.size()) std::cout << "| ";
            std::cout << out[i];
            if (i + 1 < out.size()) std::cout << ' ';
        }
        std::cout << "]\n";
    }
}

// ── §14.5  Sampling strategy comparison ──────────────────────────────────────

static void section_comparison() {
    section("§14.5  Sampling Strategy Comparison");

    // Uneven logits — one clear winner, a moderate runner-up, three low-prob tokens.
    const std::vector<float> logit_vals = {0.2f, 0.4f, 0.6f, 3.0f, 7.0f};
    Tensor logits = make_logits(logit_vals);

    const int N = 5000;
    std::mt19937 rng(100);

    auto run = [&](std::string_view label, auto sampler_fn) {
        std::map<int32_t, int> counts;
        for (int i = 0; i < N; ++i) ++counts[sampler_fn()];
        std::cout << std::format("  {:>20}: ", label);
        for (int tok = 0; tok < 5; ++tok)
            std::cout << std::format("  tok{}={:.1f}%", tok,
                                      100.0f * static_cast<float>(counts[tok]) / static_cast<float>(N));
        std::cout << '\n';
    };

    run("greedy (always)", [&]() { return greedy_sample(logits); });
    run("temp=1.0", [&]() { return temperature_sample(logits, 1.0f, rng); });
    run("temp=0.5", [&]() { return temperature_sample(logits, 0.5f, rng); });
    run("top-k=3, t=1.0", [&]() { return top_k_sample(logits, 3, 1.0f, rng); });
    run("top-p=0.9, t=1.0", [&]() { return top_p_sample(logits, 0.9f, 1.0f, rng); });

    std::cout << "\n  Greedy: always deterministic, zero diversity.\n";
    std::cout << "  Temperature: controls entropy of the full distribution.\n";
    std::cout << "  Top-k: hard cutoff at k candidates regardless of probability gaps.\n";
    std::cout << "  Top-p: soft cutoff adapts to the distribution — fewer tokens when peaked.\n";
}

// ── main ──────────────────────────────────────────────────────────────────────

int main() {
    std::cout << "Chapter 14 — Inference and Sampling\n";
    std::cout << std::string(60, '=') << '\n';

    section_temperature();
    section_top_k();
    section_top_p();
    section_generation();
    section_comparison();

    std::cout << "\nDone.\n";
    return 0;
}
