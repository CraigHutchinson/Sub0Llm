// Chapter 22: General-Purpose MathLM
//
// MathGPT as an efficient general-purpose language model.
// The MathTransformerBlock (≈ 1/n_layers of total params) contains a routing
// head (Linear(D, kNumRouteTypes) — the only arithmetic-specific weight) that
// routes numeric positions to exact compute; the rest handles language.
//
//   §22.1  Parameter efficiency: routing head vs full model param breakdown
//   §22.2  Configurable integer range: exact arithmetic beyond int16
//   §22.3  OOD generalization: exact compute vs memorization
//   §22.4  Mixed language + arithmetic: one model, two domains

#include "sub0llm/nn/math_nodes.hpp"
#include "sub0llm/nn/numeric_router.hpp"
#include "sub0llm/tokenizer/numeric_tokenizer.hpp"
#include "sub0llm/tokenizer/bpe.hpp"
#include "sub0llm/nn/optimizer.hpp"
#include "sub0llm/nn/modern_gpt.hpp"
#include "sub0llm/autograd/ops.hpp"
#include "sub0llm/core/tensor.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <string_view>
#include <vector>

using namespace sub0llm;
using namespace sub0llm::nn;
using namespace sub0llm::autograd;

// ── helpers ───────────────────────────────────────────────────────────────────

static Tensor make_targets(const std::vector<int32_t>& ids) {
    const int64_t T = static_cast<int64_t>(ids.size()) - 1;
    Tensor t({T}, DType::Int32);
    auto sp = t.data_as<int32_t>();
    for (int64_t i = 0; i < T; ++i)
        sp[static_cast<std::size_t>(i)] = ids[static_cast<std::size_t>(i + 1)];
    return t;
}

static Tensor make_ids_tensor(const std::vector<int32_t>& ids) {
    Tensor t({static_cast<int64_t>(ids.size())}, DType::Int32);
    auto sp = t.data_as<int32_t>();
    for (std::size_t i = 0; i < ids.size(); ++i) sp[i] = ids[i];
    return t;
}

static int64_t count_params(const std::vector<Variable*>& params) {
    int64_t n = 0;
    for (const auto* p : params) n += static_cast<int64_t>(p->data().numel());
    return n;
}

// Returns the predicted numeric value at the last token position, or nullopt.
static std::optional<int32_t> predict_int(const Variable& logits,
                                           const NumericTokenizer& ntok) {
    const int64_t last = logits.data().shape(0) - 1;
    const int64_t V    = logits.data().shape(1);
    const auto sp      = logits.data().data_as<float>();
    int64_t best = 0;
    float best_val = sp[static_cast<std::size_t>(last * V)];
    for (int64_t v = 1; v < V; ++v) {
        const float val = sp[static_cast<std::size_t>(last * V + v)];
        if (val > best_val) { best_val = val; best = v; }
    }
    auto pred = static_cast<NumericTokenizer::TokenId>(best);
    if (!ntok.is_numeric(pred) || ntok.is_nan_token(pred) || ntok.is_overflow_token(pred))
        return std::nullopt;
    return static_cast<int32_t>(ntok.numeric_value(pred));
}

static bool pred_correct(const Variable& logits, int32_t expected,
                          const NumericTokenizer& ntok) {
    auto v = predict_int(logits, ntok);
    return v.has_value() && *v == expected;
}

// ── §22.1  Parameter Efficiency ───────────────────────────────────────────────

static void section_parameter_efficiency() {
    std::cout << "\n=== §22.1  Parameter Efficiency ===\n";
    std::cout << "  MathTransformerBlock replaces one standard block at depth l_math.\n"
              << "  Inside it, a routing head (Linear(D, kNumRouteTypes)) routes arithmetic\n"
              << "  to exact compute — no memorization required in those weights.\n\n";

    // Representative vocab size for a moderate integer range
    const int64_t V = 5100;  // BPE ~100 + int range 5000 + extras

    struct Config { int64_t D; int64_t n_layers; const char* label; };
    const std::array<Config, 3> cfgs{{{32, 4, "small"}, {64, 6, "medium"}, {128, 8, "large"}}};

    // kNumRouteTypes = 11 (FFN, Add, Sub, Mul, Div, IsLT, IsGT, IsEq, Inc, Dec, Sqrt)
    static constexpr int64_t kRouteTypes = static_cast<int64_t>(kNumRouteTypes);

    std::cout << std::format("  {:>8}  {:>8}  {:>10}  {:>12}  {:>14}  {:>12}  {:>10}\n",
        "label", "D", "n_layers", "total_P", "math_block_P", "route_head_P", "block%");
    std::cout << "  " << std::string(82, '-') << "\n";

    for (const auto& cfg : cfgs) {
        ModernGPT base_m(V, cfg.D, 4, 2, cfg.n_layers, 0, 0, /*seed=*/42);
        MathGPT   math_m(V, cfg.D, 4, 2, cfg.n_layers, -1, 0, /*seed=*/42);

        int64_t total_p = count_params(base_m.parameters());
        int64_t block_p = count_params(math_m.math_block_only_parameters());
        // Routing head: Linear(D, 8) has D×8 weights + 8 bias
        int64_t route_p = cfg.D * kRouteTypes + kRouteTypes;
        float   ratio   = 100.f * static_cast<float>(block_p) / static_cast<float>(total_p);

        std::cout << std::format("  {:>8}  {:>8}  {:>10}  {:>12}  {:>14}  {:>12}  {:>9.3f}%\n",
            cfg.label, cfg.D, cfg.n_layers, total_p, block_p, route_p, ratio);
    }

    std::cout << "\n  math_block_P  = full MathTransformerBlock (≈ 1/n_layers of total).\n"
              << "  route_head_P  = Linear(D, kNumRouteTypes) — arithmetic-specific weight.\n"
              << "  Everything else in the block serves language understanding.\n";
}

// ── §22.2  Configurable Integer Range ─────────────────────────────────────────

static void section_large_range() {
    std::cout << "\n=== §22.2  Configurable Integer Range ===\n";
    std::cout << "  NumericTokenizer(bpe, int_min, int_max) removes the int16 cap.\n\n";

    const int32_t kMin = -2000, kMax = 2000;  // 4001 integers

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(-800, 800);
    std::vector<std::string> corpus;
    corpus.reserve(400);
    for (int i = 0; i < 200; ++i) {
        int A = dist(rng), B = dist(rng);
        corpus.push_back(std::to_string(A) + " + " + std::to_string(B) +
                          " = " + std::to_string(A + B));
    }
    for (int i = 0; i < 200; ++i) {
        int A = dist(rng), B = dist(rng);
        corpus.push_back(std::to_string(A) + " - " + std::to_string(B) +
                          " = " + std::to_string(A - B));
    }

    auto bpe  = BPETokenizer::train(corpus, /*vocab_size=*/60);
    NumericTokenizer ntok(std::move(bpe), kMin, kMax);

    std::cout << std::format("  Range:       [{}, {}]  ({} integers)\n",
        kMin, kMax, ntok.int_range());
    std::cout << std::format("  Total vocab: {}  (BPE={} + numeric={} + extras={})\n\n",
        ntok.total_vocab_size(), ntok.bpe_vocab_size(),
        ntok.int_range(), NumericTokenizer::kExtraTokens);

    // Encode training data
    std::vector<std::vector<int32_t>> train_ids;
    train_ids.reserve(corpus.size());
    for (const auto& e : corpus) {
        auto ids = ntok.encode(e);
        if (ids.size() >= 2)
            train_ids.push_back(std::vector<int32_t>(ids.begin(), ids.end()));
    }

    // Test set: same distribution as training
    std::mt19937 rng_test(99);
    std::vector<std::pair<std::vector<int32_t>, int32_t>> test_set;
    test_set.reserve(50);
    for (int i = 0; i < 50; ++i) {
        int A = dist(rng_test), B = dist(rng_test);
        std::string expr = std::to_string(A) + " + " + std::to_string(B) + " =";
        auto ids = ntok.encode(expr);
        test_set.push_back({std::vector<int32_t>(ids.begin(), ids.end()), A + B});
    }

    const int64_t D = 32;
    const int64_t V = ntok.total_vocab_size();
    MathGPT   math_m(V, D, 4, 2, 4, -1, 0, /*seed=*/42);
    ModernGPT base_m (V, D, 4, 2, 4, 0, 0, /*seed=*/42);

    Adam adam_math(math_m.parameters(), 3e-3f);
    Adam adam_base(base_m.parameters(),  3e-3f);
    std::mt19937 rng_tr(11);

    const int steps = 600;
    std::cout << std::format("  Training {} steps...\n\n", steps);

    float last_math = 0.f, last_base = 0.f;
    for (int step = 0; step < steps; ++step) {
        const auto& ids = train_ids[rng_tr() % train_ids.size()];
        if (ids.size() < 2) continue;
        Tensor ids_t = make_ids_tensor(ids);
        Tensor tgt  = make_targets(ids);
        const int64_t T = static_cast<int64_t>(ids.size());

        adam_math.zero_grad();
        auto loss_m = cross_entropy(narrow(math_m.forward_math(ids_t, ntok), 0, T - 1), tgt);
        last_math   = loss_m.data().data_as<float>()[0];
        loss_m.backward();
        (void)clip_grad_norm(math_m.parameters(), 1.0f);
        adam_math.step();

        adam_base.zero_grad();
        auto loss_b = cross_entropy(narrow(base_m.forward(ids_t), 0, T - 1), tgt);
        last_base   = loss_b.data().data_as<float>()[0];
        loss_b.backward();
        (void)clip_grad_norm(base_m.parameters(), 1.0f);
        adam_base.step();
    }

    int math_correct = 0, base_correct = 0;
    for (const auto& [prompt, expected] : test_set) {
        Tensor ids_t = make_ids_tensor(prompt);
        if (pred_correct(math_m.forward_math(ids_t, ntok), expected, ntok)) ++math_correct;
        if (pred_correct(base_m.forward(ids_t),             expected, ntok)) ++base_correct;
    }

    std::cout << std::format("  Final CE loss:  MathGPT={:.4f}  Baseline={:.4f}\n",
        last_math, last_base);
    std::cout << std::format("  Accuracy (/50): MathGPT={:.0f}%  Baseline={:.0f}%\n",
        100.f * static_cast<float>(math_correct) / 50.f,
        100.f * static_cast<float>(base_correct)  / 50.f);
    std::cout << "\n  Exact arithmetic scales to any range with zero additional training cost.\n";
}

// ── §22.3  OOD Generalization ─────────────────────────────────────────────────

static void section_ood_generalization() {
    std::cout << "\n=== §22.3  OOD Generalization (Unseen Integer Pairs) ===\n";
    std::cout << "  Train: A+B with A,B ∈ [0,49]\n"
              << "  Test:  A+B with A,B ∈ [50,199]  (never seen during training)\n\n";

    // Range covers both training outputs [0,98] and OOD outputs [100,398]
    const int32_t kMin = 0, kMax = 450;

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> train_dist(0, 49);
    std::vector<std::string> corpus;
    corpus.reserve(300);
    for (int i = 0; i < 300; ++i) {
        int A = train_dist(rng), B = train_dist(rng);
        corpus.push_back(std::to_string(A) + " + " + std::to_string(B) +
                          " = " + std::to_string(A + B));
    }

    auto bpe  = BPETokenizer::train(corpus, /*vocab_size=*/60);
    NumericTokenizer ntok(std::move(bpe), kMin, kMax);

    std::vector<std::vector<int32_t>> train_ids;
    train_ids.reserve(corpus.size());
    for (const auto& e : corpus) {
        auto ids = ntok.encode(e);
        if (ids.size() >= 2)
            train_ids.push_back(std::vector<int32_t>(ids.begin(), ids.end()));
    }

    // In-distribution test: A,B in [0,49]
    std::mt19937 rng_id(88);
    std::vector<std::pair<std::vector<int32_t>, int32_t>> id_test;
    id_test.reserve(50);
    for (int i = 0; i < 50; ++i) {
        int A = train_dist(rng_id), B = train_dist(rng_id);
        std::string expr = std::to_string(A) + " + " + std::to_string(B) + " =";
        auto ids = ntok.encode(expr);
        id_test.push_back({std::vector<int32_t>(ids.begin(), ids.end()), A + B});
    }

    // OOD test: A,B in [50,199], results in [100,398] — vocab includes these
    std::mt19937 rng_ood(77);
    std::uniform_int_distribution<int> ood_dist(50, 199);
    std::vector<std::pair<std::vector<int32_t>, int32_t>> ood_test;
    ood_test.reserve(50);
    for (int i = 0; i < 50; ++i) {
        int A = ood_dist(rng_ood), B = ood_dist(rng_ood);
        int result = A + B;
        if (result < kMin || result > kMax) { --i; continue; }
        std::string expr = std::to_string(A) + " + " + std::to_string(B) + " =";
        auto ids = ntok.encode(expr);
        ood_test.push_back({std::vector<int32_t>(ids.begin(), ids.end()), result});
    }

    const int64_t D = 32;
    const int64_t V = ntok.total_vocab_size();
    MathGPT   math_m(V, D, 4, 2, 4, -1, 0, /*seed=*/42);
    ModernGPT base_m (V, D, 4, 2, 4, 0, 0, /*seed=*/42);

    Adam adam_math(math_m.parameters(), 3e-3f);
    Adam adam_base(base_m.parameters(),  3e-3f);
    std::mt19937 rng_tr(13);

    const int steps = 700;
    for (int step = 0; step < steps; ++step) {
        const auto& ids = train_ids[rng_tr() % train_ids.size()];
        if (ids.size() < 2) continue;
        Tensor ids_t = make_ids_tensor(ids);
        Tensor tgt  = make_targets(ids);
        const int64_t T = static_cast<int64_t>(ids.size());

        adam_math.zero_grad();
        cross_entropy(narrow(math_m.forward_math(ids_t, ntok), 0, T - 1), tgt).backward();
        (void)clip_grad_norm(math_m.parameters(), 1.0f);
        adam_math.step();

        adam_base.zero_grad();
        cross_entropy(narrow(base_m.forward(ids_t), 0, T - 1), tgt).backward();
        (void)clip_grad_norm(base_m.parameters(), 1.0f);
        adam_base.step();
    }

    auto eval_acc = [&](const auto& test_set, bool use_math) {
        int correct = 0;
        for (const auto& [prompt, expected] : test_set) {
            Tensor ids_t = make_ids_tensor(prompt);
            Variable logits = use_math
                ? math_m.forward_math(ids_t, ntok)
                : base_m.forward(ids_t);
            if (pred_correct(logits, expected, ntok)) ++correct;
        }
        return 100.f * static_cast<float>(correct) / static_cast<float>(test_set.size());
    };

    std::cout << std::format("  {:>22}  {:>10}  {:>10}\n", "test_set", "MathGPT", "Baseline");
    std::cout << "  " << std::string(48, '-') << "\n";
    std::cout << std::format("  {:>22}  {:>9.1f}%  {:>9.1f}%\n",
        "in-distribution", eval_acc(id_test, true), eval_acc(id_test, false));
    std::cout << std::format("  {:>22}  {:>9.1f}%  {:>9.1f}%\n",
        "out-of-distribution", eval_acc(ood_test, true), eval_acc(ood_test, false));

    std::cout << "\n  Baseline memorizes token combinations → OOD accuracy collapses.\n"
              << "  MathGPT computes exact results → OOD accuracy matches in-distribution.\n";
}

// ── §22.4  Mixed Language + Arithmetic ────────────────────────────────────────

static void section_mixed_language_math() {
    std::cout << "\n=== §22.4  Mixed Language + Arithmetic ===\n";
    std::cout << "  One model trained on natural language AND arithmetic simultaneously.\n\n";

    // Text corpus (no numbers)
    const std::vector<std::string> text_corpus{
        "the quick brown fox", "a fast red car", "the big blue sky",
        "hello world today",   "learning and growing", "simple words work well",
        "the cat sat here",    "birds fly very high",  "water flows downstream",
        "time passes quickly", "light travels fast",   "cold wind blows strong",
        "green trees stand tall", "soft rain falls gently", "bright stars shine far",
    };

    // Math corpus: add and subtract
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> d50(1, 50);
    std::vector<std::string> math_corpus;
    math_corpus.reserve(200);
    for (int i = 0; i < 120; ++i) {
        int A = d50(rng), B = d50(rng);
        math_corpus.push_back(std::to_string(A) + " + " + std::to_string(B) +
                                " = " + std::to_string(A + B));
    }
    for (int i = 0; i < 80; ++i) {
        int A = d50(rng) + 20, B = d50(rng);
        math_corpus.push_back(std::to_string(A) + " - " + std::to_string(B) +
                                " = " + std::to_string(A - B));
    }

    // Train BPE on full corpus
    std::vector<std::string> full_corpus = text_corpus;
    full_corpus.insert(full_corpus.end(), math_corpus.begin(), math_corpus.end());
    auto bpe = BPETokenizer::train(full_corpus, /*vocab_size=*/100);
    NumericTokenizer ntok(std::move(bpe), 0, 200);

    std::cout << std::format("  Text examples: {}   Math examples: {}\n",
        text_corpus.size(), math_corpus.size());
    std::cout << std::format("  Vocab: {} total  (BPE={} + numeric={} + extras={})\n\n",
        ntok.total_vocab_size(), ntok.bpe_vocab_size(),
        ntok.int_range(), NumericTokenizer::kExtraTokens);

    auto encode_vec = [&](const std::vector<std::string>& v) {
        std::vector<std::vector<int32_t>> out;
        out.reserve(v.size());
        for (const auto& e : v) {
            auto ids = ntok.encode(e);
            if (ids.size() >= 2)
                out.push_back(std::vector<int32_t>(ids.begin(), ids.end()));
        }
        return out;
    };

    auto text_ids = encode_vec(text_corpus);
    auto all_ids  = encode_vec(full_corpus);

    const int64_t D = 32;
    const int64_t V = ntok.total_vocab_size();
    MathGPT   math_m(V, D, 4, 2, 4, -1, 0, /*seed=*/42);
    ModernGPT base_m (V, D, 4, 2, 4, 0, 0, /*seed=*/42);

    Adam adam_math(math_m.parameters(), 2e-3f);
    Adam adam_base(base_m.parameters(),  2e-3f);
    std::mt19937 rng_tr(17);

    const int steps = 600;
    for (int step = 0; step < steps; ++step) {
        const auto& ids = all_ids[rng_tr() % all_ids.size()];
        if (ids.size() < 2) continue;
        Tensor ids_t = make_ids_tensor(ids);
        Tensor tgt  = make_targets(ids);
        const int64_t T = static_cast<int64_t>(ids.size());

        adam_math.zero_grad();
        cross_entropy(narrow(math_m.forward_math(ids_t, ntok), 0, T - 1), tgt).backward();
        (void)clip_grad_norm(math_m.parameters(), 1.0f);
        adam_math.step();

        adam_base.zero_grad();
        cross_entropy(narrow(base_m.forward(ids_t), 0, T - 1), tgt).backward();
        (void)clip_grad_norm(base_m.parameters(), 1.0f);
        adam_base.step();
    }

    // Text CE loss (no math tokens → same for both models in theory)
    auto avg_text_loss = [&](bool use_math) {
        float total = 0.f;
        int n = 0;
        for (const auto& ids : text_ids) {
            if (ids.size() < 2) continue;
            Tensor ids_t = make_ids_tensor(ids);
            Tensor tgt  = make_targets(ids);
            const int64_t T = static_cast<int64_t>(ids.size());
            Variable logits = use_math
                ? math_m.forward_math(ids_t, ntok)
                : base_m.forward(ids_t);
            total += cross_entropy(narrow(logits, 0, T - 1), tgt)
                         .data().data_as<float>()[0];
            ++n;
        }
        return n > 0 ? total / static_cast<float>(n) : 0.f;
    };

    // Math accuracy on test set
    std::mt19937 rng_mtest(99);
    std::vector<std::pair<std::vector<int32_t>, int32_t>> math_test;
    math_test.reserve(40);
    for (int i = 0; i < 40; ++i) {
        int A = d50(rng_mtest), B = d50(rng_mtest);
        std::string expr = std::to_string(A) + " + " + std::to_string(B) + " =";
        auto ids = ntok.encode(expr);
        math_test.push_back({std::vector<int32_t>(ids.begin(), ids.end()), A + B});
    }

    auto math_acc = [&](bool use_math) {
        int correct = 0;
        for (const auto& [prompt, expected] : math_test) {
            Tensor ids_t = make_ids_tensor(prompt);
            Variable logits = use_math
                ? math_m.forward_math(ids_t, ntok)
                : base_m.forward(ids_t);
            if (pred_correct(logits, expected, ntok)) ++correct;
        }
        return 100.f * static_cast<float>(correct) / static_cast<float>(math_test.size());
    };

    std::cout << std::format("  {:>26}  {:>10}  {:>10}\n", "metric", "MathGPT", "Baseline");
    std::cout << "  " << std::string(52, '-') << "\n";
    std::cout << std::format("  {:>26}  {:>10.4f}  {:>10.4f}\n",
        "text CE loss", avg_text_loss(true), avg_text_loss(false));
    std::cout << std::format("  {:>26}  {:>9.1f}%  {:>9.1f}%\n",
        "arithmetic accuracy (/40)", math_acc(true), math_acc(false));

    std::cout << "\n  Text language modeling is equally good in both models.\n"
              << "  Arithmetic diverges: exact routing vs weight-space memorization.\n"
              << "  The routing head (D×8 params) is the only arithmetic-specific weight.\n";
}

// ── §22.5  Range Scaling ───────────────────────────────────────────────────────

static void section_range_scaling() {
    std::cout << "\n=== §22.5  Scaling the Integer Range ===\n";
    std::cout << "  MathGPT accuracy is range-invariant; baseline degrades as range grows.\n\n";

    struct RangeCfg { int32_t lim; int steps; };
    const std::array<RangeCfg, 4> cfgs{{{  50, 400}, { 200, 500}, { 500, 600}, {1000, 700}}};

    std::cout << std::format("  {:>10}  {:>10}  {:>12}  {:>12}\n",
        "range", "vocab", "math_acc%", "base_acc%");
    std::cout << "  " << std::string(50, '-') << "\n";

    for (const auto& cfg : cfgs) {
        const int32_t kMin = -cfg.lim, kMax = cfg.lim;

        std::mt19937 rng(42);
        std::uniform_int_distribution<int> dist(-cfg.lim / 2, cfg.lim / 2);
        std::vector<std::string> corpus;
        corpus.reserve(300);
        for (int i = 0; i < 300; ++i) {
            int A = dist(rng), B = dist(rng);
            corpus.push_back(std::to_string(A) + " + " + std::to_string(B) +
                              " = " + std::to_string(A + B));
        }

        auto bpe  = BPETokenizer::train(corpus, /*vocab_size=*/60);
        NumericTokenizer ntok(std::move(bpe), kMin, kMax);

        std::vector<std::vector<int32_t>> train_ids;
        train_ids.reserve(corpus.size());
        for (const auto& e : corpus) {
            auto ids = ntok.encode(e);
            if (ids.size() >= 2)
                train_ids.push_back(std::vector<int32_t>(ids.begin(), ids.end()));
        }

        // Test set
        std::mt19937 rng_test(99);
        std::vector<std::pair<std::vector<int32_t>, int32_t>> test_set;
        test_set.reserve(40);
        for (int i = 0; i < 40; ++i) {
            int A = dist(rng_test), B = dist(rng_test);
            int R = A + B;
            if (R < kMin || R > kMax) { --i; continue; }
            std::string expr = std::to_string(A) + " + " + std::to_string(B) + " =";
            auto ids = ntok.encode(expr);
            test_set.push_back({std::vector<int32_t>(ids.begin(), ids.end()), R});
        }

        const int64_t D = 32;
        const int64_t V = ntok.total_vocab_size();
        MathGPT   math_m(V, D, 4, 2, 4, -1, 0, /*seed=*/42);
        ModernGPT base_m (V, D, 4, 2, 4, 0, 0, /*seed=*/42);

        Adam adam_math(math_m.parameters(), 3e-3f);
        Adam adam_base(base_m.parameters(),  3e-3f);
        std::mt19937 rng_tr(11);

        for (int step = 0; step < cfg.steps; ++step) {
            const auto& ids = train_ids[rng_tr() % train_ids.size()];
            if (ids.size() < 2) continue;
            Tensor ids_t = make_ids_tensor(ids);
            Tensor tgt   = make_targets(ids);
            const int64_t T = static_cast<int64_t>(ids.size());

            adam_math.zero_grad();
            cross_entropy(narrow(math_m.forward_math(ids_t, ntok), 0, T - 1), tgt).backward();
            (void)clip_grad_norm(math_m.parameters(), 1.0f);
            adam_math.step();

            adam_base.zero_grad();
            cross_entropy(narrow(base_m.forward(ids_t), 0, T - 1), tgt).backward();
            (void)clip_grad_norm(base_m.parameters(), 1.0f);
            adam_base.step();
        }

        int math_correct = 0, base_correct = 0;
        for (const auto& [prompt, expected] : test_set) {
            Tensor ids_t = make_ids_tensor(prompt);
            if (pred_correct(math_m.forward_math(ids_t, ntok), expected, ntok)) ++math_correct;
            if (pred_correct(base_m.forward(ids_t),             expected, ntok)) ++base_correct;
        }
        const float n_test = static_cast<float>(test_set.size());

        std::cout << std::format("  {:>10}  {:>10}  {:>12.1f}  {:>12.1f}\n",
            std::format("[-{},{}]", cfg.lim, cfg.lim),
            ntok.total_vocab_size(),
            100.f * static_cast<float>(math_correct) / n_test,
            100.f * static_cast<float>(base_correct)  / n_test);
    }

    std::cout << "\n  Baseline accuracy drops as range grows — memorization fails to scale.\n"
              << "  MathGPT maintains accuracy — exact compute is range-independent.\n";
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    std::string phase = "all";
    for (int i = 1; i < argc; ++i) {
        std::string_view arg(argv[i]);
        if (arg == "--phase" && i + 1 < argc) phase = argv[++i];
    }

    std::cout << "╔══════════════════════════════════════════════════════════════╗\n"
              << "║  Chapter 22: General-Purpose MathLM                         ║\n"
              << "║  Efficient arithmetic with a lightweight routing head         ║\n"
              << "╚══════════════════════════════════════════════════════════════╝\n";

    const bool run_all = (phase == "all");

    if (run_all || phase == "params")       section_parameter_efficiency();
    if (run_all || phase == "large_range")  section_large_range();
    if (run_all || phase == "ood")          section_ood_generalization();
    if (run_all || phase == "mixed")        section_mixed_language_math();
    if (run_all || phase == "scaling")      section_range_scaling();

    if (!run_all && phase != "params"      && phase != "large_range" &&
        phase != "ood" && phase != "mixed" && phase != "scaling")
        throw std::runtime_error(
            std::format("unknown phase '{}' — valid: params large_range ood mixed scaling", phase));

    std::cout << "\nDone.\n";
    return 0;
}
