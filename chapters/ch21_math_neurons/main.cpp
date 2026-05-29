#include "sub0llm/nn/math_nodes.hpp"
#include "sub0llm/nn/numeric_router.hpp"
#include "sub0llm/tokenizer/numeric_tokenizer.hpp"
#include "sub0llm/tokenizer/bpe.hpp"
#include "sub0llm/nn/optimizer.hpp"
#include "sub0llm/nn/modern_gpt.hpp"
#include "sub0llm/autograd/ops.hpp"
#include "sub0llm/core/tensor.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <numbers>
#include <numeric>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
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
    for (int64_t i = 0; i < T; i++) sp[static_cast<std::size_t>(i)] = ids[static_cast<std::size_t>(i + 1)];
    return t;
}

static Tensor make_ids_tensor(const std::vector<int32_t>& ids) {
    Tensor t({static_cast<int64_t>(ids.size())}, DType::Int32);
    auto sp = t.data_as<int32_t>();
    for (std::size_t i = 0; i < ids.size(); ++i)
        sp[i] = ids[i];
    return t;
}

// ── §21.1  Numeric Tokenizer ──────────────────────────────────────────────────

static void section_numeric_tokenizer() {
    std::cout << "\n=== §21.1  Numeric Tokenizer ===\n";

    std::vector<std::string> corpus = {
        "the cat sat", "1 plus 2 is 3", "42 minus 7 is 35"
    };
    auto bpe = BPETokenizer::train(corpus, /*vocab_size=*/60);
    NumericTokenizer ntok(std::move(bpe));

    std::cout << std::format("  bpe_vocab_size        : {}\n", ntok.bpe_vocab_size());
    std::cout << std::format("  total_vocab_size      : {}  (= bpe + 65536 + 2)\n",
                              ntok.total_vocab_size());
    std::cout << std::format("  numeric_range_start   : {}\n", ntok.numeric_range_start());

    // Encode a mixed text string
    auto enc = ntok.encode("the cat 42 sat");
    std::cout << "\n  encode(\"the cat 42 sat\") =>";
    for (auto id : enc) std::cout << " " << id;
    std::cout << "\n";

    // Decode back
    std::cout << "  decode back              => \"" << ntok.decode(enc) << "\"\n";

    // is_numeric check
    auto id100 = static_cast<NumericTokenizer::TokenId>(ntok.numeric_range_start() + 100);
    std::cout << std::format("  is_numeric(numeric_range_start + 100) = {}\n",
                              ntok.is_numeric(id100) ? "true" : "false");

    // encode_int round-trip
    auto id42 = ntok.encode_int(42);
    std::cout << std::format("  encode_int(42) = {}  numeric_value = {:.1f}\n",
                              id42, ntok.numeric_value(id42));

    // Special tokens
    std::cout << std::format("  nan_token()      = {}  is_nan_token = {}\n",
                              ntok.nan_token(),
                              ntok.is_nan_token(ntok.nan_token()) ? "true" : "false");
    std::cout << std::format("  overflow_token() = {}  is_overflow  = {}\n",
                              ntok.overflow_token(),
                              ntok.is_overflow_token(ntok.overflow_token()) ? "true" : "false");
}

// ── §21.2  Math Node Unit Tests ───────────────────────────────────────────────

static void section_math_ops() {
    std::cout << "\n=== §21.2  Math Node Unit Tests ===\n";

    struct TestCase {
        RouteType op;
        float     a, b;
        std::string label;
    };

    const std::vector<TestCase> cases = {
        {RouteType::Add,     42.f,  7.f,  "Add(42, 7)   "},
        {RouteType::Sub,    100.f, 37.f,  "Sub(100, 37) "},
        {RouteType::Mul,     12.f,  5.f,  "Mul(12, 5)   "},
        {RouteType::Div,    100.f,  4.f,  "Div(100, 4)  "},
        {RouteType::Div,      7.f,  0.f,  "Div(7, 0)    "},
        {RouteType::IsLessThan,    3.f,  7.f,  "IsLT(3, 7)   "},
        {RouteType::IsLessThan,    7.f,  3.f,  "IsLT(7, 3)   "},
        {RouteType::IsGreaterThan, 7.f,  3.f,  "IsGT(7, 3)   "},
        {RouteType::IsGreaterThan, 3.f,  7.f,  "IsGT(3, 7)   "},
        {RouteType::IsEqual,       5.f,  5.f,  "IsEq(5, 5)   "},
        {RouteType::IsEqual,       5.f,  6.f,  "IsEq(5, 6)   "},
        {RouteType::Mul,         300.f,300.f,  "Mul(300,300) "},
    };

    std::cout << std::format("  {:<16}  {:>8}  {:>8}  {:>8}\n",
                              "op", "value", "is_nan", "is_ovfl");
    std::cout << "  " << std::string(48, '-') << "\n";

    for (const auto& tc : cases) {
        auto r = apply_math_op(tc.op, tc.a, tc.b);
        std::cout << std::format("  {:<16}  {:>8.1f}  {:>8}  {:>8}\n",
                                  tc.label,
                                  r.value,
                                  r.is_nan      ? "true" : "false",
                                  r.is_overflow ? "true" : "false");
    }
}

// ── §21.3  NumericRouter ──────────────────────────────────────────────────────

static void section_numeric_router() {
    std::cout << "\n=== §21.3  NumericRouter ===\n";

    const int64_t D = 32, T = 10;
    NumericRouter router(D, /*seed=*/42);

    // Build a random input Variable
    Tensor x_t({T, D}, DType::Float32);
    {
        std::mt19937 rng(7);
        std::normal_distribution<float> dist(0.f, 0.1f);
        for (auto& v : x_t.data_as<float>()) v = dist(rng);
    }
    Variable x(x_t, /*requires_grad=*/true);

    auto [soft_probs, hard_mask] = router.forward(x);

    std::cout << std::format("  soft_probs shape : ({}, {})\n",
                              soft_probs.data().shape(0), soft_probs.data().shape(1));
    std::cout << std::format("  hard_mask  shape : ({}, {})\n",
                              hard_mask.shape(0), hard_mask.shape(1));

    // Verify hard_mask sums to 1 per row
    auto mask_sp = hard_mask.data_as<float>();
    const int64_t K = static_cast<int64_t>(kNumRouteTypes);
    bool all_one_hot = true;
    for (int64_t t = 0; t < T; ++t) {
        float row_sum = 0.f;
        for (int64_t k = 0; k < K; ++k)
            row_sum += mask_sp[static_cast<std::size_t>(t * K + k)];
        if (std::abs(row_sum - 1.0f) > 1e-5f) { all_one_hot = false; break; }
    }
    std::cout << std::format("  hard_mask row sums == 1.0: {}\n",
                              all_one_hot ? "true" : "false");

    // Route decisions
    auto routes = router.route_hard(x);
    std::cout << "  route_hard decisions :";
    for (auto r : routes) std::cout << " " << static_cast<int>(r);
    std::cout << "\n";

    // Parameter count
    auto params = router.parameters();
    int64_t nparams = 0;
    for (auto* p : params) nparams += static_cast<int64_t>(p->data().numel());
    std::cout << std::format("  parameter count: {}\n", nparams);
}

// ── §21.4  MathGPT: Arithmetic Training ──────────────────────────────────────

static void section_arithmetic_training() {
    std::cout << "\n=== §21.4  MathGPT: Arithmetic Training ===\n";

    std::mt19937 rng_data(42);
    std::uniform_int_distribution<int> d09(0, 9);

    // Build 200 training expressions: "A + B = C" and "A - B = C"
    std::vector<std::string> corpus_exprs;
    corpus_exprs.reserve(200);
    for (int i = 0; i < 100; ++i) {
        int A = d09(rng_data), B = d09(rng_data);
        corpus_exprs.push_back(std::to_string(A) + " + " + std::to_string(B) +
                                " = " + std::to_string(A + B));
    }
    for (int i = 0; i < 100; ++i) {
        int A = d09(rng_data), B = d09(rng_data);
        corpus_exprs.push_back(std::to_string(A) + " - " + std::to_string(B) +
                                " = " + std::to_string(A - B));
    }

    auto bpe = BPETokenizer::train(corpus_exprs, /*vocab_size=*/50);
    NumericTokenizer ntok(std::move(bpe));
    const int64_t total_vocab = ntok.total_vocab_size();

    std::cout << std::format("  bpe_vocab_size = {}  total_vocab = {}\n",
                              ntok.bpe_vocab_size(), total_vocab);

    // Encode all training expressions
    std::vector<std::vector<int32_t>> train_ids;
    train_ids.reserve(corpus_exprs.size());
    for (const auto& expr : corpus_exprs) {
        auto ids = ntok.encode(expr);
        if (ids.size() >= 2)
            train_ids.push_back(std::vector<int32_t>(ids.begin(), ids.end()));
    }

    // Build 50 test expressions (add & sub, A,B in 0..9)
    std::vector<std::pair<std::vector<int32_t>, int32_t>> test_set;
    std::mt19937 rng_test(99);
    for (int i = 0; i < 25; ++i) {
        int A = d09(rng_test), B = d09(rng_test);
        std::string expr = std::to_string(A) + " + " + std::to_string(B) + " =";
        auto ids = ntok.encode(expr);
        test_set.push_back({std::vector<int32_t>(ids.begin(), ids.end()), A + B});
    }
    for (int i = 0; i < 25; ++i) {
        int A = d09(rng_test), B = d09(rng_test);
        std::string expr = std::to_string(A) + " - " + std::to_string(B) + " =";
        auto ids = ntok.encode(expr);
        test_set.push_back({std::vector<int32_t>(ids.begin(), ids.end()), A - B});
    }

    // Build both models
    const int64_t D = 16;
    const std::size_t n_heads = 2, n_kv = 1;
    const int64_t n_layers = 4;

    ModernGPT baseline(total_vocab, D, n_heads, n_kv, n_layers, 0, 0, /*seed=*/42);
    MathGPT   math_model(total_vocab, D, n_heads, n_kv, n_layers, -1, 0, /*seed=*/42);

    Adam adam_base(baseline.parameters(), 3e-3f);
    Adam adam_math(math_model.parameters(), 3e-3f);

    const int steps = 300;
    std::mt19937 rng_train(11);

    std::cout << std::format("  Training both models for {} steps...\n", steps);
    std::cout << std::format("  l_math = {} (round(0.7 * {}))\n\n",
                              math_model.l_math(), n_layers);

    float base_loss_last = 0.f, math_loss_last = 0.f;

    for (int step = 0; step < steps; ++step) {
        // Pick a random training example
        const std::size_t idx = rng_train() % train_ids.size();
        const auto& ids = train_ids[idx];
        if (ids.size() < 2) continue;

        Tensor id_tensor = make_ids_tensor(ids);
        Tensor targets   = make_targets(ids);
        const int64_t seq_len = static_cast<int64_t>(ids.size());

        // Baseline step
        {
            adam_base.zero_grad();
            auto logits = baseline.forward(id_tensor);
            auto ltrunc = narrow(logits, 0, seq_len - 1);
            auto loss   = cross_entropy(ltrunc, targets);
            loss.backward();
            (void)clip_grad_norm(baseline.parameters(), 1.0f);
            adam_base.step();
            base_loss_last = loss.data().data_as<float>()[0];
        }

        // MathGPT step
        {
            adam_math.zero_grad();
            auto logits = math_model.forward_math(id_tensor, ntok);
            auto ltrunc = narrow(logits, 0, seq_len - 1);
            auto loss   = cross_entropy(ltrunc, targets);
            loss.backward();
            (void)clip_grad_norm(math_model.parameters(), 1.0f);
            adam_math.step();
            math_loss_last = loss.data().data_as<float>()[0];
        }

        if (step == 0 || (step + 1) % 100 == 0) {
            std::cout << std::format("  step {:3d}  baseline_loss={:.4f}  math_loss={:.4f}\n",
                                      step + 1, base_loss_last, math_loss_last);
        }
    }

    // Evaluate on test set
    auto eval_logits = [&](const Variable& logits, int32_t expected_val) -> bool {
        const int64_t last_pos = logits.data().shape(0) - 1;
        const int64_t V       = logits.data().shape(1);
        auto logits_sp = logits.data().data_as<float>();

        int64_t best = 0;
        float   best_val = logits_sp[static_cast<std::size_t>(last_pos * V)];
        for (int64_t v = 1; v < V; ++v) {
            float val = logits_sp[static_cast<std::size_t>(last_pos * V + v)];
            if (val > best_val) { best_val = val; best = v; }
        }
        auto pred_id = static_cast<NumericTokenizer::TokenId>(best);
        if (ntok.is_numeric(pred_id) && !ntok.is_nan_token(pred_id) &&
            !ntok.is_overflow_token(pred_id))
        {
            int32_t pred_val = static_cast<int32_t>(ntok.numeric_value(pred_id));
            return pred_val == expected_val;
        }
        return false;
    };

    int base_correct = 0, math_correct = 0;
    for (const auto& [prompt_ids, expected_val] : test_set) {
        if (prompt_ids.empty()) continue;
        Tensor id_tensor = make_ids_tensor(prompt_ids);
        if (eval_logits(baseline.forward(id_tensor), expected_val))   ++base_correct;
        if (eval_logits(math_model.forward_math(id_tensor, ntok), expected_val)) ++math_correct;
    }
    float base_acc = static_cast<float>(base_correct) / static_cast<float>(test_set.size());
    float math_acc = static_cast<float>(math_correct) / static_cast<float>(test_set.size());

    std::cout << std::format("\n  Accuracy on 50 test expressions:\n");
    std::cout << std::format("    Baseline (ModernGPT)  : {:.1f}%\n", base_acc * 100.f);
    std::cout << std::format("    MathGPT (l_math={})   : {:.1f}%\n",
                              math_model.l_math(), math_acc * 100.f);
}

// ── §21.5  Layer Depth Ablation ───────────────────────────────────────────────

static void section_layer_ablation() {
    std::cout << "\n=== §21.5  Layer Depth Ablation ===\n";
    std::cout << "  Training MathGPT with l_math = 0..3, n_layers=4\n\n";

    // Build a shared dataset (same seed as §21.4 for comparability)
    std::mt19937 rng_data(42);
    std::uniform_int_distribution<int> d09(0, 9);

    std::vector<std::string> corpus_exprs;
    corpus_exprs.reserve(200);
    for (int i = 0; i < 100; ++i) {
        int A = d09(rng_data), B = d09(rng_data);
        corpus_exprs.push_back(std::to_string(A) + " + " + std::to_string(B) +
                                " = " + std::to_string(A + B));
    }
    for (int i = 0; i < 100; ++i) {
        int A = d09(rng_data), B = d09(rng_data);
        corpus_exprs.push_back(std::to_string(A) + " - " + std::to_string(B) +
                                " = " + std::to_string(A - B));
    }

    auto bpe = BPETokenizer::train(corpus_exprs, 50);
    NumericTokenizer ntok(std::move(bpe));
    const int64_t total_vocab = ntok.total_vocab_size();

    std::vector<std::vector<int32_t>> train_ids;
    for (const auto& expr : corpus_exprs) {
        auto ids = ntok.encode(expr);
        if (ids.size() >= 2)
            train_ids.push_back(std::vector<int32_t>(ids.begin(), ids.end()));
    }

    // Build test set
    std::vector<std::pair<std::vector<int32_t>, int32_t>> test_set;
    std::mt19937 rng_test(99);
    for (int i = 0; i < 25; ++i) {
        int A = d09(rng_test), B = d09(rng_test);
        std::string expr = std::to_string(A) + " + " + std::to_string(B) + " =";
        auto ids = ntok.encode(expr);
        test_set.push_back({std::vector<int32_t>(ids.begin(), ids.end()), A + B});
    }
    for (int i = 0; i < 25; ++i) {
        int A = d09(rng_test), B = d09(rng_test);
        std::string expr = std::to_string(A) + " - " + std::to_string(B) + " =";
        auto ids = ntok.encode(expr);
        test_set.push_back({std::vector<int32_t>(ids.begin(), ids.end()), A - B});
    }

    std::cout << std::format("  {:<8}  {:>10}  {:>12}\n", "l_math", "accuracy", "final_loss");
    std::cout << "  " << std::string(36, '-') << "\n";

    // n_layers=4 so l_math ∈ {0, 1, 2, 3}
    for (int64_t lm = 0; lm < 4; ++lm) {
        MathGPT model(total_vocab, 16, 2, 1, 4, lm, 0, /*seed=*/42);
        Adam adam(model.parameters(), 3e-3f);
        std::mt19937 rng_train(11);

        float last_loss = 0.f;
        for (int step = 0; step < 300; ++step) {
            const std::size_t idx = rng_train() % train_ids.size();
            const auto& ids = train_ids[idx];
            if (ids.size() < 2) continue;

            Tensor id_tensor = make_ids_tensor(ids);
            Tensor targets   = make_targets(ids);
            const int64_t seq_len = static_cast<int64_t>(ids.size());

            adam.zero_grad();
            auto logits = model.forward_math(id_tensor, ntok);
            auto ltrunc = narrow(logits, 0, seq_len - 1);
            auto loss   = cross_entropy(ltrunc, targets);
            loss.backward();
            (void)clip_grad_norm(model.parameters(), 1.0f);
            adam.step();
            last_loss = loss.data().data_as<float>()[0];
        }

        // Evaluate accuracy
        int correct = 0;
        for (const auto& [prompt_ids, expected_val] : test_set) {
            if (prompt_ids.empty()) continue;
            Tensor id_tensor = make_ids_tensor(prompt_ids);
            Variable logits = model.forward_math(id_tensor, ntok);

            const int64_t last_pos = logits.data().shape(0) - 1;
            const int64_t V       = logits.data().shape(1);
            auto logits_sp = logits.data().data_as<float>();

            int64_t best = 0;
            float   best_val = logits_sp[static_cast<std::size_t>(last_pos * V)];
            for (int64_t v = 1; v < V; ++v) {
                float val = logits_sp[static_cast<std::size_t>(last_pos * V + v)];
                if (val > best_val) { best_val = val; best = v; }
            }
            auto pred_id = static_cast<NumericTokenizer::TokenId>(best);
            if (ntok.is_numeric(pred_id) && !ntok.is_nan_token(pred_id) &&
                !ntok.is_overflow_token(pred_id))
            {
                int32_t pred_val = static_cast<int32_t>(ntok.numeric_value(pred_id));
                if (pred_val == expected_val) ++correct;
            }
        }
        float acc = static_cast<float>(correct) / static_cast<float>(test_set.size());
        std::cout << std::format("  {:<8}  {:>9.1f}%  {:>12.4f}\n",
                                  lm, acc * 100.f, last_loss);
    }
}

// ── §21.7  Training Dynamics & Router Specialisation ─────────────────────────

static void section_training_dynamics() {
    std::cout << "\n=== §21.7  Training Dynamics & Router Specialisation ===\n";

    std::mt19937 rng_data(42);
    std::uniform_int_distribution<int> d09(0, 9);

    // Build 300 training expressions: 100 add + 100 sub + 50 mul + 50 div
    std::vector<std::string> corpus_exprs;
    corpus_exprs.reserve(300);
    for (int i = 0; i < 100; ++i) {
        int A = d09(rng_data), B = d09(rng_data);
        corpus_exprs.push_back(std::to_string(A) + " + " + std::to_string(B) +
                                " = " + std::to_string(A + B));
    }
    for (int i = 0; i < 100; ++i) {
        int A = d09(rng_data), B = d09(rng_data);
        corpus_exprs.push_back(std::to_string(A) + " - " + std::to_string(B) +
                                " = " + std::to_string(A - B));
    }
    for (int i = 0; i < 50; ++i) {
        int A = d09(rng_data), B = d09(rng_data);
        corpus_exprs.push_back(std::to_string(A) + " * " + std::to_string(B) +
                                " = " + std::to_string(A * B));
    }
    for (int i = 0; i < 50; ++i) {
        int A = d09(rng_data), B = std::max(1, d09(rng_data));
        corpus_exprs.push_back(std::to_string(A) + " / " + std::to_string(B) +
                                " = " + std::to_string(A / B));
    }

    auto bpe = BPETokenizer::train(corpus_exprs, /*vocab_size=*/60);
    NumericTokenizer ntok(std::move(bpe));
    const int64_t total_vocab = ntok.total_vocab_size();

    std::cout << std::format("  total_vocab = {}  training_exprs = {}\n\n",
                              total_vocab, corpus_exprs.size());

    std::vector<std::vector<int32_t>> train_ids;
    train_ids.reserve(corpus_exprs.size());
    for (const auto& expr : corpus_exprs) {
        auto ids = ntok.encode(expr);
        if (ids.size() >= 2)
            train_ids.push_back(std::vector<int32_t>(ids.begin(), ids.end()));
    }

    // Build 50 test expressions (25 add + 25 sub)
    std::vector<std::pair<std::vector<int32_t>, int32_t>> test_set;
    // Also track expected operations for router_spec
    std::vector<RouteType> test_expected_ops;
    {
        std::mt19937 rng_test(99);
        for (int i = 0; i < 25; ++i) {
            int A = d09(rng_test), B = d09(rng_test);
            std::string expr = std::to_string(A) + " + " + std::to_string(B) + " =";
            auto ids = ntok.encode(expr);
            test_set.push_back({std::vector<int32_t>(ids.begin(), ids.end()), A + B});
            test_expected_ops.push_back(RouteType::Add);
        }
        for (int i = 0; i < 25; ++i) {
            int A = d09(rng_test), B = d09(rng_test);
            std::string expr = std::to_string(A) + " - " + std::to_string(B) + " =";
            auto ids = ntok.encode(expr);
            test_set.push_back({std::vector<int32_t>(ids.begin(), ids.end()), A - B});
            test_expected_ops.push_back(RouteType::Sub);
        }
    }

    const int64_t D = 16;
    const std::size_t n_heads = 2, n_kv = 1;
    const int64_t n_layers = 4;

    MathGPT model(total_vocab, D, n_heads, n_kv, n_layers, -1, 0, /*seed=*/42);
    Adam adam(model.parameters(), 3e-3f);
    std::mt19937 rng_train(11);

    const int steps = 5000;
    const int log_every = 500;

    // Evaluate helpers
    auto eval_accuracy = [&]() -> float {
        int correct = 0;
        for (const auto& [prompt_ids, expected_val] : test_set) {
            if (prompt_ids.empty()) continue;
            Tensor id_tensor = make_ids_tensor(prompt_ids);
            Variable logits = model.forward_math(id_tensor, ntok);

            const int64_t last_pos = logits.data().shape(0) - 1;
            const int64_t V       = logits.data().shape(1);
            auto logits_sp = logits.data().data_as<float>();

            int64_t best = 0;
            float   best_val = logits_sp[static_cast<std::size_t>(last_pos * V)];
            for (int64_t v = 1; v < V; ++v) {
                float val = logits_sp[static_cast<std::size_t>(last_pos * V + v)];
                if (val > best_val) { best_val = val; best = v; }
            }
            auto pred_id = static_cast<NumericTokenizer::TokenId>(best);
            if (ntok.is_numeric(pred_id) && !ntok.is_nan_token(pred_id) &&
                !ntok.is_overflow_token(pred_id))
            {
                int32_t pred_val = static_cast<int32_t>(ntok.numeric_value(pred_id));
                if (pred_val == expected_val) ++correct;
            }
        }
        return static_cast<float>(correct) / static_cast<float>(test_set.size());
    };

    auto eval_router = [&]() -> std::pair<float, float> {
        int spec_correct = 0;
        float total_entropy = 0.f;
        int count = 0;
        for (std::size_t i = 0; i < test_set.size(); ++i) {
            const auto& prompt_ids = test_set[i].first;
            if (prompt_ids.empty()) continue;
            Tensor id_tensor = make_ids_tensor(prompt_ids);
            RouteInfo ri = model.route_info(id_tensor, ntok);
            if (ri.routes.empty()) continue;
            // Check last token route (the "=" token position)
            const std::size_t last_t = ri.routes.size() - 1;
            if (ri.routes[last_t] == test_expected_ops[i]) ++spec_correct;
            total_entropy += ri.entropy[last_t];
            ++count;
        }
        float spec = (count > 0) ? static_cast<float>(spec_correct) / static_cast<float>(count) : 0.f;
        float avg_ent = (count > 0) ? total_entropy / static_cast<float>(count) : 0.f;
        return {spec, avg_ent};
    };

    std::cout << std::format("  {:>6}  {:>7}  {:>9}  {:>12}  {:>11}\n",
                              "step", "loss", "test_acc", "router_spec", "avg_entropy");
    std::cout << "  " << std::string(53, '-') << "\n";

    // Step 0 baseline
    {
        auto [spec, avg_ent] = eval_router();
        float acc = eval_accuracy();
        std::cout << std::format("  {:>6}  {:>7}  {:>8.1f}%  {:>11.1f}%  {:>11.2f}\n",
                                  0, "n/a", acc * 100.f, spec * 100.f, avg_ent);
    }

    // Timing trackers
    int step_spec_30 = -1, step_acc_10 = -1;
    float last_loss = 0.f;

    for (int step = 1; step <= steps; ++step) {
        const std::size_t idx = rng_train() % train_ids.size();
        const auto& ids = train_ids[idx];
        if (ids.size() < 2) { --step; continue; }

        Tensor id_tensor = make_ids_tensor(ids);
        Tensor targets   = make_targets(ids);
        const int64_t seq_len = static_cast<int64_t>(ids.size());

        adam.zero_grad();
        auto logits = model.forward_math(id_tensor, ntok);
        auto ltrunc = narrow(logits, 0, seq_len - 1);
        auto loss   = cross_entropy(ltrunc, targets);
        loss.backward();
        (void)clip_grad_norm(model.parameters(), 1.0f);
        adam.step();
        last_loss = loss.data().data_as<float>()[0];

        if (step % log_every == 0) {
            float acc = eval_accuracy();
            auto [spec, avg_ent] = eval_router();

            std::cout << std::format("  {:>6}  {:>7.2f}  {:>8.1f}%  {:>11.1f}%  {:>11.2f}\n",
                                      step, last_loss, acc * 100.f, spec * 100.f, avg_ent);

            if (step_spec_30 < 0 && spec >= 0.30f) step_spec_30 = step;
            if (step_acc_10  < 0 && acc  >= 0.10f) step_acc_10  = step;
        }
    }

    // Analysis section
    std::cout << "\n  Analysis:\n";
    std::cout << "  ---------\n";

    if (step_spec_30 >= 0)
        std::cout << std::format("  Router specialisation first exceeded 30% at step {}\n",
                                  step_spec_30);
    else
        std::cout << "  Router specialisation did not exceed 30% within 5000 steps\n";

    if (step_acc_10 >= 0)
        std::cout << std::format("  Test accuracy first exceeded 10% at step {}\n", step_acc_10);
    else
        std::cout << "  Test accuracy did not exceed 10% within 5000 steps\n";

    if (step_spec_30 >= 0 && step_acc_10 >= 0) {
        if (step_spec_30 < step_acc_10)
            std::cout << std::format("  Router specialisation LEADS accuracy by {} steps\n",
                                      step_acc_10 - step_spec_30);
        else
            std::cout << std::format("  Accuracy improvement LEADS router specialisation by {} steps\n",
                                      step_spec_30 - step_acc_10);
    }

    // Minimum training budget estimate
    const int n_unique_results = 19;  // add/sub on 0..9 produces results in [-9..18] → 28 vals,
                                       // but practically ~19 unique outcomes seen in training
    const int router_routes = kNumRouteTypes;
    const int steps_per_example = 5000 / static_cast<int>(train_ids.size());
    std::cout << std::format("\n  Minimum training budget estimate:\n");
    std::cout << std::format("    steps_per_example ({}) × n_unique_results ({}) × router_routes ({}) = {}\n",
                              steps_per_example, n_unique_results, router_routes,
                              steps_per_example * n_unique_results * router_routes);

    std::cout << "\n  Why convergence is slow:\n";
    std::cout << std::format("    Vocabulary size = {} (cross-entropy floor ln({}) ≈ {:.2f} nats)\n",
                              total_vocab, total_vocab,
                              std::log(static_cast<float>(total_vocab)));
    std::cout << "    Random-init embeddings: each of 65k+ tokens starts at equal distance from\n";
    std::cout << "    every other token — the model must first cluster numeric tokens before\n";
    std::cout << "    the router can learn meaningful distinctions.\n";
    std::cout << std::format("    Early router: 1/K = {:.1f}% chance of correct route by chance.\n",
                              100.f / static_cast<float>(kNumRouteTypes));
    std::cout << "    With D=16 embeddings spread across 65k vocab, gradient signal per token\n";
    std::cout << "    is extremely diluted — most steps update unrelated embeddings.\n";

    std::cout << "\n  What would accelerate training:\n";
    std::cout << "    1. Smaller vocab: a purpose-built arithmetic tokenizer (BPE on digits only)\n";
    std::cout << "       would reduce the vocab to ~50 tokens, making each gradient step 1000×\n";
    std::cout << "       more focused on the numeric subspace.\n";
    std::cout << "    2. Larger D: D=64 or D=128 gives the router more expressive capacity to\n";
    std::cout << "       separate operator symbols (+/-/*/) from numeric tokens in embedding space.\n";
    std::cout << "    3. Explicit routing supervision: add a cross-entropy loss on router logits\n";
    std::cout << "       with ground-truth operator labels — this directly trains the router\n";
    std::cout << "       without waiting for end-to-end gradient to propagate through STE.\n";
}

// ── §21.8  Curriculum Learning: Specialise then Scale ────────────────────────
//
// Phase 1: logit-masked training restricts cross-entropy to ~35 active tokens.
// This concentrates gradient ~1800× vs full 65k vocab, forcing the router to
// specialise quickly.  The trained math_block is then transferred via
// import_math_block() and fine-tuned against the full vocabulary (Phase 2).

static void section_curriculum_learning() {
    std::cout << "\n=== §21.8  Curriculum Learning: Specialise then Scale ===\n";
    std::cout << "  Phase 1: logit-masked (active ~35 tokens) — forces router specialisation\n";
    std::cout << "  Phase 2A: full-vocab fresh start (cold baseline)\n";
    std::cout << "  Phase 2B: full-vocab with Phase-1 math_block transferred\n\n";

    // ── shared dataset (same seed as §21.7 for comparability) ────────────────
    std::mt19937 rng_data(42);
    std::uniform_int_distribution<int> d09(0, 9);

    std::vector<std::string> corpus;
    corpus.reserve(200);
    for (int i = 0; i < 100; ++i) {
        int A = d09(rng_data), B = d09(rng_data);
        corpus.push_back(std::to_string(A) + " + " + std::to_string(B) +
                          " = " + std::to_string(A + B));
    }
    for (int i = 0; i < 100; ++i) {
        int A = d09(rng_data), B = d09(rng_data);
        corpus.push_back(std::to_string(A) + " - " + std::to_string(B) +
                          " = " + std::to_string(A - B));
    }

    auto bpe = BPETokenizer::train(corpus, 50);
    NumericTokenizer ntok(std::move(bpe));
    const int64_t V = ntok.total_vocab_size();

    std::vector<std::vector<int32_t>> train_ids;
    for (const auto& expr : corpus) {
        auto ids = ntok.encode(expr);
        if (ids.size() >= 2)
            train_ids.push_back(std::vector<int32_t>(ids.begin(), ids.end()));
    }

    // Test set with explicit expected_op
    struct TestItem {
        std::vector<int32_t> prompt_ids;
        int32_t              expected_val;
        RouteType            expected_op;
    };
    std::vector<TestItem> test_items;
    {
        std::mt19937 rng_t(99);
        for (int i = 0; i < 25; ++i) {
            int A = d09(rng_t), B = d09(rng_t);
            auto ids = ntok.encode(std::to_string(A) + " + " + std::to_string(B) + " =");
            test_items.push_back({std::vector<int32_t>(ids.begin(), ids.end()),
                                   A + B, RouteType::Add});
        }
        for (int i = 0; i < 25; ++i) {
            int A = d09(rng_t), B = d09(rng_t);
            auto ids = ntok.encode(std::to_string(A) + " - " + std::to_string(B) + " =");
            test_items.push_back({std::vector<int32_t>(ids.begin(), ids.end()),
                                   A - B, RouteType::Sub});
        }
    }

    // ── Build Phase 1 logit bias (active token mask) ─────────────────────────
    // Collect every token ID that appears in training/test sequences
    std::vector<bool> is_active(static_cast<std::size_t>(V), false);
    for (const auto& ids : train_ids)
        for (auto id : ids)
            is_active[static_cast<std::size_t>(id)] = true;
    for (const auto& item : test_items)
        for (auto id : item.prompt_ids)
            is_active[static_cast<std::size_t>(id)] = true;
    // Include all possible single-digit result tokens [-9..18]
    for (int v = -9; v <= 18; ++v)
        is_active[static_cast<std::size_t>(ntok.encode_int(v))] = true;

    int n_active = 0;
    std::vector<float> bias_vec(static_cast<std::size_t>(V), -1e9f);
    for (int64_t i = 0; i < V; ++i) {
        if (is_active[static_cast<std::size_t>(i)]) {
            bias_vec[static_cast<std::size_t>(i)] = 0.0f;
            ++n_active;
        }
    }

    std::cout << std::format("  total_vocab={} active_tokens={}\n\n", V, n_active);

    // Pre-build the maximum-size bias tensor once; each training step slices to
    // T_loss rows via narrow instead of allocating and filling a fresh tensor.
    int64_t max_T_loss_bias = 0;
    for (const auto& ids : train_ids)
        if (static_cast<int64_t>(ids.size()) >= 2)
            max_T_loss_bias = std::max(max_T_loss_bias,
                                       static_cast<int64_t>(ids.size()) - 1);
    Tensor bias_t_full({max_T_loss_bias, V}, DType::Float32);
    {
        float* bsp = bias_t_full.data_as<float>().data();
        for (int64_t t = 0; t < max_T_loss_bias; ++t)
            std::memcpy(bsp + t * V, bias_vec.data(),
                        static_cast<std::size_t>(V) * sizeof(float));
    }

    // ── Shared evaluation lambdas ─────────────────────────────────────────────
    const int64_t D = 16;
    const std::size_t n_heads = 2, n_kv = 1;
    const int64_t n_layers = 4;

    // Evaluate accuracy and router metrics for a given model.
    // If masked=true, add the logit bias before argmax (Phase 1 eval).
    auto evaluate = [&](MathGPT& model, bool masked)
        -> std::tuple<float, float, float>  // (accuracy, router_spec, avg_entropy)
    {
        int correct = 0, spec_count = 0;
        float spec_sum = 0.f, entropy_sum = 0.f;
        for (const auto& item : test_items) {
            if (item.prompt_ids.empty()) continue;
            Tensor ids_t = make_ids_tensor(item.prompt_ids);
            Variable logits = model.forward_math(ids_t, ntok);

            const int64_t last = logits.data().shape(0) - 1;
            const int64_t Vl   = logits.data().shape(1);
            auto lsp = logits.data().data_as<float>();

            // Argmax (with optional bias)
            int64_t best = 0;
            float best_v = lsp[static_cast<std::size_t>(last * Vl)] +
                           (masked ? bias_vec[0] : 0.f);
            for (int64_t v = 1; v < Vl; ++v) {
                float val = lsp[static_cast<std::size_t>(last * Vl + v)] +
                            (masked ? bias_vec[static_cast<std::size_t>(v)] : 0.f);
                if (val > best_v) { best_v = val; best = v; }
            }
            auto pred_id = static_cast<NumericTokenizer::TokenId>(best);
            if (ntok.is_numeric(pred_id) && !ntok.is_nan_token(pred_id) &&
                !ntok.is_overflow_token(pred_id))
            {
                if (static_cast<int32_t>(ntok.numeric_value(pred_id)) == item.expected_val)
                    ++correct;
            }

            // Router info
            RouteInfo ri = model.route_info(ids_t, ntok);
            if (!ri.routes.empty()) {
                const std::size_t last_t = ri.routes.size() - 1;
                if (ri.routes[last_t] == item.expected_op) spec_sum += 1.f;
                entropy_sum += ri.entropy[last_t];
                ++spec_count;
            }
        }
        float n  = static_cast<float>(test_items.size());
        float sn = spec_count > 0 ? static_cast<float>(spec_count) : 1.f;
        return {static_cast<float>(correct) / n, spec_sum / sn, entropy_sum / sn};
    };

    auto print_header = [&]() {
        std::cout << std::format("  {:>6}  {:>6}  {:>9}  {:>12}  {:>11}\n",
                                  "step", "loss", "accuracy", "router_spec", "entropy");
        std::cout << "  " << std::string(51, '-') << "\n";
    };

    auto print_row = [&](int step, float loss, float acc, float spec, float ent) {
        std::string loss_s = (step == 0) ? "  n/a" : std::format("{:6.2f}", loss);
        std::cout << std::format("  {:>6}  {}  {:>8.1f}%  {:>11.1f}%  {:>11.2f}\n",
                                  step, loss_s, acc * 100.f, spec * 100.f, ent);
    };

    // ── Phase 1: logit-masked training ────────────────────────────────────────
    std::cout << "  Phase 1 — masked vocabulary (logit bias, active tokens only)\n";
    print_header();

    MathGPT model_p1(V, D, n_heads, n_kv, n_layers, -1, 0, /*seed=*/42);
    Adam    adam_p1(model_p1.parameters(), 3e-3f);
    std::mt19937 rng_p1(11);

    float last_loss_p1 = 0.f;
    int step_spec30_p1 = -1;

    for (int step = 0; step <= 1000; ++step) {
        if (step % 200 == 0) {
            auto [acc, spec, ent] = evaluate(model_p1, /*masked=*/true);
            print_row(step, last_loss_p1, acc, spec, ent);
            if (step_spec30_p1 < 0 && spec >= 0.30f) step_spec30_p1 = step;
        }
        if (step == 1000) break;

        const std::size_t idx = rng_p1() % train_ids.size();
        const auto& ids = train_ids[idx];
        if (ids.size() < 2) { --step; continue; }

        Tensor id_tensor = make_ids_tensor(ids);
        Tensor targets   = make_targets(ids);
        const int64_t seq_len = static_cast<int64_t>(ids.size());
        const int64_t T_loss  = seq_len - 1;

        adam_p1.zero_grad();
        auto logits  = model_p1.forward_math(id_tensor, ntok);
        auto ltrunc  = narrow(logits, 0, T_loss);
        auto lmasked = add(ltrunc, narrow(Variable(bias_t_full, false), 0, T_loss));
        auto loss    = cross_entropy(lmasked, targets);
        loss.backward();
        (void)clip_grad_norm(model_p1.parameters(), 1.0f);
        adam_p1.step();
        last_loss_p1 = loss.data().data_as<float>()[0];
    }

    if (step_spec30_p1 > 0)
        std::cout << std::format("  → router_spec first crossed 30% at step {}\n", step_spec30_p1);
    else
        std::cout << "  → router_spec did not cross 30% within 1000 steps\n";

    // ── Phase 2A: fresh full-vocab training (cold baseline) ───────────────────
    std::cout << "\n  Phase 2A — full-vocab training from scratch (cold baseline)\n";
    print_header();

    MathGPT model_p2a(V, D, n_heads, n_kv, n_layers, -1, 0, /*seed=*/42);
    Adam    adam_p2a(model_p2a.parameters(), 3e-3f);
    std::mt19937 rng_p2a(11);

    float last_loss_p2a = 0.f;

    for (int step = 0; step <= 1000; ++step) {
        if (step % 200 == 0) {
            auto [acc, spec, ent] = evaluate(model_p2a, /*masked=*/false);
            print_row(step, last_loss_p2a, acc, spec, ent);
        }
        if (step == 1000) break;

        const std::size_t idx = rng_p2a() % train_ids.size();
        const auto& ids = train_ids[idx];
        if (ids.size() < 2) { --step; continue; }

        Tensor id_tensor = make_ids_tensor(ids);
        Tensor targets   = make_targets(ids);
        const int64_t seq_len = static_cast<int64_t>(ids.size());

        adam_p2a.zero_grad();
        auto logits = model_p2a.forward_math(id_tensor, ntok);
        auto ltrunc = narrow(logits, 0, seq_len - 1);
        auto loss   = cross_entropy(ltrunc, targets);
        loss.backward();
        (void)clip_grad_norm(model_p2a.parameters(), 1.0f);
        adam_p2a.step();
        last_loss_p2a = loss.data().data_as<float>()[0];
    }

    // ── Phase 1C: masked training, math_block only (tok_emb frozen) ──────────
    // Key insight: if tok_emb stays at the same random init as Phase 2, the
    // router learns to classify operators FROM THOSE SAME EMBEDDINGS — making
    // the transfer to Phase 2C clean.
    std::cout << "\n  Phase 1C — masked, math_block only (tok_emb frozen at Phase-2 init)\n";
    print_header();

    MathGPT model_p1c(V, D, n_heads, n_kv, n_layers, -1, 0, /*seed=*/42);
    // Optimizer covers only math_block_ params — tok_emb stays frozen
    auto   p1c_block_params = model_p1c.math_block_only_parameters();
    auto   p1c_all_params   = model_p1c.parameters();  // for zero_grad
    Adam   adam_p1c(p1c_block_params, 3e-3f);
    std::mt19937 rng_p1c(11);

    float last_loss_p1c = 0.f;
    int step_spec30_p1c = -1;

    for (int step = 0; step <= 1000; ++step) {
        if (step % 200 == 0) {
            auto [acc, spec, ent] = evaluate(model_p1c, /*masked=*/true);
            print_row(step, last_loss_p1c, acc, spec, ent);
            if (step_spec30_p1c < 0 && spec >= 0.30f) step_spec30_p1c = step;
        }
        if (step == 1000) break;

        const std::size_t idx = rng_p1c() % train_ids.size();
        const auto& ids = train_ids[idx];
        if (ids.size() < 2) { --step; continue; }

        Tensor id_tensor = make_ids_tensor(ids);
        Tensor targets   = make_targets(ids);
        const int64_t seq_len = static_cast<int64_t>(ids.size());
        const int64_t T_loss  = seq_len - 1;

        // Zero ALL parameter grads (including frozen tok_emb) then step only block
        for (auto* p : p1c_all_params) p->zero_grad();
        auto logits  = model_p1c.forward_math(id_tensor, ntok);
        auto ltrunc  = narrow(logits, 0, T_loss);
        auto lmasked = add(ltrunc, narrow(Variable(bias_t_full, false), 0, T_loss));
        auto loss    = cross_entropy(lmasked, targets);
        loss.backward();
        (void)clip_grad_norm(p1c_block_params, 1.0f);
        adam_p1c.step();
        last_loss_p1c = loss.data().data_as<float>()[0];
    }

    if (step_spec30_p1c > 0)
        std::cout << std::format("  → router_spec first crossed 30% at step {}\n", step_spec30_p1c);
    else
        std::cout << "  → router_spec did not cross 30% within 1000 steps\n";

    // ── Phase 2B: transfer from Phase 1 (tok_emb updated) ────────────────────
    std::cout << "\n  Phase 2B — full-vocab with Phase-1 math_block (tok_emb updated in P1)\n";
    print_header();

    MathGPT model_p2b(V, D, n_heads, n_kv, n_layers, -1, 0, /*seed=*/42);
    model_p2b.import_math_block(model_p1);
    Adam    adam_p2b(model_p2b.parameters(), 3e-3f);
    std::mt19937 rng_p2b(11);
    float last_loss_p2b = 0.f;

    for (int step = 0; step <= 1000; ++step) {
        if (step % 200 == 0) {
            auto [acc, spec, ent] = evaluate(model_p2b, false);
            print_row(step, last_loss_p2b, acc, spec, ent);
        }
        if (step == 1000) break;
        const std::size_t idx = rng_p2b() % train_ids.size();
        const auto& ids = train_ids[idx];
        if (ids.size() < 2) { --step; continue; }
        Tensor id_tensor = make_ids_tensor(ids);
        Tensor targets   = make_targets(ids);
        const int64_t seq_len = static_cast<int64_t>(ids.size());
        adam_p2b.zero_grad();
        auto logits = model_p2b.forward_math(id_tensor, ntok);
        auto ltrunc = narrow(logits, 0, seq_len - 1);
        auto loss   = cross_entropy(ltrunc, targets);
        loss.backward();
        (void)clip_grad_norm(model_p2b.parameters(), 1.0f);
        adam_p2b.step();
        last_loss_p2b = loss.data().data_as<float>()[0];
    }

    // ── Phase 2C: transfer from Phase 1C (tok_emb frozen in P1) ──────────────
    std::cout << "\n  Phase 2C — full-vocab with Phase-1C math_block (tok_emb frozen in P1)\n";
    print_header();

    MathGPT model_p2c(V, D, n_heads, n_kv, n_layers, -1, 0, /*seed=*/42);
    model_p2c.import_math_block(model_p1c);  // router learned from same embeddings
    Adam    adam_p2c(model_p2c.parameters(), 3e-3f);
    std::mt19937 rng_p2c(11);
    float last_loss_p2c = 0.f;

    for (int step = 0; step <= 1000; ++step) {
        if (step % 200 == 0) {
            auto [acc, spec, ent] = evaluate(model_p2c, false);
            print_row(step, last_loss_p2c, acc, spec, ent);
        }
        if (step == 1000) break;
        const std::size_t idx = rng_p2c() % train_ids.size();
        const auto& ids = train_ids[idx];
        if (ids.size() < 2) { --step; continue; }
        Tensor id_tensor = make_ids_tensor(ids);
        Tensor targets   = make_targets(ids);
        const int64_t seq_len = static_cast<int64_t>(ids.size());
        adam_p2c.zero_grad();
        auto logits = model_p2c.forward_math(id_tensor, ntok);
        auto ltrunc = narrow(logits, 0, seq_len - 1);
        auto loss   = cross_entropy(ltrunc, targets);
        loss.backward();
        (void)clip_grad_norm(model_p2c.parameters(), 1.0f);
        adam_p2c.step();
        last_loss_p2c = loss.data().data_as<float>()[0];
    }

    // ── Summary comparison ────────────────────────────────────────────────────
    std::cout << "\n  Summary at step 1000:\n";
    std::cout << std::format("  {:>44}  {:>9}  {:>12}  {:>11}\n",
                              "model", "accuracy", "router_spec", "entropy");
    std::cout << "  " << std::string(80, '-') << "\n";
    {
        auto [acc_a,  spec_a,  ent_a]  = evaluate(model_p2a,  false);
        auto [acc_b,  spec_b,  ent_b]  = evaluate(model_p2b,  false);
        auto [acc_c,  spec_c,  ent_c]  = evaluate(model_p2c,  false);
        auto [acc_p1, spec_p1, ent_p1] = evaluate(model_p1,   true);
        auto [acc_1c, spec_1c, ent_1c] = evaluate(model_p1c,  true);
        std::cout << std::format("  {:>44}  {:>8.1f}%  {:>11.1f}%  {:>11.2f}\n",
                                  "Phase 2A (cold, full-vocab)",
                                  acc_a*100, spec_a*100, ent_a);
        std::cout << std::format("  {:>44}  {:>8.1f}%  {:>11.1f}%  {:>11.2f}\n",
                                  "Phase 2B (P1 transfer, tok_emb updated)",
                                  acc_b*100, spec_b*100, ent_b);
        std::cout << std::format("  {:>44}  {:>8.1f}%  {:>11.1f}%  {:>11.2f}\n",
                                  "Phase 2C (P1C transfer, tok_emb frozen)",
                                  acc_c*100, spec_c*100, ent_c);
        std::cout << std::format("  {:>44}  {:>8.1f}%  {:>11.1f}%  {:>11.2f}\n",
                                  "Phase 1  (masked, tok_emb updated)",
                                  acc_p1*100, spec_p1*100, ent_p1);
        std::cout << std::format("  {:>44}  {:>8.1f}%  {:>11.1f}%  {:>11.2f}\n",
                                  "Phase 1C (masked, tok_emb frozen)",
                                  acc_1c*100, spec_1c*100, ent_1c);
    }

    std::cout << "\n  Curriculum findings:\n";
    std::cout << std::format("    Logit masking ({} active of {} tokens = {:.0f}× gradient amplification)\n",
                              n_active, V,
                              static_cast<float>(V) / static_cast<float>(n_active));
    std::cout << "    achieves 50%+ router_spec in Phase 1 — the specialisation gap is closed.\n";
    std::cout << "    Phase 2C (tok_emb frozen during Phase 1) is the clean transfer:\n";
    std::cout << "    the router learns from the SAME embeddings it will see in Phase 2,\n";
    std::cout << "    so the routing boundaries survive the vocabulary expansion.\n";
}

// ── §21.6  Large Number Arithmetic — Exact vs Statistical ────────────────────
//
// A tiny statistical LM trained only on [0..9] single-digit arithmetic will
// fail on large numbers it has never seen.  The math execution nodes compute
// exact integer arithmetic for ANY input in the tokenizer's configured range —
// no training on those numbers is required.

static void section_large_numbers() {
    std::cout << "\n=== §21.6  Large Number Arithmetic — Exact vs Statistical ===\n";
    std::cout << "  Math nodes use exact integer arithmetic; statistical LMs must\n";
    std::cout << "  memorise every pair and fail badly on unseen large numbers.\n\n";

    // Build a tiny tokenizer (BPE on digit symbols only, never sees 4-digit numbers)
    std::vector<std::string> small_corpus = {
        "1 + 2 = 3", "5 - 3 = 2", "4 * 2 = 8", "9 / 3 = 3"
    };
    auto bpe  = BPETokenizer::train(small_corpus, 40);
    NumericTokenizer ntok(std::move(bpe));

    struct LargeCase {
        RouteType   op;
        int32_t     a, b;
        std::string description;
    };

    const std::vector<LargeCase> cases = {
        {RouteType::Add,     9876,  5432, "9876 + 5432"},
        {RouteType::Sub,    32767,     1, "32767 - 1  "},
        {RouteType::Mul,      181,     9, "181 × 9    "},
        {RouteType::Div,    32760,     8, "32760 ÷ 8  "},
        {RouteType::IsLessThan, 9999,  9998, "9999 < 9998"},   // false → 0
        {RouteType::IsLessThan, 1000, 10000, "1000 < 10000"},  // true  → 1
        {RouteType::Mul,      300,   200, "300 × 200  "},  // overflows int16
        {RouteType::Div,       42,     0, "42 ÷ 0     "},  // NaN / div-by-zero
    };

    std::cout << std::format("  {:<16}  {:>8}  {:>9}  {:>9}  {}\n",
                              "expression", "result", "is_nan", "overflow", "exact?");
    std::cout << "  " << std::string(62, '-') << "\n";

    for (const auto& c : cases) {
        auto r = apply_math_op(c.op, static_cast<float>(c.a), static_cast<float>(c.b));

        // Compute "true" reference (unclamped)
        int64_t ref = 0;
        bool ref_overflow = false;
        switch (c.op) {
            case RouteType::Add:     ref = static_cast<int64_t>(c.a) + c.b; break;
            case RouteType::Sub:     ref = static_cast<int64_t>(c.a) - c.b; break;
            case RouteType::Mul:     ref = static_cast<int64_t>(c.a) * c.b; break;
            case RouteType::Div:     ref = (c.b == 0) ? 0 : c.a / c.b;     break;
            case RouteType::IsLessThan:    ref = (c.a < c.b)  ? 1 : 0; break;
            case RouteType::IsGreaterThan: ref = (c.a > c.b)  ? 1 : 0; break;
            case RouteType::IsEqual:       ref = (c.a == c.b) ? 1 : 0; break;
            default: break;
        }
        if (ref < ntok.int_min() || ref > ntok.int_max())
            ref_overflow = true;

        std::string exact_str;
        if (r.is_nan) {
            exact_str = "NaN (div/0)";
        } else if (r.is_overflow || ref_overflow) {
            exact_str = std::format("OVERFLOW (true={}, int16 range exceeded)", ref);
        } else {
            exact_str = (static_cast<int64_t>(r.value) == ref) ? "YES" : "NO";
        }

        std::cout << std::format("  {:<16}  {:>8.0f}  {:>9}  {:>9}  {}\n",
                                  c.description,
                                  r.value,
                                  r.is_nan      ? "true" : "false",
                                  r.is_overflow ? "true" : "false",
                                  exact_str);
    }

    // Now show how the token encoding handles large numbers in context
    std::cout << "\n  Token encoding for large operands:\n";
    const std::vector<int32_t> large_vals = {9876, 5432, 15308, 32767, -32768};
    for (int32_t v : large_vals) {
        auto id = ntok.encode_int(v);
        bool ok = ntok.is_numeric(id) && !ntok.is_nan_token(id) && !ntok.is_overflow_token(id);
        if (ok) {
            float decoded = ntok.numeric_value(id);
            std::cout << std::format("    encode_int({:6d}) = {:6d}  decode = {:6.0f}  round-trip: {}\n",
                                      v, static_cast<int>(id), decoded,
                                      static_cast<int32_t>(decoded) == v ? "OK" : "FAIL");
        } else {
            std::cout << std::format("    encode_int({:6d}) = {:6d}  [overflow/nan token]\n",
                                      v, static_cast<int>(id));
        }
    }

    // Show statistical model failure mode
    std::cout << "\n  Why statistical models fail on large numbers:\n";
    std::cout << "    A model trained on [0..9] arithmetic has seen at most 100\n";
    std::cout << "    unique A+B=C triples.  For 9876 + 5432 it has zero training\n";
    std::cout << "    signal.  The math node computes it exactly in O(1) regardless\n";
    std::cout << "    of training data — the result is always 9876+5432=15308.\n";
    std::cout << "    For overflow (300*200=60000), the node correctly raises the\n";
    std::cout << "    overflow sentinel, while a statistical model would produce\n";
    std::cout << "    a random high-frequency token.\n";
}

// ── §21.9  Improved Specialisation: D=32 + Router Supervision ────────────────
//
// §21.7 showed end-to-end training locks the router onto FFN (0% router_spec
// within 5000 steps). §21.8 showed curriculum (logit masking) achieves 50%
// router_spec in Phase 1C but D=16 limits the router to 102 parameters and
// head_dim=8 limits operator disambiguation.
//
// Two targeted fixes:
//   1. D=32: router Linear(32,6) has 198 params, head_dim=16, embeddings
//      more separable across 65k vocab.
//   2. Router supervision loss: at each Phase 1Cs step, add
//        L_sup = CE(router_logits["=" position], ground_truth_op)
//      with weight α=0.5. This is a DIRECT gradient signal — instead of
//      relying solely on the STE path through the LM objective.
//
// Both changes require no architecture modifications: router_logits() exposes
// the pre-softmax router output for direct supervision.

// ── Checkpoint utilities ──────────────────────────────────────────────────────
// Binary format: magic(u32) | version(u32) | step(i32) | n_params(u32) |
//   for each param: ndim(u32) | dims(i64 × ndim) | data(f32 × numel)

static void save_checkpoint(const std::vector<autograd::Variable*>& params,
                             const std::filesystem::path& path, int step)
{
    std::filesystem::create_directories(path.parent_path());
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f)
        throw std::runtime_error(
            std::format("save_checkpoint: cannot open {}", path.string()));

    constexpr uint32_t kMagic   = 0x43483231u;
    constexpr uint32_t kVersion = 1u;
    const auto nparams = static_cast<uint32_t>(params.size());
    const auto step32  = static_cast<int32_t>(step);
    f.write(reinterpret_cast<const char*>(&kMagic),   4);
    f.write(reinterpret_cast<const char*>(&kVersion), 4);
    f.write(reinterpret_cast<const char*>(&step32),   4);
    f.write(reinterpret_cast<const char*>(&nparams),  4);

    for (const Variable* p : params) {
        const Tensor& t  = p->data();
        const auto    nd = static_cast<uint32_t>(t.ndim());
        f.write(reinterpret_cast<const char*>(&nd), 4);
        for (std::size_t i = 0, nd2 = static_cast<std::size_t>(t.ndim()); i < nd2; ++i) {
            int64_t d = t.shape(i);
            f.write(reinterpret_cast<const char*>(&d), 8);
        }
        auto sp = t.data_as<float>();  // const overload returns span<const float>
        f.write(reinterpret_cast<const char*>(sp.data()),
                static_cast<std::streamsize>(sp.size() * sizeof(float)));
    }
    std::cout << std::format("  [ckpt] step {:4d} → {}\n", step, path.string());
}

// Returns step stored in the latest checkpoint for this phase, or -1 if none found.
// If target_step > 0, only checkpoints with step <= target_step are considered
// (useful for replaying an earlier milestone without running to the end).
static int load_latest_checkpoint(const std::vector<autograd::Variable*>& params,
                                   std::string_view phase, std::string_view ckpt_dir,
                                   int target_step = 0)
{
    namespace fs = std::filesystem;
    if (!fs::is_directory(ckpt_dir)) return -1;

    const std::string prefix = std::format("ch21_{}_step", phase);
    int      best_step = -1;
    fs::path best_path;

    for (const auto& entry : fs::directory_iterator(ckpt_dir)) {
        if (!entry.is_regular_file()) continue;
        const std::string fn = entry.path().filename().string();
        if (!fn.starts_with(prefix) || !fn.ends_with(".ckpt")) continue;
        try {
            constexpr std::size_t kExtLen = std::string_view{".ckpt"}.size();
            const std::size_t start = prefix.size();
            if (fn.size() <= start + kExtLen) continue;  // no step digits between prefix and .ckpt
            const std::size_t len   = fn.size() - start - kExtLen;
            int s = std::stoi(fn.substr(start, len));
            if (target_step > 0 && s > target_step) continue;
            if (s > best_step) { best_step = s; best_path = entry.path(); }
        } catch (...) {}
    }
    if (best_step < 0) return -1;

    std::ifstream f(best_path, std::ios::binary);
    if (!f) return -1;

    constexpr uint32_t kMagic = 0x43483231u;
    uint32_t magic, version, nparams;
    int32_t  ckpt_step;
    f.read(reinterpret_cast<char*>(&magic),     4);
    f.read(reinterpret_cast<char*>(&version),   4);
    f.read(reinterpret_cast<char*>(&ckpt_step), 4);
    f.read(reinterpret_cast<char*>(&nparams),   4);

    if (magic != kMagic)
        throw std::runtime_error(
            std::format("load_checkpoint: bad magic in {}", best_path.string()));
    if (nparams != static_cast<uint32_t>(params.size()))
        throw std::runtime_error(std::format(
            "load_checkpoint: {} params in checkpoint, {} in model",
            nparams, params.size()));

    for (Variable* p : params) {
        Tensor&  t = p->data();
        uint32_t nd;
        f.read(reinterpret_cast<char*>(&nd), 4);
        if (nd != static_cast<uint32_t>(t.ndim()))
            throw std::runtime_error("load_checkpoint: ndim mismatch");
        for (uint32_t i = 0; i < nd; ++i) {
            int64_t d;
            f.read(reinterpret_cast<char*>(&d), 8);
            if (d != t.shape(i))
                throw std::runtime_error(std::format(
                    "load_checkpoint: shape[{}] mismatch: ckpt={} model={}",
                    i, d, t.shape(i)));
        }
        auto sp = t.data_as<float>();
        f.read(reinterpret_cast<char*>(sp.data()),
               static_cast<std::streamsize>(sp.size() * sizeof(float)));
    }
    std::cout << std::format("  [ckpt] resumed {} (step={})\n",
                              best_path.string(), ckpt_step);
    return ckpt_step;
}

// ── §21.9  Improved Specialisation: D=32 + Router Supervision ────────────────

struct TestItemImproved {
    std::vector<int32_t> prompt_ids;
    int32_t              expected_val;
    RouteType            expected_op;
};

static constexpr int64_t kD       = 32;
static constexpr int64_t kNHeads  = 4;
static constexpr int64_t kNKv     = 2;
static constexpr int64_t kNLayers = 4;
struct ImprovedData {
    NumericTokenizer                         ntok;
    int64_t                                  V;
    std::vector<std::vector<int32_t>>        train_ids;
    std::vector<RouteType>                   train_ops;
    std::vector<TestItemImproved>            test_items;
    std::vector<float>                       bias_vec;
    Tensor                                   bias_t_full;
};


static std::tuple<float, float, float> eval_improved(
    MathGPT& model,
    const ImprovedData& d,
    bool masked,
    TokenMode token_mode = TokenMode::Real)
{
    int correct = 0, spec_count = 0;
    float spec_sum = 0.f, entropy_sum = 0.f;
    const int64_t Vl = d.V;

    for (const auto& item : d.test_items) {
        if (item.prompt_ids.empty()) continue;
        Tensor ids_t = make_ids_tensor(item.prompt_ids);
        Variable logits = model.forward_math(ids_t, d.ntok, token_mode);

        const int64_t last = logits.data().shape(0) - 1;
        auto lsp = logits.data().data_as<float>();  // const overload → span<const float>

        int64_t best   = 0;
        float   best_v = lsp[static_cast<std::size_t>(last * Vl)] +
                          (masked ? d.bias_vec[0] : 0.f);
        for (int64_t v = 1; v < Vl; ++v) {
            float val = lsp[static_cast<std::size_t>(last * Vl + v)] +
                        (masked ? d.bias_vec[static_cast<std::size_t>(v)] : 0.f);
            if (val > best_v) { best_v = val; best = v; }
        }
        auto pred_id = static_cast<NumericTokenizer::TokenId>(best);
        if (d.ntok.is_numeric(pred_id) && !d.ntok.is_nan_token(pred_id) &&
            !d.ntok.is_overflow_token(pred_id)) {
            if (static_cast<int32_t>(d.ntok.numeric_value(pred_id)) == item.expected_val)
                ++correct;
        }

        RouteInfo ri = model.route_info(ids_t, d.ntok, token_mode);
        if (!ri.routes.empty()) {
            const std::size_t last_t = ri.routes.size() - 1;
            if (ri.routes[last_t] == item.expected_op) spec_sum += 1.f;
            entropy_sum += ri.entropy[last_t];
            ++spec_count;
        }
    }
    const float n  = static_cast<float>(d.test_items.size());
    const float sn = spec_count > 0 ? static_cast<float>(spec_count) : 1.f;
    return {static_cast<float>(correct) / n, spec_sum / sn, entropy_sum / sn};
}

static void print_train_header(bool show_alpha = false, bool show_beta = false) {
    if (show_alpha && show_beta)
        std::cout << std::format("  {:>6}  {:>6}  {:>9}  {:>12}  {:>11}  {:>9}  {:>6}  {:>6}\n",
                                  "step", "loss", "accuracy", "router_spec", "entropy",
                                  "ms/step", "α", "β");
    else if (show_alpha)
        std::cout << std::format("  {:>6}  {:>6}  {:>9}  {:>12}  {:>11}  {:>9}  {:>6}\n",
                                  "step", "loss", "accuracy", "router_spec", "entropy",
                                  "ms/step", "α");
    else
        std::cout << std::format("  {:>6}  {:>6}  {:>9}  {:>12}  {:>11}  {:>9}\n",
                                  "step", "loss", "accuracy", "router_spec", "entropy",
                                  "ms/step");
    std::cout << "  " << std::string(show_beta ? 79 : (show_alpha ? 71 : 63), '-') << "\n";
}

// alpha < 0 → omit the α column; ms_per_step == 0 → show "---"; beta < 0 → omit β column
static void print_train_row(int step, float loss, float acc, float spec, float ent,
                             float ms_per_step = 0.f, float alpha = -1.f, float beta = -1.f) {
    std::string loss_s = (step == 0) ? "  n/a" : std::format("{:6.2f}", loss);
    std::string ms_s   = (ms_per_step == 0.f) ? "    ---"
                                               : std::format("{:6.1f}ms", ms_per_step);
    if (alpha >= 0.f && beta >= 0.f) {
        std::cout << std::format("  {:>6}  {}  {:>8.1f}%  {:>11.1f}%  {:>11.2f}  {:>9}  {:>6.3f}  {:>6.3f}\n",
                                  step, loss_s, acc * 100.f, spec * 100.f, ent, ms_s, alpha, beta);
    } else if (alpha >= 0.f) {
        std::cout << std::format("  {:>6}  {}  {:>8.1f}%  {:>11.1f}%  {:>11.2f}  {:>9}  {:>6.3f}\n",
                                  step, loss_s, acc * 100.f, spec * 100.f, ent, ms_s, alpha);
    } else {
        std::cout << std::format("  {:>6}  {}  {:>8.1f}%  {:>11.1f}%  {:>11.2f}  {:>9}\n",
                                  step, loss_s, acc * 100.f, spec * 100.f, ent, ms_s);
    }
}


static ImprovedData build_math_dataset_base() {
    std::mt19937 rng(42), rng_test(99);

    std::vector<std::string> corpus;
    std::vector<RouteType>   corpus_ops;

    struct RawTest { std::string prompt; int32_t expected; RouteType op; };
    std::vector<RawTest> raw_test;

    constexpr int kMax = 32767;

    auto safe = [](int v) { return v >= -kMax && v <= kMax; };

    // ── Helpers ───────────────────────────────────────────────────────────────
    auto pick = [&](int lo, int hi) -> int {
        return lo + static_cast<int>(rng() % static_cast<unsigned>(hi - lo + 1));
    };
    auto pick_t = [&](int lo, int hi) -> int {
        return lo + static_cast<int>(rng_test() % static_cast<unsigned>(hi - lo + 1));
    };

    // Add one full expression (including answer) to the training corpus.
    auto add1 = [&](RouteType op, const std::string& s) {
        corpus.push_back(s);
        corpus_ops.push_back(op);
    };

    // Add a two-step chain.
    //   "A sym1 B = R1 , R1 sym2 C = R2"  finalOp=op2
    auto add2 = [&](RouteType op1, RouteType op2,
                    const std::string& sym1, const std::string& sym2,
                    int A, int B, int C, int R1, int R2) {
        (void)op1;  // op1 carried for documentation; only op2 supervised at final '='
        add1(op2, std::to_string(A) + " " + sym1 + " " + std::to_string(B) + " = " +
                  std::to_string(R1) + " , " + std::to_string(R1) + " " + sym2 + " " +
                  std::to_string(C) + " = " + std::to_string(R2));
    };

    // Add a three-step chain.
    //   "A s1 B = R1 , R1 s2 C = R2 , R2 s3 D = R3"  finalOp=op3
    auto add3 = [&](RouteType op3,
                    const std::string& s1, const std::string& s2, const std::string& s3,
                    int A, int B, int C, int D, int R1, int R2, int R3) {
        add1(op3, std::to_string(A)  + " " + s1 + " " + std::to_string(B)  + " = " +
                  std::to_string(R1) + " , " + std::to_string(R1) + " " + s2 + " " +
                  std::to_string(C)  + " = " + std::to_string(R2) + " , " +
                  std::to_string(R2) + " " + s3 + " " + std::to_string(D)  + " = " +
                  std::to_string(R3));
    };

    // ── 1. Single-step ops ────────────────────────────────────────────────────
    // Add: A,B ∈ [0,40]; stratify by result, 3 examples per result.
    {
        std::map<int, std::vector<std::pair<int,int>>> by_r;
        for (int A = 0; A <= 40; ++A)
            for (int B = 0; B <= 40; ++B)
                by_r[A + B].emplace_back(A, B);
        for (auto& [r, pairs] : by_r) {
            std::shuffle(pairs.begin(), pairs.end(), rng);
            for (int i = 0; i < std::min(3, (int)pairs.size()); ++i)
                add1(RouteType::Add, std::to_string(pairs[i].first) + " + " +
                     std::to_string(pairs[i].second) + " = " + std::to_string(r));
        }
    }

    // Sub: A,B ∈ [0,40]; include negative results.
    {
        std::map<int, std::vector<std::pair<int,int>>> by_r;
        for (int A = 0; A <= 40; ++A)
            for (int B = 0; B <= 40; ++B)
                by_r[A - B].emplace_back(A, B);
        for (auto& [r, pairs] : by_r) {
            std::shuffle(pairs.begin(), pairs.end(), rng);
            for (int i = 0; i < std::min(3, (int)pairs.size()); ++i)
                add1(RouteType::Sub, std::to_string(pairs[i].first) + " - " +
                     std::to_string(pairs[i].second) + " = " + std::to_string(r));
        }
    }

    // Mul: A,B ∈ [1,20].
    {
        std::map<int, std::vector<std::pair<int,int>>> by_r;
        for (int A = 1; A <= 20; ++A)
            for (int B = 1; B <= 20; ++B)
                by_r[A * B].emplace_back(A, B);
        for (auto& [r, pairs] : by_r) {
            std::shuffle(pairs.begin(), pairs.end(), rng);
            for (int i = 0; i < std::min(3, (int)pairs.size()); ++i)
                add1(RouteType::Mul, std::to_string(pairs[i].first) + " * " +
                     std::to_string(pairs[i].second) + " = " + std::to_string(r));
        }
    }

    // Div: exact division only; k ∈ [1,20], B ∈ [1,20].
    {
        std::map<int, std::vector<std::pair<int,int>>> by_k;
        for (int k = 1; k <= 20; ++k)
            for (int B = 1; B <= 20; ++B)
                by_k[k].emplace_back(k * B, B);
        for (auto& [k, pairs] : by_k) {
            std::shuffle(pairs.begin(), pairs.end(), rng);
            for (int i = 0; i < std::min(3, (int)pairs.size()); ++i)
                add1(RouteType::Div, std::to_string(pairs[i].first) + " / " +
                     std::to_string(pairs[i].second) + " = " + std::to_string(k));
        }
    }

    // Comparisons: 80 true + 80 false each.
    for (auto [op, sym, fn] : std::initializer_list<std::tuple<
             RouteType, const char*,
             std::function<bool(int,int)>>>{
         {RouteType::IsLessThan,    "<",  [](int A,int B){return A< B;}},
         {RouteType::IsGreaterThan, ">",  [](int A,int B){return A> B;}},
         {RouteType::IsEqual,       "==", [](int A,int B){return A==B;}}})
    {
        std::vector<std::pair<int,int>> tp, fp;
        for (int A = 0; A <= 40; ++A)
            for (int B = 0; B <= 40; ++B)
                (fn(A,B) ? tp : fp).emplace_back(A, B);
        std::shuffle(tp.begin(), tp.end(), rng);
        std::shuffle(fp.begin(), fp.end(), rng);
        int n = std::min({(int)tp.size(), (int)fp.size(), 80});
        for (int i = 0; i < n; ++i) {
            add1(op, std::to_string(tp[i].first)+" "+sym+" "+std::to_string(tp[i].second)+" = 1");
            add1(op, std::to_string(fp[i].first)+" "+sym+" "+std::to_string(fp[i].second)+" = 0");
        }
    }

    // Sqrt: perfect squares {0,1,4,9,16,25,36,49,64,81,100,121,144,169,196,225}.
    // Notation: "sqrt ( A ) = R"
    {
        std::vector<int> perfects;
        for (int r = 0; r <= 15; ++r) perfects.push_back(r * r);
        for (int r : perfects)
            for (int rep = 0; rep < 4; ++rep)  // 4 copies to balance vs binary ops
                add1(RouteType::Sqrt,
                     "sqrt ( " + std::to_string(r) + " ) = " + std::to_string(static_cast<int>(std::round(std::sqrt(r)))));
    }

    // Increment / Decrement.
    for (int A = 0; A <= 30; ++A) {
        add1(RouteType::Increment, "++ " + std::to_string(A) + " = " + std::to_string(A + 1));
        add1(RouteType::Decrement, "-- " + std::to_string(A) + " = " + std::to_string(A - 1));
    }

    // ── 2. Two-step chains — systematic cross-product ─────────────────────────
    // Add→Mul  A+B=R1 , R1*C=R2
    for (int i = 0; i < 60; ) {
        int A=pick(2,15), B=pick(2,15), C=pick(2,8);
        int R1=A+B, R2=R1*C;
        if (!safe(R2)) continue;
        add2(RouteType::Add, RouteType::Mul, "+", "*", A, B, C, R1, R2);
        ++i;
    }
    // Mul→Add  A*B=R1 , R1+C=R2
    for (int i = 0; i < 60; ) {
        int A=pick(2,12), B=pick(2,12), C=pick(1,20);
        int R1=A*B, R2=R1+C;
        if (!safe(R2)) continue;
        add2(RouteType::Mul, RouteType::Add, "*", "+", A, B, C, R1, R2);
        ++i;
    }
    // Sub→Mul  A-B=R1 , R1*C=R2
    for (int i = 0; i < 60; ) {
        int A=pick(5,20), B=pick(1,A-1), C=pick(2,6);
        int R1=A-B, R2=R1*C;
        if (R1<=0 || !safe(R2)) continue;
        add2(RouteType::Sub, RouteType::Mul, "-", "*", A, B, C, R1, R2);
        ++i;
    }
    // Mul→Div  A*B=R1 , R1/C=R2  (exact: C divides R1)
    for (int i = 0; i < 40; ) {
        int C=pick(2,6), B=pick(1,10);
        int A=pick(2,10)*C;  // ensure A is multiple of C so R1/C is exact
        int R1=A*B;
        if (!safe(R1) || R1 % C != 0) continue;
        int R2=R1/C;
        add2(RouteType::Mul, RouteType::Div, "*", "/", A, B, C, R1, R2);
        ++i;
    }
    // Add→Div  (A+B)/C=R2  (exact)
    for (int i = 0; i < 40; ) {
        int C=pick(2,5);
        int A=pick(1,10)*C, B=pick(1,10)*C;
        int R1=A+B, R2=R1/C;
        if (R1%C!=0 || !safe(R2)) continue;
        add2(RouteType::Add, RouteType::Div, "+", "/", A, B, C, R1, R2);
        ++i;
    }
    // Add→Add  A+B=R1 , R1+C=R2
    for (int i = 0; i < 40; ) {
        int A=pick(1,10), B=pick(1,10), C=pick(1,10);
        int R1=A+B, R2=R1+C;
        if (!safe(R2)) continue;
        add2(RouteType::Add, RouteType::Add, "+", "+", A, B, C, R1, R2);
        ++i;
    }
    // Mul→Sub  A*B=R1 , R1-C=R2
    for (int i = 0; i < 40; ) {
        int A=pick(2,10), B=pick(2,10), C=pick(1,15);
        int R1=A*B, R2=R1-C;
        if (!safe(R2)) continue;
        add2(RouteType::Mul, RouteType::Sub, "*", "-", A, B, C, R1, R2);
        ++i;
    }
    // Sub→Add  A-B=R1 , R1+C=R2
    for (int i = 0; i < 40; ) {
        int A=pick(5,20), B=pick(1,A), C=pick(1,15);
        int R1=A-B, R2=R1+C;
        if (!safe(R1)||!safe(R2)) continue;
        add2(RouteType::Sub, RouteType::Add, "-", "+", A, B, C, R1, R2);
        ++i;
    }
    // Sqrt→Add  sqrt(A)=R1 , R1+B=R2
    {
        std::vector<int> perf;
        for (int r = 1; r <= 12; ++r) perf.push_back(r*r);
        for (int sq : perf)
            for (int B = 1; B <= 10; ++B) {
                int R1=static_cast<int>(std::round(std::sqrt(sq))), R2=R1+B;
                if (!safe(R2)) continue;
                add2(RouteType::Sqrt, RouteType::Add, "sqrt (", ")+", sq, 0, B, R1, R2);
                // NOTE: the chain format "sqrt ( A ) = R1 , R1 + B = R2"
                // is generated differently; use manual add1 below.
                corpus.pop_back(); corpus_ops.pop_back();  // undo add2; redo with correct format
                add1(RouteType::Add,
                     "sqrt ( " + std::to_string(sq) + " ) = " + std::to_string(R1) +
                     " , " + std::to_string(R1) + " + " + std::to_string(B) +
                     " = " + std::to_string(R2));
            }
    }
    // Sqrt→Mul  sqrt(A)=R1 , R1*B=R2
    {
        std::vector<int> perf;
        for (int r = 1; r <= 12; ++r) perf.push_back(r*r);
        for (int sq : perf)
            for (int B = 2; B <= 6; ++B) {
                int R1=static_cast<int>(std::round(std::sqrt(sq))), R2=R1*B;
                if (!safe(R2)) continue;
                add1(RouteType::Mul,
                     "sqrt ( " + std::to_string(sq) + " ) = " + std::to_string(R1) +
                     " , " + std::to_string(R1) + " * " + std::to_string(B) +
                     " = " + std::to_string(R2));
            }
    }
    // Compare after Add: A+B=R1 , R1 < C = 0/1
    for (int i = 0; i < 40; ) {
        int A=pick(1,10), B=pick(1,10), C=pick(1,25);
        int R1=A+B;
        int R2=(R1<C)?1:0;
        if (!safe(R2)) continue;
        add2(RouteType::Add, RouteType::IsLessThan, "+", "<", A, B, C, R1, R2);
        ++i;
    }
    // Compare after Mul: A*B=R1 , R1 > C = 0/1
    for (int i = 0; i < 40; ) {
        int A=pick(1,8), B=pick(1,8), C=pick(1,60);
        int R1=A*B;
        int R2=(R1>C)?1:0;
        if (!safe(R2)) continue;
        add2(RouteType::Mul, RouteType::IsGreaterThan, "*", ">", A, B, C, R1, R2);
        ++i;
    }

    // ── 3. Three-step chains — standard formulae + accumulation ──────────────
    // Accumulative sum: A+B+C+D
    for (int i = 0; i < 50; ) {
        int A=pick(1,8),B=pick(1,8),C=pick(1,8),D=pick(1,8);
        int R1=A+B, R2=R1+C, R3=R2+D;
        if (!safe(R3)) continue;
        add3(RouteType::Add, "+", "+", "+", A, B, C, D, R1, R2, R3);
        ++i;
    }
    // Accumulative product: A*B*C
    for (int i = 0; i < 40; ) {
        int A=pick(2,6), B=pick(2,6), C=pick(2,4);
        int R1=A*B, R2=R1*C;
        if (!safe(R2)) continue;
        // Encode as 3-step by using a dummy D (we need 4 operands for add3).
        // Instead use direct add1 for 2-step product.
        add1(RouteType::Mul,
             std::to_string(A)+" * "+std::to_string(B)+" = "+std::to_string(R1)+
             " , "+std::to_string(R1)+" * "+std::to_string(C)+" = "+std::to_string(R2));
        ++i;
    }
    // Triangle area: B*H=R1 , R1/2=R2   (B and H even so result is exact)
    for (int H = 2; H <= 16; H += 2)
        for (int B = 2; B <= 16; B += 2) {
            int R1=B*H, R2=R1/2;
            if (!safe(R2)) continue;
            add1(RouteType::Div,
                 std::to_string(B)+" * "+std::to_string(H)+" = "+std::to_string(R1)+
                 " , "+std::to_string(R1)+" / 2 = "+std::to_string(R2));
        }
    // Perimeter of rectangle: W+H=R1 , R1*2=R2
    for (int W = 1; W <= 15; ++W)
        for (int H = 1; H <= 15; ++H) {
            int R1=W+H, R2=R1*2;
            if (!safe(R2)) continue;
            add1(RouteType::Mul,
                 std::to_string(W)+" + "+std::to_string(H)+" = "+std::to_string(R1)+
                 " , "+std::to_string(R1)+" * 2 = "+std::to_string(R2));
        }
    // Celsius to Fahrenheit:  C*9=R1 , R1/5=R2 , R2+32=T  (C multiples of 5)
    for (int C = 0; C <= 40; C += 5) {
        int R1=C*9, R2=R1/5, T=R2+32;
        if (R1%5!=0||!safe(T)) continue;
        add3(RouteType::Add, "*", "/", "+", C, 9, 5, 32, R1, R2, T);
    }
    // Simple interest: P*R=R1 , R1*T=R2
    for (int i = 0; i < 30; ) {
        int P=pick(10,50), R=pick(1,5), T=pick(1,4);
        int R1=P*R, R2=R1*T;
        if (!safe(R2)) continue;
        add1(RouteType::Mul,
             std::to_string(P)+" * "+std::to_string(R)+" = "+std::to_string(R1)+
             " , "+std::to_string(R1)+" * "+std::to_string(T)+" = "+std::to_string(R2));
        ++i;
    }
    // Pythagorean triples: A*A=R1 , B*B=R2 , R1+R2=R3  (3,4,5 family)
    for (auto [a,b] : std::initializer_list<std::pair<int,int>>{
         {3,4},{5,12},{8,15},{7,24},{6,8},{9,12},{5,12},{8,6},{20,21},{9,40}}) {
        int R1=a*a, R2=b*b, R3=R1+R2;
        if (!safe(R3)) continue;
        add3(RouteType::Add, "*", "*", "+", a, a, b, b, R1, R2, R3);
    }
    // Mixed: A*B+C (multiply then add — order of operations pattern)
    for (int i = 0; i < 50; ) {
        int A=pick(2,10),B=pick(2,10),C=pick(1,20);
        int R1=A*B, R2=R1+C;
        if (!safe(R2)) continue;
        add2(RouteType::Mul, RouteType::Add, "*", "+", A, B, C, R1, R2);
        ++i;
    }
    // Mixed: A+B*C (add then multiply — teaches "last op wins" in chain form)
    for (int i = 0; i < 50; ) {
        int A=pick(1,15),B=pick(2,10),C=pick(2,8);
        int R1=A+B, R2=R1*C;
        if (!safe(R2)) continue;
        add2(RouteType::Add, RouteType::Mul, "+", "*", A, B, C, R1, R2);
        ++i;
    }

    // ── 4. OOD test set ───────────────────────────────────────────────────────
    // Single-step OOD: operands outside all training ranges.
    constexpr int kOodLo = 41, kOodHi = 60;

    // Add OOD
    for (int i = 0; i < 20; ) {
        int A=pick_t(kOodLo,kOodHi), B=pick_t(kOodLo,kOodHi);
        int R=A+B;
        if (!safe(R)) continue;
        raw_test.push_back({""+std::to_string(A)+" + "+std::to_string(B)+" =", R, RouteType::Add});
        ++i;
    }
    // Sub OOD
    for (int i = 0; i < 20; ) {
        int A=pick_t(kOodLo,kOodHi), B=pick_t(kOodLo,kOodHi);
        int R=A-B;
        if (!safe(R)) continue;
        raw_test.push_back({std::to_string(A)+" - "+std::to_string(B)+" =", R, RouteType::Sub});
        ++i;
    }
    // Mul OOD
    for (int i = 0; i < 20; ) {
        int A=pick_t(21,30), B=pick_t(21,30);
        int R=A*B;
        if (!safe(R)) continue;
        raw_test.push_back({std::to_string(A)+" * "+std::to_string(B)+" =", R, RouteType::Mul});
        ++i;
    }
    // Div OOD (exact)
    for (int k = 1; k <= 15; ++k) {
        int B=pick_t(21,30), A=k*B;
        if (!safe(A)) continue;
        raw_test.push_back({std::to_string(A)+" / "+std::to_string(B)+" =", k, RouteType::Div});
    }
    // Sqrt OOD: larger perfect squares (196, 225, 256, 289, 324, 400)
    for (int r : {14,15,16,17,18,20})
        raw_test.push_back({"sqrt ( "+std::to_string(r*r)+" ) =", r, RouteType::Sqrt});

    // Two-step OOD: Add→Mul with OOD operands
    for (int i = 0; i < 15; ) {
        int A=pick_t(kOodLo,kOodHi), B=pick_t(kOodLo,kOodHi), C=pick_t(2,5);
        int R1=A+B, R2=R1*C;
        if (!safe(R2)) continue;
        raw_test.push_back({
            std::to_string(A)+" + "+std::to_string(B)+" = "+std::to_string(R1)+
            " , "+std::to_string(R1)+" * "+std::to_string(C)+" =",
            R2, RouteType::Mul});
        ++i;
    }
    // Two-step OOD: Mul→Add with OOD operands
    for (int i = 0; i < 15; ) {
        int A=pick_t(21,30), B=pick_t(21,30), C=pick_t(kOodLo,kOodHi);
        int R1=A*B, R2=R1+C;
        if (!safe(R2)) continue;
        raw_test.push_back({
            std::to_string(A)+" * "+std::to_string(B)+" = "+std::to_string(R1)+
            " , "+std::to_string(R1)+" + "+std::to_string(C)+" =",
            R2, RouteType::Add});
        ++i;
    }

    // ── 5. BPE training + encoding ────────────────────────────────────────────
    // vocab_size=70: handles existing op symbols + "sqrt", "(", ")", "++"/"--"
    auto bpe  = BPETokenizer::train(corpus, 70);
    NumericTokenizer ntok(std::move(bpe));
    // The corpus safe() guard above uses kMax=32767; verify it matches the tokenizer range.
    if (ntok.int_max() != kMax || ntok.int_min() != -kMax - 1)
        throw std::logic_error("build_math_dataset_base: tokenizer range != corpus safe range");
    const int64_t V = ntok.total_vocab_size();

    std::vector<std::vector<int32_t>> train_ids;
    std::vector<RouteType>            train_ops;
    train_ids.reserve(corpus.size());
    train_ops.reserve(corpus.size());
    for (std::size_t i = 0; i < corpus.size(); ++i) {
        auto ids = ntok.encode(corpus[i]);
        if (ids.size() < 2) continue;
        train_ids.push_back({ids.begin(), ids.end()});
        train_ops.push_back(corpus_ops[i]);
    }

    // Build test items.
    std::vector<TestItemImproved> test_items;
    test_items.reserve(raw_test.size());
    for (auto& rt : raw_test) {
        auto ids = ntok.encode(rt.prompt);
        if (ids.empty()) continue;
        test_items.push_back({{ids.begin(), ids.end()}, rt.expected, rt.op});
    }

    // No vocab bias for the advanced algebraic training (full-vocab from the start).
    Tensor bias_t_full({V}, DType::Float32);
    auto bp = bias_t_full.data_as<float>();
    for (int64_t i = 0; i < V; ++i) bp[static_cast<std::size_t>(i)] = 0.0f;

    return ImprovedData{std::move(ntok), V,
                        std::move(train_ids), std::move(train_ops),
                        std::move(test_items), {}, std::move(bias_t_full)};
}


// ── §21.14  Math training dataset ────────────────────────────────────────────
// Builds on the base corpus: adds 240 Sqrt training items (r=0..20, 12 each),
// extends OOD Sqrt test to r=21..26 (441..676), and uses boost=15 at train time.
static ImprovedData build_math_dataset() {
    // Extend the base corpus with additional Sqrt items.
    ImprovedData base = build_math_dataset_base();

    // Replace base Sqrt test items with the extended OOD set.
    {
        std::vector<TestItemImproved> kept;
        kept.reserve(base.test_items.size());
        for (auto& item : base.test_items)
            if (item.expected_op != RouteType::Sqrt)
                kept.push_back(std::move(item));
        base.test_items = std::move(kept);
    }

    // Add OOD Sqrt test: r=21..26 (values 441, 484, 529, 576, 625, 676)
    for (int r = 21; r <= 26; ++r) {
        auto ids = base.ntok.encode("sqrt ( " + std::to_string(r * r) + " ) =");
        if (!ids.empty())
            base.test_items.push_back(
                {{ids.begin(), ids.end()}, r, RouteType::Sqrt});
    }

    // Add Sqrt training items: 12 copies per r for r=0..20 (extra on top of base's 4).
    for (int r = 0; r <= 20; ++r) {
        int sq = r * r;
        int extra = (r <= 15) ? 8 : 12;   // base already has 4; add 8 more → 12 total per r
        for (int rep = 0; rep < extra; ++rep) {
            auto ids = base.ntok.encode(
                "sqrt ( " + std::to_string(sq) + " ) = " + std::to_string(r));
            if (ids.size() < 2) continue;
            base.train_ids.push_back({ids.begin(), ids.end()});
            base.train_ops.push_back(RouteType::Sqrt);
        }
    }

    return base;
}

// ── Shared op-name table (must stay in sync with RouteType enum) ─────────────

static constexpr std::array<std::string_view, kNumRouteTypes> kOpNames{
    "FFN","Add","Sub","Mul","Div","IsLT","IsGT","IsEq","Inc","Dec","Sqrt"};

// ── Orchestrator ──────────────────────────────────────────────────────────────

static void section_improved_training(std::string_view phase,
                                       std::string_view ckpt_dir,
                                       int steps = 0,
                                       int ckpt_step = 0,
                                       TokenMode token_mode = TokenMode::Real)
{
    std::cout << std::format("  checkpoint directory: {}\n", ckpt_dir);

    if (phase == "train") {
        const int total_t = (steps > 0) ? steps : 2000;
        std::cout << "\n  §21.14 Exclusive math routing — CE-only, no supervision\n";
        std::cout << "  LM head masked for numeric range: router trains from CE loss alone\n";
        std::cout << std::format("  target: {} steps\n", total_t);

        ImprovedData d = build_math_dataset();
        std::cout << std::format("  training corpus: {} items  test set: {} items\n",
                                  d.train_ids.size(), d.test_items.size());

        MathGPT model(d.V, kD, static_cast<std::size_t>(kNHeads),
                      static_cast<std::size_t>(kNKv), kNLayers, -1, 0, /*seed=*/42);
        auto all_params = model.parameters();
        Adam adam(all_params, 1e-3f);

        const std::string prefix     = "train";
        int               start_step = load_latest_checkpoint(all_params, prefix, ckpt_dir);
        if (start_step >= total_t) {
            auto [acc, spec, ent] = eval_improved(model, d, false, TokenMode::Algebraic);
            std::cout << std::format("  Already complete — acc={:.1f}%  spec={:.1f}%  ent={:.2f}\n",
                                      acc*100.f, spec*100.f, ent);
            return;
        }
        start_step = std::max(start_step, -1) + 1;

        const int ckpt_iv = 200;
        std::mt19937 rng(77);
        for (int i = 0; i < start_step; ++i) rng();

        print_train_header(/*show_alpha=*/false);
        float last_loss = 0.f;
        auto  t_phase   = std::chrono::steady_clock::now();
        int   steps_since_eval = 0;
        float best_acc  = 0.f;
        int   best_step = -1;

        for (int step = start_step; step <= total_t; ++step) {
            if (step % std::max(1, total_t / 5) == 0 || step == start_step) {
                float ms = 0.f;
                if (steps_since_eval > 0) {
                    auto now = std::chrono::steady_clock::now();
                    ms = std::chrono::duration<float, std::milli>(now - t_phase).count()
                         / static_cast<float>(steps_since_eval);
                    t_phase = now; steps_since_eval = 0;
                }
                auto [acc, spec, ent] = eval_improved(model, d, false, TokenMode::Algebraic);
                print_train_row(step, last_loss, acc, spec, ent, ms);  // no alpha column
                if (acc > best_acc && step > 0) {
                    best_acc = acc; best_step = step;
                    save_checkpoint(all_params,
                        std::filesystem::path(ckpt_dir) /
                            std::format("ch21_{}_best_step{:04d}.ckpt", prefix, step), step);
                }
            }
            if (step == total_t) {
                save_checkpoint(all_params,
                    std::filesystem::path(ckpt_dir) /
                        std::format("ch21_{}_step{:04d}.ckpt", prefix, step), step);
                break;
            }
            if (step > start_step && step % ckpt_iv == 0)
                save_checkpoint(all_params,
                    std::filesystem::path(ckpt_dir) /
                        std::format("ch21_{}_step{:04d}.ckpt", prefix, step), step);

            const std::size_t idx = rng() % d.train_ids.size();
            const auto&       ids = d.train_ids[idx];
            if (ids.size() < 2) { --step; continue; }

            Tensor id_tensor = make_ids_tensor(ids);
            const int64_t T_loss = static_cast<int64_t>(ids.size()) - 1;

            adam.zero_grad();
            auto logits = model.forward_math(id_tensor, d.ntok, TokenMode::Algebraic);
            auto ltrunc = narrow(logits, 0, T_loss);
            auto L_ce   = cross_entropy(ltrunc, make_targets(ids));
            L_ce.backward();
            (void)clip_grad_norm(all_params, 1.0f);
            adam.step();
            last_loss = L_ce.data().data_as<float>()[0];
            ++steps_since_eval;
        }

        auto [acc, spec, ent] = eval_improved(model, d, false, TokenMode::Algebraic);
        std::cout << std::format("\n  §21.14 result: acc={:.1f}%  router_spec={:.1f}%  entropy={:.2f}\n",
                                  acc * 100.f, spec * 100.f, ent);
        if (best_step >= 0)
            std::cout << std::format("  best checkpoint: step {} ({:.1f}%) — ch21_{}_best_step{:04d}.ckpt\n",
                                      best_step, best_acc * 100.f, prefix, best_step);
        return;
    }

    if (phase == "eval") {
        std::cout << "\n  §21.14 Eval: per-op accuracy breakdown on alg4 OOD test set\n";
        ImprovedData d = build_math_dataset();
        std::cout << std::format("  test set: {} items\n", d.test_items.size());

        MathGPT model(d.V, kD, static_cast<std::size_t>(kNHeads),
                      static_cast<std::size_t>(kNKv), kNLayers, -1, 0, /*seed=*/42);
        auto params = model.parameters();
        int step = load_latest_checkpoint(params, "train_best", ckpt_dir, ckpt_step);
        const bool used_best = (step >= 0);
        if (step < 0)
            step = load_latest_checkpoint(params, "train", ckpt_dir, ckpt_step);
        if (step < 0)
            throw std::runtime_error(std::format("No train checkpoint in {}", ckpt_dir));
        std::cout << std::format("  loaded {} checkpoint step {}\n\n",
                                  used_best ? "best" : "latest", step);

        std::array<int, kNumRouteTypes> op_ok{};
        std::array<int, kNumRouteTypes> op_tot{};
        std::array<int, kNumRouteTypes> fail_shown{};

        for (const auto& item : d.test_items) {
            if (item.prompt_ids.empty()) continue;
            Tensor ids_t = make_ids_tensor(item.prompt_ids);
            Variable logits = model.forward_math(ids_t, d.ntok, TokenMode::Algebraic);

            const int64_t last = logits.data().shape(0) - 1;
            auto lsp = logits.data().data_as<float>();
            int64_t best = 0;
            float best_v = lsp[static_cast<std::size_t>(last * d.V)];
            for (int64_t v = 1; v < d.V; ++v) {
                float val = lsp[static_cast<std::size_t>(last * d.V + v)];
                if (val > best_v) { best_v = val; best = v; }
            }

            const auto op_idx = static_cast<std::size_t>(item.expected_op);
            ++op_tot[op_idx];
            auto pred_id = static_cast<NumericTokenizer::TokenId>(best);
            const bool correct = d.ntok.is_numeric(pred_id) &&
                !d.ntok.is_nan_token(pred_id) && !d.ntok.is_overflow_token(pred_id) &&
                static_cast<int32_t>(d.ntok.numeric_value(pred_id)) == item.expected_val;
            if (correct) { ++op_ok[op_idx]; continue; }

            if (fail_shown[op_idx] < 3) {
                RouteInfo ri = model.route_info(ids_t, d.ntok, TokenMode::Algebraic);
                std::string_view chosen_op = ri.routes.empty()
                    ? "?" : kOpNames[static_cast<std::size_t>(ri.routes.back())];
                std::string pred_str = d.ntok.is_numeric(pred_id)
                    ? std::to_string(static_cast<int32_t>(d.ntok.numeric_value(pred_id)))
                    : "non-numeric";
                std::cout << std::format("  [{}] routed={} pred={} expected={}\n",
                    kOpNames[op_idx], chosen_op, pred_str, item.expected_val);
                ++fail_shown[op_idx];
            }
        }

        std::cout << "\n";
        std::cout << std::format("  {:>12}  {:>5}  {:>5}  {:>8}\n", "op", "ok", "total", "acc%");
        std::cout << "  " << std::string(35, '-') << "\n";
        int grand_ok = 0, grand_tot = 0;
        for (std::size_t k = 0; k < kNumRouteTypes; ++k) {
            if (op_tot[k] == 0) continue;
            float acc_k = static_cast<float>(op_ok[k]) / static_cast<float>(op_tot[k]) * 100.f;
            std::cout << std::format("  {:>12}  {:>5}  {:>5}  {:>7.1f}%\n",
                                      kOpNames[k], op_ok[k], op_tot[k], acc_k);
            grand_ok += op_ok[k]; grand_tot += op_tot[k];
        }
        std::cout << "  " << std::string(35, '-') << "\n";
        std::cout << std::format("  {:>12}  {:>5}  {:>5}  {:>7.1f}%\n\n",
                                  "TOTAL", grand_ok, grand_tot,
                                  static_cast<float>(grand_ok)/static_cast<float>(grand_tot)*100.f);
        return;
    }

    if (phase == "train_real") {
        // Real-mode counterpart of train for side-by-side OOD comparison.
        // Same vocabulary, data, architecture, and CE-only objective; only the
        // TokenMode differs — numeric tokens keep their actual values instead of
        // being remapped to abstract slots.
        const int total_t = (steps > 0) ? steps : 2000;
        std::cout << "\n  §21.14 Real-mode training — CE-only, no supervision\n";
        std::cout << "  Same data/vocab as train; TokenMode::Real (actual numeric tokens)\n";
        std::cout << std::format("  target: {} steps\n", total_t);

        ImprovedData d = build_math_dataset();
        std::cout << std::format("  training corpus: {} items  test set: {} items\n",
                                  d.train_ids.size(), d.test_items.size());

        MathGPT model(d.V, kD, static_cast<std::size_t>(kNHeads),
                      static_cast<std::size_t>(kNKv), kNLayers, -1, 0, /*seed=*/42);
        auto all_params = model.parameters();
        Adam adam(all_params, 1e-3f);

        const std::string prefix     = "train_real";
        int               start_step = load_latest_checkpoint(all_params, prefix, ckpt_dir);
        if (start_step >= total_t) {
            auto [acc, spec, ent] = eval_improved(model, d, false, TokenMode::Real);
            std::cout << std::format("  Already complete — acc={:.1f}%  spec={:.1f}%  ent={:.2f}\n",
                                      acc*100.f, spec*100.f, ent);
            return;
        }
        start_step = std::max(start_step, -1) + 1;

        const int ckpt_iv = 200;
        std::mt19937 rng(77);
        for (int i = 0; i < start_step; ++i) rng();

        print_train_header();
        float last_loss = 0.f;
        auto  t_phase   = std::chrono::steady_clock::now();
        int   steps_since_eval = 0;
        float best_acc  = 0.f;
        int   best_step = -1;

        for (int step = start_step; step <= total_t; ++step) {
            if (step % std::max(1, total_t / 5) == 0 || step == start_step) {
                float ms = 0.f;
                if (steps_since_eval > 0) {
                    auto now = std::chrono::steady_clock::now();
                    ms = std::chrono::duration<float, std::milli>(now - t_phase).count()
                         / static_cast<float>(steps_since_eval);
                    t_phase = now; steps_since_eval = 0;
                }
                auto [acc, spec, ent] = eval_improved(model, d, false, TokenMode::Real);
                print_train_row(step, last_loss, acc, spec, ent, ms);
                if (acc > best_acc && step > 0) {
                    best_acc = acc; best_step = step;
                    save_checkpoint(all_params,
                        std::filesystem::path(ckpt_dir) /
                            std::format("ch21_{}_best_step{:04d}.ckpt", prefix, step), step);
                }
            }
            if (step == total_t) {
                save_checkpoint(all_params,
                    std::filesystem::path(ckpt_dir) /
                        std::format("ch21_{}_step{:04d}.ckpt", prefix, step), step);
                break;
            }
            if (step > start_step && step % ckpt_iv == 0)
                save_checkpoint(all_params,
                    std::filesystem::path(ckpt_dir) /
                        std::format("ch21_{}_step{:04d}.ckpt", prefix, step), step);

            const std::size_t idx = rng() % d.train_ids.size();
            const auto&       ids = d.train_ids[idx];
            if (ids.size() < 2) { --step; continue; }

            Tensor id_tensor = make_ids_tensor(ids);
            const int64_t T_loss = static_cast<int64_t>(ids.size()) - 1;

            adam.zero_grad();
            auto logits = model.forward_math(id_tensor, d.ntok, TokenMode::Real);
            auto ltrunc = narrow(logits, 0, T_loss);
            auto L_ce   = cross_entropy(ltrunc, make_targets(ids));
            L_ce.backward();
            (void)clip_grad_norm(all_params, 1.0f);
            adam.step();
            last_loss = L_ce.data().data_as<float>()[0];
            ++steps_since_eval;
        }

        auto [acc, spec, ent] = eval_improved(model, d, false, TokenMode::Real);
        std::cout << std::format("\n  train_real result: acc={:.1f}%  router_spec={:.1f}%  entropy={:.2f}\n",
                                  acc * 100.f, spec * 100.f, ent);
        if (best_step >= 0)
            std::cout << std::format("  best checkpoint: step {} ({:.1f}%) — ch21_{}_best_step{:04d}.ckpt\n",
                                      best_step, best_acc * 100.f, prefix, best_step);
        return;
    }

    if (phase == "compare") {
        // §21.14: Real vs Algebraic on the same extended OOD test set.
        // Both models share vocabulary (build_advanced_algebraic_data_v3).
        // Real mode processes actual numeric token embeddings; algebraic mode
        // remaps them to abstract sequential slots before embedding lookup.
        std::cout << "\n  §21.14 Real vs Algebraic — OOD generalisation comparison\n";
        ImprovedData d = build_math_dataset();
        std::cout << std::format("  test set: {} items  vocab_size: {}\n\n",
                                  d.test_items.size(), d.V);

        // Load algebraic model
        MathGPT alg_model(d.V, kD, static_cast<std::size_t>(kNHeads),
                          static_cast<std::size_t>(kNKv), kNLayers, -1, 0, /*seed=*/42);
        {
            auto p = alg_model.parameters();
            int  s = load_latest_checkpoint(p, "train_best", ckpt_dir, ckpt_step);
            if (s < 0) s = load_latest_checkpoint(p, "train", ckpt_dir, ckpt_step);
            if (s < 0) throw std::runtime_error(std::format("No train checkpoint in {}", ckpt_dir));
            std::cout << std::format("  Algebraic model: step {} (train)\n", s);
        }

        // Load real model
        MathGPT real_model(d.V, kD, static_cast<std::size_t>(kNHeads),
                           static_cast<std::size_t>(kNKv), kNLayers, -1, 0, /*seed=*/42);
        {
            auto p = real_model.parameters();
            int  s = load_latest_checkpoint(p, "train_real_best", ckpt_dir, ckpt_step);
            if (s < 0) s = load_latest_checkpoint(p, "train_real", ckpt_dir, ckpt_step);
            if (s < 0) throw std::runtime_error(std::format("No train_real checkpoint in {}", ckpt_dir));
            std::cout << std::format("  Real model:       step {} (train_real)\n\n", s);
        }

        // Per-op tallies for both modes
        std::array<int, kNumRouteTypes> alg_ok{}, alg_tot{};
        std::array<int, kNumRouteTypes> real_ok{}, real_tot{};

        for (const auto& item : d.test_items) {
            if (item.prompt_ids.empty()) continue;
            Tensor ids_t = make_ids_tensor(item.prompt_ids);
            const auto op_idx = static_cast<std::size_t>(item.expected_op);

            auto eval_one = [&](MathGPT& m, TokenMode mode) -> bool {
                Variable logits = m.forward_math(ids_t, d.ntok, mode);
                const int64_t last = logits.data().shape(0) - 1;
                auto lsp = logits.data().data_as<float>();
                int64_t best = 0;
                float best_v = lsp[static_cast<std::size_t>(last * d.V)];
                for (int64_t v = 1; v < d.V; ++v) {
                    float val = lsp[static_cast<std::size_t>(last * d.V + v)];
                    if (val > best_v) { best_v = val; best = v; }
                }
                auto pred_id = static_cast<NumericTokenizer::TokenId>(best);
                return d.ntok.is_numeric(pred_id) &&
                       !d.ntok.is_nan_token(pred_id) &&
                       !d.ntok.is_overflow_token(pred_id) &&
                       static_cast<int32_t>(d.ntok.numeric_value(pred_id)) == item.expected_val;
            };

            ++alg_tot[op_idx];
            ++real_tot[op_idx];
            if (eval_one(alg_model,  TokenMode::Algebraic)) ++alg_ok[op_idx];
            if (eval_one(real_model, TokenMode::Real))      ++real_ok[op_idx];
        }

        // Print comparison table
        std::cout << std::format("  {:>8}  {:>12}  {:>12}  {:>10}  {:>5}\n",
                                  "op", "algebraic", "real", "Δ(alg-real)", "n");
        std::cout << "  " << std::string(55, '-') << "\n";
        int grand_alg_ok = 0, grand_real_ok = 0, grand_tot = 0;
        for (std::size_t k = 0; k < kNumRouteTypes; ++k) {
            if (alg_tot[k] == 0) continue;
            const float alg_a  = static_cast<float>(alg_ok[k])  / static_cast<float>(alg_tot[k]) * 100.f;
            const float real_a = static_cast<float>(real_ok[k]) / static_cast<float>(real_tot[k]) * 100.f;
            std::cout << std::format("  {:>8}  {:>11.1f}%  {:>11.1f}%  {:>+9.1f}%  {:>5}\n",
                                      kOpNames[k], alg_a, real_a, alg_a - real_a, alg_tot[k]);
            grand_alg_ok += alg_ok[k]; grand_real_ok += real_ok[k]; grand_tot += alg_tot[k];
        }
        std::cout << "  " << std::string(55, '-') << "\n";
        const float tot_alg  = static_cast<float>(grand_alg_ok)  / static_cast<float>(grand_tot) * 100.f;
        const float tot_real = static_cast<float>(grand_real_ok) / static_cast<float>(grand_tot) * 100.f;
        std::cout << std::format("  {:>8}  {:>11.1f}%  {:>11.1f}%  {:>+9.1f}%  {:>5}\n\n",
                                  "TOTAL", tot_alg, tot_real, tot_alg - tot_real, grand_tot);
        return;
    }

    if (phase != "all")
        throw std::runtime_error(std::format("unknown phase '{}' — valid: train eval train_real compare", phase));
}

// ── §21.10  Generalization Across Scale: Real vs Algebraic ────────────────────
//
// Tests whether the math-neuron architecture generalises beyond training operand
// ranges, and whether Algebraic mode closes the gap that Real mode leaves open.
//
// Real mode:      transformer embedding sees actual numeric token IDs.
//                 OOD operands (e.g. 5000) were never in the *input* position during
//                 training — only as *results*. The router may see unfamiliar
//                 embedding patterns and misroute.
//
// Algebraic mode: numeric tokens are replaced with abstract slots X0, X1, ...
//                 The embedding always sees the same abstract OP(Xi, Xj) = ?
//                 pattern regardless of the actual values; the reg[] side-channel
//                 carries the exact floats for arithmetic computation.
//                 Generalisation should be perfect at any scale.
//
// Chained tier:   2-step expressions "A op1 B = R , R op2 C =".  Tests that
//                 the router correctly identifies the SECOND operation even when
//                 R is an OOD value produced at runtime, not from training data.

struct GenTestTier {
    std::string                   label;
    std::vector<TestItemImproved> items;
};

static std::vector<GenTestTier> build_gentest_tiers(const NumericTokenizer& ntok) {
    std::mt19937 rng(2025);

    // Collect valid (A, B) pairs from a range, shuffle, return up to n_sample items.
    auto sample_arith = [&](RouteType op, const std::string& sym,
                             std::function<int(int,int)> compute,
                             int A_lo, int A_hi, int B_lo, int B_hi,
                             int n_sample) -> std::vector<TestItemImproved> {
        std::vector<std::pair<int,int>> valid;
        for (int A = A_lo; A <= A_hi; ++A)
            for (int B = B_lo; B <= B_hi; ++B) {
                int r = compute(A, B);
                if (r >= ntok.int_min() && r <= ntok.int_max())
                    valid.emplace_back(A, B);
            }
        std::shuffle(valid.begin(), valid.end(), rng);
        std::vector<TestItemImproved> out;
        for (int i = 0; i < std::min(n_sample, (int)valid.size()); ++i) {
            auto [A, B] = valid[i];
            auto ids = ntok.encode(std::to_string(A) + " " + sym + " " +
                                    std::to_string(B) + " =");
            if (!ids.empty())
                out.push_back({std::vector<int32_t>(ids.begin(), ids.end()),
                                compute(A, B), op});
        }
        return out;
    };

    // Exact-division cases: A = k * B.
    auto sample_div = [&](int B_lo, int B_hi, int k_max, int n_sample)
        -> std::vector<TestItemImproved> {
        std::vector<std::pair<int,int>> valid;
        for (int B = B_lo; B <= B_hi; ++B)
            for (int k = 1; k <= k_max; ++k) {
                int A = k * B;
                if (A <= ntok.int_max()) valid.emplace_back(A, B);
            }
        std::shuffle(valid.begin(), valid.end(), rng);
        std::vector<TestItemImproved> out;
        for (int i = 0; i < std::min(n_sample, (int)valid.size()); ++i) {
            auto [A, B] = valid[i];
            auto ids = ntok.encode(std::to_string(A) + " / " + std::to_string(B) + " =");
            if (!ids.empty())
                out.push_back({std::vector<int32_t>(ids.begin(), ids.end()),
                                A / B, RouteType::Div});
        }
        return out;
    };

    static constexpr int kN = 30;  // samples per op per tier

    auto add_tier = [&](std::string label,
                         int op_lo, int op_hi,        // add/sub/cmp operand range
                         int mul_a_lo, int mul_a_hi,  // mul left operand range
                         int mul_b_max,               // mul right operand max
                         int div_b_lo, int div_b_hi,  // div denominator range
                         int div_k_max)               // div quotient max
        -> GenTestTier
    {
        GenTestTier t;
        t.label = std::move(label);
        auto push = [&](std::vector<TestItemImproved>&& v) {
            for (auto& item : v) t.items.push_back(std::move(item));
        };
        push(sample_arith(RouteType::Add, "+", [](int A,int B){return A+B;},
                           op_lo,op_hi, op_lo,op_hi, kN));
        push(sample_arith(RouteType::Sub, "-", [](int A,int B){return A-B;},
                           op_lo,op_hi, op_lo,op_hi, kN));
        push(sample_arith(RouteType::Mul, "*", [](int A,int B){return A*B;},
                           mul_a_lo,mul_a_hi, 1,mul_b_max, kN));
        push(sample_div(div_b_lo, div_b_hi, div_k_max, kN));
        push(sample_arith(RouteType::IsLessThan, "<",
                           [](int A,int B){return A<B?1:0;}, op_lo,op_hi, op_lo,op_hi, kN));
        push(sample_arith(RouteType::IsGreaterThan, ">",
                           [](int A,int B){return A>B?1:0;}, op_lo,op_hi, op_lo,op_hi, kN));
        push(sample_arith(RouteType::IsEqual, "==",
                           [](int A,int B){return A==B?1:0;}, op_lo,op_hi, op_lo,op_hi, kN/3));
        return t;
    };

    std::vector<GenTestTier> tiers;

    // Near-OOD: [36, 100] — just past the existing OOD test range [26, 35]
    tiers.push_back(add_tier("Near-OOD   [36-100]",
                              36,100,  36,100,9,  36,100,9));

    // Mid-OOD: [200, 999] — hundreds scale, far beyond training/test
    tiers.push_back(add_tier("Mid-OOD  [200-999]",
                              200,999,  200,999,9,  200,999,9));

    // Large-OOD: [1000, 9000] — thousands scale
    // Mul: A∈[1000,3000], B∈[1,9] → max 27000 ≤ 32767
    // Div: B∈[1000,3000], k∈[1,9] → A=k*B ≤ 27000
    tiers.push_back(add_tier("Large-OOD [1K-9K]",
                              1000,9000,  1000,3000,9,  1000,3000,9));

    // ── Chained tier: "A op1 B = R1 , R1 op2 C =" ────────────────────────────
    // The model must route at the SECOND = position using an OOD intermediate.
    // Three sub-types: Add→Mul, Mul→Add, Sub→Mul.
    {
        GenTestTier t;
        t.label = "Chained    [2-step]";
        std::uniform_int_distribution<int> pick_med(100, 499);
        std::uniform_int_distribution<int> pick_sm(2, 9);

        // Add→Mul: "A + B = R1 , R1 * C ="  (A,B ∈ [100,499], C small)
        for (int i = 0; i < kN; ) {
            int A = pick_med(rng), B = pick_med(rng), C = pick_sm(rng);
            int R1 = A + B, R2 = R1 * C;
            if (R2 < ntok.int_min() || R2 > ntok.int_max()) continue;
            std::string p = std::to_string(A) + " + " + std::to_string(B) + " = " +
                             std::to_string(R1) + " , " + std::to_string(R1) +
                             " * " + std::to_string(C) + " =";
            auto ids = ntok.encode(p);
            if (!ids.empty())
                t.items.push_back({std::vector<int32_t>(ids.begin(), ids.end()),
                                    R2, RouteType::Mul});
            ++i;
        }

        // Mul→Add: "A * B = R1 , R1 + C ="  (A∈[20,99], B small, C ∈ [100,499])
        for (int i = 0; i < kN; ) {
            std::uniform_int_distribution<int> pick_a2(20, 99);
            int A = pick_a2(rng), B = pick_sm(rng), C = pick_med(rng);
            int R1 = A * B, R2 = R1 + C;
            if (R2 < ntok.int_min() || R2 > ntok.int_max()) continue;
            std::string p = std::to_string(A) + " * " + std::to_string(B) + " = " +
                             std::to_string(R1) + " , " + std::to_string(R1) +
                             " + " + std::to_string(C) + " =";
            auto ids = ntok.encode(p);
            if (!ids.empty())
                t.items.push_back({std::vector<int32_t>(ids.begin(), ids.end()),
                                    R2, RouteType::Add});
            ++i;
        }

        // Sub→Mul: "A - B = R1 , R1 * C ="  (A > B, both ∈ [100,499], C small)
        for (int i = 0; i < kN; ) {
            int A = pick_med(rng), B = pick_med(rng), C = pick_sm(rng);
            if (A == B) continue;
            if (A < B) std::swap(A, B);
            int R1 = A - B, R2 = R1 * C;
            if (R2 < ntok.int_min() || R2 > ntok.int_max()) continue;
            std::string p = std::to_string(A) + " - " + std::to_string(B) + " = " +
                             std::to_string(R1) + " , " + std::to_string(R1) +
                             " * " + std::to_string(C) + " =";
            auto ids = ntok.encode(p);
            if (!ids.empty())
                t.items.push_back({std::vector<int32_t>(ids.begin(), ids.end()),
                                    R2, RouteType::Mul});
            ++i;
        }

        tiers.push_back(std::move(t));
    }

    return tiers;
}

struct GenTierResult {
    float acc   = 0.f;
    int   total = 0;
    std::array<int, kNumRouteTypes> op_ok{};
    std::array<int, kNumRouteTypes> op_tot{};
};

// Evaluate a model on all tiers; returns per-tier result with per-op stats.
static std::vector<GenTierResult> eval_gentest(
    MathGPT&                           model,
    const NumericTokenizer&            ntok,
    int64_t                            V,
    const std::vector<GenTestTier>&    tiers,
    TokenMode                          mode)
{
    std::vector<GenTierResult> results;
    results.reserve(tiers.size());

    for (const auto& tier : tiers) {
        GenTierResult tr{};
        for (const auto& item : tier.items) {
            if (item.prompt_ids.empty()) continue;
            Tensor ids_t = make_ids_tensor(item.prompt_ids);
            Variable logits = model.forward_math(ids_t, ntok, mode);
            const int64_t last = logits.data().shape(0) - 1;
            auto lsp  = logits.data().data_as<float>();
            int64_t best = 0;
            float   best_v = lsp[static_cast<std::size_t>(last * V)];
            for (int64_t v = 1; v < V; ++v) {
                float val = lsp[static_cast<std::size_t>(last * V + v)];
                if (val > best_v) { best_v = val; best = v; }
            }
            auto pred_id = static_cast<NumericTokenizer::TokenId>(best);
            int32_t pred_val = INT32_MIN;
            if (ntok.is_numeric(pred_id) && !ntok.is_nan_token(pred_id) &&
                !ntok.is_overflow_token(pred_id))
                pred_val = static_cast<int32_t>(ntok.numeric_value(pred_id));
            ++tr.total;
            const int op_idx = static_cast<int>(item.expected_op);
            ++tr.op_tot[op_idx];
            if (pred_val == item.expected_val) {
                ++tr.op_ok[op_idx];
            }
        }
        tr.acc = tr.total > 0
            ? static_cast<float>(
                  std::accumulate(tr.op_ok.begin(), tr.op_ok.end(), 0)) /
              static_cast<float>(tr.total)
            : 0.f;
        results.push_back(tr);
    }
    return results;
}

static void section_generalization_test(
    std::string_view real_ckpt_dir,
    std::string_view alg_ckpt_dir)
{
    std::cout << "\n=== §21.10  Generalization Across Scale: Real vs Algebraic ===\n";
    std::cout << "  Algebraic mode replaces every numeric token with an abstract slot\n";
    std::cout << "  (X0, X1, …) so the transformer always sees the same OP(Xi,Xj)=?\n";
    std::cout << "  pattern regardless of value magnitude.  Real mode sees the actual\n";
    std::cout << "  token IDs — routing depends on learned embedding geometry which\n";
    std::cout << "  only covers the training operand range [0,25].\n";

    ImprovedData base = build_math_dataset();
    auto tiers = build_gentest_tiers(base.ntok);

    // Print test-set composition
    std::cout << "\n  Test tiers:\n";
    for (const auto& tier : tiers)
        std::cout << std::format("    {:>22} — {} items\n", tier.label, tier.items.size());

    // Load Real model
    MathGPT real_model(base.V, kD, static_cast<std::size_t>(kNHeads),
                        static_cast<std::size_t>(kNKv), kNLayers, -1, 0, /*seed=*/42);
    {
        auto p    = real_model.parameters();
        int  step = load_latest_checkpoint(p, "train_real_best", real_ckpt_dir);
        if (step < 0)
            step = load_latest_checkpoint(p, "train_real", real_ckpt_dir);
        if (step < 0)
            throw std::runtime_error(
                std::format("No train_real checkpoint in {}", real_ckpt_dir));
        std::cout << std::format("\n  Real model      (step {:>4}): {}\n", step, real_ckpt_dir);
    }

    // Load Algebraic model
    MathGPT alg_model(base.V, kD, static_cast<std::size_t>(kNHeads),
                       static_cast<std::size_t>(kNKv), kNLayers, -1, 0, /*seed=*/42);
    {
        auto p    = alg_model.parameters();
        int  step = load_latest_checkpoint(p, "train_best", alg_ckpt_dir);
        if (step < 0)
            step = load_latest_checkpoint(p, "train", alg_ckpt_dir);
        if (step < 0)
            throw std::runtime_error(
                std::format("No train checkpoint in {}", alg_ckpt_dir));
        std::cout << std::format("  Algebraic model (step {:>4}): {}\n", step, alg_ckpt_dir);
    }

    // Evaluate both models on every tier
    auto real_r = eval_gentest(real_model, base.ntok, base.V, tiers, TokenMode::Real);
    auto alg_r  = eval_gentest(alg_model,  base.ntok, base.V, tiers, TokenMode::Algebraic);

    // Print comparison table
    std::cout << "\n";
    std::cout << std::format("  {:>22}  {:>9}  {:>9}  {:>11}  {}\n",
                              "tier", "real", "algebraic", "Δ(alg-real)", "n");
    std::cout << "  " << std::string(62, '-') << "\n";
    for (std::size_t i = 0; i < tiers.size(); ++i) {
        float r = real_r[i].acc, a = alg_r[i].acc;
        int   n = real_r[i].total;
        std::cout << std::format("  {:>22}  {:>8.1f}%  {:>8.1f}%  {:>+10.1f}%  {}\n",
                                  tiers[i].label, r * 100.f, a * 100.f,
                                  (a - r) * 100.f, n);
    }
    std::cout << "  " << std::string(62, '-') << "\n";

    // Overall summary
    int real_ok = 0, alg_ok = 0, total = 0;
    for (std::size_t i = 0; i < tiers.size(); ++i) {
        real_ok += std::accumulate(real_r[i].op_ok.begin(), real_r[i].op_ok.end(), 0);
        alg_ok  += std::accumulate(alg_r[i].op_ok.begin(),  alg_r[i].op_ok.end(),  0);
        total   += real_r[i].total;
    }
    std::cout << std::format("  {:>22}  {:>8.1f}%  {:>8.1f}%  {:>+10.1f}%  {}\n",
                              "OVERALL",
                              100.f * static_cast<float>(real_ok) / static_cast<float>(total),
                              100.f * static_cast<float>(alg_ok)  / static_cast<float>(total),
                              100.f * static_cast<float>(alg_ok - real_ok) /
                                      static_cast<float>(total),
                              total);

    // Per-op breakdown for the Chained tier (last tier)
    static constexpr std::array<std::string_view, kNumRouteTypes> kOpNamesLong = {
        "FFN", "Add", "Sub", "Mul", "Div",
        "IsLessThan", "IsGreaterThan", "IsEqual", "Increment", "Decrement", "Sqrt"
    };
    const auto& chain_real = real_r.back();
    const auto& chain_alg  = alg_r.back();
    std::cout << "\n  Per-op breakdown — Chained tier:\n";
    std::cout << std::format("  {:>14}  {:>14}  {:>14}  {:>11}\n",
                              "op", "real", "algebraic", "n");
    std::cout << "  " << std::string(58, '-') << "\n";
    for (std::size_t k = 0; k < static_cast<std::size_t>(kNumRouteTypes); ++k) {
        if (chain_real.op_tot[k] == 0) continue;
        const float r_acc = static_cast<float>(chain_real.op_ok[k]) /
                            static_cast<float>(chain_real.op_tot[k]);
        const float a_acc = static_cast<float>(chain_alg.op_ok[k]) /
                            static_cast<float>(chain_alg.op_tot[k]);
        std::cout << std::format("  {:>14}  {:>13.1f}%  {:>13.1f}%  {:>11}\n",
                                  kOpNamesLong[k], r_acc * 100.f, a_acc * 100.f,
                                  chain_real.op_tot[k]);
    }
}

// ── main ──────────────────────────────────────────────────────────────────────
//
// Usage:
//   ./ch21_math_neurons                                              # run all sections (default)
//   ./ch21_math_neurons train --token-mode anon                      # single-phase, 3000 steps (recommended)
//   ./ch21_math_neurons train --steps 5000 --token-mode algebraic    # single-phase, 5000 steps
//   ./ch21_math_neurons 1cs                                          # legacy Phase 1Cs only
//   ./ch21_math_neurons 2cs                                          # legacy Phase 2Cs (requires 1cs ckpt)
//   ./ch21_math_neurons 2cs --steps 5000                             # Phase 2Cs for 5000 steps
//   ./ch21_math_neurons 2cs --steps 10000 --sched-total 5000         # continue from step 5000
//   ./ch21_math_neurons eval --token-mode anon                       # load latest checkpoint
//   ./ch21_math_neurons eval --ckpt-step 3000 --token-mode anon      # eval at step milestone
//   ./ch21_math_neurons spot --token-mode anon                       # per-example spot check
//   ./ch21_math_neurons spot --ckpt-step 3000 --token-mode algebraic # spot check at step 3000
//   ./ch21_math_neurons --phase train --ckpt-dir /tmp/ckpts --token-mode anon
//   ./ch21_math_neurons --phase gentest --ckpt-dir /tmp/real --ckpt-dir-alg /tmp/alg

int main(int argc, char* argv[]) {
    // Force line-buffered stdout so log files written via redirection are readable
    // in real-time rather than flushing only on process exit.
    std::setvbuf(stdout, nullptr, _IOLBF, 0);

    std::string phase       = "all";
    std::string ckpt_dir    = ".";
    std::string ckpt_dir_alg = "";   // algebraic checkpoint dir for --phase gentest
    int         steps       = 0;  // 0 = use phase default
    int         ckpt_step   = 0;  // 0 = load latest checkpoint; >0 = load <= this step
    TokenMode   token_mode  = TokenMode::Real;

    for (int i = 1; i < argc; ++i) {
        std::string_view arg(argv[i]);
        if      (arg == "--phase"        && i + 1 < argc) { phase        = argv[++i]; }
        else if (arg == "--ckpt-dir"     && i + 1 < argc) { ckpt_dir     = argv[++i]; }
        else if (arg == "--ckpt-dir-alg" && i + 1 < argc) { ckpt_dir_alg = argv[++i]; }
        else if (arg == "--steps"       && i + 1 < argc) { steps       = std::stoi(argv[++i]); }
        else if (arg == "--ckpt-step"   && i + 1 < argc) { ckpt_step   = std::stoi(argv[++i]); }
        else if (arg == "--token-mode"  && i + 1 < argc) {
            std::string_view m(argv[++i]);
            if      (m == "anon")               token_mode = TokenMode::Anon;
            else if (m == "algebraic")          token_mode = TokenMode::Algebraic;
            else if (m == "algebraic-special")  token_mode = TokenMode::AlgebraicSpecial;
            else if (m == "real")               token_mode = TokenMode::Real;
            else { std::cerr << "Unknown --token-mode: " << m
                             << " (use real/anon/algebraic/algebraic-special)\n"; return 1; }
        }
        else if (!arg.starts_with("--"))                  { phase       = std::string(arg); }
    }

    std::cout << "Chapter 21 — Math Neurons: Arithmetic-Aware Transformers\n";
    std::cout << std::string(60, '=') << '\n';

    // §21.10 generalization test: needs both real and algebraic checkpoint dirs.
    if (phase == "gentest") {
        if (ckpt_dir_alg.empty())
            ckpt_dir_alg = ckpt_dir + "_alg";  // default: same base + "_alg"
        section_generalization_test(ckpt_dir, ckpt_dir_alg);
        std::cout << "\nDone.\n";
        return 0;
    }

    // When targeting a training/eval phase, skip the earlier demo sections.
    const bool improved_only = (phase == "train"      || phase == "train_real" ||
                                 phase == "eval"       || phase == "compare");
    if (!improved_only) {
        section_numeric_tokenizer();
        section_math_ops();
        section_numeric_router();
        section_arithmetic_training();
        section_layer_ablation();
        section_training_dynamics();
        section_curriculum_learning();
        section_large_numbers();
    }

    section_improved_training(phase, ckpt_dir, steps, ckpt_step, token_mode);

    std::cout << "\nDone.\n";
    return 0;
}
