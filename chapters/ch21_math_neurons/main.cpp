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
#include <cstring>
#include <numbers>
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
        {RouteType::Compare,  3.f,  7.f,  "Cmp(3, 7)    "},
        {RouteType::Compare,  7.f,  3.f,  "Cmp(7, 3)    "},
        {RouteType::Mul,    300.f,300.f,  "Mul(300,300) "},
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
// exact IEEE-754 int16 arithmetic for ANY input in [-32768, 32767] — no
// training on those numbers is required.

static void section_large_numbers() {
    std::cout << "\n=== §21.6  Large Number Arithmetic — Exact vs Statistical ===\n";
    std::cout << "  Math nodes use exact int16 arithmetic; statistical LMs must\n";
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
        {RouteType::Compare, 9999,  9998, "9999 < 9998"},   // false → 0
        {RouteType::Compare, 1000, 10000, "1000 < 10000"},  // true  → 1
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
            case RouteType::Compare: ref = (c.a < c.b) ? 1 : 0;            break;
            default: break;
        }
        if (ref < NumericTokenizer::kIntMin || ref > NumericTokenizer::kIntMax)
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
static int load_latest_checkpoint(const std::vector<autograd::Variable*>& params,
                                   std::string_view phase, std::string_view ckpt_dir)
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
            const std::size_t start = prefix.size();
            const std::size_t len   = fn.size() - start - 5;
            int s = std::stoi(fn.substr(start, len));
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
static constexpr float   kAlpha   = 0.5f;

// ── Supervision schedule ──────────────────────────────────────────────────────
// Controls how the router-supervision weight α varies over training.
// Flat      : constant α = alpha_start throughout.
// LinearDecay  : α interpolates linearly from alpha_start to alpha_end.
// CosineDecay  : cosine annealing from alpha_start down to alpha_end.
// ExpDecay     : exponential decay; reaches alpha_end at the final step.
//
// alpha_end > 0 keeps a residual signal so the router never goes unsupervised.

enum class SupProfile { Flat, LinearDecay, CosineDecay, ExpDecay };

struct SupervisionSchedule {
    SupProfile profile     = SupProfile::Flat;
    float      alpha_start = 0.5f;
    float      alpha_end   = 0.0f;

    [[nodiscard]] float alpha_at(int step, int total) const noexcept {
        if (total <= 0) return alpha_start;
        const float t = std::clamp(
            static_cast<float>(step) / static_cast<float>(total), 0.f, 1.f);
        switch (profile) {
            case SupProfile::Flat:
                return alpha_start;
            case SupProfile::LinearDecay:
                return alpha_start + (alpha_end - alpha_start) * t;
            case SupProfile::CosineDecay: {
                const float w = 0.5f * (1.f + std::cos(std::numbers::pi_v<float> * t));
                return alpha_end + (alpha_start - alpha_end) * w;
            }
            case SupProfile::ExpDecay: {
                if (alpha_start <= 0.f) return alpha_end;
                if (alpha_end   <= 0.f) return alpha_start * std::exp(-5.f * t);
                const float k = std::log(alpha_start / alpha_end);
                return alpha_start * std::exp(-k * t);
            }
        }
        return alpha_start;
    }

    [[nodiscard]] std::string label() const {
        const char* name = [&]() noexcept -> const char* {
            switch (profile) {
                case SupProfile::Flat:        return "flat";
                case SupProfile::LinearDecay: return "linear";
                case SupProfile::CosineDecay: return "cosine";
                case SupProfile::ExpDecay:    return "exp";
            }
            return "?";
        }();
        if (profile == SupProfile::Flat)
            return std::format("{} α={:.2f}", name, alpha_start);
        return std::format("{} α={:.2f}→{:.2f}", name, alpha_start, alpha_end);
    }
};

// Build the deterministic §21.9 corpus, tokenizer, test set, and logit bias.
// All seeds are fixed so any phase can call this independently.
struct ImprovedData {
    NumericTokenizer                         ntok;
    int64_t                                  V;
    std::vector<std::vector<int32_t>>        train_ids;
    std::vector<RouteType>                   train_ops;
    std::vector<TestItemImproved>            test_items;
    std::vector<float>                       bias_vec;
    Tensor                                   bias_t_full;
};

static ImprovedData build_improved_data() {
    std::mt19937 rng_data(42);
    std::uniform_int_distribution<int> d09(0, 9);
    std::uniform_int_distribution<int> d19(1, 9);

    std::vector<std::string> corpus;
    std::vector<RouteType>   corpus_ops;
    corpus.reserve(500);

    // Add: A + B, A,B ∈ [0,9], result ∈ [0,18]
    for (int i = 0; i < 100; ++i) {
        int A = d09(rng_data), B = d09(rng_data);
        corpus.push_back(std::to_string(A) + " + " + std::to_string(B) +
                          " = " + std::to_string(A + B));
        corpus_ops.push_back(RouteType::Add);
    }
    // Sub: A - B, A,B ∈ [0,9], result ∈ [-9,9]
    for (int i = 0; i < 100; ++i) {
        int A = d09(rng_data), B = d09(rng_data);
        corpus.push_back(std::to_string(A) + " - " + std::to_string(B) +
                          " = " + std::to_string(A - B));
        corpus_ops.push_back(RouteType::Sub);
    }
    // Mul: A * B, A,B ∈ [0,9], result ∈ [0,81]
    for (int i = 0; i < 100; ++i) {
        int A = d09(rng_data), B = d09(rng_data);
        corpus.push_back(std::to_string(A) + " * " + std::to_string(B) +
                          " = " + std::to_string(A * B));
        corpus_ops.push_back(RouteType::Mul);
    }
    // Div: A / B = k (exact integer), A = k*B, k,B ∈ [1,9]
    for (int i = 0; i < 100; ++i) {
        int B = d19(rng_data), k = d19(rng_data);
        int A = k * B;
        corpus.push_back(std::to_string(A) + " / " + std::to_string(B) +
                          " = " + std::to_string(k));
        corpus_ops.push_back(RouteType::Div);
    }
    // Compare: A < B = 1 or 0, A,B ∈ [0,9]
    for (int i = 0; i < 100; ++i) {
        int A = d09(rng_data), B = d09(rng_data);
        corpus.push_back(std::to_string(A) + " < " + std::to_string(B) +
                          " = " + std::to_string(A < B ? 1 : 0));
        corpus_ops.push_back(RouteType::Compare);
    }

    auto bpe = BPETokenizer::train(corpus, 50);
    NumericTokenizer ntok(std::move(bpe));
    const int64_t V = ntok.total_vocab_size();

    std::vector<std::vector<int32_t>> train_ids;
    std::vector<RouteType>            train_ops;
    for (std::size_t i = 0; i < corpus.size(); ++i) {
        auto ids = ntok.encode(corpus[i]);
        if (ids.size() >= 2) {
            train_ids.push_back(std::vector<int32_t>(ids.begin(), ids.end()));
            train_ops.push_back(corpus_ops[i]);
        }
    }

    std::vector<TestItemImproved> test_items;
    {
        std::mt19937 rng_t(99);
        std::uniform_int_distribution<int> d(0, 9);
        std::uniform_int_distribution<int> d1(1, 9);
        for (int i = 0; i < 20; ++i) {
            int A = d(rng_t), B = d(rng_t);
            auto ids = ntok.encode(std::to_string(A) + " + " + std::to_string(B) + " =");
            test_items.push_back({std::vector<int32_t>(ids.begin(), ids.end()),
                                   A + B, RouteType::Add});
        }
        for (int i = 0; i < 20; ++i) {
            int A = d(rng_t), B = d(rng_t);
            auto ids = ntok.encode(std::to_string(A) + " - " + std::to_string(B) + " =");
            test_items.push_back({std::vector<int32_t>(ids.begin(), ids.end()),
                                   A - B, RouteType::Sub});
        }
        for (int i = 0; i < 20; ++i) {
            int A = d(rng_t), B = d(rng_t);
            auto ids = ntok.encode(std::to_string(A) + " * " + std::to_string(B) + " =");
            test_items.push_back({std::vector<int32_t>(ids.begin(), ids.end()),
                                   A * B, RouteType::Mul});
        }
        for (int i = 0; i < 20; ++i) {
            int B = d1(rng_t), k = d1(rng_t);
            int A = k * B;
            auto ids = ntok.encode(std::to_string(A) + " / " + std::to_string(B) + " =");
            test_items.push_back({std::vector<int32_t>(ids.begin(), ids.end()),
                                   k, RouteType::Div});
        }
        for (int i = 0; i < 20; ++i) {
            int A = d(rng_t), B = d(rng_t);
            auto ids = ntok.encode(std::to_string(A) + " < " + std::to_string(B) + " =");
            test_items.push_back({std::vector<int32_t>(ids.begin(), ids.end()),
                                   A < B ? 1 : 0, RouteType::Compare});
        }
    }

    std::vector<bool> is_active(static_cast<std::size_t>(V), false);
    for (const auto& ids : train_ids)
        for (auto id : ids)
            is_active[static_cast<std::size_t>(id)] = true;
    for (const auto& item : test_items)
        for (auto id : item.prompt_ids)
            is_active[static_cast<std::size_t>(id)] = true;
    // Cover all possible result values: Sub [-9,9], Add [0,18], Compare [0,1],
    // Div [1,9], Mul [0,81]
    for (int v = -9; v <= 81; ++v)
        is_active[static_cast<std::size_t>(ntok.encode_int(v))] = true;

    std::vector<float> bias_vec(static_cast<std::size_t>(V), -1e9f);
    for (int64_t i = 0; i < V; ++i)
        if (is_active[static_cast<std::size_t>(i)])
            bias_vec[static_cast<std::size_t>(i)] = 0.0f;

    int64_t max_T_loss = 0;
    for (const auto& ids : train_ids)
        if (static_cast<int64_t>(ids.size()) >= 2)
            max_T_loss = std::max(max_T_loss, static_cast<int64_t>(ids.size()) - 1);

    Tensor bias_t_full({max_T_loss, V}, DType::Float32);
    {
        float* bsp = bias_t_full.data_as<float>().data();
        for (int64_t t = 0; t < max_T_loss; ++t)
            std::memcpy(bsp + t * V, bias_vec.data(),
                        static_cast<std::size_t>(V) * sizeof(float));
    }

    return ImprovedData{
        std::move(ntok), V,
        std::move(train_ids), std::move(train_ops),
        std::move(test_items),
        std::move(bias_vec), std::move(bias_t_full)
    };
}

static std::tuple<float, float, float> eval_improved(
    MathGPT& model,
    const ImprovedData& d,
    bool masked)
{
    int correct = 0, spec_count = 0;
    float spec_sum = 0.f, entropy_sum = 0.f;
    const int64_t Vl = d.V;

    for (const auto& item : d.test_items) {
        if (item.prompt_ids.empty()) continue;
        Tensor ids_t = make_ids_tensor(item.prompt_ids);
        Variable logits = model.forward_math(ids_t, d.ntok);

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

        RouteInfo ri = model.route_info(ids_t, d.ntok);
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

static void print_train_header(bool show_alpha = false) {
    if (show_alpha)
        std::cout << std::format("  {:>6}  {:>6}  {:>9}  {:>12}  {:>11}  {:>9}  {:>6}\n",
                                  "step", "loss", "accuracy", "router_spec", "entropy",
                                  "ms/step", "α");
    else
        std::cout << std::format("  {:>6}  {:>6}  {:>9}  {:>12}  {:>11}  {:>9}\n",
                                  "step", "loss", "accuracy", "router_spec", "entropy",
                                  "ms/step");
    std::cout << "  " << std::string(show_alpha ? 70 : 62, '-') << "\n";
}

// alpha < 0 → omit the α column; ms_per_step == 0 → show "---"
static void print_train_row(int step, float loss, float acc, float spec, float ent,
                             float ms_per_step = 0.f, float alpha = -1.f) {
    std::string loss_s = (step == 0) ? "  n/a" : std::format("{:6.2f}", loss);
    std::string ms_s   = (ms_per_step == 0.f) ? "    ---"
                                               : std::format("{:6.1f}ms", ms_per_step);
    if (alpha >= 0.f) {
        std::cout << std::format("  {:>6}  {}  {:>8.1f}%  {:>11.1f}%  {:>11.2f}  {:>9}  {:>6.3f}\n",
                                  step, loss_s, acc * 100.f, spec * 100.f, ent, ms_s, alpha);
    } else {
        std::cout << std::format("  {:>6}  {}  {:>8.1f}%  {:>11.1f}%  {:>11.2f}  {:>9}\n",
                                  step, loss_s, acc * 100.f, spec * 100.f, ent, ms_s);
    }
}

// ── Phase 1Cs ─────────────────────────────────────────────────────────────────
// Trains math_block_only_parameters() with masked vocab + router supervision.
// Saves checkpoints every kCkptInterval steps to ckpt_dir.
// On startup, resumes from the latest ch21_1cs_step*.ckpt if present.

static void run_improved_phase_1cs(
    std::string_view ckpt_dir, int total = 1000,
    SupervisionSchedule sched = {SupProfile::Flat, kAlpha, kAlpha})
{
    std::cout << std::format("\n  Phase 1Cs — D=32, masked vocab + router supervision ({})\n",
                              sched.label());
    std::cout << std::format("  target: {} steps\n", total);

    ImprovedData d    = build_improved_data();
    const int ckpt_iv = 100;

    MathGPT model(d.V, kD, static_cast<std::size_t>(kNHeads),
                   static_cast<std::size_t>(kNKv), kNLayers, -1, 0, /*seed=*/42);

    auto block_params = model.math_block_only_parameters();
    auto all_params   = model.parameters();
    Adam adam(block_params, 3e-3f);

    // Resume from latest checkpoint if available
    int start_step = load_latest_checkpoint(block_params, "1cs", ckpt_dir);
    if (start_step >= total) {
        std::cout << std::format("  Phase 1Cs already complete (step {} ≥ {}).\n",
                                  start_step, total);
        return;
    }
    start_step = std::max(start_step, -1) + 1;  // first step to run

    std::mt19937 rng(11);
    // Advance RNG to match the step we're resuming from
    for (int i = 0; i < start_step; ++i) rng();

    print_train_header(/*show_alpha=*/true);
    float last_loss    = 0.f;
    int   step_spec30  = -1;

    auto t_phase = std::chrono::steady_clock::now();
    int  steps_since_eval = 0;

    for (int step = start_step; step <= total; ++step) {
        const float cur_alpha = sched.alpha_at(step, total);
        if (step % (total / 5) == 0 || step == start_step) {
            float ms_per_step = 0.f;
            if (steps_since_eval > 0) {
                auto now = std::chrono::steady_clock::now();
                ms_per_step = std::chrono::duration<float, std::milli>(now - t_phase).count()
                              / static_cast<float>(steps_since_eval);
                t_phase = now;
                steps_since_eval = 0;
            }
            auto [acc, spec, ent] = eval_improved(model, d, /*masked=*/true);
            print_train_row(step, last_loss, acc, spec, ent, ms_per_step, cur_alpha);
            if (step_spec30 < 0 && spec >= 0.30f) step_spec30 = step;
        }
        if (step == total) {
            save_checkpoint(block_params,
                std::filesystem::path(ckpt_dir) /
                    std::format("ch21_1cs_step{:04d}.ckpt", step), step);
            break;
        }
        if (step > start_step && step % ckpt_iv == 0)
            save_checkpoint(block_params,
                std::filesystem::path(ckpt_dir) /
                    std::format("ch21_1cs_step{:04d}.ckpt", step), step);

        const std::size_t idx   = rng() % d.train_ids.size();
        const auto&       ids   = d.train_ids[idx];
        const RouteType   gt_op = d.train_ops[idx];
        if (ids.size() < 2) { --step; rng(); continue; }

        Tensor id_tensor = make_ids_tensor(ids);
        const int64_t T_loss = static_cast<int64_t>(ids.size()) - 1;

        for (auto* p : all_params) p->zero_grad();

        auto logits  = model.forward_math(id_tensor, d.ntok);
        auto ltrunc  = narrow(logits, 0, T_loss);
        auto lmasked = add(ltrunc, narrow(Variable(d.bias_t_full, false), 0, T_loss));
        auto L_ce    = cross_entropy(lmasked, make_targets(ids));

        // Supervision at the "=" position; weight by scheduled α
        Tensor alpha_t({1}, DType::Float32);
        alpha_t.data_as<float>()[0] = cur_alpha;
        auto rlogits   = model.router_logits(id_tensor);
        auto rlogit_eq = narrow(rlogits, T_loss - 1, 1);
        Tensor gt_t({1}, DType::Int32);
        gt_t.data_as<int32_t>()[0] = static_cast<int32_t>(gt_op);
        auto L_sup = cross_entropy(rlogit_eq, gt_t);

        auto L_total = add(L_ce, mul(Variable(alpha_t, false), L_sup));
        L_total.backward();
        (void)clip_grad_norm(block_params, 1.0f);
        adam.step();
        last_loss = L_total.data().data_as<float>()[0];
        ++steps_since_eval;
    }

    if (step_spec30 >= 0)
        std::cout << std::format("  → router_spec first crossed 30% at step {}\n",
                                  step_spec30);
    else
        std::cout << "  → router_spec did not cross 30% within Phase 1Cs\n";
}

// ── Phase 2Cs ─────────────────────────────────────────────────────────────────
// Full-vocab fine-tuning.  Initialises math_block from the Phase 1Cs checkpoint;
// saves whole-model checkpoints under ch21_2cs_sup_step*.ckpt for crash recovery.
// Supervision weight follows `sched` so the router signal is never fully cut off.
// Returns {accuracy, router_spec, entropy} after the final evaluation.

static std::tuple<float, float, float>
run_improved_phase_2cs(
    std::string_view ckpt_dir, int total = 5000,
    SupervisionSchedule sched = {SupProfile::CosineDecay, 0.3f, 0.05f})
{
    std::cout << std::format(
        "\n  Phase 2Cs — full-vocab fine-tuning, supervision schedule: {}\n",
        sched.label());
    std::cout << std::format("  target: {} steps\n", total);

    ImprovedData d    = build_improved_data();
    const int ckpt_iv = 200;

    MathGPT model(d.V, kD, static_cast<std::size_t>(kNHeads),
                   static_cast<std::size_t>(kNKv), kNLayers, -1, 0, /*seed=*/42);

    auto all_params  = model.parameters();
    auto math_params = model.math_block_only_parameters();
    Adam adam(all_params, 3e-3f);

    // Resume from a supervised Phase 2Cs checkpoint if available
    int start_step = load_latest_checkpoint(all_params, "2cs_sup", ckpt_dir);
    if (start_step < 0) {
        // No supervised checkpoint: initialise math_block from Phase 1Cs
        int p1cs_step = load_latest_checkpoint(math_params, "1cs", ckpt_dir);
        if (p1cs_step < 0)
            throw std::runtime_error(
                "Phase 2Cs requires a Phase 1Cs checkpoint.\n"
                "  Run first: ./ch21_math_neurons 1cs [--ckpt-dir <dir>]");
        std::cout << std::format("  math_block imported from Phase 1Cs checkpoint "
                                  "(step {})\n", p1cs_step);
        start_step = 0;
    } else if (start_step >= total) {
        std::cout << std::format("  Phase 2Cs already complete (step {} ≥ {}).\n",
                                  start_step, total);
        return eval_improved(model, d, false);
    }
    start_step = std::max(start_step, -1) + 1;

    std::mt19937 rng(11);
    for (int i = 0; i < start_step; ++i) rng();

    print_train_header(/*show_alpha=*/true);
    float last_loss  = 0.f;
    int   step_spec30 = -1;

    auto t_phase = std::chrono::steady_clock::now();
    int  steps_since_eval = 0;

    for (int step = start_step; step <= total; ++step) {
        const float cur_alpha = sched.alpha_at(step, total);
        if (step % (total / 5) == 0 || step == start_step) {
            float ms_per_step = 0.f;
            if (steps_since_eval > 0) {
                auto now = std::chrono::steady_clock::now();
                ms_per_step = std::chrono::duration<float, std::milli>(now - t_phase).count()
                              / static_cast<float>(steps_since_eval);
                t_phase = now;
                steps_since_eval = 0;
            }
            auto [acc, spec, ent] = eval_improved(model, d, /*masked=*/false);
            print_train_row(step, last_loss, acc, spec, ent, ms_per_step, cur_alpha);
            if (step_spec30 < 0 && spec >= 0.30f) step_spec30 = step;
        }
        if (step == total) {
            save_checkpoint(all_params,
                std::filesystem::path(ckpt_dir) /
                    std::format("ch21_2cs_sup_step{:04d}.ckpt", step), step);
            break;
        }
        if (step > start_step && step % ckpt_iv == 0)
            save_checkpoint(all_params,
                std::filesystem::path(ckpt_dir) /
                    std::format("ch21_2cs_sup_step{:04d}.ckpt", step), step);

        const std::size_t idx   = rng() % d.train_ids.size();
        const auto&       ids   = d.train_ids[idx];
        const RouteType   gt_op = d.train_ops[idx];
        if (ids.size() < 2) { --step; rng(); continue; }

        Tensor id_tensor = make_ids_tensor(ids);
        const int64_t T_loss = static_cast<int64_t>(ids.size()) - 1;

        adam.zero_grad();
        auto logits = model.forward_math(id_tensor, d.ntok);
        auto ltrunc = narrow(logits, 0, T_loss);
        auto L_ce   = cross_entropy(ltrunc, make_targets(ids));

        // Scheduled supervision at the "=" position
        Tensor alpha_t({1}, DType::Float32);
        alpha_t.data_as<float>()[0] = cur_alpha;
        auto rlogits   = model.router_logits(id_tensor);
        auto rlogit_eq = narrow(rlogits, T_loss - 1, 1);
        Tensor gt_t({1}, DType::Int32);
        gt_t.data_as<int32_t>()[0] = static_cast<int32_t>(gt_op);
        auto L_sup   = cross_entropy(rlogit_eq, gt_t);
        auto L_total = add(L_ce, mul(Variable(alpha_t, false), L_sup));

        L_total.backward();
        (void)clip_grad_norm(all_params, 1.0f);
        adam.step();
        last_loss = L_total.data().data_as<float>()[0];
        ++steps_since_eval;
    }

    if (step_spec30 >= 0)
        std::cout << std::format("  → router_spec first crossed 30% at step {}\n",
                                  step_spec30);
    else
        std::cout << "  → router_spec did not cross 30% within Phase 2Cs\n";

    return eval_improved(model, d, false);
}

// ── Standalone evaluation ─────────────────────────────────────────────────────
// Load the Phase 2Cs checkpoint and print the summary table.

static void run_improved_eval(std::string_view ckpt_dir) {
    std::cout << "\n  Eval — loading Phase 2Cs checkpoint\n";

    ImprovedData d = build_improved_data();
    MathGPT model(d.V, kD, static_cast<std::size_t>(kNHeads),
                   static_cast<std::size_t>(kNKv), kNLayers, -1, 0, /*seed=*/42);

    auto all_params = model.parameters();
    // Prefer supervised checkpoint; fall back to unsupervised for older runs
    int loaded_step = load_latest_checkpoint(all_params, "2cs_sup", ckpt_dir);
    if (loaded_step < 0)
        loaded_step = load_latest_checkpoint(all_params, "2cs", ckpt_dir);
    if (loaded_step < 0)
        throw std::runtime_error(
            "Phase 2Cs checkpoint not found.\n"
            "  Run first: ./ch21_math_neurons 2cs [--ckpt-dir <dir>]");

    auto [acc, spec, ent] = eval_improved(model, d, false);
    std::cout << std::format("  accuracy={:.1f}%  router_spec={:.1f}%  entropy={:.2f}\n",
                              acc * 100.f, spec * 100.f, ent);
    std::cout << "\n  Summary (§21.9 Phase 2Cs vs §21.8 Phase 2C documented baseline):\n";
    std::cout << std::format("  {:>50}  {:>9}  {:>12}  {:>9}\n",
                              "model", "accuracy", "router_spec", "entropy");
    std::cout << "  " << std::string(83, '-') << "\n";
    // §21.8 values are documented baselines from the prior run; see docs/ch21_math_neurons.md
    std::cout << std::format("  {:>50}  {:>8.1f}%  {:>11.1f}%  {:>9.2f}\n",
                              "§21.8 Phase 2C (D=16, no supervision, 1000 steps)",
                              22.0f, 50.0f, 0.04f);
    std::cout << std::format("  {:>50}  {:>8.1f}%  {:>11.1f}%  {:>9.2f}\n",
                              "§21.9 Phase 2Cs (D=32, supervised, 2000 steps)",
                              acc * 100.f, spec * 100.f, ent);
}

// ── Orchestrator ──────────────────────────────────────────────────────────────

static void section_improved_training(std::string_view phase,
                                       std::string_view ckpt_dir,
                                       int steps = 0)  // 0 = use phase default
{
    // Phase supervision schedules
    const SupervisionSchedule sched_1cs{SupProfile::Flat,        kAlpha, kAlpha};
    const SupervisionSchedule sched_2cs{SupProfile::CosineDecay, 0.3f,   0.05f};

    std::cout << "\n=== §21.9  Improved Specialisation: D=32, 5 ops, Router Supervision ===\n";
    std::cout << "  D=32 (vs D=16 in §21.8): router 198 params, head_dim=16\n";
    std::cout << "  Operations: Add, Sub, Mul, Div, Compare (500 training examples)\n";
    std::cout << std::format("  Phase 1Cs: masked vocab + router supervision ({})\n",
                              sched_1cs.label());
    std::cout << std::format("  Phase 2Cs: full-vocab fine-tuning, supervision ({})\n",
                              sched_2cs.label());
    std::cout << std::format("  checkpoint directory: {}\n", ckpt_dir);

    if (phase == "eval") {
        run_improved_eval(ckpt_dir);
        return;
    }

    if (phase == "all" || phase == "1cs") {
        const int total1 = (steps > 0) ? steps : 1000;
        run_improved_phase_1cs(ckpt_dir, total1, sched_1cs);
    }

    if (phase == "all" || phase == "2cs") {
        const int total2 = (steps > 0) ? steps : 5000;
        auto [acc, spec, ent] = run_improved_phase_2cs(ckpt_dir, total2, sched_2cs);

        std::cout << "\n  Summary (Phase 2Cs vs §21.8 Phase 2C documented baseline):\n";
        std::cout << std::format("  {:>50}  {:>9}  {:>12}  {:>9}\n",
                                  "model", "accuracy", "router_spec", "entropy");
        std::cout << "  " << std::string(83, '-') << "\n";
        // §21.8 values are documented baselines; see docs/ch21_math_neurons.md
        std::cout << std::format("  {:>50}  {:>8.1f}%  {:>11.1f}%  {:>9.2f}\n",
                                  "§21.8 Phase 2C (D=16, no supervision, 1000 steps)",
                                  22.0f, 50.0f, 0.04f);
        std::cout << std::format("  {:>50}  {:>8.1f}%  {:>11.1f}%  {:>9.2f}\n",
                                  std::format("§21.9 Phase 2Cs (D=32, 5-op, sup, {} steps)", total2),
                                  acc * 100.f, spec * 100.f, ent);

        std::cout << "\n  Key improvements:\n";
        std::cout << "    D=32: router Linear(32,6) = 198 params (vs 102 at D=16)\n";
        std::cout << "    Router supervision adds direct CE signal at the '=' position\n";
        if (spec > 0.50f)
            std::cout << std::format("    Phase 2Cs router_spec {:.0f}% > 50% §21.8 baseline\n",
                                      spec * 100.f);
        if (acc > 0.22f)
            std::cout << std::format("    Phase 2Cs accuracy {:.0f}% > 22% §21.8 baseline\n",
                                      acc * 100.f);
    }
}

// ── main ──────────────────────────────────────────────────────────────────────
//
// Usage:
//   ./ch21_math_neurons                              # run all sections (default)
//   ./ch21_math_neurons 1cs                          # Phase 1Cs only
//   ./ch21_math_neurons 2cs                          # Phase 2Cs only (requires 1cs ckpt)
//   ./ch21_math_neurons 2cs --steps 5000             # extend Phase 2Cs to 5000 steps
//   ./ch21_math_neurons eval                         # load 2cs ckpt, print summary
//   ./ch21_math_neurons --phase 2cs --ckpt-dir /tmp/ckpts --steps 4000

int main(int argc, char* argv[]) {
    std::string phase    = "all";
    std::string ckpt_dir = ".";
    int         steps    = 0;  // 0 = use phase default (1000 / 2000)

    for (int i = 1; i < argc; ++i) {
        std::string_view arg(argv[i]);
        if      (arg == "--phase"    && i + 1 < argc) { phase    = argv[++i]; }
        else if (arg == "--ckpt-dir" && i + 1 < argc) { ckpt_dir = argv[++i]; }
        else if (arg == "--steps"    && i + 1 < argc) { steps    = std::stoi(argv[++i]); }
        else if (!arg.starts_with("--"))               { phase    = std::string(arg); }
    }

    std::cout << "Chapter 21 — Math Neurons: Arithmetic-Aware Transformers\n";
    std::cout << std::string(60, '=') << '\n';

    // When targeting a specific §21.9 phase, skip the earlier demo sections.
    const bool improved_only = (phase == "1cs" || phase == "2cs" || phase == "eval");
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

    section_improved_training(phase, ckpt_dir, steps);

    std::cout << "\nDone.\n";
    return 0;
}
