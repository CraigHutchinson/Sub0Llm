#include "sub0llm/nn/math_nodes.hpp"
#include "sub0llm/nn/numeric_router.hpp"
#include "sub0llm/tokenizer/numeric_tokenizer.hpp"
#include "sub0llm/tokenizer/bpe.hpp"
#include "sub0llm/nn/optimizer.hpp"
#include "sub0llm/nn/modern_gpt.hpp"
#include "sub0llm/autograd/ops.hpp"
#include "sub0llm/core/tensor.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <iostream>
#include <random>
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
    const int64_t D = 48;
    const std::size_t n_heads = 2, n_kv = 1;
    const int64_t n_layers = 4;

    ModernGPT baseline(total_vocab, D, n_heads, n_kv, n_layers, 0, 0, /*seed=*/42);
    MathGPT   math_model(total_vocab, D, n_heads, n_kv, n_layers, -1, 0, /*seed=*/42);

    Adam adam_base(baseline.parameters(), 3e-3f);
    Adam adam_math(math_model.parameters(), 3e-3f);

    const int steps = 200;
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

        if (step == 0 || (step + 1) % 50 == 0) {
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
    std::cout << "  Training 4 MathGPT models with l_math = 0..3, n_layers=4\n\n";

    // Build a shared dataset
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

    for (int64_t lm = 0; lm < 4; ++lm) {
        MathGPT model(total_vocab, 48, 2, 1, 4, lm, 0, /*seed=*/42);
        Adam adam(model.parameters(), 3e-3f);
        std::mt19937 rng_train(11);

        float last_loss = 0.f;
        for (int step = 0; step < 100; ++step) {
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

// ── main ──────────────────────────────────────────────────────────────────────

int main() {
    std::cout << "Chapter 21 — Math Neurons: Arithmetic-Aware Transformers\n";
    std::cout << std::string(60, '=') << '\n';

    section_numeric_tokenizer();
    section_math_ops();
    section_numeric_router();
    section_arithmetic_training();
    section_layer_ablation();

    std::cout << "\nDone.\n";
    return 0;
}
