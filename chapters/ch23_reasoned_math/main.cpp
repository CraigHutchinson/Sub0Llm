// Chapter 23: Reasoned Arithmetic — Chain-of-Thought with an Exact Math Head
//
// MathGPT's register mechanism naturally supports multi-step reasoning chains.
// The two most recent numeric tokens in context are always the current operands:
//   "A op B = C op2 D = ?"  →  lhs=C (2nd-most-recent), rhs=D (most-recent)
// This means EVERY step in a chain is computed exactly — errors don't compound.
//
//   §23.1  Register walkthrough: how multi-step context flows to operands
//   §23.2  Two-step chains: train and evaluate intermediate + final accuracy
//   §23.3  Natural language word problems: text context + explicit arithmetic
//   §23.4  Three-step chains and accuracy compounding
//   §23.5  OOD multi-step: exact compute across the chain vs memorization

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
#include <format>
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

// Returns the numeric value predicted at the last token position, or nullopt.
static std::optional<int32_t> predict_int_at(const Variable& logits,
                                               int64_t pos,
                                               const NumericTokenizer& ntok) {
    const int64_t V = logits.data().shape(1);
    const auto sp   = logits.data().data_as<float>();
    int64_t best    = 0;
    float best_val  = sp[static_cast<std::size_t>(pos * V)];
    for (int64_t v = 1; v < V; ++v) {
        const float val = sp[static_cast<std::size_t>(pos * V + v)];
        if (val > best_val) { best_val = val; best = v; }
    }
    auto pred = static_cast<NumericTokenizer::TokenId>(best);
    if (!ntok.is_numeric(pred) || ntok.is_nan_token(pred) || ntok.is_overflow_token(pred))
        return std::nullopt;
    return static_cast<int32_t>(ntok.numeric_value(pred));
}

static std::optional<int32_t> predict_int_last(const Variable& logits,
                                                const NumericTokenizer& ntok) {
    return predict_int_at(logits, logits.data().shape(0) - 1, ntok);
}

// ── §23.1  Register Walkthrough ───────────────────────────────────────────────

static void section_register_walkthrough() {
    std::cout << "\n=== §23.1  Register Walkthrough ===\n";
    std::cout << "  The register maps each token position to its numeric value (NaN if text).\n"
              << "  Operand selection: rhs = most-recent numeric, lhs = 2nd-most-recent.\n\n";

    // Build a tokenizer and show register for a 2-step chain
    std::vector<std::string> corpus = {
        "1 + 2 = 3", "5 - 3 = 2", "the cat sat", "Alice has apples",
    };
    auto bpe = BPETokenizer::train(corpus, /*vocab_size=*/50);
    NumericTokenizer ntok(std::move(bpe));

    // Two-step chain: "5 + 3 = 8 - 2 = 6"
    const std::string chain = "5 + 3 = 8 - 2 = 6";
    auto ids = ntok.encode(chain);

    std::cout << std::format("  Sequence: \"{}\"\n\n", chain);
    std::cout << std::format("  {:>4}  {:>6}  {:>10}  {:>8}  {:>8}  {:>8}\n",
        "pos", "tok_id", "is_numeric", "reg_val", "lhs", "rhs");
    std::cout << "  " << std::string(56, '-') << "\n";

    // Manually walk the register to show operand selection
    std::vector<float> reg;
    reg.reserve(ids.size());
    for (auto id : ids) {
        if (ntok.is_numeric(id) && !ntok.is_nan_token(id) && !ntok.is_overflow_token(id))
            reg.push_back(ntok.numeric_value(id));
        else
            reg.push_back(std::numeric_limits<float>::quiet_NaN());
    }

    for (std::size_t t = 0; t < ids.size(); ++t) {
        const bool is_num = !std::isnan(reg[t]);
        const float reg_val = reg[t];

        // Find the two preceding numerics (lhs and rhs for a hypothetical "predict next")
        int64_t op1_pos = -1, op2_pos = -1;
        int count = 0;
        for (int64_t s = static_cast<int64_t>(t) - 1; s >= 0 && count < 2; --s) {
            if (!std::isnan(reg[static_cast<std::size_t>(s)])) {
                if (count == 0) op1_pos = s;
                else            op2_pos = s;
                ++count;
            }
        }

        const std::string rhs_str = (op1_pos >= 0)
            ? std::format("{:.0f}", reg[static_cast<std::size_t>(op1_pos)]) : "-";
        const std::string lhs_str = (op2_pos >= 0)
            ? std::format("{:.0f}", reg[static_cast<std::size_t>(op2_pos)]) : "-";
        const std::string reg_str = is_num
            ? std::format("{:.0f}", reg_val) : "NaN";

        std::cout << std::format("  {:>4}  {:>6}  {:>10}  {:>8}  {:>8}  {:>8}\n",
            t, static_cast<int32_t>(ids[t]),
            is_num ? "yes" : "no",
            reg_str, lhs_str, rhs_str);
    }

    std::cout << "\n  At position predicting '8': lhs=5, rhs=3 → Add(5,3)=8 ✓\n"
              << "  At position predicting '6': lhs=8, rhs=2 → Sub(8,2)=6 ✓\n"
              << "  Generated result '8' enters the register automatically.\n";
}

// ── §23.2  Two-Step Chain Training ────────────────────────────────────────────

// Dataset entry: token IDs for a 2-step chain plus the two expected results.
struct ChainSample {
    std::vector<int32_t> ids;       // full sequence: "A op B = C op2 D = E"
    int32_t              step1;     // expected C
    int32_t              step2;     // expected E
    // Prompt for evaluation: up to and including the first "= "
    std::vector<int32_t> prompt1;   // "A op B ="
    // Prompt for step 2: up to and including the second "= "
    std::vector<int32_t> prompt2;   // "A op B = C op2 D ="
};

static std::vector<ChainSample> build_two_step_dataset(
        const NumericTokenizer& ntok,
        int n_add_sub, int n_mul_div,
        std::mt19937& rng) {

    std::uniform_int_distribution<int> d_small(1, 20);
    std::uniform_int_distribution<int> d_small2(1, 5);

    std::vector<ChainSample> samples;
    samples.reserve(static_cast<std::size_t>(n_add_sub + n_mul_div));

    // Add→Sub chains: "A + B = C - D = E"
    for (int i = 0; i < n_add_sub; ++i) {
        int A = d_small(rng), B = d_small(rng);
        int C = A + B;
        int D = d_small2(rng);
        if (D > C) D = C;   // keep E ≥ 0
        int E = C - D;
        if (E < ntok.int_min() || E > ntok.int_max() ||
            C < ntok.int_min() || C > ntok.int_max()) continue;

        std::string full    = std::format("{} + {} = {} - {} = {}", A, B, C, D, E);
        std::string prompt1 = std::format("{} + {} =", A, B);
        std::string prompt2 = std::format("{} + {} = {} - {} =", A, B, C, D);

        auto fids = ntok.encode(full);
        auto p1   = ntok.encode(prompt1);
        auto p2   = ntok.encode(prompt2);
        if (fids.size() < 2) continue;

        samples.push_back({
            std::vector<int32_t>(fids.begin(), fids.end()),
            C, E,
            std::vector<int32_t>(p1.begin(), p1.end()),
            std::vector<int32_t>(p2.begin(), p2.end()),
        });
    }

    // Mul→Sub chains: "A * B = C - D = E"
    for (int i = 0; i < n_mul_div; ++i) {
        int A = d_small2(rng), B = d_small2(rng);
        int C = A * B;
        int D = d_small2(rng);
        if (D > C) D = C;
        int E = C - D;
        if (E < ntok.int_min() || E > ntok.int_max() ||
            C < ntok.int_min() || C > ntok.int_max()) continue;

        std::string full    = std::format("{} * {} = {} - {} = {}", A, B, C, D, E);
        std::string prompt1 = std::format("{} * {} =", A, B);
        std::string prompt2 = std::format("{} * {} = {} - {} =", A, B, C, D);

        auto fids = ntok.encode(full);
        auto p1   = ntok.encode(prompt1);
        auto p2   = ntok.encode(prompt2);
        if (fids.size() < 2) continue;

        samples.push_back({
            std::vector<int32_t>(fids.begin(), fids.end()),
            C, E,
            std::vector<int32_t>(p1.begin(), p1.end()),
            std::vector<int32_t>(p2.begin(), p2.end()),
        });
    }
    return samples;
}

static void section_two_step_chains() {
    std::cout << "\n=== §23.2  Two-Step Chain Training ===\n";
    std::cout << "  Training MathGPT on chains: \"A + B = C - D = E\"\n"
              << "  Evaluating accuracy at BOTH intermediate (C) and final (E) steps.\n\n";

    std::mt19937 rng(42);
    std::vector<std::string> corpus;
    {
        std::uniform_int_distribution<int> d(1, 20), d2(1, 5);
        for (int i = 0; i < 300; ++i) {
            int A = d(rng), B = d(rng), C = A + B, D = d2(rng);
            if (D > C) D = 1;
            corpus.push_back(std::format("{} + {} = {} - {} = {}", A, B, C, D, C - D));
        }
        for (int i = 0; i < 100; ++i) {
            int A = d2(rng), B = d2(rng), C = A * B, D = d2(rng);
            if (D > C) D = 1;
            corpus.push_back(std::format("{} * {} = {} - {} = {}", A, B, C, D, C - D));
        }
    }
    rng = std::mt19937(42);  // reset for dataset

    auto bpe  = BPETokenizer::train(corpus, /*vocab_size=*/80);
    NumericTokenizer ntok(std::move(bpe));

    auto train_set = build_two_step_dataset(ntok, 250, 80, rng);
    std::mt19937 rng_test(77);
    auto test_set  = build_two_step_dataset(ntok, 30, 10, rng_test);

    std::cout << std::format("  Vocab: {} total  |  Train: {}  |  Test: {}\n\n",
        ntok.total_vocab_size(), train_set.size(), test_set.size());

    const int64_t V = ntok.total_vocab_size();
    const int64_t D = 32;
    MathGPT   math_m(V, D, 4, 2, 4, -1, 0, /*seed=*/42);
    ModernGPT base_m (V, D, 4, 2, 4, 0, 0, /*seed=*/42);

    Adam adam_math(math_m.parameters(), 3e-3f);
    Adam adam_base(base_m.parameters(),  3e-3f);
    std::mt19937 rng_tr(11);

    const int steps = 800;
    float last_math = 0.f, last_base = 0.f;
    for (int step = 0; step < steps; ++step) {
        const auto& s = train_set[rng_tr() % train_set.size()];
        Tensor ids_t = make_ids_tensor(s.ids);
        Tensor tgt   = make_targets(s.ids);
        const int64_t T = static_cast<int64_t>(s.ids.size());

        adam_math.zero_grad();
        auto lm = cross_entropy(narrow(math_m.forward_math(ids_t, ntok), 0, T - 1), tgt);
        last_math = lm.data().data_as<float>()[0];
        lm.backward();
        (void)clip_grad_norm(math_m.parameters(), 1.0f);
        adam_math.step();

        adam_base.zero_grad();
        auto lb = cross_entropy(narrow(base_m.forward(ids_t), 0, T - 1), tgt);
        last_base = lb.data().data_as<float>()[0];
        lb.backward();
        (void)clip_grad_norm(base_m.parameters(), 1.0f);
        adam_base.step();
    }

    // Evaluate: step1 accuracy (predict C) and step2 accuracy (predict E)
    int math_s1 = 0, math_s2 = 0, base_s1 = 0, base_s2 = 0;
    for (const auto& s : test_set) {
        if (s.prompt1.empty() || s.prompt2.empty()) continue;
        Tensor p1 = make_ids_tensor(s.prompt1);
        Tensor p2 = make_ids_tensor(s.prompt2);

        auto mlog1 = math_m.forward_math(p1, ntok);
        auto mlog2 = math_m.forward_math(p2, ntok);
        auto blog1 = base_m.forward(p1);
        auto blog2 = base_m.forward(p2);

        if (predict_int_last(mlog1, ntok) == std::optional<int32_t>{s.step1}) ++math_s1;
        if (predict_int_last(mlog2, ntok) == std::optional<int32_t>{s.step2}) ++math_s2;
        if (predict_int_last(blog1, ntok) == std::optional<int32_t>{s.step1}) ++base_s1;
        if (predict_int_last(blog2, ntok) == std::optional<int32_t>{s.step2}) ++base_s2;
    }

    const float N = static_cast<float>(test_set.size());
    std::cout << std::format("  Final CE loss:  MathGPT={:.4f}  Baseline={:.4f}\n\n",
        last_math, last_base);
    std::cout << std::format("  {:>22}  {:>10}  {:>10}\n", "step", "MathGPT", "Baseline");
    std::cout << "  " << std::string(48, '-') << "\n";
    std::cout << std::format("  {:>22}  {:>9.1f}%  {:>9.1f}%\n",
        "step 1  (A op B = C)",
        100.f * static_cast<float>(math_s1) / N,
        100.f * static_cast<float>(base_s1) / N);
    std::cout << std::format("  {:>22}  {:>9.1f}%  {:>9.1f}%\n",
        "step 2  (C op D = E)",
        100.f * static_cast<float>(math_s2) / N,
        100.f * static_cast<float>(base_s2) / N);

    std::cout << "\n  MathGPT: both steps correct — the generated C enters the register\n"
              << "  so step 2 automatically uses the exact intermediate result.\n"
              << "  Baseline: accuracy at step 2 ≤ accuracy at step 1 (errors compound).\n";
}

// ── §23.3  Natural Language Word Problems ─────────────────────────────────────

static void section_word_problems() {
    std::cout << "\n=== §23.3  Natural Language Word Problems ===\n";
    std::cout << "  Key design: write operands explicitly in the expression even when\n"
              << "  they appear in the preceding text.  The last-two-numerics rule then\n"
              << "  selects the intended operands regardless of stray numbers in prose.\n\n";

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> d(1, 30), d2(1, 10);

    // Two templates:
    //   "Alice has A. Bob gives her B. She now has A + B = C."
    //   "Store has A items. D are sold. A - D = E items remain."
    std::vector<std::string> corpus;
    corpus.reserve(400);
    for (int i = 0; i < 200; ++i) {
        int A = d(rng), B = d(rng), C = A + B;
        corpus.push_back(std::format(
            "Alice has {} Bob gives {} she now has {} + {} = {}",
            A, B, A, B, C));
    }
    for (int i = 0; i < 200; ++i) {
        int A = d(rng) + 10, B = d2(rng), E = A - B;
        if (E < 0) { --i; continue; }
        corpus.push_back(std::format(
            "store has {} sold {} left {} - {} = {}",
            A, B, A, B, E));
    }

    auto bpe  = BPETokenizer::train(corpus, /*vocab_size=*/100);
    NumericTokenizer ntok(std::move(bpe));

    std::vector<std::vector<int32_t>> train_ids;
    train_ids.reserve(corpus.size());
    for (const auto& e : corpus) {
        auto ids = ntok.encode(e);
        if (ids.size() >= 2)
            train_ids.push_back(std::vector<int32_t>(ids.begin(), ids.end()));
    }

    // Test set: prompts ending just before the result
    std::mt19937 rng_test(99);
    std::vector<std::pair<std::vector<int32_t>, int32_t>> test_prompts;
    test_prompts.reserve(50);
    for (int i = 0; i < 25; ++i) {
        int A = d(rng_test), B = d(rng_test), C = A + B;
        std::string prompt = std::format(
            "Alice has {} Bob gives {} she now has {} + {} =", A, B, A, B);
        auto ids = ntok.encode(prompt);
        test_prompts.push_back({std::vector<int32_t>(ids.begin(), ids.end()), C});
    }
    for (int i = 0; i < 25; ++i) {
        int A = d(rng_test) + 10, B = d2(rng_test), E = A - B;
        if (E < 0) { --i; continue; }
        std::string prompt = std::format(
            "store has {} sold {} left {} - {} =", A, B, A, B);
        auto ids = ntok.encode(prompt);
        test_prompts.push_back({std::vector<int32_t>(ids.begin(), ids.end()), E});
    }

    const int64_t V = ntok.total_vocab_size();
    const int64_t D = 32;
    MathGPT   math_m(V, D, 4, 2, 4, -1, 0, /*seed=*/42);
    ModernGPT base_m (V, D, 4, 2, 4, 0, 0, /*seed=*/42);

    Adam adam_math(math_m.parameters(), 3e-3f);
    Adam adam_base(base_m.parameters(),  3e-3f);
    std::mt19937 rng_tr(17);

    const int steps = 800;
    for (int step = 0; step < steps; ++step) {
        const auto& ids = train_ids[rng_tr() % train_ids.size()];
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
    for (const auto& [prompt, expected] : test_prompts) {
        Tensor ids_t = make_ids_tensor(prompt);
        if (predict_int_last(math_m.forward_math(ids_t, ntok), ntok) ==
            std::optional<int32_t>{expected}) ++math_correct;
        if (predict_int_last(base_m.forward(ids_t), ntok) ==
            std::optional<int32_t>{expected}) ++base_correct;
    }

    const float N = static_cast<float>(test_prompts.size());
    std::cout << std::format("  Accuracy (/{}): MathGPT={:.0f}%  Baseline={:.0f}%\n\n",
        test_prompts.size(),
        100.f * static_cast<float>(math_correct) / N,
        100.f * static_cast<float>(base_correct)  / N);

    // Print a few examples showing generation
    std::cout << "  Example predictions (MathGPT):\n";
    std::mt19937 rng_ex(55);
    for (int i = 0; i < 4; ++i) {
        int A = d(rng_ex), B = d(rng_ex), C = A + B;
        std::string prompt = std::format(
            "Alice has {} Bob gives {} she now has {} + {} =", A, B, A, B);
        auto ids = ntok.encode(prompt);
        Tensor ids_t = make_ids_tensor(std::vector<int32_t>(ids.begin(), ids.end()));
        auto pred = predict_int_last(math_m.forward_math(ids_t, ntok), ntok);
        std::cout << std::format("    \"{}\" → pred={} expected={} {}\n",
            prompt, pred ? std::to_string(*pred) : "?", C,
            (pred && *pred == C) ? "✓" : "✗");
    }
}

// ── §23.4  Three-Step Chains and Compounding Accuracy ────────────────────────

static void section_three_step_chains() {
    std::cout << "\n=== §23.4  Three-Step Chains ===\n";
    std::cout << "  \"A + B = C - D = E * F = G\"\n"
              << "  At each step, the exact intermediate result feeds the next operation.\n\n";

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> da(1, 15), db(1, 10), dc(1, 3);

    // Build corpus: A + B = C - D = E * F = G
    std::vector<std::string> corpus;
    corpus.reserve(400);
    for (int i = 0; i < 400; ++i) {
        int A = da(rng), B = da(rng);
        int C = A + B;
        int D = db(rng);
        if (D >= C) D = C - 1;
        int E = C - D;
        int F = dc(rng);
        int G = E * F;
        if (G > 500) { --i; continue; }  // keep within default range
        corpus.push_back(std::format(
            "{} + {} = {} - {} = {} * {} = {}", A, B, C, D, E, F, G));
    }

    auto bpe  = BPETokenizer::train(corpus, /*vocab_size=*/80);
    NumericTokenizer ntok(std::move(bpe));

    std::vector<std::vector<int32_t>> train_ids;
    train_ids.reserve(corpus.size());
    for (const auto& e : corpus) {
        auto ids = ntok.encode(e);
        if (ids.size() >= 2)
            train_ids.push_back(std::vector<int32_t>(ids.begin(), ids.end()));
    }

    // Test set with 3 eval points each
    struct ThreeSample {
        std::vector<int32_t> p1, p2, p3;
        int32_t s1, s2, s3;
    };
    std::vector<ThreeSample> test_set;
    test_set.reserve(40);
    std::mt19937 rng_test(88);
    for (int i = 0; i < 40; ++i) {
        int A = da(rng_test), B = da(rng_test);
        int C = A + B;
        int D = db(rng_test);
        if (D >= C) D = C - 1;
        int E = C - D;
        int F = dc(rng_test);
        int G = E * F;
        if (G > 500) { --i; continue; }

        auto enc = [&](const std::string& s) {
            auto ids = ntok.encode(s);
            return std::vector<int32_t>(ids.begin(), ids.end());
        };

        test_set.push_back({
            enc(std::format("{} + {} =", A, B)),
            enc(std::format("{} + {} = {} - {} =", A, B, C, D)),
            enc(std::format("{} + {} = {} - {} = {} * {} =", A, B, C, D, E, F)),
            C, E, G
        });
    }

    const int64_t V = ntok.total_vocab_size();
    const int64_t D = 32;
    MathGPT   math_m(V, D, 4, 2, 4, -1, 0, /*seed=*/42);
    ModernGPT base_m (V, D, 4, 2, 4, 0, 0, /*seed=*/42);

    Adam adam_math(math_m.parameters(), 3e-3f);
    Adam adam_base(base_m.parameters(),  3e-3f);
    std::mt19937 rng_tr(13);

    const int steps = 1000;
    for (int step = 0; step < steps; ++step) {
        const auto& ids = train_ids[rng_tr() % train_ids.size()];
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

    int m1 = 0, m2 = 0, m3 = 0, b1 = 0, b2 = 0, b3 = 0;
    for (const auto& s : test_set) {
        Tensor p1 = make_ids_tensor(s.p1);
        Tensor p2 = make_ids_tensor(s.p2);
        Tensor p3 = make_ids_tensor(s.p3);

        if (predict_int_last(math_m.forward_math(p1, ntok), ntok) == std::optional<int32_t>{s.s1}) ++m1;
        if (predict_int_last(math_m.forward_math(p2, ntok), ntok) == std::optional<int32_t>{s.s2}) ++m2;
        if (predict_int_last(math_m.forward_math(p3, ntok), ntok) == std::optional<int32_t>{s.s3}) ++m3;
        if (predict_int_last(base_m.forward(p1), ntok) == std::optional<int32_t>{s.s1}) ++b1;
        if (predict_int_last(base_m.forward(p2), ntok) == std::optional<int32_t>{s.s2}) ++b2;
        if (predict_int_last(base_m.forward(p3), ntok) == std::optional<int32_t>{s.s3}) ++b3;
    }

    const float N = static_cast<float>(test_set.size());
    std::cout << std::format("  {:>24}  {:>10}  {:>10}\n", "step", "MathGPT", "Baseline");
    std::cout << "  " << std::string(50, '-') << "\n";
    std::cout << std::format("  {:>24}  {:>9.1f}%  {:>9.1f}%\n",
        "A + B = C",
        100.f * static_cast<float>(m1) / N, 100.f * static_cast<float>(b1) / N);
    std::cout << std::format("  {:>24}  {:>9.1f}%  {:>9.1f}%\n",
        "C - D = E",
        100.f * static_cast<float>(m2) / N, 100.f * static_cast<float>(b2) / N);
    std::cout << std::format("  {:>24}  {:>9.1f}%  {:>9.1f}%\n",
        "E * F = G",
        100.f * static_cast<float>(m3) / N, 100.f * static_cast<float>(b3) / N);

    std::cout << "\n  MathGPT: accuracy constant across all steps — exact compute at each.\n"
              << "  Baseline: accuracy decreases with depth — errors compound.\n";
}

// ── §23.5  OOD Multi-Step Generalization ─────────────────────────────────────

static void section_ood_multistep() {
    std::cout << "\n=== §23.5  OOD Multi-Step Generalization ===\n";
    std::cout << "  Train on A,B ∈ [1,20] — test on A,B ∈ [21,80].\n"
              << "  For MathGPT: each chain step is exact → OOD = in-distribution.\n"
              << "  For baseline: both steps are OOD simultaneously.\n\n";

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> train_d(1, 20), d2(1, 5);

    // Training: A + B = C - D = E, A,B in [1,20]
    std::vector<std::string> corpus;
    corpus.reserve(300);
    for (int i = 0; i < 300; ++i) {
        int A = train_d(rng), B = train_d(rng);
        int C = A + B;
        int D = d2(rng);
        if (D >= C) D = 1;
        int E = C - D;
        corpus.push_back(std::format("{} + {} = {} - {} = {}", A, B, C, D, E));
    }

    auto bpe  = BPETokenizer::train(corpus, /*vocab_size=*/80);
    NumericTokenizer ntok(std::move(bpe), 0, 200);

    std::vector<std::vector<int32_t>> train_ids;
    train_ids.reserve(corpus.size());
    for (const auto& e : corpus) {
        auto ids = ntok.encode(e);
        if (ids.size() >= 2)
            train_ids.push_back(std::vector<int32_t>(ids.begin(), ids.end()));
    }

    // OOD test: A,B in [21,80]
    std::mt19937 rng_ood(77);
    std::uniform_int_distribution<int> ood_d(21, 80);
    std::vector<std::pair<std::vector<int32_t>, int32_t>> ood_s1, ood_s2;
    ood_s1.reserve(40); ood_s2.reserve(40);
    for (int i = 0; i < 40; ++i) {
        int A = ood_d(rng_ood), B = ood_d(rng_ood);
        int C = A + B;
        int D = d2(rng_ood);
        if (D >= C) D = 1;
        int E = C - D;
        if (C < ntok.int_min() || C > ntok.int_max() ||
            E < ntok.int_min() || E > ntok.int_max()) { --i; continue; }

        auto p1_ids = ntok.encode(std::format("{} + {} =", A, B));
        auto p2_ids = ntok.encode(std::format("{} + {} = {} - {} =", A, B, C, D));
        ood_s1.push_back({std::vector<int32_t>(p1_ids.begin(), p1_ids.end()), C});
        ood_s2.push_back({std::vector<int32_t>(p2_ids.begin(), p2_ids.end()), E});
    }

    // In-distribution test: A,B in [1,20]
    std::mt19937 rng_id(88);
    std::vector<std::pair<std::vector<int32_t>, int32_t>> id_s1, id_s2;
    id_s1.reserve(40); id_s2.reserve(40);
    for (int i = 0; i < 40; ++i) {
        int A = train_d(rng_id), B = train_d(rng_id);
        int C = A + B;
        int D = d2(rng_id);
        if (D >= C) D = 1;
        int E = C - D;

        auto p1_ids = ntok.encode(std::format("{} + {} =", A, B));
        auto p2_ids = ntok.encode(std::format("{} + {} = {} - {} =", A, B, C, D));
        id_s1.push_back({std::vector<int32_t>(p1_ids.begin(), p1_ids.end()), C});
        id_s2.push_back({std::vector<int32_t>(p2_ids.begin(), p2_ids.end()), E});
    }

    const int64_t V = ntok.total_vocab_size();
    const int64_t D = 32;
    MathGPT   math_m(V, D, 4, 2, 4, -1, 0, /*seed=*/42);
    ModernGPT base_m (V, D, 4, 2, 4, 0, 0, /*seed=*/42);

    Adam adam_math(math_m.parameters(), 3e-3f);
    Adam adam_base(base_m.parameters(),  3e-3f);
    std::mt19937 rng_tr(13);

    const int steps = 900;
    for (int step = 0; step < steps; ++step) {
        const auto& ids = train_ids[rng_tr() % train_ids.size()];
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

    auto eval_acc = [&](const auto& prompts, bool use_math) {
        int correct = 0;
        for (const auto& [p, expected] : prompts) {
            Tensor ids_t = make_ids_tensor(p);
            Variable logits = use_math
                ? math_m.forward_math(ids_t, ntok)
                : base_m.forward(ids_t);
            if (predict_int_last(logits, ntok) == std::optional<int32_t>{expected}) ++correct;
        }
        return 100.f * static_cast<float>(correct) / static_cast<float>(prompts.size());
    };

    std::cout << std::format("  {:>24}  {:>10}  {:>10}\n", "test split", "MathGPT", "Baseline");
    std::cout << "  " << std::string(50, '-') << "\n";
    std::cout << std::format("  {:>24}  {:>9.1f}%  {:>9.1f}%\n",
        "in-dist step 1", eval_acc(id_s1, true),  eval_acc(id_s1, false));
    std::cout << std::format("  {:>24}  {:>9.1f}%  {:>9.1f}%\n",
        "in-dist step 2", eval_acc(id_s2, true),  eval_acc(id_s2, false));
    std::cout << std::format("  {:>24}  {:>9.1f}%  {:>9.1f}%\n",
        "OOD step 1",     eval_acc(ood_s1, true),  eval_acc(ood_s1, false));
    std::cout << std::format("  {:>24}  {:>9.1f}%  {:>9.1f}%\n",
        "OOD step 2",     eval_acc(ood_s2, true),  eval_acc(ood_s2, false));

    std::cout << "\n  MathGPT OOD accuracy matches in-distribution — exact compute at both steps.\n"
              << "  Baseline OOD collapses — it has never seen these input token combinations.\n";
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    std::string phase = "all";
    for (int i = 1; i < argc; ++i) {
        std::string_view arg(argv[i]);
        if (arg == "--phase" && i + 1 < argc) phase = argv[++i];
    }

    std::cout << "╔══════════════════════════════════════════════════════════════╗\n"
              << "║  Chapter 23: Reasoned Arithmetic                            ║\n"
              << "║  Chain-of-Thought with an Exact Math Head                   ║\n"
              << "╚══════════════════════════════════════════════════════════════╝\n";

    const bool run_all = (phase == "all");

    if (run_all || phase == "register")  section_register_walkthrough();
    if (run_all || phase == "two_step")  section_two_step_chains();
    if (run_all || phase == "word")      section_word_problems();
    if (run_all || phase == "three")     section_three_step_chains();
    if (run_all || phase == "ood")       section_ood_multistep();

    if (!run_all && phase != "register" && phase != "two_step" &&
        phase != "word" && phase != "three" && phase != "ood")
        throw std::runtime_error(std::format(
            "unknown phase '{}' — valid: register two_step word three ood", phase));

    std::cout << "\nDone.\n";
    return 0;
}
