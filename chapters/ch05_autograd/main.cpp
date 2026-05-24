// Chapter 05 — Autograd Engine
//
// Neural networks are trained by gradient descent.  Computing gradients by hand
// for every possible computation graph is impractical; instead we build a
// dynamic computation graph (define-by-run, like PyTorch) and run
// reverse-mode automatic differentiation (backprop) through it.
//
// Topics covered:
//   1. Why automatic differentiation?
//   2. The computation graph: nodes and edges
//   3. Reverse-mode AD (backpropagation) via topological sort
//   4. Gradient check: numerical vs analytical
//   5. Gradient accumulation and the reuse problem
//   6. A one-hidden-layer MLP trained on XOR
//   7. What's next: embeddings + transformer blocks

#include <cmath>
#include <format>
#include <random>
#include <vector>

#include "sub0llm/autograd/ops.hpp"
#include "sub0llm/autograd/variable.hpp"
#include "sub0llm/compat/print.hpp"
#include "sub0llm/core/ops.hpp"
#include "sub0llm/version.hpp"

using namespace sub0llm;
using namespace sub0llm::autograd;

static void section(std::string_view title) {
    std::println("\n── {} ──", title);
}

// Create a 1-D leaf variable from a vector.
static Variable leaf_1d(std::vector<float> vals, bool rg = true) {
    Tensor t = zeros({static_cast<int64_t>(vals.size())});
    auto   s = t.data_as<float>();
    for (std::size_t i = 0; i < vals.size(); ++i) s[i] = vals[i];
    return Variable(std::move(t), rg);
}

// Create a 2-D leaf variable from a flat vector.
static Variable leaf_2d(std::vector<float> vals, int64_t rows, int64_t cols,
                        bool rg = true) {
    Tensor t = zeros({rows, cols});
    auto   s = t.data_as<float>();
    for (std::size_t i = 0; i < vals.size(); ++i) s[i] = vals[i];
    return Variable(std::move(t), rg);
}

// Numerical gradient check (finite differences).
static float numerical_grad(std::function<float()> f_plus,
                             std::function<float()> f_minus,
                             float eps = 1e-3f) {
    return (f_plus() - f_minus()) / (2.0f * eps);
}

// ─────────────────────────────────────────────────────────────────────────────
int main() {
    std::println("sub0llm v{} — Chapter 05: Autograd Engine", version_string);

    // ── 1. Why automatic differentiation? ────────────────────────────────────
    section("1. Why automatic differentiation?");
    std::println(
        "Training requires dL/dθ for every parameter θ.\n"
        "Hand-deriving gradients for each architecture is error-prone.\n"
        "Autograd builds a computation graph during the forward pass\n"
        "and traverses it in reverse (backprop) to compute all gradients\n"
        "exactly — no approximation, same cost as ~2–3 forward passes."
    );

    // ── 2. The computation graph ──────────────────────────────────────────────
    section("2. The computation graph");
    std::println(
        "y = relu(x * w + b)\n"
        "\n"
        "  x ─┐\n"
        "      mul ─── add ─── relu ─── y\n"
        "  w ─┘         │\n"
        "               b\n"
        "\n"
        "Each node stores: data (forward value) + grad (backward accumulator).\n"
        "Each edge stores: a VJP closure that maps upstream_grad → input_grad."
    );

    // ── 3. Forward and backward ───────────────────────────────────────────────
    section("3. Forward and backward");

    auto x = leaf_1d({-1.0f, 0.5f, 2.0f});
    auto w = leaf_1d({ 2.0f, 3.0f, 1.0f});
    auto b = leaf_1d({ 0.5f, 0.5f, 0.5f});

    auto h   = add(mul(x, w), b);   // h = x*w + b
    auto act = relu(h);             // act = relu(h)
    auto L   = sum(act);            // L = sum of activations (scalar)

    std::println("Forward:  x*w + b = [{:.2f}, {:.2f}, {:.2f}]",
        h.data().data_as<float>()[0],
        h.data().data_as<float>()[1],
        h.data().data_as<float>()[2]);
    std::println("          relu    = [{:.2f}, {:.2f}, {:.2f}]",
        act.data().data_as<float>()[0],
        act.data().data_as<float>()[1],
        act.data().data_as<float>()[2]);
    std::println("          loss    = {:.4f}", L.data().data_as<float>()[0]);

    L.backward();

    std::println("Backward dL/dw    = [{:.2f}, {:.2f}, {:.2f}]",
        w.grad().data_as<float>()[0],
        w.grad().data_as<float>()[1],
        w.grad().data_as<float>()[2]);
    std::println("         dL/db    = [{:.2f}, {:.2f}, {:.2f}]",
        b.grad().data_as<float>()[0],
        b.grad().data_as<float>()[1],
        b.grad().data_as<float>()[2]);

    // ── 4. Gradient check ─────────────────────────────────────────────────────
    section("4. Numerical gradient check");

    // Verify dL/dw[1] numerically: L = relu(x[1]*w[1] + b[1]) (others not affected).
    const float eps = 1e-3f;
    // w[1] = 3.0; x[1] = 0.5; h[1] = 1.5+0.5 = 2.0 > 0 → relu is active
    // dL/dw[1] analytical = x[1] = 0.5
    auto compute = [&](float w1_val) {
        auto xi = leaf_1d({-1.0f, 0.5f, 2.0f}, false);
        auto wi = leaf_1d({2.0f, w1_val, 1.0f}, false);
        auto bi = leaf_1d({0.5f, 0.5f, 0.5f}, false);
        auto hi = add(mul(xi, wi), bi);
        return ops::sum(relu(hi).data());
    };
    const float num_grad = numerical_grad(
        [&] { return compute(3.0f + eps); },
        [&] { return compute(3.0f - eps); },
        eps);
    const float ana_grad = w.grad().data_as<float>()[1];
    std::println("  dL/dw[1]: analytical={:.4f}  numerical={:.4f}  err={:.2e}",
                 ana_grad, num_grad, std::abs(ana_grad - num_grad));

    // ── 5. Gradient accumulation ──────────────────────────────────────────────
    section("5. Gradient accumulation (reuse)");
    std::println(
        "When a variable is used multiple times, its gradient is\n"
        "accumulated (summed) from all paths.\n"
        "  y = x + x  →  dL/dx = 2  (two paths, each contributing 1)"
    );
    {
        auto xr = leaf_1d({3.0f});
        auto yr = sum(add(xr, xr));
        yr.backward();
        std::println("  dL/dx = {:.1f}  (expected 2.0)",
                     xr.grad().data_as<float>()[0]);
    }

    // ── 6. XOR MLP — one full training loop ───────────────────────────────────
    section("6. Training a 1-hidden-layer MLP on XOR");

    // XOR dataset: 4 samples, 2 features, 1 binary label (encoded as class 0 or 1).
    // logits shape: (4, 2), targets: (4,) int32
    const std::vector<float> X_data = {0,0, 0,1, 1,0, 1,1};
    const std::vector<int32_t> y_data = {0, 1, 1, 0};

    Tensor targets_t = zeros({4}, DType::Int32);
    for (std::size_t i = 0; i < 4u; ++i)
        targets_t.data_as<int32_t>()[i] = y_data[i];

    // Xavier init for weights.
    std::mt19937 rng(42);
    auto xavier = [&](int64_t r, int64_t c) {
        std::uniform_real_distribution<float> dist(-1.0f / std::sqrt(static_cast<float>(c)),
                                                    1.0f / std::sqrt(static_cast<float>(c)));
        std::vector<float> v(static_cast<std::size_t>(r * c));
        for (auto& val : v) val = dist(rng);
        return leaf_2d(v, r, c);
    };

    // Network: 2 → 4 → 2 (hidden=4, output=2 classes)
    // Bias is 1D (C,); bias_add broadcasts it over the batch dimension.
    auto W1 = xavier(2, 4);
    auto b1 = leaf_1d(std::vector<float>(4, 0.0f));
    auto W2 = xavier(4, 2);
    auto b2 = leaf_1d(std::vector<float>(2, 0.0f));

    auto X = leaf_2d(X_data, 4, 2, false);

    const float lr     = 0.5f;
    const int   epochs = 200;

    float last_loss = 0.0f;
    for (int epoch = 0; epoch <= epochs; ++epoch) {
        // Forward
        auto h1     = relu(bias_add(matmul(X, W1), b1));   // (4,4)
        auto logits = bias_add(matmul(h1, W2), b2);         // (4,2)
        auto loss   = cross_entropy(logits, targets_t);

        // Backward
        loss.backward();

        last_loss = loss.data().data_as<float>()[0];
        if (epoch % 50 == 0)
            std::println("  epoch {:3d}  loss = {:.4f}", epoch, last_loss);

        // Gradient descent step + zero grad
        auto step = [&](Variable& v) {
            auto vs = v.data().data_as<float>();
            const auto gs = v.grad().data_as<float>();
            for (std::size_t i = 0; i < vs.size(); ++i)
                vs[i] -= lr * gs[i];
            v.zero_grad();
        };
        step(W1); step(b1); step(W2); step(b2);
    }

    // Evaluate accuracy
    auto h1_eval  = relu(bias_add(matmul(X, W1), b1));
    auto logits_e = bias_add(matmul(h1_eval, W2), b2);
    const auto ld = logits_e.data().data_as<float>();
    int correct = 0;
    for (std::size_t i = 0; i < 4u; ++i) {
        const int pred = (ld[i * 2u + 0u] > ld[i * 2u + 1u]) ? 0 : 1;
        if (pred == y_data[i]) ++correct;
    }
    std::println("Accuracy: {}/4  (expected 4/4)", correct);

    // ── 7. What's next ────────────────────────────────────────────────────────
    section("7. What's next — Ch06: Embeddings");
    std::println(
        "The MLP above takes raw floats.  LLMs take discrete token IDs.\n"
        "Ch06 adds:\n"
        "  • Embedding table: int32 token → float32 vector (lookup + scatter grad)\n"
        "  • Positional embedding: learned or sinusoidal\n"
        "  • RoPE: rotary position encoding used in Llama/Mistral\n"
        "\nAfter Ch06 the pipeline is:\n"
        "  token_ids → embeddings → transformer blocks → logits → cross-entropy"
    );

    return 0;
}
