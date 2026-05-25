// Chapter 08 — GPT Architecture
//
// This chapter assembles the transformer building blocks from Ch05–Ch07 into
// a complete GPT-2-style language model.  We add the two missing pieces:
//   1. Layer normalisation: normalise each feature vector to zero mean / unit
//      variance, then apply learnable scale (gamma) and shift (beta).
//   2. Feed-forward network: expand by 4×, apply GELU, project back.
//
// A complete GPT block (pre-norm style):
//   x = x + attn(norm1(x))   — causal self-attention with residual
//   x = x + ffn(norm2(x))    — feed-forward with residual
//
// The GPT model stacks N such blocks, adds a final layer norm, and produces
// logits over the vocabulary using weight-tied projection.

#include <cmath>
#include <format>
#include <vector>

#include "sub0llm/autograd/ops.hpp"
#include "sub0llm/compat/print.hpp"
#include "sub0llm/core/ops.hpp"
#include "sub0llm/nn/gpt.hpp"
#include "sub0llm/version.hpp"

using namespace sub0llm;
using namespace sub0llm::autograd;
using namespace sub0llm::nn;

static void section(std::string_view title) {
    std::println("\n── {} ──", title);
}

// ─────────────────────────────────────────────────────────────────────────────
int main() {
    std::println("sub0llm v{} — Chapter 08: GPT Architecture", version_string);

    // ── 1. GELU activation ────────────────────────────────────────────────────
    section("1. GELU activation");
    std::println(
        "GELU(x) = 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))\n"
        "Smoother than ReLU; used in GPT-2/BERT.  Gradient is non-zero for x<0."
    );
    {
        Tensor xt = zeros({1, 5});
        { auto s = xt.data_as<float>(); s[0]=-2.f; s[1]=-1.f; s[2]=0.f; s[3]=1.f; s[4]=2.f; }
        Variable x(xt, true);
        auto y = gelu(x);
        sum(y).backward();
        const auto yd = y.data().data_as<float>();
        const auto gd = x.grad().data_as<float>();
        std::println("  x    : [{:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}]",
                     -2.f, -1.f, 0.f, 1.f, 2.f);
        std::println("  gelu : [{:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}]",
                     yd[0], yd[1], yd[2], yd[3], yd[4]);
        std::println("  grad : [{:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}]",
                     gd[0], gd[1], gd[2], gd[3], gd[4]);
    }

    // ── 2. Layer normalisation ────────────────────────────────────────────────
    section("2. Layer normalisation");
    std::println(
        "y = γ ⊙ (x − μ) / √(σ² + ε) + β\n"
        "Normalises each token's feature vector to zero mean / unit variance.\n"
        "γ (weight) and β (bias) are learnable per-feature parameters."
    );
    {
        LayerNorm ln(4);
        Tensor xt = zeros({2, 4});
        { auto s = xt.data_as<float>();
          s[0]=1.f; s[1]=2.f; s[2]=3.f; s[3]=4.f;
          s[4]=0.f; s[5]=10.f; s[6]=20.f; s[7]=30.f; }
        Variable x(xt, false);
        auto y = ln.forward(x);
        const auto yd = y.data().data_as<float>();
        std::println("  Input  row0: [1, 2, 3, 4]       → norm: [{:.3f}, {:.3f}, {:.3f}, {:.3f}]",
                     yd[0], yd[1], yd[2], yd[3]);
        std::println("  Input  row1: [0, 10, 20, 30]    → norm: [{:.3f}, {:.3f}, {:.3f}, {:.3f}]",
                     yd[4], yd[5], yd[6], yd[7]);
    }

    // ── 3. Linear layer ───────────────────────────────────────────────────────
    section("3. Linear layer");
    std::println("y = x @ W^T + b  —  W: (out, in), b: (out,), Xavier-uniform init.");
    {
        Linear lin(8, 4, /*seed=*/0);
        Tensor xt = randn({3, 8});
        Variable x(std::move(xt), true, "x");
        auto y = lin.forward(x);
        std::println("  x: (3,8) → y: ({},{})  params: W{} b{}",
                     y.data().shape()[0], y.data().shape()[1],
                     lin.parameters()[0]->data().numel(),
                     lin.parameters()[1]->data().numel());
    }

    // ── 4. Feed-forward network ───────────────────────────────────────────────
    section("4. Feed-forward network");
    std::println("FFN(x) = gelu(x @ W1^T + b1) @ W2^T + b2");
    std::println("Expansion factor 4: D={} → 4D={} → D={}", 8, 32, 8);
    {
        FeedForward ffn(8, /*seed=*/1);
        Tensor xt = randn({4, 8});
        Variable x(std::move(xt), true);
        auto y = ffn.forward(x);
        sum(y).backward();
        std::println("  x: (4,8) → y: ({},{})", y.data().shape()[0], y.data().shape()[1]);
        std::println("  FFN params: {} tensors", ffn.parameters().size());
    }

    // ── 5. Transformer block ──────────────────────────────────────────────────
    section("5. Transformer block");
    std::println(
        "Pre-norm GPT block:\n"
        "  h = x + MultiHeadAttn(LayerNorm(x))   (residual #1)\n"
        "  y = h + FFN(LayerNorm(h))              (residual #2)"
    );
    {
        constexpr int64_t D = 16, H = 4;
        TransformerBlock block(D, H, /*seed=*/2);
        Tensor xt = randn({6, D});
        Variable x(std::move(xt), true, "x");
        auto y = block.forward(x);
        sum(y).backward();
        std::println("  x: (6,{}) → y: ({},{})  block params: {}",
                     D, y.data().shape()[0], y.data().shape()[1],
                     block.parameters().size());
        std::println("  Input grad norm: {:.4f}", ops::norm(x.grad()));
    }

    // ── 6. GPT model ──────────────────────────────────────────────────────────
    section("6. GPT model");
    constexpr int64_t VOCAB = 128, EMBED = 32, HEADS = 4, LAYERS = 2, MAX_SEQ = 64;
    GPT model(VOCAB, EMBED, HEADS, LAYERS, MAX_SEQ, /*seed=*/42);
    std::println(
        "GPT(vocab={}, embed={}, heads={}, layers={}, max_seq={})\n"
        "Total trainable parameters: {}",
        VOCAB, EMBED, HEADS, LAYERS, MAX_SEQ, model.parameters().size()
    );

    // Forward pass.
    constexpr int64_t T = 8;
    Tensor ids = zeros({T}, DType::Int32);
    { auto s = ids.data_as<int32_t>();
      s[0]=5; s[1]=12; s[2]=3; s[3]=5; s[4]=77; s[5]=9; s[6]=1; s[7]=42; }

    auto logits = model.forward(ids);
    std::println("Input: {} token ids  →  logits: ({},{})",
                 T, logits.data().shape()[0], logits.data().shape()[1]);

    // Dummy targets for cross-entropy.
    Tensor targets = zeros({T}, DType::Int32);
    { auto s = targets.data_as<int32_t>(); s[0]=12; s[1]=3; s[2]=5; s[3]=77; s[4]=9; s[5]=1; s[6]=42; s[7]=5; }

    auto loss = cross_entropy(logits, targets);
    std::println("Cross-entropy loss: {:.4f}  (log({}) ≈ {:.4f} if uniform)",
                 loss.data().data_as<float>()[0],
                 VOCAB, std::log(static_cast<float>(VOCAB)));

    // ── 7. Backward pass ──────────────────────────────────────────────────────
    section("7. Backward pass through the full model");
    loss.backward();

    std::size_t params_with_grad = 0;
    for (auto* p : model.parameters())
        if (p->grad().numel() > 0) ++params_with_grad;
    std::println("Parameters with gradient: {}/{}", params_with_grad,
                 model.parameters().size());

    float total_grad_norm = 0.0f;
    for (auto* p : model.parameters()) {
        const float n = ops::norm(p->grad());
        total_grad_norm += n * n;
    }
    std::println("Total gradient L2 norm:   {:.4f}", std::sqrt(total_grad_norm));

    // ── 8. What's next ────────────────────────────────────────────────────────
    section("8. What's next — Ch09: Optimizers");
    std::println(
        "Ch09 adds gradient-based optimisers:\n"
        "  • SGD with momentum\n"
        "  • Adam (adaptive moment estimation)\n"
        "  • Learning-rate schedules (cosine decay, warmup)\n"
        "  • Gradient clipping\n"
        "  • A small training loop that reduces the loss on synthetic data"
    );

    return 0;
}
