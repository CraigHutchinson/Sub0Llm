// Chapter 07 — Attention Mechanisms
//
// The transformer's power comes from self-attention: every token can look at
// every other token and weigh its importance.  This chapter implements the
// full differentiable attention mechanism and plugs it into the embedding
// pipeline from Ch06.
//
// Topics covered:
//   1. Scaled dot-product attention: Q@K^T / sqrt(Dh)
//   2. Causal masking: future tokens receive -∞ before softmax
//   3. Multi-head attention: H parallel heads, per-head projections W_Q,K,V
//   4. New autograd ops: softmax, transpose2d, scale
//   5. Gradient flow through the full attention forward pass
//   6. What's next: Ch08 GPT-2 architecture

#include <cmath>
#include <format>
#include <limits>
#include <vector>

#include "sub0llm/autograd/ops.hpp"
#include "sub0llm/compat/print.hpp"
#include "sub0llm/core/ops.hpp"
#include "sub0llm/nn/attention.hpp"
#include "sub0llm/nn/embedding.hpp"
#include "sub0llm/nn/positional_encoding.hpp"
#include "sub0llm/version.hpp"

using namespace sub0llm;
using namespace sub0llm::autograd;
using namespace sub0llm::nn;

static void section(std::string_view title) {
    std::println("\n── {} ──", title);
}

// ─────────────────────────────────────────────────────────────────────────────
int main() {
    std::println("sub0llm v{} — Chapter 07: Attention Mechanisms", version_string);

    // ── 1. Scaled dot-product attention (single head) ─────────────────────────
    section("1. Scaled dot-product attention");
    std::println(
        "Q (queries), K (keys), V (values) all have shape (T, D_h).\n"
        "  scores = Q @ K^T / sqrt(D_h)     (T, T)\n"
        "  attn   = softmax(scores + mask)  (T, T)\n"
        "  output = attn @ V                (T, D_h)\n"
        "\n"
        "Dividing by sqrt(D_h) keeps the dot products from growing too large\n"
        "when D_h is big — otherwise softmax saturates and gradients vanish."
    );

    constexpr int64_t T = 4, Dh = 8;
    Tensor Q = randn({T, Dh}), K = randn({T, Dh}), V = randn({T, Dh});
    Variable vQ(Q, false), vK(K, false), vV(V, false);

    auto scores    = scale(matmul(vQ, transpose2d(vK)),
                           1.0f / std::sqrt(static_cast<float>(Dh)));
    auto attn      = softmax(scores);
    auto context   = matmul(attn, vV);

    std::println("Q,K,V shape: ({},{})  scores: ({},{})  context: ({},{})",
                 T, Dh,
                 attn.data().shape()[0], attn.data().shape()[1],
                 context.data().shape()[0], context.data().shape()[1]);

    // ── 2. Causal masking ─────────────────────────────────────────────────────
    section("2. Causal masking");
    std::println(
        "For language modelling, token i must not attend to token j > i.\n"
        "We add a mask of 0 (visible) or -inf (future) before softmax:\n"
        "  mask[i,j] = 0     if j <= i\n"
        "             -inf   if j >  i\n"
        "exp(-inf) = 0, so masked positions get exactly zero attention weight."
    );

    // Verify: position 0 should get attn weight 1.0 in its row (nothing else).
    const auto ad = attn.data().data_as<float>();
    const float neg_inf = -std::numeric_limits<float>::infinity();
    Tensor mask_t = zeros({T, T});
    {
        auto md = mask_t.data_as<float>();
        for (std::size_t i = 0; i < static_cast<std::size_t>(T); ++i)
            for (std::size_t j = i + 1; j < static_cast<std::size_t>(T); ++j)
                md[i * static_cast<std::size_t>(T) + j] = neg_inf;
    }
    Variable mask_var(mask_t, false);
    auto masked_scores = add(scores, mask_var);
    auto causal_attn   = softmax(masked_scores);
    const auto ca = causal_attn.data().data_as<float>();

    std::println("Causal attn[0, :] = [{:.3f}, {:.3f}, {:.3f}, {:.3f}]  (only pos 0 visible)",
                 ca[0], ca[1], ca[2], ca[3]);
    std::println("Causal attn[2, :] = [{:.3f}, {:.3f}, {:.3f}, {:.3f}]  (pos 0-2 visible)",
                 ca[2*T], ca[2*T+1], ca[2*T+2], ca[2*T+3]);

    // ── 3. Multi-head attention ───────────────────────────────────────────────
    section("3. Multi-head self-attention");
    std::println(
        "H heads each project to D_h = D/H dimensions.\n"
        "Each head learns to attend to a different aspect of the sequence.\n"
        "Outputs are summed: out = sum_h (A_h @ V_h) @ W_O_h"
    );

    constexpr std::size_t D = 16, H = 4;
    MultiHeadSelfAttention mha(D, H, /*seed=*/42);
    std::println("MHA: embed_dim={}, num_heads={}, head_dim={}",
                 mha.embed_dim(), mha.num_heads(), mha.head_dim());
    std::println("Parameters: {} weight matrices ({} per head × {} heads)",
                 mha.parameters().size(), 4u, H);

    // ── 4. Full attention forward + backward ──────────────────────────────────
    section("4. Forward and backward through attention");

    constexpr int64_t SEQ = 6;
    Tensor xt = randn({SEQ, static_cast<int64_t>(D)});
    Variable x(std::move(xt), true, "input");

    auto out = mha.forward(x, /*causal=*/true);
    std::println("Input:  ({},{})", x.data().shape()[0], x.data().shape()[1]);
    std::println("Output: ({},{})", out.data().shape()[0], out.data().shape()[1]);

    // Backward pass.
    sum(out).backward();

    std::println("Input grad norm: {:.4f}  (gradient flows to input embeddings)",
                 ops::norm(x.grad()));

    std::size_t params_with_grad = 0;
    for (auto* p : mha.parameters())
        if (p->grad().numel() > 0) ++params_with_grad;
    std::println("Params with gradient: {}/{}", params_with_grad, mha.parameters().size());

    // ── 5. Full pipeline: embeddings → attention ──────────────────────────────
    section("5. Full pipeline: token ids → embeddings → attention");

    constexpr int64_t VOCAB = 100, EMBED = 16, POS_MAX = 128;
    Embedding      tok_emb(VOCAB, EMBED, /*seed=*/1);
    LearnedPositionalEncoding pos_emb(POS_MAX, EMBED, /*seed=*/2);
    MultiHeadSelfAttention    attn_layer(EMBED, 4, /*seed=*/3);

    Tensor ids = zeros({SEQ}, DType::Int32);
    {
        auto s = ids.data_as<int32_t>();
        s[0]=5; s[1]=12; s[2]=3; s[3]=5; s[4]=77; s[5]=9;
    }

    auto tok_vec = tok_emb.forward(ids);        // (SEQ, EMBED)
    auto pos_vec = pos_emb.forward(SEQ);         // (SEQ, EMBED)
    auto emb     = add(tok_vec, pos_vec);        // (SEQ, EMBED)
    auto ctx     = attn_layer.forward(emb);      // (SEQ, EMBED)

    std::println("token_ids → tok_emb ({},{}) + pos_emb ({},{}) → attention ({},{})",
                 tok_vec.data().shape()[0], tok_vec.data().shape()[1],
                 pos_vec.data().shape()[0], pos_vec.data().shape()[1],
                 ctx.data().shape()[0], ctx.data().shape()[1]);

    // Backward through the entire pipeline.
    sum(ctx).backward();

    const bool tok_grad = tok_emb.weight().grad().numel() > 0;
    const bool pos_grad = pos_emb.weight().grad().numel() > 0;
    std::println("Token emb grad: {}  Positional emb grad: {}  (both trainable)",
                 tok_grad ? "yes" : "no", pos_grad ? "yes" : "no");

    // ── 6. What's next ─────────────────────────────────────────────────────────
    section("6. What's next — Ch08: GPT-2 Architecture");
    std::println(
        "A GPT-2 block adds:\n"
        "  1. Layer normalisation before attention and FFN\n"
        "  2. Residual connections: x = x + attn(norm(x))\n"
        "  3. Feed-forward network: FFN(x) = GELU(x @ W1 + b1) @ W2 + b2\n"
        "  4. Stacking N transformer blocks\n"
        "  5. Language model head: x_T @ W_emb^T → logits over vocabulary\n"
        "\n"
        "Ch08 assembles these into a trainable GPT-2-sized model."
    );

    return 0;
}
