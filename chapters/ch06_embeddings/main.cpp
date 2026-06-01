// Chapter 06 — Embeddings
//
// Before attention can operate, raw token IDs and positions must be converted
// to dense vectors.  This chapter covers the three layers that sit between the
// tokenizer and the transformer blocks:
//
//   token_ids  →  token embeddings (lookup table, learnable)
//              +  positional encoding (fixed sinusoidal OR learned)
//              →  input to first attention layer
//
// Topics covered:
//   1. Why embeddings?  Discrete IDs → continuous geometry
//   2. Embedding lookup (gather) and scatter-add backward
//   3. Sinusoidal positional encoding (Vaswani 2017)
//   4. Learned positional encoding
//   5. RoPE: Rotary Position Encoding (Su et al. 2021)
//   6. Full token + position pipeline with gradient flow
//   7. What's next: Ch07 Attention

#include <cmath>
#include <format>
#include <vector>

#include "sub0llm/autograd/ops.hpp"
#include "sub0llm/compat/print.hpp"
#include "sub0llm/core/ops.hpp"
#include "sub0llm/nn/embedding.hpp"
#include "sub0llm/nn/positional_encoding.hpp"
#include "sub0llm/tokenizer/bpe.hpp"
#include "sub0llm/version.hpp"

using namespace sub0llm;
using namespace sub0llm::autograd;
using namespace sub0llm::nn;

static void section(std::string_view title) {
    std::println("\n── {} ──", title);
}

// ─────────────────────────────────────────────────────────────────────────────
int main() {
    std::println("sub0llm v{} — Chapter 06: Embeddings", version_string);

    // ── 1. Why embeddings? ────────────────────────────────────────────────────
    section("1. Why embeddings?");
    std::println(
        "Token IDs are integers with no geometric meaning: 42 is not\n"
        "'closer' to 43 than to 100.  An embedding table E[V × D] maps\n"
        "each token to a D-dimensional vector where similar tokens end up\n"
        "close (e.g. 'king' − 'man' + 'woman' ≈ 'queen').\n"
        "\n"
        "GPT-2 uses V=50257 tokens, D=768 dims → 38M embedding params.\n"
        "Llama-3.1-8B uses V=128000, D=4096 → 524M embedding params (25%%)."
    );

    // ── 2. Embedding lookup and backward ─────────────────────────────────────
    section("2. Embedding lookup (gather + scatter-add backward)");

    constexpr int64_t VOCAB = 10;
    constexpr int64_t DIM   = 4;
    Embedding emb(VOCAB, DIM, /*seed=*/0);

    std::println("Weight matrix ({}×{}):", VOCAB, DIM);

    // Build a small token sequence: [3, 1, 4, 1, 5]
    Tensor ids = zeros({5}, DType::Int32);
    {
        auto s = ids.data_as<int32_t>();
        s[0]=3; s[1]=1; s[2]=4; s[3]=1; s[4]=5;
    }

    auto emb_out = emb.forward(ids);  // (5, 4)
    std::println("embed([3,1,4,1,5]) shape: ({},{})",
                 emb_out.data().shape()[0], emb_out.data().shape()[1]);

    // Backward: the same token '1' appears at positions 1 and 3.
    auto loss = sum(emb_out);
    loss.backward();

    // Token 1 gradient should be twice the sum of a single row upstream.
    const auto wg = emb.weight().grad().data_as<float>();
    const int64_t D = DIM;
    std::print("grad[token 1] = [");
    for (int64_t j = 0; j < D; ++j) std::print(" {:.2f}", wg[static_cast<std::size_t>(D + j)]);
    std::println(" ]  (2× upstream since token 1 appears twice)");

    // Token 0 (not in sequence) should have zero gradient.
    float tok0_grad_norm = 0.0f;
    for (int64_t j = 0; j < D; ++j) tok0_grad_norm += wg[static_cast<std::size_t>(j)] * wg[static_cast<std::size_t>(j)];
    std::println("grad[token 0] norm = {:.4f}  (expected 0 — unused token)", tok0_grad_norm);

    // ── 3. Sinusoidal positional encoding ─────────────────────────────────────
    section("3. Sinusoidal positional encoding");

    constexpr int64_t SEQ = 8;
    constexpr int64_t D6  = 8;
    const Tensor pe = sinusoidal_encoding(SEQ, D6);
    std::println("PE shape: (seq={}, dim={})", SEQ, D6);
    std::print("PE[0] = [");
    for (int64_t j = 0; j < D6; ++j)
        std::print(" {:.3f}", pe.data_as<float>()[static_cast<std::size_t>(j)]);
    std::println(" ]  (sin at pos 0 → all zeros for even dims)");
    std::print("PE[1] = [");
    for (int64_t j = 0; j < D6; ++j)
        std::print(" {:.3f}", pe.data_as<float>()[static_cast<std::size_t>(D6 + j)]);
    std::println(" ]");

    // Verify: row norms should all equal sqrt(D6/2) for sinusoidal.
    float min_norm = 1e9f, max_norm = -1e9f;
    const auto ped = pe.data_as<float>();
    for (int64_t pos = 0; pos < SEQ; ++pos) {
        float sq = 0.0f;
        for (int64_t j = 0; j < D6; ++j) sq += ped[static_cast<std::size_t>(pos * D6 + j)] * ped[static_cast<std::size_t>(pos * D6 + j)];
        min_norm = std::min(min_norm, std::sqrt(sq));
        max_norm = std::max(max_norm, std::sqrt(sq));
    }
    std::println("Row norms: min={:.3f} max={:.3f}  (all ≈ sqrt(dim/2)={:.3f})",
                 min_norm, max_norm, std::sqrt(static_cast<float>(D6) / 2.0f));

    // ── 4. Learned positional encoding ────────────────────────────────────────
    section("4. Learned positional encoding");

    LearnedPositionalEncoding lpe(/*max_seq_len=*/128, /*embed_dim=*/DIM, /*seed=*/1);
    auto pe_emb = lpe.forward(/*seq_len=*/5);   // (5, DIM)
    std::println("Learned PE shape: ({},{}) — trainable, initialised N(0, 1/√D)",
                 pe_emb.data().shape()[0], pe_emb.data().shape()[1]);

    // ── 5. RoPE ───────────────────────────────────────────────────────────────
    section("5. RoPE — Rotary Position Encoding");
    std::println(
        "RoPE rotates query/key pairs (2i, 2i+1) by angle pos/base^(2i/D).\n"
        "Key property: dot(q_rotated[pos_a], k_rotated[pos_b]) depends only\n"
        "on (pos_a − pos_b), not on the absolute positions → relative attention."
    );

    constexpr int64_t ROPE_T = 4, ROPE_D = 8;
    Tensor q = randn({ROPE_T, ROPE_D}, DType::Float32);
    Tensor k = randn({ROPE_T, ROPE_D}, DType::Float32);

    Tensor q_rot = apply_rope(q);
    Tensor k_rot = apply_rope(k);

    // Verify relative-position property: <q[i], k[j]> = <q_rot[i], k_rot[j]>
    // depends only on i-j.  Check that pairs with same offset have same inner product.
    auto dot = [&](const Tensor& a, const Tensor& b, int64_t row_a, int64_t row_b) {
        const auto as = a.data_as<float>();
        const auto bs = b.data_as<float>();
        float d = 0.0f;
        for (int64_t j = 0; j < ROPE_D; ++j)
            d += as[static_cast<std::size_t>(row_a * ROPE_D + j)] * bs[static_cast<std::size_t>(row_b * ROPE_D + j)];
        return d;
    };

    // Absolute positions change, but relative offset=1 pairs should be similar.
    std::println("Dot products (q_rot[i] · k_rot[j]):");
    for (int64_t i = 0; i < ROPE_T; ++i) {
        for (int64_t j = 0; j < ROPE_T; ++j)
            std::print(" {:7.3f}", dot(q_rot, k_rot, i, j));
        std::println("");
    }
    std::println("(Diagonal is self-attention; off-diagonal encodes relative distance)");

    // ── 6. Full pipeline: tokens → embeddings → add PE ────────────────────────
    section("6. Full embedding pipeline");

    constexpr int64_t VOCAB6 = 100, DIM6 = 16, SEQ6 = 6;
    Embedding      tok_emb(VOCAB6, DIM6, 42);
    LearnedPositionalEncoding pos_emb(512, DIM6, 99);

    // Simulate token ids from a BPE tokenizer
    Tensor token_ids = zeros({SEQ6}, DType::Int32);
    auto   ts        = token_ids.data_as<int32_t>();
    ts[0]=5; ts[1]=23; ts[2]=7; ts[3]=5; ts[4]=99; ts[5]=1;

    // Forward: x = tok_emb + pos_emb
    auto tok_vec = tok_emb.forward(token_ids);     // (SEQ6, DIM6)
    auto pos_vec = pos_emb.forward(SEQ6);           // (SEQ6, DIM6)
    auto x       = add(tok_vec, pos_vec);           // (SEQ6, DIM6)

    std::println("Input embeddings shape: ({},{}) = token + position",
                 x.data().shape()[0], x.data().shape()[1]);

    // Backward through both embedding tables.
    sum(x).backward();

    const bool tok_has_grad = tok_emb.weight().grad().numel() > 0;
    const bool pos_has_grad = pos_emb.weight().grad().numel() > 0;
    std::println("Token embedding grad: {}  Positional embedding grad: {}",
                 tok_has_grad ? "yes" : "no",
                 pos_has_grad ? "yes" : "no");

    // ── 7. What's next ─────────────────────────────────────────────────────────
    section("7. What's next — Ch07: Attention Mechanisms");
    std::println(
        "Input x (B, T, D) produced here feeds directly into:\n"
        "  Q = x @ W_Q   K = x @ W_K   V = x @ W_V\n"
        "  Attn = softmax(Q @ K^T / sqrt(D_h)) @ V\n"
        "  Output = Attn @ W_O + x  (residual connection)\n"
        "\n"
        "Ch07 implements multi-head self-attention, causal masking,\n"
        "FlashAttention-style chunked computation, and cross-attention."
    );

    return 0;
}
