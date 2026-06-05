#include "sub0llm/nn/gpt.hpp"

#include "sub0llm/core/ops.hpp"

#include <cmath>
#include <format>
#include <random>
#include <stdexcept>

namespace sub0llm::nn {

namespace {

autograd::Variable xavier_var(int64_t rows, int64_t cols,
                               std::mt19937_64& rng, std::string name) {
    const float a = std::sqrt(6.0f / static_cast<float>(rows + cols));
    std::uniform_real_distribution<float> dist(-a, a);
    Tensor w = zeros({rows, cols});
    for (auto& v : w.data_as<float>()) v = dist(rng);
    return autograd::Variable(std::move(w), true, std::move(name));
}

} // anonymous namespace

// ── Linear ────────────────────────────────────────────────────────────────────

Linear::Linear(int64_t in_features, int64_t out_features, std::uint64_t seed,
               bool alloc_weights) {
    if (in_features <= 0 || out_features <= 0)
        throw std::runtime_error(std::format(
            "Linear: in_features={} and out_features={} must be positive",
            in_features, out_features));
    in_features_  = in_features;
    out_features_ = out_features;
    b_ = autograd::Variable(zeros({out_features}), true, "linear.b");
    if (alloc_weights) {
        std::mt19937_64 rng(seed);
        W_ = xavier_var(out_features, in_features, rng, "linear.W");
    } else {
        W_ = autograd::Variable(zeros({1, 1}), false, "linear.W.elided");
    }
}

autograd::Variable Linear::forward(const autograd::Variable& x) const {
    using namespace autograd;
    auto y = bias_add(matmul(x, transpose2d(W_)), b_);
    if (lora_enabled())   // y += scaling · (x @ A) @ B
        y = add(y, scale(matmul(matmul(x, lora_A_), lora_B_), lora_scaling_));
    return y;
}

void Linear::quantize_weights() {
    const int64_t out = W_.data().shape()[0];
    const int64_t in  = W_.data().shape()[1];
    if (in % backend::cpu::QK8_0 != 0) return;   // Q8_0 needs in divisible by 32
    const int64_t nb = in / backend::cpu::QK8_0;
    wq8_.resize(static_cast<std::size_t>(out * nb));
    auto wd = W_.data().data_as<float>();
    for (int64_t r = 0; r < out; ++r)
        backend::cpu::quantize_row_q8_0(wd.data() + r * in,
                                        wq8_.data() + r * nb, in);
}

void Linear::free_f32_weights() {
    if (!q8_enabled()) return;            // f32 path still needed if not quantized
    W_ = autograd::Variable(zeros({1, 1}), false, "linear.W.freed");
}

Tensor Linear::apply_one(const Tensor& x) const {
    // Q8 fast path (Ch27): single-token inference, no LoRA. Quantize the activation
    // once, then int8 block dots against the resident Q8 weight. Uses cached dims so
    // it still works after free_f32_weights().
    const int64_t in_q = in_features_;
    if (q8_enabled() && !lora_enabled() &&
        static_cast<int64_t>(x.numel()) == in_q) {
        const int64_t out = out_features_;
        const int64_t nb  = in_q / backend::cpu::QK8_0;
        Tensor xc = x.reshape({1, in_q}).contiguous();
        thread_local std::vector<backend::cpu::BlockQ8_0> xq;
        xq.resize(static_cast<std::size_t>(nb));
        Tensor result({1, out}, DType::Float32);
        auto od = result.data_as<float>();
        backend::cpu::matvec_q8_0_q8_0(wq8_.data(), xc.data_as<float>().data(),
                                       od.data(), out, in_q, xq.data());
        auto bd = b_.data().data_as<float>();
        for (int64_t c = 0; c < out; ++c)
            od[static_cast<std::size_t>(c)] += bd[static_cast<std::size_t>(c)];
        return result;
    }

    // out = x @ Wᵀ, where W_ is (out, in).  Compute it as (W @ xᵀ)ᵀ so the large,
    // constant weight matrix is never transposed or copied per call — only the
    // tiny activation is.  Both matmuls route through the vectorised Eigen/BLAS
    // GEMV path (K = in ≥ 64).  A per-call W transpose would copy out×in floats
    // every token (e.g. 3072×1024 ≈ 3M) for a result that never changes.
    const int64_t in   = W_.data().shape()[1];
    const int64_t rows = static_cast<int64_t>(x.numel()) / in;
    Tensor xT  = x.reshape({rows, in}).transpose(0, 1).contiguous();  // (in, rows)
    Tensor out = ops::matmul(W_.data(), xT)        // (out, rows)
                   .transpose(0, 1).contiguous();  // (rows, out)
    // Add bias row-wise
    auto od = out.data_as<float>();
    auto bd = b_.data().data_as<float>();
    const int64_t N = static_cast<int64_t>(b_.data().numel());
    for (int64_t r = 0; r < rows; ++r)
        for (int64_t c = 0; c < N; ++c)
            od[static_cast<std::size_t>(r * N + c)] +=
                bd[static_cast<std::size_t>(c)];

    // LoRA term: out += scaling · (x @ A) @ B   (inference, pure Tensor path)
    if (lora_enabled()) {
        Tensor xr   = x.reshape({rows, in});
        Tensor lora = ops::matmul(ops::matmul(xr, lora_A_.data()), lora_B_.data());
        auto   ld   = lora.data_as<float>();
        for (std::size_t i = 0; i < static_cast<std::size_t>(out.numel()); ++i)
            od[i] += lora_scaling_ * ld[i];
    }
    return out;
}

void Linear::enable_lora(int64_t rank, float alpha, std::uint64_t seed) {
    if (rank <= 0) return;
    const int64_t out = W_.data().shape()[0];
    const int64_t in  = W_.data().shape()[1];
    std::mt19937_64 rng(seed);
    // A ~ N(0, 1/in); B = 0  → initial low-rank delta is exactly zero.
    Tensor a = zeros({in, rank});
    const float std_a = 1.0f / std::sqrt(static_cast<float>(in));
    std::normal_distribution<float> nd(0.0f, std_a);
    for (auto& v : a.data_as<float>()) v = nd(rng);
    lora_A_ = autograd::Variable(std::move(a), true, "linear.lora_A");
    lora_B_ = autograd::Variable(zeros({rank, out}), true, "linear.lora_B");
    lora_scaling_ = alpha / static_cast<float>(rank);
    // Freeze the base weight/bias — only the adapter trains.
    W_.set_requires_grad(false);
    b_.set_requires_grad(false);
}

std::vector<autograd::Variable*> Linear::parameters() {
    std::vector<autograd::Variable*> p{&W_, &b_};
    if (lora_enabled()) { p.push_back(&lora_A_); p.push_back(&lora_B_); }
    return p;
}

// ── LayerNorm ─────────────────────────────────────────────────────────────────

LayerNorm::LayerNorm(int64_t D, float eps) : eps_(eps) {
    if (D <= 0)
        throw std::runtime_error(std::format(
            "LayerNorm: D={} must be positive", D));
    Tensor wt = zeros({D});
    for (auto& v : wt.data_as<float>()) v = 1.0f;
    weight_ = autograd::Variable(std::move(wt), true, "ln.weight");
    bias_   = autograd::Variable(zeros({D}), true, "ln.bias");
}

autograd::Variable LayerNorm::forward(const autograd::Variable& x) const {
    return autograd::layer_norm(x, weight_, bias_, eps_);
}

std::vector<autograd::Variable*> LayerNorm::parameters() {
    return {&weight_, &bias_};
}

// ── FeedForward ───────────────────────────────────────────────────────────────

FeedForward::FeedForward(int64_t D, std::uint64_t seed)
    : fc1_(D, 4 * D, seed), fc2_(4 * D, D, seed + 1) {}

autograd::Variable FeedForward::forward(const autograd::Variable& x) const {
    return fc2_.forward(autograd::gelu(fc1_.forward(x)));
}

std::vector<autograd::Variable*> FeedForward::parameters() {
    auto p = fc1_.parameters();
    for (auto* v : fc2_.parameters()) p.push_back(v);
    return p;
}

// ── TransformerBlock ──────────────────────────────────────────────────────────

TransformerBlock::TransformerBlock(int64_t D, int64_t num_heads,
                                   std::uint64_t seed)
    : norm1_(D),
      attn_(static_cast<std::size_t>(D), static_cast<std::size_t>(num_heads),
            seed),
      norm2_(D),
      ffn_(D, seed + 1000) {}

autograd::Variable TransformerBlock::forward(const autograd::Variable& x) const {
    using namespace autograd;
    auto h = add(x, attn_.forward(norm1_.forward(x), /*causal=*/true));
    return add(h, ffn_.forward(norm2_.forward(h)));
}

std::vector<autograd::Variable*> TransformerBlock::parameters() {
    std::vector<autograd::Variable*> p;
    for (auto* v : norm1_.parameters()) p.push_back(v);
    for (auto* v : attn_.parameters())  p.push_back(v);
    for (auto* v : norm2_.parameters()) p.push_back(v);
    for (auto* v : ffn_.parameters())   p.push_back(v);
    return p;
}

// ── GPT ───────────────────────────────────────────────────────────────────────

GPT::GPT(int64_t vocab_size, int64_t embed_dim, int64_t num_heads,
         int64_t num_layers, int64_t max_seq_len, std::uint64_t seed)
    : tok_emb_(vocab_size, embed_dim, seed),
      pos_emb_(max_seq_len, embed_dim, seed + 1),
      ln_f_(embed_dim) {
    if (num_heads <= 0)
        throw std::runtime_error(std::format(
            "GPT: num_heads={} must be positive", num_heads));
    if (num_layers <= 0)
        throw std::runtime_error(std::format(
            "GPT: num_layers={} must be positive", num_layers));
    blocks_.reserve(static_cast<std::size_t>(num_layers));
    for (int64_t l = 0; l < num_layers; ++l)
        blocks_.emplace_back(embed_dim, num_heads,
                             seed + 2 + static_cast<std::uint64_t>(l) * 2000);
}

autograd::Variable GPT::forward(const Tensor& token_ids) const {
    using namespace autograd;
    if (token_ids.ndim() != 1)
        throw std::runtime_error(
            "GPT::forward: token_ids must be 1D (T,)");
    const int64_t T = token_ids.shape()[0];
    auto x = add(tok_emb_.forward(token_ids), pos_emb_.forward(T));
    for (const auto& block : blocks_)
        x = block.forward(x);
    x = ln_f_.forward(x);
    return matmul(x, transpose2d(tok_emb_.weight()));
}

std::vector<autograd::Variable*> GPT::parameters() {
    std::vector<autograd::Variable*> p;
    p.push_back(&tok_emb_.weight());
    p.push_back(&pos_emb_.weight());
    for (auto& block : blocks_)
        for (auto* v : block.parameters()) p.push_back(v);
    for (auto* v : ln_f_.parameters()) p.push_back(v);
    return p;
}

} // namespace sub0llm::nn
