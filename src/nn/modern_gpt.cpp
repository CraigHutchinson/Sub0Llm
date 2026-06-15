#include "sub0llm/nn/modern_gpt.hpp"

#include "sub0llm/autograd/ops.hpp"
#include "sub0llm/core/ops.hpp"

#include <cmath>
#include <format>
#include <limits>
#include <random>
#include <stdexcept>

namespace sub0llm::nn {

namespace {

int64_t swiglu_d_ff(int64_t D, int64_t d_ff) {
    if (d_ff > 0) return d_ff;
    const int64_t raw = (8 * D + 2) / 3;
    const int64_t r   = ((raw + 63) / 64) * 64;
    return r > 0 ? r : 64;
}

// Validates D>0 and returns D; used as first argument in SwiGLUFeedForward
// initializer list so the guard fires before any Linear is constructed.
int64_t swiglu_check_D(int64_t D) {
    if (D <= 0)
        throw std::runtime_error(
            std::format("SwiGLUFeedForward: D={} must be positive", D));
    return D;
}

// Validates vocab_size>0 and embed_dim>0; returns embed_dim for use in the
// ModernGPT initializer list before tok_emb_/ln_f_ are constructed.
int64_t mgpt_validate(int64_t vocab_size, int64_t embed_dim) {
    if (vocab_size <= 0)
        throw std::runtime_error(
            std::format("ModernGPT: vocab_size={} must be positive", vocab_size));
    if (embed_dim <= 0)
        throw std::runtime_error(
            std::format("ModernGPT: embed_dim={} must be positive", embed_dim));
    return embed_dim;
}

autograd::Variable xavier_var(int64_t rows, int64_t cols,
                               std::mt19937_64& rng, std::string name) {
    const float a = std::sqrt(6.0f / static_cast<float>(rows + cols));
    std::uniform_real_distribution<float> dist(-a, a);
    Tensor w = zeros({rows, cols});
    for (auto& v : w.data_as<float>()) v = dist(rng);
    return autograd::Variable(std::move(w), true, std::move(name));
}

// Precompute RoPE cos/sin frequency tables for T positions and head dim Dh.
// Returns tables of shape (T, Dh/2).
void compute_rope_freqs(int64_t T, int64_t Dh, float base,
                        Tensor& cos_out, Tensor& sin_out) {
    const int64_t D2 = Dh / 2;
    cos_out = zeros({T, D2});
    sin_out = zeros({T, D2});
    auto cd = cos_out.data_as<float>();
    auto sd = sin_out.data_as<float>();
    for (int64_t t = 0; t < T; ++t)
        for (int64_t i = 0; i < D2; ++i) {
            const float theta = std::pow(base, -2.0f * static_cast<float>(i) /
                                               static_cast<float>(Dh));
            const float angle = static_cast<float>(t) * theta;
            const auto ti = static_cast<std::size_t>(t * D2 + i);
            cd[ti] = std::cos(angle);
            sd[ti] = std::sin(angle);
        }
}

// Causal mask with optional sliding window.
// W = -1 → full causal attention; W >= 0 → also block j < i − W.
Tensor sliding_window_mask(int64_t T, int64_t W) {
    Tensor m = zeros({T, T});
    auto   md = m.data_as<float>();
    const float neg_inf = -std::numeric_limits<float>::infinity();
    const auto  Ts = static_cast<std::size_t>(T);
    for (std::size_t i = 0; i < Ts; ++i)
        for (std::size_t j = 0; j < Ts; ++j) {
            const auto ii = static_cast<int64_t>(i);
            const auto jj = static_cast<int64_t>(j);
            if (jj > ii || (W >= 0 && ii - jj > W))
                md[i * Ts + j] = neg_inf;
        }
    return m;
}

} // anonymous namespace

// ── RMSNorm ───────────────────────────────────────────────────────────────────

RMSNorm::RMSNorm(int64_t D, float eps) : eps_(eps) {
    if (D <= 0)
        throw std::runtime_error(std::format("RMSNorm: D={} must be positive", D));
    Tensor wt = zeros({D});
    for (auto& v : wt.data_as<float>()) v = 1.0f;
    weight_ = autograd::Variable(std::move(wt), true, "rmsnorm.weight");
}

autograd::Variable RMSNorm::forward(const autograd::Variable& x) const {
    return autograd::rms_norm(x, weight_, eps_);
}

std::vector<autograd::Variable*> RMSNorm::parameters() {
    return {&weight_};
}

int64_t RMSNorm::dim() const noexcept {
    return weight_.data().numel();
}

Tensor RMSNorm::apply_one(const Tensor& x) const {
    const int64_t D = weight_.data().numel();
    auto xd = x.data_as<float>();
    auto wd = weight_.data().data_as<float>();
    float ss = 0.0f;
    for (int64_t i = 0; i < D; ++i) {
        const float v = xd[static_cast<std::size_t>(i)];
        ss += v * v;
    }
    const float rms_inv = 1.0f / std::sqrt(ss / static_cast<float>(D) + eps_);
    Tensor out = zeros({1, D});
    auto od = out.data_as<float>();
    for (int64_t i = 0; i < D; ++i) {
        const auto idx = static_cast<std::size_t>(i);
        od[idx] = wd[idx] * xd[idx] * rms_inv;
    }
    return out;
}

// ── SwiGLUFeedForward ─────────────────────────────────────────────────────────

SwiGLUFeedForward::SwiGLUFeedForward(int64_t D, int64_t d_ff, std::uint64_t seed,
                                     bool alloc_weights)
    : gate_(swiglu_check_D(D), swiglu_d_ff(D, d_ff), seed,     alloc_weights),
      up_  (D, swiglu_d_ff(D, d_ff), seed + 1,                 alloc_weights),
      down_(swiglu_d_ff(D, d_ff), D, seed + 2,                 alloc_weights) {}

autograd::Variable SwiGLUFeedForward::forward(const autograd::Variable& x) const {
    using namespace autograd;
    auto gate_out = silu(gate_.forward(x));  // (T, d_ff)
    auto up_out   = up_.forward(x);          // (T, d_ff)
    return down_.forward(mul(gate_out, up_out));
}

Tensor SwiGLUFeedForward::apply_one(const Tensor& x) const {
    Tensor g = gate_.apply_one(x);             // (1, d_ff)
    Tensor u = up_.apply_one(x);               // (1, d_ff)
    Tensor gs = ops::mul(g, ops::sigmoid(g));  // SiLU: g * σ(g)
    return down_.apply_one(ops::mul(gs, u));   // (1, D)
}

void SwiGLUFeedForward::quantize_weights() {
    gate_.quantize_weights();
    up_.quantize_weights();
    down_.quantize_weights();
}

void SwiGLUFeedForward::free_f32_weights() {
    gate_.free_f32_weights();
    up_.free_f32_weights();
    down_.free_f32_weights();
}

std::vector<autograd::Variable*> SwiGLUFeedForward::parameters() {
    std::vector<autograd::Variable*> p;
    for (auto* v : gate_.parameters()) p.push_back(v);
    for (auto* v : up_.parameters())   p.push_back(v);
    for (auto* v : down_.parameters()) p.push_back(v);
    return p;
}

void SwiGLUFeedForward::enable_lora(int64_t rank, float alpha, std::uint64_t seed) {
    gate_.enable_lora(rank, alpha, seed);
    up_.enable_lora  (rank, alpha, seed + 1);
    down_.enable_lora(rank, alpha, seed + 2);
}

// ── GroupedQueryAttention ─────────────────────────────────────────────────────

GroupedQueryAttention::GroupedQueryAttention(std::size_t embed_dim,
                                             std::size_t n_heads,
                                             std::size_t n_kv_heads,
                                             float       rope_base,
                                             std::uint64_t seed,
                                             int64_t     window_size,
                                             std::size_t head_dim,
                                             bool        use_qk_norm,
                                             bool        alloc_weights)
    : embed_dim_(embed_dim), n_heads_(n_heads),
      n_kv_heads_(n_kv_heads), rope_base_(rope_base),
      window_size_(window_size), use_qk_norm_(use_qk_norm) {
    if (embed_dim == 0 || n_heads == 0 || n_kv_heads == 0)
        throw std::runtime_error(std::format(
            "GroupedQueryAttention: embed_dim={}, n_heads={}, n_kv_heads={} "
            "must all be positive", embed_dim, n_heads, n_kv_heads));
    if (head_dim == 0) {
        // Derive head_dim from embed_dim / n_heads — standard LLaMA/Qwen2 path.
        if (embed_dim % n_heads != 0)
            throw std::runtime_error(std::format(
                "GroupedQueryAttention: embed_dim={} must be divisible by n_heads={}",
                embed_dim, n_heads));
        head_dim_ = embed_dim / n_heads;
    } else {
        // Explicit head_dim (Qwen3-4B+): head_dim != embed_dim / n_heads is allowed.
        head_dim_ = head_dim;
    }
    if (n_heads % n_kv_heads != 0)
        throw std::runtime_error(std::format(
            "GroupedQueryAttention: n_heads={} must be divisible by n_kv_heads={}",
            n_heads, n_kv_heads));
    if (head_dim_ % 2 != 0)
        throw std::runtime_error(std::format(
            "GroupedQueryAttention: head_dim={} must be even for RoPE", head_dim_));

    const int64_t D  = static_cast<int64_t>(embed_dim_);
    const int64_t Dh = static_cast<int64_t>(head_dim_);
    std::mt19937_64 rng(seed);

    // alloc_weights=false: placeholders only — the Q8 loader fills wq_q8_ directly.
    auto make = [&](int64_t r, int64_t c, const std::string& nm) {
        return alloc_weights ? xavier_var(r, c, rng, nm)
                             : autograd::Variable(zeros({1, 1}), false, nm + ".elided");
    };
    W_Q_.reserve(n_heads_);
    W_O_.reserve(n_heads_);
    for (std::size_t h = 0; h < n_heads_; ++h) {
        W_Q_.push_back(make(D, Dh, std::format("gqa.W_Q.{}", h)));
        W_O_.push_back(make(Dh, D, std::format("gqa.W_O.{}", h)));
    }
    W_K_.reserve(n_kv_heads_);
    W_V_.reserve(n_kv_heads_);
    for (std::size_t g = 0; g < n_kv_heads_; ++g) {
        W_K_.push_back(make(D, Dh, std::format("gqa.W_K.{}", g)));
        W_V_.push_back(make(D, Dh, std::format("gqa.W_V.{}", g)));
    }

    // Qwen3 QK-norm: one RMSNorm(head_dim) each for Q and K, shared across heads.
    if (use_qk_norm_) {
        q_norm_.emplace(Dh);
        k_norm_.emplace(Dh);
    }
}

autograd::Variable GroupedQueryAttention::forward(const autograd::Variable& x,
                                                   bool causal) const {
    using namespace autograd;
    if (x.data().ndim() != 2)
        throw std::runtime_error(
            "GroupedQueryAttention::forward: x must be 2D (T, embed_dim)");
    const int64_t T = x.data().shape()[0];
    if (T < 1)
        throw std::runtime_error(std::format(
            "GroupedQueryAttention::forward: T={} must be >= 1", T));

    const float    scale_fac = 1.0f / std::sqrt(static_cast<float>(head_dim_));
    const std::size_t r    = n_heads_ / n_kv_heads_;  // heads per KV group

    // Precompute RoPE frequencies — cached by T to avoid per-step trig ops.
    if (T != rope_cache_T_) {
        compute_rope_freqs(T, static_cast<int64_t>(head_dim_), rope_base_,
                           rope_cache_cos_, rope_cache_sin_);
        rope_cache_T_ = T;
    }
    const Tensor& cos_f = rope_cache_cos_;
    const Tensor& sin_f = rope_cache_sin_;

    // Precompute K and V for each KV head (shared across query groups).
    std::vector<Variable> K_heads(n_kv_heads_), V_heads(n_kv_heads_);
    for (std::size_t g = 0; g < n_kv_heads_; ++g) {
        auto k_raw = matmul(x, W_K_[g]);          // (T, head_dim)
        auto v_raw = matmul(x, W_V_[g]);          // (T, head_dim)
        if (use_qk_norm_) k_raw = k_norm_->forward(k_raw);  // RMSNorm over head_dim
        K_heads[g] = rope(k_raw, cos_f, sin_f);
        V_heads[g] = v_raw;
    }

    // Build causal/sliding-window mask once per T — cached.
    Variable mask_var;
    if (causal) {
        if (T != mask_cache_T_) {
            mask_cache_   = sliding_window_mask(T, window_size_);
            mask_cache_T_ = T;
        }
        mask_var = Variable(mask_cache_, false);
    }

    // Accumulate output across all query heads.
    auto compute_head = [&](std::size_t h) -> Variable {
        const std::size_t g = h / r;
        auto q_raw = matmul(x, W_Q_[h]);                        // (T, head_dim)
        if (use_qk_norm_) q_raw = q_norm_->forward(q_raw);      // RMSNorm over head_dim
        auto q     = rope(q_raw, cos_f, sin_f);
        auto scores = scale(matmul(q, transpose2d(K_heads[g])), scale_fac); // (T,T)
        if (causal) scores = add(scores, mask_var);
        auto attn   = softmax(scores);                           // (T, T)
        auto ctx    = matmul(attn, V_heads[g]);                  // (T, head_dim)
        return matmul(ctx, W_O_[h]);                             // (T, D)
    };

    Variable out = compute_head(0);
    for (std::size_t h = 1; h < n_heads_; ++h)
        out = add(out, compute_head(h));
    return out;
}

// Batched bidirectional forward: x is (B·T, D), B windows stacked on the leading
// dim. Projections / RoPE / QK-norm / softmax run once over all B·T rows; only the
// score and context matmuls reshape to (B,T,·) so attention is block-diagonal.
autograd::Variable GroupedQueryAttention::forward(const autograd::Variable& x,
                                                  int64_t B, int64_t T, bool causal) const {
    using namespace autograd;
    if (causal)
        throw std::runtime_error(
            "GroupedQueryAttention batched forward: causal masking is future work (bidirectional only)");
    if (x.data().ndim() != 2 || x.data().shape()[0] != B * T ||
        x.data().shape()[1] != static_cast<int64_t>(embed_dim_))
        throw std::runtime_error(std::format(
            "GroupedQueryAttention batched forward: x must be (B*T, D) = ({}, {})",
            B * T, embed_dim_));
    if (B < 1 || T < 1)
        throw std::runtime_error("GroupedQueryAttention batched forward: B,T must be >= 1");

    const float       scale_fac = 1.0f / std::sqrt(static_cast<float>(head_dim_));
    const std::size_t r  = n_heads_ / n_kv_heads_;
    const int64_t     Dh = static_cast<int64_t>(head_dim_);
    const int64_t     D2 = Dh / 2;

    // Base (T) RoPE table (cached by T), then tile to (B·T) rows (cached by B,T).
    if (T != rope_cache_T_) {
        compute_rope_freqs(T, Dh, rope_base_, rope_cache_cos_, rope_cache_sin_);
        rope_cache_T_ = T;
        rope_btile_B_ = -1;                      // invalidate the tiled table
    }
    if (B != rope_btile_B_ || T != rope_btile_T_) {
        auto tile = [&](const Tensor& base) {
            Tensor out = zeros({B * T, D2});
            auto bd = base.data_as<float>();
            auto od = out.data_as<float>();
            for (int64_t b = 0; b < B; ++b)
                std::copy_n(bd.begin(), T * D2, od.begin() + b * T * D2);
            return out;
        };
        rope_btile_cos_ = tile(rope_cache_cos_);
        rope_btile_sin_ = tile(rope_cache_sin_);
        rope_btile_B_   = B;
        rope_btile_T_   = T;
    }
    const Tensor& cos_f = rope_btile_cos_;
    const Tensor& sin_f = rope_btile_sin_;

    // K, V per KV head: (B·T,D)→(B·T,hd), QK-norm+RoPE on K, then fold to (B,T,hd).
    std::vector<Variable> K_heads(n_kv_heads_), V_heads(n_kv_heads_);
    for (std::size_t g = 0; g < n_kv_heads_; ++g) {
        auto k_raw = matmul(x, W_K_[g]);                       // (B·T, hd)
        if (use_qk_norm_) k_raw = k_norm_->forward(k_raw);
        K_heads[g] = reshape(rope(k_raw, cos_f, sin_f), {B, T, Dh});
        V_heads[g] = reshape(matmul(x, W_V_[g]), {B, T, Dh});
    }

    auto compute_head = [&](std::size_t h) -> Variable {
        const std::size_t g = h / r;
        auto q_raw = matmul(x, W_Q_[h]);                       // (B·T, hd)
        if (use_qk_norm_) q_raw = q_norm_->forward(q_raw);
        auto q3     = reshape(rope(q_raw, cos_f, sin_f), {B, T, Dh});  // (B,T,hd)
        auto scores = scale(matmul_bt(q3, K_heads[g]), scale_fac);    // (B,T,T) batched q·Kᵀ
        auto attn2  = softmax(reshape(scores, {B * T, T}));          // row-softmax over T
        auto ctx    = matmul(reshape(attn2, {B, T, T}), V_heads[g]); // (B,T,hd) batched
        return matmul(reshape(ctx, {B * T, Dh}), W_O_[h]);           // (B·T, D)
    };

    Variable out = compute_head(0);
    for (std::size_t h = 1; h < n_heads_; ++h)
        out = add(out, compute_head(h));
    return out;
}

void GroupedQueryAttention::quantize_weights() {
    const int64_t D  = static_cast<int64_t>(embed_dim_);
    const int64_t Dh = static_cast<int64_t>(head_dim_);
    if (D % backend::cpu::QK8_0 != 0 || Dh % backend::cpu::QK8_0 != 0) return;

    // W is (in, out) row-major; emit Q8 of its transpose (out, in) row-major so each
    // output row is `in` contiguous values — what matvec_q8_0_q8_0 expects.
    auto quant_T = [](const Tensor& W, int64_t in, int64_t out,
                      std::vector<backend::cpu::BlockQ8_0>& dst) {
        const int64_t nb = in / backend::cpu::QK8_0;
        dst.resize(static_cast<std::size_t>(out * nb));
        auto wd = W.data_as<float>();
        std::vector<float> row(static_cast<std::size_t>(in));
        for (int64_t o = 0; o < out; ++o) {
            for (int64_t i = 0; i < in; ++i)
                row[static_cast<std::size_t>(i)] = wd[static_cast<std::size_t>(i * out + o)];
            backend::cpu::quantize_row_q8_0(row.data(), dst.data() + o * nb, in);
        }
    };

    wq_q8_.resize(n_heads_);    wo_q8_.resize(n_heads_);
    wk_q8_.resize(n_kv_heads_); wv_q8_.resize(n_kv_heads_);
    for (std::size_t h = 0; h < n_heads_; ++h) {
        quant_T(W_Q_[h].data(), D, Dh, wq_q8_[h]);   // (D,Dh) → (Dh,D)
        quant_T(W_O_[h].data(), Dh, D, wo_q8_[h]);   // (Dh,D) → (D,Dh)
    }
    for (std::size_t g = 0; g < n_kv_heads_; ++g) {
        quant_T(W_K_[g].data(), D, Dh, wk_q8_[g]);
        quant_T(W_V_[g].data(), D, Dh, wv_q8_[g]);
    }
    q8_ = true;
}

void GroupedQueryAttention::free_f32_weights() {
    if (!q8_) return;
    for (auto& w : W_Q_) w = autograd::Variable(zeros({1, 1}), false);
    for (auto& w : W_K_) w = autograd::Variable(zeros({1, 1}), false);
    for (auto& w : W_V_) w = autograd::Variable(zeros({1, 1}), false);
    for (auto& w : W_O_) w = autograd::Variable(zeros({1, 1}), false);
}

// int8 matvec helper: y(1,M) = Wq(M,K row-major Q8) · x(1,K).
static Tensor q8_matvec(const std::vector<backend::cpu::BlockQ8_0>& Wq,
                        const Tensor& x, int64_t M, int64_t K) {
    Tensor xc = x.reshape({1, K}).contiguous();
    thread_local std::vector<backend::cpu::BlockQ8_0> xq;
    xq.resize(static_cast<std::size_t>(K / backend::cpu::QK8_0));
    Tensor y = zeros({1, M});
    backend::cpu::matvec_q8_0_q8_0(Wq.data(), xc.data_as<float>().data(),
                                   y.data_as<float>().data(), M, K, xq.data());
    return y;
}

Tensor GroupedQueryAttention::forward_one(const Tensor& x_new,
                                           int64_t       pos,
                                           KVCache&      cache,
                                           std::size_t   layer_idx,
                                           float         rope_base_override) const {
    const float base    = (rope_base_override > 0.0f) ? rope_base_override : rope_base_;
    const int64_t Dh    = static_cast<int64_t>(head_dim_);
    const int64_t D2    = Dh / 2;
    const int64_t D     = static_cast<int64_t>(embed_dim_);
    const float scale   = 1.0f / std::sqrt(static_cast<float>(head_dim_));
    const std::size_t r = n_heads_ / n_kv_heads_;

    // RoPE cos/sin for single position `pos`
    std::vector<float> cos_p(static_cast<std::size_t>(D2));
    std::vector<float> sin_p(static_cast<std::size_t>(D2));
    for (int64_t i = 0; i < D2; ++i) {
        const float theta = std::pow(base,
            -2.0f * static_cast<float>(i) / static_cast<float>(Dh));
        const float angle = static_cast<float>(pos) * theta;
        cos_p[static_cast<std::size_t>(i)] = std::cos(angle);
        sin_p[static_cast<std::size_t>(i)] = std::sin(angle);
    }

    // Project and write K, V into cache for each KV head
    for (std::size_t g = 0; g < n_kv_heads_; ++g) {
        // W_K_[g]: (D, Dh) — x_new @ W_K_[g] = (1,D)@(D,Dh) → (1,Dh)
        Tensor k_new = q8_ ? q8_matvec(wk_q8_[g], x_new, Dh, D)
                           : ops::matmul(x_new, W_K_[g].data());
        Tensor v_new = q8_ ? q8_matvec(wv_q8_[g], x_new, Dh, D)
                           : ops::matmul(x_new, W_V_[g].data());
        // Qwen3 QK-norm: RMSNorm over head_dim before RoPE.
        if (use_qk_norm_) k_new = k_norm_->apply_one(k_new);
        // Apply RoPE to k in-place
        auto kd = k_new.data_as<float>();
        for (int64_t i = 0; i < D2; ++i) {
            const auto si = static_cast<std::size_t>(i);
            const auto si2 = static_cast<std::size_t>(i + D2);
            const float k0 = kd[si], k1 = kd[si2];
            kd[si]  = k0 * cos_p[si] - k1 * sin_p[si];
            kd[si2] = k1 * cos_p[si] + k0 * sin_p[si];
        }
        // Write row `pos` of the cache buffers
        auto kb = cache.K[layer_idx][g].data_as<float>();
        auto vb = cache.V[layer_idx][g].data_as<float>();
        auto knd = k_new.data_as<float>();
        auto vnd = v_new.data_as<float>();
        const std::size_t off = static_cast<std::size_t>(pos * Dh);
        for (std::size_t i = 0; i < static_cast<std::size_t>(Dh); ++i) {
            kb[off + i] = knd[i];
            vb[off + i] = vnd[i];
        }
    }

    // Attend over [0, pos] — the filled rows of the KV cache
    const int64_t ctx_len = pos + 1;
    Tensor out = zeros({1, D});
    auto od = out.data_as<float>();

    for (std::size_t h = 0; h < n_heads_; ++h) {
        const std::size_t g = h / r;

        // Q for this head
        Tensor q = q8_ ? q8_matvec(wq_q8_[h], x_new, Dh, D)
                       : ops::matmul(x_new, W_Q_[h].data());  // (1, Dh)
        if (use_qk_norm_) q = q_norm_->apply_one(q);     // QK-norm before RoPE
        auto qd = q.data_as<float>();
        for (int64_t i = 0; i < D2; ++i) {
            const auto si = static_cast<std::size_t>(i);
            const auto si2 = static_cast<std::size_t>(i + D2);
            const float q0 = qd[si], q1 = qd[si2];
            qd[si]  = q0 * cos_p[si] - q1 * sin_p[si];
            qd[si2] = q1 * cos_p[si] + q0 * sin_p[si];
        }

        // Scores: q @ K_cache[:ctx_len]^T  → (1, ctx_len)
        Tensor K_slice = ops::narrow(cache.K[layer_idx][g], 0, ctx_len);
        Tensor K_T     = K_slice.transpose(0, 1).contiguous();  // (Dh, ctx_len)
        Tensor scores  = ops::mul(ops::matmul(q, K_T), scale);

        // Apply sliding-window mask to scores
        if (window_size_ >= 0) {
            auto sd = scores.data_as<float>();
            const float neg_inf = -std::numeric_limits<float>::infinity();
            for (int64_t j = 0; j < ctx_len; ++j)
                if (pos - j > window_size_)
                    sd[static_cast<std::size_t>(j)] = neg_inf;
        }

        Tensor attn = ops::softmax(scores, -1);               // (1, ctx_len)
        Tensor V_slice = ops::narrow(cache.V[layer_idx][g], 0, ctx_len);
        Tensor ctx   = ops::matmul(attn, V_slice);            // (1, Dh)
        // W_O_[h]: (Dh, D) — ctx @ W_O_[h] = (1,Dh)@(Dh,D) → (1,D)
        Tensor head_out = q8_ ? q8_matvec(wo_q8_[h], ctx, D, Dh)
                              : ops::matmul(ctx, W_O_[h].data());

        auto hod = head_out.data_as<float>();
        for (std::size_t i = 0; i < static_cast<std::size_t>(D); ++i)
            od[i] += hod[i];
    }

    return out;
}

std::vector<autograd::Variable*> GroupedQueryAttention::parameters() {
    std::vector<autograd::Variable*> params;
    params.reserve(n_heads_ * 2 + n_kv_heads_ * 2);
    for (std::size_t h = 0; h < n_heads_; ++h) {
        params.push_back(&W_Q_[h]);
        params.push_back(&W_O_[h]);
    }
    for (std::size_t g = 0; g < n_kv_heads_; ++g) {
        params.push_back(&W_K_[g]);
        params.push_back(&W_V_[g]);
    }
    // QK-norm weights last, so non-QK-norm models keep their original layout.
    if (use_qk_norm_) {
        for (auto* p : q_norm_->parameters()) params.push_back(p);
        for (auto* p : k_norm_->parameters()) params.push_back(p);
    }
    return params;
}

// ── ModernTransformerBlock ────────────────────────────────────────────────────

ModernTransformerBlock::ModernTransformerBlock(int64_t     D,
                                               std::size_t n_heads,
                                               std::size_t n_kv_heads,
                                               int64_t     d_ff,
                                               std::uint64_t seed,
                                               int64_t     window_size,
                                               std::size_t head_dim,
                                               bool        use_qk_norm,
                                               bool        alloc_weights)
    : norm1_(D),
      attn_(static_cast<std::size_t>(D), n_heads, n_kv_heads, 10000.0f, seed,
            window_size, head_dim, use_qk_norm, alloc_weights),
      norm2_(D),
      ffn_(D, d_ff, seed + 1000, alloc_weights) {}

autograd::Variable ModernTransformerBlock::forward(const autograd::Variable& x) const {
    using namespace autograd;
    auto h = add(x, attn_.forward(norm1_.forward(x), /*causal=*/true));
    return add(h, ffn_.forward(norm2_.forward(h)));
}

Tensor ModernTransformerBlock::forward_one(const Tensor& x,
                                            int64_t       pos,
                                            KVCache&      cache,
                                            std::size_t   layer_idx,
                                            float         rope_base_override) const {
    Tensor h_norm = norm1_.apply_one(x);
    Tensor h_attn = attn_.forward_one(h_norm, pos, cache, layer_idx, rope_base_override);
    Tensor h      = ops::add(x, h_attn);
    Tensor h_norm2 = norm2_.apply_one(h);
    return ops::add(h, ffn_.apply_one(h_norm2));
}

std::vector<autograd::Variable*> ModernTransformerBlock::parameters() {
    std::vector<autograd::Variable*> p;
    for (auto* v : norm1_.parameters()) p.push_back(v);
    for (auto* v : attn_.parameters())  p.push_back(v);
    for (auto* v : norm2_.parameters()) p.push_back(v);
    for (auto* v : ffn_.parameters())   p.push_back(v);
    return p;
}

void ModernTransformerBlock::enable_lora(int64_t rank, float alpha, std::uint64_t seed) {
    ffn_.enable_lora(rank, alpha, seed);
}

void ModernTransformerBlock::quantize_weights() {
    attn_.quantize_weights();
    ffn_.quantize_weights();
}

void ModernTransformerBlock::free_f32_weights() {
    attn_.free_f32_weights();
    ffn_.free_f32_weights();
}

// ── ModernGPT ─────────────────────────────────────────────────────────────────

ModernGPT::ModernGPT(int64_t     vocab_size,
                     int64_t     embed_dim,
                     std::size_t n_heads,
                     std::size_t n_kv_heads,
                     int64_t     n_layers,
                     int64_t     d_ff,
                     int64_t     n_mtp,
                     std::uint64_t seed,
                     int64_t     window_size,
                     std::size_t head_dim,
                     bool        use_qk_norm,
                     bool        alloc_weights)
    : tok_emb_(vocab_size, mgpt_validate(vocab_size, embed_dim), seed, alloc_weights),
      ln_f_(embed_dim) {
    if (n_heads == 0)
        throw std::runtime_error(std::format(
            "ModernGPT: n_heads={} must be positive", n_heads));
    if (n_kv_heads == 0 || n_kv_heads > n_heads)
        throw std::runtime_error(std::format(
            "ModernGPT: n_kv_heads={} must be in [1, n_heads={}]",
            n_kv_heads, n_heads));
    if (n_layers <= 0)
        throw std::runtime_error(std::format(
            "ModernGPT: n_layers={} must be positive", n_layers));
    if (n_mtp < 0)
        throw std::runtime_error(std::format(
            "ModernGPT: n_mtp_heads={} must be >= 0", n_mtp));

    n_kv_heads_ = n_kv_heads;
    head_dim_   = head_dim > 0 ? head_dim
                               : static_cast<std::size_t>(embed_dim) / n_heads;
    cached_vocab_ = vocab_size;   // survive a tok_emb f32 free in pure-inference mode
    cached_embed_ = embed_dim;

    blocks_.reserve(static_cast<std::size_t>(n_layers));
    for (int64_t l = 0; l < n_layers; ++l)
        blocks_.emplace_back(embed_dim, n_heads, n_kv_heads, d_ff,
                             seed + 2 + static_cast<std::uint64_t>(l) * 2000,
                             window_size, head_dim, use_qk_norm, alloc_weights);

    mtp_heads_.reserve(static_cast<std::size_t>(n_mtp));
    for (int64_t k = 0; k < n_mtp; ++k)
        mtp_heads_.emplace_back(embed_dim, vocab_size,
                                seed + 2 + static_cast<std::uint64_t>(n_layers) * 2000
                                         + static_cast<std::uint64_t>(k));
}

// Tied LM head: logits (T,V) = x (T,D) · Wᵀ, where W = token embedding (V,D).
// Computed as (W · xᵀ)ᵀ so the V×D (≈152k×1024) weight is never transposed —
// transpose2d(W) would copy ~155M floats every forward AND again in backward.
// Here only the small activation xᵀ and the (T,V) result are transposed; autograd
// flows gradients to both W and x through the standard matmul/transpose2d backward.
static autograd::Variable tied_lm_head(const autograd::Variable& weight,
                                       const autograd::Variable& x) {
    return autograd::transpose2d(autograd::matmul(weight, autograd::transpose2d(x)));
}

autograd::Variable ModernGPT::forward(const Tensor& token_ids) const {
    using namespace autograd;
    if (token_ids.ndim() != 1)
        throw std::runtime_error("ModernGPT::forward: token_ids must be 1D (T,)");
    if (token_ids.shape()[0] < 1)
        throw std::runtime_error("ModernGPT::forward: token_ids must be non-empty");
    auto x = tok_emb_.forward(token_ids);
    for (const auto& block : blocks_) x = block.forward(x);
    x = ln_f_.forward(x);
    return tied_lm_head(tok_emb_.weight(), x);
}

std::vector<autograd::Variable> ModernGPT::forward_mtp(const Tensor& token_ids) const {
    using namespace autograd;
    if (token_ids.ndim() != 1)
        throw std::runtime_error("ModernGPT::forward_mtp: token_ids must be 1D (T,)");
    if (token_ids.shape()[0] < 1)
        throw std::runtime_error("ModernGPT::forward_mtp: token_ids must be non-empty");
    auto x = tok_emb_.forward(token_ids);
    for (const auto& block : blocks_) x = block.forward(x);
    x = ln_f_.forward(x);

    std::vector<Variable> logits;
    logits.reserve(1 + mtp_heads_.size());
    // Head 0: weight-tied main LM head
    logits.push_back(tied_lm_head(tok_emb_.weight(), x));
    // Heads 1…k: additional MTP heads
    for (const auto& head : mtp_heads_)
        logits.push_back(head.forward(x));
    return logits;
}

std::vector<autograd::Variable*> ModernGPT::parameters() {
    std::vector<autograd::Variable*> p;
    p.push_back(&tok_emb_.weight());
    for (auto& block : blocks_)
        for (auto* v : block.parameters()) p.push_back(v);
    for (auto* v : ln_f_.parameters()) p.push_back(v);
    for (auto& head : mtp_heads_)
        for (auto* v : head.parameters()) p.push_back(v);
    return p;
}

void ModernGPT::set_trainable_last_layers(int n) {
    const int L = static_cast<int>(blocks_.size());
    if (n < 0)  n = (L + 1) / 2;   // default: last half (ceil)
    if (n == 0) n = L;             // 0 = full model
    n = std::clamp(n, 1, L);
    const bool partial = (n < L);

    // Embedding (and the tied LM head it backs) is frozen only for partial
    // updates; full-model mode keeps everything trainable.
    tok_emb_.weight().set_requires_grad(!partial);
    for (int b = 0; b < L; ++b) {
        const bool train = (b >= L - n);
        for (auto* v : blocks_[static_cast<std::size_t>(b)].parameters())
            v->set_requires_grad(train);
    }
    // The final norm and any MTP heads are always trainable (they sit at the top).
    for (auto* v : ln_f_.parameters()) v->set_requires_grad(true);
    for (auto& head : mtp_heads_)
        for (auto* v : head.parameters()) v->set_requires_grad(true);
}

void ModernGPT::enable_episodic_lora(int last_layers, int64_t rank, float alpha) {
    // Freeze every existing base parameter; adapters added below are trainable.
    for (auto* p : parameters()) p->set_requires_grad(false);

    // Default scaling = alpha/rank = 2 (alpha = 2·rank): alpha=1 would damp the
    // adapter to 1/rank and it barely learns. Caller may override alpha (>0).
    const float eff_alpha = (alpha > 0.0f) ? alpha : 2.0f * static_cast<float>(rank);

    const int L = static_cast<int>(blocks_.size());
    int n = last_layers;
    if (n < 0)  n = (L + 1) / 2;   // default: last half
    if (n == 0) n = L;             // 0 = all blocks
    n = std::clamp(n, 1, L);

    for (int b = L - n; b < L; ++b)
        blocks_[static_cast<std::size_t>(b)].enable_lora(
            rank, eff_alpha, 9000u + static_cast<std::uint64_t>(b) * 10u);
}

KVCache ModernGPT::make_kv_cache(int64_t max_seq) const {
    KVCache cache;
    cache.max_seq = max_seq;
    cache.filled  = 0;
    const std::size_t nl = blocks_.size();
    const int64_t     Dh = static_cast<int64_t>(head_dim_);
    cache.K.resize(nl);
    cache.V.resize(nl);
    for (std::size_t l = 0; l < nl; ++l) {
        cache.K[l].resize(n_kv_heads_);
        cache.V[l].resize(n_kv_heads_);
        for (std::size_t g = 0; g < n_kv_heads_; ++g) {
            cache.K[l][g] = zeros({max_seq, Dh});
            cache.V[l][g] = zeros({max_seq, Dh});
        }
    }
    return cache;
}

void ModernGPT::quantize_for_inference(bool drop_f32) {
    for (auto& b : blocks_) b.quantize_weights();        // attention + FFN projections
    const int64_t V = tok_emb_.vocab_size();
    const int64_t D = tok_emb_.embed_dim();
    if (D % backend::cpu::QK8_0 == 0) {                   // LM head
        const int64_t nb = D / backend::cpu::QK8_0;
        lm_head_q8_.resize(static_cast<std::size_t>(V * nb));
        auto wd = tok_emb_.weight().data().data_as<float>();
        for (int64_t r = 0; r < V; ++r)
            backend::cpu::quantize_row_q8_0(wd.data() + r * D,
                                            lm_head_q8_.data() + r * nb, D);
    }
    if (drop_f32) {
        for (auto& b : blocks_) b.free_f32_weights();    // reclaim attn+FFN f32 RAM
        if (!lm_head_q8_.empty()) {
            // Free the f32 (tied) embedding table too — the embedding lookup now
            // dequantizes its row from lm_head_q8_. Reclaims V·D·4 bytes (the single
            // biggest f32 tensor), completing the ~3.76× weight-RAM reduction.
            emb_from_q8_ = true;
            tok_emb_.weight() = autograd::Variable(zeros({1, 1}), false);
        }
    }
}

Tensor ModernGPT::forward_one(int32_t token_id, KVCache& cache,
                               float rope_scaling) const {
    if (cache.full())
        throw std::runtime_error("ModernGPT::forward_one: KV cache is full");

    const int64_t D  = cached_embed_;
    const int64_t V  = cached_vocab_;
    const int64_t pos = cache.filled;

    // Embedding lookup — one row of the (tied) embedding table.
    Tensor x = zeros({1, D});
    auto xd  = x.data_as<float>();
    const std::size_t tok = static_cast<std::size_t>(token_id);
    if (emb_from_q8_) {
        // f32 table was freed: dequantize the row from the Q8 LM head (same tensor).
        const std::size_t nb = static_cast<std::size_t>(D / backend::cpu::QK8_0);
        backend::cpu::dequantize_row_q8_0(lm_head_q8_.data() + tok * nb, xd.data(), D);
    } else {
        auto wd  = tok_emb_.weight().data().data_as<float>();
        const std::size_t Dsz = static_cast<std::size_t>(D);
        for (std::size_t i = 0; i < Dsz; ++i)
            xd[i] = wd[tok * Dsz + i];
    }

    // NTK-aware RoPE base override for context extension
    float rope_base_override = 0.0f;
    if (rope_scaling > 1.0f) {
        const float Dh = static_cast<float>(head_dim_);
        rope_base_override = 10000.0f * std::pow(rope_scaling, Dh / (Dh - 2.0f));
    }

    // Forward through transformer blocks with KV cache
    for (std::size_t l = 0; l < blocks_.size(); ++l)
        x = blocks_[l].forward_one(x, pos, cache, l, rope_base_override);

    // Final RMSNorm
    x = ln_f_.apply_one(x);

    // LM head: weight-tied projection against the embedding table.
    // W_emb is (V, D) row-major, so logits = W_emb @ xᵀ.
    Tensor logits;
    if (!lm_head_q8_.empty()) {
        // Q8 int8 fast path (Ch27): the biggest single GEMV in generation.
        const int64_t nb = D / backend::cpu::QK8_0;
        Tensor xc = x.reshape({1, D}).contiguous();
        thread_local std::vector<backend::cpu::BlockQ8_0> xq;
        xq.resize(static_cast<std::size_t>(nb));
        logits = zeros({V, 1});
        backend::cpu::matvec_q8_0_q8_0(lm_head_q8_.data(), xc.data_as<float>().data(),
                                       logits.data_as<float>().data(), V, D, xq.data());
    } else {
        // matmul((V,D),(D,1)) → (V,1) needs no transpose and routes through the
        // vectorised Eigen/BLAS GEMV path (K=D≥64), unlike a scalar dot-product loop.
        logits = ops::matmul(tok_emb_.weight().data(), x.reshape({D, 1}));
    }

    ++cache.filled;
    return logits.reshape({1, V});
}

int64_t     ModernGPT::vocab_size()  const noexcept { return cached_vocab_; }
int64_t     ModernGPT::embed_dim()   const noexcept { return cached_embed_; }
std::size_t ModernGPT::num_layers()  const noexcept { return blocks_.size(); }
int64_t     ModernGPT::n_mtp_heads() const noexcept {
    return static_cast<int64_t>(mtp_heads_.size());
}
std::size_t ModernGPT::n_kv_heads()  const noexcept { return n_kv_heads_; }
std::size_t ModernGPT::head_dim()    const noexcept { return head_dim_; }

} // namespace sub0llm::nn
