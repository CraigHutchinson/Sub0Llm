#include "sub0llm/nn/optimizer.hpp"

#include "sub0llm/core/ops.hpp"
#include "sub0llm/core/thread_pool.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <format>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_set>

#if defined(SUB0LLM_AVX512) || defined(SUB0LLM_AVX2)
#  include <immintrin.h>
#endif

namespace sub0llm::nn {

namespace {

void validate_params(const std::vector<autograd::Variable*>& params,
                     std::string_view who) {
    std::unordered_set<const autograd::Variable*> seen;
    for (const auto* p : params) {
        if (!p || !p->defined())
            throw std::runtime_error(std::format(
                "{}: null or uninitialised Variable* in params", who));
        if (!seen.insert(p).second)
            throw std::runtime_error(std::format(
                "{}: duplicate Variable* in params (same pointer appears twice)",
                who));
    }
}

} // anonymous namespace

float clip_grad_norm(const std::vector<autograd::Variable*>& params,
                     float max_norm) {
    if (max_norm <= 0.0f)
        throw std::runtime_error(std::format(
            "clip_grad_norm: max_norm={} must be positive", max_norm));
    float norm_sq = 0.0f;
    for (const auto* p : params) {
        if (p->grad().numel() == 0) continue;
        const float n = ops::norm(p->grad());
        norm_sq += n * n;
    }
    const float grad_norm = std::sqrt(norm_sq);
    // Guard divisor separately so tiny max_norm values still trigger clipping.
    if (grad_norm > max_norm) {
        const float scale = max_norm / std::max(grad_norm, 1e-12f);
        for (auto* p : params) {
            if (p->grad().numel() == 0) continue;
            for (float& g : p->grad().data_as<float>()) g *= scale;
        }
    }
    return grad_norm;
}

// ── SGD ───────────────────────────────────────────────────────────────────────

SGD::SGD(std::vector<autograd::Variable*> params, float lr, float momentum)
    : params_(std::move(params)), lr_(lr), momentum_(momentum) {
    validate_params(params_, "SGD");
    velocity_.reserve(params_.size());
    for (const auto* p : params_)
        velocity_.push_back(zeros(p->data().shape(), p->data().dtype(),
                                  p->data().device()));
}

void SGD::step() {
    for (std::size_t i = 0; i < params_.size(); ++i) {
        auto* p = params_[i];
        if (p->grad().numel() == 0) continue;
        float*       ps = p->data().data_as<float>().data();
        const float* gs = p->grad().data_as<float>().data();
        float*       vs = velocity_[i].data_as<float>().data();
        const std::size_t n = static_cast<std::size_t>(p->data().numel());
        assert(p->grad().numel() == p->data().numel() && "grad shape must match param shape");

        std::size_t j = 0;
#if defined(SUB0LLM_AVX512)
        const __m512 vm  = _mm512_set1_ps(momentum_);
        const __m512 vlr = _mm512_set1_ps(-lr_);
        for (const std::size_t end = n & ~15u; j < end; j += 16) {
            __m512 vel = _mm512_fmadd_ps(vm, _mm512_loadu_ps(vs + j),
                                          _mm512_mul_ps(vlr, _mm512_loadu_ps(gs + j)));
            _mm512_storeu_ps(vs + j, vel);
            _mm512_storeu_ps(ps + j, _mm512_add_ps(_mm512_loadu_ps(ps + j), vel));
        }
#elif defined(SUB0LLM_AVX2)
        const __m256 vm  = _mm256_set1_ps(momentum_);
        const __m256 vlr = _mm256_set1_ps(-lr_);
        for (const std::size_t end = n & ~7u; j < end; j += 8) {
            __m256 vel = _mm256_fmadd_ps(vm, _mm256_loadu_ps(vs + j),
                                          _mm256_mul_ps(vlr, _mm256_loadu_ps(gs + j)));
            _mm256_storeu_ps(vs + j, vel);
            _mm256_storeu_ps(ps + j, _mm256_add_ps(_mm256_loadu_ps(ps + j), vel));
        }
#endif
        for (; j < n; ++j) {
            vs[j] = momentum_ * vs[j] - lr_ * gs[j];
            ps[j] += vs[j];
        }
    }
}

void SGD::zero_grad() {
    for (auto* p : params_) p->zero_grad();
}

// ── Adam ──────────────────────────────────────────────────────────────────────

Adam::Adam(std::vector<autograd::Variable*> params,
           float lr, float b1, float b2, float eps, float weight_decay)
    : params_(std::move(params)), lr_(lr), b1_(b1), b2_(b2), eps_(eps), wd_(weight_decay) {
    validate_params(params_, "Adam");
    if (b1 < 0.0f || b1 >= 1.0f)
        throw std::runtime_error(std::format(
            "Adam: b1={} must be in [0, 1)", b1));
    if (b2 < 0.0f || b2 >= 1.0f)
        throw std::runtime_error(std::format(
            "Adam: b2={} must be in [0, 1)", b2));
    if (eps <= 0.0f)
        throw std::runtime_error(std::format(
            "Adam: eps={} must be positive", eps));
    m_.reserve(params_.size());
    v_.reserve(params_.size());
    for (const auto* p : params_) {
        m_.push_back(zeros(p->data().shape(), p->data().dtype(), p->data().device()));
        v_.push_back(zeros(p->data().shape(), p->data().dtype(), p->data().device()));
    }
}

void Adam::step() {
    ++t_;
    const float bc1    = 1.0f - std::pow(b1_, static_cast<float>(t_));
    const float bc2    = 1.0f - std::pow(b2_, static_cast<float>(t_));
    const float omB1   = 1.0f - b1_;
    const float omB2   = 1.0f - b2_;
    const float lrBc1  = lr_ / bc1;
    const float invBc2 = 1.0f / bc2;

    for (std::size_t i = 0; i < params_.size(); ++i) {
        auto* p = params_[i];
        if (p->grad().numel() == 0) continue;
        float*       ps = p->data().data_as<float>().data();
        const float* gs = p->grad().data_as<float>().data();
        float*       mi = m_[i].data_as<float>().data();
        float*       vi = v_[i].data_as<float>().data();
        const std::size_t n = static_cast<std::size_t>(p->data().numel());
        assert(p->grad().numel() == p->data().numel() && "grad shape must match param shape");

        // Decoupled weight decay (AdamW): shrink the weight before the Adam step. wd_==0
        // (plain Adam) skips this entirely. Kept as a simple scalar pre-pass so the SIMD
        // Adam update below is unchanged.
        if (wd_ != 0.0f) {
            const float keep = 1.0f - lr_ * wd_;
            for (std::size_t k = 0; k < n; ++k) ps[k] *= keep;
        }

        std::size_t j = 0;
#if defined(SUB0LLM_AVX512)
        const __m512 vb1     = _mm512_set1_ps(b1_);
        const __m512 vomB1   = _mm512_set1_ps(omB1);
        const __m512 vb2     = _mm512_set1_ps(b2_);
        const __m512 vomB2   = _mm512_set1_ps(omB2);
        const __m512 vlrBc1  = _mm512_set1_ps(lrBc1);
        const __m512 vinvBc2 = _mm512_set1_ps(invBc2);
        const __m512 veps    = _mm512_set1_ps(eps_);
        for (const std::size_t end = n & ~15u; j < end; j += 16) {
            __m512 g  = _mm512_loadu_ps(gs + j);
            __m512 mv = _mm512_fmadd_ps(vb1, _mm512_loadu_ps(mi + j), _mm512_mul_ps(vomB1, g));
            __m512 vv = _mm512_fmadd_ps(vb2, _mm512_loadu_ps(vi + j), _mm512_mul_ps(vomB2, _mm512_mul_ps(g, g)));
            _mm512_storeu_ps(mi + j, mv);
            _mm512_storeu_ps(vi + j, vv);
            __m512 denom = _mm512_add_ps(_mm512_sqrt_ps(_mm512_mul_ps(vinvBc2, vv)), veps);
            _mm512_storeu_ps(ps + j, _mm512_sub_ps(_mm512_loadu_ps(ps + j),
                                                     _mm512_div_ps(_mm512_mul_ps(vlrBc1, mv), denom)));
        }
#elif defined(SUB0LLM_AVX2)
        const __m256 vb1     = _mm256_set1_ps(b1_);
        const __m256 vomB1   = _mm256_set1_ps(omB1);
        const __m256 vb2     = _mm256_set1_ps(b2_);
        const __m256 vomB2   = _mm256_set1_ps(omB2);
        const __m256 vlrBc1  = _mm256_set1_ps(lrBc1);
        const __m256 vinvBc2 = _mm256_set1_ps(invBc2);
        const __m256 veps    = _mm256_set1_ps(eps_);
        for (const std::size_t end = n & ~7u; j < end; j += 8) {
            __m256 g  = _mm256_loadu_ps(gs + j);
            __m256 mv = _mm256_fmadd_ps(vb1, _mm256_loadu_ps(mi + j), _mm256_mul_ps(vomB1, g));
            __m256 vv = _mm256_fmadd_ps(vb2, _mm256_loadu_ps(vi + j), _mm256_mul_ps(vomB2, _mm256_mul_ps(g, g)));
            _mm256_storeu_ps(mi + j, mv);
            _mm256_storeu_ps(vi + j, vv);
            __m256 denom = _mm256_add_ps(_mm256_sqrt_ps(_mm256_mul_ps(vinvBc2, vv)), veps);
            _mm256_storeu_ps(ps + j, _mm256_sub_ps(_mm256_loadu_ps(ps + j),
                                                     _mm256_div_ps(_mm256_mul_ps(vlrBc1, mv), denom)));
        }
#endif
        for (; j < n; ++j) {
            mi[j] = b1_ * mi[j] + omB1 * gs[j];
            vi[j] = b2_ * vi[j] + omB2 * gs[j] * gs[j];
            ps[j] -= lrBc1 * mi[j] / (std::sqrt(invBc2 * vi[j]) + eps_);
        }
    }
}

void Adam::zero_grad() {
    for (auto* p : params_) p->zero_grad();
}

// ── Muon ────────────────────────────────────────────────────────────────────────

namespace {

// Transpose a 2D float tensor (rows×cols) → (cols×rows).
Tensor transpose2d(const Tensor& X) {
    const std::int64_t r = X.shape()[0], c = X.shape()[1];
    Tensor Y = zeros({c, r}, DType::Float32, X.device());
    const float* x = X.data_as<float>().data();
    float*       y = Y.data_as<float>().data();
    for (std::int64_t i = 0; i < r; ++i)
        for (std::int64_t j = 0; j < c; ++j)
            y[j * r + i] = x[i * c + j];
    return Y;
}

} // anonymous namespace

// Quintic Newton-Schulz orthogonalisation (Jordan 2024): drives the singular values of G
// to ≈1, returning O ≈ U·Vᵀ of G's SVD. Frobenius-normalised first so the fixed quintic
// (a,b,c) coefficients are in their convergence basin; iterated on the SMALLER dimension so
// the inner products A = X·Xᵀ are min(r,c)² (cheap for tall/wide weight matrices).
Tensor newton_schulz_orthogonalize(const Tensor& G, int steps) {
    constexpr float a = 3.4445f, b = -4.7750f, c = 2.0315f;
    const std::int64_t rows = G.shape()[0], cols = G.shape()[1];
    Tensor X = ops::mul(G, 1.0f / (ops::norm(G) + 1e-7f));
    const bool transpose = rows > cols;             // work on the smaller dim
    if (transpose) X = transpose2d(X);
    for (int s = 0; s < steps; ++s) {
        Tensor A  = ops::matmul_bt(X, X);           // X·Xᵀ  (m×m, m=min(r,c))
        Tensor B  = ops::add(ops::mul(A, b), ops::mul(ops::matmul(A, A), c));  // bA + cA²
        X = ops::add(ops::mul(X, a), ops::matmul(B, X));                       // aX + B·X
    }
    if (transpose) X = transpose2d(X);
    return X;
}

Muon::Muon(std::vector<autograd::Variable*> params, float lr, float momentum, int ns_steps,
           bool nesterov, float adamw_lr, float b1, float b2, float eps, float weight_decay,
           std::vector<char> force_adamw)
    : params_(std::move(params)), lr_(lr), momentum_(momentum), ns_steps_(ns_steps),
      nesterov_(nesterov), adamw_lr_(adamw_lr), b1_(b1), b2_(b2), eps_(eps), wd_(weight_decay) {
    validate_params(params_, "Muon");
    if (ns_steps < 1) throw std::runtime_error("Muon: ns_steps must be >= 1");
    for (std::size_t pi = 0; pi < params_.size(); ++pi) {
        const auto* p = params_[pi];
        const bool forced = pi < force_adamw.size() && force_adamw[pi] != 0;
        const bool mat = p->data().shape().size() == 2 && !forced;   // 2D & not forced ⇒ Muon
        is_matrix_.push_back(mat ? 1 : 0);
        // Allocate only the state the routing needs; a tiny placeholder for the unused side.
        mom_.push_back(mat ? zeros(p->data().shape(), p->data().dtype(), p->data().device())
                           : zeros({1}, DType::Float32, p->data().device()));
        m_.push_back(mat ? zeros({1}, DType::Float32, p->data().device())
                         : zeros(p->data().shape(), p->data().dtype(), p->data().device()));
        v_.push_back(mat ? zeros({1}, DType::Float32, p->data().device())
                         : zeros(p->data().shape(), p->data().dtype(), p->data().device()));
    }
}

void Muon::step() {
    ++t_;
    const float bc1 = 1.0f - std::pow(b1_, static_cast<float>(t_));
    const float bc2 = 1.0f - std::pow(b2_, static_cast<float>(t_));
    // The Newton-Schulz orthogonalisation does many TINY matmuls (per 2D param, per NS
    // iteration). Intra-op GEMM threading's fork/join overhead dwarfs that compute (~2 ms
    // wasted per tiny matmul, ×hundreds/step → ~30× slowdown), so run them serial.
    const bool prev_intra = intra_op_threading_enabled();
    intra_op_threading_enabled() = false;
    struct RestoreIntra {
        bool v; ~RestoreIntra() { intra_op_threading_enabled() = v; }
    } restore{prev_intra};
    for (std::size_t i = 0; i < params_.size(); ++i) {
        auto* p = params_[i];
        if (p->grad().numel() == 0) continue;
        const std::int64_t n = p->data().numel();
        float*       ps = p->data().data_as<float>().data();
        const float* gs = p->grad().data_as<float>().data();

        if (is_matrix_[i]) {
            // Heavy-ball momentum, then orthogonalise (optionally Nesterov-corrected).
            float* buf = mom_[i].data_as<float>().data();
            for (std::int64_t j = 0; j < n; ++j) buf[j] = momentum_ * buf[j] + gs[j];
            Tensor geff = zeros(p->data().shape(), DType::Float32, p->data().device());
            float* ge = geff.data_as<float>().data();
            if (nesterov_) for (std::int64_t j = 0; j < n; ++j) ge[j] = gs[j] + momentum_ * buf[j];
            else           for (std::int64_t j = 0; j < n; ++j) ge[j] = buf[j];

            const Tensor O = newton_schulz_orthogonalize(geff, ns_steps_);
            const float* o = O.data_as<float>().data();
            const std::int64_t rows = p->data().shape()[0], cols = p->data().shape()[1];
            // Aspect-ratio scaling keeps the per-element update magnitude ~Adam-like across
            // tall/wide matrices (O has unit singular values → RMS ≈ 1/sqrt(max(r,c))).
            const float scale = lr_ * std::sqrt(std::max(1.0f,
                                    static_cast<float>(rows) / static_cast<float>(cols)));
            for (std::int64_t j = 0; j < n; ++j) ps[j] -= scale * o[j];
        } else {
            // AdamW fallback for 1D params (norm scales, biases).
            float* mi = m_[i].data_as<float>().data();
            float* vi = v_[i].data_as<float>().data();
            for (std::int64_t j = 0; j < n; ++j) {
                mi[j] = b1_ * mi[j] + (1.0f - b1_) * gs[j];
                vi[j] = b2_ * vi[j] + (1.0f - b2_) * gs[j] * gs[j];
                const float mh = mi[j] / bc1, vh = vi[j] / bc2;
                ps[j] -= adamw_lr_ * (mh / (std::sqrt(vh) + eps_) + wd_ * ps[j]);
            }
        }
    }
}

void Muon::zero_grad() {
    for (auto* p : params_) p->zero_grad();
}

// ── Factory ─────────────────────────────────────────────────────────────────────

std::unique_ptr<Optimizer> make_optimizer(std::string_view name,
                                          std::vector<autograd::Variable*> params, float lr) {
    if (name == "adam")  return std::make_unique<Adam>(std::move(params), lr);
    if (name == "adamw") return std::make_unique<Adam>(std::move(params), lr, 0.9f, 0.999f,
                                                       1e-8f, /*weight_decay=*/0.01f);
    if (name == "muon")  return std::make_unique<Muon>(std::move(params), lr);
    throw std::runtime_error(std::format(
        "make_optimizer: unknown optimizer '{}' (expected adam|adamw|muon)", name));
}

} // namespace sub0llm::nn
