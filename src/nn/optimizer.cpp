#include "sub0llm/nn/optimizer.hpp"

#include "sub0llm/core/ops.hpp"

#include <cassert>
#include <cmath>
#include <format>
#include <stdexcept>
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
           float lr, float b1, float b2, float eps)
    : params_(std::move(params)), lr_(lr), b1_(b1), b2_(b2), eps_(eps) {
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

} // namespace sub0llm::nn
