#include "sub0llm/nn/optimizer.hpp"

#include "sub0llm/core/ops.hpp"

#include <cassert>
#include <cmath>
#include <format>
#include <stdexcept>

namespace sub0llm::nn {

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
    velocity_.reserve(params_.size());
    for (const auto* p : params_)
        velocity_.push_back(zeros(p->data().shape(), p->data().dtype(),
                                  p->data().device()));
}

void SGD::step() {
    for (std::size_t i = 0; i < params_.size(); ++i) {
        auto* p = params_[i];
        if (p->grad().numel() == 0) continue;
        auto       ps = p->data().data_as<float>();
        const auto gs = p->grad().data_as<float>();
        auto       vs = velocity_[i].data_as<float>();
        assert(gs.size() == ps.size() && "grad shape must match param shape");
        for (std::size_t j = 0; j < ps.size(); ++j) {
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
    const float bc1 = 1.0f - std::pow(b1_, static_cast<float>(t_));
    const float bc2 = 1.0f - std::pow(b2_, static_cast<float>(t_));

    for (std::size_t i = 0; i < params_.size(); ++i) {
        auto* p = params_[i];
        if (p->grad().numel() == 0) continue;
        auto       ps = p->data().data_as<float>();
        const auto gs = p->grad().data_as<float>();
        auto       ms = m_[i].data_as<float>();
        auto       vs = v_[i].data_as<float>();
        assert(gs.size() == ps.size() && "grad shape must match param shape");
        for (std::size_t j = 0; j < ps.size(); ++j) {
            ms[j] = b1_ * ms[j] + (1.0f - b1_) * gs[j];
            vs[j] = b2_ * vs[j] + (1.0f - b2_) * gs[j] * gs[j];
            const float m_hat = ms[j] / bc1;
            const float v_hat = vs[j] / bc2;
            ps[j] -= lr_ * m_hat / (std::sqrt(v_hat) + eps_);
        }
    }
}

void Adam::zero_grad() {
    for (auto* p : params_) p->zero_grad();
}

} // namespace sub0llm::nn
