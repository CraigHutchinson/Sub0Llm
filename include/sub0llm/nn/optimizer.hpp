#pragma once

#include "sub0llm/autograd/variable.hpp"
#include "sub0llm/core/tensor.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace sub0llm::nn {

// Clip all parameter gradients in-place so the global L2 norm <= max_norm.
// Returns the pre-clip global gradient norm.
[[nodiscard]] float clip_grad_norm(const std::vector<autograd::Variable*>& params,
                                   float max_norm);

// Common runtime interface so a caller can choose Adam / AdamW / Muon by name (see
// make_optimizer). The hot path is one virtual call per step() — negligible.
class Optimizer {
public:
    virtual ~Optimizer() = default;
    virtual void                step()             = 0;
    virtual void                zero_grad()        = 0;
    virtual void                set_lr(float lr) noexcept = 0;
    [[nodiscard]] virtual float lr() const noexcept       = 0;

    // Persist/restore internal state (momentum buffers + step count) so a resumed run
    // continues with the SAME adaptive rates instead of re-warming from zero — an honest
    // resume. Default no-op (a stateless optimizer); stateful ones override. `path` is a
    // binary file written alongside the weight checkpoint. load_state returns false (leaving
    // the fresh zero state) if the file is absent or its shape doesn't match this optimizer.
    virtual void save_state(const std::string& /*path*/) const {}
    virtual bool load_state(const std::string& /*path*/) { return false; }
};

// Stochastic gradient descent with optional momentum.
//   v_{t+1} = momentum * v_t - lr * g_t
//   p_{t+1} = p_t + v_{t+1}
class SGD {
public:
    SGD(std::vector<autograd::Variable*> params, float lr, float momentum = 0.0f);

    void step();
    void zero_grad();

private:
    std::vector<autograd::Variable*> params_;
    std::vector<Tensor>              velocity_;
    float                            lr_;
    float                            momentum_;
};

// Adam optimiser with bias correction (Kingma & Ba, 2015), plus optional DECOUPLED weight
// decay (Loshchilov & Hutter, 2019) — weight_decay > 0 makes it AdamW.
//   m_{t+1} = b1 * m_t + (1 - b1) * g
//   v_{t+1} = b2 * v_t + (1 - b2) * g²
//   m_hat   = m_{t+1} / (1 - b1^{t+1})
//   v_hat   = v_{t+1} / (1 - b2^{t+1})
//   p_{t+1} = p_t - lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * p_t)
class Adam : public Optimizer {
public:
    Adam(std::vector<autograd::Variable*> params,
         float lr  = 1e-3f,
         float b1  = 0.9f,
         float b2  = 0.999f,
         float eps = 1e-8f,
         float weight_decay = 0.0f);   // > 0 ⇒ AdamW

    void step() override;
    void zero_grad() override;

    // Set the learning rate (for an LR schedule, e.g. CosineWithWarmup applied per step).
    void                set_lr(float lr) noexcept override { lr_ = lr; }
    [[nodiscard]] float lr() const noexcept override { return lr_; }

    // Persist/restore (m, v, t) — the full Adam state for an honest resume.
    void save_state(const std::string& path) const override;
    bool load_state(const std::string& path) override;

private:
    std::vector<autograd::Variable*> params_;
    std::vector<Tensor>              m_;
    std::vector<Tensor>              v_;
    int64_t                          t_{0};
    float                            lr_, b1_, b2_, eps_, wd_;
};

// Muon (Jordan, 2024): orthogonalised-momentum optimiser for 2D weight matrices. Each 2D
// param's momentum buffer is orthogonalised by a quintic Newton-Schulz iteration (≈ U·Vᵀ of
// its SVD) BEFORE the update, so the update direction has all singular values ≈ 1 — which
// empirically converges faster and lower than Adam for transformer hidden weights. NON-2D
// params (RMSNorm scales, biases) fall back to AdamW. (Embeddings/output are 2D so they also
// get Muon here; the canonical recipe routes those to AdamW — a documented refinement.)
//   buf   = momentum*buf + g ;  g_eff = nesterov ? g + momentum*buf : buf
//   O     = newton_schulz(g_eff, ns_steps)              # orthogonal
//   W    -= lr * sqrt(max(1, rows/cols)) * O            # aspect-ratio-scaled
class Muon : public Optimizer {
public:
    // force_adamw (optional, per-param): 1 ⇒ use the AdamW fallback for that param even if it
    // is 2D. The canonical Muon recipe routes the token EMBEDDING and OUTPUT head to AdamW —
    // orthogonalising their (token-frequency-skewed) gradients destroys the representation and
    // collapses training. Empty ⇒ route purely by ndim (all 2D ⇒ Muon).
    Muon(std::vector<autograd::Variable*> params,
         float lr = 0.02f, float momentum = 0.95f, int ns_steps = 5, bool nesterov = true,
         float adamw_lr = 3e-4f, float b1 = 0.9f, float b2 = 0.999f,
         float eps = 1e-8f, float weight_decay = 0.0f,
         std::vector<char> force_adamw = {});

    void step() override;
    void zero_grad() override;
    void                set_lr(float lr) noexcept override { lr_ = lr; }
    [[nodiscard]] float lr() const noexcept override { return lr_; }

private:
    std::vector<autograd::Variable*> params_;
    std::vector<char>                is_matrix_;   // 1 = 2D ⇒ Muon, 0 ⇒ AdamW fallback
    std::vector<Tensor>              mom_;         // Muon momentum (matrix params)
    std::vector<Tensor>              m_, v_;       // AdamW state (non-matrix params)
    int64_t                          t_{0};
    float                            lr_, momentum_;
    int                              ns_steps_;
    bool                             nesterov_;
    float                            adamw_lr_, b1_, b2_, eps_, wd_;
};

// Quintic Newton-Schulz orthogonalisation of a 2D matrix (Muon's core kernel; exposed for
// testing/reuse). Returns O ≈ U·Vᵀ of G's SVD — singular values driven to ≈1, so the
// Frobenius norm ‖O‖_F ≈ sqrt(min(rows,cols)). `steps` quintic iterations (5 is standard).
[[nodiscard]] Tensor newton_schulz_orthogonalize(const Tensor& G, int steps = 5);

// Factory: name ∈ {"adam","adamw","muon"} (case-sensitive). `lr` is the primary LR (Adam/
// AdamW step size, or Muon's matrix LR). AdamW uses weight_decay 0.01; Muon uses its defaults
// with adamw_lr for the non-matrix group. Throws on an unknown name.
[[nodiscard]] std::unique_ptr<Optimizer> make_optimizer(
    std::string_view name, std::vector<autograd::Variable*> params, float lr);

} // namespace sub0llm::nn
