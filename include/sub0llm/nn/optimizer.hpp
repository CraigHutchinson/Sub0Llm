#pragma once

#include "sub0llm/autograd/variable.hpp"
#include "sub0llm/core/tensor.hpp"

#include <cstdint>
#include <vector>

namespace sub0llm::nn {

// Clip all parameter gradients in-place so the global L2 norm <= max_norm.
// Returns the pre-clip global gradient norm.
[[nodiscard]] float clip_grad_norm(const std::vector<autograd::Variable*>& params,
                                   float max_norm);

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

// Adam optimiser with bias correction (Kingma & Ba, 2015).
//   m_{t+1} = b1 * m_t + (1 - b1) * g
//   v_{t+1} = b2 * v_t + (1 - b2) * g²
//   m_hat   = m_{t+1} / (1 - b1^{t+1})
//   v_hat   = v_{t+1} / (1 - b2^{t+1})
//   p_{t+1} = p_t - lr * m_hat / (sqrt(v_hat) + eps)
class Adam {
public:
    Adam(std::vector<autograd::Variable*> params,
         float lr  = 1e-3f,
         float b1  = 0.9f,
         float b2  = 0.999f,
         float eps = 1e-8f);

    void step();
    void zero_grad();

private:
    std::vector<autograd::Variable*> params_;
    std::vector<Tensor>              m_;
    std::vector<Tensor>              v_;
    int64_t                          t_{0};
    float                            lr_, b1_, b2_, eps_;
};

} // namespace sub0llm::nn
