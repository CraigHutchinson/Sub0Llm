#include "sub0llm/autograd/ops.hpp"

#include "sub0llm/core/block_pool.hpp"
#include "sub0llm/core/ops.hpp"
#include "../backends/cpu/kernels.hpp"
#include "../backends/cuda/backend.hpp"   // device dispatch for rms_norm (Stage 4 Phase 2)

#include <cmath>
#include <format>
#include <stdexcept>

namespace sub0llm::autograd {

using Shape = Tensor::Shape;

// ── Internal helpers ──────────────────────────────────────────────────────────

namespace {

bool any_grad(const Variable& a, const Variable& b) {
    return a.requires_grad() || b.requires_grad();
}

std::shared_ptr<Node> make_node(Tensor data, bool rg) {
    auto n           = std::allocate_shared<Node>(PoolAllocator<Node>{});
    n->data          = std::move(data);
    n->requires_grad = rg;
    n->is_leaf       = false;
    return n;
}

// Helper: explicitly construct Edge to avoid brace-init issues with std::function.
Edge make_edge(std::shared_ptr<Node> node,
               SmallFunction<Tensor(const Tensor&)> fn) {
    Edge e;
    e.node        = std::move(node);
    e.backward_fn = std::move(fn);
    return e;
}

} // anonymous namespace

// ── add ───────────────────────────────────────────────────────────────────────

Variable add(const Variable& a, const Variable& b) {
    auto out = make_node(ops::add(a.data(), b.data()), any_grad(a, b));
    if (out->requires_grad) {
        if (a.requires_grad())
            out->edges.push_back(make_edge(a.impl(),
                [](const Tensor& g) { return g; }));
        if (b.requires_grad())
            out->edges.push_back(make_edge(b.impl(),
                [](const Tensor& g) { return g; }));
    }
    return Variable::wrap(std::move(out));
}

// ── sub ───────────────────────────────────────────────────────────────────────

Variable sub(const Variable& a, const Variable& b) {
    auto out = make_node(ops::sub(a.data(), b.data()), any_grad(a, b));
    if (out->requires_grad) {
        if (a.requires_grad())
            out->edges.push_back(make_edge(a.impl(),
                [](const Tensor& g) { return g; }));
        if (b.requires_grad())
            out->edges.push_back(make_edge(b.impl(),
                [](const Tensor& g) { return ops::neg(g); }));
    }
    return Variable::wrap(std::move(out));
}

// ── mul (element-wise) ────────────────────────────────────────────────────────

Variable mul(const Variable& a, const Variable& b) {
    auto out = make_node(ops::mul(a.data(), b.data()), any_grad(a, b));
    if (out->requires_grad) {
        if (a.requires_grad()) {
            Tensor b_data = copy(b.data());
            out->edges.push_back(make_edge(a.impl(),
                [b_data](const Tensor& g) { return ops::mul(g, b_data); }));
        }
        if (b.requires_grad()) {
            Tensor a_data = copy(a.data());
            out->edges.push_back(make_edge(b.impl(),
                [a_data](const Tensor& g) { return ops::mul(g, a_data); }));
        }
    }
    return Variable::wrap(std::move(out));
}

// ── matmul ────────────────────────────────────────────────────────────────────
//
// C = A @ B   (A:[m,k], B:[k,n])
// dL/dA = dL/dC @ B^T
// dL/dB = A^T  @ dL/dC

Variable matmul(const Variable& a, const Variable& b) {
    auto out = make_node(ops::matmul(a.data(), b.data()), any_grad(a, b));
    if (out->requires_grad) {
        // Both gradients use the fused transposed-operand kernels — no transposed
        // copy is ever materialized in backward:
        //   dL/dA = g · Bᵀ  → matmul_bt(g, B)      (B row-major as stored)
        //   dL/dB = Aᵀ · g  → matmul_tb(A, g)      (A row-major as stored)
        if (a.requires_grad()) {
            Tensor b_snap = b.data();
            out->edges.push_back(make_edge(a.impl(),
                [b_snap](const Tensor& g) { return ops::matmul_bt(g, b_snap); }));
        }
        if (b.requires_grad()) {
            Tensor a_snap = a.data();
            out->edges.push_back(make_edge(b.impl(),
                [a_snap](const Tensor& g) { return ops::matmul_tb(a_snap, g); }));
        }
    }
    return Variable::wrap(std::move(out));
}

// ── matmul_bt ─────────────────────────────────────────────────────────────────
//
// out = A · Bᵀ   with A(M,K), B(N,K) — B stays row-major, no transpose materialized.
// Backward:  dL/dA = g · B    (already the right orientation — no transpose at all)
//            dL/dB = gᵀ · A   → matmul_tb(g, A), also transpose-free

Variable matmul_bt(const Variable& a, const Variable& b) {
    auto out = make_node(ops::matmul_bt(a.data(), b.data()), any_grad(a, b));
    if (out->requires_grad) {
        if (a.requires_grad()) {
            Tensor b_snap = b.data();
            out->edges.push_back(make_edge(a.impl(),
                [b_snap](const Tensor& g) { return ops::matmul(g, b_snap); }));
        }
        if (b.requires_grad()) {
            Tensor a_snap = a.data();
            out->edges.push_back(make_edge(b.impl(),
                [a_snap](const Tensor& g) { return ops::matmul_tb(g, a_snap); }));
        }
    }
    return Variable::wrap(std::move(out));
}

// ── sum ───────────────────────────────────────────────────────────────────────
//
// s = sum(x)   (scalar, numel=1)
// dL/dx = dL/ds * ones_like(x)

Variable sum(const Variable& x) {
    const float s  = ops::sum(x.data());
    Tensor      st = ones({1}, DType::Float32, x.data().device());
    st.data_as<float>()[0] = s;

    auto out = make_node(std::move(st), x.requires_grad());
    if (out->requires_grad) {
        const Shape  x_shape  = x.data().shape();
        const Device x_device = x.data().device();
        out->edges.push_back(make_edge(x.impl(),
            [x_shape, x_device](const Tensor& g) {
                const float upstream = g.data_as<float>()[0];
                return ops::mul(ones(x_shape, DType::Float32, x_device), upstream);
            }));
    }
    return Variable::wrap(std::move(out));
}

// ── relu ──────────────────────────────────────────────────────────────────────
//
// y = max(0, x)
// dL/dx = dL/dy * (x > 0)

Variable relu(const Variable& x) {
    // CUDA: forward dispatches but the backward mask is host pointer math — guard to a clean
    // error rather than a UB host-deref of device memory (Stage 4: not on the diffusion path).
    if (x.data().device().is_cuda())
        throw std::runtime_error("autograd::relu: not yet implemented on CUDA");
    auto out = make_node(ops::relu(x.data()), x.requires_grad());
    if (out->requires_grad) {
        Tensor x_data = copy(x.data());
        out->edges.push_back(make_edge(x.impl(),
            [x_data](const Tensor& g) {
                Tensor mask = zeros(x_data.shape(), DType::Float32, x_data.device());
                auto   ms   = mask.data_as<float>();
                const auto xs = x_data.data_as<float>();
                for (std::size_t i = 0; i < ms.size(); ++i)
                    ms[i] = xs[i] > 0.0f ? 1.0f : 0.0f;
                return ops::mul(g, mask);
            }));
    }
    return Variable::wrap(std::move(out));
}

// ── bias_add ──────────────────────────────────────────────────────────────────
//
// y[i, j] = x[i, j] + b[j]   (b broadcast over rows)
//
// Backward:
//   grad_x[i, j] = upstream[i, j]
//   grad_b[j]    = sum_i(upstream[i, j])

Variable bias_add(const Variable& x, const Variable& b) {
    const auto& xd = x.data();
    const auto& bd = b.data();
    if (xd.ndim() != 2)
        throw std::runtime_error("autograd::bias_add: x must be 2D");
    const auto N = static_cast<std::size_t>(xd.shape()[0]);
    const auto C = static_cast<std::size_t>(xd.shape()[1]);
    const auto blen = static_cast<std::size_t>(bd.numel());
    if (blen != C)
        throw std::runtime_error(std::format(
            "autograd::bias_add: bias size {} != columns {}", blen, C));

    const Tensor xc = xd.contiguous();
    Tensor out_data;
    if (xd.device().is_cuda()) {                 // Stage 4 Phase 7: on-device bias broadcast-add
        out_data = backend::cuda::bias_add(xc, bd);
    } else {
        out_data = zeros({static_cast<int64_t>(N), static_cast<int64_t>(C)});
        const auto xs = xc.data_as<float>();
        const auto bs = bd.data_as<float>();
        auto       os = out_data.data_as<float>();
        for (std::size_t i = 0; i < N; ++i)
            for (std::size_t j = 0; j < C; ++j)
                os[i * C + j] = xs[i * C + j] + bs[j];
    }

    auto out = make_node(std::move(out_data), any_grad(x, b));
    if (out->requires_grad) {
        if (x.requires_grad())
            out->edges.push_back(make_edge(x.impl(),
                [](const Tensor& g) { return g; }));
        if (b.requires_grad()) {
            const DType  b_dtype  = b.data().dtype();
            const Device b_device = b.data().device();
            out->edges.push_back(make_edge(b.impl(),
                [N, C, b_dtype, b_device](const Tensor& g) {
                    // Sum upstream over rows → (C,) gradient for bias.
                    if (b_device.is_cuda())      // Stage 4 Phase 7: on-device column sum
                        return backend::cuda::bias_add_bwd_b(g.contiguous());
                    Tensor gb  = zeros({static_cast<int64_t>(C)}, b_dtype, b_device);
                    const auto gs  = g.data_as<float>();
                    auto       gbs = gb.data_as<float>();
                    for (std::size_t i = 0; i < N; ++i)
                        for (std::size_t j = 0; j < C; ++j)
                            gbs[j] += gs[i * C + j];
                    return gb;
                }));
        }
    }
    return Variable::wrap(std::move(out));
}

// ── log_softmax ───────────────────────────────────────────────────────────────
//
// y = log(softmax(x))   [over last axis]
//
// Backward:
//   dL/dx_i = dL/dy_i - softmax(x)_i * sum_j(dL/dy_j)

Variable log_softmax(const Variable& x) {
    const auto& xd = x.data();
    if (xd.device().is_cuda())   // not yet on CUDA — clean error vs host-deref UB
        throw std::runtime_error("autograd::log_softmax: not yet implemented on CUDA");
    if (xd.ndim() > 2)
        throw std::runtime_error(
            "autograd::log_softmax: only 1D or 2D input supported");

    const Tensor xc  = xd.contiguous();
    Tensor probs     = ops::softmax(xc, -1);
    Tensor log_probs = ops::log(probs);

    auto out = make_node(std::move(log_probs), x.requires_grad());
    if (out->requires_grad) {
        Tensor probs_copy = copy(probs);
        out->edges.push_back(make_edge(x.impl(),
            [probs_copy](const Tensor& g) {
                const auto gd    = g.data_as<float>();
                const auto pd    = probs_copy.data_as<float>();
                const std::size_t total = gd.size();

                Tensor result = zeros(probs_copy.shape(), DType::Float32, probs_copy.device());
                auto   rd     = result.data_as<float>();

                if (probs_copy.ndim() <= 1) {
                    float sum_g = 0.0f;
                    for (std::size_t i = 0; i < total; ++i) sum_g += gd[i];
                    for (std::size_t i = 0; i < total; ++i)
                        rd[i] = gd[i] - pd[i] * sum_g;
                } else {
                    const auto C = static_cast<std::size_t>(probs_copy.shape()[1]);
                    const std::size_t N = total / C;
                    for (std::size_t r = 0; r < N; ++r) {
                        float sum_g = 0.0f;
                        for (std::size_t c = 0; c < C; ++c)
                            sum_g += gd[r * C + c];
                        for (std::size_t c = 0; c < C; ++c)
                            rd[r * C + c] = gd[r * C + c] - pd[r * C + c] * sum_g;
                    }
                }
                return result;
            }));
    }
    return Variable::wrap(std::move(out));
}

// ── cross_entropy ─────────────────────────────────────────────────────────────
//
// Fused NLL + log-softmax.
//   loss = -1/N * sum_i log_softmax(logits[i])[targets[i]]
//
// Backward (fused, numerically stable):
//   grad_logits[i, j] = softmax(logits[i])[j] / N
//   grad_logits[i, targets[i]] -= 1 / N

Variable cross_entropy(const Variable& logits, const Tensor& targets) {
    const auto& ld = logits.data();
    if (ld.device().is_cuda())   // diffusion uses weighted_cross_entropy (on CUDA); this one isn't
        throw std::runtime_error("autograd::cross_entropy: not yet implemented on CUDA");
    if (ld.ndim() != 2)
        throw std::runtime_error(
            "autograd::cross_entropy: logits must be 2D (N, C)");
    if (targets.ndim() != 1)
        throw std::runtime_error(
            "autograd::cross_entropy: targets must be 1D (N,)");

    const auto N = static_cast<std::size_t>(ld.shape()[0]);
    const auto C = static_cast<std::size_t>(ld.shape()[1]);

    if (targets.numel() != static_cast<std::int64_t>(N))
        throw std::runtime_error(std::format(
            "autograd::cross_entropy: targets size {} != batch size {}",
            targets.numel(), N));

    Tensor probs     = ops::softmax(ld, -1);
    Tensor log_probs = ops::log(probs);

    const auto lps = log_probs.data_as<float>();
    const auto tgs = targets.data_as<int32_t>();

    float loss_val = 0.0f;
    for (std::size_t i = 0; i < N; ++i)
        loss_val += -lps[i * C + static_cast<std::size_t>(tgs[i])];
    loss_val /= static_cast<float>(N);

    Tensor scalar = ones({1}, DType::Float32, ld.device());
    scalar.data_as<float>()[0] = loss_val;

    auto out = make_node(std::move(scalar), logits.requires_grad());
    if (out->requires_grad) {
        Tensor probs_copy = copy(probs);
        Tensor tgt_copy   = copy(targets);

        out->edges.push_back(make_edge(logits.impl(),
            [probs_copy, tgt_copy, N, C](const Tensor& g) {
                const float upstream = g.data_as<float>()[0];
                const float inv_N    = upstream / static_cast<float>(N);

                Tensor grad = copy(probs_copy);
                auto   gd   = grad.data_as<float>();
                const auto tc = tgt_copy.data_as<int32_t>();

                for (std::size_t i = 0; i < N; ++i) {
                    for (std::size_t j = 0; j < C; ++j)
                        gd[i * C + j] *= inv_N;
                    gd[i * C + static_cast<std::size_t>(tc[i])] -= inv_N;
                }
                return grad;
            }));
    }
    return Variable::wrap(std::move(out));
}

// Per-position weighted cross-entropy: loss = Σ wᵢ·CEᵢ / Σ wᵢ.  Used by episodic
// memory to weight each token by its novelty, so the gradient concentrates on the
// surprising tokens and barely touches ones the model already predicts.
Variable weighted_cross_entropy(const Variable& logits, const Tensor& targets,
                                const Tensor& weights) {
    const auto& ld = logits.data();
    if (ld.ndim() != 2)
        throw std::runtime_error("autograd::weighted_cross_entropy: logits must be 2D (N, C)");
    if (targets.ndim() != 1)
        throw std::runtime_error("autograd::weighted_cross_entropy: targets must be 1D (N,)");

    const auto N = static_cast<std::size_t>(ld.shape()[0]);
    const auto C = static_cast<std::size_t>(ld.shape()[1]);
    if (targets.numel() != static_cast<std::int64_t>(N) ||
        weights.numel() != static_cast<std::int64_t>(N))
        throw std::runtime_error("autograd::weighted_cross_entropy: targets/weights size != N");

    Tensor probs = ops::softmax(ld, -1);

    // ── CUDA path: reduce the loss on-device and dispatch the backward kernel ──
    if (ld.device().is_cuda()) {
        const Device dev = ld.device();
        const Tensor tgt_dev = targets.device() == dev ? targets : targets.to(dev);
        const Tensor w_dev   = weights.device() == dev ? weights : weights.to(dev);
        Tensor loss_dev = zeros({1}, DType::Float32, dev);
        Tensor wsum_dev = zeros({1}, DType::Float32, dev);
        backend::cuda::wce_fwd(
            reinterpret_cast<const float*>(probs.raw_ptr()),
            reinterpret_cast<const int*>(tgt_dev.raw_ptr()),
            reinterpret_cast<const float*>(w_dev.raw_ptr()),
            reinterpret_cast<float*>(loss_dev.raw_ptr()),
            reinterpret_cast<float*>(wsum_dev.raw_ptr()),
            static_cast<int>(N), static_cast<int>(C));
        const float wsum = wsum_dev.to(Device::cpu()).data_as<float>()[0];   // D2H one float

        auto out = make_node(std::move(loss_dev), logits.requires_grad());
        if (out->requires_grad) {
            Tensor probs_copy = copy(probs);
            out->edges.push_back(make_edge(logits.impl(),
                [probs_copy, tgt_dev, w_dev, N, C, wsum](const Tensor& g) {
                    Tensor grad = zeros({static_cast<int64_t>(N), static_cast<int64_t>(C)},
                                        DType::Float32, probs_copy.device());
                    backend::cuda::wce_bwd(
                        reinterpret_cast<const float*>(probs_copy.raw_ptr()),
                        reinterpret_cast<const int*>(tgt_dev.raw_ptr()),
                        reinterpret_cast<const float*>(w_dev.raw_ptr()),
                        reinterpret_cast<const float*>(g.raw_ptr()),
                        reinterpret_cast<float*>(grad.raw_ptr()),
                        static_cast<int>(N), static_cast<int>(C), wsum);
                    return grad;
                }));
        }
        return Variable::wrap(std::move(out));
    }

    Tensor log_probs = ops::log(probs);
    const auto lps = log_probs.data_as<float>();
    const auto tgs = targets.data_as<int32_t>();
    const auto wts = weights.data_as<float>();

    float wsum = 0.0f;
    for (std::size_t i = 0; i < N; ++i) wsum += wts[i];
    if (wsum <= 0.0f) wsum = static_cast<float>(N);   // fallback to uniform

    float loss_val = 0.0f;
    for (std::size_t i = 0; i < N; ++i)
        loss_val += wts[i] * (-lps[i * C + static_cast<std::size_t>(tgs[i])]);
    loss_val /= wsum;

    Tensor scalar = ones({1}, DType::Float32, ld.device());
    scalar.data_as<float>()[0] = loss_val;

    auto out = make_node(std::move(scalar), logits.requires_grad());
    if (out->requires_grad) {
        Tensor probs_copy = copy(probs);
        Tensor tgt_copy   = copy(targets);
        Tensor w_copy     = copy(weights);
        out->edges.push_back(make_edge(logits.impl(),
            [probs_copy, tgt_copy, w_copy, N, C, wsum](const Tensor& g) {
                const float upstream = g.data_as<float>()[0];
                Tensor grad = copy(probs_copy);
                auto   gd   = grad.data_as<float>();
                const auto tc = tgt_copy.data_as<int32_t>();
                const auto wc = w_copy.data_as<float>();
                for (std::size_t i = 0; i < N; ++i) {
                    const float wi = wc[i] * upstream / wsum;
                    for (std::size_t j = 0; j < C; ++j) gd[i * C + j] *= wi;
                    gd[i * C + static_cast<std::size_t>(tc[i])] -= wi;
                }
                return grad;
            }));
    }
    return Variable::wrap(std::move(out));
}

// ── gelu ─────────────────────────────────────────────────────────────────────
//
// y = gelu(x)  (tanh approximation: 0.5*x*(1+tanh(sqrt(2/π)*(x+0.044715*x³))))
// dL/dx = g * gelu'(x)
// gelu'(x) = 0.5*(1+t) + 0.5*x*(1-t²)*sqrt(2/π)*(1+3*0.044715*x²)
//            where k = sqrt(2/π)*(x+0.044715*x³), t = tanh(k)

Variable gelu(const Variable& x) {
    if (x.data().device().is_cuda())   // backward is host pointer math — guard (diffusion uses silu)
        throw std::runtime_error("autograd::gelu: not yet implemented on CUDA");
    const Tensor xc       = x.data().contiguous();
    Tensor       out_data = ops::gelu(xc);
    auto out = make_node(std::move(out_data), x.requires_grad());
    if (out->requires_grad) {
        Tensor x_snap = copy(xc);
        out->edges.push_back(make_edge(x.impl(),
            [x_snap](const Tensor& g) {
                const Tensor gc = g.contiguous();
                Tensor gx = zeros(x_snap.shape(), DType::Float32, x_snap.device());
                backend::cpu::gelu_backward_f32(
                    reinterpret_cast<const float*>(gc.raw_ptr()),
                    reinterpret_cast<const float*>(x_snap.raw_ptr()),
                    reinterpret_cast<float*>(gx.raw_ptr()),
                    static_cast<std::size_t>(gx.numel()));
                return gx;
            }));
    }
    return Variable::wrap(std::move(out));
}

// ── layer_norm ────────────────────────────────────────────────────────────────
//
// y[t,d] = weight[d] * x_hat[t,d] + bias[d]
// x_hat[t,d] = (x[t,d] - mean[t]) * rstd[t]
//
// Backward (standard LayerNorm JVP per row t):
//   dhat[j]   = g[t,j] * weight[j]
//   dx[t,j]   = (rstd[t]/D) * (D*dhat[j] - Σ_k dhat[k] - x_hat[t,j]*Σ_k dhat[k]*x_hat[t,k])
//   dweight[j] = Σ_t g[t,j] * x_hat[t,j]
//   dbias[j]   = Σ_t g[t,j]

Variable layer_norm(const Variable& x, const Variable& weight,
                    const Variable& bias, float eps) {
    const auto& xd = x.data();
    if (xd.device().is_cuda())   // diffusion uses rms_norm (on CUDA); layer_norm isn't ported
        throw std::runtime_error("autograd::layer_norm: not yet implemented on CUDA");
    if (xd.ndim() != 2)
        throw std::runtime_error("autograd::layer_norm: x must be 2D (T, D)");
    const std::size_t T = static_cast<std::size_t>(xd.shape()[0]);
    const std::size_t D = static_cast<std::size_t>(xd.shape()[1]);
    if (T == 0)
        throw std::runtime_error("autograd::layer_norm: sequence length T must be > 0");
    if (D == 0)
        throw std::runtime_error("autograd::layer_norm: feature dimension D must be > 0");
    if (weight.data().numel() != static_cast<int64_t>(D) ||
        bias.data().numel()   != static_cast<int64_t>(D))
        throw std::runtime_error(std::format(
            "autograd::layer_norm: weight/bias numel ({},{}) must equal D={}",
            weight.data().numel(), bias.data().numel(), D));

    const Tensor xc = xd.contiguous();
    const Tensor wc = weight.data().contiguous();
    const Tensor bc = bias.data().contiguous();
    const auto   xcs = xc.data_as<float>();
    const auto   ws  = wc.data_as<float>();
    const auto   bs  = bc.data_as<float>();

    Tensor x_hat  = zeros({static_cast<int64_t>(T), static_cast<int64_t>(D)});
    Tensor rstd_t = zeros({static_cast<int64_t>(T)});
    Tensor out_d  = zeros({static_cast<int64_t>(T), static_cast<int64_t>(D)});
    {
        auto xhs = x_hat.data_as<float>();
        auto rs  = rstd_t.data_as<float>();
        auto od  = out_d.data_as<float>();
        const float Df = static_cast<float>(D);
        for (std::size_t i = 0; i < T; ++i) {
            float mu = 0.0f;
            for (std::size_t j = 0; j < D; ++j) mu += xcs[i * D + j];
            mu /= Df;
            float var = 0.0f;
            for (std::size_t j = 0; j < D; ++j) {
                const float d = xcs[i * D + j] - mu;
                var += d * d;
            }
            var /= Df;
            const float rs_i = 1.0f / std::sqrt(var + eps);
            rs[i] = rs_i;
            for (std::size_t j = 0; j < D; ++j) {
                xhs[i * D + j] = (xcs[i * D + j] - mu) * rs_i;
                od[i * D + j]  = ws[j] * xhs[i * D + j] + bs[j];
            }
        }
    }

    const bool rg = x.requires_grad() || weight.requires_grad() || bias.requires_grad();
    auto out = make_node(std::move(out_d), rg);
    if (rg) {
        Tensor xh_snap = copy(x_hat);
        Tensor rs_snap = rstd_t;
        Tensor w_snap  = copy(wc);
        const Device x_dev = x.data().device();
        const Device w_dev = weight.data().device();
        const Device b_dev = bias.data().device();

        if (x.requires_grad())
            out->edges.push_back(make_edge(x.impl(),
                [xh_snap, rs_snap, w_snap, x_dev, T, D](const Tensor& g) {
                    const Tensor gc = g.contiguous();
                    const auto   gs  = gc.data_as<float>();
                    const auto   xhs = xh_snap.data_as<float>();
                    const auto   rs  = rs_snap.data_as<float>();
                    const auto   wgs = w_snap.data_as<float>();
                    Tensor gx = zeros({static_cast<int64_t>(T), static_cast<int64_t>(D)},
                                      DType::Float32, x_dev);
                    auto   gxs = gx.data_as<float>();
                    const float Df = static_cast<float>(D);
                    for (std::size_t i = 0; i < T; ++i) {
                        float s1 = 0.0f, s2 = 0.0f;
                        for (std::size_t j = 0; j < D; ++j) {
                            const float dh = gs[i * D + j] * wgs[j];
                            s1 += dh;
                            s2 += dh * xhs[i * D + j];
                        }
                        for (std::size_t j = 0; j < D; ++j) {
                            const float dh = gs[i * D + j] * wgs[j];
                            gxs[i * D + j] = rs[i] / Df *
                                (Df * dh - s1 - xhs[i * D + j] * s2);
                        }
                    }
                    return gx;
                }));
        if (weight.requires_grad())
            out->edges.push_back(make_edge(weight.impl(),
                [xh_snap, w_dev, T, D](const Tensor& g) {
                    const Tensor gc = g.contiguous();
                    const auto   gs = gc.data_as<float>();
                    const auto   xhs= xh_snap.data_as<float>();
                    Tensor gw = zeros({static_cast<int64_t>(D)}, DType::Float32, w_dev);
                    auto   gws = gw.data_as<float>();
                    for (std::size_t i = 0; i < T; ++i)
                        for (std::size_t j = 0; j < D; ++j)
                            gws[j] += gs[i * D + j] * xhs[i * D + j];
                    return gw;
                }));
        if (bias.requires_grad())
            out->edges.push_back(make_edge(bias.impl(),
                [b_dev, T, D](const Tensor& g) {
                    const Tensor gc = g.contiguous();
                    const auto   gs = gc.data_as<float>();
                    Tensor gb = zeros({static_cast<int64_t>(D)}, DType::Float32, b_dev);
                    auto   gbs = gb.data_as<float>();
                    for (std::size_t i = 0; i < T; ++i)
                        for (std::size_t j = 0; j < D; ++j)
                            gbs[j] += gs[i * D + j];
                    return gb;
                }));
    }
    return Variable::wrap(std::move(out));
}

// ── scale ─────────────────────────────────────────────────────────────────────
//
// y = alpha * x    (scalar broadcast, no learnable parameter)
// dL/dx = alpha * upstream

Variable scale(const Variable& x, float alpha) {
    const Tensor xc = x.data().contiguous();
    auto out = make_node(ops::mul(xc, alpha), x.requires_grad());
    if (out->requires_grad)
        out->edges.push_back(make_edge(x.impl(),
            [alpha](const Tensor& g) { return ops::mul(g, alpha); }));
    return Variable::wrap(std::move(out));
}

// ── transpose2d ───────────────────────────────────────────────────────────────
//
// y = x^T   (M, N) → (N, M)
// dL/dx = (dL/dy)^T

Variable transpose2d(const Variable& x) {
    if (x.data().ndim() != 2)
        throw std::runtime_error("autograd::transpose2d: input must be 2D (M, N)");
    Tensor out_data = x.data().transpose(0, 1).contiguous();
    auto out = make_node(std::move(out_data), x.requires_grad());
    if (out->requires_grad)
        out->edges.push_back(make_edge(x.impl(),
            [](const Tensor& g) { return g.transpose(0, 1).contiguous(); }));
    return Variable::wrap(std::move(out));
}

// ── reshape ───────────────────────────────────────────────────────────────────
// Zero-copy on contiguous data (Tensor::reshape shares storage). Backward reshapes
// the upstream gradient back to the input's original shape.
Variable reshape(const Variable& x, Tensor::Shape new_shape) {
    Tensor::Shape in_shape = x.data().shape();
    auto out = make_node(x.data().reshape(std::move(new_shape)), x.requires_grad());
    if (out->requires_grad)
        out->edges.push_back(make_edge(x.impl(),
            [in_shape](const Tensor& g) { return g.reshape(in_shape); }));
    return Variable::wrap(std::move(out));
}

// ── softmax ───────────────────────────────────────────────────────────────────
//
// y = softmax(x, dim=-1)   row-wise for 2D input (N, C)
//
// Backward (Jacobian-vector product):
//   dL/dx[i] = y[i] * (g[i] - dot(g[i], y[i]))  per row i

Variable softmax(const Variable& x) {
    const auto& xd = x.data();
    if (xd.ndim() != 2)
        throw std::runtime_error("autograd::softmax: only 2D input (N, C) supported");

    const Tensor xc       = xd.contiguous();
    Tensor       out_data = ops::softmax(xc, -1);

    auto out = make_node(std::move(out_data), x.requires_grad());
    if (out->requires_grad) {
        Tensor y_copy = copy(out->data);
        const std::size_t N = static_cast<std::size_t>(xd.shape()[0]);
        const std::size_t C = static_cast<std::size_t>(xd.shape()[1]);
        out->edges.push_back(make_edge(x.impl(),
            [y_copy, N, C](const Tensor& g) {
                const Tensor gc = g.contiguous();
                if (gc.ndim() != 2 ||
                    static_cast<std::size_t>(gc.shape()[0]) != N ||
                    static_cast<std::size_t>(gc.shape()[1]) != C)
                    throw std::runtime_error(std::format(
                        "softmax backward: gradient shape ({},{}) != expected ({},{})",
                        gc.shape()[0], gc.shape()[1], N, C));
                if (y_copy.device().is_cuda())     // Stage 4 Phase 7: gx = y·(g − rowsum(g·y))
                    return backend::cuda::softmax_bwd(gc, y_copy);
                const auto   gs = gc.data_as<float>();
                const auto   ys = y_copy.data_as<float>();
                Tensor gx = zeros({static_cast<int64_t>(N), static_cast<int64_t>(C)},
                                  DType::Float32, y_copy.device());
                auto gxs = gx.data_as<float>();
                for (std::size_t i = 0; i < N; ++i) {
                    float dot = 0.0f;
                    for (std::size_t j = 0; j < C; ++j)
                        dot += gs[i * C + j] * ys[i * C + j];
                    for (std::size_t j = 0; j < C; ++j)
                        gxs[i * C + j] = ys[i * C + j] * (gs[i * C + j] - dot);
                }
                return gx;
            }));
    }
    return Variable::wrap(std::move(out));
}

// ── silu ──────────────────────────────────────────────────────────────────────
//
// y = x * sigmoid(x) = x / (1 + exp(-x))
// dy/dx = sigmoid(x) * (1 + x * (1 − sigmoid(x)))

Variable silu(const Variable& x) {
    const Tensor xc = x.data().contiguous();
    Tensor out_data = ops::silu(xc);
    auto out = make_node(std::move(out_data), x.requires_grad());
    if (out->requires_grad) {
        Tensor x_snap = copy(xc);
        out->edges.push_back(make_edge(x.impl(),
            [x_snap](const Tensor& g) {
                const Tensor gc = g.contiguous();
                Tensor gx = zeros(x_snap.shape(), DType::Float32, x_snap.device());
                if (x_snap.device().is_cuda())
                    backend::cuda::silu_bwd(
                        reinterpret_cast<const float*>(gc.raw_ptr()),
                        reinterpret_cast<const float*>(x_snap.raw_ptr()),
                        reinterpret_cast<float*>(gx.raw_ptr()),
                        static_cast<int>(gx.numel()));
                else
                    backend::cpu::silu_backward_f32(
                        reinterpret_cast<const float*>(gc.raw_ptr()),
                        reinterpret_cast<const float*>(x_snap.raw_ptr()),
                        reinterpret_cast<float*>(gx.raw_ptr()),
                        static_cast<std::size_t>(gx.numel()));
                return gx;
            }));
    }
    return Variable::wrap(std::move(out));
}

// ── rms_norm ──────────────────────────────────────────────────────────────────
//
// y[t,d] = weight[d] * x_norm[t,d],  x_norm[t,d] = x[t,d] / rms[t]
// rms[t] = sqrt((1/D) * Σ_j x[t,j]² + eps)
//
// Backward:
//   h[j]     = weight[j] * g[t,j]                   (per row t)
//   σ[t]     = dot(h[t,:], x_norm[t,:])
//   dx[t,j]  = inv_rms[t]/D * (D*h[j] − x_norm[t,j] * σ[t])
//   dweight[j] = Σ_t g[t,j] * x_norm[t,j]

Variable rms_norm(const Variable& x, const Variable& weight, float eps) {
    const auto& xd = x.data();
    if (xd.ndim() != 2)
        throw std::runtime_error("autograd::rms_norm: x must be 2D (T, D)");
    const std::size_t T = static_cast<std::size_t>(xd.shape()[0]);
    const std::size_t D = static_cast<std::size_t>(xd.shape()[1]);
    if (T == 0 || D == 0)
        throw std::runtime_error("autograd::rms_norm: T and D must be > 0");
    if (weight.data().numel() != static_cast<int64_t>(D))
        throw std::runtime_error(std::format(
            "autograd::rms_norm: weight numel {} must equal D={}", weight.data().numel(), D));

    const Device dev = xd.device();
    const Tensor xc = xd.contiguous();
    Tensor       wc = weight.data().contiguous();

    Tensor x_norm  = zeros({static_cast<int64_t>(T), static_cast<int64_t>(D)}, DType::Float32, dev);
    Tensor inv_rms = zeros({static_cast<int64_t>(T)}, DType::Float32, dev);
    Tensor out_d   = zeros({static_cast<int64_t>(T), static_cast<int64_t>(D)}, DType::Float32, dev);

    if (dev.is_cuda())
        backend::cuda::rms_norm_fwd(
            reinterpret_cast<const float*>(xc.raw_ptr()),
            reinterpret_cast<const float*>(wc.raw_ptr()),
            reinterpret_cast<float*>(x_norm.raw_ptr()),
            reinterpret_cast<float*>(inv_rms.raw_ptr()),
            reinterpret_cast<float*>(out_d.raw_ptr()),
            static_cast<int>(T), static_cast<int>(D), eps);
    else
        backend::cpu::rms_norm_fwd_f32(
            reinterpret_cast<const float*>(xc.raw_ptr()),
            reinterpret_cast<const float*>(wc.raw_ptr()),
            reinterpret_cast<float*>(x_norm.raw_ptr()),
            reinterpret_cast<float*>(inv_rms.raw_ptr()),
            reinterpret_cast<float*>(out_d.raw_ptr()),
            T, D, eps);

    const bool rg = x.requires_grad() || weight.requires_grad();
    auto out = make_node(std::move(out_d), rg);
    if (rg) {
        // Move intermediates into snapshots — no deep copy; closures share the
        // same storage via shared_ptr<Storage>.  Invariant: backward kernels
        // must be read-only on xn_snap / ir_snap (no scratch reuse into those buffers).
        Tensor xn_snap = std::move(x_norm);
        Tensor ir_snap = std::move(inv_rms);
        const Device x_dev = x.data().device();
        const Device w_dev = weight.data().device();

        if (x.requires_grad())
            out->edges.push_back(make_edge(x.impl(),
                [xn_snap, ir_snap, w_snap = std::move(wc), x_dev, T, D](const Tensor& g) {
                    const Tensor gc = g.contiguous();
                    Tensor gx = zeros({static_cast<int64_t>(T), static_cast<int64_t>(D)},
                                      DType::Float32, x_dev);
                    if (x_dev.is_cuda())
                        backend::cuda::rms_norm_bwd_x(
                            reinterpret_cast<const float*>(gc.raw_ptr()),
                            reinterpret_cast<const float*>(xn_snap.raw_ptr()),
                            reinterpret_cast<const float*>(ir_snap.raw_ptr()),
                            reinterpret_cast<const float*>(w_snap.raw_ptr()),
                            reinterpret_cast<float*>(gx.raw_ptr()),
                            static_cast<int>(T), static_cast<int>(D));
                    else
                        backend::cpu::rms_norm_bwd_x_f32(
                            reinterpret_cast<const float*>(gc.raw_ptr()),
                            reinterpret_cast<const float*>(xn_snap.raw_ptr()),
                            reinterpret_cast<const float*>(ir_snap.raw_ptr()),
                            reinterpret_cast<const float*>(w_snap.raw_ptr()),
                            reinterpret_cast<float*>(gx.raw_ptr()),
                            T, D);
                    return gx;
                }));
        if (weight.requires_grad())
            out->edges.push_back(make_edge(weight.impl(),
                [xn_snap, w_dev, T, D](const Tensor& g) {
                    const Tensor gc = g.contiguous();
                    Tensor gw = zeros({static_cast<int64_t>(D)}, DType::Float32, w_dev);
                    if (w_dev.is_cuda())
                        backend::cuda::rms_norm_bwd_w(
                            reinterpret_cast<const float*>(gc.raw_ptr()),
                            reinterpret_cast<const float*>(xn_snap.raw_ptr()),
                            reinterpret_cast<float*>(gw.raw_ptr()),
                            static_cast<int>(T), static_cast<int>(D));
                    else
                        backend::cpu::rms_norm_bwd_w_f32(
                            reinterpret_cast<const float*>(gc.raw_ptr()),
                            reinterpret_cast<const float*>(xn_snap.raw_ptr()),
                            reinterpret_cast<float*>(gw.raw_ptr()),
                            T, D);
                    return gw;
                }));
    }
    return Variable::wrap(std::move(out));
}

// ── rope ──────────────────────────────────────────────────────────────────────
//
// Rotary Position Embedding applied to one head's Q or K.
//   x:         (T, Dh)      input
//   cos_freqs: (T, Dh/2)   precomputed
//   sin_freqs: (T, Dh/2)   precomputed (no grad — constants)
//
// Forward (pair i at position t):
//   out[t, 2i]   = x[t,2i]*c − x[t,2i+1]*s
//   out[t, 2i+1] = x[t,2i]*s + x[t,2i+1]*c
//
// Backward (rotate upstream by −θ, i.e. swap sin sign):
//   dx[t, 2i]   =  g[t,2i]*c + g[t,2i+1]*s
//   dx[t, 2i+1] = −g[t,2i]*s + g[t,2i+1]*c

Variable rope(const Variable& x, const Tensor& cos_freqs, const Tensor& sin_freqs) {
    const auto& xd = x.data();
    if (xd.ndim() != 2)
        throw std::runtime_error("autograd::rope: x must be 2D (T, Dh)");
    const std::size_t T  = static_cast<std::size_t>(xd.shape()[0]);
    const std::size_t Dh = static_cast<std::size_t>(xd.shape()[1]);
    if (Dh % 2 != 0)
        throw std::runtime_error("autograd::rope: head_dim Dh must be even");
    const std::size_t D2 = Dh / 2;

    const Device dev = xd.device();
    const Tensor xc  = xd.contiguous();

    // Half-split (NeoX/HF "rotate_half") convention: dimension i pairs with
    // i+D2, matching ModernGPT::forward_one() and GGUF/Qwen3 weights so the
    // training and KV-cached inference paths produce identical results.
    Tensor out_d = zeros({static_cast<int64_t>(T), static_cast<int64_t>(Dh)}, DType::Float32, dev);
    if (dev.is_cuda()) {
        // cos/sin must be device-resident for the kernel; move them if the caller passed host
        // tensors (cheap, constant). The temporaries' cudaFree synchronises the launch.
        const Tensor cf = cos_freqs.device() == dev ? cos_freqs : cos_freqs.to(dev);
        const Tensor sf = sin_freqs.device() == dev ? sin_freqs : sin_freqs.to(dev);
        backend::cuda::rope_fwd(
            reinterpret_cast<const float*>(xc.raw_ptr()),
            reinterpret_cast<const float*>(cf.raw_ptr()),
            reinterpret_cast<const float*>(sf.raw_ptr()),
            reinterpret_cast<float*>(out_d.raw_ptr()),
            static_cast<int>(T), static_cast<int>(Dh));
    } else {
        const auto xs = xc.data_as<float>();
        const auto cs = cos_freqs.data_as<float>();
        const auto ss = sin_freqs.data_as<float>();
        auto       od = out_d.data_as<float>();
        for (std::size_t t = 0; t < T; ++t)
            for (std::size_t i = 0; i < D2; ++i) {
                const float c  = cs[t * D2 + i];
                const float s  = ss[t * D2 + i];
                const float x0 = xs[t * Dh + i];
                const float x1 = xs[t * Dh + i + D2];
                od[t * Dh + i]      = x0 * c - x1 * s;
                od[t * Dh + i + D2] = x0 * s + x1 * c;
            }
    }

    auto out = make_node(std::move(out_d), x.requires_grad());
    if (out->requires_grad) {
        const Device x_dev = x.data().device();
        // Snapshot cos/sin on the input's device so the backward kernel can read them. .to(dev)
        // is a deep H2D copy for host inputs; a same-device call shares storage (kept alive here).
        Tensor cf_snap = x_dev.is_cuda() ? cos_freqs.to(x_dev) : copy(cos_freqs);
        Tensor sf_snap = x_dev.is_cuda() ? sin_freqs.to(x_dev) : copy(sin_freqs);
        out->edges.push_back(make_edge(x.impl(),
            [cf_snap, sf_snap, x_dev, T, Dh, D2](const Tensor& g) {
                const Tensor gc = g.contiguous();
                Tensor gx  = zeros({static_cast<int64_t>(T), static_cast<int64_t>(Dh)},
                                   DType::Float32, x_dev);
                if (x_dev.is_cuda()) {
                    backend::cuda::rope_bwd(
                        reinterpret_cast<const float*>(gc.raw_ptr()),
                        reinterpret_cast<const float*>(cf_snap.raw_ptr()),
                        reinterpret_cast<const float*>(sf_snap.raw_ptr()),
                        reinterpret_cast<float*>(gx.raw_ptr()),
                        static_cast<int>(T), static_cast<int>(Dh));
                    return gx;
                }
                const auto gs  = gc.data_as<float>();
                const auto cf  = cf_snap.data_as<float>();
                const auto sf  = sf_snap.data_as<float>();
                auto       gxs = gx.data_as<float>();
                for (std::size_t t = 0; t < T; ++t)
                    for (std::size_t i = 0; i < D2; ++i) {
                        const float c  = cf[t * D2 + i];
                        const float s  = sf[t * D2 + i];
                        const float g0 = gs[t * Dh + i];
                        const float g1 = gs[t * Dh + i + D2];
                        gxs[t * Dh + i]      =  g0 * c + g1 * s;
                        gxs[t * Dh + i + D2] = -g0 * s + g1 * c;
                    }
                return gx;
            }));
    }
    return Variable::wrap(std::move(out));
}

// ── narrow ────────────────────────────────────────────────────────────────────
//
// Extract rows [start, start+length) along dim 0.  Backward: scatter upstream
// gradient into a zeros tensor of the original shape.

Variable narrow(const Variable& x, int64_t start, int64_t length) {
    const auto& xd = x.data();
    if (xd.ndim() < 1)
        throw std::runtime_error("autograd::narrow: input must be at least 1D");
    const int64_t T = xd.shape(0);
    if (start < 0 || length <= 0 || start + length > T)
        throw std::runtime_error(std::format(
            "autograd::narrow: [{},{}) out of range [0,{})", start, start + length, T));

    Tensor out_data = ops::narrow(xd, start, length);
    auto out = make_node(std::move(out_data), x.requires_grad());
    if (out->requires_grad) {
        Shape orig_shape = xd.shape();
        const Device x_dev = xd.device();
        out->edges.push_back(make_edge(x.impl(),
            [start, length, orig_shape, x_dev](const Tensor& g) {
                const Tensor gc = g.contiguous();
                Tensor gx  = zeros(orig_shape, DType::Float32, x_dev);   // zeros() cudaMemsets on GPU
                const int64_t row_elems = gc.numel() / length;
                // Backward of a dim-0 narrow scatters the upstream into a contiguous slice of a
                // zeroed buffer — a single copy at offset start·row_elems.
                if (x_dev.is_cuda()) {
                    const std::size_t off = static_cast<std::size_t>(start) * row_elems * sizeof(float);
                    backend::cuda::memcpy_d2d(
                        static_cast<std::byte*>(gx.raw_ptr()) + off, gc.raw_ptr(),
                        static_cast<std::size_t>(gc.numel()) * sizeof(float), x_dev.index);
                    return gx;
                }
                const auto gs  = gc.data_as<float>();
                auto       gxs = gx.data_as<float>();
                for (int64_t i = 0; i < length; ++i)
                    for (int64_t j = 0; j < row_elems; ++j)
                        gxs[static_cast<std::size_t>((start + i) * row_elems + j)] =
                            gs[static_cast<std::size_t>(i * row_elems + j)];
                return gx;
            }));
    }
    return Variable::wrap(std::move(out));
}

// ── log_sigmoid ──────────────────────────────────────────────────────────────
Variable log_sigmoid(const Variable& x) {
    const auto& xd = x.data();
    if (xd.device().is_cuda())   // host pointer math — guard (not on the diffusion path)
        throw std::runtime_error("autograd::log_sigmoid: not yet implemented on CUDA");
    // forward: y = log(sigmoid(x)) = -log(1 + exp(-x))
    // Numerically stable: min(x,0) - log(1 + exp(-|x|))
    Tensor out_data(xd.shape(), DType::Float32, xd.device());
    {
        const auto xs = xd.data_as<float>();
        auto       ys = out_data.data_as<float>();
        for (std::size_t i = 0; i < xs.size(); ++i) {
            const float xi = xs[i];
            ys[i] = (xi >= 0.0f)
                    ? -std::log1p(std::exp(-xi))
                    : xi - std::log1p(std::exp(xi));
        }
    }
    auto out = make_node(std::move(out_data), x.requires_grad());
    if (out->requires_grad) {
        Tensor xsnap = copy(xd);
        out->edges.push_back(make_edge(x.impl(),
            [xsnap = std::move(xsnap)](const Tensor& g) {
                Tensor gx(xsnap.shape(), DType::Float32, xsnap.device());
                const auto xs  = xsnap.data_as<float>();
                const auto gs  = g.data_as<float>();
                auto       gxs = gx.data_as<float>();
                for (std::size_t i = 0; i < xs.size(); ++i) {
                    // d/dx log_sigmoid(x) = sigmoid(-x) = 1/(1+exp(x))
                    const float sig_neg = 1.0f / (1.0f + std::exp(xs[i]));
                    gxs[i] = gs[i] * sig_neg;
                }
                return gx;
            }));
    }
    return Variable::wrap(std::move(out));
}

// ── row_scale ─────────────────────────────────────────────────────────────────
//
// y[i, j] = x[i, j] * v[i, 0]   — scale each row of x by the corresponding
// scalar in v.
//
// Backward:
//   grad_x[i, j] = upstream[i, j] * v[i, 0]
//   grad_v[i, 0] = dot(upstream[i, :], x[i, :])

Variable row_scale(const Variable& x, const Variable& v) {
    const auto& xd = x.data();
    const auto& vd = v.data();
    if (xd.device().is_cuda())   // MoE routing op; host pointer math — guard (not on diffusion path)
        throw std::runtime_error("autograd::row_scale: not yet implemented on CUDA");
    if (xd.ndim() != 2)
        throw std::runtime_error("autograd::row_scale: x must be 2D (N, D)");
    if (vd.ndim() != 2 || vd.shape(1) != 1)
        throw std::runtime_error("autograd::row_scale: v must be 2D (N, 1)");
    const auto N = static_cast<std::size_t>(xd.shape(0));
    const auto D = static_cast<std::size_t>(xd.shape(1));
    if (static_cast<std::size_t>(vd.shape(0)) != N)
        throw std::runtime_error(std::format(
            "autograd::row_scale: x has {} rows but v has {} rows", N, vd.shape(0)));

    const Tensor xc = xd.contiguous();
    const Tensor vc = vd.contiguous();
    Tensor out_data = zeros({static_cast<int64_t>(N), static_cast<int64_t>(D)});
    const auto xs = xc.data_as<float>();
    const auto vs = vc.data_as<float>();
    auto       os = out_data.data_as<float>();
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = 0; j < D; ++j)
            os[i * D + j] = xs[i * D + j] * vs[i];

    const Device x_dev = xd.device();
    const Device v_dev = vd.device();
    auto out = make_node(std::move(out_data), any_grad(x, v));
    if (out->requires_grad) {
        if (x.requires_grad()) {
            out->edges.push_back(make_edge(x.impl(),
                [N, D, vc, x_dev](const Tensor& g) {
                    const auto g_sp  = g.data_as<float>();
                    const auto vc_sp = vc.data_as<float>();
                    Tensor gx = zeros({static_cast<int64_t>(N), static_cast<int64_t>(D)},
                                      DType::Float32, x_dev);
                    auto gxs = gx.data_as<float>();
                    for (std::size_t i = 0; i < N; ++i)
                        for (std::size_t j = 0; j < D; ++j)
                            gxs[i * D + j] = g_sp[i * D + j] * vc_sp[i];
                    return gx;
                }));
        }
        if (v.requires_grad()) {
            out->edges.push_back(make_edge(v.impl(),
                [N, D, xc, v_dev](const Tensor& g) {
                    const auto g_sp  = g.data_as<float>();
                    const auto xc_sp = xc.data_as<float>();
                    Tensor gv = zeros({static_cast<int64_t>(N), 1}, DType::Float32, v_dev);
                    auto gvs = gv.data_as<float>();
                    for (std::size_t i = 0; i < N; ++i) {
                        float s = 0.f;
                        for (std::size_t j = 0; j < D; ++j)
                            s += g_sp[i * D + j] * xc_sp[i * D + j];
                        gvs[i] = s;
                    }
                    return gv;
                }));
        }
    }
    return Variable::wrap(std::move(out));
}

} // namespace sub0llm::autograd
