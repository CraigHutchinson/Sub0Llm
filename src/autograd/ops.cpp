#include "sub0llm/autograd/ops.hpp"

#include "sub0llm/core/ops.hpp"

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
    auto n           = std::make_shared<Node>();
    n->data          = std::move(data);
    n->requires_grad = rg;
    n->is_leaf       = false;
    return n;
}

// Helper: explicitly construct Edge to avoid brace-init issues with std::function.
Edge make_edge(std::shared_ptr<Node> node,
               std::function<Tensor(const Tensor&)> fn) {
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
        if (a.requires_grad()) {
            Tensor b_t = b.data().transpose(0, 1).contiguous();
            out->edges.push_back(make_edge(a.impl(),
                [b_t](const Tensor& g) { return ops::matmul(g, b_t); }));
        }
        if (b.requires_grad()) {
            Tensor a_t = a.data().transpose(0, 1).contiguous();
            out->edges.push_back(make_edge(b.impl(),
                [a_t](const Tensor& g) { return ops::matmul(a_t, g); }));
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

    Tensor out_data = zeros({static_cast<int64_t>(N), static_cast<int64_t>(C)});
    const auto xs = xd.data_as<float>();
    const auto bs = bd.data_as<float>();
    auto       os = out_data.data_as<float>();
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = 0; j < C; ++j)
            os[i * C + j] = xs[i * C + j] + bs[j];

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
    if (xd.ndim() > 2)
        throw std::runtime_error(
            "autograd::log_softmax: only 1D or 2D input supported");

    Tensor probs     = ops::softmax(xd, -1);
    Tensor log_probs = ops::log(probs);

    auto out = make_node(std::move(log_probs), x.requires_grad());
    if (out->requires_grad) {
        Tensor probs_copy = copy(probs);
        out->edges.push_back(make_edge(x.impl(),
            [probs_copy](const Tensor& g) {
                const auto gd    = g.data_as<float>();
                const auto pd    = probs_copy.data_as<float>();
                const std::size_t total = gd.size();

                Tensor result = zeros(probs_copy.shape());
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

// ── scale ─────────────────────────────────────────────────────────────────────
//
// y = alpha * x    (scalar broadcast, no learnable parameter)
// dL/dx = alpha * upstream

Variable scale(const Variable& x, float alpha) {
    auto out = make_node(ops::mul(x.data(), alpha), x.requires_grad());
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

} // namespace sub0llm::autograd
