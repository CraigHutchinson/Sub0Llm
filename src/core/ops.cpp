#include "sub0llm/core/ops.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

// Naive CPU implementations — readable baseline for Chapter 1.
// Chapter 2 will dispatch to SIMD/CUDA/OpenVINO equivalents.

namespace sub0llm::ops {

namespace {

// Validate that two tensors have the same shape for element-wise ops.
void check_same_shape(const Tensor& a, const Tensor& b, std::string_view op) {
    if (a.shape() != b.shape()) {
        throw std::runtime_error(
            std::format("{}: shape mismatch {} vs {}", op,
                a.shape_str(), b.shape_str()));
    }
    if (a.dtype() != b.dtype()) {
        throw std::runtime_error(
            std::format("{}: dtype mismatch {} vs {}", op,
                dtype_name(a.dtype()), dtype_name(b.dtype())));
    }
}

// Helper: apply a binary op element-wise for float32.
template<typename F>
Tensor binary_op_f32(const Tensor& a, const Tensor& b, std::string_view op, F fn) {
    check_same_shape(a, b, op);
    if (a.dtype() != DType::Float32)
        throw std::runtime_error(std::format("{}: only float32 supported in Ch01", op));

    Tensor out(a.shape(), DType::Float32, a.device());
    auto sa  = a.data_as<float>();
    auto sb  = b.data_as<float>();
    auto dst = out.data_as<float>();
    for (std::size_t i = 0; i < dst.size(); ++i)
        dst[i] = fn(sa[i], sb[i]);
    return out;
}

} // anonymous namespace

// ── Element-wise ─────────────────────────────────────────────────────────────

Tensor add(const Tensor& a, const Tensor& b) {
    return binary_op_f32(a, b, "add", [](float x, float y) { return x + y; });
}

Tensor sub(const Tensor& a, const Tensor& b) {
    return binary_op_f32(a, b, "sub", [](float x, float y) { return x - y; });
}

Tensor mul(const Tensor& a, const Tensor& b) {
    return binary_op_f32(a, b, "mul", [](float x, float y) { return x * y; });
}

Tensor div(const Tensor& a, const Tensor& b) {
    return binary_op_f32(a, b, "div", [](float x, float y) { return x / y; });
}

Tensor add(const Tensor& a, float scalar) {
    Tensor out(a.shape(), a.dtype(), a.device());
    auto sa  = a.data_as<float>();
    auto dst = out.data_as<float>();
    for (std::size_t i = 0; i < dst.size(); ++i)
        dst[i] = sa[i] + scalar;
    return out;
}

Tensor mul(const Tensor& a, float scalar) {
    Tensor out(a.shape(), a.dtype(), a.device());
    auto sa  = a.data_as<float>();
    auto dst = out.data_as<float>();
    for (std::size_t i = 0; i < dst.size(); ++i)
        dst[i] = sa[i] * scalar;
    return out;
}

// ── Reductions ────────────────────────────────────────────────────────────────

float sum(const Tensor& t) {
    auto sp = t.data_as<float>();
    return std::accumulate(sp.begin(), sp.end(), 0.0f);
}

float mean(const Tensor& t) {
    if (t.numel() == 0) return 0.0f;
    return sum(t) / static_cast<float>(t.numel());
}

float max(const Tensor& t) {
    auto sp = t.data_as<float>();
    return *std::max_element(sp.begin(), sp.end());
}

float min(const Tensor& t) {
    auto sp = t.data_as<float>();
    return *std::min_element(sp.begin(), sp.end());
}

// ── Activations ───────────────────────────────────────────────────────────────

Tensor relu(const Tensor& t) {
    Tensor out(t.shape(), t.dtype(), t.device());
    auto si  = t.data_as<float>();
    auto dst = out.data_as<float>();
    for (std::size_t i = 0; i < dst.size(); ++i)
        dst[i] = std::max(0.0f, si[i]);
    return out;
}

Tensor gelu(const Tensor& t) {
    // GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    constexpr float kSqrt2OverPi = 0.7978845608f;
    constexpr float kCoef        = 0.044715f;
    Tensor out(t.shape(), t.dtype(), t.device());
    auto si  = t.data_as<float>();
    auto dst = out.data_as<float>();
    for (std::size_t i = 0; i < dst.size(); ++i) {
        const float x = si[i];
        dst[i] = 0.5f * x * (1.0f + std::tanh(kSqrt2OverPi * (x + kCoef * x * x * x)));
    }
    return out;
}

Tensor softmax(const Tensor& t, int dim) {
    // Numerically-stable softmax along the last dimension for 1D/2D tensors.
    // Full n-dim dispatch comes in Ch07 (attention).
    if (t.ndim() > 2 || dim != -1) {
        throw std::runtime_error("softmax: Ch01 only supports 1D/2D with dim=-1");
    }
    Tensor out(t.shape(), t.dtype(), t.device());

    const std::int64_t cols = t.shape(static_cast<std::size_t>(t.ndim() - 1));
    const std::int64_t rows = t.numel() / cols;

    auto si  = t.data_as<float>();
    auto dst = out.data_as<float>();

    for (std::int64_t r = 0; r < rows; ++r) {
        const std::size_t offset = static_cast<std::size_t>(r * cols);
        // Shift by max for stability.
        float mx = *std::max_element(si.begin() + static_cast<std::ptrdiff_t>(offset),
                                     si.begin() + static_cast<std::ptrdiff_t>(offset) + cols);
        float s = 0.0f;
        for (std::int64_t c = 0; c < cols; ++c) {
            dst[offset + static_cast<std::size_t>(c)] = std::exp(si[offset + static_cast<std::size_t>(c)] - mx);
            s += dst[offset + static_cast<std::size_t>(c)];
        }
        for (std::int64_t c = 0; c < cols; ++c)
            dst[offset + static_cast<std::size_t>(c)] /= s;
    }
    return out;
}

Tensor sigmoid(const Tensor& t) {
    Tensor out(t.shape(), t.dtype(), t.device());
    auto si  = t.data_as<float>();
    auto dst = out.data_as<float>();
    for (std::size_t i = 0; i < dst.size(); ++i)
        dst[i] = 1.0f / (1.0f + std::exp(-si[i]));
    return out;
}

// ── Matrix multiply ───────────────────────────────────────────────────────────
// Naive O(M*N*K) — Ch02 replaces with BLAS/cuBLAS/oneDNN.

Tensor matmul(const Tensor& a, const Tensor& b) {
    if (a.ndim() < 2 || b.ndim() < 2) {
        throw std::runtime_error("matmul: inputs must be at least 2D");
    }
    const std::int64_t M = a.shape(a.ndim() - 2);
    const std::int64_t K = a.shape(a.ndim() - 1);
    const std::int64_t K2 = b.shape(b.ndim() - 2);
    const std::int64_t N  = b.shape(b.ndim() - 1);
    if (K != K2) {
        throw std::runtime_error(
            std::format("matmul: inner dims must match, got {} vs {}", K, K2));
    }
    if (a.dtype() != DType::Float32)
        throw std::runtime_error("matmul: only float32 supported in Ch01");

    // For simplicity, only 2D matmul in Ch01; batched added in Ch07.
    if (a.ndim() != 2 || b.ndim() != 2) {
        throw std::runtime_error("matmul: batched matmul added in Ch07 (attention)");
    }

    Tensor out = zeros({M, N}, DType::Float32, a.device());
    auto sa  = a.data_as<float>();
    auto sb  = b.data_as<float>();
    auto dst = out.data_as<float>();

    for (std::int64_t m = 0; m < M; ++m) {
        for (std::int64_t k = 0; k < K; ++k) {
            const float a_mk = sa[static_cast<std::size_t>(m * K + k)];
            for (std::int64_t n = 0; n < N; ++n) {
                dst[static_cast<std::size_t>(m * N + n)] +=
                    a_mk * sb[static_cast<std::size_t>(k * N + n)];
            }
        }
    }
    return out;
}

// ── Unary ─────────────────────────────────────────────────────────────────────

Tensor exp(const Tensor& t) {
    Tensor out(t.shape(), t.dtype(), t.device());
    auto si  = t.data_as<float>();
    auto dst = out.data_as<float>();
    for (std::size_t i = 0; i < dst.size(); ++i) dst[i] = std::exp(si[i]);
    return out;
}

Tensor log(const Tensor& t) {
    Tensor out(t.shape(), t.dtype(), t.device());
    auto si  = t.data_as<float>();
    auto dst = out.data_as<float>();
    for (std::size_t i = 0; i < dst.size(); ++i) dst[i] = std::log(si[i]);
    return out;
}

Tensor sqrt(const Tensor& t) {
    Tensor out(t.shape(), t.dtype(), t.device());
    auto si  = t.data_as<float>();
    auto dst = out.data_as<float>();
    for (std::size_t i = 0; i < dst.size(); ++i) dst[i] = std::sqrt(si[i]);
    return out;
}

Tensor abs(const Tensor& t) {
    Tensor out(t.shape(), t.dtype(), t.device());
    auto si  = t.data_as<float>();
    auto dst = out.data_as<float>();
    for (std::size_t i = 0; i < dst.size(); ++i) dst[i] = std::abs(si[i]);
    return out;
}

Tensor neg(const Tensor& t) {
    Tensor out(t.shape(), t.dtype(), t.device());
    auto si  = t.data_as<float>();
    auto dst = out.data_as<float>();
    for (std::size_t i = 0; i < dst.size(); ++i) dst[i] = -si[i];
    return out;
}

float norm(const Tensor& t) {
    auto sp = t.data_as<float>();
    float s = 0.0f;
    for (float v : sp) s += v * v;
    return std::sqrt(s);
}

} // namespace sub0llm::ops
