#include "sub0llm/core/ops.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

#include "../backends/cpu/kernels.hpp"
#include "../backends/cuda/backend.hpp"
#include "../backends/openvino/backend.hpp"

// ── Ch02: Dispatch layer ──────────────────────────────────────────────────────
// ops.cpp is now a routing table:
//   CUDA device   → backend::cuda::*
//   OpenVINO dev  → backend::openvino::*   (stub; full in Ch14)
//   CPU           → backend::cpu::*        (AVX-512 / AVX2 / scalar)
//
// The per-backend kernel selection (AVX2 vs scalar) is resolved at compile
// time in backends/cpu/kernels.cpp — ops.cpp sees only the same function
// signatures regardless of the SIMD level.

namespace sub0llm::ops {

// ── Internal guards ────────────────────────────────────────────────────────────

namespace {

void require_f32(const Tensor& t, std::string_view op) {
    if (t.dtype() != DType::Float32)
        throw std::runtime_error(
            std::format("{}: only float32 supported, got {}", op, dtype_name(t.dtype())));
}

void require_same(const Tensor& a, const Tensor& b, std::string_view op) {
    if (a.shape() != b.shape())
        throw std::runtime_error(
            std::format("{}: shape mismatch {} vs {}", op, a.shape_str(), b.shape_str()));
    if (a.dtype() != b.dtype())
        throw std::runtime_error(
            std::format("{}: dtype mismatch {} vs {}", op, dtype_name(a.dtype()), dtype_name(b.dtype())));
    if (a.device() != b.device())
        throw std::runtime_error(
            std::format("{}: device mismatch {} vs {}", op, a.device().str(), b.device().str()));
}

void require_cpu(const Tensor& t, std::string_view op) {
    if (!t.device().is_cpu())
        throw std::runtime_error(
            std::format("{}: not yet implemented for device {}", op, t.device().str()));
}

// Dispatch a unary CPU kernel.
template<typename KernelFn>
Tensor unary_cpu(const Tensor& t, std::string_view op, KernelFn fn) {
    require_f32(t, op);
    Tensor out(t.shape(), t.dtype(), t.device());
    fn(reinterpret_cast<const float*>(t.raw_ptr()),
       reinterpret_cast<float*>(out.raw_ptr()),
       static_cast<std::size_t>(t.numel()));
    return out;
}

// Dispatch a binary CPU kernel. Callers must call require_same and require_cpu first.
template<typename KernelFn>
Tensor binary_cpu(const Tensor& a, const Tensor& b, std::string_view op, KernelFn fn) {
    require_f32(a, op);
    Tensor out(a.shape(), a.dtype(), a.device());
    fn(reinterpret_cast<const float*>(a.raw_ptr()),
       reinterpret_cast<const float*>(b.raw_ptr()),
       reinterpret_cast<float*>(out.raw_ptr()),
       static_cast<std::size_t>(a.numel()));
    return out;
}

} // anonymous namespace

// ── Element-wise binary ───────────────────────────────────────────────────────

Tensor add(const Tensor& a, const Tensor& b) {
    require_same(a, b, "add");
    if (a.device().is_cuda())     return backend::cuda::add(a, b);
    if (a.device().is_openvino()) return backend::openvino::add(a, b);
    return binary_cpu(a, b, "add", backend::cpu::add_f32);
}

Tensor sub(const Tensor& a, const Tensor& b) {
    require_same(a, b, "sub");
    require_cpu(a, "sub");
    return binary_cpu(a, b, "sub", backend::cpu::sub_f32);
}

Tensor mul(const Tensor& a, const Tensor& b) {
    require_same(a, b, "mul");
    if (a.device().is_cuda())     return backend::cuda::mul(a, b);
    return binary_cpu(a, b, "mul", backend::cpu::mul_f32);
}

Tensor div(const Tensor& a, const Tensor& b) {
    require_same(a, b, "div");
    require_cpu(a, "div");
    return binary_cpu(a, b, "div", backend::cpu::div_f32);
}

Tensor add(const Tensor& a, float scalar) {
    require_f32(a, "add_scalar");
    require_cpu(a, "add_scalar");
    Tensor out(a.shape(), a.dtype(), a.device());
    backend::cpu::add_scalar_f32(
        reinterpret_cast<const float*>(a.raw_ptr()),
        scalar,
        reinterpret_cast<float*>(out.raw_ptr()),
        static_cast<std::size_t>(a.numel()));
    return out;
}

Tensor mul(const Tensor& a, float scalar) {
    require_f32(a, "mul_scalar");
    require_cpu(a, "mul_scalar");
    Tensor out(a.shape(), a.dtype(), a.device());
    backend::cpu::mul_scalar_f32(
        reinterpret_cast<const float*>(a.raw_ptr()),
        scalar,
        reinterpret_cast<float*>(out.raw_ptr()),
        static_cast<std::size_t>(a.numel()));
    return out;
}

// ── Reductions ────────────────────────────────────────────────────────────────

float sum (const Tensor& t) { require_f32(t,"sum");  require_cpu(t,"sum");  return backend::cpu::sum_f32 (reinterpret_cast<const float*>(t.raw_ptr()), static_cast<std::size_t>(t.numel())); }
float mean(const Tensor& t) { require_cpu(t,"mean"); return t.numel() ? sum(t)/static_cast<float>(t.numel()) : 0.0f; }
float max (const Tensor& t) { require_f32(t,"max");  require_cpu(t,"max");  return backend::cpu::max_f32 (reinterpret_cast<const float*>(t.raw_ptr()), static_cast<std::size_t>(t.numel())); }
float min (const Tensor& t) { require_f32(t,"min");  require_cpu(t,"min");  return backend::cpu::min_f32 (reinterpret_cast<const float*>(t.raw_ptr()), static_cast<std::size_t>(t.numel())); }
float norm(const Tensor& t) { require_f32(t,"norm"); require_cpu(t,"norm"); return backend::cpu::norm_f32(reinterpret_cast<const float*>(t.raw_ptr()), static_cast<std::size_t>(t.numel())); }

// ── Activations ───────────────────────────────────────────────────────────────

Tensor relu   (const Tensor& t) {
    if (t.device().is_cuda()) return backend::cuda::relu(t);
    return unary_cpu(t, "relu",    backend::cpu::relu_f32);
}
Tensor sigmoid(const Tensor& t) { return unary_cpu(t, "sigmoid", backend::cpu::sigmoid_f32); }
Tensor neg    (const Tensor& t) { return unary_cpu(t, "neg",     backend::cpu::neg_f32);     }
Tensor exp    (const Tensor& t) { return unary_cpu(t, "exp",     backend::cpu::exp_f32);     }
Tensor log    (const Tensor& t) { return unary_cpu(t, "log",     backend::cpu::log_f32);     }
Tensor sqrt   (const Tensor& t) { return unary_cpu(t, "sqrt",    backend::cpu::sqrt_f32);    }
Tensor abs    (const Tensor& t) { return unary_cpu(t, "abs",     backend::cpu::abs_f32);     }

Tensor gelu(const Tensor& t) { return unary_cpu(t, "gelu", backend::cpu::gelu_f32); }
Tensor silu(const Tensor& t) { return unary_cpu(t, "silu", backend::cpu::silu_f32); }

Tensor softmax(const Tensor& t, int dim) {
    if (t.ndim() > 2 || dim != -1)
        throw std::runtime_error("softmax: Ch02 only supports 1D/2D with dim=-1; full n-dim in Ch07");
    require_f32(t, "softmax");
    const std::int64_t cols = t.shape(static_cast<std::size_t>(t.ndim() - 1));
    if (cols == 0)
        throw std::runtime_error("softmax: last dimension must be > 0");
    Tensor out(t.shape(), t.dtype(), t.device());
    const std::int64_t rows = t.numel() / cols;
    backend::cpu::softmax_rows_f32(
        reinterpret_cast<const float*>(t.raw_ptr()),
        reinterpret_cast<float*>(out.raw_ptr()),
        static_cast<std::size_t>(rows),
        static_cast<std::size_t>(cols));
    return out;
}

// ── Matrix multiply ───────────────────────────────────────────────────────────

Tensor matmul(const Tensor& a, const Tensor& b) {
    if (a.ndim() < 2 || b.ndim() < 2)
        throw std::runtime_error("matmul: inputs must be at least 2D");
    if (a.ndim() != 2 || b.ndim() != 2)
        throw std::runtime_error("matmul: batched matmul added in Ch07 (attention)");

    const auto M  = static_cast<std::size_t>(a.shape(0));
    const auto K  = static_cast<std::size_t>(a.shape(1));
    const auto K2 = static_cast<std::size_t>(b.shape(0));
    const auto N  = static_cast<std::size_t>(b.shape(1));

    if (K != K2) throw std::runtime_error(
        std::format("matmul: inner dims must match, got {} vs {}", K, K2));
    require_f32(a, "matmul");

    if (a.device().is_cuda()) return backend::cuda::matmul(a, b);
    if (a.device().is_openvino()) return backend::openvino::matmul(a, b);

    Tensor out(Tensor::Shape{static_cast<std::int64_t>(M), static_cast<std::int64_t>(N)},
               DType::Float32, a.device());

    backend::cpu::matmul_f32(
        reinterpret_cast<const float*>(a.raw_ptr()),
        reinterpret_cast<const float*>(b.raw_ptr()),
        reinterpret_cast<float*>(out.raw_ptr()),
        M, N, K);
    return out;
}

Tensor matmul_bt(const Tensor& a, const Tensor& b) {
    if (a.ndim() != 2 || b.ndim() != 2)
        throw std::runtime_error("matmul_bt: inputs must be 2D");

    const auto M  = static_cast<std::size_t>(a.shape(0));
    const auto K  = static_cast<std::size_t>(a.shape(1));
    const auto N  = static_cast<std::size_t>(b.shape(0));
    const auto K2 = static_cast<std::size_t>(b.shape(1));

    if (K != K2) throw std::runtime_error(
        std::format("matmul_bt: inner dims must match, got A(.,{}) vs B(.,{})", K, K2));
    require_f32(a, "matmul_bt");

    // The fused row-major A·Bᵀ kernel is CPU-only; other devices take the generic path.
    if (!a.device().is_cpu())
        return matmul(a, b.transpose(0, 1).contiguous());

    const Tensor ac = a.contiguous();
    const Tensor bc = b.contiguous();
    Tensor out(Tensor::Shape{static_cast<std::int64_t>(M), static_cast<std::int64_t>(N)},
               DType::Float32, a.device());

    backend::cpu::matmul_bt_f32(
        reinterpret_cast<const float*>(ac.raw_ptr()),
        reinterpret_cast<const float*>(bc.raw_ptr()),
        reinterpret_cast<float*>(out.raw_ptr()),
        M, N, K);
    return out;
}

Tensor narrow(const Tensor& t, int64_t start, int64_t length) {
    require_f32(t, "narrow");
    require_cpu(t, "narrow");
    if (t.ndim() < 1)
        throw std::runtime_error("narrow: input must be at least 1D");
    const int64_t T = t.shape(0);
    if (start < 0 || length <= 0 || start + length > T)
        throw std::runtime_error(
            std::format("narrow: [{},{}) out of range [0,{})", start, start + length, T));

    const int64_t row_elems = t.numel() / T;
    Tensor::Shape out_shape = t.shape();
    out_shape[0] = length;
    Tensor out(out_shape, t.dtype(), t.device());

    const auto src = t.data_as<float>();
    auto       dst = out.data_as<float>();
    for (int64_t i = 0; i < length; ++i)
        for (int64_t j = 0; j < row_elems; ++j)
            dst[static_cast<std::size_t>(i * row_elems + j)] =
                src[static_cast<std::size_t>((start + i) * row_elems + j)];
    return out;
}

} // namespace sub0llm::ops
