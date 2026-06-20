#include "sub0llm/core/ops.hpp"
#include "sub0llm/core/thread_pool.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
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

// Intra-op threading (Ch29 Route B): a large GEMM's M output rows are independent,
// so partition them across the shared P-core pool — each block runs the same 2D
// kernel on its rows. Row-disjoint ⇒ bitwise-identical to serial (no reduction), so
// it is parity-safe and needs no test changes. Small ops stay serial (work threshold).
using RowKernel = void (*)(const float*, const float*, float*,
                           std::size_t, std::size_t, std::size_t);
constexpr std::size_t kMatmulThreadWork = std::size_t{1} << 21;   // ~2M MACs

void run_matmul_rows(const float* A, const float* B, float* C,
                     std::size_t M, std::size_t N, std::size_t K, RowKernel kern) {
    if (intra_op_threading_enabled() && M >= 8 && M * N * K >= kMatmulThreadWork) {
        ThreadPool::global().parallel_for(
            static_cast<std::int64_t>(M),
            [=](std::int64_t r0, std::int64_t r1) {
                kern(A + static_cast<std::size_t>(r0) * K, B,
                     C + static_cast<std::size_t>(r0) * N,
                     static_cast<std::size_t>(r1 - r0), N, K);
            },
            /*grain=*/8);
    } else {
        kern(A, B, C, M, N, K);
    }
}

// Threaded Aᵀ·B (the weight-gradient GEMM dL/dB) — the backward bottleneck. Its
// output C(K,N) reduces over the contracted M dim, so M-row blocks can't write
// disjoint output; instead each P-core computes a PARTIAL C(K,N) from its M-block
// (contiguous A/B row slices → still Eigen-fast) and we sum the partials. The sum
// reorders float adds vs the serial path (≈1e-6 rel), so this is gated to large M
// (training shapes) — the small 2D unit tests stay on the exact serial kernel.
void run_matmul_tb(const float* A, const float* B, float* C,
                   std::size_t M, std::size_t N, std::size_t K) {
    auto& pool = ThreadPool::global();
    const int Tn = pool.size();
    if (!intra_op_threading_enabled() || Tn <= 1 || M < 256 || M * N * K < kMatmulThreadWork) {
        backend::cpu::matmul_tb_f32(A, B, C, M, N, K);
        return;
    }
    const std::size_t KN = K * N;
    // Reused scratch (one allocation, grows monotonically): run_matmul_tb is only
    // ever called from the single graph-building thread (backward is sequential), so
    // a plain function-static buffer is safe — the pool workers only WRITE disjoint
    // [p·KN, (p+1)·KN) slices of it, they never call run_matmul_tb. NOT thread_local
    // (that bound the workers to a different instance → the earlier crash).
    static std::vector<float> partials;
    const std::size_t need = static_cast<std::size_t>(Tn) * KN;
    if (partials.size() < need) partials.resize(need);
    const std::int64_t mchunk = (static_cast<std::int64_t>(M) + Tn - 1) / Tn;
    pool.parallel_for(Tn, [&](std::int64_t p0, std::int64_t p1) {
        for (std::int64_t p = p0; p < p1; ++p) {
            float* dst = partials.data() + static_cast<std::size_t>(p) * KN;
            const std::int64_t m0 = p * mchunk;
            const std::int64_t m1 = std::min<std::int64_t>(static_cast<std::int64_t>(M), m0 + mchunk);
            if (m1 <= m0) { std::fill_n(dst, KN, 0.0f); continue; }
            backend::cpu::matmul_tb_f32(A + static_cast<std::size_t>(m0) * K,
                                        B + static_cast<std::size_t>(m0) * N,
                                        dst, static_cast<std::size_t>(m1 - m0), N, K);
        }
    }, /*grain=*/1);
    // Serial reduce partials → C.
    std::memcpy(C, partials.data(), KN * sizeof(float));
    for (int p = 1; p < Tn; ++p) {
        const float* bp = partials.data() + static_cast<std::size_t>(p) * KN;
        for (std::size_t i = 0; i < KN; ++i) C[i] += bp[i];
    }
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
    if (a.device().is_cuda()) return backend::cuda::mul_scalar(a, scalar);
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
Tensor silu(const Tensor& t) {
    if (t.device().is_cuda()) return backend::cuda::silu(t);
    return unary_cpu(t, "silu", backend::cpu::silu_f32);
}

Tensor softmax(const Tensor& t, int dim) {
    if (t.ndim() > 2 || dim != -1)
        throw std::runtime_error("softmax: Ch02 only supports 1D/2D with dim=-1; full n-dim in Ch07");
    require_f32(t, "softmax");
    const std::int64_t cols = t.shape(static_cast<std::size_t>(t.ndim() - 1));
    if (cols == 0)
        throw std::runtime_error("softmax: last dimension must be > 0");
    if (t.device().is_cuda()) return backend::cuda::softmax(t, dim);
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

// Batched 3D matmul: (B,M,K)·(B,K,N) → (B,M,N), both operands batched with the same
// leading B. CPU/f32 only (the fused kernels are CPU-only). The 2D path is unchanged;
// the autograd VJPs (matmul_bt / matmul_tb) inherit batching by calling back here.
// 2D weight layers fold batch into M via reshape in the model, not through here.
namespace {
template<class Kernel>
Tensor matmul_batched_impl(const Tensor& a, const Tensor& b, const char* name,
                           std::int64_t out_rows, std::int64_t out_cols,
                           std::size_t M, std::size_t N, std::size_t K,
                           std::size_t a_slice, std::size_t b_slice, Kernel kern) {
    require_f32(a, name);
    require_cpu(a, name);
    require_cpu(b, name);
    const Tensor ac = a.contiguous();
    const Tensor bc = b.contiguous();
    const auto B = static_cast<std::size_t>(a.shape(0));
    Tensor out(Tensor::Shape{a.shape(0), out_rows, out_cols}, DType::Float32, a.device());
    const float* pa = reinterpret_cast<const float*>(ac.raw_ptr());
    const float* pb = reinterpret_cast<const float*>(bc.raw_ptr());
    float*       pc = reinterpret_cast<float*>(out.raw_ptr());
    // Output slice is rows×cols (M·N for matmul/bt, K·N for tb) — derive it, don't
    // assume M·N, so the transposed-A kernel's (K,N) output strides correctly.
    const std::size_t c_slice = static_cast<std::size_t>(out_rows) * static_cast<std::size_t>(out_cols);
    for (std::size_t bi = 0; bi < B; ++bi)
        kern(pa + bi * a_slice, pb + bi * b_slice, pc + bi * c_slice, M, N, K);
    return out;
}
} // namespace

Tensor matmul(const Tensor& a, const Tensor& b) {
    if (a.ndim() < 2 || b.ndim() < 2)
        throw std::runtime_error("matmul: inputs must be at least 2D");
    if (a.ndim() == 3 || b.ndim() == 3) {
        if (a.ndim() != 3 || b.ndim() != 3)
            throw std::runtime_error("matmul: batched matmul needs both operands 3D (B,M,K)·(B,K,N)");
        if (a.shape(0) != b.shape(0))
            throw std::runtime_error(std::format(
                "matmul: batch dims must match, got {} vs {}", a.shape(0), b.shape(0)));
        const auto M = static_cast<std::size_t>(a.shape(1)), K = static_cast<std::size_t>(a.shape(2));
        const auto K2 = static_cast<std::size_t>(b.shape(1)), N = static_cast<std::size_t>(b.shape(2));
        if (K != K2) throw std::runtime_error(std::format(
            "matmul: inner dims must match, got {} vs {}", K, K2));
        return matmul_batched_impl(a, b, "matmul", a.shape(1), b.shape(2),
                                   M, N, K, M * K, K * N, backend::cpu::matmul_f32);
    }
    if (a.ndim() != 2 || b.ndim() != 2)
        throw std::runtime_error("matmul: only 2D and 3D-batched are supported");

    const auto M  = static_cast<std::size_t>(a.shape(0));
    const auto K  = static_cast<std::size_t>(a.shape(1));
    const auto K2 = static_cast<std::size_t>(b.shape(0));
    const auto N  = static_cast<std::size_t>(b.shape(1));

    if (K != K2) throw std::runtime_error(
        std::format("matmul: inner dims must match, got {} vs {}", K, K2));
    require_f32(a, "matmul");

    // contiguous(): the device matmul kernel assumes row-major contiguous, but operands can be
    // transposed views (e.g. attention's matmul(Q, transpose2d(K))). No-op when already contiguous;
    // otherwise materialised on-device via the strided-copy kernel (Stage 4 Phase 7).
    if (a.device().is_cuda()) return backend::cuda::matmul(a.contiguous(), b.contiguous());
    if (a.device().is_openvino()) return backend::openvino::matmul(a, b);

    Tensor out(Tensor::Shape{static_cast<std::int64_t>(M), static_cast<std::int64_t>(N)},
               DType::Float32, a.device());

    run_matmul_rows(
        reinterpret_cast<const float*>(a.raw_ptr()),
        reinterpret_cast<const float*>(b.raw_ptr()),
        reinterpret_cast<float*>(out.raw_ptr()),
        M, N, K, backend::cpu::matmul_f32);
    return out;
}

Tensor matmul_bt(const Tensor& a, const Tensor& b) {
    if (a.ndim() == 3 || b.ndim() == 3) {
        if (a.ndim() != 3 || b.ndim() != 3)
            throw std::runtime_error("matmul_bt: batched needs both operands 3D (B,M,K)·(B,N,K)ᵀ");
        if (a.shape(0) != b.shape(0))
            throw std::runtime_error(std::format(
                "matmul_bt: batch dims must match, got {} vs {}", a.shape(0), b.shape(0)));
        const auto M = static_cast<std::size_t>(a.shape(1)), K = static_cast<std::size_t>(a.shape(2));
        const auto N = static_cast<std::size_t>(b.shape(1)), K2 = static_cast<std::size_t>(b.shape(2));
        if (K != K2) throw std::runtime_error(std::format(
            "matmul_bt: inner dims must match, got A(.,{}) vs B(.,{})", K, K2));
        return matmul_batched_impl(a, b, "matmul_bt", a.shape(1), b.shape(1),
                                   M, N, K, M * K, N * K, backend::cpu::matmul_bt_f32);
    }
    if (a.ndim() != 2 || b.ndim() != 2)
        throw std::runtime_error("matmul_bt: inputs must be 2D");

    const auto M  = static_cast<std::size_t>(a.shape(0));
    const auto K  = static_cast<std::size_t>(a.shape(1));
    const auto N  = static_cast<std::size_t>(b.shape(0));
    const auto K2 = static_cast<std::size_t>(b.shape(1));

    if (K != K2) throw std::runtime_error(
        std::format("matmul_bt: inner dims must match, got A(.,{}) vs B(.,{})", K, K2));
    require_f32(a, "matmul_bt");

    if (a.device().is_cuda())
        return backend::cuda::matmul_bt(a.contiguous(), b.contiguous());

    const Tensor ac = a.contiguous();
    const Tensor bc = b.contiguous();
    Tensor out(Tensor::Shape{static_cast<std::int64_t>(M), static_cast<std::int64_t>(N)},
               DType::Float32, a.device());

    run_matmul_rows(
        reinterpret_cast<const float*>(ac.raw_ptr()),
        reinterpret_cast<const float*>(bc.raw_ptr()),
        reinterpret_cast<float*>(out.raw_ptr()),
        M, N, K, backend::cpu::matmul_bt_f32);
    return out;
}

Tensor matmul_tb(const Tensor& a, const Tensor& b) {
    if (a.ndim() == 3 || b.ndim() == 3) {
        if (a.ndim() != 3 || b.ndim() != 3)
            throw std::runtime_error("matmul_tb: batched needs both operands 3D (B,M,K)ᵀ·(B,M,N)");
        if (a.shape(0) != b.shape(0))
            throw std::runtime_error(std::format(
                "matmul_tb: batch dims must match, got {} vs {}", a.shape(0), b.shape(0)));
        const auto M = static_cast<std::size_t>(a.shape(1)), K = static_cast<std::size_t>(a.shape(2));
        const auto M2 = static_cast<std::size_t>(b.shape(1)), N = static_cast<std::size_t>(b.shape(2));
        if (M != M2) throw std::runtime_error(std::format(
            "matmul_tb: leading dims must match, got A({},.) vs B({},.)", M, M2));
        return matmul_batched_impl(a, b, "matmul_tb", a.shape(2), b.shape(2),
                                   M, N, K, M * K, M * N, backend::cpu::matmul_tb_f32);
    }
    if (a.ndim() != 2 || b.ndim() != 2)
        throw std::runtime_error("matmul_tb: inputs must be 2D");

    const auto M  = static_cast<std::size_t>(a.shape(0));
    const auto K  = static_cast<std::size_t>(a.shape(1));
    const auto M2 = static_cast<std::size_t>(b.shape(0));
    const auto N  = static_cast<std::size_t>(b.shape(1));

    if (M != M2) throw std::runtime_error(
        std::format("matmul_tb: leading dims must match, got A({},.) vs B({},.)", M, M2));
    require_f32(a, "matmul_tb");

    if (a.device().is_cuda())
        return backend::cuda::matmul_tb(a.contiguous(), b.contiguous());

    const Tensor ac = a.contiguous();
    const Tensor bc = b.contiguous();
    Tensor out(Tensor::Shape{static_cast<std::int64_t>(K), static_cast<std::int64_t>(N)},
               DType::Float32, a.device());

    run_matmul_tb(
        reinterpret_cast<const float*>(ac.raw_ptr()),
        reinterpret_cast<const float*>(bc.raw_ptr()),
        reinterpret_cast<float*>(out.raw_ptr()),
        M, N, K);
    return out;
}

void matmul_into(const Tensor& a, const Tensor& b, Tensor& c) {
    if (a.ndim() != 2 || b.ndim() != 2 || c.ndim() != 2)
        throw std::runtime_error("matmul_into: a, b, c must be 2D");
    const auto M = static_cast<std::size_t>(a.shape(0)), K = static_cast<std::size_t>(a.shape(1));
    const auto K2 = static_cast<std::size_t>(b.shape(0)), N = static_cast<std::size_t>(b.shape(1));
    if (K != K2 || c.shape(0) != a.shape(0) || c.shape(1) != b.shape(1))
        throw std::runtime_error("matmul_into: shape mismatch (need A(M,K)·B(K,N)→C(M,N))");
    require_f32(a, "matmul_into"); require_cpu(a, "matmul_into");
    backend::cpu::matmul_f32(reinterpret_cast<const float*>(a.raw_ptr()),
                             reinterpret_cast<const float*>(b.raw_ptr()),
                             reinterpret_cast<float*>(c.raw_ptr()), M, N, K);
}

void matmul_bt_into(const Tensor& a, const Tensor& b, Tensor& c) {
    if (a.ndim() != 2 || b.ndim() != 2 || c.ndim() != 2)
        throw std::runtime_error("matmul_bt_into: a, b, c must be 2D");
    const auto M = static_cast<std::size_t>(a.shape(0)), K = static_cast<std::size_t>(a.shape(1));
    const auto N = static_cast<std::size_t>(b.shape(0)), K2 = static_cast<std::size_t>(b.shape(1));
    if (K != K2 || c.shape(0) != a.shape(0) || c.shape(1) != b.shape(0))
        throw std::runtime_error("matmul_bt_into: shape mismatch (need A(M,K)·B(N,K)ᵀ→C(M,N))");
    require_f32(a, "matmul_bt_into"); require_cpu(a, "matmul_bt_into");
    backend::cpu::matmul_bt_f32(reinterpret_cast<const float*>(a.raw_ptr()),
                                reinterpret_cast<const float*>(b.raw_ptr()),
                                reinterpret_cast<float*>(c.raw_ptr()), M, N, K);
}

void matmul_tb_into(const Tensor& a, const Tensor& b, Tensor& c) {
    if (a.ndim() != 2 || b.ndim() != 2 || c.ndim() != 2)
        throw std::runtime_error("matmul_tb_into: a, b, c must be 2D");
    const auto M = static_cast<std::size_t>(a.shape(0)), K = static_cast<std::size_t>(a.shape(1));
    const auto M2 = static_cast<std::size_t>(b.shape(0)), N = static_cast<std::size_t>(b.shape(1));
    if (M != M2 || c.shape(0) != a.shape(1) || c.shape(1) != b.shape(1))
        throw std::runtime_error("matmul_tb_into: shape mismatch (need A(M,K)ᵀ·B(M,N)→C(K,N))");
    require_f32(a, "matmul_tb_into"); require_cpu(a, "matmul_tb_into");
    backend::cpu::matmul_tb_f32(reinterpret_cast<const float*>(a.raw_ptr()),
                                reinterpret_cast<const float*>(b.raw_ptr()),
                                reinterpret_cast<float*>(c.raw_ptr()), M, N, K);
}

Tensor narrow(const Tensor& t, int64_t start, int64_t length) {
    require_f32(t, "narrow");
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

    // Narrow along dim 0 of a row-major tensor is a contiguous block — a single copy of
    // [start·row_elems, (start+length)·row_elems). On CUDA that's a device-to-device memcpy.
    if (t.device().is_cuda()) {
        const std::size_t off   = static_cast<std::size_t>(start) * row_elems * sizeof(float);
        const std::size_t bytes = static_cast<std::size_t>(length) * row_elems * sizeof(float);
        backend::cuda::memcpy_d2d(out.raw_ptr(),
                                  static_cast<const std::byte*>(t.raw_ptr()) + off,
                                  bytes, t.device().index);
        return out;
    }
    require_cpu(t, "narrow");

    const auto src = t.data_as<float>();
    auto       dst = out.data_as<float>();
    for (int64_t i = 0; i < length; ++i)
        for (int64_t j = 0; j < row_elems; ++j)
            dst[static_cast<std::size_t>(i * row_elems + j)] =
                src[static_cast<std::size_t>((start + i) * row_elems + j)];
    return out;
}

} // namespace sub0llm::ops
