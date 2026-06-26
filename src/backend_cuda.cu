// backend_cuda.cu — CUDA compute backend skeleton (part of the GPU build).
//
// Compiled by nvcc (CUDA 13, sm_120) on the multi-compiler path; the C++ engine stays
// on clang and meets this translation unit only at the extern "C" boundary. This is the
// file that grows into the full GPU backend (forward / backward / train_batch / AdamW as
// batched device kernels + CUDA-graph-captured step), eventually replacing backend_cpu.cpp
// when SUB0_COMPUTE=GPU.
//
// Phase 2a (this file's current scope): prove the nvcc build + device execution + the
// host<->device parameter mirror that the full backend relies on. The differentiable
// kernels land in Phase 2b/2c. See include/sub0/core.hpp for the API to implement and
// the memory plan in the repo notes (VRAM = hard fast budget, +shared = soft ceiling).

#include "sub0/layout.hpp"   // PARAM_FLOATS + model dims + COMPUTE_MODE / CUDA_ARCH / HAS_CUDA
#include "sub0/knob.hpp"     // Knob<T,Baked>: compile-time-baked / runtime-tunable knobs

#include <cstdio>
#include <vector>
#include <cmath>

#include <cuda_runtime.h>
#include <cublas_v2.h>

// Export macro for the CUDA backend's C-ABI entry points (the clang-built host links
// against the import lib, so the symbols must leave the DLL). The full backend will
// reuse SUB0_API from core.hpp for the engine API; this covers the extra self-test hook.
#if defined(_WIN32)
  #define SUB0_CUDA_API extern "C" __declspec(dllexport)
#else
  #define SUB0_CUDA_API extern "C" __attribute__((visibility("default")))
#endif

// The GPU backend is dense-FP only for now; ternary/BitNet stays on the CPU backend. A
// GPU build with ternary weights would silently ignore the quantization -> hard stop
// here (mirrors the CMake FATAL_ERROR guard in cmake/Backends.cmake).
// TODO(ternary-gpu): implement absmean re-quant + straight-through estimator on device
// (mirror ternarize_into() in src/backend_cpu.cpp), then lift this guard.
static_assert(!USE_TERNARY,
    "the CUDA backend is dense-FP only; ternary/BitNet is CPU-only for now (TODO(ternary-gpu)).");

namespace {

// CUDA error check for the self-test: report file:line + the error string and bail out
// of the calling function with a nonzero code. The full backend will route failures
// through the engine's fatal path instead.
#define SUB0_CUDA_CHECK(expr)                                                          \
    do {                                                                               \
        const cudaError_t _err = (expr);                                              \
        if (_err != cudaSuccess) {                                                     \
            std::fprintf(stderr, "cuda error: %s at %s:%d -> %s\n", #expr, __FILE__,   \
                         __LINE__, cudaGetErrorString(_err));                          \
            return 1;                                                                  \
        }                                                                              \
    } while (0)

// Same check for void-returning helpers (the training forward/backward): report and bail out
// (return;) instead of a code. Stream-ordered failures still surface on the next sync.
#define SUB0_CUDA_CHECK_VOID(expr)                                                     \
    do {                                                                               \
        const cudaError_t _err = (expr);                                              \
        if (_err != cudaSuccess) {                                                     \
            std::fprintf(stderr, "cuda error: %s at %s:%d -> %s\n", #expr, __FILE__,   \
                         __LINE__, cudaGetErrorString(_err));                          \
            return;                                                                    \
        }                                                                              \
    } while (0)

// Trivial device kernel: y = a*x + y over n elements (the smoke-test workload).
__global__ void axpy_kernel(float a, const float* __restrict__ x, float* __restrict__ y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}

// Device parameter mirror: the model's flat weight blob (PARAM_FLOATS floats) lives here
// during a GPU run. The host params_ptr()/checkpoint buffers sync to/from it via the
// upload/download entry points below (the device half of the sync_params hooks).
float* g_dev_params = nullptr;

// Optimizer state mirrors (Phase 2d), all PARAM_FLOATS long: the reduced gradient the device
// backward writes, the AdamW first/second moments, and a 0/1 weight-decay mask derived once from
// PARAM_LAYOUT (decay applies to matrices, not biases/norms -- matches the CPU ParamView.decay).
float* g_dev_grad  = nullptr;
float* g_dev_m     = nullptr;
float* g_dev_vel   = nullptr;
float* g_dev_decay = nullptr;
double* g_dev_normsq = nullptr;   // [1] global grad sum-of-squares (AdamW clip, double like CPU)

// cuBLAS handle for the dense GEMMs (every Linear + the lm_head). Created lazily, destroyed
// in sub0_cuda_shutdown.
cublasHandle_t g_cublas = nullptr;

// TF32 tensor-core GEMM math expressed as a sub0::Knob: the baked default is the configured
// CUDA_TF32 (a compile-time constant in a normal build; runtime-mutable under SUB0_TUNING so
// the autotuner can sweep it). MEASURED 2026-06 on sm_120: TF32 gives NO speedup at this
// model's small-K GEMMs (96/384) -- 0.98x vs FP32 -- so CUDA_TF32 defaults off. The benchmark
// and sub0_cuda_set_tf32 drive the cuBLAS handle directly to compare modes regardless of the
// baked value.
using CudaTf32 = sub0::Knob<bool, CUDA_TF32>;

// Dedicated stream for all device work so the forward can be captured into a CUDA graph
// (capturing the legacy default stream is disallowed). Plus the captured executable graph and
// the (batch,T) it was captured for -- recaptured when the shape or the GEMM math mode changes.
cudaStream_t    g_stream      = nullptr;
cudaGraphExec_t g_graph_exec  = nullptr;
int             g_graph_batch = -1;
int             g_graph_T     = -1;

// Broadcast bias add: Y[m,o] += bias[o] over all M rows. Applied after the cuBLAS GEMM
// (the legacy cublasSgemm has no bias epilogue) to add the linear's bias.
__global__ void bias_add_kernel(float* __restrict__ Y, const float* __restrict__ bias, int M, int N) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M * N) Y[idx] += bias[idx % N];
}

// --- Fast transcendental math (CUDA native intrinsics) ----------------------
// Use the hardware fast-math path -- __expf on the SFU, rsqrtf for reciprocal-sqrt --
// rather than re-deriving the CPU's software polynomial. Both are ~1e-6 approximations,
// so the GPU forward stays close to the CPU FAST_MATH path but is NOT bit-identical; the
// CPU<->GPU parity test uses a tolerance that absorbs the two approximations' difference.
// dev_tanh keeps the tanh-form GELU shape (matching the CPU's GPT-2-style approx) built on
// __expf, so the GELU stays consistent with the CPU op_gelu.
__device__ inline float dev_tanh(float x) {
    const float e = __expf(-2.f * x);
    return (1.f - e) / (1.f + e);
}
__device__ inline float dev_gelu(float v) {
    return 0.5f * v * (1.f + dev_tanh(0.7978845608f * (v + 0.0447150000f * v * v * v)));
}
// GELU derivative, kept consistent with dev_gelu (mirrors the CPU dgelu_fast tanh form) so the
// backward gradient matches the CPU reference. GELU_C = sqrt(2/pi), GELU_A = 0.044715.
__device__ inline float dev_dgelu(float v) {
    const float t  = dev_tanh(0.7978845608f * (v + 0.0447150000f * v * v * v));
    const float du = 0.7978845608f * (1.f + 3.f * 0.0447150000f * v * v);
    return 0.5f * (1.f + t) + 0.5f * v * (1.f - t * t) * du;
}

// h[m,j] = tok_emb[ids[m], j] + pos_emb[t, j], where m is the global row over batch*T and
// t = m % T is the position WITHIN the window (op_embed + op_add: first forward step).
__global__ void embed_add_kernel(const float* __restrict__ tok_emb, const float* __restrict__ pos_emb,
                                 const int* __restrict__ ids, float* __restrict__ h, int M, int T, int C) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = blockIdx.y * blockDim.y + threadIdx.y;
    if (m < M && j < C) {
        const int t = m % T;
        h[m * C + j] = tok_emb[ids[m] * C + j] + pos_emb[t * C + j];
    }
}

// y[m,j] = x[m,j] * (1/sqrt(mean_j x^2 + eps)) * gamma[j]  (op_rmsnorm forward; one row/thread).
// TODO(perf): one thread per row serializes the C-length reduction. For larger C a block-per-row
// warp/shared reduction (or a fused rmsnorm+linear epilogue) would cut memory traffic.
__global__ void rmsnorm_kernel(const float* __restrict__ x, const float* __restrict__ gamma,
                               float* __restrict__ y, int rows, int C) {
    const int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m < rows) {
        const float eps = 1e-5f;
        const float* xr = x + static_cast<size_t>(m) * C;
        float ms = 0.f;
        for (int j = 0; j < C; ++j) ms += xr[j] * xr[j];
        ms /= C;
        const float r = rsqrtf(ms + eps);              // CUDA fast reciprocal sqrt
        float* yr = y + static_cast<size_t>(m) * C;
        for (int j = 0; j < C; ++j) yr[j] = xr[j] * r * gamma[j];
    }
}

// Training RMSNorm: same as rmsnorm_kernel but also saves the per-row reciprocal-rms `rinv` that
// the backward pass needs (so backward uses the exact value forward produced).
__global__ void rmsnorm_train_kernel(const float* __restrict__ x, const float* __restrict__ gamma,
                                     float* __restrict__ y, float* __restrict__ rinv, int rows, int C) {
    const int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m < rows) {
        const float eps = 1e-5f;
        const float* xr = x + static_cast<size_t>(m) * C;
        float ms = 0.f;
        for (int j = 0; j < C; ++j) ms += xr[j] * xr[j];
        ms /= C;
        const float r = rsqrtf(ms + eps);
        rinv[m] = r;
        float* yr = y + static_cast<size_t>(m) * C;
        for (int j = 0; j < C; ++j) yr[j] = xr[j] * r * gamma[j];
    }
}

// Elementwise tanh-form GELU (op_gelu, FAST_MATH path).
__global__ void gelu_kernel(const float* __restrict__ x, float* __restrict__ y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = dev_gelu(x[i]);
}

// Elementwise add: c[i] = a[i] + b[i] (residual connections; safe in-place when c == a).
__global__ void add_kernel(const float* __restrict__ a, const float* __restrict__ b,
                           float* __restrict__ c, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

// Causal multi-head attention over a BATCH of windows (op_attn, FAST_MATH softmax). One
// TODO(perf): naive O(T^2) per thread with a float sc[SEQ_LEN] in local memory. For longer T a
// flash-attention-style tiled/online-softmax kernel (shared-mem K/V tiles, no sc[] spill) would
// be far more bandwidth- and occupancy-efficient. Fine at SEQ_LEN=64.
// thread per (window b, head h, query i): scores over j<=i WITHIN the window, __expf softmax,
// weighted sum of v. q/k/v rows are `in_stride` apart (= 3C when they are sub-blocks of the
// fused QKV buffer); the output is a packed [M,C] buffer (stride C). Windows are independent.
__global__ void attn_kernel(const float* __restrict__ q, const float* __restrict__ k,
                            const float* __restrict__ v, float* __restrict__ out,
                            int batch, int T, int C, int H, int in_stride) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;   // idx over batch*H*T
    const int per = H * T;
    const int b   = idx / per;
    if (b >= batch) return;
    const int rem = idx - b * per;
    const int h   = rem / T;
    const int i   = rem % T;
    const int    d        = C / H;
    const int    off      = h * d;
    const float  scale    = 1.0f / sqrtf(static_cast<float>(d));
    const size_t in_base  = static_cast<size_t>(b) * T * in_stride;   // q/k/v rows are in_stride apart
    const size_t out_base = static_cast<size_t>(b) * T * C;
    float sc[SEQ_LEN];
    const float* qi = q + in_base + static_cast<size_t>(i) * in_stride + off;
    float mx = -1e30f;
    for (int j = 0; j <= i; ++j) {
        const float* kj = k + in_base + static_cast<size_t>(j) * in_stride + off;
        float s = 0.f;
        for (int a = 0; a < d; ++a) s += qi[a] * kj[a];
        s *= scale; sc[j] = s; if (s > mx) mx = s;
    }
    float Z = 0.f;
    for (int j = 0; j <= i; ++j) { sc[j] = __expf(sc[j] - mx); Z += sc[j]; }   // SFU fast exp
    float* oi = out + out_base + static_cast<size_t>(i) * C + off;
    for (int a = 0; a < d; ++a) oi[a] = 0.f;
    for (int j = 0; j <= i; ++j) {
        const float pj = sc[j] / Z;
        const float* vj = v + in_base + static_cast<size_t>(j) * in_stride + off;
        for (int a = 0; a < d; ++a) oi[a] += pj * vj[a];
    }
}

// Training attention: identical math to attn_kernel but also writes the softmax weights P (one
// causal [T,T] block per (window, head), layout [batch,H,T,T]) that the backward pass consumes.
__global__ void attn_train_kernel(const float* __restrict__ q, const float* __restrict__ k,
                                  const float* __restrict__ v, float* __restrict__ out,
                                  float* __restrict__ P, int batch, int T, int C, int H, int in_stride) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;   // idx over batch*H*T
    const int per = H * T;
    const int b   = idx / per;
    if (b >= batch) return;
    const int rem = idx - b * per;
    const int h   = rem / T;
    const int i   = rem % T;
    const int    d        = C / H;
    const int    off      = h * d;
    const float  scale    = 1.0f / sqrtf(static_cast<float>(d));
    const size_t in_base  = static_cast<size_t>(b) * T * in_stride;
    const size_t out_base = static_cast<size_t>(b) * T * C;
    const size_t Prow     = ((static_cast<size_t>(b) * H + h) * T + i) * T;
    float sc[SEQ_LEN];
    const float* qi = q + in_base + static_cast<size_t>(i) * in_stride + off;
    float mx = -1e30f;
    for (int j = 0; j <= i; ++j) {
        const float* kj = k + in_base + static_cast<size_t>(j) * in_stride + off;
        float s = 0.f;
        for (int a = 0; a < d; ++a) s += qi[a] * kj[a];
        s *= scale; sc[j] = s; if (s > mx) mx = s;
    }
    float Z = 0.f;
    for (int j = 0; j <= i; ++j) { sc[j] = __expf(sc[j] - mx); Z += sc[j]; }
    float* oi = out + out_base + static_cast<size_t>(i) * C + off;
    for (int a = 0; a < d; ++a) oi[a] = 0.f;
    const float invZ = 1.f / Z;
    for (int j = 0; j <= i; ++j) {
        const float pj = sc[j] * invZ;
        P[Prow + j] = pj;
        const float* vj = v + in_base + static_cast<size_t>(j) * in_stride + off;
        for (int a = 0; a < d; ++a) oi[a] += pj * vj[a];
    }
}

// Build the fused QKV weight Wqkv[C, 3C] (row-major) from Wq,Wk,Wv [C,C]: row p holds
// [Wq[p] | Wk[p] | Wv[p]]. Materialized ONCE at upload so the three projection GEMMs collapse
// into one a . Wqkv -> [M, 3C] (better-shaped GEMM + fewer launches).
__global__ void build_qkv_kernel(const float* __restrict__ Wq, const float* __restrict__ Wk,
                                 const float* __restrict__ Wv, float* __restrict__ Wqkv, int C) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;   // over C*C elements of one source matrix
    if (idx < C * C) {
        const int p = idx / C, c = idx % C;
        const int row = p * 3 * C;
        Wqkv[row + c]         = Wq[idx];
        Wqkv[row + C + c]     = Wk[idx];
        Wqkv[row + 2 * C + c] = Wv[idx];
    }
}

// ============================================================================
//  Backward kernels (Phase 2d) -- mirror src/backend_cpu.cpp backward_node()
// ============================================================================

// Cross-entropy backward (op_cross_entropy): one thread per row m over [M,V]. Recomputes the
// row softmax, accumulates the mean loss (-log p[target], summed; host divides by M) and writes
// dlogits[m,j] = (1/M)(p - onehot). invM folds the CPU's (1/batch)*(1/T) seed since M = batch*T.
__global__ void ce_backward_kernel(const float* __restrict__ logits, const int* __restrict__ targets,
                                   float* __restrict__ dlogits, double* __restrict__ loss_acc,
                                   int M, int V, float invM) {
    const int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M) return;
    const float* lr = logits + static_cast<size_t>(m) * V;
    float* dl       = dlogits + static_cast<size_t>(m) * V;
    const int tgt   = targets[m];
    float mx = -1e30f;
    for (int j = 0; j < V; ++j) mx = fmaxf(mx, lr[j]);
    float Z = 0.f;
    for (int j = 0; j < V; ++j) Z += __expf(lr[j] - mx);
    const float invZ = 1.f / Z;
    for (int j = 0; j < V; ++j) {
        const float p = __expf(lr[j] - mx) * invZ;
        dl[j] = invM * (p - (j == tgt ? 1.f : 0.f));
    }
    const float ptgt = __expf(lr[tgt] - mx) * invZ;
    atomicAdd(loss_acc, static_cast<double>(-__logf(fmaxf(1e-9f, ptgt))));
}

// Bias gradient: dbias[o] = sum_m dY[m,o] (one thread per output column).
__global__ void bias_grad_kernel(const float* __restrict__ dY, float* __restrict__ dbias, int M, int N) {
    const int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= N) return;
    float s = 0.f;
    for (int m = 0; m < M; ++m) s += dY[static_cast<size_t>(m) * N + o];
    dbias[o] = s;
}

// RMSNorm backward (op_rmsnorm): one row/thread. ACCUMULATES dx into the running residual-stream
// gradient (+=) and dgamma (atomic, shared across rows). Mirrors the CPU formula exactly.
__global__ void rmsnorm_backward_kernel(const float* __restrict__ x, const float* __restrict__ gamma,
                                        const float* __restrict__ rinv, const float* __restrict__ dy,
                                        float* __restrict__ dx, float* __restrict__ dgamma,
                                        int rows, int C) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= rows) return;
    const float* xr  = x  + static_cast<size_t>(t) * C;
    const float* dyr = dy + static_cast<size_t>(t) * C;
    float*       dxr = dx + static_cast<size_t>(t) * C;
    float S = 0.f;
    for (int j = 0; j < C; ++j) S += dyr[j] * gamma[j] * xr[j];
    const float r = rinv[t], r3 = r * r * r;
    for (int j = 0; j < C; ++j) {
        const float xj = xr[j], dyj = dyr[j], gj = gamma[j];
        dxr[j] += r * dyj * gj - (xj * r3 / C) * S;
        atomicAdd(&dgamma[j], dyj * xj * r);
    }
}

// GELU backward (op_gelu): dx[i] = dy[i] * dgelu(x[i]). Writes (input grad buffer is fresh).
__global__ void gelu_backward_kernel(const float* __restrict__ x, const float* __restrict__ dy,
                                     float* __restrict__ dx, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dx[i] = dy[i] * dev_dgelu(x[i]);
}

// Causal attention backward (op_attn): one thread per (window b, head h) owns that head's
// columns [off, off+d), so dq/dk/dv accumulate with NO cross-thread races. q/k/v and dq/dk/dv
// are stride-in_stride sub-blocks of the fused QKV buffer; dout/att are packed (stride C). The
// dq/dk/dv buffer must be zeroed before launch. Mirrors the CPU backward exactly.
__global__ void attn_backward_kernel(const float* __restrict__ q, const float* __restrict__ k,
                                     const float* __restrict__ v, const float* __restrict__ P,
                                     const float* __restrict__ dout, float* __restrict__ dq,
                                     float* __restrict__ dk, float* __restrict__ dv,
                                     int batch, int T, int C, int H, int in_stride) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;   // over batch*H
    if (idx >= batch * H) return;
    const int b = idx / H, h = idx % H;
    const int d = C / H, off = h * d;
    const float scale = 1.0f / sqrtf(static_cast<float>(d));
    const size_t in_base  = static_cast<size_t>(b) * T * in_stride;
    const size_t out_base = static_cast<size_t>(b) * T * C;
    const size_t Pbase    = (static_cast<size_t>(b) * H + h) * T * T;
    float dP[SEQ_LEN];
    for (int i = 0; i < T; ++i) {
        const float* douti = dout + out_base + static_cast<size_t>(i) * C + off;
        for (int j = 0; j <= i; ++j) {
            const float p = P[Pbase + static_cast<size_t>(i) * T + j];
            const float* vj = v + in_base + static_cast<size_t>(j) * in_stride + off;
            float* dvj      = dv + in_base + static_cast<size_t>(j) * in_stride + off;
            float dp = 0.f;
            for (int a = 0; a < d; ++a) { dvj[a] += p * douti[a]; dp += douti[a] * vj[a]; }
            dP[j] = dp;
        }
        float dot = 0.f;
        for (int j = 0; j <= i; ++j) dot += P[Pbase + static_cast<size_t>(i) * T + j] * dP[j];
        const float* qi = q + in_base + static_cast<size_t>(i) * in_stride + off;
        float* dqi      = dq + in_base + static_cast<size_t>(i) * in_stride + off;
        for (int j = 0; j <= i; ++j) {
            const float ds = P[Pbase + static_cast<size_t>(i) * T + j] * (dP[j] - dot) * scale;
            const float* kj = k + in_base + static_cast<size_t>(j) * in_stride + off;
            float* dkj      = dk + in_base + static_cast<size_t>(j) * in_stride + off;
            for (int a = 0; a < d; ++a) { dqi[a] += ds * kj[a]; dkj[a] += ds * qi[a]; }
        }
    }
}

// Embedding backward (op_embed x2): scatter-add the residual-stream grad into tok_emb / pos_emb
// rows. Multiple rows map to the same token/position, so accumulation is atomic. One thread per
// (row m, channel j). pos = m % T (position within the window).
__global__ void embed_backward_kernel(const float* __restrict__ dh, const int* __restrict__ ids,
                                      float* __restrict__ dtok, float* __restrict__ dpos,
                                      int M, int T, int C) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = blockIdx.y * blockDim.y + threadIdx.y;
    if (m < M && j < C) {
        const float g = dh[static_cast<size_t>(m) * C + j];
        atomicAdd(&dtok[static_cast<size_t>(ids[m]) * C + j], g);
        atomicAdd(&dpos[static_cast<size_t>(m % T) * C + j], g);
    }
}

// Split the fused QKV weight gradient dWqkv[C,3C] back into the per-projection grads dWq/dWk/dWv
// [C,C] at their param-blob offsets (inverse of build_qkv_kernel). One thread per [C,C] element.
__global__ void split_dqkv_kernel(const float* __restrict__ dWqkv, float* __restrict__ dWq,
                                  float* __restrict__ dWk, float* __restrict__ dWv, int C) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < C * C) {
        const int p = idx / C, c = idx % C;
        const int row = p * 3 * C;
        dWq[idx] = dWqkv[row + c];
        dWk[idx] = dWqkv[row + C + c];
        dWv[idx] = dWqkv[row + 2 * C + c];
    }
}

// AdamW: block-reduce sum of grad^2 into a double accumulator (matches the CPU double-precision
// global-norm clip). Shared-mem reduction per block, one atomicAdd per block.
__global__ void grad_normsq_kernel(const float* __restrict__ grad, double* __restrict__ normsq, int n) {
    __shared__ double sm[256];
    const int tid = threadIdx.x;
    const int i   = blockIdx.x * blockDim.x + tid;
    double v = 0.0;
    if (i < n) { const double g = grad[i]; v = g * g; }
    sm[tid] = v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sm[tid] += sm[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(normsq, sm[0]);
}

// AdamW per-parameter update (op-for-op identical to AdamW::step on the CPU). gs = global grad
// clip scale, bc1/bc2 = bias corrections, decay = 0/1 mask (wd applies to matrices only).
__global__ void adam_step_kernel(float* __restrict__ p, const float* __restrict__ grad,
                                 float* __restrict__ m, float* __restrict__ vel,
                                 const float* __restrict__ decay, int n, float gs, float lr,
                                 float b1, float b2, float eps, float wd, float bc1, float bc2) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float g = grad[i] * gs;
    m[i]   = b1 * m[i]   + (1.f - b1) * g;
    vel[i] = b2 * vel[i] + (1.f - b2) * g * g;
    const float mhat = m[i] / bc1;
    const float vhat = vel[i] / bc2;
    p[i] -= lr * mhat / (sqrtf(vhat) + eps);
    p[i] -= lr * (decay[i] * wd) * p[i];
}

// --- Device-pointer launchers (no host alloc; drive the resident forward chain) ---
inline void ensure_stream() { if (!g_stream) cudaStreamCreate(&g_stream); }

// Invalidate the captured CUDA graph (defined below) -- a changed forward shape or GEMM math
// mode makes it stale, so it must be recaptured.
void invalidate_graph();

// Set the cuBLAS handle's math mode directly -- used to compare modes in the benchmark and to
// force a mode in the parity tests, independent of the baked knob. The mode is baked into a
// captured graph's algo, so changing it invalidates the graph.
inline void set_handle_tf32(bool on) {
    if (g_cublas) cublasSetMathMode(g_cublas, on ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH);
    invalidate_graph();
}
inline void apply_math_mode() { set_handle_tf32(CudaTf32::get()); }   // production default = baked knob
inline void ensure_cublas() {
    if (!g_cublas) {
        cublasCreate(&g_cublas);
        ensure_stream();
        cublasSetStream(g_cublas, g_stream);   // all GEMMs on the capture stream
        apply_math_mode();
    }
}
// Y[M,out] = X[M,in] . W[in,out] (+ bias) via cuBLAS. cublasSgemm is column-major, so we
// compute the column-major Y^T = W^T . X^T -- which, read back row-major, IS Y = X . W:
//   cublasSgemm(N,N, out,M,in, a, W,out, X,in, b, Y,out). Bias (no epilogue in the legacy
// API) is a cheap broadcast-add kernel afterwards. Same default stream as the kernels, so
// ordering is preserved.
// TODO(perf): migrate to cublasLt (cublasLtMatmul) with a fused bias+GELU epilogue to drop the
// separate bias_add/gelu launches. TODO(perf): at K=D_MODEL=96 these GEMMs are launch/occupancy
// bound -- FP16/BF16 storage + tensor cores (or batched-strided GEMM across layers) is the real
// lever; QKV fusion measured neutral at training scale (M=4096).
inline void launch_linear(const float* dX, const float* dW, const float* dB, float* dY,
                          int M, int in, int out) {
    ensure_cublas();
    const float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(g_cublas, CUBLAS_OP_N, CUBLAS_OP_N, out, M, in,
                &alpha, dW, out, dX, in, &beta, dY, out);
    if (dB) {
        const int n = M * out, block = 256;
        bias_add_kernel<<<(n + block - 1) / block, block, 0, g_stream>>>(dY, dB, M, out);
    }
}
inline void launch_rmsnorm(const float* dX, const float* dG, float* dY, int T, int C) {
    const int block = 64;
    rmsnorm_kernel<<<(T + block - 1) / block, block, 0, g_stream>>>(dX, dG, dY, T, C);
}
inline void launch_rmsnorm_train(const float* dX, const float* dG, float* dY, float* dRinv, int T, int C) {
    const int block = 64;
    rmsnorm_train_kernel<<<(T + block - 1) / block, block, 0, g_stream>>>(dX, dG, dY, dRinv, T, C);
}
inline void launch_gelu(const float* dX, float* dY, int n) {
    const int block = 256;
    gelu_kernel<<<(n + block - 1) / block, block, 0, g_stream>>>(dX, dY, n);
}
inline void launch_add(const float* dA, const float* dB, float* dC, int n) {
    const int block = 256;
    add_kernel<<<(n + block - 1) / block, block, 0, g_stream>>>(dA, dB, dC, n);
}
inline void launch_attn(const float* dQ, const float* dK, const float* dV, float* dOut,
                        int batch, int T, int C, int H, int in_stride) {
    const int block = 64;
    const int total = batch * H * T;
    attn_kernel<<<(total + block - 1) / block, block, 0, g_stream>>>(dQ, dK, dV, dOut, batch, T, C, H, in_stride);
}
inline void launch_attn_train(const float* dQ, const float* dK, const float* dV, float* dOut,
                              float* dP, int batch, int T, int C, int H, int in_stride) {
    const int block = 64;
    const int total = batch * H * T;
    attn_train_kernel<<<(total + block - 1) / block, block, 0, g_stream>>>(dQ, dK, dV, dOut, dP, batch, T, C, H, in_stride);
}

// Linear backward (mirrors the Op::Linear case): given the forward input X[M,in], weight W[in,out]
// and upstream grad dY[M,out], produce dX[M,in] = dY.W^T and dW[in,out] = X^T.dY via two cuBLAS
// GEMMs (same column-major trick as launch_linear), plus the bias column-sum. dW/dbias write
// straight into the param grad blob (beta=0; each weight is used once per forward). dX may be null
// for the bottom of the graph (no input grad needed).
inline void launch_linear_bwd(const float* dX_in, const float* dW_in, const float* dY,
                              float* dX, float* dW, float* dbias, int M, int in, int out) {
    const float alpha = 1.0f, beta = 0.0f;
    if (dX)
        cublasSgemm(g_cublas, CUBLAS_OP_T, CUBLAS_OP_N, in, M, out,
                    &alpha, dW_in, out, dY, out, &beta, dX, in);     // dX = dY . W^T
    cublasSgemm(g_cublas, CUBLAS_OP_N, CUBLAS_OP_T, out, in, M,
                &alpha, dY, out, dX_in, in, &beta, dW, out);         // dW = X^T . dY
    if (dbias) {
        const int block = 128;
        bias_grad_kernel<<<(out + block - 1) / block, block, 0, g_stream>>>(dY, dbias, M, out);
    }
}
inline void launch_rmsnorm_bwd(const float* x, const float* gamma, const float* rinv,
                               const float* dy, float* dx, float* dgamma, int rows, int C) {
    const int block = 64;
    rmsnorm_backward_kernel<<<(rows + block - 1) / block, block, 0, g_stream>>>(
        x, gamma, rinv, dy, dx, dgamma, rows, C);
}
inline void launch_gelu_bwd(const float* x, const float* dy, float* dx, int n) {
    const int block = 256;
    gelu_backward_kernel<<<(n + block - 1) / block, block, 0, g_stream>>>(x, dy, dx, n);
}
inline void launch_attn_bwd(const float* qkv, const float* P, const float* dout, float* dqkv,
                            int batch, int T, int C, int H, int in_stride) {
    const int block = 64;
    const int total = batch * H;
    attn_backward_kernel<<<(total + block - 1) / block, block, 0, g_stream>>>(
        qkv, qkv + C, qkv + 2 * C, P, dout, dqkv, dqkv + C, dqkv + 2 * C, batch, T, C, H, in_stride);
}

// Max minibatch the resident forward scratch is sized for: the CPU's tuned data-parallel
// width, so the GPU forward covers the same batch training uses. A forward requesting more
// windows than this is rejected.
constexpr int MAX_FWD_BATCH =
    DEFAULT_THREADS * DEFAULT_WINDOWS_PER_THREAD > 0 ? DEFAULT_THREADS * DEFAULT_WINDOWS_PER_THREAD : 8;

// Resident forward scratch: allocated once for the full M = MAX_FWD_BATCH * SEQ_LEN row
// count and reused by every sub0_cuda_forward, so the hot path has no per-call cudaMalloc
// churn. Freed by sub0_cuda_shutdown.
struct FwdScratch {
    int*   dids   = nullptr;
    float* h      = nullptr;
    float* a      = nullptr;
    float* qkv    = nullptr;            // fused [M, 3C] projections (q|k|v sub-blocks)
    float* att    = nullptr;
    float* proj   = nullptr;
    float* fbuf   = nullptr;
    float* ff1    = nullptr;
    float* gact   = nullptr;
    float* ff2    = nullptr;
    float* logits = nullptr;
    float* wqkv[N_LAYERS] = {};         // per-layer fused QKV weight [C, 3C], built once at upload
    bool   inited = false;
};
FwdScratch g_fwd;

int fwd_alloc() {
    if (g_fwd.inited) return 0;
    ensure_stream();                                  // device work + graph capture run on g_stream
    const size_t Mm = static_cast<size_t>(MAX_FWD_BATCH) * SEQ_LEN;
    const size_t MC = Mm * D_MODEL, MF = Mm * D_FF, MV = Mm * VOCAB;
    SUB0_CUDA_CHECK(cudaMalloc(&g_fwd.dids,   Mm * sizeof(int)));
    SUB0_CUDA_CHECK(cudaMalloc(&g_fwd.h,      MC * sizeof(float)));
    SUB0_CUDA_CHECK(cudaMalloc(&g_fwd.a,      MC * sizeof(float)));
    SUB0_CUDA_CHECK(cudaMalloc(&g_fwd.qkv,    Mm * 3 * D_MODEL * sizeof(float)));
    SUB0_CUDA_CHECK(cudaMalloc(&g_fwd.att,    MC * sizeof(float)));
    SUB0_CUDA_CHECK(cudaMalloc(&g_fwd.proj,   MC * sizeof(float)));
    SUB0_CUDA_CHECK(cudaMalloc(&g_fwd.fbuf,   MC * sizeof(float)));
    SUB0_CUDA_CHECK(cudaMalloc(&g_fwd.ff1,    MF * sizeof(float)));
    SUB0_CUDA_CHECK(cudaMalloc(&g_fwd.gact,   MF * sizeof(float)));
    SUB0_CUDA_CHECK(cudaMalloc(&g_fwd.ff2,    MC * sizeof(float)));
    SUB0_CUDA_CHECK(cudaMalloc(&g_fwd.logits, MV * sizeof(float)));
    for (int l = 0; l < N_LAYERS; ++l)
        SUB0_CUDA_CHECK(cudaMalloc(&g_fwd.wqkv[l], static_cast<size_t>(D_MODEL) * 3 * D_MODEL * sizeof(float)));
    g_fwd.inited = true;
    return 0;
}

void fwd_free() {
    cudaFree(g_fwd.dids);  cudaFree(g_fwd.h);    cudaFree(g_fwd.a);    cudaFree(g_fwd.qkv);
    cudaFree(g_fwd.att);   cudaFree(g_fwd.proj); cudaFree(g_fwd.fbuf); cudaFree(g_fwd.ff1);
    cudaFree(g_fwd.gact);  cudaFree(g_fwd.ff2);  cudaFree(g_fwd.logits);
    for (int l = 0; l < N_LAYERS; ++l) cudaFree(g_fwd.wqkv[l]);
    g_fwd = FwdScratch{};
}

// Resident TRAINING scratch (Phase 2d): the forward saves every activation the backward needs
// (no recompute), plus the gradient temporaries that thread the reverse pass. Sized once for the
// full M = MAX_FWD_BATCH * SEQ_LEN. Allocated lazily by train_alloc, freed by sub0_cuda_shutdown.
struct TrainScratch {
    // per-layer saved forward activations
    float* h_in [N_LAYERS] = {};   // [M,C] layer input (= rmsnorm1 input)
    float* rinv1[N_LAYERS] = {};   // [M]   rmsnorm1 reciprocal-rms
    float* a    [N_LAYERS] = {};   // [M,C] rmsnorm1 output (= qkv input)
    float* qkv  [N_LAYERS] = {};   // [M,3C] fused q|k|v (= attn input)
    float* P    [N_LAYERS] = {};   // [batch,H,T,T] attention weights
    float* att  [N_LAYERS] = {};   // [M,C] attention output (= Wo input)
    float* h_mid[N_LAYERS] = {};   // [M,C] after residual-1 (= rmsnorm2 input)
    float* rinv2[N_LAYERS] = {};   // [M]   rmsnorm2 reciprocal-rms
    float* fbuf [N_LAYERS] = {};   // [M,C] rmsnorm2 output (= W1 input)
    float* ff1  [N_LAYERS] = {};   // [M,F] pre-GELU (= GELU input)
    float* gact [N_LAYERS] = {};   // [M,F] GELU output (= W2 input)
    // final block
    float* h_final = nullptr;      // [M,C] last residual stream (= rmsnorm_f input)
    float* rinv_f  = nullptr;      // [M]
    float* a_final = nullptr;      // [M,C] rmsnorm_f output (= lm_head input)
    float* logits  = nullptr;      // [M,V]
    // gradient temporaries (reused across layers)
    float* dh      = nullptr;      // [M,C] running residual-stream grad
    float* da      = nullptr;      // [M,C]
    float* dqkv    = nullptr;      // [M,3C]
    float* datt    = nullptr;      // [M,C]
    float* dfbuf   = nullptr;      // [M,C]
    float* dff1    = nullptr;      // [M,F]
    float* dgact   = nullptr;      // [M,F]
    float* dlogits = nullptr;      // [M,V]
    float* dwqkv   = nullptr;      // [C,3C] fused QKV weight-grad temp
    double* loss   = nullptr;      // [1] accumulated cross-entropy (device)
    int*   dtargets = nullptr;     // [M] next-token targets for cross-entropy
    bool   inited  = false;
};
TrainScratch g_tr;

int train_alloc() {
    if (g_tr.inited) return 0;
    ensure_stream();
    const size_t Mm = static_cast<size_t>(MAX_FWD_BATCH) * SEQ_LEN;
    const size_t MC = Mm * D_MODEL, MF = Mm * D_FF, MV = Mm * VOCAB;
    const size_t PB = static_cast<size_t>(MAX_FWD_BATCH) * N_HEADS * SEQ_LEN * SEQ_LEN;
    for (int l = 0; l < N_LAYERS; ++l) {
        SUB0_CUDA_CHECK(cudaMalloc(&g_tr.h_in[l],  MC * sizeof(float)));
        SUB0_CUDA_CHECK(cudaMalloc(&g_tr.rinv1[l], Mm * sizeof(float)));
        SUB0_CUDA_CHECK(cudaMalloc(&g_tr.a[l],     MC * sizeof(float)));
        SUB0_CUDA_CHECK(cudaMalloc(&g_tr.qkv[l],   Mm * 3 * D_MODEL * sizeof(float)));
        SUB0_CUDA_CHECK(cudaMalloc(&g_tr.P[l],     PB * sizeof(float)));
        SUB0_CUDA_CHECK(cudaMalloc(&g_tr.att[l],   MC * sizeof(float)));
        SUB0_CUDA_CHECK(cudaMalloc(&g_tr.h_mid[l], MC * sizeof(float)));
        SUB0_CUDA_CHECK(cudaMalloc(&g_tr.rinv2[l], Mm * sizeof(float)));
        SUB0_CUDA_CHECK(cudaMalloc(&g_tr.fbuf[l],  MC * sizeof(float)));
        SUB0_CUDA_CHECK(cudaMalloc(&g_tr.ff1[l],   MF * sizeof(float)));
        SUB0_CUDA_CHECK(cudaMalloc(&g_tr.gact[l],  MF * sizeof(float)));
    }
    SUB0_CUDA_CHECK(cudaMalloc(&g_tr.h_final, MC * sizeof(float)));
    SUB0_CUDA_CHECK(cudaMalloc(&g_tr.rinv_f,  Mm * sizeof(float)));
    SUB0_CUDA_CHECK(cudaMalloc(&g_tr.a_final, MC * sizeof(float)));
    SUB0_CUDA_CHECK(cudaMalloc(&g_tr.logits,  MV * sizeof(float)));
    SUB0_CUDA_CHECK(cudaMalloc(&g_tr.dh,      MC * sizeof(float)));
    SUB0_CUDA_CHECK(cudaMalloc(&g_tr.da,      MC * sizeof(float)));
    SUB0_CUDA_CHECK(cudaMalloc(&g_tr.dqkv,    Mm * 3 * D_MODEL * sizeof(float)));
    SUB0_CUDA_CHECK(cudaMalloc(&g_tr.datt,    MC * sizeof(float)));
    SUB0_CUDA_CHECK(cudaMalloc(&g_tr.dfbuf,   MC * sizeof(float)));
    SUB0_CUDA_CHECK(cudaMalloc(&g_tr.dff1,    MF * sizeof(float)));
    SUB0_CUDA_CHECK(cudaMalloc(&g_tr.dgact,   MF * sizeof(float)));
    SUB0_CUDA_CHECK(cudaMalloc(&g_tr.dlogits, MV * sizeof(float)));
    SUB0_CUDA_CHECK(cudaMalloc(&g_tr.dwqkv,   static_cast<size_t>(D_MODEL) * 3 * D_MODEL * sizeof(float)));
    SUB0_CUDA_CHECK(cudaMalloc(&g_tr.loss,    sizeof(double)));
    SUB0_CUDA_CHECK(cudaMalloc(&g_tr.dtargets, Mm * sizeof(int)));
    g_tr.inited = true;
    return 0;
}

void train_free() {
    for (int l = 0; l < N_LAYERS; ++l) {
        cudaFree(g_tr.h_in[l]);  cudaFree(g_tr.rinv1[l]); cudaFree(g_tr.a[l]);    cudaFree(g_tr.qkv[l]);
        cudaFree(g_tr.P[l]);     cudaFree(g_tr.att[l]);   cudaFree(g_tr.h_mid[l]); cudaFree(g_tr.rinv2[l]);
        cudaFree(g_tr.fbuf[l]);  cudaFree(g_tr.ff1[l]);   cudaFree(g_tr.gact[l]);
    }
    cudaFree(g_tr.h_final); cudaFree(g_tr.rinv_f); cudaFree(g_tr.a_final); cudaFree(g_tr.logits);
    cudaFree(g_tr.dh);   cudaFree(g_tr.da);   cudaFree(g_tr.dqkv); cudaFree(g_tr.datt);
    cudaFree(g_tr.dfbuf); cudaFree(g_tr.dff1); cudaFree(g_tr.dgact); cudaFree(g_tr.dlogits);
    cudaFree(g_tr.dwqkv); cudaFree(g_tr.loss); cudaFree(g_tr.dtargets);
    g_tr = TrainScratch{};
}


// Device-side forward chain (no host transfers): assumes g_fwd.dids is populated and the
// params uploaded; writes logits into g_fwd.logits over M = batch*T rows. Shared by
// sub0_cuda_forward and the benchmark so both run the IDENTICAL kernel sequence.
void forward_device(int batch, int T) {
    const auto&  L  = sub0::PARAM_LAYOUT;
    const int    C  = D_MODEL, F = D_FF, V = VOCAB, H = N_HEADS;
    const int    M  = batch * T;
    const int    MC = M * C, MF = M * F;
    float* const base = g_dev_params;

    const float* tok_emb = base + L[0].off;
    const float* pos_emb = base + L[1].off;
    const int    fi      = 2 + 10 * N_LAYERS;          // index of ln_f
    const float* ln_f    = base + L[fi + 0].off;
    const float* lm_head = base + L[fi + 1].off;
    const float* lm_bias = base + L[fi + 2].off;

    float* h    = g_fwd.h;
    float* a    = g_fwd.a;
    float* qkv  = g_fwd.qkv;
    float* att  = g_fwd.att;
    float* proj = g_fwd.proj;
    float* fbuf = g_fwd.fbuf;
    float* ff1  = g_fwd.ff1;
    float* gact = g_fwd.gact;
    float* ff2  = g_fwd.ff2;

    // h = tok_emb[ids] + pos_emb  (over all M rows; position = row % T)
    {
        const dim3 block(16, 16);
        const dim3 grid((C + block.x - 1) / block.x, (M + block.y - 1) / block.y);
        embed_add_kernel<<<grid, block, 0, g_stream>>>(tok_emb, pos_emb, g_fwd.dids, h, M, T, C);
    }
    for (int l = 0; l < N_LAYERS; ++l) {
        const int    b0  = 2 + 10 * l;
        const float* ln1 = base + L[b0 + 0].off;
        const float* ln2 = base + L[b0 + 1].off;
        const float* Wo  = base + L[b0 + 5].off;
        const float* W1  = base + L[b0 + 6].off;
        const float* b1  = base + L[b0 + 7].off;
        const float* W2  = base + L[b0 + 8].off;
        const float* b2  = base + L[b0 + 9].off;

        launch_rmsnorm(h, ln1, a, M, C);                      // a = rmsnorm(h, ln1)
        launch_linear(a, g_fwd.wqkv[l], nullptr, qkv, M, C, 3 * C);          // fused qkv = a . [Wq|Wk|Wv]
        launch_attn(qkv, qkv + C, qkv + 2 * C, att, batch, T, C, H, 3 * C);  // q/k/v sub-blocks (stride 3C)
        launch_linear(att, Wo, nullptr, proj, M, C, C);      // proj = att . Wo
        launch_add(h, proj, h, MC);                          // h = h + proj
        launch_rmsnorm(h, ln2, fbuf, M, C);                  // f = rmsnorm(h, ln2)
        launch_linear(fbuf, W1, b1, ff1, M, C, F);        // ff1 = f . W1 + b1
        launch_gelu(ff1, gact, MF);                       // gelu
        launch_linear(gact, W2, b2, ff2, M, F, C);        // ff2 = gelu . W2 + b2
        launch_add(h, ff2, h, MC);                        // h = h + ff2
    }
    launch_rmsnorm(h, ln_f, a, M, C);                     // a = rmsnorm(h, ln_f)
    launch_linear(a, lm_head, lm_bias, g_fwd.logits, M, C, V);  // logits = a . lm_head + lm_bias
}

// Drop any captured graph (shape or math-mode change -> recapture on the next forward).
void invalidate_graph() {
    if (g_graph_exec) { cudaGraphExecDestroy(g_graph_exec); g_graph_exec = nullptr; }
    g_graph_batch = -1;
    g_graph_T     = -1;
}

// Capture forward_device(batch,T) into an executable CUDA graph (once per shape + math mode),
// collapsing the ~100 per-forward kernel launches into a single graph launch. The stream must
// be idle when called (callers streamSync between forwards). A warmup pass first lets cuBLAS do
// any lazy allocation OUTSIDE capture (cudaMalloc during capture is illegal). Returns 0 on ok.
int capture_graph(int batch, int T) {
    if (g_graph_exec && g_graph_batch == batch && g_graph_T == T) return 0;
    invalidate_graph();
    ensure_cublas();
    SUB0_CUDA_CHECK(cudaMemsetAsync(g_fwd.dids, 0, static_cast<size_t>(batch) * T * sizeof(int), g_stream));
    forward_device(batch, T);                            // warmup (safe ids=0): cuBLAS lazy alloc
    SUB0_CUDA_CHECK(cudaStreamSynchronize(g_stream));
    cudaGraph_t graph = nullptr;
    SUB0_CUDA_CHECK(cudaStreamBeginCapture(g_stream, cudaStreamCaptureModeThreadLocal));
    forward_device(batch, T);
    SUB0_CUDA_CHECK(cudaStreamEndCapture(g_stream, &graph));
    SUB0_CUDA_CHECK(cudaGraphInstantiate(&g_graph_exec, graph, 0));
    cudaGraphDestroy(graph);
    g_graph_batch = batch;
    g_graph_T     = T;
    return 0;
}

// Materialize the per-layer fused QKV weight (g_fwd.wqkv[l] = [Wq|Wk|Wv]) from the uploaded
// param blob. Called once whenever the weights change (upload/load). TODO(qkv-train): once the
// device backend trains in place, the optimizer updates Wq/Wk/Wv in g_dev_params each step, so
// this rebuild must run inside the step (cheap: N_LAYERS*C*C threads) -- or store QKV fused
// natively and slice it back for the layout-table serialization.
void build_qkv_weights() {
    const auto& L = sub0::PARAM_LAYOUT;
    const int   C = D_MODEL;
    const int   n = C * C, block = 256, grid = (n + block - 1) / block;
    for (int l = 0; l < N_LAYERS; ++l) {
        const int    b0 = 2 + 10 * l;
        const float* Wq = g_dev_params + L[b0 + 2].off;
        const float* Wk = g_dev_params + L[b0 + 3].off;
        const float* Wv = g_dev_params + L[b0 + 4].off;
        build_qkv_kernel<<<grid, block, 0, g_stream>>>(Wq, Wk, Wv, g_fwd.wqkv[l], C);
    }
}

// ============================================================================
//  Training forward + backward + AdamW (Phase 2d) -- mirrors backend_cpu.cpp
// ============================================================================

// Forward pass that SAVES every activation the backward needs into g_tr (no recompute). Same op
// sequence as forward_device but writes to the per-layer training buffers and uses the train
// variants of rmsnorm/attn (which also save rinv / the softmax weights P). Assumes g_fwd.dids is
// populated, params uploaded and g_fwd.wqkv[] built. Eager (not graph-captured): the backward
// reads these buffers, so the whole step runs as plain stream-ordered launches.
// TODO(perf): once stable, CUDA-graph-capture the whole fwd+bwd+AdamW step (topology is static)
// to collapse the ~200 launches -- the training-step analogue of the inference graph.
// TODO(mem): the saved-activation scratch is ~1.5GB at MAX_FWD_BATCH; gradient checkpointing
// (recompute a/qkv/att in backward) or FP16 activation storage would cut this sharply.
void forward_train(int batch, int T) {
    const auto&  L  = sub0::PARAM_LAYOUT;
    const int    C  = D_MODEL, F = D_FF, V = VOCAB, H = N_HEADS;
    const int    M  = batch * T;
    const int    MC = M * C, MF = M * F;
    float* const base = g_dev_params;

    const float* tok_emb = base + L[0].off;
    const float* pos_emb = base + L[1].off;
    const int    fi      = 2 + 10 * N_LAYERS;
    const float* ln_f    = base + L[fi + 0].off;
    const float* lm_head = base + L[fi + 1].off;
    const float* lm_bias = base + L[fi + 2].off;

    {   // h_in[0] = tok_emb[ids] + pos_emb
        const dim3 block(16, 16);
        const dim3 grid((C + block.x - 1) / block.x, (M + block.y - 1) / block.y);
        embed_add_kernel<<<grid, block, 0, g_stream>>>(tok_emb, pos_emb, g_fwd.dids, g_tr.h_in[0], M, T, C);
    }
    for (int l = 0; l < N_LAYERS; ++l) {
        const int    b0  = 2 + 10 * l;
        const float* ln1 = base + L[b0 + 0].off;
        const float* ln2 = base + L[b0 + 1].off;
        const float* Wo  = base + L[b0 + 5].off;
        const float* W1  = base + L[b0 + 6].off;
        const float* b1  = base + L[b0 + 7].off;
        const float* W2  = base + L[b0 + 8].off;
        const float* b2  = base + L[b0 + 9].off;
        float* const hin  = g_tr.h_in[l];
        float* const hmid = g_tr.h_mid[l];
        float* const next = (l + 1 < N_LAYERS) ? g_tr.h_in[l + 1] : g_tr.h_final;

        launch_rmsnorm_train(hin, ln1, g_tr.a[l], g_tr.rinv1[l], M, C);              // a = rmsnorm(hin,ln1)
        launch_linear(g_tr.a[l], g_fwd.wqkv[l], nullptr, g_tr.qkv[l], M, C, 3 * C);  // fused qkv
        launch_attn_train(g_tr.qkv[l], g_tr.qkv[l] + C, g_tr.qkv[l] + 2 * C,
                          g_tr.att[l], g_tr.P[l], batch, T, C, H, 3 * C);            // att + save P
        launch_linear(g_tr.att[l], Wo, nullptr, hmid, M, C, C);                      // hmid = proj
        launch_add(hin, hmid, hmid, MC);                                            // hmid = hin + proj
        launch_rmsnorm_train(hmid, ln2, g_tr.fbuf[l], g_tr.rinv2[l], M, C);         // f = rmsnorm(hmid,ln2)
        launch_linear(g_tr.fbuf[l], W1, b1, g_tr.ff1[l], M, C, F);                  // ff1 = f.W1 + b1
        launch_gelu(g_tr.ff1[l], g_tr.gact[l], MF);                                 // gelu
        launch_linear(g_tr.gact[l], W2, b2, next, M, F, C);                         // next = ff2 = gelu.W2 + b2
        launch_add(hmid, next, next, MC);                                           // next = hmid + ff2
    }
    launch_rmsnorm_train(g_tr.h_final, ln_f, g_tr.a_final, g_tr.rinv_f, M, C);      // a_final = rmsnorm(h,ln_f)
    launch_linear(g_tr.a_final, lm_head, lm_bias, g_tr.logits, M, C, V);           // logits
}

// Reverse pass: consumes the saved activations, writes the reduced gradient into g_dev_grad and
// accumulates the mean cross-entropy into g_tr.loss. dh threads the residual stream; the rmsnorm
// backward kernels accumulate into it (residual skip + through-norm paths), exactly like the CPU
// tape walk. Weight/bias grads are written straight to their PARAM_LAYOUT offsets (each weight is
// used once per forward). Loss scaling invM = 1/M makes the result equal the CPU train_batch grad.
void backward_device(int batch, int T) {
    const auto&  L  = sub0::PARAM_LAYOUT;
    const int    C  = D_MODEL, F = D_FF, V = VOCAB, H = N_HEADS;
    const int    M  = batch * T;
    const int    MC = M * C, MF = M * F;
    float* const pb = g_dev_params;
    float* const gb = g_dev_grad;
    const int    fi = 2 + 10 * N_LAYERS;

    SUB0_CUDA_CHECK_VOID(cudaMemsetAsync(gb, 0, sub0::PARAM_FLOATS * sizeof(float), g_stream));
    SUB0_CUDA_CHECK_VOID(cudaMemsetAsync(g_tr.loss, 0, sizeof(double), g_stream));

    // cross-entropy: dlogits = (1/M)(p - onehot), loss += -log p[target]
    {
        const int block = 256;
        ce_backward_kernel<<<(M + block - 1) / block, block, 0, g_stream>>>(
            g_tr.logits, g_tr.dtargets, g_tr.dlogits, g_tr.loss, M, V, 1.0f / static_cast<float>(M));
    }
    // lm_head: dW/dbias -> grad blob, da = grad into a_final
    launch_linear_bwd(g_tr.a_final, pb + L[fi + 1].off, g_tr.dlogits, g_tr.da,
                      gb + L[fi + 1].off, gb + L[fi + 2].off, M, C, V);
    // rmsnorm_f: dh starts here (grad into h_final)
    SUB0_CUDA_CHECK_VOID(cudaMemsetAsync(g_tr.dh, 0, static_cast<size_t>(MC) * sizeof(float), g_stream));
    launch_rmsnorm_bwd(g_tr.h_final, pb + L[fi + 0].off, g_tr.rinv_f, g_tr.da, g_tr.dh,
                       gb + L[fi + 0].off, M, C);

    for (int l = N_LAYERS - 1; l >= 0; --l) {
        const int b0 = 2 + 10 * l;
        // ff residual: dh = grad into layer output = d(ff2). W2 backward (input gact)
        launch_linear_bwd(g_tr.gact[l], pb + L[b0 + 8].off, g_tr.dh, g_tr.dgact,
                          gb + L[b0 + 8].off, gb + L[b0 + 9].off, M, F, C);
        launch_gelu_bwd(g_tr.ff1[l], g_tr.dgact, g_tr.dff1, MF);                    // dff1
        launch_linear_bwd(g_tr.fbuf[l], pb + L[b0 + 6].off, g_tr.dff1, g_tr.dfbuf,
                          gb + L[b0 + 6].off, gb + L[b0 + 7].off, M, C, F);          // W1 backward
        launch_rmsnorm_bwd(g_tr.h_mid[l], pb + L[b0 + 1].off, g_tr.rinv2[l], g_tr.dfbuf,
                           g_tr.dh, gb + L[b0 + 1].off, M, C);                       // dh += -> d(h_mid)
        // proj residual: dh = grad into h_mid = d(proj). Wo backward (input att)
        launch_linear_bwd(g_tr.att[l], pb + L[b0 + 5].off, g_tr.dh, g_tr.datt,
                          gb + L[b0 + 5].off, nullptr, M, C, C);
        SUB0_CUDA_CHECK_VOID(cudaMemsetAsync(g_tr.dqkv, 0, static_cast<size_t>(M) * 3 * C * sizeof(float), g_stream));
        launch_attn_bwd(g_tr.qkv[l], g_tr.P[l], g_tr.datt, g_tr.dqkv, batch, T, C, H, 3 * C);
        // qkv backward (input a): da = grad into a, dWqkv -> split into dWq/dWk/dWv
        launch_linear_bwd(g_tr.a[l], g_fwd.wqkv[l], g_tr.dqkv, g_tr.da,
                          g_tr.dwqkv, nullptr, M, C, 3 * C);
        {
            const int n = C * C, block = 256;
            split_dqkv_kernel<<<(n + block - 1) / block, block, 0, g_stream>>>(
                g_tr.dwqkv, gb + L[b0 + 2].off, gb + L[b0 + 3].off, gb + L[b0 + 4].off, C);
        }
        launch_rmsnorm_bwd(g_tr.h_in[l], pb + L[b0 + 0].off, g_tr.rinv1[l], g_tr.da,
                           g_tr.dh, gb + L[b0 + 0].off, M, C);                       // dh += -> d(h_in)
    }
    // embed backward: scatter dh into tok_emb / pos_emb grads
    {
        const dim3 block(16, 16);
        const dim3 grid((C + block.x - 1) / block.x, (M + block.y - 1) / block.y);
        embed_backward_kernel<<<grid, block, 0, g_stream>>>(
            g_tr.dh, g_fwd.dids, gb + L[0].off, gb + L[1].off, M, T, C);
    }
}

// AdamW state: the reduced grad, the two moments (zeroed), the 0/1 decay mask (built from
// PARAM_LAYOUT) and the double norm accumulator. Allocated lazily alongside the train scratch.
int opt_alloc() {
    if (g_dev_grad) return 0;
    ensure_stream();
    const size_t n = sub0::PARAM_FLOATS;
    SUB0_CUDA_CHECK(cudaMalloc(&g_dev_grad,  n * sizeof(float)));
    SUB0_CUDA_CHECK(cudaMalloc(&g_dev_m,     n * sizeof(float)));
    SUB0_CUDA_CHECK(cudaMalloc(&g_dev_vel,   n * sizeof(float)));
    SUB0_CUDA_CHECK(cudaMalloc(&g_dev_decay, n * sizeof(float)));
    SUB0_CUDA_CHECK(cudaMalloc(&g_dev_normsq, sizeof(double)));
    SUB0_CUDA_CHECK(cudaMemset(g_dev_m,   0, n * sizeof(float)));
    SUB0_CUDA_CHECK(cudaMemset(g_dev_vel, 0, n * sizeof(float)));
    // weight-decay mask: 1 for matrices (PARAM_LAYOUT decay flag), 0 for biases/norms.
    std::vector<float> mask(n, 0.0f);
    const auto& L = sub0::PARAM_LAYOUT;
    for (const auto& pv : L) {
        const float d = pv.decay ? 1.0f : 0.0f;
        const size_t cnt = static_cast<size_t>(pv.rows) * pv.cols;
        for (size_t i = 0; i < cnt; ++i) mask[pv.off + i] = d;
    }
    SUB0_CUDA_CHECK(cudaMemcpy(g_dev_decay, mask.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    return 0;
}

void opt_free() {
    cudaFree(g_dev_grad);  cudaFree(g_dev_m);    cudaFree(g_dev_vel);
    cudaFree(g_dev_decay); cudaFree(g_dev_normsq);
    g_dev_grad = g_dev_m = g_dev_vel = g_dev_decay = nullptr;
    g_dev_normsq = nullptr;
}

// One AdamW step over the whole param blob, op-for-op identical to AdamW::step on the CPU: global
// grad L2 clip (double accumulation), then the per-parameter moment update with the decay mask.
// `t` is the post-increment step counter (>=1) the host optimizer drives. Rebuilds the fused QKV
// weights (Wq/Wk/Wv just changed) and invalidates the inference graph afterwards.
void device_adam_step(float lr, long t, float b1, float b2, float eps, float wd, float clip) {
    const int n = static_cast<int>(sub0::PARAM_FLOATS);
    const int block = 256, grid = (n + block - 1) / block;
    SUB0_CUDA_CHECK_VOID(cudaMemsetAsync(g_dev_normsq, 0, sizeof(double), g_stream));
    grad_normsq_kernel<<<grid, block, 0, g_stream>>>(g_dev_grad, g_dev_normsq, n);
    double normsq = 0.0;
    SUB0_CUDA_CHECK_VOID(cudaMemcpyAsync(&normsq, g_dev_normsq, sizeof(double), cudaMemcpyDeviceToHost, g_stream));
    SUB0_CUDA_CHECK_VOID(cudaStreamSynchronize(g_stream));
    const float norm = static_cast<float>(std::sqrt(normsq));
    const float gs   = (norm > clip) ? clip / (norm + 1e-6f) : 1.0f;
    const float bc1  = 1.0f - std::pow(b1, static_cast<float>(t));
    const float bc2  = 1.0f - std::pow(b2, static_cast<float>(t));
    adam_step_kernel<<<grid, block, 0, g_stream>>>(g_dev_params, g_dev_grad, g_dev_m, g_dev_vel,
                                                   g_dev_decay, n, gs, lr, b1, b2, eps, wd, bc1, bc2);
    build_qkv_weights();      // Wq/Wk/Wv changed -> refresh the fused inference/train weight
    invalidate_graph();       // params changed -> recapture the forward graph on next inference
}

}  // namespace

// Self-test: confirm the toolchain built device code for THIS GPU and that H2D, kernel
// launch and D2H all work end to end on a buffer the size of the model's parameter blob
// (PARAM_FLOATS). Returns 0 on success. Exposed via the C ABI so the clang-built host
// driver can call it across the DLL boundary.
SUB0_CUDA_API int sub0_cuda_selftest() {
    int dev = 0;
    SUB0_CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop{};
    SUB0_CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    std::printf("cuda selftest: device %d = %s, sm_%d%d, %.1f GB VRAM (built for sm_%d)\n",
                dev, prop.name, prop.major, prop.minor,
                static_cast<double>(prop.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0),
                CUDA_ARCH);

    const int    n     = static_cast<int>(sub0::PARAM_FLOATS);
    const size_t bytes = static_cast<size_t>(n) * sizeof(float);
    std::vector<float> hx(static_cast<size_t>(n), 1.0f);
    std::vector<float> hy(static_cast<size_t>(n), 2.0f);

    float* dx = nullptr;
    float* dy = nullptr;
    SUB0_CUDA_CHECK(cudaMalloc(&dx, bytes));
    SUB0_CUDA_CHECK(cudaMalloc(&dy, bytes));
    SUB0_CUDA_CHECK(cudaMemcpy(dx, hx.data(), bytes, cudaMemcpyHostToDevice));
    SUB0_CUDA_CHECK(cudaMemcpy(dy, hy.data(), bytes, cudaMemcpyHostToDevice));

    const int threads = 256;
    const int blocks  = (n + threads - 1) / threads;
    axpy_kernel<<<blocks, threads>>>(3.0f, dx, dy, n);   // y = 3*x + y -> 5
    SUB0_CUDA_CHECK(cudaGetLastError());
    SUB0_CUDA_CHECK(cudaDeviceSynchronize());

    SUB0_CUDA_CHECK(cudaMemcpy(hy.data(), dy, bytes, cudaMemcpyDeviceToHost));
    cudaFree(dx);
    cudaFree(dy);

    int bad = 0;
    for (int i = 0; i < n; ++i)
        if (hy[static_cast<size_t>(i)] != 5.0f) ++bad;
    if (bad) {
        std::fprintf(stderr, "cuda selftest: FAIL (%d/%d elements mismatched)\n", bad, n);
        return 2;
    }
    std::printf("cuda selftest: OK (%d elements, y = 3*x + y == 5)\n", n);
    return 0;
}

// ============================================================================
//  Device parameter mirror (the device half of the sync_params hooks)
// ============================================================================
// Phase 2b: allocate the device weight blob and move it host<->device. When this file
// becomes the active backend these back sync_params_to_device/host directly; for now the
// parity tests drive them. Idempotent init; upload auto-inits.

SUB0_CUDA_API int sub0_cuda_init() {
    if (g_dev_params) return 0;
    SUB0_CUDA_CHECK(cudaMalloc(&g_dev_params, sub0::PARAM_FLOATS * sizeof(float)));
    return 0;
}

SUB0_CUDA_API void sub0_cuda_shutdown() {
    invalidate_graph();
    if (g_dev_params) cudaFree(g_dev_params);
    g_dev_params = nullptr;
    fwd_free();                                  // release the resident forward scratch too
    train_free();                                // and the training scratch (Phase 2d)
    opt_free();                                  // and the optimizer state
    if (g_cublas) { cublasDestroy(g_cublas); g_cublas = nullptr; }
    if (g_stream) { cudaStreamDestroy(g_stream); g_stream = nullptr; }
}

SUB0_CUDA_API int sub0_cuda_upload_params(const float* host) {
    if (!g_dev_params) { const int r = sub0_cuda_init(); if (r) return r; }
    if (fwd_alloc()) return 1;                   // ensure the fused-QKV weight buffers exist
    SUB0_CUDA_CHECK(cudaMemcpyAsync(g_dev_params, host, sub0::PARAM_FLOATS * sizeof(float),
                                    cudaMemcpyHostToDevice, g_stream));
    build_qkv_weights();                         // rebuild the fused [Wq|Wk|Wv] from the new weights
    invalidate_graph();                          // weights changed -> recapture the forward graph
    SUB0_CUDA_CHECK(cudaStreamSynchronize(g_stream));
    return 0;
}

SUB0_CUDA_API int sub0_cuda_download_params(float* host) {
    if (!g_dev_params) return 1;
    SUB0_CUDA_CHECK(cudaMemcpy(host, g_dev_params, sub0::PARAM_FLOATS * sizeof(float),
                               cudaMemcpyDeviceToHost));
    return 0;
}

// ============================================================================
//  Dense linear (parity building block)
// ============================================================================
// Y[T,out] = X[T,in] . W[in,out] (+ bias). Takes host pointers, runs on the device and
// returns the result -- a self-contained kernel the CPU-parity test drives. Per-call
// device allocations are fine for the test harness; the real forward keeps them resident.
// TODO(robustness): the SUB0_CUDA_CHECK early-returns leak any already-allocated dX/dW/dY/dB.
// Acceptable for the test harness (process exits); add RAII/goto-cleanup if this ever ships.
SUB0_CUDA_API int sub0_cuda_linear(const float* X, int T, int in, int out,
                                   const float* W, const float* bias, float* Y) {
    const size_t xb = static_cast<size_t>(T) * in * sizeof(float);
    const size_t wb = static_cast<size_t>(in) * out * sizeof(float);
    const size_t yb = static_cast<size_t>(T) * out * sizeof(float);
    const size_t bb = static_cast<size_t>(out) * sizeof(float);

    float* dX = nullptr;
    float* dW = nullptr;
    float* dY = nullptr;
    float* dB = nullptr;
    SUB0_CUDA_CHECK(cudaMalloc(&dX, xb));
    SUB0_CUDA_CHECK(cudaMalloc(&dW, wb));
    SUB0_CUDA_CHECK(cudaMalloc(&dY, yb));
    if (bias) SUB0_CUDA_CHECK(cudaMalloc(&dB, bb));
    SUB0_CUDA_CHECK(cudaMemcpy(dX, X, xb, cudaMemcpyHostToDevice));
    SUB0_CUDA_CHECK(cudaMemcpy(dW, W, wb, cudaMemcpyHostToDevice));
    if (bias) SUB0_CUDA_CHECK(cudaMemcpy(dB, bias, bb, cudaMemcpyHostToDevice));

    launch_linear(dX, dW, dB, dY, T, in, out);   // cuBLAS GEMM (+ bias) -- same path as the forward
    SUB0_CUDA_CHECK(cudaGetLastError());
    SUB0_CUDA_CHECK(cudaDeviceSynchronize());

    SUB0_CUDA_CHECK(cudaMemcpy(Y, dY, yb, cudaMemcpyDeviceToHost));
    cudaFree(dX);
    cudaFree(dW);
    cudaFree(dY);
    if (dB) cudaFree(dB);
    return 0;
}

// ============================================================================
//  Full forward pass (single window) -> logits, CPU-parity gated
// ============================================================================
// Runs the same op sequence as Model::forward over a BATCH of windows: M = batch*T rows.
// Every Linear / RMSNorm / GELU / residual runs over all M rows at once; attention is
// per-window causal (windows independent). Requires sub0_cuda_upload_params() first. ids is
// [batch*T]; out_logits receives [batch*T, VOCAB]. Scratch is resident.
// NOTE: this is the inference (graph-captured) path. The training path -- forward_train (saves
// activations) + backward_device + device_adam_step -- lives below (sub0_cuda_train_step), gated
// by the grad/AdamW parity tests. TODO(phase-2e): wire sub0_cuda_train_step into the train stage
// (implement the sub0:: API in a GPU engine) so SUB0_COMPUTE=GPU trains end-to-end on device;
// today the train stage still runs the CPU backend.
SUB0_CUDA_API int sub0_cuda_forward(const int* ids, int batch, int T, float* out_logits) {
    if (!g_dev_params) return 1;
    if (batch < 1 || batch > MAX_FWD_BATCH || T < 1 || T > SEQ_LEN) return 1;
    if (fwd_alloc()) return 1;
    ensure_cublas();
    if (capture_graph(batch, T)) return 1;       // capture once per shape (replay thereafter)
    const int M = batch * T;
    // H2D ids -> graph replay -> D2H logits, all ordered on the capture stream.
    SUB0_CUDA_CHECK(cudaMemcpyAsync(g_fwd.dids, ids, static_cast<size_t>(M) * sizeof(int),
                                    cudaMemcpyHostToDevice, g_stream));
    SUB0_CUDA_CHECK(cudaGraphLaunch(g_graph_exec, g_stream));
    SUB0_CUDA_CHECK(cudaMemcpyAsync(out_logits, g_fwd.logits, static_cast<size_t>(M) * VOCAB * sizeof(float),
                                    cudaMemcpyDeviceToHost, g_stream));
    SUB0_CUDA_CHECK(cudaStreamSynchronize(g_stream));
    return 0;
}

// ============================================================================
//  Training step: forward + backward + AdamW on device (Phase 2d), parity-gated
// ============================================================================
// Shared driver: upload ids+targets, run the activation-saving forward and the reverse pass into
// g_dev_grad, sync, and read back the mean cross-entropy. Requires upload_params() first.
static int run_fwd_bwd(const int* ids, const int* targets, int batch, int T, double* out_loss) {
    if (!g_dev_params) return 1;
    if (batch < 1 || batch > MAX_FWD_BATCH || T < 1 || T > SEQ_LEN) return 1;
    if (fwd_alloc() || train_alloc() || opt_alloc()) return 1;
    ensure_cublas();
    set_handle_tf32(CudaTf32::get());            // training uses the baked math mode
    const int M = batch * T;
    SUB0_CUDA_CHECK(cudaMemcpyAsync(g_fwd.dids, ids, static_cast<size_t>(M) * sizeof(int),
                                    cudaMemcpyHostToDevice, g_stream));
    SUB0_CUDA_CHECK(cudaMemcpyAsync(g_tr.dtargets, targets, static_cast<size_t>(M) * sizeof(int),
                                    cudaMemcpyHostToDevice, g_stream));
    forward_train(batch, T);
    backward_device(batch, T);
    double loss = 0.0;
    SUB0_CUDA_CHECK(cudaMemcpyAsync(&loss, g_tr.loss, sizeof(double), cudaMemcpyDeviceToHost, g_stream));
    SUB0_CUDA_CHECK(cudaStreamSynchronize(g_stream));
    SUB0_CUDA_CHECK(cudaGetLastError());
    if (out_loss) *out_loss = loss / M;          // mean cross-entropy (matches CPU train_batch)
    return 0;
}

// Forward+backward only: fills out_grad[PARAM_FLOATS] with the reduced gradient (for the
// gradient-parity test against the CPU train_batch grad). Optionally returns the mean loss.
SUB0_CUDA_API int sub0_cuda_backward(const int* ids, const int* targets, int batch, int T,
                                     float* out_grad, double* out_loss) {
    if (run_fwd_bwd(ids, targets, batch, T, out_loss)) return 1;
    SUB0_CUDA_CHECK(cudaMemcpy(out_grad, g_dev_grad, sub0::PARAM_FLOATS * sizeof(float),
                               cudaMemcpyDeviceToHost));
    return 0;
}

// AdamW step over the device gradient (after sub0_cuda_backward). `t` is the post-increment step
// counter the host optimizer maintains. Uses the CPU AdamW defaults so the two stay in lockstep.
SUB0_CUDA_API int sub0_cuda_adam_step(float lr, long t) {
    if (!g_dev_grad) return 1;
    ensure_cublas();
    device_adam_step(lr, t, 0.9f, 0.95f, 1e-8f, 0.01f, 1.0f);
    SUB0_CUDA_CHECK(cudaStreamSynchronize(g_stream));
    SUB0_CUDA_CHECK(cudaGetLastError());
    return 0;
}

// Full device training step: forward + backward + AdamW, returning the mean loss. The end-to-end
// path the GPU train stage will drive once it replaces the CPU backend.
SUB0_CUDA_API int sub0_cuda_train_step(const int* ids, const int* targets, int batch, int T,
                                       float lr, long t, double* out_loss) {
    if (run_fwd_bwd(ids, targets, batch, T, out_loss)) return 1;
    device_adam_step(lr, t, 0.9f, 0.95f, 1e-8f, 0.01f, 1.0f);
    SUB0_CUDA_CHECK(cudaStreamSynchronize(g_stream));
    SUB0_CUDA_CHECK(cudaGetLastError());
    return 0;
}

// Select TF32 tensor-core GEMM math (on != 0) vs full FP32 for subsequent forwards. Runtime
// knob -- the optimum depends on the host + GEMM shapes, so it is intended to feed the
// autotuner. Parity tests force FP32 here for a tight gate.
SUB0_CUDA_API void sub0_cuda_set_tf32(int on) {
    ensure_cublas();
    CudaTf32::set(on != 0);        // sweep the knob (a no-op in a non-tuning build)
    set_handle_tf32(on != 0);      // force the handle mode now (for the bench / parity tests)
}

// Profile the device forward: time `iters` runs of the resident kernel chain (no host
// transfers) under FP32 then TF32 math and print avg ms + the speedup. Self-contained
// (allocates the param mirror + scratch, ids = token 0; GEMM timing is data-independent).
// batch is clamped to MAX_FWD_BATCH.
SUB0_CUDA_API int sub0_cuda_benchmark(int batch, int T, int iters) {
    if (T < 1 || T > SEQ_LEN || iters < 1) return 1;
    if (batch < 1) batch = 1;
    if (batch > MAX_FWD_BATCH) batch = MAX_FWD_BATCH;
    if (sub0_cuda_init()) return 1;
    if (fwd_alloc()) return 1;
    const int M = batch * T;
    cudaMemset(g_fwd.dids, 0, static_cast<size_t>(M) * sizeof(int));   // ids = token 0

    auto time_eager = [&](bool tf32) -> double {
        set_handle_tf32(tf32);                                         // also invalidates the graph
        for (int w = 0; w < 5; ++w) forward_device(batch, T);         // warmup
        cudaStreamSynchronize(g_stream);
        cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
        cudaEventRecord(s, g_stream);
        for (int it = 0; it < iters; ++it) forward_device(batch, T);
        cudaEventRecord(e, g_stream); cudaEventSynchronize(e);
        float ms = 0.f; cudaEventElapsedTime(&ms, s, e);
        cudaEventDestroy(s); cudaEventDestroy(e);
        return static_cast<double>(ms) / iters;
    };
    auto time_graph = [&]() -> double {
        capture_graph(batch, T);                                      // FP32 (set just below)
        for (int w = 0; w < 5; ++w) cudaGraphLaunch(g_graph_exec, g_stream);
        cudaStreamSynchronize(g_stream);
        cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
        cudaEventRecord(s, g_stream);
        for (int it = 0; it < iters; ++it) cudaGraphLaunch(g_graph_exec, g_stream);
        cudaEventRecord(e, g_stream); cudaEventSynchronize(e);
        float ms = 0.f; cudaEventElapsedTime(&ms, s, e);
        cudaEventDestroy(s); cudaEventDestroy(e);
        return static_cast<double>(ms) / iters;
    };
    const double fp32 = time_eager(false);
    const double tf32 = time_eager(true);
    set_handle_tf32(false);                                           // capture the graph in FP32
    const double graph = time_graph();
    std::printf("cuda bench forward: batch=%d T=%d M=%d iters=%d | eager FP32 %.3f ms | eager TF32 %.3f ms | "
                "graph FP32 %.3f ms | graph %.2fx vs eager\n",
                batch, T, M, iters, fp32, tf32, graph, graph > 0.0 ? fp32 / graph : 0.0);
    return 0;
}
