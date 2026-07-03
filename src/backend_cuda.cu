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
#include "sub0/bench.hpp"    // adaptive_time: budget-sized measurement shared with the CPU tuner
#include "sub0/memplan.hpp"  // train_resident_bytes: the predicted footprint this file is validated against

#include <cstdio>
#include <vector>
#include <cmath>
#include <type_traits>

#include <cuda_runtime.h>
#include <cuda_bf16.h>
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

// Saved-activation element type. ACT_DTYPE selects the storage precision (configurator-baked):
// F32 today; BF16 halves the per-token train scratch on sm_80+ once the kernels are templated.
// FP32 accumulate is preserved (cuBLAS computeType / kernel math both stay float regardless).
using act_t = std::conditional_t<ACT_DTYPE == Dtype::BF16, __nv_bfloat16, float>;
[[maybe_unused]] __device__ inline float  to_f32(float v)         { return v; }
[[maybe_unused]] __device__ inline float  to_f32(__nv_bfloat16 v) { return __bfloat162float(v); }
[[maybe_unused]] __device__ inline void   st_act(float* p, float v)         { *p = v; }
[[maybe_unused]] __device__ inline void   st_act(__nv_bfloat16* p, float v) { *p = __float2bfloat16(v); }
[[maybe_unused]] __device__ inline void   atomicAdd_act(float* p, float v)         { atomicAdd(p, v); }
// bf16 atomic-add fallback: read-modify-write (no native bf16 atomic). Only the query-scheme attn
// backward uses it; the head scheme (bf16 default) is atomic-free so contention is not exercised.
[[maybe_unused]] __device__ inline void   atomicAdd_act(__nv_bfloat16* p, float v) { *p = __float2bfloat16(__bfloat162float(*p) + v); }

// Block-wide sum reduction, returned to EVERY thread (broadcast): warp-shuffle butterfly within each
// warp, then one more shuffle-reduce across per-warp partials via shared memory. BLOCK must be a
// compile-time power-of-two multiple of 32 (every caller launches with a fixed, template-known block
// size). The trailing __syncthreads() guards a second call reusing warp_sums[] in the same kernel
// (e.g. ce_backward's max-then-sum-of-exp) against a thread racing ahead into the next reduction's
// writes before every thread has finished reading this one's broadcast value.
template <int BLOCK>
[[maybe_unused]] __device__ inline float block_reduce_sum(float val) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) val += __shfl_down_sync(0xffffffffu, val, off);
    if constexpr (BLOCK > 32) {
        __shared__ float warp_sums[BLOCK / 32];
        const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
        if (lane == 0) warp_sums[wid] = val;
        __syncthreads();
        if (wid == 0) {
            val = (lane < BLOCK / 32) ? warp_sums[lane] : 0.f;
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) val += __shfl_down_sync(0xffffffffu, val, off);
            if (lane == 0) warp_sums[0] = val;
        }
        __syncthreads();
        val = warp_sums[0];
        __syncthreads();
    } else {
        val = __shfl_sync(0xffffffffu, val, 0);
    }
    return val;
}
// Same, but returns the block-wide MAX (used by ce_backward's softmax-stabilizing row max).
template <int BLOCK>
[[maybe_unused]] __device__ inline float block_reduce_max(float val) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) val = fmaxf(val, __shfl_down_sync(0xffffffffu, val, off));
    if constexpr (BLOCK > 32) {
        __shared__ float warp_maxs[BLOCK / 32];
        const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
        if (lane == 0) warp_maxs[wid] = val;
        __syncthreads();
        if (wid == 0) {
            val = (lane < BLOCK / 32) ? warp_maxs[lane] : -1e30f;
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) val = fmaxf(val, __shfl_down_sync(0xffffffffu, val, off));
            if (lane == 0) warp_maxs[0] = val;
        }
        __syncthreads();
        val = warp_maxs[0];
        __syncthreads();
    } else {
        val = __shfl_sync(0xffffffffu, val, 0);
    }
    return val;
}
// Down-cast a float buffer into the act_t store type (bf16 weight mirrors + dh16). F32: plain copy.
__global__ void f32_to_act_kernel(const float* __restrict__ x, act_t* __restrict__ y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) st_act(&y[i], x[i]);
}
// Up-cast the act_t store type back to float (used by the attention naive-vs-tiled self-check to
// bring bf16 device buffers to the host for comparison; F32 build == plain copy).
__global__ void act_to_f32_kernel(const act_t* __restrict__ x, float* __restrict__ y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = to_f32(x[i]);
}

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
unsigned char* g_dev_decay = nullptr;   // 0/1 weight-decay mask; see adam_step_kernel
double* g_dev_normsq = nullptr;   // [1] global grad sum-of-squares (AdamW clip, double like CPU)
float*  g_dev_gs     = nullptr;   // [1] grad-clip scale, computed on-device from g_dev_normsq (see
                                  // grad_clip_scale_kernel) -- avoids a host readback of normsq just
                                  // to compute this one scalar (device_adam_step).

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
// idx/bound computed in 64-bit: M*N (lm_head's bias-add is M=batch*T against N=VOCAB) overflows
// signed 32-bit at ordinary training sizes -- e.g. batch=128, T=512, VOCAB=32873 already exceeds
// INT32_MAX. Verified via this project's own generated config, not a hypothetical: a plain `int`
// product here is silent UB (observed as either a corrupted grid or a bias-add skipped for most
// rows), not a future-proofing nicety.
__global__ void bias_add_kernel(float* __restrict__ Y, const float* __restrict__ bias, int M, int N) {
    const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < static_cast<long long>(M) * N) Y[idx] += bias[idx % N];
}

// --- Fast transcendental math (CUDA native intrinsics) ----------------------
// Use the hardware fast-math path -- __expf on the SFU, rsqrtf for reciprocal-sqrt --
// rather than re-deriving the CPU's software polynomial. Both are ~1e-6 approximations,
// so the GPU forward stays close to the CPU FAST_MATH path but is NOT bit-identical; the
// CPU<->GPU parity test uses a tolerance that absorbs the two approximations' difference.
// dev_tanh keeps the tanh-form GELU shape (matching the CPU's GPT-2-style approx) built on
// __expf, so the GELU stays consistent with the CPU op_gelu.
__device__ inline float dev_tanh(float x) {
    // Clamp the exponent exactly like the CPU fast_exp ([-87,88]) BEFORE __expf. Without this a
    // large negative GELU input makes -2x large positive, __expf overflows to +Inf, and the tanh
    // becomes (1-Inf)/(1+Inf) = NaN -- harmless at small init but it poisons the weights once
    // training grows the activations (the cause of GPU-training divergence). Clamping saturates
    // to +-1 like the CPU path instead.
    float a = -2.f * x;
    a = a < -87.f ? -87.f : (a > 88.f ? 88.f : a);
    const float e = __expf(a);
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
// t = m % T is the position WITHIN the window (op_embed + op_add: first forward step). C is D_MODEL
// at every call site below -- constexpr-folded (no loop to unroll here, pure elementwise; the win is
// just the dropped runtime parameter). M/T stay runtime (genuinely vary).
__global__ void embed_add_kernel(const float* __restrict__ tok_emb, const float* __restrict__ pos_emb,
                                 const int* __restrict__ ids, float* __restrict__ h, int M, int T) {
    constexpr int C = D_MODEL;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = blockIdx.y * blockDim.y + threadIdx.y;
    if (m < M && j < C) {
        const int t = m % T;
        h[m * C + j] = tok_emb[ids[m] * C + j] + pos_emb[t * C + j];
    }
}

// Token-only embedding (RoPE path): h[m,j] = tok_emb[ids[m], j]. Position is injected later by
// the RoPE rotation inside attention, so there is no pos_emb add here.
__global__ void embed_kernel(const float* __restrict__ tok_emb, const int* __restrict__ ids,
                             float* __restrict__ h, int M) {
    constexpr int C = D_MODEL;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = blockIdx.y * blockDim.y + threadIdx.y;
    if (m < M && j < C) h[m * C + j] = tok_emb[ids[m] * C + j];
}

// Activation-typed variants: write into the saved-activation store type (act_t). F32 build keeps
// these identical to the float kernels above; BF16 stores half-width with FP32 source.
template <class A>
__global__ void embed_act_kernel(const float* __restrict__ tok_emb, const int* __restrict__ ids,
                                 A* __restrict__ h, int M) {
    constexpr int C = D_MODEL;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = blockIdx.y * blockDim.y + threadIdx.y;
    if (m < M && j < C) st_act(&h[m * C + j], tok_emb[ids[m] * C + j]);
}
template <class A>
__global__ void embed_add_act_kernel(const float* __restrict__ tok_emb, const float* __restrict__ pos_emb,
                                     const int* __restrict__ ids, A* __restrict__ h, int M, int T) {
    constexpr int C = D_MODEL;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = blockIdx.y * blockDim.y + threadIdx.y;
    if (m < M && j < C) { const int t = m % T; st_act(&h[m * C + j], tok_emb[ids[m] * C + j] + pos_emb[t * C + j]); }
}

// y[m,j] = x[m,j] * (1/sqrt(mean_j x^2 + eps)) * gamma[j]  (op_rmsnorm forward). One BLOCK per row,
// threads striding coalesced over D_MODEL -- was one thread per row (the fix and reasoning are
// identical to rmsnorm_train_act_kernel above, which has the full ncu evidence for the same
// one-thread-per-row anti-pattern; this is the plain non-training sibling used by generation/decode,
// including the per-TOKEN decode hot path where rows==1 -- under the old layout that meant 63 of 64
// threads in the launch sat idle every single token). D_MODEL used directly (constexpr, never a
// runtime size at any call site) instead of as a parameter.
template <int BLOCK>
__global__ void __launch_bounds__(BLOCK)
rmsnorm_kernel(const float* __restrict__ x, const float* __restrict__ gamma,
              float* __restrict__ y, int rows) {
    constexpr int   C    = D_MODEL;
    constexpr float invC = 1.0f / static_cast<float>(C);
    constexpr float eps  = 1e-5f;
    const int m = blockIdx.x;
    if (m >= rows) return;
    const float* xr = x + static_cast<size_t>(m) * C;
    float partial = 0.f;
    #pragma unroll
    for (int j = threadIdx.x; j < C; j += BLOCK) partial += xr[j] * xr[j];
    const float ms = block_reduce_sum<BLOCK>(partial) * invC;
    const float r  = rsqrtf(ms + eps);                  // CUDA fast reciprocal sqrt
    float* yr = y + static_cast<size_t>(m) * C;
    #pragma unroll
    for (int j = threadIdx.x; j < C; j += BLOCK) yr[j] = xr[j] * r * gamma[j];
}

// Elementwise tanh-form GELU (op_gelu, FAST_MATH path).
__global__ void gelu_kernel(const float* __restrict__ x, float* __restrict__ y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = dev_gelu(x[i]);
}
// act-typed GELU: y = gelu(x), reading/writing the store type (F32 build == gelu_kernel).
template <class A>
__global__ void gelu_act_kernel(const A* __restrict__ x, A* __restrict__ y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) st_act(&y[i], dev_gelu(to_f32(x[i])));
}

// Elementwise add: c[i] = a[i] + b[i] (residual connections; safe in-place when c == a).
__global__ void add_kernel(const float* __restrict__ a, const float* __restrict__ b,
                           float* __restrict__ c, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
// act-typed add: c = a + b in the store type (F32 build == add_kernel).
template <class A>
__global__ void add_act_kernel(const A* __restrict__ a, const A* __restrict__ b, A* __restrict__ c, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) st_act(&c[i], to_f32(a[i]) + to_f32(b[i]));
}
// act-typed RMSNorm-train: y = rmsnorm(x)*gamma, saving rinv; in X / out Y store types, FP32 math.
// One BLOCK per row, threads striding coalesced over D_MODEL -- same fix and same reasoning as
// rmsnorm_backward_act_kernel above (that comment has the full ncu evidence): one-thread-per-row put
// a warp's simultaneous loads C floats apart with no coalescing. D_MODEL is compile-time constexpr
// (never a runtime size at any call site), so it's used directly rather than taken as a parameter,
// which also lets the strided loops fully unroll and folds the /C divide into a constant multiply.
template <class X, class Y, int BLOCK>
__global__ void __launch_bounds__(BLOCK)
rmsnorm_train_act_kernel(const X* __restrict__ x, const float* __restrict__ gamma,
                         Y* __restrict__ y, float* __restrict__ rinv, int rows) {
    constexpr int   C    = D_MODEL;
    constexpr float invC = 1.0f / static_cast<float>(C);
    constexpr float eps  = 1e-5f;
    const int m = blockIdx.x;
    if (m >= rows) return;
    const X* xr = x + static_cast<size_t>(m) * C;
    float partial = 0.f;
    #pragma unroll
    for (int j = threadIdx.x; j < C; j += BLOCK) { const float v = to_f32(xr[j]); partial += v * v; }
    const float ms = block_reduce_sum<BLOCK>(partial) * invC;
    const float r = rsqrtf(ms + eps);
    if (threadIdx.x == 0) rinv[m] = r;
    Y* yr = y + static_cast<size_t>(m) * C;
    #pragma unroll
    for (int j = threadIdx.x; j < C; j += BLOCK) st_act(&yr[j], to_f32(xr[j]) * r * gamma[j]);
}

// Causal multi-head attention (op_attn) is served by the flash-style attn_fwd_tiled_kernel below
// (used for BOTH inference and training, via launch_attn / launch_attn_train_t). The naive
// one-thread-per-query variant survives only as attn_train_act_kernel, kept as the on-device parity
// REFERENCE the self-test compares the tiled kernel against (sub0_cuda_attn_check).

// Build the fused QKV weight Wqkv[C, 3C] (row-major) from Wq,Wk,Wv [C,C]: row p holds
// [Wq[p] | Wk[p] | Wv[p]]. Materialized ONCE at upload so the three projection GEMMs collapse
// into one a . Wqkv -> [M, 3C] (better-shaped GEMM + fewer launches).
// C is D_MODEL at its one call site -- constexpr-folded, so the p=idx/C, c=idx%C split (GPU integer
// div/mod has no native instruction) resolves at compile time instead of once per thread per layer.
__global__ void build_qkv_kernel(const float* __restrict__ Wq, const float* __restrict__ Wk,
                                 const float* __restrict__ Wv, float* __restrict__ Wqkv) {
    constexpr int C = D_MODEL;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;   // over C*C elements of one source matrix
    if (idx < C * C) {
        const int p = idx / C, c = idx % C;
        const int row = p * 3 * C;
        Wqkv[row + c]         = Wq[idx];
        Wqkv[row + C + c]     = Wk[idx];
        Wqkv[row + 2 * C + c] = Wv[idx];
    }
}

// Same fused layout, writing DIRECTLY to the activation (bf16) mirror instead of staging through an
// F32 intermediate -- what BF16 training uses (build_qkv_weights() below) so it never needs the
// F32 g_fwd.wqkv buffer at all. st_act rounds each element on the store, identical to building the
// F32 buffer above and converting it afterward, minus the extra read+write pass over 3*C*C elements.
template <class A>
__global__ void build_qkv_act_kernel(const float* __restrict__ Wq, const float* __restrict__ Wk,
                                     const float* __restrict__ Wv, A* __restrict__ Wqkv) {
    constexpr int C = D_MODEL;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < C * C) {
        const int p = idx / C, c = idx % C;
        const int row = p * 3 * C;
        st_act(&Wqkv[row + c],         Wq[idx]);
        st_act(&Wqkv[row + C + c],     Wk[idx]);
        st_act(&Wqkv[row + 2 * C + c], Wv[idx]);
    }
}

// RoPE forward: rotate the Q and K sub-blocks (columns [0,C) and [C,2C)) of the fused [*, 3C]
// qkv buffer in place, in interleaved pairs per head; V (cols [2C,3C)) is left alone. One thread
// per (row m, global pair pg over C/2). pos = m % T (position within the window). Mirrors the CPU
// op_rope; the math (CUDA __sincosf/powf vs CPU std::) agrees to fast-math tolerance.
//
// C, H, in_stride are D_MODEL/N_HEADS/3*D_MODEL at every call site (confirmed by grep) -- baked
// constexpr, so used directly instead of as parameters. This also folds `d = C/H` from a per-thread
// runtime division into the already-baked D_HEAD constant (GPU integer div/mod has no native
// instruction; this ran over M*D_MODEL/2 threads every layer, every forward AND backward pass under
// RoPE). T/batch stay runtime (genuinely vary, same as ce_backward_kernel's T).
__global__ void rope_kernel(float* __restrict__ qkv, int batch, int T, float theta) {
    constexpr int C = D_MODEL, in_stride = 3 * D_MODEL;
    constexpr int d = D_HEAD, half = d / 2;
    const int pg = blockIdx.x * blockDim.x + threadIdx.x;   // pair over C/2 (heads x d/2)
    const int m  = blockIdx.y * blockDim.y + threadIdx.y;   // row over M = batch*T
    if (m >= batch * T || pg >= C / 2) return;
    const int h  = pg / half, mi = pg % half;
    const int a0 = h * d + 2 * mi;
    const float ang = static_cast<float>(m % T) * powf(theta, -2.0f * mi / d);
    float sn, cs; __sincosf(ang, &sn, &cs);
    float* row = qkv + static_cast<size_t>(m) * in_stride;
    const float q0 = row[a0],     q1 = row[a0 + 1];
    row[a0]         = q0 * cs - q1 * sn;
    row[a0 + 1]     = q0 * sn + q1 * cs;
    const float k0 = row[C + a0], k1 = row[C + a0 + 1];
    row[C + a0]     = k0 * cs - k1 * sn;
    row[C + a0 + 1] = k0 * sn + k1 * cs;
}

// ============================================================================
//  Backward kernels (Phase 2d) -- mirror src/backend_cpu.cpp backward_node()
// ============================================================================

// act-typed attention forward (flash, P-free) -- q/k/v/out in the store type, FP32 softmax.
template <class A>
__global__ void attn_train_act_kernel(const A* __restrict__ q, const A* __restrict__ k,
                                       const A* __restrict__ v, A* __restrict__ out,
                                       int batch, int T, int C, int H, int in_stride) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x; const int per = H * T;
    const int b = idx / per; if (b >= batch) return;
    const int rem = idx - b * per, h = rem / T, i = rem % T, d = C / H, off = h * d;
    const float scale = 1.0f / sqrtf(static_cast<float>(d));
    const size_t in_base = static_cast<size_t>(b) * T * in_stride, out_base = static_cast<size_t>(b) * T * C;
    const A* qi = q + in_base + static_cast<size_t>(i) * in_stride + off;
    float m = -1e30f, Z = 0.f, acc[128] = {};
    for (int j = 0; j <= i; ++j) {
        const A* kj = k + in_base + static_cast<size_t>(j) * in_stride + off;
        float s = 0.f; for (int a = 0; a < d; ++a) s += to_f32(qi[a]) * to_f32(kj[a]); s *= scale;
        const float mn = fmaxf(m, s), c = __expf(m - mn), e = __expf(s - mn); Z = Z * c + e;
        const A* vj = v + in_base + static_cast<size_t>(j) * in_stride + off;
        for (int a = 0; a < d; ++a) acc[a] = acc[a] * c + e * to_f32(vj[a]); m = mn;
    }
    A* oi = out + out_base + static_cast<size_t>(i) * C + off; const float invZ = 1.f / Z;
    for (int a = 0; a < d; ++a) st_act(&oi[a], acc[a] * invZ);
}

// ---- Flash-style TILED attention forward (the hot-path replacement for attn_train_act_kernel) ----
// The naive kernel above is L1/TEX-cache-bound: every query thread re-streams ALL of K and V from
// global, so each K/V row is read O(T) times (ncu at d448/T256: L1/TEX 93%, Memory 90%, Compute 19%,
// = 92% of the forward). This tiles the keys through shared memory. A block owns TILE_Q queries of
// ONE (b,h) -- grid = (T-tiles, H, batch) -- stages each K/V tile into shared ONCE, and all TILE_Q
// queries in the block reuse it (the redundant global reads collapse by ~TILE_Q). q and the online-
// softmax accumulator live in registers (HD = D_HEAD is compile-time, so the per-channel loops fully
// unroll and the streamed data never spills to local/L1). Keys are still consumed in INCREASING j,
// exactly like the naive kernel, so the online-softmax arithmetic -- and therefore the result -- is
// bit-identical to attn_train_act_kernel (gated on device by sub0_cuda_attn_check, naive vs tiled).
// HD = head dim, TILE_Q = queries per block, TILE_K = keys staged per shared tile.
template <int HD> constexpr int attn_tile_k() { return HD <= 64 ? 64 : (HD <= 128 ? 32 : 16); }
constexpr int kAttnTileQ = 64;

// Query-tile size for the dk/dv backward kernel, which stages BOTH q and dout (2*TILE_Q*HD floats,
// plus 3 per-query stats). At the fixed 64 that overflows the 48 KB static shared-memory limit once
// HD grows (HD=96 -> 2*64*96*4 + 3*64*4 = 49920 B > 49152): shrink the staged query tile for large HD
// so it fits, exactly as attn_tile_k does for the key tile. The tile is DECOUPLED from the 64-thread
// (64-key) block -- the staging loops stride by blockDim.x and the query loop steps by TILE_Q from the
// key-tile base, so causality and the result are unchanged for any TILE_Q. (Register pressure at large
// HD is a separate, still-open perf item: 4*HD accumulators spill; a DV/DK split would help.)
template <int HD> constexpr int attn_tile_q() { return HD <= 64 ? 64 : (HD <= 128 ? 32 : 16); }

// Warp-cooperative channel split for the dq kernel: LANES threads (a power of 2) share one query,
// each owning HD/LANES channels of qr/dr/dqa, so the per-thread register need drops from 3*HD to
// 3*(HD/LANES). Engaged once the raw 3*HD array footprint APPROACHES the 255-register architectural
// cap (verified via ptxas -v -- neither #pragma unroll nor a smaller __launch_bounds__ hint change
// that cap; 255 is a hardware ceiling, not a launch-config artifact). The trigger is NOT simply
// "3*HD > 255": measured via cudaFuncGetAttributes (sub0_cuda_attn_regcheck), HD=64 (3*HD=192,
// comfortably under 255 by the raw-array count alone) still compiled to 255 regs + 56B spill --
// per-thread state beyond the three raw arrays (softmax stats, shared-memory staging pointers, loop
// induction) adds real, non-negligible register pressure the naive formula doesn't count. HD=32
// (3*HD=96) measured 0 spill, so the threshold is moved down to 3*HD >= 192 (HD >= 64) to also
// cover that case, while leaving HD=32 on the unsplit LANES=1 path. LANES=1 makes every formula in
// the kernel below reduce algebraically to the single-thread-per-query code (lane=0, HALF=HD), so
// small-HD builds are otherwise unaffected in code path or performance.
template <int HD> __host__ __device__ constexpr int attn_dq_lanes() { return (3 * HD >= 192 && HD % 2 == 0) ? 2 : 1; }

// Same channel-split idea, applied to the dk kernel below: LANES threads share one KEY, each owning
// HD/LANES channels of kr/vr/dka. dk_j = sum_i ds_ij q_i needs both the QK score (via kr) and the
// dp = <dout_i, v_j> term (via vr) to form the scalar ds_ij -- structurally the SAME shape as dq's
// s/dp pair, just with q/k roles swapped (dk owns the key, streams q/dout; dq owns the query, streams
// k/v), so the identical warp-shuffle-combine trick applies. Same empirically-grounded threshold as
// attn_dq_lanes (see its comment): kept as a separate function since the two kernels' register
// pressure could diverge in the future even though the formula is identical today.
template <int HD> __host__ __device__ constexpr int attn_dk_lanes() { return (3 * HD >= 192 && HD % 2 == 0) ? 2 : 1; }

template <class A, int HD, int TILE_K>
__global__ void __launch_bounds__(kAttnTileQ)
attn_fwd_tiled_kernel(const A* __restrict__ q, const A* __restrict__ k, const A* __restrict__ v,
                      A* __restrict__ out, int T, int C, int in_stride) {
    __shared__ float Ks[TILE_K * HD];
    __shared__ float Vs[TILE_K * HD];
    const int    b  = blockIdx.z, h = blockIdx.y;
    const int    q0 = blockIdx.x * blockDim.x;               // first query of this block
    const int    i  = q0 + threadIdx.x;                      // this thread's query (block owns one (b,h))
    const int    off = h * HD;
    const float  scale    = rsqrtf(static_cast<float>(HD));
    const size_t in_base  = static_cast<size_t>(b) * T * in_stride;
    const size_t out_base = static_cast<size_t>(b) * T * C;

    float qr[HD];                                            // this query's row, cached in registers
    if (i < T) {
        const A* qi = q + in_base + static_cast<size_t>(i) * in_stride + off;
        #pragma unroll
        for (int a = 0; a < HD; ++a) qr[a] = to_f32(qi[a]);
    }
    float m = -1e30f, Z = 0.f, acc[HD];
    #pragma unroll
    for (int a = 0; a < HD; ++a) acc[a] = 0.f;

    const int jmax = min(T, q0 + static_cast<int>(blockDim.x));   // no query in this block attends j >= jmax
    for (int j0 = 0; j0 < jmax; j0 += TILE_K) {
        const int tk = min(TILE_K, jmax - j0);
        __syncthreads();
        for (int e = threadIdx.x; e < tk * HD; e += blockDim.x) { // cooperative K/V tile load (all threads)
            const int    jl  = e / HD, a = e - jl * HD;
            const size_t row = in_base + static_cast<size_t>(j0 + jl) * in_stride + off;
            Ks[e] = to_f32(k[row + a]);
            Vs[e] = to_f32(v[row + a]);
        }
        __syncthreads();
        if (i < T) {
            const int jend = min(tk, i - j0 + 1);            // causal: only keys j <= i (<=0 -> skip tile)
            for (int jl = 0; jl < jend; ++jl) {
                const float* ks = Ks + jl * HD;
                float s = 0.f;
                #pragma unroll
                for (int a = 0; a < HD; ++a) s += qr[a] * ks[a];
                s *= scale;
                const float mn = fmaxf(m, s), c = __expf(m - mn), e2 = __expf(s - mn);
                Z = Z * c + e2;
                const float* vs = Vs + jl * HD;
                #pragma unroll
                for (int a = 0; a < HD; ++a) acc[a] = acc[a] * c + e2 * vs[a];
                m = mn;
            }
        }
    }
    if (i < T) {
        A* oi = out + out_base + static_cast<size_t>(i) * C + off;
        const float invZ = 1.f / Z;
        #pragma unroll
        for (int a = 0; a < HD; ++a) st_act(&oi[a], acc[a] * invZ);
    }
}

// C/H/in_stride constexpr-folded (D_MODEL/N_HEADS/3*D_MODEL, every call site) -- see rope_kernel above.
template <class A> __global__ void rope_act_kernel(A* __restrict__ qkv, int batch, int T, float theta) {
    constexpr int C = D_MODEL, in_stride = 3 * D_MODEL, d = D_HEAD, half = d / 2;
    const int pg = blockIdx.x * blockDim.x + threadIdx.x, m = blockIdx.y * blockDim.y + threadIdx.y;
    if (m >= batch * T || pg >= C / 2) return;
    const int h = pg / half, mi = pg % half, a0 = h * d + 2 * mi;
    const float ang = static_cast<float>(m % T) * powf(theta, -2.0f * mi / d); float sn, cs; __sincosf(ang, &sn, &cs);
    A* row = qkv + static_cast<size_t>(m) * in_stride;
    const float q0 = to_f32(row[a0]), q1 = to_f32(row[a0 + 1]); st_act(&row[a0], q0*cs-q1*sn); st_act(&row[a0+1], q0*sn+q1*cs);
    const float k0 = to_f32(row[C+a0]), k1 = to_f32(row[C+a0+1]); st_act(&row[C+a0], k0*cs-k1*sn); st_act(&row[C+a0+1], k0*sn+k1*cs);
}
template <class A> __global__ void rope_bwd_act_kernel(A* __restrict__ dq, int batch, int T, float theta) {
    constexpr int C = D_MODEL, in_stride = 3 * D_MODEL, d = D_HEAD, half = d / 2;
    const int pg = blockIdx.x * blockDim.x + threadIdx.x, m = blockIdx.y * blockDim.y + threadIdx.y;
    if (m >= batch * T || pg >= C / 2) return;
    const int h = pg / half, mi = pg % half, a0 = h * d + 2 * mi;
    const float ang = static_cast<float>(m % T) * powf(theta, -2.0f * mi / d); float sn, cs; __sincosf(ang, &sn, &cs);
    A* row = dq + static_cast<size_t>(m) * in_stride;
    const float g0 = to_f32(row[a0]), g1 = to_f32(row[a0+1]); st_act(&row[a0], g0*cs+g1*sn); st_act(&row[a0+1], -g0*sn+g1*cs);
    const float h0 = to_f32(row[C+a0]), h1 = to_f32(row[C+a0+1]); st_act(&row[C+a0], h0*cs+h1*sn); st_act(&row[C+a0+1], -h0*sn+h1*cs);
}

// Cross-entropy backward (op_cross_entropy): one BLOCK per row m over [M,V], threads striding
// coalesced over V (was one thread per row doing 3 fully sequential O(V) passes -- ncu at V=16489
// measured 30.3ms/call, 28% occupancy, 4.7% compute throughput, the same uncoalesced-warp pattern as
// the rmsnorm kernels above: adjacent threads owned adjacent ROWS, so a fixed loop iteration touched
// addresses V floats apart). Row m belongs to window b = m/T at position t = m%T. Positions
// t >= lengths[b] are PADDING (a short document padded up to T): they get zero gradient and add no
// loss. Each trained row is weighted 1/(batch*len[b]) so the loss/grad is the per-window mean over its
// real length, then the mean over the batch -- identical to the CPU train_batch reduction, and exactly
// 1/M when every window is full (len == T, lengths == nullptr), which keeps the dense gradient-parity
// gate unchanged.
//
// V is VOCAB, a baked constexpr (this kernel's one call site always passes it), used directly instead
// of as a parameter. UNLIKE D_MODEL in the rmsnorm kernels above, V is NOT unrolled -- it runs into
// the tens of thousands (32873 in the production config), so a full #pragma unroll would be a
// code-size/I-cache disaster for a handful fewer loop-branch instructions; the real win here is
// coalescing + parallelizing the O(V) work across BLOCK threads, same as the reduction itself. T stays
// a runtime parameter (unlike V, it genuinely varies -- benchmark/self-test paths call this with a
// window length shorter than SEQ_LEN).
template <int BLOCK>
__global__ void __launch_bounds__(BLOCK)
ce_backward_kernel(const float* __restrict__ logits, const int* __restrict__ targets,
                   float* __restrict__ dlogits, double* __restrict__ loss_acc,
                   int M, int T, int batch, const int* __restrict__ lengths) {
    constexpr int V = VOCAB;
    const int m = blockIdx.x;
    if (m >= M) return;
    const int b   = m / T;
    const int t   = m - b * T;
    const int len = lengths ? lengths[b] : T;
    float* dl     = dlogits + static_cast<size_t>(m) * V;
    if (t >= len) {                                  // padding row: inert (no grad, no loss)
        for (int j = threadIdx.x; j < V; j += BLOCK) dl[j] = 0.f;
        return;
    }
    const float* lr = logits + static_cast<size_t>(m) * V;
    const int tgt   = targets[m];
    float mx_partial = -1e30f;
    for (int j = threadIdx.x; j < V; j += BLOCK) mx_partial = fmaxf(mx_partial, lr[j]);
    const float mx = block_reduce_max<BLOCK>(mx_partial);
    float Z_partial = 0.f;
    for (int j = threadIdx.x; j < V; j += BLOCK) Z_partial += __expf(lr[j] - mx);
    const float Z = block_reduce_sum<BLOCK>(Z_partial);
    const float invZ = 1.f / Z;
    const float w    = 1.0f / (static_cast<float>(batch) * static_cast<float>(len));   // per-window mean
    float ptgt_partial = 0.f;                        // capture before any write (dlogits may alias logits)
    for (int j = threadIdx.x; j < V; j += BLOCK) {
        const float p = __expf(lr[j] - mx) * invZ;
        if (j == tgt) ptgt_partial = p;               // exactly one thread across the block owns j==tgt
        dl[j] = w * (p - (j == tgt ? 1.f : 0.f));    // safe in-place: reads lr[j] then writes dl[j]
    }
    const float ptgt = block_reduce_sum<BLOCK>(ptgt_partial);   // sum isolates the one nonzero contributor
    if (threadIdx.x == 0)
        atomicAdd(loss_acc, static_cast<double>(w * -__logf(fmaxf(1e-9f, ptgt))));
}

// Bias gradient: dbias[o] = sum_m dY[m,o] (one thread per output column). `accumulate` mirrors
// gemm()'s beta: false OVERWRITES dbias (every existing call site), true adds this call's column
// sum into it -- not used by any call site yet (see gemm()'s beta comment); default keeps every
// existing call's behavior unchanged.
__global__ void bias_grad_kernel(const float* __restrict__ dY, float* __restrict__ dbias, int M, int N,
                                 bool accumulate = false) {
    const int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= N) return;
    float s = 0.f;
    for (int m = 0; m < M; ++m) s += dY[static_cast<size_t>(m) * N + o];
    dbias[o] = accumulate ? dbias[o] + s : s;
}
// act-typed bias gradient: dbias[o] = sum_m dY_act[m,o], reading the bf16 store with FP32 accum. This
// kernel's memory pattern is already fine (adjacent threads o, o+1 read adjacent columns at each fixed
// m -- coalesced across the warp); unlike the row-reduction kernels above, there's no block-per-row fix
// to apply here. N is D_FF at this kernel's one call site (a baked constexpr, unlike bias_grad_kernel
// below whose N genuinely varies across ITS call sites -- confirmed by checking every one), so it's
// used directly rather than as a parameter; the M-loop stays runtime (M varies) with nothing to unroll.
template <class A>
__global__ void bias_grad_act_kernel(const A* __restrict__ dY, float* __restrict__ dbias, int M) {
    constexpr int N = D_FF;
    const int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= N) return;
    float s = 0.f;
    for (int m = 0; m < M; ++m) s += to_f32(dY[static_cast<size_t>(m) * N + o]);
    dbias[o] = s;
}

// RMSNorm backward: one BLOCK per row, threads striding coalesced over D_MODEL (was one thread per
// row -- ncu measured that at 26-28% occupancy, 3% compute throughput, 77-78% of warp cycles stalled
// on L1TEX scoreboard dependencies, because at a fixed loop iteration j, adjacent threads (adjacent
// rows) touch addresses D_MODEL floats apart -- no coalescing across the warp at all). Here thread `k`
// of the block reads column k, k+BLOCK, k+2*BLOCK, ... of the SAME row, so a warp's simultaneous loads
// land in one contiguous span; the row's S = sum_j dy[j]*gamma[j]*x[j] is then a block_reduce_sum
// instead of a sequential accumulation.
//
// D_MODEL is baked constexpr (one build = one architecture), not a runtime size -- rmsnorm always
// normalizes the full residual stream, never a differently-sized slice, at every call site in this
// file. Taking it as a compile-time constant instead of a parameter lets the strided loops fully
// unroll (their trip count is then also compile time) and folds the /D_MODEL divide into a constant
// multiply. BLOCK=64 (see launch_rmsnorm_bwd_t) evenly divides every D_MODEL this project currently
// bakes (192/448/768), so no thread does a ragged final iteration.
//
// ACCUMULATES dx into the running residual-stream gradient (+=) and dgamma (atomic). Residual input x
// is the store type; dy/dx and dgamma stay F32. F32 build: X=float; BF16: reads the half-width
// residual. Mirrors the CPU formula exactly.
template <class X, int BLOCK>
__global__ void __launch_bounds__(BLOCK)
rmsnorm_backward_act_kernel(const X* __restrict__ x, const float* __restrict__ gamma,
                            const float* __restrict__ rinv, const float* __restrict__ dy,
                            float* __restrict__ dx, float* __restrict__ dgamma, int rows) {
    constexpr int   C    = D_MODEL;
    constexpr float invC = 1.0f / static_cast<float>(C);
    const int t = blockIdx.x;
    if (t >= rows) return;
    const X*     xr  = x  + static_cast<size_t>(t) * C;
    const float* dyr = dy + static_cast<size_t>(t) * C;
    float*       dxr = dx + static_cast<size_t>(t) * C;
    float partial = 0.f;
    #pragma unroll
    for (int j = threadIdx.x; j < C; j += BLOCK) partial += dyr[j] * gamma[j] * to_f32(xr[j]);
    const float S = block_reduce_sum<BLOCK>(partial);
    const float r = rinv[t], r3 = r * r * r;
    #pragma unroll
    for (int j = threadIdx.x; j < C; j += BLOCK) {
        const float xj = to_f32(xr[j]), dyj = dyr[j], gj = gamma[j];
        dxr[j] += r * dyj * gj - (xj * r3 * invC) * S;
        atomicAdd(&dgamma[j], dyj * xj * r);
    }
}

// Naive causal attention backward (op_attn), HEAD-per-thread: one thread per (window b, head h) owns
// the whole head, so dq/dk/dv accumulate with NO atomics. Superseded on the hot path by the flash
// tiled backward below; KEPT as the on-device parity REFERENCE the tiled kernels are checked against
// (sub0_cuda_attn_bwd_check). The dq/dk/dv buffer must be zeroed before launch. q/k/v and dqkv are
// the store type (F32 build == FP32 path; BF16 reads/writes bf16 with FP32 softmax+accum).
template <class A>
__global__ void attn_backward_head_act_kernel(const A* __restrict__ q, const A* __restrict__ k,
                                          const A* __restrict__ v, const A* __restrict__ dout,
                                          A* __restrict__ dq, A* __restrict__ dk, A* __restrict__ dv,
                                          int batch, int T, int C, int H, int in_stride) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x; if (idx >= batch * H) return;
    const int b = idx / H, h = idx % H, d = C / H, off = h * d;
    const float scale = 1.0f / sqrtf(static_cast<float>(d));
    const size_t in_base = static_cast<size_t>(b) * T * in_stride, out_base = static_cast<size_t>(b) * T * C;
    for (int i = 0; i < T; ++i) {
        const A* qi = q + in_base + static_cast<size_t>(i) * in_stride + off;
        const A* di = dout + out_base + static_cast<size_t>(i) * C + off;
        float m = -1e30f, Z = 0.f;
        for (int j = 0; j <= i; ++j) { const A* kj = k + in_base + static_cast<size_t>(j) * in_stride + off;
            float s = 0.f; for (int a = 0; a < d; ++a) s += to_f32(qi[a]) * to_f32(kj[a]); s *= scale;
            const float mn = fmaxf(m, s); Z = Z * __expf(m - mn) + __expf(s - mn); m = mn; }
        const float invZ = 1.f / Z; float dot = 0.f;
        for (int j = 0; j <= i; ++j) { const A* kj = k + in_base + static_cast<size_t>(j) * in_stride + off;
            float s = 0.f; for (int a = 0; a < d; ++a) s += to_f32(qi[a]) * to_f32(kj[a]);
            const float p = __expf(s * scale - m) * invZ; const A* vj = v + in_base + static_cast<size_t>(j) * in_stride + off;
            A* dvj = dv + in_base + static_cast<size_t>(j) * in_stride + off; float dp = 0.f;
            for (int a = 0; a < d; ++a) { st_act(&dvj[a], to_f32(dvj[a]) + p * to_f32(di[a])); dp += to_f32(di[a]) * to_f32(vj[a]); }
            dot += p * dp; }
        A* dqi = dq + in_base + static_cast<size_t>(i) * in_stride + off;
        for (int j = 0; j <= i; ++j) { const A* kj = k + in_base + static_cast<size_t>(j) * in_stride + off;
            const A* vj = v + in_base + static_cast<size_t>(j) * in_stride + off; float s = 0.f, dp = 0.f;
            for (int a = 0; a < d; ++a) { s += to_f32(qi[a]) * to_f32(kj[a]); dp += to_f32(di[a]) * to_f32(vj[a]); }
            const float p = __expf(s * scale - m) * invZ, ds = p * (dp - dot) * scale;
            A* dkj = dk + in_base + static_cast<size_t>(j) * in_stride + off;
            for (int a = 0; a < d; ++a) { st_act(&dqi[a], to_f32(dqi[a]) + ds * to_f32(kj[a])); st_act(&dkj[a], to_f32(dkj[a]) + ds * to_f32(qi[a])); } }
    }
}

// ---- Flash-style TILED attention BACKWARD (the hot-path replacement for the two naive kernels) ----
// The naive kernels above are either low-parallelism (head-per-thread: only batch*H threads) or
// atomic-heavy (query-per-thread: bf16 RMW races), and both re-stream K/V/Q from global O(T) times.
// This mirrors the FlashAttention backward: THREE atomic-free kernels, each parallel over the full
// (b,h,T) grid and staging the contracted dimension through shared memory. Per-query softmax stats
// (m_i, 1/Z_i, dot_i = <dout_i, out_i>) are precomputed once, then dq is accumulated query-parallel
// (owned per query -> no atomics) and dk/dv key-parallel (owned per key -> no atomics). Uses the SAVED
// attention output `att` for dot_i (= out_i), so no value re-accumulation is needed to get it. The
// arithmetic matches the naive backward to bf16 rounding (gated on device by sub0_cuda_attn_check).
// The flat per-(b,h,i) stats index is ((b*H)+h)*T + i (H = gridDim.y).

// (1) Per-query stats: m_i (max scaled score), 1/Z_i, and dot_i = sum_{j<=i} p_ij <dout_i, v_j>
//     ( = <dout_i, out_i> ). Two tiled key-passes: pass A gets m/Z from shared K; pass B gets dot in
//     FP32 from shared K+V (recomputed p). Computing dot in fp32 -- rather than from the bf16-rounded
//     saved output -- keeps ds = p*(dp - dot) matching the naive backward through near-cancellation.
template <class A, int HD, int TILE_K>
__global__ void __launch_bounds__(kAttnTileQ)
attn_bwd_stats_kernel(const A* __restrict__ q, const A* __restrict__ k, const A* __restrict__ v,
                      const A* __restrict__ dout, float* __restrict__ stat_m, float* __restrict__ stat_invZ,
                      float* __restrict__ stat_dot, int T, int C, int in_stride) {
    __shared__ float Ks[TILE_K * HD];
    __shared__ float Vs[TILE_K * HD];
    const int    b = blockIdx.z, h = blockIdx.y;
    const int    q0 = blockIdx.x * blockDim.x, i = q0 + threadIdx.x;
    const int    off = h * HD;
    const float  scale    = rsqrtf(static_cast<float>(HD));
    const size_t in_base  = static_cast<size_t>(b) * T * in_stride;
    const size_t out_base = static_cast<size_t>(b) * T * C;
    float qr[HD], dr[HD];
    if (i < T) {
        const A* qi = q + in_base + static_cast<size_t>(i) * in_stride + off;
        const A* di = dout + out_base + static_cast<size_t>(i) * C + off;
        #pragma unroll
        for (int a = 0; a < HD; ++a) { qr[a] = to_f32(qi[a]); dr[a] = to_f32(di[a]); }
    }
    const int jmax = min(T, q0 + static_cast<int>(blockDim.x));
    float m = -1e30f, Z = 0.f;
    for (int j0 = 0; j0 < jmax; j0 += TILE_K) {                        // pass A: m, Z (shared K)
        const int tk = min(TILE_K, jmax - j0);
        __syncthreads();
        for (int e = threadIdx.x; e < tk * HD; e += blockDim.x) {
            const int jl = e / HD, a = e - jl * HD;
            Ks[e] = to_f32(k[in_base + static_cast<size_t>(j0 + jl) * in_stride + off + a]); }
        __syncthreads();
        if (i < T) { const int jend = min(tk, i - j0 + 1);
            for (int jl = 0; jl < jend; ++jl) { const float* ks = Ks + jl * HD;
                float s = 0.f;
                #pragma unroll
                for (int a = 0; a < HD; ++a) s += qr[a] * ks[a];
                s *= scale;
                const float mn = fmaxf(m, s); Z = Z * __expf(m - mn) + __expf(s - mn); m = mn; } }
    }
    const float invZ = (i < T) ? 1.f / Z : 0.f;
    float dot = 0.f;
    for (int j0 = 0; j0 < jmax; j0 += TILE_K) {                        // pass B: dot = sum p_ij <dout_i, v_j>
        const int tk = min(TILE_K, jmax - j0);
        __syncthreads();
        for (int e = threadIdx.x; e < tk * HD; e += blockDim.x) {
            const int jl = e / HD, a = e - jl * HD;
            const size_t row = in_base + static_cast<size_t>(j0 + jl) * in_stride + off;
            Ks[e] = to_f32(k[row + a]); Vs[e] = to_f32(v[row + a]); }
        __syncthreads();
        if (i < T) { const int jend = min(tk, i - j0 + 1);
            for (int jl = 0; jl < jend; ++jl) { const float* ks = Ks + jl * HD; const float* vs = Vs + jl * HD;
                float s = 0.f, dp = 0.f;
                #pragma unroll
                for (int a = 0; a < HD; ++a) { s += qr[a] * ks[a]; dp += dr[a] * vs[a]; }
                dot += __expf(s * scale - m) * invZ * dp; } }
    }
    if (i < T) {
        const size_t sidx = (static_cast<size_t>(b) * gridDim.y + h) * T + i;
        stat_m[sidx] = m; stat_invZ[sidx] = invZ; stat_dot[sidx] = dot;
    }
}

// (2) dq (query-parallel, tiled over keys): dq_i = sum_{j<=i} ds_ij k_j, owned per query (no atomics).
// LANES threads warp-cooperate per query at large HD (attn_dq_lanes<HD>() above): each owns HALF =
// HD/LANES channels of qr/dr/dqa (register need 3*HALF instead of 3*HD), and the two partial dot
// products (s, dp) that need the FULL HD range are combined via a warp-shuffle XOR between the pair.
// LANES=1 makes lane=0, HALF=HD, qi_local=threadIdx.x, and the `if constexpr` shuffle branch compile
// away entirely -- algebraically IDENTICAL to the pre-split single-thread-per-query kernel, so small-
// HD builds are unaffected in code path or performance.
template <class A, int HD, int TILE_K>
__global__ void __launch_bounds__(kAttnTileQ)
attn_bwd_dq_kernel(const A* __restrict__ q, const A* __restrict__ k, const A* __restrict__ v,
                   const A* __restrict__ dout, const float* __restrict__ stat_m,
                   const float* __restrict__ stat_invZ, const float* __restrict__ stat_dot,
                   A* __restrict__ dq, int T, int C, int in_stride) {
    constexpr int LANES = attn_dq_lanes<HD>();
    constexpr int HALF  = HD / LANES;
    __shared__ float Ks[TILE_K * HD];
    __shared__ float Vs[TILE_K * HD];
    const int    b = blockIdx.z, h = blockIdx.y;
    const int    lane     = threadIdx.x % LANES;              // which HALF-channel slice this thread owns
    const int    qi_local = threadIdx.x / LANES;               // this pair's query slot within the block
    const int    q0 = blockIdx.x * (static_cast<int>(blockDim.x) / LANES), i = q0 + qi_local;
    const int    stage_off = h * HD;                           // unshifted: Ks/Vs staging covers the FULL HD
    const int    off = stage_off + lane * HALF;                 // shifted: this thread's own channel half
    const float  scale    = rsqrtf(static_cast<float>(HD));
    const size_t in_base  = static_cast<size_t>(b) * T * in_stride;
    const size_t out_base = static_cast<size_t>(b) * T * C;
    float qr[HALF], dr[HALF], dqa[HALF];
    float mi = 0.f, invZi = 0.f, doti = 0.f;
    if (i < T) {
        const A* qi_p = q + in_base + static_cast<size_t>(i) * in_stride + off;
        const A* di_p = dout + out_base + static_cast<size_t>(i) * C + off;
        #pragma unroll
        for (int a = 0; a < HALF; ++a) { qr[a] = to_f32(qi_p[a]); dr[a] = to_f32(di_p[a]); dqa[a] = 0.f; }
        const size_t sidx = (static_cast<size_t>(b) * gridDim.y + h) * T + i;
        mi = stat_m[sidx]; invZi = stat_invZ[sidx]; doti = stat_dot[sidx];
    }
    const int jmax = min(T, q0 + static_cast<int>(blockDim.x) / LANES);
    for (int j0 = 0; j0 < jmax; j0 += TILE_K) {
        const int tk = min(TILE_K, jmax - j0);
        __syncthreads();
        for (int e = threadIdx.x; e < tk * HD; e += blockDim.x) {
            const int jl = e / HD, a = e - jl * HD;
            const size_t row = in_base + static_cast<size_t>(j0 + jl) * in_stride + stage_off;
            Ks[e] = to_f32(k[row + a]); Vs[e] = to_f32(v[row + a]); }
        __syncthreads();
        if (i < T) { const int jend = min(tk, i - j0 + 1);
            for (int jl = 0; jl < jend; ++jl) {
                const float* ks = Ks + jl * HD + lane * HALF;   // this thread's channel half of key j
                const float* vs = Vs + jl * HD + lane * HALF;
                float s = 0.f, dp = 0.f;
                #pragma unroll
                for (int a = 0; a < HALF; ++a) { s += qr[a] * ks[a]; dp += dr[a] * vs[a]; }
                if constexpr (LANES > 1) {                      // combine the two lanes' partial dot products
                    const unsigned mask = __activemask();       // pair is always both-active or both-inactive
                    s  += __shfl_xor_sync(mask, s,  1);          // (same query i -> same i<T branch outcome)
                    dp += __shfl_xor_sync(mask, dp, 1);
                }
                const float p = __expf(s * scale - mi) * invZi, ds = p * (dp - doti) * scale;
                #pragma unroll
                for (int a = 0; a < HALF; ++a) dqa[a] += ds * ks[a]; } }
    }
    if (i < T) { A* dqi = dq + in_base + static_cast<size_t>(i) * in_stride + off;
        #pragma unroll
        for (int a = 0; a < HALF; ++a) st_act(&dqi[a], dqa[a]); }
}

// (3) dk/dv (key-parallel, tiled over queries), SPLIT into two kernels by register need. The combined
// kernel held kr+vr+dka+dva = 4*HD float registers per thread -- fine at HD<=64 (256 regs, the
// architectural cap) but AT HD=96 (native's d768 config) that is 384, over the 255-register hard limit,
// so ptxas spilled ~50% of them to local memory (measured: 255 regs used, 984B stack frame, 1172B
// spill store/load -- occupancy-limiting AND adds off-register traffic every launch). dv_j = sum p_ij
// dout_i depends only on p (hence kr, for the QK score) -- NOT on v at all, so it splits cleanly into
// its own 2*HD-register kernel. dk_j = sum ds_ij q_i needs BOTH kr (for p) and vr (for dp, feeding ds)
// together, so the 4->3*HD reduction alone isn't enough at large HD -- but it turns out to have the
// SAME shape as dq's s/dp pair (two independent half-width dot products combined into one scalar), so
// it takes the identical warp-cooperative channel split (attn_dk_lanes<HD>() above), dropping the
// per-thread need to 3*(HD/LANES) and eliminating the spill entirely rather than just bounding it (see
// attn_bwd_dk_kernel below). Splitting trades one dk/dv kernel launch for two independent ones (no
// shared state between them -- same causal tiling, same stats reads); dv drops stat_dot/v/Sd entirely
// (dv doesn't need them), so it is also lighter on shared memory and global bandwidth, not just
// registers.
template <class A, int HD, int TILE_Q>
__global__ void __launch_bounds__(kAttnTileQ)
attn_bwd_dv_kernel(const A* __restrict__ q, const A* __restrict__ k,
                   const A* __restrict__ dout, const float* __restrict__ stat_m,
                   const float* __restrict__ stat_invZ,
                   A* __restrict__ dv, int T, int C, int in_stride) {
    __shared__ float Qs[TILE_Q * HD];
    __shared__ float Ds[TILE_Q * HD];
    __shared__ float Sm[TILE_Q], Sz[TILE_Q];
    const int    b = blockIdx.z, h = blockIdx.y;
    const int    k0 = blockIdx.x * blockDim.x, j = k0 + threadIdx.x;      // this thread's KEY
    const int    off = h * HD;
    const float  scale    = rsqrtf(static_cast<float>(HD));
    const size_t in_base  = static_cast<size_t>(b) * T * in_stride;
    const size_t out_base = static_cast<size_t>(b) * T * C;
    float kr[HD], dva[HD];
    if (j < T) {
        const A* kj = k + in_base + static_cast<size_t>(j) * in_stride + off;
        #pragma unroll
        for (int a = 0; a < HD; ++a) { kr[a] = to_f32(kj[a]); dva[a] = 0.f; }
    }
    for (int i0 = k0; i0 < T; i0 += TILE_Q) {                             // only queries i >= j = k0..
        const int ti = min(TILE_Q, T - i0);
        __syncthreads();
        for (int e = threadIdx.x; e < ti * HD; e += blockDim.x) {
            const int il = e / HD, a = e - il * HD;
            Qs[e] = to_f32(q[in_base + static_cast<size_t>(i0 + il) * in_stride + off + a]);
            Ds[e] = to_f32(dout[out_base + static_cast<size_t>(i0 + il) * C + off + a]); }
        for (int il = threadIdx.x; il < ti; il += blockDim.x) {
            const size_t sidx = (static_cast<size_t>(b) * gridDim.y + h) * T + (i0 + il);
            Sm[il] = stat_m[sidx]; Sz[il] = stat_invZ[sidx]; }
        __syncthreads();
        if (j < T) {
            for (int il = 0; il < ti; ++il) { const int i = i0 + il;
                if (i < j) continue;                                     // causal: key j only seen by i >= j
                const float* qs = Qs + il * HD; const float* dsr = Ds + il * HD;
                float s = 0.f;
                #pragma unroll
                for (int a = 0; a < HD; ++a) s += qs[a] * kr[a];
                const float p = __expf(s * scale - Sm[il]) * Sz[il];
                #pragma unroll
                for (int a = 0; a < HD; ++a) dva[a] += p * dsr[a]; }
        }
    }
    if (j < T) {
        A* dvj = dv + in_base + static_cast<size_t>(j) * in_stride + off;
        #pragma unroll
        for (int a = 0; a < HD; ++a) st_act(&dvj[a], dva[a]);
    }
}

// LANES threads warp-cooperate per key at large HD (attn_dk_lanes<HD>() above): each owns HALF =
// HD/LANES channels of kr/vr/dka, and the two partial dot products (s, dp) that need the FULL HD
// range are combined via a warp-shuffle XOR between the pair -- the mirror image of attn_bwd_dq_kernel
// above (there the OWNED state is q/dout and the STREAMED tiles are k/v; here the OWNED state is k/v
// and the STREAMED tiles are q/dout). LANES=1 reduces every formula below to the pre-split
// single-thread-per-key kernel (lane=0, HALF=HD), so small-HD builds are unaffected.
template <class A, int HD, int TILE_Q>
__global__ void __launch_bounds__(kAttnTileQ)
attn_bwd_dk_kernel(const A* __restrict__ q, const A* __restrict__ k, const A* __restrict__ v,
                   const A* __restrict__ dout, const float* __restrict__ stat_m,
                   const float* __restrict__ stat_invZ, const float* __restrict__ stat_dot,
                   A* __restrict__ dk, int T, int C, int in_stride) {
    constexpr int LANES = attn_dk_lanes<HD>();
    constexpr int HALF  = HD / LANES;
    __shared__ float Qs[TILE_Q * HD];
    __shared__ float Ds[TILE_Q * HD];
    __shared__ float Sm[TILE_Q], Sz[TILE_Q], Sd[TILE_Q];
    const int    b = blockIdx.z, h = blockIdx.y;
    const int    lane    = threadIdx.x % LANES;               // which HALF-channel slice this thread owns
    const int    j_local = threadIdx.x / LANES;                // this pair's key slot within the block
    const int    k0 = blockIdx.x * (static_cast<int>(blockDim.x) / LANES), j = k0 + j_local;  // this thread's KEY
    const int    stage_off = h * HD;                           // unshifted: Qs/Ds staging covers the FULL HD
    const int    off = stage_off + lane * HALF;                 // shifted: this thread's own channel half
    const float  scale    = rsqrtf(static_cast<float>(HD));
    const size_t in_base  = static_cast<size_t>(b) * T * in_stride;
    const size_t out_base = static_cast<size_t>(b) * T * C;
    float kr[HALF], vr[HALF], dka[HALF];
    if (j < T) {
        const A* kj = k + in_base + static_cast<size_t>(j) * in_stride + off;
        const A* vj = v + in_base + static_cast<size_t>(j) * in_stride + off;
        #pragma unroll
        for (int a = 0; a < HALF; ++a) { kr[a] = to_f32(kj[a]); vr[a] = to_f32(vj[a]); dka[a] = 0.f; }
    }
    for (int i0 = k0; i0 < T; i0 += TILE_Q) {                             // only queries i >= j = k0..
        const int ti = min(TILE_Q, T - i0);
        __syncthreads();
        for (int e = threadIdx.x; e < ti * HD; e += blockDim.x) {
            const int il = e / HD, a = e - il * HD;
            Qs[e] = to_f32(q[in_base + static_cast<size_t>(i0 + il) * in_stride + stage_off + a]);
            Ds[e] = to_f32(dout[out_base + static_cast<size_t>(i0 + il) * C + stage_off + a]); }
        for (int il = threadIdx.x; il < ti; il += blockDim.x) {
            const size_t sidx = (static_cast<size_t>(b) * gridDim.y + h) * T + (i0 + il);
            Sm[il] = stat_m[sidx]; Sz[il] = stat_invZ[sidx]; Sd[il] = stat_dot[sidx]; }
        __syncthreads();
        if (j < T) {
            for (int il = 0; il < ti; ++il) { const int i = i0 + il;
                if (i < j) continue;                                     // causal: key j only seen by i >= j
                const float* qs = Qs + il * HD + lane * HALF;    // this thread's channel half of query i
                const float* dsr = Ds + il * HD + lane * HALF;
                float s = 0.f, dp = 0.f;
                #pragma unroll
                for (int a = 0; a < HALF; ++a) { s += qs[a] * kr[a]; dp += dsr[a] * vr[a]; }
                if constexpr (LANES > 1) {                      // combine the two lanes' partial dot products
                    const unsigned mask = __activemask();       // pair is always both-active or both-inactive
                    s  += __shfl_xor_sync(mask, s,  1);          // (same key j -> same j<T / i<j branch outcome)
                    dp += __shfl_xor_sync(mask, dp, 1);
                }
                const float p = __expf(s * scale - Sm[il]) * Sz[il], dsc = p * (dp - Sd[il]) * scale;
                #pragma unroll
                for (int a = 0; a < HALF; ++a) dka[a] += dsc * qs[a]; }
        }
    }
    if (j < T) {
        A* dkj = dk + in_base + static_cast<size_t>(j) * in_stride + off;
        #pragma unroll
        for (int a = 0; a < HALF; ++a) st_act(&dkj[a], dka[a]);
    }
}

// Embedding backward (op_embed x2): scatter-add the residual-stream grad into tok_emb / pos_emb
// rows. Multiple rows map to the same token/position, so accumulation is atomic. One thread per
// (row m, channel j). pos = m % T (position within the window).
// C is D_MODEL at this kernel's one call site -- constexpr-folded; T stays runtime (genuinely
// varies, same reasoning as ce_backward_kernel's T).
__global__ void embed_backward_kernel(const float* __restrict__ dh, const int* __restrict__ ids,
                                      float* __restrict__ dtok, float* __restrict__ dpos,
                                      int M, int T) {
    constexpr int C = D_MODEL;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = blockIdx.y * blockDim.y + threadIdx.y;
    if (m < M && j < C) {
        const float g = dh[static_cast<size_t>(m) * C + j];
        atomicAdd(&dtok[static_cast<size_t>(ids[m]) * C + j], g);
        atomicAdd(&dpos[static_cast<size_t>(m % T) * C + j], g);
    }
}

// Token-only embedding backward (RoPE path): scatter the residual-stream grad into tok_emb only
// (no pos_emb, which carries no gradient under RoPE).
// C is D_MODEL at this kernel's one call site -- constexpr-folded.
__global__ void embed_backward_token_kernel(const float* __restrict__ dh, const int* __restrict__ ids,
                                            float* __restrict__ dtok, int M) {
    constexpr int C = D_MODEL;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = blockIdx.y * blockDim.y + threadIdx.y;
    if (m < M && j < C)
        atomicAdd(&dtok[static_cast<size_t>(ids[m]) * C + j], dh[static_cast<size_t>(m) * C + j]);
}

// Split the fused QKV weight gradient dWqkv[C,3C] back into the per-projection grads dWq/dWk/dWv
// [C,C] at their param-blob offsets (inverse of build_qkv_kernel). One thread per [C,C] element.
// C is D_MODEL at its one call site -- constexpr-folded, same div/mod-fold reasoning as above.
__global__ void split_dqkv_kernel(const float* __restrict__ dWqkv, float* __restrict__ dWq,
                                  float* __restrict__ dWk, float* __restrict__ dWv) {
    constexpr int C = D_MODEL;
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

// One-thread kernel: reads the grad L2 normsq the previous kernel just accumulated and writes the
// AdamW clip scale gs = (norm > clip) ? clip/(norm+eps) : 1 -- entirely on-device, so adam_step_kernel
// below can read it via pointer instead of the host needing normsq back just to compute this one
// scalar (see device_adam_step: this is what lets the per-step normsq readback sync disappear).
__global__ void grad_clip_scale_kernel(const double* __restrict__ normsq, float clip,
                                       float* __restrict__ gs) {
    const float norm = sqrtf(static_cast<float>(*normsq));
    *gs = (norm > clip) ? clip / (norm + 1e-6f) : 1.0f;
}

// AdamW per-parameter update (op-for-op identical to AdamW::step on the CPU). gs = global grad
// clip scale (device-resident scalar -- see grad_clip_scale_kernel above), bc1/bc2 = bias
// corrections, decay = 0/1 mask (wd applies to matrices only). decay is uint8 (a 0/1 flag needs no
// more): this array is PARAM_FLOATS long and persists for the whole run, so its storage type is pure
// overhead -- was float (4B/param, 627MB at production scale), now 1B.
__global__ void adam_step_kernel(float* __restrict__ p, const float* __restrict__ grad,
                                 float* __restrict__ m, float* __restrict__ vel,
                                 const unsigned char* __restrict__ decay, int n,
                                 const float* __restrict__ gs_ptr, float lr,
                                 float b1, float b2, float eps, float wd, float bc1, float bc2) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float g = grad[i] * (*gs_ptr);
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

// cuBLAS compute type for the FP32-storage GEMMs: when the config bakes GEMM BF16 AND the tensor
// path is enabled (CudaTf32 knob; parity tests force it off), use 32F_FAST_16BF -- inputs/outputs
// stay FP32 in memory, cuBLAS down-converts to BF16 for the tensor cores and accumulates in FP32.
// Otherwise pure FP32. Keeps the parity gate FP32-exact while making "GEMM BF16" genuinely active.
// `force_tc` requests bf16-tensor-core compute (FAST_16BF) whenever GEMM_DTYPE==BF16, independent
// of the CudaTf32 knob -- for GEMMs where cuBLAS's own algorithm heuristic (under the knob's default
// plain CUBLAS_COMPUTE_32F) does not reliably land on a tensor-core kernel by itself. Measured (nsys,
// 2026-07-02): the lm_head GEMM's awkward VOCAB-wide N (e.g. 32873) falls back to a CUDA-core SGEMM
// (magma_sgemmEx_kernel, ~60-100ms/call) while every other GEMM in the same forward/backward pass
// (QKV/attn-out/FFN, all "nicer" N like D_MODEL/D_FF) already lands on a bf16 tensor-op CUTLASS
// kernel (<1ms/call) under that SAME default compute type -- so the rest of the network is already
// running at this precision level; only the lm_head wasn't getting it. Explicit request only for
// that call site; every other caller's already-working behavior is untouched.
inline cublasComputeType_t gemm_compute(bool force_tc = false) {
    return (GEMM_DTYPE == Dtype::BF16 && (force_tc || CudaTf32::get())) ? CUBLAS_COMPUTE_32F_FAST_16BF
                                                                        : CUBLAS_COMPUTE_32F;
}
// Thin cublasGemmEx wrapper (FP32 A/B/C) honoring gemm_compute(); replaces cublasSgemm everywhere.
// beta defaults to 0 (OVERWRITE C) -- the behavior of every pre-existing call site. beta=1
// ACCUMULATES C += A.B; not used by any call site yet (added groundwork for a future row-chunked
// GEMM, e.g. splitting the lm_head backward over M -- see Wave 8 in memory), but exercised directly
// by its own dedicated test so it's verified correct before anything depends on it.
inline void gemm(cublasOperation_t opA, cublasOperation_t opB, int m, int n, int k,
                 const float* A, int lda, const float* B, int ldb, float* C, int ldc,
                 bool force_tc = false, float beta = 0.0f) {
    const float alpha = 1.0f;
    cublasGemmEx(g_cublas, opA, opB, m, n, k, &alpha, A, CUDA_R_32F, lda, B, CUDA_R_32F, ldb,
                 &beta, C, CUDA_R_32F, ldc, gemm_compute(force_tc), CUBLAS_GEMM_DEFAULT);
}
template <class T> constexpr cudaDataType_t cu_type() {
    return std::is_same_v<T, __nv_bfloat16> ? CUDA_R_16BF : CUDA_R_32F;
}
// Mixed-type GemmEx: A/B share type IN, output type OUT; FP32 accumulate always. Used for bf16
// activation GEMMs (IN=bf16 weights+acts) writing either bf16 (chained) or f32 (residual) output.
template <class IN, class OUT>
inline void gemm_t(cublasOperation_t opA, cublasOperation_t opB, int m, int n, int k,
                   const IN* A, int lda, const IN* B, int ldb, OUT* C, int ldc) {
    const float alpha = 1.0f, beta = 0.0f;
    cublasGemmEx(g_cublas, opA, opB, m, n, k, &alpha, A, cu_type<IN>(), lda, B, cu_type<IN>(), ldb,
                 &beta, C, cu_type<OUT>(), ldc, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
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
                          int M, int in, int out, bool force_tc = false) {
    ensure_cublas();
    gemm(CUBLAS_OP_N, CUBLAS_OP_N, out, M, in, dW, out, dX, in, dY, out, force_tc);
    if (dB) {
        // 64-bit: M*out (lm_head: out=VOCAB) overflows int32 well within reachable batch sizes --
        // see bias_add_kernel's comment. grid.x itself stays far under the 2^31-1 hardware cap.
        const long long n = static_cast<long long>(M) * out;
        const int block = 256;
        const int grid = static_cast<int>((n + block - 1) / block);
        bias_add_kernel<<<grid, block, 0, g_stream>>>(dY, dB, M, out);
    }
}
// act-typed linear: X(IN) . W(IN) -> Y(OUT), bias OUT. IN=act_t for bf16 GEMMs (W is the bf16
// mirror); OUT chains bf16 or lands f32 at a residual boundary. No bias on the bf16 chain GEMMs.
template <class IN, class OUT>
inline void launch_linear_t(const IN* dX, const IN* dW, OUT* dY, int M, int in, int out) {
    ensure_cublas();
    gemm_t<IN, OUT>(CUBLAS_OP_N, CUBLAS_OP_N, out, M, in, dW, out, dX, in, dY, out);
}
inline void launch_rmsnorm(const float* dX, const float* dG, float* dY, int T) {
    constexpr int kNormBlock = 64;   // one block per row; divides every D_MODEL this project bakes
    rmsnorm_kernel<kNormBlock><<<T, kNormBlock, 0, g_stream>>>(dX, dG, dY, T);
}
inline void launch_gelu(const float* dX, float* dY, int n) {
    const int block = 256;
    gelu_kernel<<<(n + block - 1) / block, block, 0, g_stream>>>(dX, dY, n);
}
inline void launch_add(const float* dA, const float* dB, float* dC, int n) {
    const int block = 256;
    add_kernel<<<(n + block - 1) / block, block, 0, g_stream>>>(dA, dB, dC, n);
}
template <class A> inline void launch_add_t(const A* dA, const A* dB, A* dC, int n) {
    const int block = 256;
    add_act_kernel<A><<<(n + block - 1) / block, block, 0, g_stream>>>(dA, dB, dC, n);
}
// act-typed FFN launchers (bf16 storage): rmsnorm f32->act, bias-add on act, gelu/gelu-bwd act.
template <class X, class Y>
inline void launch_rmsnorm_train_t(const X* dX, const float* dG, Y* dY, float* dRinv, int T) {
    constexpr int kNormBlock = 64;   // one block per row; divides every D_MODEL this project bakes
    rmsnorm_train_act_kernel<X, Y, kNormBlock><<<T, kNormBlock, 0, g_stream>>>(dX, dG, dY, dRinv, T);
}
// 64-bit index/bound -- same overflow reasoning as bias_add_kernel above (this one is reached at a
// higher batch, N=D_FF scale, but the same UB applies once it is).
template <class A>
__global__ void bias_add_act_kernel(A* __restrict__ Y, const float* __restrict__ bias, int M, int N) {
    const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < static_cast<long long>(M) * N) st_act(&Y[i], to_f32(Y[i]) + bias[i % N]);
}
template <class A> inline void launch_bias_act(A* dY, const float* dB, int M, int N) {
    const long long n = static_cast<long long>(M) * N;
    const int block = 256;
    const int grid = static_cast<int>((n + block - 1) / block);
    bias_add_act_kernel<A><<<grid, block, 0, g_stream>>>(dY, dB, M, N);
}
template <class A> inline void launch_gelu_t(const A* dX, A* dY, int n) {
    const int block = 256; gelu_act_kernel<A><<<(n + block - 1) / block, block, 0, g_stream>>>(dX, dY, n);
}
template <class A>
__global__ void gelu_backward_act_kernel(const A* __restrict__ x, const A* __restrict__ dy, A* __restrict__ dx, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) st_act(&dx[i], to_f32(dy[i]) * dev_dgelu(to_f32(x[i])));
}
template <class A> inline void launch_gelu_bwd_t(const A* x, const A* dy, A* dx, int n) {
    const int block = 256; gelu_backward_act_kernel<A><<<(n + block - 1) / block, block, 0, g_stream>>>(x, dy, dx, n);
}
inline void launch_attn(const float* dQ, const float* dK, const float* dV, float* dOut,
                        int batch, int T, int C, int H, int in_stride) {
    constexpr int HD = D_HEAD, TK = attn_tile_k<HD>();       // head dim baked -> tiled kernel (see above)
    const dim3 block(kAttnTileQ);
    const dim3 grid((T + kAttnTileQ - 1) / kAttnTileQ, H, batch);
    attn_fwd_tiled_kernel<float, HD, TK><<<grid, block, 0, g_stream>>>(dQ, dK, dV, dOut, T, C, in_stride);
}
// RoPE: rotate Q/K (forward) or dQ/dK (backward) in place over the fused [*, in_stride] buffer.
// One thread per (row, pair); grid.x over C/2 pairs, grid.y over batch*T rows. C/H/in_stride dropped
// as parameters -- constexpr-folded inside the kernels themselves, see rope_kernel above.
inline void launch_rope(float* qkv, int batch, int T) {
    const dim3 block(32, 8);
    const dim3 grid((D_MODEL / 2 + block.x - 1) / block.x, (batch * T + block.y - 1) / block.y);
    rope_kernel<<<grid, block, 0, g_stream>>>(qkv, batch, T, ROPE_THETA);
}
template <class A> inline void launch_rope_t(A* qkv, int batch, int T) {
    const dim3 block(32, 8);
    const dim3 grid((D_MODEL / 2 + block.x - 1) / block.x, (batch * T + block.y - 1) / block.y);
    rope_act_kernel<A><<<grid, block, 0, g_stream>>>(qkv, batch, T, ROPE_THETA);
}
template <class A> inline void launch_rope_bwd_t(A* dqkv, int batch, int T) {
    const dim3 block(32, 8);
    const dim3 grid((D_MODEL / 2 + block.x - 1) / block.x, (batch * T + block.y - 1) / block.y);
    rope_bwd_act_kernel<A><<<grid, block, 0, g_stream>>>(dqkv, batch, T, ROPE_THETA);
}

// Linear backward (mirrors the Op::Linear case): given the forward input X[M,in], weight W[in,out]
// and upstream grad dY[M,out], produce dX[M,in] = dY.W^T and dW[in,out] = X^T.dY via two cuBLAS
// GEMMs (same column-major trick as launch_linear), plus the bias column-sum. With the default
// accumulate=false, dW/dbias write straight into the param grad blob (beta=0; each weight is used
// once per forward). accumulate=true switches ONLY the dW GEMM and the dbias column-sum to +=
// (their M dimension is a reduction axis, so a row-chunked caller must sum partial contributions
// into a pre-zeroed output); not used by any call site yet -- see gemm()'s beta comment. dX stays
// an overwrite either way -- its rows are per-chunk DISJOINT, each written exactly once. dX may be
// null for the bottom of the graph (no input grad needed).
inline void launch_linear_bwd(const float* dX_in, const float* dW_in, const float* dY,
                              float* dX, float* dW, float* dbias, int M, int in, int out,
                              bool force_tc = false, bool accumulate = false) {
    if (dX)
        gemm(CUBLAS_OP_T, CUBLAS_OP_N, in, M, out, dW_in, out, dY, out, dX, in, force_tc);   // dX = dY . W^T
    gemm(CUBLAS_OP_N, CUBLAS_OP_T, out, in, M, dY, out, dX_in, in, dW, out, force_tc,
         accumulate ? 1.0f : 0.0f);                                                          // dW (+)= X^T . dY
    if (dbias) {
        const int block = 128;
        bias_grad_kernel<<<(out + block - 1) / block, block, 0, g_stream>>>(dY, dbias, M, out, accumulate);
    }
}
// act-typed linear backward: bf16 inputs/grads, dW lands F32 in the grad blob, dbias optional F32.
template <class IN, class DX>
inline void launch_linear_bwd_t(const IN* dX_in, const IN* dW16, const IN* dY,
                                DX* dX, float* dW, int M, int in, int out) {
    if (dX) gemm_t<IN, DX>(CUBLAS_OP_T, CUBLAS_OP_N, in, M, out, dW16, out, dY, out, dX, in);   // dX=dY.W^T
    gemm_t<IN, float>(CUBLAS_OP_N, CUBLAS_OP_T, out, in, M, dY, out, dX_in, in, dW, out);       // dW=X^T.dY
}
template <class X> inline void launch_rmsnorm_bwd_t(const X* x, const float* gamma, const float* rinv,
                               const float* dy, float* dx, float* dgamma, int rows) {
    constexpr int kNormBlock = 64;   // one block per row; divides every D_MODEL this project bakes
    rmsnorm_backward_act_kernel<X, kNormBlock><<<rows, kNormBlock, 0, g_stream>>>(
        x, gamma, rinv, dy, dx, dgamma, rows);
}
// act-typed attention forward/backward (bf16 store): same launch shape, store-typed q/k/v/att.
template <class A> inline void launch_attn_train_t(const A* dQ, const A* dK, const A* dV, A* dOut,
                              int batch, int T, int C, int H, int in_stride) {
    constexpr int HD = D_HEAD, TK = attn_tile_k<HD>();       // head dim baked -> tiled kernel (see above)
    const dim3 block(kAttnTileQ);
    const dim3 grid((T + kAttnTileQ - 1) / kAttnTileQ, H, batch);
    attn_fwd_tiled_kernel<A, HD, TK><<<grid, block, 0, g_stream>>>(dQ, dK, dV, dOut, T, C, in_stride);
}

// Per-query softmax-stats scratch for the flash backward (m_i, 1/Z_i, dot_i), each batch*H*T floats.
// Grown on demand (like the fwd/train scratch) and reused across steps; freed in train_free().
float* g_bwd_m = nullptr; float* g_bwd_invZ = nullptr; float* g_bwd_dot = nullptr;
size_t g_bwd_stats_cap = 0;
inline int ensure_bwd_stats(size_t n) {
    if (n <= g_bwd_stats_cap) return 0;
    if (g_bwd_m)    cudaFree(g_bwd_m);
    if (g_bwd_invZ) cudaFree(g_bwd_invZ);
    if (g_bwd_dot)  cudaFree(g_bwd_dot);
    g_bwd_stats_cap = 0;
    if (cudaMalloc(&g_bwd_m,    n * sizeof(float)) != cudaSuccess) return 1;
    if (cudaMalloc(&g_bwd_invZ, n * sizeof(float)) != cudaSuccess) return 1;
    if (cudaMalloc(&g_bwd_dot,  n * sizeof(float)) != cudaSuccess) return 1;
    g_bwd_stats_cap = n;
    return 0;
}
inline void free_bwd_stats() {
    if (g_bwd_m)    cudaFree(g_bwd_m);
    if (g_bwd_invZ) cudaFree(g_bwd_invZ);
    if (g_bwd_dot)  cudaFree(g_bwd_dot);
    g_bwd_m = g_bwd_invZ = g_bwd_dot = nullptr; g_bwd_stats_cap = 0;
}

// Flash attention backward: stats -> dq (query-parallel) -> dv, dk (key-parallel, register-split; see
// the comment above attn_bwd_dv_kernel), all atomic-free. dqkv need NOT be pre-zeroed (each dq_i /
// dk_j / dv_j is written exactly once, in full).
template <class A> inline void launch_attn_bwd_t(const A* qkv, const A* dout, A* dqkv,
                            int batch, int T, int C, int H, int in_stride) {
    constexpr int HD = D_HEAD, TK = attn_tile_k<HD>(), TQ = attn_tile_q<HD>();
    constexpr int DQ_LANES = attn_dq_lanes<HD>();
    constexpr int DK_LANES = attn_dk_lanes<HD>();
    if (ensure_bwd_stats(static_cast<size_t>(batch) * H * T)) return;    // OOM -> next sync surfaces it
    const dim3 block(kAttnTileQ);
    const dim3 grid((T + kAttnTileQ - 1) / kAttnTileQ, H, batch);
    // dq's block covers kAttnTileQ/DQ_LANES queries (LANES threads warp-cooperate per query at large
    // HD; see attn_dq_lanes), so its grid.x scales inversely -- more, smaller-coverage blocks. dk is
    // the same idea over keys instead of queries (attn_dk_lanes).
    const dim3 grid_dq((T + kAttnTileQ / DQ_LANES - 1) / (kAttnTileQ / DQ_LANES), H, batch);
    const dim3 grid_dk((T + kAttnTileQ / DK_LANES - 1) / (kAttnTileQ / DK_LANES), H, batch);
    const A* q = qkv; const A* k = qkv + C; const A* v = qkv + 2 * C;
    A* dq = dqkv; A* dk = dqkv + C; A* dv = dqkv + 2 * C;
    attn_bwd_stats_kernel<A, HD, TK><<<grid, block, 0, g_stream>>>(q, k, v, dout, g_bwd_m, g_bwd_invZ, g_bwd_dot, T, C, in_stride);
    attn_bwd_dq_kernel<A, HD, TK><<<grid_dq, block, 0, g_stream>>>(q, k, v, dout, g_bwd_m, g_bwd_invZ, g_bwd_dot, dq, T, C, in_stride);
    attn_bwd_dv_kernel<A, HD, TQ><<<grid, block, 0, g_stream>>>(q, k, dout, g_bwd_m, g_bwd_invZ, dv, T, C, in_stride);
    attn_bwd_dk_kernel<A, HD, TQ><<<grid_dk, block, 0, g_stream>>>(q, k, v, dout, g_bwd_m, g_bwd_invZ, g_bwd_dot, dk, T, C, in_stride);
}

// Hard cap on the minibatch the resident device scratch will size to. Decoupled from the CPU's
// data-parallel width: the GPU wants a larger batch (bigger GEMM M = better utilization). The
// scratch is sized to the ACTUAL batch requested (grow-on-demand, see fwd_alloc/train_alloc), so a
// small-batch run does not over-allocate; this is just the ceiling a request is validated against.
// MEASURED 2026-06 on sm_120 (train tok/s): 64->62k 128->105k 256->157k 384->176k 512->191k
// 768->195k 1024->209k -- throughput keeps rising but with diminishing returns past ~512. cudaMalloc
// failure (too big for VRAM) is handled gracefully (the alloc returns nonzero, the caller errors).
// This is just a sanity ceiling: the real per-run upper bound is set by VRAM (memplan::max_batch_for_vram),
// computed at tune time, so bf16's halved footprint can reach batches the old 1024 list could not.
// One number shared with the footprint model and the train stage's token-budget batch scheduler.
constexpr int MAX_FWD_BATCH = sub0::memplan::MAX_DEVICE_BATCH;

// Resident forward scratch. The batch-dependent buffers are sized to the largest batch seen so far
// (g_fwd_cap, grown on demand); the fused-QKV weight buffers (wqkv) are batch-independent and built
// once at upload. Freed by sub0_cuda_shutdown.
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
};
FwdScratch g_fwd;
size_t     g_fwd_rows = 0;             // total [M] ROW capacity the batch-dependent buffers are sized for
                                       // (batch*T row-product, NOT a batch count -- so a varied-T training
                                       // step can trade batch against T inside one reserved budget)
int        g_fwd_full = 0;             // 1 = full inference scratch allocated; 0 = dids-only (training)
long long  g_fwd_grows = 0;            // monotonic (re)allocation count -- test observability
// BF16 builds only: true when the lazily-built F32 wqkv (ensure_wqkv_f32, for sub0_cuda_forward /
// sub0_cuda_forward_one) is stale relative to the current weights -- set by build_qkv_weights() on
// every weight change, cleared once ensure_wqkv_f32() rebuilds it.
bool g_wqkv_f32_dirty = true;

// Batch-independent fused-QKV weight buffer (F32, [C,3C] per layer). BF16 builds no longer keep
// this resident during training -- build_qkv_weights() below writes the bf16 mirror (g_wqkv16)
// DIRECTLY from Wq/Wk/Wv instead of staging through this F32 buffer, so it's dead weight for a pure
// training run (~108 MiB at production dims). The F32 CUDA inference paths (sub0_cuda_forward,
// sub0_cuda_forward_one) still need it -- see ensure_wqkv_f32() below, which allocates+builds it
// lazily, on demand, only in a process that actually calls them. F32-activation builds are
// unaffected: g_wqkv16 there ALIASES this buffer directly (build_qkv_weights()'s F32 branch), so it
// stays eagerly built exactly as before.
int wqkv_alloc() {
    if constexpr (ACT_DTYPE == Dtype::BF16) return 0;   // handled by ensure_wqkv_f32() / build_qkv_weights()
    if (g_fwd.wqkv[0]) return 0;
    ensure_stream();
    for (int l = 0; l < N_LAYERS; ++l)
        SUB0_CUDA_CHECK(cudaMalloc(&g_fwd.wqkv[l], static_cast<size_t>(D_MODEL) * 3 * D_MODEL * sizeof(float)));
    return 0;
}

// Free only the batch-dependent forward buffers (for a grow-realloc); leaves wqkv intact.
void invalidate_graph();           // fwd: a grow-realloc frees buffers the captured graph references
void fwd_free_batch() {
    invalidate_graph();            // the captured forward graph references these buffers -> drop it
    cudaFree(g_fwd.dids); cudaFree(g_fwd.h);    cudaFree(g_fwd.a);    cudaFree(g_fwd.qkv);
    cudaFree(g_fwd.att);  cudaFree(g_fwd.proj); cudaFree(g_fwd.fbuf); cudaFree(g_fwd.ff1);
    cudaFree(g_fwd.gact); cudaFree(g_fwd.ff2);  cudaFree(g_fwd.logits);
    g_fwd.dids = nullptr; g_fwd.h = g_fwd.a = g_fwd.qkv = g_fwd.att = g_fwd.proj = g_fwd.fbuf =
        g_fwd.ff1 = g_fwd.gact = g_fwd.ff2 = g_fwd.logits = nullptr;
    g_fwd_rows = 0;
    g_fwd_full = 0;
}

// Ensure the forward scratch covers batch*T rows (grow-on-demand) plus the wqkv buffers. Capacity is
// the ROW PRODUCT, not the batch: any (batch, T) pair whose batch*T fits the reserved rows is served
// without reallocating, so the training loop can grow its effective batch as T shrinks inside one
// fixed token budget (batch_t*T ~= tuned_batch*SEQ_LEN). T defaults to SEQ_LEN, which reproduces the
// old "sized for `batch` full-length windows" behavior exactly for every caller that passes a plain
// batch. `full` allocates the inference activation scratch (h..logits); training passes full=false
// (it reads from g_tr, only needs dids + wqkv) -- skipping ~6 [M,C]/[M,F]/[M,V] buffers saves
// hundreds of MiB during training.
int fwd_alloc(int batch, bool full = true, int T = SEQ_LEN) {
    if (wqkv_alloc()) return 1;
    const size_t need = static_cast<size_t>(batch) * static_cast<size_t>(T);
    if (g_fwd_rows >= need && g_fwd.dids && (!full || g_fwd_full)) return 0;   // already big enough
    fwd_free_batch();                                      // grow: drop the old (smaller) buffers
    ++g_fwd_grows;
    const size_t Mm = need;
    const size_t MC = Mm * D_MODEL, MF = Mm * D_FF, MV = Mm * VOCAB;
    SUB0_CUDA_CHECK(cudaMalloc(&g_fwd.dids,   Mm * sizeof(int)));
    if (full) {
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
    }
    g_fwd_rows = Mm;
    g_fwd_full = full ? 1 : 0;
    return 0;
}

void fwd_free() {
    fwd_free_batch();
    for (int l = 0; l < N_LAYERS; ++l) { cudaFree(g_fwd.wqkv[l]); g_fwd.wqkv[l] = nullptr; }
    g_wqkv_f32_dirty = true;   // BF16: force ensure_wqkv_f32() to rebuild after the next (re)alloc
}

// Resident TRAINING scratch (Phase 2d): the forward saves every activation the backward needs
// (no recompute), plus the gradient temporaries that thread the reverse pass. Sized to a ROW budget
// (batch*T product, grown on demand -- see train_alloc). Freed by sub0_cuda_shutdown.
struct TrainScratch {
    // per-layer saved forward activations (residual stream stored bf16; transient scratch is act_t)
    act_t* h_in [N_LAYERS] = {};   // [M,C] layer input (= rmsnorm1 input)
    float* rinv1[N_LAYERS] = {};   // [M]   rmsnorm1 reciprocal-rms
    act_t* a    = nullptr;         // [M,C] rmsnorm1 output scratch (bf16) -- recomputed from h_in in backward (not per-layer)
    act_t* qkv  = nullptr;         // [M,3C] fused q|k|v scratch (bf16) -- recomputed from a in backward (not per-layer)
    act_t* att  = nullptr;         // [M,C] attention output scratch (bf16) -- recomputed from qkv in backward (not per-layer)
    act_t* h_mid[N_LAYERS] = {};   // [M,C] after residual-1 (= rmsnorm2 input)
    float* rinv2[N_LAYERS] = {};   // [M]   rmsnorm2 reciprocal-rms
    act_t* fbuf = nullptr;         // [M,C] rmsnorm2 output scratch -- recomputed from h_mid in backward (= W1 input)
    act_t* ff1  = nullptr;         // [M,F] pre-GELU scratch -- recomputed from fbuf in backward (not per-layer)
    act_t* gact = nullptr;         // [M,F] GELU output scratch -- recomputed from fbuf in backward (not per-layer)
    // final block
    act_t* h_final = nullptr;      // [M,C] last residual stream (= rmsnorm_f input)
    float* rinv_f  = nullptr;      // [M]
    float* a_final = nullptr;      // [M,C] rmsnorm_f output (= lm_head input)
    float* logits  = nullptr;      // [M,V]
    // gradient temporaries (reused across layers); dh threads the residual stream in F32
    float* dh      = nullptr;      // [M,C] running residual-stream grad
    float* da      = nullptr;      // [M,C] f32 (feeds rmsnorm1 bwd dy)
    act_t* dqkv    = nullptr;      // [M,3C] bf16
    act_t* datt    = nullptr;      // [M,C] bf16
    float* dfbuf   = nullptr;      // [M,C]
    act_t* dff1    = nullptr;      // [M,F]
    act_t* dgact   = nullptr;      // [M,F]
    float* dlogits = nullptr;      // [M,V]
    float* dwqkv   = nullptr;      // [C,3C] fused QKV weight-grad temp
    act_t* dh16    = nullptr;      // [M,C] bf16 cast of dh (FFN W2 backward operand)
    double* loss   = nullptr;      // [1] accumulated cross-entropy (device)
    int*   dtargets = nullptr;     // [M] next-token targets for cross-entropy
    int*   lengths  = nullptr;     // [MAX_FWD_BATCH] per-window trained length (padding mask for short
                                   // docs) -- constant-sized (16 KiB) so it covers ANY effective batch
                                   // the row budget admits, decoupled from the rows the scratch grows to
};
TrainScratch g_tr;
size_t       g_tr_rows = 0;        // total [M] ROW capacity (batch*T product) the buffers are sized for
long long    g_tr_grows = 0;       // monotonic (re)allocation count -- test observability
// Per-layer bf16 weight mirrors for the FFN GEMMs (built from the F32 master on upload/step).
act_t* g_w1_16[N_LAYERS] = {};     // [C,F]
act_t* g_w2_16[N_LAYERS] = {};     // [F,C]
act_t* g_wo16[N_LAYERS]  = {};     // [C,C] attention output proj mirror
act_t* g_wqkv16[N_LAYERS] = {};    // [C,3C] fused QKV mirror (bf16 acts) / alias under f32

void train_free();                 // fwd: train_alloc frees the old buffers before a grow-realloc
// Row-product capacity, exactly like fwd_alloc above: any (batch, T) with batch*T <= g_tr_rows is
// served without reallocating; T defaults to SEQ_LEN so existing plain-batch callers keep their old
// sizing bit-for-bit. Callers that vary (batch, T) per step should reserve the full budget up front
// (sub0_cuda_train_reserve) so the varying products never trigger a grow-realloc mid-run.
int train_alloc(int batch, int T = SEQ_LEN) {
    const size_t need = static_cast<size_t>(batch) * static_cast<size_t>(T);
    if (g_tr_rows >= need && g_tr.h_in[0]) return 0;       // already big enough
    if (g_tr.h_in[0]) train_free();                        // grow: drop the old (smaller) buffers
    ensure_stream();
    ++g_tr_grows;
    const size_t Mm = need;
    const size_t MC = Mm * D_MODEL, MF = Mm * D_FF, MV = Mm * VOCAB;
    // TODO(mem): BF16 activation storage (sm>=80) would roughly halve the per-layer/final scratch -- the
    // biggest remaining lever for larger batches now that qkv/att/ff1/gact are checkpointed to singles.
    for (int l = 0; l < N_LAYERS; ++l) {
        SUB0_CUDA_CHECK(cudaMalloc(&g_tr.h_in[l],  MC * sizeof(act_t)));
        SUB0_CUDA_CHECK(cudaMalloc(&g_tr.rinv1[l], Mm * sizeof(float)));
        SUB0_CUDA_CHECK(cudaMalloc(&g_tr.h_mid[l], MC * sizeof(act_t)));
        SUB0_CUDA_CHECK(cudaMalloc(&g_tr.rinv2[l], Mm * sizeof(float)));
    }
    SUB0_CUDA_CHECK(cudaMalloc(&g_tr.a,      MC * sizeof(act_t)));   // single (checkpoint scratch, bf16)
    SUB0_CUDA_CHECK(cudaMalloc(&g_tr.fbuf,   MC * sizeof(act_t)));  // single (checkpoint scratch, bf16)
    SUB0_CUDA_CHECK(cudaMalloc(&g_tr.ff1,    MF * sizeof(act_t)));  // single (checkpoint scratch, bf16)
    SUB0_CUDA_CHECK(cudaMalloc(&g_tr.gact,   MF * sizeof(act_t)));  // single (checkpoint scratch, bf16)
    SUB0_CUDA_CHECK(cudaMalloc(&g_tr.qkv,    Mm * 3 * D_MODEL * sizeof(act_t)));  // single (checkpoint scratch, bf16)
    SUB0_CUDA_CHECK(cudaMalloc(&g_tr.att,    MC * sizeof(act_t)));   // single (checkpoint scratch, bf16)
    SUB0_CUDA_CHECK(cudaMalloc(&g_tr.h_final, MC * sizeof(act_t)));
    SUB0_CUDA_CHECK(cudaMalloc(&g_tr.rinv_f,  Mm * sizeof(float)));
    SUB0_CUDA_CHECK(cudaMalloc(&g_tr.a_final, MC * sizeof(float)));
    SUB0_CUDA_CHECK(cudaMalloc(&g_tr.logits,  MV * sizeof(float)));
    SUB0_CUDA_CHECK(cudaMalloc(&g_tr.dh,      MC * sizeof(float)));
    SUB0_CUDA_CHECK(cudaMalloc(&g_tr.da,      MC * sizeof(float)));
    SUB0_CUDA_CHECK(cudaMalloc(&g_tr.dqkv,    Mm * 3 * D_MODEL * sizeof(act_t)));
    SUB0_CUDA_CHECK(cudaMalloc(&g_tr.datt,    MC * sizeof(act_t)));
    SUB0_CUDA_CHECK(cudaMalloc(&g_tr.dfbuf,   MC * sizeof(float)));
    SUB0_CUDA_CHECK(cudaMalloc(&g_tr.dff1,    MF * sizeof(act_t)));
    SUB0_CUDA_CHECK(cudaMalloc(&g_tr.dgact,   MF * sizeof(act_t)));
    g_tr.dlogits = g_tr.logits;                 // CE backward is in-place: dlogits overwrites logits [M,V]
    SUB0_CUDA_CHECK(cudaMalloc(&g_tr.dwqkv,   static_cast<size_t>(D_MODEL) * 3 * D_MODEL * sizeof(float)));
    SUB0_CUDA_CHECK(cudaMalloc(&g_tr.dh16,    MC * sizeof(act_t)));   // bf16 cast of dh for FFN W2 bwd
    SUB0_CUDA_CHECK(cudaMalloc(&g_tr.loss,    sizeof(double)));
    SUB0_CUDA_CHECK(cudaMalloc(&g_tr.dtargets, Mm * sizeof(int)));
    // lengths is indexed [0, batch) at step time, where batch can be any value up to MAX_FWD_BATCH
    // that the row budget admits (a short-T step runs MORE windows). Constant-size it to the ceiling
    // (16 KiB) instead of the reserving batch, so no legal (batch, T) pair can overrun it.
    SUB0_CUDA_CHECK(cudaMalloc(&g_tr.lengths,  static_cast<size_t>(MAX_FWD_BATCH) * sizeof(int)));
    g_tr_rows = Mm;
    return 0;
}

void train_free() {
    for (int l = 0; l < N_LAYERS; ++l) {
        cudaFree(g_tr.h_in[l]);  cudaFree(g_tr.rinv1[l]);
        cudaFree(g_tr.h_mid[l]); cudaFree(g_tr.rinv2[l]);
    }
    cudaFree(g_tr.a); cudaFree(g_tr.fbuf); cudaFree(g_tr.ff1); cudaFree(g_tr.gact); cudaFree(g_tr.qkv); cudaFree(g_tr.att);
    cudaFree(g_tr.h_final); cudaFree(g_tr.rinv_f); cudaFree(g_tr.a_final); cudaFree(g_tr.logits);
    cudaFree(g_tr.dh);   cudaFree(g_tr.da);   cudaFree(g_tr.dqkv); cudaFree(g_tr.datt);
    cudaFree(g_tr.dfbuf); cudaFree(g_tr.dff1); cudaFree(g_tr.dgact); // dlogits aliases logits (freed above)
    cudaFree(g_tr.dwqkv); cudaFree(g_tr.dh16); cudaFree(g_tr.loss); cudaFree(g_tr.dtargets); cudaFree(g_tr.lengths);
    free_bwd_stats();                                   // flash-backward per-query stats scratch
    g_tr = TrainScratch{};
    g_tr_rows = 0;
}


// ============================================================================
//  GPU incremental single-token inference (device KV-cache decode)
// ============================================================================
// The device counterpart of the CPU forward_one: a resident per-layer K/V cache + a per-token forward
// over a SINGLE row (reusing the M=1 rmsnorm/linear/gelu/add launches), so autoregressive gen runs on
// the device at O(T) per token instead of re-forwarding the whole context. Positions must stay
// < SEQ_LEN. Dense FP32 params (g_dev_params + the fused g_fwd.wqkv), same weights as sub0_cuda_forward.
float* g_kv_k = nullptr;   // [N_LAYERS * SEQ_LEN * D_MODEL] -- roped K per (layer, position)
float* g_kv_v = nullptr;   // [N_LAYERS * SEQ_LEN * D_MODEL] -- V per (layer, position)

inline int kv_alloc() {
    if (g_kv_k) return 0;
    const size_t n = static_cast<size_t>(N_LAYERS) * SEQ_LEN * D_MODEL;
    if (cudaMalloc(&g_kv_k, n * sizeof(float)) != cudaSuccess) return 1;
    if (cudaMalloc(&g_kv_v, n * sizeof(float)) != cudaSuccess) return 1;
    return 0;
}
inline void kv_free() { if (g_kv_k) cudaFree(g_kv_k); if (g_kv_v) cudaFree(g_kv_v); g_kv_k = g_kv_v = nullptr; }

// Embed one token into h[C] (+ pos_emb[pos] under Absolute; pos_emb == nullptr under RoPE).
// C is D_MODEL at this kernel's one call site (the per-token decode path) -- constexpr-folded.
__global__ void embed_one_kernel(const float* __restrict__ tok_emb, const float* __restrict__ pos_emb,
                                 int id, int pos, float* __restrict__ h) {
    constexpr int C = D_MODEL;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= C) return;
    float v = tok_emb[static_cast<size_t>(id) * C + j];
    if (pos_emb) v += pos_emb[static_cast<size_t>(pos) * C + j];
    h[j] = v;
}
// RoPE one row: rotate q (cols [0,C)) and k (cols [C,2C)) of a fused [3C] qkv row at position pos
// (mirrors rope_kernel with t = pos). V (cols [2C,3C)) untouched. C/H constexpr-folded (D_MODEL/
// N_HEADS, its one call site) -- see rope_kernel above; pos stays runtime (the decode position).
__global__ void rope_one_kernel(float* __restrict__ qkv, int pos, float theta) {
    constexpr int C = D_MODEL, d = D_HEAD, half = d / 2;
    const int pg = blockIdx.x * blockDim.x + threadIdx.x;
    if (pg >= C / 2) return;
    const int h = pg / half, mi = pg % half, a0 = h * d + 2 * mi;
    const float ang = static_cast<float>(pos) * powf(theta, -2.0f * mi / d);
    float sn, cs; __sincosf(ang, &sn, &cs);
    const float q0 = qkv[a0],     q1 = qkv[a0 + 1];     qkv[a0]     = q0 * cs - q1 * sn; qkv[a0 + 1]     = q0 * sn + q1 * cs;
    const float k0 = qkv[C + a0], k1 = qkv[C + a0 + 1]; qkv[C + a0] = k0 * cs - k1 * sn; qkv[C + a0 + 1] = k0 * sn + k1 * cs;
}
// Decode attention: the single (roped) query q[C] attends the cached K/V (rows over [SEQ_LEN,C]) for
// j=0..pos -> att[C]. One block per head; blockDim=128. Shared: the query head + scores[pos+1]. The
// math is the last-query row of op_attn, so att matches the full forward's final row to fp tolerance.
// C is D_MODEL at its one call site -- constexpr-folded. H was a dead parameter (never read in the
// body; the caller uses it only to size the launch's grid.x, one block per head, not passed in here).
template <int HD>
__global__ void attn_decode_kernel(const float* __restrict__ q, const float* __restrict__ kcache,
                                   const float* __restrict__ vcache, float* __restrict__ att, int pos) {
    constexpr int C = D_MODEL;
    extern __shared__ float sh[];                 // qh[HD] then sc[pos+1]
    float* qh = sh;
    float* sc = sh + HD;
    __shared__ float red[128];
    const int hh = blockIdx.x, off = hh * HD, tid = threadIdx.x, nt = blockDim.x;
    const float scale = rsqrtf(static_cast<float>(HD));
    for (int a = tid; a < HD; a += nt) qh[a] = q[off + a];
    __syncthreads();
    for (int j = tid; j <= pos; j += nt) {         // scaled scores q.k_j
        const float* kj = kcache + static_cast<size_t>(j) * C + off;
        float s = 0.f;
        #pragma unroll
        for (int a = 0; a < HD; ++a) s += qh[a] * kj[a];
        sc[j] = s * scale;
    }
    __syncthreads();
    float lm = -1e30f;                             // block-reduce max
    for (int j = tid; j <= pos; j += nt) lm = fmaxf(lm, sc[j]);
    red[tid] = lm; __syncthreads();
    for (int s = nt / 2; s > 0; s >>= 1) { if (tid < s) red[tid] = fmaxf(red[tid], red[tid + s]); __syncthreads(); }
    const float mx = red[0]; __syncthreads();
    float lz = 0.f;                                // block-reduce sum(exp)
    for (int j = tid; j <= pos; j += nt) lz += __expf(sc[j] - mx);
    red[tid] = lz; __syncthreads();
    for (int s = nt / 2; s > 0; s >>= 1) { if (tid < s) red[tid] += red[tid + s]; __syncthreads(); }
    const float invZ = 1.f / red[0]; __syncthreads();
    for (int j = tid; j <= pos; j += nt) sc[j] = __expf(sc[j] - mx) * invZ;   // p_j
    __syncthreads();
    for (int a = tid; a < HD; a += nt) {           // att = sum_j p_j v_j
        float acc = 0.f;
        for (int j = 0; j <= pos; ++j) acc += sc[j] * vcache[static_cast<size_t>(j) * C + off + a];
        att[off + a] = acc;
    }
}

// One decode step on the device: token `id` at window position `pos`, reusing the M=1 dense launches
// and the K/V cache. Writes logits into g_fwd.logits[0..VOCAB). Requires uploaded params + wqkv +
// fwd_alloc(1) + kv_alloc(); the caller resets pos to 0 at the start of a sequence.
void forward_one_device(int id, int pos) {
    const auto&  L  = sub0::PARAM_LAYOUT;
    const int    C  = D_MODEL, F = D_FF, V = VOCAB, H = N_HEADS;
    float* const base = g_dev_params;
    const float* tok_emb = base + L[0].off;
    const float* pos_emb = base + L[1].off;
    const int    fi      = 2 + 10 * N_LAYERS;
    const float* ln_f    = base + L[fi + 0].off;
    const float* lm_head = base + L[fi + 1].off;
    const float* lm_bias = base + L[fi + 2].off;
    float* h = g_fwd.h; float* a = g_fwd.a; float* qkv = g_fwd.qkv; float* att = g_fwd.att;
    float* proj = g_fwd.proj; float* fbuf = g_fwd.fbuf; float* ff1 = g_fwd.ff1; float* gact = g_fwd.gact; float* ff2 = g_fwd.ff2;

    { const int blk = 256; const float* pe = (POS_ENCODING == PosEncoding::Absolute) ? pos_emb : nullptr;
      embed_one_kernel<<<(C + blk - 1) / blk, blk, 0, g_stream>>>(tok_emb, pe, id, pos, h); }
    for (int l = 0; l < N_LAYERS; ++l) {
        const int    b0  = 2 + 10 * l;
        const float* ln1 = base + L[b0 + 0].off, *ln2 = base + L[b0 + 1].off;
        const float* Wo  = base + L[b0 + 5].off, *W1 = base + L[b0 + 6].off, *b1 = base + L[b0 + 7].off;
        const float* W2  = base + L[b0 + 8].off, *b2 = base + L[b0 + 9].off;
        launch_rmsnorm(h, ln1, a, 1);
        launch_linear(a, g_fwd.wqkv[l], nullptr, qkv, 1, C, 3 * C);                 // fused q|k|v (one row)
        if constexpr (POS_ENCODING == PosEncoding::Rope) {
            const int blk = 128; rope_one_kernel<<<(C / 2 + blk - 1) / blk, blk, 0, g_stream>>>(qkv, pos, ROPE_THETA);
        }
        float* kc = g_kv_k + (static_cast<size_t>(l) * SEQ_LEN + pos) * C;          // append this token's K/V
        float* vc = g_kv_v + (static_cast<size_t>(l) * SEQ_LEN + pos) * C;
        cudaMemcpyAsync(kc, qkv + C,     C * sizeof(float), cudaMemcpyDeviceToDevice, g_stream);
        cudaMemcpyAsync(vc, qkv + 2 * C, C * sizeof(float), cudaMemcpyDeviceToDevice, g_stream);
        { constexpr int HD = D_HEAD; const size_t shb = (HD + SEQ_LEN) * sizeof(float);
          attn_decode_kernel<HD><<<H, 128, shb, g_stream>>>(qkv, g_kv_k + static_cast<size_t>(l) * SEQ_LEN * C,
                                                            g_kv_v + static_cast<size_t>(l) * SEQ_LEN * C, att, pos); }
        launch_linear(att, Wo, nullptr, proj, 1, C, C);
        launch_add(h, proj, h, C);
        launch_rmsnorm(h, ln2, fbuf, 1);
        launch_linear(fbuf, W1, b1, ff1, 1, C, F);
        launch_gelu(ff1, gact, F);
        launch_linear(gact, W2, b2, ff2, 1, F, C);
        launch_add(h, ff2, h, C);
    }
    launch_rmsnorm(h, ln_f, a, 1);
    launch_linear(a, lm_head, lm_bias, g_fwd.logits, 1, C, V, /*force_tc=*/true);
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
    [[maybe_unused]] const float* pos_emb = base + L[1].off;
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

    // h = tok_emb[ids] (+ pos_emb under Absolute); position = row % T
    {
        const dim3 block(16, 16);
        const dim3 grid((C + block.x - 1) / block.x, (M + block.y - 1) / block.y);
        if constexpr (POS_ENCODING == PosEncoding::Absolute)
            embed_add_kernel<<<grid, block, 0, g_stream>>>(tok_emb, pos_emb, g_fwd.dids, h, M, T);
        else
            embed_kernel<<<grid, block, 0, g_stream>>>(tok_emb, g_fwd.dids, h, M);
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

        launch_rmsnorm(h, ln1, a, M);                         // a = rmsnorm(h, ln1)
        launch_linear(a, g_fwd.wqkv[l], nullptr, qkv, M, C, 3 * C);          // fused qkv = a . [Wq|Wk|Wv]
        if constexpr (POS_ENCODING == PosEncoding::Rope) launch_rope(qkv, batch, T);
        launch_attn(qkv, qkv + C, qkv + 2 * C, att, batch, T, C, H, 3 * C);  // q/k/v sub-blocks (stride 3C)
        launch_linear(att, Wo, nullptr, proj, M, C, C);      // proj = att . Wo
        launch_add(h, proj, h, MC);                          // h = h + proj
        launch_rmsnorm(h, ln2, fbuf, M);                     // f = rmsnorm(h, ln2)
        launch_linear(fbuf, W1, b1, ff1, M, C, F);        // ff1 = f . W1 + b1
        launch_gelu(ff1, gact, MF);                       // gelu
        launch_linear(gact, W2, b2, ff2, M, F, C);        // ff2 = gelu . W2 + b2
        launch_add(h, ff2, h, MC);                        // h = h + ff2
    }
    launch_rmsnorm(h, ln_f, a, M);                        // a = rmsnorm(h, ln_f)
    launch_linear(a, lm_head, lm_bias, g_fwd.logits, M, C, V, /*force_tc=*/true);  // logits = a . lm_head + lm_bias
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

// Materialize the per-layer fused QKV weight from the uploaded param blob. Called once whenever the
// weights change (upload/load/every AdamW step). TODO(qkv-train): this rebuild runs inside every
// device training step (cheap: N_LAYERS*C*C threads) -- or store QKV fused natively and slice it
// back for the layout-table serialization.
//
// BF16 builds write g_wqkv16 DIRECTLY from Wq/Wk/Wv (build_qkv_act_kernel) -- no F32 g_fwd.wqkv
// staging buffer at all, since training never reads it (only the bf16 mirror feeds the training
// GEMMs). That buffer is dead weight during training (~108 MiB at production dims): marks
// g_wqkv_f32_dirty so the F32 CUDA inference paths (ensure_wqkv_f32, sub0_cuda_forward /
// sub0_cuda_forward_one) rebuild their own lazily-allocated copy on next use instead of reading a
// stale one -- this is what keeps an interleaved train-then-infer call (e.g. the forward_one/full-
// forward parity check) correct.
//
// F32 builds are UNCHANGED from before this split: g_fwd.wqkv is eagerly built here (same
// build_qkv_kernel as always) and every mirror aliases the master weights directly (no copy).
void build_qkv_weights() {
    const auto& L = sub0::PARAM_LAYOUT;
    const int   C = D_MODEL;
    const int   n = C * C, block = 256, grid = (n + block - 1) / block;
    if constexpr (ACT_DTYPE == Dtype::BF16) {
        g_wqkv_f32_dirty = true;
        const int F = D_FF;
        const int nce = C * F, gce = (nce + 255) / 256;
        for (int l = 0; l < N_LAYERS; ++l) {
            const int    b0 = 2 + 10 * l;
            const float* Wq = g_dev_params + L[b0 + 2].off;
            const float* Wk = g_dev_params + L[b0 + 3].off;
            const float* Wv = g_dev_params + L[b0 + 4].off;
            if (!g_w1_16[l])   cudaMalloc(&g_w1_16[l],   static_cast<size_t>(C) * F * sizeof(act_t));
            if (!g_w2_16[l])   cudaMalloc(&g_w2_16[l],   static_cast<size_t>(F) * C * sizeof(act_t));
            if (!g_wo16[l])    cudaMalloc(&g_wo16[l],    static_cast<size_t>(C) * C * sizeof(act_t));
            if (!g_wqkv16[l])  cudaMalloc(&g_wqkv16[l],  static_cast<size_t>(C) * 3 * C * sizeof(act_t));
            f32_to_act_kernel<<<gce, 256, 0, g_stream>>>(g_dev_params + L[b0 + 6].off, g_w1_16[l], nce);
            f32_to_act_kernel<<<gce, 256, 0, g_stream>>>(g_dev_params + L[b0 + 8].off, g_w2_16[l], nce);
            { const int ncc = C * C, gcc = (ncc + 255) / 256;
              f32_to_act_kernel<<<gcc, 256, 0, g_stream>>>(g_dev_params + L[b0 + 5].off, g_wo16[l], ncc); }
            build_qkv_act_kernel<act_t><<<grid, block, 0, g_stream>>>(Wq, Wk, Wv, g_wqkv16[l]);
        }
    } else {                                             // F32: mirrors alias the master weights (no copy)
        for (int l = 0; l < N_LAYERS; ++l) {
            const int    b0 = 2 + 10 * l;
            const float* Wq = g_dev_params + L[b0 + 2].off;
            const float* Wk = g_dev_params + L[b0 + 3].off;
            const float* Wv = g_dev_params + L[b0 + 4].off;
            build_qkv_kernel<<<grid, block, 0, g_stream>>>(Wq, Wk, Wv, g_fwd.wqkv[l]);
            g_w1_16[l] = reinterpret_cast<act_t*>(g_dev_params + L[b0 + 6].off);
            g_w2_16[l] = reinterpret_cast<act_t*>(g_dev_params + L[b0 + 8].off);
            g_wo16[l]  = reinterpret_cast<act_t*>(g_dev_params + L[b0 + 5].off);
            g_wqkv16[l] = reinterpret_cast<act_t*>(g_fwd.wqkv[l]);
        }
    }
}

// Lazily (re)builds the F32 fused QKV weight for the CUDA inference-only paths (sub0_cuda_forward,
// sub0_cuda_forward_one) on a BF16 build, which otherwise keeps only the bf16 mirror resident (see
// build_qkv_weights() above). No-op on an F32 build: g_fwd.wqkv there is already kept current by
// build_qkv_weights() directly, exactly as before this split.
int ensure_wqkv_f32() {
    if constexpr (ACT_DTYPE != Dtype::BF16) return 0;
    if (!g_fwd.wqkv[0]) {
        ensure_stream();
        for (int l = 0; l < N_LAYERS; ++l)
            SUB0_CUDA_CHECK(cudaMalloc(&g_fwd.wqkv[l], static_cast<size_t>(D_MODEL) * 3 * D_MODEL * sizeof(float)));
        g_wqkv_f32_dirty = true;         // freshly (re)allocated -- garbage until populated below
    }
    if (g_wqkv_f32_dirty) {
        const auto& L = sub0::PARAM_LAYOUT;
        const int C = D_MODEL, n = C * C, block = 256, grid = (n + block - 1) / block;
        for (int l = 0; l < N_LAYERS; ++l) {
            const int b0 = 2 + 10 * l;
            build_qkv_kernel<<<grid, block, 0, g_stream>>>(g_dev_params + L[b0 + 2].off,
                g_dev_params + L[b0 + 3].off, g_dev_params + L[b0 + 4].off, g_fwd.wqkv[l]);
        }
        g_wqkv_f32_dirty = false;
    }
    return 0;
}

// ============================================================================
//  Training forward + backward + AdamW (Phase 2d) -- mirrors backend_cpu.cpp
// ============================================================================

// Forward pass that SAVES every activation the backward needs into g_tr (no recompute). Same op
// sequence as forward_device but writes to the per-layer training buffers and uses the train
// variants of rmsnorm/attn (which also save rinv / the softmax weights P). Assumes g_fwd.dids is
// populated, params uploaded and g_wqkv16[]/g_w1_16[]/g_w2_16[]/g_wo16[] built (build_qkv_weights) --
// training reads ONLY the bf16 mirrors, never the F32 g_fwd.wqkv (that buffer is inference-only, see
// ensure_wqkv_f32). Eager (not graph-captured): the backward reads these buffers, so the whole step
// runs as plain stream-ordered launches.
// NOTE(perf): deliberately NOT graph-captured. A step graph is an EFFICIENCY change -- it removes
// per-launch CPU/driver management overhead (~200 launches -> 1), not a compute change. The inference
// forward IS captured (capture_graph) because gen is launch-bound; training is COMPUTE-bound (measured
// graph 1.00x at M=4096), so removing that CPU overhead buys nothing, and the AdamW step has a host-in-
// the-loop break that prevents single-graph capture anyway (D2H grad-norm -> host clip scale + pow bias-
// correction). Revisit only if a device-side global-norm clip removes that sync AND a launch-bound regime appears.
// TODO(mem): saved-activation scratch dominates at large batch; a/qkv/att are already checkpointed
// (recomputed in backward) and acts + residual stream are bf16, so batch 512 now fits 8GB. Further
// cuts would need checkpointing the per-layer residual (h_in/h_mid) too, recomputed from the input.
void forward_train(int batch, int T) {
    const auto&  L  = sub0::PARAM_LAYOUT;
    const int    C  = D_MODEL, F = D_FF, V = VOCAB, H = N_HEADS;
    const int    M  = batch * T;
    const int    MC = M * C, MF = M * F;
    float* const base = g_dev_params;

    const float* tok_emb = base + L[0].off;
    [[maybe_unused]] const float* pos_emb = base + L[1].off;
    const int    fi      = 2 + 10 * N_LAYERS;
    const float* ln_f    = base + L[fi + 0].off;
    const float* lm_head = base + L[fi + 1].off;
    const float* lm_bias = base + L[fi + 2].off;

    {   // h_in[0] = tok_emb[ids] (+ pos_emb under Absolute)
        const dim3 block(16, 16);
        const dim3 grid((C + block.x - 1) / block.x, (M + block.y - 1) / block.y);
        if constexpr (POS_ENCODING == PosEncoding::Absolute)
            embed_add_act_kernel<act_t><<<grid, block, 0, g_stream>>>(tok_emb, pos_emb, g_fwd.dids, g_tr.h_in[0], M, T);
        else
            embed_act_kernel<act_t><<<grid, block, 0, g_stream>>>(tok_emb, g_fwd.dids, g_tr.h_in[0], M);
    }
    for (int l = 0; l < N_LAYERS; ++l) {
        const int    b0  = 2 + 10 * l;
        const float* ln1 = base + L[b0 + 0].off;
        const float* ln2 = base + L[b0 + 1].off;
        const float* W1  = base + L[b0 + 6].off;
        const float* b1  = base + L[b0 + 7].off;
        const float* W2  = base + L[b0 + 8].off;
        const float* b2  = base + L[b0 + 9].off;
        act_t* const hin  = g_tr.h_in[l];
        act_t* const hmid = g_tr.h_mid[l];
        act_t* const next = (l + 1 < N_LAYERS) ? g_tr.h_in[l + 1] : g_tr.h_final;

        launch_rmsnorm_train_t<act_t, act_t>(hin, ln1, g_tr.a, g_tr.rinv1[l], M);          // a = rmsnorm(hin,ln1) bf16
        launch_linear_t<act_t, act_t>(g_tr.a, g_wqkv16[l], g_tr.qkv, M, C, 3 * C);          // fused qkv bf16
        if constexpr (POS_ENCODING == PosEncoding::Rope) launch_rope_t<act_t>(g_tr.qkv, batch, T);
        launch_attn_train_t<act_t>(g_tr.qkv, g_tr.qkv + C, g_tr.qkv + 2 * C,
                          g_tr.att, batch, T, C, H, 3 * C);                          // attention (P-free)
        launch_linear_t<act_t, act_t>(g_tr.att, g_wo16[l], hmid, M, C, C);           // hmid = proj (bf16)
        launch_add_t<act_t>(hin, hmid, hmid, MC);                                   // hmid = hin + proj
        launch_rmsnorm_train_t<act_t, act_t>(hmid, ln2, g_tr.fbuf, g_tr.rinv2[l], M);      // fbuf bf16
        launch_linear_t<act_t, act_t>(g_tr.fbuf, g_w1_16[l], g_tr.ff1, M, C, F);
        launch_bias_act(g_tr.ff1, b1, M, F);                                       // ff1 = f.W1 + b1
        launch_gelu_t(g_tr.ff1, g_tr.gact, MF);                                     // gelu
        launch_linear_t<act_t, act_t>(g_tr.gact, g_w2_16[l], next, M, F, C);        // next = ff2 bf16
        launch_bias_act(next, b2, M, C);
        launch_add_t<act_t>(hmid, next, next, MC);                                  // next = hmid + ff2
    }
    launch_rmsnorm_train_t<act_t, float>(g_tr.h_final, ln_f, g_tr.a_final, g_tr.rinv_f, M);     // a_final f32
    launch_linear(g_tr.a_final, lm_head, lm_bias, g_tr.logits, M, C, V, /*force_tc=*/true);  // logits
}

// Reverse pass: consumes the saved activations, writes the reduced gradient into g_dev_grad and
// accumulates the mean cross-entropy into g_tr.loss. dh threads the residual stream; the rmsnorm
// backward kernels accumulate into it (residual skip + through-norm paths), exactly like the CPU
// tape walk. Weight/bias grads are written straight to their PARAM_LAYOUT offsets (each weight is
// used once per forward). Loss scaling invM = 1/M makes the result equal the CPU train_batch grad.
void backward_device(int batch, int T, const int* d_lengths) {
    const auto&  L  = sub0::PARAM_LAYOUT;
    const int    C  = D_MODEL, F = D_FF, V = VOCAB, H = N_HEADS;
    const int    M  = batch * T;
    const int    MC = M * C, MF = M * F;
    float* const pb = g_dev_params;
    float* const gb = g_dev_grad;
    const int    fi = 2 + 10 * N_LAYERS;

    SUB0_CUDA_CHECK_VOID(cudaMemsetAsync(gb, 0, sub0::PARAM_FLOATS * sizeof(float), g_stream));
    SUB0_CUDA_CHECK_VOID(cudaMemsetAsync(g_tr.loss, 0, sizeof(double), g_stream));

    // cross-entropy: per-window-mean dlogits with padding masked out (d_lengths; nullptr = all full T)
    {
        constexpr int kCeBlock = 256;   // one block per row; see ce_backward_kernel above
        ce_backward_kernel<kCeBlock><<<M, kCeBlock, 0, g_stream>>>(
            g_tr.logits, g_tr.dtargets, g_tr.dlogits, g_tr.loss, M, T, batch, d_lengths);
    }
    // lm_head: dW/dbias -> grad blob, da = grad into a_final
    launch_linear_bwd(g_tr.a_final, pb + L[fi + 1].off, g_tr.dlogits, g_tr.da,
                      gb + L[fi + 1].off, gb + L[fi + 2].off, M, C, V, /*force_tc=*/true);
    // rmsnorm_f: dh starts here (grad into h_final)
    SUB0_CUDA_CHECK_VOID(cudaMemsetAsync(g_tr.dh, 0, static_cast<size_t>(MC) * sizeof(float), g_stream));
    launch_rmsnorm_bwd_t<act_t>(g_tr.h_final, pb + L[fi + 0].off, g_tr.rinv_f, g_tr.da, g_tr.dh,
                       gb + L[fi + 0].off, M);

    for (int l = N_LAYERS - 1; l >= 0; --l) {
        const int b0 = 2 + 10 * l;
        const float* W1 = pb + L[b0 + 6].off, *b1 = pb + L[b0 + 7].off;
        // checkpoint: fbuf/ff1/gact not saved -- recompute fbuf from h_mid, then ff1/gact, before use
        launch_rmsnorm_train_t<act_t, act_t>(g_tr.h_mid[l], pb + L[b0 + 1].off, g_tr.fbuf, g_tr.rinv2[l], M);
        launch_linear_t<act_t, act_t>(g_tr.fbuf, g_w1_16[l], g_tr.ff1, M, C, F);
        launch_bias_act(g_tr.ff1, b1, M, F);
        launch_gelu_t(g_tr.ff1, g_tr.gact, MF);
        // ff residual: dh = grad into layer output = d(ff2). W2 backward (input gact); dh->bf16
        { const int n2 = M * C, bk = 256; f32_to_act_kernel<<<(n2 + bk - 1) / bk, bk, 0, g_stream>>>(g_tr.dh, g_tr.dh16, n2); }
        launch_linear_bwd_t<act_t, act_t>(g_tr.gact, g_w2_16[l], g_tr.dh16, g_tr.dgact, gb + L[b0 + 8].off, M, F, C);
        { const int bk = 128; bias_grad_kernel<<<(C + bk - 1) / bk, bk, 0, g_stream>>>(g_tr.dh, gb + L[b0 + 9].off, M, C); }
        launch_gelu_bwd_t(g_tr.ff1, g_tr.dgact, g_tr.dff1, MF);                     // dff1
        launch_linear_bwd_t<act_t, float>(g_tr.fbuf, g_w1_16[l], g_tr.dff1, g_tr.dfbuf, gb + L[b0 + 6].off, M, C, F);
        { const int bk = 128; bias_grad_act_kernel<act_t><<<(F + bk - 1) / bk, bk, 0, g_stream>>>(g_tr.dff1, gb + L[b0 + 7].off, M); }
        launch_rmsnorm_bwd_t<act_t>(g_tr.h_mid[l], pb + L[b0 + 1].off, g_tr.rinv2[l], g_tr.dfbuf,
                           g_tr.dh, gb + L[b0 + 1].off, M);                          // dh += -> d(h_mid)
        // proj residual: dh = grad into h_mid = d(proj). Wo backward (input att, bf16)
        // checkpoint: a/qkv/att not saved -- recompute a from h_in, then qkv (+rope), then attention
        launch_rmsnorm_train_t<act_t, act_t>(g_tr.h_in[l], pb + L[b0 + 0].off, g_tr.a, g_tr.rinv1[l], M);
        launch_linear_t<act_t, act_t>(g_tr.a, g_wqkv16[l], g_tr.qkv, M, C, 3 * C);
        if constexpr (POS_ENCODING == PosEncoding::Rope) launch_rope_t<act_t>(g_tr.qkv, batch, T);
        launch_attn_train_t<act_t>(g_tr.qkv, g_tr.qkv + C, g_tr.qkv + 2 * C, g_tr.att, batch, T, C, H, 3 * C);
        { const int n2 = M * C, bk = 256; f32_to_act_kernel<<<(n2 + bk - 1) / bk, bk, 0, g_stream>>>(g_tr.dh, g_tr.dh16, n2); }
        launch_linear_bwd_t<act_t, act_t>(g_tr.att, g_wo16[l], g_tr.dh16, g_tr.datt, gb + L[b0 + 5].off, M, C, C);
        // flash backward writes each dq/dk/dv exactly once (no atomics), so no pre-zero is needed.
        launch_attn_bwd_t<act_t>(g_tr.qkv, g_tr.datt, g_tr.dqkv, batch, T, C, H, 3 * C);
        // RoPE: convert dQ/dK (w.r.t. the rotated q/k attention used) back to grad w.r.t. the
        // projected q/k before the qkv-GEMM backward (inverse rotation). dV is untouched.
        if constexpr (POS_ENCODING == PosEncoding::Rope) launch_rope_bwd_t<act_t>(g_tr.dqkv, batch, T);
        // qkv backward (input a): da = grad into a (f32), dWqkv -> split into dWq/dWk/dWv
        launch_linear_bwd_t<act_t, float>(g_tr.a, g_wqkv16[l], g_tr.dqkv, g_tr.da, g_tr.dwqkv, M, C, 3 * C);
        {
            const int n = C * C, block = 256;
            split_dqkv_kernel<<<(n + block - 1) / block, block, 0, g_stream>>>(
                g_tr.dwqkv, gb + L[b0 + 2].off, gb + L[b0 + 3].off, gb + L[b0 + 4].off);
        }
        launch_rmsnorm_bwd_t<act_t>(g_tr.h_in[l], pb + L[b0 + 0].off, g_tr.rinv1[l], g_tr.da,
                           g_tr.dh, gb + L[b0 + 0].off, M);                          // dh += -> d(h_in)
    }
    // embed backward: scatter dh into tok_emb (+ pos_emb under Absolute) grads
    {
        const dim3 block(16, 16);
        const dim3 grid((C + block.x - 1) / block.x, (M + block.y - 1) / block.y);
        if constexpr (POS_ENCODING == PosEncoding::Absolute)
            embed_backward_kernel<<<grid, block, 0, g_stream>>>(
                g_tr.dh, g_fwd.dids, gb + L[0].off, gb + L[1].off, M, T);
        else
            embed_backward_token_kernel<<<grid, block, 0, g_stream>>>(
                g_tr.dh, g_fwd.dids, gb + L[0].off, M);
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
    // uint8 mask (a 0/1 flag needs no more than 1 byte -- was float, 4x the storage for the same
    // information; persistent for the whole run, so every byte here is permanent headroom lost to
    // a larger training batch, the bigger lever).
    SUB0_CUDA_CHECK(cudaMalloc(&g_dev_decay, n * sizeof(unsigned char)));
    SUB0_CUDA_CHECK(cudaMalloc(&g_dev_normsq, sizeof(double)));
    SUB0_CUDA_CHECK(cudaMalloc(&g_dev_gs, sizeof(float)));
    SUB0_CUDA_CHECK(cudaMemset(g_dev_m,   0, n * sizeof(float)));
    SUB0_CUDA_CHECK(cudaMemset(g_dev_vel, 0, n * sizeof(float)));
    // weight-decay mask: 1 for matrices (PARAM_LAYOUT decay flag), 0 for biases/norms.
    std::vector<unsigned char> mask(n, 0);
    const auto& L = sub0::PARAM_LAYOUT;
    for (const auto& pv : L) {
        const unsigned char d = pv.decay ? 1 : 0;
        const size_t cnt = static_cast<size_t>(pv.rows) * pv.cols;
        for (size_t i = 0; i < cnt; ++i) mask[pv.off + i] = d;
    }
    SUB0_CUDA_CHECK(cudaMemcpy(g_dev_decay, mask.data(), n * sizeof(unsigned char), cudaMemcpyHostToDevice));
    return 0;
}

void opt_free() {
    cudaFree(g_dev_grad);  cudaFree(g_dev_m);    cudaFree(g_dev_vel);
    cudaFree(g_dev_decay); cudaFree(g_dev_normsq); cudaFree(g_dev_gs);
    g_dev_grad = g_dev_m = g_dev_vel = nullptr;
    g_dev_decay = nullptr;
    g_dev_normsq = nullptr;
    g_dev_gs = nullptr;
}

// One AdamW step over the whole param blob, op-for-op identical to AdamW::step on the CPU: global
// grad L2 clip (double accumulation), then the per-parameter moment update with the decay mask.
// `t` is the post-increment step counter (>=1) the host optimizer drives. Rebuilds the fused QKV
// weights (Wq/Wk/Wv just changed) and invalidates the inference graph afterwards.
//
// NO host sync here (unlike the pre-2026-07 version): the clip scale gs is computed ON DEVICE by
// grad_clip_scale_kernel from g_dev_normsq and read by adam_step_kernel via pointer, so nothing here
// needs normsq back on the host at all. bc1/bc2 are already host-computable from `t` alone (no
// device dependency). This is the second of two per-step syncs removed 2026-07 -- run_fwd_bwd's loss
// readback (below) was the other -- leaving a single sync at the end of sub0_cuda_train_step.
void device_adam_step(float lr, long t, float b1, float b2, float eps, float wd, float clip) {
    const int n = static_cast<int>(sub0::PARAM_FLOATS);
    const int block = 256, grid = (n + block - 1) / block;
    SUB0_CUDA_CHECK_VOID(cudaMemsetAsync(g_dev_normsq, 0, sizeof(double), g_stream));
    grad_normsq_kernel<<<grid, block, 0, g_stream>>>(g_dev_grad, g_dev_normsq, n);
    grad_clip_scale_kernel<<<1, 1, 0, g_stream>>>(g_dev_normsq, clip, g_dev_gs);
    const float bc1  = 1.0f - std::pow(b1, static_cast<float>(t));
    const float bc2  = 1.0f - std::pow(b2, static_cast<float>(t));
    adam_step_kernel<<<grid, block, 0, g_stream>>>(g_dev_params, g_dev_grad, g_dev_m, g_dev_vel,
                                                   g_dev_decay, n, g_dev_gs, lr, b1, b2, eps, wd, bc1, bc2);
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
    // Block (sleep) the host thread on device syncs instead of spinning a core at 100%: the resident
    // training step is GPU-bound, so busy-waiting the per-step sync just wastes a CPU core (and its
    // power/heat) for no throughput. Must precede the first runtime call that creates the context; a
    // non-zero return (context already active) is harmless, so it is intentionally not checked.
    cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
    SUB0_CUDA_CHECK(cudaMalloc(&g_dev_params, sub0::PARAM_FLOATS * sizeof(float)));
    return 0;
}

SUB0_CUDA_API void sub0_cuda_shutdown() {
    invalidate_graph();
    if (g_dev_params) cudaFree(g_dev_params);
    g_dev_params = nullptr;
    fwd_free();                                  // release the resident forward scratch too
    kv_free();                                   // and the decode KV-cache
    train_free();                                // and the training scratch (Phase 2d)
    opt_free();                                  // and the optimizer state
    if (g_cublas) { cublasDestroy(g_cublas); g_cublas = nullptr; }
    if (g_stream) { cudaStreamDestroy(g_stream); g_stream = nullptr; }
}

SUB0_CUDA_API int sub0_cuda_upload_params(const float* host) {
    if (!g_dev_params) { const int r = sub0_cuda_init(); if (r) return r; }
    if (wqkv_alloc()) return 1;                  // ensure the fused-QKV weight buffers exist
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

// AdamW moment sync (for crash-safe checkpoint / resume): the optimizer state lives on the device
// during a GPU run, so the train stage round-trips it through the host adam_m/adam_v buffers
// around save/load -- mirroring the param sync above. Both are PARAM_FLOATS long.
SUB0_CUDA_API int sub0_cuda_download_opt(float* host_m, float* host_v) {
    if (!g_dev_m || !g_dev_vel) return 1;
    SUB0_CUDA_CHECK(cudaMemcpy(host_m, g_dev_m,   sub0::PARAM_FLOATS * sizeof(float), cudaMemcpyDeviceToHost));
    SUB0_CUDA_CHECK(cudaMemcpy(host_v, g_dev_vel, sub0::PARAM_FLOATS * sizeof(float), cudaMemcpyDeviceToHost));
    return 0;
}

SUB0_CUDA_API int sub0_cuda_upload_opt(const float* host_m, const float* host_v) {
    if (opt_alloc()) return 1;                   // ensure the moment buffers exist
    SUB0_CUDA_CHECK(cudaMemcpy(g_dev_m,   host_m, sub0::PARAM_FLOATS * sizeof(float), cudaMemcpyHostToDevice));
    SUB0_CUDA_CHECK(cudaMemcpy(g_dev_vel, host_v, sub0::PARAM_FLOATS * sizeof(float), cudaMemcpyHostToDevice));
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
    if (fwd_alloc(batch)) return 1;
    if (ensure_wqkv_f32()) return 1;             // BF16: (re)build the F32 mirror this path reads
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

// Reset the decode KV-cache for a new sequence: ensure the M=1 forward scratch + the K/V cache exist.
// (Positions are supplied per call, so "reset" just guarantees the buffers.) Requires uploaded params.
SUB0_CUDA_API int sub0_cuda_kv_reset() {
    if (!g_dev_params) return 1;
    if (fwd_alloc(1) || kv_alloc()) return 1;
    ensure_cublas();
    return 0;
}
// One decode step: token `id` at window position `pos` (0-based, < SEQ_LEN). Runs forward_one_device
// and copies the logits [VOCAB] to the host. Requires sub0_cuda_upload_params + sub0_cuda_kv_reset.
SUB0_CUDA_API int sub0_cuda_forward_one(int id, int pos, float* out_logits) {
    if (!g_dev_params || !g_kv_k) return 1;
    if (id < 0 || id >= VOCAB || pos < 0 || pos >= SEQ_LEN) return 1;
    if (ensure_wqkv_f32()) return 1;               // BF16: (re)build the F32 mirror this path reads
    set_handle_tf32(false);                       // tight FP32 inference math (parity with the CPU path)
    forward_one_device(id, pos);
    SUB0_CUDA_CHECK(cudaMemcpyAsync(out_logits, g_fwd.logits, VOCAB * sizeof(float), cudaMemcpyDeviceToHost, g_stream));
    SUB0_CUDA_CHECK(cudaStreamSynchronize(g_stream));
    SUB0_CUDA_CHECK(cudaGetLastError());
    return 0;
}

// On-device correctness + speed guard for the decode path: with random params, the incremental
// forward_one (KV-cache) must reproduce the FULL forward's per-position logits -- forward_one(id_t, t)
// attends the same causal context as sub0_cuda_forward()'s row t -- and then time the decode tok/s.
// Returns nonzero on a parity failure. Self-contained (uploads its own random weights).
SUB0_CUDA_API int sub0_cuda_forward_one_check(int T, int iters, double* out_maxreldiff, double* out_toks) {
    if (T < 2 || T > SEQ_LEN) return 1;
    if (iters < 1) iters = 200;
    if (sub0_cuda_init()) return 1;
    std::vector<float> hp(sub0::PARAM_FLOATS);                          // small random weights (finite logits)
    unsigned s = 0x243f6a88u;
    for (auto& x : hp) { s = s * 1664525u + 1013904223u; x = (static_cast<float>(s >> 9) / 8388608.0f - 1.0f) * 0.02f; }
    if (sub0_cuda_upload_params(hp.data())) return 1;                   // also builds the fused wqkv
    std::vector<int> ids(static_cast<size_t>(T));
    for (int& x : ids) { s = s * 1664525u + 1013904223u; x = static_cast<int>(s % static_cast<unsigned>(VOCAB)); }

    std::vector<float> full(static_cast<size_t>(T) * VOCAB);            // full-forward reference [T, V]
    if (sub0_cuda_forward(ids.data(), 1, T, full.data())) return 1;
    if (sub0_cuda_kv_reset()) return 1;
    std::vector<float> one(static_cast<size_t>(VOCAB));
    double maxrel = 0.0;
    int top1_agree = 0;                                                // do both pick the same argmax token?
    auto argmax = [V = VOCAB](const float* p) { int b = 0; for (int j = 1; j < V; ++j) if (p[j] > p[b]) b = j; return b; };
    for (int pos = 0; pos < T; ++pos) {
        if (sub0_cuda_forward_one(ids[pos], pos, one.data())) return 1;
        const float* ref = full.data() + static_cast<size_t>(pos) * VOCAB;
        double maxabs = 0.0, maxmag = 1e-30;
        for (int j = 0; j < VOCAB; ++j) {
            maxabs = std::fmax(maxabs, std::fabs(static_cast<double>(one[j]) - ref[j]));
            maxmag = std::fmax(maxmag, std::fabs(static_cast<double>(ref[j])));
        }
        maxrel = std::fmax(maxrel, maxabs / maxmag);
        if (argmax(one.data()) == argmax(ref)) ++top1_agree;
    }
    const double top1_pct = 100.0 * top1_agree / T;

    const int mid = T / 2;                                              // time decode steps at a mid position
    for (int w = 0; w < 5; ++w) forward_one_device(ids[mid], mid);
    cudaStreamSynchronize(g_stream);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0, g_stream);
    for (int it = 0; it < iters; ++it) forward_one_device(ids[mid], mid);
    cudaEventRecord(e1, g_stream); cudaEventSynchronize(e1);
    float ms = 0.f; cudaEventElapsedTime(&ms, e0, e1); cudaEventDestroy(e0); cudaEventDestroy(e1);
    const double per_ms = static_cast<double>(ms) / iters, toks = per_ms > 0 ? 1000.0 / per_ms : 0.0;

    // Under RANDOM weights the two-pass decode softmax vs the flash online-softmax diverge and amplify
    // over the layers, so the raw-logit rel diff is a poor gate; top-1 agreement (would gen pick the
    // same token?) is the gen-relevant signal, verified against a trained model separately.
    const bool ok = top1_pct >= 95.0;
    std::printf("cuda forward_one check: T=%d | top-1 agree %.1f%% (max rel diff %.2e)  %s | decode %.3f ms/tok (%.0f tok/s)\n",
                T, top1_pct, maxrel, ok ? "OK" : "FAIL", per_ms, toks);
    if (out_maxreldiff) *out_maxreldiff = maxrel;
    if (out_toks)       *out_toks       = toks;
    return ok ? 0 : 2;
}

// ============================================================================
//  Training step: forward + backward + AdamW on device (Phase 2d), parity-gated
// ============================================================================
// Shared driver: upload ids+targets, run the activation-saving forward and the reverse pass into
// g_dev_grad, sync, and read back the mean cross-entropy. Requires upload_params() first.
static int run_fwd_bwd(const int* ids, const int* targets, int batch, int T, double* out_loss,
                       const int* lengths = nullptr) {
    if (!g_dev_params) return 1;
    if (batch < 1 || batch > MAX_FWD_BATCH || T < 1 || T > SEQ_LEN) return 1;
    // Row-product sizing: a step only needs batch*T rows, so a varied-T caller that keeps
    // batch*T inside a previously reserved budget (sub0_cuda_train_reserve) never reallocates here.
    if (fwd_alloc(batch, false, T) || train_alloc(batch, T) || opt_alloc()) return 1;
    ensure_cublas();
    set_handle_tf32(CudaTf32::get());            // training uses the baked math mode
    const int M = batch * T;
    SUB0_CUDA_CHECK(cudaMemcpyAsync(g_fwd.dids, ids, static_cast<size_t>(M) * sizeof(int),
                                    cudaMemcpyHostToDevice, g_stream));
    SUB0_CUDA_CHECK(cudaMemcpyAsync(g_tr.dtargets, targets, static_cast<size_t>(M) * sizeof(int),
                                    cudaMemcpyHostToDevice, g_stream));
    const int* d_lengths = nullptr;              // null => every window is full T (dense path)
    if (lengths) {
        SUB0_CUDA_CHECK(cudaMemcpyAsync(g_tr.lengths, lengths, static_cast<size_t>(batch) * sizeof(int),
                                        cudaMemcpyHostToDevice, g_stream));
        d_lengths = g_tr.lengths;
    }
    forward_train(batch, T);
    backward_device(batch, T, d_lengths);
    double loss = 0.0;
    SUB0_CUDA_CHECK(cudaMemcpyAsync(&loss, g_tr.loss, sizeof(double), cudaMemcpyDeviceToHost, g_stream));
    SUB0_CUDA_CHECK(cudaStreamSynchronize(g_stream));
    SUB0_CUDA_CHECK(cudaGetLastError());
    if (out_loss) *out_loss = loss;              // the CE kernel already produced the mean
    return 0;
}

// Forward+backward only: fills out_grad[PARAM_FLOATS] with the reduced gradient (for the
// gradient-parity test against the CPU train_batch grad). Optionally returns the mean loss.
// `lengths` (optional) masks short-window padding exactly as the training step does.
SUB0_CUDA_API int sub0_cuda_backward(const int* ids, const int* targets, int batch, int T,
                                     float* out_grad, double* out_loss, const int* lengths) {
    if (run_fwd_bwd(ids, targets, batch, T, out_loss, lengths)) return 1;
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
// path the GPU train stage will drive once it replaces the CPU backend. `lengths` (optional) gives
// each window's trained length so short documents padded up to T train only on their real tokens.
SUB0_CUDA_API int sub0_cuda_train_step(const int* ids, const int* targets, int batch, int T,
                                       float lr, long t, double* out_loss, const int* lengths) {
    if (run_fwd_bwd(ids, targets, batch, T, out_loss, lengths)) return 1;
    device_adam_step(lr, t, 0.9f, 0.95f, 1e-8f, 0.01f, 1.0f);
    SUB0_CUDA_CHECK(cudaStreamSynchronize(g_stream));
    SUB0_CUDA_CHECK(cudaGetLastError());
    return 0;
}

// Reserve the full training row budget (batch * SEQ_LEN rows) up front. A varied-T training loop
// calls this ONCE with its nominal/tuned batch, then every per-step (batch_t, T) whose product fits
// the budget reuses the same buffers -- without this, the first short-T step would size the scratch
// to ITS product and each new per-step maximum would trigger a full grow-realloc (~30 cudaMallocs +
// graph invalidation) mid-run. Also pre-sizes the flash-backward stats scratch to its own ceiling
// (batch * N_HEADS * SEQ_LEN >= batch_t * N_HEADS * T for every admitted pair) for the same reason.
// Failure (VRAM) is a clean nonzero so the caller can fall back to the CPU path BEFORE training
// starts, instead of discovering the OOM on step 1.
SUB0_CUDA_API int sub0_cuda_train_reserve(int batch) {
    if (batch < 1 || batch > MAX_FWD_BATCH) return 1;
    // On ANY failure below, unconditionally release whatever partially succeeded before returning --
    // fwd_alloc/train_alloc/ensure_bwd_stats each fail mid-sequence (e.g. train_alloc's ~20-buffer
    // cudaMalloc chain can succeed on the first dozen and fail on a later, larger one) leaving real,
    // sizeable VRAM resident with capacity still marked 0. A caller measuring free VRAM right after a
    // failed reserve (to size a smaller fallback attempt, see GpuTrainer::enable) would otherwise see
    // that leftover partial allocation as "in use", undercounting what's really available once this
    // function's own next call (or anyone else's) frees it anyway.
    if (fwd_alloc(batch, false) || train_alloc(batch) || opt_alloc() ||
        ensure_bwd_stats(static_cast<size_t>(batch) * N_HEADS * SEQ_LEN)) {
        fwd_free_batch(); train_free(); opt_free(); free_bwd_stats();
        return 1;
    }
    return 0;
}

// Scratch observability for the row-budget tests: current row capacities and the monotonic
// (re)allocation counts (bumped every time fwd_alloc/train_alloc actually allocates, NOT on the
// capacity-hit fast path). Any pointer may be null.
SUB0_CUDA_API int sub0_cuda_scratch_stats(long long* fwd_rows, long long* tr_rows,
                                          long long* fwd_grows, long long* tr_grows) {
    if (fwd_rows)  *fwd_rows  = static_cast<long long>(g_fwd_rows);
    if (tr_rows)   *tr_rows   = static_cast<long long>(g_tr_rows);
    if (fwd_grows) *fwd_grows = g_fwd_grows;
    if (tr_grows)  *tr_grows  = g_tr_grows;
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
    if (fwd_alloc(batch)) return 1;
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

// On-device correctness + speed guard for the flash attention FORWARD. Runs the naive reference
// (attn_train_act_kernel) and the tiled kernel (attn_fwd_tiled_kernel) on the SAME random q/k/v at
// the training config, then (1) compares outputs -- both consume keys in increasing j with the same
// online-softmax recurrence, so they must agree to ~bf16 rounding -- and (2) times both for a
// dimensionless speedup RATIO. The ratio is the regression guard the Catch2 test asserts on: unlike
// an absolute-ms floor it does NOT drift with GPU clocks or host load (both kernels ride the same
// conditions), so it is a stable structural check that the tiling stayed intact. Returns nonzero
// only on a PARITY failure (a real bug); the caller chooses the minimum speedup to require.
SUB0_CUDA_API int sub0_cuda_attn_check(int batch, int T, int iters,
                                       double* out_maxreldiff, double* out_speedup) {
    if (T < 1 || T > SEQ_LEN) return 1;
    if (batch < 1) batch = 1;
    if (batch > MAX_FWD_BATCH) batch = MAX_FWD_BATCH;
    if (iters < 1) iters = 50;
    if (sub0_cuda_init()) return 1;
    const int    C = D_MODEL, H = N_HEADS, M = batch * T;
    const size_t nqkv = static_cast<size_t>(M) * 3 * C;                 // fused q|k|v, in_stride = 3C
    const size_t natt = static_cast<size_t>(M) * C;

    std::vector<float> h(nqkv);                                         // random q/k/v (host -> f32 -> act_t)
    unsigned s = 0x9e3779b9u;
    for (size_t i = 0; i < nqkv; ++i) { s = s * 1664525u + 1013904223u; h[i] = static_cast<float>(s >> 8) / 8388608.0f - 1.0f; }
    float* dqf = nullptr; act_t* dqkv = nullptr; act_t* dref = nullptr; act_t* dtes = nullptr;
    float* fref = nullptr; float* ftes = nullptr;
    SUB0_CUDA_CHECK(cudaMalloc(&dqf,  nqkv * sizeof(float)));
    SUB0_CUDA_CHECK(cudaMalloc(&dqkv, nqkv * sizeof(act_t)));
    SUB0_CUDA_CHECK(cudaMalloc(&dref, natt * sizeof(act_t)));
    SUB0_CUDA_CHECK(cudaMalloc(&dtes, natt * sizeof(act_t)));
    SUB0_CUDA_CHECK(cudaMalloc(&fref, natt * sizeof(float)));
    SUB0_CUDA_CHECK(cudaMalloc(&ftes, natt * sizeof(float)));
    SUB0_CUDA_CHECK(cudaMemcpy(dqf, h.data(), nqkv * sizeof(float), cudaMemcpyHostToDevice));
    { const int bk = 256; f32_to_act_kernel<<<static_cast<int>((nqkv + bk - 1) / bk), bk, 0, g_stream>>>(dqf, dqkv, static_cast<int>(nqkv)); }

    constexpr int HD = D_HEAD, TK = attn_tile_k<HD>();
    auto run_naive = [&] {
        const int blk = 64, total = batch * H * T;
        attn_train_act_kernel<act_t><<<(total + blk - 1) / blk, blk, 0, g_stream>>>(
            dqkv, dqkv + C, dqkv + 2 * C, dref, batch, T, C, H, 3 * C);
    };
    auto run_tiled = [&] {
        const dim3 block(kAttnTileQ), grid((T + kAttnTileQ - 1) / kAttnTileQ, H, batch);
        attn_fwd_tiled_kernel<act_t, HD, TK><<<grid, block, 0, g_stream>>>(
            dqkv, dqkv + C, dqkv + 2 * C, dtes, T, C, 3 * C);
    };
    run_naive(); run_tiled();
    SUB0_CUDA_CHECK(cudaStreamSynchronize(g_stream));
    SUB0_CUDA_CHECK(cudaGetLastError());

    { const int bk = 256, n = static_cast<int>(natt);                  // bring both outputs to f32 for the host
      act_to_f32_kernel<<<(n + bk - 1) / bk, bk, 0, g_stream>>>(dref, fref, n);
      act_to_f32_kernel<<<(n + bk - 1) / bk, bk, 0, g_stream>>>(dtes, ftes, n); }
    std::vector<float> a(natt), bvals(natt);
    SUB0_CUDA_CHECK(cudaMemcpy(a.data(),     fref, natt * sizeof(float), cudaMemcpyDeviceToHost));
    SUB0_CUDA_CHECK(cudaMemcpy(bvals.data(), ftes, natt * sizeof(float), cudaMemcpyDeviceToHost));
    double maxabs = 0.0, maxmag = 1e-30;
    for (size_t i = 0; i < natt; ++i) {
        const double d = std::fabs(static_cast<double>(a[i]) - bvals[i]), mg = std::fabs(static_cast<double>(a[i]));
        if (d > maxabs) maxabs = d;
        if (mg > maxmag) maxmag = mg;
    }
    const double reldiff = maxabs / maxmag;

    auto timed = [&](auto&& fn) -> double {                            // GPU-event time; the RATIO is the guard
        for (int w = 0; w < 5; ++w) fn();
        cudaStreamSynchronize(g_stream);
        cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
        cudaEventRecord(e0, g_stream);
        for (int it = 0; it < iters; ++it) fn();
        cudaEventRecord(e1, g_stream); cudaEventSynchronize(e1);
        float ms = 0.f; cudaEventElapsedTime(&ms, e0, e1); cudaEventDestroy(e0); cudaEventDestroy(e1);
        return static_cast<double>(ms) / iters;
    };
    const double naive_ms = timed(run_naive), tiled_ms = timed(run_tiled);
    const double speedup  = tiled_ms > 0.0 ? naive_ms / tiled_ms : 0.0;

    cudaFree(dqf); cudaFree(dqkv); cudaFree(dref); cudaFree(dtes); cudaFree(fref); cudaFree(ftes);
    const bool ok = reldiff < 5e-2;
    std::printf("cuda attn check: batch=%d T=%d HD=%d | naive %.3f ms  tiled %.3f ms = %.1fx | "
                "max rel diff %.2e  %s\n", batch, T, HD, naive_ms, tiled_ms, speedup, reldiff, ok ? "OK" : "FAIL");
    if (out_maxreldiff) *out_maxreldiff = reldiff;
    if (out_speedup)    *out_speedup    = speedup;
    return ok ? 0 : 2;
}

// On-device correctness + speed guard for the flash attention BACKWARD (the three atomic-free tiled
// kernels vs the naive head-per-thread reference, which is itself atomic-free and CPU-grad-gated).
// Runs both on the same random q/k/v/dout at the training config, compares the full dq|dk|dv grad
// buffer (must agree to bf16 rounding -- the tiled path takes dot_i from the bf16-rounded saved
// output, so it is not bit-exact but very close), and reports a dimensionless speedup RATIO (the
// stable regression guard). Returns nonzero only on a PARITY failure.
SUB0_CUDA_API int sub0_cuda_attn_bwd_check(int batch, int T, int iters,
                                           double* out_maxreldiff, double* out_speedup) {
    if (T < 1 || T > SEQ_LEN) return 1;
    if (batch < 1) batch = 1;
    if (batch > MAX_FWD_BATCH) batch = MAX_FWD_BATCH;
    if (iters < 1) iters = 30;
    if (sub0_cuda_init()) return 1;
    const int    C = D_MODEL, H = N_HEADS, M = batch * T;
    const size_t nqkv = static_cast<size_t>(M) * 3 * C, natt = static_cast<size_t>(M) * C;
    constexpr int HD = D_HEAD;

    std::vector<float> h(nqkv);                                         // random q/k/v
    unsigned s = 0x85ebca6bu;
    for (size_t i = 0; i < nqkv; ++i) { s = s * 1664525u + 1013904223u; h[i] = static_cast<float>(s >> 8) / 8388608.0f - 1.0f; }
    std::vector<float> hd(natt);                                        // random dout
    for (size_t i = 0; i < natt; ++i) { s = s * 1664525u + 1013904223u; hd[i] = static_cast<float>(s >> 8) / 8388608.0f - 1.0f; }

    float* dqf = nullptr; float* ddf = nullptr; act_t* dqkv = nullptr; act_t* ddout = nullptr;
    act_t* dref = nullptr; act_t* dtes = nullptr; float* fref = nullptr; float* ftes = nullptr;
    SUB0_CUDA_CHECK(cudaMalloc(&dqf,  nqkv * sizeof(float)));
    SUB0_CUDA_CHECK(cudaMalloc(&ddf,  natt * sizeof(float)));
    SUB0_CUDA_CHECK(cudaMalloc(&dqkv, nqkv * sizeof(act_t)));
    SUB0_CUDA_CHECK(cudaMalloc(&ddout, natt * sizeof(act_t)));
    SUB0_CUDA_CHECK(cudaMalloc(&dref, nqkv * sizeof(act_t)));
    SUB0_CUDA_CHECK(cudaMalloc(&dtes, nqkv * sizeof(act_t)));
    SUB0_CUDA_CHECK(cudaMalloc(&fref, nqkv * sizeof(float)));
    SUB0_CUDA_CHECK(cudaMalloc(&ftes, nqkv * sizeof(float)));
    SUB0_CUDA_CHECK(cudaMemcpy(dqf, h.data(),  nqkv * sizeof(float), cudaMemcpyHostToDevice));
    SUB0_CUDA_CHECK(cudaMemcpy(ddf, hd.data(), natt * sizeof(float), cudaMemcpyHostToDevice));
    { const int bk = 256;
      f32_to_act_kernel<<<static_cast<int>((nqkv + bk - 1) / bk), bk, 0, g_stream>>>(dqf, dqkv, static_cast<int>(nqkv));
      f32_to_act_kernel<<<static_cast<int>((natt + bk - 1) / bk), bk, 0, g_stream>>>(ddf, ddout, static_cast<int>(natt)); }

    auto run_naive = [&] {                                             // atomic-free head-per-thread reference
        cudaMemsetAsync(dref, 0, nqkv * sizeof(act_t), g_stream);      // it accumulates with +=
        const int blk = 64, total = batch * H;
        attn_backward_head_act_kernel<act_t><<<(total + blk - 1) / blk, blk, 0, g_stream>>>(
            dqkv, dqkv + C, dqkv + 2 * C, ddout, dref, dref + C, dref + 2 * C, batch, T, C, H, 3 * C);
    };
    auto run_tiled = [&] { launch_attn_bwd_t<act_t>(dqkv, ddout, dtes, batch, T, C, H, 3 * C); };
    run_naive(); run_tiled();
    SUB0_CUDA_CHECK(cudaStreamSynchronize(g_stream));
    SUB0_CUDA_CHECK(cudaGetLastError());

    { const int bk = 256, n = static_cast<int>(nqkv);
      act_to_f32_kernel<<<(n + bk - 1) / bk, bk, 0, g_stream>>>(dref, fref, n);
      act_to_f32_kernel<<<(n + bk - 1) / bk, bk, 0, g_stream>>>(dtes, ftes, n); }
    std::vector<float> a(nqkv), bvals(nqkv);
    SUB0_CUDA_CHECK(cudaMemcpy(a.data(),     fref, nqkv * sizeof(float), cudaMemcpyDeviceToHost));
    SUB0_CUDA_CHECK(cudaMemcpy(bvals.data(), ftes, nqkv * sizeof(float), cudaMemcpyDeviceToHost));
    double maxabs = 0.0, maxmag = 1e-30;
    for (size_t i = 0; i < nqkv; ++i) {
        const double d = std::fabs(static_cast<double>(a[i]) - bvals[i]), mg = std::fabs(static_cast<double>(a[i]));
        if (d > maxabs) maxabs = d;
        if (mg > maxmag) maxmag = mg;
    }
    const double reldiff = maxabs / maxmag;

    auto timed = [&](auto&& fn) -> double {
        for (int w = 0; w < 5; ++w) fn();
        cudaStreamSynchronize(g_stream);
        cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
        cudaEventRecord(e0, g_stream);
        for (int it = 0; it < iters; ++it) fn();
        cudaEventRecord(e1, g_stream); cudaEventSynchronize(e1);
        float ms = 0.f; cudaEventElapsedTime(&ms, e0, e1); cudaEventDestroy(e0); cudaEventDestroy(e1);
        return static_cast<double>(ms) / iters;
    };
    const double naive_ms = timed(run_naive), tiled_ms = timed(run_tiled);
    const double speedup  = tiled_ms > 0.0 ? naive_ms / tiled_ms : 0.0;

    cudaFree(dqf); cudaFree(ddf); cudaFree(dqkv); cudaFree(ddout);
    cudaFree(dref); cudaFree(dtes); cudaFree(fref); cudaFree(ftes);
    // NOTE: the naive reference ACCUMULATES dq/dk/dv in bf16 (read-modify-write per step), so it
    // carries ~sqrt(T)*bf16_eps (~6% at T=256) of accumulation noise; the tiled kernels accumulate
    // in FP32 and round once, so they are the MORE accurate of the two. This coarse gate only catches
    // gross breakage -- the tight correctness gate is the CPU-fp32 gradient parity test (cuda_tests).
    const bool ok = reldiff < 1.5e-1;
    std::printf("cuda attn-bwd check: batch=%d T=%d HD=%d | naive %.3f ms  tiled %.3f ms = %.1fx | "
                "max rel diff %.2e  %s\n", batch, T, HD, naive_ms, tiled_ms, speedup, reldiff, ok ? "OK" : "FAIL");
    if (out_maxreldiff) *out_maxreldiff = reldiff;
    if (out_speedup)    *out_speedup    = speedup;
    return ok ? 0 : 2;
}

// Verifies gemm()'s beta=1 accumulate mode and bias_grad_kernel's/launch_linear_bwd's accumulate
// flag that ride on it -- groundwork added for a future row-chunked GEMM (e.g. splitting the
// lm_head backward over M) but not yet used by any real call site, so this is its only coverage
// today. Splits a [M,in]x[M,out] backward into two row-chunks (chunk 1 overwrites dW/dbias, chunk 2
// accumulates into the SAME buffers) and compares against a single full-M reference call -- if
// dW/dbias's math correctly splits as a sum over disjoint row ranges (it does: dW[i,o] =
// sum_m X[m,i]*dY[m,o] splits cleanly at any row boundary), the two must match up to GEMM
// reassociation noise, not the wrong-shape/wrong-scale error a real accumulate-mode bug would cause.
SUB0_CUDA_API int sub0_cuda_test_accumulate_check(int M, int in, int out, double* out_maxreldiff) {
    if (M < 2 || in < 1 || out < 1) return 1;
    if (sub0_cuda_init()) return 1;
    ensure_cublas();
    const int M1 = M / 2, M2 = M - M1;

    std::vector<float> hX(static_cast<size_t>(M) * in), hdY(static_cast<size_t>(M) * out);
    unsigned s = 0x9e3779b9u;
    auto randf = [&] { s = s * 1664525u + 1013904223u; return static_cast<float>(s >> 8) / 8388608.0f - 1.0f; };
    for (auto& v : hX)  v = randf();
    for (auto& v : hdY) v = randf();

    float *dX_in = nullptr, *dY = nullptr;
    float *dW_ref = nullptr, *dbias_ref = nullptr, *dW_chunk = nullptr, *dbias_chunk = nullptr;
    SUB0_CUDA_CHECK(cudaMalloc(&dX_in, hX.size() * sizeof(float)));
    SUB0_CUDA_CHECK(cudaMalloc(&dY,    hdY.size() * sizeof(float)));
    SUB0_CUDA_CHECK(cudaMalloc(&dW_ref,      static_cast<size_t>(in) * out * sizeof(float)));
    SUB0_CUDA_CHECK(cudaMalloc(&dbias_ref,   static_cast<size_t>(out) * sizeof(float)));
    SUB0_CUDA_CHECK(cudaMalloc(&dW_chunk,    static_cast<size_t>(in) * out * sizeof(float)));
    SUB0_CUDA_CHECK(cudaMalloc(&dbias_chunk, static_cast<size_t>(out) * sizeof(float)));
    SUB0_CUDA_CHECK(cudaMemcpy(dX_in, hX.data(),  hX.size()  * sizeof(float), cudaMemcpyHostToDevice));
    SUB0_CUDA_CHECK(cudaMemcpy(dY,    hdY.data(), hdY.size() * sizeof(float), cudaMemcpyHostToDevice));

    // Reference: single full-M call, accumulate=false (the existing, already-trusted behavior).
    launch_linear_bwd(dX_in, nullptr, dY, nullptr, dW_ref, dbias_ref, M, in, out);

    // Chunked: same X/dY split at row M1 -- chunk 1 overwrites dW_chunk/dbias_chunk, chunk 2
    // accumulates into the SAME buffers. [M,in]/[M,out] are row-major, so row M1 starts at offset
    // M1*in / M1*out floats.
    launch_linear_bwd(dX_in, nullptr, dY, nullptr, dW_chunk, dbias_chunk, M1, in, out, false, false);
    launch_linear_bwd(dX_in + static_cast<size_t>(M1) * in, nullptr, dY + static_cast<size_t>(M1) * out,
                      nullptr, dW_chunk, dbias_chunk, M2, in, out, false, true);
    SUB0_CUDA_CHECK(cudaStreamSynchronize(g_stream));
    SUB0_CUDA_CHECK(cudaGetLastError());

    std::vector<float> hW_ref(static_cast<size_t>(in) * out), hW_chunk(static_cast<size_t>(in) * out);
    std::vector<float> hb_ref(out), hb_chunk(out);
    SUB0_CUDA_CHECK(cudaMemcpy(hW_ref.data(),   dW_ref,      hW_ref.size()   * sizeof(float), cudaMemcpyDeviceToHost));
    SUB0_CUDA_CHECK(cudaMemcpy(hW_chunk.data(), dW_chunk,    hW_chunk.size() * sizeof(float), cudaMemcpyDeviceToHost));
    SUB0_CUDA_CHECK(cudaMemcpy(hb_ref.data(),   dbias_ref,   hb_ref.size()   * sizeof(float), cudaMemcpyDeviceToHost));
    SUB0_CUDA_CHECK(cudaMemcpy(hb_chunk.data(), dbias_chunk, hb_chunk.size() * sizeof(float), cudaMemcpyDeviceToHost));

    double maxrel = 0.0;
    for (size_t i = 0; i < hW_ref.size(); ++i) {
        const double d = std::fabs(static_cast<double>(hW_chunk[i]) - hW_ref[i]);
        const double denom = std::max(1e-6, static_cast<double>(std::fabs(hW_ref[i])));
        maxrel = std::max(maxrel, d / denom);
    }
    for (size_t i = 0; i < hb_ref.size(); ++i) {
        const double d = std::fabs(static_cast<double>(hb_chunk[i]) - hb_ref[i]);
        const double denom = std::max(1e-6, static_cast<double>(std::fabs(hb_ref[i])));
        maxrel = std::max(maxrel, d / denom);
    }
    if (out_maxreldiff) *out_maxreldiff = maxrel;

    cudaFree(dX_in); cudaFree(dY);
    cudaFree(dW_ref); cudaFree(dbias_ref); cudaFree(dW_chunk); cudaFree(dbias_chunk);
    return 0;
}

// Register/local-memory-spill regression guard for the five flash-attention kernels: query the
// ACTUAL compiled kernel attributes via the CUDA runtime (cudaFuncGetAttributes), not by parsing
// ptxas -v text output -- deterministic and load-independent, same philosophy as the speedup RATIO
// above (not wall-clock). cudaFuncAttributes::localSizeBytes > 0 means the kernel is spilling
// per-thread state to local memory. See the dq/dv/dk register-split work: stats/dq/dv/fwd must stay
// at ZERO spill (each was driven there deliberately); dk keeps a small, bounded spill by design (its
// 3*HD register need has no further clean split -- see the comment above attn_bwd_dv_kernel). This
// function only MEASURES; a Catch2 test applies the actual pass/fail thresholds, so a future change
// to any of these kernels' resident state gets caught here before it silently re-spills at scale.
// Any out-pointer may be null. Returns nonzero only if a query itself fails (e.g. a bad device).
SUB0_CUDA_API int sub0_cuda_attn_regcheck(int* stats_regs, int* stats_spill,
                                          int* dq_regs, int* dq_spill,
                                          int* dv_regs, int* dv_spill,
                                          int* dk_regs, int* dk_spill,
                                          int* fwd_regs, int* fwd_spill) {
    if (sub0_cuda_init()) return 1;
    constexpr int HD = D_HEAD, TK = attn_tile_k<HD>(), TQ = attn_tile_q<HD>();
    cudaFuncAttributes attr;
    auto query = [&](const void* fn, int* regs, int* spill) -> bool {
        if (cudaFuncGetAttributes(&attr, fn) != cudaSuccess) return false;
        if (regs)  *regs  = attr.numRegs;
        if (spill) *spill = static_cast<int>(attr.localSizeBytes);
        return true;
    };
    bool ok = true;
    ok &= query(reinterpret_cast<const void*>(&attn_bwd_stats_kernel<act_t, HD, TK>), stats_regs, stats_spill);
    ok &= query(reinterpret_cast<const void*>(&attn_bwd_dq_kernel<act_t, HD, TK>),    dq_regs,    dq_spill);
    ok &= query(reinterpret_cast<const void*>(&attn_bwd_dv_kernel<act_t, HD, TQ>),    dv_regs,    dv_spill);
    ok &= query(reinterpret_cast<const void*>(&attn_bwd_dk_kernel<act_t, HD, TQ>),    dk_regs,    dk_spill);
    ok &= query(reinterpret_cast<const void*>(&attn_fwd_tiled_kernel<act_t, HD, TK>), fwd_regs,   fwd_spill);
    return ok ? 0 : 1;
}

// Adaptive timing bounds (shared with the CPU tuner via sub0/bench.hpp). Rather than a fixed
// iteration count -- which makes profiling time balloon as the per-step cost grows with batch --
// we size the timed run to a wall-time BUDGET so every measurement costs roughly the same. These
// are the GPU-flavoured defaults (a couple of warmups cover clock ramp + graph capture).
static constexpr sub0::bench::Budget TUNE_BUDGET{
    .budget_ms = 1200.0, .warmup = 2, .min_iters = 3, .max_iters = 60,
};

// Time a full TRAINING step (forward_train + backward_device + AdamW) at a given batch: synthetic
// finite params + ids/targets (timing is data-independent), warmups, then timed steps. Writes the
// mean step time (ms) to *out_ms. Quiet -- callers print. batch clamped to MAX_FWD_BATCH.
//   iters > 0  : run exactly that many timed steps (explicit benchmark).
//   iters <= 0 : auto-size the run to `budget_ms` of wall time -- probe once to estimate the
//                per-step cost, then run clamp(budget/est, MIN, MAX) steps. Keeps the profiling
//                wall-time ~constant across batch sizes instead of growing with the workload.
static int time_train_step(int batch, int T, int iters, double budget_ms, double* out_ms) {
    if (T < 1 || T > SEQ_LEN) return 1;
    if (batch < 1) batch = 1;
    if (batch > MAX_FWD_BATCH) batch = MAX_FWD_BATCH;
    if (sub0_cuda_init() || fwd_alloc(batch, false) || train_alloc(batch) || opt_alloc()) return 1;
    ensure_cublas();
    set_handle_tf32(CudaTf32::get());
    const int M = batch * T;

    // Synthetic, finite weights so nothing overflows; build the fused QKV from them once.
    std::vector<float> hp(sub0::PARAM_FLOATS, 0.02f);
    cudaMemcpy(g_dev_params, hp.data(), sub0::PARAM_FLOATS * sizeof(float), cudaMemcpyHostToDevice);
    build_qkv_weights();
    cudaMemset(g_dev_m, 0, sub0::PARAM_FLOATS * sizeof(float));
    cudaMemset(g_dev_vel, 0, sub0::PARAM_FLOATS * sizeof(float));
    std::vector<int> hid(static_cast<size_t>(M)), htg(static_cast<size_t>(M));
    for (int i = 0; i < M; ++i) { hid[i] = i % VOCAB; htg[i] = (i + 1) % VOCAB; }
    cudaMemcpy(g_fwd.dids,     hid.data(), static_cast<size_t>(M) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(g_tr.dtargets,  htg.data(), static_cast<size_t>(M) * sizeof(int), cudaMemcpyHostToDevice);

    long step = 0;
    auto one_step = [&]() {
        forward_train(batch, T);
        backward_device(batch, T, nullptr);
        device_adam_step(0.001f, ++step, 0.9f, 0.95f, 1e-8f, 0.01f, 1.0f);
    };
    // Time `n` steps on the stream and return the elapsed milliseconds (GPU event timing is precise).
    auto run_timed = [&](int n) -> double {
        cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
        cudaEventRecord(s, g_stream);
        for (int i = 0; i < n; ++i) one_step();
        cudaEventRecord(e, g_stream); cudaEventSynchronize(e);
        float ms = 0.f; cudaEventElapsedTime(&ms, s, e);
        cudaEventDestroy(s); cudaEventDestroy(e);
        return static_cast<double>(ms);
    };

    if (iters > 0) {                                                 // explicit fixed-iteration benchmark
        for (int w = 0; w < TUNE_BUDGET.warmup; ++w) one_step();     // warmup
        cudaStreamSynchronize(g_stream);
        const double total = run_timed(iters);
        if (out_ms) *out_ms = total / iters;
        return 0;
    }
    // Budget mode: shared adaptive sizing (warmup + probe + clamp(budget/est, min, max)).
    sub0::bench::Budget b = TUNE_BUDGET;
    if (budget_ms > 0.0) b.budget_ms = budget_ms;
    const sub0::bench::Timing t = sub0::bench::adaptive_time(one_step, run_timed, b);
    if (out_ms) *out_ms = t.per_step_ms;
    return 0;
}

// Profile a full TRAINING step at a given batch and print ms/step + throughput (tok/s). The device
// measurement primitive the GPU autotuner sweeps over batch / kernel-config knobs. Returns the step
// time in `out_ms` (nullable).
SUB0_CUDA_API int sub0_cuda_train_benchmark(int batch, int T, int iters, double* out_ms) {
    double per = 0.0;
    if (time_train_step(batch, T, iters, 0.0, &per)) return 1;
    const int M = (batch < 1 ? 1 : (batch > MAX_FWD_BATCH ? MAX_FWD_BATCH : batch)) * T;
    const double toks = per > 0.0 ? (static_cast<double>(M) * 1000.0) / per : 0.0;
    std::printf("cuda bench train: batch=%d T=%d M=%d iters=%d | step %.3f ms | %.0f tok/s\n",
                (M / T), T, M, iters, per, toks);
    if (out_ms) *out_ms = per;
    return 0;
}

// Quiet training-step timer for the autotuner: measures with the CURRENT knob state (TF32 /
// attn-backward, set via sub0_cuda_set_*), sizing the timed run to `budget_ms` of wall time so
// every batch profiles in roughly the same time. Writes mean ms/step to *out_ms, prints nothing.
SUB0_CUDA_API int sub0_cuda_time_train_step(int batch, int T, double budget_ms, double* out_ms) {
    return time_train_step(batch, T, 0, budget_ms, out_ms);
}

// Per-phase profiler: times forward_train / backward_device / device_adam_step SEPARATELY via CUDA
// events so the step cost can be attributed -- the small-GEMM forward vs the checkpoint-RECOMPUTE
// backward vs the AdamW update (which carries the grad-norm host sync). Data-independent like
// time_train_step; warms up (clock + cuBLAS) then averages `iters` timed steps. Any out ptr may be null.
SUB0_CUDA_API int sub0_cuda_train_profile(int batch, int T, int iters,
                                          double* fwd_ms, double* bwd_ms, double* adam_ms) {
    if (T < 1 || T > SEQ_LEN || iters < 1) return 1;
    if (batch < 1) batch = 1;
    if (batch > MAX_FWD_BATCH) batch = MAX_FWD_BATCH;
    if (sub0_cuda_init() || fwd_alloc(batch, false) || train_alloc(batch) || opt_alloc()) return 1;
    ensure_cublas();
    set_handle_tf32(CudaTf32::get());
    const int M = batch * T;
    std::vector<float> hp(sub0::PARAM_FLOATS, 0.02f);
    cudaMemcpy(g_dev_params, hp.data(), sub0::PARAM_FLOATS * sizeof(float), cudaMemcpyHostToDevice);
    build_qkv_weights();
    cudaMemset(g_dev_m,   0, sub0::PARAM_FLOATS * sizeof(float));
    cudaMemset(g_dev_vel, 0, sub0::PARAM_FLOATS * sizeof(float));
    std::vector<int> hid(static_cast<size_t>(M)), htg(static_cast<size_t>(M));
    for (int i = 0; i < M; ++i) { hid[i] = i % VOCAB; htg[i] = (i + 1) % VOCAB; }
    cudaMemcpy(g_fwd.dids,    hid.data(), static_cast<size_t>(M) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(g_tr.dtargets, htg.data(), static_cast<size_t>(M) * sizeof(int), cudaMemcpyHostToDevice);

    long t = 0;
    auto one = [&] {
        forward_train(batch, T);
        backward_device(batch, T, nullptr);
        device_adam_step(0.001f, ++t, 0.9f, 0.95f, 1e-8f, 0.01f, 1.0f);
    };
    for (int w = 0; w < std::max(4, iters); ++w) one();        // warm the clock + cuBLAS before timing
    cudaStreamSynchronize(g_stream);

    cudaEvent_t e0, e1, e2, e3;
    cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventCreate(&e2); cudaEventCreate(&e3);
    double fsum = 0.0, bsum = 0.0, asum = 0.0;
    for (int i = 0; i < iters; ++i) {
        cudaEventRecord(e0, g_stream); forward_train(batch, T);
        cudaEventRecord(e1, g_stream); backward_device(batch, T, nullptr);
        cudaEventRecord(e2, g_stream); device_adam_step(0.001f, ++t, 0.9f, 0.95f, 1e-8f, 0.01f, 1.0f);
        cudaEventRecord(e3, g_stream); cudaEventSynchronize(e3);
        float f = 0.f, b = 0.f, a = 0.f;
        cudaEventElapsedTime(&f, e0, e1); cudaEventElapsedTime(&b, e1, e2); cudaEventElapsedTime(&a, e2, e3);
        fsum += f; bsum += b; asum += a;
    }
    cudaEventDestroy(e0); cudaEventDestroy(e1); cudaEventDestroy(e2); cudaEventDestroy(e3);
    if (fwd_ms)  *fwd_ms  = fsum / iters;
    if (bwd_ms)  *bwd_ms  = bsum / iters;
    if (adam_ms) *adam_ms = asum / iters;
    return 0;
}

// This build's model dimensions, packaged for the pure footprint model (sub0/memplan.hpp).
static constexpr sub0::memplan::Dims kFootprintDims{
    D_MODEL, N_LAYERS, N_HEADS, D_FF, SEQ_LEN, VOCAB,
};

// Predicted resident training footprint (MiB) for `batch`, straight from the pure model. No device
// work -- callers that only need the prediction (the runtime guard, the configurator) use this.
SUB0_CUDA_API int sub0_cuda_train_predicted_mb(int batch) {
    if (batch < 1) batch = 1;
    if (batch > MAX_FWD_BATCH) batch = MAX_FWD_BATCH;
    return sub0::memplan::train_resident_mb(kFootprintDims, batch, sizeof(act_t));
}

// Currently-free dedicated VRAM in MiB. The GPU tuner budgets its batch ladder against THIS rather
// than the baked GPU_VRAM_MB spec, because the usable budget is the card total minus the CUDA
// context/driver reservation (~1 GB) the pure footprint model cannot see -- which is why a batch the
// model says "fits" the spec can still spill. We force ONLY the context (cudaFree(nullptr) is the
// canonical no-op that triggers lazy context creation) so that reservation is already subtracted from
// the figure, but we do NOT allocate the param blob or realize cuBLAS here: the previous
// sub0_cuda_init()/ensure_cublas() allocated exactly the memory we are trying to measure free (it
// drove the budget negative -- "showed -2gb"). cuBLAS workspace is allocated lazily on the first GEMM,
// AFTER this probe, so the caller reserves headroom for it. Returns 0 on failure (caller falls back).
//
// Clears any PENDING sticky CUDA error first: this is also the probe GpuTrainer::enable() calls to
// compute a fallback batch right after an optimistic sub0_cuda_train_reserve() attempt failed (a real
// cudaMalloc OOM) -- an unhandled CUDA error is sticky and poisons every subsequent call on the same
// context, including THIS one's cudaMemGetInfo, until something consumes it. Without this, the
// fallback path silently measured free_mb=0 (not genuinely zero -- cudaMemGetInfo itself failing) and
// fell back to the less-accurate static-spec-minus-headroom budget instead of the real, more
// conservative live figure, right when accuracy matters most (immediately after proving the naive
// estimate was too optimistic).
SUB0_CUDA_API int sub0_cuda_free_vram_mb() {
    cudaGetLastError();                                    // clear any pending sticky error (see above)
    cudaFree(nullptr);                                     // ensure the context exists; allocates nothing
    std::size_t free_b = 0, total = 0;
    if (cudaMemGetInfo(&free_b, &total) != cudaSuccess) return 0;
    return static_cast<int>(free_b / (1024 * 1024));
}

// Self-validating footprint probe: allocate the FULL resident training set for `batch` on a clean
// device, measure the actual VRAM it consumed (cudaMemGetInfo delta), and return that alongside the
// pure model's prediction. The gap is how far the memplan.hpp mirror has drifted from the real
// allocations in this file -- the CUDA footprint test asserts it stays tiny, so adding/removing a
// device buffer without updating the mirror fails CI instead of silently mis-predicting in the field.
// Both outputs are MiB; either pointer may be null. Leaves the scratch allocated (callers re-tune or
// shut down as usual). Returns nonzero on a device/allocation failure.
SUB0_CUDA_API int sub0_cuda_train_footprint(int batch, double* predicted_mb, double* actual_mb) {
    if (batch < 1) batch = 1;
    if (batch > MAX_FWD_BATCH) batch = MAX_FWD_BATCH;
    sub0_cuda_shutdown();                                   // clean slate so the delta is purely ours
    std::size_t free_before = 0, total = 0;
    if (cudaMemGetInfo(&free_before, &total) != cudaSuccess) return 1;
    if (sub0_cuda_init() || fwd_alloc(batch, false) || train_alloc(batch) || opt_alloc()) return 1;
    build_qkv_weights();    // BF16 builds: also resident-allocates g_w1_16/g_w2_16/g_wo16/g_wqkv16
                             // (persistent_bytes() counts these -- see memplan.hpp), so the measured
                             // delta below stays comparable to the prediction instead of undercounting it.
    cudaDeviceSynchronize();                                // ensure the allocations are physically resident
    std::size_t free_after = 0;
    if (cudaMemGetInfo(&free_after, &total) != cudaSuccess) return 1;
    const double actual = free_before > free_after ? static_cast<double>(free_before - free_after) : 0.0;
    const double predicted = static_cast<double>(sub0::memplan::train_resident_bytes(kFootprintDims, batch, sizeof(act_t)));
    constexpr double MiB = 1024.0 * 1024.0;
    if (predicted_mb) *predicted_mb = predicted / MiB;
    if (actual_mb)    *actual_mb    = actual / MiB;
    return 0;
}
