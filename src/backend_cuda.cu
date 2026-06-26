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

#include <cstdio>
#include <vector>

#include <cuda_runtime.h>

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

// Trivial device kernel: y = a*x + y over n elements (the smoke-test workload).
__global__ void axpy_kernel(float a, const float* __restrict__ x, float* __restrict__ y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}

// Device parameter mirror: the model's flat weight blob (PARAM_FLOATS floats) lives here
// during a GPU run. The host params_ptr()/checkpoint buffers sync to/from it via the
// upload/download entry points below (the device half of the sync_params hooks).
float* g_dev_params = nullptr;

// Dense linear: Y[T,out] = X[T,in] . W[in,out] (+ bias[out]). Matches op_linear's dense
// (non-ternary) path -- one thread per (t,o) output element, summing over `in` in the same
// order as the CPU. The building block of every projection and the lm_head; the batched
// cuBLASLt form arrives in 2c, this naive kernel is the parity baseline.
__global__ void linear_kernel(const float* __restrict__ X, const float* __restrict__ W,
                              const float* __restrict__ bias, float* __restrict__ Y,
                              int T, int in, int out) {
    const int o = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y * blockDim.y + threadIdx.y;
    if (t < T && o < out) {
        float acc = bias ? bias[o] : 0.0f;
        for (int p = 0; p < in; ++p) acc += X[t * in + p] * W[p * out + o];
        Y[t * out + o] = acc;
    }
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

// h[t,j] = tok_emb[ids[t], j] + pos_emb[t, j]  (op_embed + op_add: first forward step).
__global__ void embed_add_kernel(const float* __restrict__ tok_emb, const float* __restrict__ pos_emb,
                                 const int* __restrict__ ids, float* __restrict__ h, int T, int C) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y * blockDim.y + threadIdx.y;
    if (t < T && j < C) h[t * C + j] = tok_emb[ids[t] * C + j] + pos_emb[t * C + j];
}

// y[t,j] = x[t,j] * (1/sqrt(mean_j x^2 + eps)) * gamma[j]  (op_rmsnorm forward; one row/thread).
__global__ void rmsnorm_kernel(const float* __restrict__ x, const float* __restrict__ gamma,
                               float* __restrict__ y, int T, int C) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < T) {
        const float eps = 1e-5f;
        const float* xr = x + static_cast<size_t>(t) * C;
        float ms = 0.f;
        for (int j = 0; j < C; ++j) ms += xr[j] * xr[j];
        ms /= C;
        const float r = rsqrtf(ms + eps);              // CUDA fast reciprocal sqrt
        float* yr = y + static_cast<size_t>(t) * C;
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

// Causal multi-head attention (op_attn, FAST_MATH softmax). One thread per (head, query i):
// scores over j<=i, fast_exp softmax, weighted sum of v. T is bounded by SEQ_LEN.
__global__ void attn_kernel(const float* __restrict__ q, const float* __restrict__ k,
                            const float* __restrict__ v, float* __restrict__ out,
                            int T, int C, int H) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;   // idx = head*T + i
    const int h = idx / T;
    const int i = idx % T;
    if (h >= H || i >= T) return;
    const int   d     = C / H;
    const int   off   = h * d;
    const float scale = 1.0f / sqrtf(static_cast<float>(d));
    float sc[SEQ_LEN];
    const float* qi = q + static_cast<size_t>(i) * C + off;
    float mx = -1e30f;
    for (int j = 0; j <= i; ++j) {
        const float* kj = k + static_cast<size_t>(j) * C + off;
        float s = 0.f;
        for (int a = 0; a < d; ++a) s += qi[a] * kj[a];
        s *= scale; sc[j] = s; if (s > mx) mx = s;
    }
    float Z = 0.f;
    for (int j = 0; j <= i; ++j) { sc[j] = __expf(sc[j] - mx); Z += sc[j]; }   // SFU fast exp
    float* oi = out + static_cast<size_t>(i) * C + off;
    for (int a = 0; a < d; ++a) oi[a] = 0.f;
    for (int j = 0; j <= i; ++j) {
        const float pj = sc[j] / Z;
        const float* vj = v + static_cast<size_t>(j) * C + off;
        for (int a = 0; a < d; ++a) oi[a] += pj * vj[a];
    }
}

// --- Device-pointer launchers (no host alloc; drive the resident forward chain) ---
inline void launch_linear(const float* dX, const float* dW, const float* dB, float* dY,
                          int T, int in, int out) {
    const dim3 block(16, 16);
    const dim3 grid((out + block.x - 1) / block.x, (T + block.y - 1) / block.y);
    linear_kernel<<<grid, block>>>(dX, dW, dB, dY, T, in, out);
}
inline void launch_rmsnorm(const float* dX, const float* dG, float* dY, int T, int C) {
    const int block = 64;
    rmsnorm_kernel<<<(T + block - 1) / block, block>>>(dX, dG, dY, T, C);
}
inline void launch_gelu(const float* dX, float* dY, int n) {
    const int block = 256;
    gelu_kernel<<<(n + block - 1) / block, block>>>(dX, dY, n);
}
inline void launch_add(const float* dA, const float* dB, float* dC, int n) {
    const int block = 256;
    add_kernel<<<(n + block - 1) / block, block>>>(dA, dB, dC, n);
}
inline void launch_attn(const float* dQ, const float* dK, const float* dV, float* dOut,
                        int T, int C, int H) {
    const int block = 64;
    const int total = H * T;
    attn_kernel<<<(total + block - 1) / block, block>>>(dQ, dK, dV, dOut, T, C, H);
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
    if (g_dev_params) cudaFree(g_dev_params);
    g_dev_params = nullptr;
}

SUB0_CUDA_API int sub0_cuda_upload_params(const float* host) {
    if (!g_dev_params) { const int r = sub0_cuda_init(); if (r) return r; }
    SUB0_CUDA_CHECK(cudaMemcpy(g_dev_params, host, sub0::PARAM_FLOATS * sizeof(float),
                               cudaMemcpyHostToDevice));
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

    const dim3 block(16, 16);
    const dim3 grid((out + block.x - 1) / block.x, (T + block.y - 1) / block.y);
    linear_kernel<<<grid, block>>>(dX, dW, dB, dY, T, in, out);
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
// Runs the same op sequence as Model::forward in backend_cpu.cpp on the device, slicing
// the uploaded weight blob (g_dev_params) by the shared constexpr PARAM_LAYOUT. Requires
// sub0_cuda_upload_params() first. out_logits receives [T, VOCAB]. Per-call device scratch
// (Phase 2b correctness baseline); 2c makes the buffers resident and batches M = B*T.
SUB0_CUDA_API int sub0_cuda_forward(const int* ids, int T, float* out_logits) {
    if (!g_dev_params) return 1;
    const auto&  L = sub0::PARAM_LAYOUT;
    const int    C = D_MODEL, F = D_FF, V = VOCAB, H = N_HEADS;
    float* const base = g_dev_params;

    const float* tok_emb = base + L[0].off;
    const float* pos_emb = base + L[1].off;
    const int    fi      = 2 + 10 * N_LAYERS;          // index of ln_f
    const float* ln_f    = base + L[fi + 0].off;
    const float* lm_head = base + L[fi + 1].off;
    const float* lm_bias = base + L[fi + 2].off;

    const size_t TC = static_cast<size_t>(T) * C;
    const size_t TF = static_cast<size_t>(T) * F;
    const size_t TV = static_cast<size_t>(T) * V;

    int*   dids   = nullptr;
    float* h      = nullptr;
    float* a      = nullptr;
    float* q      = nullptr;
    float* kk     = nullptr;
    float* vv     = nullptr;
    float* att    = nullptr;
    float* proj   = nullptr;
    float* fbuf   = nullptr;
    float* ff1    = nullptr;
    float* gact   = nullptr;
    float* ff2    = nullptr;
    float* logits = nullptr;
    SUB0_CUDA_CHECK(cudaMalloc(&dids, static_cast<size_t>(T) * sizeof(int)));
    SUB0_CUDA_CHECK(cudaMalloc(&h,      TC * sizeof(float)));
    SUB0_CUDA_CHECK(cudaMalloc(&a,      TC * sizeof(float)));
    SUB0_CUDA_CHECK(cudaMalloc(&q,      TC * sizeof(float)));
    SUB0_CUDA_CHECK(cudaMalloc(&kk,     TC * sizeof(float)));
    SUB0_CUDA_CHECK(cudaMalloc(&vv,     TC * sizeof(float)));
    SUB0_CUDA_CHECK(cudaMalloc(&att,    TC * sizeof(float)));
    SUB0_CUDA_CHECK(cudaMalloc(&proj,   TC * sizeof(float)));
    SUB0_CUDA_CHECK(cudaMalloc(&fbuf,   TC * sizeof(float)));
    SUB0_CUDA_CHECK(cudaMalloc(&ff1,    TF * sizeof(float)));
    SUB0_CUDA_CHECK(cudaMalloc(&gact,   TF * sizeof(float)));
    SUB0_CUDA_CHECK(cudaMalloc(&ff2,    TC * sizeof(float)));
    SUB0_CUDA_CHECK(cudaMalloc(&logits, TV * sizeof(float)));
    SUB0_CUDA_CHECK(cudaMemcpy(dids, ids, static_cast<size_t>(T) * sizeof(int), cudaMemcpyHostToDevice));

    // h = tok_emb[ids] + pos_emb
    {
        const dim3 block(16, 16);
        const dim3 grid((C + block.x - 1) / block.x, (T + block.y - 1) / block.y);
        embed_add_kernel<<<grid, block>>>(tok_emb, pos_emb, dids, h, T, C);
    }

    for (int l = 0; l < N_LAYERS; ++l) {
        const int    b0  = 2 + 10 * l;
        const float* ln1 = base + L[b0 + 0].off;
        const float* ln2 = base + L[b0 + 1].off;
        const float* Wq  = base + L[b0 + 2].off;
        const float* Wk  = base + L[b0 + 3].off;
        const float* Wv  = base + L[b0 + 4].off;
        const float* Wo  = base + L[b0 + 5].off;
        const float* W1  = base + L[b0 + 6].off;
        const float* b1  = base + L[b0 + 7].off;
        const float* W2  = base + L[b0 + 8].off;
        const float* b2  = base + L[b0 + 9].off;

        launch_rmsnorm(h, ln1, a, T, C);                  // a = rmsnorm(h, ln1)
        launch_linear(a, Wq, nullptr, q,  T, C, C);       // q,k,v = a . Wq/Wk/Wv
        launch_linear(a, Wk, nullptr, kk, T, C, C);
        launch_linear(a, Wv, nullptr, vv, T, C, C);
        launch_attn(q, kk, vv, att, T, C, H);             // att = attn(q,k,v)
        launch_linear(att, Wo, nullptr, proj, T, C, C);   // proj = att . Wo
        launch_add(h, proj, h, static_cast<int>(TC));     // h = h + proj
        launch_rmsnorm(h, ln2, fbuf, T, C);               // f = rmsnorm(h, ln2)
        launch_linear(fbuf, W1, b1, ff1, T, C, F);        // ff1 = f . W1 + b1
        launch_gelu(ff1, gact, static_cast<int>(TF));     // gelu
        launch_linear(gact, W2, b2, ff2, T, F, C);        // ff2 = gelu . W2 + b2
        launch_add(h, ff2, h, static_cast<int>(TC));      // h = h + ff2
    }
    launch_rmsnorm(h, ln_f, a, T, C);                     // a = rmsnorm(h, ln_f)
    launch_linear(a, lm_head, lm_bias, logits, T, C, V);  // logits = a . lm_head + lm_bias

    SUB0_CUDA_CHECK(cudaGetLastError());
    SUB0_CUDA_CHECK(cudaDeviceSynchronize());
    SUB0_CUDA_CHECK(cudaMemcpy(out_logits, logits, TV * sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(dids); cudaFree(h); cudaFree(a); cudaFree(q); cudaFree(kk); cudaFree(vv);
    cudaFree(att); cudaFree(proj); cudaFree(fbuf); cudaFree(ff1); cudaFree(gact);
    cudaFree(ff2); cudaFree(logits);
    return 0;
}
