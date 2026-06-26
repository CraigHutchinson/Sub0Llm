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

// Trivial device kernel: y = a*x + y over n elements (the smoke-test workload).
__global__ void axpy_kernel(float a, const float* __restrict__ x, float* __restrict__ y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}

// Device parameter mirror: the model's flat weight blob (PARAM_FLOATS floats) lives here
// during a GPU run. The host params_ptr()/checkpoint buffers sync to/from it via the
// upload/download entry points below (the device half of the sync_params hooks).
float* g_dev_params = nullptr;

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
// thread per (window b, head h, query i): scores over j<=i WITHIN the window, __expf
// softmax, weighted sum of v. Windows are independent -- no cross-window attention.
__global__ void attn_kernel(const float* __restrict__ q, const float* __restrict__ k,
                            const float* __restrict__ v, float* __restrict__ out,
                            int batch, int T, int C, int H) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;   // idx over batch*H*T
    const int per = H * T;
    const int b   = idx / per;
    if (b >= batch) return;
    const int rem = idx - b * per;
    const int h   = rem / T;
    const int i   = rem % T;
    const int    d     = C / H;
    const int    off   = h * d;
    const float  scale = 1.0f / sqrtf(static_cast<float>(d));
    const size_t base  = static_cast<size_t>(b) * T * C;     // window b's first row
    float sc[SEQ_LEN];
    const float* qi = q + base + static_cast<size_t>(i) * C + off;
    float mx = -1e30f;
    for (int j = 0; j <= i; ++j) {
        const float* kj = k + base + static_cast<size_t>(j) * C + off;
        float s = 0.f;
        for (int a = 0; a < d; ++a) s += qi[a] * kj[a];
        s *= scale; sc[j] = s; if (s > mx) mx = s;
    }
    float Z = 0.f;
    for (int j = 0; j <= i; ++j) { sc[j] = __expf(sc[j] - mx); Z += sc[j]; }   // SFU fast exp
    float* oi = out + base + static_cast<size_t>(i) * C + off;
    for (int a = 0; a < d; ++a) oi[a] = 0.f;
    for (int j = 0; j <= i; ++j) {
        const float pj = sc[j] / Z;
        const float* vj = v + base + static_cast<size_t>(j) * C + off;
        for (int a = 0; a < d; ++a) oi[a] += pj * vj[a];
    }
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
inline void launch_gelu(const float* dX, float* dY, int n) {
    const int block = 256;
    gelu_kernel<<<(n + block - 1) / block, block, 0, g_stream>>>(dX, dY, n);
}
inline void launch_add(const float* dA, const float* dB, float* dC, int n) {
    const int block = 256;
    add_kernel<<<(n + block - 1) / block, block, 0, g_stream>>>(dA, dB, dC, n);
}
inline void launch_attn(const float* dQ, const float* dK, const float* dV, float* dOut,
                        int batch, int T, int C, int H) {
    const int block = 64;
    const int total = batch * H * T;
    attn_kernel<<<(total + block - 1) / block, block, 0, g_stream>>>(dQ, dK, dV, dOut, batch, T, C, H);
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
    SUB0_CUDA_CHECK(cudaMalloc(&g_fwd.q,      MC * sizeof(float)));
    SUB0_CUDA_CHECK(cudaMalloc(&g_fwd.kk,     MC * sizeof(float)));
    SUB0_CUDA_CHECK(cudaMalloc(&g_fwd.vv,     MC * sizeof(float)));
    SUB0_CUDA_CHECK(cudaMalloc(&g_fwd.att,    MC * sizeof(float)));
    SUB0_CUDA_CHECK(cudaMalloc(&g_fwd.proj,   MC * sizeof(float)));
    SUB0_CUDA_CHECK(cudaMalloc(&g_fwd.fbuf,   MC * sizeof(float)));
    SUB0_CUDA_CHECK(cudaMalloc(&g_fwd.ff1,    MF * sizeof(float)));
    SUB0_CUDA_CHECK(cudaMalloc(&g_fwd.gact,   MF * sizeof(float)));
    SUB0_CUDA_CHECK(cudaMalloc(&g_fwd.ff2,    MC * sizeof(float)));
    SUB0_CUDA_CHECK(cudaMalloc(&g_fwd.logits, MV * sizeof(float)));
    g_fwd.inited = true;
    return 0;
}

void fwd_free() {
    cudaFree(g_fwd.dids);  cudaFree(g_fwd.h);    cudaFree(g_fwd.a);    cudaFree(g_fwd.q);
    cudaFree(g_fwd.kk);    cudaFree(g_fwd.vv);   cudaFree(g_fwd.att);  cudaFree(g_fwd.proj);
    cudaFree(g_fwd.fbuf);  cudaFree(g_fwd.ff1);  cudaFree(g_fwd.gact); cudaFree(g_fwd.ff2);
    cudaFree(g_fwd.logits);
    g_fwd = FwdScratch{};
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
    float* q    = g_fwd.q;
    float* kk   = g_fwd.kk;
    float* vv   = g_fwd.vv;
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
        const float* Wq  = base + L[b0 + 2].off;
        const float* Wk  = base + L[b0 + 3].off;
        const float* Wv  = base + L[b0 + 4].off;
        const float* Wo  = base + L[b0 + 5].off;
        const float* W1  = base + L[b0 + 6].off;
        const float* b1  = base + L[b0 + 7].off;
        const float* W2  = base + L[b0 + 8].off;
        const float* b2  = base + L[b0 + 9].off;

        launch_rmsnorm(h, ln1, a, M, C);                  // a = rmsnorm(h, ln1)
        launch_linear(a, Wq, nullptr, q,  M, C, C);       // q,k,v = a . Wq/Wk/Wv
        launch_linear(a, Wk, nullptr, kk, M, C, C);
        launch_linear(a, Wv, nullptr, vv, M, C, C);
        launch_attn(q, kk, vv, att, batch, T, C, H);      // per-window causal attention
        launch_linear(att, Wo, nullptr, proj, M, C, C);   // proj = att . Wo
        launch_add(h, proj, h, MC);                       // h = h + proj
        launch_rmsnorm(h, ln2, fbuf, M, C);               // f = rmsnorm(h, ln2)
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
    if (g_cublas) { cublasDestroy(g_cublas); g_cublas = nullptr; }
    if (g_stream) { cudaStreamDestroy(g_stream); g_stream = nullptr; }
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
