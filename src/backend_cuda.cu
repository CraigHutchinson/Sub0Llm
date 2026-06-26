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
