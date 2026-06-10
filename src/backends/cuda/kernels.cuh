#pragma once
#ifdef SUB0LLM_CUDA
#include <cstddef>
#include <cstdint>

#include "sub0llm/backends/cpu/quant.hpp"   // BlockQ8_0 (shared host/device Q8_0 layout)

// CUDA kernel launcher declarations.
// These are defined in kernels.cu and are only compiled with nvcc.

namespace sub0llm::backend::cuda::kernels {

void launch_add_f32 (const float* a, const float* b, float* out, std::size_t n);
void launch_mul_f32 (const float* a, const float* b, float* out, std::size_t n);
void launch_relu_f32(const float* in, float* out, std::size_t n);
void launch_matmul_f32(const float* A, const float* B, float* C,
                       std::size_t M, std::size_t N, std::size_t K);

// Q8_0 × Q8_0 batched matmul on DEVICE pointers: Y[M,T] = Wq[M,K] · Xqᵀ, matching the CPU
// backend::cpu::matmul_q8_0_q8_0 semantics (Wq row-major (M, K/32), Xq (T, K/32), Y (M, T)
// row-major). int8 dot via __dp4a, per-block f16 scales. Caller owns the device buffers.
void launch_matmul_q8_0(const ::sub0llm::backend::cpu::BlockQ8_0* dW,
                        const ::sub0llm::backend::cpu::BlockQ8_0* dXq,
                        float* dY, int M, int K, int T);

} // namespace sub0llm::backend::cuda::kernels
#endif // SUB0LLM_CUDA
