#pragma once
#ifdef SUB0LLM_CUDA
#include <cstddef>

// CUDA kernel launcher declarations.
// These are defined in kernels.cu and are only compiled with nvcc.

namespace sub0llm::backend::cuda::kernels {

void launch_add_f32 (const float* a, const float* b, float* out, std::size_t n);
void launch_mul_f32 (const float* a, const float* b, float* out, std::size_t n);
void launch_relu_f32(const float* in, float* out, std::size_t n);
void launch_matmul_f32(const float* A, const float* B, float* C,
                       std::size_t M, std::size_t N, std::size_t K);

} // namespace sub0llm::backend::cuda::kernels
#endif // SUB0LLM_CUDA
