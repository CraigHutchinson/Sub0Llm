// CUDA kernels — compiled by nvcc only when SUB0LLM_ENABLE_CUDA=ON.
//
// Each kernel follows the standard 1D grid / 1D block pattern:
//   grid  = ceil(n / BLOCK)  blocks
//   block = BLOCK             threads
//
// For real workloads, matmul would use cuBLAS (Chapter 11).  These naive CUDA
// kernels exist to demonstrate the dispatch model and GPU programming model.

#include "kernels.cuh"
#include <cuda_runtime.h>

namespace sub0llm::backend::cuda::kernels {

namespace {
constexpr int BLOCK = 256;

__global__ void add_f32_kernel(const float* a, const float* b, float* out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

__global__ void mul_f32_kernel(const float* a, const float* b, float* out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] * b[i];
}

__global__ void relu_f32_kernel(const float* in, float* out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i] > 0.0f ? in[i] : 0.0f;
}

// Tiled matmul — 16×16 shared-memory tile.
// For production use cuBLAS; this is for pedagogical clarity.
constexpr int TILE = 16;

__global__ void matmul_f32_kernel(const float* A, const float* B, float* C,
                                   int M, int N, int K) {
    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];

    const int row = blockIdx.y * TILE + threadIdx.y;
    const int col = blockIdx.x * TILE + threadIdx.x;

    float acc = 0.0f;
    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        sA[threadIdx.y][threadIdx.x] = (row < M && t*TILE + threadIdx.x < K)
            ? A[row * K + t*TILE + threadIdx.x] : 0.0f;
        sB[threadIdx.y][threadIdx.x] = (col < N && t*TILE + threadIdx.y < K)
            ? B[(t*TILE + threadIdx.y) * N + col] : 0.0f;
        __syncthreads();

        for (int k = 0; k < TILE; ++k) acc += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = acc;
}

inline int grid(std::size_t n, int block) {
    return static_cast<int>((n + static_cast<std::size_t>(block) - 1) / static_cast<std::size_t>(block));
}

} // anonymous namespace

void launch_add_f32(const float* a, const float* b, float* out, std::size_t n) {
    add_f32_kernel<<<grid(n, BLOCK), BLOCK>>>(a, b, out, static_cast<int>(n));
}

void launch_mul_f32(const float* a, const float* b, float* out, std::size_t n) {
    mul_f32_kernel<<<grid(n, BLOCK), BLOCK>>>(a, b, out, static_cast<int>(n));
}

void launch_relu_f32(const float* in, float* out, std::size_t n) {
    relu_f32_kernel<<<grid(n, BLOCK), BLOCK>>>(in, out, static_cast<int>(n));
}

void launch_matmul_f32(const float* A, const float* B, float* C,
                       std::size_t M, std::size_t N, std::size_t K) {
    const dim3 block(TILE, TILE);
    const dim3 grid2(grid(N, TILE), grid(M, TILE));
    matmul_f32_kernel<<<grid2, block>>>(A, B, C,
        static_cast<int>(M), static_cast<int>(N), static_cast<int>(K));
}

} // namespace sub0llm::backend::cuda::kernels
