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
#include <cuda_fp16.h>
#include <cstdint>

namespace sub0llm::backend::cuda::kernels {

using ::sub0llm::backend::cpu::BlockQ8_0;

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

// ── Q8_0 × Q8_0 matmul ────────────────────────────────────────────────────────
constexpr int QK = 32;   // Q8_0 block size

// Pack 4 signed int8 into the byte lanes of an int for __dp4a (reads them back as s8).
__device__ __forceinline__ int pack4_s8(const int8_t* p) {
    return  (static_cast<int>(static_cast<unsigned char>(p[0]))      )
          | (static_cast<int>(static_cast<unsigned char>(p[1])) <<  8)
          | (static_cast<int>(static_cast<unsigned char>(p[2])) << 16)
          | (static_cast<int>(static_cast<unsigned char>(p[3])) << 24);
}

// One thread per output Y[m,t]: dot weight row m with activation column t over nb Q8 blocks.
// int8 dot via __dp4a (DP4A, native sm_61+), scaled by the product of the two f16 block scales.
__global__ void matmul_q8_0_kernel(const BlockQ8_0* __restrict__ W,
                                   const BlockQ8_0* __restrict__ X,
                                   float* __restrict__ Y, int M, int K, int T) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;   // token / column
    const int m = blockIdx.y * blockDim.y + threadIdx.y;   // output feature / row
    if (m >= M || t >= T) return;
    const int nb = K / QK;
    const BlockQ8_0* wrow = W + static_cast<long long>(m) * nb;
    const BlockQ8_0* xcol = X + static_cast<long long>(t) * nb;
    float acc = 0.0f;
    for (int i = 0; i < nb; ++i) {
        const BlockQ8_0& wb = wrow[i];
        const BlockQ8_0& xb = xcol[i];
        int idot = 0;
        #pragma unroll
        for (int j = 0; j < QK; j += 4)
            idot = __dp4a(pack4_s8(wb.qs + j), pack4_s8(xb.qs + j), idot);
        const float wd = __half2float(__ushort_as_half(wb.d));
        const float xd = __half2float(__ushort_as_half(xb.d));
        acc += wd * xd * static_cast<float>(idot);
    }
    Y[static_cast<long long>(m) * T + t] = acc;
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

void launch_matmul_q8_0(const BlockQ8_0* dW, const BlockQ8_0* dXq, float* dY,
                        int M, int K, int T) {
    const dim3 block(16, 16);
    const dim3 grid2(grid(static_cast<std::size_t>(T), block.x),
                     grid(static_cast<std::size_t>(M), block.y));
    matmul_q8_0_kernel<<<grid2, block>>>(dW, dXq, dY, M, K, T);
}

} // namespace sub0llm::backend::cuda::kernels
