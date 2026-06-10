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
#include <mma.h>
#include <cstdint>

namespace sub0llm::backend::cuda::kernels {

using ::sub0llm::backend::cpu::BlockQ8_0;
namespace wmma = nvcuda::wmma;

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

// IMMA (int8 tensor-core) Q8 matmul. One warp computes a 16×16 output tile. Q8_0's per-32
// scale fights the tensor core (which wants uniform-scale K-accumulation), so we run ONE
// 16×16×16 MMA pair per Q8 block (= 32 K-elems → two K=16 steps) into an int32 fragment,
// then apply that block's wd[m]·xd[n] scales and accumulate into a float tile in shared mem.
// Weight/activation int8 + scales are gathered to shared each block (handles the scattered
// BlockQ8_0 layout and lets WMMA load contiguous tiles). M/T tails are zero-padded.
constexpr int MMA = 16;
__global__ void matmul_q8_0_mma_kernel(const BlockQ8_0* __restrict__ W,
                                       const BlockQ8_0* __restrict__ X,
                                       float* __restrict__ Y, int M, int K, int T) {
    const int m0 = blockIdx.y * MMA;   // tile's first weight row
    const int n0 = blockIdx.x * MMA;   // tile's first token / column
    const int lane = threadIdx.x;      // 0..31, single warp
    const int nb = K / QK;

    __shared__ signed char As[MMA][QK];   // weight tile  [row][k]   (matrix_a, row-major)
    __shared__ signed char Bs[MMA][QK];   // act tile     [col][k]   (matrix_b, col-major)
    __shared__ int         Cs[MMA][MMA];  // per-block int32 dot
    __shared__ float       facc[MMA][MMA];// scaled float accumulator (persists across blocks)
    __shared__ float       wd[MMA], xd[MMA];

    for (int idx = lane; idx < MMA * MMA; idx += 32) facc[idx / MMA][idx % MMA] = 0.0f;
    __syncwarp();

    wmma::fragment<wmma::matrix_a, 16, 16, 16, signed char, wmma::row_major> a0, a1;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, signed char, wmma::col_major> b0, b1;
    wmma::fragment<wmma::accumulator, 16, 16, 16, int> c;

    for (int b = 0; b < nb; ++b) {
        for (int idx = lane; idx < MMA * QK; idx += 32) {       // gather weight int8 tile
            const int r = idx / QK, k = idx % QK, gm = m0 + r;
            As[r][k] = (gm < M) ? W[static_cast<long long>(gm) * nb + b].qs[k] : (signed char)0;
        }
        for (int idx = lane; idx < MMA * QK; idx += 32) {       // gather activation int8 tile
            const int col = idx / QK, k = idx % QK, gn = n0 + col;
            Bs[col][k] = (gn < T) ? X[static_cast<long long>(gn) * nb + b].qs[k] : (signed char)0;
        }
        if (lane < MMA) {                                       // gather per-block f16 scales
            const int gm = m0 + lane, gn = n0 + lane;
            wd[lane] = (gm < M) ? __half2float(__ushort_as_half(W[static_cast<long long>(gm) * nb + b].d)) : 0.0f;
            xd[lane] = (gn < T) ? __half2float(__ushort_as_half(X[static_cast<long long>(gn) * nb + b].d)) : 0.0f;
        }
        __syncwarp();

        wmma::fill_fragment(c, 0);
        wmma::load_matrix_sync(a0, &As[0][0],  QK);
        wmma::load_matrix_sync(a1, &As[0][16], QK);
        wmma::load_matrix_sync(b0, &Bs[0][0],  QK);
        wmma::load_matrix_sync(b1, &Bs[0][16], QK);
        wmma::mma_sync(c, a0, b0, c);
        wmma::mma_sync(c, a1, b1, c);
        wmma::store_matrix_sync(&Cs[0][0], c, MMA, wmma::mem_row_major);
        __syncwarp();

        for (int idx = lane; idx < MMA * MMA; idx += 32) {      // apply block scales, accumulate
            const int m = idx / MMA, n = idx % MMA;
            facc[m][n] += wd[m] * xd[n] * static_cast<float>(Cs[m][n]);
        }
        __syncwarp();
    }

    for (int idx = lane; idx < MMA * MMA; idx += 32) {
        const int m = idx / MMA, n = idx % MMA, gm = m0 + m, gn = n0 + n;
        if (gm < M && gn < T) Y[static_cast<long long>(gm) * T + gn] = facc[m][n];
    }
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

void launch_matmul_q8_0_mma(const BlockQ8_0* dW, const BlockQ8_0* dXq, float* dY,
                            int M, int K, int T) {
    const dim3 block(32, 1);                      // one warp per 16×16 output tile
    const dim3 grid2(grid(static_cast<std::size_t>(T), MMA),
                     grid(static_cast<std::size_t>(M), MMA));
    matmul_q8_0_mma_kernel<<<grid2, block>>>(dW, dXq, dY, M, K, T);
}

} // namespace sub0llm::backend::cuda::kernels
