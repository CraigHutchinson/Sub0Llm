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

// Row-wise softmax: one block per row, blockDim threads reduce max then Σexp in shared mem,
// matching backend::cpu::softmax_rows_f32 (max-subtract → expf → normalise). blockDim must be
// a power of two for the tree reduction. expf (not __expf) to stay close to the CPU reference.
__global__ void softmax_rows_f32_kernel(const float* __restrict__ in, float* __restrict__ out,
                                        int rows, int cols) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    const float* rin  = in  + static_cast<std::size_t>(row) * cols;
    float*       rout = out + static_cast<std::size_t>(row) * cols;

    extern __shared__ float red[];
    float local = -INFINITY;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) local = fmaxf(local, rin[i]);
    red[threadIdx.x] = local;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s]);
        __syncthreads();
    }
    const float mx = red[0];
    __syncthreads();

    float lsum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        const float e = expf(rin[i] - mx);
        rout[i] = e;
        lsum += e;
    }
    red[threadIdx.x] = lsum;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    const float inv = 1.0f / red[0];
    for (int i = threadIdx.x; i < cols; i += blockDim.x) rout[i] *= inv;
}

// ── rms_norm training kernels (autograd path) ───────────────────────────────────────────
// Distinct from the inference rmsnorm_kernel (single dh-vector, Gemma decode): these process
// T rows of width D and expose the x_norm / inv_rms intermediates the backward pass needs.
// Match backend::cpu::rms_norm_{fwd,bwd_x,bwd_w}_f32 exactly (1/sqrtf, not rsqrtf).

// Forward: one block per row. y = x·inv_rms·w; also writes x_norm = x·inv_rms and inv_rms[t].
__global__ void rms_norm_fwd_kernel(const float* __restrict__ x, const float* __restrict__ w,
                                    float* __restrict__ x_norm, float* __restrict__ inv_rms,
                                    float* __restrict__ out, int T, int D, float eps) {
    const int t = blockIdx.x;
    if (t >= T) return;
    const float* xt  = x      + static_cast<std::size_t>(t) * D;
    float*       xnt = x_norm + static_cast<std::size_t>(t) * D;
    float*       ot  = out    + static_cast<std::size_t>(t) * D;

    extern __shared__ float red[];
    float local = 0.0f;
    for (int j = threadIdx.x; j < D; j += blockDim.x) local += xt[j] * xt[j];
    red[threadIdx.x] = local;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    const float ir = 1.0f / sqrtf(red[0] / static_cast<float>(D) + eps);
    if (threadIdx.x == 0) inv_rms[t] = ir;
    for (int j = threadIdx.x; j < D; j += blockDim.x) {
        const float xn = xt[j] * ir;
        xnt[j] = xn;
        ot[j]  = w[j] * xn;
    }
}

// Backward wrt x: one block per row. sigma = Σ_j w·g·x_norm; gx = ir·w·g − (ir·sigma/D)·x_norm.
__global__ void rms_norm_bwd_x_kernel(const float* __restrict__ g, const float* __restrict__ x_norm,
                                      const float* __restrict__ inv_rms, const float* __restrict__ w,
                                      float* __restrict__ gx, int T, int D) {
    const int t = blockIdx.x;
    if (t >= T) return;
    const float* gt  = g      + static_cast<std::size_t>(t) * D;
    const float* xnt = x_norm + static_cast<std::size_t>(t) * D;
    float*       gxt = gx     + static_cast<std::size_t>(t) * D;
    const float  ir  = inv_rms[t];

    extern __shared__ float red[];
    float local = 0.0f;
    for (int j = threadIdx.x; j < D; j += blockDim.x) local += w[j] * gt[j] * xnt[j];
    red[threadIdx.x] = local;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    const float c2 = ir * red[0] / static_cast<float>(D);
    for (int j = threadIdx.x; j < D; j += blockDim.x)
        gxt[j] = ir * w[j] * gt[j] - c2 * xnt[j];
}

// Backward wrt w: one thread per column j, reduces over the T rows. gw[j] = Σ_t g[t,j]·x_norm[t,j].
// Writes (not +=) — the autograd caller passes a freshly zeroed gw, so this matches the CPU
// accumulate-into-zero result without needing cross-block atomics.
__global__ void rms_norm_bwd_w_kernel(const float* __restrict__ g, const float* __restrict__ x_norm,
                                      float* __restrict__ gw, int T, int D) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= D) return;
    float acc = 0.0f;
    for (int t = 0; t < T; ++t) {
        const std::size_t off = static_cast<std::size_t>(t) * D + j;
        acc += g[off] * x_norm[off];
    }
    gw[j] = acc;
}

// rope (half-split / NeoX) fwd+bwd over x(T,Dh) with precomputed cos/sin (T,Dh/2). One thread per
// rotated pair (t,i), i∈[0,Dh/2). Matches the autograd::rope CPU loops exactly.
//   fwd: out[t,i]=x0·c−x1·s,  out[t,i+D2]=x0·s+x1·c   (x0=x[t,i], x1=x[t,i+D2])
//   bwd: gx[t,i]=g0·c+g1·s,   gx[t,i+D2]=−g0·s+g1·c
__global__ void rope_fwd_kernel(const float* __restrict__ x, const float* __restrict__ cosf,
                                const float* __restrict__ sinf, float* __restrict__ out,
                                int T, int Dh, int D2) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= T * D2) return;
    const int t = idx / D2, i = idx % D2;
    const float c  = cosf[idx], s = sinf[idx];     // cos/sin are (T,D2) row-major → idx == t·D2+i
    const float x0 = x[static_cast<std::size_t>(t) * Dh + i];
    const float x1 = x[static_cast<std::size_t>(t) * Dh + i + D2];
    out[static_cast<std::size_t>(t) * Dh + i]      = x0 * c - x1 * s;
    out[static_cast<std::size_t>(t) * Dh + i + D2] = x0 * s + x1 * c;
}

__global__ void rope_bwd_kernel(const float* __restrict__ g, const float* __restrict__ cosf,
                                const float* __restrict__ sinf, float* __restrict__ gx,
                                int T, int Dh, int D2) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= T * D2) return;
    const int t = idx / D2, i = idx % D2;
    const float c  = cosf[idx], s = sinf[idx];
    const float g0 = g[static_cast<std::size_t>(t) * Dh + i];
    const float g1 = g[static_cast<std::size_t>(t) * Dh + i + D2];
    gx[static_cast<std::size_t>(t) * Dh + i]      =  g0 * c + g1 * s;
    gx[static_cast<std::size_t>(t) * Dh + i + D2] = -g0 * s + g1 * c;
}

// silu fwd/bwd (elementwise). silu(x)=x·σ(x); silu'(x)=σ·(1+x·(1−σ)). expf for σ, matching the
// scalar tail of backend::cpu::silu_{f32,backward_f32} (the SIMD path uses a fast σ approximation).
__global__ void silu_f32_kernel(const float* __restrict__ in, float* __restrict__ out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float x = in[i];
    out[i] = x / (1.0f + expf(-x));
}

__global__ void silu_bwd_f32_kernel(const float* __restrict__ grad_out, const float* __restrict__ x,
                                    float* __restrict__ grad_in, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float sig = 1.0f / (1.0f + expf(-x[i]));
    grad_in[i] = grad_out[i] * sig * (1.0f + x[i] * (1.0f - sig));
}

// Embedding-scatter backward: grad_w[idx[i]] += g_out[i] over N rows of width D. One thread per
// (i,j) element; atomicAdd resolves the rows multiple tokens share. grad_w must be pre-zeroed,
// matching backend::cpu::embed_bwd_f32 (which accumulates into a zeroed buffer).
__global__ void embed_bwd_f32_kernel(const float* __restrict__ g_out, const int* __restrict__ idx,
                                     float* __restrict__ g_w, int N, int D) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= N * D) return;
    const int i = t / D, j = t % D;
    atomicAdd(&g_w[static_cast<std::size_t>(idx[i]) * D + j], g_out[static_cast<std::size_t>(i) * D + j]);
}

// Scalar multiply (elementwise): out[i] = in[i] * alpha. Matches backend::cpu::mul_scalar_f32 —
// the primitive behind autograd::scale (forward y=α·x and backward g↦α·g).
__global__ void mul_scalar_f32_kernel(const float* __restrict__ in, float alpha,
                                      float* __restrict__ out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i] * alpha;
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

// Transposed-first-operand matmul: C = Aᵀ·B. A(M,K) and B(M,N) row-major, C(K,N). Contraction is
// over M (the shared leading dim). out[k,n] = Σ_m A[m,k]·B[m,n] — the weight-gradient kernel,
// matching backend::cpu::matmul_tb_f32. Same 16×16 tiling as matmul_f32_kernel; the only change is
// the left tile reads A transposed (A[m·K + k]).
__global__ void matmul_tb_f32_kernel(const float* A, const float* B, float* C,
                                     int M, int N, int K) {
    __shared__ float sA[TILE][TILE];   // sA[ty][tx] = Aᵀ[k][m] = A[m·K + k]
    __shared__ float sB[TILE][TILE];   // sB[ty][tx] = B[m][n]

    const int k = blockIdx.y * TILE + threadIdx.y;   // output row (0..K)
    const int n = blockIdx.x * TILE + threadIdx.x;   // output col (0..N)

    float acc = 0.0f;
    for (int t = 0; t < (M + TILE - 1) / TILE; ++t) {
        const int mA = t * TILE + threadIdx.x;       // contraction index for the Aᵀ tile
        sA[threadIdx.y][threadIdx.x] = (k < K && mA < M)
            ? A[static_cast<std::size_t>(mA) * K + k] : 0.0f;
        const int mB = t * TILE + threadIdx.y;       // contraction index for the B tile
        sB[threadIdx.y][threadIdx.x] = (n < N && mB < M)
            ? B[static_cast<std::size_t>(mB) * N + n] : 0.0f;
        __syncthreads();

        for (int i = 0; i < TILE; ++i) acc += sA[threadIdx.y][i] * sB[i][threadIdx.x];
        __syncthreads();
    }
    if (k < K && n < N) C[static_cast<std::size_t>(k) * N + n] = acc;
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

// T=1 Q8 GEMV (decode): one WARP per output row, lanes stream the row's nb blocks cooperatively
// (coalesced — adjacent lanes read adjacent blocks), int8 dot via __dp4a, warp-reduce the partials.
// Replaces the one-thread-per-output kernel for T=1, which read each weight row serially (poor
// coalescing, GPU under-utilized) — this saturates VRAM bandwidth, the point of GPU-resident layers.
__global__ void matmul_q8_0_gemv_kernel(const BlockQ8_0* __restrict__ W,
                                        const BlockQ8_0* __restrict__ X,
                                        float* __restrict__ Y, int M, int nb) {
    const int m = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;   // global warp id = output row
    const int lane = threadIdx.x & 31;
    if (m >= M) return;
    const BlockQ8_0* wrow = W + static_cast<long long>(m) * nb;
    float acc = 0.0f;
    for (int b = lane; b < nb; b += 32) {
        const BlockQ8_0& wb = wrow[b];
        const BlockQ8_0& xb = X[b];
        int idot = 0;
        #pragma unroll
        for (int j = 0; j < QK; j += 4) idot = __dp4a(pack4_s8(wb.qs + j), pack4_s8(xb.qs + j), idot);
        acc += __half2float(__ushort_as_half(wb.d)) * __half2float(__ushort_as_half(xb.d))
             * static_cast<float>(idot);
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if (lane == 0) Y[m] = acc;
}

// Aligned T=1 Q8 GEMV (decode): weights repacked to an ALIGNED layout — qs[M*K] contiguous int8
// (16-byte aligned per block) + scales[M*nb] f16 — so the warp reads weights as coalesced int4
// (vs the 34-byte BlockQ8_0 whose qs at +2 forces scattered byte loads ~27-45% of VRAM bw). One
// warp per row; lane streams blocks. Activation X stays BlockQ8_0 (1 row, small). The bandwidth fix.
__global__ void matmul_q8_0_gemv_aligned_kernel(const int8_t* __restrict__ Wqs,
                                                const uint16_t* __restrict__ Wsc,
                                                const BlockQ8_0* __restrict__ X,
                                                float* __restrict__ Y, int M, int K, int nb) {
    const int m = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (m >= M) return;
    const int8_t*  wqs = Wqs + static_cast<long long>(m) * K;
    const uint16_t* wsc = Wsc + static_cast<long long>(m) * nb;
    float acc = 0.0f;
    for (int b = lane; b < nb; b += 32) {
        const int4 a = *reinterpret_cast<const int4*>(wqs + b * QK);          // 16 int8 (aligned)
        const int4 c = *reinterpret_cast<const int4*>(wqs + b * QK + 16);     // next 16 int8
        const int8_t* xq = X[b].qs;
        int idot = 0;
        idot = __dp4a(a.x, pack4_s8(xq + 0),  idot);  idot = __dp4a(a.y, pack4_s8(xq + 4),  idot);
        idot = __dp4a(a.z, pack4_s8(xq + 8),  idot);  idot = __dp4a(a.w, pack4_s8(xq + 12), idot);
        idot = __dp4a(c.x, pack4_s8(xq + 16), idot);  idot = __dp4a(c.y, pack4_s8(xq + 20), idot);
        idot = __dp4a(c.z, pack4_s8(xq + 24), idot);  idot = __dp4a(c.w, pack4_s8(xq + 28), idot);
        acc += __half2float(__ushort_as_half(wsc[b])) * __half2float(__ushort_as_half(X[b].d))
             * static_cast<float>(idot);
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if (lane == 0) Y[m] = acc;
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

// ── Layer sub-kernels (f32) ─────────────────────────────────────────────────────────────
// Mirror the CPU Gemma forward (gemma.cpp): rmsnorm, NEOX RoPE, GeGLU, flash-attn decode.

// RMSNorm in one block: blockDim threads reduce Σx² in shared, then scale. Uses 1/sqrtf (not
// the approximate rsqrtf) to stay close to the CPU's 1.0f/std::sqrt. w=nullptr → no weight.
__global__ void rmsnorm_kernel(const float* __restrict__ x, const float* __restrict__ w,
                               float* __restrict__ y, int n, float eps) {
    extern __shared__ float red[];
    float local = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) local += x[i] * x[i];
    red[threadIdx.x] = local;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    const float inv = 1.0f / sqrtf(red[0] / static_cast<float>(n) + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = x[i] * inv;
        y[i] = w ? v * w[i] : v;
    }
}

// NEOX half-split RoPE on one head vector, out-of-place. One thread per rotated pair.
__global__ void rope_neox_kernel(const float* __restrict__ xin, float* __restrict__ xout,
                                 int dh, int pos, float base, const float* __restrict__ ff) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int half = dh >> 1;
    if (i >= half) return;
    float theta = static_cast<float>(pos) * powf(base, -2.0f * static_cast<float>(i) / static_cast<float>(dh));
    if (ff) theta /= ff[i];
    const float c = cosf(theta), s = sinf(theta);
    const float a = xin[i], b = xin[i + half];
    xout[i]        = a * c - b * s;
    xout[i + half] = a * s + b * c;
}

// GeGLU: out = gelu_tanh(gate) ⊙ up. Same constants as backend::cpu::gelu_f32.
__global__ void geglu_kernel(const float* __restrict__ gate, const float* __restrict__ up,
                             float* __restrict__ out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    constexpr float kK = 1.5957691216f, kC = 0.044715f;
    const float xv = gate[i];
    const float z  = kK * (xv + kC * xv * xv * xv);
    const float g  = xv / (1.0f + expf(-z));
    out[i] = g * up[i];
}

// Flash-attention decode, single query head: online softmax over kvlen cached positions.
// One block; q/acc live in shared; m,l,score scalars in shared. Attention scale = 1.0.
__global__ void flash_attn_decode_kernel(const float* __restrict__ q, const float* __restrict__ K,
                                         const float* __restrict__ V, float* __restrict__ o,
                                         int dh, int kvlen) {
    extern __shared__ float sh[];
    float* sq   = sh;                 // [dh]   query
    float* sacc = sh + dh;            // [dh]   running weighted-V accumulator
    float* sred = sh + 2 * dh;        // [blockDim] dot reduction scratch
    __shared__ float s_m, s_l, s_alpha, s_p;

    for (int i = threadIdx.x; i < dh; i += blockDim.x) { sq[i] = q[i]; sacc[i] = 0.0f; }
    if (threadIdx.x == 0) { s_m = __int_as_float(0xff800000); s_l = 0.0f; }  // -inf (no #221-D remark)
    __syncthreads();

    for (int t = 0; t < kvlen; ++t) {
        const float* Kt = K + static_cast<long long>(t) * dh;
        float local = 0.0f;
        for (int i = threadIdx.x; i < dh; i += blockDim.x) local += sq[i] * Kt[i];
        sred[threadIdx.x] = local;
        __syncthreads();
        for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
            if (threadIdx.x < s) sred[threadIdx.x] += sred[threadIdx.x + s];
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            const float score = sred[0];              // scale = 1.0
            const float m_new = fmaxf(s_m, score);
            s_alpha = expf(s_m - m_new);
            s_p     = expf(score - m_new);
            s_l     = s_l * s_alpha + s_p;
            s_m     = m_new;
        }
        __syncthreads();
        const float* Vt = V + static_cast<long long>(t) * dh;
        const float alpha = s_alpha, p = s_p;
        for (int i = threadIdx.x; i < dh; i += blockDim.x) sacc[i] = sacc[i] * alpha + p * Vt[i];
        __syncthreads();
    }
    const float invl = 1.0f / s_l;
    for (int i = threadIdx.x; i < dh; i += blockDim.x) o[i] = sacc[i] * invl;
}

// Quantize f32 → Q8_0, one warp per 32-element block: lane j owns element j; warp-reduce the
// abs-max, d = amax/127, qs[j] = round-to-nearest(x[j]/d), store d as f16. Mirrors quantize_row_q8_0.
__global__ void quantize_q8_kernel(const float* __restrict__ x, BlockQ8_0* __restrict__ y, int nb) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (warp >= nb) return;
    const float v = x[static_cast<long long>(warp) * QK + lane];
    float a = fabsf(v);
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
    const float d  = a / 127.0f;
    const float id = d > 0.0f ? 1.0f / d : 0.0f;
    y[warp].qs[lane] = static_cast<signed char>(__float2int_rn(v * id));
    if (lane == 0) y[warp].d = __half_as_ushort(__float2half(d));
}

// Fused scaled residual: out = (a + b) * s.
__global__ void add_scale_kernel(const float* __restrict__ a, const float* __restrict__ b,
                                 float* __restrict__ out, float s, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = (a[i] + b[i]) * s;
}

// ── Batched per-head decode kernels (one launch over all heads) ─────────────────────────────
// RMSNorm of head blockIdx.x (dh-slice). Identical math to rmsnorm_kernel, offset per head.
__global__ void rmsnorm_heads_kernel(const float* __restrict__ x, const float* __restrict__ w,
                                     float* __restrict__ y, int dh, float eps) {
    const long long off = static_cast<long long>(blockIdx.x) * dh;
    const float* xh = x + off;  float* yh = y + off;
    extern __shared__ float red[];
    float local = 0.0f;
    for (int i = threadIdx.x; i < dh; i += blockDim.x) local += xh[i] * xh[i];
    red[threadIdx.x] = local;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    const float inv = 1.0f / sqrtf(red[0] / static_cast<float>(dh) + eps);
    for (int i = threadIdx.x; i < dh; i += blockDim.x) {
        const float v = xh[i] * inv;
        yh[i] = w ? v * w[i] : v;
    }
}

// NEOX RoPE for n_heads contiguous dh-vectors; one thread per (head, pair).
__global__ void rope_heads_kernel(const float* __restrict__ xin, float* __restrict__ xout,
                                  int n_heads, int dh, int pos, float base, const float* __restrict__ ff) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int half = dh >> 1;
    if (idx >= n_heads * half) return;
    const int h = idx / half, i = idx % half;
    const long long off = static_cast<long long>(h) * dh;
    float theta = static_cast<float>(pos) * powf(base, -2.0f * static_cast<float>(i) / static_cast<float>(dh));
    if (ff) theta /= ff[i];
    const float c = cosf(theta), s = sinf(theta);
    const float a = xin[off + i], b = xin[off + i + half];
    xout[off + i]        = a * c - b * s;
    xout[off + i + half] = a * s + b * c;
}

// Scatter n_kv contiguous K/V head-vectors into the cache slot for `pos` ([kv_head][max_pos][dh]).
__global__ void store_kv_kernel(const float* __restrict__ kcur, const float* __restrict__ vcur,
                                float* __restrict__ kcD, float* __restrict__ vcD,
                                int n_kv, int dh, int max_pos, int pos) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_kv * dh) return;
    const int g = idx / dh, d = idx % dh;
    const long long dst = (static_cast<long long>(g) * max_pos + pos) * dh + d;
    kcD[dst] = kcur[static_cast<long long>(g) * dh + d];
    vcD[dst] = vcur[static_cast<long long>(g) * dh + d];
}

// Flash decode for query head blockIdx.x. Same online softmax as flash_attn_decode_kernel.
__global__ void flash_attn_decode_heads_kernel(const float* __restrict__ q, const float* __restrict__ kcD,
                                               const float* __restrict__ vcD, float* __restrict__ out,
                                               int dh, int kvlen, int kv_lo, int group, int max_pos) {
    const int hd = blockIdx.x, g = hd / group;
    const float* qh = q + static_cast<long long>(hd) * dh;
    const float* K  = kcD + (static_cast<long long>(g) * max_pos + kv_lo) * dh;
    const float* V  = vcD + (static_cast<long long>(g) * max_pos + kv_lo) * dh;
    float* oh = out + static_cast<long long>(hd) * dh;

    extern __shared__ float sh[];
    float* sq = sh;  float* sacc = sh + dh;  float* sred = sh + 2 * dh;
    __shared__ float s_m, s_l, s_alpha, s_p;
    for (int i = threadIdx.x; i < dh; i += blockDim.x) { sq[i] = qh[i]; sacc[i] = 0.0f; }
    if (threadIdx.x == 0) { s_m = __int_as_float(0xff800000); s_l = 0.0f; }
    __syncthreads();
    for (int t = 0; t < kvlen; ++t) {
        const float* Kt = K + static_cast<long long>(t) * dh;
        float local = 0.0f;
        for (int i = threadIdx.x; i < dh; i += blockDim.x) local += sq[i] * Kt[i];
        sred[threadIdx.x] = local;
        __syncthreads();
        for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
            if (threadIdx.x < s) sred[threadIdx.x] += sred[threadIdx.x + s];
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            const float score = sred[0];
            const float m_new = fmaxf(s_m, score);
            s_alpha = expf(s_m - m_new);  s_p = expf(score - m_new);
            s_l = s_l * s_alpha + s_p;    s_m = m_new;
        }
        __syncthreads();
        const float* Vt = V + static_cast<long long>(t) * dh;
        const float alpha = s_alpha, p = s_p;
        for (int i = threadIdx.x; i < dh; i += blockDim.x) sacc[i] = sacc[i] * alpha + p * Vt[i];
        __syncthreads();
    }
    const float invl = 1.0f / s_l;
    for (int i = threadIdx.x; i < dh; i += blockDim.x) oh[i] = sacc[i] * invl;
}

// Batched CAUSAL attention for PREFILL: one block per (token t, head h) = grid T·nH. Query (t,h)
// attends cache positions [kv_lo(p), p], p=start_pos+t (causal + sliding window). Same online
// softmax as the decode kernel; the only change is the grid carries the token index and kvlen=p-kv_lo+1.
// Q is [T, nH, dh] (per-token-contiguous, row (t·nH+h)); K/V cache [kv_head][max_pos][dh] filled for the
// T prefill positions before this launch. Matches forward_prefill's per-(token,head) attention.
__global__ void flash_attn_prefill_heads_kernel(const float* __restrict__ Q, const float* __restrict__ kcD,
                                                const float* __restrict__ vcD, float* __restrict__ out,
                                                int nH, int dh, int window, int group, int max_pos,
                                                int start_pos) {
    const int idx = blockIdx.x;                 // t*nH + h
    const int t = idx / nH, h = idx % nH, g = h / group;
    const int p = start_pos + t;
    const int kv_lo = window > 0 ? max(0, p - window + 1) : 0;
    const int kvlen = p - kv_lo + 1;
    const float* qh = Q + static_cast<long long>(idx) * dh;
    const float* K  = kcD + (static_cast<long long>(g) * max_pos + kv_lo) * dh;
    const float* V  = vcD + (static_cast<long long>(g) * max_pos + kv_lo) * dh;
    float* oh = out + static_cast<long long>(idx) * dh;

    extern __shared__ float sh[];
    float* sq = sh;  float* sacc = sh + dh;  float* sred = sh + 2 * dh;
    __shared__ float s_m, s_l, s_alpha, s_p;
    for (int i = threadIdx.x; i < dh; i += blockDim.x) { sq[i] = qh[i]; sacc[i] = 0.0f; }
    if (threadIdx.x == 0) { s_m = __int_as_float(0xff800000); s_l = 0.0f; }
    __syncthreads();
    for (int s = 0; s < kvlen; ++s) {
        const float* Ks = K + static_cast<long long>(s) * dh;
        float local = 0.0f;
        for (int i = threadIdx.x; i < dh; i += blockDim.x) local += sq[i] * Ks[i];
        sred[threadIdx.x] = local;
        __syncthreads();
        for (int r = blockDim.x >> 1; r > 0; r >>= 1) {
            if (threadIdx.x < r) sred[threadIdx.x] += sred[threadIdx.x + r];
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            const float score = sred[0];
            const float m_new = fmaxf(s_m, score);
            s_alpha = expf(s_m - m_new);  s_p = expf(score - m_new);
            s_l = s_l * s_alpha + s_p;    s_m = m_new;
        }
        __syncthreads();
        const float* Vs = V + static_cast<long long>(s) * dh;
        const float alpha = s_alpha, pp = s_p;
        for (int i = threadIdx.x; i < dh; i += blockDim.x) sacc[i] = sacc[i] * alpha + pp * Vs[i];
        __syncthreads();
    }
    const float invl = 1.0f / s_l;
    for (int i = threadIdx.x; i < dh; i += blockDim.x) oh[i] = sacc[i] * invl;
}

// ── batched-prefill primitives (T>1) ─────────────────────────────────────────────────────────
// Coalesced tiled transpose: in (M,T) row-major → out (T,M) row-major. out[t*M+m]=in[m*T+t].
// The IMMA GEMM emits (out_features, T); per-token rope/norm needs (T, out_features).
__global__ void transpose_kernel(const float* __restrict__ in, float* __restrict__ out, int M, int T) {
    __shared__ float tile[32][33];                  // +1 pad avoids shared bank conflicts
    const int m = blockIdx.y * 32 + threadIdx.y, t = blockIdx.x * 32 + threadIdx.x;
    if (m < M && t < T) tile[threadIdx.y][threadIdx.x] = in[static_cast<long long>(m) * T + t];
    __syncthreads();
    const int ot = blockIdx.x * 32 + threadIdx.y, om = blockIdx.y * 32 + threadIdx.x;
    if (ot < T && om < M) out[static_cast<long long>(ot) * M + om] = tile[threadIdx.x][threadIdx.y];
}

// NEOX RoPE for prefill: per (token t, head h) — pos = start_pos + t (varies per token). One thread
// per (t,h,pair). x is [T, nH, dh] (row t*nH+h). In-place.
__global__ void rope_prefill_kernel(float* __restrict__ x, int T, int nH, int dh, int start_pos,
                                    float base, const float* __restrict__ ff) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int half = dh >> 1;
    if (idx >= T * nH * half) return;
    const int pair = idx % half, hd = idx / half, t = hd / nH;
    const int p = start_pos + t;
    float* v = x + static_cast<long long>(hd) * dh;
    float theta = static_cast<float>(p) * powf(base, -2.0f * static_cast<float>(pair) / static_cast<float>(dh));
    if (ff) theta /= ff[pair];
    const float c = cosf(theta), s = sinf(theta), a = v[pair], b = v[pair + half];
    v[pair] = a * c - b * s;  v[pair + half] = a * s + b * c;
}

// Scatter T tokens' K/V (each [T, nKV, dh], row t*nKV+g) into the cache [kv_head][max_pos][dh] at
// positions start_pos..start_pos+T-1. One thread per (t, g, d).
__global__ void store_kv_prefill_kernel(const float* __restrict__ K, const float* __restrict__ V,
                                        float* __restrict__ kcD, float* __restrict__ vcD,
                                        int T, int nKV, int dh, int max_pos, int start_pos) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= T * nKV * dh) return;
    const int d = idx % dh, tg = idx / dh, g = tg % nKV, t = tg / nKV;
    const long long src = (static_cast<long long>(t) * nKV + g) * dh + d;
    const long long dst = (static_cast<long long>(g) * max_pos + start_pos + t) * dh + d;
    kcD[dst] = K[src];  vcD[dst] = V[src];
}

// ── q8 KV cache variants ────────────────────────────────────────────────────────────────────
// Quantize K/V head-vectors and scatter into the Q8 KV cache slot for `pos`. One warp per block;
// warp w covers a (head, block) of K (w < nKV·nbh) or V (w >= nKV·nbh). Mirrors quantize_row_q8_0.
__global__ void store_kv_q8_kernel(const float* __restrict__ kcur, const float* __restrict__ vcur,
                                   BlockQ8_0* __restrict__ kcD, BlockQ8_0* __restrict__ vcD,
                                   int nKV, int dh, int max_pos, int pos) {
    const int nbh = dh / QK;
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (warp >= nKV * nbh * 2) return;
    const bool isV = warp >= nKV * nbh;
    const int w = isV ? warp - nKV * nbh : warp;
    const int g = w / nbh, blk = w % nbh;
    const float* src = (isV ? vcur : kcur) + static_cast<long long>(g) * dh + blk * QK;
    BlockQ8_0* dst = (isV ? vcD : kcD) + (static_cast<long long>(g) * max_pos + pos) * nbh + blk;
    const float v = src[lane];
    float a = fabsf(v);
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
    const float d = a / 127.0f, id = d > 0.0f ? 1.0f / d : 0.0f;
    dst->qs[lane] = static_cast<signed char>(__float2int_rn(v * id));
    if (lane == 0) dst->d = __half_as_ushort(__float2half(d));
}

// Flash decode reading a Q8 KV cache: dequantize K/V per element via the block's f16 scale.
__global__ void flash_attn_decode_heads_q8_kernel(const float* __restrict__ q,
                                                  const BlockQ8_0* __restrict__ kcD,
                                                  const BlockQ8_0* __restrict__ vcD,
                                                  float* __restrict__ out, int dh, int kvlen,
                                                  int kv_lo, int group, int max_pos) {
    const int hd = blockIdx.x, g = hd / group, nbh = dh / QK;
    const float* qh = q + static_cast<long long>(hd) * dh;
    const BlockQ8_0* K = kcD + (static_cast<long long>(g) * max_pos + kv_lo) * nbh;
    const BlockQ8_0* V = vcD + (static_cast<long long>(g) * max_pos + kv_lo) * nbh;
    float* oh = out + static_cast<long long>(hd) * dh;

    extern __shared__ float sh[];
    float* sq = sh;  float* sacc = sh + dh;  float* sred = sh + 2 * dh;
    __shared__ float s_m, s_l, s_alpha, s_p;
    for (int i = threadIdx.x; i < dh; i += blockDim.x) { sq[i] = qh[i]; sacc[i] = 0.0f; }
    if (threadIdx.x == 0) { s_m = __int_as_float(0xff800000); s_l = 0.0f; }
    __syncthreads();
    for (int t = 0; t < kvlen; ++t) {
        const BlockQ8_0* Kt = K + static_cast<long long>(t) * nbh;
        float local = 0.0f;
        for (int i = threadIdx.x; i < dh; i += blockDim.x) {
            const BlockQ8_0& b = Kt[i >> 5];
            local += sq[i] * __half2float(__ushort_as_half(b.d)) * static_cast<float>(b.qs[i & 31]);
        }
        sred[threadIdx.x] = local;
        __syncthreads();
        for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
            if (threadIdx.x < s) sred[threadIdx.x] += sred[threadIdx.x + s];
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            const float score = sred[0];
            const float m_new = fmaxf(s_m, score);
            s_alpha = expf(s_m - m_new);  s_p = expf(score - m_new);
            s_l = s_l * s_alpha + s_p;    s_m = m_new;
        }
        __syncthreads();
        const BlockQ8_0* Vt = V + static_cast<long long>(t) * nbh;
        const float alpha = s_alpha, p = s_p;
        for (int i = threadIdx.x; i < dh; i += blockDim.x) {
            const BlockQ8_0& b = Vt[i >> 5];
            sacc[i] = sacc[i] * alpha + p * __half2float(__ushort_as_half(b.d)) * static_cast<float>(b.qs[i & 31]);
        }
        __syncthreads();
    }
    const float invl = 1.0f / s_l;
    for (int i = threadIdx.x; i < dh; i += blockDim.x) oh[i] = sacc[i] * invl;
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

void launch_softmax_rows_f32(const float* in, float* out, int rows, int cols) {
    constexpr int B = 256;                        // power of two for the tree reduction
    softmax_rows_f32_kernel<<<rows, B, B * sizeof(float)>>>(in, out, rows, cols);
}

void launch_rms_norm_fwd(const float* x, const float* w, float* x_norm, float* inv_rms,
                         float* out, int T, int D, float eps) {
    constexpr int B = 256;                        // power of two for the tree reduction
    rms_norm_fwd_kernel<<<T, B, B * sizeof(float)>>>(x, w, x_norm, inv_rms, out, T, D, eps);
}

void launch_rms_norm_bwd_x(const float* g, const float* x_norm, const float* inv_rms,
                           const float* w, float* gx, int T, int D) {
    constexpr int B = 256;
    rms_norm_bwd_x_kernel<<<T, B, B * sizeof(float)>>>(g, x_norm, inv_rms, w, gx, T, D);
}

void launch_rms_norm_bwd_w(const float* g, const float* x_norm, float* gw, int T, int D) {
    constexpr int B = 256;
    rms_norm_bwd_w_kernel<<<grid(static_cast<std::size_t>(D), B), B>>>(g, x_norm, gw, T, D);
}

void launch_rope_fwd(const float* x, const float* cosf, const float* sinf, float* out,
                     int T, int Dh) {
    const int D2 = Dh / 2;
    rope_fwd_kernel<<<grid(static_cast<std::size_t>(T) * D2, BLOCK), BLOCK>>>(
        x, cosf, sinf, out, T, Dh, D2);
}

void launch_rope_bwd(const float* g, const float* cosf, const float* sinf, float* gx,
                     int T, int Dh) {
    const int D2 = Dh / 2;
    rope_bwd_kernel<<<grid(static_cast<std::size_t>(T) * D2, BLOCK), BLOCK>>>(
        g, cosf, sinf, gx, T, Dh, D2);
}

void launch_silu_f32(const float* in, float* out, std::size_t n) {
    silu_f32_kernel<<<grid(n, BLOCK), BLOCK>>>(in, out, static_cast<int>(n));
}

void launch_silu_bwd_f32(const float* grad_out, const float* x, float* grad_in, std::size_t n) {
    silu_bwd_f32_kernel<<<grid(n, BLOCK), BLOCK>>>(grad_out, x, grad_in, static_cast<int>(n));
}

void launch_embed_bwd_f32(const float* g_out, const int* idx, float* g_w, int N, int D) {
    embed_bwd_f32_kernel<<<grid(static_cast<std::size_t>(N) * D, BLOCK), BLOCK>>>(g_out, idx, g_w, N, D);
}

void launch_mul_scalar_f32(const float* in, float alpha, float* out, std::size_t n) {
    mul_scalar_f32_kernel<<<grid(n, BLOCK), BLOCK>>>(in, alpha, out, static_cast<int>(n));
}

void launch_matmul_f32(const float* A, const float* B, float* C,
                       std::size_t M, std::size_t N, std::size_t K) {
    const dim3 block(TILE, TILE);
    const dim3 grid2(grid(N, TILE), grid(M, TILE));
    matmul_f32_kernel<<<grid2, block>>>(A, B, C,
        static_cast<int>(M), static_cast<int>(N), static_cast<int>(K));
}

void launch_matmul_tb_f32(const float* A, const float* B, float* C,
                          std::size_t M, std::size_t N, std::size_t K) {
    const dim3 block(TILE, TILE);
    const dim3 grid2(grid(N, TILE), grid(K, TILE));   // output is (K, N)
    matmul_tb_f32_kernel<<<grid2, block>>>(A, B, C,
        static_cast<int>(M), static_cast<int>(N), static_cast<int>(K));
}

void launch_matmul_q8_0(const BlockQ8_0* dW, const BlockQ8_0* dXq, float* dY,
                        int M, int K, int T) {
    if (T == 1) {                                  // decode GEMV: one warp per output row
        constexpr int B = 128;                     // 4 warps/block → 4 rows per block
        const int blocks = grid(static_cast<std::size_t>(M) * 32, B);
        matmul_q8_0_gemv_kernel<<<blocks, B>>>(dW, dXq, dY, M, K / QK);
        return;
    }
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

void launch_rmsnorm(const float* dx, const float* dw, float* dy, int n, float eps) {
    constexpr int B = 256;                        // power of two for the tree reduction
    rmsnorm_kernel<<<1, B, B * sizeof(float)>>>(dx, dw, dy, n, eps);
}

void launch_rope_neox(const float* dxin, float* dxout, int dh, int pos, float base,
                      const float* dff) {
    const int half = dh / 2;
    rope_neox_kernel<<<grid(static_cast<std::size_t>(half), BLOCK), BLOCK>>>(
        dxin, dxout, dh, pos, base, dff);
}

void launch_geglu(const float* dgate, const float* dup, float* dout, int n) {
    geglu_kernel<<<grid(static_cast<std::size_t>(n), BLOCK), BLOCK>>>(dgate, dup, dout, n);
}

void launch_flash_attn_decode(const float* dq, const float* dK, const float* dV,
                              float* dout, int dh, int kvlen) {
    constexpr int B = 128;                        // power of two for the per-position reduction
    const std::size_t shmem = (static_cast<std::size_t>(2 * dh) + B) * sizeof(float);
    flash_attn_decode_kernel<<<1, B, shmem>>>(dq, dK, dV, dout, dh, kvlen);
}

void launch_quantize_q8(const float* dx, BlockQ8_0* dy, int nb) {
    constexpr int B = 256;                        // 8 warps/block → 8 Q8 blocks per CUDA block
    const int blocks = grid(static_cast<std::size_t>(nb) * 32, B);
    quantize_q8_kernel<<<blocks, B>>>(dx, dy, nb);
}

void launch_add_scale(const float* da, const float* db, float* dout, float s, int n) {
    add_scale_kernel<<<grid(static_cast<std::size_t>(n), BLOCK), BLOCK>>>(da, db, dout, s, n);
}

void launch_rmsnorm_heads(const float* x, const float* w, float* y, int n_heads, int dh, float eps) {
    constexpr int B = 256;
    rmsnorm_heads_kernel<<<n_heads, B, B * sizeof(float)>>>(x, w, y, dh, eps);
}

void launch_rope_heads(const float* xin, float* xout, int n_heads, int dh, int pos, float base,
                       const float* ff) {
    const std::size_t total = static_cast<std::size_t>(n_heads) * (dh / 2);
    rope_heads_kernel<<<grid(total, BLOCK), BLOCK>>>(xin, xout, n_heads, dh, pos, base, ff);
}

void launch_store_kv(const float* kcur, const float* vcur, float* kcD, float* vcD,
                     int n_kv, int dh, int max_pos, int pos) {
    const std::size_t total = static_cast<std::size_t>(n_kv) * dh;
    store_kv_kernel<<<grid(total, BLOCK), BLOCK>>>(kcur, vcur, kcD, vcD, n_kv, dh, max_pos, pos);
}

void launch_flash_attn_decode_heads(const float* q, const float* kcD, const float* vcD, float* out,
                                    int n_head, int dh, int kvlen, int kv_lo, int group, int max_pos) {
    constexpr int B = 128;
    const std::size_t shmem = (static_cast<std::size_t>(2 * dh) + B) * sizeof(float);
    flash_attn_decode_heads_kernel<<<n_head, B, shmem>>>(q, kcD, vcD, out, dh, kvlen, kv_lo, group, max_pos);
}

void launch_store_kv_q8(const float* kcur, const float* vcur, BlockQ8_0* kcD, BlockQ8_0* vcD,
                        int n_kv, int dh, int max_pos, int pos) {
    constexpr int B = 256;                                  // 8 warps/block
    const std::size_t total = static_cast<std::size_t>(n_kv) * (dh / QK) * 2 * 32;
    store_kv_q8_kernel<<<grid(total, B), B>>>(kcur, vcur, kcD, vcD, n_kv, dh, max_pos, pos);
}

void launch_flash_attn_decode_heads_q8(const float* q, const BlockQ8_0* kcD, const BlockQ8_0* vcD,
                                       float* out, int n_head, int dh, int kvlen, int kv_lo,
                                       int group, int max_pos) {
    constexpr int B = 128;
    const std::size_t shmem = (static_cast<std::size_t>(2 * dh) + B) * sizeof(float);
    flash_attn_decode_heads_q8_kernel<<<n_head, B, shmem>>>(q, kcD, vcD, out, dh, kvlen, kv_lo, group, max_pos);
}

void launch_transpose(const float* in, float* out, int M, int T) {
    const dim3 block(32, 32);
    const dim3 g(grid(static_cast<std::size_t>(T), 32), grid(static_cast<std::size_t>(M), 32));
    transpose_kernel<<<g, block>>>(in, out, M, T);
}

void launch_rope_prefill(float* x, int T, int nH, int dh, int start_pos, float base, const float* ff) {
    const std::size_t total = static_cast<std::size_t>(T) * nH * (dh / 2);
    rope_prefill_kernel<<<grid(total, BLOCK), BLOCK>>>(x, T, nH, dh, start_pos, base, ff);
}

void launch_store_kv_prefill(const float* K, const float* V, float* kcD, float* vcD,
                             int T, int nKV, int dh, int max_pos, int start_pos) {
    const std::size_t total = static_cast<std::size_t>(T) * nKV * dh;
    store_kv_prefill_kernel<<<grid(total, BLOCK), BLOCK>>>(K, V, kcD, vcD, T, nKV, dh, max_pos, start_pos);
}

void launch_flash_attn_prefill_heads(const float* Q, const float* kcD, const float* vcD, float* out,
                                     int T, int nH, int dh, int window, int group, int max_pos,
                                     int start_pos) {
    constexpr int B = 128;
    const std::size_t shmem = (static_cast<std::size_t>(2 * dh) + B) * sizeof(float);
    flash_attn_prefill_heads_kernel<<<T * nH, B, shmem>>>(Q, kcD, vcD, out, nH, dh, window, group,
                                                          max_pos, start_pos);
}

void launch_matmul_q8_0_gemv_aligned(const int8_t* Wqs, const uint16_t* Wsc, const BlockQ8_0* X,
                                     float* Y, int M, int K) {
    constexpr int B = 128;                          // 4 warps/block → 4 rows
    const int blocks = grid(static_cast<std::size_t>(M) * 32, B);
    matmul_q8_0_gemv_aligned_kernel<<<blocks, B>>>(Wqs, Wsc, X, Y, M, K, K / QK);
}

} // namespace sub0llm::backend::cuda::kernels
