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

void launch_matmul_f32(const float* A, const float* B, float* C,
                       std::size_t M, std::size_t N, std::size_t K) {
    const dim3 block(TILE, TILE);
    const dim3 grid2(grid(N, TILE), grid(M, TILE));
    matmul_f32_kernel<<<grid2, block>>>(A, B, C,
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

} // namespace sub0llm::backend::cuda::kernels
