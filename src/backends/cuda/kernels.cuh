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

// Same semantics, int8 tensor cores (WMMA IMMA): one warp per 16×16 tile, one MMA pair per
// Q8 block, per-block f16 scales applied into a float accumulator. Targets the sm_120 ceiling.
void launch_matmul_q8_0_mma(const ::sub0llm::backend::cpu::BlockQ8_0* dW,
                            const ::sub0llm::backend::cpu::BlockQ8_0* dXq,
                            float* dY, int M, int K, int T);

// ── Layer sub-kernels (f32) — mirror the CPU Gemma forward (gemma.cpp) exactly ──────────
// All operate on DEVICE pointers; one launch per call. These are the building blocks of the
// on-device single-layer forward (the "go wide" GPU engine). Each is validated against its
// CPU reference (relRMS) by qbench --layers before being wired into the layer.

// RMSNorm: y = x / sqrt(mean(x²)+eps) * w  (w=nullptr → no learned weight, Gemma V path).
// Single block, blockDim-wide parallel reduction for the sum of squares. n ≤ a few thousand.
void launch_rmsnorm(const float* dx, const float* dw, float* dy, int n, float eps);

// NEOX (half-split) RoPE on one head vector of dim dh, out-of-place. pair i in [0,dh/2):
// theta = pos·base^(-2i/dh) / ff[i] (ff=nullptr → no freq scaling); rotate (x[i], x[i+dh/2]).
void launch_rope_neox(const float* dxin, float* dxout, int dh, int pos, float base,
                      const float* dff);

// GeGLU activation: out = gelu_tanh(gate) ⊙ up, with gelu(x)=x·sigmoid(kK·(x+kC·x³)),
// kK=1.5957691216, kC=0.044715 (matches backend::cpu::gelu_f32). One thread per element.
void launch_geglu(const float* dgate, const float* dup, float* dout, int n);

// Flash-attention DECODE (single query head): o[dh] = Σ_t softmax_t(q·K_t)·V_t over the
// kvlen cached positions, online-softmax, attention scale = 1.0 (Gemma). q,K,V,o device.
// One block per head; K/V are [kvlen][dh] contiguous (caller slices the window). Matches the
// per-head loop in GemmaModel::forward_one.
void launch_flash_attn_decode(const float* dq, const float* dK, const float* dV,
                              float* dout, int dh, int kvlen);

// Quantize `nb` Q8_0 blocks (nb·32 f32 activations) → BlockQ8_0 on device, matching
// backend::cpu::quantize_row_q8_0 (per-32 amax scale d=amax/127, round-to-nearest, f16 d).
// One warp per block. Feeds the device Q8 matmul (which consumes Q8 activations).
void launch_quantize_q8(const float* dx, ::sub0llm::backend::cpu::BlockQ8_0* dy, int nb);

// Fused scaled residual: out[i] = (a[i] + b[i]) * s. One thread per element. Used for Gemma's
// per-layer (cur + attn_out)·layer_output_scale.
void launch_add_scale(const float* da, const float* db, float* dout, float s, int n);

// ── Batched per-head ops (decode) — one launch over all heads, not one launch per head ──────
// The decode forward was launch-overhead-bound (~16 q-heads × rmsnorm+rope etc. = dozens of tiny
// launches/layer). These do the SAME math for `n_heads` contiguous dh-vectors in ONE launch.

// RMSNorm of each of `n_heads` contiguous dh-slices (shared per-head weight w, or nullptr).
// One block per head. In-place safe (y may equal x).
void launch_rmsnorm_heads(const float* x, const float* w, float* y, int n_heads, int dh, float eps);

// NEOX RoPE of each of `n_heads` contiguous dh-vectors at the same `pos`. In-place safe.
void launch_rope_heads(const float* xin, float* xout, int n_heads, int dh, int pos, float base,
                       const float* ff);

// Scatter `n_kv` contiguous dh-vectors of K and V into the KV cache slot for `pos`
// (cache layout [kv_head][max_pos][dh]) in one launch (replaces 2·n_kv D2D copies).
void launch_store_kv(const float* kcur, const float* vcur, float* kcD, float* vcD,
                     int n_kv, int dh, int max_pos, int pos);

// Flash-attention decode for ALL `n_head` query heads in one launch (one block per head).
// Head hd reads kv-head g=hd/group from the cache windowed at [kv_lo, kv_lo+kvlen). scale=1.0.
void launch_flash_attn_decode_heads(const float* q, const float* kcD, const float* vcD, float* out,
                                    int n_head, int dh, int kvlen, int kv_lo, int group, int max_pos);

} // namespace sub0llm::backend::cuda::kernels
#endif // SUB0LLM_CUDA
