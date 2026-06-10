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

} // namespace sub0llm::backend::cuda::kernels
#endif // SUB0LLM_CUDA
