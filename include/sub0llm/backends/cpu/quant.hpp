#pragma once

// quant.hpp (Ch27) — Q8_0 weight storage + two matmul strategies for keeping
// quantized weights resident (≈4× less RAM than dequantizing to f32).
//
// The question this module exists to ANSWER BY MEASUREMENT (not assumption):
// given Q8_0 weights, is it faster to
//   (A) dequantize-on-the-fly  — expand each weight row to f32 in registers,
//       then do an f32·f32 dot against the f32 activation, or
//   (B) quantized matmul       — quantize the activation to Q8_0 once, then do
//       int8·int8 block dots scaled by the per-block f16 scales.
// (A) reuses the f32 ALU and adds a dequant cost; (B) uses int8 SIMD (maddubs/
// VNNI) at higher throughput but pays to quantize the activation. Which wins
// depends on shape and CPU — so `sub0llm-qbench` profiles both per model.

#include <cstddef>
#include <cstdint>

namespace sub0llm::backend::cpu {

// GGUF Q8_0 block: 32 int8 quants sharing one f16 scale → 34 bytes/32 weights
// = 1.0625 bytes/weight (vs 4 for f32).
inline constexpr int QK8_0 = 32;

struct BlockQ8_0 {
    uint16_t d;            // f16 scale
    int8_t   qs[QK8_0];    // quantized weights
};
static_assert(sizeof(BlockQ8_0) == 34, "Q8_0 block must be 34 bytes (matches GGUF)");

// f16 <-> f32 scalar helpers (used for the per-block scales).
[[nodiscard]] float    f16_to_f32(uint16_t h) noexcept;
[[nodiscard]] uint16_t f32_to_f16(float f) noexcept;

// ── f32 reduction / BLAS-ish primitives ────────────────────────────────────────
// SIMD helpers for the per-token "glue" between GEMVs (RMSNorm, attention, argmax) that
// the compiler can't auto-vectorize: a float-add reduction needs reassociation, which is
// forbidden without -ffast-math. These do an explicit 8-wide reduction + scalar remainder
// (sizes are runtime — head_dim 256/512, D 3840, V 262144 — so the bulk is SIMD, the tail
// scalar). AVX2 instructions run on AVX-512 hosts too, so one path covers both.
[[nodiscard]] float sum_squares_f32(const float* x, int64_t n) noexcept;                  // Σ x²
[[nodiscard]] float dot_f32(const float* a, const float* b, int64_t n) noexcept;          // Σ aᵢbᵢ
void axpy_f32(float alpha, const float* x, float* y, int64_t n) noexcept;                  // y += α·x
// Index of the first maximum (matches std::max_element's first-occurrence semantics).
[[nodiscard]] int64_t argmax_f32(const float* x, int64_t n) noexcept;

// Quantize / dequantize a row of `k` floats (k must be a multiple of QK8_0).
void quantize_row_q8_0  (const float* x, BlockQ8_0* y, int64_t k) noexcept;
void dequantize_row_q8_0(const BlockQ8_0* x, float* y, int64_t k) noexcept;

// Dot of one Q8_0 weight row with an f32 activation — DEQUANT-ON-THE-FLY (A).
[[nodiscard]] float dot_q8_0_f32(const BlockQ8_0* w, const float* x, int64_t k) noexcept;

// (A') dequant-on-the-fly to an f16 intermediate, then f32 dot. Completes the
// dequant-target grid; for a GEMV this is ≤ (A) (no reuse to amortize).
[[nodiscard]] float dot_q8_0_f16(const BlockQ8_0* w, const float* x, int64_t k) noexcept;
void matvec_q8_0_f16(const BlockQ8_0* W, const float* x, float* y,
                     int64_t M, int64_t K) noexcept;

// Dot of two Q8_0 rows (activation pre-quantized) — QUANTIZED int8 dot (B).
// nblocks = k / QK8_0.
[[nodiscard]] float dot_q8_0_q8_0(const BlockQ8_0* a, const BlockQ8_0* b, int64_t nblocks) noexcept;

// ── f16 weight storage (2 B/weight) ─────────────────────────────────────────────
//
// The middle ground: half the RAM of f32, no per-block scales, and one F16C
// instruction converts 8 weights to f32. Near-lossless for f16/bf16-native models
// (vs Q8's ~0.4% error); 2× the RAM of Q8. Profiled alongside the Q8 paths.
void  quantize_row_f16(const float* x, uint16_t* y, int64_t k) noexcept;
[[nodiscard]] float dot_f16_f32(const uint16_t* w, const float* x, int64_t k) noexcept;
void  matvec_f16_f32(const uint16_t* W, const float* x, float* y,
                     int64_t M, int64_t K) noexcept;

// GEMV y[M] = W[M,K]·x[K], W stored as Q8_0 (row-major, K a multiple of QK8_0).
//   _f32  : dequant-on-the-fly (A) — x stays f32.
//   _q8_0 : quantized (B) — quantizes x into the caller-provided scratch
//           (xq must hold K/QK8_0 BlockQ8_0), then int8 dots.
void matvec_q8_0_f32 (const BlockQ8_0* W, const float* x, float* y,
                      int64_t M, int64_t K) noexcept;
void matvec_q8_0_q8_0(const BlockQ8_0* W, const float* x, float* y,
                      int64_t M, int64_t K, BlockQ8_0* xq) noexcept;

// Batched (prompt processing) Q8 GEMM: Y[M,T] = W[M,K]·Xᵀ, W and the T activation
// columns both Q8. Iterates rows outer / columns inner so each weight row is loaded
// ONCE and reused across all T columns (cache locality) — W's memory traffic is M·K,
// not T·M·K as it would be doing T independent GEMVs. Xq is (T, K/QK8_0) row-major,
// Y is (M, T) row-major. This is where tiling pays off (T>1); for T=1 it's a GEMV.
void matmul_q8_0_q8_0(const BlockQ8_0* W, const BlockQ8_0* Xq, float* Y,
                      int64_t M, int64_t K, int64_t T) noexcept;

// ONE weight row (nb = K/QK8_0 blocks) · T activation columns (Xq is T×nb row-major,
// each column contiguous), writing out[0..T). Register-blocks 4 columns at a time so each
// weight block is loaded, |w|-folded, and f16-scale-decoded ONCE and reused across the 4
// columns (vs a per-column GEMV that redoes that weight-side work T times) — the
// compute-bound prefill-GEMM win. Callers parallelize over the M weight rows.
void gemm_row_q8_0(const BlockQ8_0* w, const BlockQ8_0* Xq, float* out,
                   int64_t nb, int64_t T) noexcept;

// As above, but with the T·nb activation block scales pre-decoded (xsd[col*nb + i] =
// f16_fast(Xq[col*nb + i].d), via decode_q8_block_scales). The per-(column,block) f16→f32
// decode is otherwise redone for every one of the M weight rows; hoisting it out — decode
// once per matmul, reuse across rows — removes that redundant work from the inner loop.
// Bitwise-identical to the inline-decode overload (same scale values).
void gemm_row_q8_0(const BlockQ8_0* w, const BlockQ8_0* Xq, const float* xsd,
                   float* out, int64_t nb, int64_t T) noexcept;

// Decode `count` Q8 block scales (f16 → f32) into `scales`. Call once per prefill matmul to
// feed the hoisted gemm_row_q8_0; `count` is T·(K/QK8_0) for a T-column activation tile.
void decode_q8_block_scales(const BlockQ8_0* X, float* scales, int64_t count) noexcept;

} // namespace sub0llm::backend::cpu
