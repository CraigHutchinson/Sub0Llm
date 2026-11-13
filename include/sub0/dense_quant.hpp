// Bounded-memory scalar GGUF projection baseline. GGML stores each output row contiguously.
#pragma once
#include "sub0/gguf.hpp"
#include <algorithm>
#include <limits>

namespace sub0::denseq {
enum class Iq1Kernel { Baseline, Scalar, Native, NativeAll, FusedIQ2, PackedQ8K, PackedQ16 };
// Caller owns weights, inputs, outputs and scratch; scratch must be reserved once before decode.
// x and y must not overlap. This baseline accepts whole quant blocks per row.
[[nodiscard]] inline bool project(std::uint32_t type, std::span<const std::uint8_t> raw,
                                  std::span<const float> x, std::span<float> y,
                                  std::vector<float>& scratch,Iq1Kernel iq1=Iq1Kernel::Baseline) {
    if(iq1==Iq1Kernel::PackedQ8K || iq1==Iq1Kernel::PackedQ16)return false; // requires a prepared activation row via project_q8k
    const auto spec = gguf::block_spec(type);
    const auto decode = gguf::dequant_fn(type);
    if (!decode || spec.elems == 0 || x.empty() || x.size() % spec.elems != 0 ||
        scratch.capacity() < spec.elems) return false;
    const std::size_t blocks = x.size() / spec.elems;
    if (blocks > std::numeric_limits<std::size_t>::max() / spec.bytes) return false;
    const std::size_t row_bytes = blocks * spec.bytes;
    if (y.size() > raw.size() / row_bytes) return false;
    if(type==static_cast<std::uint32_t>(quant::IQ1S::type) && iq1!=Iq1Kernel::Baseline){
        for(std::size_t row=0;row<y.size();++row){
            const auto* packed=raw.data()+row*row_bytes;
            y[row]=(iq1==Iq1Kernel::Native || iq1==Iq1Kernel::NativeAll || iq1==Iq1Kernel::FusedIQ2)?quant::IQ1S::dot_row<true>(packed,x.data(),x.size()):quant::IQ1S::dot_row<false>(packed,x.data(),x.size());
        }
        return true;
    }
    if(type==static_cast<std::uint32_t>(quant::IQ2XXS::type) && iq1==Iq1Kernel::FusedIQ2){
        for(std::size_t row=0;row<y.size();++row)
            y[row]=quant::IQ2XXS::dot_row<true>(raw.data()+row*row_bytes,x.data(),x.size());
        return true;
    }
    for (std::size_t row = 0; row < y.size(); ++row) {
        float sum = 0.f;
#if defined(__AVX2__)
        const bool vector_dot=(iq1==Iq1Kernel::NativeAll || iq1==Iq1Kernel::FusedIQ2) && spec.elems%8==0;
        __m256 accum=_mm256_setzero_ps();
#endif
        for (std::size_t b = 0; b < blocks; ++b) {
            const auto bytes = raw.subspan(row * row_bytes + b * spec.bytes, spec.bytes);
            if (!decode(bytes, spec.elems, scratch)) return false;
#if defined(__AVX2__)
            if(vector_dot){
                for(std::size_t j=0;j<spec.elems;j+=8)
                    accum=_mm256_add_ps(accum,_mm256_mul_ps(_mm256_loadu_ps(x.data()+b*spec.elems+j),_mm256_loadu_ps(scratch.data()+j)));
                continue;
            }
#endif
            for (std::size_t j = 0; j < spec.elems; ++j)
                sum += x[b * spec.elems + j] * scratch[j];
        }
#if defined(__AVX2__)
        if(vector_dot){alignas(32) float lanes[8];_mm256_store_ps(lanes,accum);for(float v:lanes)sum+=v;}
#endif
        y[row] = sum;
    }
    return true;
}

// Batched projection for prompt processing. Computes y[tok*y_stride + row] = dot(row, x[tok*in ...]) for every token and every
// output row, decoding each weight block ONCE and reusing it for all tokens in a tile. Each token keeps its own accumulator and
// adds in the same block and lane order as project(), so every output is bit-identical to a per-token project() call.
// `rows` rows starting at `raw` are produced (callers split rows across workers by passing a sub-span and y offset, as for
// project()). Caller owns all buffers; `scratch` must have capacity for one block. Tile is the token tile (accumulators and the
// tile's activations stay hot); tokens beyond one tile repeat the decode, so keep Tile near the typical prompt chunk.
template<std::size_t Tile = 16>
[[nodiscard]] inline bool project_batch(std::uint32_t type, std::span<const std::uint8_t> raw, const float* x, std::size_t in,
                                        std::size_t n_tokens, float* y, std::size_t rows, std::size_t y_stride,
                                        std::vector<float>& scratch, Iq1Kernel iq1 = Iq1Kernel::Baseline) {
    static_assert(Tile > 0 && Tile <= 64);
    if(iq1==Iq1Kernel::PackedQ8K || iq1==Iq1Kernel::PackedQ16)return false; // requires a prepared activation row via project_q8k
    const auto spec = gguf::block_spec(type);
    const auto decode = gguf::dequant_fn(type);
    if (!decode || spec.elems == 0 || in == 0 || in % spec.elems != 0 || n_tokens == 0 || scratch.capacity() < spec.elems) return false;
    const std::size_t blocks = in / spec.elems;
    if (blocks > std::numeric_limits<std::size_t>::max() / spec.bytes) return false;
    const std::size_t row_bytes = blocks * spec.bytes;
    if (rows > raw.size() / row_bytes || y_stride < rows || !x || !y ||
        n_tokens > std::numeric_limits<std::size_t>::max() / in ||
        (y_stride && n_tokens > std::numeric_limits<std::size_t>::max() / y_stride)) return false;
    const bool iq1_native = type == static_cast<std::uint32_t>(quant::IQ1S::type) && iq1 != Iq1Kernel::Baseline;
#if defined(__AVX2__)
    const bool vector_dot = !iq1_native && (iq1 == Iq1Kernel::NativeAll || iq1 == Iq1Kernel::FusedIQ2) && spec.elems % 8 == 0;
#endif
    for (std::size_t first = 0; first < n_tokens; first += Tile) {
        const std::size_t n = std::min(Tile, n_tokens - first);
        const float* xt = x + first * in;
        for (std::size_t row = 0; row < rows; ++row) {
            const std::uint8_t* packed = raw.data() + row * row_bytes;
            float result[Tile];
            if (iq1_native) {
                if (iq1 == Iq1Kernel::Native || iq1 == Iq1Kernel::NativeAll || iq1 == Iq1Kernel::FusedIQ2) quant::IQ1S::dot_row_batch<true, Tile>(packed, xt, in, in, n, result);
                else quant::IQ1S::dot_row_batch<false, Tile>(packed, xt, in, in, n, result);
            } else if (type == static_cast<std::uint32_t>(quant::IQ2XXS::type) && iq1 == Iq1Kernel::FusedIQ2) {
                // Reuse the qualified core until an IQ2 batch core preserves its exact arithmetic.
                for (std::size_t t = 0; t < n; ++t)
                    result[t] = quant::IQ2XXS::dot_row<true>(packed, xt + t * in, in);
            } else {
                float sum[Tile];
                for (std::size_t t = 0; t < n; ++t) sum[t] = 0.f;
#if defined(__AVX2__)
                __m256 accum[Tile];
                for (std::size_t t = 0; t < n; ++t) accum[t] = _mm256_setzero_ps();
#endif
                for (std::size_t b = 0; b < blocks; ++b) {
                    if (!decode(std::span<const std::uint8_t>(packed + b * spec.bytes, spec.bytes), spec.elems, scratch)) return false;
#if defined(__AVX2__)
                    if (vector_dot) {
                        for (std::size_t t = 0; t < n; ++t)
                            for (std::size_t j = 0; j < spec.elems; j += 8)
                                accum[t] = _mm256_add_ps(accum[t], _mm256_mul_ps(_mm256_loadu_ps(xt + t * in + b * spec.elems + j), _mm256_loadu_ps(scratch.data() + j)));
                        continue;
                    }
#endif
                    for (std::size_t t = 0; t < n; ++t)
                        for (std::size_t j = 0; j < spec.elems; ++j) sum[t] += xt[t * in + b * spec.elems + j] * scratch[j];
                }
#if defined(__AVX2__)
                if (vector_dot)
                    for (std::size_t t = 0; t < n; ++t) { alignas(32) float lanes[8]; _mm256_store_ps(lanes, accum[t]); for (float v : lanes) sum[t] += v; }
#endif
                for (std::size_t t = 0; t < n; ++t) result[t] = sum[t];
            }
            for (std::size_t t = 0; t < n; ++t) y[(first + t) * y_stride + row] = result[t];
        }
    }
    return true;
}
} // namespace sub0::denseq
