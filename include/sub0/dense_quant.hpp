// Bounded-memory scalar GGUF projection baseline. GGML stores each output row contiguously.
#pragma once
#include "sub0/gguf.hpp"
#include <limits>

namespace sub0::denseq {
enum class Iq1Kernel { Baseline, Scalar, Native, NativeAll };
// Caller owns weights, inputs, outputs and scratch; scratch must be reserved once before decode.
// x and y must not overlap. This baseline accepts whole quant blocks per row.
[[nodiscard]] inline bool project(std::uint32_t type, std::span<const std::uint8_t> raw,
                                  std::span<const float> x, std::span<float> y,
                                  std::vector<float>& scratch,Iq1Kernel iq1=Iq1Kernel::Baseline) {
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
            y[row]=(iq1==Iq1Kernel::Native || iq1==Iq1Kernel::NativeAll)?quant::IQ1S::dot_row<true>(packed,x.data(),x.size()):quant::IQ1S::dot_row<false>(packed,x.data(),x.size());
        }
        return true;
    }
    for (std::size_t row = 0; row < y.size(); ++row) {
        float sum = 0.f;
#if defined(__AVX2__)
        const bool vector_dot=iq1==Iq1Kernel::NativeAll && spec.elems%8==0;
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
} // namespace sub0::denseq
