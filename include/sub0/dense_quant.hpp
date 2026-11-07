// Bounded-memory scalar GGUF projection baseline. GGML stores each output row contiguously.
#pragma once
#include "sub0/gguf.hpp"
#include <limits>

namespace sub0::denseq {
// Caller owns weights, inputs, outputs and scratch; scratch must be reserved once before decode.
// x and y must not overlap. This baseline accepts whole quant blocks per row.
[[nodiscard]] inline bool project(std::uint32_t type, std::span<const std::uint8_t> raw,
                                  std::span<const float> x, std::span<float> y,
                                  std::vector<float>& scratch) {
    const auto spec = gguf::block_spec(type);
    const auto decode = gguf::dequant_fn(type);
    if (!decode || spec.elems == 0 || x.empty() || x.size() % spec.elems != 0 ||
        scratch.capacity() < spec.elems) return false;
    const std::size_t blocks = x.size() / spec.elems;
    if (blocks > std::numeric_limits<std::size_t>::max() / spec.bytes) return false;
    const std::size_t row_bytes = blocks * spec.bytes;
    if (y.size() > raw.size() / row_bytes) return false;
    for (std::size_t row = 0; row < y.size(); ++row) {
        float sum = 0.f;
        for (std::size_t b = 0; b < blocks; ++b) {
            const auto bytes = raw.subspan(row * row_bytes + b * spec.bytes, spec.bytes);
            if (!decode(bytes, spec.elems, scratch)) return false;
            for (std::size_t j = 0; j < spec.elems; ++j)
                sum += x[b * spec.elems + j] * scratch[j];
        }
        y[row] = sum;
    }
    return true;
}
} // namespace sub0::denseq
