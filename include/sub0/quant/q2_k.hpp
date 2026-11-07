// Q2_K scalar reference decoder. Layout transcribed from ggml-quants.c::dequantize_row_q2_K.
#pragma once
#include "sub0/quant/format.hpp"
#include <cstddef>

namespace sub0::quant {
struct Q2K {
    static constexpr TensorType type = TensorType::Q2_K;
    static constexpr std::string_view name = "Q2_K";
    static constexpr BlockSpec block{256, 84};

    [[nodiscard]] static inline bool dequantize(std::span<const std::uint8_t> raw, std::uint64_t n,
                                                 std::vector<float>& out) {
        const std::uint64_t blocks = (n + block.elems - 1) / block.elems;
        if (raw.size() < blocks * block.bytes) return false;
        out.resize(static_cast<std::size_t>(n));
        std::uint64_t written = 0;
        for (std::uint64_t b = 0; b < blocks; ++b) {
            const std::uint8_t* blk = raw.data() + b * block.bytes;
            const float d = f16_to_f32(static_cast<std::uint16_t>(blk[80] | (blk[81] << 8)));
            const float dmin = f16_to_f32(static_cast<std::uint16_t>(blk[82] | (blk[83] << 8)));
            const std::uint8_t* qs = blk + 16;
            int is = 0;
            for (int half = 0; half < 2; ++half) {
                int shift = 0;
                for (int j = 0; j < 4; ++j) {
                    for (int repeat = 0; repeat < 2; ++repeat) {
                        const std::uint8_t sc = blk[is++];
                        const float dl = d * (sc & 0x0f);
                        const float ml = dmin * (sc >> 4);
                        const std::uint8_t* q = qs + half * 32 + repeat * 16;
                        for (int l = 0; l < 16 && written < n; ++l, ++written)
                            out[static_cast<std::size_t>(written)] = dl * ((q[l] >> shift) & 3) - ml;
                    }
                    shift += 2;
                }
            }
        }
        return true;
    }
};
static_assert(FormatModule<Q2K>);
} // namespace sub0::quant
