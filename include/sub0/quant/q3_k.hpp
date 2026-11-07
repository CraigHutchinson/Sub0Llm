// Q3_K scalar reference decoder, following ggml-quants.c::dequantize_row_q3_K.
#pragma once
#include "sub0/quant/format.hpp"
#include <cstddef>

namespace sub0::quant {
struct Q3K {
    static constexpr TensorType type = TensorType::Q3_K;
    static constexpr std::string_view name = "Q3_K";
    static constexpr BlockSpec block{256, 110};

    [[nodiscard]] static inline bool dequantize(std::span<const std::uint8_t> raw, std::uint64_t n,
                                                 std::vector<float>& out) {
        const std::uint64_t blocks = (n + block.elems - 1) / block.elems;
        if (raw.size() < blocks * block.bytes) return false;
        out.resize(static_cast<std::size_t>(n));
        for (std::uint64_t ib = 0; ib < blocks; ++ib) {
            const std::uint8_t* blk = raw.data() + ib * block.bytes;
            const std::uint8_t* hm = blk;
            const std::uint8_t* q = blk + 32;
            const std::uint8_t* packed_scales = blk + 96;
            const float d = f16_to_f32(static_cast<std::uint16_t>(blk[108] | (blk[109] << 8)));

            std::uint32_t aux[4] = {};
            for (int i = 0; i < 3; ++i) {
                for (int b = 0; b < 4; ++b)
                    aux[i] |= static_cast<std::uint32_t>(packed_scales[4 * i + b]) << (8 * b);
            }
            constexpr std::uint32_t kmask1 = 0x03030303u;
            constexpr std::uint32_t kmask2 = 0x0f0f0f0fu;
            const std::uint32_t tmp = aux[2];
            aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
            aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
            aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
            aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);

            const auto scale_at = [&aux](int i) {
                return static_cast<int>((aux[i / 4] >> (8 * (i % 4))) & 0xffu) - 32;
            };
            int is = 0;
            std::uint8_t mask = 1;
            std::uint64_t written = ib * block.elems;
            for (int half = 0; half < 2; ++half) {
                int shift = 0;
                for (int j = 0; j < 4; ++j) {
                    const float dl0 = d * static_cast<float>(scale_at(is++));
                    for (int lane = 0; lane < 16; ++lane, ++written) {
                        const int qv = static_cast<int>((q[lane] >> shift) & 3u) - ((hm[lane] & mask) ? 0 : 4);
                        if (written < n) out[static_cast<std::size_t>(written)] = dl0 * static_cast<float>(qv);
                    }
                    const float dl1 = d * static_cast<float>(scale_at(is++));
                    for (int lane = 0; lane < 16; ++lane, ++written) {
                        const int qv = static_cast<int>((q[lane + 16] >> shift) & 3u) - ((hm[lane + 16] & mask) ? 0 : 4);
                        if (written < n) out[static_cast<std::size_t>(written)] = dl1 * static_cast<float>(qv);
                    }
                    shift += 2;
                    mask = static_cast<std::uint8_t>(mask << 1);
                }
                q += 32;
            }
        }
        return true;
    }
};
static_assert(FormatModule<Q3K>);
} // namespace sub0::quant
