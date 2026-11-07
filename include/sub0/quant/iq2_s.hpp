// IQ2_S scalar reference decoder, following ggml-quants.c::dequantize_row_iq2_s.
#pragma once
#include "sub0/gguf_quant_tables.hpp" // shared IQ sign masks, retained here during incremental table migration
#include "sub0/quant/format.hpp"
#include "sub0/quant/iq2_s_grid.hpp"
#include <cstddef>

namespace sub0::quant {
struct IQ2S {
    static constexpr TensorType type = TensorType::IQ2_S;
    static constexpr std::string_view name = "IQ2_S";
    static constexpr BlockSpec block{256, 82};

    [[nodiscard]] static inline bool dequantize(std::span<const std::uint8_t> raw, std::uint64_t n,
                                                 std::vector<float>& out) {
        const std::uint64_t blocks = (n + block.elems - 1) / block.elems;
        if (raw.size() < blocks * block.bytes) return false;
        out.resize(static_cast<std::size_t>(n));
        for (std::uint64_t ib = 0; ib < blocks; ++ib) {
            const std::uint8_t* blk = raw.data() + ib * block.bytes;
            const float d = f16_to_f32(static_cast<std::uint16_t>(blk[0] | (blk[1] << 8)));
            const std::uint8_t* qs = blk + 2;
            const std::uint8_t* qh = blk + 66;
            const std::uint8_t* signs = qs + block.elems / 8;
            const std::uint8_t* scales = blk + 74;
            std::uint64_t written = ib * block.elems;
            for (int ib32 = 0; ib32 < 8; ++ib32) {
                const float db[2] = {
                    d * (0.5f + static_cast<float>(scales[ib32] & 0x0fu)) * 0.25f,
                    d * (0.5f + static_cast<float>(scales[ib32] >> 4)) * 0.25f,
                };
                for (int l = 0; l < 4; ++l) {
                    const std::uint16_t grid_index = static_cast<std::uint16_t>(qs[l]) |
                        static_cast<std::uint16_t>((static_cast<std::uint16_t>(qh[ib32]) << (8 - 2 * l)) & 0x300u);
                    const std::uint64_t grid = IQ2S_GRID[grid_index];
                    const std::uint8_t sign_bits = signs[l];
                    for (int j = 0; j < 8; ++j, ++written) {
                        if (written < n) {
                            const float mag = static_cast<float>((grid >> (8 * j)) & 0xffu);
                            const float sign = sign_bits & gguf::KMASK_IQ2XS[j] ? -1.f : 1.f;
                            out[static_cast<std::size_t>(written)] = db[l / 2] * mag * sign;
                        }
                    }
                }
                qs += 4;
                signs += 4;
            }
        }
        return true;
    }
};
static_assert(FormatModule<IQ2S>);
} // namespace sub0::quant
