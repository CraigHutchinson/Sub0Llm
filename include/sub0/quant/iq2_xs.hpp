// IQ2_XS scalar reference decoder, following ggml-quants.c::dequantize_row_iq2_xs.
#pragma once
#include "sub0/gguf_quant_tables.hpp" // shared IQ sign codebook, retained here during incremental table migration
#include "sub0/quant/format.hpp"
#include "sub0/quant/iq2_xs_grid.hpp"
#include <cstddef>

namespace sub0::quant {
struct IQ2XS {
    static constexpr TensorType type = TensorType::IQ2_XS;
    static constexpr std::string_view name = "IQ2_XS";
    static constexpr BlockSpec block{256, 74};

    [[nodiscard]] static inline bool dequantize(std::span<const std::uint8_t> raw, std::uint64_t n,
                                                 std::vector<float>& out) {
        const std::uint64_t blocks = (n + block.elems - 1) / block.elems;
        if (raw.size() < blocks * block.bytes) return false;
        out.resize(static_cast<std::size_t>(n));
        for (std::uint64_t ib = 0; ib < blocks; ++ib) {
            const std::uint8_t* blk = raw.data() + ib * block.bytes;
            const float d = f16_to_f32(static_cast<std::uint16_t>(blk[0] | (blk[1] << 8)));
            const std::uint8_t* qs = blk + 2;
            const std::uint8_t* scales = blk + 66;
            std::uint64_t written = ib * block.elems;
            for (int ib32 = 0; ib32 < 8; ++ib32) {
                const float db[2] = {
                    d * (0.5f + static_cast<float>(scales[ib32] & 0x0fu)) * 0.25f,
                    d * (0.5f + static_cast<float>(scales[ib32] >> 4)) * 0.25f,
                };
                for (int l = 0; l < 4; ++l) {
                    const std::uint8_t* q = qs + 2 * (4 * ib32 + l);
                    const std::uint16_t packed = static_cast<std::uint16_t>(q[0] | (q[1] << 8));
                    const std::uint64_t grid = IQ2XS_GRID[packed & 511u];
                    const std::uint8_t signs = gguf::KSIGNS_IQ2XS[packed >> 9];
                    for (int j = 0; j < 8; ++j, ++written) {
                        if (written < n) {
                            const float mag = static_cast<float>((grid >> (8 * j)) & 0xffu);
                            const float sign = (signs >> j) & 1u ? -1.f : 1.f;
                            out[static_cast<std::size_t>(written)] = db[l / 2] * mag * sign;
                        }
                    }
                }
            }
        }
        return true;
    }
};
static_assert(FormatModule<IQ2XS>);
} // namespace sub0::quant
