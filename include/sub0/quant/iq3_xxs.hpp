// IQ3_XXS scalar reference decoder, following ggml-quants.c::dequantize_row_iq3_xxs.
#pragma once
#include "sub0/gguf_quant_tables.hpp" // shared IQ sign codebook, retained here during incremental table migration
#include "sub0/quant/format.hpp"
#include "sub0/quant/iq3_xxs_grid.hpp"
#include <cstddef>

namespace sub0::quant {
struct IQ3XXS {
    static constexpr TensorType type = TensorType::IQ3_XXS;
    static constexpr std::string_view name = "IQ3_XXS";
    static constexpr BlockSpec block{256, 98};

    [[nodiscard]] static inline bool dequantize(std::span<const std::uint8_t> raw, std::uint64_t n,
                                                 std::vector<float>& out) {
        const std::uint64_t blocks = (n + block.elems - 1) / block.elems;
        if (raw.size() < blocks * block.bytes) return false;
        out.resize(static_cast<std::size_t>(n));
        for (std::uint64_t ib = 0; ib < blocks; ++ib) {
            const std::uint8_t* blk = raw.data() + ib * block.bytes;
            const float d = f16_to_f32(static_cast<std::uint16_t>(blk[0] | (blk[1] << 8)));
            const std::uint8_t* qs = blk + 2;
            const std::uint8_t* scales_and_signs = qs + block.elems / 4;
            std::uint64_t written = ib * block.elems;
            for (int ib32 = 0; ib32 < 8; ++ib32) {
                const std::uint8_t* aux = scales_and_signs + 4 * ib32;
                const std::uint32_t aux32 = static_cast<std::uint32_t>(aux[0]) |
                    (static_cast<std::uint32_t>(aux[1]) << 8) |
                    (static_cast<std::uint32_t>(aux[2]) << 16) |
                    (static_cast<std::uint32_t>(aux[3]) << 24);
                const float db = d * (0.5f + static_cast<float>(aux32 >> 28)) * 0.5f;
                for (int l = 0; l < 4; ++l) {
                    const std::uint8_t signs = gguf::KSIGNS_IQ2XS[(aux32 >> (7 * l)) & 127u];
                    const std::uint32_t grid1 = IQ3XXS_GRID[qs[2 * l]];
                    const std::uint32_t grid2 = IQ3XXS_GRID[qs[2 * l + 1]];
                    for (int j = 0; j < 4; ++j) {
                        const std::uint64_t i0 = written + static_cast<std::uint64_t>(j);
                        const std::uint64_t i1 = written + static_cast<std::uint64_t>(j + 4);
                        const float m0 = static_cast<float>((grid1 >> (8 * j)) & 0xffu);
                        const float m1 = static_cast<float>((grid2 >> (8 * j)) & 0xffu);
                        if (i0 < n) out[static_cast<std::size_t>(i0)] = db * m0 * ((signs >> j) & 1u ? -1.f : 1.f);
                        if (i1 < n) out[static_cast<std::size_t>(i1)] = db * m1 * ((signs >> (j + 4)) & 1u ? -1.f : 1.f);
                    }
                    written += 8;
                }
                qs += 8;
            }
        }
        return true;
    }
};
static_assert(FormatModule<IQ3XXS>);
} // namespace sub0::quant
