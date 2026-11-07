// IQ3_S scalar reference decoder, following ggml-quants.c::dequantize_row_iq3_s.
#pragma once
#include "sub0/gguf_quant_tables.hpp" // shared IQ sign masks, retained here during incremental table migration
#include "sub0/quant/format.hpp"
#include "sub0/quant/iq3_s_grid.hpp"
#include <cstddef>

namespace sub0::quant {
struct IQ3S {
    static constexpr TensorType type = TensorType::IQ3_S;
    static constexpr std::string_view name = "IQ3_S";
    static constexpr BlockSpec block{256, 110};

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
            const std::uint8_t* signs = blk + 74;
            const std::uint8_t* scales = blk + 106;
            std::uint64_t written = ib * block.elems;
            for (int ib64 = 0; ib64 < 4; ++ib64) {
                const float db[2] = {
                    d * (1.f + 2.f * static_cast<float>(scales[ib64] & 0x0fu)),
                    d * (1.f + 2.f * static_cast<float>(scales[ib64] >> 4)),
                };
                for (int half = 0; half < 2; ++half) {
                    const std::uint8_t high = qh[2 * ib64 + half];
                    const std::uint8_t* q = qs + 16 * ib64 + 8 * half;
                    const std::uint8_t* sign_group = signs + 8 * ib64 + 4 * half;
                    for (int l = 0; l < 4; ++l) {
                        const std::uint16_t i0 = static_cast<std::uint16_t>(q[2 * l]) |
                            static_cast<std::uint16_t>((static_cast<std::uint16_t>(high) << (8 - 2 * l)) & 0x100u);
                        const std::uint16_t i1 = static_cast<std::uint16_t>(q[2 * l + 1]) |
                            static_cast<std::uint16_t>((static_cast<std::uint16_t>(high) << (7 - 2 * l)) & 0x100u);
                        const std::uint32_t grid0 = IQ3S_GRID[i0];
                        const std::uint32_t grid1 = IQ3S_GRID[i1];
                        const std::uint8_t sign_bits = sign_group[l];
                        for (int j = 0; j < 4; ++j) {
                            const float m0 = static_cast<float>((grid0 >> (8 * j)) & 0xffu);
                            const float m1 = static_cast<float>((grid1 >> (8 * j)) & 0xffu);
                            if (written + static_cast<std::uint64_t>(j) < n)
                                out[static_cast<std::size_t>(written + j)] = db[half] * m0 *
                                    ((sign_bits & gguf::KMASK_IQ2XS[j]) ? -1.f : 1.f);
                            if (written + static_cast<std::uint64_t>(j + 4) < n)
                                out[static_cast<std::size_t>(written + j + 4)] = db[half] * m1 *
                                    ((sign_bits & gguf::KMASK_IQ2XS[j + 4]) ? -1.f : 1.f);
                        }
                        written += 8;
                    }
                }
            }
        }
        return true;
    }
};
static_assert(FormatModule<IQ3S>);
} // namespace sub0::quant
