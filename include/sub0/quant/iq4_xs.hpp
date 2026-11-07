// IQ4_XS scalar reference decoder, following ggml-quants.c::dequantize_row_iq4_xs.
#pragma once
#include "sub0/gguf_quant_tables.hpp" // shared IQ4_NL value table, retained during incremental migration
#include "sub0/quant/format.hpp"
#include <cstddef>

namespace sub0::quant {
struct IQ4XS {
    static constexpr TensorType type = TensorType::IQ4_XS;
    static constexpr std::string_view name = "IQ4_XS";
    static constexpr BlockSpec block{256, 136};

    [[nodiscard]] static inline bool dequantize(std::span<const std::uint8_t> raw, std::uint64_t n,
                                                 std::vector<float>& out) {
        const std::uint64_t blocks = (n + block.elems - 1) / block.elems;
        if (raw.size() < blocks * block.bytes) return false;
        out.resize(static_cast<std::size_t>(n));
        for (std::uint64_t ib = 0; ib < blocks; ++ib) {
            const std::uint8_t* blk = raw.data() + ib * block.bytes;
            const float d = f16_to_f32(static_cast<std::uint16_t>(blk[0] | (blk[1] << 8)));
            const std::uint16_t scales_h = static_cast<std::uint16_t>(blk[2] | (blk[3] << 8));
            const std::uint8_t* scales_l = blk + 4;
            const std::uint8_t* qs = blk + 8;
            std::uint64_t written = ib * block.elems;
            for (int group = 0; group < 8; ++group) {
                const int ls = ((scales_l[group / 2] >> (4 * (group % 2))) & 0x0f) |
                    (((scales_h >> (2 * group)) & 0x03) << 4);
                const float dl = d * static_cast<float>(ls - 32);
                for (int j = 0; j < 16; ++j) {
                    const float lo = dl * gguf::KVALUES_IQ4NL[qs[j] & 0x0fu];
                    const float hi = dl * gguf::KVALUES_IQ4NL[qs[j] >> 4];
                    if (written + static_cast<std::uint64_t>(j) < n)
                        out[static_cast<std::size_t>(written + j)] = lo;
                    if (written + static_cast<std::uint64_t>(j + 16) < n)
                        out[static_cast<std::size_t>(written + j + 16)] = hi;
                }
                written += 32;
                qs += 16;
            }
        }
        return true;
    }
};
static_assert(FormatModule<IQ4XS>);
} // namespace sub0::quant
