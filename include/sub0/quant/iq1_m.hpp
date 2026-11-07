// IQ1_M scalar reference decoder, following ggml-quants.c::dequantize_row_iq1_m.
#pragma once
#include "sub0/gguf_quant_tables.hpp" // IQ1_S ternary grid and delta are shared with IQ1_M
#include "sub0/quant/format.hpp"
#include <cstddef>

namespace sub0::quant {
struct IQ1M {
    static constexpr TensorType type = TensorType::IQ1_M;
    static constexpr std::string_view name = "IQ1_M";
    static constexpr BlockSpec block{256, 56};

    [[nodiscard]] static inline bool dequantize(std::span<const std::uint8_t> raw, std::uint64_t n,
                                                 std::vector<float>& out) {
        const std::uint64_t blocks = (n + block.elems - 1) / block.elems;
        if (raw.size() < blocks * block.bytes) return false;
        out.resize(static_cast<std::size_t>(n));
        for (std::uint64_t ib = 0; ib < blocks; ++ib) {
            const std::uint8_t* blk = raw.data() + ib * block.bytes;
            const std::uint8_t* qs = blk;
            const std::uint8_t* qh = blk + 32;
            const std::uint8_t* scales = blk + 48;
            std::uint16_t sc[4];
            for (int i = 0; i < 4; ++i)
                sc[i] = static_cast<std::uint16_t>(scales[2 * i] | (scales[2 * i + 1] << 8));
            const std::uint16_t d_bits = static_cast<std::uint16_t>(
                (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0u) |
                ((sc[2] >> 4) & 0x0f00u) | (sc[3] & 0xf000u));
            const float d = f16_to_f32(d_bits);
            std::uint64_t written = ib * block.elems;
            for (int ib32 = 0; ib32 < 8; ++ib32) {
                const std::uint16_t sb = sc[ib32 / 2];
                const int shift = 6 * (ib32 % 2);
                const float dl1 = d * (2.f * static_cast<float>((sb >> shift) & 0x7u) + 1.f);
                const float dl2 = d * (2.f * static_cast<float>((sb >> (shift + 3)) & 0x7u) + 1.f);
                const std::uint8_t h0 = qh[0], h1 = qh[1];
                const std::uint16_t indices[4] = {
                    static_cast<std::uint16_t>(qs[0] | ((static_cast<std::uint16_t>(h0) << 8) & 0x700u)),
                    static_cast<std::uint16_t>(qs[1] | ((static_cast<std::uint16_t>(h0) << 4) & 0x700u)),
                    static_cast<std::uint16_t>(qs[2] | ((static_cast<std::uint16_t>(h1) << 8) & 0x700u)),
                    static_cast<std::uint16_t>(qs[3] | ((static_cast<std::uint16_t>(h1) << 4) & 0x700u)),
                };
                const float delta[4] = {
                    (h0 & 0x08u) ? -gguf::IQ1S_DELTA : gguf::IQ1S_DELTA,
                    (h0 & 0x80u) ? -gguf::IQ1S_DELTA : gguf::IQ1S_DELTA,
                    (h1 & 0x08u) ? -gguf::IQ1S_DELTA : gguf::IQ1S_DELTA,
                    (h1 & 0x80u) ? -gguf::IQ1S_DELTA : gguf::IQ1S_DELTA,
                };
                for (int l = 0; l < 4; ++l) {
                    const std::uint64_t grid = gguf::IQ1S_GRID[indices[l]];
                    const float scale = l < 2 ? dl1 : dl2;
                    for (int j = 0; j < 8; ++j, ++written) {
                        if (written < n) {
                            const std::uint8_t byte = static_cast<std::uint8_t>(grid >> (8 * j));
                            const int q = byte < 0x80u ? static_cast<int>(byte) : static_cast<int>(byte) - 256;
                            out[static_cast<std::size_t>(written)] = scale * (static_cast<float>(q) + delta[l]);
                        }
                    }
                }
                qs += 4;
                qh += 2;
            }
        }
        return true;
    }
};
static_assert(FormatModule<IQ1M>);
} // namespace sub0::quant
