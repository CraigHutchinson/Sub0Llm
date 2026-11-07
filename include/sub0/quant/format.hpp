// Common metadata and compile-time contract for independently maintained GGML format modules.
#pragma once
#include <concepts>
#include <cstring>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace sub0::quant {
enum class TensorType : std::uint32_t {
    F32 = 0, F16 = 1, Q8_0 = 8, Q3_K = 11, Q2_K = 10, IQ2_XS = 17, IQ3_XXS = 18,
    IQ2_S = 22, IQ3_S = 21, IQ4_XS = 23, IQ1_M = 29,
    Q4_K = 12, Q5_K = 13, Q6_K = 14, IQ2_XXS = 16, IQ1_S = 19, IQ4_NL = 20, BF16 = 30,
};
struct BlockSpec { std::uint64_t elems = 0, bytes = 0; };
inline float f16_to_f32(std::uint16_t h) {
    std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000u) << 16;
    std::uint32_t exp = (h >> 10) & 0x1fu, mant = h & 0x3ffu, bits;
    if (exp == 0) {
        if (mant == 0) bits = sign;
        else { int e = -1; do { mant <<= 1; ++e; } while (!(mant & 0x400u));
               mant &= 0x3ffu; bits = sign | (static_cast<std::uint32_t>(127 - 15 - e) << 23) | (mant << 13); }
    } else if (exp == 0x1fu) bits = sign | 0x7f800000u | (mant << 13);
    else bits = sign | ((exp + (127u - 15u)) << 23) | (mant << 13);
    float f; std::memcpy(&f, &bits, sizeof f); return f;
}

template<class M>
concept FormatModule = requires(std::span<const std::uint8_t> raw, std::uint64_t n, std::vector<float>& out) {
    { M::type } -> std::convertible_to<TensorType>;
    { M::name } -> std::convertible_to<std::string_view>;
    { M::block.elems } -> std::convertible_to<std::uint64_t>;
    { M::block.bytes } -> std::convertible_to<std::uint64_t>;
    { M::dequantize(raw, n, out) } -> std::same_as<bool>;
};
} // namespace sub0::quant
