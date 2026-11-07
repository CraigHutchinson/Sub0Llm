// Incremental format-module list. Legacy formats migrate here one at a time per docs/QUANT_FORMATS.md.
#pragma once
#include "sub0/quant/q2_k.hpp"
#include "sub0/quant/iq3_xxs.hpp"
#include "sub0/quant/iq2_xs.hpp"
#include "sub0/quant/iq2_s.hpp"
#include "sub0/quant/iq3_s.hpp"
#include "sub0/quant/iq1_m.hpp"
#include "sub0/quant/iq1_s.hpp"
#include "sub0/quant/iq4_xs.hpp"
#include "sub0/quant/q3_k.hpp"
#include <array>

namespace sub0::quant {
using DequantFn = bool (*)(std::span<const std::uint8_t>, std::uint64_t, std::vector<float>&);
struct ModuleInfo { TensorType type; std::string_view name; BlockSpec block; DequantFn dequantize; };
inline constexpr std::array<ModuleInfo, 9> kModules = {
    ModuleInfo{Q2K::type, Q2K::name, Q2K::block, &Q2K::dequantize},
    ModuleInfo{IQ3XXS::type, IQ3XXS::name, IQ3XXS::block, &IQ3XXS::dequantize},
    ModuleInfo{IQ2XS::type, IQ2XS::name, IQ2XS::block, &IQ2XS::dequantize},
    ModuleInfo{IQ2S::type, IQ2S::name, IQ2S::block, &IQ2S::dequantize},
    ModuleInfo{IQ3S::type, IQ3S::name, IQ3S::block, &IQ3S::dequantize},
    ModuleInfo{IQ1M::type, IQ1M::name, IQ1M::block, &IQ1M::dequantize},
    ModuleInfo{IQ1S::type, IQ1S::name, IQ1S::block, &IQ1S::dequantize},
    ModuleInfo{IQ4XS::type, IQ4XS::name, IQ4XS::block, &IQ4XS::dequantize},
    ModuleInfo{Q3K::type, Q3K::name, Q3K::block, &Q3K::dequantize},
};
}
