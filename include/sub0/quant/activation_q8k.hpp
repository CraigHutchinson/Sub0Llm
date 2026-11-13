// Q8_K-compatible activation scratch, derived from pinned llama.cpp cf659f24
// ggml-quants.c::quantize_row_q8_K_ref. Runtime scratch only; never serialized.
#pragma once
#include <algorithm>
#include <array>
#include <bit>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <span>
#include <vector>
#include <type_traits>
#include <utility>

namespace sub0::quant {
struct Q8KBlock {
    float d=0;
    std::array<std::int8_t,256> qs{};
    std::array<std::int16_t,16> bsums{};
};
struct Q16Block {float d=0;std::array<std::int16_t,256> qs{};};
// One buffer per independent session/activation. Construct before decode; quantize never resizes.
// A prepared row may be read concurrently by output-row workers until the next quantize call.
template<class Block,int Limit>
class ActivationRow {
    static_assert((std::is_same_v<Block,Q8KBlock> && Limit==127) || (std::is_same_v<Block,Q16Block> && Limit==32767));
    std::vector<Block> storage_;
    std::size_t width_=0;
public:
    explicit ActivationRow(std::size_t maximum_width=0):storage_(maximum_width/256) {}
    ActivationRow(const ActivationRow&)=default;
    ActivationRow& operator=(const ActivationRow&)=default;
    ActivationRow(ActivationRow&& other) noexcept:storage_(std::move(other.storage_)),width_(std::exchange(other.width_,0)) {}
    ActivationRow& operator=(ActivationRow&& other) noexcept {
        if(this!=&other){storage_=std::move(other.storage_);width_=std::exchange(other.width_,0);}
        return *this;
    }
    [[nodiscard]] std::size_t width() const noexcept {return width_;}
    [[nodiscard]] std::span<const Block> blocks() const noexcept {return {storage_.data(),width_/256};}
    // Signed first-largest max and nearest-even rounding match the pinned reference.
    // Invalid input invalidates the prepared view. Reference-unrepresentable tiny reciprocals
    // are rejected rather than passing an infinity to the rounding conversion.
    [[nodiscard]] bool quantize(std::span<const float> x) noexcept {
        width_=0;
        if(x.empty() || x.size()%256 || x.size()/256>storage_.size() || std::fegetround()!=FE_TONEAREST)return false;
        for(std::size_t b=0;b<x.size()/256;++b){
            auto& out=storage_[b];const float* input=x.data()+256*b;
            float maximum=0,absolute=0;
            for(int j=0;j<256;++j){
                if(!std::isfinite(input[j]))return false;
                const float a=std::fabs(input[j]);
                if(a>absolute){absolute=a;maximum=input[j];}
            }
            if(absolute==0){out.d=0;out.qs.fill(0);if constexpr(Limit==127)out.bsums.fill(0);continue;}
            const float inverse=-static_cast<float>(Limit)/maximum;
            if(!std::isfinite(inverse) || inverse==0)return false;
            out.d=1.f/inverse;
            if(!std::isfinite(out.d))return false;
            for(int sub=0;sub<16;++sub){
                int sum=0;
                for(int j=0;j<16;++j){
                    const int index=sub*16+j;
                    // Round the float product before integer conversion. A fused multiply+bias
                    // would change half-way cases compared with the reference nearest_int(float).
                    const int rounded=static_cast<int>(std::nearbyint(input[index]*inverse));
                    const auto q=static_cast<typename decltype(out.qs)::value_type>(std::clamp(rounded,-Limit,Limit));
                    out.qs[index]=q;sum+=q;
                }
                if constexpr(Limit==127)out.bsums[sub]=static_cast<std::int16_t>(sum);
            }
        }
        width_=x.size();return true;
    }
};
using Q8KActivations=ActivationRow<Q8KBlock,127>;
using Q16Activations=ActivationRow<Q16Block,32767>;
}
