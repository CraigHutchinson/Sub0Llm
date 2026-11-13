// Packed dense IQ1_S/IQ2_XXS x Q8_K. Derived from llama.cpp cf659f24,
// ggml-cpu/quants.c::ggml_vec_dot_iq{1_s,2_xxs}_q8_K_generic.
#pragma once
#include "sub0/quant/activation_q8k.hpp"
#include "sub0/quant/iq1_s.hpp"
#include "sub0/quant/iq2_xxs.hpp"
#include <bit>
#include <limits>

namespace sub0::denseq {
namespace q8k_detail {
// The AVX2 pairwise multiply cannot saturate: folded IQ1 magnitudes <=9,
// IQ2 magnitudes <=127, activations <=127; each pair fits signed int16.
static_assert([] {
    for(auto grid:gguf::IQ1S_GRID)for(int j=0;j<8;++j){
        const auto q=std::bit_cast<std::int8_t>(static_cast<std::uint8_t>(grid>>(8*j)));
        if(q < -1 || q > 1)return false;
    }
    for(auto grid:gguf::IQ2XXS_GRID)for(int j=0;j<8;++j)if((grid>>(8*j)&255)>127)return false;
    return true;
}());
inline constexpr auto signs=[] {
    std::array<std::uint64_t,128> result{};
    for(std::size_t i=0;i<result.size();++i)
        for(int j=0;j<8;++j)result[i]|=std::uint64_t{(gguf::KSIGNS_IQ2XS[i]>>j)&1?255u:1u}<<(8*j);
    return result;
}();
#if defined(__AVX2__)
[[nodiscard]] inline __m256i group(std::uint64_t a,std::uint64_t b,std::uint64_t c,std::uint64_t d) noexcept {
    return _mm256_set_epi64x(std::bit_cast<long long>(d),std::bit_cast<long long>(c),std::bit_cast<long long>(b),std::bit_cast<long long>(a));
}
[[nodiscard]] inline __m256i dot32(__m256i magnitude,__m256i signed_input) noexcept {
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
    return _mm256_dpbusd_epi32(_mm256_setzero_si256(),magnitude,signed_input);
#else
    // Table magnitudes <=127 and Q8 input excludes -128: adjacent products fit int16.
    return _mm256_madd_epi16(_mm256_maddubs_epi16(magnitude,signed_input),_mm256_set1_epi16(1));
#endif
}
[[nodiscard]] inline int reduce(__m256i a) noexcept {
    __m128i s=_mm_add_epi32(_mm256_castsi256_si128(a),_mm256_extracti128_si256(a,1));
    s=_mm_add_epi32(s,_mm_shuffle_epi32(s,0x4e));
    s=_mm_add_epi32(s,_mm_shuffle_epi32(s,0xb1));
    return _mm_cvtsi128_si32(s);
}
#endif
// Unpack only one integer group into registers. No floating weights or unpacked tensor storage.
template<bool Native,bool IQ1>
[[nodiscard]] inline float row(const std::uint8_t* packed,std::span<const quant::Q8KBlock> x) noexcept {
    float total=0;
    for(std::size_t b=0;b<x.size();++b){
        const auto* block=packed+b*(IQ1?50:66);const auto& activation=x[b];
        const float d=quant::IQ2XXS::scale(block)*activation.d;
        int integer=0;
#if defined(__AVX2__)
        __m256i accum0=_mm256_setzero_si256(),accum1=_mm256_setzero_si256();
#endif
        for(int g=0;g<8;++g){
            std::uint64_t grids[4];int odd=0,offset=0;std::uint64_t sign_words[4]{};
            if constexpr(IQ1){
                const auto f=quant::IQ1S::fields(block,32*g);odd=2*((f.qh>>12)&7)+1;
                for(int l=0;l<4;++l)grids[l]=quant::IQ1S::grid(f,l);
                offset=(f.qh&0x8000)?-1:1;
            }else{
                const auto f=quant::IQ2XXS::fields(block,32*g);odd=2*(f.aux[1]>>28)+1;
                for(int l=0;l<4;++l){grids[l]=quant::IQ2XXS::grid(f,l);sign_words[l]=signs[quant::IQ2XXS::sign_index(f,l)];}
            }
#if defined(__AVX2__)
            if constexpr(Native){
                const auto w=group(grids[0],grids[1],grids[2],grids[3]);
                const auto input=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(activation.qs.data()+32*g));
                __m256i magnitude,adjusted;
                if constexpr(IQ1){
                    // (grid + sign/8) = (8*grid + sign)/8. Fold the offset into
                    // small signed integer weights instead of loading activation sums per group.
                    auto folded=_mm256_add_epi8(w,w);
                    folded=_mm256_add_epi8(folded,folded);
                    folded=_mm256_add_epi8(folded,folded);
                    folded=_mm256_add_epi8(folded,_mm256_set1_epi8(static_cast<char>(offset)));
                    magnitude=_mm256_abs_epi8(folded);adjusted=_mm256_sign_epi8(input,folded);
                }
                else {magnitude=w;adjusted=_mm256_sign_epi8(input,group(sign_words[0],sign_words[1],sign_words[2],sign_words[3]));}
                const auto partial=_mm256_mullo_epi32(dot32(magnitude,adjusted),_mm256_set1_epi32(odd));
                if(g&1)accum1=_mm256_add_epi32(accum1,partial);
                else accum0=_mm256_add_epi32(accum0,partial);
                continue;
            }
#endif
            int sum=0;
            for(int l=0;l<4;++l)for(int j=0;j<8;++j){
                const auto byte=static_cast<std::uint8_t>(grids[l]>>(8*j));
                const int w=IQ1?8*static_cast<int>(std::bit_cast<std::int8_t>(byte))+offset:
                    static_cast<int>(byte)*((sign_words[l]>>(8*j)&255)==255?-1:1);
                sum+=w*static_cast<int>(activation.qs[32*g+8*l+j]);
            }
            integer+=odd*sum;
        }
#if defined(__AVX2__)
        if constexpr(Native)integer=reduce(_mm256_add_epi32(accum0,accum1));
#endif
        if constexpr(IQ1)total+=d*(static_cast<float>(integer)*0.125f);
        else total+=d*static_cast<float>(integer);
    }
    if constexpr(!IQ1)total*=0.125f;
    return total;
}
}
[[nodiscard]] inline bool q8k_supported(std::uint32_t type) noexcept {return type==19 || type==16;}
// Prepare x once before splitting output rows among workers. All buffers are borrowed for this call.
// Outputs must not overlap weights or x. Concurrent sessions own distinct x/output buffers.
template<bool Native=true>
[[nodiscard]] inline bool project_q8k(std::uint32_t type,std::span<const std::uint8_t> raw,
                                     const quant::Q8KActivations& x,std::span<float> y) noexcept {
    if(!q8k_supported(type) || !x.width() || x.width()%256)return false;
    const std::size_t bytes=type==19?quant::IQ1S::block.bytes:quant::IQ2XXS::block.bytes;
    if(x.blocks().size()>std::numeric_limits<std::size_t>::max()/bytes)return false;
    const auto row_bytes=x.blocks().size()*bytes;
    if(y.size()>raw.size()/row_bytes)return false;
    for(std::size_t r=0;r<y.size();++r)
        y[r]=type==19?q8k_detail::row<Native,true>(raw.data()+r*row_bytes,x.blocks()):
                       q8k_detail::row<Native,false>(raw.data()+r*row_bytes,x.blocks());
    return true;
}
}
