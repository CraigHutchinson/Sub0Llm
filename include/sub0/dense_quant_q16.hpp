// Accuracy-oriented packed IQ x signed-int16 activations. Weight encoding is unchanged.
#pragma once
#include "sub0/dense_quant_q8k.hpp"
namespace sub0::denseq {
namespace q16_detail {
#if defined(__AVX2__)
[[nodiscard]] inline std::int64_t reduce(__m256i a) noexcept {
    const auto s=_mm256_add_epi64(_mm256_cvtepi32_epi64(_mm256_castsi256_si128(a)),_mm256_cvtepi32_epi64(_mm256_extracti128_si256(a,1)));
    const auto pair=_mm_add_epi64(_mm256_castsi256_si128(s),_mm256_extracti128_si256(s,1));
    return _mm_cvtsi128_si64(pair)+_mm_extract_epi64(pair,1);
}
[[nodiscard]] inline __m256i dot(__m256i accum,__m256i weights,__m256i input) noexcept {
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
    return _mm256_dpwssd_epi32(accum,weights,input);
#else
    return _mm256_add_epi32(accum,_mm256_madd_epi16(weights,input));
#endif
}
#endif
// IQ2: at most 16 products per int32 lane before the final int64 reduction.
// 16*43*31*32767 < INT32_MAX. IQ1 folded weights are bounded by 9*15.
static_assert([]{for(auto g:gguf::IQ2XXS_GRID)for(int j=0;j<8;++j)if((g>>(8*j)&255)>43)return false;return true;}());
template<bool Native,bool IQ1>
[[nodiscard]] inline float row(const std::uint8_t* packed,std::span<const quant::Q16Block> x) noexcept {
    float total=0;
    for(std::size_t b=0;b<x.size();++b){
        const auto* block=packed+b*(IQ1?50:66);const auto& activation=x[b];
        std::int64_t integer=0;
#if defined(__AVX2__)
        __m256i lo=_mm256_setzero_si256(),hi=_mm256_setzero_si256();
#endif
        for(int g=0;g<8;++g){
            std::uint64_t grids[4],signs[4]{};int odd=0,offset=0;
            if constexpr(IQ1){const auto f=quant::IQ1S::fields(block,32*g);odd=2*((f.qh>>12)&7)+1;offset=(f.qh&0x8000)?-1:1;
                for(int l=0;l<4;++l)grids[l]=quant::IQ1S::grid(f,l);
            }else{const auto f=quant::IQ2XXS::fields(block,32*g);odd=2*(f.aux[1]>>28)+1;
                for(int l=0;l<4;++l){grids[l]=quant::IQ2XXS::grid(f,l);signs[l]=q8k_detail::signs[quant::IQ2XXS::sign_index(f,l)];}
            }
#if defined(__AVX2__)
            if constexpr(Native){
                auto w=q8k_detail::group(grids[0],grids[1],grids[2],grids[3]);
                if constexpr(IQ1){w=_mm256_add_epi8(w,w);w=_mm256_add_epi8(w,w);w=_mm256_add_epi8(w,w);w=_mm256_add_epi8(w,_mm256_set1_epi8(static_cast<char>(offset)));}
                else w=_mm256_sign_epi8(w,q8k_detail::group(signs[0],signs[1],signs[2],signs[3]));
                const auto scale=_mm256_set1_epi16(static_cast<short>(odd));
                const auto wl=_mm256_mullo_epi16(_mm256_cvtepi8_epi16(_mm256_castsi256_si128(w)),scale);
                const auto wh=_mm256_mullo_epi16(_mm256_cvtepi8_epi16(_mm256_extracti128_si256(w,1)),scale);
                lo=dot(lo,wl,_mm256_loadu_si256(reinterpret_cast<const __m256i*>(activation.qs.data()+32*g)));
                hi=dot(hi,wh,_mm256_loadu_si256(reinterpret_cast<const __m256i*>(activation.qs.data()+32*g+16)));
                continue;
            }
#endif
            for(int l=0;l<4;++l)for(int j=0;j<8;++j){const auto byte=static_cast<std::uint8_t>(grids[l]>>(8*j));
                const int w=IQ1?8*static_cast<int>(std::bit_cast<std::int8_t>(byte))+offset:
                    static_cast<int>(byte)*((signs[l]>>(8*j)&255)==255?-1:1);
                integer+=static_cast<std::int64_t>(w*odd)*activation.qs[32*g+8*l+j];
            }
        }
#if defined(__AVX2__)
        if constexpr(Native)integer=reduce(lo)+reduce(hi);
#endif
        total+=(quant::IQ2XXS::scale(block)*activation.d)*(static_cast<float>(integer)*0.125f);
    }
    return total;
}
}
template<bool Native=true>
[[nodiscard]] inline bool project_q16(std::uint32_t type,std::span<const std::uint8_t> raw,
                                     const quant::Q16Activations& x,std::span<float> y) noexcept {
    if(!q8k_supported(type) || !x.width() || x.width()%256)return false;
    const std::size_t bytes=type==19?quant::IQ1S::block.bytes:quant::IQ2XXS::block.bytes;
    if(x.blocks().size()>std::numeric_limits<std::size_t>::max()/bytes)return false;
    const auto row_bytes=x.blocks().size()*bytes;
    if(y.size()>raw.size()/row_bytes)return false;
    for(std::size_t r=0;r<y.size();++r)y[r]=type==19?q16_detail::row<Native,true>(raw.data()+r*row_bytes,x.blocks()):q16_detail::row<Native,false>(raw.data()+r*row_bytes,x.blocks());
    return true;
}
}
