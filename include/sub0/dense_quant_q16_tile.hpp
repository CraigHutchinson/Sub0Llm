// IQ1_S Q16: two output rows x up to two active sessions. GGUF weights stay packed.
#pragma once
#include "sub0/dense_quant_q16.hpp"

namespace sub0::denseq {
inline constexpr std::size_t IQ1_Q16_TILE_MAX_SESSIONS=2;
template<int Bits> inline constexpr bool iq1_q16_tile_available=[] {
    static_assert(Bits==0 || Bits==256 || Bits==512);
    if constexpr(Bits!=512)return true; // 0 is scalar; 256 has a scalar portable fallback.
#if defined(__AVX512VNNI__) && defined(__AVX512BW__)
    return true;
#else
    return false;
#endif
}();
namespace q16_tile_detail {
// Same-sized codebook, not an unpacked tensor: keep the exact byte representation of 8*q.
inline constexpr auto grid8=[] {
    std::array<std::uint64_t,2048> result{};
    for(std::size_t i=0;i<result.size();++i)for(int j=0;j<8;++j){
        const int q=std::bit_cast<std::int8_t>(static_cast<std::uint8_t>(gguf::IQ1S_GRID[i]>>(8*j)));
        result[i]|=std::uint64_t{static_cast<std::uint8_t>(8*q)}<<(8*j);
    }
    return result;
}();
template<bool Scaled>
[[nodiscard]] inline std::uint64_t grid(const quant::IQ1S::Fields& f,int l) noexcept {
    if constexpr(Scaled)return grid8[quant::IQ1S::index(f,l)];
    else return quant::IQ1S::grid(f,l);
}
// |sum| <=256*9*15*32767=1,132,427,520: the WHOLE IQ1 block fits int32.
// This bound deliberately does not apply to IQ2_XXS.
template<std::size_t Sessions,std::size_t Rows,bool Scaled>
inline void scalar(const std::uint8_t* raw,std::size_t row_bytes,
                   std::span<const quant::Q16Activations* const> x,float* y,std::size_t stride) noexcept {
    float total[Sessions][Rows]{};
    for(std::size_t b=0;b<x[0]->blocks().size();++b){
        std::int32_t sum[Sessions][Rows]{};
        for(std::size_t r=0;r<Rows;++r){
            const auto* block=raw+r*row_bytes+b*50;
            for(int g=0;g<8;++g){
                const auto f=quant::IQ1S::fields(block,32*g);
                const int odd=2*((f.qh>>12)&7)+1,offset=(f.qh&0x8000)?-1:1;
                for(int l=0;l<4;++l){const auto code=grid<Scaled>(f,l);
                    for(int j=0;j<8;++j){
                        const int q=std::bit_cast<std::int8_t>(static_cast<std::uint8_t>(code>>(8*j)));
                        const int w=((Scaled?q:8*q)+offset)*odd;
                        for(std::size_t s=0;s<Sessions;++s)sum[s][r]+=w*x[s]->blocks()[b].qs[32*g+8*l+j];
                    }
                }
            }
            const float dw=quant::IQ2XXS::scale(block);
            for(std::size_t s=0;s<Sessions;++s)total[s][r]+=(dw*x[s]->blocks()[b].d)*(static_cast<float>(sum[s][r])*0.125f);
        }
    }
    for(std::size_t s=0;s<Sessions;++s)for(std::size_t r=0;r<Rows;++r)y[s*stride+r]=total[s][r];
}
#if defined(__AVX2__)
template<int Bits> struct Ops;
template<> struct Ops<256> {
    using Vector=__m256i;
    struct Weights {Vector lo,hi;};
    using Input=Weights;
    [[nodiscard]] static Vector zero() noexcept {return _mm256_setzero_si256();}
    [[nodiscard]] static Weights weights(__m256i w,int odd) noexcept {
        const auto scale=_mm256_set1_epi16(static_cast<short>(odd));
        return {_mm256_mullo_epi16(_mm256_cvtepi8_epi16(_mm256_castsi256_si128(w)),scale),
                _mm256_mullo_epi16(_mm256_cvtepi8_epi16(_mm256_extracti128_si256(w,1)),scale)};
    }
    [[nodiscard]] static Input input(const std::int16_t* x) noexcept {
        return {_mm256_loadu_si256(reinterpret_cast<const __m256i*>(x)),_mm256_loadu_si256(reinterpret_cast<const __m256i*>(x+16))};
    }
    [[nodiscard]] static Vector add(Vector a,Weights w,Input x) noexcept {return q16_detail::dot(q16_detail::dot(a,w.lo,x.lo),w.hi,x.hi);}
    [[nodiscard]] static int reduce(Vector a) noexcept {return q8k_detail::reduce(a);}
};
#if defined(__AVX512VNNI__) && defined(__AVX512BW__)
template<> struct Ops<512> {
    using Vector=__m512i;
    using Weights=Vector;using Input=Vector;
    [[nodiscard]] static Vector zero() noexcept {return _mm512_setzero_si512();}
    [[nodiscard]] static Weights weights(__m256i w,int odd) noexcept {return _mm512_mullo_epi16(_mm512_cvtepi8_epi16(w),_mm512_set1_epi16(static_cast<short>(odd)));}
    [[nodiscard]] static Input input(const std::int16_t* x) noexcept {return _mm512_loadu_si512(x);}
    [[nodiscard]] static Vector add(Vector a,Weights w,Input x) noexcept {return _mm512_dpwssd_epi32(a,w,x);}
    [[nodiscard]] static int reduce(Vector a) noexcept {return _mm512_reduce_add_epi32(a);}
};
#endif
template<int Bits,bool Scaled,std::size_t Sessions,std::size_t Rows>
inline void native(const std::uint8_t* raw,std::size_t row_bytes,
                   std::span<const quant::Q16Activations* const> x,float* y,std::size_t stride) noexcept {
    using O=Ops<Bits>;using V=typename O::Vector;
    float total[Sessions][Rows]{};
    for(std::size_t b=0;b<x[0]->blocks().size();++b){
        V sum[Sessions][Rows];
        const std::uint8_t* blocks[Rows];float dw[Rows];
        for(std::size_t r=0;r<Rows;++r){blocks[r]=raw+r*row_bytes+b*50;dw[r]=quant::IQ2XXS::scale(blocks[r]);}
        for(std::size_t s=0;s<Sessions;++s)for(std::size_t r=0;r<Rows;++r)sum[s][r]=O::zero();
        for(int g=0;g<8;++g){
            typename O::Weights weights[Rows];
            for(std::size_t r=0;r<Rows;++r){
                const auto f=quant::IQ1S::fields(blocks[r],32*g);
                auto w=q8k_detail::group(grid<Scaled>(f,0),grid<Scaled>(f,1),grid<Scaled>(f,2),grid<Scaled>(f,3));
                if constexpr(!Scaled){w=_mm256_add_epi8(w,w);w=_mm256_add_epi8(w,w);w=_mm256_add_epi8(w,w);}
                w=_mm256_add_epi8(w,_mm256_set1_epi8(static_cast<char>((f.qh&0x8000)?-1:1)));
                weights[r]=O::weights(w,2*((f.qh>>12)&7)+1);
            }
            for(std::size_t s=0;s<Sessions;++s){
                const auto input=O::input(x[s]->blocks()[b].qs.data()+32*g);
                for(std::size_t r=0;r<Rows;++r)sum[s][r]=O::add(sum[s][r],weights[r],input);
            }
        }
        for(std::size_t s=0;s<Sessions;++s)for(std::size_t r=0;r<Rows;++r)
            total[s][r]+=(dw[r]*x[s]->blocks()[b].d)*(static_cast<float>(O::reduce(sum[s][r]))*0.125f);
    }
    for(std::size_t s=0;s<Sessions;++s)for(std::size_t r=0;r<Rows;++r)y[s*stride+r]=total[s][r];
}
#endif
template<int Bits,bool Scaled,std::size_t Sessions,std::size_t Rows>
inline void tile(const std::uint8_t* raw,std::size_t row_bytes,std::span<const quant::Q16Activations* const> x,float* y,std::size_t stride) noexcept {
#if defined(__AVX2__)
    if constexpr(Bits!=0)native<Bits,Scaled,Sessions,Rows>(raw,row_bytes,x,y,stride);
    else
#endif
        scalar<Sessions,Rows,Scaled>(raw,row_bytes,x,y,stride);
}
}
// Borrowed prepared views and weights outlive this call; output cannot alias either.
// Output is [active_session][stride], with rows meaningful values. Inactive lanes are never run.
// Bits 512 is refused when unavailable; 256 has a portable scalar fallback.
template<int Bits=256,bool Scaled=true>
[[nodiscard]] inline bool project_iq1_q16_tile(std::span<const std::uint8_t> raw,
        std::span<const quant::Q16Activations* const> x,std::span<float> y,std::size_t rows,std::size_t stride) noexcept {
    if constexpr(!iq1_q16_tile_available<Bits>)return false;
    else {
        if(x.empty() || x.size()>IQ1_Q16_TILE_MAX_SESSIONS || !rows || stride<rows || !x[0] || !x[0]->width())return false;
        const auto width=x[0]->width();
        for(const auto* p:x)if(!p || p->width()!=width || width%256)return false;
        if(width/256>std::numeric_limits<std::size_t>::max()/quant::IQ1S::block.bytes)return false;
        const auto row_bytes=width/256*quant::IQ1S::block.bytes;
        if(rows>raw.size()/row_bytes || (x.size()==2 && stride>std::numeric_limits<std::size_t>::max()-rows))return false;
        if(y.size()<(x.size()-1)*stride+rows)return false;
        for(std::size_t r=0;r<rows;r+=2){
            const auto* packed=raw.data()+r*row_bytes;float* out=y.data()+r;
            if(x.size()==1){
                if(rows-r>=2)q16_tile_detail::tile<Bits,Scaled,1,2>(packed,row_bytes,x,out,stride);
                else q16_tile_detail::tile<Bits,Scaled,1,1>(packed,row_bytes,x,out,stride);
            }else{
                if(rows-r>=2)q16_tile_detail::tile<Bits,Scaled,2,2>(packed,row_bytes,x,out,stride);
                else q16_tile_detail::tile<Bits,Scaled,2,1>(packed,row_bytes,x,out,stride);
            }
        }
        return true;
    }
}
}
