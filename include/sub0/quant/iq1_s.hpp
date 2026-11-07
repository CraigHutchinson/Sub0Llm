// IQ1_S: ggml-quants.c::dequantize_row_iq1_s, pinned cf659f240b232e15700b1721bbb43c1d973c52e9.
#pragma once
#include "sub0/quant/format.hpp"
#include "sub0/gguf_quant_tables.hpp"
#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace sub0::quant {
struct IQ1S {
    static constexpr TensorType type=TensorType::IQ1_S;
    static constexpr std::string_view name="IQ1_S";
    static constexpr BlockSpec block{256,50};
    struct Fields {
        std::uint16_t d_bits=0,qh=0;
        const std::uint8_t* qs=nullptr; // borrowed from the caller's mapped plane
    };
    [[nodiscard]] static Fields fields(const std::uint8_t* plane,std::uint64_t p) noexcept {
        const auto* blk=plane+(p/block.elems)*block.bytes;
        const auto group=(p%block.elems)/32;
        Fields f;f.qs=blk+2+4*group;
        std::memcpy(&f.d_bits,blk,2);std::memcpy(&f.qh,blk+34+2*group,2);return f;
    }
    [[nodiscard]] static int index(const Fields& f,int l) noexcept {
        return static_cast<int>(f.qs[l]|(((f.qh>>(3*l))&7)<<8));
    }
    [[nodiscard]] static std::uint64_t grid(const Fields& f,int l) noexcept {
        return gguf::IQ1S_GRID[static_cast<std::size_t>(index(f,l))];
    }
    [[nodiscard]] static float scale(const Fields& f,float d) noexcept {
        return d*static_cast<float>(2*((f.qh>>12)&7)+1);
    }
    [[nodiscard]] static float delta(const Fields& f) noexcept {
        return (f.qh&0x8000)?-gguf::IQ1S_DELTA:gguf::IQ1S_DELTA;
    }
    [[nodiscard]] static bool dequantize(std::span<const std::uint8_t> raw,std::uint64_t n,std::vector<float>& out) {
        const auto blocks=n/block.elems+(n%block.elems!=0);
        if(blocks>raw.size()/block.bytes || n>out.max_size())return false;
        out.resize(static_cast<std::size_t>(n));
        for(std::uint64_t b=0;b<blocks;++b){
            const auto base=b*block.elems;const float d=f16_to_f32(fields(raw.data(),base).d_bits);
            for(int ib=0;ib<8;++ib){
                const auto f=fields(raw.data(),base+32*ib);const float dl=scale(f,d),offset=delta(f);
                for(int l=0;l<4;++l){
                    const auto g=grid(f,l);
                    for(int j=0;j<8;++j){
                        const auto at=base+32*ib+8*l+j;if(at>=n)break;
                        const auto q=static_cast<std::int8_t>((g>>(8*j))&255);
                        out[static_cast<std::size_t>(at)]=dl*(static_cast<float>(q)+offset);
                    }
                }
            }
        }
        return true;
    }
    // Internal row core: caller validates whole blocks, input width and encoded span first.
    // Float inputs are retained. SIMD changes accumulation order, never packed weight decoding.
    template<bool Native>
    [[nodiscard]] static float dot_row(const std::uint8_t* row,const float* x,std::size_t width) noexcept {
        float sum=0;
#if defined(__AVX2__)
        __m256 accum=_mm256_setzero_ps();
#endif
        for(std::size_t b=0;b<width/block.elems;++b){
            const auto base=b*block.elems;const float d=f16_to_f32(fields(row,base).d_bits);
            for(int ib=0;ib<8;++ib){
                const auto f=fields(row,base+32*ib);const float dl=scale(f,d),offset=delta(f);
                for(int l=0;l<4;++l){
                    const auto g=grid(f,l);const auto at=base+32*ib+8*l;
#if defined(__AVX2__)
                    if constexpr(Native){
                        const auto q=_mm256_cvtepi8_epi32(_mm_cvtsi64_si128(static_cast<long long>(g)));
                        const auto w=_mm256_mul_ps(_mm256_set1_ps(dl),_mm256_add_ps(_mm256_cvtepi32_ps(q),_mm256_set1_ps(offset)));
                        accum=_mm256_add_ps(accum,_mm256_mul_ps(_mm256_loadu_ps(x+at),w));
                    } else
#endif
                    {
                        for(int j=0;j<8;++j){
                            const auto q=static_cast<std::int8_t>((g>>(8*j))&255);
                            const float w=dl*(static_cast<float>(q)+offset);sum+=x[at+j]*w;
                        }
                    }
                }
            }
        }
#if defined(__AVX2__)
        if constexpr(Native){alignas(32) float lanes[8];_mm256_store_ps(lanes,accum);for(float v:lanes)sum+=v;}
#endif
        return sum;
    }
};
static_assert(FormatModule<IQ1S>);
}
