// IQ2_XXS: ggml-quants.c::dequantize_row_iq2_xxs, pinned cf659f240b232e15700b1721bbb43c1d973c52e9.
#pragma once
#include "sub0/quant/format.hpp"
#include "sub0/gguf_quant_tables.hpp"
#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace sub0::quant {
struct IQ2XXS {
    static constexpr TensorType type=TensorType::IQ2_XXS;
    static constexpr std::string_view name="IQ2_XXS";
    static constexpr BlockSpec block{256,66};
    // Explicit little-endian loads also accept unaligned mapped rows.
    [[nodiscard]] static std::uint32_t word(const std::uint8_t* p) noexcept {
        return static_cast<std::uint32_t>(p[0]) | static_cast<std::uint32_t>(p[1])<<8 |
               static_cast<std::uint32_t>(p[2])<<16 | static_cast<std::uint32_t>(p[3])<<24;
    }
    [[nodiscard]] static float scale(const std::uint8_t* p) noexcept {
        return f16_to_f32(static_cast<std::uint16_t>(p[0]|p[1]<<8));
    }
    [[nodiscard]] static float group_scale(float d,std::uint32_t packed) noexcept {
        return d*(0.5f+static_cast<float>(packed>>28))*0.25f;
    }
    struct Fields {
        std::uint16_t d_bits=0;
        std::uint32_t aux[2]={0,0};
    };
    [[nodiscard]] static Fields fields(const std::uint8_t* plane,std::uint64_t p) noexcept {
        const auto* row=plane+(p/block.elems)*block.bytes;
        const auto* group=row+2+8*((p%block.elems)/32);
        return {static_cast<std::uint16_t>(row[0]|row[1]<<8),{word(group),word(group+4)}};
    }
    [[nodiscard]] static int sign_index(const Fields& f,int l) noexcept {
        return static_cast<int>((f.aux[1]>>(7*l))&127);
    }
    [[nodiscard]] static std::uint64_t grid(const Fields& f,int l) noexcept {
        return gguf::IQ2XXS_GRID[(f.aux[0]>>(8*l))&255];
    }
    [[nodiscard]] static bool dequantize(std::span<const std::uint8_t> raw,std::uint64_t n,std::vector<float>& out) {
        const auto blocks=n/block.elems+(n%block.elems!=0);
        if(blocks>raw.size()/block.bytes || n>out.max_size())return false;
        out.resize(static_cast<std::size_t>(n));
        for(std::uint64_t b=0;b<blocks;++b){
            const auto* row=raw.data()+b*block.bytes;const float d=scale(row);
            for(int ib=0;ib<8;++ib){
                const auto f=fields(row,32*ib);const auto packed=f.aux[1];
                const float db=group_scale(d,packed);
                for(int l=0;l<4;++l){
                    const auto code=grid(f,l);
                    const auto signs=gguf::KSIGNS_IQ2XS[sign_index(f,l)];
                    const auto base=b*block.elems+32*ib+8*l;
                    if(base+8<=n){
                        for(int j=0;j<8;++j){
                            const float mag=static_cast<float>((code>>(8*j))&255);
                            const float sign=1.f-2.f*static_cast<float>((signs>>j)&1);
                            out[static_cast<std::size_t>(base+j)]=db*mag*sign;
                        }
                    }else{
                        for(int j=0;j<8 && base+j<n;++j){
                            const float mag=static_cast<float>((code>>(8*j))&255);
                            out[static_cast<std::size_t>(base+j)]=db*mag*((signs>>j)&1?-1.f:1.f);
                        }
                    }
                }
            }
        }
        return true;
    }
    // Caller validates whole blocks and the encoded/input spans. No allocation or scratch writes.
    // The native core preserves NativeAll's eight-lane accumulation order and float activations.
    template<bool Native>
    [[nodiscard]] static float dot_row(const std::uint8_t* row,const float* x,std::size_t width) noexcept {
        float sum=0;
#if defined(__AVX2__)
        __m256 accum=_mm256_setzero_ps();
        const auto shifts=_mm256_setr_epi32(0,1,2,3,4,5,6,7);
#endif
        for(std::size_t b=0;b<width/block.elems;++b){
            const auto* block_row=row+b*block.bytes;const float d=scale(block_row);
            for(int ib=0;ib<8;++ib){
                const auto f=fields(block_row,32*ib);const auto packed=f.aux[1];
                const float db=group_scale(d,packed);
                for(int l=0;l<4;++l){
                    const auto code=grid(f,l);
                    const auto signs=gguf::KSIGNS_IQ2XS[sign_index(f,l)];
                    const auto at=b*block.elems+32*ib+8*l;
#if defined(__AVX2__)
                    if constexpr(Native){
                        const auto magnitude=_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_cvtsi64_si128(static_cast<long long>(code))));
                        const auto bits=_mm256_and_si256(_mm256_srlv_epi32(_mm256_set1_epi32(signs),shifts),_mm256_set1_epi32(1));
                        const auto negative=_mm256_castsi256_ps(_mm256_slli_epi32(bits,31));
                        const auto weights=_mm256_xor_ps(_mm256_mul_ps(_mm256_set1_ps(db),magnitude),negative);
                        accum=_mm256_add_ps(accum,_mm256_mul_ps(_mm256_loadu_ps(x+at),weights));
                    }else
#endif
                    {
                        for(int j=0;j<8;++j){
                            const float mag=static_cast<float>((code>>(8*j))&255);
                            const float weight=db*mag*((signs>>j)&1?-1.f:1.f);
                            sum+=x[at+j]*weight;
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
static_assert(FormatModule<IQ2XXS>);
}
