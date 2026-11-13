// Independent Q8_K activation and packed IQ1_S/IQ2_XXS projection oracle.
#pragma once
#include "sub0/dense_quant.hpp"
#include "sub0/dense_quant_q8k.hpp"
#include <algorithm>
#include <bit>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace sub0bench {
inline void q8k_require(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }

// Pinned ggml quantize_row_q8_K_ref: signed extremum and magic-bias rounding.
inline sub0::quant::Q8KBlock q8k_reference(const float* x) {
    sub0::quant::Q8KBlock out{};
    float amax=0.f, extremum=0.f;
    for (int i=0;i<256;++i) if (std::fabs(x[i])>amax) { amax=std::fabs(x[i]); extremum=x[i]; }
    if (amax==0.f) return out;
    const float inv=-127.f/extremum;
    out.d=1.f/inv;
    for(int i=0;i<256;++i) {
        // The reference calls nearest_int(float) with an already-rounded product.
        // A volatile store prevents contraction with its subsequent bias addition.
        volatile float product=x[i]*inv;
        const float biased=product+12582912.f;
        const int q=std::min(127,static_cast<int>(std::bit_cast<std::uint32_t>(biased)&0x007fffffu)-0x00400000);
        out.qs[static_cast<std::size_t>(i)]=static_cast<std::int8_t>(q);
        out.bsums[static_cast<std::size_t>(i/16)]+=static_cast<std::int16_t>(q);
    }
    return out;
}
inline double q8k_half(std::uint16_t h) {
    const int e=(h>>10)&31, m=h&1023;
    return (h&0x8000?-1.:1.)*std::ldexp(e?1.+m/1024.:m/1024.,e?e-15:-14);
}

struct Q8KFixture {
    static constexpr std::size_t rows=32;
    std::uint32_t type;
    std::size_t width,blockBytes;
    std::vector<std::uint8_t> packed;
    std::vector<float> x,y,scratch;
    std::vector<double> expected;
    sub0::quant::Q8KActivations activations;

    Q8KFixture(std::uint32_t format,std::size_t n):type(format),width(n),blockBytes(format==19?50:66),
        packed(1+rows*(n/256)*blockBytes),x(n),y(rows),expected(rows),activations(n) {
        q8k_require(format==19||format==16,"unsupported fixture type");
        scratch.reserve(256);
        for(std::size_t i=0;i<n;++i)x[i]=static_cast<float>(static_cast<int>(i*17%101)-50)/37.f;
        for(std::size_t b=0;b<rows*n/256;++b) {
            auto* p=packed.data()+1+b*blockBytes;
            constexpr std::uint16_t ds[]{0,0x0001,0x0400,0x3400,0x3c00,0xbc00};
            const auto h=ds[b%6];p[0]=static_cast<std::uint8_t>(h);p[1]=static_cast<std::uint8_t>(h>>8);
            for(int g=0;g<8;++g) {
                if(type==19) {
                    std::uint16_t qh=static_cast<std::uint16_t>(((b+g)%8)<<12|((b+g)%2)<<15);
                    for(int l=0;l<4;++l) {
                        const auto index=(b*32+static_cast<std::size_t>(g)*4+l)%2048;
                        p[2+4*g+l]=static_cast<std::uint8_t>(index);
                        qh|=static_cast<std::uint16_t>((index>>8)<<(3*l));
                    }
                    p[34+2*g]=static_cast<std::uint8_t>(qh);p[35+2*g]=static_cast<std::uint8_t>(qh>>8);
                } else {
                    std::uint32_t high=static_cast<std::uint32_t>((b+g)%16)<<28;
                    for(int l=0;l<4;++l) {
                        p[2+8*g+l]=static_cast<std::uint8_t>((b*32+g*4+l)%256);
                        high|=static_cast<std::uint32_t>((b+11*g+29*l)%128)<<(7*l);
                    }
                    for(int j=0;j<4;++j)p[6+8*g+j]=static_cast<std::uint8_t>(high>>(8*j));
                }
            }
        }
        prepare();
        // Independent packed-wire decoder and double dot against reconstructed Q8_K activations.
        for(std::size_t b=0;b<n/256;++b) {
            const auto ref=q8k_reference(x.data()+256*b);
            const auto& actual=activations.blocks()[b];
            q8k_require(std::bit_cast<std::uint32_t>(ref.d)==std::bit_cast<std::uint32_t>(actual.d) &&
                         ref.qs==actual.qs && ref.bsums==actual.bsums,"Q8_K encoder differs from reference");
        }
        for(std::size_t row=0;row<rows;++row)for(std::size_t b=0;b<n/256;++b) {
            const auto* p=packed.data()+1+(row*n/256+b)*blockBytes;
            const double d=q8k_half(static_cast<std::uint16_t>(p[0]|p[1]<<8));
            const auto& a=activations.blocks()[b];
            for(int g=0;g<8;++g)for(int l=0;l<4;++l)for(int j=0;j<8;++j) {
                double w;
                if(type==19) {
                    const unsigned qh=p[34+2*g]|p[35+2*g]<<8;
                    const auto index=p[2+4*g+l]|((qh>>(3*l)&7)<<8);
                    const auto code=sub0::gguf::IQ1S_GRID[index];
                    const auto q=std::bit_cast<std::int8_t>(static_cast<std::uint8_t>(code>>(8*j)));
                    w=d*(2*(qh>>12&7)+1)*(static_cast<double>(q)+(qh&0x8000?-0.125:0.125));
                } else {
                    const auto* q=p+2+8*g;
                    const std::uint32_t high=static_cast<std::uint32_t>(q[4])|static_cast<std::uint32_t>(q[5])<<8|
                        static_cast<std::uint32_t>(q[6])<<16|static_cast<std::uint32_t>(q[7])<<24;
                    const auto code=sub0::gguf::IQ2XXS_GRID[q[l]];
                    const auto signs=sub0::gguf::KSIGNS_IQ2XS[(high>>(7*l))&127];
                    w=d*(0.5+static_cast<double>(high>>28))*0.25*static_cast<double>((code>>(8*j))&255)*((signs>>j)&1?-1.:1.);
                }
                expected[row]+=w*static_cast<double>(a.d)*a.qs[static_cast<std::size_t>(g*32+l*8+j)];
            }
        }
        run_prepared<false>(); check();
        run_prepared<true>(); check();
        q8k_require(!sub0::denseq::project_q8k<false>(type,{packed.data()+1,packed.size()-2},activations,y),"truncated weights accepted");
        q8k_require(!sub0::denseq::project_q8k<true>(1,{packed.data()+1,packed.size()-1},activations,y),"unsupported format accepted");
    }
    void check() const {
        double err=0,norm=0;
        for(std::size_t i=0;i<rows;++i) {
            q8k_require(std::isfinite(y[i]),"nonfinite Q8_K projection");
            const double e=y[i]-expected[i];err+=e*e;norm+=expected[i]*expected[i];
        }
        q8k_require(std::sqrt(err)<=2e-5*std::max(1.,std::sqrt(norm)),"Q8_K independent double dot mismatch");
    }
    void prepare() {
        q8k_require(activations.quantize(x) && activations.width()==width && activations.blocks().size()==width/256,
                     "Q8_K quantize rejected valid input");
    }
    template<bool Native> void run_prepared() {
        q8k_require(sub0::denseq::project_q8k<Native>(type,{packed.data()+1,packed.size()-1},activations,y),"Q8_K projection rejected");
    }
    void run_float() {
        q8k_require(sub0::denseq::project(type,{packed.data()+1,packed.size()-1},x,y,scratch,
                                          sub0::denseq::Iq1Kernel::FusedIQ2),"float projection rejected");
    }
};

inline void check_q8k_activation_contract() {
    sub0::quant::Q8KActivations a(512);
    std::vector<float> x(256,0.f);
    const auto verify=[&] {
        q8k_require(a.quantize(x),"Q8_K edge quantization rejected");
        const auto ref=q8k_reference(x.data());
        const auto& got=a.blocks()[0];
        q8k_require(ref.qs==got.qs && ref.bsums==got.bsums &&
                     std::bit_cast<std::uint32_t>(ref.d)==std::bit_cast<std::uint32_t>(got.d),"Q8_K edge quantization mismatch");
        q8k_require(std::none_of(got.qs.begin(),got.qs.end(),[](std::int8_t q) { return q == -128; }),
                     "Q8_K encoder emitted -128");
    };
    verify();
    x[0]=127.f;x[1]=-127.f;x[2]=0.5f;x[3]=1.5f;x[4]=-0.5f;x[5]=-1.5f;verify();
    x[0]=-127.f;x[1]=127.f;verify();
    std::fill(x.begin(),x.end(),0.f);
    x[0]=16.9456787109375f;x[1]=-6.604811668395996f;verify();
    q8k_require(a.blocks()[0].qs[1]==50,"Q8_K multiply rounded after bias");
    std::fill(x.begin(),x.end(),0.f);verify();
    std::vector<float> oversized(768,1.f);
    q8k_require(!a.quantize(oversized) && a.width()==0,"oversized activation accepted or stale");
    q8k_require(!a.quantize(std::span<const float>{}) && a.width()==0,"empty activation accepted or stale");
    q8k_require(!a.quantize(std::span<const float>{x.data(),255}) && a.width()==0,"partial block accepted or stale");
    for(float invalid:{std::numeric_limits<float>::quiet_NaN(),std::numeric_limits<float>::infinity(),
                       -std::numeric_limits<float>::infinity(),std::numeric_limits<float>::denorm_min()}) {
        x[0]=invalid;
        q8k_require(!a.quantize(x) && a.width()==0,"invalid activation accepted or stale");
        x[0]=0.f;
    }
    if (std::fesetround(FE_DOWNWARD)==0) {
        const bool accepted=a.quantize(x);
        const int restored=std::fesetround(FE_TONEAREST);
        q8k_require(restored==0 && !accepted && a.width()==0,"unsupported rounding mode accepted");
    }
    verify();
    sub0::quant::Q8KActivations unprepared(256);
    std::vector<std::uint8_t> raw(50);
    std::vector<float> y(1);
    q8k_require(!sub0::denseq::project_q8k<false>(19,raw,unprepared,y),"unprepared input accepted");
}
} // namespace sub0bench
