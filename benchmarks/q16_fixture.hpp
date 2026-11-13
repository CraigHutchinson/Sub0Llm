// Independent Q16 activation and packed IQ1_S/IQ2_XXS projection oracle.
#pragma once
#include "q8k_fixture.hpp"
#include "sub0/dense_quant_q16.hpp"
#include <algorithm>
#include <bit>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

namespace sub0bench {
inline sub0::quant::Q16Block q16_reference(const float* input) {
    sub0::quant::Q16Block result{};
    float max_abs=0.f, extremum=0.f;
    for(int i=0;i<256;++i) if(std::fabs(input[i])>max_abs) {
        max_abs=std::fabs(input[i]);extremum=input[i];
    }
    if(max_abs==0.f)return result;
    const float inverse=-32767.f/extremum;
    result.d=1.f/inverse;
    for(int i=0;i<256;++i) {
        volatile float product=input[i]*inverse;
        result.qs[static_cast<std::size_t>(i)]=static_cast<std::int16_t>(std::nearbyint(product));
    }
    return result;
}

struct Q16Fixture {
    Q8KFixture base;
    std::vector<float> y;
    std::vector<double> expected;
    sub0::quant::Q16Activations activations;

    Q16Fixture(std::uint32_t type,std::size_t width):base(type,width),y(base.rows),expected(base.rows),activations(width) {
        prepare();
        for(std::size_t b=0;b<width/256;++b) {
            const auto reference=q16_reference(base.x.data()+256*b);
            const auto& got=activations.blocks()[b];
            q8k_require(reference.qs==got.qs &&
                std::bit_cast<std::uint32_t>(reference.d)==std::bit_cast<std::uint32_t>(got.d),"Q16 encoder mismatch");
        }
        // Decode packed bytes directly, independently of production accessors and dot implementations.
        for(std::size_t row=0;row<base.rows;++row)for(std::size_t b=0;b<width/256;++b) {
            const auto* p=base.packed.data()+1+(row*width/256+b)*base.blockBytes;
            const double d=q8k_half(static_cast<std::uint16_t>(p[0]|p[1]<<8));
            const auto& a=activations.blocks()[b];
            for(int g=0;g<8;++g)for(int l=0;l<4;++l)for(int j=0;j<8;++j) {
                double w;
                if(type==19) {
                    const unsigned qh=p[34+2*g]|p[35+2*g]<<8;
                    const auto index=p[2+4*g+l]|((qh>>(3*l)&7)<<8);
                    const auto grid=sub0::gguf::IQ1S_GRID[index];
                    const auto q=std::bit_cast<std::int8_t>(static_cast<std::uint8_t>(grid>>(8*j)));
                    w=d*(2*(qh>>12&7)+1)*(static_cast<double>(q)+(qh&0x8000?-0.125:0.125));
                } else {
                    const auto* q=p+2+8*g;
                    const std::uint32_t high=static_cast<std::uint32_t>(q[4])|static_cast<std::uint32_t>(q[5])<<8|
                        static_cast<std::uint32_t>(q[6])<<16|static_cast<std::uint32_t>(q[7])<<24;
                    const auto grid=sub0::gguf::IQ2XXS_GRID[q[l]];
                    const auto signs=sub0::gguf::KSIGNS_IQ2XS[(high>>(7*l))&127];
                    w=d*(0.5+static_cast<double>(high>>28))*0.25*static_cast<double>((grid>>(8*j))&255)*((signs>>j)&1?-1.:1.);
                }
                expected[row]+=w*static_cast<double>(a.d)*a.qs[static_cast<std::size_t>(g*32+l*8+j)];
            }
        }
        run<false>();check();run<true>();check();
        q8k_require(!sub0::denseq::project_q16<false>(type,{base.packed.data()+1,base.packed.size()-2},activations,y),"Q16 truncated weights accepted");
        q8k_require(!sub0::denseq::project_q16<true>(1,{base.packed.data()+1,base.packed.size()-1},activations,y),"Q16 unsupported format accepted");
    }
    void prepare() {
        q8k_require(activations.quantize(base.x) && activations.width()==base.width &&
            activations.blocks().size()==base.width/256,"Q16 valid input rejected");
    }
    template<bool Native> void run() {
        q8k_require(sub0::denseq::project_q16<Native>(base.type,{base.packed.data()+1,base.packed.size()-1},activations,y),"Q16 projection rejected");
    }
    void check() const {
        double error=0,norm=0;
        for(std::size_t i=0;i<y.size();++i) {
            q8k_require(std::isfinite(y[i]),"nonfinite Q16 projection");
            const double e=y[i]-expected[i];error+=e*e;norm+=expected[i]*expected[i];
        }
        q8k_require(std::sqrt(error)<=2e-5*std::max(1.,std::sqrt(norm)),"Q16 independent double dot mismatch");
    }
};

inline void check_q16_activation_contract() {
    sub0::quant::Q16Activations a(512);
    std::vector<float> x(256,0.f);
    const auto verify=[&] {
        q8k_require(a.quantize(x) && a.width()==256,"Q16 edge input rejected");
        const auto ref=q16_reference(x.data());const auto& got=a.blocks()[0];
        q8k_require(ref.qs==got.qs && std::bit_cast<std::uint32_t>(ref.d)==std::bit_cast<std::uint32_t>(got.d),"Q16 edge encoder mismatch");
        q8k_require(std::none_of(got.qs.begin(),got.qs.end(),[](std::int16_t q){return q==-32768;}),"Q16 emitted -32768");
    };
    verify();
    x[0]=32767.f;x[1]=-32767.f;x[2]=0.5f;x[3]=1.5f;x[4]=-0.5f;x[5]=-1.5f;verify();
    x[0]=-32767.f;x[1]=32767.f;verify();
    std::fill(x.begin(),x.end(),0.f);verify();
    std::vector<float> oversized(768,1.f);
    q8k_require(!a.quantize(oversized) && a.width()==0,"Q16 oversized or stale");
    q8k_require(!a.quantize(std::span<const float>{}) && a.width()==0,"Q16 empty or stale");
    q8k_require(!a.quantize(std::span<const float>{x.data(),255}) && a.width()==0,"Q16 partial block or stale");
    for(float bad:{std::numeric_limits<float>::quiet_NaN(),std::numeric_limits<float>::infinity(),
                   -std::numeric_limits<float>::infinity(),std::numeric_limits<float>::denorm_min()}) {
        x[0]=bad;q8k_require(!a.quantize(x) && a.width()==0,"Q16 invalid input or stale");x[0]=0.f;
    }
    if(std::fesetround(FE_DOWNWARD)==0) {
        const bool accepted=a.quantize(x);const int restored=std::fesetround(FE_TONEAREST);
        q8k_require(restored==0 && !accepted && a.width()==0,"Q16 rounding mode accepted");
    }
    verify();
    sub0::quant::Q16Activations unprepared(256);
    std::vector<std::uint8_t> raw(50);std::vector<float> y(1);
    q8k_require(!sub0::denseq::project_q16<false>(19,raw,unprepared,y),"Q16 unprepared input accepted");
}

inline void check_q16_wide_accumulator() {
    // All signs positive and every group uses the largest-sum IQ2 grid, forcing
    // the whole 256-value block sum beyond signed 32-bit range.
    int largest=-1;std::uint8_t best=0;
    for(std::size_t i=0;i<256;++i) {
        const auto grid=sub0::gguf::IQ2XXS_GRID[i];int sum=0;
        for(int j=0;j<8;++j)sum+=static_cast<int>((grid>>(8*j))&255);
        if(sum>largest){largest=sum;best=static_cast<std::uint8_t>(i);}
    }
    std::vector<std::uint8_t> packed(66,0);packed[1]=0x3c;
    for(int g=0;g<8;++g) {
        for(int l=0;l<4;++l)packed[2+8*g+l]=best;
        packed[9+8*g]=0xf0; // scale 31/8, sign indices zero
    }
    std::vector<float> x(256,1.f),y(1);
    sub0::quant::Q16Activations a(256);
    q8k_require(a.quantize(x),"Q16 wide-accumulator input rejected");
    const std::int64_t integer=static_cast<std::int64_t>(largest)*32*31*32767;
    q8k_require(integer>std::numeric_limits<std::int32_t>::max(),"Q16 stress case is too small");
    const double expected=static_cast<double>(integer)/32767.0/8.0;
    q8k_require(sub0::denseq::project_q16<false>(16,packed,a,y) &&
        std::fabs(y[0]-expected)<2e-5*std::fabs(expected),"Q16 scalar accumulator overflow");
    q8k_require(sub0::denseq::project_q16<true>(16,packed,a,y) &&
        std::fabs(y[0]-expected)<2e-5*std::fabs(expected),"Q16 native accumulator overflow");
}
} // namespace sub0bench
