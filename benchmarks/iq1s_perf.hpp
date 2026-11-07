// Warm-cache row slices with independent packed-byte decode/dot; not end-to-end model speed.
#pragma once
#include <bench_runner.hpp>
#include "sub0/dense_quant.hpp"
#include <bit>
#include <cmath>
#include <stdexcept>

namespace {
struct Iq1Fixture {
    static constexpr std::size_t rows=32;
    std::size_t width;
    std::vector<std::uint8_t> packed;
    std::vector<float> x,y,scratch;
    std::vector<double> expected;
    explicit Iq1Fixture(std::size_t n):width(n),packed(1+rows*n/256*50),x(n+1),y(rows),expected(rows){
        scratch.reserve(256);
        for(std::size_t i=0;i<n;++i)x[i+1]=static_cast<float>(static_cast<int>(i*17%101)-50)/37.f;
        constexpr std::uint16_t scales[]{0,0x0001,0x0400,0x3400,0x3c00,0xbc00};
        for(std::size_t b=0;b<rows*n/256;++b){
            auto* p=packed.data()+1+b*50;const auto d=scales[b%6];p[0]=d&255;p[1]=d>>8;
            for(int ib=0;ib<8;++ib){
                std::uint16_t qh=static_cast<std::uint16_t>(((b+ib)%8)<<12|((b+ib)%2)<<15);
                for(int l=0;l<4;++l){const auto idx=(b*32+ib*4+l)%2048;p[2+4*ib+l]=idx&255;qh|=static_cast<std::uint16_t>((idx>>8)<<(3*l));}
                p[34+2*ib]=qh&255;p[35+2*ib]=qh>>8;
            }
        }
        // This oracle does not call IQ1S fields/dequantize/dot or denseq::project.
        // Use double accumulation, with byte-order and scale equations derived from the wire layout.
        for(std::size_t row=0;row<rows;++row)for(std::size_t b=0;b<n/256;++b){
            const auto* p=packed.data()+1+(row*n/256+b)*50;
            const auto h=static_cast<std::uint16_t>(p[0]|p[1]<<8);
            const double d=half(h);
            for(int ib=0;ib<8;++ib){const unsigned qh=p[34+2*ib]|p[35+2*ib]<<8;
                for(int l=0;l<4;++l){const auto idx=p[2+4*ib+l]|((qh>>(3*l)&7)<<8);const auto code=sub0::gguf::IQ1S_GRID[idx];
                    for(int j=0;j<8;++j){const auto q=std::bit_cast<std::int8_t>(static_cast<std::uint8_t>(code>>(8*j)));
                        const double w=d*(2*(qh>>12&7)+1)*(static_cast<double>(q)+(qh>>15?-0.125:0.125));
                        expected[row]+=x[1+b*256+ib*32+l*8+j]*w;
                    }
                }
            }
        }
        for(auto k:{sub0::denseq::Iq1Kernel::Baseline,sub0::denseq::Iq1Kernel::Scalar,sub0::denseq::Iq1Kernel::Native}){
            run(k);double squared_error=0,squared_reference=0;
            for(std::size_t i=0;i<rows;++i){
                if(!std::isfinite(y[i]))throw std::runtime_error("nonfinite IQ1_S result");
                const double e=y[i]-expected[i];squared_error+=e*e;squared_reference+=expected[i]*expected[i];
            }
            // A component close to zero can have cancellation error even in the original scalar
            // baseline. Use vector relative L2 against the independent double dot, not relative
            // error divided by that one near-zero component. Production logit limits are unchanged.
            if(std::sqrt(squared_error)>2e-5*std::max(1.,std::sqrt(squared_reference)))
                throw std::runtime_error("IQ1_S independent double dot fixture mismatch");
        }
        if(sub0::denseq::project(19,{packed.data()+1,packed.size()-2},{x.data()+1,width},y,scratch))throw std::runtime_error("accepted truncated projection");
        if(sub0::denseq::project(19,{packed.data()+1,packed.size()-1},{x.data()+1,width-1},y,scratch))throw std::runtime_error("accepted partial-block row");
        std::vector<float> decoded;
        for(std::size_t count:{std::size_t{0},std::size_t{1},std::size_t{255},std::size_t{256},std::size_t{257}}){
            if(!sub0::quant::IQ1S::dequantize({packed.data()+1,packed.size()-1},count,decoded)||decoded.size()!=count)throw std::runtime_error("partial dequantization rejected");
        }
        if(sub0::quant::IQ1S::dequantize({packed.data()+1,49},1,decoded))throw std::runtime_error("truncated dequantization accepted");
    }
    static double half(std::uint16_t h){
        const int sign=h&0x8000?-1:1,exponent=h>>10&31,mantissa=h&1023;
        return sign*std::ldexp(exponent?1.+mantissa/1024.:mantissa/1024.,exponent?exponent-15:-14);
    }
    void run(sub0::denseq::Iq1Kernel k){
        if(!sub0::denseq::project(19,{packed.data()+1,packed.size()-1},{x.data()+1,width},y,scratch,k))throw std::runtime_error("projection rejected valid fixture");
    }
};
}
inline void add_iq1s_benchmarks(sub0bench::BenchRunner& runner){
    Iq1Fixture small(256),qwen(5120),down(17408);
    for(auto* f:{&small,&qwen,&down})for(auto k:{sub0::denseq::Iq1Kernel::Baseline,sub0::denseq::Iq1Kernel::Scalar,sub0::denseq::Iq1Kernel::Native}){
        const auto label=k==sub0::denseq::Iq1Kernel::Baseline?"baseline":k==sub0::denseq::Iq1Kernel::Scalar?"fused scalar":"fused native";
        const auto name="IQ1_S 32x"+std::to_string(f->width)+", "+label;
        runner.run(name.c_str(),[&]{f->run(k);ankerl::nanobench::doNotOptimizeAway(f->y.data());});
    }
}
