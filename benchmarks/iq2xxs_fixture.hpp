// Warm-cache IQ2_XXS row slices; independent wire decoder and double-dot oracle.
#pragma once
#include "sub0/dense_quant.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace sub0bench {
struct Iq2xxsFixture {
    static constexpr std::size_t rows=32;
    std::size_t width;
    std::vector<std::uint8_t> packed;
    std::vector<float> x,y,scratch,reference_weights;
    std::vector<double> expected;
    explicit Iq2xxsFixture(std::size_t n):width(n),packed(1+rows*n/256*66),x(n+1),y(rows),expected(rows){
        scratch.reserve(256);
        for(std::size_t i=0;i<n;++i)x[i+1]=static_cast<float>(static_cast<int>(i*17%101)-50)/37.f;
        constexpr std::uint16_t scales[]{0,0x8000,0x0001,0x0400,0x3400,0xbc00,0x7bff,0xfbff};
        for(std::size_t b=0;b<rows*n/256;++b){
            auto* p=packed.data()+1+b*66;const auto h=scales[b%8];p[0]=h&255;p[1]=h>>8;
            for(int ib=0;ib<8;++ib){
                const auto group=b*8+ib;
                std::uint32_t high=static_cast<std::uint32_t>((b+ib)%16)<<28;
                for(int l=0;l<4;++l){
                    p[2+8*ib+l]=static_cast<std::uint8_t>((group*4+l)%256);
                    high|=static_cast<std::uint32_t>((b+11*ib+29*l)%128)<<(7*l);
                }
                for(int j=0;j<4;++j)p[6+8*ib+j]=static_cast<std::uint8_t>(high>>(8*j));
            }
        }
        // Frozen wire-layout equations, independent of IQ2XXS accessors/dequantize/dot.
        reference_weights.resize(rows*n);
        for(std::size_t at=0;at<rows*n;++at){
            const auto* p=packed.data()+1+at/256*66;const auto ib=at%256/32,l=at%32/8,j=at%8;
            const unsigned h=p[0]|p[1]<<8;const int exponent=h>>10&31;
            const double magnitude=std::ldexp(exponent?1.+(h&1023)/1024.:(h&1023)/1024.,exponent?exponent-15:-14);
            const float d=static_cast<float>(h&0x8000?-magnitude:magnitude);
            const auto* q=p+2+8*ib;
            const std::uint32_t high=static_cast<std::uint32_t>(q[4])|static_cast<std::uint32_t>(q[5])<<8|static_cast<std::uint32_t>(q[6])<<16|static_cast<std::uint32_t>(q[7])<<24;
            const float db=d*(0.5f+static_cast<float>(high>>28))*0.25f;
            const auto code=sub0::gguf::IQ2XXS_GRID[q[l]],signs=static_cast<std::uint64_t>(sub0::gguf::KSIGNS_IQ2XS[(high>>(7*l))&127]);
            const float weight=db*static_cast<float>((code>>(8*j))&255)*((signs>>j)&1?-1.f:1.f);
            reference_weights[at]=weight;
            expected[at/n]+=static_cast<double>(weight)*x[1+at%n];
        }
        std::vector<float> decoded;
        if(!sub0::quant::IQ2XXS::dequantize({packed.data()+1,packed.size()-1},rows*n,decoded) ||
           std::memcmp(decoded.data(),reference_weights.data(),decoded.size()*sizeof(float)))throw std::runtime_error("IQ2_XXS independent wire decode mismatch");
        for(std::size_t count:{std::size_t{0},std::size_t{1},std::size_t{7},std::size_t{8},std::size_t{31},std::size_t{32},std::size_t{255},std::size_t{256},std::size_t{257},std::size_t{511}}){
            if(!sub0::quant::IQ2XXS::dequantize({packed.data()+1,packed.size()-1},count,decoded) || decoded.size()!=count ||
               std::memcmp(decoded.data(),reference_weights.data(),count*sizeof(float)))throw std::runtime_error("IQ2_XXS tail decode mismatch");
        }
        if(sub0::quant::IQ2XXS::dequantize({packed.data()+1,65},1,decoded) ||
           sub0::quant::IQ2XXS::dequantize({packed.data()+1,packed.size()-1},std::numeric_limits<std::uint64_t>::max(),decoded))throw std::runtime_error("IQ2_XXS invalid decode accepted");
        for(auto mode:{sub0::denseq::Iq1Kernel::Baseline,sub0::denseq::Iq1Kernel::NativeAll,sub0::denseq::Iq1Kernel::FusedIQ2}){
            run(mode);check(y);
        }
        run(sub0::denseq::Iq1Kernel::NativeAll);const auto prior=y;
        run(sub0::denseq::Iq1Kernel::FusedIQ2);
        if(std::memcmp(prior.data(),y.data(),rows*sizeof(float)))throw std::runtime_error("IQ2_XXS fused changed prior vector reduction");
        for(std::size_t row=0;row<rows;++row)y[row]=sub0::quant::IQ2XXS::dot_row<false>(packed.data()+1+row*n/256*66,x.data()+1,n);
        check(y);
        const auto capacity=scratch.capacity();run(sub0::denseq::Iq1Kernel::FusedIQ2);
        if(scratch.capacity()!=capacity)throw std::runtime_error("IQ2_XXS scratch grew");
        for(auto mode:{sub0::denseq::Iq1Kernel::NativeAll,sub0::denseq::Iq1Kernel::FusedIQ2}){
            if(sub0::denseq::project(16,{packed.data()+1,packed.size()-2},{x.data()+1,width},y,scratch,mode) ||
               sub0::denseq::project(16,{packed.data()+1,packed.size()-1},{x.data()+1,width-1},y,scratch,mode))throw std::runtime_error("IQ2_XXS invalid projection accepted");
        }
    }
    void check(const std::vector<float>& actual)const{
        double error=0,norm=0;
        for(std::size_t r=0;r<rows;++r){
            if(!std::isfinite(actual[r]))throw std::runtime_error("IQ2_XXS nonfinite dot");
            const double delta=actual[r]-expected[r];error+=delta*delta;norm+=expected[r]*expected[r];
        }
        if(std::sqrt(error)>2e-5*std::max(1.,std::sqrt(norm)))throw std::runtime_error("IQ2_XXS independent double dot mismatch");
    }
    void run(sub0::denseq::Iq1Kernel mode){
        if(!sub0::denseq::project(16,{packed.data()+1,packed.size()-1},{x.data()+1,width},y,scratch,mode))throw std::runtime_error("IQ2_XXS projection rejected");
    }
};
}
