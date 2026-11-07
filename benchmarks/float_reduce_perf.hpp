#pragma once
#include "sub0/dense_quant.hpp"
#include <cmath>
#include <stdexcept>

namespace sub0bench {
struct FloatReduceFixture {
    static constexpr std::size_t rows=4;
    std::uint32_t type;
    std::size_t width;
    std::vector<std::uint8_t> raw;
    std::vector<float> x,y,scratch;
    std::vector<double> expected;
    FloatReduceFixture(std::uint32_t t,std::size_t n):type(t),width(n),x(n+1),y(rows),expected(rows){
        const auto block=sub0::gguf::block_spec(type);const auto bytes=n/block.elems*block.bytes;
        raw.resize(1+rows*bytes);scratch.reserve(256);
        std::uint32_t seed=0x91347u+t;
        // Both positive/negative byte ranges exclude exponent-31 F16 scales. IQ1_M's
        // assembled high scale nibble is likewise finite. Other packed bits remain varied.
        for(auto& v:raw){seed=seed*1664525u+1013904223u;v=static_cast<std::uint8_t>(0x20+(seed>>16&63)|(seed>>24&128));}
        for(std::size_t i=0;i<n;++i)x[i+1]=static_cast<float>(static_cast<int>(i*17%101)-50)/37.f;
        std::vector<float> weights;
        for(std::size_t r=0;r<rows;++r){
            if(!sub0::gguf::dequant_fn(type)({raw.data()+1+r*bytes,bytes},n,weights))throw std::runtime_error("fixture decode");
            for(std::size_t j=0;j<n;++j)expected[r]+=static_cast<double>(weights[j])*x[j+1];
        }
        // Decoder fidelity is checked separately against pinned ggml real rows. This
        // oracle isolates the changed reduction using independent double accumulation.
        for(auto mode:{sub0::denseq::Iq1Kernel::Native,sub0::denseq::Iq1Kernel::NativeAll}){
            run(mode);double error=0,norm=0;
            for(std::size_t r=0;r<rows;++r){if(!std::isfinite(y[r]))throw std::runtime_error("nonfinite fixture");const double d=y[r]-expected[r];error+=d*d;norm+=expected[r]*expected[r];}
            if(std::sqrt(error)>2e-5*std::max(1.,std::sqrt(norm)))throw std::runtime_error("independent float reduction mismatch");
            if(sub0::denseq::project(type,{raw.data()+1,raw.size()-2},{x.data()+1,width},y,scratch,mode))throw std::runtime_error("truncated projection accepted");
            if(sub0::denseq::project(type,{raw.data()+1,raw.size()-1},{x.data()+1,width-1},y,scratch,mode))throw std::runtime_error("partial row accepted");
        }
    }
    void run(sub0::denseq::Iq1Kernel mode){
        if(!sub0::denseq::project(type,{raw.data()+1,raw.size()-1},{x.data()+1,width},y,scratch,mode))throw std::runtime_error("fixture projection rejected");
    }
};
inline void add_float_reduce_benchmarks(BenchRunner& runner){
    for(const auto& entry:sub0::gguf::kDequant)for(std::size_t width:{256,5120}){
        FloatReduceFixture fixture(static_cast<std::uint32_t>(entry.type),width);
        for(auto mode:{sub0::denseq::Iq1Kernel::Native,sub0::denseq::Iq1Kernel::NativeAll}){
            const auto name=std::string(sub0::gguf::find_format(fixture.type)->name)+" float 4x"+std::to_string(width)+", "+(mode==sub0::denseq::Iq1Kernel::Native?"prior":"vector");
            runner.run(name.c_str(),[&]{fixture.run(mode);ankerl::nanobench::doNotOptimizeAway(fixture.y.data());});
        }
    }
}
}
