// Independent Q16 oracle, session isolation and C++ allocation gate.
#include "../../../benchmarks/q16_fixture.hpp"
#include <atomic>
#include <barrier>
#include <bit>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <thread>

namespace {
std::atomic<bool> count_allocations{false};
std::atomic<std::size_t> allocations{0};
}
void* operator new(std::size_t n) {
    if(count_allocations.load(std::memory_order_relaxed))++allocations;
    if(void* p=std::malloc(n?n:1))return p;
    throw std::bad_alloc();
}
void* operator new[](std::size_t n) {return ::operator new(n);}
void operator delete(void* p) noexcept {std::free(p);}
void operator delete[](void* p) noexcept {std::free(p);}
void operator delete(void* p,std::size_t) noexcept {std::free(p);}
void operator delete[](void* p,std::size_t) noexcept {std::free(p);}

void check_concurrent_q16(const sub0bench::Q16Fixture& fixture) {
    using sub0::denseq::project_q16;
    const auto& base=fixture.base;
    std::vector<float> x2(base.width);
    for(std::size_t i=0;i<x2.size();++i)x2[i]=base.x[(i+1)%x2.size()];
    sub0::quant::Q16Activations a(base.width),b(base.width);
    sub0bench::q8k_require(a.quantize(base.x) && b.quantize(x2),"Q16 concurrent setup failed");
    const std::span<const std::uint8_t> weights{base.packed.data()+1,base.packed.size()-1};
    std::vector<float> expectedA(base.rows),expectedB(base.rows),outA(base.rows),outB(base.rows);
    sub0bench::q8k_require(project_q16<true>(base.type,weights,a,expectedA) &&
                           project_q16<true>(base.type,weights,b,expectedB),"Q16 serial references failed");
    std::barrier start(2);std::atomic<bool> ok{true};
    const auto work=[&](const auto& prepared,auto& out,const auto& expected) {
        start.arrive_and_wait();
        for(int repeat=0;repeat<8;++repeat) {
            if(!project_q16<true>(base.type,weights,prepared,out)){ok=false;return;}
            for(std::size_t i=0;i<out.size();++i)
                if(std::bit_cast<std::uint32_t>(out[i])!=std::bit_cast<std::uint32_t>(expected[i]))ok=false;
        }
    };
    std::jthread first([&]{work(a,outA,expectedA);});std::jthread second;
    try {second=std::jthread([&]{work(b,outB,expectedB);});}
    catch(...){start.arrive_and_drop();throw;}
    first.join();second.join();
    sub0bench::q8k_require(ok.load(),"Q16 concurrent sessions differ from serial");
}

int main() try {
    sub0bench::check_q16_activation_contract();
    {
        sub0::quant::Q16Activations source(256);std::vector<float> input(256,1.f);
        sub0bench::q8k_require(source.quantize(input),"move setup failed");
        auto moved=std::move(source);
        sub0bench::q8k_require(source.width()==0 && source.blocks().empty() && moved.width()==256,"move leaves invalid prepared span");
        source=std::move(moved);
        sub0bench::q8k_require(moved.width()==0 && moved.blocks().empty() && source.width()==256,"move assignment leaves invalid prepared span");
    }
    sub0bench::check_q16_wide_accumulator();
    for(std::size_t width:{std::size_t{256},std::size_t{5120},std::size_t{17408}})
        for(std::uint32_t type:{std::uint32_t{19},std::uint32_t{16}}) {
            sub0bench::Q16Fixture f(type,width);
            allocations=0;count_allocations=true;
            f.prepare();f.run<false>();f.run<true>();
            count_allocations=false;
            sub0bench::q8k_require(allocations==0,"allocation in Q16 quantize/projection hot path");
            check_concurrent_q16(f);
        }
    std::puts("Q16 encoder and projection fixtures passed");
} catch(const std::exception& e) {std::fprintf(stderr,"%s\n",e.what());return 1;}
