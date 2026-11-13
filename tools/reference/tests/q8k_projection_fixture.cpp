// Independent oracle and no-allocation gate; no Catch dependency.
#include "../../../benchmarks/q8k_fixture.hpp"
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
    if(count_allocations.load(std::memory_order_relaxed)) ++allocations;
    if(void* p=std::malloc(n?n:1))return p;
    throw std::bad_alloc();
}
void* operator new[](std::size_t n) { return ::operator new(n); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p,std::size_t) noexcept { std::free(p); }
void operator delete[](void* p,std::size_t) noexcept { std::free(p); }

// Independent sessions hold separate prepared inputs and outputs, sharing only immutable weights.
void check_concurrent_sessions(const sub0bench::Q8KFixture& fixture) {
    using sub0::denseq::project_q8k;
    std::vector<float> x2(fixture.width);
    for(std::size_t i=0;i<x2.size();++i)x2[i]=fixture.x[(i+1)%x2.size()];
    sub0::quant::Q8KActivations a(fixture.width),b(fixture.width);
    sub0bench::q8k_require(a.quantize(fixture.x) && b.quantize(x2),"concurrent setup failed");
    const std::span<const std::uint8_t> weights{fixture.packed.data()+1,fixture.packed.size()-1};
    std::vector<float> expectedA(fixture.rows),expectedB(fixture.rows),outA(fixture.rows),outB(fixture.rows);
    sub0bench::q8k_require(project_q8k<true>(fixture.type,weights,a,expectedA) &&
                            project_q8k<true>(fixture.type,weights,b,expectedB),"concurrent reference failed");
    std::barrier start(2);
    std::atomic<bool> ok{true};
    const auto work=[&](const auto& prepared,auto& out,const auto& expected) {
        start.arrive_and_wait();
        for(int repeat=0;repeat<8;++repeat) {
            if(!project_q8k<true>(fixture.type,weights,prepared,out)) { ok=false; return; }
            for(std::size_t i=0;i<out.size();++i)
                if(std::bit_cast<std::uint32_t>(out[i])!=std::bit_cast<std::uint32_t>(expected[i])) ok=false;
        }
    };
    std::jthread first([&]{work(a,outA,expectedA);});
    std::jthread second;
    try { second=std::jthread([&]{work(b,outB,expectedB);}); }
    catch (...) { start.arrive_and_drop(); throw; }
    first.join();second.join();
    sub0bench::q8k_require(ok.load(),"concurrent independent sessions differ from serial");
}

int main() try {
    sub0bench::check_q8k_activation_contract();
    {
        sub0::quant::Q8KActivations source(256);std::vector<float> input(256,1.f);
        sub0bench::q8k_require(source.quantize(input),"move setup failed");
        auto moved=std::move(source);
        sub0bench::q8k_require(source.width()==0 && source.blocks().empty() && moved.width()==256,"move leaves invalid prepared span");
        source=std::move(moved);
        sub0bench::q8k_require(moved.width()==0 && moved.blocks().empty() && source.width()==256,"move assignment leaves invalid prepared span");
    }
    for(std::size_t width:{std::size_t{256},std::size_t{5120},std::size_t{17408}})
        for(std::uint32_t type:{std::uint32_t{19},std::uint32_t{16}}) {
            sub0bench::Q8KFixture f(type,width);
            allocations=0;count_allocations=true;
            f.prepare();f.run_prepared<false>();f.run_prepared<true>();
            count_allocations=false;
            sub0bench::q8k_require(allocations==0,"allocation in quantize/projection hot path");
            check_concurrent_sessions(f);
        }
    std::puts("Q8_K encoder and projection fixtures passed");
} catch(const std::exception& e) {std::fprintf(stderr,"%s\n",e.what());return 1;}
