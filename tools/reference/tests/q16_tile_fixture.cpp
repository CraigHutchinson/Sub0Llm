#include "../../../benchmarks/q16_tile_fixture.hpp"
#include <atomic>
#include <barrier>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <thread>
namespace {std::atomic<bool> counting{false};std::atomic<std::size_t> allocations{0};}
void* operator new(std::size_t n){if(counting.load(std::memory_order_relaxed))++allocations;if(auto* p=std::malloc(n?n:1))return p;throw std::bad_alloc();}
void* operator new[](std::size_t n){return ::operator new(n);}
void operator delete(void* p)noexcept{std::free(p);}
void operator delete[](void* p)noexcept{std::free(p);}
void operator delete(void* p,std::size_t)noexcept{std::free(p);}
void operator delete[](void* p,std::size_t)noexcept{std::free(p);}
template<int Bits,bool Scaled=true> void verify(sub0bench::Q16TileFixture& f){
    if constexpr(sub0::denseq::iq1_q16_tile_available<Bits>){
        std::fill(f.output.begin(),f.output.end(),12345.f);
        allocations=0;counting=true;f.prepare();f.run<Bits,Scaled>();counting=false;
        sub0bench::q8k_require(allocations==0,"allocation in tile hot path");f.check();
    }
}
void check_extreme_block(){
    int best=-1;unsigned index=0;
    for(unsigned i=0;i<2048;++i){int sum=0;for(int j=0;j<8;++j){const int q=std::bit_cast<std::int8_t>(static_cast<std::uint8_t>(sub0::gguf::IQ1S_GRID[i]>>(8*j)));sum+=std::abs(8*q+1);}if(sum>best){best=sum;index=i;}}
    std::vector<std::uint8_t> raw(100);std::vector<float> x(256);float expected=0;
    for(int row=0;row<2;++row){auto* wire=raw.data()+row*50;wire[1]=0x3c;
        const unsigned high=0x7000|((index>>8)*0x249);
        for(int g=0;g<8;++g){wire[34+2*g]=high&255;wire[35+2*g]=high>>8;
            for(int l=0;l<4;++l){wire[2+4*g+l]=index&255;for(int j=0;j<8;++j){const int q=std::bit_cast<std::int8_t>(static_cast<std::uint8_t>(sub0::gguf::IQ1S_GRID[index]>>(8*j)));x[32*g+8*l+j]=8*q+1>0?1.f:-1.f;}}
        }
    }
    expected=static_cast<float>(best*32*15)*0.125f;
    sub0::quant::Q16Activations a(256);sub0bench::q8k_require(a.quantize(x),"extreme activation refused");
    const std::array<const sub0::quant::Q16Activations*,2> inputs{&a,&a};std::array<float,4> out{};
    const auto check=[&]{for(float v:out)sub0bench::q8k_require(std::abs(v-expected)<2e-5f*expected,"extreme IQ1 accumulation failed");};
    out.fill(12345.f);
    sub0bench::q8k_require(sub0::denseq::project_iq1_q16_tile<0>(raw,inputs,out,2,2),"extreme scalar refused");check();
    out.fill(12345.f);
    sub0bench::q8k_require(sub0::denseq::project_iq1_q16_tile<256>(raw,inputs,out,2,2),"extreme 256 refused");check();
    if constexpr(sub0::denseq::iq1_q16_tile_available<512>){out.fill(12345.f);sub0bench::q8k_require(sub0::denseq::project_iq1_q16_tile<512>(raw,inputs,out,2,2),"extreme 512 refused");check();}
}
int main()try{
    check_extreme_block();
    using sub0bench::q8k_require;using sub0::denseq::project_iq1_q16_tile;
    for(std::size_t width:{256,5120,17408})for(std::size_t sessions:{1,2})for(std::size_t rows:{1,2,3,31,32}){
        sub0bench::Q16TileFixture f(width,sessions,rows);
        verify<0,false>(f);verify<0>(f);verify<256,false>(f);verify<256>(f);verify<512,false>(f);verify<512>(f);
    }
    for(std::size_t width:{0,264,17664}){
        bool rejected=false;try{sub0bench::Q16TileFixture invalid(width);}catch(const std::runtime_error&){rejected=true;}
        q8k_require(rejected,"invalid fixture width accepted");
    }
    sub0bench::Q16TileFixture f(256);
    // Runtime occupancy can drop without reading or writing the inactive session.
    f.sessions=1;f.inputs[1]=nullptr;verify<256>(f);verify<512>(f);
    f.sessions=2;f.inputs[1]=&f.second;
    const auto reject=[&](auto weights,auto inputs,auto out,std::size_t rows,std::size_t stride){
        std::fill(f.output.begin(),f.output.end(),-7654.f);
        q8k_require(!project_iq1_q16_tile(weights,inputs,out,rows,stride),"invalid tile accepted");
        q8k_require(std::all_of(f.output.begin(),f.output.end(),[](float v){return v==-7654.f;}),"refused tile wrote output");
    };
    f.run<256>();f.check();
    reject(f.weights(),f.active(),std::span<float>(f.output),0,f.stride);
    reject(f.weights(),f.active(),std::span<float>(f.output),f.rows,f.rows-1);
    reject(f.weights().first(f.weights().size()-1),f.active(),std::span<float>(f.output),f.rows,f.stride);
    reject(f.weights(),f.active(),std::span<float>(f.output).first(f.rows),f.rows,f.stride);
    reject(f.weights(),f.active(),std::span<float>(f.output),f.rows,SIZE_MAX);
    reject(f.weights(),std::span<const sub0::quant::Q16Activations* const>{},std::span<float>(f.output),f.rows,f.stride);
    std::array<const sub0::quant::Q16Activations*,3> three{f.inputs[0],f.inputs[1],f.inputs[0]};
    reject(f.weights(),std::span<const sub0::quant::Q16Activations* const>(three),std::span<float>(f.output),f.rows,f.stride);
    sub0::quant::Q16Activations unprepared(256),different(512);std::vector<float> x(512,1.f);
    q8k_require(different.quantize(x),"mismatch setup");
    for(const auto* bad:std::array<const sub0::quant::Q16Activations*,3>{nullptr,&unprepared,&different}){
        const std::array<const sub0::quant::Q16Activations*,2> inputs{f.inputs[0],bad};
        reject(f.weights(),std::span<const sub0::quant::Q16Activations* const>(inputs),std::span<float>(f.output),f.rows,f.stride);
    }
    if constexpr(!sub0::denseq::iq1_q16_tile_available<512>)q8k_require(!project_iq1_q16_tile<512>(f.weights(),f.active(),f.output,f.rows,f.stride),"unavailable ISA accepted");
    std::fill(f.output.begin(),f.output.end(),12345.f);f.run<256>();f.check();
    // Two callers share only immutable weights and prepared inputs; own their outputs.
    auto second_output=f.output;std::atomic<bool> ok{true};std::barrier start(2);
    const auto work=[&](std::span<float> out){start.arrive_and_wait();for(int i=0;i<32;++i){std::fill(out.begin(),out.end(),12345.f);if(!project_iq1_q16_tile(f.weights(),f.active(),out,f.rows,f.stride) || !std::equal(out.begin(),out.end(),f.output.begin()))ok=false;}};
    // Keep the expected vector immutable while both workers write separate buffers.
    auto first_output=f.output;
    std::jthread a([&]{work(first_output);});std::jthread b;
    try{b=std::jthread([&]{work(second_output);});}catch(...){start.arrive_and_drop();throw;}
    a.join();b.join();q8k_require(ok,"concurrent tile differs");
    std::puts("Q16 tile oracle, tails, occupancy, refusal, allocation and isolation passed");
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
