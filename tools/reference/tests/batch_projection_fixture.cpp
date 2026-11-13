// Integration gate: serial/batched arithmetic across formats, modes, widths and tile tails.
#include "sub0/dense_quant.hpp"
#include <bit>
#include <cmath>
#include <cstdio>
#include <stdexcept>

using sub0::denseq::Iq1Kernel;
void require(bool value) { if (!value) throw std::runtime_error("batch projection contract"); }
template<std::size_t Tile>
void check(std::uint32_t type, std::size_t width, Iq1Kernel kernel) {
    constexpr std::size_t rows=3,tokens=17,stride=5;
    const auto spec=sub0::gguf::block_spec(type);
    std::vector<std::uint8_t> packed(1+rows*(width/spec.elems)*spec.bytes);
    for(std::size_t i=1;i<packed.size();++i)packed[i]=static_cast<std::uint8_t>((i*37+i/7)%64);
    std::vector<float> x(tokens*width),serial(tokens*stride,123.f),batch(serial),scratch;
    scratch.reserve(256);
    for(std::size_t i=0;i<x.size();++i)x[i]=static_cast<float>(static_cast<int>(i*17%101)-50)/37.f;
    const std::span<const std::uint8_t> raw{packed.data()+1,packed.size()-1};
    for(std::size_t t=0;t<tokens;++t)
        require(sub0::denseq::project(type,raw,{x.data()+t*width,width},{serial.data()+t*stride,rows},scratch,kernel));
    require(sub0::denseq::project_batch<Tile>(type,raw,x.data(),width,tokens,batch.data(),rows,stride,scratch,kernel));
    for(std::size_t i=0;i<serial.size();++i){
        require(std::isfinite(serial[i]));
        require(std::bit_cast<std::uint32_t>(serial[i])==std::bit_cast<std::uint32_t>(batch[i]));
    }
    require(!sub0::denseq::project_batch<Tile>(type,raw.first(raw.size()-1),x.data(),width,tokens,batch.data(),rows,stride,scratch,kernel));
    require(!sub0::denseq::project_batch<Tile>(type,raw,x.data(),width,tokens,batch.data(),rows,rows-1,scratch,kernel));
    require(!sub0::denseq::project_batch<Tile>(type,raw,nullptr,width,tokens,batch.data(),rows,stride,scratch,kernel));
    require(!sub0::denseq::project_batch<Tile>(type,raw,x.data(),width,std::numeric_limits<std::size_t>::max(),batch.data(),rows,stride,scratch,kernel));
}
int main() try {
    for(const auto& entry:sub0::gguf::kDequant)
        for(auto width:{std::size_t{256},std::size_t{5120},std::size_t{17408}})
            for(auto kernel:{Iq1Kernel::Baseline,Iq1Kernel::Scalar,Iq1Kernel::Native,Iq1Kernel::NativeAll,Iq1Kernel::FusedIQ2}) {
                check<8>(static_cast<std::uint32_t>(entry.type),width,kernel);
                check<16>(static_cast<std::uint32_t>(entry.type),width,kernel);
                check<32>(static_cast<std::uint32_t>(entry.type),width,kernel);
            }
    std::puts("batch projection parity/validation passed");
} catch(const std::exception& e) {std::fprintf(stderr,"%s\n",e.what());return 1;}
