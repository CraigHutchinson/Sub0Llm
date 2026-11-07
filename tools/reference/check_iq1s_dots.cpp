// Independent real-row oracle: unmodified pinned ggml dequantization, then double dot.
#include "ggml.h"
#include "sub0/file_map.hpp"
#include "sub0/dense_quant.hpp"
#include <array>
#include <cmath>
#include <cstdio>
#include <memory>
int main(int argc,char**argv)try{
    if(argc!=2)return 2;
    std::unique_ptr<ggml_context,decltype(&ggml_free)> context(ggml_init({16*1024*1024,nullptr,true}),ggml_free);
    if(!context)throw std::runtime_error("ggml init");
    sub0::FileMap map;std::string error;if(!map.open(argv[1],error))throw std::runtime_error(error);
    const sub0::gguf::Reader r({map.data(),map.size()});if(!r.ok())throw std::runtime_error("GGUF parse");
    const auto* traits=ggml_get_type_traits(GGML_TYPE_IQ1_S);
    std::array<double,3> squared_error{},max_error{};double squared_reference=0;std::size_t rows=0;
    for(const auto& t:r.tensors()){
        if(t.type_raw!=19 || t.dims.size()!=2)continue;
        const auto n=static_cast<std::size_t>(t.dims[0]),height=static_cast<std::size_t>(t.dims[1]);
        if(!n || n%256 || !height)throw std::runtime_error("invalid IQ1_S shape");
        const auto raw=r.tensor_bytes(t);const auto row_bytes=n/256*50;
        std::vector<float> weights(n),decoded,x(n),scratch,y(1);scratch.reserve(256);
        for(std::size_t i=0;i<n;++i)x[i]=static_cast<float>(static_cast<int>(i*17%101)-50)/37.f;
        for(const auto row:{std::size_t{0},height/2,height-1}){
            const auto packed=raw.subspan(row*row_bytes,row_bytes);traits->to_float(packed.data(),weights.data(),static_cast<std::int64_t>(n));
            if(!sub0::quant::IQ1S::dequantize(packed,n,decoded)||std::memcmp(weights.data(),decoded.data(),n*sizeof(float)))throw std::runtime_error("real row decoder mismatch");
            double gold=0;for(std::size_t j=0;j<n;++j)gold+=static_cast<double>(weights[j])*x[j];
            squared_reference+=gold*gold;
            for(int k=0;k<3;++k){
                if(!sub0::denseq::project(19,packed,x,y,scratch,static_cast<sub0::denseq::Iq1Kernel>(k)))throw std::runtime_error("real row projection rejected");
                if(!std::isfinite(y[0]))throw std::runtime_error("nonfinite real row");
                const double diff=y[0]-gold;squared_error[k]+=diff*diff;max_error[k]=std::max(max_error[k],std::abs(diff));
            }
            ++rows;
        }
    }
    bool passed=rows>0;std::printf("{\"rows\":%zu,\"kernels\":[",rows);
    for(int k=0;k<3;++k){const double l2=std::sqrt(squared_error[k])/std::max(1.,std::sqrt(squared_reference));passed&=l2<=2e-5;
        std::printf("%s{\"kernel\":%d,\"relative_l2\":%.12g,\"max_abs\":%.12g}",k?",":"",k,l2,max_error[k]);}
    std::printf("],\"passed\":%s}\n",passed?"true":"false");return passed?0:1;
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
