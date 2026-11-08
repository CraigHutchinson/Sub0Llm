// Independent oracle: pinned ggml decode and double dots. Re-encode real rows for absent types.
#include "ggml.h"
#include "sub0/file_map.hpp"
#include "sub0/dense_quant.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <map>
#include <memory>

struct RowStats {
    std::size_t rows=0;
    double reference_norm=0;
    std::array<double,4> errors{},maximum{};
};
int main(int argc,char**argv)try{
    if(argc!=2)return 2;
    std::unique_ptr<ggml_context,decltype(&ggml_free)> context(ggml_init({16*1024*1024,nullptr,true}),ggml_free);
    if(!context)throw std::runtime_error("ggml init");
    sub0::FileMap map;std::string error;if(!map.open(argv[1],error))throw std::runtime_error(error);
    const sub0::gguf::Reader reader({map.data(),map.size()});if(!reader.ok())throw std::runtime_error("GGUF parse");
    std::map<std::pair<std::uint32_t,bool>,RowStats> stats;
    auto check=[&](std::uint32_t type,std::span<const std::uint8_t> raw,std::size_t n,bool reencoded){
        const auto* traits=ggml_get_type_traits(static_cast<ggml_type>(type));
        if(!traits || !traits->to_float)throw std::runtime_error("missing ggml decoder");
        std::vector<float> weights(n),decoded,x(n),scratch,y(1);scratch.reserve(256);
        traits->to_float(raw.data(),weights.data(),static_cast<std::int64_t>(n));
        if(!sub0::gguf::dequant_fn(type)(raw,n,decoded)||std::memcmp(weights.data(),decoded.data(),n*sizeof(float)))throw std::runtime_error("decoder mismatch");
        double gold=0;for(std::size_t j=0;j<n;++j){x[j]=static_cast<float>(static_cast<int>(j*17%101)-50)/37.f;gold+=static_cast<double>(weights[j])*x[j];}
        auto& s=stats[{type,reencoded}];++s.rows;s.reference_norm+=gold*gold;
        float prior_vector=0;
        for(int k=0;k<4;++k){
            const auto mode=k==0?sub0::denseq::Iq1Kernel::Baseline:k==1?sub0::denseq::Iq1Kernel::Native:k==2?sub0::denseq::Iq1Kernel::NativeAll:sub0::denseq::Iq1Kernel::FusedIQ2;
            if(!sub0::denseq::project(type,raw,x,y,scratch,mode)||!std::isfinite(y[0]))throw std::runtime_error("invalid row result");
            if(k==2)prior_vector=y[0];
            if(k==3 && std::memcmp(&prior_vector,y.data(),sizeof(float)))throw std::runtime_error("fused path differs from prior vector dot");
            const double d=y[0]-gold;s.errors[k]+=d*d;s.maximum[k]=std::max(s.maximum[k],std::abs(d));
        }
    };
    const sub0::gguf::TensorInfo* source=nullptr;
    for(const auto& t:reader.tensors()){
        if(!sub0::gguf::dequant_fn(t.type_raw)||t.dims.size()!=2)continue;
        const auto block=sub0::gguf::block_spec(t.type_raw);
        const auto n=static_cast<std::size_t>(t.dims[0]),height=static_cast<std::size_t>(t.dims[1]);
        if(!n || n%block.elems || !height)throw std::runtime_error("invalid quantized shape");
        if(!source && n>=5120)source=&t;
        const auto raw=reader.tensor_bytes(t);const auto bytes=n/block.elems*block.bytes;
        for(const auto row:{std::size_t{0},height/2,height-1})check(t.type_raw,raw.subspan(row*bytes,bytes),n,false);
    }
    if(!source)throw std::runtime_error("missing source rows");
    for(const auto& entry:sub0::gguf::kDequant){
        const auto type=static_cast<std::uint32_t>(entry.type);if(stats.contains({type,false}))continue;
        const auto target=static_cast<ggml_type>(type);
        if(ggml_quantize_requires_imatrix(target))throw std::runtime_error("absent format needs importance matrix");
        const auto source_width=static_cast<std::size_t>(source->dims[0]),height=static_cast<std::size_t>(source->dims[1]);
        const auto sb=sub0::gguf::block_spec(source->type_raw);const auto source_bytes=source_width/sb.elems*sb.bytes;
        const auto raw=reader.tensor_bytes(*source);const auto* traits=ggml_get_type_traits(static_cast<ggml_type>(source->type_raw));
        std::vector<float> values(source_width);const auto block=sub0::gguf::block_spec(type);std::vector<std::uint8_t> packed(5120/block.elems*block.bytes);
        for(const auto row:{std::size_t{0},height/2,height-1}){
            traits->to_float(raw.data()+row*source_bytes,values.data(),static_cast<std::int64_t>(source_width));
            if(ggml_quantize_chunk(target,values.data(),packed.data(),0,1,5120,nullptr)!=packed.size())throw std::runtime_error("quantized row size");
            check(type,packed,5120,true);
        }
    }
    bool passed=!stats.empty(),first=true;std::size_t total=0,native_rows=0;
    std::printf("{\"source_tensor\":\"%s\",\"formats\":[",source->name.c_str());
    for(const auto& [key,s]:stats){
        total+=s.rows;if(!key.second)native_rows+=s.rows;
        std::printf("%s{\"type\":%u,\"reencoded\":%s,\"rows\":%zu,\"kernels\":[",first?"":",",key.first,key.second?"true":"false",s.rows);first=false;
        for(int k=0;k<4;++k){const double l2=std::sqrt(s.errors[k])/std::max(1.,std::sqrt(s.reference_norm));passed&=l2<=2e-5;
            std::printf("%s{\"kernel\":%d,\"enum_value\":%d,\"relative_l2\":%.12g,\"max_abs\":%.12g}",k?",":"",k,k==0?0:k==1?2:k==2?3:4,l2,s.maximum[k]);}
        std::printf("]}");
    }
    std::printf("],\"rows\":%zu,\"native_rows\":%zu,\"bit_exact_prior\":true,\"passed\":%s}\n",total,native_rows,passed?"true":"false");ggml_quantize_free();return passed?0:1;
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
