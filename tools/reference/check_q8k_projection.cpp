// Real-GGUF row gate. The float decode exists only in this independent validation tool;
// inference consumes packed weights directly. Activation encoding has its own pinned-reference gate.
#include "sub0/dense_quant.hpp"
#include "sub0/dense_quant_q8k.hpp"
#include "sub0/dense_quant_q16.hpp"
#include "sub0/file_map.hpp"
#include <cmath>
#include <cstdio>
#include <map>
#include <stdexcept>

struct Stats {std::size_t rows=0;double error=0,norm=0,maximum=0;};
int main(int argc,char** argv) try {
    if(argc!=2)throw std::runtime_error("usage: check_q8k_projection MODEL.gguf");
    sub0::FileMap file;std::string error;
    if(!file.open(argv[1],error))throw std::runtime_error(error);
    sub0::gguf::Reader reader({file.data(),file.size()});
    if(!reader.ok())throw std::runtime_error("invalid GGUF");
    std::map<std::pair<std::uint32_t,int>,Stats> results;
    for(const auto& t:reader.tensors()){
        if(!sub0::denseq::q8k_supported(t.type_raw) || t.dims.size()!=2)continue;
        const auto width=static_cast<std::size_t>(t.dims[0]),height=static_cast<std::size_t>(t.dims[1]);
        if(!width || width%256 || !height)throw std::runtime_error("invalid tensor shape");
        const auto block=sub0::gguf::block_spec(t.type_raw);const auto bytes=width/256*block.bytes;
        const auto raw=reader.tensor_bytes(t);
        const auto check=[&]<class Activation>(int precision){
            Activation input(width);std::vector<float> x(width),weights,y(1),scalar(1);
            for(std::size_t i=0;i<width;++i)x[i]=static_cast<float>(static_cast<int>(i*17%101)-50)/37.f;
            if(!input.quantize(x))throw std::runtime_error("activation encoding failed");
            for(auto r:{std::size_t{0},height/2,height-1}){
                const auto row=raw.subspan(r*bytes,bytes);
                bool ok=false;
                if constexpr(std::same_as<Activation,sub0::quant::Q8KActivations>)
                    ok=sub0::denseq::project_q8k<true>(t.type_raw,row,input,y) && sub0::denseq::project_q8k<false>(t.type_raw,row,input,scalar);
                else ok=sub0::denseq::project_q16<true>(t.type_raw,row,input,y) && sub0::denseq::project_q16<false>(t.type_raw,row,input,scalar);
                if(!ok || !sub0::gguf::dequant_fn(t.type_raw)(row,width,weights))throw std::runtime_error("projection failed");
                double expected=0;
                for(std::size_t i=0;i<width;++i){const auto& q=input.blocks()[i/256];
                    expected+=static_cast<double>(weights[i])*static_cast<double>(q.d)*q.qs[i%256];}
                auto& stats=results[{t.type_raw,precision}];++stats.rows;stats.norm+=expected*expected;
                const double delta=static_cast<double>(y[0])-expected;
                stats.error+=delta*delta;stats.maximum=std::max(stats.maximum,std::abs(delta));
                if(!std::isfinite(y[0]) || y[0]!=scalar[0])throw std::runtime_error("scalar/native mismatch");
            }
        };
        check.template operator()<sub0::quant::Q8KActivations>(8);
        check.template operator()<sub0::quant::Q16Activations>(16);
    }
    bool passed=results.size()==4,first=true;
    std::printf("{\"formats\":[");
    for(const auto& [key,s]:results){
        const double relative=std::sqrt(s.error)/std::max(1.,std::sqrt(s.norm));passed&=relative<=2e-5;
        std::printf("%s{\"type\":%u,\"activation_bits\":%d,\"rows\":%zu,\"relative_l2\":%.12g,\"max_abs\":%.12g}",first?"":",",key.first,key.second,s.rows,relative,s.maximum);first=false;
    }
    std::printf("],\"scalar_native_exact\":true,\"passed\":%s}\n",passed?"true":"false");
    return passed?0:1;
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
