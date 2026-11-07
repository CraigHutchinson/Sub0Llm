#include "../../qwen_phase_metrics.hpp"
#include "../../qwen_projection_profile.hpp"
#include <string>
int main(int argc,char**argv){
    if(argc!=3)return 2;
    if(std::string(argv[2])=="profile"){
        sub0::phase::ProjectionProfile p;
        p.record(19,100,200,2.,true);p.record(19,150,300,3.,true);
        p.decoding=true;p.record(19,100,200,4.,true);p.record(8,50,80,1.,false);
        bool rejected=false;try{p.record(64,0,0,0.,true);}catch(const std::runtime_error&){rejected=true;}
        if(!rejected)return 3;
        p.write(argv[1]);return 0;
    }
    const bool batched=std::string(argv[2])=="batch";
    sub0::phase::Metrics metrics;metrics.reserve(13,8);metrics.setup_ms=20.;
    metrics.prefill_ms.assign(batched?1:13,batched?130.:10.);
    metrics.decode_ms.assign(8,2.);metrics.top_ids.assign((batched?1:13)+8,1);
    metrics.write(argv[1],8,128,batched,true);
}
