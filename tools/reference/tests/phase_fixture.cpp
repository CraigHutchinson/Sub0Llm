#include "../../qwen_phase_metrics.hpp"
#include <string>
int main(int argc,char**argv){
    if(argc!=3)return 2;
    const bool batched=std::string(argv[2])=="batch";
    sub0::phase::Metrics metrics;metrics.reserve(13,8);metrics.setup_ms=20.;
    metrics.prefill_ms.assign(batched?1:13,batched?130.:10.);
    metrics.decode_ms.assign(8,2.);metrics.top_ids.assign((batched?1:13)+8,1);
    metrics.write(argv[1],8,128,batched,true);
}
