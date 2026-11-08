#include "../../qwen_quality_metrics.hpp"
#include <array>
#include <limits>
#include <string>
int main(int argc,char**argv){
    if(argc!=2)return 2;
    const std::string prefix=argv[1];const std::array<int,2> inputs{0,1},targets{0,2};
    for(bool shifted:{false,true}){
        sub0::quality::Scores scores;scores.reserve(2);
        const float shift=shifted?1000.f:0.f;
        const std::array<float,3> a{shift,shift+1,shift-1},b{shift+2,shift+4,shift+1};
        scores.add(a,0);scores.add(b,2);scores.write(prefix+(shifted?"-shift.json":"-base.json"),inputs,targets);
    }
    sub0::quality::Scores uniform;uniform.reserve(1);std::vector<float> zeros(248320);
    uniform.add(zeros,42);const std::array<int,1> one{42};uniform.write(prefix+"-uniform.json",one,one);
    int rejected=0;
    for(int mode=0;mode<6;++mode){
        try{
            sub0::quality::Scores s;s.reserve(1);std::vector<float> v(3);
            if(mode==0)s.add(v,-1);
            if(mode==1)s.add(v,3);
            if(mode==2){v[0]=std::numeric_limits<float>::quiet_NaN();s.add(v,0);}
            if(mode==3){v[0]=std::numeric_limits<float>::infinity();s.add(v,0);}
            if(mode==4)s.add({},0);
            if(mode==5){s.add(v,0);v.resize(4);s.add(v,0);}
        }catch(const std::runtime_error&){++rejected;}
    }
    return rejected==6?0:3;
}
