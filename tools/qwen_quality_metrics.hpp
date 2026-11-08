// Teacher-forced raw-logit NLL. No sampling, implicit BOS, or token filtering.
#pragma once
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <span>
#include <stdexcept>
#include <vector>

namespace sub0::quality {
struct Scores {
    std::size_t vocabulary=0;
    std::vector<double> nll;
    std::vector<int> top_ids,scored_targets;
    void reserve(std::size_t count){nll.reserve(count);top_ids.reserve(count);scored_targets.reserve(count);}
    void add(std::span<const float> logits,int target){
        if(logits.empty() || target<0 || static_cast<std::size_t>(target)>=logits.size())throw std::runtime_error("invalid score target/vocabulary");
        if(vocabulary && vocabulary!=logits.size())throw std::runtime_error("score vocabulary changed");
        vocabulary=logits.size();
        for(float x:logits)if(!std::isfinite(x))throw std::runtime_error("nonfinite score logit");
        const auto best=std::max_element(logits.begin(),logits.end());const double maximum=*best;
        double sum=0;for(float x:logits)sum+=std::exp(static_cast<double>(x)-maximum);
        const double loss=std::log(sum)+maximum-static_cast<double>(logits[static_cast<std::size_t>(target)]);
        if(!std::isfinite(loss))throw std::runtime_error("nonfinite NLL");
        nll.push_back(loss);scored_targets.push_back(target);top_ids.push_back(static_cast<int>(best-logits.begin()));
    }
    void write(const std::string& path,std::span<const int> inputs,std::span<const int> targets)const{
        if(inputs.size()!=nll.size() || targets.size()!=nll.size() || nll.empty())throw std::runtime_error("incomplete score sequence");
        if(!std::equal(targets.begin(),targets.end(),scored_targets.begin(),scored_targets.end()))throw std::runtime_error("score target trace changed");
        std::ofstream out(path,std::ios::binary);if(!out)throw std::runtime_error("score output open");
        out<<std::setprecision(17)<<"{\"schema\":\"qwen-nll-1\",\"vocabulary\":"<<vocabulary;
        auto array=[&](const char* name,const auto& data){out<<",\""<<name<<"\":[";bool first=true;for(auto x:data){if(!first)out<<',';first=false;out<<x;}out<<']';};
        array("input_ids",inputs);array("target_ids",targets);array("nll",nll);array("top_ids",top_ids);
        out<<"}\n";out.close();if(!out)throw std::runtime_error("score output write");
    }
};
}
