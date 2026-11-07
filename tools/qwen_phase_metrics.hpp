// Shared native phase definitions for the mapped proving ground and external reference harness.
#pragma once
#include <chrono>
#include <cstdint>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <vector>

namespace sub0::phase {
using Clock = std::chrono::steady_clock;
inline double milliseconds(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now()-start).count();
}
struct Metrics {
    double setup_ms=0;
    std::size_t prompt_tokens=0;
    std::uint64_t weight_bytes_per_eval=0, projection_flops_per_eval=0;
    std::vector<double> prefill_ms, decode_ms;
    std::vector<int> top_ids;
    void reserve(std::size_t prompt, std::size_t decode) {
        prompt_tokens=prompt;prefill_ms.reserve(prompt);decode_ms.reserve(decode);top_ids.reserve(prompt+decode);
    }
    void write(const std::string& path,int workers,int context,bool batched,bool f32_attention) const {
        std::ofstream out(path,std::ios::binary);if(!out)throw std::runtime_error("phase metrics open");
        out<<std::setprecision(12)<<"{\"schema\":\"qwen-phase-1\",\"workers\":"<<workers
           <<",\"context_capacity\":"<<context<<",\"prompt_tokens\":"<<prompt_tokens<<",\"decode_evaluations\":"<<decode_ms.size()
           <<",\"weight_bytes_per_eval\":"<<weight_bytes_per_eval<<",\"projection_flops_per_eval\":"<<projection_flops_per_eval<<",\"setup_ms\":"<<setup_ms
           <<",\"batched_prefill\":"<<(batched?"true":"false")<<",\"f32_attention\":"<<(f32_attention?"true":"false");
        auto array=[&](const char* name,const auto& values){out<<",\""<<name<<"\":[";bool first=true;for(auto value:values){if(!first)out<<',';first=false;out<<value;}out<<']';};
        array("prefill_ms",prefill_ms);array("decode_ms",decode_ms);array("top_ids",top_ids);
        out<<"}\n";out.close();if(!out)throw std::runtime_error("phase metrics write");
    }
};
}
