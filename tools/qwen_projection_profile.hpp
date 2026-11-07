// Inclusive projection wall time on the calling thread; worker times are not summed.
#pragma once
#include "qwen_phase_metrics.hpp"
#include <array>

namespace sub0::phase {
struct ProjectionProfile {
    struct Entry {
        std::uint64_t calls=0, failures=0, encoded_bytes=0, logical_flops=0;
        double wall_ms=0;
    };
    // GGUF type IDs used by this proving ground are 0..30. Fail closed on new IDs.
    std::array<std::array<Entry,64>,2> entries{};
    bool decoding=false;
    void record(std::uint32_t type,std::uint64_t bytes,std::uint64_t flops,double ms,bool ok) {
        if(type>=entries[0].size())throw std::runtime_error("projection profile type outside table");
        auto& e=entries[decoding?1:0][type];
        ++e.calls;e.failures+=!ok;e.encoded_bytes+=bytes;e.logical_flops+=flops;e.wall_ms+=ms;
    }
    void write(const std::string& path) const {
        std::ofstream out(path,std::ios::binary);if(!out)throw std::runtime_error("projection profile open");
        out<<std::setprecision(12)<<"{\"schema\":\"qwen-projection-profile-1\",\"entries\":[";
        bool first=true;
        for(std::size_t phase=0;phase<entries.size();++phase)for(std::size_t type=0;type<entries[phase].size();++type){
            const auto& e=entries[phase][type];if(!e.calls)continue;
            if(!first)out<<',';first=false;
            out<<"{\"phase\":\""<<(phase?"decode":"prefill")<<"\",\"type\":"<<type
               <<",\"calls\":"<<e.calls<<",\"failures\":"<<e.failures<<",\"encoded_bytes\":"<<e.encoded_bytes
               <<",\"logical_flops\":"<<e.logical_flops<<",\"wall_ms\":"<<e.wall_ms<<'}';
        }
        out<<"]}\n";out.close();if(!out)throw std::runtime_error("projection profile write");
    }
};
}
