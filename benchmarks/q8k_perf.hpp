// Warm-cache 32-row projections; preparation included in a separate measurement.
#pragma once
#include <bench_runner.hpp>
#include "q8k_fixture.hpp"
#include "q16_fixture.hpp"
namespace sub0bench {
inline void add_q8k_benchmarks(BenchRunner& runner) {
    for(std::size_t width:{std::size_t{256},std::size_t{5120},std::size_t{17408}})
        for(std::uint32_t type:{std::uint32_t{19},std::uint32_t{16}}) {
            Q8KFixture f(type,width);
            const std::string name=std::string(type==19?"IQ1_S":"IQ2_XXS")+" 32x"+std::to_string(width);
            runner.run((name+", float native").c_str(),[&]{f.run_float();ankerl::nanobench::doNotOptimizeAway(f.y.data());});
            runner.run((name+", Q8_K prepared").c_str(),[&]{f.run_prepared<true>();ankerl::nanobench::doNotOptimizeAway(f.y.data());});
            runner.run((name+", Q8_K quantize+project").c_str(),[&]{f.prepare();f.run_prepared<true>();ankerl::nanobench::doNotOptimizeAway(f.y.data());});
            Q16Fixture high(type,width);
            runner.run((name+", Q16 prepared").c_str(),[&]{high.run<true>();ankerl::nanobench::doNotOptimizeAway(high.y.data());});
            runner.run((name+", Q16 quantize+project").c_str(),[&]{high.prepare();high.run<true>();ankerl::nanobench::doNotOptimizeAway(high.y.data());});
        }
}
} // namespace sub0bench
