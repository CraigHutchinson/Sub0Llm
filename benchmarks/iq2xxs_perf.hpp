#pragma once
#include <bench_runner.hpp>
#include "iq2xxs_fixture.hpp"
namespace sub0bench {
template<class Runner>
void add_iq2xxs_benchmarks(Runner& runner){
    for(std::size_t width:{256,5120,17408}){
        Iq2xxsFixture fixture(width);
        for(auto mode:{sub0::denseq::Iq1Kernel::NativeAll,sub0::denseq::Iq1Kernel::FusedIQ2}){
            const auto name="IQ2_XXS float 32x"+std::to_string(width)+", "+(mode==sub0::denseq::Iq1Kernel::NativeAll?"vector":"fused");
            runner.run(name.c_str(),[&]{fixture.run(mode);ankerl::nanobench::doNotOptimizeAway(fixture.y.data());});
        }
    }
}
}
