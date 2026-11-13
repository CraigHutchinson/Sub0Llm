// Synthetic warm-cache projections; one operation covers all active sessions.
#define ANKERL_NANOBENCH_IMPLEMENT
#include <bench_runner.hpp>
#include "q16_tile_fixture.hpp"
#include <cstdio>
int main(int argc,char** argv)try{
    sub0bench::BenchRunner runner(argc,argv,"IQ1 Q16 row/session tile");runner.bench().unit("session batch");
    for(std::size_t width:{256,5120,17408})for(std::size_t sessions:{1,2}){
        sub0bench::Q16TileFixture f(width,sessions);
        const auto name="IQ1_S 32x"+std::to_string(width)+" s"+std::to_string(sessions);
        runner.run((name+", float serial").c_str(),[&]{f.run_float();ankerl::nanobench::doNotOptimizeAway(f.output.data());});
        runner.run((name+", Q16 serial").c_str(),[&]{f.run_serial();ankerl::nanobench::doNotOptimizeAway(f.output.data());});
        runner.run((name+", Q16 serial prepare+project").c_str(),[&]{f.prepare();f.run_serial();ankerl::nanobench::doNotOptimizeAway(f.output.data());});
        runner.run((name+", tile256 unscaled").c_str(),[&]{f.run<256,false>();ankerl::nanobench::doNotOptimizeAway(f.output.data());});
        runner.run((name+", tile256 scaled").c_str(),[&]{f.run<256>();ankerl::nanobench::doNotOptimizeAway(f.output.data());});
        runner.run((name+", tile256 prepare+project").c_str(),[&]{f.prepare();f.run<256>();ankerl::nanobench::doNotOptimizeAway(f.output.data());});
        if constexpr(sub0::denseq::iq1_q16_tile_available<512>){
            runner.run((name+", tile512 scaled").c_str(),[&]{f.run<512>();ankerl::nanobench::doNotOptimizeAway(f.output.data());});
            runner.run((name+", tile512 prepare+project").c_str(),[&]{f.prepare();f.run<512>();ankerl::nanobench::doNotOptimizeAway(f.output.data());});
        }
    }
    return runner.finish();
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
