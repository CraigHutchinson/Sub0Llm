// Local Zen4/x86 AVX512 roof probes; descriptive measured ceilings, not engine benchmarks.
#include <immintrin.h>
#include <omp.h>
#include <sched.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cstdint>
#include <stdexcept>
using Clock=std::chrono::steady_clock;
static void pin(int tid){cpu_set_t mask;CPU_ZERO(&mask);CPU_SET(tid,&mask);if(sched_setaffinity(0,sizeof(mask),&mask))std::abort();}
int main() try {
    constexpr std::uint64_t iterations=500000000;
    constexpr std::size_t count=1u<<28;
    constexpr int passes=32,threads=8;
    omp_set_dynamic(0);
    std::vector<float> input(count,1.f);
    volatile float sink[threads]{};volatile std::int32_t sink_i[threads]{};
    double fp[3]{},integer[3]{},memory[3]{};
    for(int run=0;run<3;++run){
        auto start=Clock::now();
        #pragma omp parallel num_threads(threads)
        {
            const int tid=omp_get_thread_num();if(omp_get_num_threads()!=threads)std::abort();pin(tid);
            __m512 values[8];for(int k=0;k<8;++k)values[k]=_mm512_set1_ps(static_cast<float>(k+tid+1));
            const __m512 scale=_mm512_set1_ps(.999999f),offset=_mm512_set1_ps(.000001f);
            for(std::uint64_t i=0;i<iterations;++i)for(int k=0;k<8;++k)values[k]=_mm512_fmadd_ps(values[k],scale,offset);
            __m512 sum=_mm512_setzero_ps();for(int k=0;k<8;++k)sum=_mm512_add_ps(sum,values[k]);
            sink[tid]=_mm512_reduce_add_ps(sum);
        }
        double seconds=std::chrono::duration<double>(Clock::now()-start).count();
        fp[run]=static_cast<double>(iterations)*8*16*2*threads/seconds/1e9;
        start=Clock::now();
        #pragma omp parallel num_threads(threads)
        {
            const int tid=omp_get_thread_num();if(omp_get_num_threads()!=threads)std::abort();pin(tid);
            __m512i values[8];for(int k=0;k<8;++k)values[k]=_mm512_set1_epi32(k+tid);
            // Runtime reads prevent folding the dot operands into compile-time constants.
            static volatile char seed=1;const __m512i x=_mm512_set1_epi8(seed),y=_mm512_set1_epi8(seed);
            for(std::uint64_t i=0;i<iterations;++i)for(int k=0;k<8;++k)values[k]=_mm512_dpbusd_epi32(values[k],x,y);
            __m512i sum=_mm512_setzero_si512();for(int k=0;k<8;++k)sum=_mm512_add_epi32(sum,values[k]);
            sink_i[tid]=_mm512_reduce_add_epi32(sum);
        }
        seconds=std::chrono::duration<double>(Clock::now()-start).count();
        integer[run]=static_cast<double>(iterations)*8*64*2*threads/seconds/1e9;
        start=Clock::now();
        #pragma omp parallel num_threads(threads)
        {
            const int tid=omp_get_thread_num();if(omp_get_num_threads()!=threads)std::abort();pin(tid);float total=0;
            const std::size_t base=count*tid/threads,end=count*(tid+1)/threads;
            for(int pass=0;pass<passes;++pass){
                __m512 values[8];for(auto& v:values)v=_mm512_setzero_ps();
                for(std::size_t i=base;i<end;i+=128)for(int k=0;k<8;++k)values[k]=_mm512_add_ps(values[k],_mm512_loadu_ps(input.data()+i+16*k));
                __m512 sum=_mm512_setzero_ps();for(auto v:values)sum=_mm512_add_ps(sum,v);
                total+=_mm512_reduce_add_ps(sum);
            }
            sink[tid]=total;
        }
        seconds=std::chrono::duration<double>(Clock::now()-start).count();
        memory[run]=static_cast<double>(count)*sizeof(float)*passes/seconds/1e9;
    }
    std::printf("{\"schema\":\"host-roofs-1\",\"cpus\":[0,1,2,3,4,5,6,7],\"fp32_gflops\":[%.9g,%.9g,%.9g],\"int8_dot_gops\":[%.9g,%.9g,%.9g],\"read_gb_s\":[%.9g,%.9g,%.9g]}\n",fp[0],fp[1],fp[2],integer[0],integer[1],integer[2],memory[0],memory[1],memory[2]);
    (void)sink;(void)sink_i;
} catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
