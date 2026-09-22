// sub0llm-bench-gemv -- kernel-only microbenchmark for the shared GEMV primitive (include/sub0/gemv.hpp)
// at every real Qwen4 backbone projection shape, bf16 weights, 1..8 threads.
//
// WHY THIS EXISTS: O2 made the backbone GEMV one primitive, vectorized it (-26% decode single-threaded),
// then found that splitting it across threads made decode SLOWER, and slower with every thread added --
// against a measured 79 GB/s on 8 P-cores vs 30 GB/s on one. Diagnosing that through the 37 GiB decode
// costs minutes per data point and mixes every phase. This isolates the primitive so thread count,
// placement (OMP_PLACES / OMP_PROC_BIND) and wake-up policy (KMP_BLOCKTIME) can be varied in seconds.
//
// Each shape is timed as the decode sees it: many DIFFERENT weight matrices in turn (a pool larger than
// L3, like consecutive layers), so every call streams its weights from DRAM rather than from cache.

#include "sub0/gemv.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string_view>
#include <vector>

namespace {

struct Shape {
    const char* name;
    int         in;
    int         out;
};
// The real axes: hidden 2560, GDN conv_dim 10240 / value_dim 6144, QSA q_width 6144 / kv 512, d_ff 640,
// 512 experts, vocab 248,320.
constexpr Shape kShapes[] = {
    {"gdn in_qkv", 2560, 10240}, {"gdn in_z", 2560, 6144},  {"gdn out", 6144, 2560},
    {"qsa k/v", 2560, 512},      {"shared gate", 2560, 640}, {"shared down", 640, 2560},
    {"router", 2560, 512},       {"lm_head", 2560, 248320},
    {"gr down", 10240, 320},     {"gr up", 320, 10240},       // Gated Residual mix, 96 calls/token
};

template <int Threads>
double time_shape(const Shape& sh, const std::vector<std::vector<sub0::bf16>>& pool, const float* x,
                  float* y, double min_seconds) {
    using clock = std::chrono::steady_clock;
    std::size_t calls = 0;
    const auto t0 = clock::now();
    double el = 0.0;
    do {
        for (const auto& w : pool) sub0::gemv::axpy<Threads>(x, sub0::Bf16CPtr{w.data()}, sh.in, sh.out, y);
        calls += pool.size();
        el = std::chrono::duration<double>(clock::now() - t0).count();
    } while (el < min_seconds);
    return el / static_cast<double>(calls) * 1e6;   // us per GEMV
}

}  // namespace

int main(int argc, char** argv) {
    double min_seconds = 0.5;
    for (int i = 1; i < argc; ++i)
        if (std::string_view(argv[i]) == "--seconds" && i + 1 < argc) min_seconds = std::atof(argv[++i]);

    std::mt19937 rng(11);
    std::uniform_int_distribution<int> bits(0x3c00, 0x3f80);   // bf16 patterns in ~[0.0078, 1)
    std::printf("%-12s %6s %7s | %12s %12s %12s %12s %12s %12s   (us/GEMV / GB/s)\n", "shape", "in", "out",
                "1 thr", "2 thr", "4 thr", "8 thr", "12 thr", "16 thr");
    for (const Shape& sh : kShapes) {
        const std::size_t elems = static_cast<std::size_t>(sh.in) * static_cast<std::size_t>(sh.out);
        const std::size_t bytes = elems * sizeof(sub0::bf16);
        // Enough distinct matrices to overflow L3 (36 MiB) three times over, capped for lm_head.
        const std::size_t n = std::max<std::size_t>(1, std::min<std::size_t>(64, (3ull * 36 << 20) / bytes + 1));
        std::vector<std::vector<sub0::bf16>> pool(n, std::vector<sub0::bf16>(elems));
        for (auto& m : pool)
            for (auto& v : m) v.bits = static_cast<std::uint16_t>(bits(rng));
        std::vector<float> x(static_cast<std::size_t>(sh.in)), y(static_cast<std::size_t>(sh.out));
        for (auto& v : x) v = static_cast<float>(rng() % 1000) / 1000.f - 0.5f;

        const double t[6] = {time_shape<1>(sh, pool, x.data(), y.data(), min_seconds),
                             time_shape<2>(sh, pool, x.data(), y.data(), min_seconds),
                             time_shape<4>(sh, pool, x.data(), y.data(), min_seconds),
                             time_shape<8>(sh, pool, x.data(), y.data(), min_seconds),
                             time_shape<12>(sh, pool, x.data(), y.data(), min_seconds),
                             time_shape<16>(sh, pool, x.data(), y.data(), min_seconds)};
        std::printf("%-12s %6d %7d |", sh.name, sh.in, sh.out);
        for (const double us : t) std::printf(" %7.0f/%4.1f", us, static_cast<double>(bytes) / (us * 1e-6) / 1e9);
        std::printf("\n");
    }
    return 0;
}
