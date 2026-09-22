// sub0llm-bench-moeqd -- kernel-only microbenchmark for the fused quantized MoE GEMV (B35,
// include/sub0/moe_quant_dot.hpp), at the real Qwen4 plane shapes.
//
// WHY THIS EXISTS: the only other way to time these kernels is a full real-artifact decode -- a 37 GiB
// sidecar, minutes per data point, and a number that mixes the kernel with every other stage. That is
// the right FINAL gate (scripts/run_perf_suite.py --stage perf), but the wrong inner loop for
// iterating on a kernel three times per docs/OPTIMIZATION_PROCESS.md S4. This isolates the kernel and
// reports it against BOTH roofs (S1), so every iteration says not just "faster" but "how far from the
// ceiling" -- the question that actually decides whether to keep going.
//
// Two memory modes, because they answer different questions:
//   hot     one plane, re-read: cache-resident, so the time is COMPUTE (unpack + MAC) alone.
//   stream  a pool of distinct planes larger than L3 (36 MiB on this host), each read once per sweep:
//           the shape decode actually has -- every routed expert's plane is touched once per token.
// If stream ~= hot, the kernel is compute-bound and the GB/s column is an OUTCOME, not a limit.
//
// Weights are random bytes with each block's f16 scale patched sane -- the same generator
// tests/moe_quant_tests.cpp uses. Every index these formats encode is in range by construction, so
// random bytes exercise the full unpack path (all grid entries, all sign patterns), which a
// structured fixture would not.
//
// Output checksum: the sum of every output row, printed per case. Two kernel variants that compute the
// same thing print the same checksum to float-rounding; a layout or sign error moves it by orders of
// magnitude. It is a smoke check for the bench, NOT the correctness gate -- that is
// tests/moe_quant_tests.cpp's "[moequant]" cases.

#include "sub0/moe_quant_dot.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <span>
#include <string_view>
#include <vector>

namespace {

using sub0::gguf::TensorType;
namespace moeqd = sub0::moeqd;

// Roofline ceilings for this host, from docs/optimization/roofline_post_b35.md -- ONE core, since decode
// runs the MoE resolve on MOE_DECODE_THREADS = 1 (B29).
constexpr double kDramGBs   = 30.0;    // measured, docs/BACKBONE_PRECISION.md S2c
constexpr double kAvx2GMacs = 112.0;   // vpmaddwd: 16 MAC/instr x 2 ports x 3.5 GHz

std::vector<std::uint8_t> make_plane(TensorType type, std::uint64_t elems, std::uint32_t seed) {
    const sub0::gguf::BlockSpec spec = sub0::gguf::block_spec(static_cast<std::uint32_t>(type));
    const std::uint64_t blocks = elems / spec.elems;
    std::vector<std::uint8_t> raw(static_cast<std::size_t>(blocks * spec.bytes));
    std::mt19937 rng(seed);
    for (auto& b : raw) b = static_cast<std::uint8_t>(rng() & 0xFFu);
    for (std::uint64_t b = 0; b < blocks; ++b) {
        const auto d_bits = static_cast<std::uint16_t>(0x3000u + (rng() & 0x0FFFu));
        std::memcpy(raw.data() + b * spec.bytes, &d_bits, sizeof d_bits);
    }
    return raw;
}

const char* name(TensorType t) {
    switch (t) {
        case TensorType::IQ1_S:   return "IQ1_S";
        case TensorType::IQ2_XXS: return "IQ2_XXS";
        default:                  return "IQ4_NL";
    }
}

struct Shape {
    const char* role;
    int         rows;
    int         row_elems;
};
constexpr Shape kShapes[] = {{"gate/up", 640, 2560}, {"down", 2560, 640}};   // real Qwen4 axes

void run_case(TensorType type, const Shape& sh, bool stream, double min_seconds) {
    const auto type_raw = static_cast<std::uint32_t>(type);
    const std::uint64_t elems = static_cast<std::uint64_t>(sh.rows) * static_cast<std::uint64_t>(sh.row_elems);
    const std::uint64_t plane_bytes = moeqd::plane_bytes(type_raw, elems);

    // stream: enough distinct planes to overflow L3 four times over, so nothing survives a sweep.
    constexpr std::uint64_t kStreamBytes = 4ull * 36ull * 1024ull * 1024ull;
    const std::size_t n_planes = stream ? static_cast<std::size_t>((kStreamBytes + plane_bytes - 1) / plane_bytes) : 1;
    std::vector<std::vector<std::uint8_t>> planes;
    planes.reserve(n_planes);
    for (std::size_t i = 0; i < n_planes; ++i)
        planes.push_back(make_plane(type, elems, 1000u + static_cast<std::uint32_t>(i) + 97u * type_raw));

    std::vector<float> x(static_cast<std::size_t>(sh.row_elems));
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto& v : x) v = nd(rng);
    moeqd::ActBlocks xq;
    xq.quantize(x.data(), sh.row_elems);
    std::vector<float> out(static_cast<std::size_t>(sh.rows));

    double checksum = 0.0;
    auto one = [&](std::size_t i) {
        if (!moeqd::gemv_plane(type_raw, std::span<const std::uint8_t>(planes[i]), sh.rows, sh.row_elems, xq,
                               out.data())) {
            std::fprintf(stderr, "gemv_plane refused %s %s\n", name(type), sh.role);
            std::exit(1);
        }
    };
    for (std::size_t i = 0; i < n_planes; ++i) one(i);   // warm-up: page in every plane once
    for (float v : out) checksum += v;

    using clock = std::chrono::steady_clock;
    std::uint64_t calls = 0;
    const auto t0 = clock::now();
    double elapsed = 0.0;
    do {
        for (std::size_t i = 0; i < n_planes; ++i) one(i);
        calls += n_planes;
        elapsed = std::chrono::duration<double>(clock::now() - t0).count();
    } while (elapsed < min_seconds);

    const double us    = elapsed / static_cast<double>(calls) * 1e6;
    const double gbs   = static_cast<double>(plane_bytes) / (us * 1e-6) / 1e9;
    const double gmacs = static_cast<double>(elems) / (us * 1e-6) / 1e9;
    std::printf("%-8s %-8s %-6s %9.1f us  %6.2f GB/s (%4.1f%%)  %6.2f GMAC/s (%4.1f%%)  chk %.6e\n",
                name(type), sh.role, stream ? "stream" : "hot", us, gbs, 100.0 * gbs / kDramGBs, gmacs,
                100.0 * gmacs / kAvx2GMacs, checksum);
}

}  // namespace

int main(int argc, char** argv) {
    double min_seconds = 1.0;
    bool do_hot = true, do_stream = true;
    for (int i = 1; i < argc; ++i) {
        const std::string_view a = argv[i];
        if (a == "--hot") do_stream = false;
        else if (a == "--stream") do_hot = false;
        else if (a == "--seconds" && i + 1 < argc) min_seconds = std::atof(argv[++i]);
        else {
            std::fprintf(stderr, "usage: %s [--hot|--stream] [--seconds S]\n", argv[0]);
            return 2;
        }
    }
    std::printf("fused MoE GEMV, one thread; ceilings %.0f GB/s DRAM, %.0f GMAC/s AVX2 int8\n", kDramGBs,
                kAvx2GMacs);
    for (const TensorType t : {TensorType::IQ1_S, TensorType::IQ2_XXS, TensorType::IQ4_NL})
        for (const Shape& sh : kShapes) {
            if (do_hot) run_case(t, sh, false, min_seconds);
            if (do_stream) run_case(t, sh, true, min_seconds);
        }
    return 0;
}
