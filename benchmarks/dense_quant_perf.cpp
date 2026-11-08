// Shared sub0perf entry point. Synthetic warm-cache Q8_0 output-row slices, not model tok/s.
#define ANKERL_NANOBENCH_IMPLEMENT
#include <bench_runner.hpp>
#include "sub0/dense_quant.hpp"
#include "iq1s_perf.hpp"
#include "iq2xxs_perf.hpp"
#include "float_reduce_perf.hpp"
#include <bit>
#include <cmath>
#include <stdexcept>

namespace {
struct Fixture {
    std::size_t width;
    static constexpr std::size_t rows = 32;
    std::vector<std::uint8_t> weights;
    std::vector<float> input, output, reference, scratch;

    explicit Fixture(std::size_t n) : width(n), weights(rows * n / 32 * 34),
        input(n), output(rows), reference(rows) {
        scratch.reserve(256);
        for (std::size_t i = 0; i < input.size(); ++i)
            input[i] = static_cast<float>(static_cast<int>(i % 29) - 14) / 16.f;
        for (std::size_t b = 0; b < rows * n / 32; ++b) {
            weights[b * 34] = 0;
            weights[b * 34 + 1] = 0x30; // little-endian binary16 scale = 0.125
            for (std::size_t j = 0; j < 32; ++j)
                weights[b * 34 + 2 + j] = static_cast<std::uint8_t>((b * 13 + j * 7) % 256);
        }
        project();
        hand_written();
        // Independent fixture decoder/dot: rejects incorrect kernels before any timing/list output.
        for (std::size_t i = 0; i < rows; ++i)
            if (!std::isfinite(output[i]) || output[i] != reference[i])
                throw std::runtime_error("Q8_0 projection fixture differs from independent scalar reference");
    }

    void project() {
        if (!sub0::denseq::project(8, weights, input, output, scratch))
            throw std::runtime_error("Q8_0 projection rejected fixture");
    }

    void hand_written() {
        // Fast scalar form for these fixed-scale fixtures: direct packed-byte dot, no scratch decode.
        // It does not call the project's decoder or projection and is never weakened to match it.
        for (std::size_t row = 0; row < rows; ++row) {
            float sum = 0.f;
            for (std::size_t b = 0; b < width / 32; ++b)
                for (std::size_t j = 0; j < 32; ++j) {
                    const auto q = std::bit_cast<std::int8_t>(weights[(row * width / 32 + b) * 34 + 2 + j]);
                    sum += input[b * 32 + j] * (0.125f * static_cast<float>(q));
                }
            reference[row] = sum;
        }
    }
};
}

int main(int argc, char** argv) try {
    Fixture small(256), qwen_width(5120);
    sub0bench::BenchRunner runner(argc, argv, "sub0llm quantized projection (warm-cache slices)");
    runner.bench().unit("projection");
    runner.run("Q8_0 32x256, hand-written fixed-scale", [&] {
        small.hand_written(); ankerl::nanobench::doNotOptimizeAway(small.reference.data());
    });
    runner.run("Q8_0 32x256, denseq", [&] {
        small.project(); ankerl::nanobench::doNotOptimizeAway(small.output.data());
    });
    runner.run("Q8_0 32x5120, hand-written fixed-scale", [&] {
        qwen_width.hand_written(); ankerl::nanobench::doNotOptimizeAway(qwen_width.reference.data());
    });
    runner.run("Q8_0 32x5120, denseq", [&] {
        qwen_width.project(); ankerl::nanobench::doNotOptimizeAway(qwen_width.output.data());
    });
    add_iq1s_benchmarks(runner);
    sub0bench::add_iq2xxs_benchmarks(runner);
    sub0bench::add_float_reduce_benchmarks(runner);
    return runner.finish();
} catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
}
