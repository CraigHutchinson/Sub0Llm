// CPU Muon kernel timing at small and training-scale square/wide/tall shapes.
#include "sub0/muon.hpp"

#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <random>
#include <string>
#include <utility>
#include <vector>

TEST_CASE("Muon orthogonalization with prepared scratch", "[muon][benchmark]") {
    for (auto [rows, cols] : {std::pair{96, 96}, {96, 384}, {384, 96},
                             {448, 448}, {448, 1792}, {1792, 448}}) {
        const auto mn = static_cast<std::size_t>(rows) * cols;
        const auto m = static_cast<std::size_t>(std::min(rows, cols));
        std::vector<float> input(mn), output(mn);
        std::vector<float> scratch(sub0::muon::scratch_floats(mn, m * m));
        std::mt19937 rng(42);
        std::normal_distribution<float> normal(0.f, 1.f);
        for (float& v : input) v = normal(rng);

        BENCHMARK(std::to_string(rows) + "x" + std::to_string(cols)) {
            sub0::muon::newton_schulz5(input.data(), rows, cols, output.data(), scratch);
            return output[mn / 2];
        };
    }
}
