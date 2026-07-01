// frontend_bench.cpp — Catch2 microbenchmarks for the engine-free frontend hot paths the configurator
// runs over the WHOLE corpus at ingest (per-byte cost matters at GBs). Statistical timing (Catch2 does
// warmup + samples + mean/stddev) for the CPU optimization work. Device throughput is measured elsewhere
// (`sub0-cuda-selftest bench`, cudaEvent-based).
//
// Run: cmake -DSUB0_BUILD_BENCHMARKS=ON ... && cmake --build ... --target sub0_frontend_bench
//      ./sub0_frontend_bench

#include <catch2/catch_test_macros.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>

#include "sub0/tokenizer.hpp"

#include <string>
#include <string_view>
#include <vector>

namespace {
// A small, representative corpus to learn a tokenizer from (mixed case, punctuation, quotes, newline).
constexpr std::string_view kCorpus =
    "The quick brown fox jumps over the lazy dog. The dog ran and the cat slept quietly.\n"
    "\"Hello,\" she said, and they smiled. Once there was a little girl who loved to play outside.\n"
    "CamelCaseWords and code_like_tokens and numbers 1234 test the segmentation boundaries.\n";

// The text encoded each iteration -- the per-call work the configurator does per corpus chunk.
const std::string kText =
    "The quick brown fox jumps over the lazy dog, and they said \"hello\" once more outside.";
}  // namespace

TEST_CASE("frontend tokenizer throughput", "[!benchmark]") {
    const sub0::tok::Tokenizer t = sub0::tok::learn(kCorpus);

    // encode = normalize + truecase + JOIN/Viterbi segmentation -- the throughput-critical ingest path.
    BENCHMARK("encode") {
        return sub0::tok::encode(t, kText);
    };

    // The full lossless round-trip (the fuzz-tested invariant), timed end to end.
    BENCHMARK("encode + detokenize round-trip") {
        return sub0::tok::detokenize(t, sub0::tok::encode(t, kText));
    };
}
