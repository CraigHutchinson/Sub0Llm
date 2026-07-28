// frontend_bench.cpp — Catch2 microbenchmarks for the engine-free frontend hot paths the configurator
// runs over the WHOLE corpus at ingest (per-byte cost matters at GBs). Statistical timing (Catch2 does
// warmup + samples + mean/stddev) for the CPU optimization work. Device throughput is measured elsewhere
// (`sub0-cuda-selftest bench`, cudaEvent-based).
//
// These benches are the MEASUREMENT + GATING harness for the tokenizer-throughput workstream (see the
// per-pass plan): they isolate the three corpus passes the configurator actually runs (Scan::add_names,
// Scan::add_words, and the Pass-3 full encode) plus the runtime encode/round-trip, on a corpus large
// enough (~MBs) that a real optimization moves the mean well outside the sample stddev. To A/B a change,
// build this target with the optimization compiled in behind its compile-time gate and compare the same
// named benchmark's mean before/after IN ONE PROCESS INVOCATION (the project's own perf notes: separate
// sequential process runs on this laptop carry real thermal confounds; Catch2's interleaved samples do
// not). Correctness of any such change is gated separately by the byte/token-identical differential test
// (tests/ws6_simd_differential_tests.cpp) and the frontend round-trip suite -- NOT by these numbers.
//
// Run: cmake -DSUB0_BUILD_BENCHMARKS=ON ... && cmake --build ... --target sub0_frontend_bench
//      ./sub0_frontend_bench

#include <catch2/catch_test_macros.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>

#include "sub0/casing.hpp"
#include "sub0/tokenizer.hpp"

#include <cmath>
#include <random>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace {
// A small, representative seed (mixed case, punctuation, quotes, contractions, code-like tokens,
// digits, newline) -- the shapes the truecaser/JOIN encoder actually branch on.
constexpr std::string_view kSeed =
    "The quick brown fox jumps over the lazy dog. The dog ran and the cat slept quietly.\n"
    "\"Hello,\" she said, and they smiled. Once there was a little girl who loved to play outside.\n"
    "CamelCaseWords and code_like_tokens and numbers 1234 test the segmentation boundaries.\n"
    "Don't forget the well-known non-commercial cases; f(x) and a[i] glue their brackets too.\n";

// Build a ~target_bytes corpus by repeating the seed. Real corpora are Zipfian (a small head of words
// carries most occurrences), and repetition reproduces exactly that head -- which is what the per-word
// encode cache the throughput work targets is meant to exploit, so this is a fair proxy for the
// cache-hit rate, not an artificially easy one. Sized once, reused across benchmarks (never rebuilt
// inside a timed lambda).
std::string make_corpus(std::size_t target_bytes) {
    std::string c;
    c.reserve(target_bytes + kSeed.size());
    while (c.size() < target_bytes) c.append(kSeed);
    return c;
}

// The text encoded each iteration by the runtime-encode benches -- the per-call work the engine does
// per prompt and the configurator does per corpus chunk.
const std::string kText =
    "The quick brown fox jumps over the lazy dog, and they said \"hello\" once more outside.";

// A DIVERSE, Zipf-distributed corpus -- the honest scale proxy the repeated-seed corpus is not. The
// seed corpus has a tiny fixed vocabulary, so the per-word encode cache hits ~100% and never grows;
// that flatters the cache and hides its map-growth/rehash cost. Here ~`vocab` distinct pseudo-words
// are emitted with a head-heavy (cube-of-uniform) rank bias, reproducing a real corpus's long tail:
// the cache grows to realistic size AND still has a dominant head, so the measured win is what a
// 50 GB+ corpus would actually see, not a best case. Deterministic (fixed seed) for stable timing.
std::string make_diverse_corpus(std::size_t target_bytes, unsigned vocab) {
    std::mt19937 rng(12345u);
    std::vector<std::string> words;
    words.reserve(vocab);
    for (unsigned i = 0; i < vocab; ++i) {
        const int len = 3 + static_cast<int>(rng() % 10);
        std::string w;
        w.reserve(static_cast<std::size_t>(len));
        for (int k = 0; k < len; ++k) w.push_back(static_cast<char>('a' + rng() % 26));
        if (i % 7 == 0) w[0] = static_cast<char>('A' + rng() % 26);   // a realistic minority are capitalised
        words.push_back(std::move(w));
    }
    const char punct[4] = {'.', ',', ';', '\n'};
    std::string c;
    c.reserve(target_bytes + 16);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    while (c.size() < target_bytes) {
        const double x = u(rng);
        const unsigned r = static_cast<unsigned>(static_cast<double>(vocab) * x * x * x);   // head-heavy rank
        c += words[std::min(r, vocab - 1)];
        if (rng() % 12 == 0) c.push_back(punct[rng() % 4]);
        c.push_back(' ');
    }
    return c;
}
}  // namespace

TEST_CASE("frontend tokenizer throughput", "[!benchmark]") {
    const sub0::tok::Tokenizer t = sub0::tok::learn(kSeed);

    // encode = normalize + truecase + JOIN/Viterbi segmentation -- the throughput-critical ingest path.
    BENCHMARK("encode") {
        return sub0::tok::encode(t, kText);
    };

    // The full lossless round-trip (the fuzz-tested invariant), timed end to end.
    BENCHMARK("encode + detokenize round-trip") {
        return sub0::tok::detokenize(t, sub0::tok::encode(t, kText));
    };
}

// Per-pass corpus throughput: the three passes the configurator streams over the WHOLE corpus. Each is
// isolated so the throughput work can attribute a win (or regression) to the pass it changed, instead of
// only seeing the blended end-to-end number. The corpus is a few MB so per-byte cost dominates the fixed
// per-call overhead; divide the Catch2 mean by the corpus size for a MB/s figure.
TEST_CASE("frontend corpus passes throughput", "[!benchmark]") {
    constexpr std::size_t kCorpusBytes = 4u << 20;   // ~4 MB -- stable timing, quick to build
    const std::string corpus = make_corpus(kCorpusBytes);

    // Pass 1 -- name detection. A fresh Scan each iteration (add_names accumulates into its tables, so
    // reusing one across samples would grow them unboundedly and skew the timing); the construction cost
    // is negligible against a multi-MB scan.
    BENCHMARK("pass1 add_names (4 MB)") {
        sub0::tok::Scan s;
        s.add_names(corpus);
        return s.raw_bytes;
    };

    // Pass 2 -- truecase + unique-word table build. Needs the derived attested set (pass 1's output), so
    // compute it once outside the timed loop; the bench then measures only add_words, the pass the
    // per-word classification cost (the WS6-identified dominant term) lives in.
    sub0::tok::Scan names;
    names.add_names(corpus);
    const std::unordered_set<std::string> attested = sub0::tok::derive_attested(names);
    BENCHMARK("pass2 add_words (4 MB)") {
        sub0::tok::Scan s;
        s.add_words(corpus, attested);
        return s.word_syms.size();
    };

    // Pass 3 -- the full encode the configurator runs to emit corpus.tok: normalize + truecase +
    // encode_join + per-word Viterbi. Historically the dominant corpus pass (it re-ran the per-word
    // Viterbi for every occurrence); encode_join now memoises each word's id sequence by its byte key
    // (the call-local word cache), which collapses that cost on the Zipf head -- this benchmark is the
    // gate that proves and guards that win.
    const sub0::tok::Tokenizer t = sub0::tok::learn(corpus);
    BENCHMARK("pass3 full encode (4 MB)") {
        return sub0::tok::encode(t, corpus);
    };

    // Attribution for passes 2 and 3 (both run these shared per-byte primitives): normalize_text and
    // truecase_tokenize. Isolating them shows how much of a pass's cost is the transform vs. the pass's
    // own table/encode work, so the second optimization pass targets the real hot spot instead of
    // guessing. normalize_text allocates an n-sized classify buffer + the output string each call;
    // truecase_tokenize allocates the 4-bytes-per-symbol int stream -- both are candidates for reused
    // scratch (AGENTS.md §1) if they dominate.
    BENCHMARK("normalize_text (4 MB)") {
        long replaced = 0;
        return sub0::casing::normalize_text(corpus, replaced);
    };
    long _r = 0;
    const std::string norm = sub0::casing::normalize_text(corpus, _r);
    BENCHMARK("truecase_tokenize (4 MB)") {
        return sub0::casing::truecase_tokenize(norm, attested, nullptr);
    };
}

// Cache-hit locality: encoding a highly repetitive in-vocabulary stream is the Zipf-head scenario the
// per-word encode cache collapses (a word seen before looks up its ids instead of re-running Viterbi --
// gigatoken's central lever). This isolates the cache's best case so a regression that weakened or broke
// the memoisation shows up here as a sharp slowdown even if the 4 MB blended number barely moved.
TEST_CASE("frontend encode cache-hit locality", "[!benchmark]") {
    const sub0::tok::Tokenizer t = sub0::tok::learn(kSeed);
    const std::string repeated = make_corpus(64u << 10);   // 64 KB of the repeated seed
    BENCHMARK("encode repeated-word stream (64 KB)") {
        return sub0::tok::encode(t, repeated);
    };
}

// Realistic-scale throughput on a DIVERSE Zipfian corpus (see make_diverse_corpus): the honest test of
// the memoisation win at 50 GB+. If the per-word cache's map growth/rehash on a large unique-word set
// were a real cost, pass3 here would lag the repetitive-corpus pass3 by more than the extra unique-word
// Viterbi accounts for -- this benchmark is what would expose that (and gate a fix such as reserving
// the cache map). pass2's word-table build faces the same growth, so it is measured alongside.
TEST_CASE("frontend diverse-corpus throughput (Zipfian)", "[!benchmark]") {
    constexpr std::size_t kBytes = 4u << 20;      // ~4 MB, matches the repetitive pass benches
    constexpr unsigned    kVocab = 20000;         // realistic distinct-word count for a few MB
    const std::string corpus = make_diverse_corpus(kBytes, kVocab);
    const sub0::tok::Tokenizer t = sub0::tok::learn(corpus);

    sub0::tok::Scan names;
    names.add_names(corpus);
    const std::unordered_set<std::string> attested = sub0::tok::derive_attested(names);
    BENCHMARK("pass2 add_words diverse (4 MB)") {
        sub0::tok::Scan s;
        s.add_words(corpus, attested);
        return s.word_syms.size();
    };
    BENCHMARK("pass3 full encode diverse (4 MB)") {
        return sub0::tok::encode(t, corpus);
    };
}
