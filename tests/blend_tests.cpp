// blend_tests.cpp -- engine-free tests for the weighted per-window corpus blender
// (include/sub0/blend.hpp). No model: validates the source-selection statistics, the
// single-source rng-neutrality invariant (the resume-determinism guarantee), and that a blended
// draw's window stays inside the CHOSEN source's bounds + documents.

#include <catch2/catch_test_macros.hpp>

#include "sub0/blend.hpp"

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <random>
#include <span>
#include <vector>

using sub0::BlendSource;

namespace {

// A source backed by a caller-owned int32 token buffer + doc index. Returned by value; the buffers
// live in the fixture so the borrowed spans stay valid for the test's lifetime.
BlendSource make_source(const std::vector<int>& toks, const std::vector<std::uint64_t>& docs,
                        double weight, const std::vector<std::uint8_t>& mask = {}) {
    BlendSource s;
    s.view   = sub0::TokView::over_int32(toks.data(), toks.size());
    s.docs   = std::span<const std::uint64_t>(docs);
    s.mask   = std::span<const std::uint8_t>(mask);
    s.weight = weight;
    return s;
}

}  // namespace

TEST_CASE("blend: pick_source converges to the configured weight ratio", "[blend]") {
    std::vector<int> a(1000, 1), b(1000, 2);
    std::vector<std::uint64_t> da{0}, db{0};
    std::vector<BlendSource> sources{ make_source(a, da, 0.8), make_source(b, db, 0.2) };

    std::mt19937 rng(12345);
    int c0 = 0;
    const int N = 200000;
    for (int i = 0; i < N; ++i) c0 += (sub0::pick_source(rng, sources) == 0);
    const double frac0 = static_cast<double>(c0) / N;
    REQUIRE(frac0 > 0.79);
    REQUIRE(frac0 < 0.81);
}

TEST_CASE("blend: a single source consumes NO rng draw (resume determinism)", "[blend]") {
    std::vector<int> a(100, 7);
    std::vector<std::uint64_t> da{0};
    std::vector<BlendSource> one{ make_source(a, da, 1.0) };

    // The rng state after pick_source must equal the untouched state: a single-source blend must not
    // perturb the window-sampling stream the resume point depends on.
    std::mt19937 rng(999), ref(999);
    for (int i = 0; i < 50; ++i) REQUIRE(sub0::pick_source(rng, one) == 0);
    REQUIRE(rng == ref);   // identical engine state -> zero draws consumed
}

TEST_CASE("blend: sample_blend keeps each window inside the chosen source and one document", "[blend]") {
    // Source 0: three 20-token documents. Source 1: one 50-token document.
    std::vector<int> a(60), b(50);
    std::iota(a.begin(), a.end(), 0);
    std::iota(b.begin(), b.end(), 1000);
    std::vector<std::uint64_t> da{0, 20, 40}, db{0};
    std::vector<BlendSource> sources{ make_source(a, da, 0.5), make_source(b, db, 0.5) };

    std::mt19937 rng(7);
    for (int it = 0; it < 5000; ++it) {
        const sub0::BlendDraw d = sub0::sample_blend(rng, 8, sources);
        REQUIRE((d.src == 0 || d.src == 1));
        const BlendSource& src = sources[static_cast<std::size_t>(d.src)];
        const std::size_t end = d.win.start + static_cast<std::size_t>(d.win.len);
        REQUIRE(d.win.len >= 1);
        REQUIRE(end < src.view.size());                          // window (incl. shifted target) in bounds
        // The whole window [start, start+len] stays within one document of the chosen source.
        const std::size_t k = sub0::doc_of(src.docs, d.win.start);
        const std::uint64_t doc_end = (k + 1 < src.docs.size()) ? src.docs[k + 1] : src.view.size();
        REQUIRE(end < doc_end);
    }
}

TEST_CASE("blend: masked() reflects whether a source carries a loss mask", "[blend]") {
    std::vector<int> a(10, 1);
    std::vector<std::uint64_t> da{0};
    std::vector<std::uint8_t> m(10, 1);
    REQUIRE_FALSE(make_source(a, da, 1.0).masked());
    REQUIRE(make_source(a, da, 1.0, m).masked());
}

TEST_CASE("blend: doc_of finds the document owning a corpus position", "[blend]") {
    // Three docs: [0,20), [20,40), [40,60).
    const std::vector<std::uint64_t> docs{0, 20, 40};
    CHECK(sub0::doc_of(docs, 0)  == 0);   // first position of doc 0
    CHECK(sub0::doc_of(docs, 19) == 0);   // last position of doc 0
    CHECK(sub0::doc_of(docs, 20) == 1);   // exactly on a boundary -> the NEXT doc, not the previous
    CHECK(sub0::doc_of(docs, 39) == 1);
    CHECK(sub0::doc_of(docs, 40) == 2);
    CHECK(sub0::doc_of(docs, 59) == 2);   // last position of the last doc

    // Single document: every position resolves to doc 0.
    const std::vector<std::uint64_t> one_doc{0};
    CHECK(sub0::doc_of(one_doc, 0)   == 0);
    CHECK(sub0::doc_of(one_doc, 999) == 0);
}
