// blend_tests.cpp -- engine-free tests for BlendSource (include/sub0/blend.hpp) and the staged,
// epoch-fair deficit scheduler (include/sub0/blend_schedule.hpp). No model: validates the scheduler's
// fairness convergence (the whole reason it replaced the old flat-weight picker), the stage-transition
// starvation fix, the rng-neutrality invariant (now unconditional -- source selection is fully
// deterministic, not just for a single source), and that a blended draw's window stays inside the
// CHOSEN source's bounds + documents.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "sub0/blend.hpp"
#include "sub0/window.hpp"
#include "sub0/blend_schedule.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <span>
#include <string>
#include <vector>

using sub0::BlendSource;
using sub0::BlendFairness;
using sub0::ResolvedSchedule;

namespace {

// A source backed by a caller-owned int32 token buffer + doc index. Returned by value; the buffers
// live in the fixture so the borrowed spans stay valid for the test's lifetime.
BlendSource make_source(std::string name, const std::vector<int>& toks,
                        const std::vector<std::uint64_t>& docs,
                        const std::vector<std::uint8_t>& mask = {}) {
    BlendSource s;
    s.name = std::move(name);
    s.view = sub0::TokView::over_int32(toks.data(), toks.size());
    s.docs = std::span<const std::uint64_t>(docs);
    s.mask = std::span<const std::uint8_t>(mask);
    return s;
}

// A ResolvedSchedule with ONE stage covering [0, +inf), rates given in `sources` order.
ResolvedSchedule single_stage(std::initializer_list<double> rates) {
    ResolvedSchedule sched;
    sched.until_epoch = { std::numeric_limits<double>::infinity() };
    sched.rate = { std::vector<double>(rates) };
    return sched;
}

}  // namespace

TEST_CASE("blend: pick_source_staged converges each source's own epoch progress toward equal, "
         "regardless of size", "[blend]") {
    // A 1000x size ratio, mirroring (at a tractable scale) the real incident this scheduler fixes: a
    // small curriculum blended against a huge base corpus. Equal target rates (the default schema).
    std::vector<int> small(100, 1), big(100000, 2);
    std::vector<std::uint64_t> d_small{0}, d_big{0};
    std::vector<BlendSource> sources{ make_source("small", small, d_small),
                                      make_source("big",   big,   d_big) };
    const ResolvedSchedule sched = single_stage({1.0, 1.0});

    BlendFairness fair(sources.size());
    std::mt19937 rng(1);
    for (int i = 0; i < 20000; ++i)
        sub0::sample_blend_staged(rng, fair, sources, sched, /*T=*/4, /*frac_epoch=*/0.0);

    const double progress_small = fair.drawn_tokens[0] / static_cast<double>(sources[0].view.size());
    const double progress_big   = fair.drawn_tokens[1] / static_cast<double>(sources[1].view.size());
    // Under the OLD flat-weight scheme, "small" would be many hundreds of epochs deep while "big" was
    // still at a small fraction of one -- here both must track each other closely.
    REQUIRE(progress_small > 0.0);
    REQUIRE(progress_big > 0.0);
    REQUIRE(std::abs(progress_small - progress_big) < 0.05);
}

TEST_CASE("blend: pick_source_staged with unequal target rates converges to THAT ratio, not 1:1",
         "[blend]") {
    std::vector<int> a(1000, 1), b(1000, 2);
    std::vector<std::uint64_t> da{0}, db{0};
    std::vector<BlendSource> sources{ make_source("a", a, da), make_source("b", b, db) };
    const ResolvedSchedule sched = single_stage({3.0, 1.0});   // a should epoch 3x faster than b

    BlendFairness fair(sources.size());
    std::mt19937 rng(2);
    for (int i = 0; i < 20000; ++i)
        sub0::sample_blend_staged(rng, fair, sources, sched, /*T=*/4, /*frac_epoch=*/0.0);

    const double progress_a = fair.drawn_tokens[0] / static_cast<double>(sources[0].view.size());
    const double progress_b = fair.drawn_tokens[1] / static_cast<double>(sources[1].view.size());
    REQUIRE(progress_b > 0.0);
    const double ratio = progress_a / progress_b;
    REQUIRE(ratio > 2.7);
    REQUIRE(ratio < 3.3);
}

TEST_CASE("blend: pick_source_staged never picks a source with rate<=0 in the current stage",
         "[blend]") {
    std::vector<int> a(1000, 1), b(1000, 2);
    std::vector<std::uint64_t> da{0}, db{0};
    std::vector<BlendSource> sources{ make_source("a", a, da), make_source("b", b, db) };
    const ResolvedSchedule sched = single_stage({1.0, 0.0});   // "b" is declared but inactive

    BlendFairness fair(sources.size());
    std::mt19937 rng(3);
    for (int i = 0; i < 500; ++i) {
        const int s = sub0::pick_source_staged(fair, sources, sched, 0.0);
        REQUIRE(s == 0);
    }
    REQUIRE(fair.drawn_tokens[1] == 0.0);
}

TEST_CASE("blend: sample_blend_staged consumes NO extra rng beyond window sampling, for any source "
         "count (resume determinism)", "[blend]") {
    // Source selection is now fully deterministic (argmin over accumulated state), not a weighted-random
    // draw -- a real, intentional change from the old pick_source, which drew one rng sample per call for
    // >=2 sources. Verify the new invariant: sample_blend_staged's rng consumption exactly matches
    // sample_window's own, whether there is 1 source or several.
    std::vector<int> a(1000, 1), b(1000, 2), c(1000, 3);
    std::vector<std::uint64_t> da{0}, db{0}, dc{0};

    for (int n_sources : {1, 2, 3}) {
        std::vector<BlendSource> sources{ make_source("a", a, da) };
        std::vector<double> rates{1.0};
        if (n_sources >= 2) { sources.push_back(make_source("b", b, db)); rates.push_back(1.0); }
        if (n_sources >= 3) { sources.push_back(make_source("c", c, dc)); rates.push_back(1.0); }
        ResolvedSchedule sched;
        sched.until_epoch = { std::numeric_limits<double>::infinity() };
        sched.rate = { rates };

        BlendFairness fair(sources.size());
        std::mt19937 rng(42), ref(42);
        for (int i = 0; i < 50; ++i) {
            sub0::sample_blend_staged(rng, fair, sources, sched, /*T=*/4, /*frac_epoch=*/0.0);
            sub0::sample_window(ref, /*T=*/4, sources[0].view.size(), sources[0].docs);
        }
        // Not a claim the two engines are on the same DRAW (different sources may have been chosen for
        // the staged call) -- only that the NUMBER of draws consumed matches exactly, i.e. picking added
        // zero rng cost regardless of source count.
        REQUIRE(rng == ref);
    }
}

TEST_CASE("blend: a stage transition with a rate DECREASE bounds starvation instead of leaving it "
         "unbounded", "[blend]") {
    // Reproduces the design-review regression: a source's rate tapers from 0.5 to 0.05 at a stage
    // boundary. Without the transition clamp, the source's normalized progress (progress/rate) jumps
    // ~10x at the boundary and it would not be redrawn for thousands of windows. With the clamp, it
    // must be redrawn again "soon" (bounded, not unbounded).
    std::vector<int> a(100000, 1), b(100000, 2);
    std::vector<std::uint64_t> da{0}, db{0};
    std::vector<BlendSource> sources{ make_source("a", a, da), make_source("b", b, db) };

    ResolvedSchedule sched;
    sched.until_epoch = { 0.5, std::numeric_limits<double>::infinity() };
    sched.rate = { {0.5, 0.5}, {0.95, 0.05} };   // stage 2 tapers "b" from equal to 5%

    BlendFairness fair(sources.size());
    std::mt19937 rng(4);
    // Run stage 1 to a comparable progress on both sources.
    for (int i = 0; i < 4000; ++i)
        sub0::sample_blend_staged(rng, fair, sources, sched, /*T=*/8, /*frac_epoch=*/0.3);
    REQUIRE(fair.drawn_tokens[1] > 0.0);   // "b" got real draws in stage 1

    // Cross into stage 2 (frac_epoch >= 0.5) and confirm "b" is drawn again within a bounded window,
    // not starved for thousands of draws the way the un-clamped math would produce.
    bool b_drawn_again = false;
    for (int i = 0; i < 200 && !b_drawn_again; ++i) {
        const int s = sub0::pick_source_staged(fair, sources, sched, 0.5);
        if (s == 1) b_drawn_again = true;
        sub0::sample_window(rng, 8, sources[static_cast<std::size_t>(s)].view.size(),
                            sources[static_cast<std::size_t>(s)].docs);
        fair.drawn_tokens[static_cast<std::size_t>(s)] += 8;
    }
    REQUIRE(b_drawn_again);
}

TEST_CASE("blend: a source newly activated mid-schedule starts at progress 0 and gets an initial "
         "catch-up burst", "[blend]") {
    std::vector<int> a(100000, 1), b(100000, 2);
    std::vector<std::uint64_t> da{0}, db{0};
    std::vector<BlendSource> sources{ make_source("a", a, da), make_source("b", b, db) };

    ResolvedSchedule sched;
    sched.until_epoch = { 0.5, std::numeric_limits<double>::infinity() };
    sched.rate = { {1.0, 0.0}, {1.0, 1.0} };   // "b" only activates in stage 2

    BlendFairness fair(sources.size());
    std::mt19937 rng(5);
    for (int i = 0; i < 2000; ++i)
        sub0::sample_blend_staged(rng, fair, sources, sched, /*T=*/8, /*frac_epoch=*/0.3);
    REQUIRE(fair.drawn_tokens[1] == 0.0);   // never drawn in stage 1

    // Stage 2: "b" starts at progress 0 while "a" is already well ahead -- it should win argmin
    // immediately (the intended burst), not wait.
    const int s = sub0::pick_source_staged(fair, sources, sched, 0.5);
    REQUIRE(s == 1);
}

TEST_CASE("blend: resolve_stage picks the stage covering frac_epoch, boundaries and end handled",
         "[blend]") {
    ResolvedSchedule sched;
    sched.until_epoch = { 0.5, 0.9, std::numeric_limits<double>::infinity() };
    sched.rate = { {1.0}, {1.0}, {1.0} };

    CHECK(sub0::resolve_stage(sched, 0.0)  == 0);
    CHECK(sub0::resolve_stage(sched, 0.49) == 0);
    CHECK(sub0::resolve_stage(sched, 0.5)  == 1);   // exactly on a boundary -> the NEXT stage
    CHECK(sub0::resolve_stage(sched, 0.89) == 1);
    CHECK(sub0::resolve_stage(sched, 0.9)  == 2);
    CHECK(sub0::resolve_stage(sched, 1000.0) == 2);  // "end" sentinel covers everything past it
}

TEST_CASE("blend: sample_blend_staged keeps each window inside the chosen source and one document",
         "[blend]") {
    // Source 0: three 20-token documents. Source 1: one 50-token document.
    std::vector<int> a(60), b(50);
    std::iota(a.begin(), a.end(), 0);
    std::iota(b.begin(), b.end(), 1000);
    std::vector<std::uint64_t> da{0, 20, 40}, db{0};
    std::vector<BlendSource> sources{ make_source("a", a, da), make_source("b", b, db) };
    const ResolvedSchedule sched = single_stage({0.5, 0.5});

    BlendFairness fair(sources.size());
    std::mt19937 rng(7);
    for (int it = 0; it < 5000; ++it) {
        const sub0::BlendDraw d = sub0::sample_blend_staged(rng, fair, sources, sched, 8, 0.0);
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
    REQUIRE_FALSE(make_source("a", a, da).masked());
    REQUIRE(make_source("a", a, da, m).masked());
}

TEST_CASE("blend: carry_forward_by_name matches progress by NAME, not position", "[blend]") {
    // The --blend-config-replace scenario this exists for: a schedule swap that keeps the same NUMBER of
    // sources but changes what an index means. "base" keeps its progress even though it moved from index
    // 0 to index 1; "scratch" is dropped (no longer declared); "op" is new and starts at 0.
    const std::vector<std::string> old_names{"base", "scratch"};
    const std::vector<double>      old_tokens{1000.0, 250.0};
    const std::vector<std::string> new_names{"op", "base"};   // reordered, "scratch"->"op"

    const std::vector<double> out = sub0::carry_forward_by_name(old_names, old_tokens, new_names);
    REQUIRE(out.size() == 2);
    CHECK(out[0] == 0.0);       // "op" is new
    CHECK(out[1] == 1000.0);    // "base" carried forward despite moving index 0 -> 1
}

TEST_CASE("blend: carry_forward_by_name is a no-op when names and order are unchanged", "[blend]") {
    const std::vector<std::string> names{"base", "scratch"};
    const std::vector<double> tokens{500.0, 42.0};
    const std::vector<double> out = sub0::carry_forward_by_name(names, tokens, names);
    CHECK(out == tokens);
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

// --- corpus subsetting: train on a distributed FRACTION of the documents ------------------------
// Replaces the deleted fineweb_smoke.txt prefix file. The properties that make a subset usable are
// not "it is smaller" -- they are distribution, determinism, and boundary safety. Each is pinned.

TEST_CASE("corpus subset: selects ~the requested fraction, spread across the whole corpus",
          "[frontend][window][subset]") {
    constexpr std::size_t NDOCS = 20000;
    for (double f : {0.05, 0.25, 0.5, 0.9}) {
        std::size_t n = 0, first_half = 0;
        for (std::size_t d = 0; d < NDOCS; ++d)
            if (sub0::doc_in_subset(d, 1234u, f)) { ++n; if (d < NDOCS / 2) ++first_half; }
        const double got = static_cast<double>(n) / static_cast<double>(NDOCS);
        CHECK(got == Catch::Approx(f).margin(0.02));            // right size
        // ...and SPREAD, which is the whole point: a prefix file would put 100% in the first half.
        // Each half should hold about half the selected documents.
        CHECK(static_cast<double>(first_half) / static_cast<double>(n) == Catch::Approx(0.5).margin(0.05));
    }
}

TEST_CASE("corpus subset: deterministic in (seed, fraction), and different seeds differ",
          "[frontend][window][subset]") {
    // Reproducing a run must need only the seed and the fraction -- no subset file to ship or drift.
    for (std::size_t d = 0; d < 500; ++d)
        CHECK(sub0::doc_in_subset(d, 7u, 0.3) == sub0::doc_in_subset(d, 7u, 0.3));
    int differ = 0;
    for (std::size_t d = 0; d < 2000; ++d)
        if (sub0::doc_in_subset(d, 7u, 0.3) != sub0::doc_in_subset(d, 8u, 0.3)) ++differ;
    CHECK(differ > 100);                                        // a different seed is a different subset
}

TEST_CASE("corpus subset: fraction is monotone, so subsets nest", "[frontend][window][subset]") {
    // A document selected at 25% must still be selected at 50%: the threshold is a comparison against
    // one hash per document, not a re-draw. That makes 10/25/50/100% a genuine nested sweep rather
    // than four unrelated samples -- which is what a learning-curve study needs.
    for (std::size_t d = 0; d < 5000; ++d)
        if (sub0::doc_in_subset(d, 42u, 0.25)) CHECK(sub0::doc_in_subset(d, 42u, 0.5));
}

TEST_CASE("corpus subset: the complement is the exact held-out set", "[frontend][window][subset]") {
    // Every document is in exactly one of the two, so "the 75% not trained on" is unseen BY
    // CONSTRUCTION -- a stronger guarantee than the positional train/val tail split.
    std::size_t in = 0, out = 0;
    for (std::size_t d = 0; d < 3000; ++d)
        (sub0::doc_in_subset(d, 99u, 0.25) ? in : out)++;
    CHECK(in + out == 3000);
    CHECK(in > 600);
    CHECK(in < 900);
}

TEST_CASE("corpus subset: fraction 1.0 is bit-for-bit the unsubsetted sampler",
          "[frontend][window][subset]") {
    // The default path must not shift by even one draw, or every existing result moves under it.
    std::vector<std::uint64_t> docs;
    for (std::uint64_t d = 0; d < 400; ++d) docs.push_back(d * 50);
    std::mt19937 a(2024u), b(2024u);
    for (int i = 0; i < 500; ++i) {
        const sub0::Window wa = sub0::sample_window(a, 8, 20000, docs);
        const sub0::Window wb = sub0::sample_window(b, 8, 20000, docs, 1.0, 12345u);
        REQUIRE(wa.start == wb.start);
        REQUIRE(wa.len   == wb.len);
    }
}

TEST_CASE("corpus subset: sampled windows come ONLY from selected documents, and stay in-document",
          "[frontend][window][subset]") {
    // The safety property. A subset must never be a reason a window straddles a document boundary --
    // that is the unsoundness the <|endoftext|>-only scan was introduced to remove.
    constexpr std::size_t NDOC = 600, DOCLEN = 50, TRAIN = NDOC * DOCLEN;
    std::vector<std::uint64_t> docs;
    for (std::uint64_t d = 0; d < NDOC; ++d) docs.push_back(d * DOCLEN);
    const double f = 0.1;
    const std::uint64_t seed = 555u;
    std::mt19937 rng(31337u);
    for (int i = 0; i < 4000; ++i) {
        const sub0::Window w = sub0::sample_window(rng, 8, TRAIN, docs, f, seed);
        const std::size_t k = sub0::doc_of(docs, w.start);
        CHECK(sub0::doc_in_subset(k, seed, f));                       // from a selected document
        CHECK(w.start + static_cast<std::size_t>(w.len) <= docs[k] + DOCLEN);   // and inside it
    }
}
