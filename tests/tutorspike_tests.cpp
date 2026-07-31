// tutorspike_tests.cpp -- the mastery surface's arithmetic (include/sub0/tutorspike.hpp).
//
// Engine-free: the surface is fed numbers, so its behaviour is fully testable with no model. That is
// the point of keeping it that way -- the parts that can be wrong here (a velocity normalised by the
// wrong denominator, a transfer term that silently includes the entry's own learning, a sign flip that
// turns forgetting into mastery) are arithmetic, and every one of them would still produce a plausible
// heat map.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "sub0/tutorspike.hpp"

#include <cmath>
#include <algorithm>
#include <cstdio>
#include <numeric>
#include <filesystem>
#include <fstream>
#include <limits>
#include <random>
#include <string>

using sub0::tutor::Surface;
using sub0::tutor::DriftProbe;
using sub0::tutor::Manifest;

TEST_CASE("surface: velocity is normalised by APPLIED LEARNING, not by visits", "[tutor]") {
    // The normalization trap, as an assertion. Two entries learn the SAME amount of nelbo, but entry 1
    // was trained at a tenth of the learning rate. Per-visit normalization would call them equal; the
    // correct reading is that entry 1 is learning ten times FASTER per unit of learning applied to it.
    // That is the whole reason the denominator exists: a down-weighted entry must not look mastered
    // merely because it was down-weighted.
    Surface s;
    s.reset(2, /*typical_visit_applied=*/0.f);       // 0 => update velocity on every visit

    s.record(0, /*nelbo=*/5.0f, /*lr=*/0.001f, /*tokens=*/100);   // applied 0.1
    s.record(0, /*nelbo=*/4.0f, /*lr=*/0.001f, /*tokens=*/100);   // applied 0.2, nelbo -1.0

    s.record(1, /*nelbo=*/5.0f, /*lr=*/0.010f, /*tokens=*/100);   // applied 1.0
    s.record(1, /*nelbo=*/4.0f, /*lr=*/0.010f, /*tokens=*/100);   // applied 2.0, nelbo -1.0

    REQUIRE(s.at(0).velocity == Catch::Approx(10.0f));   // 1.0 nelbo / 0.1 applied
    REQUIRE(s.at(1).velocity == Catch::Approx(1.0f));    // 1.0 nelbo / 1.0 applied
    REQUIRE(s.at(0).velocity > s.at(1).velocity);
}

TEST_CASE("surface: falling nelbo is POSITIVE velocity, rising is negative", "[tutor]") {
    // Sign convention, pinned. A regressing (being-forgotten) entry must come out NEGATIVE so the
    // eventual weighting rule can tell it apart from a mastered one -- both of which sit near zero
    // velocity if the sign is dropped, which is exactly the confusion the scheme exists to avoid.
    Surface s;
    s.reset(2, 0.f);
    s.record(0, 5.0f, 0.001f, 100);
    s.record(0, 4.0f, 0.001f, 100);        // learning
    s.record(1, 4.0f, 0.001f, 100);
    s.record(1, 5.0f, 0.001f, 100);        // forgetting
    REQUIRE(s.at(0).velocity > 0.f);
    REQUIRE(s.at(1).velocity < 0.f);
}

TEST_CASE("surface: an unlearnable entry reads HIGH level and ~ZERO velocity", "[tutor]") {
    // The population the spike exists to separate. A level-based rule sees only the first number and
    // up-weights this forever; the velocity reading is what makes it distinguishable from an entry that
    // is merely difficult. Note the mastered case lands in the SAME velocity band -- level is what tells
    // those two apart, which is why the surface has to carry both.
    Surface s;
    s.reset(2, 0.f);
    for (int i = 0; i < 5; ++i) {
        s.record(0, 9.0f, 0.001f, 100);    // unlearnable: high, flat
        s.record(1, 0.5f, 0.001f, 100);    // mastered:    low,  flat
    }
    REQUIRE(std::fabs(s.at(0).velocity) < 1e-5f);
    REQUIRE(std::fabs(s.at(1).velocity) < 1e-5f);
    REQUIRE(s.at(0).nelbo > s.at(1).nelbo);            // only the LEVEL separates them
}

TEST_CASE("surface: transfer needs a post reading, and excludes the entry's own learning", "[tutor]") {
    // The conflict/reinforcement measurement. Without a post-update reading the between-visit change
    // still contains the previous visit's own learning and is not attributable to anything, so no
    // transfer is recorded at all -- silently attributing it would be the worst outcome.
    Surface s;
    s.reset(2, 0.f);

    s.record(0, 5.0f, 0.001f, 100);
    s.add_global_applied(10.0);
    s.record(0, 4.5f, 0.001f, 100);
    REQUIRE_FALSE(s.at(0).has_transfer());              // no post reading -> no claim made

    // With a post reading the split is well-defined: the entry's own update took it 5.0 -> 4.6, and it
    // then arrived at its next visit at 4.4. The 0.2 improvement in between happened while this entry
    // was NOT being trained, so it came from the rest of the corpus: reinforcement, and negative by the
    // sign convention.
    Surface t;
    t.reset(1, 0.f);
    t.record(0, 5.0f, 0.001f, 100);
    t.record_post(0, 4.6f);
    t.add_global_applied(10.0);
    t.record(0, 4.4f, 0.001f, 100);
    REQUIRE(t.at(0).has_transfer());
    REQUIRE(t.at(0).transfer == Catch::Approx((4.4f - 4.6f) / 10.0f));
    REQUIRE(t.at(0).transfer < 0.f);                   // improved unaided => reinforcement
}

TEST_CASE("surface: a CONFLICTED entry reads positive transfer", "[tutor]") {
    // The other sign, and the case the whole idea is for: this entry degraded while it was not being
    // trained, so something else in the corpus is displacing it. Invisible to any level-based rule, and
    // invisible to velocity too -- velocity only sees what happens across its own visits.
    Surface s;
    s.reset(1, 0.f);
    s.record(0, 3.0f, 0.001f, 100);
    s.record_post(0, 2.8f);
    s.add_global_applied(5.0);
    s.record(0, 3.4f, 0.001f, 100);                    // came back WORSE than it was left
    REQUIRE(s.at(0).transfer == Catch::Approx((3.4f - 2.8f) / 5.0f));
    REQUIRE(s.at(0).transfer > 0.f);
}

TEST_CASE("surface: coverage counts visited documents", "[tutor]") {
    // Poisson sampling leaves a large fraction of the corpus untouched early on, and an unvisited entry
    // has zero velocity for a reason that has nothing to do with learning. Coverage is what keeps
    // "never asked" from being read as "nothing to learn".
    Surface s;
    s.reset(4, 0.f);
    REQUIRE(s.coverage() == Catch::Approx(0.0));
    s.record(0, 5.0f, 0.001f, 100);
    s.record(2, 5.0f, 0.001f, 100);
    REQUIRE(s.coverage() == Catch::Approx(0.5));
}

TEST_CASE("surface: the velocity mark threshold defers division by a tiny denominator", "[tutor]") {
    // With a threshold set, velocity is not recomputed until enough NEW applied learning has landed.
    // Dividing a float-limited numerator by a near-zero denominator produces a large number made
    // entirely of rounding, which would read as a spectacular learning rate.
    Surface s;
    s.reset(1, /*typical_visit_applied=*/0.1f);        // threshold = 4 * 0.1 = 0.4
    s.record(0, 5.0f, 0.001f, 100);                    // applied 0.1 (baseline)
    s.record(0, 4.9f, 0.001f, 100);                    // +0.1 -- below threshold
    REQUIRE(s.at(0).velocity == Catch::Approx(0.0f));
    s.record(0, 4.8f, 0.001f, 100);
    s.record(0, 4.7f, 0.001f, 100);
    s.record(0, 4.6f, 0.001f, 100);                    // cumulative +0.4 -- now it fires
    REQUIRE(s.at(0).velocity > 0.f);
}

TEST_CASE("drift probe: excluded documents are identified and the floor is a MAGNITUDE", "[tutor]") {
    DriftProbe p;
    p.reset(/*n_docs=*/100, /*stride=*/10);
    REQUIRE(p.active());
    REQUIRE(p.docs().size() == 10);
    REQUIRE(p.is_probe(0));
    REQUIRE(p.is_probe(30));
    REQUIRE_FALSE(p.is_probe(31));

    // First round only establishes the baseline -- there is nothing to compare against yet.
    const std::vector<float> a{ 5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 5.0f };
    REQUIRE(p.observe(a, 1.0) == Catch::Approx(0.0));

    // Half the probes drift up and half down. A signed mean would cancel to zero and report "no drift"
    // while the model was in fact moving under every one of them -- the floor has to be a magnitude.
    const std::vector<float> b{ 5.2f, 4.8f, 5.2f, 4.8f, 5.2f, 4.8f, 5.2f, 4.8f, 5.2f, 4.8f };
    const double floor_ = p.observe(b, 2.0);
    REQUIRE(floor_ == Catch::Approx(0.2 / 2.0).epsilon(1e-4));   // mean |delta| per unit applied
}

TEST_CASE("surface: a post reading of exactly 0.0 is still a reading", "[tutor]") {
    // A 0-sentinel would classify this as absent. 0.0 is a LEGITIMATE loss here: cross_entropy returns
    // exactly 0 for a fully loss-masked window and the per-window readout reports that faithfully, so a
    // masked-source window can produce one. The failure would be silent and one-directional -- transfer
    // simply never recorded for those entries, reading as "no conflict found".
    Surface s;
    s.reset(1, 0.f);
    s.record(0, 3.0f, 0.001f, 100);
    s.record_post(0, 0.0f);                            // a real reading whose value happens to be zero
    REQUIRE(s.at(0).has_post());
    s.add_global_applied(4.0);
    s.record(0, 1.0f, 0.001f, 100);
    REQUIRE(s.at(0).has_transfer());
    REQUIRE(s.at(0).transfer == Catch::Approx((1.0f - 0.0f) / 4.0));
}

TEST_CASE("surface: a consumed post reading does not leak into the NEXT interval", "[tutor]") {
    // record() clears the flag after use. Without that, an entry re-scored once would keep attributing
    // every later interval against that same stale post value, and the transfer term would drift
    // further from meaning anything the longer the run went on.
    Surface s;
    s.reset(1, 0.f);
    s.record(0, 5.0f, 0.001f, 100);
    s.record_post(0, 4.0f);
    s.add_global_applied(2.0);
    s.record(0, 4.5f, 0.001f, 100);                    // consumes the post reading
    REQUIRE(s.at(0).transfer_n == 1);
    REQUIRE_FALSE(s.at(0).has_post());
    s.add_global_applied(2.0);
    s.record(0, 6.0f, 0.001f, 100);                    // no post this time -> no new claim
    REQUIRE(s.at(0).transfer_n == 1);
}

TEST_CASE("surface: event overflow is COUNTED, never silently dropped", "[tutor]") {
    // The buffer is fixed so the per-step path cannot allocate, which means it can fill. A diagnostic
    // stream with an unrecorded hole is worse than no stream: the hole is invisible downstream and reads
    // as an absence of events rather than an absence of recording.
    Surface s;
    s.reset(1, 0.f);
    s.reserve_events(2);
    for (int i = 0; i < 10; ++i) s.record(0, 5.0f - 0.1f * static_cast<float>(i), 0.001f, 100);
    REQUIRE(s.events().size() == 2);
    REQUIRE(s.dropped_events() > 0);
}

TEST_CASE("surface: coverage divides by REACHABLE documents", "[tutor]") {
    // The validation split and the drift probes are permanently out of training's reach, so dividing by
    // the total document count makes 100% unattainable and prints a fixed shortfall that looks like a
    // coverage failure. On the real corpus that shortfall (8.9%) coincidentally matched the old
    // sampler's genuine Poisson gap almost exactly.
    Surface s;
    s.reset(10, 0.f);
    s.set_eligible(4);                                  // 6 of 10 documents are unreachable by design
    s.record(0, 1.f, 0.001f, 10);
    s.record(1, 1.f, 0.001f, 10);
    s.record(2, 1.f, 0.001f, 10);
    s.record(3, 1.f, 0.001f, 10);
    REQUIRE(s.coverage() == Catch::Approx(1.0));        // every reachable document visited
}

TEST_CASE("drift probe: a non-finite score cannot poison the floor", "[tutor]") {
    // nelbo_cpu_each writes NaN for a window it cannot grade. A single NaN propagates through the sum,
    // making the whole floor NaN -- which then compares false against every velocity and silently
    // disables the guard that says which readings are real.
    DriftProbe p;
    p.reset(30, 10);
    REQUIRE(p.docs().size() == 3);
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const std::vector<float> a{ 5.0f, nan, 5.0f };
    REQUIRE(p.observe(a, 1.0) == Catch::Approx(0.0));   // baseline
    const std::vector<float> b{ 5.4f, nan, 5.2f };
    const double f = p.observe(b, 1.0);
    REQUIRE(std::isfinite(f));
    REQUIRE(f == Catch::Approx(0.3).epsilon(1e-6));     // mean of |0.4| and |0.2|, the NaN skipped
}

TEST_CASE("epoch plan: every token is covered exactly once per epoch", "[tutor]") {
    // The defining property, and the whole reason for replacing independent sampling. Under sampling
    // WITH replacement a token's per-epoch visit count is Poisson -- some tokens are trained several
    // times and ~1/e of documents not at all, which the surface reports as zero velocity and which is
    // indistinguishable from "nothing left to learn". A permutation makes the count exactly one.
    sub0::tutor::EpochPlan plan;
    // Three documents of 10, 25 and 7 tokens; the trailing phantom boundary mirrors corpus.tok.
    const std::vector<std::uint64_t> docs{ 0, 10, 35, 42 };
    plan.build(docs, /*train_tok=*/42, /*T=*/8, [](std::size_t) { return false; });
    REQUIRE_FALSE(plan.empty());

    std::vector<int> covered(42, 0);
    std::mt19937 rng(5);
    const std::size_t n = plan.size();
    for (std::size_t i = 0; i < n; ++i) {
        const auto& s = plan.next(rng);
        for (int t = 0; t < s.len; ++t) ++covered[static_cast<std::size_t>(s.start) + static_cast<std::size_t>(t)];
    }
    // Every position that can be an INPUT with a target inside its own document is covered EXACTLY
    // once -- never twice (which would silently double that token's weight in an epoch) and never more.
    // The documented exception: a document whose input count leaves a remainder of exactly one token
    // drops that token, since a 1-token window has no context. Doc 0 here has 10 tokens => 9 inputs =>
    // 8 + remainder 1, so position 8 is the dropped case and is asserted as such rather than waved past.
    for (std::size_t d = 0; d + 1 < docs.size(); ++d) {
        const std::size_t inputs = docs[d + 1] - docs[d] - 1;
        const std::size_t rem    = inputs % 8;
        for (std::size_t p = docs[d]; p + 1 < docs[d + 1]; ++p) {
            const bool dropped = (rem == 1) && (p == docs[d + 1] - 2);
            REQUIRE(covered[p] == (dropped ? 0 : 1));
        }
    }
    // And no window straddles a document boundary -- the property window.hpp exists to guarantee and
    // which a naive tiling of the whole corpus would silently break.
    for (std::size_t d = 0; d + 1 < docs.size(); ++d)
        REQUIRE(covered[docs[d + 1] - 1] == 0);
}

TEST_CASE("epoch plan: skipped documents contribute no windows at all", "[tutor]") {
    // Probe exclusion is built into the plan rather than done by rejection sampling. Exactness matters:
    // a probe trained even occasionally stops being a pure drift reading, and under the old rejection
    // loop a bounded retry could let one through.
    sub0::tutor::EpochPlan plan;
    const std::vector<std::uint64_t> docs{ 0, 10, 20, 30 };
    plan.build(docs, 30, 8, [](std::size_t d) { return d == 1; });
    std::mt19937 rng(6);
    for (std::size_t i = 0; i < plan.size(); ++i) {
        const auto& s = plan.next(rng);
        REQUIRE(s.doc != 1u);
        REQUIRE((s.start < 10 || s.start >= 20));
    }
}

TEST_CASE("epoch plan: the permutation reshuffles and repeats at the epoch boundary", "[tutor]") {
    sub0::tutor::EpochPlan plan;
    const std::vector<std::uint64_t> docs{ 0, 64 };
    plan.build(docs, 64, 8, [](std::size_t) { return false; });
    const std::size_t n = plan.size();
    REQUIRE(n > 4);
    std::mt19937 rng(7);
    std::vector<std::uint64_t> first, second;
    for (std::size_t i = 0; i < n; ++i) first.push_back(plan.next(rng).start);
    REQUIRE(plan.epoch() == 1);
    for (std::size_t i = 0; i < n; ++i) second.push_back(plan.next(rng).start);
    REQUIRE(plan.epoch() == 2);
    // Same SET both epochs (coverage is exact every epoch), different ORDER (it really reshuffles).
    auto a = first, b = second;
    std::sort(a.begin(), a.end()); std::sort(b.begin(), b.end());
    REQUIRE(a == b);
    REQUIRE(first != second);
}

TEST_CASE("surface: the sidecar round-trips EXACTLY", "[tutor]") {
    // TUTOR.md is explicit that the surface is feedback state and must be restored exactly, or a
    // matched-arm A/B quietly stops being matched. Nothing tested that until now -- the format was
    // written, read back by a live run, and assumed. Every field is compared, not a summary of them,
    // because a partial restore (say, everything but the velocity marks) would resume and look fine.
    Surface a;
    a.reset(64, 0.5f);
    std::mt19937 rng(31);
    std::uniform_real_distribution<float> u(0.5f, 8.0f);
    for (int pass = 0; pass < 6; ++pass)
        for (std::size_t d = 0; d < 64; ++d) {
            a.add_global_applied(0.75);
            a.record(d, u(rng), 0.01f, 100);
            if (d % 3 == 0) a.record_post(d, u(rng));
        }

    const std::string path = "tutor_roundtrip.bin";
    REQUIRE(a.save(path));
    Surface b;
    REQUIRE(b.load(path));
    REQUIRE(b.size() == a.size());
    REQUIRE(b.global_applied() == a.global_applied());
    for (std::size_t d = 0; d < a.size(); ++d) {
        const auto& x = a.at(d);
        const auto& y = b.at(d);
        REQUIRE(y.visits == x.visits);
        REQUIRE(y.tokens == x.tokens);
        REQUIRE(y.applied == x.applied);
        REQUIRE(y.applied_since == x.applied_since);
        REQUIRE(y.nelbo == x.nelbo);
        REQUIRE(y.nelbo_mark == x.nelbo_mark);
        REQUIRE(y.velocity == x.velocity);
        REQUIRE(y.nelbo_post == x.nelbo_post);
        REQUIRE(y.flags == x.flags);
        REQUIRE(y.transfer == x.transfer);
        REQUIRE(y.transfer_n == x.transfer_n);
        REQUIRE(y.global_mark == x.global_mark);
    }
    std::remove(path.c_str());
}

TEST_CASE("surface: a foreign or truncated file is REFUSED, not misread", "[tutor]") {
    // Refusing beats resuming against mismatched state, which is the whole reason the header carries a
    // magic, a version AND sizeof(Entry): the version cannot catch a layout change made by someone who
    // forgot to bump the version, which is exactly the person the check exists for.
    const std::string path = "tutor_foreign.bin";
    {
        std::ofstream os(path, std::ios::binary);
        const char junk[64] = { 'n', 'o', 't', 'a', 's', 'u', 'r', 'f' };
        os.write(junk, sizeof(junk));
    }
    Surface s;
    REQUIRE_FALSE(s.load(path));
    std::remove(path.c_str());

    // A truncated real file: header intact, payload short. Must also fail rather than leave half the
    // entries at their defaults, which would read as "these documents were never visited".
    Surface a;
    a.reset(32, 0.f);
    for (std::size_t d = 0; d < 32; ++d) a.record(d, 3.f, 0.01f, 50);
    const std::string tpath = "tutor_trunc.bin";
    REQUIRE(a.save(tpath));
    {
        const auto full = std::filesystem::file_size(tpath);
        std::filesystem::resize_file(tpath, full / 2);
    }
    // And the failed load must not have MUTATED the target: "false means start fresh" is the caller's
    // contract, so a surface left resized to the stale file's shape would be indexed wrongly forever.
    Surface c;
    c.reset(99, 0.f);
    REQUIRE_FALSE(c.load(tpath));
    REQUIRE(c.size() == 99);
    std::remove(tpath.c_str());
}

TEST_CASE("epoch plan: documents() agrees with for_each_document", "[tutor]") {
    // One implementation of "which documents does the plan reach", not two. The coverage denominator bug
    // came from a second, independently-maintained answer to that question.
    sub0::tutor::EpochPlan plan;
    const std::vector<std::uint64_t> docs{ 0, 40, 90, 140, 200 };
    plan.build(docs, 200, 16, [](std::size_t d) { return d == 2; });
    std::vector<std::uint32_t> seen;
    plan.for_each_document([&](std::uint32_t d) { seen.push_back(d); });
    REQUIRE(seen.size() == plan.documents());
    REQUIRE(std::is_sorted(seen.begin(), seen.end()));
    REQUIRE(std::find(seen.begin(), seen.end(), 2u) == seen.end());   // the skipped one
}

TEST_CASE("manifest: doc_count_matches expects the trailing phantom boundary", "[tutor]") {
    // The join between the splice's ordinals and corpus.tok's document index. scan_doc_boundaries pushes
    // one extra start for the corpus's final EOS, so a correctly joined corpus has total_docs + 1
    // starts. Equality would reject every correct corpus, and a >= test would accept a corpus that had
    // silently gained documents -- either way the labels would attach to the wrong documents.
    Manifest m;
    m.populations = { "a", "b" };
    m.doc_pop     = { 0, 0, 1, 1 };
    REQUIRE(m.doc_count_matches(5));
    REQUIRE_FALSE(m.doc_count_matches(4));
    REQUIRE_FALSE(m.doc_count_matches(6));
}

TEST_CASE("weighter: neutral until it has readings, then shaped by |velocity|", "[tutor][weight]") {
    using sub0::tutor::Weighter;
    Weighter w;
    sub0::tutor::Entry e{};

    // No scale yet -> exactly neutral. Early training must not be steered by a measurement that does
    // not exist; a controller that guesses before it can measure is the thing the read-only stage was
    // built to avoid ever shipping.
    REQUIRE_FALSE(w.active());
    REQUIRE(w.raw_weight(e) == 1.f);

    // Build a population of readings so the median scale is well defined.
    std::vector<sub0::tutor::Entry> pop(200);
    for (std::size_t i = 0; i < pop.size(); ++i) {
        pop[i].visits = 3;
        pop[i].velocity = 0.10f + 0.001f * static_cast<float>(i);   // median ~0.20
    }
    w.refresh(pop);
    REQUIRE(w.active());
    REQUIRE(w.scale() == Catch::Approx(0.20f).epsilon(0.05));

    sub0::tutor::Entry fast{}; fast.visits = 3; fast.velocity =  2.00f;   // still learning fast
    sub0::tutor::Entry flat{}; flat.visits = 3; flat.velocity =  0.001f;  // mastered OR unlearnable
    sub0::tutor::Entry back{}; back.visits = 3; back.velocity = -1.00f;   // being FORGOTTEN

    REQUIRE(w.raw_weight(fast) > w.raw_weight(flat));
    // Forgetting must weight UP, not down. Using the magnitude is what buys this: a signed rule would
    // drive a regressing entry toward zero weight, which is the opposite of rehearsal and exactly the
    // failure TUTOR.md's table calls out.
    REQUIRE(w.raw_weight(back) > w.raw_weight(flat));
    // Bounded at both ends. The floor is the safety property: a maximally down-weighted entry still
    // receives a quarter of normal learning, so it can always recover. Without it, an entry driven to
    // zero freezes its own NELBO, which confirms the low weight forever.
    REQUIRE(w.raw_weight(fast) <= Weighter::W_MAX);
    REQUIRE(w.raw_weight(flat) >= Weighter::W_MIN);

    // An entry with no reading stays neutral even once a scale exists.
    sub0::tutor::Entry unseen{}; unseen.visits = 1; unseen.velocity = 0.f;
    REQUIRE(w.raw_weight(unseen) == 1.f);
}

TEST_CASE("weighter: batch weights normalise to mean 1", "[tutor][weight]") {
    // The property that makes the A/B interpretable at all. Weighting redistributes gradient; if the
    // total also changed, the Tutor arm would be training at a different EFFECTIVE learning rate than
    // the control and no difference in final loss could be attributed to the redistribution -- which is
    // the hypothesis under test. TUTOR.md names this drift as a risk with "nothing logging it".
    std::vector<float> w{ 4.0f, 0.25f, 1.0f, 2.0f, 0.5f };
    sub0::tutor::Weighter::normalise(w);
    const double mean = std::accumulate(w.begin(), w.end(), 0.0) / static_cast<double>(w.size());
    REQUIRE(mean == Catch::Approx(1.0).epsilon(1e-6));
    // Relative ordering is preserved -- normalisation rescales, it does not reshape.
    REQUIRE(w[0] > w[3]);
    REQUIRE(w[3] > w[2]);
    REQUIRE(w[2] > w[4]);
    REQUIRE(w[4] > w[1]);

    // Degenerate input must not produce NaN weights that would poison a whole step's gradient.
    std::vector<float> z{ 0.f, 0.f, 0.f };
    sub0::tutor::Weighter::normalise(z);
    for (float x : z) REQUIRE(x == 1.f);
}
