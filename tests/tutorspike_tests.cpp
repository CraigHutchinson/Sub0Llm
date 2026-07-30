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
