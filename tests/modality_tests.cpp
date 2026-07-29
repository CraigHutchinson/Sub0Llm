// modality_tests.cpp — engine-free tests for sub0::modality: the per-codepoint spacing-modality
// calibration (scan, merge, serialize, contradiction flagging). Links sub0_frontend only.
#include <catch2/catch_test_macros.hpp>

#include "sub0/modality.hpp"

#include <sstream>
#include <string>

using namespace sub0::modality;

TEST_CASE("modality: neighbour classification (SS/SG/GS/GG)", "[modality]") {
    ModalityStats st;
    add_modality(st, "a, b.");            // ',' and '.' are both glue-before, space-after (closers)
    REQUIRE(st.chars[','].dominant() == GS);
    REQUIRE(st.chars['.'].dominant() == GS);
    add_modality(st, "(a)");              // '(' opener (space-before, glue-after); ')' closer
    REQUIRE(st.chars['('].dominant() == SG);
    REQUIRE(st.chars[')'].dominant() == GS);
    add_modality(st, "x=y");              // '=' interior (glue both sides)
    REQUIRE(st.chars['='].dominant() == GG);
}

TEST_CASE("modality: UTF-8 symbols tallied, letters excluded", "[modality]") {
    ModalityStats st;
    // "£5 and 10€ here"  (£ = C2 A3, € = E2 82 AC) plus "café" whose é (C3 A9) is a letter.
    add_modality(st, "\xC2\xA3" "5 and 10\xE2\x82\xAC here and caf\xC3\xA9 too");
    REQUIRE(st.chars.count(0x00A3) == 1);          // £ scanned
    REQUIRE(st.chars.count(0x20AC) == 1);          // € scanned
    REQUIRE(st.chars[0x00A3].dominant() == SG);    // £5  -> currency prefix (opener)
    REQUIRE(st.chars[0x20AC].dominant() == GS);    // 10€ -> currency suffix (closer)
    REQUIRE(st.chars.count(0x00E9) == 0);          // é is a letter, NOT tallied
    REQUIRE(st.chars.count('a') == 0);             // ASCII letters never tallied
}

TEST_CASE("modality: merge is additive and order-free", "[modality]") {
    ModalityStats a, b;
    add_modality(a, "x, y; z.");
    add_modality(b, "p) q( r=s");
    ModalityStats ab = a; merge(ab, b);
    ModalityStats ba = b; merge(ba, a);
    REQUIRE(ab.chars.size() == ba.chars.size());
    REQUIRE(ab.scanned_bytes == ba.scanned_bytes);
    for (const auto& [cp, cm] : ab.chars) REQUIRE(cm.n == ba.chars.at(cp).n);   // same counts either order
}

TEST_CASE("modality: serialize -> deserialize round-trips the ledger", "[modality]") {
    ModalityStats a;
    add_modality(a, "some, text; with. mixed (punct) and 5\xC2\xA3 symbols!");
    std::stringstream ss;
    serialize(a, ss);
    ModalityStats b;
    REQUIRE(deserialize(b, ss));
    REQUIRE(b.scanned_bytes == a.scanned_bytes);
    REQUIRE(b.chars.size() == a.chars.size());
    for (const auto& [cp, cm] : a.chars) REQUIRE(b.chars.at(cp).n == cm.n);
}

TEST_CASE("modality: a corpus that flips a dominant modality is flagged", "[modality]") {
    ModalityStats prior, fresh;
    for (int i = 0; i < 1000; ++i) add_modality(prior, "x, ");   // ',' -> GS (glue-before, space-after)
    for (int i = 0; i < 1000; ++i) add_modality(fresh, "x ,y");  // ',' -> SG (space-before, glue-after)
    REQUIRE(prior.chars[','].dominant() == GS);
    REQUIRE(fresh.chars[','].dominant() == SG);
    const auto bad = find_contradictions(prior, fresh);
    bool comma_flagged = false;
    for (const auto& c : bad) if (c.cp == ',') { comma_flagged = true; REQUIRE(c.prior_dom == GS); REQUIRE(c.fresh_dom == SG); }
    REQUIRE(comma_flagged);
    // An agreeing corpus flags nothing.
    ModalityStats same; for (int i = 0; i < 1000; ++i) add_modality(same, "x, ");
    REQUIRE(find_contradictions(prior, same).empty());
}
