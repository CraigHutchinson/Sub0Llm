// tokenizer_tests.cpp — BPE tokenizer round-trip contract tests.
//
// The tokenizer is learned at build time by sub0-configure and serialized to
// tokenizer.tok. These tests load that artifact and pin the encode/detokenize
// contract: detokenize(encode(x)) reproduces x for in-alphabet text, including
// truecasing of capitalized words and all-caps words.

#include <catch2/catch_test_macros.hpp>

#include "sub0/core.hpp"
#include "sub0/casing.hpp"
#include "sub0/tokenizer.hpp"

#include <string>
#include <vector>

namespace {

// Load the generated tokenizer once for the whole suite.
bool tokenizer_ready() {
    static const bool ok = sub0::load_tokenizer(sub0::default_tokenizer());
    return ok;
}

}  // namespace

TEST_CASE("tokenizer loads from the generated artifact", "[tokenizer]") {
    REQUIRE(tokenizer_ready());
}

TEST_CASE("encode emits in-range ids and detokenizes back to lowercase text", "[tokenizer]") {
    REQUIRE(tokenizer_ready());
    const std::string text = "the little cat sat on the mat.";

    const std::vector<int> ids = sub0::encode(text);
    REQUIRE_FALSE(ids.empty());
    for (int id : ids) {
        REQUIRE(id >= 0);
        REQUIRE(id < VOCAB);
    }
    REQUIRE(sub0::detokenize(ids) == text);
}

TEST_CASE("truecasing preserves capitalization through the round-trip", "[tokenizer]") {
    REQUIRE(tokenizer_ready());
    // A capitalized name, a sentence-initial capital and an all-caps word all
    // collapse to a lowercase base form plus a marker, then re-expand exactly.
    const std::string text = "Lily went to the park. SHE was very happy!";
    REQUIRE(sub0::detokenize(sub0::encode(text)) == text);
}

TEST_CASE("case markers stay atomic so the word token is case-shared", "[tokenizer]") {
    REQUIRE(tokenizer_ready());
    // The point of truecasing: "They" must encode as a standalone <|cap|> marker
    // followed by the *same* token sequence as lowercase "they". If BPE were allowed
    // to merge the marker into the word, "They" would get its own token and the tail
    // would differ -- the per-case vocab split this design exists to prevent.
    const std::vector<int> low = sub0::encode("they");
    const std::vector<int> cap = sub0::encode("They");
    REQUIRE_FALSE(low.empty());
    REQUIRE(cap.size() == low.size() + 1);
    REQUIRE(std::vector<int>(cap.begin() + 1, cap.end()) == low);
}

TEST_CASE("normalize_text folds typographic glyphs to ASCII", "[casing]") {
    long r = 0;
    // “ I don’t ” – wait…  (curly double/single quotes, en dash, ellipsis)
    const std::string in = "\xE2\x80\x9C" "I don\xE2\x80\x99t\xE2\x80\x9D \xE2\x80\x93 wait\xE2\x80\xA6";
    REQUIRE(sub0::casing::normalize_text(in, r) == "\"I don't\" - wait...");
    REQUIRE(r == 5);
}

TEST_CASE("word_unit_end keeps contractions and accented words whole", "[casing]") {
    using namespace sub0::casing;
    // Interior apostrophe -> one unit covering all of "don't".
    const std::vector<int> dont = truecase_tokenize("don't", {}, nullptr);
    REQUIRE(word_unit_end(dont, 0) == dont.size());
    // ñ (C3 B1) is word material -> "piñata" is one unit, not split at the accent.
    const std::vector<int> pin = truecase_tokenize("pi\xC3\xB1" "ata", {}, nullptr);
    REQUIRE(word_unit_end(pin, 0) == pin.size());
    // A trailing apostrophe is not interior -> it splits off ("dogs" + "'").
    const std::vector<int> dogs = truecase_tokenize("dogs'", {}, nullptr);
    REQUIRE(word_unit_end(dogs, 0) == 4);
    // v2: a digit run is its own unit -> a direct digit<->letter transition splits ("foo123" -> "foo"|"123").
    const std::vector<int> foo = truecase_tokenize("foo123", {}, nullptr);
    REQUIRE(word_unit_end(foo, 0) == 3);                 // "foo" ends before the digit run
    REQUIRE(word_unit_end(foo, 3) == foo.size());        // "123" is one whole digit unit
    // A connector still binds ACROSS the class boundary -> "covid-19" stays one unit.
    const std::vector<int> cov = truecase_tokenize("covid-19", {}, nullptr);
    REQUIRE(word_unit_end(cov, 0) == cov.size());
}

TEST_CASE("typographic input round-trips through the build tokenizer", "[tokenizer]") {
    REQUIRE(tokenizer_ready());
    long r = 0;
    const std::string in = "\xE2\x80\x9C" "I don\xE2\x80\x99t\xE2\x80\x9D";  // “I don’t”
    REQUIRE(sub0::detokenize(sub0::encode(in)) == sub0::casing::normalize_text(in, r));
}

TEST_CASE("a name that shadows a noun is withheld from case collapse", "[tokenizer]") {
    // The inverse of the "They"/"they" sharing contract above. When a capitalized
    // word dominates its lowercase form in the corpus, derive_attested withholds that
    // form from case collapse, so the name stays a distinct *verbatim* token (e.g. the
    // dog "Spot") rather than folding onto <|cap|>spot. Whether a word qualifies is a
    // property of the corpus, so this builds a tiny deterministic in-memory corpus and
    // learns a tokenizer on it directly -- no dependency on the baked production vocab.
    //
    // "Spot" appears only capitalized mid-sentence (always preceded by a lowercase
    // word, the name-use signal) with a single lowercase "spot"; "they" appears only
    // lowercase but for one sentence-initial "They" (which is not a name use).
    std::string corpus;
    for (int i = 0; i < 200; ++i)
        corpus += "the dog Spot ran and Spot played with Spot all day .\n";
    corpus += "i found a spot on the wall .\n";   // lone lowercase use of the shadowed noun
    for (int i = 0; i < 200; ++i)
        corpus += "they went home and they slept and they were happy .\n";
    corpus += "They went home today .\n";          // lone sentence-initial cap (not a name use)

    const sub0::tok::Tokenizer t = sub0::tok::learn(corpus);

    // The mechanism under test: "spot" (name-dominated) is withheld; "they" is attested.
    REQUIRE(t.attested.count("spot") == 0);
    REQUIRE(t.attested.count("they") == 1);

    // The withheld name stays a verbatim capitalized token: NOT <|cap|> + the noun.
    const std::vector<int> name = sub0::tok::encode(t, "Spot");
    const std::vector<int> noun = sub0::tok::encode(t, "spot");
    REQUIRE_FALSE(name.empty());
    REQUIRE(name != noun);
    const bool encoded_as_cap_plus_noun =
        name.size() == noun.size() + 1 &&
        name.front() == sub0::casing::TOK_CAP &&
        std::vector<int>(name.begin() + 1, name.end()) == noun;
    REQUIRE_FALSE(encoded_as_cap_plus_noun);
    REQUIRE(sub0::tok::detokenize(t, name) == "Spot");  // capital preserved on the round-trip

    // Control: the attested word DOES case-share -- "They" = <|cap|> + "they".
    const std::vector<int> they_low = sub0::tok::encode(t, "they");
    const std::vector<int> they_cap = sub0::tok::encode(t, "They");
    REQUIRE(they_cap.size() == they_low.size() + 1);
    REQUIRE(they_cap.front() == sub0::casing::TOK_CAP);
    REQUIRE(std::vector<int>(they_cap.begin() + 1, they_cap.end()) == they_low);
}

TEST_CASE("encode is deterministic", "[tokenizer]") {
    REQUIRE(tokenizer_ready());
    const std::string text = "the dog ran and the cat slept";
    REQUIRE(sub0::encode(text) == sub0::encode(text));
}
