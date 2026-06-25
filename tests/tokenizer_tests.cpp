// tokenizer_tests.cpp — BPE tokenizer round-trip contract tests.
//
// The tokenizer is learned at build time by sub0-configure and serialized to
// tokenizer.bin. These tests load that artifact and pin the encode/detokenize
// contract: detokenize(encode(x)) reproduces x for in-alphabet text, including
// truecasing of capitalized words and all-caps words.

#include <catch2/catch_test_macros.hpp>

#include "sub0/core.hpp"
#include "sub0/casing.hpp"

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
}

TEST_CASE("typographic input round-trips through the build tokenizer", "[tokenizer]") {
    REQUIRE(tokenizer_ready());
    long r = 0;
    const std::string in = "\xE2\x80\x9C" "I don\xE2\x80\x99t\xE2\x80\x9D";  // “I don’t”
    REQUIRE(sub0::detokenize(sub0::encode(in)) == sub0::casing::normalize_text(in, r));
}

TEST_CASE("a name that shadows a noun is withheld from case collapse", "[tokenizer]") {
    REQUIRE(tokenizer_ready());
    // "Spot" (the dog) appears capitalized mid-sentence far more than the noun
    // "spot" appears lowercase, so the configurator withholds it from `attested`:
    // "Spot" stays a distinct verbatim token rather than folding onto <|cap|>spot.
    // This is the inverse of the "They"/"they" case-sharing contract above.
    const std::vector<int> name = sub0::encode("Spot");
    const std::vector<int> noun = sub0::encode("spot");
    REQUIRE_FALSE(name.empty());
    REQUIRE(name != noun);
    const bool encoded_as_cap_plus_noun =
        name.size() == noun.size() + 1 &&
        std::vector<int>(name.begin() + 1, name.end()) == noun;
    REQUIRE_FALSE(encoded_as_cap_plus_noun);
    REQUIRE(sub0::detokenize(name) == "Spot");  // capital preserved on the round-trip
}

TEST_CASE("encode is deterministic", "[tokenizer]") {
    REQUIRE(tokenizer_ready());
    const std::string text = "the dog ran and the cat slept";
    REQUIRE(sub0::encode(text) == sub0::encode(text));
}
