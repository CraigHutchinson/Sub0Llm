// tokenizer_tests.cpp — BPE tokenizer round-trip contract tests.
//
// The tokenizer is learned at build time by sub0-configure and serialized to
// tokenizer.bin. These tests load that artifact and pin the encode/detokenize
// contract: detokenize(encode(x)) reproduces x for in-alphabet text, including
// truecasing of capitalized words and all-caps words.

#include <catch2/catch_test_macros.hpp>

#include "sub0/core.hpp"

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

TEST_CASE("encode is deterministic", "[tokenizer]") {
    REQUIRE(tokenizer_ready());
    const std::string text = "the dog ran and the cat slept";
    REQUIRE(sub0::encode(text) == sub0::encode(text));
}
