// test_sentencepiece.cpp (Ch27) — SPM tokenizer algorithm, with a tiny hand-built
// vocab (the live parity vs llama-tokenize on the Gemma GGUF is the end-to-end gate).

#include "sub0llm/tokenizer/sentencepiece.hpp"

#include <catch2/catch_test_macros.hpp>

#include <format>
#include <string>
#include <vector>

using sub0llm::SPTokenizer;

namespace {
// Vocab: specials + a few pieces + all 256 byte-fallback tokens. ▁ = U+2581.
SPTokenizer make_tok() {
    std::vector<std::string> toks = {"<unk>", "<bos>", "<eos>", "a", "b", "ab",
                                     "\xe2\x96\x81", "\xe2\x96\x81""a"};
    std::vector<float> scores = {0, 0, 0, -3.0f, -3.0f, -1.0f, -2.0f, -0.5f};
    for (int i = 0; i < 256; ++i) {                 // byte tokens <0x00>..<0xFF>
        toks.push_back(std::format("<0x{:02X}>", i));
        scores.push_back(-10.0f);                   // worse than any real merge
    }
    return SPTokenizer::from_vocab(toks, scores, /*bos*/1, /*eos*/2, /*add_bos*/true);
}
} // namespace

TEST_CASE("SPM prepends BOS and merges by score", "[spm]") {
    auto t = make_tok();
    // "ab": a,b → merge "ab"(id 5, score -1 beats keeping a,b). BOS(1) prepended.
    REQUIRE(t.encode("ab") == std::vector<int32_t>{1, 5});
}

TEST_CASE("SPM maps space to the word-marker piece", "[spm]") {
    auto t = make_tok();
    // " a" → ▁a (id 7). BOS prepended.
    REQUIRE(t.encode(" a") == std::vector<int32_t>{1, 7});
}

TEST_CASE("SPM byte-fallback for unknown characters", "[spm]") {
    auto t = make_tok();
    // 'z' (0x7A) isn't a piece → byte token <0x7A> = id 8 + 0x7A = 122.
    const auto ids = t.encode("z");
    REQUIRE(ids.size() == 2);          // BOS + one byte token
    REQUIRE(ids[0] == 1);
    REQUIRE(ids[1] == 8 + 0x7A);
}

TEST_CASE("SPM decode round-trips word-marker to space and byte tokens to bytes", "[spm]") {
    auto t = make_tok();
    REQUIRE(t.decode({1, 6}) == " ");           // BOS skipped, ▁ (id 6) → space
    REQUIRE(t.decode({7}) == " a");              // ▁a → " a"
    REQUIRE(t.decode({5}) == "ab");
    REQUIRE(t.decode({8 + 0x7A}) == "z");        // byte token → raw byte
}
