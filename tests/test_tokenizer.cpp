#include <catch2/catch_test_macros.hpp>

#include "sub0llm/tokenizer/bpe.hpp"

using namespace sub0llm;

// ── Vocabulary and token registration ────────────────────────────────────────

TEST_CASE("add_special_token - deduplication", "[tokenizer]") {
    auto tok = BPETokenizer::train({"hello"}, 20);
    const auto id1 = tok.add_special_token("<pad>");
    const auto id2 = tok.add_special_token("<pad>");
    REQUIRE(id1 == id2);
    REQUIRE(tok.token_id("<pad>") == id1);
}

TEST_CASE("token_str out of range throws", "[tokenizer]") {
    auto tok = BPETokenizer::train({"a"}, 5);
    REQUIRE_THROWS_AS(tok.token_str(9999), std::out_of_range);
}

TEST_CASE("token_id for unknown returns -1", "[tokenizer]") {
    auto tok = BPETokenizer::train({"hello"}, 10);
    REQUIRE(tok.token_id("XXXXXXXX") == -1);
}

// ── Training ─────────────────────────────────────────────────────────────────

TEST_CASE("train - vocab grows with merges", "[tokenizer][train]") {
    // Use a repetitive corpus so many merges are possible.
    auto tok = BPETokenizer::train(
        {"the cat sat on the mat the cat ate the rat"}, 50);
    // Starting vocab <= unique chars; after training it grows with merges.
    REQUIRE(tok.vocab_size() > 5);
    REQUIRE(tok.num_merges() > 0);
}

TEST_CASE("train - all input bytes are in vocabulary", "[tokenizer][train]") {
    auto tok = BPETokenizer::train({"abc def"}, 20);
    // Every unique character in the corpus must be in the vocabulary
    for (char c : std::string{"abcdef"}) {
        REQUIRE(tok.token_id(std::string(1, c)) != -1);
    }
}

TEST_CASE("train - merge rules are non-empty for non-trivial corpus", "[tokenizer][train]") {
    auto tok = BPETokenizer::train(
        {"aaabdaaabac", "aaabdaaabac", "aaabdaaabac"}, 15);
    REQUIRE(tok.num_merges() > 0);
}

// ── Encode ────────────────────────────────────────────────────────────────────

TEST_CASE("encode - non-empty output", "[tokenizer][encode]") {
    auto tok = BPETokenizer::train({"hello world"}, 20);
    const auto ids = tok.encode("hello");
    REQUIRE_FALSE(ids.empty());
}

TEST_CASE("encode - all ids are in vocabulary", "[tokenizer][encode]") {
    auto tok = BPETokenizer::train({"hello world"}, 30);
    const auto ids = tok.encode("hello world");
    for (auto id : ids) {
        REQUIRE(id >= 0);
        REQUIRE(static_cast<std::size_t>(id) < tok.vocab_size());
    }
}

TEST_CASE("encode - empty string returns empty", "[tokenizer][encode]") {
    auto tok = BPETokenizer::train({"hello"}, 15);
    REQUIRE(tok.encode("").empty());
}

// ── Decode ────────────────────────────────────────────────────────────────────

TEST_CASE("decode - roundtrip for in-vocabulary text", "[tokenizer][roundtrip]") {
    const std::string corpus_word = "hello";
    auto tok = BPETokenizer::train({corpus_word}, 20);

    // Train on single word; the full word may be merged into one token.
    const auto ids = tok.encode(corpus_word);
    const std::string decoded = tok.decode(ids);
    REQUIRE(decoded == corpus_word);
}

TEST_CASE("decode - space marker is restored", "[tokenizer][roundtrip]") {
    auto tok = BPETokenizer::train({"hello world foo bar"}, 40);
    const std::string text = "hello world";
    const auto ids = tok.encode(text);
    const std::string decoded = tok.decode(ids);
    REQUIRE(decoded == text);
}

TEST_CASE("decode - out-of-range id throws", "[tokenizer][decode]") {
    auto tok = BPETokenizer::train({"a"}, 5);
    const std::vector<BPETokenizer::TokenId> bad{9999};
    REQUIRE_THROWS_AS(tok.decode(bad), std::runtime_error);
}

// ── Save / load ───────────────────────────────────────────────────────────────

TEST_CASE("save and reload - vocab identical", "[tokenizer][io]") {
    auto tok = BPETokenizer::train({"hello world goodbye"}, 30);
    const std::filesystem::path dir = "/tmp/sub0llm_tok_test";
    tok.save(dir);

    auto tok2 = BPETokenizer::load(dir / "vocab.json", dir / "merges.txt");

    REQUIRE(tok2.vocab_size() == tok.vocab_size());
    REQUIRE(tok2.num_merges() == tok.num_merges());

    // Every token in original must appear with the same id in reloaded.
    for (std::size_t i = 0; i < tok.vocab_size(); ++i) {
        const auto id = static_cast<BPETokenizer::TokenId>(i);
        REQUIRE(tok2.token_id(tok.token_str(id)) == id);
    }
}

TEST_CASE("save and reload - encode roundtrip preserved", "[tokenizer][io]") {
    auto tok = BPETokenizer::train({"hello world"}, 25);
    const std::filesystem::path dir = "/tmp/sub0llm_tok_test2";
    tok.save(dir);
    auto tok2 = BPETokenizer::load(dir / "vocab.json", dir / "merges.txt");

    const std::string text = "hello";
    const auto ids1 = tok.encode(text);
    const auto ids2 = tok2.encode(text);
    REQUIRE(ids1 == ids2);
}

// ── EOS token ────────────────────────────────────────────────────────────────

TEST_CASE("trained tokenizer has EOS token", "[tokenizer][special]") {
    auto tok = BPETokenizer::train({"hello"}, 15);
    REQUIRE(tok.eos_id() >= 0);
    REQUIRE(tok.token_str(tok.eos_id()) == "<|endoftext|>");
}
