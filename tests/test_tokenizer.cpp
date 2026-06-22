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

// ── Char-level tokenizer (TRAINING_DESIGN §13.6) ──────────────────────────────

TEST_CASE("char_level - round-trips text preserving whitespace and newlines", "[tokenizer][char]") {
    const std::string text = "To be,\n  or not to be.\n";
    auto tok = BPETokenizer::char_level({text});

    REQUIRE(tok.num_merges() == 0);                      // zero merges — pure char-level
    const auto ids = tok.encode(text);
    // One token per Unicode code point (here all single-byte): exact length, exact round-trip.
    REQUIRE(ids.size() == text.size());
    REQUIRE(tok.decode(ids) == text);                    // spaces, indent and '\n' survive
}

TEST_CASE("char_level - space and newline are first-class tokens", "[tokenizer][char]") {
    auto tok = BPETokenizer::char_level({"a b\nc"});
    REQUIRE(tok.token_id(" ")  >= 0);
    REQUIRE(tok.token_id("\n") >= 0);
}

TEST_CASE("char_level - survives save/load via empty-merges detection", "[tokenizer][char]") {
    const std::string text = "hark!\nwho goes there?\n";
    auto tok = BPETokenizer::char_level({text});

    const auto dir = std::filesystem::temp_directory_path() / "sub0llm_chartok_test";
    std::filesystem::remove_all(dir);
    tok.save(dir);
    auto loaded = BPETokenizer::load(dir / "vocab.json", dir / "merges.txt");

    // load() must re-detect char-level (no merges + literal whitespace) so encode does
    // NOT fall back to the whitespace-collapsing pre-tokenizer.
    REQUIRE(loaded.decode(loaded.encode(text)) == text);
    std::filesystem::remove_all(dir);
}

// ── Word-level tokenizer (granularity spectrum, TRAINING_DESIGN §13.6) ─────────

TEST_CASE("word_level - round-trips, one token per word + punctuation + whitespace", "[tokenizer][word]") {
    const std::string text = "To be, or not\n  to be.\n";
    auto tok = BPETokenizer::word_level({text});

    REQUIRE(tok.num_merges() == 0);
    const auto ids = tok.encode(text);
    REQUIRE(tok.decode(ids) == text);                 // exact round-trip, structure preserved
    // "be" appears twice but is ONE vocab entry; "be," is split into "be" + ",".
    REQUIRE(tok.token_id("be") >= 0);
    REQUIRE(tok.token_id("not") >= 0);
    REQUIRE(tok.token_id(",")  >= 0);
    REQUIRE(tok.token_id("\n") >= 0);
}

TEST_CASE("word_level - contractions stay whole, repeats collapse to one token", "[tokenizer][word]") {
    auto tok = BPETokenizer::word_level({"I'll go and go and go"});
    REQUIRE(tok.token_id("I'll") >= 0);               // apostrophe kept inside the word
    const auto ids = tok.encode("go go");
    REQUIRE(ids.size() == 3);                         // go, <space>, go — "go" is one shared id
    REQUIRE(ids[0] == ids[2]);
}

TEST_CASE("word_level - LEADING quotes don't weld to the word (apostrophe is word-internal only)",
          "[tokenizer][word]") {
    // TinyStories dialogue ('What's, ''Oh) used to spawn rare 'Word / ''Word tokens because a leading
    // apostrophe started a word run. The leading quote must now be its own token; the contraction stays.
    auto tok = BPETokenizer::word_level({"''What's that?' said the dog. What's up"});
    REQUIRE(tok.decode(tok.encode("''What's")) == "''What's");   // still lossless
    REQUIRE(tok.token_id("'")      >= 0);                         // the quote is its own token
    REQUIRE(tok.token_id("What's") >= 0);                        // the word, unwelded
    REQUIRE(tok.token_id("''What's") < 0);                       // the glued variant no longer exists
    REQUIRE(tok.token_id("'What's") < 0);
    // "What's" before a quote and after one are the SAME id now (statistics no longer fragmented).
    const auto a = tok.encode("''What's"), b = tok.encode(" What's");
    REQUIRE(a.back() == b.back());
}

TEST_CASE("word_level - accented Latin letters stay inside the word (no accent split)",
          "[tokenizer][word]") {
    // ñ/é/ï are multi-byte UTF-8; the old ASCII-only letter test split them out, fragmenting
    // "piñata"->"pi"+"ñ"+"ata". Accented Latin letters must be word-internal.
    auto tok = BPETokenizer::word_level({"the piñata at the café was naïve"});
    REQUIRE(tok.decode(tok.encode("piñata")) == "piñata");   // lossless
    REQUIRE(tok.token_id("piñata") >= 0);                    // ONE token, not split
    REQUIRE(tok.token_id("café")   >= 0);
    REQUIRE(tok.token_id("naïve")  >= 0);
    REQUIRE(tok.token_id("ñ") < 0);                          // the accent is NOT its own token
}

TEST_CASE("word_level - survives save/load and is distinguished from char-level", "[tokenizer][word]") {
    const std::string text = "hark, who goes there?\n";
    auto tok = BPETokenizer::word_level({text});

    const auto dir = std::filesystem::temp_directory_path() / "sub0llm_wordtok_test";
    std::filesystem::remove_all(dir);
    tok.save(dir);
    auto loaded = BPETokenizer::load(dir / "vocab.json", dir / "merges.txt");
    // load() must re-detect word-level (multi-char word token present) — NOT char-level —
    // so encode splits by word, giving the same ids and an exact round-trip.
    REQUIRE(loaded.encode(text) == tok.encode(text));
    REQUIRE(loaded.decode(loaded.encode(text)) == text);
    std::filesystem::remove_all(dir);
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
