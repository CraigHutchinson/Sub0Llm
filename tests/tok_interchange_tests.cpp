// tok_interchange_tests.cpp — WS7: sub0::tok::to_interchange, the struct-producing half of the
// export pathway (JSON/TSV formatting lives in tools/sub0llm-tokenizer.cpp, not here -- see
// include/sub0/tok_interchange.hpp's own doc comment).

#include <catch2/catch_test_macros.hpp>

#include "sub0/tok_interchange.hpp"

#include <unordered_set>

using sub0::tok::Interchange;
using sub0::tok::Tokenizer;
using sub0::tok::TokenKind;

namespace {
const std::string kCorpus = [] {
    std::string c;
    for (int i = 0; i < 30; ++i)
        c += "the quick brown fox jumps over the lazy dog .\n"
             "she said , \" hello world \" and they left .\n"
             "save_scan_state and non-commercial and NASA's team .\n";
    return c;
}();
}  // namespace

TEST_CASE("to_interchange: dense, id-ordered, covers the whole vocab", "[tok][interchange]") {
    const Tokenizer t = sub0::tok::learn(kCorpus);
    const Interchange ic = sub0::tok::to_interchange(t);
    REQUIRE(static_cast<int>(ic.tokens.size()) == t.vocab);
    for (int id = 0; id < t.vocab; ++id)
        REQUIRE(ic.tokens[static_cast<std::size_t>(id)].id == id);   // dense, id-ordered -- ic.tokens[id].id == id
}

TEST_CASE("to_interchange: byte/marker/piece kinds split at the right boundaries", "[tok][interchange]") {
    const Tokenizer t = sub0::tok::learn(kCorpus);
    const Interchange ic = sub0::tok::to_interchange(t);
    for (int id = 0; id < 256; ++id)
        REQUIRE(ic.tokens[static_cast<std::size_t>(id)].kind == TokenKind::Byte);
    for (int id = 256; id < t.n_base; ++id)
        REQUIRE(ic.tokens[static_cast<std::size_t>(id)].kind == TokenKind::Marker);
    for (int id = t.n_base; id < t.vocab; ++id)
        REQUIRE(ic.tokens[static_cast<std::size_t>(id)].kind == TokenKind::Piece);
}

TEST_CASE("to_interchange: byte text is the single raw byte, id == byte value", "[tok][interchange]") {
    const Tokenizer t = sub0::tok::learn(kCorpus);
    const Interchange ic = sub0::tok::to_interchange(t);
    REQUIRE(ic.tokens[static_cast<std::size_t>('A')].text == "A");
    REQUIRE(ic.tokens[0].text.size() == 1);
    REQUIRE(static_cast<unsigned char>(ic.tokens[0].text[0]) == 0);   // NUL byte, id 0
}

TEST_CASE("to_interchange: known markers get their real literal text", "[tok][interchange]") {
    const Tokenizer t = sub0::tok::learn(kCorpus);
    const Interchange ic = sub0::tok::to_interchange(t);
    REQUIRE(ic.tokens[static_cast<std::size_t>(sub0::casing::TOK_EOS)].text == "<|endoftext|>");
    REQUIRE(ic.tokens[static_cast<std::size_t>(sub0::casing::TOK_TURN_START)].text == "<|im_start|>");
    REQUIRE(ic.tokens[static_cast<std::size_t>(sub0::casing::TOK_TURN_END)].text == "<|im_end|>");
    REQUIRE(ic.tokens[static_cast<std::size_t>(sub0::casing::TOK_GLUE_OPAREN)].text == "(");
    REQUIRE(ic.tokens[static_cast<std::size_t>(sub0::casing::TOK_GLUE_CPAREN)].text == ")");
    REQUIRE(ic.tokens[static_cast<std::size_t>(sub0::casing::TOK_NEWLINE)].text == "\n");
    // A reserved (not-yet-assigned) marker still gets a total, non-empty placeholder -- never the
    // wrapped-byte garbage detokenize_join's own reserved-id guard exists to avoid (see casing.hpp).
    REQUIRE(ic.tokens[static_cast<std::size_t>(sub0::casing::TOK_RESERVED_0)].text == "<|reserved_0|>");
}

TEST_CASE("to_interchange: piece text matches the tokenizer's own expansion bytes", "[tok][interchange]") {
    const Tokenizer t = sub0::tok::learn(kCorpus);
    const Interchange ic = sub0::tok::to_interchange(t);
    REQUIRE(t.vocab > t.n_base);   // sanity: this corpus actually learned some pieces
    for (int id = t.n_base; id < t.vocab; ++id) {
        std::string expect;
        for (int code : t.expansion[static_cast<std::size_t>(id)]) expect.push_back(static_cast<char>(code & 0xFF));
        REQUIRE(ic.tokens[static_cast<std::size_t>(id)].text == expect);
    }
}

TEST_CASE("to_interchange: special ids -- only eos_id is set, matching this scheme's design", "[tok][interchange]") {
    const Tokenizer t = sub0::tok::learn(kCorpus);
    const Interchange ic = sub0::tok::to_interchange(t);
    REQUIRE(ic.eos_id == sub0::casing::TOK_EOS);
    REQUIRE(ic.bos_id == -1);   // no BOS concept in this scheme
    REQUIRE(ic.pad_id == -1);   // no padding token (fixed-length training windows)
    REQUIRE(ic.unk_id == -1);   // byte-level base alphabet is never "unknown"
}

// marker_literal() (tok_interchange.cpp, internal) switches over the enum TYPE specifically so
// -Wswitch catches a future marker added without a table entry -- this test is the runtime
// counterpart: every marker id in [TOK_EOS, TOK_MARKER_COUNT) must produce a non-empty literal.
// NOT asserting global uniqueness: TOK_ODQUOTE/TOK_CDQUOTE deliberately share `"` (that's the whole
// reason the scheme needs two distinct ids for one printable character -- direction lives in the
// id, not the text), so a text collision between exactly those two is expected, not a bug.
TEST_CASE("to_interchange: every marker has a non-empty literal", "[tok][interchange]") {
    const Tokenizer t = sub0::tok::learn(kCorpus);
    const Interchange ic = sub0::tok::to_interchange(t);
    std::unordered_set<std::string> seen;
    for (int id = sub0::casing::TOK_EOS; id < sub0::casing::TOK_MARKER_COUNT; ++id) {
        const std::string& txt = ic.tokens[static_cast<std::size_t>(id)].text;
        REQUIRE_FALSE(txt.empty());
        REQUIRE(txt != "<|unknown_marker|>");   // the exhaustive switch's unreachable fallback
        const bool is_quote = id == sub0::casing::TOK_ODQUOTE || id == sub0::casing::TOK_CDQUOTE;
        if (!is_quote) REQUIRE(seen.insert(txt).second);   // every OTHER marker's literal is unique
    }
}
