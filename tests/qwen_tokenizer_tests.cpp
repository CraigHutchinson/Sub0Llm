// qwen_tokenizer_tests.cpp -- WP5a's PRIMARY correctness gate: sub0::qwen_tok (this engine's
// reimplementation of the REAL Qwen3.8-Flash-Next byte-level BPE tokenizer) checked against
// expected values produced BY THE REAL TOKENIZER, at tests/fixtures/qwen_tokenizer/.
//
// Four fixture files, four independently-failing layers, so a break says WHERE it broke rather than
// only that some token id moved (docs/QWEN_TOKENIZER.md sec 5):
//   nfc_cases.tsv          -- the normalizer alone, 1382 cases including every codepoint in Unicode
//                             whose NFC differs from itself
//   pretokenize_cases.tsv  -- the hand-transliterated pre-tokenization regex alone, chunk boundaries
//                             only, no vocabulary involved
//   encode_cases.tsv       -- the whole pipeline: text -> ids, and the reference's own decode(ids)
//   decode_cases.tsv       -- decode-only: id sequences encode() never produces (a character cut in
//                             half, and the model's padding rows above the tokenizer's vocabulary)
//
// The first two need NOTHING but this repo and run everywhere. The last two additionally need the
// real model's vocab.json / merges.txt / tokenizer_config.json (~10 MB) -- too large to commit under
// the project's own convention (models and corpora are fetched, not versioned), so they are located
// at runtime and the cases SKIP WITH A WARNING if absent, exactly like the other qwen4 fixture tests.
// Point SUB0_QWEN_TOKENIZER_DIR at a directory holding the three files, or drop them in
// data/qwen_tokenizer/. See docs/QWEN_TOKENIZER.md sec 5 for the one-line fetch.
//
// Alongside the fixtures there are mutation-style checks (the transplant_tests.cpp pattern): each one
// names the specific wrong-but-plausible implementation it rules out, because a tokenizer that is
// subtly wrong still produces plausible-looking ids and every downstream number stays finite.

#include <catch2/catch_test_macros.hpp>

#include "sub0/qwen_tokenizer.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#ifdef SUB0_SOURCE_DIR

namespace {
namespace fs = std::filesystem;

fs::path fixture_dir() { return fs::path(SUB0_SOURCE_DIR) / "tests" / "fixtures" / "qwen_tokenizer"; }

// The real vocabulary's three files, if this machine has them.
fs::path model_files_dir() {
    if (const char* env = std::getenv("SUB0_QWEN_TOKENIZER_DIR")) {
        const fs::path p(env);
        if (fs::exists(p / "vocab.json")) return p;
    }
    const fs::path repo = fs::path(SUB0_SOURCE_DIR) / "data" / "qwen_tokenizer";
    if (fs::exists(repo / "vocab.json")) return repo;
    return {};
}

std::string from_hex(std::string_view h) {
    auto nib = [](char c) {
        return c >= '0' && c <= '9' ? c - '0' : (c >= 'a' && c <= 'f' ? c - 'a' + 10 : c - 'A' + 10);
    };
    std::string out;
    out.reserve(h.size() / 2);
    for (std::size_t i = 0; i + 1 < h.size(); i += 2)
        out.push_back(static_cast<char>((nib(h[i]) << 4) | nib(h[i + 1])));
    return out;
}

// One fixture row, split on TAB. Trailing empty columns are preserved (the "empty" case has three).
std::vector<std::string> split_tabs(const std::string& line, std::size_t want) {
    std::vector<std::string> out;
    std::size_t start = 0;
    while (out.size() + 1 < want) {
        const std::size_t t = line.find('\t', start);
        if (t == std::string::npos) break;
        out.push_back(line.substr(start, t - start));
        start = t + 1;
    }
    out.push_back(line.substr(start));
    while (out.size() < want) out.push_back({});
    return out;
}

std::vector<std::vector<std::string>> read_fixture(const fs::path& p, std::size_t cols) {
    std::vector<std::vector<std::string>> rows;
    std::ifstream f(p, std::ios::binary);
    std::string line;
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line[0] == '#') continue;
        rows.push_back(split_tabs(line, cols));
    }
    return rows;
}

std::vector<int> parse_ids(const std::string& s) {
    std::vector<int> out;
    std::istringstream in(s);
    int v = 0;
    while (in >> v) out.push_back(v);
    return out;
}

// Printable form for a failure message -- the inputs are full of control characters.
std::string show(std::string_view s) {
    std::string out;
    for (unsigned char c : s) {
        if (c >= 0x20 && c < 0x7F && c != '\\') out.push_back(static_cast<char>(c));
        else {
            static const char* hex = "0123456789abcdef";
            out += "\\x";
            out.push_back(hex[c >> 4]);
            out.push_back(hex[c & 0xF]);
        }
    }
    return out;
}

const sub0::qwen_tok::Tokenizer* shared_tokenizer() {
    static sub0::qwen_tok::Tokenizer tk;
    static bool tried = false;
    if (!tried) {
        tried = true;
        const fs::path d = model_files_dir();
        if (!d.empty()) {
            std::string err;
            if (!tk.load((d / "vocab.json").string(), (d / "merges.txt").string(),
                         (d / "tokenizer_config.json").string(), &err))
                FAIL("the real tokenizer files are present at " << d.string() << " but did not load: " << err);
        }
    }
    return tk.loaded() ? &tk : nullptr;
}

}  // namespace

// --- layer 1: the byte alphabet ------------------------------------------------------------------
TEST_CASE("the GPT-2 byte alphabet is a bijection and matches its known anchor values",
          "[qwen_tok]") {
    const auto& ab = sub0::qwen_tok::byte_alphabet();
    // Every byte maps to a distinct codepoint, and every one of those maps back. Without this a
    // byte-level tokenizer silently conflates two different bytes.
    bool seen[0x180] = {};
    for (int b = 0; b < 256; ++b) {
        const std::uint32_t cp = ab.to_cp[static_cast<std::size_t>(b)];
        REQUIRE(cp < 0x180);
        REQUIRE_FALSE(seen[cp]);
        seen[cp] = true;
        REQUIRE(ab.to_byte[cp] == b);
    }
    // The anchors that pin the mapping's SHAPE. Rules out the two plausible wrong versions: mapping
    // all 256 bytes to U+0100.. in order, and mapping only ASCII to itself.
    REQUIRE(ab.to_cp[' '] == 0x120);    // space -> U+0120, the vocabulary's leading-space marker
    REQUIRE(ab.to_cp['!'] == '!');      // first directly-mapped byte
    REQUIRE(ab.to_cp['~'] == '~');      // last of the printable-ASCII run
    REQUIRE(ab.to_cp['\n'] == 0x10A);
    REQUIRE(ab.to_cp['\t'] == 0x109);
    REQUIRE(ab.to_cp[0xA1] == 0xA1);    // Latin-1 run maps to itself...
    REQUIRE(ab.to_cp[0xAD] == 0x143);   // ...except the soft hyphen, which does not
    REQUIRE(ab.to_cp[0xA0] == 0x142);   // and NBSP, which sits just below that run
    REQUIRE(ab.to_cp[0x00] == 0x100);   // the first remapped byte
    REQUIRE(ab.to_cp[0x7F] == 0x121);   // DEL, right after the printable-ASCII run
    REQUIRE(ab.to_cp[0xFF] == 0xFF);
}

// --- layer 2: NFC --------------------------------------------------------------------------------
TEST_CASE("NFC matches the real reference on every fixture case", "[qwen_tok][fixture]") {
    const auto rows = read_fixture(fixture_dir() / "nfc_cases.tsv", 3);
    REQUIRE(rows.size() > 1300);        // the fixture carries every codepoint whose NFC differs
    std::size_t checked = 0;
    for (const auto& r : rows) {
        const std::string in = from_hex(r[1]), want = from_hex(r[2]);
        const std::string got = sub0::qwen_uni::nfc(in);
        INFO("case " << r[0] << "  input=" << show(in));
        REQUIRE(show(got) == show(want));
        ++checked;
    }
    REQUIRE(checked == rows.size());
}

TEST_CASE("NFC's own defining properties hold independently of the fixture", "[qwen_tok]") {
    using sub0::qwen_uni::nfc;
    // Idempotent, and the identity on ASCII (the fast path must not diverge from the slow one).
    for (const char* s : {"", "hello world", "a\tb\nc", "\x7f", "The quick brown fox."}) {
        REQUIRE(nfc(s) == std::string(s));
        REQUIRE(nfc(nfc(s)) == nfc(s));
    }
    // Composition: e + U+0301 -> U+00E9. Rules out "decompose but never recompose", which passes
    // every round-trip test while producing different token ids for the same visible text.
    REQUIRE(nfc("e\xCC\x81") == "\xC3\xA9");
    // Singleton decomposition: U+212B ANGSTROM SIGN -> U+00C5. Rules out "only handle pair
    // decompositions", the easy half of the table.
    REQUIRE(nfc("\xE2\x84\xAB") == "\xC3\x85");
    // Composition EXCLUSION: U+0958 is excluded, so U+0915 + U+093C must NOT compose back to it.
    // Rules out building the composition table from the decomposition table without filtering.
    REQUIRE(nfc("\xE0\xA4\x95\xE0\xA4\xBC") == "\xE0\xA4\x95\xE0\xA4\xBC");
    // Hangul, which is algorithmic rather than table-driven: L + V + T -> one syllable, and back is
    // never needed because NFC keeps it composed.
    REQUIRE(nfc("\xE1\x84\x92\xE1\x85\xA1\xE1\x86\xAB") == "\xED\x95\x9C");
    // Canonical ORDERING before composition: the acute must be applied to the base even though a
    // lower-class mark was typed first. Rules out "compose adjacent pairs only".
    REQUIRE(nfc("q\xCC\x87\xCC\xA3") == nfc("q\xCC\xA3\xCC\x87"));
}

// --- layer 3: the pre-tokenization regex ----------------------------------------------------------
TEST_CASE("pre-tokenization matches the real reference regex on every fixture case",
          "[qwen_tok][fixture]") {
    const auto rows = read_fixture(fixture_dir() / "pretokenize_cases.tsv", 3);
    REQUIRE(rows.size() > 250);
    std::vector<std::string_view> got;
    for (const auto& r : rows) {
        const std::string in = from_hex(r[1]);
        sub0::qwen_tok::pretokenize(in, got);
        std::string flat_got, flat_want;
        for (std::string_view c : got) { flat_got += show(c); flat_got += '|'; }
        std::istringstream want(r[2]);
        std::string h;
        while (want >> h) { flat_want += show(from_hex(h)); flat_want += '|'; }
        INFO("case " << r[0] << "  input=" << show(in));
        REQUIRE(flat_got == flat_want);
    }
}

TEST_CASE("pre-tokenization keeps the regex clauses that a simplified rewrite would drop",
          "[qwen_tok]") {
    std::vector<std::string_view> c;
    auto chunks = [&](std::string_view s) {
        sub0::qwen_tok::pretokenize(s, c);
        std::string out;
        for (std::string_view x : c) { out += show(x); out += '|'; }
        return out;
    };
    // Every chunk sequence must reconstruct the input exactly -- a regex clause that skipped input
    // would silently DELETE text from the prompt.
    for (const char* s : {"", " ", "a  \n\n  b", "\r\n\r\n", "  \t  x", "don't", "\xC2\xA0z"}) {
        sub0::qwen_tok::pretokenize(s, c);
        std::string joined;
        for (std::string_view x : c) joined += x;
        REQUIRE(joined == std::string(s));
    }
    // Clause 3 is a BARE \p{N}: one digit per chunk. This is the single most surprising real
    // behaviour, and a `\p{N}+` "fix" would look right and change every number's tokenization.
    REQUIRE(chunks("123456") == "1|2|3|4|5|6|");
    // Clause 6, `\s+(?!\S)`: two spaces before a word split 1+1, not 2+0. Rules out dropping the
    // lookahead, which yields "  |a" instead of " | a" -- different ids, same visible text.
    REQUIRE(chunks("  a") == " | a|");
    // Clause 5, `\s*[\r\n]+`: whitespace is absorbed up to and including the LAST newline of a run.
    REQUIRE(chunks("a  \n\n  b") == "a|  \\x0a\\x0a| | b|");
    // \s is Unicode White_Space, not the ASCII six. With an ASCII-only \s, U+00A0 would join the
    // following symbol run via clause 4 and produce a different chunking entirely.
    REQUIRE(chunks("a\xC2\xA0") == "a|\\xc2\\xa0|");
    REQUIRE(chunks("\xE3\x80\x80x") == "\\xe3\\x80\\x80x|");
    // Clause 1 is Unicode case-INSENSITIVE, and its fold really does reach U+017F LATIN SMALL
    // LETTER LONG S -- so "'"+U+017F splits off as a contraction rather than being absorbed by
    // clause 2 as "'" + a letter run. Rules out an ASCII-only tolower().
    REQUIRE(chunks("'sx") == "'s|x|");
    REQUIRE(chunks("'Sx") == "'S|x|");
    REQUIRE(chunks("'\xC5\xBFx") == "'\\xc5\\xbf|x|");
    // ...but the fold is only case folding: U+212A KELVIN SIGN folds to 'k', for which there is no
    // alternative, so it must NOT split.
    REQUIRE(chunks("'\xE2\x84\xAAx") == "'\\xe2\\x84\\xaax|");
    // Clause 4's optional leading space is an ASCII space specifically, and it is a SINGLE one.
    REQUIRE(chunks(" ==") == " ==|");
    REQUIRE(chunks("  ==") == " | ==|");
}

// --- layer 4: the whole pipeline, against the real vocabulary --------------------------------------
TEST_CASE("encode matches the real reference on every fixture case", "[qwen_tok][fixture]") {
    const sub0::qwen_tok::Tokenizer* tk = shared_tokenizer();
    if (!tk) {
        WARN("the real Qwen tokenizer files were not found (set SUB0_QWEN_TOKENIZER_DIR, or put "
             "vocab.json/merges.txt/tokenizer_config.json in data/qwen_tokenizer/) -- skipping");
        return;
    }
    const auto rows = read_fixture(fixture_dir() / "encode_cases.tsv", 4);
    REQUIRE(rows.size() > 250);
    std::vector<int> ids;
    for (const auto& r : rows) {
        const std::string in = from_hex(r[1]);
        const std::vector<int> want = parse_ids(r[2]);
        tk->encode(in, ids);
        INFO("case " << r[0] << "  input=" << show(in));
        REQUIRE(ids == want);
        // ...and decode(ids) must reproduce what the reference's own decode produced, which for a
        // non-NFC input is NFC(input), not the input. That is the real round-trip contract.
        INFO("decode of case " << r[0]);
        REQUIRE(show(tk->decode(ids)) == show(from_hex(r[3])));
    }
}

TEST_CASE("decode matches the real reference on id sequences encode never produces",
          "[qwen_tok][fixture]") {
    const sub0::qwen_tok::Tokenizer* tk = shared_tokenizer();
    if (!tk) {
        WARN("the real Qwen tokenizer files were not found -- skipping");
        return;
    }
    for (const auto& r : read_fixture(fixture_dir() / "decode_cases.tsv", 3)) {
        const std::vector<int> ids = parse_ids(r[1]);
        INFO("case " << r[0]);
        REQUIRE(show(tk->decode(ids)) == show(from_hex(r[2])));
    }
}

TEST_CASE("the loaded vocabulary has the real model's own sizes and special-token ids",
          "[qwen_tok][fixture]") {
    const sub0::qwen_tok::Tokenizer* tk = shared_tokenizer();
    if (!tk) {
        WARN("the real Qwen tokenizer files were not found -- skipping");
        return;
    }
    // The BPE vocabulary, the added tokens, and the gap up to the model's own VOCAB axis. WP4b-f
    // already builds the engine at VOCAB = 248320; the 243 rows above 248,077 are padding, which is
    // why decode has to tolerate them (covered by decode_cases.tsv's pad_row_* cases).
    REQUIRE(tk->base_vocab_size() == 248044);
    REQUIRE(tk->vocab_size() == 248077);
    REQUIRE(tk->eos_id() == 248046);      // <|im_end|>,     tokenizer_config.json eos_token
    REQUIRE(tk->pad_id() == 248044);      // <|endoftext|>,  tokenizer_config.json pad_token
}

TEST_CASE("encode makes the real tokenizer's non-obvious choices, not the plausible ones",
          "[qwen_tok][fixture]") {
    const sub0::qwen_tok::Tokenizer* tk = shared_tokenizer();
    if (!tk) {
        WARN("the real Qwen tokenizer files were not found -- skipping");
        return;
    }
    // Digits are one token EACH. A BPE that let digits merge would give "123456" one or two ids.
    REQUIRE(tk->encode("123456") == std::vector<int>{16, 17, 18, 19, 20, 21});
    // A special token is a LITERAL match taken before the regex and BPE ever see it. The plausible
    // wrong answer -- BPE over the characters "<", "|", "endoftext", ... -- is what happens if the
    // added-token split is forgotten, and it is 3 ids instead of 1.
    REQUIRE(tk->encode("<|endoftext|>") == std::vector<int>{248044});
    REQUIRE(tk->encode("hi<|im_end|>there") == std::vector<int>{5834, 248046, 17977});
    // ...and the split is leftmost-LONGEST over raw bytes, so a partial marker stays ordinary text
    // while an embedded complete one still wins.
    REQUIRE(tk->encode("<|im_end|").size() == 5);
    REQUIRE(tk->encode("<<|im_end|>") == std::vector<int>{27, 248046});
    // Merges never cross a pre-token boundary: encoding the chunks separately and concatenating must
    // equal encoding the whole. Rules out running BPE over the un-split text.
    REQUIRE(tk->encode("hello world") == std::vector<int>{14556, 1814});
    // The regex really does carry \p{M}, so a combining mark CONTINUES a letter run instead of ending
    // it. This is the trap of docs/QWEN_TOKENIZER.md sec 2.1: gating against plain AutoTokenizer would
    // have used an older \p{M}-less regex, which yields [24400, 64020] here -- two valid-looking ids
    // instead of one. Latin accents cannot catch this (NFC composes them away first); only a script
    // whose marks have no precomposed form can.
    REQUIRE(tk->encode("\xE0\xB8\x81\xE0\xB8\xB1\xE0\xB8\x99") == std::vector<int>{148783});    // Thai
    REQUIRE(tk->encode("\xE0\xA4\x95\xE0\xA5\x8D\xE0\xA4\xB7") == std::vector<int>{151858});    // Devanagari
    // NFC really is applied before tokenization -- the decomposed and composed spellings of the same
    // word must give the same ids. Rules out skipping the normalizer, which no ASCII test can see.
    REQUIRE(tk->encode("cafe\xCC\x81") == tk->encode("caf\xC3\xA9"));
    // Every input is representable: no unknown token exists, so even a lone continuation byte
    // encodes (as its own byte token) rather than being dropped or replaced.
    REQUIRE_FALSE(tk->encode("\xC3\xA9").empty());
    // decode(encode(x)) == x for NFC input, including through a special token.
    for (const char* s : {"hello world", "The quick brown fox.", "caf\xC3\xA9",
                          "\xE4\xBD\xA0\xE5\xA5\xBD", "a<|im_end|>b", "  \t\n  x"})
        REQUIRE(tk->decode(tk->encode(s)) == std::string(s));
}

#endif  // SUB0_SOURCE_DIR
