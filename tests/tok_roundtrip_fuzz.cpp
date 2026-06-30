// tok_roundtrip_fuzz.cpp — generative round-trip hardening for sub0::tok.
//
// The unit suite (tok_lib_tests.cpp) pins specific WORKED examples. This file is the
// GENERATIVE net that catches what curated cases miss, so a round-trip regression
// surfaces HERE (fast, deterministic, with an expected-vs-actual diff) instead of deep
// in a multi-hundred-MB corpus run that takes minutes to fail:
//   1. dogfood   — round-trip the project's OWN C++ source (real punctuation, CamelCase,
//                  snake_case, hyphenated identifiers, indentation, string/char literals),
//                  with the tokenizer LEARNED on that same source (a realistic attested set
//                  and merge table, not a toy corpus).
//   2. fuzz      — deterministic random-mutation stress (byte flip/insert/delete/dup on seed
//                  strings) + pure random byte blobs. A FIXED seed makes any failure exactly
//                  reproducible; the report prints the offending input escaped.
//   3. patterns  — a NON-failing report (Catch2 WARN) of how real code/prose actually
//                  tokenizes, to spot surprising structure (newline+indentation falling back
//                  to verbatim bytes, JOIN density, SPELL groups) that should guide the design.
//
// Contract under test (everywhere): detokenize(encode(x)) == normalize_text(x).

#include <catch2/catch_test_macros.hpp>

#include "sub0/casing.hpp"
#include "sub0/tokenizer.hpp"

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

using sub0::tok::Tokenizer;

namespace {

// ---------------------------------------------------------------------------
//  Shared helpers
// ---------------------------------------------------------------------------

// Make control bytes visible so a diff line is readable in the test log: \n \t \r as
// escapes, other non-printables as \xHH, everything else verbatim.
std::string vis(std::string_view s) {
    static const char* hex = "0123456789abcdef";
    std::string o;
    o.reserve(s.size());
    for (unsigned char c : s) {
        if      (c == '\n') o += "\\n";
        else if (c == '\t') o += "\\t";
        else if (c == '\r') o += "\\r";
        else if (c < 0x20 || c == 0x7F) { o += "\\x"; o += hex[c >> 4]; o += hex[c & 0xF]; }
        else o += static_cast<char>(c);
    }
    return o;
}

struct RtResult {
    bool        ok = true;
    std::string report;  // populated only on failure (expected-vs-actual around first divergence)
};

// Render an id stream symbolically so the token structure is legible: markers as <J>/<NL>/
// <OQ>/<[…]>, case as <C>/<U>, a content word as {its-text}, a standalone byte as its char.
// Used by the pattern-breakdown report to SEE how a construct tokenizes (token count = pieces).
std::string describe_tokens(const Tokenizer& t, const std::vector<int>& ids) {
    std::string o;
    for (int id : ids) {
        if      (id == t.join_id)        o += "<J>";
        else if (id == t.newline_id)     o += "<NL>";
        else if (id == t.para_id)        o += "<P>";
        else if (id == t.cap_id)         o += "<C>";
        else if (id == t.up_id)          o += "<U>";
        else if (id == t.odquote_id)     o += "<OQ>";
        else if (id == t.cdquote_id)     o += "<CQ>";
        else if (id == t.spell_start_id) o += "<[";
        else if (id == t.spell_end_id)   o += "]>";
        else if (id >= 0 && id < static_cast<int>(t.expansion.size())) {
            std::string w;
            for (int code : t.expansion[static_cast<std::size_t>(id)]) w += static_cast<char>(code);
            o += '{'; o += vis(w); o += '}';
        }
        o += ' ';
    }
    return o;
}

// The single round-trip check, with a configurator-style first-divergence diff on failure.
RtResult round_trip(const Tokenizer& t, const std::string& x, std::string_view label = {}) {
    long r = 0;
    const std::string expect = sub0::casing::normalize_text(x, r);
    const std::string actual = sub0::tok::detokenize(t, sub0::tok::encode(t, x));
    if (actual == expect) return {true, {}};

    std::size_t p = 0;
    while (p < actual.size() && p < expect.size() && actual[p] == expect[p]) ++p;
    const std::size_t lo = p > 32 ? p - 32 : 0;
    std::string rep = "ROUND-TRIP FAIL";
    if (!label.empty()) { rep += " ["; rep += label; rep += "]"; }
    rep += " at byte " + std::to_string(p) +
           " (expect " + std::to_string(expect.size()) + "B / actual " + std::to_string(actual.size()) + "B)\n";
    rep += "  input:    ..." + vis(std::string_view(x).substr(lo < x.size() ? lo : 0, 80)) + "...\n";
    rep += "  expected: ..." + vis(std::string_view(expect).substr(lo, 80)) + "...\n";
    rep += "  actual:   ..." + vis(std::string_view(actual).substr(lo, 80)) + "...\n";
    return {false, std::move(rep)};
}

// A general corpus with enough lowercase bulk to attest the common words (so the truecasing
// CAP/UP collapse — the path that produced the "NASA's" bug — is actually exercised), plus
// capitals, all-caps, quotes, digits and code-like punctuation.
const std::string kCorpus = [] {
    std::string c;
    for (int i = 0; i < 60; ++i) {
        c += "the quick brown fox jumps over the lazy dog .\n"
             "she said , \" hello world \" and they left .\n"
             "they went home and they slept ; it was 1999 .\n"
             "a cat sat on the mat and the dog ran away .\n"
             "the commons attribution license is non commercial here .\n"
             "save scan state and load model from the cache please .\n";
        c += "The dog ran and the cat slept .\n"
             "They were happy and she was very glad .\n"
             "THE END of the day was finally near .\n";
    }
    return c;
}();

// ---------------------------------------------------------------------------
//  Dogfood corpus: the project's own C++ source
// ---------------------------------------------------------------------------

#ifdef SUB0_SOURCE_DIR

// Gather the project's own source files (capped) so we both LEARN on real code and
// round-trip each file. Skips the build tree and VCS metadata.
std::vector<std::filesystem::path> source_files(std::size_t max_files = 64) {
    namespace fs = std::filesystem;
    std::vector<fs::path> out;
    const std::array<const char*, 4> roots{"src", "include", "tools", "tests"};
    auto is_src = [](const fs::path& p) {
        const std::string e = p.extension().string();
        return e == ".cpp" || e == ".hpp" || e == ".h" || e == ".cu" || e == ".cc" || e == ".inl";
    };
    for (const char* root : roots) {
        const fs::path base = fs::path(SUB0_SOURCE_DIR) / root;
        std::error_code ec;
        if (!fs::exists(base, ec)) continue;
        for (auto it = fs::recursive_directory_iterator(base, ec);
             it != fs::recursive_directory_iterator(); it.increment(ec)) {
            if (ec) break;
            if (it->is_regular_file(ec) && is_src(it->path())) {
                out.push_back(it->path());
                if (out.size() >= max_files) return out;
            }
        }
    }
    return out;
}

std::string read_file(const std::filesystem::path& p) {
    std::ifstream is(p, std::ios::binary);
    return {std::istreambuf_iterator<char>(is), std::istreambuf_iterator<char>()};
}

#endif  // SUB0_SOURCE_DIR

}  // namespace

// ---------------------------------------------------------------------------
//  Worked examples — constructs that real corpora throw at the encoder
// ---------------------------------------------------------------------------

TEST_CASE("round-trip: worked examples across constructs", "[tok][fuzz]") {
    const Tokenizer t = sub0::tok::learn(kCorpus);
    const std::vector<std::string> cases = {
        // hyphen-joined, capitalized, CamelCase compounds (the user's example)
        "Commons Attribution-NonCommercial-NoDerivs",
        "Creative Commons Attribution-ShareAlike 4.0 International",
        // CamelCase / snake_case identifiers (source-code shapes)
        "ThisIsMyAwesomeFunction",
        "save_scan_state(corpus, &out_tokens);",
        "auto bpe_encode_word = [&](const Tokenizer& t) { return t.n_base; };",
        "std::vector<std::pair<int,int>> merges;  // ordered (left,right)",
        // punctuation density, contractions, possessives
        "don't, won't, can't — it's the model's output…",
        "URL: https://example.com/a/b?x=1&y=2#frag",
        "email me @ x%y/z+~ (100% sure): a/b = c+d",
        // quotes and nesting
        "She said, \"Don't touch the NASA's antidisestablishment thing!\"",
        "\"start\" middle \"end\"",
        // whitespace shapes
        "line one\nline two\n\npara two\n\tindented\n    four spaces",
        // numbers, mixed
        "v1.2.3-rc4 build #4567 (0xDEADBEEF)",
    };
    int fails = 0;
    std::string reports;
    for (const std::string& x : cases) {
        const RtResult rt = round_trip(t, x, x.substr(0, 24));
        if (!rt.ok && fails++ < 8) reports += rt.report + "\n";
    }
    INFO(reports);
    CHECK(fails == 0);
}

// ---------------------------------------------------------------------------
//  Dogfood — round-trip the project's own source, tokenizer learned on it
// ---------------------------------------------------------------------------

#ifdef SUB0_SOURCE_DIR
TEST_CASE("round-trip: dogfood the project's own C++ source", "[tok][fuzz][dogfood]") {
    const std::vector<std::filesystem::path> files = source_files();
    REQUIRE_FALSE(files.empty());                       // SUB0_SOURCE_DIR must point at the tree

    // Learn on the concatenated source (capped) so the attested set + merges reflect real
    // code, then round-trip each file under that tokenizer.
    std::string corpus;
    std::vector<std::pair<std::string, std::string>> contents;  // (name, text)
    for (const auto& p : files) {
        std::string text = read_file(p);
        if (text.empty()) continue;
        if (corpus.size() < 1u << 21) corpus += text;   // cap the learn set at ~2 MB
        contents.emplace_back(p.filename().string(), std::move(text));
    }
    const Tokenizer t = sub0::tok::learn(corpus, {.vocab_target = 4096});

    int fails = 0;
    std::string reports;
    for (const auto& [name, text] : contents) {
        const RtResult rt = round_trip(t, text, name);
        if (!rt.ok && fails++ < 6) reports += rt.report + "\n";
    }
    INFO("dogfood files: " + std::to_string(contents.size()) + "\n" + reports);
    CHECK(fails == 0);
}
#endif

// ---------------------------------------------------------------------------
//  Fuzz — deterministic random-mutation stress + random byte blobs
// ---------------------------------------------------------------------------

namespace {

// Seed strings the mutator perturbs; chosen to sit near the encoder's branch points
// (quotes, newlines, markers' worth of case, glued punctuation, multibyte/UTF-8 bytes).
const std::vector<std::string> kSeeds = {
    "the dog ran .",
    "She said, \"hello\" to NASA's team.",
    "a\nb\n\nc\td   e",
    "Commons-Attribution-NoDerivs",
    "don't can't won't it's",
    "x\"y\"z \"q\" w",
    "THE END of days",
    "caf\xC3\xA9" " pi\xC3\xB1" "ata na\xC3\xAF" "ve",   // UTF-8 multibyte letters (café piñata naïve)
    "save_scan_state CamelCaseWord",
    "   leading and trailing   ",
};

// Apply `nmut` random single-byte mutations (flip / insert / delete / duplicate) to `s`.
std::string mutate(std::string s, std::mt19937& rng, int nmut) {
    std::uniform_int_distribution<int> op(0, 3), byte(0, 255);
    for (int m = 0; m < nmut; ++m) {
        if (s.empty()) { s.push_back(static_cast<char>(byte(rng))); continue; }
        std::uniform_int_distribution<std::size_t> pos(0, s.size() - 1);
        const std::size_t i = pos(rng);
        switch (op(rng)) {
            case 0: s[i] = static_cast<char>(byte(rng));                       break;  // flip
            case 1: s.insert(s.begin() + static_cast<std::ptrdiff_t>(i),
                             static_cast<char>(byte(rng)));                    break;  // insert
            case 2: s.erase(s.begin() + static_cast<std::ptrdiff_t>(i));       break;  // delete
            case 3: s.insert(s.begin() + static_cast<std::ptrdiff_t>(i), s[i]);break;  // duplicate
        }
    }
    return s;
}

}  // namespace

TEST_CASE("round-trip: fuzz mutated seeds", "[tok][fuzz]") {
    const Tokenizer t = sub0::tok::learn(kCorpus);
    std::mt19937 rng(0x5EED5EEDu);                      // fixed -> reproducible
    int fails = 0;
    std::string reports;
    for (int iter = 0; iter < 4000; ++iter) {
        const std::string& seed = kSeeds[static_cast<std::size_t>(iter) % kSeeds.size()];
        const std::string x = mutate(seed, rng, 1 + (iter % 6));
        const RtResult rt = round_trip(t, x, "fuzz#" + std::to_string(iter));
        if (!rt.ok && fails++ < 8) reports += rt.report + "\n";
    }
    INFO(reports);
    CHECK(fails == 0);
}

TEST_CASE("round-trip: fuzz random byte blobs", "[tok][fuzz]") {
    const Tokenizer t = sub0::tok::learn(kCorpus);
    std::mt19937 rng(0xB10BB10Bu);
    std::uniform_int_distribution<int> len(0, 200), byte(0, 255);
    int fails = 0;
    std::string reports;
    for (int iter = 0; iter < 4000; ++iter) {
        std::string x(static_cast<std::size_t>(len(rng)), '\0');
        for (char& c : x) c = static_cast<char>(byte(rng));
        const RtResult rt = round_trip(t, x, "blob#" + std::to_string(iter));
        if (!rt.ok && fails++ < 8) reports += rt.report + "\n";
    }
    INFO(reports);
    CHECK(fails == 0);
}

// ---------------------------------------------------------------------------
//  Pattern breakdowns — SEE how surprising constructs tokenize (non-failing)
// ---------------------------------------------------------------------------

TEST_CASE("patterns: token breakdown of surprising constructs", "[tok][patterns]") {
    const Tokenizer t = sub0::tok::learn(kCorpus);
    const std::vector<std::string> examples = {
        "Commons Attribution-NonCommercial-NoDerivs",   // hyphen-joined + CamelCase
        "ThisIsMyAwesomeFunction",                      // CamelCase identifier
        "save_scan_state",                              // snake_case identifier
        "std::vector<int>",                             // glued code punctuation
        "the sunday morning",                           // does "sunday" = sun+day?
        "She said \"hello\" .",                         // directional quotes
    };
    std::string rep = "\n=== token breakdowns ('{}'=word/byte, <J>=glue, <C/U>=case) ===\n";
    for (const std::string& x : examples) {
        const std::vector<int> ids = sub0::tok::encode(t, x);
        rep += "  \"" + vis(x) + "\"  (" + std::to_string(ids.size()) + " tok)\n    " +
               describe_tokens(t, ids) + "\n";
    }
    WARN(rep);
    SUCCEED();
}

// ---------------------------------------------------------------------------
//  Patterns — non-failing report of how real source actually tokenizes
// ---------------------------------------------------------------------------

#ifdef SUB0_SOURCE_DIR
TEST_CASE("patterns: how the project's own source tokenizes", "[tok][patterns]") {
    std::string corpus;
    for (const auto& p : source_files()) {
        std::string text = read_file(p);
        if (corpus.size() < 1u << 21) corpus += text;
    }
    REQUIRE_FALSE(corpus.empty());
    const Tokenizer t = sub0::tok::learn(corpus, {.vocab_target = 4096});

    long r = 0;
    const std::string norm = sub0::casing::normalize_text(corpus, r);
    const std::vector<int> ids = sub0::tok::encode(t, norm);

    long long n_join = 0, n_nl = 0, n_para = 0, n_odq = 0, n_cdq = 0, n_spell = 0, n_spell_sub = 0;
    long long ws_space = 0, ws_tab = 0, ws_nl = 0, ws_cr = 0;     // verbatim whitespace BYTES
    bool in_spell = false;
    for (int id : ids) {
        if      (id == t.join_id)        ++n_join;
        else if (id == t.newline_id)     ++n_nl;
        else if (id == t.para_id)        ++n_para;
        else if (id == t.odquote_id)     ++n_odq;
        else if (id == t.cdquote_id)     ++n_cdq;
        else if (id == t.spell_start_id) { ++n_spell; in_spell = true; }
        else if (id == t.spell_end_id)   in_spell = false;
        else if (id >= 0 && id < 256) {
            if      (id == ' ')  ++ws_space;
            else if (id == '\t') ++ws_tab;
            else if (id == '\n') ++ws_nl;
            else if (id == '\r') ++ws_cr;
        }
        if (in_spell && id != t.spell_start_id) ++n_spell_sub;
    }
    // The "indentation" surprise: a '\n' immediately followed by a space/tab can never become a
    // lone NEWLINE token (the gap is >1), so it falls back to verbatim whitespace bytes. Count
    // it straight from the normalized text — this is the dominant source-code inefficiency.
    long long indented_lines = 0, total_lines = 0;
    for (std::size_t i = 0; i + 1 < norm.size(); ++i)
        if (norm[i] == '\n') { ++total_lines; if (norm[i + 1] == ' ' || norm[i + 1] == '\t') ++indented_lines; }

    const double bpt = ids.empty() ? 0.0 : static_cast<double>(norm.size()) / static_cast<double>(ids.size());
    std::string rep = "\n=== source tokenization patterns ===\n";
    rep += "bytes=" + std::to_string(norm.size()) + " tokens=" + std::to_string(ids.size()) +
           "  bytes/token=" + std::to_string(bpt) + "\n";
    rep += "markers: JOIN=" + std::to_string(n_join) + " NEWLINE=" + std::to_string(n_nl) +
           " PARA=" + std::to_string(n_para) + " ODQ=" + std::to_string(n_odq) +
           " CDQ=" + std::to_string(n_cdq) + " SPELL=" + std::to_string(n_spell) +
           " (avg sub " + std::to_string(n_spell ? static_cast<double>(n_spell_sub) / static_cast<double>(n_spell) : 0.0) + ")\n";
    rep += "verbatim ws bytes: space=" + std::to_string(ws_space) + " tab=" + std::to_string(ws_tab) +
           " nl=" + std::to_string(ws_nl) + " cr=" + std::to_string(ws_cr) + "\n";
    rep += "indented lines: " + std::to_string(indented_lines) + " / " + std::to_string(total_lines) +
           "  (each defeats NEWLINE -> verbatim '\\n'+indent bytes)\n";
    WARN(rep);
    SUCCEED();
}
#endif
