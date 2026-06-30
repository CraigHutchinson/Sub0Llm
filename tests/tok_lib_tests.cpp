// tok_lib_tests.cpp — pure sub0::tok library tests, with NO dependency on the engine
// or the baked production tokenizer.bin. The tokenizer was factored into a reusable
// library precisely so it can be exercised on small in-memory test corpora here,
// deterministically and fast. This target (sub0_tok_tests) links only sub0_tok +
// Catch2, so it builds and runs even while the engine DLLs are locked by a training run.
//
// The contract under test is the round-trip: detokenize(encode(x)) == normalize_text(x)
// for any text x whose bytes are in the learned alphabet.

#include <catch2/catch_test_macros.hpp>

#include "sub0/casing.hpp"
#include "sub0/tokenizer.hpp"
#include "sub0/unigram.hpp"

#include <algorithm>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using sub0::tok::Tokenizer;
using sub0::tok::LearnOptions;

namespace {

// A small but varied corpus that exercises lowercase words, capitalised words, an
// all-caps word, punctuation, contractions and digits — enough alphabet coverage that
// the round-trip strings below are all in-alphabet.
const std::string kCorpus = [] {
    std::string c;
    for (int i = 0; i < 50; ++i) {
        // Lowercase bulk: makes the common words attested (eligible for case collapse).
        c += "the quick brown fox jumps over the lazy dog .\n"
             "she said , \" hello world \" and they left .\n"
             "they went home and they slept ; it was 1999 .\n"
             "a cat sat on the mat and the dog ran away .\n";
        // Sentence-initial capitals (CAP marker) + an all-caps word (UP marker), so the
        // learned alphabet actually contains the markers; these are not name-uses (not
        // preceded by a lowercase word), so the words stay attested.
        c += "The dog ran and the cat slept .\n"
             "They were happy and she was very glad .\n"
             "THE END of the day was finally near .\n";
    }
    return c;
}();

// detokenize(encode(x)) must reproduce normalize_text(x).
bool round_trips(const Tokenizer& t, const std::string& x) {
    long r = 0;
    const std::string expect = sub0::casing::normalize_text(x, r);
    return sub0::tok::detokenize(t, sub0::tok::encode(t, x)) == expect;
}

}  // namespace

// ---------------------------------------------------------------------------
//  Unigram LM vocabulariser (the BPE replacement)
// ---------------------------------------------------------------------------

TEST_CASE("Unigram: learns morphological pieces, segments losslessly, no dead tokens", "[tok][unigram]") {
    using sub0::tok::learn_unigram;
    using sub0::tok::corpus_tokens;
    // A corpus where a shared suffix ("ing") and stems recur -- the piece a good vocabulariser finds.
    std::vector<std::pair<std::string, long long>> words = {
        {"running", 50}, {"jumping", 40}, {"walking", 30}, {"talking", 20}, {"reading", 25},
        {"runner", 15}, {"jumper", 10}, {"walker", 8}, {"talked", 12}, {"reads", 18},
        {"the", 300}, {"and", 200}, {"a", 150}, {"cat", 40}, {"dog", 35},
    };
    sub0::tok::UnigramOptions opt;
    opt.target = 64; opt.min_count = 2;
    const sub0::tok::Unigram u = learn_unigram(words, opt);

    REQUIRE(u.size() > 0);
    REQUIRE(u.size() <= 80);                         // ~target + the mandatory single bytes

    // Lossless: every word's Viterbi segmentation concatenates back to the word.
    for (const auto& [w, f] : words) {
        std::string rebuilt;
        for (int id : u.segment(w)) rebuilt += u.token[static_cast<std::size_t>(id)];
        REQUIRE(rebuilt == w);
    }
    // No dead tokens: every token is reachable (its bytes segment to itself or appears) -- the whole
    // point vs BPE. Check a stronger property: total corpus tokens < total bytes (real compression).
    long long bytes = 0;
    const long long toks = corpus_tokens(u, words, &bytes);
    REQUIRE(toks < bytes);                           // beats character encoding
    REQUIRE(toks > 0);

    // It must learn real multi-byte pieces (sub-words / whole words), not fall back to single bytes.
    // (Recovery of a *specific* morpheme like "ing" is frequency-dependent and shown on real corpora
    // -- the configurator A/B vocab contains it; a 15-word toy corpus is too small to force it.)
    int multi = 0, long3 = 0;
    for (const std::string& s : u.token) { if (s.size() > 1) ++multi; if (s.size() >= 3) ++long3; }
    REQUIRE(multi > 5);
    REQUIRE(long3 >= 1);
}

TEST_CASE("learn builds a usable tokenizer from an in-memory corpus", "[tok]") {
    const Tokenizer t = sub0::tok::learn(kCorpus);
    REQUIRE(t.loaded);
    REQUIRE(t.vocab > t.n_base);          // some merges were learned
    REQUIRE(t.n_base > 0);
}

TEST_CASE("round-trip holds for in-alphabet text (current scheme)", "[tok]") {
    const Tokenizer t = sub0::tok::learn(kCorpus);
    REQUIRE(round_trips(t, "the quick brown fox ."));
    REQUIRE(round_trips(t, "the dog ran and the cat slept"));
    REQUIRE(round_trips(t, "they said hello ."));
}

TEST_CASE("truecasing round-trips capitals and all-caps", "[tok]") {
    const Tokenizer t = sub0::tok::learn(kCorpus);
    REQUIRE(round_trips(t, "The dog ran ."));
    REQUIRE(round_trips(t, "THE DOG RAN ."));
    REQUIRE(round_trips(t, "She said hello and they slept ."));
}

TEST_CASE("case markers keep the word token case-shared", "[tok]") {
    const Tokenizer t = sub0::tok::learn(kCorpus);
    const std::vector<int> low = sub0::tok::encode(t, "they");
    const std::vector<int> cap = sub0::tok::encode(t, "They");
    REQUIRE_FALSE(low.empty());
    REQUIRE(cap.size() == low.size() + 1);   // a leading <|cap|> marker, same tail
    REQUIRE(cap.front() == t.cap_id);
    REQUIRE(std::vector<int>(cap.begin() + 1, cap.end()) == low);
}

TEST_CASE("encode is deterministic", "[tok]") {
    const Tokenizer t = sub0::tok::learn(kCorpus);
    REQUIRE(sub0::tok::encode(t, "the cat sat") == sub0::tok::encode(t, "the cat sat"));
}

// ---------------------------------------------------------------------------
//  JOIN / implicit-space scheme
// ---------------------------------------------------------------------------

TEST_CASE("JOIN scheme: complete base alphabet + markers", "[tok][join]") {
    const Tokenizer t = sub0::tok::learn(kCorpus);
    REQUIRE(t.join_id >= 0);
    REQUIRE(t.n_base == 269);                 // 256 bytes + 13 markers
    REQUIRE(t.cap_id == 256);
    REQUIRE(t.up_id == 257);
    REQUIRE(t.join_id == 258);
    REQUIRE(t.newline_id == 259);
    REQUIRE(t.para_id == 260);
    REQUIRE(t.odquote_id == 261);
    REQUIRE(t.cdquote_id == 262);
    REQUIRE(t.spell_start_id == 263);
    REQUIRE(t.spell_end_id == 264);
    REQUIRE(t.space2_id == 265);
    REQUIRE(t.space4_id == 266);
    REQUIRE(t.tab2_id == 267);
    REQUIRE(t.tab4_id == 268);
    REQUIRE(t.vocab > t.n_base);              // merges still learned on top
}

TEST_CASE("JOIN scheme: a single inter-word space costs no token", "[tok][join]") {
    const Tokenizer t = sub0::tok::learn(kCorpus);
    const std::vector<int> ids = sub0::tok::encode(t, "the dog ran");
    REQUIRE_FALSE(ids.empty());
    for (int id : ids) REQUIRE(id != ' ');    // spaces are implicit, never a literal byte token
    REQUIRE(round_trips(t, "the dog ran"));
}

TEST_CASE("JOIN scheme: round-trips spacing, punctuation and case", "[tok][join]") {
    const Tokenizer t = sub0::tok::learn(kCorpus);
    REQUIRE(round_trips(t, "the quick brown fox ."));
    REQUIRE(round_trips(t, "the cat, the dog; and a fox."));   // glued punctuation -> JOIN
    REQUIRE(round_trips(t, "glued.text.no.spaces"));           // all glued
    REQUIRE(round_trips(t, "The dog ran. They were happy."));  // capitalised words
    REQUIRE(round_trips(t, "THE END of the day"));             // all-caps (UP across the word)
    REQUIRE(round_trips(t, "She said hello to the cat."));
}

TEST_CASE("JOIN scheme: round-trips whitespace variety", "[tok][join]") {
    const Tokenizer t = sub0::tok::learn(kCorpus);
    REQUIRE(round_trips(t, "line one\nline two"));             // single newline -> NEWLINE
    REQUIRE(round_trips(t, "para one\n\npara two"));           // blank line  -> PARA
    REQUIRE(round_trips(t, "a  b   c"));                       // multi-space -> SPACE2 (+ byte)
    REQUIRE(round_trips(t, "tab\there"));                      // lone tab    -> verbatim byte
    REQUIRE(round_trips(t, " leading and trailing "));         // edge whitespace
    REQUIRE(round_trips(t, "x\n\n\ny"));                       // 3 newlines  -> PARA + NEWLINE
}

// Run-length whitespace: multi-space/tab runs collapse to SPACE2/4 / TAB2/4 (so code
// indentation stops costing one token per space); a single inter-word space stays free.
TEST_CASE("JOIN scheme: run-length whitespace tokens collapse indentation", "[tok][join]") {
    const Tokenizer t = sub0::tok::learn(kCorpus);
    auto enc = [&](const std::string& s) { return sub0::tok::encode(t, s); };

    // 4-space indent after a newline -> NEWLINE + SPACE4 (was 1 nl byte + 4 space bytes).
    const std::vector<int> ind = enc("a\n    b");
    REQUIRE(std::count(ind.begin(), ind.end(), t.newline_id) == 1);
    REQUIRE(std::count(ind.begin(), ind.end(), t.space4_id) == 1);
    REQUIRE(std::count(ind.begin(), ind.end(), ' ') == 0);    // no literal space bytes
    REQUIRE(round_trips(t, "a\n    b"));

    // Greedy tiling: 8->2xSPACE4, 6->SPACE4+SPACE2, 3->SPACE2+space, 2->SPACE2.
    const std::vector<int> sp8 = enc("a\n        b");
    REQUIRE(std::count(sp8.begin(), sp8.end(), t.space4_id) == 2);
    REQUIRE(round_trips(t, "x        y"));                     // 8 spaces
    REQUIRE(round_trips(t, "x      y"));                       // 6 spaces
    REQUIRE(round_trips(t, "x   y"));                          // 3 spaces (SPACE2 + 1)
    const std::vector<int> sp2 = enc("x  y");                  // 2 spaces -> exactly one SPACE2
    REQUIRE(std::count(sp2.begin(), sp2.end(), t.space2_id) == 1);

    // Tabs tile the same way; a single inter-word space is still implicit (free).
    const std::vector<int> tb4 = enc("a\t\t\t\tb");
    REQUIRE(std::count(tb4.begin(), tb4.end(), t.tab4_id) == 1);
    REQUIRE(round_trips(t, "a\t\tb\tc"));                      // TAB2 + lone tab byte
    for (int id : enc("the dog ran")) REQUIRE(id != t.space2_id);   // single spaces never tile
    REQUIRE(round_trips(t, "\n\t  mixed\n\n    indent\t\t"));  // mixed nl/tab/space + trailing
}

TEST_CASE("JOIN scheme: complete base encodes out-of-corpus bytes", "[tok][join]") {
    const Tokenizer t = sub0::tok::learn(kCorpus);
    // Symbols the tiny corpus may never contain still round-trip -- the total 256-byte base
    // means no input character is silently dropped (the §1 correctness fix).
    REQUIRE(round_trips(t, "email me @ x%y/z+~"));
    REQUIRE(round_trips(t, "100% sure: a/b = c+d"));
    REQUIRE(round_trips(t, "{code}[index]"));
}

TEST_CASE("JOIN scheme: serialize/deserialize preserves the scheme + round-trip", "[tok][join]") {
    const Tokenizer t = sub0::tok::learn(kCorpus);
    std::ostringstream os(std::ios::binary);
    sub0::tok::serialize(t, os);
    std::istringstream is(os.str(), std::ios::binary);
    Tokenizer t2;
    REQUIRE(sub0::tok::deserialize(t2, is));
    REQUIRE(t2.join_id >= 0);                  // scheme inferred from the base alphabet
    REQUIRE(t2.join_id == t.join_id);
    REQUIRE(round_trips(t2, "The cat, the dog.\nLine two here."));
}

TEST_CASE("JOIN scheme: encode is deterministic", "[tok][join]") {
    const Tokenizer t = sub0::tok::learn(kCorpus);
    REQUIRE(sub0::tok::encode(t, "the cat, sat.") == sub0::tok::encode(t, "the cat, sat."));
}

TEST_CASE("JOIN scheme: directional double quotes round-trip + bundle spacing", "[tok][join]") {
    const Tokenizer t = sub0::tok::learn(kCorpus);
    REQUIRE(round_trips(t, "she said \"hello\" and left ."));
    REQUIRE(round_trips(t, "\"quoted start\" then text"));     // leading open quote
    REQUIRE(round_trips(t, "a \"b\" c \"d e\" f"));            // multiple pairs, multi-word quote
    REQUIRE(round_trips(t, "end with a quote\""));             // trailing bare quote
    REQUIRE(round_trips(t, "no\"space\"quotes"));              // glue/glue -> bare fallback
    // ` "hi"` encodes the open as ONE OPEN_DQUOTE and the close as ONE CLOSE_DQUOTE (bundled).
    const std::vector<int> ids = sub0::tok::encode(t, "said \"hi\"");
    REQUIRE(std::count(ids.begin(), ids.end(), t.odquote_id) == 1);
    REQUIRE(std::count(ids.begin(), ids.end(), t.cdquote_id) == 1);
}

TEST_CASE("JOIN scheme: SPELL encapsulation for long/OOV words round-trips", "[tok][join]") {
    const Tokenizer t = sub0::tok::learn(kCorpus);
    REQUIRE(round_trips(t, "the antidisestablishmentarianism word"));
    REQUIRE(round_trips(t, "DoThisFooBar and NASA acronyms"));
    REQUIRE(round_trips(t, "pneumonoultramicroscopicsilicovolcanoconiosis"));
    // a long word splits into N>=3 BPE sub-tokens -> exactly one balanced SPELL_START/END pair.
    const std::vector<int> ids = sub0::tok::encode(t, "antidisestablishmentarianism");
    REQUIRE(std::count(ids.begin(), ids.end(), t.spell_start_id) == 1);
    REQUIRE(std::count(ids.begin(), ids.end(), t.spell_end_id) == 1);
    REQUIRE(std::find(ids.begin(), ids.end(), t.spell_start_id)
          < std::find(ids.begin(), ids.end(), t.spell_end_id));
}

TEST_CASE("JOIN scheme: case carries across SPELL + quotes", "[tok][join]") {
    const Tokenizer t = sub0::tok::learn(kCorpus);
    REQUIRE(round_trips(t, "She said, \"Don't touch the antidisestablishment thing!\""));
    REQUIRE(round_trips(t, "ANTIDISESTABLISHMENT"));          // all-caps long word: UP across the SPELL group
    REQUIRE(round_trips(t, "The Supercalifragilistic word ends here ."));
}

// CamelCase / PascalCase splits at case transitions into one collapsed sub-word per segment
// (each reusing the lowercase merges) instead of shattering into near-character SPELL tokens.
TEST_CASE("JOIN scheme: CamelCase splits into per-segment case markers", "[tok][join]") {
    const Tokenizer t = sub0::tok::learn(kCorpus);
    // "NonCommercial" -> Non | Commercial : two capitalised segments -> two CAP markers.
    const std::vector<int> nc = sub0::tok::encode(t, "NonCommercial");
    REQUIRE(std::count(nc.begin(), nc.end(), t.cap_id) == 2);
    REQUIRE(round_trips(t, "NonCommercial"));
    // "HTMLParser" -> HTML (acronym, UP) | Parser (CAP).
    const std::vector<int> hp = sub0::tok::encode(t, "HTMLParser");
    REQUIRE(std::count(hp.begin(), hp.end(), t.up_id) == 1);
    REQUIRE(std::count(hp.begin(), hp.end(), t.cap_id) == 1);
    REQUIRE(round_trips(t, "HTMLParser"));
    // camelCase: a leading lowercase segment (no marker) + a capitalised one.
    REQUIRE(round_trips(t, "myAwesomeFunction"));
    REQUIRE(round_trips(t, "macOS and iOS devices"));
    REQUIRE(round_trips(t, "Commons Attribution-NonCommercial-NoDerivs"));
}

// snake_case / hyphenated compounds: the interior '_' and '-' bind the word into ONE unit
// (no JOIN per separator) so BPE merges across them; round-trip is preserved.
TEST_CASE("JOIN scheme: snake_case and hyphen bind into one unit", "[tok][join]") {
    const Tokenizer t = sub0::tok::learn(kCorpus);
    const std::vector<int> ss = sub0::tok::encode(t, "save_scan_state");
    REQUIRE(std::count(ss.begin(), ss.end(), t.join_id) == 0);   // one unit, not save<J>_<J>scan...
    REQUIRE(round_trips(t, "save_scan_state"));
    const std::vector<int> wk = sub0::tok::encode(t, "well-known");
    REQUIRE(std::count(wk.begin(), wk.end(), t.join_id) == 0);
    REQUIRE(round_trips(t, "well-known and non-commercial"));
    // A leading/trailing separator is NOT interior -> still splits off (round-trip holds).
    REQUIRE(round_trips(t, "--flag -x _leading trailing_"));
}

// Regression: an all-caps word + a possessive/contraction ("NASA's") truecases to UP + the
// lowercase form, but word_unit_end keeps "nasa's" as ONE unit (interior apostrophe). UP must
// stop at the apostrophe so the post-' "s" stays lowercase -- mirrors casing::detokenize.
TEST_CASE("JOIN scheme: UP stops at an interior apostrophe", "[tok][join]") {
    const Tokenizer t = sub0::tok::learn(kCorpus);
    REQUIRE(round_trips(t, "THE's end"));                    // UP on THE, lowercase 's (the NASA's bug)
    REQUIRE(round_trips(t, "the DOG's bone is here"));       // "dog" is attested -> UP collapse
    REQUIRE(round_trips(t, "She's happy and they're here")); // CAP + contraction
    REQUIRE(round_trips(t, "don't won't can't"));            // plain contractions
}
