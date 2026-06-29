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

#include <algorithm>
#include <sstream>
#include <string>
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
    const Tokenizer t = sub0::tok::learn(kCorpus, {.join_scheme = true});
    REQUIRE(t.join_scheme);
    REQUIRE(t.n_base == 265);                 // 256 bytes + CAP,UP,JOIN,NEWLINE,PARA,ODQUOTE,CDQUOTE,SPELL_START,SPELL_END
    REQUIRE(t.cap_id == 256);
    REQUIRE(t.up_id == 257);
    REQUIRE(t.join_id == 258);
    REQUIRE(t.newline_id == 259);
    REQUIRE(t.para_id == 260);
    REQUIRE(t.odquote_id == 261);
    REQUIRE(t.cdquote_id == 262);
    REQUIRE(t.spell_start_id == 263);
    REQUIRE(t.spell_end_id == 264);
    REQUIRE(t.vocab > t.n_base);              // merges still learned on top
}

TEST_CASE("JOIN scheme: a single inter-word space costs no token", "[tok][join]") {
    const Tokenizer t = sub0::tok::learn(kCorpus, {.join_scheme = true});
    const std::vector<int> ids = sub0::tok::encode(t, "the dog ran");
    REQUIRE_FALSE(ids.empty());
    for (int id : ids) REQUIRE(id != ' ');    // spaces are implicit, never a literal byte token
    REQUIRE(round_trips(t, "the dog ran"));
}

TEST_CASE("JOIN scheme: round-trips spacing, punctuation and case", "[tok][join]") {
    const Tokenizer t = sub0::tok::learn(kCorpus, {.join_scheme = true});
    REQUIRE(round_trips(t, "the quick brown fox ."));
    REQUIRE(round_trips(t, "the cat, the dog; and a fox."));   // glued punctuation -> JOIN
    REQUIRE(round_trips(t, "glued.text.no.spaces"));           // all glued
    REQUIRE(round_trips(t, "The dog ran. They were happy."));  // capitalised words
    REQUIRE(round_trips(t, "THE END of the day"));             // all-caps (UP across the word)
    REQUIRE(round_trips(t, "She said hello to the cat."));
}

TEST_CASE("JOIN scheme: round-trips whitespace variety", "[tok][join]") {
    const Tokenizer t = sub0::tok::learn(kCorpus, {.join_scheme = true});
    REQUIRE(round_trips(t, "line one\nline two"));             // single newline -> NEWLINE
    REQUIRE(round_trips(t, "para one\n\npara two"));           // blank line  -> PARA
    REQUIRE(round_trips(t, "a  b   c"));                       // multi-space -> verbatim
    REQUIRE(round_trips(t, "tab\there"));                      // tab         -> verbatim
    REQUIRE(round_trips(t, " leading and trailing "));         // edge whitespace
    REQUIRE(round_trips(t, "x\n\n\ny"));                       // 3 newlines  -> verbatim
}

TEST_CASE("JOIN scheme: complete base encodes out-of-corpus bytes", "[tok][join]") {
    const Tokenizer t = sub0::tok::learn(kCorpus, {.join_scheme = true});
    // Symbols the tiny corpus may never contain still round-trip -- the total 256-byte base
    // means no input character is silently dropped (the §1 correctness fix).
    REQUIRE(round_trips(t, "email me @ x%y/z+~"));
    REQUIRE(round_trips(t, "100% sure: a/b = c+d"));
    REQUIRE(round_trips(t, "{code}[index]"));
}

TEST_CASE("JOIN scheme: serialize/deserialize preserves the scheme + round-trip", "[tok][join]") {
    const Tokenizer t = sub0::tok::learn(kCorpus, {.join_scheme = true});
    std::ostringstream os(std::ios::binary);
    sub0::tok::serialize(t, os);
    std::istringstream is(os.str(), std::ios::binary);
    Tokenizer t2;
    REQUIRE(sub0::tok::deserialize(t2, is));
    REQUIRE(t2.join_scheme);                  // scheme inferred from the base alphabet
    REQUIRE(t2.join_id == t.join_id);
    REQUIRE(round_trips(t2, "The cat, the dog.\nLine two here."));
}

TEST_CASE("JOIN scheme: encode is deterministic", "[tok][join]") {
    const Tokenizer t = sub0::tok::learn(kCorpus, {.join_scheme = true});
    REQUIRE(sub0::tok::encode(t, "the cat, sat.") == sub0::tok::encode(t, "the cat, sat."));
}

TEST_CASE("JOIN scheme: directional double quotes round-trip + bundle spacing", "[tok][join]") {
    const Tokenizer t = sub0::tok::learn(kCorpus, {.join_scheme = true});
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
    const Tokenizer t = sub0::tok::learn(kCorpus, {.join_scheme = true});
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
    const Tokenizer t = sub0::tok::learn(kCorpus, {.join_scheme = true});
    REQUIRE(round_trips(t, "She said, \"Don't touch the antidisestablishment thing!\""));
    REQUIRE(round_trips(t, "ANTIDISESTABLISHMENT"));          // all-caps long word: UP across the SPELL group
    REQUIRE(round_trips(t, "The Supercalifragilistic word ends here ."));
}

// Regression: an all-caps word + a possessive/contraction ("NASA's") truecases to UP + the
// lowercase form, but word_unit_end keeps "nasa's" as ONE unit (interior apostrophe). UP must
// stop at the apostrophe so the post-' "s" stays lowercase -- mirrors casing::detokenize.
TEST_CASE("JOIN scheme: UP stops at an interior apostrophe", "[tok][join]") {
    const Tokenizer t = sub0::tok::learn(kCorpus, {.join_scheme = true});
    REQUIRE(round_trips(t, "THE's end"));                    // UP on THE, lowercase 's (the NASA's bug)
    REQUIRE(round_trips(t, "the DOG's bone is here"));       // "dog" is attested -> UP collapse
    REQUIRE(round_trips(t, "She's happy and they're here")); // CAP + contraction
    REQUIRE(round_trips(t, "don't won't can't"));            // plain contractions
}
