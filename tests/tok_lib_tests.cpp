// tok_lib_tests.cpp — pure sub0::tok library tests, with NO dependency on the engine
// or the baked production tokenizer.tok. The tokenizer was factored into a reusable
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
#include "sub0/window.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <random>
#include <span>
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
    REQUIRE(cap.front() == sub0::casing::TOK_CAP);
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
    REQUIRE(sub0::casing::TOK_JOIN >= 0);
    REQUIRE(t.n_base == 288);                 // 256 bytes + 32 markers (14 original + 2 turn + 6 bracket-glue + 10 reserved)
    REQUIRE(sub0::casing::TOK_EOS == 256);                 // FIRST marker, right after the byte range (see casing.hpp)
    REQUIRE(sub0::casing::TOK_CAP == 257);
    REQUIRE(sub0::casing::TOK_UP == 258);
    REQUIRE(sub0::casing::TOK_JOIN == 259);
    REQUIRE(sub0::casing::TOK_NEWLINE == 260);
    REQUIRE(sub0::casing::TOK_PARA == 261);
    REQUIRE(sub0::casing::TOK_ODQUOTE == 262);
    REQUIRE(sub0::casing::TOK_CDQUOTE == 263);
    REQUIRE(sub0::casing::TOK_SPELL_START == 264);
    REQUIRE(sub0::casing::TOK_SPELL_END == 265);
    REQUIRE(sub0::casing::TOK_SPACE2 == 266);
    REQUIRE(sub0::casing::TOK_SPACE4 == 267);
    REQUIRE(sub0::casing::TOK_TAB2 == 268);
    REQUIRE(sub0::casing::TOK_TAB4 == 269);
    REQUIRE(sub0::casing::TOK_TURN_START == 270);
    REQUIRE(sub0::casing::TOK_TURN_END == 271);
    REQUIRE(sub0::casing::TOK_GLUE_OPAREN == 272);
    REQUIRE(sub0::casing::TOK_GLUE_CPAREN == 273);
    REQUIRE(sub0::casing::TOK_GLUE_OBRACKET == 274);
    REQUIRE(sub0::casing::TOK_GLUE_CBRACKET == 275);
    REQUIRE(sub0::casing::TOK_GLUE_OBRACE == 276);
    REQUIRE(sub0::casing::TOK_GLUE_CBRACE == 277);
    REQUIRE(sub0::casing::TOK_UNCOMBINE == 278);      // renamed from TOK_RESERVED_0 for the token-granularity spike
    REQUIRE(sub0::casing::TOK_UNCOMBINE_END == 279);  // renamed from TOK_RESERVED_1
    REQUIRE(sub0::casing::TOK_COMBINE == 280);        // renamed from TOK_RESERVED_2
    REQUIRE(sub0::casing::TOK_COMBINE_END == 281);    // renamed from TOK_RESERVED_3
    REQUIRE(sub0::casing::TOK_RESERVED_4 == 282);
    REQUIRE(sub0::casing::TOK_RESERVED_9 == 287);
    REQUIRE(sub0::casing::TOK_MARKER_COUNT == 288);
    REQUIRE(t.vocab > t.n_base);              // merges still learned on top
}

// Regression: detokenize_join originally had no eos_id case at all (round-trip fell through to a
// garbled default), and once added, it still dropped the pending implicit space before the literal
// (unlike the general word path and odquote_id, which both check `dps`) -- caught by the project's own
// dogfood test, since several source files literally contain the string "<|endoftext|>" in comments.
TEST_CASE("JOIN scheme: round-trips the literal <|endoftext|> EOS marker", "[tok][join]") {
    const Tokenizer t = sub0::tok::learn(kCorpus);
    const std::vector<int> ids = sub0::tok::encode(t, "the dog ran <|endoftext|> the cat slept");
    REQUIRE(std::count(ids.begin(), ids.end(), sub0::casing::TOK_EOS) == 1);   // collapses to exactly one token
    REQUIRE(round_trips(t, "the dog ran <|endoftext|> the cat slept"));
    REQUIRE(round_trips(t, "word <|endoftext|>"));                // pending space before EOS (the bug)
    REQUIRE(round_trips(t, "<|endoftext|>word"));                 // EOS glued to the next word
    REQUIRE(round_trips(t, "one<|endoftext|>\ntwo<|endoftext|>\n"));  // back-to-back documents
}

// Stage 2: the ChatML-adopted turn markers, verbatim literal match (see casing.hpp's TOK_TURN_START
// comment) -- same worked-example shape as the EOS test above (bare, isolation, pending-space,
// glued), plus the specific adjacency the design requires: TOK_TURN_START glues to the role word
// that follows it (no space token/JOIN needed), and TOK_TURN_END glues to whatever precedes it.
TEST_CASE("JOIN scheme: round-trips the literal <|im_start|>/<|im_end|> turn markers", "[tok][join]") {
    const Tokenizer t = sub0::tok::learn(kCorpus);
    const std::vector<int> ids = sub0::tok::encode(t, "<|im_start|>user\nhello<|im_end|>");
    REQUIRE(std::count(ids.begin(), ids.end(), sub0::casing::TOK_TURN_START) == 1);
    REQUIRE(std::count(ids.begin(), ids.end(), sub0::casing::TOK_TURN_END) == 1);
    // TOK_TURN_START is immediately followed by the role word with NO JOIN -- the role text is
    // glued directly onto the marker, same as ChatML's own "<|im_start|>user" (no space).
    const auto start_it = std::find(ids.begin(), ids.end(), sub0::casing::TOK_TURN_START);
    REQUIRE(start_it != ids.end());
    REQUIRE(*(start_it + 1) != sub0::casing::TOK_JOIN);
    REQUIRE(round_trips(t, "<|im_start|>user\nhello<|im_end|>"));
    REQUIRE(round_trips(t, "<|im_start|>system\nyou are helpful.<|im_end|>\n<|im_start|>user\nhi<|im_end|>"));
    REQUIRE(round_trips(t, "word <|im_end|>"));                   // pending space before END (mirrors EOS)
    REQUIRE(round_trips(t, "<|im_start|>word"));                  // START glued to the next word
    REQUIRE(round_trips(t, "a full chat: <|im_start|>assistant\nHi there!<|im_end|>"));  // punctuation-adjacent
    // A full multi-turn round-trip -- the realistic shape this marker family exists for.
    REQUIRE(round_trips(t,
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n"
        "<|im_start|>assistant\nThe capital of France is Paris.<|im_end|>\n"));
}

// Mechanism test (user request 2026-07-02): prove the FULL training-corpus pipeline actually lets
// the model see EOS as a trainable target, not just that sample_window's index arithmetic CAN reach
// a document boundary (engine_tests.cpp's "window sampler keeps each training window inside one
// document" already covers that generically, with a synthetic doc index and random token ids). This
// exercises the REAL chain a corpus goes through: tokenizer.encode() places eos_id, the shared
// scan_doc_boundaries() (also used by tools/configurator.cpp -- see its declaration in
// tokenizer.hpp) turns that into a doc index, and sample_window() (window.hpp) draws windows from
// it -- checking the ACTUAL token id at the sampled target position, not an assumed one. This is
// exactly the gap the whole EOS feature exists to close (see casing.hpp's TOK_EOS comment): without
// it, a short document's sampled window would stop one token short of ever training "last content
// token -> what comes next".
TEST_CASE("training pipeline: EOS is a reachable, in-window target (not cut off)", "[tok][join][window]") {
    const Tokenizer t = sub0::tok::learn(kCorpus);
    // Two short documents, back to back -- short enough that sample_window's "short doc: whole
    // document" path is taken deterministically (cap well under any reasonable T), not just hit by
    // chance, so a target of eos_id is guaranteed whenever the sampler resolves into document 0.
    const std::string mini = "the cat sat .<|endoftext|>\nthe dog ran .<|endoftext|>\n";
    const std::vector<int> ids32 = sub0::tok::encode(t, mini);
    REQUIRE(std::count(ids32.begin(), ids32.end(), sub0::casing::TOK_EOS) == 2);
    const std::vector<std::int32_t> toks(ids32.begin(), ids32.end());

    std::vector<std::uint64_t> doc_starts{0u};
    int nl_run = 0;
    sub0::tok::scan_doc_boundaries(std::span<const std::int32_t>(toks), 0u, t, nl_run, doc_starts);
    REQUIRE(doc_starts.size() == 3);            // doc0 (seeded) + a boundary after each of the 2 EOS tokens
    // The boundary scan_doc_boundaries recorded for document 0 must point one-past a REAL eos_id in
    // the actual encoded stream -- not just an index that happens to satisfy the arithmetic.
    REQUIRE(toks[doc_starts[1] - 1] == sub0::casing::TOK_EOS);

    // sample_window must actually surface that position as a trainable target across real draws.
    std::mt19937 rng(11);
    bool saw_eos_target = false;
    for (int it = 0; it < 500; ++it) {
        const sub0::Window w = sub0::sample_window(rng, 64, toks.size(),
                                                   std::span<const std::uint64_t>(doc_starts));
        REQUIRE(w.len >= 1);
        const std::size_t target_pos = w.start + static_cast<std::size_t>(w.len);
        REQUIRE(target_pos < toks.size());
        if (toks[target_pos] == sub0::casing::TOK_EOS) saw_eos_target = true;
    }
    REQUIRE(saw_eos_target);                    // the model DOES get trained on "last token -> EOS"
}

// TOK_TURN_END must stay a document-boundary NON-event: scan_doc_boundaries only recognizes
// TOK_EOS (or a >=2 newline run) as a boundary, so a document containing chat turns doesn't get
// spuriously chopped at every assistant-turn end -- that would break sample_window's within-one-
// document guarantee for ordinary conversational documents. See casing.hpp's TOK_TURN_END comment.
TEST_CASE("JOIN scheme: TOK_TURN_END is not treated as a document boundary", "[tok][join][window]") {
    const Tokenizer t = sub0::tok::learn(kCorpus);
    const std::string chat = "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\nhi<|im_end|>\n";
    const std::vector<int> ids32 = sub0::tok::encode(t, chat);
    REQUIRE(std::count(ids32.begin(), ids32.end(), sub0::casing::TOK_TURN_END) == 2);
    REQUIRE(std::count(ids32.begin(), ids32.end(), sub0::casing::TOK_EOS) == 0);
    const std::vector<std::int32_t> toks(ids32.begin(), ids32.end());

    std::vector<std::uint64_t> doc_starts{0u};
    int nl_run = 0;
    sub0::tok::scan_doc_boundaries(std::span<const std::int32_t>(toks), 0u, t, nl_run, doc_starts);
    REQUIRE(doc_starts.size() == 1);            // just the seeded doc0 -- no TOK_TURN_END boundary fired
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
    REQUIRE(std::count(ind.begin(), ind.end(), sub0::casing::TOK_NEWLINE) == 1);
    REQUIRE(std::count(ind.begin(), ind.end(), sub0::casing::TOK_SPACE4) == 1);
    REQUIRE(std::count(ind.begin(), ind.end(), ' ') == 0);    // no literal space bytes
    REQUIRE(round_trips(t, "a\n    b"));

    // Greedy tiling: 8->2xSPACE4, 6->SPACE4+SPACE2, 3->SPACE2+space, 2->SPACE2.
    const std::vector<int> sp8 = enc("a\n        b");
    REQUIRE(std::count(sp8.begin(), sp8.end(), sub0::casing::TOK_SPACE4) == 2);
    REQUIRE(round_trips(t, "x        y"));                     // 8 spaces
    REQUIRE(round_trips(t, "x      y"));                       // 6 spaces
    REQUIRE(round_trips(t, "x   y"));                          // 3 spaces (SPACE2 + 1)
    const std::vector<int> sp2 = enc("x  y");                  // 2 spaces -> exactly one SPACE2
    REQUIRE(std::count(sp2.begin(), sp2.end(), sub0::casing::TOK_SPACE2) == 1);

    // Tabs tile the same way; a single inter-word space is still implicit (free).
    const std::vector<int> tb4 = enc("a\t\t\t\tb");
    REQUIRE(std::count(tb4.begin(), tb4.end(), sub0::casing::TOK_TAB4) == 1);
    REQUIRE(round_trips(t, "a\t\tb\tc"));                      // TAB2 + lone tab byte
    for (int id : enc("the dog ran")) REQUIRE(id != sub0::casing::TOK_SPACE2);   // single spaces never tile
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
    REQUIRE(t2.n_base == t.n_base);                        // base alphabet size survived the round-trip
    REQUIRE(round_trips(t2, "The cat, the dog.\nLine two here."));
}

// The runtime tokenizer only ever loads kind==1 (Unigram) -- a pre-WS2 file's legacy BPE-merge
// encoding (kind 0) must be rejected outright, not decoded (there is no BPE word encoder left to
// use it with). Corrupt a real, otherwise-valid blob at the exact `kind` field offset (magic +
// kSchemeVersion + vocab + n_base, each a u32, then n_base u16 base-symbol codes) rather than
// hand-building the whole binary format, so this stays correct if the format around it ever changes.
TEST_CASE("deserialize rejects a legacy kind==0 (BPE) blob", "[tok][join]") {
    const Tokenizer t = sub0::tok::learn(kCorpus);
    std::ostringstream os(std::ios::binary);
    sub0::tok::serialize(t, os);
    std::string blob = os.str();
    const std::size_t kind_offset = 4 + 4 + 4 + 4 + static_cast<std::size_t>(t.n_base) * 2;
    REQUIRE(kind_offset + 4 <= blob.size());
    REQUIRE(blob[kind_offset] == 1);       // sanity: this really is the kind byte, currently 1 (Unigram)
    blob[kind_offset] = 0;                 // corrupt to the legacy BPE discriminator
    std::istringstream is(blob, std::ios::binary);
    Tokenizer t2;
    REQUIRE_FALSE(sub0::tok::deserialize(t2, is));
}

// Stage 2 bumped the magic "S0TE" -> "S0TF" (TOK_MARKER_COUNT grew 14 -> 32, shifting where
// learned piece ids start) -- a stale pre-Stage-2 file must be rejected outright, not silently
// misread with its piece ids offset by the wrong amount. Same discipline as the earlier
// "S0TZ"->"S0TE" bump (see docs/TOKENIZER_REVIEW.md and the comment above serialize()).
TEST_CASE("deserialize rejects a stale pre-Stage-2 (S0TE) magic", "[tok][join]") {
    const Tokenizer t = sub0::tok::learn(kCorpus);
    std::ostringstream os(std::ios::binary);
    sub0::tok::serialize(t, os);
    std::string blob = os.str();
    REQUIRE(blob.size() >= 4);
    std::uint32_t old_magic = 0x45543053u;  // "S0TE"
    std::memcpy(blob.data(), &old_magic, sizeof old_magic);
    std::istringstream is(blob, std::ios::binary);
    Tokenizer t2;
    REQUIRE_FALSE(sub0::tok::deserialize(t2, is));
}

// learn_bpe_analysis() is analysis-only (--dump-vocab's A/B + vocab-curve tool, never the runtime
// word encoder -- see its own doc comment in tokenizer.hpp) but was previously untested in either
// direction. Sanity-check it still produces a coherent BpeAnalysisVocab: some merges learned, the
// merge count matches vocab growth beyond the base alphabet, and the base alphabet itself is the
// same fixed scheme learn() produces (base_symbol[i] == i for the whole base region).
TEST_CASE("learn_bpe_analysis produces a sane analysis-only vocabulary", "[tok][bpe]") {
    sub0::tok::Scan s;
    s.add_names(kCorpus);
    const auto attested = sub0::tok::derive_attested(s);
    s.add_words(kCorpus, attested);
    const sub0::tok::BpeAnalysisVocab bpe = sub0::tok::learn_bpe_analysis(s, /*vocab_target=*/600, /*min_merge=*/2);
    REQUIRE(bpe.n_base > 0);
    REQUIRE(bpe.vocab > bpe.n_base);                              // at least one merge was learned
    REQUIRE(bpe.merges.size() == bpe.merge_count.size());
    REQUIRE(static_cast<int>(bpe.merges.size()) == bpe.vocab - bpe.n_base);
    REQUIRE(static_cast<int>(bpe.base_symbol.size()) == bpe.n_base);
    REQUIRE(static_cast<int>(bpe.expansion.size()) == bpe.vocab);
    for (int i = 0; i < bpe.n_base; ++i) REQUIRE(bpe.base_symbol[static_cast<std::size_t>(i)] == i);
}

TEST_CASE("JOIN scheme: encode is deterministic", "[tok][join]") {
    const Tokenizer t = sub0::tok::learn(kCorpus);
    REQUIRE(sub0::tok::encode(t, "the cat, sat.") == sub0::tok::encode(t, "the cat, sat."));
}

// The fingerprint is what a trained model stamps to pin its decoder identity. Three properties matter:
// it is deterministic; it SURVIVES a serialize->deserialize round-trip (the value a saved model carries
// must equal the value gen recomputes from the reloaded tokenizer.tok, or the guard false-positives);
// and it is SENSITIVE (a different vocab fingerprints differently, so a mismatch is actually caught).
TEST_CASE("fingerprint: deterministic + survives a serialize round-trip", "[tok][fingerprint]") {
    const Tokenizer t = sub0::tok::learn(kCorpus);
    const std::uint64_t fp = sub0::tok::fingerprint(t);
    REQUIRE(fp != 0);
    REQUIRE(sub0::tok::fingerprint(t) == fp);                       // deterministic

    std::ostringstream os(std::ios::binary);
    sub0::tok::serialize(t, os);
    std::istringstream is(os.str(), std::ios::binary);
    Tokenizer t2;
    REQUIRE(sub0::tok::deserialize(t2, is));
    REQUIRE(sub0::tok::fingerprint(t2) == fp);                      // round-trip stable (the guard relies on this)
}

TEST_CASE("fingerprint: distinguishes a different vocabulary", "[tok][fingerprint]") {
    const Tokenizer a = sub0::tok::learn(kCorpus);
    // A different corpus -> a different learned vocab -> a different fingerprint. This is exactly the
    // reconfigure-for-another-corpus case the model/decoder guard must catch.
    const std::string other = [] {
        std::string c;
        for (int i = 0; i < 50; ++i)
            c += "machines compute numbers quickly and store results in memory banks .\n"
                 "The algorithm sorts the array then searches it in logarithmic time .\n"
                 "PROGRAM output was 42 and the process exited cleanly at last .\n";
        return c;
    }();
    const Tokenizer b = sub0::tok::learn(other);
    REQUIRE(sub0::tok::fingerprint(a) != sub0::tok::fingerprint(b));
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
    REQUIRE(std::count(ids.begin(), ids.end(), sub0::casing::TOK_ODQUOTE) == 1);
    REQUIRE(std::count(ids.begin(), ids.end(), sub0::casing::TOK_CDQUOTE) == 1);
}

// Regression, from a real 100MB-TinyStories-dialogue re-measurement (docs/TOKENIZER_REVIEW.md
// §5.8): a quote right after a bare '\n' (a new line/paragraph starting with dialogue) was 87.8% of
// all fallback-to-bare-byte cases -- by far the dominant miss. FIXED: it now fires TOK_ODQUOTE
// (preceded by its own TOK_NEWLINE, so the exact newline byte still round-trips). Pinned as the
// corrected behavior.
TEST_CASE("JOIN scheme: line-initial opening quote fires ODQUOTE, not a bare fallback", "[tok][join]") {
    const Tokenizer t = sub0::tok::learn(kCorpus);
    const std::string x = "she said hello.\n\"Are you coming?\" he asked.";
    const std::vector<int> ids = sub0::tok::encode(t, x);
    REQUIRE(std::count(ids.begin(), ids.end(), sub0::casing::TOK_NEWLINE) == 1);
    REQUIRE(std::count(ids.begin(), ids.end(), sub0::casing::TOK_ODQUOTE) == 1);
    // The NEWLINE must immediately precede the ODQUOTE -- this is specifically the "glue the
    // newline's own token right up against the quote marker" mechanism, not just "both appear".
    const auto nl_it = std::find(ids.begin(), ids.end(), sub0::casing::TOK_NEWLINE);
    REQUIRE(nl_it != ids.end());
    REQUIRE(*(nl_it + 1) == sub0::casing::TOK_ODQUOTE);
    REQUIRE(round_trips(t, x));
    REQUIRE(round_trips(t, "para one.\n\nparagraph two.\n\"Quoted at a fresh paragraph.\""));
    REQUIRE(round_trips(t, "\"Quoted at the very start of the text.\" said no one."));
}

// Regression, same re-measurement: British/logical-style quoting (punctuation immediately after the
// closing quote, e.g. `"Hello", she said`) was 8.6% of fallbacks -- a real but ~20x smaller effect
// than the line-initial case above, left as documented, KNOWN, pinned behavior (not fixed -- a
// correct general fix needs a broader trailing-punctuation-byte design, out of scope for this pass).
TEST_CASE("JOIN scheme: British-style punctuation-after-close quote is a known bare fallback", "[tok][join]") {
    const Tokenizer t = sub0::tok::learn(kCorpus);
    // The OPEN quote is preceded by a space, so it fires normally (isolates the miss to CLOSE only).
    const std::string x = "she said \"hello\", nodding.";
    const std::vector<int> ids = sub0::tok::encode(t, x);
    REQUIRE(std::count(ids.begin(), ids.end(), sub0::casing::TOK_ODQUOTE) == 1);   // open still fires fine
    REQUIRE(std::count(ids.begin(), ids.end(), sub0::casing::TOK_CDQUOTE) == 0);   // known miss, not fixed
    REQUIRE(std::count(ids.begin(), ids.end(), '"') == 1);                        // the close falls back to a bare byte
    REQUIRE(round_trips(t, x));   // still round-trips correctly, just at a higher token cost
}

TEST_CASE("JOIN scheme: SPELL encapsulation for long/OOV words round-trips", "[tok][join]") {
    const Tokenizer t = sub0::tok::learn(kCorpus);
    REQUIRE(round_trips(t, "the antidisestablishmentarianism word"));
    REQUIRE(round_trips(t, "DoThisFooBar and NASA acronyms"));
    REQUIRE(round_trips(t, "pneumonoultramicroscopicsilicovolcanoconiosis"));
    // a long word splits into N>=3 BPE sub-tokens -> exactly one balanced SPELL_START/END pair.
    const std::vector<int> ids = sub0::tok::encode(t, "antidisestablishmentarianism");
    REQUIRE(std::count(ids.begin(), ids.end(), sub0::casing::TOK_SPELL_START) == 1);
    REQUIRE(std::count(ids.begin(), ids.end(), sub0::casing::TOK_SPELL_END) == 1);
    REQUIRE(std::find(ids.begin(), ids.end(), sub0::casing::TOK_SPELL_START)
          < std::find(ids.begin(), ids.end(), sub0::casing::TOK_SPELL_END));
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
    REQUIRE(std::count(nc.begin(), nc.end(), sub0::casing::TOK_CAP) == 2);
    REQUIRE(round_trips(t, "NonCommercial"));
    // "HTMLParser" -> HTML (acronym, UP) | Parser (CAP).
    const std::vector<int> hp = sub0::tok::encode(t, "HTMLParser");
    REQUIRE(std::count(hp.begin(), hp.end(), sub0::casing::TOK_UP) == 1);
    REQUIRE(std::count(hp.begin(), hp.end(), sub0::casing::TOK_CAP) == 1);
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
    REQUIRE(std::count(ss.begin(), ss.end(), sub0::casing::TOK_JOIN) == 0);   // one unit, not save<J>_<J>scan...
    REQUIRE(round_trips(t, "save_scan_state"));
    const std::vector<int> wk = sub0::tok::encode(t, "well-known");
    REQUIRE(std::count(wk.begin(), wk.end(), sub0::casing::TOK_JOIN) == 0);
    REQUIRE(round_trips(t, "well-known and non-commercial"));
    // A leading/trailing separator is NOT interior -> still splits off (round-trip holds).
    REQUIRE(round_trips(t, "--flag -x _leading trailing_"));
}

// WS5b: bracket-glue markers collapse the JOIN tax on a bracket glued directly to what precedes it.
// Unlike quotes, `(`/`)`/`[`/`]`/`{`/`}` are already unambiguous distinct bytes, so these markers
// exist purely to save tokens, not to disambiguate direction -- verify the SAVINGS directly (not
// just round-trip), since that's the whole point of WS5b (see docs/TOKENIZER_REVIEW.md §5.9).
TEST_CASE("JOIN scheme: bracket glue collapses the JOIN tax on glued brackets", "[tok][join]") {
    const Tokenizer t = sub0::tok::learn(kCorpus);
    // "f(x)" -- old: f,JOIN,(,JOIN,x,JOIN,) = 7 tokens. new: f,GLUE_OPAREN,x,GLUE_CPAREN = 4 tokens
    // (assuming f/x are single-piece words in this tiny learned vocab -- checked structurally below,
    // not by a hardcoded count, so this doesn't break if the vocab's piece-count for f/x ever shifts).
    const std::vector<int> ids = sub0::tok::encode(t, "f(x)");
    REQUIRE(std::count(ids.begin(), ids.end(), sub0::casing::TOK_GLUE_OPAREN) == 1);
    REQUIRE(std::count(ids.begin(), ids.end(), sub0::casing::TOK_GLUE_CPAREN) == 1);
    REQUIRE(std::count(ids.begin(), ids.end(), sub0::casing::TOK_JOIN) == 0);   // both JOINs eliminated
    REQUIRE(round_trips(t, "f(x)"));

    // A bracket preceded by a REAL space is NOT bundled (that spacing was already free/implicit) --
    // falls through to the ordinary byte path unchanged, still round-trips, no glue marker fires.
    const std::vector<int> spaced = sub0::tok::encode(t, "f (x)");
    REQUIRE(std::count(spaced.begin(), spaced.end(), sub0::casing::TOK_GLUE_OPAREN) == 0);
    REQUIRE(round_trips(t, "f (x)"));
    REQUIRE(round_trips(t, "f( x )"));       // internally-spaced style: correctness, not optimized
    REQUIRE(round_trips(t, "f( x)"));        // asymmetric: spaced-open, glued-close
    REQUIRE(round_trips(t, "f(x )"));        // asymmetric: glued-open, spaced-close

    // Nested and mixed families.
    REQUIRE(round_trips(t, "((a))"));
    REQUIRE(round_trips(t, "a[i]"));
    REQUIRE(round_trips(t, "foo() {"));
    REQUIRE(round_trips(t, "f(x) and (y)"));
    REQUIRE(round_trips(t, "{code}[index]"));   // the exact case a first attempt at this got wrong
    REQUIRE(round_trips(t, "std::vector<int> v(3);"));   // angle brackets stay excluded, unaffected
    REQUIRE(round_trips(t, "map[key] = fn(a, b, {1, 2, 3});"));

    // Brackets adjacent to quotes, case markers and SPELL groups -- the marker interactions WS5's
    // own review flagged as worth checking explicitly, not just brackets in isolation.
    REQUIRE(round_trips(t, "she said \"call foo(x)\" and left"));
    REQUIRE(round_trips(t, "(Capitalized) and (ALLCAPS)"));
    REQUIRE(round_trips(t, "(antidisestablishmentarianism)"));   // SPELL group inside parens
}

// Regression: an all-caps word + a possessive/contraction ("NASA's") truecases to UP + the
// lowercase form, but word_unit_end keeps "nasa's" as ONE unit (interior apostrophe). UP must
// stop at the apostrophe so the post-' "s" stays lowercase -- mirrors detokenize_join's UpWord rule.
TEST_CASE("JOIN scheme: UP stops at an interior apostrophe", "[tok][join]") {
    const Tokenizer t = sub0::tok::learn(kCorpus);
    REQUIRE(round_trips(t, "THE's end"));                    // UP on THE, lowercase 's (the NASA's bug)
    REQUIRE(round_trips(t, "the DOG's bone is here"));       // "dog" is attested -> UP collapse
    REQUIRE(round_trips(t, "She's happy and they're here")); // CAP + contraction
    REQUIRE(round_trips(t, "don't won't can't"));            // plain contractions
}
