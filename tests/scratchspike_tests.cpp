// scratchspike_tests.cpp -- engine-free tests for the scratch-token (context-translation) curriculum
// (include/sub0/scratchspike.hpp). No model: validates the synthetic data's STRUCTURE -- OOV
// generation, task-trace shapes, the loss mask (define+injected content masked, ops+answer graded),
// the drilled/held-out OOV split, and that build_dataset's document index lines up with the window
// sampler's contract. Mirrors spellspike_tests.cpp.

#include <catch2/catch_test_macros.hpp>

#include "sub0/scratchspike.hpp"
#include "sub0/window.hpp"

#include <algorithm>
#include <random>
#include <span>
#include <string>
#include <unordered_set>

using sub0::tok::Tokenizer;
namespace ss  = sub0::scratchspike;
namespace cas = sub0::casing;

namespace {
const std::string kCorpus = [] {
    std::string c;
    for (int i = 0; i < 40; ++i)
        c += "the quick brown fox jumps over the lazy dog and the cat sat on the mat .\n"
             "she said hello to the whole wide world and they all went home again .\n"
             "commons attribution license non commercial share alike international here .\n"
             "save scan state load model cache please and thank you very much indeed .\n";
    return c;
}();
}  // namespace

TEST_CASE("scratchspike: gen_oov yields genuinely OOV byte strings (never a single vocab piece)", "[scratchspike]") {
    const Tokenizer t = sub0::tok::learn(kCorpus);
    std::mt19937_64 rng(1);
    for (int i = 0; i < 200; ++i) {
        const std::string o = ss::gen_oov(rng, t);
        REQUIRE((o.size() >= 3 && o.size() <= 6));
        REQUIRE(t.piece_index.find(o) == t.piece_index.end());   // not representable as one token
        for (char c : o) REQUIRE((c >= 'a' && c <= 'z'));
        // its fragments are its byte codes (all < 256 base byte tokens)
        for (int b : ss::oov_bytes(o)) REQUIRE((b >= 0 && b < 256));
    }
}

TEST_CASE("scratchspike: nth_char trace resolves a pre-bound slot; answer is the Nth byte", "[scratchspike]") {
    const Tokenizer t = sub0::tok::learn(kCorpus);
    const std::string oov = "zqwbx";
    const std::vector<int> bytes = ss::oov_bytes(oov);
    const int slot = ss::scratch_slot(0);

    for (int n = 0; n < static_cast<int>(bytes.size()); ++n) {
        const ss::Task k = ss::nth_char_task(t, oov, n);
        REQUIRE(k.answer_byte == bytes[static_cast<std::size_t>(n)]);
        REQUIRE(k.mask.size() == k.trace.size());

        // Layout: [# n <slot>] [UNCOMBINE <bytes> UNCOMBINE_END] [= answer EOS]. No OOV bytes in the
        //         stream -- the only path to the chars is resolving the slot. Checked by exact POSITION
        //         (a byte value can appear both as an injected char AND as the answer).
        std::size_t i = 0;
        REQUIRE((k.trace[i] == ss::Q_NTH && k.mask[i] == 0)); ++i;            // query: context (masked)
        REQUIRE((k.trace[i] == '0' + n && k.mask[i] == 0)); ++i;
        REQUIRE((k.trace[i] == slot && k.mask[i] == 0)); ++i;                  // reference the slot (1 token!)
        REQUIRE((k.trace[i] == cas::TOK_UNCOMBINE && k.mask[i] == 1)); ++i;    // resolve: GRADED
        for (std::size_t b = 0; b < bytes.size(); ++b, ++i) REQUIRE(k.mask[i] == 0);   // injected: masked
        REQUIRE((k.trace[i] == cas::TOK_UNCOMBINE_END && k.mask[i] == 0)); ++i;
        REQUIRE((k.trace[i] == ss::SEP && k.mask[i] == 1)); ++i;
        REQUIRE((k.trace[i] == k.answer_byte && k.mask[i] == 1)); ++i;         // the graded answer
        REQUIRE((k.trace[i] == cas::TOK_EOS && k.mask[i] == 1));

        REQUIRE(k.prompt == std::vector<int>{ss::Q_NTH, '0' + n, slot});       // fed live; model generates from UNCOMBINE on
    }
}

TEST_CASE("scratchspike: round-trip trace resolves the slot then recombines its fragments", "[scratchspike]") {
    const Tokenizer t = sub0::tok::learn(kCorpus);
    const std::string oov = "vmkpq";
    const std::vector<int> bytes = ss::oov_bytes(oov);
    const int slot = ss::scratch_slot(0);
    const ss::Task k = ss::roundtrip_task(t, oov);
    REQUIRE(k.mask.size() == k.trace.size());
    REQUIRE(k.prompt == std::vector<int>{ss::Q_RT, slot});

    // uncombine (resolve) then combine (back). The COMBINE-back leg's copied fragments are GRADED (the
    // model must reproduce the resolved bytes); the recombined slot at the end is injected (masked).
    REQUIRE(std::count(k.trace.begin(), k.trace.end(), cas::TOK_UNCOMBINE) == 1);
    REQUIRE(std::count(k.trace.begin(), k.trace.end(), cas::TOK_COMBINE) == 1);   // recombine only (no in-stream define)
    REQUIRE(k.trace[k.trace.size() - 2] == slot);   // recombine result == the same slot
    REQUIRE(k.mask[k.mask.size() - 2] == 0);        // ... which is injected (masked)
    REQUIRE(k.trace.back() == cas::TOK_EOS);

    std::size_t ce = k.trace.size();
    for (std::size_t i = k.trace.size(); i-- > 0;) if (k.trace[i] == cas::TOK_COMBINE_END) { ce = i; break; }
    REQUIRE(ce < k.trace.size());
    for (std::size_t b = 0; b < bytes.size(); ++b) REQUIRE(k.mask[ce - 1 - b] == 1);   // copied fragments trained
}

TEST_CASE("scratchspike: make_oov_split is a disjoint, deterministic partition of unique OOVs", "[scratchspike]") {
    const Tokenizer t = sub0::tok::learn(kCorpus);
    const ss::OovSplit s = ss::make_oov_split(t, 40, 0.7, 2024);
    std::unordered_set<std::string> seen;
    for (const std::string& o : s.drilled)  REQUIRE(seen.insert(o).second);   // unique across both sets
    for (const std::string& o : s.held_out) REQUIRE(seen.insert(o).second);
    REQUIRE(static_cast<int>(s.drilled.size()) == 28);   // 0.7 * 40
    REQUIRE(static_cast<int>(s.held_out.size()) == 12);
    REQUIRE(ss::make_oov_split(t, 40, 0.7, 2024).drilled == s.drilled);   // deterministic
}

TEST_CASE("scratchspike: build_dataset doc index matches the window sampler's contract", "[scratchspike]") {
    const Tokenizer t = sub0::tok::learn(kCorpus);
    const ss::OovSplit s = ss::make_oov_split(t, 20, 0.7, 7);
    ss::DatasetOptions opt; opt.tasks_per_oov = 4;
    const ss::Dataset ds = ss::build_dataset(t, s, opt);

    REQUIRE(ds.mask.size() == ds.tokens.size());
    REQUIRE(ds.doc_starts.front() == 0u);
    REQUIRE(ds.doc_starts.back() == static_cast<std::uint64_t>(ds.tokens.size()));
    for (std::size_t i = 1; i < ds.doc_starts.size(); ++i) REQUIRE(ds.doc_starts[i] > ds.doc_starts[i - 1]);
    REQUIRE(static_cast<int>(ds.doc_starts.size()) == static_cast<int>(s.drilled.size()) * opt.tasks_per_oov + 1);

    // Every window stays within one task document.
    std::mt19937 rng(3);
    for (int it = 0; it < 200; ++it) {
        const sub0::Window w = sub0::sample_window(rng, 16, ds.tokens.size(),
                                                   std::span<const std::uint64_t>(ds.doc_starts));
        const std::size_t end = w.start + static_cast<std::size_t>(w.len);
        REQUIRE(end < ds.tokens.size());
        const auto k = static_cast<std::size_t>(
            std::upper_bound(ds.doc_starts.begin(), ds.doc_starts.end(),
                             static_cast<std::uint64_t>(w.start)) - ds.doc_starts.begin()) - 1;
        REQUIRE(end < ds.doc_starts[k + 1]);
    }
}
