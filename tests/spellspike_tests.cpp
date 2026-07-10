// spellspike_tests.cpp -- engine-free tests for the combine/uncombine logic curriculum
// (include/sub0/spellspike.hpp). Validates the synthetic training data's STRUCTURE with no model:
// task-trace shapes, that answers match the tokenizer's own expansion, the drilled/held-out split,
// and that build_dataset's document index lines up with the window sampler's contract.

#include <catch2/catch_test_macros.hpp>

#include "sub0/spellspike.hpp"
#include "sub0/window.hpp"

#include <algorithm>
#include <random>
#include <string>
#include <unordered_set>

using sub0::tok::Tokenizer;
namespace ss = sub0::spellspike;
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

// first task word (a learned piece of length 2..9)
int first_task_word(const Tokenizer& t) {
    for (int id = t.n_base; id < t.vocab; ++id) if (ss::is_task_word(t, id)) return id;
    return -1;
}
}  // namespace

TEST_CASE("spellspike: is_task_word selects short learned word pieces only", "[spellspike]") {
    const Tokenizer t = sub0::tok::learn(kCorpus);
    for (int id = 0; id < t.n_base; ++id) REQUIRE_FALSE(ss::is_task_word(t, id));   // bytes+markers: no
    int found = 0;
    for (int id = t.n_base; id < t.vocab; ++id)
        if (ss::is_task_word(t, id)) {
            const auto len = t.expansion[static_cast<std::size_t>(id)].size();
            REQUIRE((len >= 2 && len <= 9));
            ++found;
        }
    REQUIRE(found > 0);
}

TEST_CASE("spellspike: nth_char task trace + answer match the tokenizer's expansion", "[spellspike]") {
    const Tokenizer t = sub0::tok::learn(kCorpus);
    const int w = first_task_word(t);
    REQUIRE(w >= 0);
    const std::vector<int> bytes = t.expansion[static_cast<std::size_t>(w)];
    for (int n = 0; n < static_cast<int>(bytes.size()); ++n) {
        const ss::Task k = ss::nth_char_task(t, w, n);
        // prompt: [#, '0'+n, word]
        REQUIRE(k.prompt == std::vector<int>{ss::Q_NTH, '0' + n, w});
        // answer is the nth byte
        REQUIRE(k.answer_byte == bytes[static_cast<std::size_t>(n)]);
        // trace shape: prompt, UNCOMBINE, <bytes>, UNCOMBINE_END, SEP, answer, EOS
        REQUIRE(k.trace[3] == cas::TOK_UNCOMBINE);
        const std::vector<int> injected(k.trace.begin() + 4,
                                        k.trace.begin() + 4 + static_cast<std::ptrdiff_t>(bytes.size()));
        REQUIRE(injected == bytes);
        REQUIRE(k.trace[4 + static_cast<int>(bytes.size())] == cas::TOK_UNCOMBINE_END);
        REQUIRE(k.trace[k.trace.size() - 3] == ss::SEP);
        REQUIRE(k.trace[k.trace.size() - 2] == k.answer_byte);
        REQUIRE(k.trace.back() == cas::TOK_EOS);
    }
}

TEST_CASE("spellspike: count task answer equals the true character count (digit)", "[spellspike]") {
    const Tokenizer t = sub0::tok::learn(kCorpus);
    const int w = first_task_word(t);
    const std::string txt = ss::word_text(t, w);
    // count of the word's own first character must be >= 1 and match.
    const int x = static_cast<unsigned char>(txt[0]);
    const int truth = static_cast<int>(std::count(txt.begin(), txt.end(), static_cast<char>(x)));
    const ss::Task k = ss::count_task(t, w, x);
    REQUIRE(k.prompt == std::vector<int>{ss::Q_CNT, x, w});
    REQUIRE(k.answer_byte == '0' + std::min(truth, 9));
    // a character not in the word counts to 0.
    const ss::Task k0 = ss::count_task(t, w, '#');   // '#' won't appear inside a word
    REQUIRE(k0.answer_byte == '0');
}

TEST_CASE("spellspike: round-trip trace uncombines then combines back to the same word", "[spellspike]") {
    const Tokenizer t = sub0::tok::learn(kCorpus);
    const int w = first_task_word(t);
    const std::vector<int> bytes = t.expansion[static_cast<std::size_t>(w)];
    const ss::Task k = ss::roundtrip_task(t, w);
    REQUIRE(k.prompt == std::vector<int>{ss::Q_RT, w});
    // ~, word, UNCOMBINE, <bytes>, UNCOMBINE_END, COMBINE, <bytes>, COMBINE_END, word, EOS
    REQUIRE(k.trace[0] == ss::Q_RT);
    REQUIRE(k.trace[1] == w);
    REQUIRE(k.trace[2] == cas::TOK_UNCOMBINE);
    REQUIRE(k.trace[k.trace.size() - 2] == w);       // combine result == original word
    REQUIRE(k.trace.back() == cas::TOK_EOS);
    REQUIRE(std::count(k.trace.begin(), k.trace.end(), cas::TOK_COMBINE) == 1);
    REQUIRE(std::count(k.trace.begin(), k.trace.end(), cas::TOK_COMBINE_END) == 1);
}

TEST_CASE("spellspike: the loss mask grades the ops+answer, never the harness-injected content", "[spellspike]") {
    const Tokenizer t = sub0::tok::learn(kCorpus);
    const int w = first_task_word(t);
    const std::size_t len = t.expansion[static_cast<std::size_t>(w)].size();

    // nth-char layout: [Q, n, word, UNCOMBINE, <len bytes>, UNCOMBINE_END, SEP, answer, EOS].
    // Checked by exact POSITION (not token value -- a byte value can legitimately appear both as an
    // injected char AND as the answer): prompt + injected span masked (0), ops + answer trained (1).
    {
        const ss::Task k = ss::nth_char_task(t, w, 0);
        REQUIRE(k.mask.size() == k.trace.size());
        REQUIRE(k.trace.size() == 3 + 1 + len + 1 + 3);
        std::size_t i = 0;
        REQUIRE((k.mask[i] == 0 && k.mask[i + 1] == 0 && k.mask[i + 2] == 0)); i += 3;   // prompt
        REQUIRE((k.trace[i] == cas::TOK_UNCOMBINE && k.mask[i] == 1)); ++i;               // request: trained
        for (std::size_t b = 0; b < len; ++b, ++i) REQUIRE(k.mask[i] == 0);               // injected bytes: masked
        REQUIRE((k.trace[i] == cas::TOK_UNCOMBINE_END && k.mask[i] == 0)); ++i;           // injected end: masked
        REQUIRE((k.trace[i] == ss::SEP && k.mask[i] == 1)); ++i;                          // trained
        REQUIRE((k.trace[i] == k.answer_byte && k.mask[i] == 1)); ++i;                    // the graded answer
        REQUIRE((k.trace[i] == cas::TOK_EOS && k.mask[i] == 1));                          // trained
    }
    // round-trip layout: [~, word, UNCOMBINE, <bytes>, UNCOMBINE_END, COMBINE, <copied bytes>,
    // COMBINE_END, word, EOS]. Injected uncombine span + combine-result masked (0); the COPIED
    // fragments inside the combine region are trained (1).
    {
        const ss::Task k = ss::roundtrip_task(t, w);
        REQUIRE(k.mask.size() == k.trace.size());
        std::size_t i = 0;
        REQUIRE((k.mask[i] == 0 && k.mask[i + 1] == 0)); i += 2;                          // prompt
        REQUIRE((k.trace[i] == cas::TOK_UNCOMBINE && k.mask[i] == 1)); ++i;
        for (std::size_t b = 0; b < len; ++b, ++i) REQUIRE(k.mask[i] == 0);               // injected bytes: masked
        REQUIRE((k.trace[i] == cas::TOK_UNCOMBINE_END && k.mask[i] == 0)); ++i;
        REQUIRE((k.trace[i] == cas::TOK_COMBINE && k.mask[i] == 1)); ++i;
        for (std::size_t b = 0; b < len; ++b, ++i) REQUIRE(k.mask[i] == 1);               // copied fragments: TRAINED
        REQUIRE((k.trace[i] == cas::TOK_COMBINE_END && k.mask[i] == 1)); ++i;
        REQUIRE((k.trace[i] == w && k.mask[i] == 0)); ++i;                                // combine result: masked
        REQUIRE((k.trace[i] == cas::TOK_EOS && k.mask[i] == 1));
    }
}

TEST_CASE("spellspike: split_task_words is a disjoint, exhaustive partition of the task words", "[spellspike]") {
    const Tokenizer t = sub0::tok::learn(kCorpus);
    const ss::WordSplit s = ss::split_task_words(t, 0.7, 42);
    std::unordered_set<int> seen;
    int total = 0;
    for (int id = t.n_base; id < t.vocab; ++id) if (ss::is_task_word(t, id)) ++total;
    for (int id : s.drilled)  { REQUIRE(ss::is_task_word(t, id)); REQUIRE(seen.insert(id).second); }
    for (int id : s.held_out) { REQUIRE(ss::is_task_word(t, id)); REQUIRE(seen.insert(id).second); }
    REQUIRE(static_cast<int>(seen.size()) == total);
    REQUIRE_FALSE(s.drilled.empty());
    REQUIRE_FALSE(s.held_out.empty());
    REQUIRE(ss::split_task_words(t, 0.7, 42).drilled == s.drilled);   // deterministic
}

TEST_CASE("spellspike: build_dataset doc index matches the window sampler's contract", "[spellspike]") {
    const Tokenizer t = sub0::tok::learn(kCorpus);
    const ss::WordSplit s = ss::split_task_words(t, 0.7, 42);
    ss::DatasetOptions opt; opt.tasks_per_word = 4;
    const ss::Dataset ds = ss::build_dataset(t, s, opt);

    REQUIRE(ds.mask.size() == ds.tokens.size());   // the parallel loss mask stays aligned
    REQUIRE(ds.doc_starts.front() == 0u);
    REQUIRE(ds.doc_starts.back() == static_cast<std::uint64_t>(ds.tokens.size()));
    for (std::size_t i = 1; i < ds.doc_starts.size(); ++i) REQUIRE(ds.doc_starts[i] > ds.doc_starts[i - 1]);
    REQUIRE(static_cast<int>(ds.doc_starts.size()) == static_cast<int>(s.drilled.size()) * opt.tasks_per_word + 1);

    // Every window stays within one task document.
    std::mt19937 rng(7);
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
