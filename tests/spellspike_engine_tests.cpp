// spellspike_engine_tests.cpp -- the engine-side harness for the combine/uncombine token-granularity
// spike. Train a tiny model on the synthetic logic curriculum (spellspike.hpp), then measure -- via
// the LIVE decode interceptor (decode.hpp's kv_decode_ops) -- whether it learned to INVOKE uncombine
// and USE the injected characters to answer character-level questions it otherwise couldn't. The
// headline is HELD-OUT accuracy: a correct answer for a word never seen in training is only possible
// by using the harness-provided characters, not memory. Round-trip (uncombine then combine back)
// tests the combine op + copy fidelity.
//
// Tagged "[.spellspike]" (hidden): trains hundreds of steps, so NOT in the default ctest sweep --
// invoke with `sub0_tests "[spellspike]"`. This is a SPIKE: assertions only guard "it ran + finite
// numbers"; the reported metrics (via WARN) are the deliverable.

#include <catch2/catch_test_macros.hpp>

#include "sub0/core.hpp"
#include "sub0/decode.hpp"
#include "sub0/spellspike.hpp"
#include "sub0/window.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <random>
#include <span>
#include <string>
#include <vector>

namespace {

namespace ss  = sub0::spellspike;
namespace cas = sub0::casing;
using sub0::tok::Tokenizer;

constexpr int    kBatch        = 16;
constexpr int    kWindowT      = 40;     // task traces are short; must be <= SEQ_LEN
constexpr int    kEvalRounds   = 10;
constexpr int    kStepsPerEval = 300;
constexpr int    kMaxEvalWords = 40;
constexpr int    kTasksPerWord = 16;
constexpr double kDrilledFrac  = 0.75;
constexpr float  kLr           = 0.003f;

// The two deterministic tokenizer-layer ops the decode interceptor calls. `tk` is captured by ref.
struct Ops {
    const Tokenizer& tk;
    std::vector<int> expand(int token) const {
        if (token < tk.n_base || token >= tk.vocab) return {token};
        return tk.expansion[static_cast<std::size_t>(token)];        // word piece -> its byte codes
    }
    std::vector<int> combine(const std::vector<int>& frags) const {
        std::string key;
        for (int f : frags) key.push_back(static_cast<char>(f & 0xFF));
        const auto it = tk.piece_index.find(key);
        return it != tk.piece_index.end() ? std::vector<int>{it->second} : frags;  // exact -> token, else identity
    }
};

// Run the live interceptor from a task prompt and return the produced context.
std::vector<int> run_task(const Ops& ops, const std::vector<int>& prompt) {
    return sub0::kv_decode_ops(prompt, /*max_steps=*/kWindowT, cas::TOK_EOS,
                               [&](int t) { return ops.expand(t); },
                               [&](const std::vector<int>& f) { return ops.combine(f); });
}

// The model's answer token = the token right after the '=' separator (nth/count tasks).
int answer_after_sep(const std::vector<int>& out) {
    for (std::size_t i = 0; i + 1 < out.size(); ++i) if (out[i] == ss::SEP) return out[i + 1];
    return -1;
}
// The combine result = the token right after the last TOK_COMBINE_END (round-trip).
int combine_result(const std::vector<int>& out) {
    for (std::size_t i = out.size(); i-- > 0;) if (out[i] == cas::TOK_COMBINE_END) return i + 1 < out.size() ? out[i + 1] : -1;
    return -1;
}

struct Acc { int ok = 0, n = 0; double rate() const { return n ? static_cast<double>(ok) / n : 0.0; } };

// Evaluate the three task types over a word subset. Nth-char is tested at every valid index; count at
// the word's own characters + a couple of absent ones; round-trip once per word.
struct TaskAcc { Acc nth, cnt, rt; };
TaskAcc eval_tasks(const Tokenizer& tk, const Ops& ops, const std::vector<int>& words) {
    TaskAcc a;
    const int n = std::min<int>(kMaxEvalWords, static_cast<int>(words.size()));
    std::mt19937 rng(123);
    for (int i = 0; i < n; ++i) {
        const int w = words[static_cast<std::size_t>(i)];
        const std::vector<int> bytes = tk.expansion[static_cast<std::size_t>(w)];
        const int len = static_cast<int>(bytes.size());
        for (int pos = 0; pos < len; ++pos) {                        // nth-char, every index
            const ss::Task k = ss::nth_char_task(tk, w, pos);
            a.nth.ok += (answer_after_sep(run_task(ops, k.prompt)) == k.answer_byte); ++a.nth.n;
        }
        for (int c = 0; c < 3; ++c) {                                // count: mix present + absent chars
            const int x = (c == 0) ? bytes[static_cast<std::size_t>(rng() % static_cast<unsigned>(len))]
                                   : static_cast<int>('a' + (rng() % 26));
            const ss::Task k = ss::count_task(tk, w, x);
            a.cnt.ok += (answer_after_sep(run_task(ops, k.prompt)) == k.answer_byte); ++a.cnt.n;
        }
        const ss::Task rt = ss::roundtrip_task(tk, w);               // round-trip
        a.rt.ok += (combine_result(run_task(ops, rt.prompt)) == w); ++a.rt.n;
    }
    return a;
}

void train_steps(const ss::Dataset& ds, sub0::AdamW& opt, int steps, std::mt19937& rng) {
    std::vector<std::size_t> starts(kBatch);
    std::vector<int>         lens(kBatch);
    for (int s = 0; s < steps; ++s) {
        for (int b = 0; b < kBatch; ++b) {
            const sub0::Window w = sub0::sample_window(rng, kWindowT, ds.tokens.size(),
                                                       std::span<const std::uint64_t>(ds.doc_starts));
            starts[static_cast<std::size_t>(b)] = w.start;
            lens[static_cast<std::size_t>(b)]   = w.len;
        }
        opt.zero_grad();
        // loss_mask grades ONLY the invoked ops + the answer + the copy, never the harness-injected
        // characters -- the fix for the memorization the unmasked run exhibited.
        (void)sub0::train_batch(ds.tokens.data(), starts.data(), kBatch, kWindowT, lens.data(), ds.mask.data());
        opt.step();
    }
}

}  // namespace

TEST_CASE("spellspike: a tiny model doing char-reasoning via uncombine (drilled vs held-out)", "[.spellspike]") {
    REQUIRE(kWindowT <= SEQ_LEN);

    Tokenizer tk;
    {
        std::ifstream is(sub0::default_tokenizer(), std::ios::binary);
        REQUIRE(is.good());
        REQUIRE(sub0::tok::deserialize(tk, is));
    }
    REQUIRE(tk.vocab == VOCAB);
    const Ops ops{tk};

    const ss::WordSplit split = ss::split_task_words(tk, kDrilledFrac, /*seed=*/2024);
    REQUIRE_FALSE(split.drilled.empty());
    REQUIRE_FALSE(split.held_out.empty());

    ss::DatasetOptions dopt; dopt.tasks_per_word = kTasksPerWord; dopt.seed = 99;
    const ss::Dataset ds = ss::build_dataset(tk, split, dopt);
    REQUIRE(ds.tokens.size() > static_cast<std::size_t>(kWindowT));

    sub0::build_model();
    std::string report = "\n=== spellspike combine/uncombine (VOCAB=" + std::to_string(VOCAB) +
                         " D_MODEL=" + std::to_string(D_MODEL) + " tied=" + std::to_string(USE_TIED_EMBEDDINGS) +
                         " task_words=" + std::to_string(split.drilled.size() + split.held_out.size()) +
                         " drilled=" + std::to_string(split.drilled.size()) +
                         " held_out=" + std::to_string(split.held_out.size()) + ") ===\n"
                         "  (held-out = words NEVER trained; a correct answer there PROVES the model uses\n"
                         "   the harness-injected characters, not memorized letters)\n";

    sub0::AdamW opt(kLr);
    std::mt19937 rng(1);
    double last_drilled_nth = 0.0, last_held_nth = 0.0;
    for (int r = 0; r < kEvalRounds; ++r) {
        train_steps(ds, opt, kStepsPerEval, rng);
        const TaskAcc d = eval_tasks(tk, ops, split.drilled);
        const TaskAcc h = eval_tasks(tk, ops, split.held_out);
        last_drilled_nth = d.nth.rate(); last_held_nth = h.nth.rate();
        char line[320];
        std::snprintf(line, sizeof line,
            "  step %5d | DRILLED nth=%.3f cnt=%.3f rt=%.3f | HELD-OUT nth=%.3f cnt=%.3f rt=%.3f\n",
            (r + 1) * kStepsPerEval, d.nth.rate(), d.cnt.rate(), d.rt.rate(),
            h.nth.rate(), h.cnt.rate(), h.rt.rate());
        report += line;
    }
    WARN(report);

    // Spike guards only (metrics are the deliverable): it ran, produced finite rates, and drilled
    // nth-char reasoning moved off the floor.
    REQUIRE(std::isfinite(last_drilled_nth));
    REQUIRE(std::isfinite(last_held_nth));
    REQUIRE(last_drilled_nth > 0.15);
}
