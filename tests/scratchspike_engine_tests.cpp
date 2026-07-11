// scratchspike_engine_tests.cpp -- the engine-side harness for the SCRATCH TOKEN (context-translation)
// spike. Train a tiny model on the synthetic curriculum (scratchspike.hpp), then measure -- via the
// LIVE decode interceptor (decode.hpp's kv_decode_generate) driven by a STATEFUL binding table -- whether
// it learned to resolve a reference to a dynamically-bound, embedding-less SCRATCH SLOT by invoking
// uncombine, and whether that GENERALIZES to HELD-OUT OOVs never bound in training (the clean
// no-memorization probe: a correct char-level answer for an OOV the model never saw bound is only
// possible by resolving the slot's in-context binding via the interceptor's side table).
//
// The scratch-specific machinery lives ENTIRELY in the interceptor's callbacks: because
// kv_decode_generate takes expand/combine as std::function, a stateful ScratchOps (owning slot->fragments)
// supplies the binding table with NO engine or decode.hpp change. This validates the compression
// mechanism (an OOV costs 1 scratch token per reference, its bytes recovered on demand) BEFORE any
// content-derived-embedding engine work. Tagged "[.scratchspike]" (hidden): trains hundreds of steps,
// so NOT in the default ctest sweep -- invoke with `sub0_tests "[scratchspike]"`.

#include <catch2/catch_test_macros.hpp>

#include "sub0/core.hpp"
#include "sub0/decode.hpp"
#include "sub0/scratchspike.hpp"
#include "sub0/window.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <random>
#include <span>
#include <string>
#include <vector>

namespace {

namespace ss  = sub0::scratchspike;
namespace cas = sub0::casing;
using sub0::tok::Tokenizer;

constexpr int    kBatch        = 16;
constexpr int    kWindowT      = 40;
constexpr int    kEvalRounds   = 10;
constexpr int    kStepsPerEval = 300;
constexpr int    kOovPool      = 400;   // many distinct OOVs -> force the general index op, not memorization
constexpr double kDrilledFrac  = 0.7;
constexpr float  kLr           = 0.003f;

// The stateful interceptor: a per-context BINDING TABLE (slot -> fragments) plus the two deterministic
// tokenizer-layer ops the decode loop calls. Reset per eval task; the harness pre-binds the scratch
// slot to the task's OOV (simulating a prior in-context definition), then the model must resolve
// references to it. combine also returns an EXISTING binding's slot (the round-trip's inverse leg).
struct ScratchOps {
    const Tokenizer&              tk;
    std::vector<std::vector<int>> bindings;   // bindings[i] are the fragments of slot SCRATCH_BASE+i

    void reset() { bindings.clear(); }
    void bind(int slot, std::vector<int> frags) {
        const std::size_t i = static_cast<std::size_t>(slot - ss::SCRATCH_BASE);
        if (bindings.size() <= i) bindings.resize(i + 1);
        bindings[i] = std::move(frags);
    }
    std::vector<int> expand(int token) const {
        const int s = token - ss::SCRATCH_BASE;
        if (s >= 0 && s < static_cast<int>(bindings.size()) && !bindings[static_cast<std::size_t>(s)].empty())
            return bindings[static_cast<std::size_t>(s)];                 // scratch slot -> its bound fragments
        if (token >= tk.n_base && token < tk.vocab) return tk.expansion[static_cast<std::size_t>(token)];
        return {token};
    }
    std::vector<int> combine(const std::vector<int>& frags) {
        std::string key;
        for (int f : frags) key.push_back(static_cast<char>(f & 0xFF));
        const auto it = tk.piece_index.find(key);
        if (it != tk.piece_index.end()) return {it->second};             // exact vocab piece -> that token
        for (std::size_t i = 0; i < bindings.size(); ++i)
            if (bindings[i] == frags) return {ss::SCRATCH_BASE + static_cast<int>(i)};   // existing binding -> its slot
        if (static_cast<int>(bindings.size()) < ss::SCRATCH_POOL) {      // model-driven bind: assign a free slot
            bindings.push_back(frags);
            return {ss::SCRATCH_BASE + static_cast<int>(bindings.size()) - 1};
        }
        return frags;                                                    // pool full -> no compression
    }
};

// Run the live interceptor from a task: reset the table, pre-bind slot i to the task's binds[i]
// (single task binds slot 0; a multi task binds all K), then greedy decode (topk=1) with the
// binding-table callbacks. Returns the full produced context.
std::vector<int> run_task(ScratchOps& ops, const ss::Task& k) {
    ops.reset();
    for (int i = 0; i < static_cast<int>(k.binds.size()); ++i)
        ops.bind(ss::scratch_slot(i), ss::oov_bytes(k.binds[static_cast<std::size_t>(i)]));
    std::vector<int> ctx = k.prompt;
    std::mt19937 rng(0);
    sub0::kv_decode_generate(ctx, /*n=*/kWindowT, /*temp=*/1.f, /*topk=*/1, rng, cas::TOK_EOS,
                             /*use_gpu=*/false, /*on_token=*/{},
                             [&](int t) { return ops.expand(t); },
                             [&](const std::vector<int>& f) { return ops.combine(f); });
    return ctx;
}

int answer_after_sep(const std::vector<int>& out) {
    for (std::size_t i = 0; i + 1 < out.size(); ++i) if (out[i] == ss::SEP) return out[i + 1];
    return -1;
}
int combine_result(const std::vector<int>& out) {
    for (std::size_t i = out.size(); i-- > 0;) if (out[i] == cas::TOK_COMBINE_END) return i + 1 < out.size() ? out[i + 1] : -1;
    return -1;
}

struct Acc { int ok = 0, n = 0; double rate() const { return n ? static_cast<double>(ok) / n : 0.0; } };
struct TaskAcc { Acc nth, rt; };

// nth-char at every index of the OOV + one round-trip per OOV, over an OOV subset (single-binding).
TaskAcc eval_tasks(const Tokenizer& tk, ScratchOps& ops, const std::vector<std::string>& oovs) {
    TaskAcc a;
    const int slot0 = ss::scratch_slot(0);
    for (const std::string& oov : oovs) {
        const std::vector<int> bytes = ss::oov_bytes(oov);
        for (int pos = 0; pos < static_cast<int>(bytes.size()); ++pos) {
            const ss::Task k = ss::nth_char_task(tk, oov, pos);
            a.nth.ok += (answer_after_sep(run_task(ops, k)) == k.answer_byte); ++a.nth.n;
        }
        const ss::Task rt = ss::roundtrip_task(tk, oov);
        a.rt.ok += (combine_result(run_task(ops, rt)) == slot0); ++a.rt.n;   // recombine -> same slot
    }
    return a;
}

// Multi-binding: K slots bound per context, query one at random. Measures whether resolution stays
// correct amid K-1 competing bindings (attention + KV interference) vs the single-binding baseline.
Acc eval_multi(const Tokenizer& tk, ScratchOps& ops, const std::vector<std::string>& oovs, int K, unsigned seed) {
    Acc a;
    std::mt19937_64 rng(seed);
    for (int qi = 0; qi < static_cast<int>(oovs.size()); ++qi) {
        const ss::Task k = ss::pick_multi_task(tk, oovs, qi, K, rng);
        a.ok += (answer_after_sep(run_task(ops, k)) == k.answer_byte); ++a.n;
    }
    return a;
}

// #1 model-driven binding: no pre-bind; the model emits combine to define the OOV, and success = the
// interceptor's resulting binding holds the RIGHT bytes (the model copied the OOV correctly).
Acc eval_define(const Tokenizer& tk, ScratchOps& ops, const std::vector<std::string>& oovs) {
    Acc a;
    for (const std::string& oov : oovs) {
        const ss::Task k = ss::define_task(tk, oov);
        run_task(ops, k);   // k.binds empty -> the model's own combine performs the binding
        a.ok += (!ops.bindings.empty() && ops.bindings[0] == ss::oov_bytes(oov)); ++a.n;
    }
    return a;
}

// #2 reason-over-slots: K tagged slots, query a tag -> the model must output the matching slot.
Acc eval_select(const Tokenizer& tk, ScratchOps& ops, const std::vector<std::string>& oovs, int K, unsigned seed) {
    Acc a;
    std::mt19937_64 rng(seed);
    for (int qi = 0; qi < static_cast<int>(oovs.size()); ++qi) {
        const ss::Task k = ss::pick_select_task(tk, oovs, qi, K, rng);
        a.ok += (answer_after_sep(run_task(ops, k)) == k.answer_byte); ++a.n;
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
        (void)sub0::train_batch(ds.tokens.data(), starts.data(), kBatch, kWindowT, lens.data(), ds.mask.data());
        opt.step();
    }
}

}  // namespace

TEST_CASE("scratchspike: a tiny model resolving dynamically-bound scratch tokens (drilled vs held-out)", "[.scratchspike]") {
    REQUIRE(kWindowT <= SEQ_LEN);

    Tokenizer tk;
    {
        std::ifstream is(sub0::default_tokenizer(), std::ios::binary);
        REQUIRE(is.good());
        REQUIRE(sub0::tok::deserialize(tk, is));
    }
    REQUIRE(tk.vocab == VOCAB);
    ScratchOps ops{tk, {}};

    const ss::OovSplit split = ss::make_oov_split(tk, kOovPool, kDrilledFrac, /*seed=*/2024);
    REQUIRE_FALSE(split.drilled.empty());
    REQUIRE_FALSE(split.held_out.empty());

    ss::DatasetOptions dopt; dopt.tasks_per_oov = 12; dopt.seed = 99;
    const ss::Dataset ds = ss::build_dataset(tk, split, dopt);
    REQUIRE(ds.tokens.size() > static_cast<std::size_t>(kWindowT));

    sub0::build_model();
    std::string report = "\n=== scratchspike context-translation (VOCAB=" + std::to_string(VOCAB) +
                         " D_MODEL=" + std::to_string(D_MODEL) + " tied=" + std::to_string(USE_TIED_EMBEDDINGS) +
                         " oov_pool=" + std::to_string(kOovPool) +
                         " drilled=" + std::to_string(split.drilled.size()) +
                         " held_out=" + std::to_string(split.held_out.size()) + ") ===\n"
                         "  (held-out = OOVs never bound in training; a correct answer there PROVES the\n"
                         "   model resolves the scratch slot via its in-context binding, not memory)\n";

    sub0::AdamW opt(kLr);
    std::mt19937 rng(1);
    double last_held_nth = 0.0;
    for (int r = 0; r < kEvalRounds; ++r) {
        train_steps(ds, opt, kStepsPerEval, rng);
        const TaskAcc d = eval_tasks(tk, ops, split.drilled);
        const TaskAcc h = eval_tasks(tk, ops, split.held_out);
        last_held_nth = h.nth.rate();
        char line[256];
        std::snprintf(line, sizeof line,
            "  step %5d | DRILLED nth=%.3f rt=%.3f | HELD-OUT nth=%.3f rt=%.3f\n",
            (r + 1) * kStepsPerEval, d.nth.rate(), d.rt.rate(), h.nth.rate(), h.rt.rate());
        report += line;
    }
    WARN(report);

    REQUIRE(std::isfinite(last_held_nth));
}

TEST_CASE("scratchspike: multi-binding disambiguation amid competing slots (drilled vs held-out)", "[.scratchspike]") {
    REQUIRE(kWindowT <= SEQ_LEN);
    constexpr int kK = 3;   // slots bound per context

    Tokenizer tk;
    {
        std::ifstream is(sub0::default_tokenizer(), std::ios::binary);
        REQUIRE(is.good());
        REQUIRE(sub0::tok::deserialize(tk, is));
    }
    REQUIRE(tk.vocab == VOCAB);
    REQUIRE(kK <= ss::SCRATCH_POOL);
    ScratchOps ops{tk, {}};

    const ss::OovSplit split = ss::make_oov_split(tk, kOovPool, kDrilledFrac, /*seed=*/2024);
    ss::DatasetOptions dopt; dopt.tasks_per_oov = 12; dopt.seed = 99;
    const ss::Dataset ds = ss::build_dataset_multi(tk, split, kK, dopt);
    REQUIRE(ds.tokens.size() > static_cast<std::size_t>(kWindowT));

    sub0::build_model();
    std::string report = "\n=== scratchspike MULTI-binding (K=" + std::to_string(kK) + " slots/context, VOCAB=" +
                         std::to_string(VOCAB) + " D_MODEL=" + std::to_string(D_MODEL) +
                         " drilled=" + std::to_string(split.drilled.size()) +
                         " held_out=" + std::to_string(split.held_out.size()) + ") ===\n"
                         "  (query one of K bound slots; the model must resolve the RIGHT one amid the others)\n";

    sub0::AdamW opt(kLr);
    std::mt19937 rng(1);
    double last_held = 0.0;
    for (int r = 0; r < kEvalRounds; ++r) {
        train_steps(ds, opt, kStepsPerEval, rng);
        const Acc d = eval_multi(tk, ops, split.drilled, kK, /*seed=*/7);
        const Acc h = eval_multi(tk, ops, split.held_out, kK, /*seed=*/11);
        last_held = h.rate();
        char line[192];
        std::snprintf(line, sizeof line, "  step %5d | DRILLED nth=%.3f | HELD-OUT nth=%.3f\n",
                      (r + 1) * kStepsPerEval, d.rate(), h.rate());
        report += line;
    }
    WARN(report);

    REQUIRE(std::isfinite(last_held));
}

// Shared trainer for the two model-drives-it tests below: load tokenizer, split OOVs, build the given
// dataset, train, and report drilled vs held-out via the supplied evaluator. Returns the last held-out.
template <class BuildDs, class Eval>
double run_scratch_experiment(const char* title, const char* legend, BuildDs build_ds, Eval eval) {
    Tokenizer tk;
    {
        std::ifstream is(sub0::default_tokenizer(), std::ios::binary);
        REQUIRE(is.good());
        REQUIRE(sub0::tok::deserialize(tk, is));
    }
    REQUIRE(tk.vocab == VOCAB);
    ScratchOps ops{tk, {}};

    const ss::OovSplit split = ss::make_oov_split(tk, kOovPool, kDrilledFrac, /*seed=*/2024);
    ss::DatasetOptions dopt; dopt.tasks_per_oov = 12; dopt.seed = 99;
    const ss::Dataset ds = build_ds(tk, split, dopt);
    REQUIRE(ds.tokens.size() > static_cast<std::size_t>(kWindowT));

    sub0::build_model();
    std::string report = std::string("\n=== ") + title + " (VOCAB=" + std::to_string(VOCAB) +
                         " D_MODEL=" + std::to_string(D_MODEL) + " drilled=" + std::to_string(split.drilled.size()) +
                         " held_out=" + std::to_string(split.held_out.size()) + ") ===\n  " + legend + "\n";
    sub0::AdamW opt(kLr);
    std::mt19937 rng(1);
    double last_held = 0.0;
    for (int r = 0; r < kEvalRounds; ++r) {
        train_steps(ds, opt, kStepsPerEval, rng);
        const Acc d = eval(tk, ops, split.drilled);
        const Acc h = eval(tk, ops, split.held_out);
        last_held = h.rate();
        char line[160];
        std::snprintf(line, sizeof line, "  step %5d | DRILLED %.3f | HELD-OUT %.3f\n",
                      (r + 1) * kStepsPerEval, d.rate(), h.rate());
        report += line;
    }
    WARN(report);
    return last_held;
}

TEST_CASE("scratchspike: MODEL-DRIVEN binding -- the model emits combine to define an OOV", "[.scratchspike]") {
    const double held = run_scratch_experiment(
        "scratchspike #1 MODEL-DRIVEN binding",
        "(no pre-bind: the model emits combine over the OOV; success = it copied it so the slot binds right)",
        [](const Tokenizer& tk, const ss::OovSplit& s, const ss::DatasetOptions& o) { return ss::build_dataset_define(tk, s, o); },
        [](const Tokenizer& tk, ScratchOps& ops, const std::vector<std::string>& oovs) { return eval_define(tk, ops, oovs); });
    REQUIRE(std::isfinite(held));
}

TEST_CASE("scratchspike: REASON-OVER-SLOTS -- select the slot carrying a queried tag", "[.scratchspike]") {
    constexpr int kK = 3;
    const double held = run_scratch_experiment(
        "scratchspike #2 REASON-OVER-SLOTS (select, K=3)",
        "(K tagged slots; query a tag -> output the matching slot: slots as distinct in-context entities)",
        [](const Tokenizer& tk, const ss::OovSplit& s, const ss::DatasetOptions& o) { return ss::build_dataset_select(tk, s, kK, o); },
        [](const Tokenizer& tk, ScratchOps& ops, const std::vector<std::string>& oovs) { return eval_select(tk, ops, oovs, kK, 7); });
    REQUIRE(std::isfinite(held));
}
