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
    std::vector<int> combine(const std::vector<int>& frags) const {
        std::string key;
        for (int f : frags) key.push_back(static_cast<char>(f & 0xFF));
        const auto it = tk.piece_index.find(key);
        if (it != tk.piece_index.end()) return {it->second};             // exact vocab piece -> that token
        for (std::size_t i = 0; i < bindings.size(); ++i)
            if (bindings[i] == frags) return {ss::SCRATCH_BASE + static_cast<int>(i)};   // existing binding -> its slot
        return frags;                                                    // (eval pre-binds; unseen OOV stays fragments)
    }
};

// Run the live interceptor from a task prompt: reset the table, pre-bind slot 0 to this OOV, then greedy
// decode (topk=1) with the binding-table callbacks. Returns the full produced context.
std::vector<int> run_task(ScratchOps& ops, const std::string& oov, const std::vector<int>& prompt) {
    ops.reset();
    ops.bind(ss::scratch_slot(0), ss::oov_bytes(oov));
    std::vector<int> ctx = prompt;
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

// nth-char at every index of the OOV + one round-trip per OOV, over an OOV subset.
TaskAcc eval_tasks(const Tokenizer& tk, ScratchOps& ops, const std::vector<std::string>& oovs) {
    TaskAcc a;
    const int slot0 = ss::scratch_slot(0);
    for (const std::string& oov : oovs) {
        const std::vector<int> bytes = ss::oov_bytes(oov);
        for (int pos = 0; pos < static_cast<int>(bytes.size()); ++pos) {
            const ss::Task k = ss::nth_char_task(tk, oov, pos);
            a.nth.ok += (answer_after_sep(run_task(ops, oov, k.prompt)) == k.answer_byte); ++a.nth.n;
        }
        const ss::Task rt = ss::roundtrip_task(tk, oov);
        a.rt.ok += (combine_result(run_task(ops, oov, rt.prompt)) == slot0); ++a.rt.n;   // recombine -> same slot
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
