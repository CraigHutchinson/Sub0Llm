// persistent_scratchspike_engine_tests.cpp -- SPIKE: does resolve-a-named-reference
// (persistent_scratchspike::multi_nth_char_task) generalize to held-out OOVs when bound to the PERSISTENT
// (id >= VOCAB) slot range at K well beyond the ephemeral pool's hard 6-slot ceiling? Mirrors
// scratchspike_engine_tests.cpp's own methodology (drilled-vs-held-out split, live kv_decode_generate
// eval) as closely as possible for a direct, comparable result -- the only structural differences are the
// id range (base+i, not SCRATCH_BASE+i) and how training is driven: sub0::train_batch does NOT accept
// PersistentBindings (a single global, not a per-window array like train_batch's own win_binds -- see
// persistent_slots_engine_tests.cpp's own comment on this), so training here manually loops over the
// batch's windows via the low-level forward/backward API. This is the exact same "an unbatchable binding
// mechanism needs its own driving loop" shape scratchspike_engine_tests.cpp's own train_ce_steps already
// established for CharEncoder's un-batchable enc_w gradient -- reused here for the analogous reason.
//
// Tagged "[.persistent_scratchspike]" (hidden, like "[.scratchspike]"): trains hundreds of steps per
// TEST_CASE, not in the default ctest sweep. Invoke explicitly: `sub0_tests "[persistent_scratchspike]"`
// (drop the leading dot -- Catch2's tag filter matches on the name, the dot only controls default-hiding).

#include <catch2/catch_test_macros.hpp>

#include "sub0/core.hpp"
#include "sub0/decode.hpp"
#include "sub0/persistent_scratchspike.hpp"
#include "sub0/scratch_slots.hpp"
#include "sub0/scratchspike.hpp"
#include "sub0/window.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <random>
#include <span>
#include <string>
#include <vector>

namespace {

namespace pss = sub0::persistent_scratchspike;
namespace ss  = sub0::scratchspike;
namespace cas = sub0::casing;
using sub0::tok::Tokenizer;

constexpr int    kBatch        = 16;
constexpr int    kEvalRounds   = 10;
constexpr int    kStepsPerEval = 300;
constexpr int    kOovPool      = 400;   // many distinct OOVs -> force the general index op, not memorization
constexpr double kDrilledFrac  = 0.7;
// Same LR-scaling rule as scratchspike_engine_tests.cpp (tuned at d128, scaled ~1/width so the recipe
// transfers across model scale -- see that file's own comment + docs/SCRATCH_TOKENS.md).
constexpr float  kLr           = 0.003f * (128.0f / static_cast<float>(D_MODEL));

// Test-only stateful binding table for the PERSISTENT range -- analogous to sub0::ScratchTable
// (scratch.hpp) but for id >= base. NOT a production component (unlike scratchspike_engine_tests.cpp's
// own ScratchOps, which reuses the REAL ScratchTable gen_stage drives): a real production persistent-slot
// table needs an actual compound-word cache/DB, explicitly out of scope for this spike (see project
// memory persistent-slot-selection-problem-backlog and docs/DETERMINISTIC_MECHANISMS.md's "persistent
// compound-word cache" note). This is deliberately the minimal thing needed to drive
// kv_decode_generate's expand/combine + sub0::set_persistent_bindings() for one eval task.
struct PersistentTable {
    int                            base = 0;   // VOCAB
    std::vector<std::vector<int>> bindings;   // bindings[i] = fragments of base+i

    void reset() { bindings.clear(); }
    void bind(int slot, std::vector<int> frags) {
        const int i = slot - base;
        if (i < 0) return;
        if (static_cast<int>(bindings.size()) <= i) bindings.resize(static_cast<std::size_t>(i) + 1);
        bindings[static_cast<std::size_t>(i)] = std::move(frags);
    }
    sub0::PersistentBindings to_bindings() const {
        return sub0::PersistentBindings{ std::span<const std::vector<int>>(bindings), base, sub0::SlotEncoding::MeanPool };
    }
    // Fulfil a TOK_UNCOMBINE on a persistent id -> its bound fragments; anything else -> identity.
    std::vector<int> expand(int token) const {
        const int s = token - base;
        if (s >= 0 && s < static_cast<int>(bindings.size()) && !bindings[static_cast<std::size_t>(s)].empty())
            return bindings[static_cast<std::size_t>(s)];
        return {token};
    }
    // Never actually invoked by this attend-only curriculum (no task asks the model to BIND anything) --
    // supplied only because kv_decode_generate requires BOTH expand and combine truthy to enable
    // interception at all. Identity fallback (no compression), matching ScratchTable::combine's own
    // "can't mint" case.
    std::vector<int> combine(const std::vector<int>& frags) const { return frags; }
};

// Run one task via the LIVE interceptor + persistent-binding dispatch: pre-bind slots 0..K-1 from
// k.binds, install the table via set_persistent_bindings (read by backend_cpu.cpp's forward_one for
// EVERY token fed through kv_decode_generate, including the persistent ids in the primed prompt), greedy
// decode, uninstall. Mirrors scratchspike_engine_tests.cpp's own run_task exactly.
std::vector<int> run_task(PersistentTable& ops, const pss::Task& k) {
    ops.reset();
    for (int i = 0; i < static_cast<int>(k.binds.size()); ++i)
        ops.bind(ops.base + i, ss::oov_bytes(k.binds[static_cast<std::size_t>(i)]));
    const sub0::PersistentBindings pb = ops.to_bindings();
    sub0::set_persistent_bindings(&pb);
    std::vector<int> ctx = k.prompt;
    std::mt19937 rng(0);
    // kv_decode_generate's precondition is ctx.size()+n <= SEQ_LEN; 64 is comfortably above the longest
    // actual generation needed (resolve region ~8 tokens + answer region ~3), but clamp defensively so a
    // large K against a build with a small SEQ_LEN degrades to a smaller (still ample) budget instead of
    // violating the precondition.
    const int n = std::min(64, SEQ_LEN - static_cast<int>(ctx.size()) - 1);
    sub0::kv_decode_generate(ctx, n, /*temp=*/1.f, /*topk=*/1, rng, cas::TOK_EOS,
                             /*use_gpu=*/false, /*on_token=*/{},
                             [&](int t) { return ops.expand(t); },
                             [&](const std::vector<int>& f) { return ops.combine(f); });
    sub0::set_persistent_bindings(nullptr);
    return ctx;
}

int answer_after_sep(const std::vector<int>& out) {
    for (std::size_t i = 0; i + 1 < out.size(); ++i) if (out[i] == pss::SEP) return out[i + 1];
    return -1;
}

struct Acc { int ok = 0, n = 0; double rate() const { return n ? static_cast<double>(ok) / n : 0.0; } };

Acc eval_multi(PersistentTable& ops, const std::vector<std::string>& oovs, int K, unsigned seed) {
    Acc a;
    std::mt19937_64 rng(seed);
    for (int qi = 0; qi < static_cast<int>(oovs.size()); ++qi) {
        const pss::Task k = pss::pick_multi_task(ops.base, oovs, qi, K, rng);
        a.ok += (answer_after_sep(run_task(ops, k)) == k.answer_byte); ++a.n;
    }
    return a;
}

// Cold optimizer per TEST_CASE -- the AdamW moment arenas are GLOBAL and persist across test cases in one
// process; a prior test's training would otherwise warm-start this one. Same rationale + same fix as
// scratchspike_engine_tests.cpp's own reset_opt_state.
void reset_opt_state() {
    const std::size_t n = sub0::trainable_floats();
    std::fill(sub0::adam_m_ptr(), sub0::adam_m_ptr() + n, 0.f);
    std::fill(sub0::adam_v_ptr(), sub0::adam_v_ptr() + n, 0.f);
}

// Manual per-window forward/backward training loop -- train_batch does not accept PersistentBindings (a
// single global, not a per-window array), so this drives the low-level API directly, one window at a
// time, accumulating gradient across the batch before opt.step() -- the exact shape
// scratchspike_engine_tests.cpp's own train_ce_steps already established for the analogous reason
// (CharEncoder's un-batchable enc_w gradient), reused here. `window` must be >= the longest task trace in
// `ds` (the K-sweep needs a wider window than a fixed small K would -- same caveat train_steps' own
// `window` parameter documents in scratchspike_engine_tests.cpp).
void train_steps(const pss::Dataset& ds, sub0::AdamW& opt, int steps, std::mt19937& rng, int window) {
    std::vector<int> masked_tgt(static_cast<std::size_t>(window));
    for (int s = 0; s < steps; ++s) {
        opt.zero_grad();
        for (int b = 0; b < kBatch; ++b) {
            const sub0::Window w = sub0::sample_window(rng, window, ds.tokens.size(),
                                                       std::span<const std::uint64_t>(ds.doc_starts));
            const int Tb = w.len;
            for (int i = 0; i < Tb; ++i) {
                const std::size_t p = w.start + static_cast<std::size_t>(i) + 1;
                masked_tgt[static_cast<std::size_t>(i)] = ds.mask[p] ? ds.tokens[p] : sub0::LOSS_IGNORE_INDEX;
            }
            const std::size_t doc = static_cast<std::size_t>(
                std::upper_bound(ds.doc_starts.begin(), ds.doc_starts.end(),
                                 static_cast<std::uint64_t>(w.start)) - ds.doc_starts.begin()) - 1;
            const sub0::PersistentBindings pb{ std::span<const std::vector<int>>(ds.doc_bindings[doc]),
                                               VOCAB, sub0::SlotEncoding::MeanPool };
            sub0::set_persistent_bindings(&pb);
            sub0::graph_reset();
            sub0::Node* logits = sub0::forward(ds.tokens.data() + w.start, Tb);
            sub0::Node* loss   = sub0::cross_entropy(logits, masked_tgt.data());
            sub0::backward(loss, 1.f / static_cast<float>(kBatch));
            sub0::set_persistent_bindings(nullptr);
        }
        sub0::reduce_gradients();
        opt.step();
    }
}

// Shared body for the K-sweep points below: load the tokenizer, split OOVs, build the dataset at the
// given K, train + report drilled/held-out resolve accuracy every kStepsPerEval steps.
void run_persistent_multi_experiment(int K) {
    // Worst-case trace length for this K: K preamble refs + 3 query tokens + up to 8 resolve tokens
    // (TOK_UNCOMBINE + up to 6 fragment bytes + TOK_UNCOMBINE_END) + 3 answer tokens (SEP + byte + EOS).
    // Generous margin over that, capped at SEQ_LEN (train_steps' own `window` contract).
    const int window = std::min(SEQ_LEN, K + 24);
    REQUIRE(window >= K + 14);   // otherwise this build's SEQ_LEN is too small to even fit one task at this K

    Tokenizer tk;
    {
        std::ifstream is(sub0::default_tokenizer(), std::ios::binary);
        REQUIRE(is.good());
        REQUIRE(sub0::tok::deserialize(tk, is));
    }
    REQUIRE(tk.vocab == VOCAB);

    const ss::OovSplit split = ss::make_oov_split(tk, kOovPool, kDrilledFrac, /*seed=*/2024);
    REQUIRE_FALSE(split.drilled.empty());
    REQUIRE_FALSE(split.held_out.empty());
    REQUIRE(static_cast<int>(split.drilled.size()) > K);   // enough distinct OOVs to fill K distractors

    pss::DatasetOptions dopt; dopt.tasks_per_oov = 12; dopt.seed = 99;
    const pss::Dataset ds = pss::build_dataset_multi(VOCAB, split, K, dopt);
    REQUIRE(ds.tokens.size() > static_cast<std::size_t>(window));

    sub0::build_model();
    reset_opt_state();
    PersistentTable ops; ops.base = VOCAB;

    std::string report = "\n=== persistent_scratchspike MULTI-binding (K=" + std::to_string(K) +
                         " slots/context, PERSISTENT range [VOCAB.." + std::to_string(VOCAB + K - 1) +
                         "], VOCAB=" + std::to_string(VOCAB) + " D_MODEL=" + std::to_string(D_MODEL) +
                         " window=" + std::to_string(window) +
                         " drilled=" + std::to_string(split.drilled.size()) +
                         " held_out=" + std::to_string(split.held_out.size()) + ") ===\n"
                         "  (query one of K persistent-bound slots; the model must resolve the RIGHT one\n"
                         "   amid the others -- held-out = OOVs never bound in training)\n";

    sub0::AdamW opt(kLr);
    std::mt19937 rng(1);
    double last_held = 0.0;
    for (int r = 0; r < kEvalRounds; ++r) {
        train_steps(ds, opt, kStepsPerEval, rng, window);
        const Acc d = eval_multi(ops, split.drilled, K, /*seed=*/7);
        const Acc h = eval_multi(ops, split.held_out, K, /*seed=*/11);
        last_held = h.rate();
        char line[192];
        std::snprintf(line, sizeof line, "  step %5d | DRILLED nth=%.3f | HELD-OUT nth=%.3f\n",
                      (r + 1) * kStepsPerEval, d.rate(), h.rate());
        report += line;
    }
    WARN(report);

    REQUIRE(std::isfinite(last_held));
}

}  // namespace

TEST_CASE("persistent_scratchspike: multi-binding resolve at K=6 (baseline -- matches the ephemeral "
         "pool's own maximum pool size, sanity-checking the mechanism is equivalent)",
         "[.persistent_scratchspike]") {
    run_persistent_multi_experiment(6);
}

TEST_CASE("persistent_scratchspike: multi-binding resolve at K=30 (well past the ephemeral pool's "
         "hard 6-slot ceiling -- the actual question this spike exists to answer)",
         "[.persistent_scratchspike]") {
    run_persistent_multi_experiment(30);
}
