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
#include "sub0/scratch.hpp"
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
// Base LR 0.003 is tuned for the d128 spike; scale it ~1/width for larger builds so the recipe TRANSFERS
// across model scale. A d128-tuned LR lands a 2x-wider model in the degenerate always-S0 basin and it never
// escapes -- VALIDATED: at d256 both auto-config and the scaled-working shape collapsed at 0.003 but
// recovered (K=2 peak 0.83) at 0.0015 == 0.003 * 128/256. Identical to 0.003 at d128 (the default build).
// See docs/SCRATCH_TOKENS.md "Status -- the scale run".
constexpr float  kLr           = 0.003f * (128.0f / static_cast<float>(D_MODEL));

// The stateful interceptor is the PRODUCTION binding table (sub0::ScratchTable, include/sub0/scratch.hpp)
// -- the spike exercises the exact component gen_stage drives, no duplicate. Reset per eval task; the
// harness pre-binds a scratch slot to the task's OOV (a prior in-context definition), then the model must
// resolve references to it. combine returns an existing binding's slot, or (allow_bind) mints a fresh one.
using ScratchOps = sub0::ScratchTable;

// Run the live interceptor from a task: reset the table, pre-bind slot i to the task's binds[i]
// (single task binds slot 0; a multi task binds all K), then greedy decode (topk=1) with the
// binding-table callbacks. Returns the full produced context.
std::vector<int> run_task(ScratchOps& ops, const ss::Task& k, bool content_embed = false,
                          sub0::SlotEncoding enc = sub0::SlotEncoding::MeanPool, const float* enc_w = nullptr) {
    ops.reset();
    for (int i = 0; i < static_cast<int>(k.binds.size()); ++i)
        ops.bind(ss::scratch_slot(i), ss::oov_bytes(k.binds[static_cast<std::size_t>(i)]));
    // Match training: if the model was trained WITH content-derived embeddings, feed this task's bindings
    // to the decode forward_one too (else a bound slot embeds from the fixed row, a train/eval mismatch).
    sub0::ScratchBindings binds = ops.to_bindings(enc);
    binds.enc_w = enc_w;   // CharEncoder weights (eval needs no grad -> enc_w_grad stays null)
    if (content_embed) sub0::set_scratch_bindings(&binds);
    std::vector<int> ctx = k.prompt;
    std::mt19937 rng(0);
    sub0::kv_decode_generate(ctx, /*n=*/kWindowT, /*temp=*/1.f, /*topk=*/1, rng, cas::TOK_EOS,
                             /*use_gpu=*/false, /*on_token=*/{},
                             [&](int t) { return ops.expand(t); },
                             [&](const std::vector<int>& f) { return ops.combine(f); });
    if (content_embed) sub0::set_scratch_bindings(nullptr);
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

// #4 CONTENT reason-over-slots: query a first-letter -> the slot whose OOV starts with it (needs content
// AND order -- MeanPool/CharEncoder are provably permutation-invariant so cannot address this; Hash is the
// order-sensitive candidate, see TODO(order-sensitive-slot-encoding) in scratch_slots.hpp).
Acc eval_content(const Tokenizer& tk, ScratchOps& ops, const std::vector<std::string>& oovs, int K, unsigned seed,
                 bool content_embed = false, sub0::SlotEncoding enc = sub0::SlotEncoding::MeanPool,
                 const float* enc_w = nullptr) {
    Acc a;
    std::mt19937_64 rng(seed);
    for (int qi = 0; qi < static_cast<int>(oovs.size()); ++qi) {
        const ss::Task k = ss::pick_content_select_task(tk, oovs, qi, K, rng);
        a.ok += (answer_after_sep(run_task(ops, k, content_embed, enc, enc_w)) == k.answer_byte); ++a.n;
    }
    return a;
}

// #4b CONTAINS reason-over-slots (order-agnostic): query a char -> the slot whose OOV contains it.
Acc eval_contains(const Tokenizer& tk, ScratchOps& ops, const std::vector<std::string>& oovs, int K, unsigned seed,
                  bool content_embed = false, sub0::SlotEncoding enc = sub0::SlotEncoding::MeanPool,
                  const float* enc_w = nullptr) {
    Acc a;
    std::mt19937_64 rng(seed);
    for (int qi = 0; qi < static_cast<int>(oovs.size()); ++qi) {
        const ss::Task k = ss::pick_content_contains_task(tk, oovs, qi, K, rng);
        a.ok += (answer_after_sep(run_task(ops, k, content_embed, enc, enc_w)) == k.answer_byte); ++a.n;
    }
    return a;
}

// #4c CONTAINS via thinking-uncombine (Route A): the model RESOLVES each slot (emits <slot> UNCOMBINE; the
// live interceptor injects the spelling from the binding table) then answers -- NO content embeddings. The
// regime is a mix of THREE outcomes (a slot / ANS_NONE / ANS_ALL), so we tally overall AND per-kind: the
// breakdown shows whether the model learned genuine per-word checking (all three) or a shortcut. `picker`
// selects the task variant (plain resolve-then-answer, or the CoT per-slot-verdict scaffold); train and eval
// must use the MATCHING picker so the generated protocol matches what was trained.
using ReasonPicker = ss::Task (*)(const Tokenizer&, const std::vector<std::string>&, int, int, std::mt19937_64&);
struct ReasonAcc { Acc overall, one, non, uni; };   // non = NONE-kind, uni = ALL-kind (universal)
ReasonAcc eval_contains_reason(const Tokenizer& tk, ScratchOps& ops, const std::vector<std::string>& oovs,
                               int K, unsigned seed, ReasonPicker picker) {
    ReasonAcc a;
    std::mt19937_64 rng(seed);
    for (int qi = 0; qi < static_cast<int>(oovs.size()); ++qi) {
        const ss::Task k = picker(tk, oovs, qi, K, rng);
        const bool ok = (answer_after_sep(run_task(ops, k)) == k.answer_byte);
        a.overall.ok += ok; ++a.overall.n;
        Acc& kind = (k.answer_byte == ss::ANS_NONE) ? a.non : (k.answer_byte == ss::ANS_ALL) ? a.uni : a.one;
        kind.ok += ok; ++kind.n;
    }
    return a;
}

// Single-threaded CharEncoder training: the learned enc_w [C,C] has NO per-thread grad reduction (a
// multi-threaded train_batch would race on enc_w_grad), so this runs the forward+backward loop on one
// thread, accumulating the model grad (reduce_gradients + opt.step) AND the encoder grad (a plain SGD on
// enc_w). use_ce=false trains the plain baseline (no content embeddings) the SAME single-threaded way, so
// the A/B is apples-to-apples. Mirrors train_batch's masking (loss_mask -> LOSS_IGNORE_INDEX targets).
struct EncAdam { std::vector<float> m, v; long t = 0; };
void train_ce_steps(const ss::Dataset& ds, sub0::AdamW& opt, std::vector<float>& enc_w,
                    std::vector<float>& enc_w_grad, EncAdam& ea, float enc_lr, int steps,
                    std::mt19937& rng, bool use_ce, sub0::SlotEncoding enc = sub0::SlotEncoding::CharEncoder) {
    std::vector<int> masked_tgt(kWindowT);
    for (int s = 0; s < steps; ++s) {
        std::fill(enc_w_grad.begin(), enc_w_grad.end(), 0.f);
        opt.zero_grad();
        for (int b = 0; b < kBatch; ++b) {
            const sub0::Window w = sub0::sample_window(rng, kWindowT, ds.tokens.size(),
                                                       std::span<const std::uint64_t>(ds.doc_starts));
            const int Tb = w.len;
            for (int i = 0; i < Tb; ++i) {
                const std::size_t p = w.start + static_cast<std::size_t>(i) + 1;
                masked_tgt[static_cast<std::size_t>(i)] = ds.mask[p] ? ds.tokens[p] : sub0::LOSS_IGNORE_INDEX;
            }
            sub0::ScratchBindings binds{};
            if (use_ce) {
                const std::size_t doc = static_cast<std::size_t>(
                    std::upper_bound(ds.doc_starts.begin(), ds.doc_starts.end(),
                                     static_cast<std::uint64_t>(w.start)) - ds.doc_starts.begin()) - 1;
                binds = sub0::ScratchBindings{ std::span<const std::vector<int>>(ds.doc_bindings[doc]),
                                               enc, enc_w.data(), enc_w_grad.data() };
                sub0::set_scratch_bindings(&binds);
            }
            sub0::graph_reset();
            sub0::Node* logits = sub0::forward(ds.tokens.data() + w.start, Tb);
            sub0::Node* loss   = sub0::cross_entropy(logits, masked_tgt.data());
            sub0::backward(loss, 1.f / static_cast<float>(kBatch));   // seed averages over the batch
            if (use_ce) sub0::set_scratch_bindings(nullptr);
        }
        sub0::reduce_gradients();
        opt.step();
        if (use_ce) {   // AdamW on the encoder weights (matches the model's optimizer class)
            ++ea.t;
            const float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
            const float bc1 = 1.f - std::pow(b1, static_cast<float>(ea.t));
            const float bc2 = 1.f - std::pow(b2, static_cast<float>(ea.t));
            for (std::size_t i = 0; i < enc_w.size(); ++i) {
                const float g = enc_w_grad[i];
                ea.m[i] = b1 * ea.m[i] + (1.f - b1) * g;
                ea.v[i] = b2 * ea.v[i] + (1.f - b2) * g * g;
                enc_w[i] -= enc_lr * (ea.m[i] / bc1) / (std::sqrt(ea.v[i] / bc2) + eps);
            }
        }
    }
}

// Each TEST_CASE must start from a COLD optimizer. build_model re-inits the WEIGHTS deterministically
// (fixed-seed init_weights), but the AdamW moment arenas (g_param_m/g_param_v) are GLOBAL and persist
// across test cases in one process -- a prior test's training would otherwise warm-start this one via
// leftover momentum, inflating convergence (this is exactly why an earlier "multi-binding = 1.000"
// reading was really warm-started by the single-binding test that ran before it). Zero them so every
// experiment is independent + reproducible whether run alone or as part of the suite.
void reset_opt_state() {
    const std::size_t n = sub0::trainable_floats();
    std::fill(sub0::adam_m_ptr(), sub0::adam_m_ptr() + n, 0.f);
    std::fill(sub0::adam_v_ptr(), sub0::adam_v_ptr() + n, 0.f);
}

// `content_embed` on -> each window is trained WITH content-derived slot embeddings: its document's
// bindings (ds.doc_bindings) are threaded to train_batch so a bound scratch slot embeds from its fragments.
// `enc` selects the encoder (MeanPool default, matching every existing call site); Hash has no learned
// params so -- unlike CharEncoder's train_ce_steps -- this stays the normal (possibly multi-threaded)
// train_batch path, no separate single-threaded harness needed. `window` is the training-window width
// (default kWindowT; the K-sweep widens it for long local-CoT traces at high K -- must be <= SEQ_LEN).
void train_steps(const ss::Dataset& ds, sub0::AdamW& opt, int steps, std::mt19937& rng,
                 bool content_embed = false, int window = kWindowT,
                 sub0::SlotEncoding enc = sub0::SlotEncoding::MeanPool) {
    std::vector<std::size_t> starts(kBatch);
    std::vector<int>         lens(kBatch);
    std::vector<sub0::ScratchBindings>        binds(kBatch);
    std::vector<const sub0::ScratchBindings*> binds_ptr(kBatch);
    for (int s = 0; s < steps; ++s) {
        for (int b = 0; b < kBatch; ++b) {
            const sub0::Window w = sub0::sample_window(rng, window, ds.tokens.size(),
                                                       std::span<const std::uint64_t>(ds.doc_starts));
            starts[static_cast<std::size_t>(b)] = w.start;
            lens[static_cast<std::size_t>(b)]   = w.len;
            if (content_embed) {
                const std::size_t doc = static_cast<std::size_t>(
                    std::upper_bound(ds.doc_starts.begin(), ds.doc_starts.end(),
                                     static_cast<std::uint64_t>(w.start)) - ds.doc_starts.begin()) - 1;
                binds[static_cast<std::size_t>(b)] = sub0::ScratchBindings{
                    std::span<const std::vector<int>>(ds.doc_bindings[doc]), enc };
                binds_ptr[static_cast<std::size_t>(b)] = &binds[static_cast<std::size_t>(b)];
            }
        }
        opt.zero_grad();
        (void)sub0::train_batch(ds.tokens.data(), starts.data(), kBatch, window, lens.data(), ds.mask.data(),
                                content_embed ? binds_ptr.data() : nullptr);
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
    ScratchOps ops{&tk, true, {}};

    const ss::OovSplit split = ss::make_oov_split(tk, kOovPool, kDrilledFrac, /*seed=*/2024);
    REQUIRE_FALSE(split.drilled.empty());
    REQUIRE_FALSE(split.held_out.empty());

    ss::DatasetOptions dopt; dopt.tasks_per_oov = 12; dopt.seed = 99;
    const ss::Dataset ds = ss::build_dataset(tk, split, dopt);
    REQUIRE(ds.tokens.size() > static_cast<std::size_t>(kWindowT));

    sub0::build_model();
    reset_opt_state();   // cold optimizer per test (see reset_opt_state)
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
    ScratchOps ops{&tk, true, {}};

    const ss::OovSplit split = ss::make_oov_split(tk, kOovPool, kDrilledFrac, /*seed=*/2024);
    ss::DatasetOptions dopt; dopt.tasks_per_oov = 12; dopt.seed = 99;
    const ss::Dataset ds = ss::build_dataset_multi(tk, split, kK, dopt);
    REQUIRE(ds.tokens.size() > static_cast<std::size_t>(kWindowT));

    sub0::build_model();
    reset_opt_state();   // cold optimizer per test (see reset_opt_state)
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
    ScratchOps ops{&tk, true, {}};

    const ss::OovSplit split = ss::make_oov_split(tk, kOovPool, kDrilledFrac, /*seed=*/2024);
    ss::DatasetOptions dopt; dopt.tasks_per_oov = 12; dopt.seed = 99;
    const ss::Dataset ds = build_ds(tk, split, dopt);
    REQUIRE(ds.tokens.size() > static_cast<std::size_t>(kWindowT));

    sub0::build_model();
    reset_opt_state();   // cold optimizer per test (see reset_opt_state)
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

TEST_CASE("scratchspike: CONTENT reason-over-slots -- select the slot whose OOV starts with a letter", "[.scratchspike]") {
    constexpr int kK = 3;
    const double held = run_scratch_experiment(
        "scratchspike #4 CONTENT reason-over-slots (first-letter, K=3)",
        "(K slots, NO tags; query a first-letter -> the slot whose OOV starts with it: needs slot CONTENT,\n"
        "   which a generic reserved-id embedding lacks -- the case that motivates content-derived embeddings)",
        [](const Tokenizer& tk, const ss::OovSplit& s, const ss::DatasetOptions& o) { return ss::build_dataset_content(tk, s, kK, o); },
        [](const Tokenizer& tk, ScratchOps& ops, const std::vector<std::string>& oovs) { return eval_content(tk, ops, oovs, kK, 7); });
    REQUIRE(std::isfinite(held));
}

// The payoff of the content-derived-embedding engine change: the SAME content task (#4) that fails at
// chance WITHOUT it, trained AND evaluated WITH content-derived slot embeddings (each bound slot embeds
// from its fragments). If the embeddings carry the content signal, held-out should move off 1/K.
TEST_CASE("scratchspike: CONTENT reasoning WITH content-derived slot embeddings (the engine change)", "[.scratchspike]") {
    constexpr int kK = 3;
    Tokenizer tk;
    {
        std::ifstream is(sub0::default_tokenizer(), std::ios::binary);
        REQUIRE(is.good());
        REQUIRE(sub0::tok::deserialize(tk, is));
    }
    REQUIRE(tk.vocab == VOCAB);
    ScratchOps ops{&tk, true, {}};

    const ss::OovSplit split = ss::make_oov_split(tk, kOovPool, kDrilledFrac, /*seed=*/2024);
    ss::DatasetOptions dopt; dopt.tasks_per_oov = 12; dopt.seed = 99;
    const ss::Dataset ds = ss::build_dataset_content(tk, split, kK, dopt);
    REQUIRE(ds.doc_bindings.size() + 1 == ds.doc_starts.size());   // per-doc bindings are carried

    sub0::build_model();
    reset_opt_state();
    std::string report = "\n=== scratchspike #4-CE CONTENT reasoning WITH content-derived embeddings (K=3) ===\n"
                         "  (identical task to #4 which failed at chance ~1/K WITHOUT the engine change)\n";
    sub0::AdamW opt(kLr);
    std::mt19937 rng(1);
    double last_held = 0.0;
    for (int r = 0; r < kEvalRounds; ++r) {
        train_steps(ds, opt, kStepsPerEval, rng, /*content_embed=*/true);
        const Acc d = eval_content(tk, ops, split.drilled,  kK, /*seed=*/7,  /*content_embed=*/true);
        const Acc h = eval_content(tk, ops, split.held_out, kK, /*seed=*/11, /*content_embed=*/true);
        last_held = h.rate();
        char line[160];
        std::snprintf(line, sizeof line, "  step %5d | DRILLED %.3f | HELD-OUT %.3f\n",
                      (r + 1) * kStepsPerEval, d.rate(), h.rate());
        report += line;
    }
    WARN(report);
    REQUIRE(std::isfinite(last_held));
}

// A/B on the ORDER-SENSITIVE content task (#4, "starts with X"): MeanPool (proven above to sit at chance --
// mean-pooling discards order entirely) vs Hash (RoPE-style positional binding, see the
// TODO(order-sensitive-slot-encoding) comment in scratch_slots.hpp and project memory
// meanpool-alternatives-prior-art-and-math). Hash has NO learned params, so both arms use the SAME
// train_steps path (no separate single-threaded harness like CharEncoder needed). If positional binding is
// the missing lever, WITH-Hash should move held-out off ~1/K where WITH-MeanPool could not.
//
// MULTI-SEED: build_model()'s weight init is deterministic (fixed-seed, not parameterized here), so the
// only independently-variable axis is training ORDER (the window-sampling rng) -- run each arm across a
// few different training-order seeds and report all of them plus the mean, rather than trusting a single
// trajectory. A real finding should hold up across all seeds, not just the one originally tried.
TEST_CASE("scratchspike: CONTENT (starts-with) reasoning -- Hash/RoPE positional-binding A/B", "[.scratchspike]") {
    constexpr int kK = 3;
    constexpr unsigned kSeeds[] = {1, 2, 3};
    constexpr int kNumSeeds = 3;
    Tokenizer tk;
    {
        std::ifstream is(sub0::default_tokenizer(), std::ios::binary);
        REQUIRE(is.good());
        REQUIRE(sub0::tok::deserialize(tk, is));
    }
    REQUIRE(tk.vocab == VOCAB);
    ScratchOps ops{&tk, true, {}};

    const ss::OovSplit split = ss::make_oov_split(tk, kOovPool, kDrilledFrac, /*seed=*/2024);
    ss::DatasetOptions dopt; dopt.tasks_per_oov = 12; dopt.seed = 99;
    const ss::Dataset ds = ss::build_dataset_content(tk, split, kK, dopt);
    REQUIRE(ds.doc_bindings.size() + 1 == ds.doc_starts.size());

    auto run = [&](sub0::SlotEncoding enc, unsigned seed, double& drilled) {
        sub0::build_model();
        reset_opt_state();
        sub0::AdamW opt(kLr);
        std::mt19937 rng(seed);
        double held = 0.0;
        for (int r = 0; r < kEvalRounds; ++r) {
            train_steps(ds, opt, kStepsPerEval, rng, /*content_embed=*/true, kWindowT, enc);
            drilled = eval_content(tk, ops, split.drilled,  kK, /*seed=*/7,  /*content_embed=*/true, enc).rate();
            held    = eval_content(tk, ops, split.held_out, kK, /*seed=*/11, /*content_embed=*/true, enc).rate();
        }
        return held;
    };

    std::string report = "\n=== scratchspike #4 CONTENT (starts-with) via Hash/RoPE positional binding "
                         "(K=3) @3000, multi-seed ===\n";
    char line[224];
    double sum_mp = 0.0, sum_hash = 0.0;
    for (unsigned seed : kSeeds) {
        double d_mp = 0.0, d_hash = 0.0;
        const double h_mp   = run(sub0::SlotEncoding::MeanPool, seed, d_mp);
        const double h_hash = run(sub0::SlotEncoding::Hash, seed, d_hash);
        sum_mp += h_mp; sum_hash += h_hash;
        std::snprintf(line, sizeof line,
            "  seed %u | MeanPool: drilled %.3f held-out %.3f | Hash: drilled %.3f held-out %.3f\n",
            seed, d_mp, h_mp, d_hash, h_hash);
        report += line;
    }
    const double mean_mp = sum_mp / static_cast<double>(kNumSeeds);
    const double mean_hash = sum_hash / static_cast<double>(kNumSeeds);
    std::snprintf(line, sizeof line, "  MEAN held-out (chance = 0.333): MeanPool %.3f | Hash %.3f\n",
                 mean_mp, mean_hash);
    report += line;
    WARN(report);
    REQUIRE(std::isfinite(mean_mp));
    REQUIRE(std::isfinite(mean_hash));
}

// A/B on the ORDER-SENSITIVE content task (#4, "starts with X") via ConvPool (Candidate 3: Kim et al.
// char-CNN style width-2 conv + relu + maxpool, see scratch_slots.hpp and project memory
// meanpool-alternatives-prior-art-and-math). Unlike Hash, ConvPool has LEARNED params (enc_w, packed
// [2,C,C]) with no per-thread grad reduction, so this reuses train_ce_steps' single-threaded harness
// (same shape as CharEncoder's own A/B) instead of train_steps' normal multi-threaded path. Multi-seeded
// from the start this time (the Hash spike's own first single run was later shown to be on the favorable
// end of a noisy spread -- see that finding above).
TEST_CASE("scratchspike: CONTENT (starts-with) reasoning -- ConvPool (char-CNN) A/B", "[.scratchspike]") {
    constexpr int kK = 3;
    constexpr unsigned kSeeds[] = {1, 2, 3};
    constexpr int kNumSeeds = 3;
    Tokenizer tk;
    {
        std::ifstream is(sub0::default_tokenizer(), std::ios::binary);
        REQUIRE(is.good());
        REQUIRE(sub0::tok::deserialize(tk, is));
    }
    REQUIRE(tk.vocab == VOCAB);
    ScratchOps ops{&tk, true, {}};

    const ss::OovSplit split = ss::make_oov_split(tk, kOovPool, kDrilledFrac, /*seed=*/2024);
    ss::DatasetOptions dopt; dopt.tasks_per_oov = 12; dopt.seed = 99;
    const ss::Dataset ds = ss::build_dataset_content(tk, split, kK, dopt);
    REQUIRE(ds.doc_bindings.size() + 1 == ds.doc_starts.size());

    const int C = D_MODEL;
    std::vector<float> enc_w(static_cast<std::size_t>(2) * C * C), enc_w_grad(enc_w.size());

    auto run = [&](bool ce, unsigned seed, double& drilled) {
        sub0::build_model();
        reset_opt_state();
        std::mt19937 wr(seed * 1000u + 7u);
        std::normal_distribution<float> wnd(0.f, 1.f / std::sqrt(static_cast<float>(C)));
        for (float& x : enc_w) x = wnd(wr);
        EncAdam ea; ea.m.assign(enc_w.size(), 0.f); ea.v.assign(enc_w.size(), 0.f);
        sub0::AdamW opt(kLr);
        std::mt19937 rng(seed);
        double held = 0.0;
        for (int r = 0; r < kEvalRounds; ++r) {
            train_ce_steps(ds, opt, enc_w, enc_w_grad, ea, /*enc_lr=*/kLr, kStepsPerEval, rng, ce,
                           sub0::SlotEncoding::ConvPool);
            drilled = eval_content(tk, ops, split.drilled,  kK, /*seed=*/7,  ce, sub0::SlotEncoding::ConvPool,
                                   ce ? enc_w.data() : nullptr).rate();
            held    = eval_content(tk, ops, split.held_out, kK, /*seed=*/11, ce, sub0::SlotEncoding::ConvPool,
                                   ce ? enc_w.data() : nullptr).rate();
        }
        return held;
    };

    std::string report = "\n=== scratchspike #4 CONTENT (starts-with) via ConvPool (char-CNN) "
                         "(K=3) @3000, multi-seed ===\n";
    char line[224];
    double sum_plain = 0.0, sum_cp = 0.0;
    for (unsigned seed : kSeeds) {
        double d_plain = 0.0, d_cp = 0.0;
        const double h_plain = run(false, seed, d_plain);
        const double h_cp    = run(true,  seed, d_cp);
        sum_plain += h_plain; sum_cp += h_cp;
        std::snprintf(line, sizeof line,
            "  seed %u | plain: drilled %.3f held-out %.3f | ConvPool: drilled %.3f held-out %.3f\n",
            seed, d_plain, h_plain, d_cp, h_cp);
        report += line;
    }
    const double mean_plain = sum_plain / static_cast<double>(kNumSeeds);
    const double mean_cp = sum_cp / static_cast<double>(kNumSeeds);
    std::snprintf(line, sizeof line, "  MEAN held-out (chance = 0.333): plain %.3f | ConvPool %.3f\n",
                 mean_plain, mean_cp);
    report += line;
    WARN(report);
    REQUIRE(std::isfinite(mean_plain));
    REQUIRE(std::isfinite(mean_cp));
}

// A/B on an ORDER-AGNOSTIC content task (which slot CONTAINS char X): same task/everything, trained+eval'd
// WITHOUT vs WITH content-derived embeddings. FINDING at d128: BOTH stay at chance (~1/K) and match closely
// -- mean-pool embeddings do NOT deliver usable content reasoning here (the model converges to a degenerate
// fixed-slot output; a specific char's presence is too diluted in a mean of ~5 byte rows to extract). This
// is NOT a wiring no-op: "scratch embed: train_batch applies per-window content bindings" proves the content
// bindings change the forward. The result MOTIVATES the reserved learned CharEncoder arm (per-char features)
// over mean-pool -- exactly the pluggable extension the scaffold exists for. (First-letter #4 also needs
// ORDER, which mean-pool discards entirely.)
TEST_CASE("scratchspike: CONTAINS reasoning -- content-derived embeddings A/B (order-agnostic payoff)", "[.scratchspike]") {
    constexpr int kK = 3;
    Tokenizer tk;
    {
        std::ifstream is(sub0::default_tokenizer(), std::ios::binary);
        REQUIRE(is.good());
        REQUIRE(sub0::tok::deserialize(tk, is));
    }
    REQUIRE(tk.vocab == VOCAB);
    ScratchOps ops{&tk, true, {}};

    const ss::OovSplit split = ss::make_oov_split(tk, kOovPool, kDrilledFrac, /*seed=*/2024);
    ss::DatasetOptions dopt; dopt.tasks_per_oov = 12; dopt.seed = 99;
    const ss::Dataset ds = ss::build_dataset_contains(tk, split, kK, dopt);

    auto run = [&](bool ce, double& drilled) {
        sub0::build_model();
        reset_opt_state();
        sub0::AdamW opt(kLr);
        std::mt19937 rng(1);
        double held = 0.0;
        for (int r = 0; r < kEvalRounds; ++r) {
            train_steps(ds, opt, kStepsPerEval, rng, ce);
            held    = eval_contains(tk, ops, split.held_out, kK, /*seed=*/11, ce).rate();
            drilled = eval_contains(tk, ops, split.drilled,  kK, /*seed=*/7,  ce).rate();
        }
        return held;
    };
    double d_without = 0, d_with = 0;
    const double without = run(false, d_without);
    const double with_ce = run(true,  d_with);
    char line[288];
    std::snprintf(line, sizeof line,
        "\n=== scratchspike CONTAINS content reasoning (K=3) @3000 (chance = %.3f) ===\n"
        "  WITHOUT content-embed: drilled %.3f  held-out %.3f\n"
        "  WITH    content-embed: drilled %.3f  held-out %.3f\n",
        1.0 / kK, d_without, without, d_with, with_ce);
    WARN(line);
    REQUIRE(std::isfinite(with_ce));
}

// The real content-embedding test: the SAME order-agnostic CONTAINS task, but the slot embeds via the
// LEARNED CharEncoder (a per-fragment [C,C] projection + relu, sum-pooled) instead of mean-pool -- which
// has capacity to keep a char's presence decodable. A/B: plain (no content) vs CharEncoder, both trained
// single-threaded (the encoder grad has no per-thread reduction). If the learned encoder is the lever,
// WITH should beat plain/chance where mean-pool could not.
TEST_CASE("scratchspike: CONTAINS reasoning -- CharEncoder A/B (learned per-fragment projection)", "[.scratchspike]") {
    constexpr int kK = 3;
    Tokenizer tk;
    {
        std::ifstream is(sub0::default_tokenizer(), std::ios::binary);
        REQUIRE(is.good());
        REQUIRE(sub0::tok::deserialize(tk, is));
    }
    REQUIRE(tk.vocab == VOCAB);
    ScratchOps ops{&tk, true, {}};

    const ss::OovSplit split = ss::make_oov_split(tk, kOovPool, kDrilledFrac, /*seed=*/2024);
    ss::DatasetOptions dopt; dopt.tasks_per_oov = 12; dopt.seed = 99;
    const ss::Dataset ds = ss::build_dataset_contains(tk, split, kK, dopt);

    const int C = D_MODEL;
    std::vector<float> enc_w(static_cast<std::size_t>(C) * C), enc_w_grad(static_cast<std::size_t>(C) * C);

    auto run = [&](bool ce) {
        sub0::build_model();
        reset_opt_state();
        std::mt19937 wr(123);
        std::normal_distribution<float> nd(0.f, 1.f / std::sqrt(static_cast<float>(C)));   // [C,C] linear init
        for (float& x : enc_w) x = nd(wr);
        EncAdam ea; ea.m.assign(enc_w.size(), 0.f); ea.v.assign(enc_w.size(), 0.f);
        sub0::AdamW opt(kLr);
        std::mt19937 rng(1);
        double held = 0.0;
        for (int r = 0; r < kEvalRounds; ++r) {
            train_ce_steps(ds, opt, enc_w, enc_w_grad, ea, /*enc_lr=*/kLr, kStepsPerEval, rng, ce);
            held = eval_contains(tk, ops, split.held_out, kK, /*seed=*/11, ce,
                                 sub0::SlotEncoding::CharEncoder, ce ? enc_w.data() : nullptr).rate();
        }
        return held;
    };
    const double without = run(false);
    const double with_ce = run(true);
    char line[224];
    std::snprintf(line, sizeof line,
        "\n=== scratchspike CONTAINS via CharEncoder (K=3) @3000 (chance = %.3f) ===\n"
        "  WITHOUT content (plain) = %.3f  |  WITH CharEncoder = %.3f\n",
        1.0 / kK, without, with_ce);
    WARN(line);
    REQUIRE(std::isfinite(with_ce));
}

// A produced-context stringifier (bytes as chars; slots as S0/S1..; markers named) -- lets the report
// show what the model actually GENERATES, so a chance result is read as "reasoning too hard" only when
// the resolve protocol (slot -> UNCOMBINE -> injected spelling -> = answer) is well-formed.
std::string trace_str(const std::vector<int>& out) {
    std::string s;
    for (int t : out) {
        if (t == ss::SEP) s += "= ";
        else if (t == ss::ANS_NONE) s += "NONE ";
        else if (t == ss::ANS_ALL) s += "ALL ";
        else if (t == ss::VERD_YES) s += "+ ";
        else if (t == ss::VERD_NO) s += "- ";
        else if (t == ss::Q_HAS) s += "% ";
        else if (t == cas::TOK_UNCOMBINE) s += "<U>";
        else if (t == cas::TOK_UNCOMBINE_END) s += "</U> ";
        else if (t == cas::TOK_EOS) s += "<EOS>";
        else if (ss::scratch_slot(0) <= t && t < ss::scratch_slot(ss::SCRATCH_POOL)) s += "S" + std::to_string(t - ss::scratch_slot(0)) + " ";
        else if (t >= 32 && t < 127) s += static_cast<char>(t);
        else s += "?" + std::to_string(t);
    }
    return s;
}

// Train Route A on a prebuilt dataset (the plain-reason or the CoT-verdict variant) and eval with the
// MATCHING picker; returns PEAK held-out (the capability can peak mid-training then overfit/decay, so
// last-round understates it). Appends per-round overall + per-kind (one/none/all) rates and a few generated
// held-out traces to `report`. The per-kind `one` rate is the localization signal; overall clearing the
// ~0.40 presence-heuristic ceiling without it just means the model detects presence, not which slot.
double route_a_run(const Tokenizer& tk, ScratchOps& ops, const ss::OovSplit& split, int K,
                   const ss::Dataset& ds, ReasonPicker picker, std::string& report, int window = kWindowT,
                   double* out_best_one = nullptr) {
    REQUIRE(ds.tokens.size() > static_cast<std::size_t>(window));
    sub0::build_model();
    reset_opt_state();
    sub0::AdamW opt(kLr);
    std::mt19937 rng(1);
    char head[224];
    std::snprintf(head, sizeof head, "\n--- K=%d (mix 60/20/20; presence-heuristic ceiling ~0.40, lr = %.4f, win %d) ---\n", K, kLr, window);
    report += head;
    double last_held = 0.0, best_held = 0.0, best_one = 0.0;
    int best_step = 0;
    for (int r = 0; r < kEvalRounds; ++r) {
        train_steps(ds, opt, kStepsPerEval, rng, /*content_embed=*/false, window);   // plain masked training
        const ReasonAcc d = eval_contains_reason(tk, ops, split.drilled,  K, /*seed=*/7,  picker);
        const ReasonAcc h = eval_contains_reason(tk, ops, split.held_out, K, /*seed=*/11, picker);
        last_held = h.overall.rate();
        if (h.overall.rate() > best_held) { best_held = h.overall.rate(); best_step = (r + 1) * kStepsPerEval; }
        if (h.one.rate() > best_one) best_one = h.one.rate();
        char line[144];
        std::snprintf(line, sizeof line,
                      "  step %5d | DRILLED %.3f | HELD-OUT %.3f  [one %.2f none %.2f all %.2f]\n",
                      (r + 1) * kStepsPerEval, d.overall.rate(), h.overall.rate(),
                      h.one.rate(), h.non.rate(), h.uni.rate());
        report += line;
    }
    if (out_best_one) *out_best_one = best_one;
    char best[128];
    std::snprintf(best, sizeof best, "  => BEST held-out %.3f @ step %d (best one %.3f, last %.3f)\n",
                  best_held, best_step, best_one, last_held);
    report += best;
    std::mt19937_64 drng(11);
    for (int s = 0; s < 4 && s < static_cast<int>(split.held_out.size()); ++s) {
        const ss::Task k = picker(tk, split.held_out, s, K, drng);
        const int ans = k.answer_byte;
        const std::string want = (ans == ss::ANS_NONE) ? "NONE" : (ans == ss::ANS_ALL) ? "ALL"
                                 : "S" + std::to_string(ans - ss::scratch_slot(0));
        report += "    want " + want + " | " + trace_str(run_task(ops, k)) + "\n";
    }
    return best_held;
}

// ROUTE A payoff: can THINKING-UNCOMBINE solve the content question the bare slot embedding could not?
// Train (PLAIN masked training, NO content embeddings) on the reasoning curriculum where the model
// resolves every slot before answering; eval HELD-OUT OOVs (never bound in training -> a correct answer
// cannot be memorization, only genuine in-context resolution + selection). This is the route the spelling
// spike proved for in-vocab words, applied to per-context scratch bindings.
//
// The regime is ONE-heavy 60/20/20 over ONE (a char in exactly one slot -> that slot) / NONE (ANS_NONE) /
// ALL (ANS_ALL). NONE/ALL exist to KILL the elimination shortcut: with "exactly one matches" guaranteed,
// K=2 collapses to a single check + inference ("not word 0 -> word 1") that does NOT compose to K=3.
//
// FINDING -- the task DECOMPOSES, and the wall is LOCALIZATION, not checking. The resolve PROTOCOL is
// flawless at every K/size (emits <slot> UNCOMBINE, interceptor injects each spelling -- verified in the
// dumped traces, held-out too). On top of that:
//   * PRESENCE DETECTION (NONE / ALL -- is the char in no / every word) is learned nearly PERFECTLY (~1.0)
//     at K=2 AND K=3 -- proof the model now genuinely checks EVERY word (the point of adding NONE/ALL).
//   * LOCALIZATION (ONE -- WHICH slot has it) is the real wall: the per-kind `one` rate stays at ~random
//     slot-guessing (~0.45 at K=2, ~0.25 at K=3) even when ONE is the majority class. With an EQUAL 1/3
//     mix the model doesn't even try -- it takes a "present->ALL, absent->NONE" global-OR heuristic (~0.67
//     overall, one~0.05); ONE-heavy forces it to point, and it points ~randomly. So it can tell you a char
//     is present but not pin down which word -- an argmax/pointer op it can't do at this scale.
// (Earlier ONE-ONLY regime: the old K=2 "0.72" was that elimination shortcut, not localization -- which is
// why it never composed to K=3.) SCALE CAVEAT (why kLr scales with D_MODEL above): a d128-tuned LR does NOT
// transfer -- at 0.003 a 2x-wider d256 collapsed; 0.003*128/256 recovered it. A scale run must re-tune the
// recipe first. OPEN: does localization clear with scale (d256 ONE-heavy), more heads (parallel pointer
// lanes), or a pointer-specific curriculum? Reported via WARN (noisy; peaks then overfits -> BEST held-out
// + per-kind breakdown); the hard assert only pins the run is finite. See docs/SCRATCH_TOKENS.md.
TEST_CASE("scratchspike: CONTAINS reasoning -- thinking-uncombine (Route A, exact resolution)", "[.scratchspike]") {
    Tokenizer tk;
    {
        std::ifstream is(sub0::default_tokenizer(), std::ios::binary);
        REQUIRE(is.good());
        REQUIRE(sub0::tok::deserialize(tk, is));
    }
    REQUIRE(tk.vocab == VOCAB);
    ScratchOps ops{&tk, true, {}};

    const ss::OovSplit split = ss::make_oov_split(tk, kOovPool, kDrilledFrac, /*seed=*/2024);
    ss::DatasetOptions dopt; dopt.tasks_per_oov = 12; dopt.seed = 99;

    std::string report = "\n=== scratchspike CONTAINS via thinking-uncombine (Route A: resolve then select) ===\n"
                         "  (held-out = never-bound OOVs; a correct answer PROVES resolve-then-select, not memorization)\n";
    const double h2 = route_a_run(tk, ops, split, 2, ss::build_dataset_contains_via(tk, split, 2, dopt, ss::pick_content_contains_reason_task),
                                  ss::pick_content_contains_reason_task, report);   // presence solved; localization is the wall
    const double h3 = route_a_run(tk, ops, split, 3, ss::build_dataset_contains_via(tk, split, 3, dopt, ss::pick_content_contains_reason_task),
                                  ss::pick_content_contains_reason_task, report);   // one-rate ~random, not K-specific
    WARN(report);
    REQUIRE(std::isfinite(h2));
    REQUIRE(std::isfinite(h3));
}

// LOCALIZATION SCAFFOLD: the SAME ONE-heavy contains regime, but the model emits a per-slot verdict (+/-)
// right after resolving each slot ("show your work"). This turns the final answer from a hidden global
// recall ("which spelling matched -> which slot") into "copy the slot that got a +", a short-range
// induction -- directly targeting the LOCALIZATION wall the plain-reason regime can't clear (one-rate
// ~random at d128 AND d256). The signal is the per-kind `one` rate: if explicit verdicts crack localization
// it should rise well above random (0.5 at K=2, 0.33 at K=3).
TEST_CASE("scratchspike: CONTAINS reasoning -- chain-of-thought verdicts (localization scaffold)", "[.scratchspike]") {
    Tokenizer tk;
    {
        std::ifstream is(sub0::default_tokenizer(), std::ios::binary);
        REQUIRE(is.good());
        REQUIRE(sub0::tok::deserialize(tk, is));
    }
    REQUIRE(tk.vocab == VOCAB);
    ScratchOps ops{&tk, true, {}};

    const ss::OovSplit split = ss::make_oov_split(tk, kOovPool, kDrilledFrac, /*seed=*/2024);
    ss::DatasetOptions dopt; dopt.tasks_per_oov = 12; dopt.seed = 99;

    std::string report = "\n=== scratchspike CONTAINS via chain-of-thought verdicts (per-slot +/-) ===\n"
                         "  (watch the per-kind `one` rate: does explicit per-slot checking crack localization?)\n";
    const double h2 = route_a_run(tk, ops, split, 2, ss::build_dataset_contains_via(tk, split, 2, dopt, ss::pick_content_contains_cot_task),
                                  ss::pick_content_contains_cot_task, report);
    const double h3 = route_a_run(tk, ops, split, 3, ss::build_dataset_contains_via(tk, split, 3, dopt, ss::pick_content_contains_cot_task),
                                  ss::pick_content_contains_cot_task, report);
    WARN(report);
    REQUIRE(std::isfinite(h2));
    REQUIRE(std::isfinite(h3));
}

// LOCAL-QUERY variant: the CoT verdict, but the queried char is RESTATED immediately before each verdict
// (so the check is over adjacent tokens: <bytes_i> <c> <verdict>) and the redundant front slot-list is
// dropped. This removes BOTH crutches the plain CoT lacked -- far-off query recall AND segment isolation --
// leaving the verdict a purely LOCAL "is this char in these recent bytes". THE crux test: if the per-kind
// `one` rate now clears random, the wall was non-local binding (and localization is crackable by locality);
// if it stays ~random, segment isolation itself is the wall. (Minor confound: also drops the front list, but
// that list is redundant -- the resolve phase re-emits every slot.)
TEST_CASE("scratchspike: CONTAINS reasoning -- CoT with LOCAL query restatement", "[.scratchspike]") {
    Tokenizer tk;
    {
        std::ifstream is(sub0::default_tokenizer(), std::ios::binary);
        REQUIRE(is.good());
        REQUIRE(sub0::tok::deserialize(tk, is));
    }
    REQUIRE(tk.vocab == VOCAB);
    ScratchOps ops{&tk, true, {}};

    const ss::OovSplit split = ss::make_oov_split(tk, kOovPool, kDrilledFrac, /*seed=*/2024);
    ss::DatasetOptions dopt; dopt.tasks_per_oov = 12; dopt.seed = 99;

    std::string report = "\n=== scratchspike CONTAINS via CoT + LOCAL query restatement (<bytes> <c> <verdict>) ===\n"
                         "  (crux: does making the per-slot check purely LOCAL lift the `one` rate off random?)\n";
    const double h2 = route_a_run(tk, ops, split, 2, ss::build_dataset_contains_via(tk, split, 2, dopt, ss::pick_content_contains_cot_local_task),
                                  ss::pick_content_contains_cot_local_task, report);
    const double h3 = route_a_run(tk, ops, split, 3, ss::build_dataset_contains_via(tk, split, 3, dopt, ss::pick_content_contains_cot_local_task),
                                  ss::pick_content_contains_cot_local_task, report);
    WARN(report);
    REQUIRE(std::isfinite(h2));
    REQUIRE(std::isfinite(h3));
}

// CONFOUND CONTROL for the local-query win: drops the front slot-list (like local mode) but does NOT restate
// the query. If the local win were just from the shorter context, this would also lift the `one` rate; if
// (as expected) the restatement is the driver, this stays ~random like the plain CoT -- isolating LOCALITY
// (query adjacent to the segment) as the mechanism that cracks localization.
TEST_CASE("scratchspike: CONTAINS reasoning -- CoT control (drop front list, no restatement)", "[.scratchspike]") {
    Tokenizer tk;
    {
        std::ifstream is(sub0::default_tokenizer(), std::ios::binary);
        REQUIRE(is.good());
        REQUIRE(sub0::tok::deserialize(tk, is));
    }
    REQUIRE(tk.vocab == VOCAB);
    ScratchOps ops{&tk, true, {}};

    const ss::OovSplit split = ss::make_oov_split(tk, kOovPool, kDrilledFrac, /*seed=*/2024);
    ss::DatasetOptions dopt; dopt.tasks_per_oov = 12; dopt.seed = 99;

    std::string report = "\n=== scratchspike CONTAINS CoT control (front list dropped, NO restatement) ===\n"
                         "  (isolates the confound: if `one` stays ~random, the restatement -- not the drop -- is the driver)\n";
    const double h2 = route_a_run(tk, ops, split, 2, ss::build_dataset_contains_via(tk, split, 2, dopt, ss::pick_content_contains_cot_ctrl_task),
                                  ss::pick_content_contains_cot_ctrl_task, report);
    const double h3 = route_a_run(tk, ops, split, 3, ss::build_dataset_contains_via(tk, split, 3, dopt, ss::pick_content_contains_cot_ctrl_task),
                                  ss::pick_content_contains_cot_ctrl_task, report);
    WARN(report);
    REQUIRE(std::isfinite(h2));
    REQUIRE(std::isfinite(h3));
}

// MAXIMAL-K SWEEP: does the local-grounding win HOLD as K grows toward the scratch-pool limit (SCRATCH_POOL
// slots), or does the final aggregation ("which of K verdicts is the +") break down? Because the per-slot
// check is LOCAL, per-slot accuracy should be K-independent; the risk is the argmax-over-K-verdicts copy.
// Each K needs a longer trace (~5 + 11K worst case), so this wants a wide-SEQ_LEN build (run at seq128 for
// the full K=2..6); it auto-skips any K whose worst-case trace would not fit SEQ_LEN. The signal is the
// per-K peak held-out `one` rate (random = 1/K); a clean run stays high across all K.
TEST_CASE("scratchspike: CONTAINS reasoning -- maximal-K sweep (local grounding, K=2..pool)", "[.scratchspike]") {
    Tokenizer tk;
    {
        std::ifstream is(sub0::default_tokenizer(), std::ios::binary);
        REQUIRE(is.good());
        REQUIRE(sub0::tok::deserialize(tk, is));
    }
    REQUIRE(tk.vocab == VOCAB);
    ScratchOps ops{&tk, true, {}};

    const ss::OovSplit split = ss::make_oov_split(tk, kOovPool, kDrilledFrac, /*seed=*/2024);
    ss::DatasetOptions dopt; dopt.tasks_per_oov = 12; dopt.seed = 99;

    const int window = SEQ_LEN - 1;   // widest training window this model allows (whole-doc for these traces)
    std::string report = "\n=== scratchspike CONTAINS maximal-K sweep (local grounding; SEQ_LEN=" +
                         std::to_string(SEQ_LEN) + ", pool=" + std::to_string(ss::SCRATCH_POOL) + ") ===\n"
                         "  (per-K peak held-out `one` rate; random = 1/K. Does localization hold as K grows?)\n";
    std::string summary = "  -- summary (K : peak held-out one-rate | random 1/K) --\n";
    bool ran_any = false;
    for (int K = 2; K <= ss::SCRATCH_POOL; ++K) {
        const int worst_trace = 5 + 11 * K;                 // 2 (query) + K*(<=11) + 3 (answer)
        if (worst_trace >= SEQ_LEN) {                       // would overflow context -> skip (need a wider build)
            summary += "    K=" + std::to_string(K) + " : SKIPPED (trace ~" + std::to_string(worst_trace) +
                       " >= SEQ_LEN " + std::to_string(SEQ_LEN) + ")\n";
            continue;
        }
        double best_one = 0.0;
        const double ov = route_a_run(tk, ops, split, K,
                                      ss::build_dataset_contains_via(tk, split, K, dopt, ss::pick_content_contains_cot_local_task),
                                      ss::pick_content_contains_cot_local_task, report, window, &best_one);
        char sline[96];
        std::snprintf(sline, sizeof sline, "    K=%d : one %.3f  (overall %.3f | random %.3f)\n",
                      K, best_one, ov, 1.0 / K);
        summary += sline;
        ran_any = true;
    }
    WARN(report + "\n" + summary);
    REQUIRE(ran_any);
}

// The MERGE validation: train on the exact production curriculum (build_dataset_scratch with contains_k>0 --
// the resolution + associative + CONTENT-reasoning mix that --scratch-mix now blends) and confirm ALL THREE
// held-out capabilities are learned together, with no destructive interference. Uses contains_k=3 so the
// local-CoT content trace (~38 tokens) fits the default window.
TEST_CASE("scratchspike: COMBINED production curriculum -- resolution + associative + content together", "[.scratchspike]") {
    constexpr int kK = 3;
    constexpr int kContainsK = 3;
    Tokenizer tk;
    {
        std::ifstream is(sub0::default_tokenizer(), std::ios::binary);
        REQUIRE(is.good());
        REQUIRE(sub0::tok::deserialize(tk, is));
    }
    REQUIRE(tk.vocab == VOCAB);
    ScratchOps ops{&tk, true, {}};

    const ss::OovSplit split = ss::make_oov_split(tk, kOovPool, kDrilledFrac, /*seed=*/2024);
    ss::DatasetOptions dopt; dopt.tasks_per_oov = 12; dopt.seed = 99;
    const ss::Dataset ds = ss::build_dataset_scratch(tk, split, kK, dopt, kContainsK);

    sub0::build_model();
    reset_opt_state();
    std::string report = "\n=== scratchspike COMBINED production curriculum (resolution + associative + content, K=3) ===\n"
                         "  (one model, the mix --scratch-mix trains; ALL THREE held-out capabilities must hold together)\n";
    sub0::AdamW opt(kLr);
    std::mt19937 rng(1);
    // A production-realistic budget: content and resolution are BOTH hard and compete, so a short run can't
    // max both (a 3000-step run starved whichever got the minority share). Real training is 10x+ longer.
    constexpr int kRounds = 12, kStepsPer = 600;   // 7200 steps
    double last_res = 0.0;
    for (int r = 0; r < kRounds; ++r) {
        train_steps(ds, opt, kStepsPer, rng);
        const Acc res = eval_multi(tk, ops, split.held_out, kK, /*seed=*/7);
        const Acc rec = eval_select(tk, ops, split.held_out, kK, /*seed=*/11);
        const ReasonAcc con = eval_contains_reason(tk, ops, split.held_out, kContainsK, /*seed=*/13,
                                                   ss::pick_content_contains_cot_local_task);
        last_res = res.rate();
        char line[200];
        std::snprintf(line, sizeof line,
                      "  step %5d | HELD-OUT resolution=%.3f  associative=%.3f  content[one %.2f/all %.3f]\n",
                      (r + 1) * kStepsPer, res.rate(), rec.rate(), con.one.rate(), con.overall.rate());
        report += line;
    }
    WARN(report);
    REQUIRE(std::isfinite(last_res));
}
