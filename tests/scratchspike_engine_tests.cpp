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
constexpr float  kLr           = 0.003f;

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

// #4 CONTENT reason-over-slots: query a first-letter -> the slot whose OOV starts with it (needs content).
Acc eval_content(const Tokenizer& tk, ScratchOps& ops, const std::vector<std::string>& oovs, int K, unsigned seed,
                 bool content_embed = false) {
    Acc a;
    std::mt19937_64 rng(seed);
    for (int qi = 0; qi < static_cast<int>(oovs.size()); ++qi) {
        const ss::Task k = ss::pick_content_select_task(tk, oovs, qi, K, rng);
        a.ok += (answer_after_sep(run_task(ops, k, content_embed)) == k.answer_byte); ++a.n;
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

// Single-threaded CharEncoder training: the learned enc_w [C,C] has NO per-thread grad reduction (a
// multi-threaded train_batch would race on enc_w_grad), so this runs the forward+backward loop on one
// thread, accumulating the model grad (reduce_gradients + opt.step) AND the encoder grad (a plain SGD on
// enc_w). use_ce=false trains the plain baseline (no content embeddings) the SAME single-threaded way, so
// the A/B is apples-to-apples. Mirrors train_batch's masking (loss_mask -> LOSS_IGNORE_INDEX targets).
struct EncAdam { std::vector<float> m, v; long t = 0; };
void train_ce_steps(const ss::Dataset& ds, sub0::AdamW& opt, std::vector<float>& enc_w,
                    std::vector<float>& enc_w_grad, EncAdam& ea, float enc_lr, int steps,
                    std::mt19937& rng, bool use_ce) {
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
                                               sub0::SlotEncoding::CharEncoder, enc_w.data(), enc_w_grad.data() };
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
void train_steps(const ss::Dataset& ds, sub0::AdamW& opt, int steps, std::mt19937& rng, bool content_embed = false) {
    std::vector<std::size_t> starts(kBatch);
    std::vector<int>         lens(kBatch);
    std::vector<sub0::ScratchBindings>        binds(kBatch);
    std::vector<const sub0::ScratchBindings*> binds_ptr(kBatch);
    for (int s = 0; s < steps; ++s) {
        for (int b = 0; b < kBatch; ++b) {
            const sub0::Window w = sub0::sample_window(rng, kWindowT, ds.tokens.size(),
                                                       std::span<const std::uint64_t>(ds.doc_starts));
            starts[static_cast<std::size_t>(b)] = w.start;
            lens[static_cast<std::size_t>(b)]   = w.len;
            if (content_embed) {
                const std::size_t doc = static_cast<std::size_t>(
                    std::upper_bound(ds.doc_starts.begin(), ds.doc_starts.end(),
                                     static_cast<std::uint64_t>(w.start)) - ds.doc_starts.begin()) - 1;
                binds[static_cast<std::size_t>(b)] = sub0::ScratchBindings{
                    std::span<const std::vector<int>>(ds.doc_bindings[doc]), sub0::SlotEncoding::MeanPool };
                binds_ptr[static_cast<std::size_t>(b)] = &binds[static_cast<std::size_t>(b)];
            }
        }
        opt.zero_grad();
        (void)sub0::train_batch(ds.tokens.data(), starts.data(), kBatch, kWindowT, lens.data(), ds.mask.data(),
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

// The MERGE validation: train on the exact production curriculum (build_dataset_scratch -- the 50/50 mix
// of resolution + associative reasoning that --scratch-mix blends) and confirm BOTH capabilities are
// learned together (no destructive interference between the two working features at this scale).
TEST_CASE("scratchspike: COMBINED production curriculum -- resolution + associative reasoning together", "[.scratchspike]") {
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
    const ss::Dataset ds = ss::build_dataset_scratch(tk, split, kK, dopt);

    sub0::build_model();
    reset_opt_state();
    std::string report = "\n=== scratchspike COMBINED production curriculum (resolution + associative, K=3) ===\n"
                         "  (one model, the 50/50 mix --scratch-mix trains; both held-out capabilities must hold)\n";
    sub0::AdamW opt(kLr);
    std::mt19937 rng(1);
    double last_res = 0.0;
    for (int r = 0; r < kEvalRounds; ++r) {
        train_steps(ds, opt, kStepsPerEval, rng);
        const Acc res = eval_multi(tk, ops, split.held_out, kK, /*seed=*/7);
        const Acc rec = eval_select(tk, ops, split.held_out, kK, /*seed=*/11);
        last_res = res.rate();
        char line[160];
        std::snprintf(line, sizeof line, "  step %5d | HELD-OUT resolution=%.3f  associative=%.3f\n",
                      (r + 1) * kStepsPerEval, res.rate(), rec.rate());
        report += line;
    }
    WARN(report);
    REQUIRE(std::isfinite(last_res));
}
