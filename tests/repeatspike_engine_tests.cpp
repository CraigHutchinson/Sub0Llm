// repeatspike_engine_tests.cpp -- the engine-side A/B for the repeat-mention spike (sub0/repeatspike.hpp):
// does HARNESS-DRIVEN collapse of a REPEATED mention (mention 1 spelled in full, mentions 2/3 replaced by
// the bound slot -- no request from the model) beat raw FUZZY copying (all three mentions spelled out,
// answer via plain in-context lookup), on HELD-OUT OOVs never bound in training? Tagged "[.repeatspike]"
// (hidden): trains two separate models, not in the default ctest sweep -- invoke with
// `sub0_tests "[repeatspike]"`.

#include <catch2/catch_test_macros.hpp>

#include "sub0/core.hpp"
#include "sub0/decode.hpp"
#include "sub0/scratch.hpp"
#include "sub0/repeatspike.hpp"
#include "sub0/window.hpp"

#include <algorithm>
#include <fstream>
#include <random>
#include <span>
#include <string>
#include <vector>

namespace {

namespace ss  = sub0::scratchspike;
namespace rs  = sub0::repeatspike;
namespace cas = sub0::casing;

constexpr int    kBatch        = 16;
constexpr int    kWindowT      = 40;
constexpr int    kEvalRounds   = 10;
constexpr int    kStepsPerEval = 300;
constexpr int    kOovPool      = 400;
constexpr double kDrilledFrac  = 0.7;
// Same LR-vs-D_MODEL scaling scratchspike_engine_tests.cpp uses (see its comment) -- this spike shares
// the same task shape (nth-char resolution) and model scale sensitivity.
constexpr float  kLr           = 0.003f * (128.0f / static_cast<float>(D_MODEL));

void reset_opt_state() {
    const std::size_t n = sub0::trainable_floats();
    std::fill(sub0::adam_m_ptr(), sub0::adam_m_ptr() + n, 0.f);
    std::fill(sub0::adam_v_ptr(), sub0::adam_v_ptr() + n, 0.f);
}

void train_steps(const ss::Dataset& ds, sub0::AdamW& opt, int steps, std::mt19937& rng) {
    std::vector<std::size_t> starts(kBatch);
    for (int s = 0; s < steps; ++s) {
        for (int b = 0; b < kBatch; ++b)
            starts[static_cast<std::size_t>(b)] =
                sub0::sample_window(rng, kWindowT, ds.tokens.size(), std::span<const std::uint64_t>(ds.doc_starts)).start;
        opt.zero_grad();
        (void)sub0::train_batch(ds.tokens.data(), starts.data(), kBatch, kWindowT, nullptr, ds.mask.data(), nullptr);
        opt.step();
    }
}

int answer_after_sep(const std::vector<int>& out) {
    for (std::size_t i = 0; i + 1 < out.size(); ++i) if (out[i] == ss::SEP) return out[i + 1];
    return -1;
}

struct Acc { int ok = 0, n = 0; double rate() const { return n ? static_cast<double>(ok) / n : 0.0; } };

// FUZZY held-out eval: no interceptor at all -- the answer must come from plain in-context lookup over
// the three fully-spelled mentions already in the prompt.
Acc eval_fuzzy(const std::vector<std::string>& held_out, std::mt19937_64& rng) {
    Acc a;
    std::uniform_int_distribution<std::size_t> pick(0, held_out.size() - 1);
    for (int i = 0; i < 60; ++i) {
        const std::string& oov = held_out[pick(rng)];
        std::uniform_int_distribution<int> pn(0, static_cast<int>(oov.size()) - 1);
        const int n = pn(rng);
        const ss::Task t = rs::repeat_task_fuzzy(oov, n);
        std::vector<int> ctx = t.prompt;
        std::mt19937 grng(0);
        sub0::kv_decode_generate(ctx, kWindowT, 1.f, 1, grng, cas::TOK_EOS, false);
        a.ok += (answer_after_sep(ctx) == t.answer_byte); ++a.n;
    }
    return a;
}

// COLLAPSE held-out eval: the SAME production ScratchTable interceptor scratchspike_engine_tests.cpp
// uses -- pre-bind slot 0 to the held-out OOV (mirrors mention 1 having just bound it), then the model
// must UNCOMBINE to resolve mentions 2/3's slot reference and answer.
Acc eval_collapse(sub0::ScratchTable& ops, const std::vector<std::string>& held_out, std::mt19937_64& rng) {
    Acc a;
    std::uniform_int_distribution<std::size_t> pick(0, held_out.size() - 1);
    for (int i = 0; i < 60; ++i) {
        const std::string& oov = held_out[pick(rng)];
        std::uniform_int_distribution<int> pn(0, static_cast<int>(oov.size()) - 1);
        const int n = pn(rng);
        const ss::Task t = rs::repeat_task_collapse(oov, n);
        ops.reset();
        ops.bind(ss::scratch_slot(0), ss::oov_bytes(oov));
        std::vector<int> ctx = t.prompt;
        std::mt19937 grng(0);
        sub0::kv_decode_generate(ctx, kWindowT, 1.f, 1, grng, cas::TOK_EOS, false, /*on_token=*/{},
                                 [&](int tok) { return ops.expand(tok); },
                                 [&](const std::vector<int>& f) { return ops.combine(f); });
        a.ok += (answer_after_sep(ctx) == t.answer_byte); ++a.n;
    }
    return a;
}

}  // namespace

TEST_CASE("repeatspike: harness-collapse vs fuzzy repeat-copy on held-out OOVs", "[.repeatspike]") {
    sub0::tok::Tokenizer tk;
    {
        std::ifstream is(sub0::default_tokenizer(), std::ios::binary);
        REQUIRE(is.good());
        REQUIRE(sub0::tok::deserialize(tk, is));
    }
    REQUIRE(tk.vocab == VOCAB);
    const ss::OovSplit split = ss::make_oov_split(tk, kOovPool, kDrilledFrac, /*seed=*/2027);
    ss::DatasetOptions opt; opt.tasks_per_oov = 12; opt.seed = 20260716;
    const ss::Dataset fuzzy_ds    = rs::build_dataset_fuzzy(split, opt);
    const ss::Dataset collapse_ds = rs::build_dataset_collapse(split, opt);
    REQUIRE(fuzzy_ds.tokens.size() > static_cast<std::size_t>(kWindowT));
    REQUIRE(collapse_ds.tokens.size() > static_cast<std::size_t>(kWindowT));

    std::string report = "\n=== repeatspike: FUZZY (repeat-copy) vs COLLAPSE (harness-bound, no model "
                         "request) on Nth-char-of-a-3x-repeated-OOV, held-out (d" + std::to_string(D_MODEL) +
                         ") ===\n";

    // Arm A: FUZZY.
    sub0::build_model();
    reset_opt_state();
    {
        sub0::AdamW opt_fuzzy(kLr);
        std::mt19937 trng(1);
        report += "  FUZZY   (all 3 mentions spelled out, plain lookup):\n   ";
        for (int r = 0; r < kEvalRounds; ++r) {
            train_steps(fuzzy_ds, opt_fuzzy, kStepsPerEval, trng);
            std::mt19937_64 ev(4242ULL + static_cast<std::uint64_t>(r));
            const Acc a = eval_fuzzy(split.held_out, ev);
            char c[32]; std::snprintf(c, sizeof c, " s%d=%.2f", (r + 1) * kStepsPerEval, a.rate());
            report += c;
        }
        report += "\n";
    }

    // Arm B: COLLAPSE (fresh model + fresh optimizer state -- no cross-arm contamination).
    sub0::build_model();
    reset_opt_state();
    sub0::ScratchTable ops;
    {
        sub0::AdamW opt_collapse(kLr);
        std::mt19937 trng(1);
        report += "  COLLAPSE (mention 1 full + harness-bound, 2/3 collapsed, resolve via UNCOMBINE):\n   ";
        for (int r = 0; r < kEvalRounds; ++r) {
            train_steps(collapse_ds, opt_collapse, kStepsPerEval, trng);
            std::mt19937_64 ev(4242ULL + static_cast<std::uint64_t>(r));
            const Acc a = eval_collapse(ops, split.held_out, ev);
            char c[32]; std::snprintf(c, sizeof c, " s%d=%.2f", (r + 1) * kStepsPerEval, a.rate());
            report += c;
        }
        report += "\n";
    }
    report += "  (held-out OOVs never bound in training; a correct answer needs tracking/resolution, "
             "not memorisation)\n";
    WARN(report);
    REQUIRE(true);
}
