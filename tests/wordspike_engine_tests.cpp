// wordspike_engine_tests.cpp -- the engine-side proof for the natural-prose scratch-slot spike
// (sub0/wordspike.hpp), in two parts:
//
//   1. A fast, always-run differential ("[wordspike]") proving the live word_collapse splice
//      (decode.hpp's kv_decode_generate) is mechanically correct: given a context ending in a just-
//      completed TOK_SPELL_START..END span, it truncates the span and substitutes the callback's
//      replacement, leaving the prefix untouched. No model, no training.
//   2. A hidden capstone ("[.wordspike]") training two real models to test whether HARNESS-DRIVEN
//      name-collapse (mention 1 spelled in full, mention 2 replaced by the bound slot -- no request
//      from the model, mirroring repeatspike.hpp's own proven mechanism) composes with op_curriculum's
//      arithmetic-collapse inside REAL GSM8K-shaped prose (built via tok::encode, so genuine
//      TOK_CAP/TOK_SPELL_START/END/TOK_JOIN markers appear exactly as real tokenization produces them),
//      by comparing op-delegation accuracy under COLLAPSE (shorter, slot-based context) vs FUZZY (longer,
//      fully-spelled context). The collapse decision itself is resolved by the harness BEFORE generation
//      starts in this eval, matching training's own teacher-forced shape and repeatspike's precedent --
//      see eval_arm's header comment for why asking free generation to produce the collapse point itself
//      would test an unsupported claim (mask=0 means neither arm ever trains that position at all).
// Tagged "[.wordspike]": trains two separate models, not in the default ctest sweep -- invoke with
// `sub0_tests "[wordspike]"`.

#include <catch2/catch_test_macros.hpp>

#include "sub0/core.hpp"
#include "sub0/decode.hpp"
#include "sub0/scratch.hpp"
#include "sub0/wordspike.hpp"
#include "sub0/nodes.hpp"
#include "sub0/node_frame.hpp"
#include "sub0/window.hpp"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <random>
#include <span>
#include <string>
#include <vector>

namespace {

namespace ws  = sub0::wordspike;
namespace nd  = sub0::nodes;
namespace cas = sub0::casing;
using sub0::tok::Tokenizer;

constexpr int   kBatch        = 16;
// Natural-prose documents (two invented names + connective prose + digits + an op frame) run much
// longer than repeatspike's terse control-symbol passages -- measured max 91 tokens (FUZZY arm, names
// spelled twice) over 200 sampled documents, so 100 gives safe headroom while staying far cheaper than
// SEQ_LEN-1 would. Sizing it too small (originally 64) let sample_window truncate BEFORE the graded
// "= [op math] <result>" ending for longer documents, starving the op-routing signal entirely --
// exactly the failure mode gsm8k_engine_tests.cpp's own train() avoids by using SEQ_LEN-1 (that
// curriculum has no comparable per-document length variance to exploit for a tighter bound).
constexpr int   kWindowT      = 100;
constexpr int   kEvalRounds   = 10;
constexpr int   kStepsPerEval = 300;
constexpr int   kMaxDigits    = 2;
// Same LR-vs-D_MODEL scaling every other spike test in this codebase uses (scratchspike/repeatspike).
constexpr float kLr           = 0.003f * (128.0f / static_cast<float>(D_MODEL));

void reset_opt_state() {
    const std::size_t n = sub0::trainable_floats();
    std::fill(sub0::adam_m_ptr(), sub0::adam_m_ptr() + n, 0.f);
    std::fill(sub0::adam_v_ptr(), sub0::adam_v_ptr() + n, 0.f);
}

void train_steps(const ws::Dataset& ds, sub0::AdamW& opt, int steps, std::mt19937& rng) {
    std::vector<std::size_t> starts(kBatch);
    for (int s = 0; s < steps; ++s) {
        for (int b = 0; b < kBatch; ++b)
            starts[static_cast<std::size_t>(b)] =
                sub0::sample_window(rng, kWindowT, ds.tokens.size(),
                                    std::span<const std::uint64_t>(ds.doc_starts)).start;
        opt.zero_grad();
        (void)sub0::train_batch(ds.tokens.data(), starts.data(), kBatch, kWindowT, nullptr, ds.mask.data(), nullptr);
        opt.step();
    }
}

struct Score { int op_ok = 0, n = 0; };

// Shared eval body: build the SAME production op-collapse callback gen_stage.cpp's op_on branch and
// blended_capstone_engine_tests.cpp's eval_op use, and grade the op-delegation continuation -- the
// downstream capability that composes with name-collapse in the same document.
//
// The name-collapse decision itself is resolved by the HARNESS before generation starts, exactly
// mirroring emit_doc's own teacher-forced shape and repeatspike.hpp's repeat_task_collapse (mentions 2/3
// are fed as already-resolved slot tokens, never something free generation is asked to produce -- see
// wordspike.hpp's EvalProblem comment for why an earlier version that asked generation to produce the
// collapse itself was testing an unsupported claim). `collapse`: true builds the prompt with Name1's
// second mention as the bound slot token (pre-bound in `scratch`, matching training's mask=0 slot
// insertion); false (FUZZY control) builds it with Name1's second mention spelled out in full, matching
// FUZZY's longer, uncollapsed context. Both arms then generate identically from "A+B=" onward -- the
// comparison is whether the shorter, collapsed context helps or hurts the SAME op-delegation skill.
Score eval_arm(const Tokenizer& tk, bool collapse, int n_eval, unsigned seed) {
    Score sc;
    std::mt19937_64 ev(seed);
    std::mt19937 grng(0);
    for (int i = 0; i < n_eval; ++i) {
        const ws::EvalProblem pr = ws::gen_eval(ev, tk, kMaxDigits);

        sub0::ScratchTable scratch;
        std::vector<int> ctx;
        auto gtext = [&](const std::string& s) { for (int t : sub0::tok::encode(tk, s)) ctx.push_back(t); };
        gtext(pr.name1 + " has "); gtext(pr.A + " eggs. ");
        gtext(pr.name2 + " gives " + pr.B + " eggs to ");
        if (collapse) {
            std::vector<int> frags; for (char c : pr.name1) frags.push_back(static_cast<unsigned char>(c));
            scratch.bind(sub0::SCRATCH_SLOT_BASE, std::move(frags));
            ctx.push_back(sub0::SCRATCH_SLOT_BASE);
        } else {
            gtext(pr.name1);
        }
        gtext(". " + pr.A + "+" + pr.B + "=");
        if (static_cast<int>(ctx.size()) + 16 >= SEQ_LEN) continue;

        auto inner = nd::make_compute_callback(nd::builtin(), [&scratch](int s) { return scratch.value(s); });
        auto compute = [&scratch, inner](const std::vector<int>& c) -> std::vector<int> {
            const std::vector<int> r = inner(c);
            if (r.empty()) return r;
            std::string val;
            for (int t : r) if (t != nd::FRAME_OPEN && t != nd::FRAME_CLOSE) val.push_back(static_cast<char>(t));
            const int idx = static_cast<int>(scratch.bindings.size());
            if (val.empty() || idx >= sub0::SCRATCH_SLOT_COUNT) return r;
            std::vector<int> frags; for (char ch : val) frags.push_back(static_cast<unsigned char>(ch));
            scratch.bind(sub0::SCRATCH_SLOT_BASE + idx, std::move(frags));
            return { sub0::SCRATCH_SLOT_BASE + idx };
        };

        sub0::kv_decode_generate(ctx, 24, 1.f, 1, grng, cas::TOK_EOS, false, {}, {}, {}, compute,
                                 nd::FRAME_CLOSE, {});
        ++sc.n;

        // Op-delegation: the LAST bound slot's value should equal gold (mirrors blended_capstone's eval_op).
        const bool did = !scratch.bindings.empty();
        const std::string got = did ? scratch.value(sub0::SCRATCH_SLOT_BASE +
                                                     static_cast<int>(scratch.bindings.size()) - 1)
                                    : std::string{};
        sc.op_ok += (got == pr.gold);
    }
    return sc;
}

}  // namespace

// Fast, always-run differential (no training, no dependence on model quality): proves the retroactive
// resize-and-refeed splice itself is mechanically correct -- decode.hpp's kv_decode_generate, given a
// context that ends exactly at a just-completed TOK_SPELL_START..END compound word and a word_collapse
// callback that reports a recurrence, must truncate the span and replace it with the callback's
// replacement token(s), leaving everything before the span untouched. Uses the "forward-pass resolution
// of a prompt that ends in a completed marker" path (decode.hpp) so this needs no real sampling to land
// on a specific repeat -- the hand-built context IS the "just generated" state.
TEST_CASE("wordspike: the live word-collapse splice truncates a completed SPELL span and substitutes "
         "the replacement, leaving the prefix untouched", "[wordspike]") {
    sub0::build_model();
    const int p1 = 10, p2 = 20, p3 = 30;                 // an arbitrary 3-piece "word", byte-range ids
    const int replacement = sub0::SCRATCH_SLOT_BASE;      // the substituted token the callback reports
    std::vector<int> ctx = { 5, 6, 7 };                   // some unrelated prefix
    const std::size_t open = ctx.size();                  // where TOK_SPELL_START will land
    ctx.push_back(cas::TOK_SPELL_START);
    ctx.push_back(p1); ctx.push_back(p2); ctx.push_back(p3);
    ctx.push_back(cas::TOK_SPELL_END);
    const std::vector<int> prefix = { 5, 6, 7 };          // expected to survive untouched

    bool saw_span = false;
    std::vector<int> seen_span;
    auto word_collapse = [&](const std::vector<int>& span) -> std::vector<int> {
        saw_span = true; seen_span = span;
        return { replacement };
    };
    std::mt19937 rng(0);
    sub0::kv_decode_generate(ctx, /*n=*/1, 1.f, 1, rng, /*eos_id=*/-1, /*use_gpu=*/false,
                             /*on_token=*/{}, /*expand=*/{}, /*combine=*/{}, /*compute=*/{},
                             /*compute_marker=*/-1, word_collapse);

    REQUIRE(saw_span);
    CHECK(seen_span == std::vector<int>{ p1, p2, p3 });
    REQUIRE(ctx.size() >= open + 1);
    CHECK(std::equal(prefix.begin(), prefix.end(), ctx.begin()));      // prefix untouched
    CHECK(ctx[open] == replacement);                                  // the whole SPELL span -> one token
    // The 5-slot span (START + 3 pieces + END) collapsed to 1 -- ctx is strictly shorter at this point
    // than it would be with the span left alone, before any further (n=1) generation appends past it.
}

TEST_CASE("wordspike: name-collapse vs fuzzy re-spelling in natural GSM8K-style prose, "
         "composed with op-delegation", "[.wordspike]") {
    Tokenizer tk;
    {
        std::ifstream is(sub0::default_tokenizer(), std::ios::binary);
        REQUIRE(is.good());
        REQUIRE(sub0::tok::deserialize(tk, is));
    }
    REQUIRE(tk.vocab == VOCAB);

    ws::Options opt; opt.seed = 20260718; opt.n_examples = 3000; opt.max_digits = kMaxDigits;
    const ws::Dataset fuzzy_ds    = ws::build_dataset(tk, opt, ws::Arm::Fuzzy);
    const ws::Dataset collapse_ds = ws::build_dataset(tk, opt, ws::Arm::Collapse);
    REQUIRE(fuzzy_ds.tokens.size() > static_cast<std::size_t>(kWindowT));
    REQUIRE(collapse_ds.tokens.size() > static_cast<std::size_t>(kWindowT));

    std::string report = "\n=== wordspike: op-delegation accuracy with FUZZY (name re-spelled, longer "
        "context) vs COLLAPSE (harness-bound slot, shorter context) in natural prose (d" +
        std::to_string(D_MODEL) + ") ===\n";

    double fuzzy_op = 0.0, collapse_op = 0.0;

    sub0::build_model(); reset_opt_state();
    {
        sub0::AdamW opt_fuzzy(kLr);
        std::mt19937 trng(1);
        report += "  FUZZY:   ";
        for (int r = 0; r < kEvalRounds; ++r) {
            train_steps(fuzzy_ds, opt_fuzzy, kStepsPerEval, trng);
            const Score s = eval_arm(tk, false, 60, 4242u + static_cast<unsigned>(r));
            fuzzy_op = s.n ? double(s.op_ok) / s.n : 0.0;
            char c[32];
            std::snprintf(c, sizeof c, " s%d(op=%.2f)", (r + 1) * kStepsPerEval, fuzzy_op);
            report += c;
        }
        report += "\n";
    }

    sub0::build_model(); reset_opt_state();
    {
        sub0::AdamW opt_collapse(kLr);
        std::mt19937 trng(1);
        report += "  COLLAPSE:";
        for (int r = 0; r < kEvalRounds; ++r) {
            train_steps(collapse_ds, opt_collapse, kStepsPerEval, trng);
            const Score s = eval_arm(tk, true, 60, 4242u + static_cast<unsigned>(r));
            collapse_op = s.n ? double(s.op_ok) / s.n : 0.0;
            char c[32];
            std::snprintf(c, sizeof c, " s%d(op=%.2f)", (r + 1) * kStepsPerEval, collapse_op);
            report += c;
        }
        report += "\n";
    }
    report += "  (fresh random names/digits every eval draw -- effectively held-out by construction, "
             "same intent as scratchspike/repeatspike's explicit drilled/held-out split; the name-carry "
             "mechanism itself is proven separately by the fast differential test above and by a real "
             "CLI train+gen session, not by free generation here -- see EvalProblem's header comment)\n";
    WARN(report);

    CHECK(std::isfinite(fuzzy_op));
    CHECK(std::isfinite(collapse_op));
    CHECK(collapse_op > 0.3);   // op-delegation must still work when composed with a pre-collapsed name
}
