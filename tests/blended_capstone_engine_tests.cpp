// blended_capstone_engine_tests.cpp -- does ONE model, trained via the REAL production epoch-fair
// scheduler (sub0::sample_blend_staged, blend_schedule.hpp) on a schedule that blends the scratch
// (scratchspike) and op-delegation (op_curriculum) curricula together, retain BOTH capabilities at
// solo-baseline quality? Complements the two single-mechanism capstones (scratchspike_engine_tests.cpp's
// [.scratchspike] tests, gsm8k_engine_tests.cpp's [.gsm8kcapstone]/[.chaincapstone]), which each only
// ever train their model on ONE curriculum alone -- this is the first test that exercises the scheduler
// over >1 generator source at once, the way a real --blend-config run combining "scratchspike" +
// "op_curriculum" sources does (see e.g. models/ce256_prod_fixed/blend_schedule.json for a live example
// of the single-source shape this generalizes).
//
// Two TEST_CASEs:
//  * A fast, model-free check that the shared scratch-slot pool (sub0::ScratchTable) allocates
//    independent, non-colliding slots for an OOV binding vs an op-collapse result within one session --
//    the one genuinely NEW correctness risk blending introduces at GEN time. Both mechanisms mint into
//    the SAME finite SCRATCH_SLOT_COUNT range via the same `bindings` vector (scratch.hpp): scratchspike
//    mints on an unrecognized OOV via combine(), op_curriculum mints on a resolved result via gen_stage.cpp's
//    op_on branch (`scratch.bind(SCRATCH_SLOT_BASE + scratch.bindings.size(), ...)`) -- gen_stage.cpp
//    already composes both unconditionally (sp_on/op_on are independent booleans), so this pins that the
//    shared counter genuinely doesn't collide, not just that the code compiles.
//  * The capstone itself (hidden, [.blendedcapstone]): trains three models at a MATCHED per-mechanism
//    step budget -- scratch-only, op-only, and blended -- and scores each on its own solo methodology
//    (scratchspike's drilled/held-out nth-char resolution; op_curriculum's delegation/exact-match via
//    gen_eval, mirroring gen_stage.cpp's sub0_eval_stage collapse callback exactly), so "does blending
//    hurt" is a real A/B against a matched-budget solo baseline trained THIS run, not eyeballed against
//    a different test file's numbers from a different session.

#include <catch2/catch_test_macros.hpp>

#include "sub0/core.hpp"
#include "sub0/decode.hpp"
#include "sub0/scratch.hpp"
#include "sub0/scratchspike.hpp"
#include "sub0/op_curriculum.hpp"
#include "sub0/blend_schedule.hpp"
#include "sub0/nodes.hpp"
#include "sub0/node_frame.hpp"

#include <fstream>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace {

namespace ss  = sub0::scratchspike;
namespace oc  = sub0::op_curriculum;
namespace nd  = sub0::nodes;
namespace cas = sub0::casing;
using sub0::tok::Tokenizer;

constexpr int    kBatch        = 16;
constexpr int    kWindowT      = 40;
constexpr int    kOovPool      = 400;
constexpr double kDrilledFrac  = 0.7;
constexpr int    kOpExamples   = 2000;
constexpr double kChainFrac    = 0.4;
constexpr int    kMaxDigits    = 2;
constexpr int    kEvalRounds   = 8;
constexpr int    kStepsPerEval = 250;   // per-mechanism solo budget; the blended run doubles its TOTAL
                                        // budget so each mechanism still gets ~this much exposure under
                                        // the equal-weight epoch-fair scheduler -- a matched-budget A/B,
                                        // not "blended gets diluted vs solo by construction".
// Same 1/width LR-transfer rule scratchspike_engine_tests.cpp documents (a d128-tuned LR collapses a
// wider model into a degenerate basin) -- kept identical so results are comparable to that file's own.
constexpr float  kLr = 0.003f * (128.0f / static_cast<float>(D_MODEL));

void reset_opt_state() {
    const std::size_t n = sub0::trainable_floats();
    std::fill(sub0::adam_m_ptr(), sub0::adam_m_ptr() + n, 0.f);
    std::fill(sub0::adam_v_ptr(), sub0::adam_v_ptr() + n, 0.f);
}

Tokenizer load_tk() {
    Tokenizer tk;
    std::ifstream is(sub0::default_tokenizer(), std::ios::binary);
    REQUIRE(is.good());
    REQUIRE(sub0::tok::deserialize(tk, is));
    REQUIRE(tk.vocab == VOCAB);
    // eval_op calls sub0::encode() (the engine's global-tokenizer-backed encode, matching gen_stage.cpp's
    // own prompt-encoding call) -- that needs sub0::load_tokenizer() to have populated the global g_tok
    // first (train_stage.cpp's own on-demand-encode precedent, engine_core.cpp:275); the LOCAL `tk` above
    // is a separate copy this file's own dataset builders (ss::build_dataset/oc::build_dataset) consume
    // directly, unrelated to the global one.
    REQUIRE(sub0::load_tokenizer(sub0::default_tokenizer()));
    return tk;
}

// ---- scratch-side scoring: same methodology as scratchspike_engine_tests.cpp's run_task/eval_tasks
// (copied, not shared -- matches this project's convention of self-contained capstone test files; e.g.
// gsm8k_engine_tests.cpp keeps its own local reset_opt_state/byte_encode rather than importing them). ---
using ScratchOps = sub0::ScratchTable;

std::vector<int> run_scratch_task(ScratchOps& ops, const ss::Task& k) {
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
struct Acc { int ok = 0, n = 0; double rate() const { return n ? static_cast<double>(ok) / n : 0.0; } };
Acc eval_scratch_nth(const Tokenizer& tk, ScratchOps& ops, const std::vector<std::string>& oovs) {
    Acc a;
    for (const std::string& oov : oovs) {
        const std::vector<int> bytes = ss::oov_bytes(oov);
        for (int pos = 0; pos < static_cast<int>(bytes.size()); ++pos) {
            const ss::Task k = ss::nth_char_task(tk, oov, pos);
            a.ok += (answer_after_sep(run_scratch_task(ops, k)) == k.answer_byte); ++a.n;
        }
    }
    return a;
}

// ---- op-side scoring: mirrors gen_stage.cpp's sub0_eval_stage EXACTLY (same production collapse
// callback shape -- resolve the op, bind the result to the next free scratch slot, inject that token;
// the last bound slot's value is the model's answer). ----
struct OpScore { int exact = 0, delegated = 0, n = 0; double rate() const { return n ? static_cast<double>(exact) / n : 0.0; } };
OpScore eval_op(int n_eval, int maxdig, unsigned seed) {
    OpScore sc;
    ScratchOps scratch;
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
    std::mt19937_64 ev(seed);
    std::mt19937 grng(0);
    for (int i = 0; i < n_eval; ++i) {
        const oc::EvalProblem pr = oc::gen_eval(ev, maxdig);
        std::vector<int> ctx = sub0::encode(pr.prompt);
        if (static_cast<int>(ctx.size()) + 8 >= SEQ_LEN) continue;
        scratch.reset();
        sub0::kv_decode_generate(ctx, 24, 1.f, 1, grng, cas::TOK_EOS, false, {}, {}, {}, compute, nd::FRAME_CLOSE);
        const bool did = !scratch.bindings.empty();
        const std::string got = did ? scratch.value(sub0::SCRATCH_SLOT_BASE + static_cast<int>(scratch.bindings.size()) - 1)
                                    : std::string{};
        sc.delegated += did;
        sc.exact     += (got == pr.gold);
        ++sc.n;
    }
    return sc;
}

// ---- training: draws windows via the REAL production scheduler (sample_blend_staged) over an
// arbitrary set of BlendSources -- one source degenerates to plain per-window sampling (the solo
// baselines below), two exercises the actual blend a production --blend-config run would drive. No
// content-embed here (deliberately -- that axis is already exhaustively A/B'd in
// scratchspike_engine_tests.cpp; this file isolates the NEW variable, blending two generators). ----
void train_steps(std::vector<sub0::BlendSource>& sources, sub0::ResolvedSchedule& sched,
                 sub0::BlendFairness& fair, sub0::AdamW& opt, int steps, std::mt19937& rng) {
    std::vector<int> data(static_cast<std::size_t>(kBatch) * (kWindowT + 1));
    std::vector<std::uint8_t> mask(data.size());
    std::vector<std::size_t> starts(kBatch);
    std::vector<int> lens(kBatch);
    for (int s = 0; s < steps; ++s) {
        for (int b = 0; b < kBatch; ++b) {
            const sub0::BlendDraw d = sub0::sample_blend_staged(rng, fair, sources, sched, kWindowT,
                                                                /*frac_epoch=*/0.0);   // single "end" stage
            const sub0::BlendSource& src = sources[static_cast<std::size_t>(d.src)];
            const std::size_t base = static_cast<std::size_t>(b) * (kWindowT + 1);
            const std::size_t n = static_cast<std::size_t>(d.win.len) + 1;
            src.view.copy_to(d.win.start, n, &data[base]);
            for (std::size_t k = 0; k < n; ++k)
                mask[base + k] = src.masked() ? src.mask[d.win.start + k] : std::uint8_t{1};
            starts[static_cast<std::size_t>(b)] = base;
            lens[static_cast<std::size_t>(b)] = d.win.len;
        }
        opt.zero_grad();
        (void)sub0::train_batch(data.data(), starts.data(), kBatch, kWindowT, lens.data(), mask.data());
        opt.step();
    }
}

sub0::ScheduleStage equal_weight_stage(std::initializer_list<std::pair<std::string, double>> w) {
    return sub0::ScheduleStage{ std::numeric_limits<double>::infinity(),
                               std::vector<std::pair<std::string, double>>(w) };
}

}  // namespace

TEST_CASE("blended: OOV bindings and op-collapse results allocate independent scratch slots", "[blended]") {
    ScratchOps scratch;   // allow_bind defaults true (scratch.hpp)

    // Mint two OOV bindings via combine() -- mirrors gen_stage.cpp's sp_on/expand-combine path.
    const std::vector<int> oov_a = { 'f', 'o', 'o' }, oov_b = { 'b', 'a', 'r' };
    const std::vector<int> bound_a = scratch.combine(oov_a);
    REQUIRE(bound_a.size() == 1);
    REQUIRE(bound_a[0] == sub0::SCRATCH_SLOT_BASE);
    const std::vector<int> bound_b = scratch.combine(oov_b);
    REQUIRE(bound_b.size() == 1);
    REQUIRE(bound_b[0] == sub0::SCRATCH_SLOT_BASE + 1);

    // Mint an op-collapse result the SAME way gen_stage.cpp's op_on branch does: bind to
    // scratch.bindings.size() (the next free index) -- must land on slot 2, not collide with either OOV.
    const int idx = static_cast<int>(scratch.bindings.size());
    REQUIRE(idx == 2);
    const std::vector<int> frags = { '4', '2' };
    scratch.bind(sub0::SCRATCH_SLOT_BASE + idx, frags);

    CHECK(scratch.value(sub0::SCRATCH_SLOT_BASE)     == "foo");
    CHECK(scratch.value(sub0::SCRATCH_SLOT_BASE + 1) == "bar");
    CHECK(scratch.value(sub0::SCRATCH_SLOT_BASE + 2) == "42");
    // expand() (decode-time lookup) resolves the op-collapsed slot exactly like an OOV-bound one -- op
    // results and OOV bindings are indistinguishable to expand(), by design (the symbol is opaque to
    // the model either way; only the WRITER differs).
    CHECK(scratch.expand(sub0::SCRATCH_SLOT_BASE + 2) == frags);

    // Re-combining an already-bound OOV must still resolve to its OWN slot (0), not be confused with the
    // op-collapsed slot 2 that landed at a numerically-adjacent index.
    CHECK(scratch.combine(oov_a) == bound_a);
}

TEST_CASE("blended capstone: one model trained on scratch+op via the real epoch-fair scheduler "
         "retains both capabilities", "[.blendedcapstone]") {
    Tokenizer tk = load_tk();

    const ss::OovSplit split = ss::make_oov_split(tk, kOovPool, kDrilledFrac, /*seed=*/2024);
    REQUIRE_FALSE(split.drilled.empty());
    REQUIRE_FALSE(split.held_out.empty());
    ss::DatasetOptions sdopt; sdopt.tasks_per_oov = 12; sdopt.seed = 99;
    const ss::Dataset scratch_ds = ss::build_dataset(tk, split, sdopt);
    REQUIRE(scratch_ds.tokens.size() > static_cast<std::size_t>(kWindowT));

    oc::Options oopt; oopt.seed = 4242; oopt.n_examples = kOpExamples; oopt.chain_frac = kChainFrac;
    oopt.max_digits = kMaxDigits;
    const oc::Dataset op_ds = oc::build_dataset(tk, oopt);
    REQUIRE(op_ds.tokens.size() > static_cast<std::size_t>(kWindowT));

    sub0::BlendSource scratch_src{ "scratch",
        sub0::TokView::over_int32(scratch_ds.tokens.data(), scratch_ds.tokens.size()),
        std::span<const std::uint64_t>(scratch_ds.doc_starts), std::span<const std::uint8_t>(scratch_ds.mask), {} };
    sub0::BlendSource op_src{ "op",
        sub0::TokView::over_int32(op_ds.tokens.data(), op_ds.tokens.size()),
        std::span<const std::uint64_t>(op_ds.doc_starts), std::span<const std::uint8_t>(op_ds.mask), {} };

    std::string report = "\n=== blended capstone (d" + std::to_string(D_MODEL) + "): scratch (" +
        std::to_string(split.drilled.size()) + " drilled OOVs) + op-delegation (" +
        std::to_string(op_ds.doc_starts.size() - 1) + " problems), equal-weight epoch-fair blend ===\n";

    const int solo_steps = kEvalRounds * kStepsPerEval;

    // --- solo scratch baseline: same dataset, alone ---
    Acc solo_scratch_drilled, solo_scratch_held;
    {
        sub0::build_model(); reset_opt_state();
        sub0::AdamW opt(kLr);
        std::mt19937 rng(1);
        std::vector<sub0::BlendSource> srcs{ scratch_src };
        std::vector<std::string> names{ "scratch" };
        sub0::ScheduleSpec spec; spec.stages.push_back(equal_weight_stage({ {"scratch", 1.0} }));
        sub0::ResolvedSchedule sched = sub0::resolve_schedule(spec, std::span<const std::string>(names));
        sub0::BlendFairness fair(srcs.size());
        for (int r = 0; r < kEvalRounds; ++r) train_steps(srcs, sched, fair, opt, kStepsPerEval, rng);
        ScratchOps ops{ &tk, true, {} };
        solo_scratch_drilled = eval_scratch_nth(tk, ops, split.drilled);
        solo_scratch_held    = eval_scratch_nth(tk, ops, split.held_out);
        report += "  [scratch-solo, " + std::to_string(solo_steps) + " steps] DRILLED nth=" +
            std::to_string(solo_scratch_drilled.rate()) + "  HELD-OUT nth=" + std::to_string(solo_scratch_held.rate()) + "\n";
    }

    // --- solo op baseline: same dataset, alone ---
    OpScore solo_op_indist, solo_op_ood;
    {
        sub0::build_model(); reset_opt_state();
        sub0::AdamW opt(kLr);
        std::mt19937 rng(1);
        std::vector<sub0::BlendSource> srcs{ op_src };
        std::vector<std::string> names{ "op" };
        sub0::ScheduleSpec spec; spec.stages.push_back(equal_weight_stage({ {"op", 1.0} }));
        sub0::ResolvedSchedule sched = sub0::resolve_schedule(spec, std::span<const std::string>(names));
        sub0::BlendFairness fair(srcs.size());
        for (int r = 0; r < kEvalRounds; ++r) train_steps(srcs, sched, fair, opt, kStepsPerEval, rng);
        solo_op_indist = eval_op(200, kMaxDigits, 4242u);
        solo_op_ood    = eval_op(200, kMaxDigits + 1, 911u);   // OOD digit-width, matches gsm8k capstone
        report += "  [op-solo, " + std::to_string(solo_steps) + " steps] in-dist delegated=" +
            std::to_string(solo_op_indist.delegated / double(solo_op_indist.n)) + " exact=" +
            std::to_string(solo_op_indist.rate()) + "  OOD exact=" + std::to_string(solo_op_ood.rate()) + "\n";
    }

    // --- blended: BOTH sources, drawn by the real scheduler, DOUBLE the total step budget so each
    // mechanism still gets ~solo_steps of its own exposure under equal-weight epoch-fair scheduling ---
    Acc blend_scratch_drilled, blend_scratch_held;
    OpScore blend_op_indist, blend_op_ood;
    {
        sub0::build_model(); reset_opt_state();
        sub0::AdamW opt(kLr);
        std::mt19937 rng(1);
        std::vector<sub0::BlendSource> srcs{ scratch_src, op_src };
        std::vector<std::string> names{ "scratch", "op" };
        sub0::ScheduleSpec spec;
        spec.stages.push_back(equal_weight_stage({ {"scratch", 1.0}, {"op", 1.0} }));
        sub0::ResolvedSchedule sched = sub0::resolve_schedule(spec, std::span<const std::string>(names));
        sub0::BlendFairness fair(srcs.size());
        for (int r = 0; r < kEvalRounds; ++r) train_steps(srcs, sched, fair, opt, 2 * kStepsPerEval, rng);
        ScratchOps ops{ &tk, true, {} };
        blend_scratch_drilled = eval_scratch_nth(tk, ops, split.drilled);
        blend_scratch_held    = eval_scratch_nth(tk, ops, split.held_out);
        blend_op_indist = eval_op(200, kMaxDigits, 4242u);
        blend_op_ood    = eval_op(200, kMaxDigits + 1, 911u);
        report += "  [BLENDED, " + std::to_string(2 * solo_steps) +
            " steps total, ~" + std::to_string(solo_steps) + "/mechanism]\n"
            "    scratch: DRILLED nth=" + std::to_string(blend_scratch_drilled.rate()) +
            "  HELD-OUT nth=" + std::to_string(blend_scratch_held.rate()) + "\n"
            "    op:      in-dist delegated=" + std::to_string(blend_op_indist.delegated / double(blend_op_indist.n)) +
            " exact=" + std::to_string(blend_op_indist.rate()) + "  OOD exact=" + std::to_string(blend_op_ood.rate()) + "\n";
    }

    WARN(report);

    // The blended function-validation claim: neither capability collapses when trained together vs
    // alone. A generous "not badly broken" bar (half the solo rate, floored at a real-signal minimum)
    // rather than a tight regression pin -- this is a single-seed run and both mechanisms have real
    // step-to-step variance (see scratchspike/gsm8k capstones' own WARN-report-first convention); the
    // report above carries the full numbers for a human to judge quality, this assertion catches an
    // outright REGRESSION (blending silently breaking one mechanism), not fine-grained drift.
    CHECK(std::isfinite(blend_scratch_held.rate()));
    CHECK(std::isfinite(blend_op_indist.rate()));
    CHECK(blend_scratch_held.rate() > std::max(0.15, solo_scratch_held.rate() * 0.5));
    CHECK(blend_op_indist.rate()   > std::max(0.15, solo_op_indist.rate() * 0.5));
}
