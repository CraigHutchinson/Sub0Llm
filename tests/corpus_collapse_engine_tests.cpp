// corpus_collapse_engine_tests.cpp -- two kinds of proof for sub0/corpus_collapse.hpp:
//
//   1. Fast, always-run, engine-free tests ("[corpus_collapse]") for build_dataset itself: proves the
//      per-document ScratchTable reset, the val-split clamp, long-document prefix truncation, and the
//      masking discipline (mask=1 except at a substituted slot) are all correct, using hand-built
//      TokView/doc_starts fixtures -- no model, no tokenizer file, no training.
//   2. A hidden capstone ("[.corpus_collapse]") training THREE real models to test whether blending a
//      corpus_collapse source alongside "base" measurably helps or hurts held-out NELBO on ordinary real
//      corpus text, via the REAL production scheduler (sample_blend_staged) -- matched-budget A/B/C,
//      mirroring blended_capstone_engine_tests.cpp's own methodology. The third arm (2026-07-21,
//      build_dataset_markers) tests docs/FACTSPIKE.md's "SPELL marker finding" at real-corpus scale: does
//      binding a collapsed slot to the marker-INCLUSIVE span (SPELL_START/pieces/SPELL_END, what a real
//      forward pass over the word actually processes) instead of word_span's marker-stripped pieces
//      change held-out NELBO, the way it decisively closed a reconstruction-fidelity gap in factspike's
//      toy model.
//
// See docs/CORPUS_COLLAPSE.md for the full design record and the capstone's real result.

#include <catch2/catch_test_macros.hpp>

#include "sub0/core.hpp"
#include "sub0/corpus_collapse.hpp"
#include "sub0/blend_schedule.hpp"
#include "sub0/casing.hpp"

#include <chrono>
#include <cstdio>
#include <fstream>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace {

namespace cas = sub0::casing;
using sub0::tok::Tokenizer;

// A 5-token compound-word span: TOK_SPELL_START <p0> <p1> <p2> TOK_SPELL_END -- matches exactly what
// encode_join emits for a 3+-piece OOV/compound word (see scratch.hpp's detail::word_span).
std::vector<int> spell_span(int p0, int p1, int p2) {
    return { cas::TOK_SPELL_START, p0, p1, p2, cas::TOK_SPELL_END };
}

}  // namespace

TEST_CASE("corpus_collapse: per-document ScratchTable reset -- the same repeated OOV in two documents "
         "does not collapse across the document boundary", "[corpus_collapse]") {
    // Each document independently: mention 1 (spelled, 5 raw tokens) + mention 2 (the same bytes,
    // collapses to 1 slot token). If the table incorrectly persisted across documents, doc 1's own
    // FIRST mention would wrongly collapse too (since doc 0 already bound the identical bytes).
    std::vector<int> corpus;
    const std::vector<int> span = spell_span(10, 20, 30);
    corpus.insert(corpus.end(), span.begin(), span.end());
    corpus.insert(corpus.end(), span.begin(), span.end());   // doc 0: mention 1 + mention 2
    const std::uint64_t doc1_start = corpus.size();
    corpus.insert(corpus.end(), span.begin(), span.end());
    corpus.insert(corpus.end(), span.begin(), span.end());   // doc 1: mention 1 + mention 2 (same bytes)

    const std::vector<std::uint64_t> doc_starts = { 0, doc1_start };   // raw corpus.tok convention
    const sub0::TokView view = sub0::TokView::over_int32(corpus.data(), corpus.size());

    Tokenizer tk;   // default-constructed: empty piece_index, so every span is treated as a bind candidate
    sub0::corpus_collapse::Options opt; opt.seed = 1; opt.n_docs = 10;
    const sub0::corpus_collapse::Dataset ds =
        sub0::corpus_collapse::build_dataset(tk, view, doc_starts, corpus.size(), opt);

    REQUIRE(ds.doc_starts.size() == 3);   // sentinel(0) + one boundary per document
    REQUIRE(ds.doc_bindings.size() == 2);

    // Both documents collapse identically: 5 raw tokens (mention 1) + 1 slot token (mention 2) == 6.
    CHECK(ds.doc_starts[1] - ds.doc_starts[0] == 6);
    CHECK(ds.doc_starts[2] - ds.doc_starts[1] == 6);   // NOT 2 -- would be 2 if the table leaked across docs

    const std::size_t d0 = static_cast<std::size_t>(ds.doc_starts[0]);
    const std::size_t d1 = static_cast<std::size_t>(ds.doc_starts[1]);
    // doc 0: mention 1 spelled out verbatim (mask=1 x5), mention 2 collapsed to the bound slot (mask=0).
    for (int i = 0; i < 5; ++i) { CHECK(ds.tokens[d0 + static_cast<std::size_t>(i)] == span[static_cast<std::size_t>(i)]); CHECK(ds.mask[d0 + static_cast<std::size_t>(i)] == 1); }
    CHECK(ds.tokens[d0 + 5] == sub0::SCRATCH_SLOT_BASE);
    CHECK(ds.mask[d0 + 5] == 0);
    // doc 1: mention 1 is ALSO spelled out verbatim (fresh table -- proves no cross-document leak).
    for (int i = 0; i < 5; ++i) { CHECK(ds.tokens[d1 + static_cast<std::size_t>(i)] == span[static_cast<std::size_t>(i)]); CHECK(ds.mask[d1 + static_cast<std::size_t>(i)] == 1); }
    CHECK(ds.tokens[d1 + 5] == sub0::SCRATCH_SLOT_BASE);
    CHECK(ds.mask[d1 + 5] == 0);
}

TEST_CASE("corpus_collapse: the train/val split clamp truncates a document's end to train_tok, "
         "mirroring window.hpp's own clamp", "[corpus_collapse]") {
    // A single document of 10 ordinary (non-marker) tokens; doc_starts declares a SECOND document
    // starting at 10 (so `de` initially resolves to 10 via doc_starts[1]), but train_tok=5 clips it.
    const std::vector<int> corpus = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    const std::vector<std::uint64_t> doc_starts = { 0, 10 };
    const sub0::TokView view = sub0::TokView::over_int32(corpus.data(), corpus.size());

    Tokenizer tk;
    sub0::corpus_collapse::Options opt; opt.seed = 1; opt.n_docs = 10;
    const sub0::corpus_collapse::Dataset ds =
        sub0::corpus_collapse::build_dataset(tk, view, doc_starts, /*train_tok=*/5, opt);

    REQUIRE(ds.doc_starts.size() == 2);   // only doc 0 is sampled/emitted -- doc 1 starts at/past train_tok
    CHECK(ds.tokens.size() == 5);
    for (int i = 0; i < 5; ++i) CHECK(ds.tokens[static_cast<std::size_t>(i)] == corpus[static_cast<std::size_t>(i)]);
}

TEST_CASE("corpus_collapse: a document longer than max_doc_tokens is prefix-truncated, not dropped -- "
         "a recurrence past the cutoff is simply never seen, not an error", "[corpus_collapse]") {
    std::vector<int> corpus;
    const std::vector<int> span = spell_span(10, 20, 30);
    corpus.insert(corpus.end(), span.begin(), span.end());          // mention 1 (5 tokens, positions 0-4)
    corpus.insert(corpus.end(), { 100, 101, 102, 103, 104 });        // filler (5 tokens, positions 5-9)
    corpus.insert(corpus.end(), span.begin(), span.end());          // mention 2 (5 tokens, positions 10-14) -- past the cap

    const std::vector<std::uint64_t> doc_starts = { 0 };
    const sub0::TokView view = sub0::TokView::over_int32(corpus.data(), corpus.size());

    Tokenizer tk;
    sub0::corpus_collapse::Options opt; opt.seed = 1; opt.n_docs = 10; opt.max_doc_tokens = 10;
    const sub0::corpus_collapse::Dataset ds =
        sub0::corpus_collapse::build_dataset(tk, view, doc_starts, corpus.size(), opt);

    REQUIRE(ds.doc_starts.size() == 2);
    // Mention 2 never enters the truncated window, so nothing collapses -- exactly the 10 raw tokens
    // survive, all graded (mask=1), no slot token anywhere.
    CHECK(ds.tokens.size() == 10);
    for (std::uint8_t m : ds.mask) CHECK(m == 1);
    bool saw_slot = false;
    for (int t : ds.tokens) if (sub0::is_scratch_slot(t)) saw_slot = true;
    CHECK_FALSE(saw_slot);
}

TEST_CASE("corpus_collapse: mask is 1 everywhere except exactly the substituted slot position -- "
         "an ordinary recognized vocab piece is never treated as a bind candidate", "[corpus_collapse]") {
    // A tokenizer that recognizes byte 42 as an ordinary single vocab piece (piece_index populated) --
    // combine_recurrence must therefore treat repeated occurrences of it as ordinary text, never binding
    // a slot for it, regardless of how many times it recurs.
    Tokenizer tk;
    std::string key(1, static_cast<char>(42));
    tk.piece_index[key] = 42;

    const std::vector<int> corpus = { 42, 42, 42, cas::TOK_SPELL_START, 10, 20, 30, cas::TOK_SPELL_END,
                                      cas::TOK_SPELL_START, 10, 20, 30, cas::TOK_SPELL_END };
    const std::vector<std::uint64_t> doc_starts = { 0 };
    const sub0::TokView view = sub0::TokView::over_int32(corpus.data(), corpus.size());

    sub0::corpus_collapse::Options opt; opt.seed = 1; opt.n_docs = 10;
    const sub0::corpus_collapse::Dataset ds =
        sub0::corpus_collapse::build_dataset(tk, view, doc_starts, corpus.size(), opt);

    REQUIRE(ds.doc_starts.size() == 2);
    // 3 ordinary "42" tokens (graded, never bind) + 5 (mention 1, graded) + 1 (mention 2, collapsed).
    REQUIRE(ds.tokens.size() == 9);
    for (int i = 0; i < 8; ++i) CHECK(ds.mask[static_cast<std::size_t>(i)] == 1);   // everything but the last
    CHECK(ds.mask[8] == 0);                                                        // exactly the slot position
    CHECK(ds.tokens[8] == sub0::SCRATCH_SLOT_BASE);
}

// ---- capstone: does blending corpus_collapse alongside "base" measurably help or hurt ordinary graded
// prediction on REAL corpus text (TinyStories), matched-budget, via the REAL production scheduler
// (sample_blend_staged) -- same discipline as blended_capstone_engine_tests.cpp's own A/B. Requires this
// test binary to be built against a TinyStories-configured corpus.tok (this project's own default
// analysis corpus); skips with a clear message otherwise rather than failing on an unrelated build. ----

namespace {

void reset_opt_state() {
    const std::size_t n = sub0::trainable_floats();
    std::fill(sub0::adam_m_ptr(), sub0::adam_m_ptr() + n, 0.f);
    std::fill(sub0::adam_v_ptr(), sub0::adam_v_ptr() + n, 0.f);
}

constexpr int   kBatch        = 16;
constexpr int   kWindowT      = 64;
constexpr int   kStepsPerEval = 150;
constexpr int   kEvalRounds   = 4;      // 600 steps/arm -- a directional signal, not a production run
constexpr int   kValBatches   = 20;     // 320 windows per val_nelbo measurement
constexpr float kLr           = 0.003f * (128.0f / static_cast<float>(D_MODEL));

sub0::ScheduleStage equal_weight_stage(std::initializer_list<std::pair<std::string, double>> w) {
    return sub0::ScheduleStage{ std::numeric_limits<double>::infinity(),
                               std::vector<std::pair<std::string, double>>(w) };
}

// Renders document `doc_idx` of a corpus_collapse Dataset with each collapsed slot marked inline as
// "<~original bytes~>" (using that document's own doc_bindings to resolve the slot back to what it
// stands for) -- a qualitative, human-readable check that a real recurring word in real corpus prose is
// actually what got collapsed, not just that the token counts/masks line up. Not needed for correctness
// (the unit tests above already pin the mechanics); this is for a human to eyeball real examples, per
// the three-pillar policy's "not accuracy alone" spirit and this doc's own "Gaps" section.
std::string render_marked(const sub0::corpus_collapse::Dataset& ds, std::size_t doc_idx) {
    const std::size_t b = static_cast<std::size_t>(ds.doc_starts[doc_idx]);
    const std::size_t e = static_cast<std::size_t>(ds.doc_starts[doc_idx + 1]);
    const std::vector<std::vector<int>>& bindings = ds.doc_bindings[doc_idx];
    std::vector<int> disp;
    for (std::size_t i = b; i < e; ++i) {
        const int t = ds.tokens[i];
        if (!sub0::is_scratch_slot(t)) { disp.push_back(t); continue; }
        const int s = t - sub0::SCRATCH_SLOT_BASE;
        auto lit = [&](char c) { disp.push_back(static_cast<int>(static_cast<unsigned char>(c))); };
        lit('<'); disp.push_back(cas::TOK_JOIN); lit('~'); disp.push_back(cas::TOK_JOIN);
        if (s >= 0 && s < static_cast<int>(bindings.size())) {
            bool first = true;
            for (int frag : bindings[static_cast<std::size_t>(s)]) {
                if (!first) disp.push_back(cas::TOK_JOIN);
                disp.push_back(frag);
                first = false;
            }
        }
        disp.push_back(cas::TOK_JOIN); lit('~'); disp.push_back(cas::TOK_JOIN); lit('>');
    }
    return sub0::detokenize(disp);
}

void train_steps(std::vector<sub0::BlendSource>& sources, sub0::ResolvedSchedule& sched,
                 sub0::BlendFairness& fair, sub0::AdamW& opt, int steps, std::mt19937& rng) {
    std::vector<int> data(static_cast<std::size_t>(kBatch) * (kWindowT + 1));
    std::vector<std::uint8_t> mask(data.size());
    std::vector<std::size_t> starts(kBatch);
    std::vector<int> lens(kBatch);
    for (int s = 0; s < steps; ++s) {
        for (int b = 0; b < kBatch; ++b) {
            const sub0::BlendDraw d = sub0::sample_blend_staged(rng, fair, sources, sched, kWindowT, 0.0);
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

// Flat (non-doc-aware) held-out NELBO: reuses train_batch's own returned mean cross-entropy loss as the
// measurement (the exact training objective, not a hand-rolled re-derivation) -- the backward pass it
// computes is simply discarded (never followed by opt.step()), harmless since the next real training
// step always zero_grads first.
double eval_val_nelbo(sub0::TokView val_span, std::mt19937& rng) {
    std::vector<int> data(static_cast<std::size_t>(kBatch) * (kWindowT + 1));
    std::vector<std::size_t> starts(kBatch);
    std::uniform_int_distribution<std::size_t> uni(0, val_span.size() - (kWindowT + 1));
    double total = 0.0;
    for (int r = 0; r < kValBatches; ++r) {
        for (int b = 0; b < kBatch; ++b) {
            const std::size_t s = uni(rng);
            val_span.copy_to(s, kWindowT + 1, &data[static_cast<std::size_t>(b) * (kWindowT + 1)]);
            starts[static_cast<std::size_t>(b)] = static_cast<std::size_t>(b) * (kWindowT + 1);
        }
        total += sub0::train_batch(data.data(), starts.data(), kBatch, kWindowT);
    }
    return total / kValBatches;
}

}  // namespace

TEST_CASE("corpus_collapse capstone: does blending a real-corpus collapse curriculum alongside base "
         "measurably help or hurt held-out NELBO, matched total budget", "[.corpus_collapse]") {
    // Trains against WHATEVER corpus.tok this build was configured with -- there is no reliable way to
    // detect "is this specifically TinyStories" from the file alone (any correctly-configured build's own
    // corpus.tok trivially matches its own baked VOCAB by construction, so a vocab check can't tell them
    // apart). Run this against a TinyStories-configured build (e.g. out/build/d196check, this project's
    // own default analysis corpus) for the intended, reported-in-docs/CORPUS_COLLAPSE.md result; running
    // it elsewhere is harmless (just trains/evaluates against that build's own corpus instead) but won't
    // match that doc's numbers. The only genuine bail-out: no usable corpus.tok at all (e.g. a
    // --corpus-pretok 0 / on-demand-only build has none).
    sub0::TokMap tok(sub0::default_corpus_tok());
    if (!tok.ok() || tok.vocab() != VOCAB) {
        WARN("corpus_collapse capstone: no usable corpus.tok in this build ('" +
            std::string(sub0::default_corpus_tok()) + "') -- skipping. See docs/CORPUS_COLLAPSE.md.");
        return;
    }
    Tokenizer tk;
    { std::ifstream is(sub0::default_tokenizer(), std::ios::binary); REQUIRE(sub0::tok::deserialize(tk, is)); }
    REQUIRE(tk.vocab == VOCAB);
    // render_marked() below calls sub0::detokenize(), which reads the GLOBAL tokenizer state, not the
    // LOCAL `tk` above (that's a separate copy this file's own build_dataset call consumes directly) --
    // without this, the qualitative sample dump silently renders blank/garbage text (a real bug this
    // comment now documents, found by actually reading the WARN output, not assumed correct).
    REQUIRE(sub0::load_tokenizer(sub0::default_tokenizer()));

    const sub0::TokView data = tok.tokens();
    const std::span<const std::uint64_t> doc_index = tok.doc_starts();
    REQUIRE_FALSE(doc_index.empty());
    constexpr double kValFraction = 0.02;   // small held-out tail, this is a directional capstone not a
                                            // production split -- the real train_stage.cpp uses VAL_FRACTION
    const std::size_t val_tokens = std::max<std::size_t>(static_cast<std::size_t>(kWindowT) + 2,
        static_cast<std::size_t>(kValFraction * static_cast<double>(data.size())));
    const std::size_t val_start = data.size() - val_tokens;
    const sub0::TokView train_span = data.first(val_start);
    const sub0::TokView val_span   = data.subspan(val_start);

    sub0::corpus_collapse::Options copt;
    copt.seed = 20260718; copt.n_docs = 2000; copt.max_doc_tokens = kWindowT;
    const auto t0 = std::chrono::steady_clock::now();
    const sub0::corpus_collapse::Dataset cc_ds =
        sub0::corpus_collapse::build_dataset(tk, train_span, doc_index, val_start, copt);
    const auto t1 = std::chrono::steady_clock::now();
    const double build_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    REQUIRE(cc_ds.tokens.size() > static_cast<std::size_t>(kWindowT));

    // Qualitative sample: show a handful of real collapsed documents (see render_marked's own comment).
    // Also reports what FRACTION of the sampled real documents exhibited a recurring compound word at
    // all -- a useful statistic independent of the shown examples.
    {
        const std::size_t n_docs = cc_ds.doc_bindings.size();
        std::size_t n_with_collapse = 0;
        std::string samples;
        int shown = 0;
        for (std::size_t d = 0; d < n_docs; ++d) {
            const bool has_collapse = !cc_ds.doc_bindings[d].empty();
            if (has_collapse) {
                ++n_with_collapse;
                if (shown < 5) {
                    samples += "  [" + std::to_string(shown + 1) + "] " + render_marked(cc_ds, d) + "\n";
                    ++shown;
                }
            }
        }
        std::string qual = "\n=== corpus_collapse: real TinyStories examples (<~word~> = collapsed slot, "
            "showing what it resolves to) ===\n" + samples +
            "  (" + std::to_string(n_with_collapse) + "/" + std::to_string(n_docs) +
            " sampled documents had >=1 recurring compound word to collapse)\n";
        WARN(qual);
    }

    // Marker-inclusive dataset (docs/FACTSPIKE.md's "SPELL marker finding") -- SAME sampled documents/seed
    // as cc_ds above (build_dataset_markers takes identical Options), differing only in whether a
    // collapsed slot's bound content includes SPELL_START/SPELL_END or not.
    const sub0::corpus_collapse::Dataset cc_ds_markers =
        sub0::corpus_collapse::build_dataset_markers(tk, train_span, doc_index, val_start, copt);
    REQUIRE(cc_ds_markers.tokens.size() > static_cast<std::size_t>(kWindowT));

    sub0::BlendSource base_src{ "base", train_span, doc_index, {}, {} };
    sub0::BlendSource cc_src{ "collapse",
        sub0::TokView::over_int32(cc_ds.tokens.data(), cc_ds.tokens.size()),
        std::span<const std::uint64_t>(cc_ds.doc_starts), std::span<const std::uint8_t>(cc_ds.mask),
        std::span<const std::vector<std::vector<int>>>(cc_ds.doc_bindings) };
    sub0::BlendSource cc_src_markers{ "collapse_markers",
        sub0::TokView::over_int32(cc_ds_markers.tokens.data(), cc_ds_markers.tokens.size()),
        std::span<const std::uint64_t>(cc_ds_markers.doc_starts), std::span<const std::uint8_t>(cc_ds_markers.mask),
        std::span<const std::vector<std::vector<int>>>(cc_ds_markers.doc_bindings) };

    double base_only_nelbo = 0.0, blended_nelbo = 0.0, blended_markers_nelbo = 0.0;

    sub0::build_model(); reset_opt_state();
    {
        sub0::AdamW opt(kLr);
        std::mt19937 rng(1);
        std::vector<sub0::BlendSource> srcs{ base_src };
        std::vector<std::string> names{ "base" };
        sub0::ScheduleSpec spec; spec.stages.push_back(equal_weight_stage({ {"base", 1.0} }));
        sub0::ResolvedSchedule sched = sub0::resolve_schedule(spec, std::span<const std::string>(names));
        sub0::BlendFairness fair(srcs.size());
        for (int r = 0; r < kEvalRounds; ++r) train_steps(srcs, sched, fair, opt, kStepsPerEval, rng);
        std::mt19937 vrng(2);
        base_only_nelbo = eval_val_nelbo(val_span, vrng);
    }

    sub0::build_model(); reset_opt_state();
    {
        sub0::AdamW opt(kLr);
        std::mt19937 rng(1);
        std::vector<sub0::BlendSource> srcs{ base_src, cc_src };
        std::vector<std::string> names{ "base", "collapse" };
        sub0::ScheduleSpec spec; spec.stages.push_back(equal_weight_stage({ {"base", 1.0}, {"collapse", 1.0} }));
        sub0::ResolvedSchedule sched = sub0::resolve_schedule(spec, std::span<const std::string>(names));
        sub0::BlendFairness fair(srcs.size());
        for (int r = 0; r < kEvalRounds; ++r) train_steps(srcs, sched, fair, opt, kStepsPerEval, rng);
        std::mt19937 vrng(2);
        blended_nelbo = eval_val_nelbo(val_span, vrng);
    }

    sub0::build_model(); reset_opt_state();
    {
        sub0::AdamW opt(kLr);
        std::mt19937 rng(1);
        std::vector<sub0::BlendSource> srcs{ base_src, cc_src_markers };
        std::vector<std::string> names{ "base", "collapse_markers" };
        sub0::ScheduleSpec spec;
        spec.stages.push_back(equal_weight_stage({ {"base", 1.0}, {"collapse_markers", 1.0} }));
        sub0::ResolvedSchedule sched = sub0::resolve_schedule(spec, std::span<const std::string>(names));
        sub0::BlendFairness fair(srcs.size());
        for (int r = 0; r < kEvalRounds; ++r) train_steps(srcs, sched, fair, opt, kStepsPerEval, rng);
        std::mt19937 vrng(2);
        blended_markers_nelbo = eval_val_nelbo(val_span, vrng);
    }

    const std::size_t cc_bytes = cc_ds.tokens.size() * sizeof(int) + cc_ds.mask.size();
    char buf[800];
    std::snprintf(buf, sizeof buf,
        "\n=== corpus_collapse capstone (d%d, corpus '%s') ===\n"
        "  build_dataset: %d docs sampled -> %zu tokens, %.1f ms\n"
        "  memory: collapse Dataset ~%.2f MB  vs  base corpus %.1f MB (token count only, mmap'd not resident)\n"
        "  held-out NELBO (%d val batches, %d windows/batch): base-only=%.4f  base+collapse=%.4f  "
        "base+collapse_markers=%.4f\n"
        "  delta vs base-only: collapse=%.4f  collapse_markers=%.4f  (+ve = hurt; docs/FACTSPIKE.md's "
        "SPELL marker finding predicts collapse_markers should do LESS harm / more good than collapse)\n",
        D_MODEL, sub0::default_corpus_tok(), copt.n_docs, cc_ds.tokens.size(), build_ms,
        static_cast<double>(cc_bytes) / (1024.0 * 1024.0),
        static_cast<double>(train_span.size() * sizeof(int)) / (1024.0 * 1024.0),
        kValBatches, kBatch, base_only_nelbo, blended_nelbo, blended_markers_nelbo,
        blended_nelbo - base_only_nelbo, blended_markers_nelbo - base_only_nelbo);
    WARN(std::string(buf));

    CHECK(std::isfinite(base_only_nelbo));
    CHECK(std::isfinite(blended_nelbo));
    CHECK(std::isfinite(blended_markers_nelbo));
    // Equal-weight blending is a deliberately AGGRESSIVE exposure (a production schedule would likely
    // weight this source much lower) -- this is a "not badly broken" gate, not a tight regression pin;
    // the WARN report above carries the real numbers for a human to judge (three-pillar policy: not
    // NELBO alone -- see the build_ms/memory lines too).
    CHECK(blended_nelbo < base_only_nelbo + 0.5);
}
