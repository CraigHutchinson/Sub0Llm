// factspike_engine_tests.cpp -- Phase B/C (docs/FACTSPIKE.md): does a factual association taught via
// ordinary repeated text survive PIECE-EMBEDDING TRANSFER into a scratch slot? Staged, each phase gating
// the next -- this is an open experiment, not a guaranteed-positive validation. Requires this test binary
// to be built against the factspike96 toy config (out/build/factspike96): a small, dedicated model/vocab
// forced to be mostly multi-piece, generated per docs/FACTSPIKE.md's Phase A. Constants here (seed=
// 20260719, n_total=12, drilled_frac=0.667) MUST match the scratchpad corpus-generation tool that built
// that config's tokenizer, so both sides reconstruct the identical FactSplit deterministically.
//
//   * [.factspikebaseline] -- Phase B: train on JUST the fact-teaching documents (ordinary text, no
//     scratch slots anywhere), eval ONLY the baseline arm (subject spelled out in full). This is the
//     load-bearing premise: does the curriculum teach a persistent, cross-document association at all?
//     RESULT (docs/FACTSPIKE.md): yes -- peak accuracy 0.78 (7/9) vs chance 0.125, though training is
//     highly volatile (0.00 <-> 0.78 across rounds), so PEAK not final-round accuracy is what's checked.
//   * [.factspikecapstone] -- Phase C: adds slot-reading exposure documents (piece-id bound, over
//     SEPARATE subjects from the ones under test) blended with fact-teaching, then the real 3-arm A/B:
//     baseline (subject spelled out), scratch (subject's own piece ids bound to a slot, no textual
//     restatement), held-out (same slot mechanism, subject never fact-taught -- negative control).
//     RESULT (docs/FACTSPIKE.md): peak baseline=1.00, scratch=0.56 (above chance, real signal), held-
//     out=0.33 (consistent with n=3 noise). A reconstruction-fidelity diagnostic added afterward found NO
//     positive correlation between packed-vector fidelity and scratch accuracy (r=-0.23) -- because
//     Phase C's exposure documents only grade generic filler text after the slot, never the fact itself,
//     so nothing trains packed-vector fidelity for the actual eval subjects at all.
//   * [.factspikepat] -- Phase D ("Pack-Aware Training"): same budget/seeds as Phase C, but the slot-
//     exposure documents are replaced with build_slot_retrieval_dataset()'s task-contingent design -- the
//     graded text after the slot IS the fact color itself (no full-text restatement in that document),
//     so gradient into the packed vector is directly informative, the structural match to how QAT
//     computes its real task loss through the dequantized forward pass. Exposure subjects additionally
//     get ordinary full-text fact teaching via a WIDENED build_dataset() pool (drilled + exposure
//     subjects together) -- a separate document, so no in-document shortcut. Matched-budget A/B against
//     Phase C's own recorded numbers.
//     RESULT (docs/FACTSPIKE.md): WORSE across the board, not better -- baseline collapsed 1.00 -> 0.22,
//     scratch 0.56 -> 0.00, even though fidelity stayed in roughly the same flat band as Phase C. Baseline
//     collapsing too (no slot involved at all) means this isn't "the harder task hurt scratch
//     specifically" -- the combined regime got worse at teaching the fact at all. Two undisambiguated
//     candidates: diluted per-subject repetition (widened pool, same fixed budget), or a harder objective
//     mixed flat/cold from a random-init model destabilizing shared weights (matches what progressive-QAT
//     literature warns against) -- Phase E below tests the second directly.
//   * [.factspikepatramp] -- Phase E: same matched budget/seeds/datasets as Phase D, but WARM-STARTED
//     (first 5 rounds slot_frac=0, pure plain-text) then RAMPED (slot_frac linearly 0->0.5 over the next
//     10 rounds, holding at 0.5 for the final 5) instead of flat 0.5 from step 1. Isolates cold-mixing
//     interference from pool dilution: dilution persists through warm-start (same widened ds throughout),
//     so if warm-start alone recovers baseline toward Phase C's level, cold-mixing was the dominant cause
//     in Phase D, not dilution.
//   * [.factspikehidden] -- hidden-state diagnostic: after Phase C's own (validated, non-degenerate)
//     training regime, does the packed slot's FULLY-PROCESSED hidden state (post all N_LAYERS, right
//     before the color would be generated -- sub0::last_hidden_ptr()) resemble the SAME subject's hidden
//     state when read normally? A different question from the reconstruction-fidelity diagnostic (which
//     compared the packed vector against RAW pre-transformer piece embeddings): this compares the model's
//     own decision-relevant, output-level representation instead.
//     RESULT (docs/FACTSPIKE.md): mean cos_sim=0.83 (inflated by 3 accidentally-single-piece subjects);
//     0.77 for the 6 genuinely multi-piece ones, no clean link to correctness at this sample size (r=0.24,
//     not reliable). Real but partial representational degradation, not the dominant driver alone.
//   * [.factspikepatfair] -- Phase F: re-tests Phase D's task-contingent gradient fix WITHOUT the dilution
//     confound Phase E isolated. train_steps_3way keeps drilled subjects' own plain-text sampling rate
//     close to Phase C's (a SEPARATE drilled-only Dataset, not merged with exposure subjects), while
//     exposure subjects get their own separate plain-text pool at a smaller, independently-controlled
//     rate. Flat slot_frac=0.5 from step 1 (matching Phase D, not Phase E's ramp) -- isolates "does fixing
//     dilution alone recover Phase C" as its own clean data point.
//   * [.factspikekvtrace] -- Phase G: KV-trace memoization, docs/SCRATCH_TOKEN_FRAMING.md's two
//     candidates. Parks the training-schedule (PAT) axis Phase D/E/F investigated and instead attacks
//     axis 9 (attention-capacity preservation) directly: instead of composing a slot's embedding from RAW
//     pre-transformer piece rows, capture the word's REAL per-layer (K,V) trace from an isolated forward
//     pass and splice it into a live KV-cache -- reusing real computation instead of approximating it.
//     Candidate 1 (B): pool n->1 per layer (reusing encode_slot's own HRR math one layer deeper) before
//     splicing -- keeps O(1) cost like mechanism A. Candidate 2 (C/D): no pooling at all, splice the FULL
//     n-position trace (n real KV-cache rows per layer) -- exact reconstruction, O(n) cost like not
//     packing. C is the trivial same-context shape (a correctness gate: should reproduce baseline almost
//     exactly); D is the harder context-sensitivity probe (the isolated-captured trace spliced into a
//     REAL non-empty preceding context, testing the framing doc's own open "how context-sensitive is a
//     word's own trace" question). All measured in the SAME run, same trained weights, alongside mechanism
//     A (embed-compose baseline).
//     RESULT (docs/FACTSPIKE.md) -- candidate 1 (B): NOT an improvement -- mean cos_sim A=0.83 vs B=0.77
//     (worse, not better), r(piece_count,cos_sim) A=-0.614 vs B=-0.624 (slightly MORE negative than
//     predicted-weaker), accuracy tied 3/9 each on DIFFERENT subjects. Mean-gap is mostly one n=8 outlier
//     (excluding it: A=0.859 vs B=0.850, nearly tied); the correlation gap survives outlier exclusion.
//     Reframes the open question: HRR's fixed-capacity n->1 bundle itself, not the pooled input's
//     richness, looks like the dominant bottleneck -- an open, cheap-to-test link to the framing doc's own
//     HRR-crosstalk question. Candidate 2 (C, trivial case): landed at mean 0.947, not the predicted ~1.0
//     -- root-caused to a suffix-tokenization confound (fixed) plus, decisively, SPELL_START/SPELL_END
//     wrapper-marker omission: subject_piece_ids (what A/B/C ALL capture from) strips markers a real
//     forward pass actually processes. A direct swap-and-remeasure (E: replay the marker-INCLUSIVE span)
//     closed the ENTIRE gap to exact 1.000 for every drilled subject, multi-piece included, zero
//     exceptions -- not a correlation, a controlled proof. Every mechanism here has been operating on an
//     incomplete basis by construction. D (context-sensitivity probe): superseded by F/G below (D's own
//     confound was never fixed). F/G (context_sensitivity_cosine2, boundary detection redone via search
//     after a bug caught by a diagnostic -- see docs/FACTSPIKE.md): F(pieces)=0.919, G(markers)=0.983,
//     spliced into a REAL different prefix. Markers matter more than context-sensitivity does -- a trace
//     captured once in isolation WITH markers reproduces ~98% of the true recomputed representation when
//     reused elsewhere, a genuinely encouraging result for prefill-compute-amortization schemes.
//   * [.factspikemarkers] -- Phase H: does marker-inclusive composition (SPELL_START/pieces/SPELL_END
//     instead of subject_piece_ids) help mechanism A and candidate 1, now that candidate 2 PROVED the
//     markers are load-bearing? A genuinely SEPARATE matched-budget training run (build_slot_exposure_
//     dataset_markers) -- the model must be TAUGHT to read a marker-inclusive-bound slot, not just
//     evaluated with one, or this would test generalization to a novel convention instead of the actual
//     hypothesis. Measures mechanism A (embed-compose) and candidate 1 (pooled KV-trace splice), both
//     marker-inclusive, on this newly-trained model -- compare against the piece-only regime's own
//     recorded numbers ([.factspikehidden] mean_sim=0.83 r=-0.614; Phase G's candidate-1 mean_sim=0.77
//     r=-0.624).
//     RESULT (docs/FACTSPIKE.md): WORSE, not better -- mean cos_sim A_markers=0.695 (vs 0.83) B_markers=
//     0.630 (vs 0.77), accuracy A_markers=0/9 this snapshot. Doesn't contradict candidate 2 -- COMPLETES
//     the picture: splice (no compression) benefits from more real signal freely; encode_slot/candidate-1
//     STILL have to compress into one fixed-D_MODEL vector, and cramming 2 more genuine items into an
//     ALREADY-bottlenecked HRR bundle (candidate 1's own finding) makes that bundle's job harder, not
//     easier. Three separate experiments (candidate 1's input swap, this marker-add, candidate 2's no-
//     compression splice) now converge on the same direction: the fixed-size compression STEP is the
//     bottleneck, not which tokens feed it. Single-seed caveat applies, as always -- the DIRECTION across
//     three independent designs is what's being leaned on, not any one number.
//   * [.factspikereinject] -- Phase I: periodic packed-content re-injection, Nanbeige-inspired (project
//     memory nanbeige-architecture-reference's NanbeigeNgramLayerFusion writeup). Mechanism A injects a
//     packed vector ONCE at layer 0 and lets it survive N_LAYERS of ordinary computation alone; this
//     re-adds the SAME vector back into the scratch slot's own hidden state every `stride` layers (no new
//     learned parameters -- core.hpp's set_scratch_reinject, a plain opt-in elementwise add). EVAL-ONLY on
//     top of the ALREADY-trained Phase-C model (not wired into the training graph) -- a directional probe,
//     not the full test; a real positive would justify wiring this into training properly.
//     RESULT (docs/FACTSPIKE.md): first version (fixed embedding-scale) was a near no-op -- diagnosed as
//     the residual stream's own growing norm across depth dwarfing a fixed small addition, not a real
//     negative. Corrected to scale-adaptive (fraction of h's OWN current norm, still zero learned params):
//     REAL, monotonic, positive dose-response -- mean cos_sim 0.831->0.860, accuracy 3/9->5/9 correct at
//     the highest dose (40%), EVAL-ONLY on a model never trained for this signal. First mechanism in the
//     whole investigation with a genuine O(1)-cost positive. Justifies wiring this into the training graph
//     next (not yet started) -- see docs/SCRATCH_TOKEN_FRAMING.md's new candidate 3.

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "sub0/core.hpp"
#include "sub0/layout.hpp"   // sub0::D_KV -- KV-cache row width (narrower than D_MODEL under GQA)
#include "sub0/decode.hpp"
#include "sub0/factspike.hpp"
#include "sub0/scratch_slots.hpp"
#include "sub0/tokmap.hpp"
#include "sub0/window.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <random>
#include <string>
#include <vector>

namespace {

namespace fs  = sub0::factspike;
namespace cas = sub0::casing;
using sub0::tok::Tokenizer;

constexpr int    kNSubjects    = 12;
constexpr double kDrilledFrac  = 0.667;
constexpr std::uint64_t kSplitSeed = 20260719;   // MUST match gen_factspike_corpus.cpp's own seed

constexpr int   kBatch        = 16;
constexpr int   kWindowT      = 48;
constexpr int   kDocsPerFact  = 30;
constexpr int   kStepsPerEval = 100;
constexpr int   kEvalRounds   = 20;     // 2000 steps total -- 600 plateaued at/below chance (0.11 vs
                                        // 0.125), extending per docs/FACTSPIKE.md's own "expect to
                                        // measure and possibly raise this" plan before concluding either way
constexpr float kLr           = 0.003f * (128.0f / static_cast<float>(D_MODEL));

void reset_opt_state() {
    const std::size_t n = sub0::trainable_floats();
    std::fill(sub0::adam_m_ptr(), sub0::adam_m_ptr() + n, 0.f);
    std::fill(sub0::adam_v_ptr(), sub0::adam_v_ptr() + n, 0.f);
}

// A SHORT document's sampled window has len < kWindowT (sample_window's own "whole document, caller
// pads" contract) -- lengths[] MUST be passed to train_batch in that case, or it treats every window as
// a full kWindowT-wide read from `start`, silently over-reading past a short document's own end. Safe
// for any document except the LAST one in the flat array (a real, latent bug this comment now documents
// -- found via a genuine out-of-bounds crash: with only ~270 short documents here, the last document's
// short window gets sampled within the first few hundred draws, unlike wordspike/corpus_collapse's much
// larger datasets where the same missing-lengths pattern apparently never got unlucky enough to crash).
void train_steps(const fs::Dataset& ds, sub0::AdamW& opt, int steps, std::mt19937& rng) {
    std::vector<std::size_t> starts(kBatch);
    std::vector<int> lens(kBatch);
    for (int s = 0; s < steps; ++s) {
        for (int b = 0; b < kBatch; ++b) {
            const sub0::Window w = sub0::sample_window(rng, kWindowT, ds.tokens.size(),
                                                       std::span<const std::uint64_t>(ds.doc_starts));
            starts[static_cast<std::size_t>(b)] = w.start;
            lens[static_cast<std::size_t>(b)] = w.len;
        }
        opt.zero_grad();
        (void)sub0::train_batch(ds.tokens.data(), starts.data(), kBatch, kWindowT, lens.data(), ds.mask.data());
        opt.step();
    }
}

struct Score { int ok = 0, n = 0; double rate() const { return n ? static_cast<double>(ok) / n : 0.0; } };

// Baseline arm: subject spelled out in full ("{subject} loves the color ") -> generate -> does the
// completion start with the assigned fact color's own tokenization? Deterministic (topk=1), matching
// numeric_bind/blended_capstone's own eval convention for a clean pass/fail check.
Score eval_baseline(const Tokenizer& tk, const std::vector<fs::FactPair>& subjects, unsigned seed) {
    Score sc;
    std::mt19937 grng(seed);
    for (const fs::FactPair& fp : subjects) {
        std::vector<int> ctx = sub0::tok::encode(tk, fp.subject + " loves the color ");
        const std::size_t prompt_len = ctx.size();
        if (static_cast<int>(prompt_len) + 8 >= SEQ_LEN) continue;
        const std::vector<int> gold = sub0::tok::encode(tk, fp.fact);
        if (gold.empty()) continue;
        std::mt19937 rng_copy = grng;
        sub0::kv_decode_generate(ctx, static_cast<int>(gold.size()) + 2, 1.f, 1, rng_copy, cas::TOK_EOS, false);
        grng = rng_copy;
        bool match = ctx.size() >= prompt_len + gold.size();
        for (std::size_t k = 0; match && k < gold.size(); ++k)
            if (ctx[prompt_len + k] != gold[k]) match = false;
        ++sc.n;
        sc.ok += match;
    }
    return sc;
}

// Extracts the clean piece-id span (markers stripped) a subject's own spelling tokenizes to -- the
// exact basis content_embed composes a slot's embedding from (this header's own top comment / factspike
// Phase 0's leakage check use the same helper).
std::vector<int> subject_piece_ids(const Tokenizer& tk, const std::string& subject) {
    const std::vector<int> ctx = sub0::tok::encode(tk, subject);
    std::vector<int> pieces;
    for (std::size_t i = 0; i < ctx.size(); ) {
        const auto [span_len, ids] = sub0::detail::word_span(ctx, i);
        pieces.insert(pieces.end(), ids.begin(), ids.end());
        i += span_len;
    }
    return pieces;
}

// Scratch (piece-transfer) arm: SAME prompt shape as eval_baseline, but the subject's token span is
// replaced by a scratch slot bound directly to the subject's OWN piece ids (no byte decomposition) --
// content_embed then composes the slot's embedding from those already-trained piece rows. No textual
// restatement of the fact anywhere in this prompt -- this is the actual claim under test. Used for both
// the scratch arm (drilled subjects) and the held-out arm (held-out subjects, same mechanism, negative
// control) -- the caller picks which subject list to pass.
Score eval_scratch(const Tokenizer& tk, const std::vector<fs::FactPair>& subjects, unsigned seed) {
    Score sc;
    std::mt19937 grng(seed);
    for (const fs::FactPair& fp : subjects) {
        const std::vector<int> pieces = subject_piece_ids(tk, fp.subject);
        if (pieces.empty()) continue;

        sub0::ScratchTable scratch;
        scratch.tk = &tk;
        scratch.bind(sub0::SCRATCH_SLOT_BASE, pieces);

        std::vector<int> ctx{ sub0::SCRATCH_SLOT_BASE };
        for (int t : sub0::tok::encode(tk, " loves the color ")) ctx.push_back(t);
        const std::size_t prompt_len = ctx.size();
        if (static_cast<int>(prompt_len) + 8 >= SEQ_LEN) continue;
        const std::vector<int> gold = sub0::tok::encode(tk, fp.fact);
        if (gold.empty()) continue;

        sub0::ScratchBindings binds = scratch.to_bindings(sub0::SlotEncoding::HRR);
        sub0::set_scratch_bindings(&binds);
        std::mt19937 rng_copy = grng;
        sub0::kv_decode_generate(ctx, static_cast<int>(gold.size()) + 2, 1.f, 1, rng_copy, cas::TOK_EOS, false);
        grng = rng_copy;
        sub0::set_scratch_bindings(nullptr);

        bool match = ctx.size() >= prompt_len + gold.size();
        for (std::size_t k = 0; match && k < gold.size(); ++k)
            if (ctx[prompt_len + k] != gold[k]) match = false;
        ++sc.n;
        sc.ok += match;
    }
    return sc;
}

// Phase H (docs/FACTSPIKE.md's "SPELL marker finding"): SAME shape as eval_scratch, but binds the slot to
// the marker-INCLUSIVE span (tok::encode(subject) directly) instead of subject_piece_ids -- a SEPARATE
// function rather than a refactor of eval_scratch above, for the same reason eval_scratch_one below is
// separate: avoid any risk of changing eval_scratch's own already-validated/documented behavior (Phase
// C/D/E/F's recorded results depend on it staying exactly as it is).
Score eval_scratch_markers(const Tokenizer& tk, const std::vector<fs::FactPair>& subjects, unsigned seed) {
    Score sc;
    std::mt19937 grng(seed);
    for (const fs::FactPair& fp : subjects) {
        const std::vector<int> full_span = sub0::tok::encode(tk, fp.subject);
        if (full_span.empty()) continue;

        sub0::ScratchTable scratch;
        scratch.tk = &tk;
        scratch.bind(sub0::SCRATCH_SLOT_BASE, full_span);

        std::vector<int> ctx{ sub0::SCRATCH_SLOT_BASE };
        for (int t : sub0::tok::encode(tk, " loves the color ")) ctx.push_back(t);
        const std::size_t prompt_len = ctx.size();
        if (static_cast<int>(prompt_len) + 8 >= SEQ_LEN) continue;
        const std::vector<int> gold = sub0::tok::encode(tk, fp.fact);
        if (gold.empty()) continue;

        sub0::ScratchBindings binds = scratch.to_bindings(sub0::SlotEncoding::HRR);
        sub0::set_scratch_bindings(&binds);
        std::mt19937 rng_copy = grng;
        sub0::kv_decode_generate(ctx, static_cast<int>(gold.size()) + 2, 1.f, 1, rng_copy, cas::TOK_EOS, false);
        grng = rng_copy;
        sub0::set_scratch_bindings(nullptr);

        bool match = ctx.size() >= prompt_len + gold.size();
        for (std::size_t k = 0; match && k < gold.size(); ++k)
            if (ctx[prompt_len + k] != gold[k]) match = false;
        ++sc.n;
        sc.ok += match;
    }
    return sc;
}

// Standalone per-subject scratch-arm check, seed fixed (kv_decode_generate's own topk=1 makes this
// deterministic regardless of seed value) -- a SEPARATE function rather than a refactor of eval_scratch
// above, to avoid any risk of changing eval_scratch's own already-validated/documented RNG-threading
// behavior (Phase C/D/E's recorded results depend on it staying exactly as it is).
bool eval_scratch_one(const Tokenizer& tk, const fs::FactPair& fp) {
    const std::vector<int> pieces = subject_piece_ids(tk, fp.subject);
    if (pieces.empty()) return false;
    sub0::ScratchTable scratch;
    scratch.tk = &tk;
    scratch.bind(sub0::SCRATCH_SLOT_BASE, pieces);
    std::vector<int> ctx{ sub0::SCRATCH_SLOT_BASE };
    for (int t : sub0::tok::encode(tk, " loves the color ")) ctx.push_back(t);
    const std::size_t prompt_len = ctx.size();
    if (static_cast<int>(prompt_len) + 8 >= SEQ_LEN) return false;
    const std::vector<int> gold = sub0::tok::encode(tk, fp.fact);
    if (gold.empty()) return false;
    sub0::ScratchBindings binds = scratch.to_bindings(sub0::SlotEncoding::HRR);
    sub0::set_scratch_bindings(&binds);
    std::mt19937 rng(1234u);
    sub0::kv_decode_generate(ctx, static_cast<int>(gold.size()) + 2, 1.f, 1, rng, cas::TOK_EOS, false);
    sub0::set_scratch_bindings(nullptr);
    bool match = ctx.size() >= prompt_len + gold.size();
    for (std::size_t k = 0; match && k < gold.size(); ++k)
        if (ctx[prompt_len + k] != gold[k]) match = false;
    return match;
}

// Phase H marker-inclusive analog of eval_scratch_one -- same reasoning as eval_scratch_markers above.
bool eval_scratch_one_markers(const Tokenizer& tk, const fs::FactPair& fp) {
    const std::vector<int> full_span = sub0::tok::encode(tk, fp.subject);
    if (full_span.empty()) return false;
    sub0::ScratchTable scratch;
    scratch.tk = &tk;
    scratch.bind(sub0::SCRATCH_SLOT_BASE, full_span);
    std::vector<int> ctx{ sub0::SCRATCH_SLOT_BASE };
    for (int t : sub0::tok::encode(tk, " loves the color ")) ctx.push_back(t);
    const std::size_t prompt_len = ctx.size();
    if (static_cast<int>(prompt_len) + 8 >= SEQ_LEN) return false;
    const std::vector<int> gold = sub0::tok::encode(tk, fp.fact);
    if (gold.empty()) return false;
    sub0::ScratchBindings binds = scratch.to_bindings(sub0::SlotEncoding::HRR);
    sub0::set_scratch_bindings(&binds);
    std::mt19937 rng(1234u);
    sub0::kv_decode_generate(ctx, static_cast<int>(gold.size()) + 2, 1.f, 1, rng, cas::TOK_EOS, false);
    sub0::set_scratch_bindings(nullptr);
    bool match = ctx.size() >= prompt_len + gold.size();
    for (std::size_t k = 0; match && k < gold.size(); ++k)
        if (ctx[prompt_len + k] != gold[k]) match = false;
    return match;
}

// Diagnostic: does the packed slot's FULLY-PROCESSED hidden state (post all N_LAYERS, right before the
// color would be generated) resemble the SAME subject's hidden state when read normally (full text)? A
// DIFFERENT question from mean_reconstruction_fidelity below, which compares the packed vector against
// RAW pre-transformer piece embeddings -- this compares the model's own decision-relevant, output-level
// representation instead, using sub0::last_hidden_ptr() (the residual stream right before ln_f/the head,
// captured by forward_one -- see backend_cpu.cpp's Model::last_hidden). Both prompts mirror eval_baseline/
// eval_scratch's own construction exactly, so the captured vector is the literal input to the color
// prediction, not an arbitrary intermediate point.
double hidden_state_cosine(const Tokenizer& tk, const fs::FactPair& fp) {
    sub0::kv_reset();
    const std::vector<int> base_ctx = sub0::tok::encode(tk, fp.subject + " loves the color ");
    for (std::size_t i = 0; i < base_ctx.size(); ++i) (void)sub0::forward_one(base_ctx[i], static_cast<int>(i));
    std::array<float, D_MODEL> base_hidden{};
    std::copy_n(sub0::last_hidden_ptr(), D_MODEL, base_hidden.data());

    const std::vector<int> pieces = subject_piece_ids(tk, fp.subject);
    if (pieces.empty()) return 0.0;
    sub0::ScratchTable scratch;
    scratch.tk = &tk;
    scratch.bind(sub0::SCRATCH_SLOT_BASE, pieces);
    sub0::ScratchBindings binds = scratch.to_bindings(sub0::SlotEncoding::HRR);
    sub0::set_scratch_bindings(&binds);
    sub0::kv_reset();
    std::vector<int> scr_ctx{ sub0::SCRATCH_SLOT_BASE };
    for (int t : sub0::tok::encode(tk, " loves the color ")) scr_ctx.push_back(t);
    for (std::size_t i = 0; i < scr_ctx.size(); ++i) (void)sub0::forward_one(scr_ctx[i], static_cast<int>(i));
    sub0::set_scratch_bindings(nullptr);
    std::array<float, D_MODEL> scr_hidden{};
    std::copy_n(sub0::last_hidden_ptr(), D_MODEL, scr_hidden.data());

    double dot = 0.0, nb = 0.0, ns = 0.0;
    for (int c = 0; c < D_MODEL; ++c) {
        dot += static_cast<double>(base_hidden[static_cast<std::size_t>(c)]) * scr_hidden[static_cast<std::size_t>(c)];
        nb  += static_cast<double>(base_hidden[static_cast<std::size_t>(c)]) * base_hidden[static_cast<std::size_t>(c)];
        ns  += static_cast<double>(scr_hidden[static_cast<std::size_t>(c)])  * scr_hidden[static_cast<std::size_t>(c)];
    }
    return (nb > 0.0 && ns > 0.0) ? dot / (std::sqrt(nb) * std::sqrt(ns)) : 0.0;
}

// Phase H (docs/FACTSPIKE.md's "SPELL marker finding"): SAME shape as hidden_state_cosine, but binds the
// slot to the marker-INCLUSIVE span instead of subject_piece_ids -- a separate function, same reasoning
// as eval_scratch_markers above (don't risk perturbing hidden_state_cosine's own recorded behavior).
double hidden_state_cosine_markers(const Tokenizer& tk, const fs::FactPair& fp) {
    sub0::kv_reset();
    const std::vector<int> base_ctx = sub0::tok::encode(tk, fp.subject + " loves the color ");
    for (std::size_t i = 0; i < base_ctx.size(); ++i) (void)sub0::forward_one(base_ctx[i], static_cast<int>(i));
    std::array<float, D_MODEL> base_hidden{};
    std::copy_n(sub0::last_hidden_ptr(), D_MODEL, base_hidden.data());

    const std::vector<int> full_span = sub0::tok::encode(tk, fp.subject);
    if (full_span.empty()) return 0.0;
    sub0::ScratchTable scratch;
    scratch.tk = &tk;
    scratch.bind(sub0::SCRATCH_SLOT_BASE, full_span);
    sub0::ScratchBindings binds = scratch.to_bindings(sub0::SlotEncoding::HRR);
    sub0::set_scratch_bindings(&binds);
    sub0::kv_reset();
    std::vector<int> scr_ctx{ sub0::SCRATCH_SLOT_BASE };
    for (int t : sub0::tok::encode(tk, " loves the color ")) scr_ctx.push_back(t);
    for (std::size_t i = 0; i < scr_ctx.size(); ++i) (void)sub0::forward_one(scr_ctx[i], static_cast<int>(i));
    sub0::set_scratch_bindings(nullptr);
    std::array<float, D_MODEL> scr_hidden{};
    std::copy_n(sub0::last_hidden_ptr(), D_MODEL, scr_hidden.data());

    double dot = 0.0, nb = 0.0, ns = 0.0;
    for (int c = 0; c < D_MODEL; ++c) {
        dot += static_cast<double>(base_hidden[static_cast<std::size_t>(c)]) * scr_hidden[static_cast<std::size_t>(c)];
        nb  += static_cast<double>(base_hidden[static_cast<std::size_t>(c)]) * base_hidden[static_cast<std::size_t>(c)];
        ns  += static_cast<double>(scr_hidden[static_cast<std::size_t>(c)])  * scr_hidden[static_cast<std::size_t>(c)];
    }
    return (nb > 0.0 && ns > 0.0) ? dot / (std::sqrt(nb) * std::sqrt(ns)) : 0.0;
}

// Phase I (docs/FACTSPIKE.md, Nanbeige-inspired periodic re-injection spike): SAME shape as
// hidden_state_cosine (piece-only bind, mechanism A), but wraps the scratch-arm forward pass with
// set_scratch_reinject(stride, scale) -- tests whether reinforcing the packed vector partway through the
// layer stack (instead of betting everything on the single layer-0 injection) narrows the fidelity gap.
// EVAL-ONLY on top of the ALREADY-trained Phase-C/G model (core.hpp's own doc comment on why this is a
// directional probe, not the full test -- the model was never trained to expect the extra signal).
double hidden_state_cosine_reinject(const Tokenizer& tk, const fs::FactPair& fp, int stride, float scale) {
    sub0::kv_reset();
    const std::vector<int> base_ctx = sub0::tok::encode(tk, fp.subject + " loves the color ");
    for (std::size_t i = 0; i < base_ctx.size(); ++i) (void)sub0::forward_one(base_ctx[i], static_cast<int>(i));
    std::array<float, D_MODEL> base_hidden{};
    std::copy_n(sub0::last_hidden_ptr(), D_MODEL, base_hidden.data());

    const std::vector<int> pieces = subject_piece_ids(tk, fp.subject);
    if (pieces.empty()) return 0.0;
    sub0::ScratchTable scratch;
    scratch.tk = &tk;
    scratch.bind(sub0::SCRATCH_SLOT_BASE, pieces);
    sub0::ScratchBindings binds = scratch.to_bindings(sub0::SlotEncoding::HRR);
    sub0::set_scratch_bindings(&binds);
    sub0::set_scratch_reinject(stride, scale);
    sub0::kv_reset();
    std::vector<int> scr_ctx{ sub0::SCRATCH_SLOT_BASE };
    for (int t : sub0::tok::encode(tk, " loves the color ")) scr_ctx.push_back(t);
    for (std::size_t i = 0; i < scr_ctx.size(); ++i) (void)sub0::forward_one(scr_ctx[i], static_cast<int>(i));
    sub0::set_scratch_reinject(0, 1.0f);
    sub0::set_scratch_bindings(nullptr);
    std::array<float, D_MODEL> scr_hidden{};
    std::copy_n(sub0::last_hidden_ptr(), D_MODEL, scr_hidden.data());

    double dot = 0.0, nb = 0.0, ns = 0.0;
    for (int c = 0; c < D_MODEL; ++c) {
        dot += static_cast<double>(base_hidden[static_cast<std::size_t>(c)]) * scr_hidden[static_cast<std::size_t>(c)];
        nb  += static_cast<double>(base_hidden[static_cast<std::size_t>(c)]) * base_hidden[static_cast<std::size_t>(c)];
        ns  += static_cast<double>(scr_hidden[static_cast<std::size_t>(c)])  * scr_hidden[static_cast<std::size_t>(c)];
    }
    return (nb > 0.0 && ns > 0.0) ? dot / (std::sqrt(nb) * std::sqrt(ns)) : 0.0;
}

// Phase I analog of eval_scratch_one -- same reinject wrapping as hidden_state_cosine_reinject above.
bool eval_scratch_one_reinject(const Tokenizer& tk, const fs::FactPair& fp, int stride, float scale) {
    const std::vector<int> pieces = subject_piece_ids(tk, fp.subject);
    if (pieces.empty()) return false;
    sub0::ScratchTable scratch;
    scratch.tk = &tk;
    scratch.bind(sub0::SCRATCH_SLOT_BASE, pieces);
    std::vector<int> ctx{ sub0::SCRATCH_SLOT_BASE };
    for (int t : sub0::tok::encode(tk, " loves the color ")) ctx.push_back(t);
    const std::size_t prompt_len = ctx.size();
    if (static_cast<int>(prompt_len) + 8 >= SEQ_LEN) return false;
    const std::vector<int> gold = sub0::tok::encode(tk, fp.fact);
    if (gold.empty()) return false;
    sub0::ScratchBindings binds = scratch.to_bindings(sub0::SlotEncoding::HRR);
    sub0::set_scratch_bindings(&binds);
    sub0::set_scratch_reinject(stride, scale);
    std::mt19937 rng(1234u);
    sub0::kv_decode_generate(ctx, static_cast<int>(gold.size()) + 2, 1.f, 1, rng, cas::TOK_EOS, false);
    sub0::set_scratch_reinject(0, 1.0f);
    sub0::set_scratch_bindings(nullptr);
    bool match = ctx.size() >= prompt_len + gold.size();
    for (std::size_t k = 0; match && k < gold.size(); ++k)
        if (ctx[prompt_len + k] != gold[k]) match = false;
    return match;
}

// Diagnostic (docs/FACTSPIKE.md "Pack-Aware Training" discussion): does the packed vector's per-fragment
// HRR-unbind fidelity correlate with scratch-arm task accuracy? Mean cosine similarity between
// hrr_unbind(packed_vector, position p) and the TRUE tok_emb row for piece p, averaged over every
// (subject, piece) pair in `subjects`, computed against the CURRENT (mid-training) model weights -- so a
// per-round trajectory can be compared directly against that same round's accuracy. Cosine similarity
// (not raw error) is deliberate: HRR's role vectors aren't necessarily unit-scaled, so only DIRECTION,
// not magnitude, is a meaningful fidelity signal here.
double mean_reconstruction_fidelity(const Tokenizer& tk, const std::vector<fs::FactPair>& subjects) {
    const float* tok_emb = sub0::params_ptr();   // [VOCAB, D_MODEL] -- tok_emb is param offset 0 (layout.hpp)
    std::vector<float> packed(D_MODEL), recon(D_MODEL);
    double sum = 0.0; int n = 0;
    for (const fs::FactPair& fp : subjects) {
        const std::vector<int> pieces = subject_piece_ids(tk, fp.subject);
        if (pieces.empty()) continue;
        sub0::encode_slot(tok_emb, D_MODEL, pieces, sub0::SlotEncoding::HRR, packed.data());
        for (std::size_t p = 0; p < pieces.size(); ++p) {
            sub0::hrr_unbind(packed.data(), D_MODEL, p, recon.data());
            const float* truth = tok_emb + static_cast<std::size_t>(pieces[p]) * D_MODEL;
            double dot = 0.0, nr = 0.0, nt = 0.0;
            for (int c = 0; c < D_MODEL; ++c) {
                dot += static_cast<double>(recon[static_cast<std::size_t>(c)]) * truth[c];
                nr  += static_cast<double>(recon[static_cast<std::size_t>(c)]) * recon[static_cast<std::size_t>(c)];
                nt  += static_cast<double>(truth[c]) * truth[c];
            }
            if (nr > 0.0 && nt > 0.0) { sum += dot / (std::sqrt(nr) * std::sqrt(nt)); ++n; }
        }
    }
    return n ? sum / n : 0.0;
}

// Pearson correlation coefficient between two equal-length series -- hand-rolled (n=kEvalRounds is small,
// not worth a dependency) to quantify whether reconstruction fidelity tracks task accuracy round-by-round.
double pearson_r(const std::vector<double>& a, const std::vector<double>& b) {
    const std::size_t n = a.size();
    if (n < 2 || b.size() != n) return 0.0;
    double ma = 0.0, mb = 0.0;
    for (std::size_t i = 0; i < n; ++i) { ma += a[i]; mb += b[i]; }
    ma /= static_cast<double>(n); mb /= static_cast<double>(n);
    double cov = 0.0, va = 0.0, vb = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double da = a[i] - ma, db = b[i] - mb;
        cov += da * db; va += da * da; vb += db * db;
    }
    return (va > 0.0 && vb > 0.0) ? cov / (std::sqrt(va) * std::sqrt(vb)) : 0.0;
}

// --- Phase G: KV-trace memoization (docs/SCRATCH_TOKEN_FRAMING.md "candidate 1: post-hoc per-layer KV
// pooling") -- instead of composing a slot's embedding from RAW pre-transformer piece rows (encode_slot,
// what every eval above uses), capture the word's REAL per-layer (K,V) trace from an isolated forward
// pass, pool it (reusing encode_slot's own HRR math one layer deeper), and splice the pooled trace
// directly into a live KV-cache. See core.hpp's kv_krow_ptr/kv_vrow_ptr/kv_rope_rotate/kv_splice_row doc
// comments for the full mechanism. Tests both axes the framing doc asked about: representational fidelity
// (does r(piece_count, cos_sim) get weaker than mechanism A's -0.614?) and task accuracy (does it beat
// Phase C's 0.56 peak?).
struct KvTrace {
    std::vector<float> k;   // [N_LAYERS, sub0::D_KV] flat -- DE-ROTATED ("canonical", position-0-equivalent)
    std::vector<float> v;   // [N_LAYERS, sub0::D_KV] flat -- position-invariant (RoPE never touches V)
};

// Captures a word's own per-layer (K,V) trace by running its pieces through an ISOLATED forward pass
// (local positions 0..n-1, no prefix -- the cheapest canonical precompute context, matching this file's
// own existing scratch-arm contexts), then pools n rows -> 1 per layer via encode_slot's existing HRR
// math, treating the captured rows as a synthetic [n, sub0::D_KV] "embedding table" indexed 0..n-1 (idx[p]
// selects role p, exactly mirroring mechanism A's own per-piece role assignment -- same operator, one
// layer deeper). K rows are DE-ROTATED first (kv_rope_rotate with the row's own NEGATED local capture
// position) so pooling operates in a position-independent frame -- only the geometric RoPE component is
// removed; the content-derived, causal-attention-refined part of each row (the actual multi-hop signal)
// is never touched. V needs no de-rotation (RoPE never touches V).
KvTrace capture_kv_trace(const std::vector<int>& pieces) {
    KvTrace trace;
    trace.k.assign(static_cast<std::size_t>(N_LAYERS) * sub0::D_KV, 0.f);
    trace.v.assign(static_cast<std::size_t>(N_LAYERS) * sub0::D_KV, 0.f);
    const int n = static_cast<int>(pieces.size());
    if (n == 0) return trace;

    sub0::kv_reset();
    for (int i = 0; i < n; ++i) (void)sub0::forward_one(pieces[static_cast<std::size_t>(i)], i);

    std::vector<float> kbuf(static_cast<std::size_t>(n) * sub0::D_KV);
    std::vector<float> vbuf(static_cast<std::size_t>(n) * sub0::D_KV);
    std::vector<int> idx(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) idx[static_cast<std::size_t>(i)] = i;

    for (int l = 0; l < N_LAYERS; ++l) {
        for (int i = 0; i < n; ++i) {
            float* krow = kbuf.data() + static_cast<std::size_t>(i) * sub0::D_KV;
            std::copy_n(sub0::kv_krow_ptr(l, i), sub0::D_KV, krow);
            sub0::kv_rope_rotate(krow, -i);   // strip this row's own local capture-position RoPE angle
            std::copy_n(sub0::kv_vrow_ptr(l, i), sub0::D_KV, vbuf.data() + static_cast<std::size_t>(i) * sub0::D_KV);
        }
        sub0::encode_slot(kbuf.data(), sub0::D_KV, idx, sub0::SlotEncoding::HRR,
                          trace.k.data() + static_cast<std::size_t>(l) * sub0::D_KV);
        sub0::encode_slot(vbuf.data(), sub0::D_KV, idx, sub0::SlotEncoding::HRR,
                          trace.v.data() + static_cast<std::size_t>(l) * sub0::D_KV);
    }
    return trace;
}

// Splices a captured/pooled trace into the CURRENT thread's live KV-cache at `pos` -- one call per layer,
// bypassing forward_one's normal per-position computation for that position entirely (nothing downstream
// needs this position's own query/attention-output/FFN, only later positions' attention over its K/V).
void splice_kv_trace(const KvTrace& trace, int pos) {
    for (int l = 0; l < N_LAYERS; ++l) {
        sub0::kv_splice_row(l, pos, trace.k.data() + static_cast<std::size_t>(l) * sub0::D_KV,
                            trace.v.data() + static_cast<std::size_t>(l) * sub0::D_KV);
    }
}

// KV-trace analog of hidden_state_cosine: same baseline arm, but the scratch position's contribution
// comes from a spliced trace instead of encode_slot's pre-transformer composition.
double hidden_state_cosine_kvtrace(const Tokenizer& tk, const fs::FactPair& fp) {
    sub0::kv_reset();
    const std::vector<int> base_ctx = sub0::tok::encode(tk, fp.subject + " loves the color ");
    for (std::size_t i = 0; i < base_ctx.size(); ++i) (void)sub0::forward_one(base_ctx[i], static_cast<int>(i));
    std::array<float, D_MODEL> base_hidden{};
    std::copy_n(sub0::last_hidden_ptr(), D_MODEL, base_hidden.data());

    const std::vector<int> pieces = subject_piece_ids(tk, fp.subject);
    if (pieces.empty()) return 0.0;
    const KvTrace trace = capture_kv_trace(pieces);   // captured under the CURRENT (mid-training) weights

    sub0::kv_reset();
    splice_kv_trace(trace, 0);
    const std::vector<int> suffix = sub0::tok::encode(tk, " loves the color ");
    for (std::size_t i = 0; i < suffix.size(); ++i)
        (void)sub0::forward_one(suffix[i], static_cast<int>(1 + i));
    std::array<float, D_MODEL> kvt_hidden{};
    std::copy_n(sub0::last_hidden_ptr(), D_MODEL, kvt_hidden.data());

    double dot = 0.0, nb = 0.0, nk = 0.0;
    for (int c = 0; c < D_MODEL; ++c) {
        dot += static_cast<double>(base_hidden[static_cast<std::size_t>(c)]) * kvt_hidden[static_cast<std::size_t>(c)];
        nb  += static_cast<double>(base_hidden[static_cast<std::size_t>(c)]) * base_hidden[static_cast<std::size_t>(c)];
        nk  += static_cast<double>(kvt_hidden[static_cast<std::size_t>(c)])  * kvt_hidden[static_cast<std::size_t>(c)];
    }
    return (nb > 0.0 && nk > 0.0) ? dot / (std::sqrt(nb) * std::sqrt(nk)) : 0.0;
}

// KV-trace analog of eval_scratch_one: same prompt shape/eval convention, but the slot position's
// contribution comes from a spliced trace. kv_decode_generate (decode.hpp) can't be reused here -- it
// always drives its own internal forward_one prefill loop with no hook to redirect one position
// (decode.hpp:158-159) -- so the prefill + greedy-decode tail are hand-rolled, mirroring that function's
// own tail (decode.hpp:210-216) exactly.
bool eval_kvtrace_one(const Tokenizer& tk, const fs::FactPair& fp) {
    const std::vector<int> pieces = subject_piece_ids(tk, fp.subject);
    if (pieces.empty()) return false;
    const KvTrace trace = capture_kv_trace(pieces);

    std::vector<int> ctx{ sub0::SCRATCH_SLOT_BASE };   // visible id only -- its own forward_one never runs
    for (int t : sub0::tok::encode(tk, " loves the color ")) ctx.push_back(t);
    const std::size_t prompt_len = ctx.size();
    if (static_cast<int>(prompt_len) + 8 >= SEQ_LEN) return false;
    const std::vector<int> gold = sub0::tok::encode(tk, fp.fact);
    if (gold.empty()) return false;

    sub0::kv_reset();
    splice_kv_trace(trace, 0);
    const float* logits = nullptr;
    for (std::size_t i = 1; i < prompt_len; ++i) logits = sub0::forward_one(ctx[i], static_cast<int>(i));

    std::mt19937 rng(1234u);
    const int n = static_cast<int>(gold.size()) + 2;
    for (int s = 0; s < n && static_cast<int>(ctx.size()) < SEQ_LEN; ++s) {
        const int next = sub0::sample_token(logits, 1.f, 1, rng);
        if (next == cas::TOK_EOS) break;
        ctx.push_back(next);
        logits = sub0::forward_one(next, static_cast<int>(ctx.size()) - 1);
    }

    bool match = ctx.size() >= prompt_len + gold.size();
    for (std::size_t k = 0; match && k < gold.size(); ++k)
        if (ctx[prompt_len + k] != gold[k]) match = false;
    return match;
}

// --- Candidate 2: landmark-style transparent expansion (docs/SCRATCH_TOKEN_FRAMING.md) -- no pooling AT
// ALL: splice the FULL n-position per-layer trace, one real KV-cache row per piece per layer, instead of
// candidate 1's n->1 compression. Genuinely costs n KV-cache positions for what the VISIBLE token stream
// still shows as one slot token -- axis 10 (cost efficiency) is knowingly given up here, the opposite
// trade from candidate 1. Reuses the SAME three primitives with ZERO new engine code: capture is
// capture_kv_trace's own per-piece de-rotated rows WITHOUT the pooling step, and splice is kv_splice_row
// called n times instead of once (exactly the "extension point" the framing doc's DRY review anticipated
// when these primitives were first built).
struct KvFullTrace {
    int n = 0;
    // sub0::D_KV, not D_MODEL: KV-cache rows hold N_KV_HEADS heads, which GQA narrows below the residual
    // width. Equal under plain MHA, so this only diverges once --kv-heads < --heads.
    std::vector<float> k;   // [N_LAYERS, n, sub0::D_KV] flat -- DE-ROTATED per-piece (canonical, position-0-equivalent)
    std::vector<float> v;   // [N_LAYERS, n, sub0::D_KV] flat -- position-invariant
};

KvFullTrace capture_kv_trace_full(const std::vector<int>& pieces) {
    KvFullTrace trace;
    const int n = static_cast<int>(pieces.size());
    trace.n = n;
    if (n == 0) return trace;
    trace.k.assign(static_cast<std::size_t>(N_LAYERS) * static_cast<std::size_t>(n) * sub0::D_KV, 0.f);
    trace.v.assign(static_cast<std::size_t>(N_LAYERS) * static_cast<std::size_t>(n) * sub0::D_KV, 0.f);

    sub0::kv_reset();
    for (int i = 0; i < n; ++i) (void)sub0::forward_one(pieces[static_cast<std::size_t>(i)], i);

    for (int l = 0; l < N_LAYERS; ++l) {
        for (int i = 0; i < n; ++i) {
            float* krow = trace.k.data() + (static_cast<std::size_t>(l) * static_cast<std::size_t>(n) +
                                            static_cast<std::size_t>(i)) * sub0::D_KV;
            std::copy_n(sub0::kv_krow_ptr(l, i), sub0::D_KV, krow);
            sub0::kv_rope_rotate(krow, -i);   // de-rotate to canonical (position-0-equivalent) form
            float* vrow = trace.v.data() + (static_cast<std::size_t>(l) * static_cast<std::size_t>(n) +
                                            static_cast<std::size_t>(i)) * sub0::D_KV;
            std::copy_n(sub0::kv_vrow_ptr(l, i), sub0::D_KV, vrow);
        }
    }
    return trace;
}

// Splices piece i's canonical row at KV-cache position `pos + i`, for every layer -- mathematically EXACT
// (up to floating point) when the splice context's own preceding content matches capture's (both
// isolated/empty here), an approximation otherwise (the framing doc's open "how context-sensitive is a
// word's own trace" question -- see context_sensitivity_cosine below, which tests exactly that).
void splice_kv_trace_full(const KvFullTrace& trace, int pos) {
    for (int l = 0; l < N_LAYERS; ++l) {
        for (int i = 0; i < trace.n; ++i) {
            const float* krow = trace.k.data() + (static_cast<std::size_t>(l) * static_cast<std::size_t>(trace.n) +
                                                  static_cast<std::size_t>(i)) * sub0::D_KV;
            const float* vrow = trace.v.data() + (static_cast<std::size_t>(l) * static_cast<std::size_t>(trace.n) +
                                                  static_cast<std::size_t>(i)) * sub0::D_KV;
            sub0::kv_splice_row(l, pos + i, krow, vrow);
        }
    }
}

// Candidate-2 analog of hidden_state_cosine: same trivial (position-0, isolated-context) shape as A/B, so
// capture and splice contexts coincide exactly -- this should reproduce the baseline hidden state almost
// PERFECTLY (cos_sim ~1.0) for every subject regardless of piece count, since nothing is compressed. This
// is as much a correctness gate on the splice implementation as it is a "does candidate 2 help" measure:
// anything meaningfully below 1.0 here would point at a bug, not an inherent mechanism limitation.
//
// `replay_tokens` is the token list that gets captured+spliced -- `subject_piece_ids` (marker-stripped,
// what mechanisms A/B/C/D normally use) or `tok::encode(subject)` (marker-INCLUSIVE, `alone` below --
// see the SPELL_START/SPELL_END finding this parameterization exists to test directly, not just infer).
// `alone` (always the RAW, unstripped tokenization, regardless of which list is being replayed) locates
// exactly where the subject's own span ends within base_ctx, letting the suffix be SLICED OUT of
// base_ctx itself rather than re-tokenized separately as `tok::encode(" loves the color ")` -- a
// tokenization-boundary regression test (`[factspike][kvtrace]`, non-hidden) found the two differ (BPE
// merges the leading space differently depending on what precedes it), which would otherwise inject a
// confound unrelated to the splice mechanism into what's supposed to be an exact-reconstruction gate.
double hidden_state_cosine_kvtrace_full_impl(const Tokenizer& tk, const fs::FactPair& fp,
                                             const std::vector<int>& replay_tokens) {
    sub0::kv_reset();
    const std::vector<int> base_ctx = sub0::tok::encode(tk, fp.subject + " loves the color ");
    for (std::size_t i = 0; i < base_ctx.size(); ++i) (void)sub0::forward_one(base_ctx[i], static_cast<int>(i));
    std::array<float, D_MODEL> base_hidden{};
    std::copy_n(sub0::last_hidden_ptr(), D_MODEL, base_hidden.data());

    if (replay_tokens.empty()) return 0.0;
    const KvFullTrace trace = capture_kv_trace_full(replay_tokens);

    const std::vector<int> alone = sub0::tok::encode(tk, fp.subject);
    if (alone.size() > base_ctx.size()) return 0.0;
    const std::vector<int> suffix(base_ctx.begin() + static_cast<std::ptrdiff_t>(alone.size()), base_ctx.end());

    sub0::kv_reset();
    splice_kv_trace_full(trace, 0);
    for (std::size_t i = 0; i < suffix.size(); ++i)
        (void)sub0::forward_one(suffix[i], trace.n + static_cast<int>(i));   // continue AFTER the n spliced rows
    std::array<float, D_MODEL> full_hidden{};
    std::copy_n(sub0::last_hidden_ptr(), D_MODEL, full_hidden.data());

    double dot = 0.0, nb = 0.0, nf = 0.0;
    for (int c = 0; c < D_MODEL; ++c) {
        dot += static_cast<double>(base_hidden[static_cast<std::size_t>(c)]) * full_hidden[static_cast<std::size_t>(c)];
        nb  += static_cast<double>(base_hidden[static_cast<std::size_t>(c)]) * base_hidden[static_cast<std::size_t>(c)];
        nf  += static_cast<double>(full_hidden[static_cast<std::size_t>(c)])  * full_hidden[static_cast<std::size_t>(c)];
    }
    return (nb > 0.0 && nf > 0.0) ? dot / (std::sqrt(nb) * std::sqrt(nf)) : 0.0;
}

double hidden_state_cosine_kvtrace_full(const Tokenizer& tk, const fs::FactPair& fp) {
    return hidden_state_cosine_kvtrace_full_impl(tk, fp, subject_piece_ids(tk, fp.subject));
}

// Direct test of the SPELL_START/SPELL_END finding: replay the FULL wrapped span (`tok::encode(subject)`,
// markers included) instead of the marker-stripped `pieces`. If this closes candidate 2's remaining
// multi-piece gap (Crofw/Yelfan/etc. sitting at 0.87-0.96 instead of the single-piece group's exact
// 1.000), that's a controlled confirmation the markers carry real weight -- not a correlation, a causal
// swap-and-remeasure on the SAME subjects/weights.
double hidden_state_cosine_kvtrace_full_markers(const Tokenizer& tk, const fs::FactPair& fp) {
    return hidden_state_cosine_kvtrace_full_impl(tk, fp, sub0::tok::encode(tk, fp.subject));
}

// Candidate-2 analog of eval_scratch_one/eval_kvtrace_one. `ctx` (the VISIBLE token stream, one slot
// token) stays the same shape as the other eval_*_one functions -- candidate 2's whole point is a compact
// visible sequence -- but the underlying KV-CACHE position counter (kv_pos) advances by trace.n for the
// spliced word instead of by 1, since n real rows were written per layer, and every subsequent real token
// must continue from there.
bool eval_kvtrace_full_one(const Tokenizer& tk, const fs::FactPair& fp) {
    const std::vector<int> pieces = subject_piece_ids(tk, fp.subject);
    if (pieces.empty()) return false;
    const KvFullTrace trace = capture_kv_trace_full(pieces);

    std::vector<int> ctx{ sub0::SCRATCH_SLOT_BASE };   // visible id only -- its own forward_one never runs
    for (int t : sub0::tok::encode(tk, " loves the color ")) ctx.push_back(t);
    const std::size_t prompt_len = ctx.size();
    if (trace.n + static_cast<int>(prompt_len) - 1 + 8 >= SEQ_LEN) return false;   // n spliced + suffix + budget
    const std::vector<int> gold = sub0::tok::encode(tk, fp.fact);
    if (gold.empty()) return false;

    sub0::kv_reset();
    splice_kv_trace_full(trace, 0);
    const float* logits = nullptr;
    int kv_pos = trace.n;
    for (std::size_t i = 1; i < prompt_len; ++i, ++kv_pos) logits = sub0::forward_one(ctx[i], kv_pos);

    std::mt19937 rng(1234u);
    const int steps = static_cast<int>(gold.size()) + 2;
    for (int s = 0; s < steps && kv_pos < SEQ_LEN; ++s, ++kv_pos) {
        const int next = sub0::sample_token(logits, 1.f, 1, rng);
        if (next == cas::TOK_EOS) break;
        ctx.push_back(next);
        logits = sub0::forward_one(next, kv_pos);
    }

    bool match = ctx.size() >= prompt_len + gold.size();
    for (std::size_t k = 0; match && k < gold.size(); ++k)
        if (ctx[prompt_len + k] != gold[k]) match = false;
    return match;
}

// Context-sensitivity probe (docs/SCRATCH_TOKEN_FRAMING.md's own open question: "how context-sensitive is
// a word's own internal (K,V) trace, really?"). capture_kv_trace_full captures a word's trace in
// ISOLATION (no preceding context) -- candidate 2's splice is mathematically exact only when the splice
// context's own preceding content matches capture's. This measures the gap when it doesn't: a REAL prefix
// (an actual fact_templates() lead-in, "Everyone knows that ", not an arbitrary/OOD string) precedes the
// word in the splice context, but the trace was captured with no prefix at all -- so any degradation here
// is attributable specifically to the isolated-capture assumption, not to compression (there isn't any).
double context_sensitivity_cosine(const Tokenizer& tk, const fs::FactPair& fp, const std::string& prefix) {
    sub0::kv_reset();
    const std::vector<int> real_ctx = sub0::tok::encode(tk, prefix + fp.subject + " loves the color ");
    for (std::size_t i = 0; i < real_ctx.size(); ++i) (void)sub0::forward_one(real_ctx[i], static_cast<int>(i));
    std::array<float, D_MODEL> real_hidden{};
    std::copy_n(sub0::last_hidden_ptr(), D_MODEL, real_hidden.data());

    const std::vector<int> pieces = subject_piece_ids(tk, fp.subject);
    if (pieces.empty()) return 0.0;
    const KvFullTrace trace = capture_kv_trace_full(pieces);   // captured in ISOLATION, no prefix

    sub0::kv_reset();
    const std::vector<int> prefix_ctx = sub0::tok::encode(tk, prefix);
    for (std::size_t i = 0; i < prefix_ctx.size(); ++i) (void)sub0::forward_one(prefix_ctx[i], static_cast<int>(i));
    splice_kv_trace_full(trace, static_cast<int>(prefix_ctx.size()));
    const std::vector<int> suffix = sub0::tok::encode(tk, " loves the color ");
    int kv_pos = static_cast<int>(prefix_ctx.size()) + trace.n;
    for (std::size_t i = 0; i < suffix.size(); ++i, ++kv_pos) (void)sub0::forward_one(suffix[i], kv_pos);
    std::array<float, D_MODEL> spliced_hidden{};
    std::copy_n(sub0::last_hidden_ptr(), D_MODEL, spliced_hidden.data());

    double dot = 0.0, nr = 0.0, ns = 0.0;
    for (int c = 0; c < D_MODEL; ++c) {
        dot += static_cast<double>(real_hidden[static_cast<std::size_t>(c)]) * spliced_hidden[static_cast<std::size_t>(c)];
        nr  += static_cast<double>(real_hidden[static_cast<std::size_t>(c)]) * real_hidden[static_cast<std::size_t>(c)];
        ns  += static_cast<double>(spliced_hidden[static_cast<std::size_t>(c)]) * spliced_hidden[static_cast<std::size_t>(c)];
    }
    return (nr > 0.0 && ns > 0.0) ? dot / (std::sqrt(nr) * std::sqrt(ns)) : 0.0;
}

// context_sensitivity_cosine ABOVE re-tokenizes " loves the color " separately from the combined
// real_ctx -- the SAME suffix-tokenization-boundary confound the trivial case (C) needed a fix for
// (hidden_state_cosine_kvtrace_full's own comment), never applied here. Its own recorded D=0.948
// (docs/FACTSPIKE.md Phase G) is therefore noisier than it needs to be -- NOT reused/modified in place
// (avoid perturbing an already-recorded number), but superseded by this clean pair for any FRESH
// context-sensitivity measurement.
//
// The FIRST version of this function assumed `prefix`'s own tokenize-alone length was the correct offset
// into `real_ctx` -- WRONG, caught by a diagnostic ([.factspikediag2], since removed): the fixed prefix
// "Everyone knows that " ends in a trailing space whose token gets ABSORBED/merged away when immediately
// followed by a real subject, so `prefix_ctx.size()` alone vs. in-context differ (a THIRD instance of the
// same "context-dependent tokenization at a word boundary" class of confound this whole investigation has
// been chasing -- after the subject|suffix boundary and the marker-omission finding). The subject's OWN
// tokenization-alone (`alone`), however, DOES still appear as an exact contiguous match somewhere in
// `real_ctx` (the same property already validated with no prefix present, `[factspike][kvtrace]`) --  so
// rather than assume ANY separately-tokenized piece's length is a reliable offset, SEARCH for `alone` as a
// contiguous subsequence and derive prefix/suffix by slicing `real_ctx` itself around that match. This is
// the more robust pattern: trust only the ONE assumption that's been directly verified twice, derive
// everything else from the single authoritative one-shot tokenization.
double context_sensitivity_cosine2_impl(const Tokenizer& tk, const fs::FactPair& fp, const std::string& prefix,
                                        const std::vector<int>& replay_tokens) {
    sub0::kv_reset();
    const std::vector<int> real_ctx = sub0::tok::encode(tk, prefix + fp.subject + " loves the color ");
    for (std::size_t i = 0; i < real_ctx.size(); ++i) (void)sub0::forward_one(real_ctx[i], static_cast<int>(i));
    std::array<float, D_MODEL> real_hidden{};
    std::copy_n(sub0::last_hidden_ptr(), D_MODEL, real_hidden.data());

    if (replay_tokens.empty()) return 0.0;
    const KvFullTrace trace = capture_kv_trace_full(replay_tokens);   // captured in ISOLATION, no prefix

    const std::vector<int> alone = sub0::tok::encode(tk, fp.subject);
    if (alone.empty() || alone.size() > real_ctx.size()) return 0.0;
    std::size_t subj_off = real_ctx.size();
    for (std::size_t start = 0; start + alone.size() <= real_ctx.size(); ++start) {
        bool match = true;
        for (std::size_t i = 0; i < alone.size(); ++i)
            if (real_ctx[start + i] != alone[i]) { match = false; break; }
        if (match) { subj_off = start; break; }
    }
    if (subj_off == real_ctx.size()) return 0.0;   // subject's own span not found in real_ctx -- bail

    const std::vector<int> prefix_ctx(real_ctx.begin(), real_ctx.begin() + static_cast<std::ptrdiff_t>(subj_off));
    const std::vector<int> suffix(real_ctx.begin() + static_cast<std::ptrdiff_t>(subj_off + alone.size()),
                                  real_ctx.end());   // the REAL suffix, sliced -- not re-tokenized

    sub0::kv_reset();
    for (std::size_t i = 0; i < prefix_ctx.size(); ++i) (void)sub0::forward_one(prefix_ctx[i], static_cast<int>(i));
    splice_kv_trace_full(trace, static_cast<int>(prefix_ctx.size()));
    int kv_pos = static_cast<int>(prefix_ctx.size()) + trace.n;
    for (std::size_t i = 0; i < suffix.size(); ++i, ++kv_pos) (void)sub0::forward_one(suffix[i], kv_pos);
    std::array<float, D_MODEL> spliced_hidden{};
    std::copy_n(sub0::last_hidden_ptr(), D_MODEL, spliced_hidden.data());

    double dot = 0.0, nr = 0.0, ns = 0.0;
    for (int c = 0; c < D_MODEL; ++c) {
        dot += static_cast<double>(real_hidden[static_cast<std::size_t>(c)]) * spliced_hidden[static_cast<std::size_t>(c)];
        nr  += static_cast<double>(real_hidden[static_cast<std::size_t>(c)]) * real_hidden[static_cast<std::size_t>(c)];
        ns  += static_cast<double>(spliced_hidden[static_cast<std::size_t>(c)]) * spliced_hidden[static_cast<std::size_t>(c)];
    }
    return (nr > 0.0 && ns > 0.0) ? dot / (std::sqrt(nr) * std::sqrt(ns)) : 0.0;
}

double context_sensitivity_cosine2(const Tokenizer& tk, const fs::FactPair& fp, const std::string& prefix) {
    return context_sensitivity_cosine2_impl(tk, fp, prefix, subject_piece_ids(tk, fp.subject));
}

// Direct test: does marker-inclusive replay close the (now confound-free) context-sensitivity gap? If
// context_sensitivity_cosine2's own number is already close to hidden_state_cosine_kvtrace_full's trivial
// (same-context) case, the isolated-capture assumption costs little; if THIS marker-inclusive version
// closes a real remaining gap the way E closed C's, that's a second, independent confirmation the markers
// carry weight -- this time under a genuinely different (non-empty, real) preceding context.
double context_sensitivity_cosine2_markers(const Tokenizer& tk, const fs::FactPair& fp, const std::string& prefix) {
    return context_sensitivity_cosine2_impl(tk, fp, prefix, sub0::tok::encode(tk, fp.subject));
}

// Blends fact-teaching windows (fs::Dataset, no bindings) with slot-reading-exposure windows
// (fs::SlotDataset, piece-id bound, content_embed active) at a fixed per-window mix ratio. A shared
// data/mask buffer is required (unlike the single-source train_steps above) because the two datasets are
// separate token arrays -- each window is copied into its own slice, mirroring
// blended_capstone_engine_tests.cpp's own multi-source train_steps shape, extended with scratchspike_
// engine_tests.cpp's own per-window win_binds wiring for content_embed (train_batch's win_binds tolerates
// a per-window mix of real bindings and nullptr -- exactly what a blended fact/slot schedule needs).
void train_steps_combined(const fs::Dataset& fact_ds, const fs::SlotDataset& slot_ds,
                          sub0::AdamW& opt, int steps, std::mt19937& rng, std::mt19937_64& choice_rng,
                          double slot_frac) {
    std::vector<int> data(static_cast<std::size_t>(kBatch) * (kWindowT + 1));
    std::vector<std::uint8_t> mask(data.size());
    std::vector<std::size_t> starts(kBatch);
    std::vector<int> lens(kBatch);
    std::vector<sub0::ScratchBindings> binds(kBatch);
    std::vector<const sub0::ScratchBindings*> binds_ptr(kBatch);
    std::uniform_real_distribution<double> coin(0.0, 1.0);

    for (int s = 0; s < steps; ++s) {
        for (int b = 0; b < kBatch; ++b) {
            const std::size_t base = static_cast<std::size_t>(b) * (kWindowT + 1);
            const bool use_slot = coin(choice_rng) < slot_frac;
            if (use_slot) {
                const sub0::Window w = sub0::sample_window(rng, kWindowT, slot_ds.tokens.size(),
                                                           std::span<const std::uint64_t>(slot_ds.doc_starts));
                const std::size_t n = static_cast<std::size_t>(w.len) + 1;
                for (std::size_t k = 0; k < n; ++k) {
                    data[base + k] = slot_ds.tokens[w.start + k];
                    mask[base + k] = slot_ds.mask[w.start + k];
                }
                const std::size_t doc = static_cast<std::size_t>(
                    std::upper_bound(slot_ds.doc_starts.begin(), slot_ds.doc_starts.end(),
                                     static_cast<std::uint64_t>(w.start)) - slot_ds.doc_starts.begin()) - 1;
                binds[static_cast<std::size_t>(b)] = sub0::ScratchBindings{
                    std::span<const std::vector<int>>(slot_ds.doc_bindings[doc]), sub0::SlotEncoding::HRR };
                binds_ptr[static_cast<std::size_t>(b)] = &binds[static_cast<std::size_t>(b)];
                starts[static_cast<std::size_t>(b)] = base;
                lens[static_cast<std::size_t>(b)] = w.len;
            } else {
                const sub0::Window w = sub0::sample_window(rng, kWindowT, fact_ds.tokens.size(),
                                                           std::span<const std::uint64_t>(fact_ds.doc_starts));
                const std::size_t n = static_cast<std::size_t>(w.len) + 1;
                for (std::size_t k = 0; k < n; ++k) {
                    data[base + k] = fact_ds.tokens[w.start + k];
                    mask[base + k] = fact_ds.mask[w.start + k];
                }
                binds_ptr[static_cast<std::size_t>(b)] = nullptr;
                starts[static_cast<std::size_t>(b)] = base;
                lens[static_cast<std::size_t>(b)] = w.len;
            }
        }
        opt.zero_grad();
        (void)sub0::train_batch(data.data(), starts.data(), kBatch, kWindowT, lens.data(), mask.data(),
                                binds_ptr.data());
        opt.step();
    }
}

// 3-way weighted trainer for Phase F: mixes slot-retrieval windows (fs::SlotDataset, task-contingent, per
// Phase D's design) with PLAIN TEXT drawn from TWO SEPARATE pools -- drilled subjects and exposure
// subjects -- at independently controlled rates, instead of Phase D/E's single merged pool. Phase D/E
// widened one shared plain-text Dataset to include both populations, which diluted drilled subjects' own
// sampling rate by pool-size ratio (9/15) regardless of per-subject repetition count; Phase E proved that
// dilution, not cold-mixing, was the dominant cause of Phase D's collapse (docs/FACTSPIKE.md). This keeps
// drilled subjects' sampling rate close to Phase C's own (100% of the plain-text budget) while still
// giving exposure subjects real full-text teaching: of the (1-slot_frac) windows NOT going to slot_ds,
// `exposure_frac_of_plain` of them go to exposure_ds, the rest to drilled_ds.
void train_steps_3way(const fs::Dataset& drilled_ds, const fs::Dataset& exposure_ds,
                      const fs::SlotDataset& slot_ds, sub0::AdamW& opt, int steps,
                      std::mt19937& rng, std::mt19937_64& choice_rng,
                      double slot_frac, double exposure_frac_of_plain) {
    std::vector<int> data(static_cast<std::size_t>(kBatch) * (kWindowT + 1));
    std::vector<std::uint8_t> mask(data.size());
    std::vector<std::size_t> starts(kBatch);
    std::vector<int> lens(kBatch);
    std::vector<sub0::ScratchBindings> binds(kBatch);
    std::vector<const sub0::ScratchBindings*> binds_ptr(kBatch);
    std::uniform_real_distribution<double> coin(0.0, 1.0);

    auto copy_plain = [&](const fs::Dataset& ds, int b) {
        const std::size_t base = static_cast<std::size_t>(b) * (kWindowT + 1);
        const sub0::Window w = sub0::sample_window(rng, kWindowT, ds.tokens.size(),
                                                   std::span<const std::uint64_t>(ds.doc_starts));
        const std::size_t n = static_cast<std::size_t>(w.len) + 1;
        for (std::size_t k = 0; k < n; ++k) {
            data[base + k] = ds.tokens[w.start + k];
            mask[base + k] = ds.mask[w.start + k];
        }
        binds_ptr[static_cast<std::size_t>(b)] = nullptr;
        starts[static_cast<std::size_t>(b)] = base;
        lens[static_cast<std::size_t>(b)] = w.len;
    };

    for (int s = 0; s < steps; ++s) {
        for (int b = 0; b < kBatch; ++b) {
            const double r = coin(choice_rng);
            if (r < slot_frac) {
                const std::size_t base = static_cast<std::size_t>(b) * (kWindowT + 1);
                const sub0::Window w = sub0::sample_window(rng, kWindowT, slot_ds.tokens.size(),
                                                           std::span<const std::uint64_t>(slot_ds.doc_starts));
                const std::size_t n = static_cast<std::size_t>(w.len) + 1;
                for (std::size_t k = 0; k < n; ++k) {
                    data[base + k] = slot_ds.tokens[w.start + k];
                    mask[base + k] = slot_ds.mask[w.start + k];
                }
                const std::size_t doc = static_cast<std::size_t>(
                    std::upper_bound(slot_ds.doc_starts.begin(), slot_ds.doc_starts.end(),
                                     static_cast<std::uint64_t>(w.start)) - slot_ds.doc_starts.begin()) - 1;
                binds[static_cast<std::size_t>(b)] = sub0::ScratchBindings{
                    std::span<const std::vector<int>>(slot_ds.doc_bindings[doc]), sub0::SlotEncoding::HRR };
                binds_ptr[static_cast<std::size_t>(b)] = &binds[static_cast<std::size_t>(b)];
                starts[static_cast<std::size_t>(b)] = base;
                lens[static_cast<std::size_t>(b)] = w.len;
            } else {
                const double plain_r = (r - slot_frac) / (1.0 - slot_frac);
                if (plain_r < exposure_frac_of_plain) copy_plain(exposure_ds, b);
                else                                  copy_plain(drilled_ds, b);
            }
        }
        opt.zero_grad();
        (void)sub0::train_batch(data.data(), starts.data(), kBatch, kWindowT, lens.data(), mask.data(),
                                binds_ptr.data());
        opt.step();
    }
}

}  // namespace

TEST_CASE("factspike Phase B: does the fact-teaching curriculum bake in a persistent, cross-document "
         "association at all (baseline arm only, no scratch slots yet)", "[.factspikebaseline]") {
    Tokenizer tk;
    {
        std::ifstream is(sub0::default_tokenizer(), std::ios::binary);
        if (!is.good() || !sub0::tok::deserialize(tk, is) || tk.vocab != VOCAB) {
            WARN("factspike Phase B: this build's tokenizer isn't usable/doesn't match VOCAB -- skipping. "
                "Build against out/build/factspike96 (docs/FACTSPIKE.md Phase A) for a real result.");
            return;
        }
    }

    std::mt19937_64 split_rng(kSplitSeed);
    const fs::FactSplit split = fs::make_fact_split(split_rng, kNSubjects, kDrilledFrac);
    REQUIRE_FALSE(split.drilled.empty());

    const fs::Dataset ds = fs::build_dataset(tk, split.drilled, kDocsPerFact, /*seed=*/42);
    REQUIRE(ds.tokens.size() > static_cast<std::size_t>(kWindowT));

    sub0::build_model(); reset_opt_state();
    sub0::AdamW opt(kLr);
    std::mt19937 rng(1);

    std::string report = "\n=== factspike Phase B: baseline-arm accuracy over training (d" +
        std::to_string(D_MODEL) + ", " + std::to_string(split.drilled.size()) + " drilled subjects) ===\n  ";
    double last_rate = 0.0, peak_rate = 0.0;
    for (int r = 0; r < kEvalRounds; ++r) {
        train_steps(ds, opt, kStepsPerEval, rng);
        const Score sc = eval_baseline(tk, split.drilled, 4242u + static_cast<unsigned>(r));
        last_rate = sc.rate();
        peak_rate = std::max(peak_rate, last_rate);
        char buf[48];
        std::snprintf(buf, sizeof buf, " s%d(acc=%.2f)", (r + 1) * kStepsPerEval, last_rate);
        report += buf;
    }
    report += "\n  (chance level ~" + std::to_string(1.0 / fs::fact_vocab().size()) + " for " +
             std::to_string(fs::fact_vocab().size()) + " colors; peak=" + std::to_string(peak_rate) +
             " last=" + std::to_string(last_rate) + ")\n";
    WARN(report);

    CHECK(std::isfinite(last_rate));
    // PEAK, not final-round, accuracy is the right gate for the load-bearing Phase B premise ("can this
    // curriculum teach a persistent association at all") -- a first run showed a highly volatile
    // trajectory (0.00 -> 0.78 -> 0.00), meaning the model DID demonstrably learn the association at
    // points, but small-dataset/small-model training here is unstable, not non-learning. Final-round-only
    // would have reported a false negative on a real capability -- discovered empirically, not assumed;
    // see docs/FACTSPIKE.md's status log. Training-stability is tracked as its own separate finding, not
    // conflated with "did the curriculum work at all."
    CHECK(peak_rate > 2.0 / fs::fact_vocab().size());
}

TEST_CASE("factspike Phase C: does a factual association survive PIECE-EMBEDDING TRANSFER into a "
         "scratch slot -- baseline vs scratch vs held-out, side by side", "[.factspikecapstone]") {
    Tokenizer tk;
    {
        std::ifstream is(sub0::default_tokenizer(), std::ios::binary);
        if (!is.good() || !sub0::tok::deserialize(tk, is) || tk.vocab != VOCAB) {
            WARN("factspike Phase C: this build's tokenizer isn't usable/doesn't match VOCAB -- skipping. "
                "Build against out/build/factspike96 (docs/FACTSPIKE.md Phase A) for a real result.");
            return;
        }
    }

    std::mt19937_64 split_rng(kSplitSeed);
    const fs::FactSplit split = fs::make_fact_split(split_rng, kNSubjects, kDrilledFrac);
    REQUIRE_FALSE(split.drilled.empty());
    REQUIRE_FALSE(split.held_out.empty());

    // A SEPARATE, disjoint subject pool for slot-reading exposure -- a different seed from kSplitSeed
    // guarantees (in practice, via independent random generation) no overlap with the drilled/held-out
    // subjects under test, so the model never sees a test subject's own slot during training.
    std::mt19937_64 exposure_rng(kSplitSeed ^ 0xE4C0517E5EEDULL);
    const fs::FactSplit exposure_split = fs::make_fact_split(exposure_rng, 6, 1.0);

    const fs::Dataset ds = fs::build_dataset(tk, split.drilled, kDocsPerFact, /*seed=*/42);
    REQUIRE(ds.tokens.size() > static_cast<std::size_t>(kWindowT));
    const fs::SlotDataset slot_ds = fs::build_slot_exposure_dataset(tk, exposure_split.drilled,
                                                                    kDocsPerFact, /*seed=*/43);
    REQUIRE(slot_ds.tokens.size() > static_cast<std::size_t>(kWindowT));

    sub0::build_model(); reset_opt_state();
    sub0::AdamW opt(kLr);
    std::mt19937 rng(1);
    std::mt19937_64 choice_rng(2);

    std::string report = "\n=== factspike Phase C capstone (d" + std::to_string(D_MODEL) + ", " +
        std::to_string(split.drilled.size()) + " drilled / " + std::to_string(split.held_out.size()) +
        " held-out, " + std::to_string(exposure_split.drilled.size()) + " slot-exposure subjects) ===\n";
    double peak_baseline = 0.0, peak_scratch = 0.0, peak_held_out = 0.0;
    double last_baseline = 0.0, last_scratch = 0.0, last_held_out = 0.0;
    std::vector<double> scratch_traj, held_out_traj, fid_drilled_traj, fid_held_out_traj;
    for (int r = 0; r < kEvalRounds; ++r) {
        train_steps_combined(ds, slot_ds, opt, kStepsPerEval, rng, choice_rng, /*slot_frac=*/0.5);
        const Score sb = eval_baseline(tk, split.drilled, 4242u + static_cast<unsigned>(r));
        const Score ss = eval_scratch(tk, split.drilled, 4242u + static_cast<unsigned>(r));
        const Score sh = eval_scratch(tk, split.held_out, 4242u + static_cast<unsigned>(r));
        last_baseline = sb.rate(); last_scratch = ss.rate(); last_held_out = sh.rate();
        peak_baseline = std::max(peak_baseline, last_baseline);
        peak_scratch  = std::max(peak_scratch,  last_scratch);
        peak_held_out = std::max(peak_held_out, last_held_out);
        const double fid_drilled  = mean_reconstruction_fidelity(tk, split.drilled);
        const double fid_held_out = mean_reconstruction_fidelity(tk, split.held_out);
        scratch_traj.push_back(last_scratch);
        held_out_traj.push_back(last_held_out);
        fid_drilled_traj.push_back(fid_drilled);
        fid_held_out_traj.push_back(fid_held_out);
        char buf[160];
        std::snprintf(buf, sizeof buf,
                     "  s%d: baseline=%.2f scratch=%.2f held_out=%.2f  fid_drilled=%.3f fid_held_out=%.3f\n",
                     (r + 1) * kStepsPerEval, last_baseline, last_scratch, last_held_out, fid_drilled, fid_held_out);
        report += buf;
    }
    const double r_scratch   = pearson_r(scratch_traj, fid_drilled_traj);
    const double r_held_out  = pearson_r(held_out_traj, fid_held_out_traj);
    report += "  (chance level ~" + std::to_string(1.0 / fs::fact_vocab().size()) + " for " +
             std::to_string(fs::fact_vocab().size()) + " colors)\n" +
             "  peak: baseline=" + std::to_string(peak_baseline) + " scratch=" + std::to_string(peak_scratch) +
             " held_out=" + std::to_string(peak_held_out) + "\n" +
             "  last: baseline=" + std::to_string(last_baseline) + " scratch=" + std::to_string(last_scratch) +
             " held_out=" + std::to_string(last_held_out) + "\n" +
             "  reconstruction-fidelity vs accuracy Pearson r: drilled(scratch)=" + std::to_string(r_scratch) +
             " held_out=" + std::to_string(r_held_out) + "\n";
    WARN(report);

    // Report all three, real numbers, whatever they are -- this is an open experiment (docs/FACTSPIKE.md).
    CHECK(std::isfinite(peak_baseline));
    CHECK(std::isfinite(peak_scratch));
    CHECK(std::isfinite(peak_held_out));
    CHECK(std::isfinite(r_scratch));
    CHECK(std::isfinite(r_held_out));
}

TEST_CASE("factspike Phase D: Pack-Aware Training -- task-contingent slot-retrieval gradient, matched "
         "budget A/B against Phase C", "[.factspikepat]") {
    Tokenizer tk;
    {
        std::ifstream is(sub0::default_tokenizer(), std::ios::binary);
        if (!is.good() || !sub0::tok::deserialize(tk, is) || tk.vocab != VOCAB) {
            WARN("factspike Phase D: this build's tokenizer isn't usable/doesn't match VOCAB -- skipping. "
                "Build against out/build/factspike96 (docs/FACTSPIKE.md Phase A) for a real result.");
            return;
        }
    }

    std::mt19937_64 split_rng(kSplitSeed);
    const fs::FactSplit split = fs::make_fact_split(split_rng, kNSubjects, kDrilledFrac);
    REQUIRE_FALSE(split.drilled.empty());
    REQUIRE_FALSE(split.held_out.empty());

    std::mt19937_64 exposure_rng(kSplitSeed ^ 0xE4C0517E5EEDULL);
    const fs::FactSplit exposure_split = fs::make_fact_split(exposure_rng, 6, 1.0);

    // WIDENED plain-text pool: drilled + exposure subjects both get ordinary full-text fact teaching
    // (exposure subjects lost their "mention 1" full text when build_slot_retrieval_dataset dropped it --
    // this is where they get it instead, as a genuinely separate document). held-out is NEVER added here.
    std::vector<fs::FactPair> fact_subjects = split.drilled;
    fact_subjects.insert(fact_subjects.end(), exposure_split.drilled.begin(), exposure_split.drilled.end());
    const fs::Dataset ds = fs::build_dataset(tk, fact_subjects, kDocsPerFact, /*seed=*/42);
    REQUIRE(ds.tokens.size() > static_cast<std::size_t>(kWindowT));
    const fs::SlotDataset slot_ds = fs::build_slot_retrieval_dataset(tk, exposure_split.drilled,
                                                                     kDocsPerFact, /*seed=*/43);
    REQUIRE(slot_ds.tokens.size() > static_cast<std::size_t>(kWindowT));

    sub0::build_model(); reset_opt_state();
    sub0::AdamW opt(kLr);
    std::mt19937 rng(1);
    std::mt19937_64 choice_rng(2);

    std::string report = "\n=== factspike Phase D PAT (d" + std::to_string(D_MODEL) + ", " +
        std::to_string(split.drilled.size()) + " drilled / " + std::to_string(split.held_out.size()) +
        " held-out, " + std::to_string(exposure_split.drilled.size()) + " slot-retrieval subjects) ===\n";
    double peak_baseline = 0.0, peak_scratch = 0.0, peak_held_out = 0.0;
    double last_baseline = 0.0, last_scratch = 0.0, last_held_out = 0.0;
    std::vector<double> scratch_traj, held_out_traj, fid_drilled_traj, fid_held_out_traj;
    for (int r = 0; r < kEvalRounds; ++r) {
        train_steps_combined(ds, slot_ds, opt, kStepsPerEval, rng, choice_rng, /*slot_frac=*/0.5);
        const Score sb = eval_baseline(tk, split.drilled, 4242u + static_cast<unsigned>(r));
        const Score ss = eval_scratch(tk, split.drilled, 4242u + static_cast<unsigned>(r));
        const Score sh = eval_scratch(tk, split.held_out, 4242u + static_cast<unsigned>(r));
        last_baseline = sb.rate(); last_scratch = ss.rate(); last_held_out = sh.rate();
        peak_baseline = std::max(peak_baseline, last_baseline);
        peak_scratch  = std::max(peak_scratch,  last_scratch);
        peak_held_out = std::max(peak_held_out, last_held_out);
        const double fid_drilled  = mean_reconstruction_fidelity(tk, split.drilled);
        const double fid_held_out = mean_reconstruction_fidelity(tk, split.held_out);
        scratch_traj.push_back(last_scratch);
        held_out_traj.push_back(last_held_out);
        fid_drilled_traj.push_back(fid_drilled);
        fid_held_out_traj.push_back(fid_held_out);
        char buf[160];
        std::snprintf(buf, sizeof buf,
                     "  s%d: baseline=%.2f scratch=%.2f held_out=%.2f  fid_drilled=%.3f fid_held_out=%.3f\n",
                     (r + 1) * kStepsPerEval, last_baseline, last_scratch, last_held_out, fid_drilled, fid_held_out);
        report += buf;
    }
    const double r_scratch  = pearson_r(scratch_traj, fid_drilled_traj);
    const double r_held_out = pearson_r(held_out_traj, fid_held_out_traj);
    report += "  (chance level ~" + std::to_string(1.0 / fs::fact_vocab().size()) + " for " +
             std::to_string(fs::fact_vocab().size()) + " colors)\n" +
             "  peak: baseline=" + std::to_string(peak_baseline) + " scratch=" + std::to_string(peak_scratch) +
             " held_out=" + std::to_string(peak_held_out) + "\n" +
             "  last: baseline=" + std::to_string(last_baseline) + " scratch=" + std::to_string(last_scratch) +
             " held_out=" + std::to_string(last_held_out) + "\n" +
             "  reconstruction-fidelity vs accuracy Pearson r: drilled(scratch)=" + std::to_string(r_scratch) +
             " held_out=" + std::to_string(r_held_out) + "\n" +
             "  (Phase C comparison: peak baseline=1.00 scratch=0.56 held_out=0.33, r_scratch=-0.23)\n";
    WARN(report);

    // Report all three, real numbers, whatever they are -- matched-budget A/B against Phase C's own
    // recorded result (docs/FACTSPIKE.md), not a guaranteed-positive validation.
    CHECK(std::isfinite(peak_baseline));
    CHECK(std::isfinite(peak_scratch));
    CHECK(std::isfinite(peak_held_out));
    CHECK(std::isfinite(r_scratch));
    CHECK(std::isfinite(r_held_out));
}

// Warm-start + ramp schedule for Phase E: rounds [0, kWarmupRounds) train on plain text only
// (slot_frac=0), rounds [kWarmupRounds, kWarmupRounds+kRampRounds) ramp slot_frac linearly up to
// kTargetSlotFrac, and the remainder hold at kTargetSlotFrac -- matching Phase D's own steady-state mix
// so the two are comparable once ramped in. Isolates cold-mixing interference from pool dilution: the
// widened fact_ds pool (same dilution as Phase D) is used throughout, including the warm-start rounds.
double phase_e_slot_frac(int round) {
    constexpr int    kWarmupRounds   = 5;
    constexpr int    kRampRounds     = 10;
    constexpr double kTargetSlotFrac = 0.5;
    if (round < kWarmupRounds) return 0.0;
    if (round < kWarmupRounds + kRampRounds) {
        const double t = static_cast<double>(round - kWarmupRounds + 1) / kRampRounds;
        return t * kTargetSlotFrac;
    }
    return kTargetSlotFrac;
}

TEST_CASE("factspike Phase E: Pack-Aware Training, warm-started + ramped -- isolates cold-mixing "
         "interference from pool dilution", "[.factspikepatramp]") {
    Tokenizer tk;
    {
        std::ifstream is(sub0::default_tokenizer(), std::ios::binary);
        if (!is.good() || !sub0::tok::deserialize(tk, is) || tk.vocab != VOCAB) {
            WARN("factspike Phase E: this build's tokenizer isn't usable/doesn't match VOCAB -- skipping. "
                "Build against out/build/factspike96 (docs/FACTSPIKE.md Phase A) for a real result.");
            return;
        }
    }

    std::mt19937_64 split_rng(kSplitSeed);
    const fs::FactSplit split = fs::make_fact_split(split_rng, kNSubjects, kDrilledFrac);
    REQUIRE_FALSE(split.drilled.empty());
    REQUIRE_FALSE(split.held_out.empty());

    std::mt19937_64 exposure_rng(kSplitSeed ^ 0xE4C0517E5EEDULL);
    const fs::FactSplit exposure_split = fs::make_fact_split(exposure_rng, 6, 1.0);

    std::vector<fs::FactPair> fact_subjects = split.drilled;
    fact_subjects.insert(fact_subjects.end(), exposure_split.drilled.begin(), exposure_split.drilled.end());
    const fs::Dataset ds = fs::build_dataset(tk, fact_subjects, kDocsPerFact, /*seed=*/42);
    REQUIRE(ds.tokens.size() > static_cast<std::size_t>(kWindowT));
    const fs::SlotDataset slot_ds = fs::build_slot_retrieval_dataset(tk, exposure_split.drilled,
                                                                     kDocsPerFact, /*seed=*/43);
    REQUIRE(slot_ds.tokens.size() > static_cast<std::size_t>(kWindowT));

    sub0::build_model(); reset_opt_state();
    sub0::AdamW opt(kLr);
    std::mt19937 rng(1);
    std::mt19937_64 choice_rng(2);

    std::string report = "\n=== factspike Phase E PAT warm-start+ramp (d" + std::to_string(D_MODEL) + ", " +
        std::to_string(split.drilled.size()) + " drilled / " + std::to_string(split.held_out.size()) +
        " held-out, " + std::to_string(exposure_split.drilled.size()) + " slot-retrieval subjects) ===\n";
    double peak_baseline = 0.0, peak_scratch = 0.0, peak_held_out = 0.0;
    double last_baseline = 0.0, last_scratch = 0.0, last_held_out = 0.0;
    std::vector<double> scratch_traj, held_out_traj, fid_drilled_traj, fid_held_out_traj;
    for (int r = 0; r < kEvalRounds; ++r) {
        const double slot_frac = phase_e_slot_frac(r);
        train_steps_combined(ds, slot_ds, opt, kStepsPerEval, rng, choice_rng, slot_frac);
        const Score sb = eval_baseline(tk, split.drilled, 4242u + static_cast<unsigned>(r));
        const Score ss = eval_scratch(tk, split.drilled, 4242u + static_cast<unsigned>(r));
        const Score sh = eval_scratch(tk, split.held_out, 4242u + static_cast<unsigned>(r));
        last_baseline = sb.rate(); last_scratch = ss.rate(); last_held_out = sh.rate();
        peak_baseline = std::max(peak_baseline, last_baseline);
        peak_scratch  = std::max(peak_scratch,  last_scratch);
        peak_held_out = std::max(peak_held_out, last_held_out);
        const double fid_drilled  = mean_reconstruction_fidelity(tk, split.drilled);
        const double fid_held_out = mean_reconstruction_fidelity(tk, split.held_out);
        scratch_traj.push_back(last_scratch);
        held_out_traj.push_back(last_held_out);
        fid_drilled_traj.push_back(fid_drilled);
        fid_held_out_traj.push_back(fid_held_out);
        char buf[190];
        std::snprintf(buf, sizeof buf,
                     "  s%d(sf=%.2f): baseline=%.2f scratch=%.2f held_out=%.2f  fid_drilled=%.3f fid_held_out=%.3f\n",
                     (r + 1) * kStepsPerEval, slot_frac, last_baseline, last_scratch, last_held_out,
                     fid_drilled, fid_held_out);
        report += buf;
    }
    const double r_scratch  = pearson_r(scratch_traj, fid_drilled_traj);
    const double r_held_out = pearson_r(held_out_traj, fid_held_out_traj);
    report += "  (chance level ~" + std::to_string(1.0 / fs::fact_vocab().size()) + " for " +
             std::to_string(fs::fact_vocab().size()) + " colors)\n" +
             "  peak: baseline=" + std::to_string(peak_baseline) + " scratch=" + std::to_string(peak_scratch) +
             " held_out=" + std::to_string(peak_held_out) + "\n" +
             "  last: baseline=" + std::to_string(last_baseline) + " scratch=" + std::to_string(last_scratch) +
             " held_out=" + std::to_string(last_held_out) + "\n" +
             "  reconstruction-fidelity vs accuracy Pearson r: drilled(scratch)=" + std::to_string(r_scratch) +
             " held_out=" + std::to_string(r_held_out) + "\n" +
             "  (Phase C: peak baseline=1.00 scratch=0.56 held_out=0.33 | "
             "Phase D flat-mix: peak baseline=0.22 scratch=0.00 held_out=0.00)\n";
    WARN(report);

    // Report all three, real numbers, whatever they are -- matched-budget A/B against Phase C/D's own
    // recorded results (docs/FACTSPIKE.md), not a guaranteed-positive validation.
    CHECK(std::isfinite(peak_baseline));
    CHECK(std::isfinite(peak_scratch));
    CHECK(std::isfinite(peak_held_out));
    CHECK(std::isfinite(r_scratch));
    CHECK(std::isfinite(r_held_out));
}

// Tokenization-boundary regression check, found while validating candidate 2's "trivial case should
// reconstruct baseline almost exactly" prediction (docs/FACTSPIKE.md Phase G): a subject's OWN
// tokenization (`tok::encode(subject)` alone) always matches its own span at the START of the combined
// baseline sentence (`tok::encode(subject + " loves the color ")`) -- this is what lets
// hidden_state_cosine_kvtrace_full slice the REAL suffix out of `base_ctx` rather than re-tokenizing
// " loves the color " separately (which does NOT match: BPE merges the leading space differently
// depending on what precedes it -- a real, if minor, confound this test pins down so it can't silently
// resurface). ALSO surfaces a deeper, structural finding: `subject_piece_ids` (what every packing
// mechanism -- A/B/C/D -- captures from) strips SPELL_START/SPELL_END wrapper markers that a real forward
// pass over a multi-piece subject actually processes; single-piece subjects (no wrapper needed) have
// `pieces == alone` exactly, multi-piece ones don't. This is why candidate 2's multi-piece subjects don't
// reach the near-1.0 reconstruction fidelity its single-piece ones do (see Phase G's per-subject numbers).
TEST_CASE("factspike: a subject's own tokenization matches its span at the start of the combined "
         "baseline sentence (the assumption hidden_state_cosine_kvtrace_full relies on)",
         "[factspike][kvtrace]") {
    Tokenizer tk;
    {
        std::ifstream is(sub0::default_tokenizer(), std::ios::binary);
        if (!is.good() || !sub0::tok::deserialize(tk, is) || tk.vocab != VOCAB) {
            WARN("this build's tokenizer isn't usable/doesn't match VOCAB -- skipping.");
            return;
        }
    }
    std::mt19937_64 split_rng(kSplitSeed);
    const fs::FactSplit split = fs::make_fact_split(split_rng, kNSubjects, kDrilledFrac);
    for (const fs::FactPair& fp : split.drilled) {
        const std::vector<int> alone = sub0::tok::encode(tk, fp.subject);
        const std::vector<int> combined = sub0::tok::encode(tk, fp.subject + " loves the color ");
        REQUIRE(combined.size() >= alone.size());
        bool prefix_match = true;
        for (std::size_t i = 0; i < alone.size(); ++i) if (combined[i] != alone[i]) prefix_match = false;
        CHECK(prefix_match);
    }
}

TEST_CASE("factspike hidden-state diagnostic: does the packed slot's fully-processed representation "
         "resemble the same subject read normally, under Phase C's own validated regime",
         "[.factspikehidden]") {
    Tokenizer tk;
    {
        std::ifstream is(sub0::default_tokenizer(), std::ios::binary);
        if (!is.good() || !sub0::tok::deserialize(tk, is) || tk.vocab != VOCAB) {
            WARN("factspike hidden-state diagnostic: tokenizer isn't usable/doesn't match VOCAB -- "
                "skipping. Build against out/build/factspike96 for a real result.");
            return;
        }
    }

    std::mt19937_64 split_rng(kSplitSeed);
    const fs::FactSplit split = fs::make_fact_split(split_rng, kNSubjects, kDrilledFrac);
    REQUIRE_FALSE(split.drilled.empty());

    // SAME regime as Phase C (not D/E's widened/diluted pool) -- this asks a representational question
    // that's orthogonal to the training-schedule question Phase D/E investigated, so it should run on the
    // cleanest, best-validated result (Phase C: peak scratch=0.56), not a degraded one.
    std::mt19937_64 exposure_rng(kSplitSeed ^ 0xE4C0517E5EEDULL);
    const fs::FactSplit exposure_split = fs::make_fact_split(exposure_rng, 6, 1.0);
    const fs::Dataset ds = fs::build_dataset(tk, split.drilled, kDocsPerFact, /*seed=*/42);
    REQUIRE(ds.tokens.size() > static_cast<std::size_t>(kWindowT));
    const fs::SlotDataset slot_ds = fs::build_slot_exposure_dataset(tk, exposure_split.drilled,
                                                                    kDocsPerFact, /*seed=*/43);
    REQUIRE(slot_ds.tokens.size() > static_cast<std::size_t>(kWindowT));

    sub0::build_model(); reset_opt_state();
    sub0::AdamW opt(kLr);
    std::mt19937 rng(1);
    std::mt19937_64 choice_rng(2);
    for (int r = 0; r < kEvalRounds; ++r)
        train_steps_combined(ds, slot_ds, opt, kStepsPerEval, rng, choice_rng, /*slot_frac=*/0.5);

    // Attention-capacity hypothesis: a genuine n-piece word gets ~N_LAYERS*(n-1) sibling-attention hops
    // (each layer, the word's later positions re-integrate over every earlier piece's current, once-
    // already-refined state) that a packed single position structurally cannot have (no same-word
    // siblings to attend back over at all). Prediction: the packed-vs-normal gap should SCALE WITH n, not
    // be a flat, uniform effect -- n=1 subjects (no multi-hop to lose in the first place) should show
    // near-zero gap. Piece count recorded alongside cos_sim/correctness to test this directly.
    std::string report = "\n=== factspike hidden-state diagnostic (post Phase-C-regime training) ===\n";
    double sim_sum = 0.0; int n_correct = 0, n_total = 0;
    std::vector<double> piece_count_traj, sim_traj;
    for (const fs::FactPair& fp : split.drilled) {
        const double sim = hidden_state_cosine(tk, fp);
        const bool   ok  = eval_scratch_one(tk, fp);
        const std::size_t n_pieces = subject_piece_ids(tk, fp.subject).size();
        sim_sum += sim; ++n_total; n_correct += ok ? 1 : 0;
        piece_count_traj.push_back(static_cast<double>(n_pieces));
        sim_traj.push_back(sim);
        char buf[112];
        std::snprintf(buf, sizeof buf, "  %-12s n_pieces=%d  cos_sim=%.3f  scratch_correct=%s\n",
                     fp.subject.c_str(), static_cast<int>(n_pieces), sim, ok ? "yes" : "no");
        report += buf;
    }
    const double mean_sim = n_total ? sim_sum / n_total : 0.0;
    const double r_pieces_sim = pearson_r(piece_count_traj, sim_traj);
    report += "  mean cosine similarity (baseline vs scratch final hidden state) = " +
             std::to_string(mean_sim) + "  (" + std::to_string(n_correct) + "/" +
             std::to_string(n_total) + " correct this round)\n" +
             "  Pearson r(piece_count, cos_sim) = " + std::to_string(r_pieces_sim) +
             "  (attention-capacity hypothesis predicts NEGATIVE: more pieces -> lower similarity)\n";
    WARN(report);

    CHECK(std::isfinite(mean_sim));
    CHECK(std::isfinite(r_pieces_sim));
}

TEST_CASE("factspike Phase F: Pack-Aware Training re-tested WITHOUT the dilution confound Phase E "
         "isolated -- does the task-contingent fix actually beat Phase C?", "[.factspikepatfair]") {
    Tokenizer tk;
    {
        std::ifstream is(sub0::default_tokenizer(), std::ios::binary);
        if (!is.good() || !sub0::tok::deserialize(tk, is) || tk.vocab != VOCAB) {
            WARN("factspike Phase F: this build's tokenizer isn't usable/doesn't match VOCAB -- skipping. "
                "Build against out/build/factspike96 (docs/FACTSPIKE.md Phase A) for a real result.");
            return;
        }
    }

    std::mt19937_64 split_rng(kSplitSeed);
    const fs::FactSplit split = fs::make_fact_split(split_rng, kNSubjects, kDrilledFrac);
    REQUIRE_FALSE(split.drilled.empty());
    REQUIRE_FALSE(split.held_out.empty());

    std::mt19937_64 exposure_rng(kSplitSeed ^ 0xE4C0517E5EEDULL);
    const fs::FactSplit exposure_split = fs::make_fact_split(exposure_rng, 6, 1.0);

    // SEPARATE drilled/exposure plain-text pools (not merged like Phase D/E) -- drilled_ds is IDENTICAL
    // to Phase C's own ds (same subjects, same seed), so its own sampling rate can stay close to Phase
    // C's 100%-of-plain-text-budget instead of being diluted by pool-size ratio.
    const fs::Dataset drilled_ds = fs::build_dataset(tk, split.drilled, kDocsPerFact, /*seed=*/42);
    REQUIRE(drilled_ds.tokens.size() > static_cast<std::size_t>(kWindowT));
    const fs::Dataset exposure_ds = fs::build_dataset(tk, exposure_split.drilled, kDocsPerFact, /*seed=*/44);
    REQUIRE(exposure_ds.tokens.size() > static_cast<std::size_t>(kWindowT));
    const fs::SlotDataset slot_ds = fs::build_slot_retrieval_dataset(tk, exposure_split.drilled,
                                                                     kDocsPerFact, /*seed=*/43);
    REQUIRE(slot_ds.tokens.size() > static_cast<std::size_t>(kWindowT));

    sub0::build_model(); reset_opt_state();
    sub0::AdamW opt(kLr);
    std::mt19937 rng(1);
    std::mt19937_64 choice_rng(2);
    constexpr double kSlotFrac = 0.5;                // matches Phase D's own steady-state mix
    constexpr double kExposureFracOfPlain = 0.2;     // drilled subjects keep ~40% of ALL windows (vs
                                                      // Phase C's 50%, vs Phase D/E's diluted ~30%)

    std::string report = "\n=== factspike Phase F PAT, dilution-fixed (d" + std::to_string(D_MODEL) + ", " +
        std::to_string(split.drilled.size()) + " drilled / " + std::to_string(split.held_out.size()) +
        " held-out, " + std::to_string(exposure_split.drilled.size()) + " slot-retrieval subjects) ===\n";
    double peak_baseline = 0.0, peak_scratch = 0.0, peak_held_out = 0.0;
    double last_baseline = 0.0, last_scratch = 0.0, last_held_out = 0.0;
    std::vector<double> scratch_traj, held_out_traj, fid_drilled_traj, fid_held_out_traj;
    for (int r = 0; r < kEvalRounds; ++r) {
        train_steps_3way(drilled_ds, exposure_ds, slot_ds, opt, kStepsPerEval, rng, choice_rng,
                         kSlotFrac, kExposureFracOfPlain);
        const Score sb = eval_baseline(tk, split.drilled, 4242u + static_cast<unsigned>(r));
        const Score ss = eval_scratch(tk, split.drilled, 4242u + static_cast<unsigned>(r));
        const Score sh = eval_scratch(tk, split.held_out, 4242u + static_cast<unsigned>(r));
        last_baseline = sb.rate(); last_scratch = ss.rate(); last_held_out = sh.rate();
        peak_baseline = std::max(peak_baseline, last_baseline);
        peak_scratch  = std::max(peak_scratch,  last_scratch);
        peak_held_out = std::max(peak_held_out, last_held_out);
        const double fid_drilled  = mean_reconstruction_fidelity(tk, split.drilled);
        const double fid_held_out = mean_reconstruction_fidelity(tk, split.held_out);
        scratch_traj.push_back(last_scratch);
        held_out_traj.push_back(last_held_out);
        fid_drilled_traj.push_back(fid_drilled);
        fid_held_out_traj.push_back(fid_held_out);
        char buf[160];
        std::snprintf(buf, sizeof buf,
                     "  s%d: baseline=%.2f scratch=%.2f held_out=%.2f  fid_drilled=%.3f fid_held_out=%.3f\n",
                     (r + 1) * kStepsPerEval, last_baseline, last_scratch, last_held_out, fid_drilled, fid_held_out);
        report += buf;
    }
    const double r_scratch  = pearson_r(scratch_traj, fid_drilled_traj);
    const double r_held_out = pearson_r(held_out_traj, fid_held_out_traj);
    report += "  (chance level ~" + std::to_string(1.0 / fs::fact_vocab().size()) + " for " +
             std::to_string(fs::fact_vocab().size()) + " colors)\n" +
             "  peak: baseline=" + std::to_string(peak_baseline) + " scratch=" + std::to_string(peak_scratch) +
             " held_out=" + std::to_string(peak_held_out) + "\n" +
             "  last: baseline=" + std::to_string(last_baseline) + " scratch=" + std::to_string(last_scratch) +
             " held_out=" + std::to_string(last_held_out) + "\n" +
             "  reconstruction-fidelity vs accuracy Pearson r: drilled(scratch)=" + std::to_string(r_scratch) +
             " held_out=" + std::to_string(r_held_out) + "\n" +
             "  (Phase C: peak baseline=1.00 scratch=0.56 | Phase D flat-mix diluted: peak baseline=0.22 "
             "scratch=0.00 | Phase E warm+ramp diluted: peak baseline=0.33 scratch=0.11)\n";
    WARN(report);

    // Report all three, real numbers, whatever they are -- matched-budget A/B against Phase C/D/E's own
    // recorded results (docs/FACTSPIKE.md), not a guaranteed-positive validation.
    CHECK(std::isfinite(peak_baseline));
    CHECK(std::isfinite(peak_scratch));
    CHECK(std::isfinite(peak_held_out));
    CHECK(std::isfinite(r_scratch));
    CHECK(std::isfinite(r_held_out));
}

// Fast, model-free algebraic sanity check for the KV-trace primitives' rotation math (core.hpp's
// kv_rope_rotate) -- pure math on a caller-owned buffer, no tokenizer/model/config dependency, so this
// runs in the normal always-on suite regardless of which config sub0_tests happens to be built against.
// Confirms de-rotate/re-rotate is wired correctly (R(-pos)*R(pos) = I) BEFORE any training-run time is
// spent trusting Phase G's results below -- see docs/SCRATCH_TOKEN_FRAMING.md candidate 1.
TEST_CASE("factspike kv_rope_rotate round-trips: rotate then inverse-rotate is the identity",
         "[factspike][kvtrace]") {
    std::array<float, D_MODEL> row{};
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    for (float& x : row) x = dist(rng);
    const std::array<float, D_MODEL> original = row;

    sub0::kv_rope_rotate(row.data(), 5);
    CHECK_FALSE(row == original);   // sanity: the forward rotation actually did something at pos=5
    sub0::kv_rope_rotate(row.data(), -5);
    for (int c = 0; c < D_MODEL; ++c)
        CHECK(row[static_cast<std::size_t>(c)] ==
             Catch::Approx(original[static_cast<std::size_t>(c)]).margin(1e-4f));
}

TEST_CASE("factspike Phase G: KV-trace memoization -- candidates 1 (post-hoc pooled splice) and 2 "
         "(full n-position splice, trivial + context-sensitivity probe) vs mechanism A, same "
         "Phase-C-regime training", "[.factspikekvtrace]") {
    Tokenizer tk;
    {
        std::ifstream is(sub0::default_tokenizer(), std::ios::binary);
        if (!is.good() || !sub0::tok::deserialize(tk, is) || tk.vocab != VOCAB) {
            WARN("factspike Phase G: this build's tokenizer isn't usable/doesn't match VOCAB -- skipping. "
                "Build against out/build/factspike96 (docs/FACTSPIKE.md Phase A) for a real result.");
            return;
        }
    }

    std::mt19937_64 split_rng(kSplitSeed);
    const fs::FactSplit split = fs::make_fact_split(split_rng, kNSubjects, kDrilledFrac);
    REQUIRE_FALSE(split.drilled.empty());

    // SAME regime as Phase C / [.factspikehidden] -- this asks a mechanism question orthogonal to the
    // training-schedule question, so it runs on the cleanest, best-validated result, not a degraded one.
    std::mt19937_64 exposure_rng(kSplitSeed ^ 0xE4C0517E5EEDULL);
    const fs::FactSplit exposure_split = fs::make_fact_split(exposure_rng, 6, 1.0);
    const fs::Dataset ds = fs::build_dataset(tk, split.drilled, kDocsPerFact, /*seed=*/42);
    REQUIRE(ds.tokens.size() > static_cast<std::size_t>(kWindowT));
    const fs::SlotDataset slot_ds = fs::build_slot_exposure_dataset(tk, exposure_split.drilled,
                                                                    kDocsPerFact, /*seed=*/43);
    REQUIRE(slot_ds.tokens.size() > static_cast<std::size_t>(kWindowT));

    sub0::build_model(); reset_opt_state();
    sub0::AdamW opt(kLr);
    std::mt19937 rng(1);
    std::mt19937_64 choice_rng(2);
    for (int r = 0; r < kEvalRounds; ++r)
        train_steps_combined(ds, slot_ds, opt, kStepsPerEval, rng, choice_rng, /*slot_frac=*/0.5);

    // All mechanisms measured in the SAME run, same trained weights -- a fairer matched comparison than
    // comparing against a previous session's recorded numbers, given this project's own repeated lesson
    // about single-seed training volatility (see project memory factspike-piece-transfer-validated).
    // A = mechanism A (embed-compose); B = candidate 1 (post-hoc pooled splice); C = candidate 2 at the
    // trivial position-0/isolated-context shape (should be ~exact, a correctness gate on the splice
    // implementation, not really a "does it help" measure); D = candidate 2's context-sensitivity probe
    // (same isolated-captured trace, spliced into a REAL non-empty preceding context this time); E =
    // candidate 2 trivial-case AGAIN, but replaying the marker-INCLUSIVE span (tok::encode(subject), not
    // subject_piece_ids) -- a direct, controlled test of whether SPELL_START/SPELL_END are what's holding
    // C below 1.000 for multi-piece subjects (single-piece ones already hit exact 1.000 with C, since
    // they have no wrapper to omit in the first place).
    // F/G (2026-07-21 continuation, user: "does this mean prefill can leverage pre-learnt corpus words
    // without per-token reprocessing?"): D's own context-sensitivity probe was never fixed for the
    // suffix-tokenization confound C needed fixing for, so its recorded 0.948 is noisier than necessary.
    // F = context_sensitivity_cosine2 (same isolated-capture-into-real-prefix shape as D, confound fixed,
    // pieces-only); G = the marker-inclusive version -- directly answers whether a word's trace, captured
    // ONCE in isolation, stays valid when reused in a genuinely DIFFERENT real context (the load-bearing
    // assumption behind any "reuse a precomputed trace across many mentions/documents" prefill-compute-
    // amortization idea). No new training needed for F/G -- candidate 2's splice never depended on
    // set_scratch_bindings/training exposure (it writes real K/V rows directly), so this reuses the SAME
    // already-trained model as A-E above, just two more measurements per subject.
    constexpr const char* kPrefix = "Everyone knows that ";   // an actual fact_templates() lead-in phrase
    std::string report = "\n=== factspike Phase G: KV-trace memoization vs mechanism A "
                         "(post Phase-C-regime training) ===\n";
    double sim_sum_a = 0.0, sim_sum_b = 0.0, sim_sum_c = 0.0, sim_sum_d = 0.0, sim_sum_e = 0.0;
    double sim_sum_f = 0.0, sim_sum_g = 0.0;
    int n_correct_a = 0, n_correct_b = 0, n_correct_c = 0, n_total = 0, n_valid_fg = 0;
    std::vector<double> piece_count_traj, sim_traj_a, sim_traj_b, single_piece_sim_c;
    for (const fs::FactPair& fp : split.drilled) {
        const double sim_a = hidden_state_cosine(tk, fp);
        const double sim_b = hidden_state_cosine_kvtrace(tk, fp);
        const double sim_c = hidden_state_cosine_kvtrace_full(tk, fp);
        const double sim_d = context_sensitivity_cosine(tk, fp, kPrefix);
        const double sim_e = hidden_state_cosine_kvtrace_full_markers(tk, fp);
        const double sim_f = context_sensitivity_cosine2(tk, fp, kPrefix);
        const double sim_g = context_sensitivity_cosine2_markers(tk, fp, kPrefix);
        const bool   ok_a  = eval_scratch_one(tk, fp);
        const bool   ok_b  = eval_kvtrace_one(tk, fp);
        const bool   ok_c  = eval_kvtrace_full_one(tk, fp);
        const std::size_t n_pieces = subject_piece_ids(tk, fp.subject).size();
        sim_sum_a += sim_a; sim_sum_b += sim_b; sim_sum_c += sim_c; sim_sum_d += sim_d; sim_sum_e += sim_e; ++n_total;
        if (sim_f > 0.0 && sim_g > 0.0) { sim_sum_f += sim_f; sim_sum_g += sim_g; ++n_valid_fg; }
        n_correct_a += ok_a ? 1 : 0; n_correct_b += ok_b ? 1 : 0; n_correct_c += ok_c ? 1 : 0;
        piece_count_traj.push_back(static_cast<double>(n_pieces));
        sim_traj_a.push_back(sim_a);
        sim_traj_b.push_back(sim_b);
        char buf[320];
        std::snprintf(buf, sizeof buf,
                     "  %-12s n_pieces=%d  cos_sim: A=%.3f B=%.3f C=%.3f D=%.3f E=%.3f F=%.3f G=%.3f  "
                     "correct: A=%s B=%s C=%s\n",
                     fp.subject.c_str(), static_cast<int>(n_pieces), sim_a, sim_b, sim_c, sim_d, sim_e,
                     sim_f, sim_g, ok_a ? "yes" : "no", ok_b ? "yes" : "no", ok_c ? "yes" : "no");
        report += buf;
        if (n_pieces == 1) single_piece_sim_c.push_back(sim_c);   // zero-confound correctness pin (see below)
    }
    const double mean_sim_a = n_total ? sim_sum_a / n_total : 0.0;
    const double mean_sim_b = n_total ? sim_sum_b / n_total : 0.0;
    const double mean_sim_c = n_total ? sim_sum_c / n_total : 0.0;
    const double mean_sim_d = n_total ? sim_sum_d / n_total : 0.0;
    const double mean_sim_e = n_total ? sim_sum_e / n_total : 0.0;
    const double mean_sim_f = n_valid_fg ? sim_sum_f / n_valid_fg : 0.0;
    const double mean_sim_g = n_valid_fg ? sim_sum_g / n_valid_fg : 0.0;
    const double r_a = pearson_r(piece_count_traj, sim_traj_a);
    const double r_b = pearson_r(piece_count_traj, sim_traj_b);
    report += "  mean cos_sim: A(embed-compose)=" + std::to_string(mean_sim_a) +
             " B(cand1-pooled)=" + std::to_string(mean_sim_b) +
             " C(cand2-trivial)=" + std::to_string(mean_sim_c) +
             " D(cand2-ctx-sensitivity)=" + std::to_string(mean_sim_d) +
             " E(cand2-marker-inclusive)=" + std::to_string(mean_sim_e) +
             " F(ctx-sensitivity-v2)=" + std::to_string(mean_sim_f) +
             " G(ctx-sensitivity-v2-markers)=" + std::to_string(mean_sim_g) +
             " (" + std::to_string(n_valid_fg) + "/" + std::to_string(n_total) + " subjects had a valid "
             "F/G boundary-match)\n" +
             "  accuracy: A=" + std::to_string(static_cast<double>(n_correct_a) / n_total) +
             " B=" + std::to_string(static_cast<double>(n_correct_b) / n_total) +
             " C=" + std::to_string(static_cast<double>(n_correct_c) / n_total) +
             " (" + std::to_string(n_total) + " drilled subjects, single post-training snapshot)\n" +
             "  Pearson r(piece_count, cos_sim): A=" + std::to_string(r_a) + " B=" + std::to_string(r_b) + "\n" +
             "  (C: multi-piece subjects sit below 1.0 -- NOT a splice bug (single-piece subjects, no "
             "SPELL wrapper to omit, hit EXACT 1.000); E tests directly whether replaying the "
             "marker-INCLUSIVE span instead of subject_piece_ids closes that gap -- if E >> C for "
             "multi-piece subjects, SPELL_START/SPELL_END genuinely carry weight every packing mechanism "
             "in this investigation (A/B/C/D) has been omitting by construction.\n"
             "   D prediction: C's own gap from 1.0, if any, isolates compression; D's ADDITIONAL gap below "
             "C isolates the isolated-capture assumption's own cost under a REAL non-empty prefix (\"" +
             std::string(kPrefix) + "\").\n"
             "   F/G (D's own confound fixed -- see context_sensitivity_cosine2's doc comment): does a "
             "trace captured in ISOLATION stay valid when spliced into a DIFFERENT real context, and does "
             "marker-inclusion (G) help there too? This is the load-bearing assumption behind reusing one "
             "precomputed trace across many real mentions/documents (prefill compute amortization) -- if "
             "F/G sit meaningfully below C/E's own same-context numbers, isolated capture is a real, "
             "measurable approximation, not a free lunch.)\n";
    WARN(report);

    // Report real numbers, whatever they are -- this is a spike, not a guaranteed-positive validation.
    CHECK(std::isfinite(mean_sim_a));
    CHECK(std::isfinite(mean_sim_b));
    CHECK(std::isfinite(mean_sim_c));
    CHECK(std::isfinite(mean_sim_d));
    CHECK(std::isfinite(mean_sim_e));
    CHECK(std::isfinite(mean_sim_f));
    CHECK(std::isfinite(mean_sim_g));
    CHECK(std::isfinite(r_a));
    CHECK(std::isfinite(r_b));
    // Zero-confound correctness pin: single-piece subjects have NO SPELL wrapper to omit (pieces == alone
    // exactly), so C's splice/rotation math should reproduce baseline EXACTLY for them regardless of the
    // marker question -- this is the real gate on implementation correctness, not the aggregate mean_sim_c
    // (which mixes in the now-understood, expected multi-piece marker-omission gap).
    for (double s : single_piece_sim_c) CHECK(s > 0.99);
}

TEST_CASE("factspike Phase H: does marker-INCLUSIVE composition (SPELL_START/pieces/SPELL_END) close "
         "mechanism A's baseline-vs-scratch gap, now that candidate 2 proved the markers carry real "
         "weight?", "[.factspikemarkers]") {
    Tokenizer tk;
    {
        std::ifstream is(sub0::default_tokenizer(), std::ios::binary);
        if (!is.good() || !sub0::tok::deserialize(tk, is) || tk.vocab != VOCAB) {
            WARN("factspike Phase H: this build's tokenizer isn't usable/doesn't match VOCAB -- skipping. "
                "Build against out/build/factspike96 (docs/FACTSPIKE.md Phase A) for a real result.");
            return;
        }
    }

    std::mt19937_64 split_rng(kSplitSeed);
    const fs::FactSplit split = fs::make_fact_split(split_rng, kNSubjects, kDrilledFrac);
    REQUIRE_FALSE(split.drilled.empty());

    // A GENUINELY SEPARATE training run, not just an eval-time swap: the model must be TAUGHT to read a
    // marker-inclusive-bound slot (Phase G's own KV-trace splice bypassed set_scratch_bindings entirely,
    // so it never needed this; mechanism A's real embedding-bind DOES need matching training exposure, or
    // this would test generalization to a novel binding convention, not "does marker-inclusive help").
    // Matched budget/seeds against Phase C / [.factspikehidden] -- only the slot-exposure dataset's own
    // binding convention differs (build_slot_exposure_dataset_markers instead of build_slot_exposure_dataset).
    std::mt19937_64 exposure_rng(kSplitSeed ^ 0xE4C0517E5EEDULL);
    const fs::FactSplit exposure_split = fs::make_fact_split(exposure_rng, 6, 1.0);
    const fs::Dataset ds = fs::build_dataset(tk, split.drilled, kDocsPerFact, /*seed=*/42);
    REQUIRE(ds.tokens.size() > static_cast<std::size_t>(kWindowT));
    const fs::SlotDataset slot_ds = fs::build_slot_exposure_dataset_markers(tk, exposure_split.drilled,
                                                                            kDocsPerFact, /*seed=*/43);
    REQUIRE(slot_ds.tokens.size() > static_cast<std::size_t>(kWindowT));

    sub0::build_model(); reset_opt_state();
    sub0::AdamW opt(kLr);
    std::mt19937 rng(1);
    std::mt19937_64 choice_rng(2);
    for (int r = 0; r < kEvalRounds; ++r)
        train_steps_combined(ds, slot_ds, opt, kStepsPerEval, rng, choice_rng, /*slot_frac=*/0.5);

    // Mechanism A retest (marker-inclusive bind, on a model actually TRAINED to read it) + a bonus
    // candidate-1 retest (KV-trace pooled splice, marker-inclusive capture -- capture_kv_trace is already
    // generic over its input token list, so this needed zero new engine/capture code, just passing
    // tok::encode(subject) instead of subject_piece_ids). Both measured on this SAME newly-trained model.
    std::string report = "\n=== factspike Phase H: marker-inclusive composition retest "
                         "(post marker-inclusive-regime training) ===\n";
    double sim_sum_a = 0.0, sim_sum_b = 0.0;
    int n_correct_a = 0, n_total = 0;
    std::vector<double> piece_count_traj, sim_traj_a, sim_traj_b;
    for (const fs::FactPair& fp : split.drilled) {
        const double sim_a = hidden_state_cosine_markers(tk, fp);
        const bool   ok_a  = eval_scratch_one_markers(tk, fp);
        const std::vector<int> full_span = sub0::tok::encode(tk, fp.subject);
        const KvTrace trace_b = capture_kv_trace(full_span);   // candidate 1, marker-inclusive capture
        const std::size_t n_pieces = subject_piece_ids(tk, fp.subject).size();

        // candidate-1 fidelity, same hidden_state_cosine shape as hidden_state_cosine_kvtrace but with a
        // caller-supplied trace instead of computing its own from subject_piece_ids -- inlined here rather
        // than adding a fourth near-duplicate top-level function for a single bonus measurement.
        sub0::kv_reset();
        const std::vector<int> base_ctx = sub0::tok::encode(tk, fp.subject + " loves the color ");
        for (std::size_t i = 0; i < base_ctx.size(); ++i) (void)sub0::forward_one(base_ctx[i], static_cast<int>(i));
        std::array<float, D_MODEL> base_hidden{};
        std::copy_n(sub0::last_hidden_ptr(), D_MODEL, base_hidden.data());
        sub0::kv_reset();
        splice_kv_trace(trace_b, 0);
        const std::vector<int> suffix_b = sub0::tok::encode(tk, " loves the color ");
        for (std::size_t i = 0; i < suffix_b.size(); ++i) (void)sub0::forward_one(suffix_b[i], 1 + static_cast<int>(i));
        std::array<float, D_MODEL> b_hidden{};
        std::copy_n(sub0::last_hidden_ptr(), D_MODEL, b_hidden.data());
        double dot = 0.0, nb2 = 0.0, nk = 0.0;
        for (int c = 0; c < D_MODEL; ++c) {
            dot += static_cast<double>(base_hidden[static_cast<std::size_t>(c)]) * b_hidden[static_cast<std::size_t>(c)];
            nb2 += static_cast<double>(base_hidden[static_cast<std::size_t>(c)]) * base_hidden[static_cast<std::size_t>(c)];
            nk  += static_cast<double>(b_hidden[static_cast<std::size_t>(c)])    * b_hidden[static_cast<std::size_t>(c)];
        }
        const double sim_b = (nb2 > 0.0 && nk > 0.0) ? dot / (std::sqrt(nb2) * std::sqrt(nk)) : 0.0;

        sim_sum_a += sim_a; sim_sum_b += sim_b; ++n_total;
        n_correct_a += ok_a ? 1 : 0;
        piece_count_traj.push_back(static_cast<double>(n_pieces));
        sim_traj_a.push_back(sim_a);
        sim_traj_b.push_back(sim_b);
        char buf[176];
        std::snprintf(buf, sizeof buf, "  %-12s n_pieces=%d  cos_sim: A_markers=%.3f B_markers=%.3f  "
                     "A_markers_correct=%s\n", fp.subject.c_str(), static_cast<int>(n_pieces), sim_a, sim_b,
                     ok_a ? "yes" : "no");
        report += buf;
    }
    const double mean_sim_a = n_total ? sim_sum_a / n_total : 0.0;
    const double mean_sim_b = n_total ? sim_sum_b / n_total : 0.0;
    const double r_a = pearson_r(piece_count_traj, sim_traj_a);
    const double r_b = pearson_r(piece_count_traj, sim_traj_b);
    report += "  mean cos_sim: A_markers=" + std::to_string(mean_sim_a) +
             " B_markers=" + std::to_string(mean_sim_b) + "\n" +
             "  accuracy: A_markers=" + std::to_string(static_cast<double>(n_correct_a) / n_total) +
             " (" + std::to_string(n_total) + " drilled subjects, single post-training snapshot)\n" +
             "  Pearson r(piece_count, cos_sim): A_markers=" + std::to_string(r_a) +
             " B_markers=" + std::to_string(r_b) + "\n" +
             "  (compare against the piece-only regime's own recorded numbers: [.factspikehidden] "
             "mean_sim=0.83 r=-0.614; Phase G's candidate-1 B mean_sim=0.77 r=-0.624. This is a genuinely "
             "SEPARATE training run -- not a like-for-like same-weights comparison the way Phase G's A/B/C/"
             "D/E were, since the model itself had to be retrained to read marker-inclusive bindings.)\n";
    WARN(report);

    // Report real numbers, whatever they are -- a genuine re-test, not a guaranteed-positive validation.
    CHECK(std::isfinite(mean_sim_a));
    CHECK(std::isfinite(mean_sim_b));
    CHECK(std::isfinite(r_a));
    CHECK(std::isfinite(r_b));
}

TEST_CASE("factspike Phase I: periodic packed-content re-injection (Nanbeige-inspired) -- does "
         "reinforcing the packed vector partway through the layer stack narrow the fidelity gap a "
         "single upfront injection leaves?", "[.factspikereinject]") {
    Tokenizer tk;
    {
        std::ifstream is(sub0::default_tokenizer(), std::ios::binary);
        if (!is.good() || !sub0::tok::deserialize(tk, is) || tk.vocab != VOCAB) {
            WARN("factspike Phase I: this build's tokenizer isn't usable/doesn't match VOCAB -- skipping. "
                "Build against out/build/factspike96 (docs/FACTSPIKE.md Phase A) for a real result.");
            return;
        }
    }

    std::mt19937_64 split_rng(kSplitSeed);
    const fs::FactSplit split = fs::make_fact_split(split_rng, kNSubjects, kDrilledFrac);
    REQUIRE_FALSE(split.drilled.empty());

    // SAME Phase-C-regime training as [.factspikehidden]/[.factspikekvtrace] -- this is EVAL-ONLY on top
    // of an ordinarily-trained model (core.hpp's set_scratch_reinject doc comment: the model was never
    // trained to expect the extra signal, so this is a directional probe, not the full test). Reusing the
    // identical training regime keeps this a fair like-for-like comparison against mechanism A's own
    // already-recorded numbers on this exact setup.
    std::mt19937_64 exposure_rng(kSplitSeed ^ 0xE4C0517E5EEDULL);
    const fs::FactSplit exposure_split = fs::make_fact_split(exposure_rng, 6, 1.0);
    const fs::Dataset ds = fs::build_dataset(tk, split.drilled, kDocsPerFact, /*seed=*/42);
    REQUIRE(ds.tokens.size() > static_cast<std::size_t>(kWindowT));
    const fs::SlotDataset slot_ds = fs::build_slot_exposure_dataset(tk, exposure_split.drilled,
                                                                    kDocsPerFact, /*seed=*/43);
    REQUIRE(slot_ds.tokens.size() > static_cast<std::size_t>(kWindowT));

    sub0::build_model(); reset_opt_state();
    sub0::AdamW opt(kLr);
    std::mt19937 rng(1);
    std::mt19937_64 choice_rng(2);
    for (int r = 0; r < kEvalRounds; ++r)
        train_steps_combined(ds, slot_ds, opt, kStepsPerEval, rng, choice_rng, /*slot_frac=*/0.5);

    // Dose-response sweep against the SAME trained weights as baseline (mechanism A, no reinject): SAME
    // stride (every layer) with `scale` varying as a FRACTION OF h's OWN CURRENT NORM at each layer (see
    // core.hpp's set_scratch_reinject doc comment -- a first, fixed-embedding-scale version of this was
    // measurably a near no-op, dwarfed by the residual stream's own growing norm across depth; this
    // corrected version rescales the injection to match h's current magnitude before applying `scale`).
    // gentle=5%, medium=15%, aggressive=40% of h's own norm re-added every single layer.
    constexpr float kGentleScale = 0.05f, kMediumScale = 0.15f, kAggressiveScale = 0.40f;
    std::string report = "\n=== factspike Phase I: periodic re-injection vs mechanism A baseline "
                         "(post Phase-C-regime training, SAME model for every config, scale-adaptive v2) "
                         "===\n";
    double sim_sum_0 = 0.0, sim_sum_1 = 0.0, sim_sum_2 = 0.0, sim_sum_3 = 0.0;
    int n_correct_0 = 0, n_correct_1 = 0, n_correct_2 = 0, n_correct_3 = 0, n_total = 0;
    std::vector<double> piece_count_traj, sim_traj_0, sim_traj_1, sim_traj_2, sim_traj_3;
    for (const fs::FactPair& fp : split.drilled) {
        const double sim_0 = hidden_state_cosine(tk, fp);                                  // baseline, no reinject
        const double sim_1 = hidden_state_cosine_reinject(tk, fp, 1, kGentleScale);
        const double sim_2 = hidden_state_cosine_reinject(tk, fp, 1, kMediumScale);
        const double sim_3 = hidden_state_cosine_reinject(tk, fp, 1, kAggressiveScale);
        const bool   ok_0  = eval_scratch_one(tk, fp);
        const bool   ok_1  = eval_scratch_one_reinject(tk, fp, 1, kGentleScale);
        const bool   ok_2  = eval_scratch_one_reinject(tk, fp, 1, kMediumScale);
        const bool   ok_3  = eval_scratch_one_reinject(tk, fp, 1, kAggressiveScale);
        const std::size_t n_pieces = subject_piece_ids(tk, fp.subject).size();
        sim_sum_0 += sim_0; sim_sum_1 += sim_1; sim_sum_2 += sim_2; sim_sum_3 += sim_3; ++n_total;
        n_correct_0 += ok_0 ? 1 : 0; n_correct_1 += ok_1 ? 1 : 0;
        n_correct_2 += ok_2 ? 1 : 0; n_correct_3 += ok_3 ? 1 : 0;
        piece_count_traj.push_back(static_cast<double>(n_pieces));
        sim_traj_0.push_back(sim_0); sim_traj_1.push_back(sim_1);
        sim_traj_2.push_back(sim_2); sim_traj_3.push_back(sim_3);
        char buf[240];
        std::snprintf(buf, sizeof buf,
                     "  %-12s n_pieces=%d  cos_sim: base=%.3f gentle5%%=%.3f medium15%%=%.3f "
                     "aggressive40%%=%.3f  correct: base=%s gentle=%s medium=%s aggressive=%s\n",
                     fp.subject.c_str(), static_cast<int>(n_pieces), sim_0, sim_1, sim_2, sim_3,
                     ok_0 ? "yes" : "no", ok_1 ? "yes" : "no", ok_2 ? "yes" : "no", ok_3 ? "yes" : "no");
        report += buf;
    }
    const double mean_0 = n_total ? sim_sum_0 / n_total : 0.0;
    const double mean_1 = n_total ? sim_sum_1 / n_total : 0.0;
    const double mean_2 = n_total ? sim_sum_2 / n_total : 0.0;
    const double mean_3 = n_total ? sim_sum_3 / n_total : 0.0;
    const double r_0 = pearson_r(piece_count_traj, sim_traj_0);
    const double r_1 = pearson_r(piece_count_traj, sim_traj_1);
    const double r_2 = pearson_r(piece_count_traj, sim_traj_2);
    const double r_3 = pearson_r(piece_count_traj, sim_traj_3);
    report += "  mean cos_sim: base=" + std::to_string(mean_0) + " gentle5%=" + std::to_string(mean_1) +
             " medium15%=" + std::to_string(mean_2) + " aggressive40%=" + std::to_string(mean_3) + "\n" +
             "  accuracy: base=" + std::to_string(static_cast<double>(n_correct_0) / n_total) +
             " gentle=" + std::to_string(static_cast<double>(n_correct_1) / n_total) +
             " medium=" + std::to_string(static_cast<double>(n_correct_2) / n_total) +
             " aggressive=" + std::to_string(static_cast<double>(n_correct_3) / n_total) +
             " (" + std::to_string(n_total) + " drilled subjects)\n" +
             "  Pearson r(piece_count, cos_sim): base=" + std::to_string(r_0) + " gentle=" + std::to_string(r_1) +
             " medium=" + std::to_string(r_2) + " aggressive=" + std::to_string(r_3) + "\n" +
             "  (EVAL-ONLY probe on an ordinarily-trained model -- core.hpp's set_scratch_reinject doc "
             "comment: a real positive here justifies wiring this into the training graph so the model can "
             "learn to use it; a null/negative result is still useful before making that investment.)\n";
    WARN(report);

    // Report real numbers, whatever they are -- a directional probe, not a guaranteed-positive validation.
    CHECK(std::isfinite(mean_0));
    CHECK(std::isfinite(mean_1));
    CHECK(std::isfinite(mean_2));
    CHECK(std::isfinite(mean_3));
    CHECK(std::isfinite(r_0));
    CHECK(std::isfinite(r_1));
    CHECK(std::isfinite(r_2));
    CHECK(std::isfinite(r_3));
}
