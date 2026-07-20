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

#include <catch2/catch_test_macros.hpp>

#include "sub0/core.hpp"
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

    std::string report = "\n=== factspike hidden-state diagnostic (post Phase-C-regime training) ===\n";
    double sim_sum = 0.0; int n_correct = 0, n_total = 0;
    for (const fs::FactPair& fp : split.drilled) {
        const double sim = hidden_state_cosine(tk, fp);
        const bool   ok  = eval_scratch_one(tk, fp);
        sim_sum += sim; ++n_total; n_correct += ok ? 1 : 0;
        char buf[96];
        std::snprintf(buf, sizeof buf, "  %-12s cos_sim=%.3f  scratch_correct=%s\n",
                     fp.subject.c_str(), sim, ok ? "yes" : "no");
        report += buf;
    }
    const double mean_sim = n_total ? sim_sum / n_total : 0.0;
    report += "  mean cosine similarity (baseline vs scratch final hidden state) = " +
             std::to_string(mean_sim) + "  (" + std::to_string(n_correct) + "/" +
             std::to_string(n_total) + " correct this round)\n";
    WARN(report);

    CHECK(std::isfinite(mean_sim));
}
